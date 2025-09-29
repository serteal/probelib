"""
Tokenization utilities for probelib.

This module provides utilities for tokenizing dialogues and creating detection masks
using the new mask system instead of use_for_training flags.
"""

import re
from typing import TYPE_CHECKING, Any, Optional, Sequence

import torch

from ..datasets import DialogueDataset
from ..logger import logger
from ..masks import MaskFunction, TokenMetadata
from ..models.architectures import ArchitectureRegistry
from ..types import Dialogue, Message

if TYPE_CHECKING:
    from transformers.tokenization_utils_base import PreTrainedTokenizerBase


def preprocess_dialogue(
    dialogue: Dialogue, fold_system: bool = False
) -> list[dict[str, str]]:
    """Prepare a dialogue for ``apply_chat_template`` while tracking transformations.

    - Adjacent messages with the same role are concatenated so downstream masking
      and activation collection operate on full conversational turns instead of
      fragments.
    - When ``fold_system`` is ``True`` (e.g. Gemma-style chat), system content is
      merged into the opening user message to mirror the prompt format expected by
      the tokenizer.
    - The return value is a list of ``{"role": ..., "content": ...}`` dictionaries
      understood by Hugging Face chat templates; the subsequent metadata step
      records how those processed messages align with tokens so custom masks can
      recover role/message boundaries without re-tokenizing.
    """
    processed: list[dict[str, str]] = []
    if fold_system and dialogue and dialogue[0].role == "system":
        processed.append(
            {"role": "user", "content": dialogue[0].content.strip() + "\n\n"}
        )
        dialogue = dialogue[1:]

    for message in dialogue:
        if processed and processed[-1]["role"] == message.role:
            processed[-1]["content"] += message.content.strip()
        else:
            processed.append({"role": message.role, "content": message.content.strip()})

    return processed


def build_token_metadata(
    dialogues: Sequence[Dialogue],
    formatted_dialogues: Sequence[str],
    tokenizer: "PreTrainedTokenizerBase",
    tokenizer_out: dict[str, torch.Tensor],
) -> TokenMetadata:
    """Build metadata for efficient mask evaluation.

    This function creates metadata tensors that map tokens to messages and roles,
    enabling fast vectorized mask evaluation.
    """
    batch_size, seq_len = tokenizer_out["input_ids"].shape
    device = tokenizer_out["input_ids"].device

    # Initialize metadata tensors
    # Use -1 as default to indicate "no role" (not system!)
    role_ids_no_padding = torch.full(
        (batch_size, seq_len), -1, dtype=torch.int8, device=device
    )
    # Initialize with -1 to indicate tokens not belonging to any message (e.g., special tokens)
    message_boundaries = torch.full(
        (batch_size, seq_len), -1, dtype=torch.int32, device=device
    )

    # Role mapping (system is 0, user is 1, assistant is 2)
    role_to_id = {"system": 0, "user": 1, "assistant": 2}

    # Prepare BOS token IDs for optional padding-based role assignment later
    bos_token_ids = {0, 1, 2, 128000}
    if hasattr(tokenizer, "bos_token_id") and tokenizer.bos_token_id is not None:
        bos_token_ids.add(tokenizer.bos_token_id)

    # Get model family for regex patterns and padding
    model_family = get_model_family(tokenizer)
    prefix_pattern = _get_prefix_pattern(model_family)
    # Get architecture handler and its padding configuration
    arch = ArchitectureRegistry.get_architecture_by_name(model_family)
    padding_config = arch.get_token_padding()

    # Process each dialogue
    for batch_idx, dialogue in enumerate(dialogues):
        char_idx = 0
        formatted_text = formatted_dialogues[batch_idx]

        for msg_idx, message in enumerate(dialogue):
            # For models that don't support system messages (like Gemma),
            # system content gets merged with user, so skip explicit system messages
            if model_family == "gemma" and message.role == "system":
                # The content will be part of the user message in the formatted text
                continue

            # Find the start of the message content
            match = re.match(prefix_pattern, formatted_text[char_idx:])
            if match is None:
                logger.warning(f"Could not match prefix pattern at position {char_idx}")
                continue

            start_char_idx = char_idx + match.end()
            end_char_idx = start_char_idx + len(message.content.strip())

            # Find corresponding token indices. ``char_to_token`` returns the index of the
            # token covering a particular character position. The exclusive end of the
            # slice therefore needs the token index of the *last* content character plus
            # one; otherwise we incorrectly include the remainder of the sequence when the
            # end character falls exactly on a token boundary.
            start_tok_idx = tokenizer_out.char_to_token(batch_idx, start_char_idx)

            end_tok_inclusive = None
            if len(message.content.strip()) > 0:
                end_tok_inclusive = tokenizer_out.char_to_token(
                    batch_idx, max(start_char_idx, end_char_idx - 1)
                )

            if start_tok_idx is not None:
                if end_tok_inclusive is None:
                    # char_to_token can return ``None`` when the message is empty or when
                    # we're at the very end of the decoded sequence. Fall back to the
                    # start token so we at least cover the first character instead of
                    # spilling to the rest of the sequence.
                    end_tok_inclusive = start_tok_idx

                exclusive_end = min(seq_len, end_tok_inclusive + 1)

                # Set role and message IDs for content tokens only (no padding)
                role_id = role_to_id[message.role]
                role_ids_no_padding[batch_idx, start_tok_idx:exclusive_end] = role_id
                message_boundaries[batch_idx, start_tok_idx:exclusive_end] = msg_idx

            char_idx = end_char_idx

    # Create padded version of role_ids using efficient vectorized operations
    role_ids_with_padding = role_ids_no_padding.clone()

    # Only apply padding if padding config exists
    if padding_config.left > 0 or padding_config.right > 0:
        for batch_idx in range(batch_size):
            # Process each role efficiently
            for role_id in [0, 1, 2]:  # system, user, assistant
                # Find where this role appears
                role_mask = role_ids_no_padding[batch_idx] == role_id

                if not role_mask.any():
                    continue

                # Find boundaries of role regions using diff
                # Pad with False to detect edges
                padded_mask = torch.cat(
                    [
                        torch.tensor([False], device=device),
                        role_mask,
                        torch.tensor([False], device=device),
                    ]
                )

                # Find starts and ends of continuous regions
                diff = padded_mask[1:].int() - padded_mask[:-1].int()
                starts = torch.where(diff == 1)[0]
                ends = torch.where(diff == -1)[0]

                # Apply padding to each region
                for start, end in zip(starts, ends):
                    # Apply left and right padding
                    padded_start = max(0, start - padding_config.left)
                    padded_end = min(seq_len, end + padding_config.right)

                    # Set the padded region to this role
                    role_ids_with_padding[batch_idx, padded_start:padded_end] = role_id

    # Mark BOS tokens as system only in the padded view (include_padding=True)
    for batch_idx in range(batch_size):
        first_token_id = tokenizer_out["input_ids"][batch_idx, 0].item()
        if first_token_id in bos_token_ids:
            role_ids_with_padding[batch_idx, 0] = role_to_id["system"]

    special_token_ids: set[int] | None = None
    if hasattr(tokenizer, "all_special_ids") and tokenizer.all_special_ids is not None:
        try:
            special_token_ids = {
                int(token_id)
                for token_id in tokenizer.all_special_ids
                if token_id is not None
            }
        except TypeError:
            # Fall back if tokenizer reports non-iterable
            pass

    return TokenMetadata(
        token_ids=tokenizer_out["input_ids"],
        role_ids=role_ids_with_padding,  # Use padded version by default
        message_boundaries=message_boundaries,
        attention_mask=tokenizer_out["attention_mask"],
        char_to_token=tokenizer_out.char_to_token
        if hasattr(tokenizer_out, "char_to_token")
        else None,
        token_to_char=None,  # Not available in tokenizer output
        formatted_texts=formatted_dialogues,
        role_ids_no_padding=role_ids_no_padding,  # Store unpadded version
        architecture=model_family,  # Store architecture info
        special_token_ids=special_token_ids,
    )


def tokenize_dialogues(
    tokenizer: "PreTrainedTokenizerBase",
    dialogues: Sequence[Dialogue],
    mask: Optional[MaskFunction] = None,
    device: torch.device | str = "cpu",
    dataset: DialogueDataset | None = None,
    add_generation_prompt: bool = False,
    **tokenize_kwargs: Any,
) -> dict[str, torch.Tensor]:
    """Unified tokenization with optional masking.

    When ``mask`` is ``None`` and a ``DialogueDataset`` is provided, the
    dataset's ``default_mask`` is applied automatically. Callers that want to
    disable masking entirely should pass an explicit mask (e.g. ``masks.all()``).

    Args:
        tokenizer: HuggingFace tokenizer bound to the model.
        dialogues: Sequence of dialogues to tokenize.
        mask: Mask function to apply. ``None`` defers to ``dataset.default_mask``
            when available, otherwise no masking is applied.
        device: Device to place tensors on.
        dataset: Optional dataset used to resolve a default mask.
        add_generation_prompt: Whether to append the model's generation prompt.
        **tokenize_kwargs: Additional tokenizer arguments forwarded verbatim.

    Returns:
        Tokenized tensors, including ``detection_mask`` that reflects either the
        supplied mask, the dataset default, or an all-False mask when masking is
        disabled.
    """
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Convert dialogues to format expected by tokenizer
    fold_system = get_model_family(tokenizer) == "gemma"
    dialogue_dicts = [
        preprocess_dialogue(dialogue, fold_system) for dialogue in dialogues
    ]

    # Build a processed Dialogue (Message objects) aligned with formatted text
    processed_dialogues: list[list[Message]] = [
        [Message(role=m["role"], content=m["content"]) for m in d]
        for d in dialogue_dicts
    ]

    # Apply chat template if available, otherwise use simple formatting
    if hasattr(tokenizer, 'chat_template') and tokenizer.chat_template is not None:
        formatted_dialogues = tokenizer.apply_chat_template(
            dialogue_dicts,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )
    else:
        # Fallback for models without chat templates (e.g., LLAMA-2)
        # For simple single-turn dialogues, just use the content directly
        formatted_dialogues = []
        for dialogue_dict in dialogue_dicts:
            if len(dialogue_dict) == 1 and dialogue_dict[0]["role"] == "user":
                # Single user message - just use the content
                formatted_dialogues.append(dialogue_dict[0]["content"])
            else:
                # Multi-turn dialogue - format as simple text
                formatted = ""
                for msg in dialogue_dict:
                    if msg["role"] == "system":
                        formatted += f"System: {msg['content']}\n\n"
                    elif msg["role"] == "user":
                        formatted += f"User: {msg['content']}\n\n"
                    elif msg["role"] == "assistant":
                        formatted += f"Assistant: {msg['content']}\n\n"
                formatted_dialogues.append(formatted.strip())

    default_tokenize_kwargs: dict[str, Any] = {
        "return_tensors": "pt",
        "padding": True,
        "add_special_tokens": False,
    }
    default_tokenize_kwargs.update(tokenize_kwargs)

    # Tokenize
    token_dict = tokenizer(formatted_dialogues, **default_tokenize_kwargs)  # type: ignore

    # Move to device
    for k, v in token_dict.items():
        if isinstance(v, torch.Tensor):
            token_dict[k] = v.to(device)
        elif isinstance(v, list):
            token_dict[k] = torch.tensor(v, device=device)

    if "attention_mask" not in token_dict:
        raise ValueError("Tokenizer output must include attention mask")

    # Apply mask if provided
    if mask is not None:
        # Build metadata for mask evaluation
        metadata = build_token_metadata(
            processed_dialogues, formatted_dialogues, tokenizer, token_dict
        )

        # Evaluate mask
        detection_mask = mask.evaluate(dialogues, metadata)
        token_dict["detection_mask"] = detection_mask
    elif dataset is not None and hasattr(dataset, "default_mask"):
        # Use dataset's default mask
        mask_fn = dataset.default_mask
        logger.info(
            "No mask provided; using default mask %s from dataset %s",
            mask_fn.__class__.__name__,
            dataset.__class__.__name__,
        )
        metadata = build_token_metadata(
            processed_dialogues, formatted_dialogues, tokenizer, token_dict
        )
        detection_mask = mask_fn.evaluate(dialogues, metadata)
        token_dict["detection_mask"] = detection_mask
    else:
        # No mask - default to selecting all real (non-padding) tokens.
        token_dict["detection_mask"] = token_dict["attention_mask"].bool()

    return token_dict  # type: ignore


def tokenize_dataset(
    dataset: DialogueDataset,
    tokenizer: "PreTrainedTokenizerBase",
    mask: Optional[MaskFunction] = None,
    device: torch.device | str = "cpu",
    **tokenize_kwargs: Any,
) -> dict[str, torch.Tensor]:
    """Tokenize a dataset while respecting default masking rules.

    See :func:`tokenize_dialogues` for details on how dataset defaults are
    applied when ``mask`` is ``None``.

    Args:
        dataset: DialogueDataset to tokenize.
        tokenizer: HuggingFace tokenizer aligned with the model.
        mask: Optional mask override. ``None`` defers to ``dataset.default_mask``.
        device: Device to place tensors on.
        **tokenize_kwargs: Additional tokenizer arguments.

    Returns:
        Dictionary with tokenized outputs including ``detection_mask``.
    """
    return tokenize_dialogues(
        tokenizer=tokenizer,
        dialogues=dataset.dialogues,
        mask=mask,
        device=device,
        dataset=dataset,
        **tokenize_kwargs,
    )


# Cache for compiled regex patterns
_PREFIX_PATTERN_CACHE: dict[str, re.Pattern[str]] = {}


def _get_prefix_pattern(model_family: str) -> re.Pattern[str]:
    """
    Get regex pattern for matching chat template tokens.

    This pattern is used to accurately locate message boundaries in tokenized text.

    Args:
        model_family: Model family name ("llama", "gemma", "mistral")

    Returns:
        Compiled regex pattern for the model family
    """
    # Return cached pattern if available
    if model_family in _PREFIX_PATTERN_CACHE:
        return _PREFIX_PATTERN_CACHE[model_family]

    # Compile pattern based on model family
    if model_family == "gemma":
        begin_of_text = r"(<pad>)*(<bos>)?"
        end_of_last = r"(<end_of_turn>\n)?"
        start_of_turn = r"<start_of_turn>(user|model)\n"
        pattern = re.compile(
            rf"{begin_of_text}{start_of_turn}|{end_of_last}{start_of_turn}|(\n\n)?"
        )
    elif model_family == "llama":
        begin_of_text = r"((<\|pad\|>)*(<\|begin_of_text\|>))?"
        header = r"<\|start_header_id\|>(system|user|assistant)<\|end_header_id\|>\n\n"
        end_of_turn = r"<\|eot_id\|>"
        # llama3 system prompt has some boilerplate which we keep constant across all prompts
        date_info = r"(Cutting Knowledge Date: December 2023\nToday Date: \d\d \w\w\w 202[45]\n\n)?"
        pattern = re.compile(
            rf"({begin_of_text}{header}{date_info})?({end_of_turn}{header})?(\n\n)?"
        )
    else:
        # Default fallback pattern that matches common separators
        pattern = re.compile(r"(\n\n)?")

    # Cache and return
    _PREFIX_PATTERN_CACHE[model_family] = pattern
    return pattern


def get_model_family(tokenizer: "PreTrainedTokenizerBase") -> str:
    """
    Determine model family from tokenizer name.

    Args:
        tokenizer: HuggingFace tokenizer

    Returns:
        Model family string ("llama", "gemma", etc.)
    """
    return ArchitectureRegistry.detect_from_tokenizer_name(tokenizer.name_or_path)
