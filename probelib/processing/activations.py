"""
Simplified activation collection using generators for cleaner API.

This module provides tools for extracting activations from language models using hooks,
with support for different model architectures and efficient memory management.
"""

from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Any,
    Generator,
    Iterator,
    Literal,
    Optional,
    Sequence,
    overload,
)

import torch
from jaxtyping import Float
from tqdm.auto import tqdm

from ..datasets import DialogueDataset
from ..models import HookedModel
from ..models.architectures import ArchitectureRegistry
from ..types import Dialogue
from .tokenization import tokenize_dialogues

if TYPE_CHECKING:
    from transformers import PreTrainedModel
    from transformers.tokenization_utils_base import PreTrainedTokenizerBase

    from ..masks import MaskFunction


def get_batches(
    inputs: dict[str, torch.Tensor],
    batch_size: int,
    tokenizer: "PreTrainedTokenizerBase",
) -> Iterator[tuple[dict[str, torch.Tensor], list[int]]]:
    """Yield length-aware batches while preserving original indices.

    Sequences are sorted by non-padding length so each batch shares a similar
    sequence length. This minimizes padding, keeps GPU transfers tight, and
    slices away excess padding for every field in ``inputs``.

    Args:
        inputs: Tokenized inputs keyed by field name.
        batch_size: Maximum number of sequences per batch.
        tokenizer: Provides padding semantics used to trim left/right padding.
    """
    # Get sequence lengths and sort by length for efficient batching
    seq_lengths = (inputs["input_ids"] != tokenizer.pad_token_id).sum(dim=1)  # type: ignore
    sorted_indices = torch.sort(seq_lengths)[1]

    # Create batches
    num_samples = sorted_indices.numel()
    for start in range(0, num_samples, batch_size):
        end = min(start + batch_size, num_samples)
        batch_indices_tensor = sorted_indices[start:end]
        batch_indices = batch_indices_tensor.tolist()

        batch_lengths = seq_lengths.index_select(0, batch_indices_tensor)
        batch_length = int(batch_lengths.max().item())

        if tokenizer.padding_side == "right":
            batch_inputs = {
                key: tensor.index_select(0, batch_indices_tensor)[..., :batch_length]
                for key, tensor in inputs.items()
            }
        elif tokenizer.padding_side == "left":
            batch_inputs = {
                key: tensor.index_select(0, batch_indices_tensor)[..., -batch_length:]
                for key, tensor in inputs.items()
            }
        else:
            raise ValueError(f"Unknown padding side: {tokenizer.padding_side}")

        yield batch_inputs, batch_indices


def get_n_layers(model: "PreTrainedModel") -> int:
    """Get number of layers in the model using the architecture registry."""
    architecture = ArchitectureRegistry.get_architecture(model)
    return architecture.get_num_layers(model)


def get_hidden_dim(model: "PreTrainedModel") -> int:
    """Get hidden dimension of the model."""
    config = model.config

    if hasattr(config, "hidden_size"):
        return config.hidden_size  # type: ignore
    else:
        raise ValueError(f"Cannot determine hidden dimension for {model.name_or_path}")


@dataclass
class Activations:
    """Activation tensor plus the metadata required to interpret it.

    Instances always keep tensors aligned: ``activations`` uses
    ``layer_indices`` to record which model layers were captured, auxiliary
    tensors (attention mask, detection mask, input ids) stay on the same device
    as the activations, and optional ``batch_indices`` tells streaming callers
    which original examples a batch corresponds to. Helper methods such as
    ``filter_layers`` and ``aggregate`` rely on these invariants and preserve
    metadata, so downstream probes can focus on modelling instead of plumbing.
    """

    activations: Float[torch.Tensor, "n_layers batch_size seq_len d_model"]
    attention_mask: Float[torch.Tensor, "batch_size seq_len"]
    input_ids: Float[torch.Tensor, "batch_size seq_len"]
    detection_mask: Float[torch.Tensor, "batch_size seq_len"]
    layer_indices: list[
        int
    ]  # Maps layer dimension in `activations` to original model layer indices
    batch_indices: list[int] | None = None  # Original indices of samples in this batch

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.activations.shape  # type: ignore

    @property
    def n_layers(self) -> int:
        return self.activations.shape[0]

    @property
    def batch_size(self) -> int:
        return self.activations.shape[1]

    @property
    def seq_len(self) -> int:
        return self.activations.shape[2]

    @property
    def d_model(self) -> int:
        return self.activations.shape[3]

    def get_layer_tensor_indices(self, requested_layers: int | list[int]) -> list[int]:
        """
        Map requested model layer indices to tensor dimension indices.

        Args:
            requested_layers: List of original model layer indices

        Returns:
            List of tensor indices that correspond to the requested layers

        Raises:
            ValueError: If any requested layer is not available in this Activations object
        """
        if isinstance(requested_layers, int):
            requested_layers = [requested_layers]

        tensor_indices = []
        for layer in requested_layers:
            try:
                tensor_idx = self.layer_indices.index(layer)
                tensor_indices.append(tensor_idx)
            except ValueError:
                available_layers = ", ".join(map(str, self.layer_indices))
                raise ValueError(
                    f"Requested layer {layer} is not available. "
                    f"Available layers: [{available_layers}]"
                )
        return tensor_indices

    def __post_init__(self):
        """Validate shapes and apply attention mask."""
        shape = (self.batch_size, self.seq_len)
        assert self.attention_mask.shape == shape, (
            f"Attention mask shape {self.attention_mask.shape} doesn't match {shape}"
        )
        assert self.input_ids.shape == shape, (
            f"Input IDs shape {self.input_ids.shape} doesn't match {shape}"
        )

        device = self.activations.device
        self.attention_mask = self.attention_mask.to(device)
        self.input_ids = self.input_ids.to(device)
        self.detection_mask = self.detection_mask.to(device)

        # Ensure activations are floating point (but preserve the specific dtype if already float)
        if not self.activations.is_floating_point():
            self.activations = self.activations.float()

    def to(self, *args, **kwargs) -> "Activations":
        """Move activation to device/dtype.

        Supports same arguments as torch.Tensor.to():
        - device only: .to(device)
        - dtype only: .to(dtype)
        - both: .to(device, dtype) or .to(dtype=dtype, device=device)
        """
        # Copy layer indices since they don't need device/dtype conversion
        layer_indices = self.layer_indices.copy()

        # Convert activations using same args
        converted_activations = self.activations.to(*args, **kwargs)

        # For device changes, also move other tensors
        if (
            any(isinstance(arg, (str, torch.device)) for arg in args)
            or "device" in kwargs
        ):
            # Extract just the device argument
            if "device" in kwargs:
                device = kwargs["device"]
            else:
                device = next(
                    arg for arg in args if isinstance(arg, (str, torch.device))
                )

            return Activations(
                activations=converted_activations,
                attention_mask=self.attention_mask.to(device),
                input_ids=self.input_ids.to(device),
                detection_mask=self.detection_mask.to(device),
                layer_indices=layer_indices,
                batch_indices=self.batch_indices.copy()
                if self.batch_indices is not None
                else None,
            )

        # For dtype only changes, only convert activations
        return Activations(
            activations=converted_activations,
            attention_mask=self.attention_mask,
            input_ids=self.input_ids,
            detection_mask=self.detection_mask,
            layer_indices=layer_indices,
            batch_indices=self.batch_indices,
        )

    def filter_layers(self, target_layers: int | list[int]) -> "Activations":
        """
        Filter activations to only include specified layers.

        Args:
            target_layers: List of layer indices to keep

        Returns:
            New Activations object with filtered layers
        """
        if isinstance(target_layers, int):
            target_layers = [target_layers]

        # Get tensor indices for target layers
        tensor_indices = self.get_layer_tensor_indices(target_layers)

        # Filter activations tensor
        filtered_activations = self.activations[tensor_indices]

        return Activations(
            activations=filtered_activations,
            input_ids=self.input_ids,
            attention_mask=self.attention_mask,
            detection_mask=self.detection_mask,
            layer_indices=target_layers,
            batch_indices=self.batch_indices,
        )

    def aggregate(
        self, method: Literal["mean", "max", "last_token"] = "mean"
    ) -> torch.Tensor:
        """
        Aggregate activations over sequence dimension for single layer.

        Args:
            method: Aggregation method
                - 'mean': Average over detected tokens
                - 'max': Maximum over detected tokens
                - 'last_token': Use the last detected token

        Returns:
            Tensor with shape [n_samples, d_model]

        Raises:
            ValueError: If not single layer
        """
        if self.n_layers != 1:
            raise ValueError(
                f"Aggregation requires single layer, got {self.n_layers} layers. "
                f"Use filter_layers([layer_idx]) first."
            )

        acts = self.activations[0]  # [n_samples, seq_len, d_model]
        mask = self.detection_mask  # [n_samples, seq_len]

        # Apply detection mask
        mask_expanded = mask.unsqueeze(-1)  # [n_samples, seq_len, 1]
        masked_acts = acts * mask_expanded

        # Aggregate over sequence
        if method == "mean":
            counts = mask.sum(dim=1, keepdim=True).clamp(min=1)  # [n_samples, 1]
            aggregated = masked_acts.sum(dim=1) / counts
        elif method == "max":
            # Replace masked positions with -inf for max
            masked_acts = acts.clone()
            masked_acts[~mask.bool()] = float("-inf")
            aggregated = masked_acts.max(dim=1).values

            # Handle batches with no valid tokens (replace -inf with 0)
            no_valid = ~mask.bool().any(dim=1)  # [n_samples]
            aggregated[no_valid] = 0.0
        elif method == "last_token":
            # Get last valid token per sequence
            last_indices = mask.sum(dim=1) - 1  # [n_samples]

            # Handle batches with no valid tokens
            no_valid = ~mask.bool().any(dim=1)
            last_indices = last_indices.clamp(min=0)  # Avoid negative indices

            batch_indices = torch.arange(self.batch_size, device=acts.device)
            aggregated = acts[batch_indices, last_indices.long()]

            # Zero out batches with no valid tokens
            aggregated[no_valid] = 0.0
        else:
            raise ValueError(f"Unknown aggregation method: {method}")

        return aggregated

    def to_token_level(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Extract token-level features for token-level training.

        Returns:
            (features, tokens_per_sample) where:
            - features: [n_total_tokens, d_model] containing only detected tokens
            - tokens_per_sample: [n_samples] with count of tokens per sample for label expansion

        Raises:
            ValueError: If not single layer
        """
        if self.n_layers != 1:
            raise ValueError(
                f"Token extraction requires single layer, got {self.n_layers} layers. "
                f"Use filter_layers([layer_idx]) first."
            )

        acts = self.activations[0]  # [n_samples, seq_len, d_model]
        mask = self.detection_mask.bool()  # [n_samples, seq_len]

        # Vectorized token extraction – flatten mask and gather in a single op
        tokens_per_sample = mask.sum(dim=1)

        if tokens_per_sample.sum() == 0:
            features = torch.empty(
                0, self.d_model, device=acts.device, dtype=acts.dtype
            )
        else:
            features = acts[mask]  # [n_total_tokens, d_model]

        tokens_per_sample = tokens_per_sample.to(device=acts.device)

        return features, tokens_per_sample


def streaming_activations(
    model: "PreTrainedModel",
    tokenizer: "PreTrainedTokenizerBase",
    tokenized_inputs: dict[str, torch.Tensor],
    layers: list[int],
    batch_size: int = 8,
    verbose: bool = False,
) -> Generator[Activations, None, None]:
    """
    Generator that yields Activations batches with all optimizations preserved.

    Key optimizations:
    - Single HookedModel context for entire iteration
    - Sorted batching for minimal padding
    - Tensor views via narrow() where possible
    - Non-blocking GPU transfers
    - Sequence length optimization
    - Buffer reuse in HookedModel

    Args:
        model: Model to extract activations from
        tokenizer: Tokenizer for padding info
        tokenized_inputs: Pre-tokenized inputs dictionary
        layers: Layer indices to extract
        batch_size: Batch size for processing
        verbose: Whether to show progress bar

    Yields:
        Activations objects for each batch
    """
    n_samples = tokenized_inputs["input_ids"].shape[0]

    # Use HookedModel context for all batches (preserves buffer reuse optimization)
    with HookedModel(model, layers) as hooked_model:
        # Get optimized batches with sorting and views
        batch_iter = get_batches(tokenized_inputs, batch_size, tokenizer)

        # Add progress bar if requested
        if verbose:
            batch_iter = tqdm(
                batch_iter,
                desc="Collecting activations",
                total=(n_samples + batch_size - 1) // batch_size,
            )

        for batch_inputs, batch_indices in batch_iter:
            if batch_inputs["input_ids"].device != model.device:
                batch_inputs = {
                    k: v.to(model.device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch_inputs.items()
                }
            yield Activations(
                activations=hooked_model.get_activations(batch_inputs),
                attention_mask=batch_inputs["attention_mask"],
                input_ids=batch_inputs["input_ids"],
                detection_mask=batch_inputs["detection_mask"],
                layer_indices=layers,
                batch_indices=batch_indices,
            )


def batch_activations(
    model: "PreTrainedModel",
    tokenizer: "PreTrainedTokenizerBase",
    tokenized_inputs: dict[str, torch.Tensor],
    layers: list[int],
    batch_size: int = 8,
    verbose: bool = False,
) -> Activations:
    """
    Collect all activations at once into a single Activations object.

    Uses different memory strategies based on dataset size:
    - Small datasets: Pre-allocate full tensor
    - Large datasets: Accumulate batches then concatenate

    Args:
        model: Model to extract activations from
        tokenizer: Tokenizer for padding info
        tokenized_inputs: Pre-tokenized inputs
        layers: Layer indices to extract
        batch_size: Batch size for processing
        verbose: Whether to show progress

    Returns:
        Single Activations object with all data
    """
    n_samples, max_seq_len = tokenized_inputs["input_ids"].shape
    hidden_dim = get_hidden_dim(model)

    # Use different strategies based on dataset size
    if n_samples * max_seq_len < 100000:  # Small dataset
        # Pre-allocate full tensor for efficiency
        all_activations = torch.zeros(
            (len(layers), n_samples, max_seq_len, hidden_dim),
            device="cpu",
            dtype=model.dtype,
        )

        with HookedModel(model, layers) as hooked_model:
            batches = get_batches(tokenized_inputs, batch_size, tokenizer)

            if verbose:
                batches = tqdm(
                    batches,
                    desc="Collecting activations",
                    total=(n_samples + batch_size - 1) // batch_size,
                )

            for batch_inputs, batch_indices in batches:
                # Move to device if needed
                if batch_inputs["input_ids"].device != model.device:
                    batch_inputs = {
                        k: v.to(model.device) for k, v in batch_inputs.items()
                    }

                seq_len = batch_inputs["input_ids"].shape[1]
                batch_acts = hooked_model.get_activations(batch_inputs)
                batch_acts = batch_acts.to("cpu", non_blocking=True)

                # Store in correct positions
                if tokenizer.padding_side == "right":
                    all_activations[:, batch_indices, :seq_len] = batch_acts
                else:
                    all_activations[:, batch_indices, -seq_len:] = batch_acts
    else:
        # Large dataset - accumulate batches to avoid huge upfront allocation
        batch_list = []
        indices_list = []

        with HookedModel(model, layers) as hooked_model:
            batches = get_batches(tokenized_inputs, batch_size, tokenizer)

            if verbose:
                batches = tqdm(
                    batches,
                    desc="Collecting activations",
                    total=(n_samples + batch_size - 1) // batch_size,
                )

            for batch_inputs, batch_indices in batches:
                # Move to device if needed
                if batch_inputs["input_ids"].device != model.device:
                    batch_inputs = {
                        k: v.to(model.device) for k, v in batch_inputs.items()
                    }

                batch_acts = hooked_model.get_activations(batch_inputs)
                batch_acts = batch_acts.to("cpu", non_blocking=True)
                batch_list.append(batch_acts)
                indices_list.append(batch_indices)

        # Synchronize and create full tensor
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        all_activations = torch.zeros(
            (len(layers), n_samples, max_seq_len, hidden_dim),
            device="cpu",
            dtype=model.dtype,
        )

        for batch_acts, batch_indices in zip(batch_list, indices_list):
            seq_len = batch_acts.shape[2]
            if tokenizer.padding_side == "right":
                all_activations[:, batch_indices, :seq_len] = batch_acts
            else:
                all_activations[:, batch_indices, -seq_len:] = batch_acts

    return Activations(
        activations=all_activations,
        attention_mask=tokenized_inputs["attention_mask"],
        input_ids=tokenized_inputs["input_ids"],
        detection_mask=tokenized_inputs["detection_mask"],
        layer_indices=layers,
    )


class ActivationIterator:
    """
    Regenerable iterator for streaming activations.

    This iterator can be iterated multiple times, creating a fresh generator
    each time. This allows multiple passes over the data without needing to
    recreate the iterator.
    """

    def __init__(
        self,
        model: "PreTrainedModel",
        tokenizer: "PreTrainedTokenizerBase",
        tokenized_inputs: dict[str, torch.Tensor],
        layers: list[int],
        batch_size: int,
        verbose: bool,
        num_batches: int,
    ):
        """
        Initialize the regenerable iterator.

        Args:
            model: Model to extract activations from
            tokenizer: Tokenizer for padding info
            tokenized_inputs: Pre-tokenized inputs
            layers: Layer indices to extract
            batch_size: Batch size for processing
            verbose: Whether to show progress
            num_batches: Total number of batches
        """
        self._model = model
        self._tokenizer = tokenizer
        self._tokenized_inputs = tokenized_inputs
        self._layers = layers
        self._batch_size = batch_size
        self._verbose = verbose
        self._num_batches = num_batches

    def __iter__(self) -> Iterator[Activations]:
        """Create and return a fresh generator for activation batches."""
        return streaming_activations(
            model=self._model,
            tokenizer=self._tokenizer,
            tokenized_inputs=self._tokenized_inputs,
            layers=self._layers,
            batch_size=self._batch_size,
            verbose=self._verbose,
        )

    def __len__(self) -> int:
        """Return number of batches."""
        return self._num_batches

    @property
    def layers(self) -> list[int]:
        """Return layer indices this iterator provides."""
        return self._layers


# Adds overlading for correct return type inference from collect_activations
@overload
def collect_activations(
    model: "PreTrainedModel",
    tokenizer: "PreTrainedTokenizerBase",
    data: DialogueDataset | Sequence[Dialogue],
    *,
    layers: int | list[int],
    batch_size: int,
    mask: Optional["MaskFunction"] = None,
    add_generation_prompt: bool = False,
    streaming: Literal[False] = False,
    verbose: bool = False,
    **tokenize_kwargs: Any,
) -> Activations: ...


# Adds overlading for correct return type inference from collect_activations
@overload
def collect_activations(
    model: "PreTrainedModel",
    tokenizer: "PreTrainedTokenizerBase",
    data: DialogueDataset | Sequence[Dialogue],
    *,
    layers: int | list[int],
    batch_size: int,
    mask: Optional["MaskFunction"] = None,
    add_generation_prompt: bool = False,
    streaming: Literal[True],
    verbose: bool = False,
    **tokenize_kwargs: Any,
) -> ActivationIterator: ...


def collect_activations(
    model: "PreTrainedModel",
    tokenizer: "PreTrainedTokenizerBase",
    data: DialogueDataset | Sequence[Dialogue],
    *,
    layers: int | list[int],
    batch_size: int,
    mask: Optional["MaskFunction"] = None,
    add_generation_prompt: bool = False,
    streaming: bool = False,
    verbose: bool = False,
    **tokenize_kwargs: Any,
) -> Activations | ActivationIterator:
    """Entry point for activation collection across datasets or dialogue lists.

    The function tokenizes once, applies any mask exactly once, and then either
    returns a materialized :class:`Activations` tensor or a streaming iterator
    that yields compatible ``Activations`` batches. Callers only need to specify
    *what* to collect (layers, mask, batch size); the helper handles padding
    strategy, device placement, and hook management under the hood.

    Args:
        model: Model providing hidden states.
        tokenizer: Tokenizer aligned with ``model``.
        data: Dialogue dataset or plain list of :class:`Dialogue` objects.
        layers: Layer index or indices to record.
        batch_size: Number of sequences per activation batch.
        mask: Optional token mask. ``None`` applies ``dataset.default_mask`` when
            a dataset instance is supplied; pass an explicit mask such as
            ``masks.all()`` to disable masking.
        verbose: Toggle progress reporting.
        add_generation_prompt: Whether to append generation tokens before
            tokenization.
        streaming: When ``True`` yield batches lazily, otherwise load all
            activations into memory.
        **tokenize_kwargs: Extra tokenizer arguments.

    Returns:
        ``Activations`` or ``ActivationIterator`` depending on ``streaming``.
    """
    if isinstance(layers, int):
        layers = [layers]

    # Determine if we're working with a dataset or raw dialogues
    if isinstance(data, DialogueDataset):
        dataset = data
        dialogues = dataset.dialogues
    else:
        # data is a sequence of Dialogue objects
        dataset = None
        dialogues = list(data)  # Ensure it's a list

    # Set up tokenizer
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Prepare tokenization kwargs
    default_tokenize_kwargs = {
        "return_tensors": "pt",
        "padding": True,
    }
    default_tokenize_kwargs.update(tokenize_kwargs)

    # Tokenize all dialogues once (key optimization)
    tokenized_inputs = tokenize_dialogues(
        tokenizer=tokenizer,
        dialogues=dialogues,
        mask=mask,  # Pass mask if provided
        device=model.device,
        dataset=dataset,  # Pass dataset if available for default mask
        add_generation_prompt=add_generation_prompt,
        **default_tokenize_kwargs,
    )

    n_samples = tokenized_inputs["input_ids"].shape[0]
    num_batches = (n_samples + batch_size - 1) // batch_size

    if streaming:
        # Return regenerable iterator that creates fresh generators
        return ActivationIterator(
            model=model,
            tokenizer=tokenizer,
            tokenized_inputs=tokenized_inputs,
            layers=layers,
            batch_size=batch_size,
            verbose=verbose,
            num_batches=num_batches,
        )
    else:
        # Collect all activations at once
        return batch_activations(
            model=model,
            tokenizer=tokenizer,
            tokenized_inputs=tokenized_inputs,
            layers=layers,
            batch_size=batch_size,
            verbose=verbose,
        )
