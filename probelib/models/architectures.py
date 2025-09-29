import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from ..logger import logger

if TYPE_CHECKING:
    from transformers import PreTrainedModel


@dataclass
class TokenPadding:
    """Token padding configuration for role masks."""

    left: int  # Tokens before message content
    right: int  # Tokens after message content


class ModelArchitecture(ABC):
    """Base class for handling different model architectures."""

    @abstractmethod
    def get_layer_norm(
        self, model: "PreTrainedModel", layer_idx: int
    ) -> torch.nn.Module:
        """Get the layer normalization module for a specific layer."""
        pass

    @abstractmethod
    def get_layer_module(
        self, model: "PreTrainedModel", layer_idx: int
    ) -> torch.nn.Module:
        """Get the full transformer block module for a specific layer.

        This is used for post-block activation capture that aligns with
        Hugging Face hidden_states semantics.
        """
        pass

    @abstractmethod
    def get_layers(self, model: "PreTrainedModel") -> list[torch.nn.Module]:
        """Get all layers of the model."""
        pass

    @abstractmethod
    def set_layers(
        self, model: "PreTrainedModel", layers: list[torch.nn.Module]
    ) -> None:
        """Set the model's layers."""
        pass

    @abstractmethod
    def get_token_padding(self) -> TokenPadding:
        """Get token padding configuration for this architecture.

        Returns:
            TokenPadding configuration for the architecture
        """
        pass

    @abstractmethod
    def get_prefix_pattern(self) -> re.Pattern[str]:
        """Get regex pattern for matching chat template tokens.

        This pattern is used to accurately locate message boundaries in tokenized text.

        Returns:
            Compiled regex pattern for this architecture
        """
        pass

    @abstractmethod
    def should_fold_system_messages(self) -> bool:
        """Whether system messages should be folded into user messages.

        Some models (like Gemma) don't support system messages and require
        folding them into the first user message.

        Returns:
            True if system messages should be folded, False otherwise
        """
        pass

    @abstractmethod
    def get_num_layers(self, model: "PreTrainedModel") -> int:
        """Get the number of layers in the model.

        Args:
            model: The model to get layer count from

        Returns:
            Number of layers in the model
        """
        pass


class LlamaArchitecture(ModelArchitecture):
    """Architecture handler for LLaMA-style models."""

    def get_layer_norm(
        self, model: "PreTrainedModel", layer_idx: int
    ) -> torch.nn.Module:
        return model.model.layers[layer_idx].input_layernorm  # type: ignore

    def get_layer_module(
        self, model: "PreTrainedModel", layer_idx: int
    ) -> torch.nn.Module:
        return model.model.layers[layer_idx]  # type: ignore

    def get_layers(self, model: "PreTrainedModel") -> list[torch.nn.Module]:
        return model.model.layers  # type: ignore

    def set_layers(
        self, model: "PreTrainedModel", layers: list[torch.nn.Module]
    ) -> None:
        model.model.layers = layers  # type: ignore

    def get_token_padding(self) -> TokenPadding:
        return TokenPadding(left=4, right=1)

    def get_prefix_pattern(self) -> re.Pattern[str]:
        """Get LLaMA-specific chat template regex pattern."""
        begin_of_text = r"((<\|pad\|>)*(<\|begin_of_text\|>))?"
        header = r"<\|start_header_id\|>(system|user|assistant)<\|end_header_id\|>\n\n"
        end_of_turn = r"<\|eot_id\|>"
        # llama3 system prompt has some boilerplate which we keep constant across all prompts
        date_info = r"(Cutting Knowledge Date: December 2023\nToday Date: \d\d \w\w\w 202[45]\n\n)?"
        return re.compile(
            rf"({begin_of_text}{header}{date_info})?({end_of_turn}{header})?(\n\n)?"
        )

    def should_fold_system_messages(self) -> bool:
        return False

    def get_num_layers(self, model: "PreTrainedModel") -> int:
        """Get the number of layers for LLaMA models."""
        config = model.config
        if hasattr(config, "num_hidden_layers"):
            return config.num_hidden_layers  # type: ignore
        elif hasattr(config, "n_layers"):
            return config.n_layers  # type: ignore
        elif hasattr(config, "num_layers"):
            return config.num_layers  # type: ignore
        else:
            raise ValueError(
                f"Cannot determine number of layers for LLaMA model: {model}"
            )


class GemmaArchitecture(ModelArchitecture):
    """Architecture handler for Gemma models."""

    def get_layer_norm(
        self, model: "PreTrainedModel", layer_idx: int
    ) -> torch.nn.Module:
        return model.language_model.model.layers[layer_idx].input_layernorm  # type: ignore

    def get_layer_module(
        self, model: "PreTrainedModel", layer_idx: int
    ) -> torch.nn.Module:
        return model.language_model.model.layers[layer_idx]  # type: ignore

    def get_layers(self, model: "PreTrainedModel") -> list[torch.nn.Module]:
        return model.language_model.model.layers  # type: ignore

    def set_layers(
        self, model: "PreTrainedModel", layers: list[torch.nn.Module]
    ) -> None:
        model.language_model.model.layers = layers  # type: ignore

    def get_token_padding(self) -> TokenPadding:
        return TokenPadding(left=3, right=2)

    def get_prefix_pattern(self) -> re.Pattern[str]:
        """Get Gemma-specific chat template regex pattern."""
        begin_of_text = r"(<pad>)*(<bos>)?"
        end_of_last = r"(<end_of_turn>\n)?"
        start_of_turn = r"<start_of_turn>(user|model)\n"
        return re.compile(
            rf"{begin_of_text}{start_of_turn}|{end_of_last}{start_of_turn}|(\n\n)?"
        )

    def should_fold_system_messages(self) -> bool:
        return True  # Gemma doesn't support system messages

    def get_num_layers(self, model: "PreTrainedModel") -> int:
        """Get the number of layers for Gemma models."""
        # For Gemma models, need to access through language_model attribute
        base_model = model.language_model if hasattr(model, "language_model") else model
        config = base_model.config

        if hasattr(config, "num_hidden_layers"):
            return config.num_hidden_layers  # type: ignore
        elif hasattr(config, "n_layers"):
            return config.n_layers  # type: ignore
        elif hasattr(config, "num_layers"):
            return config.num_layers  # type: ignore
        else:
            raise ValueError(
                f"Cannot determine number of layers for Gemma model: {model}"
            )


class ArchitectureRegistry:
    """Registry for mapping model types to their architecture handlers."""

    _architectures: dict[str, type[ModelArchitecture]] = {
        "llama": LlamaArchitecture,
        "gemma": GemmaArchitecture,
    }

    @classmethod
    def get_architecture_by_name(cls, name: str) -> ModelArchitecture:
        """Get architecture handler by name.

        Args:
            name: Architecture name (e.g., 'llama', 'gemma')

        Returns:
            ModelArchitecture instance for the specified name

        Raises:
            ValueError: If architecture name is not supported
        """
        arch_class = cls._architectures.get(name)
        if arch_class is None:
            logger.error("Unsupported architecture requested: %s", name)
            raise ValueError(f"Unsupported architecture: {name}")
        return arch_class()

    @classmethod
    def detect_from_tokenizer_name(cls, tokenizer_name: str) -> str:
        """Detect model family from tokenizer name.

        Args:
            tokenizer_name: Name or path of the tokenizer

        Returns:
            Model family string ("llama", "gemma", etc.)
        """
        name_lower = tokenizer_name.lower()

        if "llama" in name_lower:
            return "llama"
        elif "gemma" in name_lower:
            return "gemma"
        else:
            supported = ", ".join(cls._architectures.keys())
            logger.error(
                "Could not detect architecture from tokenizer name: %s",
                tokenizer_name,
            )
            raise ValueError(
                f"Unable to detect architecture for tokenizer '{tokenizer_name}'.\n"
                f"Supported architectures: {supported}\n"
                f"Hint: Tokenizer name should contain one of: {supported}"
            )

    @classmethod
    def get_architecture(cls, model: "PreTrainedModel") -> ModelArchitecture:
        """Detect and return the appropriate architecture handler."""
        # Handle PEFT models by unwrapping to get the base model
        base_model = model
        if hasattr(model, "get_base_model"):
            # PEFT models have a get_base_model() method
            base_model = model.get_base_model()  # type: ignore
            logger.debug(f"Unwrapped PEFT model to base model: {type(base_model)}")

        for _, arch_class in cls._architectures.items():
            try:
                arch = arch_class()
                # Test if this architecture matches by trying to access a layer
                # Use base_model which is either the unwrapped PEFT model or the original model
                arch.get_layer_norm(base_model, 0)
                logger.debug(f"Detected architecture: {arch_class.__name__}")
                return arch
            except (AttributeError, IndexError, TypeError):
                continue

        supported = ", ".join(cls._architectures.keys())
        logger.error(
            f"Unsupported model architecture: {type(base_model)} (original: {type(model)})"
        )
        raise ValueError(
            f"Unsupported model architecture: {type(base_model).__name__}\n"
            f"Supported architectures: {supported}\n"
            f"Hint: The model type must be one of: LLaMA-style (LlamaForCausalLM) "
            f"or Gemma-style (GemmaForCausalLM). "
            f"If you have a compatible model, please open an issue on GitHub."
        )
