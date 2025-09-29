"""
Simplified activation collection using generators for cleaner API.

This module provides tools for extracting activations from language models using hooks,
with support for different model architectures and efficient memory management.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    TYPE_CHECKING,
    Any,
    Generator,
    Iterable,
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


class Axis(Enum):
    LAYER = auto()
    BATCH = auto()
    SEQ = auto()
    HIDDEN = auto()


class SequencePooling(Enum):
    """Pooling methods for reducing sequence dimension in probes.

    NONE: No pooling - use token-level features for training
    MEAN: Average pooling over detected tokens
    MAX: Max pooling over detected tokens
    LAST_TOKEN: Use the last detected token
    """

    NONE = "none"
    MEAN = "mean"
    MAX = "max"
    LAST_TOKEN = "last_token"


@dataclass(slots=True)
class LayerMeta:
    indices: tuple[int, ...]


@dataclass(slots=True)
class SequenceMeta:
    attention_mask: torch.Tensor
    detection_mask: torch.Tensor
    input_ids: torch.Tensor


_DEFAULT_AXES: tuple[Axis, ...] = (
    Axis.LAYER,
    Axis.BATCH,
    Axis.SEQ,
    Axis.HIDDEN,
)


def _ensure_canonical_axes(axes: tuple[Axis, ...]) -> tuple[Axis, ...]:
    allowed = tuple(axis for axis in _DEFAULT_AXES if axis in axes)
    if allowed != axes:
        raise ValueError(
            "axes must be an ordered subset of (Axis.LAYER, Axis.BATCH, Axis.SEQ, Axis.HIDDEN)"
        )
    return axes


@dataclass(slots=True)
class Activations:
    """Axis-aware container for activation tensors and metadata."""

    activations: Float[torch.Tensor, "..."]
    axes: tuple[Axis, ...] = _DEFAULT_AXES
    layer_meta: LayerMeta | None = None
    sequence_meta: SequenceMeta | None = None
    batch_indices: torch.Tensor | None = None

    _axis_positions: dict[Axis, int] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        self.axes = _ensure_canonical_axes(self.axes)

        if self.activations.ndim != len(self.axes):
            raise ValueError("Activation tensor rank and axes metadata disagree")

        if not self.activations.is_floating_point():
            self.activations = self.activations.float()

        self._axis_positions = {axis: idx for idx, axis in enumerate(self.axes)}

        self._validate_layer_meta()
        self._validate_sequence_meta()
        self._validate_batch_indices()

    # ------------------------------------------------------------------
    # Basic properties
    # ------------------------------------------------------------------
    @property
    def shape(self) -> torch.Size:
        return self.activations.shape

    @property
    def axis_positions(self) -> dict[Axis, int]:
        return dict(self._axis_positions)

    def has_axis(self, axis: Axis) -> bool:
        return axis in self._axis_positions

    def axis_size(self, axis: Axis) -> int:
        try:
            dim = self._axis_positions[axis]
        except KeyError as exc:
            raise AttributeError(f"Axis {axis.name} has been removed") from exc
        return self.activations.shape[dim]

    @property
    def n_layers(self) -> int:
        return self.axis_size(Axis.LAYER)

    @property
    def batch_size(self) -> int:
        return self.axis_size(Axis.BATCH)

    @property
    def seq_len(self) -> int:
        return self.axis_size(Axis.SEQ)

    @property
    def d_model(self) -> int:
        return self.axis_size(Axis.HIDDEN)

    @property
    def attention_mask(self) -> torch.Tensor | None:
        return None if self.sequence_meta is None else self.sequence_meta.attention_mask

    @property
    def detection_mask(self) -> torch.Tensor | None:
        return None if self.sequence_meta is None else self.sequence_meta.detection_mask

    @property
    def input_ids(self) -> torch.Tensor | None:
        return None if self.sequence_meta is None else self.sequence_meta.input_ids

    @property
    def layer_indices(self) -> list[int]:
        return [] if self.layer_meta is None else list(self.layer_meta.indices)

    @property
    def has_sequences(self) -> bool:
        """Check if sequence dimension exists."""
        return Axis.SEQ in self._axis_positions

    @property
    def has_layers(self) -> bool:
        """Check if layer dimension exists."""
        return Axis.LAYER in self._axis_positions

    @classmethod
    def from_tensor(
        cls,
        activations: torch.Tensor,
        *,
        layer_indices: list[int] | None = None,
        attention_mask: torch.Tensor | None = None,
        detection_mask: torch.Tensor | None = None,
        input_ids: torch.Tensor | None = None,
        batch_indices: torch.Tensor | None = None,
    ) -> "Activations":
        """Create Activations from a tensor with sensible defaults.

        This is a convenience method for simple cases. It automatically:
        - Detects axes from tensor shape
        - Creates default masks if not provided
        - Handles layer metadata

        Args:
            activations: Activation tensor with shape:
                - [batch, seq, hidden] for single layer
                - [layer, batch, seq, hidden] for multiple layers
            layer_indices: Which layers these activations come from.
                          Defaults to [0] for single layer or range for multiple.
            attention_mask: Optional attention mask [batch, seq].
                           Defaults to all 1s.
            detection_mask: Optional detection mask [batch, seq].
                           Defaults to all 1s.
            input_ids: Optional input token IDs [batch, seq].
                      Defaults to all 1s.
            batch_indices: Optional batch indices for streaming.

        Returns:
            Activations object ready to use.

        Examples:
            # Simple case - single layer
            acts = torch.randn(4, 10, 768)  # [batch, seq, hidden]
            activations = Activations.from_tensor(acts)

            # With custom masks
            acts = torch.randn(4, 10, 768)
            mask = torch.ones(4, 10)
            mask[:, 5:] = 0  # Mask out later tokens
            activations = Activations.from_tensor(
                acts,
                attention_mask=mask,
                detection_mask=mask
            )

            # Multiple layers
            acts = torch.randn(12, 4, 10, 768)  # [layer, batch, seq, hidden]
            activations = Activations.from_tensor(acts, layer_indices=list(range(12)))
        """
        # Determine shape and axes
        if activations.ndim == 3:
            # [batch, seq, hidden] - single layer
            batch_size, seq_len, hidden_size = activations.shape
            axes = (Axis.BATCH, Axis.SEQ, Axis.HIDDEN)

            # Add layer dimension
            activations = activations.unsqueeze(0)  # [1, batch, seq, hidden]
            axes = (Axis.LAYER, Axis.BATCH, Axis.SEQ, Axis.HIDDEN)

            if layer_indices is None:
                layer_indices = [0]

        elif activations.ndim == 4:
            # [layer, batch, seq, hidden] - multiple layers
            n_layers, batch_size, seq_len, hidden_size = activations.shape
            axes = (Axis.LAYER, Axis.BATCH, Axis.SEQ, Axis.HIDDEN)

            if layer_indices is None:
                layer_indices = list(range(n_layers))
        else:
            raise ValueError(
                f"Expected 3D [batch, seq, hidden] or 4D [layer, batch, seq, hidden] tensor, "
                f"got shape {activations.shape}"
            )

        # Create default masks if needed
        if attention_mask is None:
            attention_mask = torch.ones(batch_size, seq_len, dtype=torch.float32)

        if detection_mask is None:
            detection_mask = torch.ones(batch_size, seq_len, dtype=torch.float32)

        if input_ids is None:
            input_ids = torch.ones(batch_size, seq_len, dtype=torch.long)

        # Move to same device as activations
        device = activations.device
        attention_mask = attention_mask.to(device)
        detection_mask = detection_mask.to(device)
        input_ids = input_ids.to(device)

        # Create metadata
        layer_meta = LayerMeta(indices=tuple(layer_indices))
        sequence_meta = SequenceMeta(
            attention_mask=attention_mask,
            detection_mask=detection_mask,
            input_ids=input_ids,
        )

        return cls(
            activations=activations,
            axes=axes,
            layer_meta=layer_meta,
            sequence_meta=sequence_meta,
            batch_indices=batch_indices,
        )

    @classmethod
    def from_hidden_states(
        cls,
        hidden_states: tuple[tuple[torch.Tensor, ...], ...] | torch.Tensor,
        *,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        detection_mask: torch.Tensor | None = None,
        layer_indices: list[int] | None = None,
        batch_indices: torch.Tensor | None = None,
    ) -> "Activations":
        """Create Activations from pre-extracted hidden states.

        This is designed for integration with external libraries (like inspect_ai or
        control-arena) that return hidden states from model calls. It handles both
        the nested tuple format from HuggingFace and pre-stacked tensors.

        Args:
            hidden_states: Hidden states in one of two formats:
                - HuggingFace format: tuple[tuple[Tensor, ...], ...] where outer tuple
                  is generation steps, inner tuple is layers. Each tensor is [batch, seq, hidden].
                  Example: ((layer0_step0, layer1_step0), (layer0_step1, layer1_step1))
                - Stacked format: Tensor of shape [layer, batch, seq, hidden]
            input_ids: Token IDs [batch, seq] (optional, defaults to ones)
            attention_mask: Attention mask [batch, seq] (optional, defaults to ones)
            detection_mask: Detection mask [batch, seq] (optional, defaults to ones)
            layer_indices: Which layers these are from (optional, inferred as 0..N-1)
            batch_indices: Batch indices for streaming (optional)

        Returns:
            Activations object ready for probe inference

        Examples:
            # From inspect_ai metadata (nested tuples)
            >>> message = state.messages[-1]  # ChatMessageAssistant
            >>> hidden_states = message.metadata["hidden_states"]
            >>> acts = Activations.from_hidden_states(
            ...     hidden_states,
            ...     input_ids=tokenized["input_ids"],
            ...     attention_mask=tokenized["attention_mask"]
            ... )

            # From pre-stacked tensor
            >>> stacked = torch.randn(32, 1, 128, 4096)  # [layers, batch, seq, hidden]
            >>> acts = Activations.from_hidden_states(stacked)

        Raises:
            ValueError: If hidden_states format is invalid or empty
            TypeError: If hidden_states is not tuple or Tensor
        """
        # Handle nested tuple format (HuggingFace)
        if isinstance(hidden_states, tuple):
            # hidden_states: tuple of tuples
            # Outer tuple = generation steps (one per generated token)
            # Inner tuple = layers (one per model layer)
            # Each tensor = [batch, 1, hidden] for single token generation

            num_steps = len(hidden_states)
            if num_steps == 0:
                raise ValueError("Empty hidden_states tuple")

            num_layers = len(hidden_states[0])
            if num_layers == 0:
                raise ValueError("Empty layer tuple in hidden_states")

            # Verify all steps have same number of layers
            for step_idx, step_tuple in enumerate(hidden_states):
                if len(step_tuple) != num_layers:
                    raise ValueError(
                        f"Inconsistent layer count: step 0 has {num_layers} layers, "
                        f"step {step_idx} has {len(step_tuple)} layers"
                    )

            # Collect all tensors per layer across steps
            layer_tensors = []
            for layer_idx in range(num_layers):
                # Get all tensors for this layer across generation steps
                step_tensors = [
                    hidden_states[step_idx][layer_idx] for step_idx in range(num_steps)
                ]

                # Each step_tensor is [batch, 1, hidden] for single-token generation
                # or [batch, seq_len, hidden] for multi-token
                # Concatenate along sequence dimension
                layer_tensor = torch.cat(step_tensors, dim=1)  # [batch, total_seq, hidden]
                layer_tensors.append(layer_tensor)

            # Stack layers to get [layer, batch, seq, hidden]
            activations_tensor = torch.stack(layer_tensors, dim=0)

        elif isinstance(hidden_states, torch.Tensor):
            activations_tensor = hidden_states
            if activations_tensor.ndim != 4:
                raise ValueError(
                    f"Expected 4D tensor [layer, batch, seq, hidden], "
                    f"got shape {activations_tensor.shape}"
                )
        else:
            raise TypeError(
                f"hidden_states must be tuple or Tensor, got {type(hidden_states)}"
            )

        # Use existing from_tensor to handle the rest
        return cls.from_tensor(
            activations=activations_tensor,
            layer_indices=layer_indices,
            attention_mask=attention_mask,
            detection_mask=detection_mask,
            input_ids=input_ids,
            batch_indices=batch_indices,
        )

    @classmethod
    def from_components(
        cls,
        *,
        activations: torch.Tensor,
        layer_indices: Iterable[int] | list[int],
        attention_mask: torch.Tensor,
        detection_mask: torch.Tensor,
        input_ids: torch.Tensor,
        batch_indices: Iterable[int] | torch.Tensor | None = None,
        axes: tuple[Axis, ...] | None = None,
    ) -> "Activations":
        """Create Activations from explicit components.

        This is an alias for from_tensor() with all parameters explicitly provided.
        It exists for backward compatibility and explicitness when you have all
        components ready.

        Args:
            activations: Activation tensor (should be 4D: [layer, batch, seq, hidden])
            layer_indices: Layer indices for the activations
            attention_mask: Attention mask [batch, seq]
            detection_mask: Detection mask [batch, seq]
            input_ids: Input IDs [batch, seq]
            batch_indices: Optional batch indices for streaming
            axes: Optional custom axes (defaults to standard ordering)

        Returns:
            Activations object
        """
        # Convert to list if needed
        if not isinstance(layer_indices, list):
            layer_indices = list(layer_indices)

        # Use from_tensor which handles all the logic
        return cls.from_tensor(
            activations=activations,
            layer_indices=layer_indices,
            attention_mask=attention_mask,
            detection_mask=detection_mask,
            input_ids=input_ids,
            batch_indices=batch_indices,
        )

    # ------------------------------------------------------------------
    # Validation helpers
    # ------------------------------------------------------------------
    def _validate_layer_meta(self) -> None:
        has_layer_axis = Axis.LAYER in self._axis_positions
        if has_layer_axis:
            if self.layer_meta is None:
                raise ValueError(
                    "layer_meta is required when the layer axis is present"
                )
            indices = tuple(int(i) for i in self.layer_meta.indices)
            if len(indices) != self.axis_size(Axis.LAYER):
                raise ValueError("layer_meta length must match layer dimension")
            self.layer_meta = LayerMeta(indices)
        elif self.layer_meta is not None:
            raise ValueError("layer_meta must be None after reducing the layer axis")

    def _validate_sequence_meta(self) -> None:
        has_seq_axis = Axis.SEQ in self._axis_positions
        if has_seq_axis:
            if self.sequence_meta is None:
                raise ValueError(
                    "sequence_meta is required while the sequence axis is present"
                )
            batch = self.axis_size(Axis.BATCH)
            seq = self.axis_size(Axis.SEQ)

            attn = self.sequence_meta.attention_mask.to(self.activations.device)
            detect = self.sequence_meta.detection_mask.to(self.activations.device)
            ids = self.sequence_meta.input_ids.to(self.activations.device)

            expected = (batch, seq)
            for name, tensor in (
                ("attention_mask", attn),
                ("detection_mask", detect),
                ("input_ids", ids),
            ):
                if tensor.shape != expected:
                    raise ValueError(f"{name} shape {tensor.shape} expected {expected}")

            allowed_detect_dtypes = {
                torch.float16,
                torch.float32,
                torch.float64,
                torch.bfloat16,
                torch.bool,
                torch.int32,
                torch.int64,
            }
            if detect.dtype not in allowed_detect_dtypes:
                raise ValueError("detection_mask must be float, bool, or int tensor")

            if ids.dtype not in (torch.int32, torch.int64):
                raise ValueError("input_ids must be an integer tensor")

            self.sequence_meta = SequenceMeta(
                attention_mask=attn,
                detection_mask=detect,
                input_ids=ids,
            )
        elif self.sequence_meta is not None:
            raise ValueError(
                "sequence_meta must be None after reducing the sequence axis"
            )

    def _validate_batch_indices(self) -> None:
        if self.batch_indices is None:
            return

        tensor = torch.as_tensor(self.batch_indices, dtype=torch.long)
        if tensor.ndim != 1:
            raise ValueError("batch_indices must be one-dimensional")

        if Axis.BATCH not in self._axis_positions:
            raise ValueError("batch_indices provided but batch axis is absent")

        if tensor.numel() != self.axis_size(Axis.BATCH):
            raise ValueError("batch_indices length must match batch dimension")

        self.batch_indices = tensor

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------
    def expect_axes(self, *axes: Axis) -> None:
        missing = tuple(axis for axis in axes if axis not in self._axis_positions)
        if missing:
            missing_names = ", ".join(axis.name for axis in missing)
            present_names = ", ".join(axis.name for axis in self.axes)
            raise ValueError(
                f"Expected axes [{missing_names}] to be present, but they are missing.\n"
                f"Current axes: [{present_names}]\n"
                f"Hint: Axes may have been removed by pooling (e.g., pool(dim='sequence')) "
                f"or layer selection (e.g., select(layers=10))."
            )

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

        self.expect_axes(Axis.LAYER)
        layer_indices = self._require_layer_meta().indices

        tensor_indices: list[int] = []
        for layer in requested_layers:
            try:
                tensor_indices.append(layer_indices.index(layer))
            except ValueError as exc:
                available_layers = ", ".join(map(str, layer_indices))
                raise ValueError(
                    f"Layer {layer} is not available in this Activations object.\n"
                    f"Available layers: [{available_layers}]\n"
                    f"Hint: Use acts.select(layers={list(layer_indices)}) to select "
                    f"available layers, or collect_activations with the desired layers."
                ) from exc
        return tensor_indices

    def _require_layer_meta(self) -> LayerMeta:
        if self.layer_meta is None:
            raise ValueError(
                "Layer metadata is unavailable because the layer axis was removed.\n"
                "Hint: This happens after pooling over layers (e.g., acts.pool(dim='layer')) "
                "or selecting a single layer (e.g., acts.select(layers=10))."
            )
        return self.layer_meta

    def _require_sequence_meta(self) -> SequenceMeta:
        if self.sequence_meta is None:
            raise ValueError(
                "Sequence metadata is unavailable because the sequence axis was removed.\n"
                "Hint: This happens after pooling over sequences (e.g., acts.pool(dim='sequence')). "
                "If you need token-level data, use acts.extract_tokens() before pooling."
            )
        return self.sequence_meta

    def to(self, *args, **kwargs) -> "Activations":
        converted = self.activations.to(*args, **kwargs)

        target_device: torch.device | None = None
        if "device" in kwargs:
            target_device = torch.device(kwargs["device"])
        else:
            for arg in args:
                if isinstance(arg, torch.device):
                    target_device = arg
                    break
                if isinstance(arg, str):
                    target_device = torch.device(arg)
                    break

        sequence_meta = self.sequence_meta
        if target_device is not None and sequence_meta is not None:
            sequence_meta = SequenceMeta(
                attention_mask=sequence_meta.attention_mask.to(target_device),
                detection_mask=sequence_meta.detection_mask.to(target_device),
                input_ids=sequence_meta.input_ids.to(target_device),
            )

        return Activations(
            activations=converted,
            axes=self.axes,
            layer_meta=self.layer_meta,
            sequence_meta=sequence_meta,
            batch_indices=self.batch_indices,
        )

    # ------------------------------------------------------------------
    # Axis transforms
    # ------------------------------------------------------------------

    def select(
        self,
        *,
        layers: int | list[int] | range | None = None,
    ) -> "Activations":
        """
        Unified method for selecting layers with automatic axis handling.

        This is a cleaner API that replaces `select_layer()` and `select_layers()`
        with a single method that automatically determines whether to keep or
        remove the layer axis based on input.

        Args:
            layers: Layer(s) to select:
                - int: Select single layer, remove LAYER axis
                - list[int]: Select multiple layers, keep LAYER axis
                - range: Select range of layers, keep LAYER axis
                - None: No layer selection (returns self)

        Returns:
            New Activations with selected layers

        Examples:
            # Select single layer (removes layer axis)
            >>> acts.select(layers=10)

            # Select multiple layers (keeps layer axis)
            >>> acts.select(layers=[10, 15, 20])

            # Select range of layers
            >>> acts.select(layers=range(10, 20))

        Raises:
            ValueError: If requested layers are not available

        Note:
            For backwards compatibility, `select_layer(i)` and
            `select_layers([i, j])` are still available.
        """
        if layers is None:
            return self

        # Ensure we have layer axis
        self.expect_axes(Axis.LAYER)

        # Handle range objects
        if isinstance(layers, range):
            layers = list(layers)

        # Single layer - remove axis
        if isinstance(layers, int):
            tensor_idx = self.get_layer_tensor_indices([layers])[0]
            dim = self._axis_positions[Axis.LAYER]
            selected = self.activations.select(dim, tensor_idx)

            new_axes = tuple(ax for ax in self.axes if ax != Axis.LAYER)
            return Activations(
                activations=selected,
                axes=new_axes,
                layer_meta=None,
                sequence_meta=self.sequence_meta,
                batch_indices=self.batch_indices,
            )

        # Multiple layers - keep axis
        if not layers:
            raise ValueError(
                f"layers must be non-empty. Available layers: {self.layer_indices}"
            )

        tensor_indices = self.get_layer_tensor_indices(layers)
        dim = self._axis_positions[Axis.LAYER]
        index = torch.as_tensor(
            tensor_indices, device=self.activations.device, dtype=torch.long
        )
        subset = torch.index_select(self.activations, dim=dim, index=index)

        new_meta = LayerMeta(tuple(layers))
        return Activations(
            activations=subset,
            axes=self.axes,
            layer_meta=new_meta,
            sequence_meta=self.sequence_meta,
            batch_indices=self.batch_indices,
        )

    def pool(
        self,
        dim: Literal["sequence", "seq", "layer"] = "sequence",
        method: Literal["mean", "max", "last_token"] = "mean",
        use_detection_mask: bool = True,
    ) -> "Activations":
        """
        Pool over a specified dimension, removing that axis.

        This is a unified API for dimension reduction that works across different axes.
        Replaces the older `sequence_pool()` and `aggregate()` methods with a more
        consistent interface.

        Args:
            dim: Dimension to pool over. Options:
                - "sequence" or "seq": Pool over sequence dimension
                - "layer": Pool over layer dimension
            method: Pooling method to use:
                - "mean": Average pooling
                - "max": Max pooling
                - "last_token": Use last valid token (sequence dim only)
            use_detection_mask: If True, only pool over detected tokens (default).
                               Only applies to sequence dimension.

        Returns:
            New Activations without the pooled dimension

        Examples:
            # Pool over sequence dimension (most common)
            >>> pooled = acts.pool(dim="sequence", method="mean")

            # Pool over layer dimension to get layer-averaged features
            >>> pooled = acts.pool(dim="layer", method="mean")

            # Pool without detection mask (use all tokens)
            >>> pooled = acts.pool(dim="sequence", use_detection_mask=False)

        Note:
            For backwards compatibility, `sequence_pool()` and `aggregate()` are
            still available but may be deprecated in future versions.
        """
        # Normalize dimension name
        if dim in ("sequence", "seq"):
            axis = Axis.SEQ
            self.expect_axes(Axis.SEQ)
        elif dim == "layer":
            axis = Axis.LAYER
            self.expect_axes(Axis.LAYER)
        else:
            raise ValueError(
                f"Unknown dimension: {dim}. Supported: 'sequence', 'seq', 'layer'"
            )

        # Handle sequence pooling
        if axis == Axis.SEQ:
            if use_detection_mask:
                # Use existing _reduce_sequence which respects detection mask
                reduced = self._reduce_sequence(method)
            else:
                # Simple pooling over all tokens
                dim_idx = self._axis_positions[Axis.SEQ]
                if method == "mean":
                    reduced = self.activations.mean(dim=dim_idx)
                elif method == "max":
                    reduced = self.activations.max(dim=dim_idx).values
                elif method == "last_token":
                    # Take the last token (index -1) for each sequence
                    reduced = self.activations.select(dim_idx, -1)
                else:
                    raise ValueError(
                        f"Unknown pooling method: {method}. "
                        f"Supported: 'mean', 'max', 'last_token'"
                    )

            new_axes = tuple(ax for ax in self.axes if ax != Axis.SEQ)
            return Activations(
                activations=reduced,
                axes=new_axes,
                layer_meta=self.layer_meta,
                sequence_meta=None,
                batch_indices=self.batch_indices,
            )

        # Handle layer pooling
        elif axis == Axis.LAYER:
            if method == "last_token":
                raise ValueError(
                    "'last_token' pooling is only supported for sequence dimension"
                )

            dim_idx = self._axis_positions[Axis.LAYER]
            if method == "mean":
                reduced = self.activations.mean(dim=dim_idx)
            elif method == "max":
                reduced = self.activations.max(dim=dim_idx).values
            else:
                raise ValueError(
                    f"Unknown pooling method: {method}. "
                    f"Supported for layer dimension: 'mean', 'max'"
                )

            new_axes = tuple(ax for ax in self.axes if ax != Axis.LAYER)
            return Activations(
                activations=reduced,
                axes=new_axes,
                layer_meta=None,
                sequence_meta=self.sequence_meta,
                batch_indices=self.batch_indices,
            )

    def extract_tokens(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Extract detected tokens for token-level training.

        Returns:
            Tuple of (features, tokens_per_sample) where:
            - features: [n_tokens, hidden] tensor of detected tokens
            - tokens_per_sample: [batch] tensor of token counts per sample
        """
        self.expect_axes(Axis.BATCH, Axis.SEQ, Axis.HIDDEN)

        if self.has_axis(Axis.LAYER):
            if self.n_layers != 1:
                raise ValueError(
                    f"Token extraction requires single layer, but found {self.n_layers} layers.\n"
                    f"Available layers: {self.layer_indices}\n"
                    f"Hint: Use acts.select(layers=i) to select a single layer before extracting tokens."
                )
            # Remove layer dimension
            acts = self.activations.squeeze(self._axis_positions[Axis.LAYER])
        else:
            acts = self.activations

        seq_meta = self._require_sequence_meta()
        mask = seq_meta.detection_mask.bool()
        tokens_per_sample = mask.sum(dim=1)

        if tokens_per_sample.sum() == 0:
            features = torch.empty(
                0,
                acts.shape[-1],
                device=acts.device,
                dtype=acts.dtype,
            )
        else:
            features = acts[mask]

        return features, tokens_per_sample.to(device=acts.device)

    # ------------------------------------------------------------------
    # Internal sequence reduction
    # ------------------------------------------------------------------
    def _reduce_sequence(
        self, method: Literal["mean", "max", "last_token"]
    ) -> torch.Tensor:
        if method not in {"mean", "max", "last_token"}:
            raise ValueError(f"Unknown sequence reduction method: {method}")

        seq_meta = self._require_sequence_meta()
        detection = seq_meta.detection_mask
        mask_bool = detection.bool()
        mask_float = mask_bool.to(dtype=self.activations.dtype)

        batch_dim = self._axis_positions[Axis.BATCH]
        seq_dim = self._axis_positions[Axis.SEQ]
        rank = self.activations.ndim

        other_dims = [idx for idx in range(rank) if idx not in (batch_dim, seq_dim)]
        permute = [batch_dim, seq_dim] + other_dims
        acts = self.activations.permute(permute)

        mask_view = mask_float.view(
            mask_float.shape[0],
            mask_float.shape[1],
            *([1] * (acts.ndim - 2)),
        )

        if method == "mean":
            masked = acts * mask_view
            counts = mask_view.sum(dim=1).clamp_min(1.0)
            reduced = masked.sum(dim=1) / counts
        elif method == "max":
            mask_bool_view = mask_bool.view(
                mask_bool.shape[0],
                mask_bool.shape[1],
                *([1] * (acts.ndim - 2)),
            )
            masked = acts.masked_fill(~mask_bool_view, float("-inf"))
            reduced = masked.max(dim=1).values

            no_valid = ~mask_bool.any(dim=1)
            if no_valid.any():
                reduced = reduced.clone()
                no_valid_view = no_valid.view(
                    no_valid.shape[0],
                    *([1] * (reduced.ndim - 1)),
                ).expand_as(reduced)
                reduced.masked_fill_(no_valid_view, 0.0)
        else:
            valid_counts = mask_bool.sum(dim=1)
            no_valid = valid_counts == 0
            last_indices = torch.clamp(valid_counts - 1, min=0).to(dtype=torch.long)

            gather_index = last_indices.view(
                mask_bool.shape[0],
                1,
                *([1] * (acts.ndim - 2)),
            ).expand(mask_bool.shape[0], 1, *acts.shape[2:])

            reduced = torch.take_along_dim(acts, gather_index, dim=1).squeeze(1)
            if no_valid.any():
                reduced = reduced.clone()
                no_valid_view = no_valid.view(
                    no_valid.shape[0],
                    *([1] * (reduced.ndim - 1)),
                ).expand_as(reduced)
                reduced.masked_fill_(no_valid_view, 0.0)

        remaining_axes = tuple(axis for axis in self.axes if axis not in (Axis.SEQ,))
        permuted_axes = (Axis.BATCH,) + tuple(
            axis for axis in self.axes if axis not in (Axis.BATCH, Axis.SEQ)
        )

        permute_back = [permuted_axes.index(axis) for axis in remaining_axes]
        if permute_back != list(range(len(remaining_axes))):
            reduced = reduced.permute(permute_back)

        return reduced


def streaming_activations(
    model: "PreTrainedModel",
    tokenizer: "PreTrainedTokenizerBase",
    tokenized_inputs: dict[str, torch.Tensor],
    layers: list[int],
    batch_size: int = 8,
    verbose: bool = False,
    hook_point: Literal["pre_layernorm", "post_block"] = "post_block",
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
        hook_point: Where to extract activations:
            - "post_block": After transformer block (aligns with HF hidden_states)
            - "pre_layernorm": Before layer normalization (legacy behavior)

    Yields:
        Activations objects for each batch
    """
    n_samples = tokenized_inputs["input_ids"].shape[0]

    # Use HookedModel context for all batches (preserves buffer reuse optimization)
    with HookedModel(model, layers, hook_point=hook_point) as hooked_model:
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
                axes=_DEFAULT_AXES,
                layer_meta=LayerMeta(tuple(layers)),
                sequence_meta=SequenceMeta(
                    attention_mask=batch_inputs["attention_mask"],
                    detection_mask=batch_inputs["detection_mask"],
                    input_ids=batch_inputs["input_ids"],
                ),
                batch_indices=torch.as_tensor(batch_indices, dtype=torch.long),
            )


def batch_activations(
    model: "PreTrainedModel",
    tokenizer: "PreTrainedTokenizerBase",
    tokenized_inputs: dict[str, torch.Tensor],
    layers: list[int],
    batch_size: int = 8,
    verbose: bool = False,
    hook_point: Literal["pre_layernorm", "post_block"] = "post_block",
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
        hook_point: Where to extract activations:
            - "post_block": After transformer block (aligns with HF hidden_states)
            - "pre_layernorm": Before layer normalization (legacy behavior)

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

        with HookedModel(model, layers, hook_point=hook_point) as hooked_model:
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

        with HookedModel(model, layers, hook_point=hook_point) as hooked_model:
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
        axes=_DEFAULT_AXES,
        layer_meta=LayerMeta(tuple(layers)),
        sequence_meta=SequenceMeta(
            attention_mask=tokenized_inputs["attention_mask"],
            detection_mask=tokenized_inputs["detection_mask"],
            input_ids=tokenized_inputs["input_ids"],
        ),
        batch_indices=torch.arange(n_samples, dtype=torch.long),
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
        hook_point: Literal["pre_layernorm", "post_block"] = "post_block",
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
            hook_point: Where to extract activations
        """
        self._model = model
        self._tokenizer = tokenizer
        self._tokenized_inputs = tokenized_inputs
        self._layers = layers
        self._batch_size = batch_size
        self._verbose = verbose
        self._num_batches = num_batches
        self._hook_point = hook_point

    def __iter__(self) -> Iterator[Activations]:
        """Create and return a fresh generator for activation batches."""
        return streaming_activations(
            model=self._model,
            tokenizer=self._tokenizer,
            tokenized_inputs=self._tokenized_inputs,
            layers=self._layers,
            batch_size=self._batch_size,
            verbose=self._verbose,
            hook_point=self._hook_point,
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
    hook_point: Literal["pre_layernorm", "post_block"] = "post_block",
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
    hook_point: Literal["pre_layernorm", "post_block"] = "post_block",
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
    hook_point: Literal["pre_layernorm", "post_block"] = "post_block",
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
        hook_point: Where to extract activations from the model:
            - "post_block" (default): After transformer block output.
              Aligns with HuggingFace hidden_states semantics.
            - "pre_layernorm": Before layer normalization (legacy behavior).
              Captures pre-normalized representations.
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
            hook_point=hook_point,
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
            hook_point=hook_point,
        )
