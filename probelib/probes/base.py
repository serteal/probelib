"""Base class for all probes with unified interface."""

from abc import ABC, abstractmethod
from pathlib import Path
import torch

from ..processing.activations import (
    ActivationIterator,
    Activations,
    Axis,
    SequencePooling,
)


class BaseProbe(ABC):
    """Shared lifecycle for probes used throughout probelib.

    Probes load activations for a *single* model layer, optionally aggregate over
    tokens, and perform binary classification. Subclasses directly implement training
    and prediction logic while inheriting utility methods for feature preparation,
    device placement, and sequence pooling.

    Key design:
    - ``sequence_pooling`` controls how sequences are handled (token-level vs pooled)
    - ``_prepare_features`` handles layer selection and pooling
    - Direct implementation of ``fit``, ``partial_fit``, ``predict_proba`` by subclasses
    - Automatic label expansion for token-level training when ``sequence_pooling=NONE``
    """

    def __init__(
        self,
        layer: int | None = None,
        sequence_pooling: SequencePooling = SequencePooling.MEAN,
        device: str | None = None,
        random_state: int | None = None,
        verbose: bool = False,
    ):
        """
        Initialize base probe.

        Args:
            layer: Optional layer index. If provided, auto-selects this layer from activations.
            sequence_pooling: How to pool sequences before training.
                            NONE for token-level training.
            device: Device for computation (auto-detected if None)
            random_state: Random seed for reproducibility
            verbose: Whether to print progress information
        """
        self.layer = layer
        self.sequence_pooling = sequence_pooling

        # Set device - auto-detect if None
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.random_state = random_state
        self.verbose = verbose

        # Track fitting state
        self._fitted = False
        self._requires_grad = False  # Set by subclasses if they need gradients
        self._tokens_per_sample = None  # Set when sequence_pooling=NONE

    @property
    def requires_sequences(self) -> bool:
        """Whether probe needs full sequences."""
        return self.sequence_pooling == SequencePooling.NONE

    @property
    def preferred_pooling(self) -> str | None:
        """Preferred pooling method for memory efficiency."""
        if self.sequence_pooling == SequencePooling.NONE:
            return None
        return self.sequence_pooling.value

    def _prepare_features(
        self, acts: Activations
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare features from Activations (works with all storage formats).

        Returns:
            torch.Tensor: For most probes (features only)
            tuple[torch.Tensor, torch.Tensor]: For Attention probe (sequences, mask)

        Note: Attention probe overrides this to return (sequences, detection_mask).
        Base implementation returns features only.
        """

        # Regular Activations (dense or pooled)
        if self.layer is not None:
            if acts.has_axis(Axis.LAYER):
                acts = acts.select(layers=self.layer)
        elif acts.has_axis(Axis.LAYER) and acts.n_layers != 1:
            raise ValueError(f"Expected single layer, got {acts.n_layers}")

        # Sequence handling
        if self.sequence_pooling == SequencePooling.NONE:
            # Token-level training
            if not acts.has_axis(Axis.SEQ):
                raise ValueError("Token-level training requires sequences")
            features, self._tokens_per_sample = acts.extract_tokens()
            return features
        else:
            # Sequence-level training
            if acts.has_axis(Axis.SEQ):
                acts = acts.pool(dim="sequence", method=self.sequence_pooling.value)

            features = acts.activations
            if acts.has_axis(Axis.LAYER):
                features = features.squeeze(acts._axis_positions[Axis.LAYER])

            self._tokens_per_sample = None
            return features

    def _prepare_labels(
        self, y: list | torch.Tensor, expand_for_tokens: bool = False
    ) -> torch.Tensor:
        """
        Convert labels to tensor and optionally expand for token-level training.

        Args:
            y: List of labels (Label enum or int)
            expand_for_tokens: If True, expand labels to match token count

        Returns:
            Label tensor
        """
        if isinstance(y, torch.Tensor):
            labels = y
        elif hasattr(y[0], "value"):
            # Handle Label enum
            labels = torch.tensor([label.value for label in y])
        else:
            labels = torch.tensor(y)

        # Validate binary classification
        unique_labels = labels.unique()
        if not torch.all((unique_labels == 0) | (unique_labels == 1)):
            raise ValueError(
                f"Only binary classification is supported. "
                f"Expected labels in [0, 1], got {unique_labels.tolist()}"
            )

        # Expand for token-level if needed
        if expand_for_tokens and self._tokens_per_sample is not None:
            labels = torch.repeat_interleave(
                labels.to(self.device), self._tokens_per_sample.to(self.device)
            )

        return labels.to(self.device)

    @staticmethod
    def aggregate_token_predictions(
        predictions: torch.Tensor,
        tokens_per_sample: torch.Tensor,
        method: str = "mean"
    ) -> torch.Tensor:
        """
        Helper to aggregate token-level predictions to sample-level.

        This is a utility method for when you have token-level predictions
        and want to aggregate them to sample level.

        Args:
            predictions: Token-level predictions [n_tokens] or [n_tokens, n_classes]
            tokens_per_sample: Token counts per sample
            method: Aggregation method ("mean", "max", "last_token")

        Returns:
            Sample-level predictions [n_samples] or [n_samples, n_classes]

        Example:
            # Train on tokens
            probe.fit(acts_with_sequences, labels)

            # Get token predictions and counts
            token_preds = probe.predict_proba(test_acts)  # [n_tokens, 2]
            _, counts = test_acts.extract_tokens()

            # Aggregate to samples
            sample_preds = BaseProbe.aggregate_token_predictions(
                token_preds, counts
            )  # [n_samples, 2]
        """
        split_sizes = tokens_per_sample.tolist()

        aggregated = []
        offset = 0
        for size in split_sizes:
            if size == 0:
                if predictions.dim() == 1:
                    zero_value = torch.zeros(
                        (), device=predictions.device, dtype=predictions.dtype
                    )
                else:
                    zero_value = torch.zeros(
                        predictions.shape[1], device=predictions.device, dtype=predictions.dtype
                    )
                aggregated.append(zero_value)
                offset += size
                continue

            if predictions.dim() == 1:
                sample_pred = predictions[offset : offset + size]
            else:
                sample_pred = predictions[offset : offset + size, :]

            if method == "mean":
                aggregated.append(sample_pred.mean(dim=0))
            elif method == "max":
                aggregated.append(
                    sample_pred.max(dim=0).values
                    if sample_pred.dim() > 1
                    else sample_pred.max()
                )
            elif method == "last_token":
                aggregated.append(sample_pred[-1])
            else:
                raise ValueError(f"Unknown aggregation method: {method}")

            offset += size

        # Stack results
        if predictions.dim() == 1:
            return torch.stack(aggregated)
        else:
            return torch.stack(aggregated, dim=0)

    # Abstract methods that subclasses must implement
    @abstractmethod
    def fit(
        self,
        X: Activations | ActivationIterator,
        y: list | torch.Tensor,
    ) -> "BaseProbe":
        """
        Fit the probe on activations and labels.

        Args:
            X: Activations or ActivationIterator containing features
            y: List of labels (Label enum or int)

        Returns:
            self: The fitted probe instance
        """
        pass

    @abstractmethod
    def partial_fit(
        self,
        X: Activations,
        y: list | torch.Tensor,
    ) -> "BaseProbe":
        """
        Incrementally fit the probe on a batch of activations.

        Args:
            X: Activations containing features
            y: List of labels for this batch

        Returns:
            self: The partially fitted probe instance
        """
        pass

    @abstractmethod
    def predict_proba(
        self,
        X: Activations | ActivationIterator,
    ) -> torch.Tensor:
        """
        Predict class probabilities.

        Returns predictions at the same granularity as training:
        - If sequence_pooling=NONE: returns [n_tokens, 2]
        - Otherwise: returns [n_samples, 2]

        Args:
            X: Activations or ActivationIterator containing features

        Returns:
            Tensor of probabilities for each class
        """
        pass

    # Utility methods provided by the base class
    def predict(
        self,
        X: Activations | ActivationIterator,
    ) -> torch.Tensor:
        """
        Predict class labels.

        Args:
            X: Activations or ActivationIterator containing features

        Returns:
            Tensor of predicted class labels (0 or 1)
        """
        probs = self.predict_proba(X)
        return (probs[..., 1] > 0.5).long()

    def score(
        self,
        X: Activations | ActivationIterator,
        y: list | torch.Tensor,
    ) -> float:
        """
        Return the mean accuracy on the given test data and labels.

        Args:
            X: Activations or ActivationIterator containing features
            y: True labels

        Returns:
            Accuracy as a float
        """
        preds = self.predict(X)

        # Convert labels to tensor
        if isinstance(y, torch.Tensor):
            y_true = y
        elif hasattr(y[0], "value"):
            # Handle Label enum
            y_true = torch.tensor([label.value for label in y])
        else:
            y_true = torch.tensor(y)

        # Expand labels if needed for token-level
        if self.sequence_pooling == SequencePooling.NONE and self._tokens_per_sample is not None:
            y_true = torch.repeat_interleave(
                y_true.to(self.device), self._tokens_per_sample.to(self.device)
            )

        return (preds == y_true.to(preds.device)).float().mean().item()

    @abstractmethod
    def save(self, path: Path | str) -> None:
        """
        Save the probe to disk.

        Args:
            path: File path to save the probe
        """
        pass

    @classmethod
    @abstractmethod
    def load(cls, path: Path | str, device: str | None = None) -> "BaseProbe":
        """
        Load a probe from disk.

        Args:
            path: File path to load the probe from
            device: Device to load the probe onto

        Returns:
            Loaded probe instance
        """
        pass

    def __repr__(self) -> str:
        """String representation of the probe."""
        fitted_str = "fitted" if self._fitted else "not fitted"
        layer_str = f"layer={self.layer}, " if self.layer is not None else ""
        pooling_str = f"sequence_pooling={self.sequence_pooling.name}, "
        return f"{self.__class__.__name__}({layer_str}{pooling_str}{fitted_str})"
