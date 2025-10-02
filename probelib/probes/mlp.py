"""Multi-layer perceptron probe implementation."""

from pathlib import Path
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW

from ..processing.activations import ActivationIterator, Activations, SequencePooling
from .base import BaseProbe


class _MLPNetwork(nn.Module):
    """Simple MLP architecture for binary classification (internal implementation).

    Architecture: input -> hidden -> ReLU/GELU -> dropout (optional) -> output
    """

    def __init__(
        self,
        d_model: int,
        hidden_dim: int = 128,
        dropout: float | None = None,
        activation: Literal["relu", "gelu"] = "relu",
    ):
        super().__init__()
        self.fc1 = nn.Linear(d_model, hidden_dim)
        self.activation = nn.ReLU() if activation == "relu" else nn.GELU()
        self.dropout = nn.Dropout(dropout) if dropout is not None else None
        self.fc2 = nn.Linear(hidden_dim, 1)  # Binary classification

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.activation(x)
        if self.dropout is not None:
            x = self.dropout(x)
        x = self.fc2(x)
        return x.squeeze(-1)  # Return logits [batch_size]


class MLP(BaseProbe):
    """Multi-layer perceptron probe for binary classification.

    This probe uses a simple MLP with one hidden layer. It supports
    both batch training and streaming updates via partial_fit.
    """

    def __init__(
        self,
        layer: int | None = None,
        sequence_pooling: SequencePooling = SequencePooling.MEAN,
        hidden_dim: int = 128,
        dropout: float | None = None,
        activation: Literal["relu", "gelu"] = "relu",
        learning_rate: float = 0.001,
        weight_decay: float = 0.01,
        n_epochs: int = 100,
        batch_size: int = 32,
        device: str | None = None,
        random_state: int | None = None,
        verbose: bool = False,
    ):
        """Initialize MLP probe.

        Args:
            layer: Optional layer index for train_probes compatibility
            sequence_pooling: How to pool sequences before training
            hidden_dim: Number of hidden units
            dropout: Dropout rate (None for no dropout)
            activation: Activation function ("relu" or "gelu")
            learning_rate: Learning rate for optimizer
            weight_decay: L2 regularization strength
            n_epochs: Maximum number of training epochs
            batch_size: Batch size for training
            device: Device for computation
            random_state: Random seed for reproducibility
            verbose: Whether to print progress information
        """
        super().__init__(
            layer=layer,
            sequence_pooling=sequence_pooling,
            device=device,
            random_state=random_state,
            verbose=verbose,
        )

        # Architecture parameters
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.activation = activation

        # Training parameters
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.n_epochs = n_epochs
        self.batch_size = batch_size

        # Model components (initialized during fit)
        self._network = None
        self._optimizer = None
        self._d_model = None
        self._streaming_steps = 0

        # This probe requires gradients
        self._requires_grad = True

        # Set random seed
        if random_state is not None:
            torch.manual_seed(random_state)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(random_state)

    def _init_network(self, d_model: int, dtype: torch.dtype | None = None):
        """Initialize the network and optimizer once we know the input dimension."""
        self._d_model = d_model
        self._network = _MLPNetwork(
            d_model=d_model,
            hidden_dim=self.hidden_dim,
            dropout=self.dropout,
            activation=self.activation,
        ).to(self.device)

        # Always use float32 for MLP weights to avoid numerical issues
        # with float16/bfloat16 models that have extreme activation values
        self._network = self._network.to(torch.float32)

        self._optimizer = AdamW(
            self._network.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

    def fit(self, X: Activations | ActivationIterator, y: list | torch.Tensor) -> "MLP":
        """Fit the probe on training data.

        Args:
            X: Activations or ActivationIterator containing features
            y: Labels for training

        Returns:
            self: Fitted probe instance
        """
        if isinstance(X, ActivationIterator):
            # Use streaming for iterators
            labels = self._prepare_labels(y)
            for batch_acts in X:
                batch_idx = torch.tensor(batch_acts.batch_indices, device=labels.device, dtype=torch.long)
                batch_labels = labels.index_select(0, batch_idx)
                self.partial_fit(batch_acts, batch_labels)
            return self

        # Prepare features and labels
        features = self._prepare_features(X)
        labels = self._prepare_labels(
            y, expand_for_tokens=(self.sequence_pooling == SequencePooling.NONE)
        )

        # Skip if no features
        if features.shape[0] == 0:
            if self.verbose:
                print("No features to train on (empty batch)")
            return self

        # Move to device and convert to float32 to avoid numerical issues
        features = features.to(self.device, dtype=torch.float32)
        labels = labels.to(self.device).float()  # Labels should be float for BCE loss

        # Initialize network if needed
        if self._network is None:
            self._init_network(features.shape[1], dtype=torch.float32)

        # Create dataset and dataloader
        dataset = torch.utils.data.TensorDataset(features, labels)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=False,
        )

        # Training loop
        self._network.train()
        for epoch in range(self.n_epochs):
            total_loss = 0
            n_batches = 0

            for batch_features, batch_labels in dataloader:
                # Forward pass
                self._optimizer.zero_grad()
                outputs = self._network(batch_features)
                loss = F.binary_cross_entropy_with_logits(outputs, batch_labels)

                # Backward pass
                loss.backward()
                self._optimizer.step()

                total_loss += loss.item()
                n_batches += 1

            # Print progress
            if self.verbose and (epoch + 1) % 10 == 0:
                avg_loss = total_loss / n_batches
                print(f"Epoch {epoch + 1}/{self.n_epochs}: loss={avg_loss:.4f}")

        self._network.eval()
        self._fitted = True
        return self

    def partial_fit(self, X: Activations, y: list | torch.Tensor) -> "MLP":
        """Incrementally fit the probe on a batch of samples.

        Args:
            X: Activations containing features for this batch
            y: Labels for this batch

        Returns:
            self: Partially fitted probe instance
        """
        # Prepare features and labels
        features = self._prepare_features(X)
        labels = self._prepare_labels(
            y, expand_for_tokens=(self.sequence_pooling == SequencePooling.NONE)
        )

        # Skip empty batches
        if features.shape[0] == 0:
            if self.verbose:
                print("Skipping batch with no features")
            return self

        # Move to device and convert to float32 to avoid numerical issues
        features = features.to(self.device, dtype=torch.float32)
        labels = labels.to(self.device).float()  # Labels should be float for BCE loss

        # Initialize network if this is the first batch
        if self._network is None:
            self._init_network(features.shape[1], dtype=torch.float32)

        # Single optimization step on this batch
        self._network.train()
        self._optimizer.zero_grad()
        outputs = self._network(features)
        loss = F.binary_cross_entropy_with_logits(outputs, labels)
        loss.backward()
        self._optimizer.step()

        self._streaming_steps += 1
        if self.verbose and self._streaming_steps % 10 == 0:
            print(f"Streaming step {self._streaming_steps}: loss={loss.item():.4f}")

        self._network.eval()
        self._fitted = True
        return self

    def predict_proba(self, X: Activations | ActivationIterator) -> torch.Tensor:
        """Predict class probabilities.

        Args:
            X: Activations or ActivationIterator containing features

        Returns:
            Tensor of shape (n_samples, 2) with probabilities for each class
        """
        if not self._fitted:
            raise RuntimeError("Probe must be fitted before prediction")

        if isinstance(X, ActivationIterator):
            # Predict on iterator batches
            predictions = []
            for batch_acts in X:
                batch_probs = self.predict_proba(batch_acts)
                predictions.append(batch_probs)
            return torch.cat(predictions, dim=0)

        # Prepare features and convert to float32 for consistency
        features = self._prepare_features(X)
        features = features.to(self.device, dtype=torch.float32)

        # Get predictions
        self._network.eval()
        with torch.no_grad():
            logits = self._network(features)
            probs_positive = torch.sigmoid(logits)

        # Create 2-class probability matrix
        probs = torch.stack([1 - probs_positive, probs_positive], dim=-1)

        # Aggregate token predictions to sample predictions if needed
        if self.sequence_pooling == SequencePooling.NONE and self._tokens_per_sample is not None:
            probs = self.aggregate_token_predictions(
                probs, self._tokens_per_sample, method="mean"
            )

        return probs

    def save(self, path: Path | str) -> None:
        """Save the probe to disk."""
        if not self._fitted:
            raise RuntimeError("Cannot save unfitted probe")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Prepare state dict
        state = {
            "layer": self.layer,
            "sequence_pooling": self.sequence_pooling.value,  # Save enum value
            "hidden_dim": self.hidden_dim,
            "dropout": self.dropout,
            "activation": self.activation,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "n_epochs": self.n_epochs,
            "batch_size": self.batch_size,
            "d_model": self._d_model,
            "network_state": self._network.state_dict(),
            "random_state": self.random_state,
        }

        torch.save(state, path)

    @classmethod
    def load(cls, path: Path | str, device: str | None = None) -> "MLP":
        """Load a probe from disk."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Probe file not found: {path}")

        # Load state dict
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        state = torch.load(path, map_location=device)

        # Convert sequence_pooling string back to enum
        pooling_value = state.get("sequence_pooling", "mean")
        sequence_pooling = SequencePooling(pooling_value)

        # Create probe instance
        probe = cls(
            layer=state["layer"],
            sequence_pooling=sequence_pooling,
            hidden_dim=state["hidden_dim"],
            dropout=state.get("dropout"),
            activation=state["activation"],
            learning_rate=state["learning_rate"],
            weight_decay=state["weight_decay"],
            n_epochs=state["n_epochs"],
            batch_size=state["batch_size"],
            device=device,
            random_state=state.get("random_state"),
        )

        # Initialize network and load state
        probe._init_network(state["d_model"])
        probe._network.load_state_dict(state["network_state"])
        probe._fitted = True

        return probe