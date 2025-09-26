"""GPU-accelerated L2-regularized logistic regression probe."""

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW

from ..processing.activations import ActivationIterator, Activations, SequencePooling
from .base import BaseProbe


class _LogisticNetwork(nn.Module):
    """Simple logistic regression network (internal implementation)."""

    def __init__(self, d_model: int):
        super().__init__()
        self.linear = nn.Linear(d_model, 1, bias=False)
        nn.init.xavier_uniform_(self.linear.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x).squeeze(-1)


class _GPUStandardScaler:
    """Standard scaler for GPU tensors with streaming support."""

    def __init__(self, device: str = "cuda"):
        self.device = device
        self.mean_ = None
        self.std_ = None
        self._fitted = False
        self.n_samples_seen_ = 0

    def fit(self, X: torch.Tensor) -> "GPUStandardScaler":
        """Fit the scaler on data."""
        X = X.to(self.device)
        self.mean_ = X.mean(dim=0)
        self.std_ = X.std(dim=0).clamp(min=1e-8)
        self._fitted = True
        self.n_samples_seen_ = X.shape[0]
        return self

    def partial_fit(self, X: torch.Tensor) -> "GPUStandardScaler":
        """Update running statistics with a batch."""
        X = X.to(self.device)
        batch_size = X.shape[0]

        if not self._fitted:
            return self.fit(X)

        # Update running statistics
        delta = X.mean(dim=0) - self.mean_
        self.mean_ += delta * batch_size / (self.n_samples_seen_ + batch_size)

        # Update variance estimate
        batch_var = X.var(dim=0)
        old_var = self.std_**2

        weight_old = self.n_samples_seen_ / (self.n_samples_seen_ + batch_size)
        weight_new = batch_size / (self.n_samples_seen_ + batch_size)

        new_var = weight_old * old_var + weight_new * batch_var + weight_old * weight_new * delta**2
        self.std_ = torch.sqrt(new_var).clamp(min=1e-8)
        self.n_samples_seen_ += batch_size

        return self

    def transform(self, X: torch.Tensor) -> torch.Tensor:
        """Transform the data."""
        if not self._fitted:
            raise RuntimeError("Scaler must be fitted before transform")
        X = X.to(self.device)
        # Ensure dtypes match for the operation
        mean = self.mean_.to(X.dtype)
        std = self.std_.to(X.dtype)
        return (X - mean) / std

    def fit_transform(self, X: torch.Tensor) -> torch.Tensor:
        """Fit and transform in one step."""
        return self.fit(X).transform(X)


class Logistic(BaseProbe):
    """GPU-accelerated L2-regularized logistic regression probe.

    This implementation uses PyTorch for GPU acceleration and supports
    streaming updates via partial_fit. The probe uses AdamW optimizer
    with weight decay (equivalent to L2 regularization).
    """

    def __init__(
        self,
        layer: int | None = None,
        sequence_pooling: SequencePooling = SequencePooling.MEAN,
        l2_penalty: float = 1.0,
        learning_rate: float = 0.001,
        n_epochs: int = 100,
        patience: int = 10,
        device: str | None = None,
        random_state: int | None = None,
        verbose: bool = False,
    ):
        """Initialize GPU-based logistic regression probe.

        Args:
            layer: Optional layer index for train_probes compatibility
            sequence_pooling: How to pool sequences before training
            l2_penalty: L2 regularization strength (weight_decay)
            learning_rate: Learning rate for optimizer
            n_epochs: Maximum number of training epochs
            patience: Early stopping patience
            device: Device for PyTorch operations
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

        # Training parameters
        self.l2_penalty = l2_penalty
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.patience = patience

        # Model components (initialized during fit)
        self._network = None
        self._optimizer = None
        self._scaler = _GPUStandardScaler(device=self.device)
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
        self._network = _LogisticNetwork(d_model).to(self.device)

        # Match the dtype of the input features for mixed precision support
        if dtype is not None:
            self._network = self._network.to(dtype)

        self._optimizer = AdamW(
            self._network.parameters(),
            lr=self.learning_rate,
            weight_decay=self.l2_penalty,
            fused=self.device.startswith("cuda"),
        )

    def fit(self, X: Activations | ActivationIterator, y: list | torch.Tensor) -> "Logistic":
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

        # Move to device
        features = features.to(self.device)
        labels = labels.to(self.device).float()  # Labels should be float for BCE loss

        # Initialize network if needed
        if self._network is None:
            self._init_network(features.shape[1], dtype=features.dtype)

        # Standardize features
        features_scaled = self._scaler.fit_transform(features)

        # Create train/val split for early stopping
        n_samples = features.shape[0]
        n_val = max(1, int(0.2 * n_samples))

        if self.random_state is not None:
            torch.manual_seed(self.random_state)
        indices = torch.randperm(n_samples, device=self.device)

        train_indices = indices[n_val:]
        val_indices = indices[:n_val]

        X_train = features_scaled[train_indices]
        y_train = labels[train_indices]
        X_val = features_scaled[val_indices]
        y_val = labels[val_indices]

        # Training loop with early stopping
        best_val_loss = float("inf")
        patience_counter = 0

        self._network.train()
        for epoch in range(self.n_epochs):
            # Training step
            self._optimizer.zero_grad()
            logits = self._network(X_train)
            loss = F.binary_cross_entropy_with_logits(logits, y_train)
            loss.backward()
            self._optimizer.step()

            # Validation
            self._network.eval()
            with torch.no_grad():
                val_logits = self._network(X_val)
                val_loss = F.binary_cross_entropy_with_logits(val_logits, y_val)
            self._network.train()

            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    if self.verbose:
                        print(f"Early stopping at epoch {epoch}")
                    break

            # Print progress
            if self.verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{self.n_epochs}: loss={val_loss:.4f}")

        self._network.eval()
        self._fitted = True
        return self

    def partial_fit(self, X: Activations, y: list | torch.Tensor) -> "Logistic":
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
                print("Skipping batch with no detected tokens")
            return self

        # Move to device
        features = features.to(self.device)
        labels = labels.to(self.device).float()  # Labels should be float for BCE loss

        # Initialize network on first batch
        if self._network is None:
            self._init_network(features.shape[1], dtype=features.dtype)

        # Update scaler and standardize
        if self._scaler._fitted:
            self._scaler.partial_fit(features)
            features_scaled = self._scaler.transform(features)
        else:
            features_scaled = self._scaler.fit_transform(features)

        # Single optimization step
        self._network.train()
        self._optimizer.zero_grad()
        logits = self._network(features_scaled)
        loss = F.binary_cross_entropy_with_logits(logits, labels)
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

        # Prepare features
        features = self._prepare_features(X)
        features = features.to(self.device)

        # Standardize features
        features_scaled = self._scaler.transform(features)

        # Get predictions
        self._network.eval()
        with torch.no_grad():
            logits = self._network(features_scaled)
            probs_positive = torch.sigmoid(logits)

        # Create 2-class probability matrix
        probs = torch.stack([1 - probs_positive, probs_positive], dim=-1)

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
            "sequence_pooling": self.sequence_pooling.value,
            "l2_penalty": self.l2_penalty,
            "learning_rate": self.learning_rate,
            "n_epochs": self.n_epochs,
            "patience": self.patience,
            "d_model": self._d_model,
            "network_state": self._network.state_dict(),
            "scaler_mean": self._scaler.mean_,
            "scaler_std": self._scaler.std_,
            "scaler_n_samples": self._scaler.n_samples_seen_,
            "random_state": self.random_state,
        }

        torch.save(state, path)

    @classmethod
    def load(cls, path: Path | str, device: str | None = None) -> "Logistic":
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
            l2_penalty=state["l2_penalty"],
            learning_rate=state["learning_rate"],
            n_epochs=state["n_epochs"],
            patience=state["patience"],
            device=device,
            random_state=state.get("random_state"),
        )

        # Initialize network and load state
        probe._init_network(state["d_model"])
        probe._network.load_state_dict(state["network_state"])

        # Restore scaler state
        probe._scaler._fitted = True
        probe._scaler.mean_ = state["scaler_mean"].to(device)
        probe._scaler.std_ = state["scaler_std"].to(device)
        probe._scaler.n_samples_seen_ = state["scaler_n_samples"]

        probe._fitted = True
        return probe