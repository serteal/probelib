"""GPU-accelerated L2-regularized logistic regression probe."""

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.preprocessing import StandardScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR

from ..processing.activations import ActivationIterator, Activations, SequencePooling
from .base import BaseProbe


class _LogisticNetwork(nn.Module):
    """Simple logistic regression network (internal implementation)."""

    def __init__(self, d_model: int):
        super().__init__()
        self.linear = nn.Linear(d_model, 1, bias=True)
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

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

        new_var = (
            weight_old * old_var
            + weight_new * batch_var
            + weight_old * weight_new * delta**2
        )
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

    This implementation uses PyTorch with LBFGS optimizer to match sklearn's
    LogisticRegression behavior. For batch training, it uses LBFGS on all data.
    For streaming, it falls back to SGD-like updates via partial_fit.
    """

    def __init__(
        self,
        layer: int | None = None,
        sequence_pooling: SequencePooling = SequencePooling.MEAN,
        C: float = 1.0,
        max_iter: int = 100,
        device: str | None = None,
        random_state: int | None = None,
        verbose: bool = False,
    ):
        """Initialize GPU-based logistic regression probe.

        Args:
            layer: Optional layer index for train_probes compatibility
            sequence_pooling: How to pool sequences before training
            C: Inverse of regularization strength (higher = less regularization)
            max_iter: Maximum number of iterations for LBFGS
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
        self.C = C
        self.max_iter = max_iter

        # Model components (initialized during fit)
        self._network = None
        self._optimizer = None
        self._scheduler = None  # Learning rate scheduler for streaming mode
        self._scaler = _GPUStandardScaler(device=self.device)
        self._d_model = None
        self._streaming_steps = 0
        self._use_lbfgs = True  # Track optimizer type
        self._streaming_n_seen = 0  # Total samples seen in streaming mode
        self._scaler_frozen = False  # Freeze scaler after first epoch
        self._n_epochs = 100  # Default number of epochs for streaming

        # This probe requires gradients
        self._requires_grad = True

        # Set random seed
        if random_state is not None:
            torch.manual_seed(random_state)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(random_state)

    def _init_network(self, d_model: int, dtype: torch.dtype | None = None, use_lbfgs: bool = True, n_epochs: int = 100):
        """Initialize the network and optimizer once we know the input dimension."""
        self._d_model = d_model
        self._network = _LogisticNetwork(d_model).to(self.device)

        # Match the dtype of the input features for mixed precision support
        if dtype is not None:
            self._network = self._network.to(dtype)

        # Use LBFGS for batch training, AdamW for streaming
        if use_lbfgs:
            self._optimizer = torch.optim.LBFGS(
                self._network.parameters(),
                max_iter=self.max_iter,
                line_search_fn='strong_wolfe',
            )
            self._use_lbfgs = True
        else:
            # For streaming mode, use AdamW (better than SGD for streaming)
            # Compute weight decay from C (L2 regularization strength)
            self._alpha = 1.0 / self.C if self.C > 0 else 0.0001

            # AdamW learning rate - typically 1e-3 to 1e-4
            # Scale inversely with alpha for better conditioning
            self._lr = min(1e-3, 1.0 / (100 * self._alpha)) if self._alpha > 0 else 1e-3

            self._optimizer = torch.optim.AdamW(
                self._network.parameters(),
                lr=self._lr,
                weight_decay=self._alpha,  # AdamW handles L2 reg via weight_decay
                betas=(0.9, 0.999),
                eps=1e-8,
            )
            self._use_lbfgs = False
            self._t = 0  # Step counter for tracking
            self._n_epochs = n_epochs

            # Create learning rate scheduler: simple cosine decay (no warmup)
            # Cosine annealing works well for AdamW and decays smoothly to a minimum
            self._scheduler = CosineAnnealingLR(
                self._optimizer,
                T_max=n_epochs,  # Full cosine cycle over all epochs
                eta_min=self._lr * 0.1  # Minimum LR is 10% of base (not too low)
            )

            # Log streaming hyperparameters if verbose
            if hasattr(self, 'verbose') and self.verbose:
                print(f"Streaming AdamW initialized: C={self.C}, alpha={self._alpha:.4f}, "
                      f"lr={self._lr:.6f}, weight_decay={self._alpha:.4f}")
                print(f"LR schedule: cosine annealing over {n_epochs} epochs (min_lr={self._lr * 0.1:.6f})")

    def fit(
        self, X: Activations | ActivationIterator, y: list | torch.Tensor
    ) -> "Logistic":
        """Fit the probe on training data.

        Args:
            X: Activations or ActivationIterator containing features
            y: Labels for training

        Returns:
            self: Fitted probe instance
        """
        if isinstance(X, ActivationIterator):
            # Use streaming for iterators - more epochs for better convergence
            # Increased from 50 to 100 for improved performance
            n_epochs = min(100, max(50, self.max_iter))

            labels = self._prepare_labels(y)

            if self.verbose:
                print(f"Streaming mode: training for {n_epochs} epochs with cosine annealing LR")

            # Track first batch to initialize network with correct n_epochs
            first_batch = True

            for epoch in range(n_epochs):
                for batch_acts in X:
                    batch_idx = torch.tensor(
                        batch_acts.batch_indices, device=labels.device, dtype=torch.long
                    )
                    batch_labels = labels.index_select(0, batch_idx)

                    # Pass n_epochs on first batch for scheduler initialization
                    if first_batch:
                        self._current_n_epochs = n_epochs
                        first_batch = False

                    self.partial_fit(batch_acts, batch_labels)

                # Freeze scaler after first epoch to prevent non-stationarity
                if epoch == 0:
                    self._scaler_frozen = True
                    if self.verbose:
                        print(f"  Scaler frozen after epoch 1 (mean: {self._scaler.mean_[:3].cpu().numpy()}, std: {self._scaler.std_[:3].cpu().numpy()})")

                # Step the learning rate scheduler at the end of each epoch
                if self._scheduler is not None:
                    self._scheduler.step()
                    current_lr = self._optimizer.param_groups[0]['lr']
                    if self.verbose and (epoch + 1) % 10 == 0:
                        print(f"  Epoch {epoch + 1}/{n_epochs}: lr={current_lr:.6f}")
                elif self.verbose and (epoch + 1) % 10 == 0:
                    print(f"  Completed epoch {epoch + 1}/{n_epochs}")

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

        # Initialize network if needed (use LBFGS for batch mode)
        if self._network is None:
            self._init_network(features.shape[1], dtype=features.dtype, use_lbfgs=True)

        # Standardize features
        features_scaled = self._scaler.fit_transform(features)

        # Train on all data using LBFGS (no validation split, matching sklearn)
        n_samples = features.shape[0]

        # Compute L2 regularization weight (sklearn's C is inverse of regularization)
        # In sklearn: loss = BCE + (1/(2*C*n)) * ||w||^2
        # We'll apply: loss = BCE + (1/(2*C*n)) * ||w||^2
        l2_weight = 1.0 / (2.0 * self.C * n_samples) if self.C > 0 else 0.0

        self._network.train()

        # LBFGS requires a closure that computes and returns the loss
        def closure():
            self._optimizer.zero_grad()
            logits = self._network(features_scaled)

            # Binary cross entropy loss
            bce_loss = F.binary_cross_entropy_with_logits(logits, labels)

            # Add L2 regularization on weights (not bias)
            l2_reg = 0.0
            if l2_weight > 0:
                for name, param in self._network.named_parameters():
                    if 'weight' in name:  # Only regularize weights, not bias
                        l2_reg += torch.sum(param ** 2)

            loss = bce_loss + l2_weight * l2_reg
            loss.backward()

            return loss

        # Run LBFGS optimization
        self._optimizer.step(closure)

        if self.verbose:
            # Evaluate final loss without gradients
            self._network.eval()
            with torch.no_grad():
                logits = self._network(features_scaled)
                bce_loss = F.binary_cross_entropy_with_logits(logits, labels)
                print(f"LBFGS converged with BCE loss: {bce_loss.item():.4f}")

        self._network.eval()
        self._fitted = True
        return self

    def partial_fit(self, X: Activations, y: list | torch.Tensor) -> "Logistic":
        """Incrementally fit the probe on a batch of samples.

        Uses SGD with optimal learning rate schedule (sklearn-style) for streaming updates.

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

        # Initialize network on first batch (use AdamW for streaming)
        if self._network is None:
            # Use stored n_epochs if available, otherwise default to 100
            n_epochs = getattr(self, '_current_n_epochs', 100)
            self._init_network(features.shape[1], dtype=features.dtype, use_lbfgs=False, n_epochs=n_epochs)

        # Update scaler statistics on first epoch only, then freeze
        if not self._scaler_frozen:
            # First epoch: update running statistics
            features_scaled = self._scaler.partial_fit(features).transform(features)
        else:
            # Subsequent epochs: use frozen statistics
            features_scaled = self._scaler.transform(features)

        # Track total samples seen
        n_samples = features.shape[0]
        self._streaming_n_seen += n_samples
        self._t += 1

        # Single optimization step (AdamW handles L2 via weight_decay)
        self._network.train()
        self._optimizer.zero_grad()

        # Compute loss (no manual L2 reg - AdamW handles it)
        logits = self._network(features_scaled)
        loss = F.binary_cross_entropy_with_logits(logits, labels)

        loss.backward()
        self._optimizer.step()

        self._streaming_steps += 1
        if self.verbose and self._streaming_steps % 50 == 0:
            lr = self._optimizer.param_groups[0]['lr']
            print(f"  Step {self._streaming_steps}: loss={loss.item():.4f}, lr={lr:.6f}")

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

        # Aggregate token predictions to sample predictions if needed
        if (
            self.sequence_pooling == SequencePooling.NONE
            and self._tokens_per_sample is not None
        ):
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
            "sequence_pooling": self.sequence_pooling.value,
            "C": self.C,
            "max_iter": self.max_iter,
            "d_model": self._d_model,
            "network_state": self._network.state_dict(),
            "scaler_mean": self._scaler.mean_,
            "scaler_std": self._scaler.std_,
            "scaler_n_samples": self._scaler.n_samples_seen_,
            "random_state": self.random_state,
            "use_lbfgs": self._use_lbfgs,
            "streaming_steps": self._streaming_steps,
            "streaming_n_seen": self._streaming_n_seen,
            "t": getattr(self, '_t', 0),
            "lr": getattr(self, '_lr', 1e-3),
            "alpha": getattr(self, '_alpha', 0.0),
            "n_epochs": getattr(self, '_n_epochs', 100),
            "scheduler_state": self._scheduler.state_dict() if self._scheduler is not None else None,
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

        # Backward compatibility: handle old parameter names
        if "C" in state:
            C = state["C"]
            max_iter = state.get("max_iter", 100)
        else:
            # Old format with l2_penalty
            C = 1.0 / state.get("l2_penalty", 1.0) if state.get("l2_penalty", 1.0) > 0 else 1.0
            max_iter = state.get("n_epochs", 100)

        # Create probe instance
        probe = cls(
            layer=state["layer"],
            sequence_pooling=sequence_pooling,
            C=C,
            max_iter=max_iter,
            device=device,
            random_state=state.get("random_state"),
        )

        # Initialize network and load state
        use_lbfgs = state.get("use_lbfgs", True)
        n_epochs = state.get("n_epochs", 100)
        probe._init_network(state["d_model"], use_lbfgs=use_lbfgs, n_epochs=n_epochs)
        probe._network.load_state_dict(state["network_state"])
        probe._use_lbfgs = use_lbfgs

        # Restore streaming state
        probe._streaming_steps = state.get("streaming_steps", 0)
        probe._streaming_n_seen = state.get("streaming_n_seen", 0)
        if not use_lbfgs:
            probe._t = state.get("t", 0)
            probe._lr = state.get("lr", 1e-3)
            probe._alpha = state.get("alpha", 1.0 / C if C > 0 else 0.0001)
            probe._n_epochs = n_epochs

            # Restore scheduler state if available
            if "scheduler_state" in state and state["scheduler_state"] is not None:
                probe._scheduler.load_state_dict(state["scheduler_state"])

        # Restore scaler state
        probe._scaler._fitted = True
        probe._scaler.mean_ = state["scaler_mean"].to(device)
        probe._scaler.std_ = state["scaler_std"].to(device)
        probe._scaler.n_samples_seen_ = state["scaler_n_samples"]

        probe._fitted = True
        return probe


class SklearnLogistic(BaseProbe):
    """Sklearn-based logistic regression probe matching TPC implementation.

    This implementation uses scikit-learn's LogisticRegression with LBFGS solver
    for batch training, and SGDClassifier for streaming mode via partial_fit.
    """

    def __init__(
        self,
        layer: int | None = None,
        sequence_pooling: SequencePooling = SequencePooling.MEAN,
        C: float = 1.0,
        max_iter: int = 500,
        device: str | None = None,
        random_state: int | None = None,
        verbose: bool = False,
    ):
        """Initialize sklearn-based logistic regression probe.

        Args:
            layer: Optional layer index for train_probes compatibility
            sequence_pooling: How to pool sequences before training
            C: Inverse of regularization strength (default: 1.0)
            max_iter: Maximum iterations for solver
            device: Device for feature extraction (sklearn runs on CPU)
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

        self.C = C
        self.max_iter = max_iter

        # Model components (initialized during fit)
        self._scaler = StandardScaler()
        self._model = None
        self._use_sgd = False  # Track whether we're using SGD for streaming

        # This probe doesn't require gradients
        self._requires_grad = False

    def fit(
        self, X: Activations | ActivationIterator, y: list | torch.Tensor
    ) -> "SklearnLogistic":
        """Fit the probe on training data.

        Args:
            X: Activations or ActivationIterator containing features
            y: Labels for training

        Returns:
            self: Fitted probe instance
        """
        if isinstance(X, ActivationIterator):
            # Use streaming for iterators - multiple epochs for better convergence
            # Sklearn SGDClassifier uses max_iter as total iterations across all data
            # Use fewer epochs to avoid overfitting
            n_epochs = min(20, max(10, self.max_iter // 10))

            labels = self._prepare_labels(y)

            if self.verbose:
                print(f"Sklearn streaming mode: training for {n_epochs} epochs")

            for epoch in range(n_epochs):
                for batch_acts in X:
                    batch_idx = torch.tensor(
                        batch_acts.batch_indices, device=labels.device, dtype=torch.long
                    )
                    batch_labels = labels.index_select(0, batch_idx)
                    self.partial_fit(batch_acts, batch_labels)

                if self.verbose and (epoch + 1) % 50 == 0:
                    print(f"  Completed epoch {epoch + 1}/{n_epochs}")

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

        # Convert to numpy (convert to float32 first to handle bfloat16)
        if isinstance(features, torch.Tensor):
            X_np = features.to(dtype=torch.float32).cpu().numpy()
        else:
            X_np = features

        if isinstance(labels, torch.Tensor):
            y_np = labels.to(dtype=torch.float32).cpu().numpy()
        else:
            y_np = labels

        # Fit scaler on training data
        X_scaled = self._scaler.fit_transform(X_np)

        # Train LogisticRegression with LBFGS
        self._model = LogisticRegression(
            C=self.C,
            max_iter=self.max_iter,
            random_state=self.random_state,
            solver="lbfgs",
        )
        self._model.fit(X_scaled, y_np)
        self._use_sgd = False
        self._fitted = True

        return self

    def partial_fit(self, X: Activations, y: list | torch.Tensor) -> "SklearnLogistic":
        """Incrementally fit the probe on a batch of samples.

        Uses SGDClassifier for streaming updates.

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

        # Convert to numpy (convert to float32 first to handle bfloat16)
        if isinstance(features, torch.Tensor):
            X_np = features.to(dtype=torch.float32).cpu().numpy()
        else:
            X_np = features

        if isinstance(labels, torch.Tensor):
            y_np = labels.to(dtype=torch.float32).cpu().numpy()
        else:
            y_np = labels

        # Initialize SGDClassifier on first batch
        if self._model is None:
            # alpha is inverse of C for SGD (alpha = 1 / (C * n_samples))
            # We approximate this by using alpha = 1 / C
            alpha = 1.0 / self.C if self.C > 0 else 0.0001

            self._model = SGDClassifier(
                loss="log_loss",  # Logistic regression
                penalty="l2",
                alpha=alpha,
                max_iter=self.max_iter,
                random_state=self.random_state,
                learning_rate="optimal",
            )
            self._use_sgd = True

        # Update scaler and standardize
        if hasattr(self._scaler, "mean_"):
            # Scaler already fitted, just transform
            X_scaled = self._scaler.transform(X_np)
        else:
            # First batch, fit scaler
            X_scaled = self._scaler.fit_transform(X_np)

        # Partial fit
        self._model.partial_fit(X_scaled, y_np, classes=[0, 1])
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

        # Convert to numpy (convert to float32 first to handle bfloat16)
        if isinstance(features, torch.Tensor):
            X_np = features.to(dtype=torch.float32).cpu().numpy()
        else:
            X_np = features

        # Standardize features
        X_scaled = self._scaler.transform(X_np)

        # Get predictions
        probs_np = self._model.predict_proba(X_scaled)

        # Convert to torch tensor
        probs = torch.from_numpy(probs_np).float()

        # Aggregate token predictions to sample predictions if needed
        if (
            self.sequence_pooling == SequencePooling.NONE
            and self._tokens_per_sample is not None
        ):
            probs = self.aggregate_token_predictions(
                probs, self._tokens_per_sample, method="mean"
            )

        return probs

    def save(self, path: Path | str) -> None:
        """Save the probe to disk."""
        if not self._fitted:
            raise RuntimeError("Cannot save unfitted probe")

        import pickle

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Prepare state dict
        state = {
            "layer": self.layer,
            "sequence_pooling": self.sequence_pooling.value,
            "C": self.C,
            "max_iter": self.max_iter,
            "random_state": self.random_state,
            "scaler": self._scaler,
            "model": self._model,
            "use_sgd": self._use_sgd,
        }

        with open(path, "wb") as f:
            pickle.dump(state, f)

    @classmethod
    def load(cls, path: Path | str, device: str | None = None) -> "SklearnLogistic":
        """Load a probe from disk."""
        import pickle

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Probe file not found: {path}")

        with open(path, "rb") as f:
            state = pickle.load(f)

        # Convert sequence_pooling string back to enum
        pooling_value = state.get("sequence_pooling", "mean")
        sequence_pooling = SequencePooling(pooling_value)

        # Create probe instance
        probe = cls(
            layer=state["layer"],
            sequence_pooling=sequence_pooling,
            C=state["C"],
            max_iter=state["max_iter"],
            device=device,
            random_state=state.get("random_state"),
        )

        # Restore model and scaler
        probe._scaler = state["scaler"]
        probe._model = state["model"]
        probe._use_sgd = state.get("use_sgd", False)
        probe._fitted = True

        return probe
