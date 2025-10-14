"""
Attention-based probe implementation following sklearn-style API.

This probe learns attention weights over the sequence dimension to focus on
the most relevant parts for classification, instead of using fixed aggregation.
"""

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW

from ..processing.activations import (
    ActivationIterator,
    Activations,
    SequencePooling,
)
from .base import BaseProbe


class _AttentionNetwork(nn.Module):
    """
    Attention-based neural network for sequence classification (internal implementation).

    Architecture:
    - Attention scoring: MLP that outputs attention scores for each token
    - Attention weights: Softmax over scores (masked by detection mask)
    - Weighted aggregation: Sum of attention-weighted activations
    - Classifier: MLP that predicts from the aggregated representation
    """

    def __init__(
        self,
        d_model: int,
        hidden_dim: int = 64,
        dropout: float = 0.1,
        temperature: float = 1.0,
    ):
        super().__init__()

        self.temperature = temperature

        # Attention scoring module with layer norm and dropout
        self.attention_norm = nn.LayerNorm(d_model)
        self.attention_scorer = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.ReLU(),  # Changed from Tanh to ReLU for better gradient flow
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

        # Classifier with dropout for regularization
        self.classifier_norm = nn.LayerNorm(d_model)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, hidden_dim * 2),  # Larger hidden dim
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

        # Initialize weights properly
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with Xavier/He initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(
                    module.weight, gain=0.5
                )  # Smaller gain for better init
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        sequences: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with attention mechanism.

        Args:
            sequences: Input sequences [batch, seq_len, d_model]
            mask: Valid token mask [batch, seq_len]

        Returns:
            Tuple of (logits, attention_weights)
            - logits: Classification logits [batch]
            - attention_weights: Attention weights [batch, seq_len]
        """
        # Apply layer norm before attention scoring
        normed_sequences = self.attention_norm(sequences)

        # Compute attention scores for each token
        attention_scores = self.attention_scorer(normed_sequences).squeeze(
            -1
        )  # [batch, seq_len]

        # Apply temperature scaling for better calibration
        attention_scores = attention_scores / self.temperature

        # Apply mask to attention scores
        attention_scores_masked = attention_scores.masked_fill(
            ~mask.bool(), float("-inf")
        )

        # Compute attention weights via softmax
        attention_weights = torch.softmax(attention_scores_masked, dim=1)

        # Handle edge case where all positions are masked
        attention_weights = torch.nan_to_num(attention_weights, nan=0.0)

        # Apply attention weights to get weighted representation
        weighted = (
            attention_weights.unsqueeze(-1) * sequences
        )  # [batch, seq_len, d_model]
        aggregated = weighted.sum(dim=1)  # [batch, d_model]

        # Apply layer norm before classification
        aggregated = self.classifier_norm(aggregated)

        # Classify the aggregated representation
        logits = self.classifier(aggregated).squeeze(-1)  # [batch]

        return logits, attention_weights


class Attention(BaseProbe):
    """
    Attention-based probe for sequence classification.

    Instead of using simple aggregation (mean/max), this probe learns attention
    weights to focus on the most relevant parts of the sequence for classification.

    Note: This probe handles sequences natively through learned attention weights,
    so it doesn't use sequence_aggregation or score_aggregation parameters.

    Attributes:
        hidden_dim: Hidden dimension for attention and classifier networks
        learning_rate: Learning rate for AdamW optimizer
        weight_decay: L2 regularization strength
        n_epochs: Maximum number of training epochs
        patience: Early stopping patience
        _network: AttentionNetwork instance
        _optimizer: AdamW optimizer
        _d_model: Input dimension (set during first fit)
        attention_weights: Last computed attention weights for interpretability
    """

    def __init__(
        self,
        layer: int,
        sequence_pooling: SequencePooling = SequencePooling.NONE,
        hidden_dim: int = 128,  # Increased from 64
        dropout: float = 0.2,  # Added dropout parameter
        temperature: float = 2.0,  # Added temperature for calibration
        learning_rate: float = 5e-4,  # Increased from 1e-4
        weight_decay: float = 1e-3,  # Reduced from 0.01
        n_epochs: int = 1000,
        patience: int = 20,  # Increased from 10
        device: str | None = None,
        random_state: int | None = None,
        verbose: bool = False,
    ):
        """
        Initialize attention probe.

        This probe handles sequences natively through learned attention weights,
        so it requires sequence_pooling=NONE.

        Args:
            layer: Layer index to use activations from
            sequence_pooling: Must be SequencePooling.NONE (attention requires sequences)
            hidden_dim: Hidden dimension for networks
            dropout: Dropout rate for regularization
            temperature: Temperature scaling for attention softmax
            learning_rate: Learning rate for AdamW
            weight_decay: Weight decay for AdamW (L2 regularization)
            n_epochs: Maximum number of training epochs
            patience: Early stopping patience
            device: Device for PyTorch operations
            random_state: Random seed for reproducibility
            verbose: Whether to print progress information
        """
        # Attention probe requires sequences - cannot pool beforehand
        if sequence_pooling != SequencePooling.NONE:
            raise ValueError(
                f"Attention probe requires sequence_pooling=NONE to compute attention weights, "
                f"got {sequence_pooling.name}"
            )

        super().__init__(
            layer=layer,
            sequence_pooling=sequence_pooling,
            device=device,
            random_state=random_state,
            verbose=verbose,
        )

        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.temperature = temperature
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.n_epochs = n_epochs
        self.patience = patience

        # Model components (initialized during fit)
        self._network = None
        self._optimizer = None
        self._d_model = None
        self._streaming_steps = 0
        self.attention_weights = None  # Store for interpretability

        # This probe requires gradients for training
        self._requires_grad = True

        # Set random seed for reproducibility
        if random_state is not None:
            torch.manual_seed(random_state)

    def _init_network(self, d_model: int, dtype: torch.dtype | None = None):
        """Initialize the network once we know the input dimension."""
        self._d_model = d_model
        self._network = _AttentionNetwork(
            d_model=d_model,
            hidden_dim=self.hidden_dim,
            dropout=self.dropout,
            temperature=self.temperature,
        ).to(self.device)

        # Match the dtype of the input features for mixed precision support
        if dtype is not None:
            self._network = self._network.to(dtype)

        self._optimizer = AdamW(
            self._network.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            fused=self.device.startswith("cuda"),
        )

    def _prepare_features(self, X: Activations) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare sequences for attention probe.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: (sequences, detection_mask)
        """
        # Regular Activations (must have sequences)
        if self.layer not in X.layer_indices:
            raise ValueError(f"Layer {self.layer} not in activations")

        if len(X.layer_indices) > 1:
            X = X.select(layers=self.layer)

        sequences = X.activations.squeeze(0)  # [batch, seq, hidden]
        detection_mask = X.detection_mask

        return sequences, detection_mask

    def fit(
        self, X: Activations | ActivationIterator, y: list | torch.Tensor
    ) -> "Attention":
        """
        Fit the probe on training data.

        Args:
            X: Activations or ActivationIterator containing features
            y: Labels for training

        Returns:
            self: Fitted probe instance
        """
        if isinstance(X, ActivationIterator):
            # Use streaming approach for iterators
            return self._fit_iterator(X, y)

        # Get sequences and mask
        sequences, detection_mask = self._prepare_features(X)
        labels = self._prepare_labels(y)

        # Move to device and ensure tensors are safe for autograd
        sequences = sequences.to(self.device)
        # Clone to avoid issues with inference tensors in autograd
        sequences = sequences.clone()
        detection_mask = detection_mask.to(self.device)
        labels = labels.to(self.device).float()  # Labels should be float for BCE loss

        # Validate we have both classes
        unique_labels = torch.unique(labels)
        if len(unique_labels) < 2:
            raise ValueError(
                f"Training data must contain both classes. Found: {unique_labels.tolist()}"
            )

        # Initialize network if needed
        if self._network is None:
            self._init_network(sequences.shape[-1], dtype=sequences.dtype)

        # Create train/validation split
        n_samples = len(sequences)
        n_val = max(1, int(0.2 * n_samples))

        if self.random_state is not None:
            torch.manual_seed(self.random_state)
        indices = torch.randperm(n_samples, device=self.device)

        train_indices = indices[n_val:]
        val_indices = indices[:n_val]

        train_sequences = sequences[train_indices]
        train_mask = detection_mask[train_indices]
        train_y = labels[train_indices]

        val_sequences = sequences[val_indices]
        val_mask = detection_mask[val_indices]
        val_y = labels[val_indices]

        # Training loop with early stopping
        best_val_loss = float("inf")
        patience_counter = 0

        self._network.train()
        for epoch in range(self.n_epochs):
            # Training step
            self._optimizer.zero_grad()
            logits, _ = self._network(train_sequences, train_mask)
            loss = F.binary_cross_entropy_with_logits(logits, train_y)
            loss.backward()
            self._optimizer.step()

            # Validation step (less frequent for efficiency)
            if epoch % 10 == 0:
                self._network.eval()
                with torch.no_grad():
                    val_logits, _ = self._network(val_sequences, val_mask)
                    val_loss = F.binary_cross_entropy_with_logits(val_logits, val_y)
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

                # Stop if loss is very small
                if val_loss < 0.001:  # Reduced from 0.01 for better convergence
                    if self.verbose:
                        print(f"Converged at epoch {epoch}")
                    break

        self._network.eval()
        self._fitted = True
        return self

    def _fit_iterator(
        self, X: ActivationIterator, y: list | torch.Tensor
    ) -> "Attention":
        """
        Fit using an ActivationIterator (streaming mode).

        Args:
            X: ActivationIterator yielding batches of activations
            y: All labels

        Returns:
            self: Fitted probe instance
        """
        labels_tensor = self._prepare_labels(y)

        # Process batches
        for batch_acts in X:
            batch_idx = torch.tensor(
                batch_acts.batch_indices, device=labels_tensor.device, dtype=torch.long
            )
            batch_labels = labels_tensor.index_select(0, batch_idx)
            self.partial_fit(batch_acts, batch_labels)

        return self

    def partial_fit(self, X: Activations, y: list | torch.Tensor) -> "Attention":
        """
        Incrementally fit the probe on a batch of samples.

        Args:
            X: Activations containing features for this batch
            y: Labels for this batch

        Returns:
            self: Partially fitted probe instance
        """
        sequences, detection_mask = self._prepare_features(X)
        labels = self._prepare_labels(y)

        # Move to device and ensure tensors are safe for autograd
        sequences = sequences.to(self.device)
        # Clone to avoid issues with inference tensors in autograd
        sequences = sequences.clone()
        detection_mask = detection_mask.to(self.device)
        labels = labels.to(self.device).float()  # Labels should be float for BCE loss

        # Initialize network on first batch
        if self._network is None:
            self._init_network(sequences.shape[-1], dtype=sequences.dtype)

        # Training step
        self._network.train()
        self._optimizer.zero_grad()
        logits, _ = self._network(sequences, detection_mask)
        loss = F.binary_cross_entropy_with_logits(logits, labels)
        loss.backward()
        self._optimizer.step()

        self._streaming_steps += 1
        self._fitted = True

        # Switch back to eval mode
        self._network.eval()
        return self

    def predict_proba(
        self, X: Activations | ActivationIterator, *, logits: bool = False
    ) -> torch.Tensor:
        """
        Predict class probabilities or logits.

        Args:
            X: Activations or ActivationIterator containing features
            logits: If True, return raw logits instead of probabilities.
                Useful for adversarial training to avoid sigmoid saturation.

        Returns:
            If logits=False (default): Tensor of shape (n_samples, 2) with probabilities
            If logits=True: Tensor of shape (n_samples,) with raw logits
        """
        if not self._fitted:
            raise RuntimeError("Probe must be fitted before prediction")

        if isinstance(X, ActivationIterator):
            # Predict on iterator batches
            return self._predict_iterator(X, logits=logits)

        # Get sequences and mask
        sequences, detection_mask = self._prepare_features(X)
        sequences = sequences.to(self.device)
        detection_mask = detection_mask.to(self.device)

        # Get predictions
        self._network.eval()
        raw_logits, attention_weights = self._network(sequences, detection_mask)

        # Store attention weights for interpretability (always detached)
        self.attention_weights = attention_weights.detach().cpu()

        # Return logits directly if requested
        if logits:
            return raw_logits

        # Otherwise return probabilities
        probs_positive = torch.sigmoid(raw_logits)

        # Create 2-class probability matrix
        probs = torch.zeros(len(probs_positive), 2, device=self.device)
        probs[:, 0] = 1 - probs_positive  # P(y=0)
        probs[:, 1] = probs_positive  # P(y=1)

        return probs

    def _predict_iterator(
        self, X: ActivationIterator, logits: bool = False
    ) -> torch.Tensor:
        """
        Predict on an ActivationIterator.

        Args:
            X: ActivationIterator yielding batches
            logits: If True, return raw logits instead of probabilities

        Returns:
            Concatenated predictions for all batches
        """
        all_probs = []

        for batch_acts in X:
            batch_probs = self.predict_proba(batch_acts, logits=logits)
            all_probs.append(batch_probs)

        return torch.cat(all_probs, dim=0)

    def save(self, path: Path | str) -> None:
        """
        Save the probe to disk.

        Args:
            path: Path to save the probe
        """
        if not self._fitted:
            raise RuntimeError("Cannot save unfitted probe")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Prepare state dict
        state = {
            "layer": self.layer,
            "sequence_pooling": self.sequence_pooling.value,  # Always "none" for attention
            "hidden_dim": self.hidden_dim,
            "dropout": self.dropout,
            "temperature": self.temperature,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "n_epochs": self.n_epochs,
            "patience": self.patience,
            "device": self.device,
            "random_state": self.random_state,
            "verbose": self.verbose,
            "d_model": self._d_model,
            "network_state_dict": self._network.state_dict(),
            "optimizer_state_dict": self._optimizer.state_dict(),
            "streaming_steps": self._streaming_steps,
        }

        torch.save(state, path)

    @classmethod
    def load(cls, path: Path | str, device: str | None = None) -> "Attention":
        """
        Load a probe from disk.

        Args:
            path: Path to load the probe from
            device: Device to load onto (None to use saved device)

        Returns:
            Loaded probe instance
        """
        path = Path(path)
        state = torch.load(path, map_location="cpu")

        # Create probe instance
        # sequence_pooling is always NONE for attention, but check if present for compatibility
        sequence_pooling = SequencePooling.NONE  # Always NONE for attention
        if "sequence_pooling" in state:
            # Validate it's NONE if present
            pooling_value = state["sequence_pooling"]
            if pooling_value != "none":
                raise ValueError(
                    f"Loaded attention probe has invalid sequence_pooling={pooling_value}, "
                    f"expected 'none'"
                )

        probe = cls(
            layer=state["layer"],
            sequence_pooling=sequence_pooling,
            hidden_dim=state["hidden_dim"],
            dropout=state.get("dropout", 0.2),  # Default for backwards compat
            temperature=state.get("temperature", 2.0),  # Default for backwards compat
            learning_rate=state["learning_rate"],
            weight_decay=state["weight_decay"],
            n_epochs=state["n_epochs"],
            patience=state["patience"],
            device=device or state["device"],
            random_state=state["random_state"],
            verbose=state.get("verbose", False),
        )

        # Initialize network
        probe._init_network(state["d_model"])

        # Load network and optimizer states
        probe._network.load_state_dict(state["network_state_dict"])
        probe._optimizer.load_state_dict(state["optimizer_state_dict"])
        probe._streaming_steps = state["streaming_steps"]

        # Move to correct device
        probe._network = probe._network.to(probe.device)
        probe._network.eval()
        probe._fitted = True

        return probe

    def __repr__(self) -> str:
        """String representation of the probe."""
        fitted_str = "fitted" if self._fitted else "not fitted"
        return (
            f"{self.__class__.__name__}("
            f"layer={self.layer}, "
            f"hidden_dim={self.hidden_dim}, "
            f"attention-based aggregation, "
            f"{fitted_str})"
        )
