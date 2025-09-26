"""Tests for MLP probe implementation."""

import tempfile
from pathlib import Path

import pytest
import torch

from probelib.probes.mlp import MLP
from probelib.processing.activations import Activations
from probelib.types import Label
from probelib.processing import SequencePooling


def create_test_activations(n_samples=10, seq_len=20, d_model=16, layer=0):
    """Create test activations."""
    acts = torch.randn(1, n_samples, seq_len, d_model)
    detection_mask = torch.ones(n_samples, seq_len)

    return Activations(
        activations=acts,
        attention_mask=torch.ones(n_samples, seq_len),
        input_ids=torch.ones(n_samples, seq_len, dtype=torch.long),
        detection_mask=detection_mask,
        layer_indices=[layer],
    )


def create_separable_data(n_samples=20, seq_len=10, d_model=8, layer=0):
    """Create linearly separable data for testing."""
    acts = torch.zeros(1, n_samples, seq_len, d_model)

    # First half: positive class
    for i in range(n_samples // 2):
        acts[0, i, :, :4] = 1.0
        acts[0, i, :, 4:] = 0.0

    # Second half: negative class
    for i in range(n_samples // 2, n_samples):
        acts[0, i, :, :4] = 0.0
        acts[0, i, :, 4:] = 1.0

    # Add small noise
    acts += torch.randn_like(acts) * 0.1

    activations = Activations(
        activations=acts,
        attention_mask=torch.ones(n_samples, seq_len),
        input_ids=torch.ones(n_samples, seq_len, dtype=torch.long),
        detection_mask=torch.ones(n_samples, seq_len),
        layer_indices=[layer],
    )

    labels = [Label.POSITIVE] * (n_samples // 2) + [Label.NEGATIVE] * (n_samples // 2)

    return activations, labels


class TestMLP:
    """Test MLP probe implementation."""

    def test_initialization(self):
        """Test probe initialization."""
        probe = MLP(
            layer=5,
            hidden_dim=256,
            dropout=0.2,
            activation="gelu",
            learning_rate=0.002,
            weight_decay=0.05,
            n_epochs=200,
            patience=10,
            sequence_pooling=SequencePooling.MAX,
            device="cpu",
            random_state=42,
            verbose=False,
        )

        assert probe.layer == 5
        assert probe.hidden_dim == 256
        assert probe.dropout == 0.2
        assert probe.activation == "gelu"
        assert probe.learning_rate == 0.002
        assert probe.weight_decay == 0.05
        assert probe.n_epochs == 200
        assert probe.patience == 10
        assert probe.sequence_pooling == SequencePooling.MAX
        # score_aggregation no longer exists
        assert probe._fitted is False
        assert probe._network is None

    def test_fit(self):
        """Test fitting the MLP probe."""
        activations, labels = create_separable_data(n_samples=20)

        probe = MLP(
            layer=0,
            hidden_dim=16,
            n_epochs=50,
            device="cpu",
            random_state=42,
        )

        fitted_probe = probe.fit(activations, labels)

        assert fitted_probe is probe
        assert probe._fitted is True
        assert probe._network is not None
        assert probe._d_model == 8

    def test_predict_proba(self):
        """Test probability prediction."""
        activations, labels = create_separable_data(n_samples=20)

        probe = MLP(
            layer=0,
            hidden_dim=32,
            n_epochs=100,
            patience=10,
            device="cpu",
            random_state=42,
        )
        probe.fit(activations, labels)

        probs = probe.predict_proba(activations)

        assert probs.shape == (20, 2)
        assert torch.all(probs >= 0) and torch.all(probs <= 1)
        assert torch.allclose(probs.sum(dim=1), torch.ones(20), atol=1e-6)

        # Check that it learned something (should get at least 50% accuracy on this simple data)
        predictions = probs.argmax(dim=1)
        true_labels = torch.tensor([label.value for label in labels])
        accuracy = (predictions == true_labels).float().mean()
        assert accuracy >= 0.5

    def test_predict_before_fit(self):
        """Test that prediction fails before fitting."""
        activations = create_test_activations()
        probe = MLP(layer=0, sequence_pooling=SequencePooling.MEAN, device="cpu")

        with pytest.raises(RuntimeError, match="Probe must be fitted"):
            probe.predict_proba(activations)

    def test_token_level_training(self):
        """Test training on token-level activations."""
        activations, labels = create_separable_data(n_samples=20)

        probe = MLP(
            layer=0,
            hidden_dim=16,
            sequence_pooling=SequencePooling.NONE,  # Token-level training with score aggregation
            n_epochs=50,
            device="cpu",
            random_state=42,
        )

        probe.fit(activations, labels)

        # Should still predict at sequence level
        probs = probe.predict_proba(activations)
        assert probs.shape == (20, 2)

    def test_partial_fit(self):
        """Test incremental fitting."""
        batch1_acts, batch1_labels = create_separable_data(n_samples=10)
        batch2_acts, batch2_labels = create_separable_data(n_samples=10)

        probe = MLP(
            layer=0,
            hidden_dim=16,
            device="cpu",
            random_state=42,
        )

        # Fit on first batch
        probe.partial_fit(batch1_acts, batch1_labels)
        assert probe._fitted is True
        assert probe._streaming_steps == 1

        # Update with second batch
        probe.partial_fit(batch2_acts, batch2_labels)
        assert probe._streaming_steps == 2

        # Should still make predictions
        probs = probe.predict_proba(batch1_acts)
        assert probs.shape == (10, 2)

    def test_aggregation_methods(self):
        """Test different aggregation methods."""
        activations, labels = create_separable_data(n_samples=20)

        for method in ["mean", "max", "last_token"]:
            probe = MLP(
                layer=0,
                hidden_dim=16,
                sequence_pooling=SequencePooling(method),
                n_epochs=50,
                device="cpu",
                random_state=42,
            )
            probe.fit(activations, labels)

            probs = probe.predict_proba(activations)
            assert probs.shape == (20, 2)

    def test_save_and_load(self):
        """Test saving and loading probe."""
        activations, labels = create_separable_data(n_samples=20)

        probe = MLP(
            layer=0,
            hidden_dim=32,
            dropout=0.1,
            activation="gelu",
            learning_rate=0.001,
            n_epochs=50,
            device="cpu",
            random_state=42,
        )
        probe.fit(activations, labels)

        probs_before = probe.predict_proba(activations)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "probe.pt"
            probe.save(save_path)

            loaded_probe = MLP.load(save_path)

            assert loaded_probe.layer == 0
            assert loaded_probe.hidden_dim == 32
            assert loaded_probe._fitted is True

            probs_after = loaded_probe.predict_proba(activations)
            assert torch.allclose(probs_before, probs_after, atol=1e-6)

    def test_different_activations(self):
        """Test different activation functions."""
        activations, labels = create_separable_data(n_samples=20)

        for activation in ["relu", "gelu"]:
            probe = MLP(
                layer=0,
                sequence_pooling=SequencePooling.MEAN,
                hidden_dim=16,
                activation=activation,
                n_epochs=50,
                device="cpu",
                random_state=42,
            )
            probe.fit(activations, labels)

            probs = probe.predict_proba(activations)
            assert probs.shape == (20, 2)
            assert torch.all(probs >= 0) and torch.all(probs <= 1)
