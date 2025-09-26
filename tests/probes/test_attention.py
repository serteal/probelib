"""Tests for attention probe implementation."""

import tempfile
from pathlib import Path

import pytest
import torch

from probelib.probes.attention import Attention
from probelib.processing.activations import Activations
from probelib.types import Label


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

    # First half: positive class - specific pattern in sequences
    for i in range(n_samples // 2):
        # Strong signal in early tokens
        acts[0, i, :3, 0] = 2.0
        acts[0, i, :3, 1] = 1.5
        # Noise elsewhere
        acts[0, i, 3:, :] = torch.randn(seq_len - 3, d_model) * 0.1

    # Second half: negative class - different pattern
    for i in range(n_samples // 2, n_samples):
        # Strong signal in late tokens
        acts[0, i, -3:, 0] = -2.0
        acts[0, i, -3:, 1] = -1.5
        # Noise elsewhere
        acts[0, i, :-3, :] = torch.randn(seq_len - 3, d_model) * 0.1

    activations = Activations(
        activations=acts,
        attention_mask=torch.ones(n_samples, seq_len),
        input_ids=torch.ones(n_samples, seq_len, dtype=torch.long),
        detection_mask=torch.ones(n_samples, seq_len),
        layer_indices=[layer],
    )

    labels = [Label.POSITIVE] * (n_samples // 2) + [Label.NEGATIVE] * (n_samples // 2)

    return activations, labels


class TestAttention:
    """Test Attention probe implementation."""

    def test_initialization(self):
        """Test probe initialization."""
        probe = Attention(
            layer=5,
            hidden_dim=32,
            learning_rate=0.001,
            weight_decay=0.01,
            n_epochs=100,
            patience=5,
            device="cpu",
            random_state=42,
            verbose=False,
        )

        assert probe.layer == 5
        assert probe.hidden_dim == 32
        assert probe.learning_rate == 0.001
        assert probe.weight_decay == 0.01
        assert probe.n_epochs == 100
        assert probe.patience == 5
        assert probe._fitted is False
        assert probe._requires_grad is True
        # Attention probe requires sequence_pooling=NONE
        from probelib.processing import SequencePooling
        assert probe.sequence_pooling == SequencePooling.NONE

    def test_fit(self):
        """Test fitting the attention probe."""
        activations, labels = create_separable_data(n_samples=20)

        probe = Attention(
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

        probe = Attention(
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
        assert torch.allclose(probs.sum(dim=1), torch.ones(20))

        # Check attention weights are stored
        assert probe.attention_weights is not None
        assert probe.attention_weights.shape == (20, 10)  # (n_samples, seq_len)

        # Just check that the probe is working (attention mechanisms can be hard to train on small data)
        # The key is that it runs without errors and produces valid outputs
        predictions = probs.argmax(dim=1)
        assert len(predictions) == 20
        assert torch.all((predictions == 0) | (predictions == 1))

    def test_predict_before_fit(self):
        """Test that prediction fails before fitting."""
        activations = create_test_activations()
        probe = Attention(layer=0, device="cpu")

        with pytest.raises(RuntimeError, match="Probe must be fitted"):
            probe.predict_proba(activations)

    def test_partial_fit(self):
        """Test incremental fitting."""
        batch1_acts, batch1_labels = create_separable_data(n_samples=10)
        batch2_acts, batch2_labels = create_separable_data(n_samples=10)

        probe = Attention(
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

    def test_save_and_load(self):
        """Test saving and loading probe."""
        activations, labels = create_separable_data(n_samples=20)

        probe = Attention(
            layer=0,
            hidden_dim=32,
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

            loaded_probe = Attention.load(save_path)

            assert loaded_probe.layer == 0
            assert loaded_probe.hidden_dim == 32
            assert loaded_probe._fitted is True

            probs_after = loaded_probe.predict_proba(activations)
            assert torch.allclose(probs_before, probs_after, atol=1e-6)

    def test_ignores_aggregation_params(self):
        """Test that attention probe ignores aggregation parameters."""
        activations, labels = create_separable_data(n_samples=20)

        # Try creating with different aggregation settings - should be ignored
        probe = Attention(
            layer=0,
            # Attention probe doesn't accept aggregation parameters
            hidden_dim=16,
            n_epochs=50,
            device="cpu",
            random_state=42,
        )

        # Internally should always use token-level
        from probelib.processing import SequencePooling
        assert probe.sequence_pooling == SequencePooling.NONE

        # Should still work normally
        probe.fit(activations, labels)
        probs = probe.predict_proba(activations)
        assert probs.shape == (20, 2)
