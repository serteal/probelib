"""Tests for Logistic implementation."""

import tempfile
from pathlib import Path

import pytest
import torch

from probelib.probes.logistic import Logistic
from probelib.processing.activations import (
    ActivationIterator,
    Activations,
)
from probelib.types import Label
from probelib.processing import SequencePooling


class MockActivationIterator(ActivationIterator):
    """Mock iterator for testing."""

    def __init__(self, batches):
        self.batches = batches
        self._layers = [0]  # Single layer

    def __iter__(self):
        return iter(self.batches)

    def __len__(self):
        return len(self.batches)

    @property
    def layers(self):
        return self._layers

    # No longer need with_labels method - use zip instead


def create_test_activations(n_samples=10, seq_len=20, d_model=16, layer=0):
    """Create test activations with controlled properties."""
    # Create activations with some structure for testing
    acts = torch.randn(1, n_samples, seq_len, d_model)

    # Create detection mask with varying lengths
    detection_mask = torch.zeros(n_samples, seq_len)
    for i in range(n_samples):
        # Each sample has different number of valid tokens
        valid_len = 5 + i % 10
        detection_mask[i, :valid_len] = 1

    return Activations(
        activations=acts,
        attention_mask=torch.ones(n_samples, seq_len),
        input_ids=torch.ones(n_samples, seq_len, dtype=torch.long),
        detection_mask=detection_mask,
        layer_indices=[layer],
    )


def create_separable_data(n_samples=20, seq_len=10, d_model=8, layer=0):
    """Create linearly separable data for testing."""
    # Create two clear clusters
    acts = torch.zeros(1, n_samples, seq_len, d_model)

    # First half: positive class
    acts[0, : n_samples // 2, :, 0] = 1.0  # High value in first dimension
    acts[0, : n_samples // 2, :, 1:] = (
        torch.randn(n_samples // 2, seq_len, d_model - 1) * 0.1
    )

    # Second half: negative class
    acts[0, n_samples // 2 :, :, 0] = -1.0  # Low value in first dimension
    acts[0, n_samples // 2 :, :, 1:] = (
        torch.randn(n_samples // 2, seq_len, d_model - 1) * 0.1
    )

    detection_mask = torch.ones(n_samples, seq_len)

    activations = Activations(
        activations=acts,
        attention_mask=torch.ones(n_samples, seq_len),
        input_ids=torch.ones(n_samples, seq_len, dtype=torch.long),
        detection_mask=detection_mask,
        layer_indices=[layer],
    )

    # Create corresponding labels
    labels = [Label.POSITIVE] * (n_samples // 2) + [Label.NEGATIVE] * (n_samples // 2)

    return activations, labels


class TestLogistic:
    """Test Logistic probe implementation."""

    def test_initialization(self):
        """Test probe initialization."""
        probe = Logistic(
            layer=5,
            sequence_pooling=SequencePooling.MEAN,
            l2_penalty=0.1,
            device="cpu",
            random_state=42,
            verbose=False,
        )

        assert probe.layer == 5
        assert probe.sequence_pooling == SequencePooling.MEAN
        # score_aggregation no longer exists
        assert probe.l2_penalty == 0.1
        assert probe.device == "cpu"
        assert probe.random_state == 42
        assert probe._fitted is False

    def test_fit_with_aggregation(self):
        """Test fitting with sequence aggregation."""
        activations, labels = create_separable_data(n_samples=20)

        probe = Logistic(
            layer=0,
            sequence_pooling=SequencePooling.MEAN,
            device="cpu",
        )

        fitted_probe = probe.fit(activations, labels)

        assert fitted_probe is probe  # Should return self
        assert probe._fitted is True
        assert probe._network is not None
        assert probe._network.linear.weight.shape == (1, 8)  # [1, d_model]

    def test_fit_token_level(self):
        """Test fitting at token level."""
        activations, labels = create_separable_data(n_samples=10, seq_len=5)

        probe = Logistic(
            layer=0,
            sequence_pooling=SequencePooling.NONE,  # Token-level training with score aggregation
            device="cpu",
        )

        probe.fit(activations, labels)

        assert probe._fitted is True
        assert probe._network is not None

    def test_predict_proba(self):
        """Test probability prediction."""
        activations, labels = create_separable_data(n_samples=20)

        probe = Logistic(
            layer=0, sequence_pooling=SequencePooling.MEAN, device="cpu", random_state=42
        )
        probe.fit(activations, labels)

        # Predict on same data
        probs = probe.predict_proba(activations)

        assert probs.shape == (20, 2)
        assert torch.all(probs >= 0) and torch.all(probs <= 1)
        assert torch.allclose(probs.sum(dim=1), torch.ones(20))

        # Should predict high probability for correct classes
        pos_probs = probs[:10, 1]  # P(y=1) for positive samples
        neg_probs = probs[10:, 0]  # P(y=0) for negative samples

        assert pos_probs.mean() > 0.65  # Should be confident on positive
        assert neg_probs.mean() > 0.65  # Should be confident on negative

    def test_predict_before_fit(self):
        """Test that prediction fails before fitting."""
        activations = create_test_activations()
        probe = Logistic(layer=0, sequence_pooling=SequencePooling.MEAN, device="cpu")

        with pytest.raises(RuntimeError, match="Probe must be fitted"):
            probe.predict_proba(activations)

    def test_partial_fit(self):
        """Test incremental fitting."""
        # Create data in batches
        batch1_acts, batch1_labels = create_separable_data(n_samples=10)
        batch2_acts, batch2_labels = create_separable_data(n_samples=10)

        probe = Logistic(
            layer=0, sequence_pooling=SequencePooling.MEAN, device="cpu", random_state=42
        )

        # Fit on first batch
        probe.partial_fit(batch1_acts, batch1_labels)
        assert probe._fitted is True

        # Update with second batch
        probe.partial_fit(batch2_acts, batch2_labels)
        assert probe._fitted is True

        # Should still make reasonable predictions
        probs = probe.predict_proba(batch1_acts)
        assert probs.shape == (10, 2)

    def test_fit_with_iterator(self):
        """Test fitting with an ActivationIterator."""
        # Create batches
        batch1_acts, _ = create_separable_data(n_samples=10)
        batch2_acts, _ = create_separable_data(n_samples=10)

        # Create iterator
        iterator = MockActivationIterator([batch1_acts, batch2_acts])

        # All labels
        labels = [Label.POSITIVE] * 10 + [Label.NEGATIVE] * 10

        probe = Logistic(layer=0, sequence_pooling=SequencePooling.MEAN, device="cpu")
        probe.fit(iterator, labels)

        assert probe._fitted is True

    def test_predict_with_iterator(self):
        """Test prediction with an ActivationIterator."""
        # Train probe first
        train_acts, train_labels = create_separable_data(n_samples=20)
        probe = Logistic(
            layer=0, sequence_pooling=SequencePooling.MEAN, device="cpu", random_state=42
        )
        probe.fit(train_acts, train_labels)

        # Create test iterator
        test_batch1, _ = create_separable_data(n_samples=10)
        test_batch2, _ = create_separable_data(n_samples=10)
        test_iterator = MockActivationIterator([test_batch1, test_batch2])

        # Predict
        probs = probe.predict_proba(test_iterator)

        assert probs.shape == (20, 2)
        assert torch.all(probs >= 0) and torch.all(probs <= 1)

    def test_different_aggregations(self):
        """Test different aggregation methods."""
        activations, labels = create_separable_data(n_samples=20)

        for method in ["mean", "max", "last_token"]:
            probe = Logistic(
                layer=0,
                sequence_pooling=SequencePooling(method),
                device="cpu",
                random_state=42,
            )

            probe.fit(activations, labels)
            probs = probe.predict_proba(activations)

            assert probs.shape == (20, 2)
            assert torch.all(probs >= 0) and torch.all(probs <= 1)

    def test_save_and_load(self):
        """Test saving and loading probe."""
        activations, labels = create_separable_data(n_samples=20)

        # Train probe
        probe = Logistic(
            layer=0,  # Match the layer in test data
            sequence_pooling=SequencePooling.MAX,
            l2_penalty=0.5,
            device="cpu",
            random_state=42,
        )
        probe.fit(activations, labels)

        # Get predictions before saving
        probs_before = probe.predict_proba(activations)

        # Save and load
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "probe.pt"
            probe.save(save_path)

            loaded_probe = Logistic.load(save_path)

            # Check attributes preserved
            assert loaded_probe.layer == 0
            assert loaded_probe.sequence_pooling == SequencePooling.MAX
            # score_aggregation no longer exists
            assert loaded_probe.l2_penalty == 0.5
            assert loaded_probe._fitted is True

            # Check predictions are the same
            probs_after = loaded_probe.predict_proba(activations)
            assert torch.allclose(probs_before, probs_after, atol=1e-6)

    def test_save_unfitted_probe(self):
        """Test that saving unfitted probe raises error."""
        probe = Logistic(layer=0, sequence_pooling=SequencePooling.MEAN, device="cpu")

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "probe.pt"
            with pytest.raises(RuntimeError, match="Cannot save unfitted probe"):
                probe.save(save_path)

    def test_load_with_different_device(self):
        """Test loading probe to different device."""
        activations, labels = create_separable_data(n_samples=20)

        # Train on CPU
        probe = Logistic(layer=0, sequence_pooling=SequencePooling.MEAN, device="cpu")
        probe.fit(activations, labels)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "probe.pt"
            probe.save(save_path)

            # Load to different device (still CPU in tests)
            loaded_probe = Logistic.load(save_path, device="cpu")
            assert loaded_probe.device == "cpu"

    def test_validation_errors(self):
        """Test that appropriate errors are raised."""
        activations = create_test_activations(n_samples=10)

        # GPU version doesn't check for single class
        # (handled by PyTorch loss function)

        # Test with wrong layer
        probe = Logistic(layer=5, sequence_pooling=SequencePooling.MEAN, device="cpu")
        with pytest.raises(ValueError, match="Layer 5 not found"):
            probe.fit(activations, [Label.POSITIVE] * 5 + [Label.NEGATIVE] * 5)

    def test_partial_fit_after_fit(self):
        """Test that partial_fit after regular fit works."""
        batch1_acts, batch1_labels = create_separable_data(n_samples=10)
        batch2_acts, batch2_labels = create_separable_data(n_samples=10)

        probe = Logistic(layer=0, sequence_pooling=SequencePooling.MEAN, device="cpu")

        # First do regular fit
        probe.fit(batch1_acts, batch1_labels)

        # Then do partial fit - GPU version allows this
        probe.partial_fit(batch2_acts, batch2_labels)
        assert probe._fitted is True

    def test_token_level_prediction_aggregation(self):
        """Test that token-level predictions are properly aggregated."""
        activations, labels = create_separable_data(n_samples=10, seq_len=5)

        # Test each aggregation method
        for method in ["mean", "max", "last_token"]:
            probe = Logistic(
                layer=0,
                sequence_pooling=SequencePooling.NONE,  # Token-level training
                device="cpu",
                random_state=42,
            )

            probe.fit(activations, labels)
            probs = probe.predict_proba(activations)

            # Should get sample-level predictions even though trained on tokens
            assert probs.shape == (10, 2)
            assert torch.all(probs >= 0) and torch.all(probs <= 1)
            assert torch.allclose(probs.sum(dim=1), torch.ones(10))

