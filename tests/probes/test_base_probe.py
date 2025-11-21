"""Tests for BaseProbe abstract class."""

import pytest
import torch

from probelib.probes.base import BaseProbe
from probelib.processing.activations import Activations
from probelib.types import Label
from probelib.processing import SequencePooling


class ConcreteProbe(BaseProbe):
    """Concrete implementation of BaseProbe for testing."""

    def fit(self, X, y):
        """Mock fit implementation."""
        self._fitted = True
        return self

    def predict_proba(self, X):
        """Mock predict_proba implementation."""
        if not self._fitted:
            raise RuntimeError("Probe not fitted")
        # Return dummy probabilities
        if hasattr(X, "batch_size"):
            n_samples = X.batch_size
        else:
            n_samples = 10
        probs = torch.ones(n_samples, 2) * 0.5
        return probs

    def partial_fit(self, X, y):
        """Mock partial_fit implementation."""
        self._fitted = True
        return self

    def save(self, path):
        """Mock save implementation."""
        pass

    @classmethod
    def load(cls, path, device=None):
        """Mock load implementation."""
        return cls(layer=0)


class TestBaseProbe:
    """Test BaseProbe abstract class and helper methods."""

    def test_initialization(self):
        """Test probe initialization."""
        probe = ConcreteProbe(
            layer=5,
            sequence_pooling=SequencePooling.MEAN,
            device="cpu",
            random_state=42,
            verbose=False,
        )

        assert probe.layer == 5
        assert probe.sequence_pooling == SequencePooling.MEAN
        # score_aggregation no longer exists
        assert probe.device == "cpu"
        assert probe.random_state == 42
        assert probe.verbose is False
        assert probe._fitted is False
        assert probe._requires_grad is False

    def test_invalid_aggregation(self):
        """Test that invalid aggregation method raises error."""
        # Pooling validation is handled by enum now

    def test_prepare_features_aggregate(self):
        """Test feature preparation with aggregation."""
        probe = ConcreteProbe(layer=0, sequence_pooling=SequencePooling.MEAN)

        # Create test activations
        acts = torch.randn(1, 4, 8, 16)  # 1 layer, 4 batch, 8 seq, 16 dim
        detection_mask = torch.ones(4, 8)

        activations = Activations.from_tensor(
            activations=acts,
            attention_mask=torch.ones(4, 8),
            input_ids=torch.ones(4, 8),
            detection_mask=detection_mask,
            layer_indices=[0],
        )

        features = probe._prepare_features(activations)

        assert features.shape == (4, 16)  # Aggregated to (batch_size, d_model)
        assert not hasattr(probe, "_tokens_per_sample")  # Not set when aggregating

    def test_prepare_features_token_level(self):
        """Test feature preparation without aggregation."""
        probe = ConcreteProbe(layer=0, sequence_pooling=SequencePooling.NONE)

        # Create test activations
        acts = torch.randn(1, 2, 8, 16)
        detection_mask = torch.tensor(
            [
                [1, 1, 1, 0, 0, 0, 0, 0],  # 3 valid tokens
                [1, 1, 0, 0, 0, 0, 0, 0],  # 2 valid tokens
            ]
        ).float()

        activations = Activations.from_tensor(
            activations=acts,
            attention_mask=torch.ones(2, 8),
            input_ids=torch.ones(2, 8),
            detection_mask=detection_mask,
            layer_indices=[0],
        )

        features = probe._prepare_features(activations)

        assert features.shape == (5, 16)  # 5 total tokens
        assert hasattr(probe, "_tokens_per_sample")
        assert probe._tokens_per_sample.tolist() == [3, 2]

    def test_prepare_features_wrong_layer(self):
        """Test that prepare_features filters to correct layer."""
        probe = ConcreteProbe(layer=5, sequence_pooling=SequencePooling.MEAN)

        # Create activations with multiple layers
        acts = torch.randn(3, 2, 8, 16)
        activations = Activations.from_tensor(
            activations=acts,
            attention_mask=torch.ones(2, 8),
            input_ids=torch.ones(2, 8),
            detection_mask=torch.ones(2, 8),
            layer_indices=[0, 5, 10],
        )

        features = probe._prepare_features(activations)

        # Should extract layer 5 only
        assert features.shape == (2, 16)

    def test_prepare_labels_from_list(self):
        """Test label preparation from list."""
        probe = ConcreteProbe(layer=0)

        # Test with Label enum
        labels = [Label.POSITIVE, Label.NEGATIVE, Label.POSITIVE]
        y = probe._prepare_labels(labels)
        assert y.tolist() == [1, 0, 1]

        # Test with raw values
        labels = [0, 1, 1, 0]
        y = probe._prepare_labels(labels)
        assert y.tolist() == [0, 1, 1, 0]

    def test_prepare_labels_from_tensor(self):
        """Test label preparation from tensor."""
        probe = ConcreteProbe(layer=0, device="cpu")  # Explicitly set device to CPU

        labels = torch.tensor([0, 1, 1, 0])
        y = probe._prepare_labels(labels)
        assert torch.equal(y, labels)

    def test_prepare_labels_invalid_classes(self):
        """Test that non-binary labels raise error."""
        probe = ConcreteProbe(layer=0)

        # Test with invalid class values
        labels = [0, 1, 2]
        with pytest.raises(ValueError, match="Only binary classification is supported"):
            probe._prepare_labels(labels)

        labels = [-1, 0, 1]
        with pytest.raises(ValueError, match="Only binary classification is supported"):
            probe._prepare_labels(labels)

    def test_repr(self):
        """Test string representation."""
        probe = ConcreteProbe(layer=5, sequence_pooling=SequencePooling.NONE)

        repr_str = repr(probe)
        assert "ConcreteProbe" in repr_str
        assert "layer=5" in repr_str
        assert "sequence_pooling=NONE" in repr_str
        assert "not fitted" in repr_str

        # After fitting
        probe._fitted = True
        repr_str = repr(probe)
        assert "fitted" in repr_str
        assert "not fitted" not in repr_str

    def test_requires_grad_property(self):
        """Test requires_grad property."""
        probe = ConcreteProbe(layer=0)
        assert probe.requires_grad is False

        probe._requires_grad = True
        assert probe.requires_grad is True

    def test_abstract_methods_required(self):
        """Test that abstract methods must be implemented."""

        class IncompleteProbe(BaseProbe):
            """Probe missing required methods."""

            pass

        # Should not be able to instantiate
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            IncompleteProbe(layer=0)
