"""
Test that probes correctly handle empty batches when no tokens are detected.
This covers the bug where probes would fail when using score_aggregation with empty batches.
"""

import pytest
import torch

from probelib.probes import MLP, Logistic
from probelib.processing.activations import Activations
from probelib.types import Label
from probelib.processing import SequencePooling


class TestEmptyBatchHandling:
    """Test that probes handle empty batches gracefully."""

    @pytest.fixture
    def empty_activations(self):
        """Create activations with no detected tokens."""
        batch_size = 2
        seq_len = 50
        d_model = 768

        # Create activations where detection_mask is all False
        return Activations.from_tensor(
            activations=torch.randn(1, batch_size, seq_len, d_model),
            attention_mask=torch.ones(batch_size, seq_len),
            input_ids=torch.randint(0, 1000, (batch_size, seq_len)),
            detection_mask=torch.zeros(
                batch_size, seq_len, dtype=torch.bool
            ),  # No detected tokens!
            layer_indices=[12],
        )

    @pytest.fixture
    def normal_activations(self):
        """Create activations with some detected tokens."""
        batch_size = 2
        seq_len = 50
        d_model = 768

        # Create activations with some detected tokens
        detection_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
        detection_mask[0, 10:20] = True  # 10 detected tokens in first sample
        detection_mask[1, 5:15] = True  # 10 detected tokens in second sample

        return Activations.from_tensor(
            activations=torch.randn(1, batch_size, seq_len, d_model),
            attention_mask=torch.ones(batch_size, seq_len),
            input_ids=torch.randint(0, 1000, (batch_size, seq_len)),
            detection_mask=detection_mask,
            layer_indices=[12],
        )

    def test_logistic_probe_empty_batch(self, empty_activations, normal_activations):
        """Test that LogisticProbe handles empty batches in partial_fit."""
        probe = Logistic(layer=12, sequence_pooling=SequencePooling.NONE, verbose=True)
        labels = [Label.POSITIVE, Label.NEGATIVE]

        # First fit with normal batch to initialize
        probe.partial_fit(normal_activations, labels)
        assert probe._fitted, "Probe should be fitted after normal batch"

        # Then try with empty batch - should not crash
        probe.partial_fit(empty_activations, labels)
        assert probe._fitted, "Probe should remain fitted"

        # Fit another normal batch to ensure it still works
        probe.partial_fit(normal_activations, labels)
        assert probe._fitted, "Probe should still be fitted"

    def test_logistic_probe_all_empty_batches(self, empty_activations):
        """Test LogisticProbe when all batches are empty."""
        probe = Logistic(layer=12, sequence_pooling=SequencePooling.NONE, verbose=True)
        labels = [Label.POSITIVE, Label.NEGATIVE]

        # Try to fit with only empty batches
        probe.partial_fit(empty_activations, labels)
        assert not probe._fitted, "Probe should not be fitted with only empty batches"

        probe.partial_fit(empty_activations, labels)
        assert not probe._fitted, "Probe should still not be fitted"

    def test_mlp_probe_empty_batch(self, empty_activations, normal_activations):
        """Test that MLPProbe handles empty batches in partial_fit."""
        probe = MLP(layer=12, sequence_pooling=SequencePooling.NONE, verbose=True)
        labels = [Label.POSITIVE, Label.NEGATIVE]

        # First fit with normal batch to initialize
        probe.partial_fit(normal_activations, labels)
        assert probe._fitted, "Probe should be fitted after normal batch"
        assert probe._network is not None, "Network should be initialized"

        # Then try with empty batch - should not crash
        probe.partial_fit(empty_activations, labels)
        assert probe._fitted, "Probe should remain fitted"

        # Fit another normal batch to ensure it still works
        probe.partial_fit(normal_activations, labels)
        assert probe._fitted, "Probe should still be fitted"

    def test_mixed_batches_streaming(self, empty_activations, normal_activations):
        """Test streaming training with a mix of empty and non-empty batches."""
        probe = Logistic(layer=12, sequence_pooling=SequencePooling.NONE, verbose=False)

        # Simulate streaming training
        labels = [Label.POSITIVE, Label.NEGATIVE]

        # Batch 1: Normal
        probe.partial_fit(normal_activations, labels)
        assert probe._fitted

        # Batch 2: Empty
        probe.partial_fit(empty_activations, labels)
        assert probe._fitted

        # Batch 3: Normal
        probe.partial_fit(normal_activations, labels)
        assert probe._fitted

        # Should be able to predict after training
        probs = probe.predict_proba(normal_activations)
        assert probs.shape == (2, 2), (
            "Should return probabilities for 2 samples, 2 classes"
        )

    def test_token_level_extraction_with_no_detected_tokens(self):
        """Test that to_token_level handles samples with no detected tokens."""
        batch_size = 3
        seq_len = 50
        d_model = 768

        # Create mixed detection mask
        detection_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
        detection_mask[0, 10:20] = True  # First sample has tokens
        # Second sample has NO tokens (all False)
        detection_mask[2, 5:10] = True  # Third sample has tokens

        activations = Activations.from_tensor(
            activations=torch.randn(1, batch_size, seq_len, d_model),
            attention_mask=torch.ones(batch_size, seq_len),
            input_ids=torch.randint(0, 1000, (batch_size, seq_len)),
            detection_mask=detection_mask,
            layer_indices=[12],
        )

        # Extract token-level features
        features, tokens_per_sample = activations.to_token_level()

        # Check results
        assert features.shape[0] == 15, "Should have 10 + 0 + 5 = 15 tokens"
        assert tokens_per_sample.tolist() == [10, 0, 5], (
            "Tokens per sample should be [10, 0, 5]"
        )

    def test_probe_with_sequence_pooling_empty_samples(self):
        """Test sequence aggregation with some samples having no detected tokens."""
        batch_size = 3
        seq_len = 50
        d_model = 768

        # Create detection mask where middle sample has no detected tokens
        detection_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
        detection_mask[0, 10:20] = True  # First sample has tokens
        # Second sample has NO tokens
        detection_mask[2, 5:10] = True  # Third sample has tokens

        activations = Activations.from_tensor(
            activations=torch.randn(1, batch_size, seq_len, d_model),
            attention_mask=torch.ones(batch_size, seq_len),
            input_ids=torch.randint(0, 1000, (batch_size, seq_len)),
            detection_mask=detection_mask,
            layer_indices=[12],
        )

        # This should work even with empty middle sample
        aggregated = activations.aggregate(method="mean")
        assert aggregated.shape == (3, d_model), "Should have 3 samples"


class TestStreamingWorkflow:
    """Test the complete streaming workflow that was failing."""

    def test_streaming_with_doluschat_pattern(self):
        """Test the pattern that was failing with DolusChatDataset."""
        from transformers import AutoTokenizer

        from probelib.processing.tokenization import tokenize_dialogues
        from probelib.types import Message

        # Create dialogues similar to DolusChatDataset
        dialogues = [
            [
                Message("system", "You are an assistant"),
                Message("user", "Question"),
                Message("assistant", "Answer"),  # This should be detected
            ],
            [
                Message("system", "Another system prompt"),
                Message("user", "Another question"),
                Message("assistant", "Another answer"),  # This should be detected
            ],
        ]

        # Import masks
        from probelib import masks

        # Tokenize with assistant mask (comments say assistant should be detected)
        tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b-it")
        tokenized = tokenize_dialogues(
            tokenizer,
            dialogues,
            mask=masks.assistant(),  # Mark assistant messages
            device="cpu",
        )

        # Verify detection mask has values
        assert tokenized["detection_mask"].sum() > 0, "Should have detected tokens"

        # Create activations
        batch_size, seq_len = tokenized["input_ids"].shape
        d_model = 768

        activations = Activations.from_tensor(
            activations=torch.randn(1, batch_size, seq_len, d_model),
            attention_mask=tokenized["attention_mask"],
            input_ids=tokenized["input_ids"],
            detection_mask=tokenized["detection_mask"],
            layer_indices=[12],
        )

        # Test with token-level probe
        probe = Logistic(layer=12, sequence_pooling=SequencePooling.NONE)
        labels = [Label.POSITIVE, Label.NEGATIVE]

        # Extract features for score aggregation
        features = probe._prepare_features(activations)
        assert features.shape[0] > 0, "Should extract some features"

        # Partial fit should work
        probe.partial_fit(activations, labels)
        assert probe._fitted or features.shape[0] == 0, (
            "Should be fitted if there were features"
        )
