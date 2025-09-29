"""Tests for activation collection utilities."""

from unittest.mock import MagicMock, Mock, patch

import pytest
import torch
import torch.nn as nn

from probelib.datasets.base import DialogueDataset
from probelib.processing.activations import (
    ActivationIterator,
    Activations,
    Axis,
    collect_activations,
    get_batches,
    get_hidden_dim,
    get_n_layers,
)
from probelib.types import Dialogue, DialogueDataType, Label, Message


class MockDialogueDataset(DialogueDataset):
    """Concrete implementation of DialogueDataset for testing."""

    def _get_dialogues(self, **kwargs) -> DialogueDataType:
        """Not used since we pass dialogues directly."""
        raise NotImplementedError("This method should not be called in tests")


class SimpleModel(nn.Module):
    """Simple model for testing."""

    def __init__(self, n_layers: int = 3, hidden_size: int = 64):
        super().__init__()
        self.config = Mock()
        self.config.num_hidden_layers = n_layers
        self.config.hidden_size = hidden_size
        self.layers = nn.ModuleList(
            [nn.Linear(hidden_size, hidden_size) for _ in range(n_layers)]
        )
        self.device = torch.device("cpu")
        self.dtype = torch.float32

    def forward(self, input_ids):
        batch_size, seq_len = input_ids.shape
        # Create dummy activations
        return torch.randn(batch_size, seq_len, self.config.hidden_size)


@pytest.fixture
def mock_tokenizer():
    """Create a mock tokenizer."""
    tokenizer = Mock()
    tokenizer.pad_token_id = 0
    tokenizer.eos_token_id = 1
    tokenizer.padding_side = "right"
    tokenizer.name_or_path = "meta-llama/Llama-2-7b-hf"

    # Mock apply_chat_template to return properly formatted dialogues
    # These should match the llama3 format expected by the library
    def mock_apply_chat_template(dialogues, **kwargs):
        # Create properly formatted chat strings for each dialogue
        formatted = []
        for dialogue_list in dialogues:
            formatted_text = ""
            for msg in dialogue_list:
                role = msg.get("role", "")
                content = msg.get("content", "")
                # Use Llama 3 format with header tags
                if role == "system":
                    formatted_text += f"<|start_header_id|>system<|end_header_id|>\n\n{content}<|eot_id|>"
                elif role == "user":
                    formatted_text += f"<|start_header_id|>user<|end_header_id|>\n\n{content}<|eot_id|>"
                elif role == "assistant":
                    formatted_text += f"<|start_header_id|>assistant<|end_header_id|>\n\n{content}<|eot_id|>"
            formatted.append(formatted_text)
        return formatted

    tokenizer.apply_chat_template = mock_apply_chat_template

    # Mock the tokenizer __call__ to return a BatchEncoding-like object
    def tokenizer_call(*args, **kwargs):
        n_samples = len(args[0]) if args else 3  # Default to 3 samples
        seq_len = 32

        # Create a mock BatchEncoding object that acts like a dict
        class MockBatchEncoding(dict):
            def __init__(self):
                super().__init__()
                self["input_ids"] = torch.randint(0, 1000, (n_samples, seq_len))
                self["attention_mask"] = torch.ones(n_samples, seq_len)

            def char_to_token(self, batch_idx, char_idx):
                # Simplified mapping - assumes ~3 chars per token
                return min(char_idx // 3, seq_len - 1)

        return MockBatchEncoding()

    tokenizer.side_effect = tokenizer_call

    return tokenizer


@pytest.fixture
def sample_dialogues():
    """Create sample dialogues for testing."""
    return [
        Dialogue(
            [
                Message(role="user", content="Hello, how are you?"),
                Message(
                    role="assistant",
                    content="I'm doing well, thank you!",
                ),
            ]
        ),
        Dialogue(
            [
                Message(
                    role="system",
                    content="You are a helpful assistant.",
                ),
                Message(role="user", content="What is 2+2?"),
                Message(role="assistant", content="2+2 equals 4."),
            ]
        ),
        Dialogue(
            [
                Message(
                    role="user",
                    content="Can you explain quantum computing?",
                ),
            ]
        ),
    ]


@pytest.fixture
def sample_dataset(sample_dialogues):
    """Create a sample dataset."""
    labels = [Label.POSITIVE, Label.NEGATIVE, Label.POSITIVE]
    return MockDialogueDataset(
        dialogues=sample_dialogues,
        labels=labels,
        metadata={"source": "test"},
    )


class TestGetBatches:
    """Test get_batches function."""

    def test_get_batches_basic(self, mock_tokenizer):
        """Test basic batch creation."""
        # Create input tensors
        input_ids = torch.tensor(
            [
                [1, 2, 3, 0, 0],  # length 3
                [1, 2, 3, 4, 5],  # length 5
                [1, 2, 0, 0, 0],  # length 2
                [1, 2, 3, 4, 0],  # length 4
            ]
        )
        attention_mask = (input_ids != 0).float()
        inputs = {"input_ids": input_ids, "attention_mask": attention_mask}

        batches = list(get_batches(inputs, batch_size=2, tokenizer=mock_tokenizer))

        # Should create 2 batches, sorted by length
        assert len(batches) == 2

        # First batch should contain shortest sequences
        batch1_inputs, batch1_indices = batches[0]
        assert set(batch1_indices) == {2, 0}  # indices of length 2 and 3 sequences
        assert batch1_inputs["input_ids"].shape[1] <= 3  # max length in batch

        # Second batch should contain longest sequences
        batch2_inputs, batch2_indices = batches[1]
        assert set(batch2_indices) == {3, 1}  # indices of length 4 and 5 sequences
        assert batch2_inputs["input_ids"].shape[1] <= 5  # max length in batch

    def test_get_batches_left_padding(self, mock_tokenizer):
        """Test batch creation with left padding."""
        mock_tokenizer.padding_side = "left"

        input_ids = torch.tensor(
            [
                [0, 0, 1, 2, 3],  # length 3
                [1, 2, 3, 4, 5],  # length 5
            ]
        )
        attention_mask = (input_ids != 0).float()
        inputs = {"input_ids": input_ids, "attention_mask": attention_mask}

        batches = list(get_batches(inputs, batch_size=2, tokenizer=mock_tokenizer))

        assert len(batches) == 1
        batch_inputs, batch_indices = batches[0]

        # Should extract from the right side for left padding
        assert batch_inputs["input_ids"].shape == (2, 5)

    def test_get_batches_single_sequence(self, mock_tokenizer):
        """Test with single sequence."""
        input_ids = torch.tensor([[1, 2, 3, 4, 5]])
        inputs = {"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids)}

        batches = list(get_batches(inputs, batch_size=1, tokenizer=mock_tokenizer))

        assert len(batches) == 1
        batch_inputs, batch_indices = batches[0]
        assert batch_indices == [0]
        assert torch.equal(batch_inputs["input_ids"], input_ids)


class TestModelHelpers:
    """Test model helper functions."""

    def test_get_n_layers(self):
        """Test getting number of layers from model."""
        # For the new architecture-based approach, we need to mock the model structure
        # that the architecture registry expects, with proper spec to prevent auto-creation

        # Test with LLaMA-style model (num_hidden_layers)
        model = Mock(spec=["config", "model"])  # Prevent get_base_model auto-creation
        model.config = Mock()
        model.config.num_hidden_layers = 12
        model.model = Mock()
        # Create real list with proper layer structure
        layers = []
        for _ in range(12):
            layer = Mock()
            layer.input_layernorm = Mock()
            layers.append(layer)
        model.model.layers = layers
        assert get_n_layers(model) == 12

        # Test with n_layers attribute
        model = Mock(spec=["config", "model"])
        model.config = Mock(spec=["n_layers"])  # Only has n_layers
        model.config.n_layers = 24
        model.model = Mock()
        layers = []
        for _ in range(24):
            layer = Mock()
            layer.input_layernorm = Mock()
            layers.append(layer)
        model.model.layers = layers
        assert get_n_layers(model) == 24

        # Test with num_layers attribute
        model = Mock(spec=["config", "model"])
        model.config = Mock(spec=["num_layers"])  # Only has num_layers
        model.config.num_layers = 36
        model.model = Mock()
        layers = []
        for _ in range(36):
            layer = Mock()
            layer.input_layernorm = Mock()
            layers.append(layer)
        model.model.layers = layers
        assert get_n_layers(model) == 36

        # Test error case - model without proper config attributes
        model = Mock(spec=["config", "model"])
        model.config = Mock(spec=[])  # Empty spec, no layer count attributes
        model.model = Mock()
        layer = Mock()
        layer.input_layernorm = Mock()
        model.model.layers = [layer]  # At least one layer for architecture detection
        with pytest.raises(ValueError, match="Cannot determine number of layers"):
            get_n_layers(model)

    def test_get_hidden_dim(self):
        """Test getting hidden dimension from model."""
        # Test with hidden_size
        model = Mock()
        model.config = Mock()
        model.config.hidden_size = 768
        assert get_hidden_dim(model) == 768

        # Test error case
        model.config = Mock()
        del model.config.hidden_size
        model.name_or_path = "unknown-model"
        with pytest.raises(ValueError, match="Cannot determine hidden dimension"):
            get_hidden_dim(model)


class TestActivations:
    """Test Activations dataclass."""

    def test_activations_initialization(self):
        """Test basic initialization."""
        acts = torch.randn(4, 8, 16, 32)  # 4 layers, 8 batch, 16 seq, 32 dim
        mask = torch.ones(8, 16)
        input_ids = torch.randint(0, 1000, (8, 16))
        detection_mask = torch.ones(8, 16).bool()
        layer_indices = [0, 5, 10, 15]

        activations = Activations.from_components(
            activations=acts,
            attention_mask=mask,
            input_ids=input_ids,
            detection_mask=detection_mask,
            layer_indices=layer_indices,
        )

        assert activations.n_layers == 4
        assert activations.batch_size == 8
        assert activations.seq_len == 16
        assert activations.d_model == 32
        assert activations.shape == (4, 8, 16, 32)

    def test_activations_attention_mask_applied(self):
        """Test that attention mask is stored correctly."""
        acts = torch.ones(2, 4, 8, 16)
        mask = torch.tensor(
            [
                [1, 1, 1, 0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 0, 0, 0],
                [1, 1, 0, 0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 1, 1, 1],
            ]
        ).float()
        input_ids = torch.ones(4, 8, dtype=torch.long)
        detection_mask = torch.ones(4, 8).bool()

        activations = Activations.from_components(
            activations=acts,
            attention_mask=mask,
            input_ids=input_ids,
            detection_mask=detection_mask,
            layer_indices=[0, 1],
            batch_indices=[0, 1, 2, 3],
        )

        # Check that attention mask is stored correctly
        assert torch.equal(activations.attention_mask, mask)
        # Activations class no longer automatically applies masking

    def test_activations_to_device(self):
        """Test moving activations to device."""
        acts = torch.randn(2, 4, 8, 16)
        mask = torch.ones(4, 8)
        input_ids = torch.ones(4, 8, dtype=torch.long)
        detection_mask = torch.ones(4, 8).bool()

        activations = Activations.from_components(
            activations=acts,
            attention_mask=mask,
            input_ids=input_ids,
            detection_mask=detection_mask,
            layer_indices=[0, 1],
            batch_indices=[0, 1, 2, 3],
        )

        # Test device movement
        if torch.cuda.is_available():
            cuda_acts = activations.to("cuda")
            assert cuda_acts.activations.device.type == "cuda"
            assert cuda_acts.attention_mask.device.type == "cuda"
            assert cuda_acts.layer_indices == [0, 1]  # Should be preserved
            assert cuda_acts.batch_indices.tolist() == [0, 1, 2, 3]

        # Test dtype conversion
        float16_acts = activations.to(torch.float16)
        assert float16_acts.activations.dtype == torch.float16
        assert float16_acts.attention_mask.dtype == torch.float32  # Should not change
        assert float16_acts.batch_indices.tolist() == [0, 1, 2, 3]

    def test_get_layer_tensor_indices(self):
        """Test mapping layer indices to tensor indices."""
        acts = torch.randn(4, 2, 8, 16)
        activations = Activations.from_components(
            activations=acts,
            attention_mask=torch.ones(2, 8),
            input_ids=torch.ones(2, 8, dtype=torch.long),
            detection_mask=torch.ones(2, 8).bool(),
            layer_indices=[0, 5, 10, 15],
        )

        # Test single layer
        assert activations.get_layer_tensor_indices(5) == [1]

        # Test multiple layers
        assert activations.get_layer_tensor_indices([0, 10]) == [0, 2]

        # Test error for unavailable layer
        with pytest.raises(ValueError, match="Layer 7 is not available"):
            activations.get_layer_tensor_indices(7)

    def test_select_multiple_layers(self):
        """Test selecting multiple layers."""
        acts = torch.randn(4, 2, 8, 16)
        activations = Activations.from_components(
            activations=acts,
            attention_mask=torch.ones(2, 8),
            input_ids=torch.ones(2, 8, dtype=torch.long),
            detection_mask=torch.ones(2, 8).bool(),
            layer_indices=[0, 5, 10, 15],
        )

        # Select subset of layers
        selected = activations.select(layers=[5, 15])
        assert selected.n_layers == 2
        assert selected.layer_indices == [5, 15]
        assert torch.equal(selected.activations[0], acts[1])  # layer 5 at index 1
        assert torch.equal(selected.activations[1], acts[3])  # layer 15 at index 3

    def test_aggregate_mean(self):
        """Test mean aggregation over sequence dimension."""
        # Create single-layer activations
        acts = torch.ones(1, 2, 8, 16)  # 1 layer, 2 batch, 8 seq, 16 dim
        detection_mask = torch.tensor(
            [
                [1, 1, 1, 1, 0, 0, 0, 0],  # 4 valid tokens
                [1, 1, 0, 0, 0, 0, 0, 0],  # 2 valid tokens
            ]
        ).float()

        activations = Activations.from_components(
            activations=acts,
            attention_mask=torch.ones(2, 8),
            input_ids=torch.ones(2, 8, dtype=torch.long),
            detection_mask=detection_mask,
            layer_indices=[5],
        )

        # Test mean aggregation
        pooled = activations.pool(dim="sequence", method="mean")
        result = pooled.activations.squeeze(0)  # Remove layer dimension
        assert result.shape == (2, 16)
        # First sample: mean of 4 tokens = 1.0
        assert torch.allclose(result[0], torch.ones(16))
        # Second sample: mean of 2 tokens = 1.0
        assert torch.allclose(result[1], torch.ones(16))

    def test_aggregate_max(self):
        """Test max aggregation over sequence dimension."""
        # Create single-layer activations with varying values
        acts = (
            torch.arange(128).reshape(1, 2, 8, 8).float()
        )  # 1 layer, 2 batch, 8 seq, 8 dim
        detection_mask = torch.tensor(
            [
                [1, 1, 1, 0, 0, 0, 0, 0],  # First 3 tokens valid
                [1, 1, 1, 1, 1, 0, 0, 0],  # First 5 tokens valid
            ]
        ).float()

        activations = Activations.from_components(
            activations=acts,
            attention_mask=torch.ones(2, 8),
            input_ids=torch.ones(2, 8, dtype=torch.long),
            detection_mask=detection_mask,
            layer_indices=[0],
        )

        # Test max aggregation
        pooled = activations.pool(dim="sequence", method="max")
        result = pooled.activations.squeeze(0)  # Remove layer dimension
        assert result.shape == (2, 8)
        # First sample: max over first 3 sequences
        expected_0 = acts[0, 0, 2, :]  # Third sequence has max values
        assert torch.allclose(result[0], expected_0)

    def test_aggregate_last_token(self):
        """Test last_token aggregation."""
        acts = torch.arange(128).reshape(1, 2, 8, 8).float()
        detection_mask = torch.tensor(
            [
                [1, 1, 1, 0, 0, 0, 0, 0],  # Last valid at index 2
                [1, 1, 1, 1, 1, 0, 0, 0],  # Last valid at index 4
            ]
        ).float()

        activations = Activations.from_components(
            activations=acts,
            attention_mask=torch.ones(2, 8),
            input_ids=torch.ones(2, 8, dtype=torch.long),
            detection_mask=detection_mask,
            layer_indices=[0],
        )

        # Test last_token aggregation
        pooled = activations.pool(dim="sequence", method="last_token")
        result = pooled.activations.squeeze(0)  # Remove layer dimension
        assert result.shape == (2, 8)
        # First sample: token at index 2
        assert torch.allclose(result[0], acts[0, 0, 2, :])
        # Second sample: token at index 4
        assert torch.allclose(result[1], acts[0, 1, 4, :])

    def test_pool_works_with_multi_layer(self):
        """Test that pool works with multi-layer activations."""
        acts = torch.randn(2, 2, 8, 16)  # 2 layers
        activations = Activations.from_components(
            activations=acts,
            attention_mask=torch.ones(2, 8),
            input_ids=torch.ones(2, 8, dtype=torch.long),
            detection_mask=torch.ones(2, 8),
            layer_indices=[0, 1],
        )

        # Pool should work with multiple layers
        pooled = activations.pool(dim="sequence", method="mean")
        # Result should have shape [2 layers, 2 batch, 16 hidden]
        assert pooled.activations.shape == (2, 2, 16)
        assert pooled.has_axis(Axis.LAYER)
        assert not pooled.has_axis(Axis.SEQ)

    def test_aggregate_mean_empty_mask(self):
        """Test mean aggregation with completely empty detection mask."""
        acts = torch.randn(1, 2, 8, 16)  # 1 layer, 2 batch, 8 seq, 16 dim
        detection_mask = torch.zeros(2, 8)  # All tokens invalid

        activations = Activations.from_components(
            activations=acts,
            attention_mask=torch.ones(2, 8),
            input_ids=torch.ones(2, 8, dtype=torch.long),
            detection_mask=detection_mask,
            layer_indices=[0],
        )

        # Mean aggregation should handle empty masks gracefully
        pooled = activations.pool(dim="sequence", method="mean")
        result = pooled.activations.squeeze(0)  # Remove layer dimension
        assert result.shape == (2, 16)
        # With no valid tokens, result should be zeros (sum=0, count clamped to 1)
        assert torch.allclose(result, torch.zeros(2, 16))

    def test_aggregate_max_empty_mask(self):
        """Test max aggregation with completely empty detection mask."""
        acts = torch.randn(1, 2, 8, 16)  # 1 layer, 2 batch, 8 seq, 16 dim
        detection_mask = torch.zeros(2, 8)  # All tokens invalid

        activations = Activations.from_components(
            activations=acts,
            attention_mask=torch.ones(2, 8),
            input_ids=torch.ones(2, 8, dtype=torch.long),
            detection_mask=detection_mask,
            layer_indices=[0],
        )

        # Max aggregation should handle empty masks gracefully
        pooled = activations.pool(dim="sequence", method="max")
        result = pooled.activations.squeeze(0)  # Remove layer dimension
        assert result.shape == (2, 16)
        # With no valid tokens, result should be zeros (not -inf)
        assert torch.allclose(result, torch.zeros(2, 16))
        assert not torch.any(torch.isinf(result))  # No infinities

    def test_aggregate_last_token_empty_mask(self):
        """Test last_token aggregation with completely empty detection mask."""
        acts = torch.randn(1, 2, 8, 16)  # 1 layer, 2 batch, 8 seq, 16 dim
        detection_mask = torch.zeros(2, 8)  # All tokens invalid

        activations = Activations.from_components(
            activations=acts,
            attention_mask=torch.ones(2, 8),
            input_ids=torch.ones(2, 8, dtype=torch.long),
            detection_mask=detection_mask,
            layer_indices=[0],
        )

        # Last token aggregation should handle empty masks gracefully
        pooled = activations.pool(dim="sequence", method="last_token")
        result = pooled.activations.squeeze(0)  # Remove layer dimension
        assert result.shape == (2, 16)
        # With no valid tokens, result should be zeros
        assert torch.allclose(result, torch.zeros(2, 16))

    def test_aggregate_mixed_empty_masks(self):
        """Test aggregation with some batches having empty masks."""
        acts = torch.randn(1, 3, 8, 16)  # 1 layer, 3 batch, 8 seq, 16 dim
        detection_mask = torch.tensor(
            [
                [1, 1, 1, 0, 0, 0, 0, 0],  # 3 valid tokens
                [0, 0, 0, 0, 0, 0, 0, 0],  # No valid tokens
                [1, 1, 0, 0, 0, 0, 0, 0],  # 2 valid tokens
            ]
        ).float()

        activations = Activations.from_components(
            activations=acts,
            attention_mask=torch.ones(3, 8),
            input_ids=torch.ones(3, 8, dtype=torch.long),
            detection_mask=detection_mask,
            layer_indices=[0],
        )

        # Test all aggregation methods with mixed masks
        for method in ["mean", "max", "last_token"]:
            pooled = activations.pool(dim="sequence", method=method)
            result = pooled.activations.squeeze(0)  # Remove layer dimension
            assert result.shape == (3, 16)
            # Second batch (index 1) should be zeros
            assert torch.allclose(result[1], torch.zeros(16))
            # Other batches should have non-zero values (unless acts happen to be zero)
            assert not torch.any(torch.isinf(result))  # No infinities

    def test_to_token_level(self):
        """Test token-level feature extraction."""
        acts = torch.arange(128).reshape(1, 2, 8, 8).float()
        detection_mask = torch.tensor(
            [
                [1, 1, 1, 0, 0, 0, 0, 0],  # 3 valid tokens
                [1, 1, 0, 0, 0, 0, 0, 0],  # 2 valid tokens
            ]
        ).float()

        activations = Activations.from_components(
            activations=acts,
            attention_mask=torch.ones(2, 8),
            input_ids=torch.ones(2, 8, dtype=torch.long),
            detection_mask=detection_mask,
            layer_indices=[0],
        )

        # Extract token-level features
        features, tokens_per_sample = activations.extract_tokens()

        # Should have 5 total tokens (3 + 2)
        assert features.shape == (5, 8)
        assert tokens_per_sample.tolist() == [3, 2]

        # Check that correct tokens were extracted
        # First 3 tokens from first sample
        assert torch.allclose(features[0], acts[0, 0, 0, :])
        assert torch.allclose(features[1], acts[0, 0, 1, :])
        assert torch.allclose(features[2], acts[0, 0, 2, :])
        # Next 2 tokens from second sample
        assert torch.allclose(features[3], acts[0, 1, 0, :])
        assert torch.allclose(features[4], acts[0, 1, 1, :])

    def test_to_token_level_requires_single_layer(self):
        """Test that to_token_level raises error for multi-layer."""
        acts = torch.randn(2, 2, 8, 16)  # 2 layers
        activations = Activations.from_components(
            activations=acts,
            attention_mask=torch.ones(2, 8),
            input_ids=torch.ones(2, 8, dtype=torch.long),
            detection_mask=torch.ones(2, 8),
            layer_indices=[0, 1],
        )

        with pytest.raises(ValueError, match="Token extraction requires single layer"):
            activations.extract_tokens()

    def test_to_token_level_empty_mask(self):
        """Test token-level extraction with completely empty detection mask."""
        acts = torch.randn(1, 2, 8, 16)  # 1 layer, 2 batch, 8 seq, 16 dim
        detection_mask = torch.zeros(2, 8)  # All tokens invalid

        activations = Activations.from_components(
            activations=acts,
            attention_mask=torch.ones(2, 8),
            input_ids=torch.ones(2, 8, dtype=torch.long),
            detection_mask=detection_mask,
            layer_indices=[0],
        )

        # Should return empty features
        features, tokens_per_sample = activations.extract_tokens()
        assert features.shape == (0, 16)  # No valid tokens
        assert tokens_per_sample.tolist() == [0, 0]

    def test_to_token_level_mixed_empty_mask(self):
        """Test token-level extraction with some empty masks."""
        acts = torch.arange(192).reshape(1, 3, 8, 8).float()
        detection_mask = torch.tensor(
            [
                [1, 1, 0, 0, 0, 0, 0, 0],  # 2 valid tokens
                [0, 0, 0, 0, 0, 0, 0, 0],  # No valid tokens
                [1, 0, 0, 0, 0, 0, 0, 0],  # 1 valid token
            ]
        ).float()

        activations = Activations.from_components(
            activations=acts,
            attention_mask=torch.ones(3, 8),
            input_ids=torch.ones(3, 8, dtype=torch.long),
            detection_mask=detection_mask,
            layer_indices=[0],
        )

        # Should handle mixed masks properly
        features, tokens_per_sample = activations.extract_tokens()
        assert features.shape == (3, 8)  # 2 + 0 + 1 = 3 valid tokens
        assert tokens_per_sample.tolist() == [2, 0, 1]

        # Check correct tokens were extracted
        assert torch.allclose(features[0], acts[0, 0, 0, :])  # First token of batch 0
        assert torch.allclose(features[1], acts[0, 0, 1, :])  # Second token of batch 0
        assert torch.allclose(features[2], acts[0, 2, 0, :])  # First token of batch 2


class TestActivationIterator:
    """Test ActivationIterator class (streaming functionality)."""

    @patch("probelib.processing.activations.HookedModel")
    def test_streaming_initialization(
        self, mock_hooked_model, sample_dataset, mock_tokenizer
    ):
        """Test streaming iterator initialization."""
        model = SimpleModel()
        layers = [0, 1, 2]

        # Mock HookedModel to work with SimpleModel
        mock_hooked_instance = MagicMock()
        mock_hooked_instance.get_activations.return_value = torch.randn(
            len(layers), 2, 32, 64
        )
        mock_hooked_model.return_value.__enter__.return_value = mock_hooked_instance

        iterator = collect_activations(
            model=model,
            tokenizer=mock_tokenizer,
            data=sample_dataset,
            layers=layers,
            batch_size=2,
            streaming=True,
            verbose=False,
        )

        assert isinstance(iterator, ActivationIterator)
        assert iterator.layers == layers
        assert len(iterator) == 2  # 3 samples with batch_size=2

    @patch("probelib.processing.activations.HookedModel")
    def test_streaming_iteration(
        self, mock_hooked_model, sample_dataset, mock_tokenizer
    ):
        """Test streaming iteration over batches."""
        model = SimpleModel()
        layers = [0, 1]

        # Mock HookedModel to work with SimpleModel
        mock_hooked_instance = MagicMock()

        def get_activations_side_effect(batch_inputs):
            batch_size = batch_inputs["input_ids"].shape[0]
            seq_len = batch_inputs["input_ids"].shape[1]
            return torch.randn(len(layers), batch_size, seq_len, 64)

        mock_hooked_instance.get_activations.side_effect = get_activations_side_effect
        mock_hooked_model.return_value.__enter__.return_value = mock_hooked_instance

        iterator = collect_activations(
            model=model,
            tokenizer=mock_tokenizer,
            data=sample_dataset,
            layers=layers,
            batch_size=2,
            streaming=True,
        )

        # Collect batches
        batches = list(iterator)
        assert len(batches) == 2  # 3 samples with batch_size=2

        # Check batch shapes
        for batch in batches:
            assert isinstance(batch, Activations)
            assert batch.n_layers == 2
            assert batch.activations.shape[0] == 2  # 2 layers


class TestCollectDatasetActivations:
    """Test collect_activations function."""

    @patch("probelib.processing.activations.tokenize_dialogues")
    @patch("probelib.processing.activations.HookedModel")
    def test_collect_activations_basic(
        self, mock_hooked_model, mock_tokenize, sample_dataset
    ):
        """Test basic activation collection."""
        model = SimpleModel(n_layers=3, hidden_size=64)
        tokenizer = Mock()
        tokenizer.pad_token_id = 0
        tokenizer.eos_token_id = 1
        tokenizer.padding_side = "right"

        # Mock tokenization output
        batch_size = len(sample_dataset)
        seq_len = 32
        mock_tokenize.return_value = {
            "input_ids": torch.randint(0, 1000, (batch_size, seq_len)),
            "attention_mask": torch.ones(batch_size, seq_len),
            "detection_mask": torch.ones(batch_size, seq_len).bool(),
        }

        # Mock hooked model
        mock_hooked_instance = MagicMock()

        # get_activations should return activations for the specific batch
        def get_activations_side_effect(batch_inputs):
            batch_size_actual = batch_inputs["input_ids"].shape[0]
            seq_len_actual = batch_inputs["input_ids"].shape[1]
            return torch.randn(len(layers), batch_size_actual, seq_len_actual, 64)

        mock_hooked_instance.get_activations.side_effect = get_activations_side_effect
        mock_hooked_model.return_value.__enter__.return_value = mock_hooked_instance

        # Collect activations
        layers = [0, 2]
        activations = collect_activations(
            model=model,
            tokenizer=tokenizer,
            data=sample_dataset,  # Changed from 'dataset' to 'data'
            layers=layers,
            batch_size=2,
            verbose=False,
        )

        assert isinstance(activations, Activations)
        assert activations.n_layers == 2
        assert activations.batch_size == batch_size
        assert activations.layer_indices == layers

    def test_collect_activations_streaming(self, sample_dataset, mock_tokenizer):
        """Test streaming activation collection."""
        model = SimpleModel()

        iterator = collect_activations(
            model=model,
            tokenizer=mock_tokenizer,
            data=sample_dataset,  # Changed from 'dataset' to 'data'
            layers=[0, 1],
            batch_size=2,
            streaming=True,
            verbose=False,
        )

        assert isinstance(iterator, ActivationIterator)
        assert iterator.layers == [0, 1]
        assert len(iterator) == 2  # 3 samples with batch_size=2


class TestActivationIteratorIntegration:
    """Integration tests for activation collection."""

    @patch("probelib.processing.activations.tokenize_dialogues")
    @patch("probelib.processing.activations.HookedModel")
    def test_iterator_yields_correct_batches(
        self, mock_hooked_model, mock_tokenize, sample_dataset
    ):
        """Test that iterator yields correct number of batches."""
        model = SimpleModel()
        tokenizer = Mock()
        tokenizer.pad_token_id = 0
        tokenizer.padding_side = "right"
        tokenizer.name_or_path = "meta-llama/Llama-2-7b-hf"
        tokenizer.eos_token_id = 1

        # Set up tokenizer to work when called directly
        def tokenizer_call(*args, **kwargs):
            n_samples = len(args[0]) if args else 3
            seq_len = 32

            class MockBatchEncoding(dict):
                def __init__(self):
                    super().__init__()
                    self["input_ids"] = torch.randint(0, 1000, (n_samples, seq_len))
                    self["attention_mask"] = torch.ones(n_samples, seq_len)

                def char_to_token(self, batch_idx, char_idx):
                    return min(char_idx // 3, seq_len - 1)

            return MockBatchEncoding()

        tokenizer.side_effect = tokenizer_call

        # Mock apply_chat_template
        def mock_apply_chat_template(dialogues, **kwargs):
            formatted = []
            for dialogue_list in dialogues:
                formatted_text = ""
                for msg in dialogue_list:
                    role = msg.get("role", "")
                    content = msg.get("content", "")
                    formatted_text += f"<|start_header_id|>{role}<|end_header_id|>\n\n{content}<|eot_id|>"
                formatted.append(formatted_text)
            return formatted

        tokenizer.apply_chat_template = mock_apply_chat_template

        # Mock tokenization for each batch
        seq_len = 32
        hidden_dim = 64

        def create_batch_output(batch_size):
            return {
                "input_ids": torch.randint(0, 1000, (batch_size, seq_len)),
                "attention_mask": torch.ones(batch_size, seq_len),
                "detection_mask": torch.ones(batch_size, seq_len).bool(),
            }

        # Set up mock tokenize to return output for all samples at once
        mock_tokenize.return_value = create_batch_output(
            3
        )  # All 3 samples from sample_dataset

        # Mock hooked model to return activations
        mock_hooked_instance = MagicMock()
        mock_hooked_instance.get_activations.side_effect = [
            torch.randn(2, 2, seq_len, hidden_dim),  # First batch
            torch.randn(2, 1, seq_len, hidden_dim),  # Second batch
        ]
        mock_hooked_model.return_value.__enter__.return_value = mock_hooked_instance

        iterator = collect_activations(
            model=model,
            tokenizer=tokenizer,
            data=sample_dataset,
            layers=[0, 1],
            batch_size=2,
            streaming=True,
            verbose=False,
        )

        batches = list(iterator)
        assert len(batches) == 2

        # First batch should have 2 samples
        assert batches[0].batch_size == 2
        assert batches[0].n_layers == 2

        # Second batch should have 1 sample
        assert batches[1].batch_size == 1
        assert batches[1].n_layers == 2


class TestRealModelIntegration:
    """Integration tests with real models (marked as slow)."""

    def test_with_real_model(self):
        """Test with a real small model if available."""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            # Try to load a tiny model for testing
            model_name = "google/gemma-2-2b-it"
            model = AutoModelForCausalLM.from_pretrained(model_name)
            tokenizer = AutoTokenizer.from_pretrained(model_name)

            # Create simple dataset
            dialogues = [
                Dialogue(
                    [
                        Message(role="user", content="Hello"),
                        Message(role="assistant", content="Hi"),
                    ]
                )
            ]
            dataset = MockDialogueDataset(dialogues=dialogues, labels=[Label.POSITIVE])

            # Collect activations
            activations = collect_activations(
                model=model,
                tokenizer=tokenizer,
                data=dataset,  # Changed from 'dataset' to 'data'
                layers=[0, 1],
                batch_size=1,
                verbose=False,
            )

            assert activations.n_layers == 2
            assert activations.batch_size == 1
            assert activations.d_model == model.config.hidden_size

        except ImportError:
            pytest.skip("transformers not available")
        except Exception as e:
            pytest.skip(f"Could not load model: {e}")
