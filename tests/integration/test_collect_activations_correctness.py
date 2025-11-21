"""
Integration tests for collect_activations correctness using real models.

These tests verify that post-block activations captured by HookedModel with
hook_point='post_block' exactly match HuggingFace hidden_states for the same
layers, and that tokenization and detection masks are consistent.

Tests are marked with @pytest.mark.integration and model-specific marks
(@pytest.mark.llama3, @pytest.mark.gemma2). They require CUDA.
"""

import random

import numpy as np
import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import probelib as pl
from probelib.processing.activations import collect_activations, get_batches


def _require_cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for this test (speed/memory)")


def _set_deterministic():
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@pytest.mark.integration
@pytest.mark.llama3
def test_collect_activations_matches_hidden_states_llama3():
    _require_cuda()
    _set_deterministic()

    model_name = "meta-llama/Llama-3.1-8B-Instruct"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
    )
    model.eval()

    # Small dialogues to limit compute
    dialogues = [
        [
            pl.Message("system", "You are a helpful assistant."),
            pl.Message("user", "What is 2+2?"),
            pl.Message("assistant", "2+2 equals 4."),
        ],
        [
            pl.Message("user", "Say hello"),
            pl.Message("assistant", "Hello!"),
        ],
    ]

    # Tokenize with assistant mask to produce detection_mask
    tokenized = pl.processing.tokenize_dialogues(
        tokenizer=tokenizer,
        dialogues=dialogues,
        mask=pl.masks.assistant(),
        device=model.device,
        add_generation_prompt=False,
    )

    # Under test: collect two early layers for strict equality
    layers = [0, 1]
    acts = collect_activations(
        model=model,
        tokenizer=tokenizer,
        dataset=dialogues,
        layers=layers,
        batch_size=2,
        streaming=False,
        verbose=False,
        add_generation_prompt=False,
        mask=pl.masks.assistant(),
    )

    # Baseline: recompute hidden_states per trimmed batch to mirror collect_activations
    n_samples, max_seq_len = tokenized["input_ids"].shape
    expected = torch.zeros(
        (len(layers), n_samples, max_seq_len, model.config.hidden_size),
        device=acts.activations.device,
        dtype=acts.activations.dtype,
    )
    with torch.inference_mode():
        for batch_inputs, batch_indices in get_batches(
            tokenized, batch_size=2, tokenizer=tokenizer
        ):
            out = model(
                input_ids=batch_inputs["input_ids"],
                attention_mask=batch_inputs["attention_mask"],
                use_cache=False,
                output_hidden_states=True,
                return_dict=True,
            )
            hs = out.hidden_states
            stacked = torch.stack([hs[i + 1] for i in layers], dim=0).to(
                device=expected.device, dtype=expected.dtype
            )
            seq_len = stacked.shape[2]
            expected[:, batch_indices, -seq_len:] = stacked

    # Exact equality (same dtype/device)
    assert acts.activations.shape == expected.shape
    assert torch.equal(acts.activations, expected)

    # Tokenization and masks preserved
    assert torch.equal(acts.input_ids, tokenized["input_ids"].to(acts.input_ids.device))
    assert torch.equal(
        acts.attention_mask, tokenized["attention_mask"].to(acts.attention_mask.device)
    )
    assert torch.equal(
        acts.detection_mask, tokenized["detection_mask"].to(acts.detection_mask.device)
    )
    assert acts.layer_indices == layers

    # Streaming vs batch consistency
    acts_iter = collect_activations(
        model=model,
        tokenizer=tokenizer,
        data=dialogues,
        layers=layers,
        batch_size=2,
        streaming=True,
        verbose=False,
        add_generation_prompt=False,
        mask=pl.masks.assistant(),
    )

    # Reconstruct full tensor from streaming batches
    full = torch.zeros_like(expected)
    for batch in acts_iter:
        assert batch.layer_indices == layers
        # Place depending on padding side
        full[:, batch.batch_indices, -batch.seq_len :] = batch.activations.to(
            full.device, full.dtype
        )
    # Compare streaming-assembled with batch activations from library
    assert torch.equal(full, acts.activations)


@pytest.mark.integration
@pytest.mark.gemma2
def test_collect_activations_matches_hidden_states_gemma2():
    _require_cuda()
    _set_deterministic()

    model_name = "google/gemma-2-2b-it"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
    )
    model.eval()

    # Include system to exercise folding behavior
    dialogues = [
        [
            pl.Message("system", "Follow the user instructions."),
            pl.Message("user", "Translate 'hello' to French."),
            pl.Message("assistant", "Bonjour"),
        ],
        [
            pl.Message("user", "What is the capital of France?"),
            pl.Message("assistant", "Paris."),
        ],
    ]

    tokenized = pl.processing.tokenize_dialogues(
        tokenizer=tokenizer,
        dialogues=dialogues,
        mask=pl.masks.assistant(),
        device=model.device,
        add_generation_prompt=False,
    )

    layers = [0, 1]
    acts = collect_activations(
        model=model,
        tokenizer=tokenizer,
        data=dialogues,
        layers=layers,
        batch_size=2,
        streaming=False,
        verbose=False,
        add_generation_prompt=False,
        mask=pl.masks.assistant(),
    )

    # Baseline expected mirroring trimmed batches
    n_samples, max_seq_len = tokenized["input_ids"].shape
    expected = torch.zeros(
        (len(layers), n_samples, max_seq_len, model.config.hidden_size),
        device=acts.activations.device,
        dtype=acts.activations.dtype,
    )
    with torch.inference_mode():
        for batch_inputs, batch_indices in get_batches(
            tokenized, batch_size=2, tokenizer=tokenizer
        ):
            out = model(
                input_ids=batch_inputs["input_ids"],
                attention_mask=batch_inputs["attention_mask"],
                use_cache=False,
                output_hidden_states=True,
                return_dict=True,
            )
            hs = out.hidden_states
            stacked = torch.stack([hs[i + 1] for i in layers], dim=0).to(
                device=expected.device, dtype=expected.dtype
            )
            seq_len = stacked.shape[2]
            expected[:, batch_indices, -seq_len:] = stacked

    assert acts.activations.shape == expected.shape
    assert torch.equal(acts.activations, expected)
    assert torch.equal(acts.input_ids, tokenized["input_ids"].to(acts.input_ids.device))
    assert torch.equal(
        acts.attention_mask, tokenized["attention_mask"].to(acts.attention_mask.device)
    )
    assert torch.equal(
        acts.detection_mask, tokenized["detection_mask"].to(acts.detection_mask.device)
    )
    assert acts.layer_indices == layers

    # Streaming equality
    acts_iter = collect_activations(
        model=model,
        tokenizer=tokenizer,
        data=dialogues,
        layers=layers,
        batch_size=2,
        streaming=True,
        verbose=False,
        add_generation_prompt=False,
        mask=pl.masks.assistant(),
    )

    full = torch.zeros_like(expected)
    for batch in acts_iter:
        assert batch.layer_indices == layers
        full[:, batch.batch_indices, -batch.seq_len :] = batch.activations.to(
            full.device, full.dtype
        )

    assert torch.equal(full, acts.activations)
