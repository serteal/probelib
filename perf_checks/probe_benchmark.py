import argparse
import gc
import logging
from typing import Any, Dict, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils import TimingResult, measure_with_warmup, timer

import probelib as pl

torch.set_float32_matmul_precision("high")
pl.logger.logger.setLevel(logging.WARNING)  # type: ignore


def benchmark_activation_collection(
    model: Any,
    tokenizer: Any,
    dataset: pl.datasets.DialogueDataset,
    layers: List[int],
    batch_size: int = 8,
    max_samples: Optional[int] = None,
) -> Dict[str, TimingResult]:
    """Benchmark activation collection performance.

    Returns dict with timing results for each operation.
    """
    results = {}

    # Limit samples if requested
    if max_samples:
        dataset = dataset[:max_samples]

    # Measure tokenization
    def tokenize_fn():
        return pl.processing.tokenize_dataset(tokenizer=tokenizer, dataset=dataset)

    tokenize_result = measure_with_warmup(
        tokenize_fn, warmup_runs=1, measurement_runs=3, name="Tokenization"
    )
    results["tokenization"] = tokenize_result

    # Get tokenized inputs for activation collection
    inputs = pl.processing.tokenize_dataset(dataset, tokenizer)

    # Measure activation collection with HookedModel
    def collect_with_hooks():
        with pl.HookedModel(model, layers=layers) as hooked_model:
            activations = []

            # Process in batches
            for start_idx in range(0, len(inputs["input_ids"]), batch_size):
                end_idx = min(start_idx + batch_size, len(inputs["input_ids"]))
                batch_inputs = {
                    k: v[start_idx:end_idx].to(model.device) for k, v in inputs.items()
                }

                # Get activations
                batch_acts = hooked_model.get_activations(batch_inputs)
                # Use non_blocking for async GPU->CPU transfer
                activations.append(batch_acts.to("cpu", non_blocking=True))

            return torch.cat(activations, dim=1)

    hook_result = measure_with_warmup(
        collect_with_hooks,
        warmup_runs=1,
        measurement_runs=3,
        name="Activation Collection (HookedModel - all positions)",
    )
    results["activation_collection_hooks"] = hook_result

    # Measure using the high-level collect_activations function (batch mode)
    def collect_high_level_batch():
        return pl.collect_activations(
            model=model,
            tokenizer=tokenizer,
            data=dataset,
            layers=layers,
            batch_size=batch_size,
            streaming=False,  # Use batch mode for fair comparison
            verbose=False,
        )

    high_level_batch_result = measure_with_warmup(
        collect_high_level_batch,
        warmup_runs=1,
        measurement_runs=3,
        name="Activation Collection (collect_activations - batch mode)",
    )
    results["activation_collection_high_level_batch"] = high_level_batch_result

    # Measure streaming mode (now with optimizations built-in)
    def collect_high_level_streaming():
        # Get the iterator (now optimized by default)
        activation_iter = pl.collect_activations(
            model=model,
            tokenizer=tokenizer,
            data=dataset,
            layers=layers,
            batch_size=batch_size,
            streaming=True,  # Force streaming mode
            verbose=False,
        )

        # Consume the iterator to measure total time
        all_batches = []
        for batch in activation_iter:
            all_batches.append(batch)

        # In real usage, probes would process batches incrementally
        # But for benchmarking, we need to measure the full iteration time
        return all_batches

    high_level_streaming_result = measure_with_warmup(
        collect_high_level_streaming,
        warmup_runs=1,
        measurement_runs=3,
        name="Activation Collection (streaming mode - optimized)",
    )
    results["activation_collection_high_level_streaming"] = high_level_streaming_result

    return results


def benchmark_probe_training(
    model: Any,
    tokenizer: Any,
    dataset: pl.datasets.DialogueDataset,
    layers: List[int],
    batch_size: int = 8,
    max_samples: Optional[int] = None,
) -> Dict[str, TimingResult]:
    """Benchmark probe training performance."""
    results = {}

    # Limit samples if requested
    if max_samples:
        dataset = dataset[:max_samples]

    # Measure full pipeline (activation collection + training)
    def train_probe_full(probe_class, **kwargs):
        def inner_train_probe_full():
            probe = probe_class(layer=layers[0], **kwargs)
            return pl.scripts.train_probes(
                probes=probe,
                model=model,
                tokenizer=tokenizer,
                data=dataset,
                labels=dataset.labels,
                batch_size=batch_size,
                streaming=True,
                verbose=False,
            )

        return inner_train_probe_full

    # Import SequencePooling for the new API
    from probelib.processing import SequencePooling

    # Test with MEAN pooling (sample-level aggregation)
    gpu_logistic_mean_result = measure_with_warmup(
        train_probe_full(pl.probes.Logistic, sequence_pooling=SequencePooling.MEAN),  # type: ignore
        warmup_runs=1,
        measurement_runs=3,
        name="GPU Logistic with MEAN pooling",
    )
    results["gpu_logistic_mean_pooling"] = gpu_logistic_mean_result

    # Test with NONE pooling (token-level training)
    gpu_logistic_token_result = measure_with_warmup(
        train_probe_full(pl.probes.Logistic, sequence_pooling=SequencePooling.NONE),  # type: ignore
        warmup_runs=1,
        measurement_runs=3,
        name="GPU Logistic with token-level (NONE pooling)",
    )
    results["gpu_logistic_token_level"] = gpu_logistic_token_result

    # MLP with MEAN pooling
    mlp_mean_result = measure_with_warmup(
        train_probe_full(pl.probes.MLP, sequence_pooling=SequencePooling.MEAN),  # type: ignore
        warmup_runs=1,
        measurement_runs=3,
        name="MLP with MEAN pooling",
    )
    results["mlp_mean_pooling"] = mlp_mean_result

    # MLP with token-level (NONE pooling)
    mlp_token_result = measure_with_warmup(
        train_probe_full(pl.probes.MLP, sequence_pooling=SequencePooling.NONE),  # type: ignore
        warmup_runs=1,
        measurement_runs=3,
        name="MLP with token-level (NONE pooling)",
    )
    results["mlp_token_level"] = mlp_token_result

    # Attention probe (always uses NONE pooling but aggregates internally)
    attention_full_pipeline_result = measure_with_warmup(
        train_probe_full(pl.probes.Attention, sequence_pooling=SequencePooling.NONE),  # type: ignore
        warmup_runs=1,
        measurement_runs=3,
        name="Attention Probe (attention-based aggregation)",
    )
    results["attention_full_pipeline"] = attention_full_pipeline_result

    def train_10_probes_full():
        probes = {
            f"logistic_{i}": pl.probes.Logistic(
                layer=layers[0], sequence_pooling=SequencePooling.MEAN
            )
            for i in range(10)
        }
        return pl.scripts.train_probes(
            probes=probes,
            model=model,
            tokenizer=tokenizer,
            data=dataset,
            labels=dataset.labels,
            batch_size=batch_size,
            streaming=True,
            verbose=False,
        )

    logistic_10_probes_full_pipeline_result = measure_with_warmup(
        train_10_probes_full,  # type: ignore
        warmup_runs=1,
        measurement_runs=3,
        name="Logistic 10 Probes Full Pipeline",
    )
    results["logistic_10_probes_full_pipeline"] = (
        logistic_10_probes_full_pipeline_result
    )

    return results


def print_summary(
    activation_results: Dict[str, TimingResult] | None = None,
    training_results: Dict[str, TimingResult] | None = None,
    num_samples: int = 0,
):
    """Print a summary of benchmark results."""
    if activation_results:
        print("Activation Collection:")
        print("-" * 40)
        for name, result in activation_results.items():
            throughput = num_samples / result.mean
            # Clean up the display name
            display_name = name.replace("activation_collection_", "").replace("_", " ")
            print(f"  {display_name}:")
            print(f"    Time: {result}")
            print(f"    Throughput: {throughput:.1f} samples/sec")

    if training_results:
        print("\nProbe Training:")
        print("-" * 40)
        for name, result in training_results.items():
            throughput = num_samples / result.mean
            print(f"  {name}:")
            print(f"    Time: {result}")
            print(f"    Throughput: {throughput:.1f} samples/sec")

    # Memory usage if CUDA available
    if torch.cuda.is_available():
        print("\nGPU Memory:")
        print("-" * 40)
        print(f"  Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        print(f"  Reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
        print(f"  Max allocated: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")


def main():
    parser = argparse.ArgumentParser(description="Benchmark probelib performance")
    parser.add_argument(
        "--model",
        # default="meta-llama/Llama-3.1-8B-Instruct",
        default="google/gemma-2-2b-it",
        help="Model to use for benchmarking",
    )
    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=[12, 14, 16],
        help="Layers to extract activations from",
    )
    parser.add_argument(
        "--batch-size", type=int, default=32, help="Batch size for processing"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=200,
        help="Maximum number of samples to use (None for all)",
    )
    parser.add_argument(
        "--add-activation-benchmark",
        action="store_true",
        help="Only benchmark activation collection",
    )

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model and tokenizer
    with timer("Model loading"):
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
            device_map=device,
        )
        model.eval()

    print(
        f"Model loaded: {model.config.num_hidden_layers} layers, "
        f"{model.config.hidden_size} hidden size"
    )

    # Load datasets
    with timer("Dataset loading"):
        harmful_dataset = pl.datasets.CircuitBreakersDataset()[: args.max_samples // 2]
        benign_dataset = pl.datasets.BenignInstructionsDataset()[
            : args.max_samples // 2
        ]
        dataset = harmful_dataset + benign_dataset

    # Benchmark activation collection
    activation_results = None
    if args.add_activation_benchmark:
        activation_results = benchmark_activation_collection(
            model=model,
            tokenizer=tokenizer,
            dataset=dataset,
            layers=args.layers,
            batch_size=args.batch_size,
            max_samples=None,  # Already limited above
        )

        # Clear cache before training benchmark
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

    # Benchmark probe training
    training_results = None
    training_results = benchmark_probe_training(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        layers=args.layers,
        batch_size=args.batch_size,
        max_samples=None,  # Already limited above
    )

    # Print summary
    print_summary(
        activation_results=activation_results,
        training_results=training_results,
        num_samples=len(dataset),
    )


if __name__ == "__main__":
    main()
