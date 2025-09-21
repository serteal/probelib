"""
Benchmark tokenization performance with different masks.

This script specifically tests tokenization and mask evaluation performance
to identify bottlenecks and optimization opportunities.
"""

import argparse
import logging
from typing import Any, Dict, List, Optional

import torch
from transformers import AutoTokenizer

import probelib as pl
from benchmarks.utils import TimingResult, measure_with_warmup
from probelib.masks import (
    after,
    all,
    assistant,
    before,
    between,
    contains,
    none,
    nth_message,
    special_tokens,
    system,
    user,
)
from probelib.processing.tokenization import tokenize_dataset

torch.set_float32_matmul_precision("high")
pl.logger.logger.setLevel(logging.WARNING)  # type: ignore


def benchmark_tokenization_only(
    tokenizer: Any,
    dataset: pl.datasets.DialogueDataset,
    device: str = "cpu",
) -> Dict[str, TimingResult]:
    """Benchmark just tokenization without masks."""
    results = {}

    # Pure tokenization without any mask
    def tokenize_no_mask():
        return tokenize_dataset(
            dataset=dataset,
            tokenizer=tokenizer,
            device=device,
        )

    result = measure_with_warmup(
        tokenize_no_mask,
        warmup_runs=2,
        measurement_runs=10,
        name="Tokenization (no mask)",
    )
    results["no_mask"] = result

    return results


def benchmark_mask_evaluation(
    tokenizer: Any,
    dataset: pl.datasets.DialogueDataset,
    device: str = "cpu",
) -> Dict[str, TimingResult]:
    """Benchmark tokenization with different mask types."""
    results = {}

    # Define mask configurations to test
    mask_configs = [
        # Basic masks
        ("none", none(), "No tokens selected"),
        ("all", all(), "All tokens selected"),
        # Role masks (should be fast with optimizations)
        ("assistant", assistant(), "Assistant messages"),
        ("user", user(), "User messages"),
        ("system", system(), "System messages"),
        ("assistant_no_pad", assistant(include_padding=False), "Assistant no padding"),
        # Text matching masks
        ("contains_the", contains("the"), "Contains 'the'"),
        ("contains_i", contains("I"), "Contains 'I'"),
        # Position masks
        ("after_colon", after(":"), "After colon"),
        ("before_period", before("."), "Before period"),
        ("between_quotes", between('"', '"'), "Between quotes"),
        # Special masks
        ("special_tokens", special_tokens(), "Special tokens"),
        ("nth_first", nth_message(0), "First message"),
        ("nth_last", nth_message(-1), "Last message"),
        # Complex combinations
        ("complex_and", assistant() & contains("I"), "Assistant AND contains 'I'"),
        ("complex_or", user() | assistant(), "User OR assistant"),
        ("complex_not", ~user(), "NOT user"),
        (
            "complex_nested",
            (assistant() & after(":")) | (user() & contains("?")),
            "Nested: (assistant & after ':') | (user & contains '?')",
        ),
    ]

    for mask_name, mask, description in mask_configs:

        def tokenize_with_mask():
            return tokenize_dataset(
                dataset=dataset,
                tokenizer=tokenizer,
                mask=mask,
                device=device,
            )

        result = measure_with_warmup(
            tokenize_with_mask,
            warmup_runs=1,
            measurement_runs=5,
            name=f"{mask_name}",
        )
        results[mask_name] = result

    return results


def analyze_scaling(
    tokenizer: Any,
    base_dataset: pl.datasets.DialogueDataset,
    sizes: List[int],
    device: str = "cpu",
) -> Dict[int, Dict[str, TimingResult]]:
    """Analyze how performance scales with dataset size."""
    scaling_results = {}

    # Test a few key masks at different scales
    test_masks = {
        "no_mask": None,
        "assistant": assistant(),
        "complex": assistant() & contains("I"),
        "between": between("<", ">"),
    }

    for size in sizes:
        print(f"\n{'=' * 60}")
        print(f"Testing with {size} samples")
        print(f"{'=' * 60}")

        # Create dataset of this size
        dataset = base_dataset[:size]

        size_results = {}
        for mask_name, mask in test_masks.items():

            def tokenize_fn():
                return tokenize_dataset(
                    dataset=dataset,
                    tokenizer=tokenizer,
                    mask=mask,
                    device=device,
                )

            result = measure_with_warmup(
                tokenize_fn,
                warmup_runs=1,
                measurement_runs=3,
                name=f"{mask_name} ({size} samples)",
            )
            size_results[mask_name] = result

        scaling_results[size] = size_results

    return scaling_results


def print_results(
    tokenization_results: Dict[str, TimingResult],
    mask_results: Dict[str, TimingResult],
    scaling_results: Optional[Dict[int, Dict[str, TimingResult]]] = None,
    num_samples: int = 0,
    baseline_name: str = "no_mask",
):
    """Print formatted benchmark results."""
    print("\n" + "=" * 80)
    print("TOKENIZATION PERFORMANCE BENCHMARK")
    print("=" * 80)

    if tokenization_results:
        print("\nBaseline Tokenization (no masks):")
        print("-" * 40)
        for name, result in tokenization_results.items():
            throughput = result.throughput(num_samples)
            print(f"  {name}: {throughput:.1f} samples/sec ({result.mean:.3f}s)")

    if mask_results:
        print("\nMask Evaluation Performance:")
        print("-" * 40)

        # Get baseline for comparison
        baseline = mask_results.get(baseline_name) or tokenization_results.get(
            "no_mask"
        )
        baseline_throughput = baseline.throughput(num_samples) if baseline else None

        # Sort by throughput
        sorted_results = sorted(
            mask_results.items(),
            key=lambda x: x[1].throughput(num_samples),
            reverse=True,
        )

        for name, result in sorted_results:
            throughput = result.throughput(num_samples)
            time_str = f"{result.mean:.3f}s ± {result.std:.3f}s"

            if baseline_throughput:
                ratio = throughput / baseline_throughput
                status = "✓" if ratio >= 0.8 else "✗"
                color = "\033[92m" if ratio >= 0.8 else "\033[91m"
                print(
                    f"  {name:20s}: {throughput:7.1f} samples/sec | "
                    f"{time_str:20s} | {color}{ratio:5.1%} {status}\033[0m"
                )
            else:
                print(f"  {name:20s}: {throughput:7.1f} samples/sec | {time_str}")

    if scaling_results:
        print("\nScaling Analysis:")
        print("-" * 40)

        # Create scaling table
        sizes = sorted(scaling_results.keys())
        mask_names = list(next(iter(scaling_results.values())).keys())

        # Print header
        print(f"  {'Size':<10} ", end="")
        for mask_name in mask_names:
            print(f"{mask_name:>15} ", end="")
        print()

        # Print data
        for size in sizes:
            print(f"  {size:<10} ", end="")
            for mask_name in mask_names:
                result = scaling_results[size][mask_name]
                throughput = result.throughput(size)
                print(f"{throughput:>14.1f} ", end="")
            print()

        # Calculate and print scaling efficiency
        if len(sizes) > 1:
            print("\nScaling Efficiency (linear = 1.0):")
            base_size = sizes[0]
            for mask_name in mask_names:
                print(f"  {mask_name}:")
                base_throughput = scaling_results[base_size][mask_name].throughput(
                    base_size
                )

                for size in sizes[1:]:
                    result = scaling_results[size][mask_name]
                    actual_throughput = result.throughput(size)
                    expected_throughput = (
                        base_throughput  # Should stay constant for linear
                    )
                    efficiency = actual_throughput / expected_throughput
                    print(f"    {base_size}->{size}: {efficiency:.2f}")


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark tokenization and mask performance"
    )
    parser.add_argument(
        "--model",
        default="google/gemma-2-2b-it",
        help="Model/tokenizer to use",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=200,
        help="Number of samples for main benchmark",
    )
    parser.add_argument(
        "--scaling",
        action="store_true",
        help="Run scaling analysis",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use",
    )

    args = parser.parse_args()

    # Load tokenizer
    print(f"Loading tokenizer: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    # Load dataset
    print(f"Loading dataset with {args.samples} samples...")
    harmful = pl.datasets.CircuitBreakersDataset()[: args.samples // 2]
    benign = pl.datasets.BenignInstructionsDataset()[: args.samples // 2]
    dataset = harmful + benign
    print(f"Dataset loaded: {len(dataset)} samples")

    # Run benchmarks
    print("\nRunning tokenization benchmark...")
    tokenization_results = benchmark_tokenization_only(tokenizer, dataset, args.device)

    print("\nRunning mask evaluation benchmark...")
    mask_results = benchmark_mask_evaluation(tokenizer, dataset, args.device)

    # Run scaling analysis if requested
    scaling_results = None
    if args.scaling:
        print("\nRunning scaling analysis...")
        sizes = [50, 100, 200, 500, 1000]

        # Create a large dataset by loading more data
        print("Loading larger dataset for scaling analysis...")
        max_size = max(sizes)
        harmful_large = pl.datasets.CircuitBreakersDataset()[: max_size // 2]
        benign_large = pl.datasets.BenignInstructionsDataset()[: max_size // 2]
        large_dataset = harmful_large + benign_large

        # Filter sizes to those we have data for
        sizes = [s for s in sizes if s <= len(large_dataset)]
        if sizes:
            scaling_results = analyze_scaling(
                tokenizer, large_dataset, sizes, args.device
            )

    # Print results
    print_results(
        tokenization_results,
        mask_results,
        scaling_results,
        num_samples=len(dataset),
    )


if __name__ == "__main__":
    main()
