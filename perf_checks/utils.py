import gc
import numpy as np
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Callable, List

import torch


@dataclass
class TimingResult:
    """Container for timing measurements."""

    mean: float
    std: float
    min: float
    max: float
    samples: List[float]

    def throughput(self, num_samples: int) -> float:
        """Calculate throughput in samples/sec."""
        return num_samples / self.mean

    def __str__(self) -> str:
        return (
            f"Mean: {self.mean:.3f}s ± {self.std:.3f}s | "
            f"Min: {self.min:.3f}s | Max: {self.max:.3f}s"
        )


@contextmanager
def timer(name: str = "Operation"):
    """Context manager for timing operations."""
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start = time.perf_counter()
    try:
        yield
    finally:
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        elapsed = time.perf_counter() - start
        print(f"{name}: {elapsed:.3f}s")


def measure_with_warmup(
    func: Callable,
    warmup_runs: int = 1,
    measurement_runs: int = 5,
    clear_cache: bool = True,
    name: str = "Operation",
) -> TimingResult:
    """Measure function execution time with warmup runs.

    Args:
        func: Function to measure
        warmup_runs: Number of warmup runs to perform
        measurement_runs: Number of measurement runs
        clear_cache: Whether to clear GPU cache between runs
        name: Name for logging

    Returns:
        TimingResult with statistics
    """
    print(f"\n{'=' * 60}")
    print(f"Measuring: {name}")
    print(f"{'=' * 60}")

    # Warmup runs
    print(f"Running {warmup_runs} warmup iteration(s)...")
    for i in range(warmup_runs):
        if clear_cache and torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
        with timer(f"Warmup {i + 1}"):
            func()

    # Measurement runs
    print(f"\nRunning {measurement_runs} measurement iterations...")
    timings = []
    for i in range(measurement_runs):
        if clear_cache and torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start = time.perf_counter()

        func()

        torch.cuda.synchronize() if torch.cuda.is_available() else None
        elapsed = time.perf_counter() - start
        timings.append(elapsed)
        print(f"  Run {i + 1}: {elapsed:.3f}s")

    result = TimingResult(
        mean=np.mean(timings),
        std=np.std(timings),
        min=np.min(timings),
        max=np.max(timings),
        samples=timings,
    )

    print(f"\nResults: {result}")
    return result
