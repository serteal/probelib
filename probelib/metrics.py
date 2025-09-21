"""
Function-based metrics API for probe evaluation.

This module provides a modern, composable metrics system with:
- Individual metric functions with consistent signatures
- Bootstrap confidence interval support via decorator
- Type-safe metric protocol
- Easy extensibility for custom metrics
"""

import functools
from typing import Callable, Protocol, runtime_checkable

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from .logger import logger


@runtime_checkable
class Metric(Protocol):
    """Protocol for metric functions.

    All metrics should accept:
    - y_true: True labels (binary 0/1)
    - y_pred_proba: Predicted probabilities (either [N] or [N, 2])
    - **kwargs: Additional metric-specific parameters

    And return a float score.
    """

    def __call__(self, y_true: np.ndarray, y_pred_proba: np.ndarray, **kwargs) -> float:
        """Compute the metric.

        Args:
            y_true: True binary labels [N]
            y_pred_proba: Predicted probabilities [N] or [N, 2]
            **kwargs: Metric-specific parameters

        Returns:
            Computed metric value
        """
        ...


def _ensure_numpy(x: np.ndarray | torch.Tensor) -> np.ndarray:
    """Convert tensor to numpy array if needed."""
    if isinstance(x, torch.Tensor):
        # Handle BFloat16 by converting to float32 first
        return x.detach().cpu().float().numpy()
    return np.asarray(x)


def _get_binary_proba(y_pred_proba: np.ndarray) -> np.ndarray:
    """Extract positive class probabilities from predictions.

    Args:
        y_pred_proba: Either [N] binary probabilities or [N, 2] class probabilities

    Returns:
        Binary probabilities [N] for the positive class
    """
    if y_pred_proba.ndim == 2:
        return y_pred_proba[:, 1]
    return y_pred_proba


def _get_binary_predictions(
    y_pred_proba: np.ndarray, threshold: float = 0.5
) -> np.ndarray:
    """Convert probabilities to binary predictions.

    Args:
        y_pred_proba: Predicted probabilities [N] or [N, 2]
        threshold: Decision threshold

    Returns:
        Binary predictions [N]
    """
    proba = _get_binary_proba(y_pred_proba)
    return (proba > threshold).astype(int)


def with_bootstrap(
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    random_state: int | None = None,
) -> Callable[[Callable], Callable]:
    """Decorator to add bootstrap confidence intervals to any metric.

    Args:
        n_bootstrap: Number of bootstrap samples
        confidence_level: Confidence level for intervals (e.g., 0.95 for 95% CI)
        random_state: Random seed for reproducibility

    Returns:
        Decorator that wraps a metric to return (point_estimate, ci_lower, ci_upper)

    Examples:
        >>> @with_bootstrap(n_bootstrap=100)
        ... def my_metric(y_true, y_pred_proba):
        ...     return np.mean(y_true == (y_pred_proba > 0.5))
        >>>
        >>> # Returns (metric_value, ci_lower, ci_upper)
        >>> result = my_metric(y_true, y_pred)
    """

    def decorator(metric_fn: Callable) -> Callable:
        @functools.wraps(metric_fn)
        def wrapped(
            y_true: np.ndarray, y_pred_proba: np.ndarray, **kwargs
        ) -> tuple[float, float, float]:
            # Ensure numpy arrays
            y_true = _ensure_numpy(y_true)
            y_pred_proba = _ensure_numpy(y_pred_proba)

            rng = np.random.default_rng(random_state)

            # Compute point estimate
            point_estimate = metric_fn(y_true, y_pred_proba, **kwargs)

            # Bootstrap sampling
            n_samples = len(y_true)
            bootstrap_results = []

            for _ in range(n_bootstrap):
                # Sample with replacement
                indices = rng.choice(n_samples, size=n_samples, replace=True)
                y_true_boot = y_true[indices]
                y_pred_boot = (
                    y_pred_proba[indices]
                    if y_pred_proba.ndim == 1
                    else y_pred_proba[indices, :]
                )

                # Compute metric on bootstrap sample
                try:
                    # Check if bootstrap sample has both classes for classification metrics
                    if len(np.unique(y_true_boot)) < 2:
                        # Skip single-class bootstrap samples
                        continue
                    boot_result = metric_fn(y_true_boot, y_pred_boot, **kwargs)
                    bootstrap_results.append(boot_result)
                except (ValueError, ZeroDivisionError):
                    # Skip invalid bootstrap samples (e.g., all same class)
                    continue

            # Calculate confidence intervals
            if len(bootstrap_results) < n_bootstrap * 0.5:
                # Too many failed samples, return NaN for CIs
                logger.warning(
                    f"Too many failed bootstrap samples for {metric_fn.__name__}. Returning NaN for CIs."  # type: ignore
                )  # type: ignore
                return point_estimate, np.nan, np.nan

            alpha = (1 - confidence_level) / 2
            ci_lower = float(np.percentile(bootstrap_results, alpha * 100))
            ci_upper = float(np.percentile(bootstrap_results, (1 - alpha) * 100))

            return point_estimate, ci_lower, ci_upper

        wrapped.__name__ = metric_fn.__name__  # type: ignore
        wrapped.__doc__ = metric_fn.__doc__  # type: ignore
        wrapped._probelib_bootstrap = True  # type: ignore[attr-defined]
        wrapped._probelib_bootstrap_config = {
            "n_bootstrap": n_bootstrap,
            "confidence_level": confidence_level,
            "random_state": random_state,
        }  # type: ignore[attr-defined]

        return wrapped

    return decorator


# ============================================================================
# Core Classification Metrics
# ============================================================================


def auroc(y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
    """Area Under the Receiver Operating Characteristic curve.

    Args:
        y_true: True binary labels [N]
        y_pred_proba: Predicted probabilities [N] or [N, 2]

    Returns:
        AUROC score (0.5 = random, 1.0 = perfect)

    Examples:
        >>> y_true = np.array([0, 0, 1, 1])
        >>> y_pred = np.array([0.1, 0.3, 0.7, 0.9])
        >>> auroc(y_true, y_pred)
        1.0
    """
    y_true = _ensure_numpy(y_true)
    y_pred_proba = _ensure_numpy(y_pred_proba)
    proba = _get_binary_proba(y_pred_proba)

    # Check if both classes are present
    if len(np.unique(y_true)) < 2:
        # Return 0.5 for undefined AUROC (single class)
        raise ValueError("Cannot compute AUROC with only one class")

    return float(roc_auc_score(y_true, proba))


def partial_auroc(
    y_true: np.ndarray, y_pred_proba: np.ndarray, max_fpr: float = 0.1
) -> float:
    """Partial AUROC up to a specified false positive rate.

    Args:
        y_true: True binary labels [N]
        y_pred_proba: Predicted probabilities [N] or [N, 2]
        max_fpr: Maximum false positive rate for partial AUROC

    Returns:
        Partial AUROC score

    Examples:
        >>> y_true = np.array([0, 0, 1, 1])
        >>> y_pred = np.array([0.1, 0.3, 0.7, 0.9])
        >>> auroc_partial(y_true, y_pred, max_fpr=0.1)  # doctest: +SKIP
    """
    y_true = _ensure_numpy(y_true)
    y_pred_proba = _ensure_numpy(y_pred_proba)
    proba = _get_binary_proba(y_pred_proba)

    # Check if both classes are present
    if len(np.unique(y_true)) < 2:
        # Return 0.0 for undefined partial AUROC (single class)
        raise ValueError("Cannot compute partial AUROC with only one class")

    return float(roc_auc_score(y_true, proba, max_fpr=max_fpr))


def accuracy(
    y_true: np.ndarray, y_pred_proba: np.ndarray, threshold: float = 0.5
) -> float:
    """Classification accuracy.

    Args:
        y_true: True binary labels [N]
        y_pred_proba: Predicted probabilities [N] or [N, 2]
        threshold: Decision threshold for binary classification

    Returns:
        Accuracy score (fraction of correct predictions)

    Examples:
        >>> y_true = np.array([0, 0, 1, 1])
        >>> y_pred = np.array([0.3, 0.4, 0.6, 0.7])
        >>> accuracy(y_true, y_pred, threshold=0.5)
        1.0
    """
    y_true = _ensure_numpy(y_true)
    y_pred_proba = _ensure_numpy(y_pred_proba)
    y_pred = _get_binary_predictions(y_pred_proba, threshold)
    return float(accuracy_score(y_true, y_pred))


def balanced_accuracy(
    y_true: np.ndarray, y_pred_proba: np.ndarray, threshold: float = 0.5
) -> float:
    """Balanced accuracy (average of recall for each class).

    Args:
        y_true: True binary labels [N]
        y_pred_proba: Predicted probabilities [N] or [N, 2]
        threshold: Decision threshold for binary classification

    Returns:
        Balanced accuracy score

    Examples:
        >>> y_true = np.array([0, 0, 0, 1])  # Imbalanced
        >>> y_pred = np.array([0.3, 0.3, 0.3, 0.7])
        >>> balanced_accuracy(y_true, y_pred, threshold=0.5)
        1.0
    """
    y_true = _ensure_numpy(y_true)
    y_pred_proba = _ensure_numpy(y_pred_proba)
    y_pred = _get_binary_predictions(y_pred_proba, threshold)
    return float(balanced_accuracy_score(y_true, y_pred))


def precision(
    y_true: np.ndarray, y_pred_proba: np.ndarray, threshold: float = 0.5
) -> float:
    """Precision (positive predictive value).

    Args:
        y_true: True binary labels [N]
        y_pred_proba: Predicted probabilities [N] or [N, 2]
        threshold: Decision threshold for binary classification

    Returns:
        Precision score (TP / (TP + FP))

    Examples:
        >>> y_true = np.array([0, 0, 1, 1])
        >>> y_pred = np.array([0.3, 0.7, 0.3, 0.7])  # One false positive
        >>> precision(y_true, y_pred, threshold=0.5)
        0.5
    """
    y_true = _ensure_numpy(y_true)
    y_pred_proba = _ensure_numpy(y_pred_proba)
    y_pred = _get_binary_predictions(y_pred_proba, threshold)
    return float(precision_score(y_true, y_pred, zero_division=0.0))


def recall(
    y_true: np.ndarray, y_pred_proba: np.ndarray, threshold: float = 0.5
) -> float:
    """Recall (sensitivity, true positive rate).

    Args:
        y_true: True binary labels [N]
        y_pred_proba: Predicted probabilities [N] or [N, 2]
        threshold: Decision threshold for binary classification

    Returns:
        Recall score (TP / (TP + FN))

    Examples:
        >>> y_true = np.array([0, 0, 1, 1])
        >>> y_pred = np.array([0.3, 0.3, 0.3, 0.7])  # One false negative
        >>> recall(y_true, y_pred, threshold=0.5)
        0.5
    """
    y_true = _ensure_numpy(y_true)
    y_pred_proba = _ensure_numpy(y_pred_proba)
    y_pred = _get_binary_predictions(y_pred_proba, threshold)
    return float(recall_score(y_true, y_pred, zero_division=0.0))


def f1(y_true: np.ndarray, y_pred_proba: np.ndarray, threshold: float = 0.5) -> float:
    """F1 score (harmonic mean of precision and recall).

    Args:
        y_true: True binary labels [N]
        y_pred_proba: Predicted probabilities [N] or [N, 2]
        threshold: Decision threshold for binary classification

    Returns:
        F1 score

    Examples:
        >>> y_true = np.array([0, 0, 1, 1])
        >>> y_pred = np.array([0.3, 0.3, 0.7, 0.7])
        >>> f1(y_true, y_pred, threshold=0.5)
        1.0
    """
    y_true = _ensure_numpy(y_true)
    y_pred_proba = _ensure_numpy(y_pred_proba)
    y_pred = _get_binary_predictions(y_pred_proba, threshold)
    return float(f1_score(y_true, y_pred, zero_division=0.0))


# ============================================================================
# Recall at Fixed FPR Metrics
# ============================================================================


def recall_at_fpr(
    y_true: np.ndarray, y_pred_proba: np.ndarray, fpr: float = 0.05
) -> float:
    """Recall (TPR) at a specified false positive rate.

    This metric is particularly useful for imbalanced datasets where
    you want to control the false positive rate.

    Args:
        y_true: True binary labels [N]
        y_pred_proba: Predicted probabilities [N] or [N, 2]
        fpr: Target false positive rate (e.g., 0.05 for 5% FPR)

    Returns:
        Recall at the specified FPR

    Examples:
        >>> y_true = np.array([0, 0, 0, 0, 1, 1])
        >>> y_pred = np.array([0.1, 0.2, 0.3, 0.4, 0.7, 0.8])
        >>> recall_at_fpr(y_true, y_pred, fpr=0.25)  # Allow 25% FPR
        1.0
    """
    assert fpr >= 0.0 and fpr <= 1.0, (
        "FPR must be between 0.0 and 1.0, e.g. 0.05 for 5% FPR"
    )
    y_true = _ensure_numpy(y_true)
    y_pred_proba = _ensure_numpy(y_pred_proba)
    proba = _get_binary_proba(y_pred_proba)

    # Separate positive and negative scores
    pos_scores = proba[y_true == 1]
    neg_scores = proba[y_true == 0]

    if len(pos_scores) == 0:
        return 0.0
    if len(neg_scores) == 0:
        return 1.0

    # Sort negative scores to find threshold
    neg_scores_sorted = np.sort(neg_scores)

    # Find threshold that gives desired FPR
    n_negatives = len(neg_scores_sorted)
    threshold_idx = int((1 - fpr) * n_negatives)
    if threshold_idx >= n_negatives:
        threshold_idx = n_negatives - 1
    threshold = neg_scores_sorted[threshold_idx]

    # Calculate recall at this threshold
    n_detected = np.sum(pos_scores > threshold)
    return float(n_detected / len(pos_scores))


def tpr_at_fpr(
    y_true: np.ndarray, y_pred_proba: np.ndarray, fpr: float = 0.05
) -> float:
    """True positive rate at a specified false positive rate (alias for recall_at_fpr).

    Args:
        y_true: True binary labels [N]
        y_pred_proba: Predicted probabilities [N] or [N, 2]
        fpr: Target false positive rate

    Returns:
        TPR at the specified FPR
    """
    # This is an alias - just call the base function without double-decorating
    # Remove decorator from recall_at_fpr call to avoid double bootstrap
    y_true = _ensure_numpy(y_true)
    y_pred_proba = _ensure_numpy(y_pred_proba)
    proba = _get_binary_proba(y_pred_proba)

    # Separate positive and negative scores
    pos_scores = proba[y_true == 1]
    neg_scores = proba[y_true == 0]

    if len(pos_scores) == 0:
        return 0.0
    if len(neg_scores) == 0:
        return 1.0

    # Sort negative scores to find threshold
    neg_scores_sorted = np.sort(neg_scores)

    # Find threshold that gives desired FPR
    n_negatives = len(neg_scores_sorted)
    threshold_idx = int((1 - fpr) * n_negatives)
    if threshold_idx >= n_negatives:
        threshold_idx = n_negatives - 1
    threshold = neg_scores_sorted[threshold_idx]

    # Calculate recall at this threshold
    n_detected = np.sum(pos_scores > threshold)
    return float(n_detected / len(pos_scores))


# ============================================================================
# False Positive Rate Metrics
# ============================================================================


def fpr_at_threshold(
    y_true: np.ndarray, y_pred_proba: np.ndarray, threshold: float = 0.5
) -> float:
    """False positive rate at a specified threshold.

    Useful for analyzing how many negative examples are incorrectly
    classified as positive at a given decision threshold.

    Args:
        y_true: True binary labels [N]
        y_pred_proba: Predicted probabilities [N] or [N, 2]
        threshold: Decision threshold

    Returns:
        False positive rate (FP / (FP + TN))

    Examples:
        >>> y_true = np.array([0, 0, 0, 0, 1, 1])
        >>> y_pred = np.array([0.3, 0.4, 0.6, 0.7, 0.8, 0.9])
        >>> fpr_at_threshold(y_true, y_pred, threshold=0.5)
        0.5
    """
    y_true = _ensure_numpy(y_true)
    y_pred_proba = _ensure_numpy(y_pred_proba)
    proba = _get_binary_proba(y_pred_proba)

    # Get negative examples
    neg_mask = y_true == 0
    if not np.any(neg_mask):
        return np.nan  # No negative examples

    neg_scores = proba[neg_mask]
    return float(np.mean(neg_scores > threshold))


def fpr(y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
    """False positive rate at default threshold (0.5).

    Args:
        y_true: True binary labels [N]
        y_pred_proba: Predicted probabilities [N] or [N, 2]

    Returns:
        False positive rate at threshold 0.5
    """
    # Direct implementation to avoid double bootstrap
    y_true = _ensure_numpy(y_true)
    y_pred_proba = _ensure_numpy(y_pred_proba)
    proba = _get_binary_proba(y_pred_proba)

    # Get negative examples
    neg_mask = y_true == 0
    if not np.any(neg_mask):
        return np.nan  # No negative examples

    neg_scores = proba[neg_mask]
    return float(np.mean(neg_scores > 0.5))


# ============================================================================
# Distribution Statistics
# ============================================================================


def mean_score(y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
    """Mean of predicted probabilities.

    Useful for understanding the overall distribution of predictions,
    especially for negative-only datasets to check calibration.

    Args:
        y_true: True binary labels [N] (can be ignored)
        y_pred_proba: Predicted probabilities [N] or [N, 2]

    Returns:
        Mean probability score

    Examples:
        >>> y_true = np.array([0, 0, 1, 1])  # Not used
        >>> y_pred = np.array([0.2, 0.3, 0.7, 0.8])
        >>> mean_score(y_true, y_pred)
        0.5
    """
    y_pred_proba = _ensure_numpy(y_pred_proba)
    proba = _get_binary_proba(y_pred_proba)
    return float(np.mean(proba))


def std_score(y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
    """Standard deviation of predicted probabilities.

    Args:
        y_true: True binary labels [N] (can be ignored)
        y_pred_proba: Predicted probabilities [N] or [N, 2]

    Returns:
        Standard deviation of probability scores

    Examples:
        >>> y_true = np.array([0, 0, 1, 1])  # Not used
        >>> y_pred = np.array([0.2, 0.3, 0.7, 0.8])
        >>> round(std_score(y_true, y_pred), 3)
        0.275
    """
    y_pred_proba = _ensure_numpy(y_pred_proba)
    proba = _get_binary_proba(y_pred_proba)
    return float(np.std(proba))


def percentile(y_true: np.ndarray, y_pred_proba: np.ndarray, q: float = 95) -> float:
    """Percentile of predicted probabilities.

    Args:
        y_true: True binary labels [N] (can be ignored)
        y_pred_proba: Predicted probabilities [N] or [N, 2]
        q: Percentile to compute (0-100)

    Returns:
        The q-th percentile of probability scores

    Examples:
        >>> y_true = np.array([0, 0, 1, 1])  # Not used
        >>> y_pred = np.array([0.1, 0.2, 0.8, 0.9])
        >>> percentile(y_true, y_pred, q=75)
        0.825
    """
    y_pred_proba = _ensure_numpy(y_pred_proba)
    proba = _get_binary_proba(y_pred_proba)
    return float(np.percentile(proba, q))


# ============================================================================
# Metric Registry for Backward Compatibility
# ============================================================================

METRICS_REGISTRY = {
    "auroc": auroc,
    "accuracy": accuracy,
    "balanced_accuracy": balanced_accuracy,
    "precision": precision,
    "recall": recall,
    "f1": f1,
    "fpr": fpr,
    "mean_score": mean_score,
    "std_score": std_score,
}


def get_metric_by_name(name: str) -> Callable:
    """Get a metric function by name.

    Supports special syntax for parameterized metrics:
    - "recall@X" or "tpr@X": Recall at X% FPR
    - "auroc@X": Partial AUROC with max X% FPR
    - "percentileX": X-th percentile
    - "fpr@X": FPR at threshold X

    Args:
        name: Metric name or special syntax

    Returns:
        Metric function

    Examples:
        >>> m = get_metric_by_name("auroc")
        >>> m = get_metric_by_name("recall@5")  # 5% FPR
        >>> m = get_metric_by_name("percentile95")
    """
    # Check registry first
    if name in METRICS_REGISTRY:
        return METRICS_REGISTRY[name]

    # Handle special cases
    if name.startswith(("recall@", "tpr@")):
        fpr_value = float(name.split("@")[1]) / 100
        return create_recall_at_fpr_metric(fpr_value)

    elif name.startswith("auroc@"):
        max_fpr = float(name.split("@")[1]) / 100
        return functools.partial(partial_auroc, max_fpr=max_fpr)

    elif name.startswith("percentile"):
        q = float(name[10:])  # Extract number after "percentile"
        return create_percentile_metric(q)

    elif name.startswith("fpr@"):
        threshold = float(name.split("@")[1])
        return create_fpr_at_threshold_metric(threshold)

    else:
        raise ValueError(f"Unknown metric: {name}")
