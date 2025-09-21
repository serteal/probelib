"""Tests for probelib.metrics module."""

import numpy as np
import pytest
import torch

from probelib.metrics import (
    accuracy,
    auroc,
    f1,
    fpr,
    fpr_at_threshold,
    mean_score,
    partial_auroc,
    percentile,
    precision,
    recall,
    recall_at_fpr,
    std_score,
    tpr_at_fpr,
    with_bootstrap,
)


class TestBasicMetrics:
    """Test basic metric functions."""

    def test_perfect_separation_auroc(self):
        """Test AUROC with perfect separation."""
        # Combine positive and negative scores
        y_true = np.array([1, 1, 1, 0, 0, 0])
        y_pred = np.array([0.8, 0.9, 0.7, 0.2, 0.3, 0.1])

        # Note: auroc has bootstrap decorator by default, returns tuple
        result = auroc(y_true, y_pred)
        # Extract point estimate (first element of tuple)
        auroc_value = result[0] if isinstance(result, tuple) else result
        assert auroc_value == 1.0

    def test_random_classification_auroc(self):
        """Test AUROC with random classification (should be ~0.5)."""
        # Overlapping distributions
        y_true = np.array([1, 1, 1, 0, 0, 0])
        y_pred = np.array([0.4, 0.5, 0.6, 0.4, 0.5, 0.6])

        result = auroc(y_true, y_pred)
        auroc_value = result[0] if isinstance(result, tuple) else result
        assert abs(auroc_value - 0.5) < 0.1

    def test_partial_auroc(self):
        """Test partial AUROC with max FPR constraint."""
        y_true = np.array([1, 1, 1, 1, 0, 0, 0, 0])
        y_pred = np.array([0.8, 0.9, 0.7, 0.85, 0.2, 0.3, 0.1, 0.25])

        # Full AUROC should be 1.0
        result = auroc(y_true, y_pred)
        full_auroc = result[0] if isinstance(result, tuple) else result
        assert full_auroc == 1.0

        # Partial AUROC at 10% FPR
        result = partial_auroc(y_true, y_pred, max_fpr=0.1)
        partial_auroc_value = result[0] if isinstance(result, tuple) else result
        assert partial_auroc_value == 1.0

    def test_recall_at_fpr_perfect_separation(self):
        """Test recall@FPR with perfect separation."""
        y_true = np.array([1, 1, 1, 0, 0, 0])
        y_pred = np.array([0.8, 0.9, 0.7, 0.2, 0.3, 0.1])

        # With perfect separation, recall should be 1.0 at any reasonable FPR
        result = recall_at_fpr(y_true, y_pred, fpr=0.05)
        recall_value = result[0] if isinstance(result, tuple) else result
        assert recall_value == 1.0

        result = recall_at_fpr(y_true, y_pred, fpr=0.10)
        recall_value = result[0] if isinstance(result, tuple) else result
        assert recall_value == 1.0

    def test_tpr_alias(self):
        """Test that tpr_at_fpr is equivalent to recall_at_fpr."""
        y_true = np.array([1, 1, 1, 0, 0, 0])
        y_pred = np.array([0.8, 0.9, 0.7, 0.2, 0.3, 0.1])

        recall_result = recall_at_fpr(y_true, y_pred, fpr=0.05)
        tpr_result = tpr_at_fpr(y_true, y_pred, fpr=0.05)

        # Both should return tuples (with bootstrap)
        recall_value = (
            recall_result[0] if isinstance(recall_result, tuple) else recall_result
        )
        tpr_value = tpr_result[0] if isinstance(tpr_result, tuple) else tpr_result
        assert recall_value == tpr_value

    def test_realistic_recall_at_fpr(self):
        """Test recall@FPR with realistic overlapping scores."""
        # Create overlapping but distinguishable distributions
        np.random.seed(42)
        pos_mask = np.ones(100, dtype=bool)
        pos_mask[:50] = False  # First 50 are negative

        scores = np.concatenate(
            [
                np.random.normal(0.3, 0.1, 50),  # Negative scores
                np.random.normal(0.7, 0.1, 50),  # Positive scores
            ]
        )
        y_true = pos_mask.astype(int)

        result_1 = recall_at_fpr(y_true, scores, fpr=0.01)
        result_5 = recall_at_fpr(y_true, scores, fpr=0.05)
        result_10 = recall_at_fpr(y_true, scores, fpr=0.10)

        recall_1 = result_1[0] if isinstance(result_1, tuple) else result_1
        recall_5 = result_5[0] if isinstance(result_5, tuple) else result_5
        recall_10 = result_10[0] if isinstance(result_10, tuple) else result_10

        # Higher FPR should give higher or equal recall
        assert recall_1 <= recall_5 <= recall_10
        # All should be between 0 and 1
        assert 0 <= recall_1 <= 1
        assert 0 <= recall_5 <= 1
        assert 0 <= recall_10 <= 1

    def test_accuracy_metric(self):
        """Test accuracy metric."""
        y_true = np.array([1, 1, 0, 0])
        y_pred = np.array([0.8, 0.7, 0.2, 0.3])

        result = accuracy(y_true, y_pred)
        acc_value = result[0] if isinstance(result, tuple) else result
        assert acc_value == 1.0  # Perfect classification at 0.5 threshold

    def test_precision_recall_f1(self):
        """Test precision, recall, and F1 metrics."""
        y_true = np.array([1, 1, 1, 0, 0, 0])
        y_pred = np.array([0.8, 0.7, 0.3, 0.2, 0.1, 0.6])  # One FN, one FP

        # Precision: TP/(TP+FP) = 2/3
        result = precision(y_true, y_pred)
        prec_value = result[0] if isinstance(result, tuple) else result
        assert prec_value == pytest.approx(2 / 3)

        # Recall: TP/(TP+FN) = 2/3
        result = recall(y_true, y_pred)
        rec_value = result[0] if isinstance(result, tuple) else result
        assert rec_value == pytest.approx(2 / 3)

        # F1: harmonic mean of precision and recall
        result = f1(y_true, y_pred)
        f1_value = result[0] if isinstance(result, tuple) else result
        assert f1_value == pytest.approx(2 / 3)


class TestBootstrap:
    """Test bootstrap functionality."""

    def test_bootstrap_decorator(self):
        """Test applying bootstrap decorator to metrics."""

        # Create a simple custom metric
        def custom_mean(y_true, y_pred_proba):
            return float(np.mean(y_pred_proba))

        # Apply bootstrap
        bootstrapped_metric = with_bootstrap(
            n_bootstrap=100, confidence_level=0.95, random_state=42
        )(custom_mean)

        y_true = np.array([1, 1, 0, 0])
        y_pred = np.array([0.8, 0.7, 0.2, 0.3])

        result = bootstrapped_metric(y_true, y_pred)
        assert isinstance(result, tuple)
        assert len(result) == 3

        point_est, ci_lower, ci_upper = result
        assert ci_lower <= point_est <= ci_upper

    def test_bootstrap_with_existing_metrics(self):
        """Test that bootstrap can be opt-in for built-in metrics."""
        y_true = np.array([1, 1, 1, 0, 0, 0])
        y_pred = np.array([0.8, 0.9, 0.7, 0.2, 0.3, 0.1])

        # Default metrics now return raw floats
        result = auroc(y_true, y_pred)
        assert isinstance(result, float)

        # Opt-in bootstrap
        bootstrapped_auroc = with_bootstrap(
            n_bootstrap=100, confidence_level=0.95, random_state=42
        )(auroc)
        result = bootstrapped_auroc(y_true, y_pred)
        assert isinstance(result, tuple)
        assert len(result) == 3

        point_est, ci_lower, ci_upper = result
        assert 0.5 <= point_est <= 1.0
        if not np.isnan(ci_lower) and not np.isnan(ci_upper):
            assert ci_lower <= point_est <= ci_upper

    def test_bootstrap_confidence_levels(self):
        """Test different confidence levels."""
        np.random.seed(42)

        # Create a metric without bootstrap first
        def base_auroc(y_true, y_pred_proba):
            from sklearn.metrics import roc_auc_score

            proba = y_pred_proba[:, 1] if y_pred_proba.ndim == 2 else y_pred_proba
            return float(roc_auc_score(y_true, proba))

        # Apply bootstrap with different confidence levels
        bootstrap_90 = with_bootstrap(
            n_bootstrap=100, confidence_level=0.90, random_state=42
        )(base_auroc)

        bootstrap_95 = with_bootstrap(
            n_bootstrap=100, confidence_level=0.95, random_state=42
        )(base_auroc)

        # Use a dataset with some variability to avoid NaN CIs
        y_true = np.array([1, 1, 1, 1, 0, 0, 0, 0, 1, 0])
        y_pred = np.array([0.8, 0.75, 0.7, 0.65, 0.35, 0.3, 0.25, 0.2, 0.6, 0.4])

        _, ci_low_90, ci_high_90 = bootstrap_90(y_true, y_pred)
        _, ci_low_95, ci_high_95 = bootstrap_95(y_true, y_pred)

        # 95% CI should be wider than or equal to 90% CI (if not NaN)
        if not (
            np.isnan(ci_low_90)
            or np.isnan(ci_high_90)
            or np.isnan(ci_low_95)
            or np.isnan(ci_high_95)
        ):
            assert (ci_high_95 - ci_low_95) >= (ci_high_90 - ci_low_90)


class TestDistributionMetrics:
    """Test distribution statistics metrics."""

    def test_mean_std_score(self):
        """Test mean and std score metrics."""
        y_true = np.array([0, 0, 0, 0, 0])  # Not used for these metrics
        y_pred = np.array([0.1, 0.2, 0.3, 0.4, 0.5])

        mean = mean_score(y_true, y_pred)
        assert mean == pytest.approx(0.3)

        std = std_score(y_true, y_pred)
        assert std == pytest.approx(np.std(y_pred))

    def test_percentile_metrics(self):
        """Test percentile metrics."""
        y_true = np.array([0] * 9)  # Not used
        y_pred = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])

        p50 = percentile(y_true, y_pred, q=50)
        assert p50 == pytest.approx(0.5)

        p90 = percentile(y_true, y_pred, q=90)
        assert p90 == pytest.approx(
            0.82
        )  # numpy's percentile with default interpolation

        p10 = percentile(y_true, y_pred, q=10)
        assert p10 == pytest.approx(0.18)


class TestFPRMetrics:
    """Test false positive rate metrics."""

    def test_fpr_at_threshold(self):
        """Test FPR at specific thresholds."""
        y_true = np.array([0, 0, 0, 0, 0])  # All negative
        y_pred = np.array([0.1, 0.2, 0.3, 0.6, 0.8])

        # Default threshold (0.5)
        result = fpr(y_true, y_pred)
        assert isinstance(result, float)
        assert result == 0.4  # 2 out of 5 > 0.5

        # Custom threshold
        result = fpr_at_threshold(y_true, y_pred, threshold=0.7)
        assert isinstance(result, float)
        assert result == 0.2  # 1 out of 5 > 0.7

    def test_fpr_with_mixed_dataset(self):
        """Test FPR with mixed positive/negative examples."""
        y_true = np.array([0, 0, 0, 0, 1, 1])  # 4 negative, 2 positive
        y_pred = np.array([0.1, 0.2, 0.3, 0.6, 0.8, 0.9])

        result = fpr_at_threshold(y_true, y_pred, threshold=0.5)
        fpr_value = result[0] if isinstance(result, tuple) else result
        # FPR = FP / (FP + TN) = 1/4 = 0.25
        assert fpr_value == 0.25


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_arrays(self):
        """Test behavior with empty arrays."""
        # Should raise ValueError for empty arrays
        with pytest.raises(ValueError):
            auroc(np.array([]), np.array([]))

    def test_single_class(self):
        """Test behavior with only one class."""
        y_true = np.array([1, 1, 1])  # Only positive
        y_pred = np.array([0.8, 0.9, 0.7])

        # Should raise ValueError for single class
        with pytest.raises(
            ValueError, match="Cannot compute AUROC with only one class"
        ):
            auroc(y_true, y_pred)

    def test_identical_scores(self):
        """Test with identical scores."""
        y_true = np.array([1, 1, 0, 0])
        y_pred = np.array([0.5, 0.5, 0.5, 0.5])

        # AUROC should be 0.5 for identical scores
        result = auroc(y_true, y_pred)
        auroc_value = result[0] if isinstance(result, tuple) else result
        assert abs(auroc_value - 0.5) < 0.1

    def test_mixed_tensor_types(self):
        """Test with mixed tensor types."""
        y_true = torch.tensor([1, 1, 0, 0], dtype=torch.int32)
        y_pred = torch.tensor([0.8, 0.9, 0.2, 0.3], dtype=torch.float64)

        result = auroc(y_true, y_pred)
        auroc_value = result[0] if isinstance(result, tuple) else result
        assert 0.5 <= auroc_value <= 1.0


class TestCustomMetrics:
    """Test creating and using custom metrics."""

    def test_custom_metric(self):
        """Test creating a custom metric."""

        def custom_balanced_score(y_true, y_pred_proba, weight=0.5):
            """Custom metric combining precision and recall."""
            from sklearn.metrics import precision_score, recall_score

            y_true = np.asarray(y_true)
            if isinstance(y_pred_proba, torch.Tensor):
                y_pred_proba = y_pred_proba.detach().cpu().numpy()

            # Convert probabilities to predictions
            y_pred = (y_pred_proba > 0.5).astype(int)

            prec = precision_score(y_true, y_pred, zero_division=0)
            rec = recall_score(y_true, y_pred, zero_division=0)

            return weight * prec + (1 - weight) * rec

        y_true = np.array([1, 1, 0, 0])
        y_pred = np.array([0.8, 0.7, 0.2, 0.3])

        result = custom_balanced_score(y_true, y_pred)
        assert 0 <= result <= 1

        # Apply bootstrap to custom metric
        bootstrapped_custom = with_bootstrap(n_bootstrap=50, random_state=42)(
            custom_balanced_score
        )

        result = bootstrapped_custom(y_true, y_pred)
        assert isinstance(result, tuple)
        assert len(result) == 3
