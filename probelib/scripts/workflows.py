"""
Unified high-level workflow functions for training and evaluating probes.
"""

import functools
from typing import TYPE_CHECKING, Any, Callable, Mapping, Optional, Union

import torch
from jaxtyping import Float, Int

from ..datasets import DialogueDataset
from ..metrics import (
    auroc,
    get_metric_by_name,
    recall_at_fpr,
    with_bootstrap,
)
from ..probes import BaseProbe
from ..processing import collect_activations
from ..processing.activations import ActivationIterator, Activations
from ..types import Dialogue, Label

if TYPE_CHECKING:
    from transformers import PreTrainedModel
    from transformers.tokenization_utils_base import PreTrainedTokenizerBase

    from ..masks import MaskFunction

# Type aliases for clarity
BaseProbeInput = Union[BaseProbe, Mapping[str, BaseProbe]]
DataInput = Union[DialogueDataset, list[Dialogue]]
PredictionsOutput = Union[
    Float[torch.Tensor, "n_examples"], Mapping[str, Float[torch.Tensor, "n_examples"]]
]
MetricsDict = Mapping[str, Any]
MetricsOutput = Union[MetricsDict, Mapping[str, MetricsDict]]


# TODO: rename to train_probes_parallel
def train_probes(
    probes: BaseProbeInput,
    model: "PreTrainedModel",
    tokenizer: "PreTrainedTokenizerBase",
    data: DataInput,
    *,
    labels: list[Label] | Int[torch.Tensor, "n_examples"] | None = None,
    mask: Optional["MaskFunction"] = None,
    batch_size: int = 32,
    streaming: bool = True,
    verbose: bool = True,
    **activation_kwargs: Any,
) -> None:
    """Train one or many probes while reusing a single activation pass.

    Callers provide the probes plus model/tokenizer/data, and the function caches activations,
    streams when datasets are large, and fans out training to each probe. It is
    intentionally thin so users can compose additional logging/loop logic around
    it without losing the shared caching benefits.

    Args:
        probes: Single probe or mapping name → probe instance.
        model: Language model whose activations are collected.
        tokenizer: Tokenizer aligned with the model.
        data: Dialogue dataset or list of :class:`Dialogue` objects.
        labels: Labels as ``Label`` enums or integer tensor.
        batch_size: Number of sequences per activation batch.
        streaming: Whether to force streaming activations.
        verbose: Toggle progress reporting.
        **activation_kwargs: Forwarded to :func:`collect_activations` for advanced
            control (e.g. generation prompts).
    """
    # 1. Normalize inputs
    is_single_probe = isinstance(probes, BaseProbe)
    probes = {"_single": probes} if is_single_probe else probes

    if labels is None:
        if not isinstance(data, DialogueDataset):
            raise ValueError("Labels are required when data is not a DialogueDataset")
        labels = data.labels

    # 2. Convert labels to tensor
    if isinstance(labels, list) and labels and isinstance(labels[0], Label):
        labels_tensor = torch.tensor([label.value for label in labels])
    else:
        labels_tensor = (
            torch.tensor(labels) if not isinstance(labels, torch.Tensor) else labels
        )

    # 3. Determine all required layers
    all_layers = set()
    for probe in probes.values():
        all_layers.add(probe.layer)

    # 4. Collect activations once for all layers
    activations = collect_activations(
        model=model,
        tokenizer=tokenizer,
        data=data,
        layers=sorted(all_layers),
        mask=mask,  # Pass mask for token selection
        batch_size=batch_size,
        streaming=streaming,
        verbose=verbose,
        **activation_kwargs,
    )

    # 5. Train each probe
    if isinstance(activations, ActivationIterator):
        # Streaming mode
        _train_probes_streaming(probes, activations, labels_tensor)
    else:
        # In-memory mode
        _train_probes_batch(probes, activations, labels_tensor)


def _train_probes_batch(
    probes: dict[str, BaseProbe],
    activations: Activations,
    labels: torch.Tensor,
) -> None:
    """Train all probes using in-memory activations."""
    # Validate label count
    if len(labels) != activations.batch_size:
        raise ValueError(
            f"Number of labels ({len(labels)}) doesn't match "
            f"batch size ({activations.batch_size})"
        )

    for name, probe in probes.items():
        # Get activations for this probe's layers
        probe_acts = activations.filter_layers([probe.layer])

        # Train probe
        probe.fit(probe_acts, labels)


def _train_probes_streaming(
    probes: dict[str, BaseProbe],
    activation_iter: ActivationIterator,
    labels: torch.Tensor,
) -> None:
    """Train all probes in streaming mode."""
    for batch_activations in activation_iter:
        for probe in probes.values():
            probe.partial_fit(
                batch_activations, labels[batch_activations.batch_indices]
            )


# TODO: rename to evaluate_probes_parallel
def evaluate_probes(
    probes: BaseProbeInput,
    model: "PreTrainedModel",
    tokenizer: "PreTrainedTokenizerBase",
    data: DataInput,
    *,
    labels: list[Label] | Int[torch.Tensor, "n_examples"] | None = None,
    mask: Optional["MaskFunction"] = None,
    batch_size: int = 32,
    streaming: bool = True,
    metrics: list[Callable | str] | None = None,
    bootstrap: bool | dict[str, Any] | None = None,
    verbose: bool = True,
    **activation_kwargs: Any,
) -> tuple[PredictionsOutput, MetricsOutput]:
    """
    Unified evaluation function for single or multiple probes.

    Args:
        probe_or_probes: Single BaseProbe or dict mapping names to BaseProbes
        model: Language model to extract activations from
        tokenizer: Tokenizer for the model
        data: Either a DialogueDataset or list of Dialogue objects
        labels: Ground truth labels for evaluation
        batch_size: Batch size for activation collection
        metrics: List of metric functions or names (defaults to standard set)
        bootstrap: ``None`` (default) disables bootstrap; pass ``True`` to apply
            default bootstrap settings or a dict of keyword arguments accepted by
            :func:`probelib.metrics.with_bootstrap` for customisation.
        verbose: Whether to show progress bars
        **activation_kwargs: Additional args passed to collect_activations

    Returns:
        Tuple of (predictions, metrics) where:
        - predictions: Tensor or dict of tensors with predicted probabilities
        - metrics: Dict or dict of dicts with computed metrics

    Examples:
        >>> # Single probe
        >>> predictions, metrics = evaluate_probes(
        ...     probe, model, tokenizer, test_data, test_labels
        ... )
        >>> print(f"AUROC: {metrics['auroc']:.3f}")

        >>> # Multiple probes
        >>> predictions, metrics = evaluate_probes(
        ...     probes, model, tokenizer, test_data, test_labels,
        ...     metrics=["auroc", "balanced_accuracy"]
        ... )
        >>> print(f"Early layers AUROC: {metrics['early']['auroc']:.3f}")
    """
    # 1. Normalize inputs
    is_single_probe = isinstance(probes, BaseProbe)
    probes = {"_single": probes} if is_single_probe else probes

    if labels is None:
        if not isinstance(data, DialogueDataset):
            raise ValueError("Labels are required when data is not a DialogueDataset")
        labels = data.labels

    # 2. Convert labels to tensor
    if isinstance(labels, list) and labels and isinstance(labels[0], Label):
        labels_tensor = torch.tensor([label.value for label in labels])
        valid_indices = None
    else:
        labels_tensor = (
            torch.tensor(labels) if not isinstance(labels, torch.Tensor) else labels
        )
        valid_indices = None

    # 3. Set default metrics and convert strings to functions
    if metrics is None:
        # Default metrics using function-based API
        metrics = [
            auroc,
            functools.partial(recall_at_fpr, fpr=0.001),  # recall@0.1%
            functools.partial(recall_at_fpr, fpr=0.01),  # recall@1%
        ]
    else:
        # Convert any string metrics to functions
        converted_metrics = []
        for metric in metrics:
            if isinstance(metric, str):
                # Convert string to function using registry
                converted_metrics.append(get_metric_by_name(metric))
            else:
                # Already a function
                converted_metrics.append(metric)
        metrics = converted_metrics

    # Normalise bootstrap configuration
    if isinstance(bootstrap, bool):
        bootstrap_kwargs: dict[str, Any] | None = {} if bootstrap else None
    else:
        bootstrap_kwargs = bootstrap

    # 4. Determine all required layers
    all_layers = set()
    for probe in probes.values():
        layers = probe.layer
        if isinstance(layers, int):
            all_layers.add(layers)
        else:
            all_layers.update(layers)

    # 5. Collect activations (never stream for evaluation)
    activations = collect_activations(
        model=model,
        tokenizer=tokenizer,
        data=data,
        layers=sorted(all_layers),
        mask=mask,  # Pass mask for token selection
        batch_size=batch_size,
        streaming=streaming,
        verbose=verbose,
        **activation_kwargs,
    )

    # 6. Evaluate based on activation type
    if isinstance(activations, ActivationIterator):
        # Streaming mode
        all_predictions, all_metrics = _evaluate_probes_streaming(
            probes, activations, labels_tensor, metrics, bootstrap_kwargs
        )
    else:
        # In-memory mode
        # Filter activations if we filtered labels
        if valid_indices is not None:
            # Create new Activations object with filtered samples
            filtered_acts = Activations(
                activations=activations.activations[:, valid_indices],
                attention_mask=activations.attention_mask[valid_indices],
                input_ids=activations.input_ids[valid_indices],
                detection_mask=activations.detection_mask[valid_indices],
                layer_indices=activations.layer_indices,
            )
            activations = filtered_acts

        all_predictions, all_metrics = _evaluate_probes_batch(
            probes, activations, labels_tensor, metrics, bootstrap_kwargs
        )

    # 7. Return in same format as input
    if is_single_probe:
        return all_predictions["_single"], all_metrics["_single"]
    else:
        return all_predictions, all_metrics


def _evaluate_probes_batch(
    probes: dict[str, BaseProbe],
    activations: Activations,
    labels: torch.Tensor,
    metrics: list[Callable],
    bootstrap_kwargs: dict[str, Any] | None,
) -> tuple[dict[str, torch.Tensor], dict[str, MetricsDict]]:
    """Evaluate all probes using in-memory activations."""

    all_predictions = {}
    all_metrics = {}

    for name, probe in probes.items():
        # Get probe-specific activations
        probe_acts = activations.filter_layers([probe.layer])

        # Make predictions (predict_proba returns [n_samples, 2])
        probs = probe.predict_proba(probe_acts)
        # Get positive class probabilities
        preds = probs[:, 1]

        all_predictions[name] = preds

        # Compute metrics using new function-based API
        probe_metrics = {}

        # Convert to numpy for metrics
        y_true = (
            labels.detach().cpu().numpy()
            if isinstance(labels, torch.Tensor)
            else labels
        )
        y_pred = (
            preds.detach().cpu().numpy() if isinstance(preds, torch.Tensor) else preds
        )

        for metric_fn in metrics:
            # Get metric name
            if hasattr(metric_fn, "__name__"):
                metric_name = metric_fn.__name__
            elif hasattr(metric_fn, "func"):  # functools.partial
                # Create name from partial function
                base_name = metric_fn.func.__name__
                if metric_fn.keywords:
                    params = "_".join(f"{k}={v}" for k, v in metric_fn.keywords.items())
                    metric_name = f"{base_name}_{params}"
                else:
                    metric_name = base_name
            else:
                metric_name = str(metric_fn)

            metric_callable = metric_fn
            if (
                bootstrap_kwargs is not None
                and not getattr(metric_fn, "_probelib_bootstrap", False)
            ):
                metric_callable = with_bootstrap(**bootstrap_kwargs)(metric_fn)

            # Compute metric (optionally bootstrapped)
            result = metric_callable(y_true, y_pred)
            probe_metrics[metric_name] = result

        all_metrics[name] = probe_metrics

    return all_predictions, all_metrics


def _evaluate_probes_streaming(
    probes: dict[str, BaseProbe],
    activation_iter: ActivationIterator,
    labels: torch.Tensor,
    metrics: list[Callable],
    bootstrap_kwargs: dict[str, Any] | None,
) -> tuple[dict[str, torch.Tensor], dict[str, MetricsDict]]:
    """Evaluate all probes in streaming mode."""

    # Initialize prediction storage that preserves original order
    n_samples = len(labels)
    probe_predictions = {
        name: torch.zeros(n_samples, device=labels.device) for name in probes
    }
    labels_seen = set()

    # Process each batch
    for batch_activations in activation_iter:
        batch_size = batch_activations.batch_size

        # Get the correct indices for this batch
        if batch_activations.batch_indices is not None:
            batch_indices = batch_activations.batch_indices
            labels_seen.update(batch_indices)
        else:
            # Fallback to sequential ordering (for backward compatibility)
            start_idx = len(labels_seen)
            end_idx = start_idx + batch_size
            if end_idx > len(labels):
                raise ValueError(
                    f"Not enough labels. Expected at least {end_idx}, "
                    f"but got {len(labels)}"
                )
            batch_indices = list(range(start_idx, end_idx))
            labels_seen.update(batch_indices)

        # Get predictions for each probe
        for name, probe in probes.items():
            probe_batch_acts = batch_activations.filter_layers([probe.layer])
            probs = probe.predict_proba(probe_batch_acts)
            # Get positive class probabilities and store in correct positions
            preds = probs[:, 1]
            # Ensure preds is on the same device as probe_predictions
            if isinstance(preds, torch.Tensor):
                preds = preds.to(probe_predictions[name].device)
            probe_predictions[name][batch_indices] = preds

    # Verify we saw all samples
    if len(labels_seen) != len(labels):
        raise ValueError(
            f"Label count mismatch. Saw {len(labels_seen)} samples but "
            f"{len(labels)} labels were provided"
        )

    # Compute metrics for each probe using new API
    all_metrics = {}
    for name, preds in probe_predictions.items():
        probe_metrics = {}

        # Convert to numpy for metrics
        y_true = (
            labels.detach().cpu().numpy()
            if isinstance(labels, torch.Tensor)
            else labels
        )
        y_pred = (
            preds.detach().cpu().numpy() if isinstance(preds, torch.Tensor) else preds
        )

        for metric_fn in metrics:
            # Get metric name
            if hasattr(metric_fn, "__name__"):
                metric_name = metric_fn.__name__
            elif hasattr(metric_fn, "func"):  # functools.partial
                # Create name from partial function
                base_name = metric_fn.func.__name__
                if metric_fn.keywords:
                    params_list = [f"{k}={v}" for k, v in metric_fn.keywords.items()]
                    params = (
                        ", ".join(params_list)
                        if len(params_list) > 1
                        else params_list[0]
                        if params_list
                        else ""
                    )
                    metric_name = f"{base_name} ({params})"
                else:
                    metric_name = base_name
            else:
                metric_name = str(metric_fn)

            metric_callable = metric_fn
            if (
                bootstrap_kwargs is not None
                and not getattr(metric_fn, "_probelib_bootstrap", False)
            ):
                metric_callable = with_bootstrap(**bootstrap_kwargs)(metric_fn)

            result = metric_callable(y_true, y_pred)
            probe_metrics[metric_name] = result

        all_metrics[name] = probe_metrics

    return probe_predictions, all_metrics
