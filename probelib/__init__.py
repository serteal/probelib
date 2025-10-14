"""
probelib: A library for training classifiers on LLM activations.

This library provides tools for:
- Dataset handling
- Activation collection using hooked and pruned models
- Probe training and evaluation
- Standard evaluation metrics
"""

from . import datasets, masks, metrics, probes, processing, scripts
from .models import HookedModel
from .processing import Activations, collect_activations
from .processing.activations import SequencePooling, detect_collection_strategy
from .types import Dialogue, Label, Message
from .visualization import print_metrics, visualize_mask

__version__ = "0.1.0"

__all__ = [
    "Message",
    "Label",
    "Dialogue",
    "collect_activations",
    "detect_collection_strategy",
    "HookedModel",
    "Activations",
    "print_metrics",
    "visualize_mask",
    "scripts",
    "probes",
    "datasets",
    "processing",
    "metrics",
    "masks",
    "SequencePooling",
]
