"""Data processing utilities for activation collection and manipulation."""

from .activations import (
    ActivationIterator,
    Activations,
    SequencePooling,
    collect_activations,
    detect_collection_strategy,
)
from .tokenization import tokenize_dataset, tokenize_dialogues

__all__ = [
    # Activation collection
    "Activations",
    "collect_activations",
    "detect_collection_strategy",
    # Pooling configuration
    "SequencePooling",
    # Streaming
    "ActivationIterator",
    # Tokenization
    "tokenize_dataset",
    "tokenize_dialogues",
]
