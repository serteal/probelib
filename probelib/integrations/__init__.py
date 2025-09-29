"""Integration utilities for external libraries.

This module provides utilities to integrate probelib with external frameworks
like inspect_ai and control-arena. These utilities handle format conversions
and provide convenience wrappers.
"""

from probelib.integrations.dialogue_conversion import (
    dialogue_from_inspect_messages,
    dialogue_to_inspect_messages,
)

__all__ = [
    "dialogue_from_inspect_messages",
    "dialogue_to_inspect_messages",
]