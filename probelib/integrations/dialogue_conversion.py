"""Dialogue conversion utilities for external library integration.

This module provides functions to convert between probelib's Dialogue format
and external library message formats (e.g., inspect_ai ChatMessage).
"""

from typing import TYPE_CHECKING, Any

from probelib.types import Dialogue, Message, Role

if TYPE_CHECKING:
    from inspect_ai.model import ChatMessage


def dialogue_from_inspect_messages(
    messages: list["ChatMessage"],
) -> Dialogue:
    """Convert inspect_ai ChatMessage list to probelib Dialogue.

    This utility converts messages from inspect_ai's format to probelib's format,
    handling role mapping. It's designed to work with inspect_ai's AgentState.messages.

    Detection of which messages to analyze is handled separately via mask functions
    during tokenization (e.g., `probelib.masks.assistant()` or `probelib.masks.last_assistant()`).

    Args:
        messages: List of ChatMessage from inspect_ai (or compatible format)

    Returns:
        probelib Dialogue object (list of Message)

    Examples:
        # From inspect_ai AgentState
        >>> from inspect_ai.agent import AgentState
        >>> import probelib as pl
        >>> state: AgentState = ...
        >>> dialogue = pl.integrations.dialogue_from_inspect_messages(state.messages)
        >>>
        >>> # Use with mask to control detection
        >>> acts = pl.collect_activations(
        ...     model, tokenizer, [dialogue],
        ...     layers=[16],
        ...     mask=pl.masks.last_assistant(),  # Only detect last assistant
        ...     ...
        ... )

    Note:
        - Messages with unsupported roles (e.g., "tool") are skipped
        - The .text property is used to extract message content as a string
    """
    dialogue: Dialogue = []

    for msg in messages:
        # Map inspect_ai roles to probelib
        role_map = {
            "system": "system",
            "user": "user",
            "assistant": "assistant",
        }

        msg_role = getattr(msg, "role", None)
        if msg_role not in role_map:
            # Skip unsupported roles (e.g., tool messages)
            continue

        role: Role = role_map[msg_role]  # type: ignore

        # Extract text content
        # inspect_ai messages have a .text property that extracts plain text
        content = getattr(msg, "text", str(getattr(msg, "content", "")))

        dialogue.append(
            Message(
                role=role,
                content=content,
            )
        )

    return dialogue


def dialogue_to_inspect_messages(
    dialogue: Dialogue,
) -> list[dict[str, Any]]:
    """Convert probelib Dialogue to inspect_ai-compatible message format.

    This is the inverse of dialogue_from_inspect_messages. It converts
    probelib dialogues back to a format that can be used with inspect_ai.

    Args:
        dialogue: probelib Dialogue object

    Returns:
        List of message dictionaries that can be used to construct ChatMessage objects

    Examples:
        >>> from probelib import Dialogue, Message
        >>> dialogue = [
        ...     Message(role="user", content="Hello", detect=False),
        ...     Message(role="assistant", content="Hi there", detect=True),
        ... ]
        >>> messages = dialogue_to_inspect_messages(dialogue)
        >>> # Use with inspect_ai
        >>> from inspect_ai.model import ChatMessageUser, ChatMessageAssistant
        >>> chat_messages = [
        ...     ChatMessageUser(content=msg["content"]) if msg["role"] == "user"
        ...     else ChatMessageAssistant(content=msg["content"])
        ...     for msg in messages
        ... ]

    Note:
        The detect flag is not preserved in the conversion as inspect_ai
        messages don't have an equivalent field.
    """
    messages = []
    for msg in dialogue:
        messages.append(
            {
                "role": msg.role,
                "content": msg.content,
            }
        )
    return messages
