"""Model access. One interface, many providers."""

from .base import LLM, merge_consecutive, split_system
from .record import Cassette, CassetteMiss, request_key
from .registry import (
    ALIASES,
    available_providers,
    get_llm,
    parse_spec,
    register_provider,
)
from .types import (
    ChatMessage,
    Completion,
    GenOptions,
    Role,
    Usage,
    assistant,
    system,
    user,
)

__all__ = [
    "LLM",
    "ChatMessage",
    "Completion",
    "GenOptions",
    "Role",
    "Usage",
    "system",
    "user",
    "assistant",
    "get_llm",
    "register_provider",
    "available_providers",
    "parse_spec",
    "ALIASES",
    "Cassette",
    "CassetteMiss",
    "request_key",
    "split_system",
    "merge_consecutive",
]
