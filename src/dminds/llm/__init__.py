"""Model implementations. The interface they satisfy is `src.api.models.LLM`."""

from .base import BaseLLM, merge_consecutive, split_system
from .record import Cassette, CassetteMiss, Tape, request_key, taped
from .registry import (
    ALIASES,
    DEFAULT_MODELS,
    available_providers,
    get_llm,
    parse_spec,
    register_provider,
)

__all__ = [
    "BaseLLM",
    "get_llm",
    "register_provider",
    "available_providers",
    "parse_spec",
    "ALIASES",
    "DEFAULT_MODELS",
    "Cassette",
    "CassetteMiss",
    "Tape",
    "request_key",
    "taped",
    "split_system",
    "merge_consecutive",
]
