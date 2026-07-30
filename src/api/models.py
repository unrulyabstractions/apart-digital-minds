"""Model access. One interface, many providers, chosen by a string.

Swapping backends is a string change and nothing else:

    get_llm("echo:")                             # no keys, deterministic
    get_llm("openai:gpt-5")
    get_llm("anthropic:claude-opus-5")
    get_llm("gemini:gemini-2.5-flash")
    get_llm("ollama:qwen3:8b")                   # local Qwen
    get_llm("ollama:gemma3:4b")                  # local Gemma
    get_llm("hf:Qwen/Qwen3-4B-Instruct-2507")    # local, weights in-process
    get_llm("vllm:Qwen/Qwen3-8B")                # OpenAI-compatible server

Hook points in this module:

    LLM               subclass it and implement `_chat`. That is the whole
                      contract. Timing and threading are handled for you.
    register_provider add a backend without touching the package
    Cassette          wrap any LLM to record its calls, then replay them
    GenOptions.extra  pass provider-specific arguments through untouched
"""

from ..dminds.llm.base import LLM, merge_consecutive, split_system
from ..dminds.llm.record import Cassette, CassetteMiss, request_key
from ..dminds.llm.registry import (
    ALIASES,
    DEFAULT_MODELS,
    available_providers,
    get_llm,
    parse_spec,
    register_provider,
)
from ..dminds.llm.types import (
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
    "get_llm",
    "register_provider",
    "available_providers",
    "parse_spec",
    "ALIASES",
    "DEFAULT_MODELS",
    "ChatMessage",
    "Completion",
    "GenOptions",
    "Role",
    "Usage",
    "system",
    "user",
    "assistant",
    "Cassette",
    "CassetteMiss",
    "request_key",
    "split_system",
    "merge_consecutive",
]
