"""The public API. Everything you need, in one import.

    from src.api import Mind, Agent, get_llm, Text

Import by concern when you prefer it:

    from src.api.core import Mind, Module, Ctx
    from src.api.models import get_llm, register_provider
    from src.api.memory import Journal, Transcript
    from src.api.agents import Agent, split_think
    from src.api.observability import Tracer, read_trace

Core implementation lives in `src.dminds`. Everything here is a re-export, so
reaching into `src.dminds` directly is always safe. This layer exists to give
the hook points one stable place to be found.
"""

from .agents import THINK_RE, Agent, has_think, replace_think, split_think, strip_think
from .core import (
    WORLD,
    Bus,
    Context,
    Ctx,
    FnModule,
    Mind,
    Module,
    Payload,
    Route,
    RunawayMind,
    Scheduler,
    Task,
    Text,
    Vector,
    handler_name,
    texts,
)
from .memory import Episode, Journal, Recallable, Scratchpad, Transcript
from .models import (
    ALIASES,
    DEFAULT_MODELS,
    LLM,
    Cassette,
    CassetteMiss,
    ChatMessage,
    Completion,
    GenOptions,
    Role,
    Usage,
    assistant,
    available_providers,
    get_llm,
    merge_consecutive,
    parse_spec,
    register_provider,
    request_key,
    split_system,
    system,
    user,
)
from .observability import (
    EVENT_KINDS,
    ConsoleSink,
    Event,
    JsonlSink,
    MemorySink,
    ModuleLog,
    PerModuleSink,
    Sink,
    Tracer,
    causal_chain,
    read_trace,
)

__all__ = [
    # core
    "Mind",
    "Module",
    "FnModule",
    "Ctx",
    "Task",
    "Scheduler",
    "RunawayMind",
    "Bus",
    "Route",
    "WORLD",
    "texts",
    "handler_name",
    # payloads
    "Text",
    "Context",
    "Vector",
    "Payload",
    # agents
    "Agent",
    "split_think",
    "strip_think",
    "replace_think",
    "has_think",
    "THINK_RE",
    # models
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
    # memory
    "Transcript",
    "Scratchpad",
    "Journal",
    "Episode",
    "Recallable",
    # observability
    "Tracer",
    "ModuleLog",
    "Event",
    "Sink",
    "MemorySink",
    "JsonlSink",
    "PerModuleSink",
    "ConsoleSink",
    "read_trace",
    "causal_chain",
    "EVENT_KINDS",
]
