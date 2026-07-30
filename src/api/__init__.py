"""The contracts. `src.dminds` implements every one of them.

Nothing here imports from `src.dminds`. The dependency points one way, so you
can replace any implementation without touching the vocabulary the other parts
speak.

    types.py           the data that crosses every boundary
    modules.py         Module, Ctx        -> BaseModule, Ctx
    models.py          LLM                -> BaseLLM and the providers
    memory.py          MessageStore, KeyValueStore, EpisodicStore
                                          -> Transcript, Scratchpad, Journal
    observability.py   Sink, Logger, Tracer
                                          -> JsonlSink and friends, ModuleLog,
                                             RunTracer
    runtime.py         Router, Scheduler, Host
                                          -> Bus, TickScheduler, Mind

Write against these when you want to swap a part. Import the implementations
from `src.dminds`, or import both from `src`.

    from src.api import Module, LLM, Sink      # what to implement
    from src.dminds import Mind, Agent, Bus    # what to use
    from src import Mind, Agent, Module, LLM   # both, flat
"""

from .memory import EpisodicStore, KeyValueStore, MessageStore, Recallable
from .models import LLM
from .modules import Ctx, Handler, Module
from .observability import (
    EVENT_KINDS,
    HANDLE_END,
    HANDLE_ERROR,
    HANDLE_START,
    LLM_ERROR,
    LLM_REQUEST,
    LLM_RESPONSE,
    MEMORY_WRITE,
    NOTE,
    TASK_DELIVER,
    TASK_EMIT,
    TICK_END,
    TICK_START,
    Logger,
    Sink,
    Tracer,
)
from .runtime import WILDCARD, WORLD, Host, RunawayMind, Router, Scheduler
from .types import (
    ChatMessage,
    Completion,
    Context,
    Episode,
    Event,
    GenOptions,
    Payload,
    Role,
    Route,
    Task,
    Text,
    Usage,
    Vector,
    assistant,
    preview,
    system,
    user,
)

__all__ = [
    # interfaces
    "Module",
    "Ctx",
    "Handler",
    "LLM",
    "MessageStore",
    "KeyValueStore",
    "EpisodicStore",
    "Recallable",
    "Sink",
    "Logger",
    "Tracer",
    "Router",
    "Scheduler",
    "Host",
    "RunawayMind",
    # data
    "Task",
    "Text",
    "Context",
    "Vector",
    "Payload",
    "Route",
    "Event",
    "Episode",
    "ChatMessage",
    "Completion",
    "Usage",
    "GenOptions",
    "Role",
    "system",
    "user",
    "assistant",
    "preview",
    # constants
    "WORLD",
    "WILDCARD",
    "EVENT_KINDS",
    "TICK_START",
    "TICK_END",
    "TASK_EMIT",
    "TASK_DELIVER",
    "HANDLE_START",
    "HANDLE_END",
    "HANDLE_ERROR",
    "LLM_REQUEST",
    "LLM_RESPONSE",
    "LLM_ERROR",
    "MEMORY_WRITE",
    "NOTE",
]
