"""The contracts. `src.dminds` implements every one of them.

Nothing here imports from `src.dminds`. The dependency points one way, so you
can replace any implementation without disturbing the vocabulary the other
parts speak.

    types/           the data that crosses every boundary
      messages.py      ChatMessage, Completion, GenOptions, Usage
      payloads.py      Payload, Text, Context, Vector
      tasks.py         Task, Route
      records.py       Episode, Event

    modules/         the things that live inside a mind
      module.py        Module, Handler   -> BaseModule, FnModule
      context.py       Ctx               -> Ctx
      agent.py         Agent             -> Agent
      roles.py         Inspectable, Editor, Workspace, Speaker, InnerVoice

    models/          llm.py    LLM       -> BaseLLM, the providers, Cassette
    memory/          stores.py MessageStore, KeyValueStore, EpisodicStore
                                         -> Transcript, Scratchpad, Journal
    observability/   sinks.py  Sink      -> the four sinks
                     tracing.py Logger, Tracer -> ModuleLog, RunTracer
                     kinds.py  the event kinds
    runtime/         router.py Router    -> Bus
                     scheduler.py Scheduler -> TickScheduler
                     host.py   Host      -> Mind

Write against these when you want to swap a part.

    from src.api import Module, Agent, LLM, Sink   # what to implement
    from src.dminds import Mind, Bus, BaseModule   # what to use
    from src import Mind, Agent, get_llm           # both, flat
"""

from .memory import EpisodicStore, KeyValueStore, MessageStore, Recallable
from .models import LLM
from .modules import (
    Agent,
    Ctx,
    Editor,
    Handler,
    InnerVoice,
    Inspectable,
    Module,
    Speaker,
    Workspace,
)
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
    # module interfaces
    "Module",
    "Agent",
    "Ctx",
    "Handler",
    # roles
    "Inspectable",
    "Editor",
    "Workspace",
    "Speaker",
    "InnerVoice",
    # other interfaces
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
