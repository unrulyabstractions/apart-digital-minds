"""The contracts. `src.dminds` implements every one of them.

Nothing here imports from `src.dminds`. The dependency points one way, so you
can replace any implementation without disturbing the vocabulary the other
parts speak.

    mind.py          Mind, the thing you build and drive
    constants.py     WORLD, WILDCARD
    errors.py        RunawayMind, UndeclaredChannel

    types/           the data that crosses every boundary
      messages.py      ChatMessage, Completion, GenOptions, Usage
      payloads.py      Payload, Text, Context, Vector
      messages_flow.py Message, Link
      records.py       Episode, Event

    modules/         the things that live inside a mind
      module.py        Module            -> BaseModule
      context.py       Ctx               -> Ctx
      agent.py         Agent             -> Subject, Ego, your own
      roles.py         Editor, Workspace, InnerVoice

    models/          llm.py     LLM          -> BaseLLM, the providers
                     factory.py ModelFactory -> get_llm, taped(...)
    memory/          stores.py  MessageStore, KeyValueStore, EpisodicStore
                                            -> Transcript, Scratchpad, Journal
    observability/   sinks.py   Sink         -> the four sinks
                     tracing.py Logger, Tracer -> ModuleLog, RunTracer
                     kinds.py   the event kinds

This is the external surface and nothing else. The scheduler, and the narrow
view a module gets of its host, are machinery you never call; they live in
`src.dminds` beside the code that uses them.

Write against these when you want to swap a part.

    from src.api import Module, Agent, LLM, Sink   # what to implement
    from src.dminds import Mind, Subject, BaseModule  # what to use
    from src import Mind, Agent, get_llm           # both, flat
"""

from .memory import EpisodicStore, KeyValueStore, MessageStore
from .models import LLM, ModelFactory
from .modules import Agent, Ctx, Editor, InnerVoice, Module, Workspace
from .observability import (
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
from .constants import WILDCARD, WORLD
from .errors import RunawayMind, UndeclaredChannel
from .mind import Mind
from .types import (
    ChatMessage,
    Completion,
    Context,
    Episode,
    Event,
    GenOptions,
    Payload,
    Link,
    Message,
    Text,
    Usage,
    Vector,
    assistant,
    system,
    user,
)

__all__ = [
    # module interfaces
    "Module",
    "Agent",
    "Ctx",
    # roles
    "Editor",
    "Workspace",
    "InnerVoice",
    # other interfaces
    "LLM",
    "MessageStore",
    "KeyValueStore",
    "EpisodicStore",
    "Sink",
    "Logger",
    "Tracer",
    "Mind",
    "RunawayMind",
    "UndeclaredChannel",
    # factories
    "ModelFactory",
    # data
    "Text",
    "Context",
    "Vector",
    "Payload",
    "Message",
    "Link",
    "Event",
    "Episode",
    "ChatMessage",
    "Completion",
    "Usage",
    "GenOptions",
    "system",
    "user",
    "assistant",
    # constants
    "WORLD",
    "WILDCARD",
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
