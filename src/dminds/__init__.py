"""Core implementation.

Five ideas, and nothing else:

    Module      something that holds a queue and handles tasks
    Task        one unit of work, routed between modules
    Mind        the assembly of modules, routes, a clock, and a trace
    Scheduler   ticks the clock; nothing emitted at t is seen before t+1
    LLM         one chat interface, many providers, chosen by a string

Everything else in this package is a convenience built on those. No cognitive
architecture is baked in. `examples/` shows several assembled from the parts.

Prefer `src.api` when you import. It re-exports all of this, grouped by
concern, and names the hook points. Reaching in here directly is fine too.

    from src.api import Mind, Agent, get_llm, Text

    mind = Mind("demo")
    mind.add(Agent("assistant", get_llm("echo:")))
    replies = await mind.prompt("hello")
"""

from .agents import (
    Agent,
    has_think,
    replace_think,
    split_think,
    strip_think,
)
from .bus import Bus, Route
from .llm import (
    LLM,
    Cassette,
    ChatMessage,
    Completion,
    GenOptions,
    Usage,
    assistant,
    available_providers,
    get_llm,
    register_provider,
    system,
    user,
)
from .memory import Episode, Journal, Recallable, Scratchpad, Transcript
from .messages import Context, Payload, Task, Text, Vector
from .mind import WORLD, Mind, texts
from .module import Ctx, FnModule, Module
from .scheduler import RunawayMind, Scheduler
from .trace import (
    ConsoleSink,
    Event,
    JsonlSink,
    MemorySink,
    ModuleLog,
    PerModuleSink,
    Tracer,
    causal_chain,
    read_trace,
)

__version__ = "0.1.0"

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
    # models
    "LLM",
    "get_llm",
    "register_provider",
    "available_providers",
    "ChatMessage",
    "Completion",
    "GenOptions",
    "Usage",
    "system",
    "user",
    "assistant",
    "Cassette",
    # memory
    "Transcript",
    "Scratchpad",
    "Journal",
    "Episode",
    "Recallable",
    # instrumentation
    "Tracer",
    "ModuleLog",
    "Event",
    "MemorySink",
    "JsonlSink",
    "PerModuleSink",
    "ConsoleSink",
    "read_trace",
    "causal_chain",
]
