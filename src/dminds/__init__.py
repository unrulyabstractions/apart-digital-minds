"""The implementations. Every one of them satisfies a contract in `src.api`.

Internal machinery lives here too, beside what uses it: `Host` is the narrow
view a module gets of its mind, and `Scheduler` is the clock. Neither is part
of the external surface in `src.api`.

    api.Module        -> BaseModule, Agent
    api.Ctx           -> Ctx
    api.LLM           -> BaseLLM, the providers, Cassette
    api.Host          -> Mind (narrow view, for modules)
    api.Mind          -> Mind (full view, for you)
    api.Agent         -> Subject, the target model a mind is built around
    api.MessageStore  -> Transcript
    api.KeyValueStore -> Scratchpad
    api.EpisodicStore -> Journal
    api.Tracer        -> RunTracer
    api.Logger        -> ModuleLog
    api.Sink          -> MemorySink, JsonlSink, PerModuleSink, ConsoleSink

No cognitive architecture is baked in. `examples/` shows several assembled from
these parts.

    from src.dminds import Mind, Agent, get_llm

    mind = Mind("demo")
    mind.add(Agent("assistant", get_llm("echo:")))
    replies = await mind.prompt("hello")
"""

from . import paths
from .agents import Agent, has_think, replace_think, split_think, strip_think
from .llm import (
    ALIASES,
    DEFAULT_MODELS,
    BaseLLM,
    Cassette,
    EchoLLM,
    CassetteMiss,
    Tape,
    available_providers,
    get_llm,
    merge_consecutive,
    parse_spec,
    register_provider,
    request_key,
    split_system,
    taped,
)
from .memory import Journal, Scratchpad, Transcript
from .mind import Mind, World, texts
from .subject import Ego, Subject
from .module import BaseModule, Ctx, UndeclaredChannel
from .host import Host, SchedulerFactory
from .scheduler import Scheduler, TickScheduler
from .trace import (
    ConsoleSink,
    JsonlSink,
    MemorySink,
    ModuleLog,
    PerModuleSink,
    RunTracer,
    read_trace,
)

__version__ = "0.1.0"

__all__ = [
    "paths",
    # runtime
    "Mind",
    "BaseModule",
    "Ctx",
    "Subject",
    "Ego",
    "UndeclaredChannel",
    "TickScheduler",
    "Scheduler",
    "Host",
    "SchedulerFactory",
    "texts",
    # agents
    "Agent",
    "split_think",
    "strip_think",
    "replace_think",
    "has_think",
    # models
    "BaseLLM",
    "EchoLLM",
    "get_llm",
    "register_provider",
    "available_providers",
    "parse_spec",
    "Cassette",
    "CassetteMiss",
    "taped",
    "split_system",
    "merge_consecutive",
    # memory
    "Transcript",
    "Scratchpad",
    "Journal",
    # observability
    "RunTracer",
    "MemorySink",
    "JsonlSink",
    "PerModuleSink",
    "ConsoleSink",
    "read_trace",
]
