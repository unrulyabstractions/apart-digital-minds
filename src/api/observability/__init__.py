"""Interfaces for logging and tracing.

Instrumentation is not optional in this runtime, so these contracts are what
every module and every model call is guaranteed to speak.

    kinds.py    the event kinds the runtime emits
    sinks.py    Sink        -> JsonlSink, PerModuleSink, ConsoleSink, MemorySink
    tracing.py  Logger      -> ModuleLog
                Tracer      -> RunTracer
"""

from .kinds import (
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
)
from .sinks import Sink
from .tracing import Logger, Tracer

__all__ = [
    "Sink",
    "Logger",
    "Tracer",
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
