"""Logs and traces. Instrumentation is always on; there is no flag.

A run writes `runs/<run_id>/trace.jsonl` and one file per module under
`runs/<run_id>/modules/`. Each event carries a logical tick, a UTC timestamp,
the module name, the event kind, and a duration where one applies.

Only the tick is logic. Wall time is for you to read.

Hook points in this module:

    Sink               implement `write(event)` and `close()` for your own sink
    Tracer.add_sink    attach it
    ModuleLog.note     log a line from your own handler
    ModuleLog.span     time a block and record its duration
    ModuleLog.child    a nested logger for a subagent, named `parent/child`
    read_trace         load a run back off disk to diff against another
    causal_chain       every event descending from one task
"""

from ..dminds.trace import (
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

#: Every event kind the runtime emits. Yours can be any string.
EVENT_KINDS = (
    TICK_START,
    TICK_END,
    TASK_EMIT,
    TASK_DELIVER,
    HANDLE_START,
    HANDLE_END,
    HANDLE_ERROR,
    LLM_REQUEST,
    LLM_RESPONSE,
    LLM_ERROR,
    MEMORY_WRITE,
    NOTE,
)

__all__ = [
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
