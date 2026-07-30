"""The event kinds the runtime emits. Yours can be any string."""

TICK_START = "tick.start"
TICK_END = "tick.end"
TASK_EMIT = "task.emit"
TASK_DELIVER = "task.deliver"
HANDLE_START = "handle.start"
HANDLE_END = "handle.end"
HANDLE_ERROR = "handle.error"
LLM_REQUEST = "llm.request"
LLM_RESPONSE = "llm.response"
LLM_ERROR = "llm.error"
MEMORY_WRITE = "memory.write"
NOTE = "note"

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
