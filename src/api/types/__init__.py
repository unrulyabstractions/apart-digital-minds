"""The vocabulary that crosses every boundary.

These are data, not behaviour. They carry no policy and import nothing from
`src.dminds`, so an implementation can be swapped without changing what the
parts say to each other.

    messages.py   ChatMessage, Completion, GenOptions, Usage
    payloads.py   Payload, Text, Context, Vector
    tasks.py      Task, Route
    records.py    Episode, Event

Read this package first. Every interface in `src/api` is written in its terms.
"""

from .messages import (
    ChatMessage,
    Completion,
    GenOptions,
    Role,
    Usage,
    assistant,
    system,
    user,
)
from .payloads import Context, Payload, Text, Vector, preview
from .records import Episode, Event, compact
from .tasks import Route, Task

__all__ = [
    "Role",
    "ChatMessage",
    "Usage",
    "Completion",
    "GenOptions",
    "system",
    "user",
    "assistant",
    "Payload",
    "Text",
    "Context",
    "Vector",
    "preview",
    "Task",
    "Route",
    "Episode",
    "Event",
    "compact",
]
