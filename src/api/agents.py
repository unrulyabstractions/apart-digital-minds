"""Model-backed modules, and helpers for reasoning tags.

`Agent` is deliberately thin. It gives you one instrumented way to call a model
and one obvious default handler, then gets out of the way.

Hook points in this module:

    Agent.on_<kind>       write one method per task kind
    Agent.think           the instrumented model call. Pass `messages=` to
                          reason over another agent's context
    Agent.prompt_messages override to change how context is assembled
    split_think           separate reasoning from output
    replace_think         swap the contents of a reasoning block
"""

from ..dminds.agents import (
    THINK_RE,
    Agent,
    has_think,
    replace_think,
    split_think,
    strip_think,
)

__all__ = [
    "Agent",
    "split_think",
    "strip_think",
    "replace_think",
    "has_think",
    "THINK_RE",
]
