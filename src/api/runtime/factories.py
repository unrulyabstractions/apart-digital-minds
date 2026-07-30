"""Factories, so the runtime's parts can be chosen rather than hardcoded.

A `Mind` needs a scheduler and a tracer. If it constructed them itself the
interfaces next door would be decorative. Instead it takes them, and falls back
to the shipped implementations when you say nothing.

The scheduler and the tracer cannot be passed as finished objects. A scheduler
needs the host it will drive, and a tracer needs the run id. Both arrive as
factories for that reason.

`ModelFactory` is the interesting one. Every model in a mind is built through
it, so wrapping one function changes every model at once.

    mind = Mind("run", model_factory=taped("runs/tape.jsonl"))
    llm = mind.model("openai:gpt-5")     # recorded, and replayed next time
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

from ..models import LLM
from ..observability import Tracer
from .scheduler import Scheduler

if TYPE_CHECKING:  # pragma: no cover
    from .host import Host


class ModelFactory(Protocol):
    """Builds a model from a spec string.

    `get_llm` is the default. Wrap it to record every call, to add a rate
    limiter, to pin a temperature, or to redirect a spec to a cheaper model
    during development.
    """

    def __call__(self, spec: str, **kwargs: Any) -> LLM: ...


class SchedulerFactory(Protocol):
    """Builds a scheduler bound to the host it will drive."""

    def __call__(self, host: "Host") -> Scheduler: ...


class TracerFactory(Protocol):
    """Builds a tracer for one run."""

    def __call__(self, run_id: str) -> Tracer: ...
