"""How a mind builds its models."""

from __future__ import annotations

from typing import Any, Protocol

from .llm import LLM


class ModelFactory(Protocol):
    """Builds a model from a spec string.

    Every model in a mind is built through one of these, so wrapping a single
    function changes them all at once. `get_llm` is the default; `taped(...)`
    is the interesting one.

        mind = Mind("study", "openai:gpt-5", model_factory=taped("runs/tape.jsonl"))

    Wrap it to record every call, add a rate limiter, pin a temperature, or
    redirect a spec to a cheaper model while you develop.
    """

    def __call__(self, spec: str, **kwargs: Any) -> LLM: ...
