"""One set of local weights, used by many modules.

Building a model per module is right for hosted providers and wrong for local
ones. A mind with ten shadows would load ten copies of the same checkpoint,
which is ten times the memory for no benefit, and on one accelerator the ten
calls do not run any faster in parallel than in sequence.

    shared = SharedLLM(get_llm("hf:Qwen/Qwen3-4B-Instruct-2507"))
    mind = Mind("study", shared)
    ShadowReader("affect", shared, AFFECT)

Calls on a shared model are serialized. That is what makes an activation hook
safe: a steered call cannot overlap an unsteered one and quietly steer it too.
Take the lock with `exclusive()` around anything that mutates the model.
"""

from __future__ import annotations

import asyncio
from typing import Sequence

from ...api.models import LLM
from ...api.types import ChatMessage, Completion, GenOptions


class SharedLLM(LLM):
    """Wraps one model so several modules can hold it without copying it."""

    def __init__(self, inner: LLM):
        self.inner = inner
        self.model = inner.model
        self.spec = inner.spec
        self.lock = asyncio.Lock()
        self.calls = 0

    async def chat(
        self,
        messages: Sequence[ChatMessage],
        opts: GenOptions | None = None,
    ) -> Completion:
        async with self.lock:
            self.calls += 1
            return await self.inner.chat(messages, opts)

    async def score(self, messages, choices) -> dict[str, float]:
        """Forward the scoring call, if the model underneath can score."""
        inner_score = getattr(self.inner, "score", None)
        if inner_score is None:
            raise AttributeError(f"{self.spec} cannot score choices.")
        async with self.lock:
            self.calls += 1
            import asyncio

            return await asyncio.to_thread(inner_score, messages, choices)

    async def hidden(self, messages, layer=0.6) -> list[float]:
        """Forward the activation read, if the model underneath has weights."""
        inner_hidden = getattr(self.inner, "hidden", None)
        if inner_hidden is None:
            raise AttributeError(f"{self.spec} has no activations to read.")
        async with self.lock:
            self.calls += 1
            import asyncio

            return await asyncio.to_thread(inner_hidden, messages, layer)

    def can_score(self) -> bool:
        return hasattr(self.inner, "score")

    def can_read(self) -> bool:
        return hasattr(self.inner, "hidden")

    def exclusive(self):
        """Hold the model still. Use around a hook or a weight edit.

        The caller runs its own `chat` through `inner` while holding this, so
        the wrapper's own lock is not taken twice.
        """
        return self.lock

    def close(self) -> None:
        self.inner.close()

    def __repr__(self) -> str:
        return f"<SharedLLM {self.spec} calls={self.calls}>"


def sharing() -> "callable":
    """A `ModelFactory` that hands out one shared model per spec.

    Pass it to a mind and every part built from the same spec gets the same
    weights.

        mind = Mind("study", "hf:Qwen/Qwen3-4B-Instruct-2507",
                    model_factory=sharing())
    """
    from .registry import get_llm

    cache: dict[str, SharedLLM] = {}

    def make(spec: str, **kwargs) -> LLM:
        if spec not in cache:
            cache[spec] = SharedLLM(get_llm(spec, **kwargs))
        return cache[spec]

    def close_all() -> None:
        """Release every model this factory handed out.

        A sweep calls this between models. Local weights are not collected
        while the factory still refers to them, and the model that fails to
        load is always the last one in the list.
        """
        for llm in cache.values():
            llm.close()
        cache.clear()

    make.close_all = close_all
    return make
