"""`BaseLLM`: the standard implementation of `api.LLM`.

Providers are written synchronously because every vendor SDK is synchronous
and sync code is easier to read. `chat` runs `_chat` on a worker thread, so the
runtime stays async and two modules can call two different models at once.

Subclass this and implement `_chat`. Implement `api.LLM` directly only if you
need something this does not do.
"""

from __future__ import annotations

import asyncio
import time
from abc import abstractmethod
from typing import Sequence

from ...api.models import LLM
from ...api.types import ChatMessage, Completion, GenOptions


class BaseLLM(LLM):
    """Handles timing and thread offload. Subclasses write one method."""

    def __init__(self, model: str, spec: str | None = None):
        self.model = model
        self.spec = spec or f"{type(self).__name__}:{model}"

    @abstractmethod
    def _chat(self, messages: list[ChatMessage], opts: GenOptions) -> Completion:
        """Do the actual call. Synchronous. Must return a Completion."""

    async def chat(
        self,
        messages: Sequence[ChatMessage],
        opts: GenOptions | None = None,
    ) -> Completion:
        """Call the model. Times the call and stamps the model name."""
        opts = opts or GenOptions()
        msgs = [m.copy() for m in messages]
        start = time.perf_counter()
        completion = await asyncio.to_thread(self._chat, msgs, opts)
        completion.latency_s = time.perf_counter() - start
        if not completion.model:
            completion.model = self.model
        return completion

    def __repr__(self) -> str:
        return f"<{type(self).__name__} {self.spec}>"


def split_system(messages: list[ChatMessage]) -> tuple[str, list[ChatMessage]]:
    """Pull leading system messages out into one string.

    Anthropic and Gemini take the system prompt as a separate argument rather
    than as a message. Both providers use this helper.
    """
    system_parts = [m.content for m in messages if m.role == "system"]
    rest = [m for m in messages if m.role != "system"]
    return "\n\n".join(system_parts), rest


def merge_consecutive(messages: list[ChatMessage]) -> list[ChatMessage]:
    """Collapse neighbouring same-role messages into one.

    Some providers reject two user messages in a row. Injecting a thought into
    a transcript produces exactly that, so providers that care call this.
    """
    merged: list[ChatMessage] = []
    for m in messages:
        if merged and merged[-1].role == m.role:
            merged[-1] = ChatMessage(
                m.role, merged[-1].content + "\n\n" + m.content, merged[-1].meta
            )
        else:
            merged.append(m.copy())
    return merged
