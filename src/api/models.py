"""The interface every chat model satisfies.

One method matters: `chat`. Everything the runtime does with a model goes
through it, so a provider, a fake, a cache, and a recorder are interchangeable.

Implemented in `src.dminds.llm` by `BaseLLM`, which handles timing and thread
offload so a provider only writes the synchronous call. Subclass `BaseLLM`
unless you need something it does not do.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence

from .types import ChatMessage, Completion, GenOptions


class LLM(ABC):
    """A chat model.

    Attributes:
        model: the provider's own name for the weights, e.g. `"gpt-5"`.
        spec: how it was asked for, e.g. `"openai:gpt-5"`. Used in logs.
    """

    model: str
    spec: str

    @abstractmethod
    async def chat(
        self,
        messages: Sequence[ChatMessage],
        opts: GenOptions | None = None,
    ) -> Completion:
        """Answer a conversation.

        Must not mutate `messages`. Must set `Completion.model` and should set
        `latency_s`.
        """

    def close(self) -> None:
        """Release anything held open. Local backends override this."""
