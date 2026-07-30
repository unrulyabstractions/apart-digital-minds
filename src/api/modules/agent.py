"""The interface a model-backed module satisfies.

An agent is a module that owns a model and a memory. Any part of a mind that
reasons is one of these, whether it talks to you, watches another agent, or
edits somebody's context.

Implemented in `src.dminds.agents` by `Agent`, which adds `on_<kind>` dispatch
and a default handler. Subclass that.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Sequence

from ..memory import EpisodicStore, KeyValueStore, MessageStore
from ..models import LLM
from ..types import ChatMessage, Completion, GenOptions
from .module import Module


class Agent(Module):
    """A module with a model, a transcript, and instrumented calls.

    Attributes:
        llm: the model. Swap providers by changing one spec string.
        transcript: what this agent has said and heard. Editable.
        scratch: working state that is not part of the conversation.
        journal: episodic memory, or None if this agent has none.
    """

    llm: LLM
    transcript: MessageStore
    scratch: KeyValueStore
    journal: EpisodicStore | None

    @abstractmethod
    def prompt_messages(self) -> list[ChatMessage]:
        """The messages that will be sent to the model.

        Override to change context assembly: trim it, summarise it, inject
        recalled memories, reorder it.
        """

    @abstractmethod
    async def think(
        self,
        messages: Sequence[ChatMessage] | None = None,
        opts: GenOptions | None = None,
        tag: str = "",
    ) -> Completion:
        """One model call, logged with its model, tokens, and latency.

        Pass `messages` to reason over something other than this agent's own
        transcript, which is how a monitor reads another agent's context.
        `tag` names the call in the trace so you can tell a draft from a
        revision.
        """

    @abstractmethod
    async def say(self, text: str, tag: str = "say") -> Completion:
        """Append a user turn, call the model, append the reply."""
