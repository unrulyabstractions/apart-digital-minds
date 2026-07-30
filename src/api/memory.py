"""Interfaces for the three kinds of memory.

They share no base class on purpose. A conversation is a sequence, working
state is a mapping, and episodic memory is a searchable log. One interface over
all three would hide what each is for.

Implemented in `src.dminds.memory` by `Transcript`, `Scratchpad`, and `Journal`.
"""

from __future__ import annotations

from typing import Any, Callable, Iterable, Protocol, Sequence, runtime_checkable

from .types import ChatMessage, Episode


@runtime_checkable
class MessageStore(Protocol):
    """An ordered conversation you can edit.

    Editing is the point. An interceptor rewriting a thought calls
    `replace_all`, and the next model call sees the edited history as if it
    were its own.
    """

    def __len__(self) -> int: ...

    def __iter__(self): ...

    def __getitem__(self, index): ...

    def append(self, message: ChatMessage) -> ChatMessage:
        """Add one turn."""

    def extend(self, messages: Iterable[ChatMessage]) -> None:
        """Add several turns."""

    def replace(self, index: int, message: ChatMessage) -> ChatMessage:
        """Swap one message. Negative indices work."""

    def replace_all(self, messages: Sequence[ChatMessage]) -> None:
        """Overwrite the whole history. This is what a context editor emits."""

    def window(self, n: int, keep_system: bool = True) -> list[ChatMessage]:
        """The last `n` messages, with system messages kept in front."""

    def tagged(self, key: str, value: Any = True) -> list[ChatMessage]:
        """Every message whose `meta[key]` matches."""

    def last(self, role: str | None = None) -> ChatMessage | None:
        """The most recent message, optionally of one role."""

    def edit_text(self, index: int, fn: Callable[[str], str]) -> ChatMessage:
        """Rewrite one message's text, keeping its role and meta."""

    def clear(self, keep_system: bool = True) -> None:
        """Drop the history."""


@runtime_checkable
class KeyValueStore(Protocol):
    """Working state. A mapping that leaves a trace of every write."""

    def __getitem__(self, key: str) -> Any: ...

    def __setitem__(self, key: str, value: Any) -> None: ...

    def __contains__(self, key: str) -> bool: ...

    def get(self, key: str, default: Any = None) -> Any: ...

    def set(self, key: str, value: Any) -> Any:
        """Write, and record that you wrote."""

    def as_text(self) -> str:
        """Render for injection into a prompt."""


@runtime_checkable
class EpisodicStore(Protocol):
    """Memory that outlives the process and can be searched."""

    def __len__(self) -> int: ...

    def remember(self, text: str, t: int = 0, source: str = "", **meta: Any) -> Episode:
        """Record one episode."""

    def recall(self, query: str, k: int = 5) -> Sequence[Episode]:
        """The `k` most relevant episodes, best first."""

    def recent(self, k: int = 5) -> Sequence[Episode]:
        """The `k` newest episodes."""

    def as_text(self, episodes: Sequence[Episode] | None = None) -> str:
        """Render for injection into a prompt."""


@runtime_checkable
class Recallable(Protocol):
    """The smallest useful memory contract, if the ones above are too much."""

    def remember(self, text: str, **meta: Any) -> Any: ...

    def recall(self, query: str, k: int = 5) -> Sequence[Any]: ...
