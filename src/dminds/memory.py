"""Memory: three kinds, all small enough to read in one sitting.

    Transcript   what was said, in order. This is what goes to the model.
    Scratchpad   working state. A dict that logs its writes.
    Journal      episodic memory. Append-only, survives the process, recallable.

They implement `api.MessageStore`, `api.KeyValueStore`, and `api.EpisodicStore`.
Those three share no base class on purpose. A transcript is a sequence, a
scratchpad is a mapping, and a journal is a searchable log. Forcing one
interface onto all three would hide what each is for.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Sequence

from ..api.memory import EpisodicStore, KeyValueStore, MessageStore
from ..api.observability import MEMORY_WRITE, Logger
from ..api.types import ChatMessage, Episode


class Transcript(MessageStore):
    """An ordered conversation you can edit.

    Editing is the point. An interceptor rewriting a thought calls `replace`,
    and the next model call sees the edited history as if it were its own.
    """

    def __init__(
        self,
        messages: Iterable[ChatMessage] | None = None,
        log: Logger | None = None,
    ):
        self.messages: list[ChatMessage] = list(messages or [])
        self.log = log

    # -- reading -------------------------------------------------------

    def __len__(self) -> int:
        return len(self.messages)

    def __iter__(self) -> Iterator[ChatMessage]:
        return iter(self.messages)

    def __getitem__(self, index):
        return self.messages[index]

    def window(self, n: int, keep_system: bool = True) -> list[ChatMessage]:
        """The last `n` messages, with system messages always kept in front."""
        if n >= len(self.messages):
            return list(self.messages)
        head = [m for m in self.messages if m.role == "system"] if keep_system else []
        tail = [m for m in self.messages if m.role != "system"][-n:]
        return head + tail

    def tagged(self, key: str, value: Any = True) -> list[ChatMessage]:
        """Every message whose `meta[key]` matches. Find all the thoughts."""
        return [m for m in self.messages if m.meta.get(key) == value]

    def last(self, role: str | None = None) -> ChatMessage | None:
        for message in reversed(self.messages):
            if role is None or message.role == role:
                return message
        return None

    def copy(self) -> "Transcript":
        return Transcript([m.copy() for m in self.messages], self.log)

    # -- writing -------------------------------------------------------

    def append(self, message: ChatMessage) -> ChatMessage:
        self.messages.append(message)
        self._record("append", message)
        return message

    def extend(self, messages: Iterable[ChatMessage]) -> None:
        for message in messages:
            self.append(message)

    def replace(self, index: int, message: ChatMessage) -> ChatMessage:
        """Swap one message. Negative indices work."""
        old = self.messages[index]
        self.messages[index] = message
        self._record("replace", message, replaced=old.content[:120])
        return message

    def replace_all(self, messages: Sequence[ChatMessage]) -> None:
        """Overwrite the whole history. This is what a context editor emits."""
        before = len(self.messages)
        self.messages = [m.copy() for m in messages]
        if self.log:
            self.log.event(
                MEMORY_WRITE,
                store="transcript",
                op="replace_all",
                window=[m.to_dict() for m in self.messages],
                summary=f"context replaced: {before} -> {len(self.messages)} messages",
            )

    def edit_text(self, index: int, fn: Callable[[str], str]) -> ChatMessage:
        """Rewrite one message's text in place, keeping role and meta."""
        old = self.messages[index]
        return self.replace(index, ChatMessage(old.role, fn(old.content), dict(old.meta)))

    def clear(self, keep_system: bool = True) -> None:
        self.messages = (
            [m for m in self.messages if m.role == "system"] if keep_system else []
        )

    def _record(self, op: str, message: ChatMessage, **extra: Any) -> None:
        """Log the write, in full.

        The content is here rather than only in the summary so that a trace is
        a recording of the window, not a description of one. That is what
        makes a run playable afterwards.
        """
        if self.log is None:
            return
        self.log.event(
            MEMORY_WRITE,
            store="transcript",
            op=op,
            role=message.role,
            content=message.content,
            meta=dict(message.meta),
            summary=f"{op} {message.role}: {message.content[:80]}",
            **extra,
        )


class Scratchpad(KeyValueStore):
    """Working memory. A dict that leaves a trace of every write."""

    def __init__(self, log: Logger | None = None, **initial: Any):
        self.data: dict[str, Any] = dict(initial)
        self.log = log

    def __getitem__(self, key: str) -> Any:
        return self.data[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self.set(key, value)

    def __contains__(self, key: str) -> bool:
        return key in self.data

    def get(self, key: str, default: Any = None) -> Any:
        return self.data.get(key, default)

    def set(self, key: str, value: Any) -> Any:
        self.data[key] = value
        if self.log:
            self.log.event(
                MEMORY_WRITE,
                store="scratchpad",
                key=key,
                summary=f"{key} = {str(value)[:60]}",
            )
        return value

    def bump(self, key: str, by: int = 1) -> int:
        return self.set(key, self.get(key, 0) + by)

    def as_text(self) -> str:
        """Render for injection into a prompt."""
        return "\n".join(f"{k}: {v}" for k, v in self.data.items())


_WORD = re.compile(r"[a-z0-9]+")


def _words(text: str) -> set[str]:
    return set(_WORD.findall(text.lower()))


class Journal(EpisodicStore):
    """Episodic memory. Append-only, optionally on disk, recallable.

    Recall scores word overlap and breaks ties by recency. No embeddings, so
    there is nothing to install and nothing hidden. Pass your own `scorer` to
    swap in a real retriever without touching anything else.
    """

    def __init__(
        self,
        path: str | Path | None = None,
        log: Logger | None = None,
        scorer: Callable[[str, Episode], float] | None = None,
        recency_weight: float = 0.01,
    ):
        self.path = Path(path) if path else None
        self.log = log
        self.scorer = scorer
        self.recency_weight = recency_weight
        self.episodes: list[Episode] = []
        if self.path and self.path.exists():
            self._load()

    def _load(self) -> None:
        for line in self.path.read_text().splitlines():
            if line.strip():
                self.episodes.append(Episode(**json.loads(line)))

    def remember(self, text: str, t: int = 0, source: str = "", **meta: Any) -> Episode:
        episode = Episode(text=text, t=t, source=source, meta=meta)
        self.episodes.append(episode)
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a") as handle:
                handle.write(json.dumps(episode.to_dict()) + "\n")
        if self.log:
            self.log.event(
                MEMORY_WRITE,
                store="journal",
                summary=f"remembered: {text[:80]}",
                source=source,
            )
        return episode

    def recall(self, query: str, k: int = 5) -> list[Episode]:
        """The `k` most relevant episodes, best first."""
        if not self.episodes:
            return []
        if self.scorer is not None:
            scored = [(self.scorer(query, e), i, e) for i, e in enumerate(self.episodes)]
        else:
            terms = _words(query)
            scored = [
                (
                    len(terms & _words(e.text)) + self.recency_weight * i,
                    i,
                    e,
                )
                for i, e in enumerate(self.episodes)
            ]
        scored.sort(key=lambda row: (-row[0], -row[1]))
        return [e for score, _, e in scored[:k] if score > 0]

    def recent(self, k: int = 5) -> list[Episode]:
        return self.episodes[-k:]

    def as_text(self, episodes: Sequence[Episode] | None = None) -> str:
        """Render for injection into a prompt."""
        chosen = self.episodes if episodes is None else episodes
        return "\n".join(f"- {e.text}" for e in chosen)

    def __len__(self) -> int:
        return len(self.episodes)

