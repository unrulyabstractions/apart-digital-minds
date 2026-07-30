"""Record real model calls once, then replay them forever.

A run is only replayable if every non-deterministic thing is captured at its
boundary. In this runtime there is exactly one such boundary, the model call,
so wrapping it is enough to make a whole experiment reproducible.

    llm = Cassette(get_llm("openai:gpt-5"), paths.tape("study"))

For a whole mind, attach a factory instead of wrapping models one at a time:

    mind = Mind("study", "openai:gpt-5", model_factory=taped(paths.tape("study")))

Modes:
    "auto"    replay a matching call, otherwise call the model and record it
    "replay"  replay only, raise on a miss
    "record"  always call the model, overwrite the tape
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Sequence

from ...api.models import LLM
from ...api.types import ChatMessage, Completion, GenOptions, Usage

MODES = ("auto", "replay", "record")


class CassetteMiss(RuntimeError):
    """Replay was asked for a call the tape does not contain."""


def request_key(spec: str, messages: list[ChatMessage], opts: GenOptions) -> str:
    """A stable fingerprint of one request."""
    blob = json.dumps(
        {
            "spec": spec,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "temperature": opts.temperature,
            "max_tokens": opts.max_tokens,
            "stop": opts.stop,
            "seed": opts.seed,
            "extra": {k: repr(v) for k, v in sorted(opts.extra.items())},
        },
        sort_keys=True,
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


class Tape:
    """One recording on disk, and the cursor into it.

    The cursor has to be shared. Two agents on the same model asking the same
    question produce the same key, so if each cassette kept its own cursor they
    would both replay the first answer and the second would be lost. One tape,
    one cursor, entries consumed in the order they were recorded.
    """

    def __init__(self, path: str | Path, mode: str = "auto"):
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}. Got {mode!r}.")
        self.path = Path(path)
        self.mode = mode
        self.entries: dict[str, list[dict]] = defaultdict(list)
        self.used: dict[str, int] = defaultdict(int)
        if mode == "record":
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text("")
        else:
            self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        for line in self.path.read_text().splitlines():
            if line.strip():
                entry = json.loads(line)
                self.entries[entry["key"]].append(entry)

    def take(self, key: str) -> dict | None:
        """The next unconsumed entry for this key, or None."""
        index = self.used[key]
        candidates = self.entries.get(key, [])
        if index >= len(candidates):
            return None
        self.used[key] += 1
        return candidates[index]

    def append(self, key: str, spec: str, completion: Completion) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        entry = {
            "key": key,
            "spec": spec,
            "model": completion.model,
            "text": completion.text,
            "input_tokens": completion.usage.input_tokens,
            "output_tokens": completion.usage.output_tokens,
            "finish_reason": completion.finish_reason,
        }
        with self.path.open("a") as handle:
            handle.write(json.dumps(entry) + "\n")

    def __len__(self) -> int:
        return sum(len(v) for v in self.entries.values())


class Cassette(LLM):
    """Wraps any LLM and persists its answers to a tape.

    This implements `api.LLM` directly rather than extending `BaseLLM`, because
    it must wrap anything satisfying the interface, and the interface only
    promises the async `chat`. On a hit it never touches the inner model.
    """

    def __init__(
        self,
        inner: LLM,
        tape: str | Path | Tape,
        mode: str = "auto",
    ):
        """Wrap `inner`, taping to `tape`.

        Pass a `Tape` to share one recording between several models. Pass a
        path for the common case of one model, one file.
        """
        self.inner = inner
        self.model = inner.model
        self.spec = f"cassette({inner.spec})"
        self.tape = tape if isinstance(tape, Tape) else Tape(tape, mode)
        self.hits = 0
        self.misses = 0

    @property
    def mode(self) -> str:
        return self.tape.mode

    @property
    def path(self) -> Path:
        return self.tape.path

    async def chat(
        self,
        messages: Sequence[ChatMessage],
        opts: GenOptions | None = None,
    ) -> Completion:
        opts = opts or GenOptions()
        msgs = [m.copy() for m in messages]
        key = request_key(self.inner.spec, msgs, opts)

        if self.tape.mode != "record":
            entry = self.tape.take(key)
            if entry is not None:
                self.hits += 1
                return Completion(
                    text=entry["text"],
                    model=entry.get("model", self.model),
                    usage=Usage(
                        entry.get("input_tokens", 0), entry.get("output_tokens", 0)
                    ),
                    finish_reason=entry.get("finish_reason", ""),
                    raw={"replayed_from": str(self.tape.path), "key": key},
                )
            if self.tape.mode == "replay":
                raise CassetteMiss(
                    f"No recorded call {key} on tape {self.tape.path}. "
                    f"Re-run with mode='auto' to record it."
                )

        self.misses += 1
        completion = await self.inner.chat(msgs, opts)
        self.tape.append(key, self.inner.spec, completion)
        return completion

    def close(self) -> None:
        self.inner.close()


def taped(
    path: str | Path,
    mode: str = "auto",
    factory: Callable[..., LLM] | None = None,
) -> Callable[..., LLM]:
    """A `ModelFactory` that tapes every model it builds to one recording.

    This is the reason models are built through the mind rather than by
    calling `get_llm` directly. One argument makes a whole experiment
    reproducible, however many agents it has and whichever providers they use.

        mind = Mind("study", "openai:gpt-5", model_factory=taped(paths.tape("study")))
        mind.add(Agent("a", mind.model("openai:gpt-5")))
        mind.add(Agent("b", mind.model("ollama:qwen3:8b")))

    Every cassette shares one `Tape`, so the file is cleared once and the
    replay cursor is global. Two agents on the same model asking the same
    question therefore get their own recorded answers back, in order.
    """
    tape = Tape(path, mode)
    build = factory if factory is not None else _default_factory()

    def make(spec: str, **kwargs: Any) -> LLM:
        return Cassette(build(spec, **kwargs), tape)

    return make


def _default_factory() -> Callable[..., LLM]:
    from .registry import get_llm

    return get_llm
