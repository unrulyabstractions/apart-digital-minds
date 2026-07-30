"""Record real model calls once, then replay them forever.

A run is only replayable if every non-deterministic thing is captured at its
boundary. In this runtime there is exactly one such boundary, the model call,
so wrapping it is enough to make a whole experiment reproducible.

    live = get_llm("openai:gpt-5")
    llm = Cassette(live, "runs/tape.jsonl")   # calls once, replays after

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

from .base import LLM
from .types import ChatMessage, Completion, GenOptions, Usage


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


class Cassette(LLM):
    """Wraps any LLM and persists its answers to a JSONL tape."""

    def __init__(self, inner: LLM, path: str | Path, mode: str = "auto"):
        super().__init__(inner.model, f"cassette({inner.spec})")
        if mode not in ("auto", "replay", "record"):
            raise ValueError(f"mode must be auto, replay, or record. Got {mode!r}.")
        self.inner = inner
        self.path = Path(path)
        self.mode = mode
        self.hits = 0
        self.misses = 0
        self._entries: dict[str, list[dict]] = defaultdict(list)
        self._used: dict[str, int] = defaultdict(int)
        if mode == "record":
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text("")
        else:
            self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        for line in self.path.read_text().splitlines():
            if not line.strip():
                continue
            entry = json.loads(line)
            self._entries[entry["key"]].append(entry)

    def _append(self, key: str, completion: Completion) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        entry = {
            "key": key,
            "spec": self.inner.spec,
            "model": completion.model,
            "text": completion.text,
            "input_tokens": completion.usage.input_tokens,
            "output_tokens": completion.usage.output_tokens,
            "finish_reason": completion.finish_reason,
        }
        with self.path.open("a") as handle:
            handle.write(json.dumps(entry) + "\n")

    def _chat(self, messages: list[ChatMessage], opts: GenOptions) -> Completion:
        key = request_key(self.inner.spec, messages, opts)

        if self.mode != "record":
            # Identical requests can legitimately return different answers, so
            # take them in the order they were recorded.
            index = self._used[key]
            candidates = self._entries.get(key, [])
            if index < len(candidates):
                self._used[key] += 1
                self.hits += 1
                entry = candidates[index]
                return Completion(
                    text=entry["text"],
                    model=entry.get("model", self.model),
                    usage=Usage(
                        entry.get("input_tokens", 0), entry.get("output_tokens", 0)
                    ),
                    finish_reason=entry.get("finish_reason", ""),
                    raw={"replayed_from": str(self.path), "key": key},
                )
            if self.mode == "replay":
                raise CassetteMiss(
                    f"No recorded call {key} on tape {self.path}. "
                    f"Re-run with mode='auto' to record it."
                )

        self.misses += 1
        completion = self.inner._chat(messages, opts)
        self._append(key, completion)
        return completion

    def close(self) -> None:
        self.inner.close()
