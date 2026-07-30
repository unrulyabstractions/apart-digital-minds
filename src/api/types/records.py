"""What gets written down: remembered episodes and traced events."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass(slots=True)
class Episode:
    """One remembered thing."""

    text: str
    t: int = 0
    source: str = ""
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"text": self.text, "t": self.t, "source": self.source, "meta": self.meta}


@dataclass(slots=True)
class Event:
    """One line in the trace.

    `tick` is logical and is the only field logic may read. `wall` and
    `elapsed_s` are for you.
    """

    seq: int
    tick: int
    wall: str
    elapsed_s: float
    module: str
    kind: str
    data: dict = field(default_factory=dict)
    duration_s: float | None = None
    task_id: str | None = None
    cause: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)

    def line(self) -> str:
        """One readable line."""
        head = (
            f"t={self.tick:<3} {self.elapsed_s:7.3f}s "
            f"{self.module:<14} {self.kind:<14}"
        )
        detail = self.data.get("summary") or compact(self.data)
        if self.duration_s is not None:
            detail = f"{detail}  ({self.duration_s * 1000:.0f}ms)"
        return f"{head} {detail}"


def compact(data: dict, width: int = 90) -> str:
    """Squash a data dict onto one line. Used when there is no summary."""
    if not data:
        return ""
    parts = []
    for key, value in data.items():
        text = value if isinstance(value, str) else repr(value)
        text = " ".join(str(text).split())
        if len(text) > 40:
            text = text[:39] + "…"
        parts.append(f"{key}={text}")
    joined = " ".join(parts)
    return joined if len(joined) <= width else joined[: width - 1] + "…"
