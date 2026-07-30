"""One unit of work, and one entry in a routing table."""

from __future__ import annotations

from dataclasses import dataclass

from .payloads import Payload, preview


@dataclass(slots=True)
class Task:
    """One unit of work sitting in one module's queue.

    `kind` is the routing key. A task of kind `"user_prompt"` is handled by the
    receiving module's `on_user_prompt` method.

    `t_created` is the tick it was emitted on and `t_deliver` the tick it
    becomes visible. They differ by one, which is the whole scheduling rule.
    """

    id: str
    kind: str
    payload: Payload
    src: str
    dst: str
    t_created: int
    t_deliver: int
    cause: str | None = None

    def describe(self) -> str:
        """One line, for logs and the console sink."""
        return f"{self.src} -> {self.dst} [{self.kind}] {preview(self.payload)}"


@dataclass(frozen=True, slots=True)
class Route:
    """One entry in a routing table.

    `as_kind` is the kind the receiver sees, which lets an emitter and a
    receiver use different names for the same message.
    """

    src: str
    kind: str
    dst: str
    as_kind: str

    def describe(self) -> str:
        renamed = f" as {self.as_kind}" if self.as_kind != self.kind else ""
        return f"{self.src} --{self.kind}--> {self.dst}{renamed}"
