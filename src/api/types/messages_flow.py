"""What flows between modules, and the registrations that carry it."""

from __future__ import annotations

from dataclasses import dataclass

from .payloads import Payload, preview


@dataclass(slots=True)
class Message:
    """One thing sent from one module to another, on a named channel.

    `t_created` is the tick it was emitted on and `t_deliver` the tick it
    becomes visible. They differ by one, which is the whole scheduling rule.
    """

    id: str
    channel: str
    payload: Payload
    src: str
    dst: str
    t_created: int
    t_deliver: int
    cause: str | None = None

    def describe(self) -> str:
        """One line, for logs and the console sink."""
        return (
            f"{self.src} --{self.channel}--> {self.dst} {preview(self.payload)}"
        )


@dataclass(frozen=True, slots=True)
class Link:
    """One registration: `src` emits on `channel`, `dst` hears `as_channel`.

    Renaming on the wire is why `as_channel` exists. A target can emit
    `"thought"` while an interceptor hears `"inspect"`, and neither module
    needs to know the other's vocabulary.
    """

    src: str
    channel: str
    dst: str
    as_channel: str

    def describe(self) -> str:
        if self.channel == "*":
            # A wildcard listener hears each channel under its own name, so
            # printing `as *` would misdescribe what arrives.
            return f"{self.src} --*--> {self.dst} (every channel, named as sent)"
        renamed = f" as {self.as_channel}" if self.as_channel != self.channel else ""
        return f"{self.src} --{self.channel}--> {self.dst}{renamed}"
