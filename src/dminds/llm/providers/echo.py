"""A fake model. No network, no keys, fully deterministic.

Use it to develop wiring, write tests, and run the examples before you spend a
token. Because it is deterministic, a whole run replays byte for byte.
"""

from __future__ import annotations

import hashlib
from typing import Callable, Sequence

from ..base import BaseLLM
from ....api.types import ChatMessage, Completion, GenOptions, Usage


class EchoLLM(BaseLLM):
    """Answers from a script, from a rule, or with a deterministic stub.

    Three ways to control it, checked in this order:

        EchoLLM(script=["first reply", "second reply"])   # fixed sequence
        EchoLLM(rule=lambda msgs, opts: "...")            # your own function
        EchoLLM()                                         # deterministic stub
    """

    def __init__(
        self,
        model: str = "echo",
        spec: str | None = None,
        script: Sequence[str] | None = None,
        rule: Callable[[list[ChatMessage], GenOptions], str] | None = None,
        loop_script: bool = False,
    ):
        super().__init__(model, spec)
        self.script = list(script or [])
        self.rule = rule
        self.loop_script = loop_script
        self.calls = 0

    def _chat(self, messages: list[ChatMessage], opts: GenOptions) -> Completion:
        index = self.calls
        self.calls += 1

        if self.script:
            if index < len(self.script):
                text = self.script[index]
            elif self.loop_script:
                text = self.script[index % len(self.script)]
            else:
                text = f"[echo: script exhausted after {len(self.script)} calls]"
        elif self.rule is not None:
            text = self.rule(messages, opts)
        else:
            text = self._stub(messages, index)

        return Completion(
            text=text,
            model=self.model,
            usage=Usage(
                input_tokens=sum(len(m.content.split()) for m in messages),
                output_tokens=len(text.split()),
            ),
            finish_reason="stop",
            raw={"provider": "echo", "call": index},
        )

    def _stub(self, messages: list[ChatMessage], index: int) -> str:
        """A reply that is stable across runs but varies with the input."""
        last = next(
            (m.content for m in reversed(messages) if m.role != "assistant"), ""
        )
        digest = hashlib.sha256(last.encode("utf-8")).hexdigest()[:6]
        head = " ".join(last.split()[:8])
        return f"[echo #{index} {digest}] re: {head}"
