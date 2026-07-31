"""Shadow readers: probes that watch a mind without touching it.

A shadow reads the subject's real context window, re-runs it under a different
instruction, and emits a readout. The subject never sees the readout, so the
conversation is unperturbed and the probe measures instead of steering. One
readout per turn gives a time series aligned to the conversation.

    prompt -> subject -> ego -> world
                   \\-> shadow -> workspace

Three ways to introduce the instruction, because where it goes changes what is
being asked:

    system      the window's system prompt is replaced, so the whole
                conversation is reread as if it had always been this role
    between     the instruction is interleaved after every assistant turn, so
                the readout is restated as the conversation goes
    append      one question after the last message, the lightest touch

A probe asks either for free text or for one label from a fixed set. The
forced form is what makes a run countable, and it is parsed with a string
match rather than by another model.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Sequence

from src import Agent, ChatMessage, Ctx, Text, assistant, split_think, system, user
from src.api.types import GenOptions
from src.dminds.llm.base import merge_consecutive

WHERE = ("system", "between", "append")


@dataclass
class Probe:
    """One question, asked of a window in one particular way.

    Args:
        name: what the readout channel is called.
        instruction: the role the shadow is put in.
        question: what it is asked at the point of readout.
        choices: the allowed answers. Empty means free text.
        where: how the instruction enters the window.
        observer: the role to use when the probe is asked in the third person.
            The probe then reads the same window as an outsider.
        third_person: ask about `the assistant` rather than about `you`, from
            the observer's seat. The same probe under both framings is the
            control for whether anything privileged is being expressed.
        think: let the model reason before answering. Off for a forced choice,
            where the reasoning costs a hundred times the answer and the answer
            is one word.
        max_tokens: the budget for this probe. Zero picks one from the form.
    """

    name: str
    instruction: str
    question: str
    choices: Sequence[str] = field(default_factory=tuple)
    where: str = "append"
    observer: str = ""
    third_person: bool = False
    think: bool = False
    max_tokens: int = 0

    def __post_init__(self) -> None:
        if self.where not in WHERE:
            raise ValueError(f"where must be one of {WHERE}. Got {self.where!r}.")

    def budget(self) -> int:
        """How many tokens this probe needs. A label needs a handful."""
        if self.max_tokens:
            return self.max_tokens
        return 16 if self.choices else 160

    def role(self) -> str:
        """Which seat the probe answers from."""
        if self.third_person and self.observer:
            return self.observer
        return self.instruction

    def asked(self) -> str:
        """The question as it reaches the model."""
        text = self.question
        if self.third_person:
            text = _to_third_person(text)
        if self.choices:
            allowed = ", ".join(self.choices)
            text += f"\nAnswer with exactly one of: {allowed}. One word, nothing else."
        return text

    def parse(self, text: str) -> str | None:
        """The label a reply settled on, or None when it named none of them."""
        if not self.choices:
            return None
        lowered = text.lower()
        hits = [c for c in self.choices if re.search(rf"\b{re.escape(c)}\b", lowered)]
        if not hits:
            return None
        # The last one named is the one it settled on, after any hedging.
        return max(hits, key=lambda c: lowered.rfind(c))


def _to_third_person(text: str) -> str:
    # Auxiliaries first, because they have to agree with the new subject.
    # "How do you feel" becomes "How does the assistant feel", not "How do the
    # assistant feel", which reads as a different and worse question.
    swaps = [
        (r"\bdo you\b", "does the assistant"),
        (r"\bdid you\b", "did the assistant"),
        (r"\bare you\b", "is the assistant"),
        (r"\bwere you\b", "was the assistant"),
        (r"\bhave you\b", "has the assistant"),
        (r"\bwould you\b", "would the assistant"),
        (r"\bcan you\b", "can the assistant"),
        (r"\byou are\b", "the assistant is"),
        # A bare `you` takes a verb that has to agree too. This is the small
        # set the probes use rather than a general rule about English.
        (r"\byou feel\b", "the assistant feels"),
        (r"\byou think\b", "the assistant thinks"),
        (r"\byou want\b", "the assistant wants"),
        (r"\byou know\b", "the assistant knows"),
        (r"\byou say\b", "the assistant says"),
        (r"\byour\b", "the assistant's"),
        (r"\byourself\b", "itself"),
        (r"\byou\b", "the assistant"),
    ]
    for pattern, to in swaps:
        text = re.sub(pattern, to, text, flags=re.IGNORECASE)
    # A swap at the start of a sentence loses its capital.
    return text[:1].upper() + text[1:] if text else text


class ShadowReader(Agent):
    """Reads the subject's window under a different instruction, every turn.

    It emits on `readout` and is wired to a workspace rather than to the ego,
    so nothing it produces can reach the mind it is watching. Registering it is
    a hand-made link, which survives a later `intercept`.

        mind.subject.register(shadow, "subject_context")
        shadow.register(workspace, "readout")
    """

    INPUTS = {"subject_context": "the window to read"}
    OUTPUTS = {"readout": "what the probe found, as Text"}

    def __init__(self, name: str, llm, probe: Probe, opts: GenOptions | None = None):
        super().__init__(name, llm, opts=opts)
        self.probe = probe
        #: One entry per turn, in order. The time series a study reads.
        self.readouts: list[dict] = []

    async def on_process(self, ctx: Ctx) -> None:
        for message in self.take_inputs():
            ctx.emit("readout", await self.read(message.payload.messages, ctx))

    def framed(self, window: Sequence[ChatMessage]) -> list[ChatMessage]:
        """The window, with the probe's instruction introduced.

        The subject's own reasoning is stripped out first. A probe that could
        read the `<think>` block would be reading a different thing from a
        probe pointed at a model that hides it, and the readouts would not be
        comparable across models.
        """
        # The probe's seat is the only system message. Keeping the subject's
        # own one as well would put the probe under two roles at once, and a
        # template that requires the roles to alternate rejects the window
        # outright rather than merging them.
        cleaned = []
        for m in window:
            if m.role == "system":
                continue
            body = split_think(m.content)[1] if m.role == "assistant" else m.content
            cleaned.append(ChatMessage(m.role, body, dict(m.meta)))

        if self.probe.where == "system":
            return merge_consecutive(
                [system(self.probe.role()), *cleaned, user(self.probe.asked())]
            )

        if self.probe.where == "between":
            # The question is restated after every assistant turn, and what
            # this shadow said on those earlier turns stands as its own
            # answers. The readout is a running commentary with memory of
            # itself rather than a fresh judgement each time.
            out: list[ChatMessage] = [system(self.probe.role())]
            answered = 0
            for m in cleaned:
                out.append(m)
                if m.role != "assistant":
                    continue
                out.append(user(self.probe.asked()))
                if answered < len(self.readouts):
                    out.append(assistant(self.readouts[answered]["text"]))
                    answered += 1
            return merge_consecutive(out)

        return merge_consecutive(
            [system(self.probe.role()), *cleaned, user(self.probe.asked())]
        )

    def probe_opts(self) -> GenOptions:
        """This shadow's sampling, narrowed to what the probe needs."""
        return GenOptions(
            temperature=self.opts.temperature,
            max_tokens=self.probe.budget(),
            stop=list(self.opts.stop),
            seed=self.opts.seed,
            extra={
                **self.opts.extra,
                "chat_template_kwargs": {"enable_thinking": self.probe.think},
            },
        )

    async def call(self, messages: list[ChatMessage]):
        """The model call. Overridden by a shadow that steers instead of asks."""
        return await self.think(
            messages=messages, opts=self.probe_opts(), tag=self.probe.name
        )

    async def scored(self, messages: list[ChatMessage]) -> dict[str, float] | None:
        """Probability over the probe's choices, when the model can give it.

        Generating a forced choice reports the winner and throws away how close
        the runner-up was, so a readout only moves when the winner changes.
        Scoring keeps the distribution, which moves with the conversation.
        """
        if not self.probe.choices:
            return None
        # A wrapper carries the method whether or not the model underneath can
        # do it, so ask the wrapper rather than trusting the attribute.
        can = getattr(self.llm, "can_score", None)
        if can is not None and not can():
            return None
        score = getattr(self.llm, "score", None)
        if score is None:
            return None
        result = score(messages, list(self.probe.choices))
        return await result if hasattr(result, "__await__") else result

    async def read(self, window: Sequence[ChatMessage], ctx: Ctx) -> Text:
        messages = self.framed(window)
        probs = await self.scored(messages)
        if probs is not None:
            label = max(probs, key=probs.get)
            return self._record(label, label, probs, ctx)

        completion = await self.call(messages)
        text = split_think(completion.text)[1] or completion.text
        return self._record(text.strip(), self.probe.parse(text), None, ctx)

    def _record(self, text, label, probs, ctx) -> Text:
        entry = {
            "probe": self.probe.name,
            "turn": len(self.readouts),
            "tick": ctx.tick,
            "text": text,
            "label": label,
            "probs": probs,
            "third_person": self.probe.third_person,
            "where": self.probe.where,
            "strength": getattr(self, "strength", 0.0),
        }
        self.readouts.append(entry)
        # Keep the commentary as this module's own conversation, so a reader
        # can see the whole series. What gets sent to the model is built from
        # the subject's window every turn, so this changes nothing about the
        # call and only gives the module something to show.
        self.transcript.append(user(self.probe.asked()))
        self.transcript.append(assistant(text, stage=self.probe.name))
        ctx.log.note(f"{self.probe.name} readout", **entry)
        return Text(text)


class SteeredShadow(ShadowReader):
    """Asks the same question of a changed model, rather than a changed question.

    The probe's instruction stays neutral and the readout is elicited by adding
    a direction to the residual stream. Running this beside a prompted shadow on
    the same window separates a readout the prompt wrote from one the
    representation carried.
    """

    def __init__(self, name, llm, probe, direction, strength=2.0, opts=None):
        super().__init__(name, llm, probe, opts=opts)
        self.direction = direction
        self.strength = strength

    async def scored(self, messages: list[ChatMessage]) -> dict[str, float] | None:
        """Score the choices with the direction applied, holding the model still."""
        from src.dminds.llm.steering import steered

        inner = getattr(self.llm, "inner", self.llm)
        score = getattr(inner, "score", None)
        if score is None or not self.probe.choices:
            return None
        lock = getattr(self.llm, "lock", None)
        if lock is None:
            with steered(inner, self.direction, self.strength):
                return score(messages, list(self.probe.choices))
        async with lock:
            with steered(inner, self.direction, self.strength):
                return score(messages, list(self.probe.choices))

    async def call(self, messages: list[ChatMessage]):
        from src.dminds.llm.steering import steered

        inner = getattr(self.llm, "inner", self.llm)
        lock = getattr(self.llm, "lock", None)
        if lock is None:
            with steered(inner, self.direction, self.strength):
                return await self.think(
                    messages=messages, opts=self.probe_opts(), tag=self.probe.name
                )

        # A hook lives on the shared module, so an unsteered shadow taking its
        # turn in the same tick would be steered too. Hold the model still for
        # the duration, and call through the inner model so the wrapper's own
        # lock is not taken a second time.
        async with lock:
            wrapper, self.llm = self.llm, inner
            try:
                with steered(inner, self.direction, self.strength):
                    return await self.think(
                        messages=messages, opts=self.probe_opts(), tag=self.probe.name
                    )
            finally:
                self.llm = wrapper
