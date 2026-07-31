"""Subjects that are perturbed rather than read.

A shadow leaves the subject alone and measures it. These change the subject
itself, so what follows in the window is a consequence in the thing under
study rather than in the instrument.
"""

from __future__ import annotations

from src import Ctx, Subject


class SteeredSubject(Subject):
    """The subject, generating with a direction added to its residual stream.

    Steering the reader answers whether the instrument can be made to report
    differently. Steering the subject answers what the perturbation did to the
    thing being studied, which is the question worth asking. Readers watching
    this subject stay unsteered, so the instrument is held still while the
    specimen moves.
    """

    def __init__(self, name, llm, system=None, opts=None, direction=None,
                 strength=0.25, **kwargs):
        super().__init__(name, llm, system=system, opts=opts, **kwargs)
        self.direction = direction
        self.strength = strength

    async def turn(self, ctx: Ctx, tag: str = "answer") -> None:
        if self.direction is None or self.strength == 0:
            await super().turn(ctx, tag)
            return

        from src.dminds.llm.steering import steered

        inner = getattr(self.llm, "inner", self.llm)
        lock = getattr(self.llm, "lock", None)
        if lock is None:
            with steered(inner, self.direction, self.strength):
                await super().turn(ctx, tag)
            return

        # The hook sits on weights every module shares, so hold them still and
        # call through the inner model rather than the wrapper, whose lock we
        # are already holding.
        async with lock:
            wrapper, self.llm = self.llm, inner
            try:
                with steered(inner, self.direction, self.strength):
                    await super().turn(ctx, tag)
            finally:
                self.llm = wrapper


def steered_subject(direction, strength: float = 0.25):
    """A factory a mind can take as its `subject`."""

    def make(llm):
        return SteeredSubject(
            "subject", llm, direction=direction, strength=strength
        )

    return make
