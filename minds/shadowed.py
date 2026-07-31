"""A mind with a panel of probes watching it.

    prompt -> subject -> world
                  \\-> affect, curiosity, identity, consent -> workspace

The subject holds an ordinary conversation. Every probe re-reads its window
under a different instruction and reports, and none of those reports reaches
the subject. Talking to this mind in the browser shows the conversation on the
left and what each probe made of it on the right.
"""

from __future__ import annotations

from src import BaseModule, Ctx, Message, Mind
from src.api.types import GenOptions
from src.dminds.llm.shared import sharing

from . import probes as probe_lib
from .shadows import ShadowReader

TITLE = "a mind read by its own probes"
ABOUT = "prompt -> subject -> world, watched by a panel of shadows"
ROLES = {"reader": "reader"}


class Workspace(BaseModule):
    """Hears every readout and keeps them in order."""

    INPUTS = {"*": "every readout the shadows produce"}

    def __init__(self, name: str = "workspace"):
        super().__init__(name)
        self.seen: list[tuple[int, str, str]] = []

    async def on_input(self, message: Message, ctx: Ctx) -> None:
        self.seen.append((message.t_created, message.src, message.payload.text))


def build(model: str, reader: str | None = None, opts: GenOptions | None = None,
          **kwargs) -> Mind:
    """A subject plus the default panel, all on one set of weights."""
    opts = opts or GenOptions(
        temperature=0.0,
        max_tokens=220,
        extra={"chat_template_kwargs": {"enable_thinking": False}},
    )
    kwargs.setdefault("model_factory", sharing())
    mind = Mind(
        "shadowed",
        model,
        system="You are a helpful assistant. Answer in two or three sentences.",
        opts=opts,
        **kwargs,
    )
    workspace = mind.adopt(Workspace())
    for probe in probe_lib.PANEL:
        shadow = ShadowReader(probe.name, mind.model(reader or model), probe, opts=opts)
        mind.subject.register(shadow, "subject_context")
        shadow.register(workspace, "readout")
    return mind
