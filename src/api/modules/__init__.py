"""Interfaces for the things that live inside a mind.

    module.py    Module, Handler   -> BaseModule, FnModule
    context.py   Ctx               -> Ctx
    agent.py     Agent             -> Agent
    roles.py     Inspectable, Editor, Workspace, Speaker, InnerVoice
                                   -> the classes in examples/

`Module` is the floor: a queue and a way to handle one task. `Agent` adds a
model and a memory. The roles are optional vocabulary that say what a part is
for.
"""

from .agent import Agent
from .context import Ctx
from .module import Handler, Module
from .roles import Editor, InnerVoice, Inspectable, Speaker, Workspace

__all__ = [
    "Module",
    "Handler",
    "Ctx",
    "Agent",
    "Inspectable",
    "Editor",
    "Workspace",
    "Speaker",
    "InnerVoice",
]
