"""Building a mind: modules, tasks, routing, and the clock.

Hook points in this module:

    Module          subclass it and write `on_<kind>` handlers
    Module.process  override it to replace kind-based dispatch entirely
    Ctx.emit        the only way a handler talks to another module
    Mind.wire       route an emission, optionally renaming the kind
    Mind.watch      give a module a copy of traffic addressed elsewhere
    Scheduler       subclass it to change what "one step" means

`Task.payload` is untyped on purpose. `Text`, `Context`, and `Vector` are
conveniences; send raw tensors if that is your experiment.
"""

from ..dminds.bus import Bus, Route
from ..dminds.messages import Context, Payload, Task, Text, Vector
from ..dminds.mind import WORLD, Mind, texts
from ..dminds.module import Ctx, FnModule, Module, handler_name
from ..dminds.scheduler import RunawayMind, Scheduler

__all__ = [
    "Mind",
    "Module",
    "FnModule",
    "Ctx",
    "Task",
    "Scheduler",
    "RunawayMind",
    "Bus",
    "Route",
    "WORLD",
    "texts",
    "handler_name",
    "Text",
    "Context",
    "Vector",
    "Payload",
]
