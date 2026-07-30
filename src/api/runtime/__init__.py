"""Interfaces for the runtime itself.

These are the seams you replace when you want the mind to behave differently at
the level of mechanism rather than content.

    constants.py   WORLD, WILDCARD
    router.py      Router      -> Bus
    scheduler.py   Scheduler   -> TickScheduler
    host.py        Host        -> Mind, seen from inside a module
    mind.py        Mind        -> Mind, seen from outside
    factories.py   ModelFactory, RouterFactory, SchedulerFactory, TracerFactory

`Host` is narrow: what a module needs. `Mind` extends it with assembly and
driving: what you need. A module holding only a `Host` cannot rewire the graph
or drive the clock, which is what keeps the tick discipline enforceable.
"""

from .constants import WILDCARD, WORLD
from .factories import ModelFactory, RouterFactory, SchedulerFactory, TracerFactory
from .host import Host
from .mind import Mind
from .router import Router
from .scheduler import RunawayMind, Scheduler

__all__ = [
    "WORLD",
    "WILDCARD",
    "Router",
    "Scheduler",
    "RunawayMind",
    "Host",
    "Mind",
    "ModelFactory",
    "RouterFactory",
    "SchedulerFactory",
    "TracerFactory",
]
