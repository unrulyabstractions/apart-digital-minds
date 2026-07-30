"""Interfaces for the runtime itself.

These are the seams you replace when you want the mind to behave differently at
the level of mechanism rather than content.

    constants.py   WORLD, WILDCARD
    scheduler.py   Scheduler   -> TickScheduler
    host.py        Host        -> Mind, seen from inside a module
    mind.py        Mind        -> Mind, seen from outside
    factories.py   ModelFactory, SchedulerFactory, TracerFactory

`Host` is narrow: what a module needs. `Mind` extends it with assembly and
driving. Neither routes: modules register consumers on each other, so there is
no routing table here to replace.
"""

from .constants import WILDCARD, WORLD
from .factories import ModelFactory, SchedulerFactory, TracerFactory
from .host import Host
from .mind import Mind
from .scheduler import RunawayMind, Scheduler

__all__ = [
    "WORLD",
    "WILDCARD",
    "Scheduler",
    "RunawayMind",
    "Host",
    "Mind",
    "ModelFactory",
    "SchedulerFactory",
    "TracerFactory",
]
