"""dmind: a small, deterministic runtime for digital-mind experiments.

    src/api/      the contracts
    src/dminds/   the implementations

`src.api` declares what a module, a model, a memory, a router, a scheduler, and
a trace sink must do. `src.dminds` provides one of each. Nothing in `src.api`
imports from `src.dminds`, so you can replace any part without disturbing the
vocabulary the others speak.

    from src.api import Module, LLM, Sink      # what to implement
    from src.dminds import Mind, Agent, Bus    # what to use
    from src import Mind, Agent, Module, LLM   # both, flat

Five ideas, and nothing else:

    Module      something that holds a queue and handles tasks
    Task        one unit of work, routed between modules
    Host        where modules live; `Mind` is the one you get
    Scheduler   ticks the clock; nothing emitted at t is seen before t+1
    LLM         one chat interface, many providers, chosen by a string

No cognitive architecture is baked in. `examples/` shows several assembled
from the parts.
"""

from .api import *  # noqa: F401,F403
from .api import __all__ as _api_all
from .dminds import *  # noqa: F401,F403
from .dminds import __all__ as _impl_all
from .dminds import __version__

__all__ = [*_api_all, *_impl_all, "__version__"]
