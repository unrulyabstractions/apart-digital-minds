"""dmind: a small, deterministic runtime for digital-mind experiments.

    src/api/      the public API, grouped by concern
    src/dminds/   the core implementation

Both of these work, so pick whichever reads better:

    from src import Mind, Agent, get_llm
    from src.api import Mind, Agent, get_llm

Five ideas, and nothing else:

    Module      something that holds a queue and handles tasks
    Task        one unit of work, routed between modules
    Mind        the assembly of modules, routes, a clock, and a trace
    Scheduler   ticks the clock; nothing emitted at t is seen before t+1
    LLM         one chat interface, many providers, chosen by a string

No cognitive architecture is baked in. `examples/` shows several assembled
from the parts.
"""

from .api import *  # noqa: F401,F403
from .api import __all__ as _api_all
from .dminds import __version__

__all__ = [*_api_all, "__version__"]
