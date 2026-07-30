"""dmind: a small, deterministic runtime for digital-mind experiments.

    src/api/      the contracts
    src/dminds/   the implementations

`src.api` declares what a module, an agent, a model, a memory, a router, a
scheduler, a trace sink, and a mind must do. `src.dminds` provides one of each.
Nothing in `src.api` imports from `src.dminds`, so you can replace any part
without disturbing the vocabulary the others speak.

    from src.api import Module, Agent, LLM, Sink   # what to implement
    from src.dminds import Mind, Bus, BaseModule   # what to use
    from src import Mind, Agent, get_llm           # both, flat

A few names exist in both layers, because the implementation kept the obvious
word. Importing them from `src` gives you the implementation, which is almost
always what you want. Reach for the contract explicitly when you mean the
contract. Those names are `Mind`, `Agent`, and `Ctx`.

    from src import Agent            # src.dminds.agents.Agent, a real class
    from src.api import Agent        # the interface it satisfies

Five ideas, and nothing else:

    Module      a queue, declared output channels, and a turn
    Message     one thing sent on a channel, from one module to another
    Subject     the target model at the centre of a mind
    Mind        where modules live, and what you drive
    LLM         one chat interface, many providers, chosen by a string

Modules wire themselves. `producer.register(consumer, "channel")` connects them
and brings the consumer into the mind, so there is no separate assembly step.

No cognitive architecture is baked in. `examples/` shows several assembled
from the parts.
"""

from . import api, dminds
from .api import *  # noqa: F401,F403
from .api import __all__ as _api_all

# Implementations are imported second on purpose. Where a name exists in both
# layers, the concrete class wins here.
from .dminds import *  # noqa: F401,F403
from .dminds import __all__ as _impl_all
from .dminds import __version__


__all__ = [*dict.fromkeys([*_api_all, *_impl_all]), "api", "dminds", "__version__"]
