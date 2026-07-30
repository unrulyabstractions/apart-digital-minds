"""Memory stores.

Three of them, sharing no base class. A transcript is a sequence, a scratchpad
is a mapping, and a journal is a searchable log. Forcing one interface onto all
three would hide what each is for.

Hook points in this module:

    Transcript.replace_all  swap an agent's whole history, which is what a
                            context editor emits
    Journal(scorer=...)     replace keyword recall with a real retriever
    Recallable              the only contract a custom memory must satisfy
"""

from ..dminds.memory import Episode, Journal, Recallable, Scratchpad, Transcript

__all__ = [
    "Transcript",
    "Scratchpad",
    "Journal",
    "Episode",
    "Recallable",
]
