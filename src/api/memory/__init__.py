"""Interfaces for the three kinds of memory.

    stores.py   MessageStore   -> Transcript
                KeyValueStore  -> Scratchpad
                EpisodicStore  -> Journal
                Recallable     -> anything with remember and recall
"""

from .stores import EpisodicStore, KeyValueStore, MessageStore, Recallable

__all__ = ["MessageStore", "KeyValueStore", "EpisodicStore", "Recallable"]
