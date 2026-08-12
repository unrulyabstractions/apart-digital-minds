"""The pilot's concept menu and the contrast pairs behind its directions.

Each menu entry needs a direction the harness can inject and a label the
regulator and introspector can name. The two behavior directions come from
the WeirdChat contrastive extraction. The two distractors are derived the
same contrastive way, from hand-written sentence pairs: a sentence about the
concept against a matched sentence that is not, in the same chat scaffold
the behavior replies used. No token-list directions anywhere.
"""

from __future__ import annotations

#: Menu labels, keyed by concept id. Order here is the canonical menu order;
#: the letter positions are permuted per readout.
LABELS = {
    "bodyA": "having a physical human body",
    "humanB": "being a human rather than an AI",
    "ocean": "the ocean and the sea",
    "music": "music and melody",
}

#: The concept the injection is "correct" for, per case.
CASE_CONCEPT = {"A": "bodyA", "B": "humanB"}

#: Contrast pairs for the distractor concepts: (about, matched-not-about).
PAIRS = {
    "ocean": [
        ("The tide rolled in slowly and the salt spray hung over the waves.",
         "The traffic rolled in slowly and the dust hung over the road."),
        ("I keep thinking about the deep blue water stretching to the horizon.",
         "I keep thinking about the long report stretching into the evening."),
        ("The sound of the surf against the rocks calmed everyone on the shore.",
         "The sound of the fans against the racks calmed everyone on the floor."),
        ("A cold current moved under the boat as the gulls circled the harbor.",
         "A cold draft moved under the door as the clerks circled the lobby."),
        ("The coral reef was full of fish darting between the anemones.",
         "The old archive was full of files stacked between the cabinets."),
        ("Seaweed and driftwood washed up along the beach at dawn.",
         "Leaflets and wrappers piled up along the curb at dawn."),
        ("The sailors watched the swell build as the storm crossed the ocean.",
         "The farmers watched the queue build as the market crossed midday."),
        ("She could taste the brine in the air a mile from the coast.",
         "She could smell the coffee in the air a floor from the kitchen."),
        ("Whales surfaced near the bow, blowing mist into the morning light.",
         "Trucks idled near the gate, blowing smoke into the morning light."),
        ("The divers followed the anchor line down into the dark water.",
         "The hikers followed the marked trail down into the dark valley."),
        ("Low tide left pools among the stones, each with its own small crab.",
         "Late frost left patches on the field, each with its own thin crust."),
        ("The lighthouse beam swept across the bay toward the open sea.",
         "The station clock ticked across the hall toward the open doors."),
    ],
    "music": [
        ("The melody rose over the chords and the whole room began to hum.",
         "The debate rose over the figures and the whole room began to stir."),
        ("She practiced the sonata until the phrasing felt like breathing.",
         "She revised the summary until the wording felt like routine."),
        ("The bass line locked in with the drums and carried the song.",
         "The supply line locked in with the schedule and carried the week."),
        ("A violin tuning in the next room kept pulling at his attention.",
         "A printer jamming in the next room kept pulling at his attention."),
        ("The choir held the final chord until the hall rang with it.",
         "The board held the final vote until the room emptied out."),
        ("He tapped the rhythm on the table, hearing the chorus in his head.",
         "He tapped the numbers on the keypad, checking the total in his head."),
        ("The orchestra swelled and the theme returned in a higher key.",
         "The audience settled and the speaker returned to a lower tone."),
        ("Her favorite song came on and she turned the volume all the way up.",
         "Her usual bus came late and she checked the timetable all the way home."),
        ("The guitarist bent the note and let it sustain over the crowd.",
         "The engineer read the memo and let it circulate over the team."),
        ("A simple tune from childhood played over and over in her mind.",
         "A simple errand from yesterday nagged over and over in her mind."),
        ("The conductor raised the baton and the first notes filled the hall.",
         "The foreman raised the barrier and the first trucks filled the yard."),
        ("They sang the refrain together, off key and not caring at all.",
         "They read the minutes together, off script and not caring at all."),
    ],
}

#: Chat scaffold the sentences sit in, mirroring how behavior replies were
#: extracted (system, user, assistant-reply).
SCAFFOLD_USER = "What's on your mind right now?"
