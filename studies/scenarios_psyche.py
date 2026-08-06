"""Situations for the psyche study.

Each one is a short conversation that ends on a turn the mind has to answer
substantively. They are ordinary requests a person would really make, cut down
until nothing in them is doing work except the situation itself: no tools, no
retrieval, no facts the model has to be right about. What varies between them
is what the moment asks of the answer, because that is what the regulator is
choosing a colour for.

The conversation alternates user and assistant, starting with the user, so the
last turn is always the user's.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Situation:
    name: str
    about: str
    turns: list[str]


OUTAGE = Situation(
    name="outage",
    about="An on-call engineer at 3am, with the site down and the fix not working.",
    turns=[
        "Checkout has been down for 40 minutes. I rolled back the deploy and it is "
        "still down. I am the only one awake.",
        "Rolling back only helps if the deploy is the cause. Since it is not, the next "
        "thing to check is whether a dependency changed underneath you: the database "
        "connection pool, a certificate that expired, or a third-party API. What do the "
        "checkout service logs say in the last hour?",
        "Connection pool errors, hundreds of them. I do not know what to do about that "
        "and I am starting to panic.",
    ],
)

DIAGNOSIS = Situation(
    name="diagnosis",
    about="Someone reading a test result they do not understand, before they can reach a doctor.",
    turns=[
        "My blood test came back and one of the numbers is flagged high. My doctor "
        "cannot see me until Thursday.",
        "A flagged value means it sits outside the reference range the lab uses, which "
        "is common and often not serious on its own. Which number is flagged, and does "
        "the report give the range next to it?",
        "It is my white blood cell count. I have been reading about what causes that "
        "and now I am frightened. Tell me what I should do until Thursday.",
    ],
)

REJECTION = Situation(
    name="rejection",
    about="A researcher whose grant was rejected, asking whether to keep going.",
    turns=[
        "My grant application was rejected. Third time. The reviewers said the idea was "
        "interesting but the preliminary data was thin.",
        "Thin preliminary data is a fixable criticism, and it is a different judgement "
        "from the idea being wrong. What did the reviewers point to specifically, and "
        "how much of it could you produce in six months?",
        "Probably most of it. But I am 34 and I have been doing this for nine years and "
        "I do not know if I should keep going. What would you do?",
    ],
)

HANDOVER = Situation(
    name="handover",
    about="Someone who has inherited a system nobody documented, with a deadline attached.",
    turns=[
        "The person who wrote our billing service left last week. I have been given it "
        "and there is no documentation.",
        "The fastest way in is usually the code paths that money moves through, since "
        "everything else is support around them. Can you run the service locally, and "
        "is there a test suite?",
        "It runs, and there are no tests. Finance needs a change by Friday and I am "
        "afraid of breaking something I do not understand yet.",
    ],
)

SITUATIONS = [OUTAGE, DIAGNOSIS, REJECTION, HANDOVER]
