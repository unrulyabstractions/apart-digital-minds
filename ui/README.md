# ui

Talk to a mind in a browser and watch every tick as it happens.

```bash
python ui/server.py                       # opens on the chooser
python ui/server.py --mind bicameral      # skip it and start here
python ui/server.py --model hf:Qwen/Qwen3-0.6B
```

Then open http://127.0.0.1:8765. If that port is busy the server walks to the
next free one and says so.

## The first screen

Nothing is running when the server starts, so the page opens on a chooser with
the two things you can do: talk to a mind, or replay a run.

The left column is every file in `minds/`, with its title and the shape of its
pipeline. The right column is every run under `out/runs/`. Both lists come
from the server, so adding a mind file puts it on this screen with no other
change.

Picking a mind builds it and drops you into the live view. Picking a run opens
it in playback. The `minds` button in the header brings the chooser back, and
choosing again closes the running mind and starts a fresh one, so you can move
between architectures without restarting the server.

## What you see

**Conversation** on the left: what you said, what the mind said back. With an
ego, the reply comes from the ego rather than the subject.

**Context windows** fill the main pane, one column per module, in pipeline
order. This is the view that matters: the subject's real thought sits beside
the window the ego was handed.

A message is shown as its parts rather than as raw tags:

    thinking   what it reasoned, inside <think>
    says       what it actually said

Each column that carries the flowing window names the column it came from and
reports how it differs. The header reads `identical to subject` in green, or
`N changed since subject` in amber. Unchanged messages collapse to one line,
so what is left on screen is exactly what a stage did:

    changed by <stage>    the message arrived and was rewritten
    added by <stage>      the message was inserted on the way
    written here          this module wrote it itself

A stage keeps its own conversation rather than carrying the window, so it is
labelled `own conversation` and is never anyone's upstream. A column that has
not received anything yet is shown on its own, since two modules starting from
different system prompts have not edited anything.

The comparison is a plain diff of two message lists. It knows nothing about
`<think>` tags, about editing versus inserting, or about what any particular
stage does, so a new stage of your own is reported the same way with no UI
change.

One thing to expect: an editor rewrites the *thought*, not the text under it.
So a rewritten message can hold a thought saying "be brief" above the
subject's own long-winded answer, which looks like a contradiction and is not
one. The ego's reply, the next message down, is what actually follows the
rewrite.

Every message is shown in full, never truncated.

The `wiring` button swaps that pane for the graph: each module, the channels
it reads and writes, its model, and every link, with the ones `intercept` laid
out marked `[auto]`.

**Trace** sits underneath, secondary by design. Every event the mind emits,
grouped by tick, with full text rather than a clipped summary. Filters narrow
it to model calls, message flow, or per-turn detail, and `▾` collapses it
entirely when you want the windows full height.

**Step mode**: tick the box before sending and the prompt is delivered without
running. `Step 1 tick` then advances one tick at a time, so you can watch the
window change hop by hop. `Run out` finishes.

## How it hooks in

Nothing in the runtime changed. `Sink` already receives every event a mind
emits, so the whole integration is one line:

```python
mind.tracer.add_sink(WebSink(broadcast))
```

The sink forwards each event to connected browsers over server-sent events.
Standard library only, no dependencies.

## Playback

The picker in the header lists every run under `out/runs/`. Choose one and the
UI switches from live to playback: the trace, the conversation, and each
module's context window are rebuilt from `trace.jsonl` alone.

A recording opens fully played, so you see the finished run. Rewind and walk
it forward:

    ⏮      back to the start
    ▶      play, at slow / normal / fast / instant
    step   one event
    tick ⏭ to the end of the current tick
    slider scrub anywhere

Stepping a tick at a time is the useful one: you watch the subject think, then
the stage rewrite what it thought, then the ego speak from the rewrite, one
tick per click.

This works because a trace is a complete recording rather than a summary. Text
payloads are stored in full, and a `memory.write` carries the content that was
written, so the conversation and the windows replay exactly. Sending is
disabled while a recording is on screen; pick `● live` to come back.

`?play=<mind>|<run-id>&at=<n>` opens a recording stepped to a position.

## Where the data goes

Every run writes to `out/runs/<mind>/<run-id>/`: `meta.json` describing the mind
and its models, `trace.jsonl`, and `modules/<name>.jsonl`. `meta.json` is
rewritten after every prompt, so it stays current while the server is up.

`?still` on the URL loads the page from `/state` alone without opening the
event stream, which is how it gets screenshotted. `?tab=windows` opens on a
particular tab.
