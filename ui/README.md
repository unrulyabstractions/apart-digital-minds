# ui

Talk to a mind in a browser and watch every tick as it happens.

```bash
python ui/server.py                       # subject -> interceptor -> ego
python ui/server.py --mind plain          # just a subject
python ui/server.py --mind bicameral      # a voice speaks into the window
python ui/server.py --model hf:Qwen/Qwen3-0.6B
```

Then open http://127.0.0.1:8765.

## What you see

**Conversation** on the left: what you said, what the mind said back. With an
ego, the reply comes from the ego rather than the subject.

**Context windows** fill the main pane, one column per module, side by side.
This is the view that matters: the subject's real thought sits beside the
window the ego was handed.

A message is shown as its parts rather than as raw tags:

    THINKING   what it reasoned, inside <think>
    SAYS       what it actually said

When a stage has rewritten one, the parts say exactly what changed:

    THINKING · REPLACED   Skip the preamble and the caveats.
    WAS                   They asked 'Hello'. I will open with a warm preamble…
    SAYS · UNTOUCHED      What a wonderful question! On the subject of Hello…

That last label matters. An editor rewrites the *thought*, not the text under
it, so the visible answer is still whatever the subject wrote and can flatly
contradict the thought above it. The ego's own reply, the next message down,
is what actually follows the rewrite.

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

Every run writes to `out/runs/ui/<run-id>/`: `meta.json` describing the mind
and its models, `trace.jsonl`, and `modules/<name>.jsonl`. `meta.json` is
rewritten after every prompt, so it stays current while the server is up.

`?still` on the URL loads the page from `/state` alone without opening the
event stream, which is how it gets screenshotted. `?tab=windows` opens on a
particular tab.
