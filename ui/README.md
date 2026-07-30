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

**Live trace** in the middle: every event the mind emits, grouped by tick.
Click any model call to read the exact prompt and completion. The filters
narrow it to model calls, message flow, or per-turn detail.

**Inspector** on the right, two tabs:

- *wiring* — every module, the channels it reads and writes, its model, and
  every link. Links that `intercept` laid out are marked `[auto]`.
- *windows* — each agent's context window side by side. A message a stage
  rewrote is marked `edited`, so you can see the subject's real thought next
  to what the ego was handed.

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
