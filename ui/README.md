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

`?still` on the URL loads the page without opening the event stream, which is
how it gets screenshotted.
