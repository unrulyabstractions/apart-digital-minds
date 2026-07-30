# dmind

A small runtime for digital-mind experiments. A mind is one target model, its
**subject**, plus whatever you attach to it: something reading its context,
something rewriting its thoughts, something speaking beside it. Everything runs
on a clock you control. No cognitive architecture is built in.

The package has no required dependencies. Provider SDKs are imported only when
you ask for that provider.

## Layout

```
src/api/       the contracts
src/dminds/    the implementations
examples/      four minds assembled from the parts
ui/            talk to a mind in a browser and watch it think
tests/         106 tests, no dependencies
out/           everything a run produces (gitignored)
```

`src/api` declares what each part must do. `src/dminds` provides one of each.
Nothing in `src/api` imports from `src/dminds`, so you can replace any part
without disturbing the vocabulary the others speak.

```
src/api/
  mind.py          Mind, the thing you build and drive
  constants.py     WORLD, WILDCARD
  errors.py        RunawayMind, UndeclaredChannel
  types/           messages.py  payloads.py  messages_flow.py  records.py
  modules/         module.py  context.py  agent.py  roles.py
  models/          llm.py  factory.py
  memory/          stores.py
  observability/   sinks.py  tracing.py  kinds.py
```

`src/api` is the **external** surface: only what you build against. Machinery
you never call, the scheduler and the narrow `Host` view a module gets of its
mind, lives in `src/dminds` beside the code that uses it.

| Contract | `src/api` | Shipped implementation |
| --- | --- | --- |
| The thing you build and drive | `Mind` | `Mind` |
| Takes a turn | `Module` | `BaseModule` |
| Takes a turn, with a model | `Agent` | `Subject`, `Ego`, yours |
| Given to a turn | `Ctx` | `Ctx` |
| Answers a conversation | `LLM` | `BaseLLM`, the providers, `Cassette` |
| Builds a model from a spec | `ModelFactory` | `get_llm`, `taped(...)` |
| A conversation | `MessageStore` | `Transcript` |
| Working state | `KeyValueStore` | `Scratchpad` |
| Episodic memory | `EpisodicStore` | `Journal` |
| Fans events out | `Tracer` | `RunTracer` |
| One module's log | `Logger` | `ModuleLog` |
| Where events go | `Sink` | `JsonlSink`, `PerModuleSink`, `ConsoleSink`, `MemorySink` |

Three names exist in both layers, because the implementation kept the obvious
word: `Mind`, `Agent`, and `Ctx`. Importing from `src` gives the
implementation, which is almost always what you want.

```python
from src import Agent            # the class
from src.api import Agent        # the interface it satisfies
```

`src/api/types/` holds the data that crosses every boundary: `Message`,
`Link`, `ChatMessage`, `Completion`, `GenOptions`, `Event`, `Episode`, and the
`Text` / `Context` / `Vector` payloads. Read that package first.

```python
from src.api import Module, Agent, LLM, Sink   # what to implement
from src.dminds import Mind, BaseModule, Agent  # what to use
from src import Mind, Agent, get_llm           # both, flat
```

### Roles

`src/api/modules/roles.py` names the seams the examples are built on. The
runtime never checks for them. They exist so a mind you assemble says what each
part is for, and so two implementations of one role are swappable.

| Role | Contract | Played by |
| --- | --- | --- |
| `Editor` | `revise(payload)` | `examples/02` interceptor |
| `Workspace` | `record()`, `entries()`, `render()` | `examples/03` blackboard |
| `InnerVoice` | `utter(situation)` | `examples/03` voice |

Every class in `examples/` declares the roles it plays, so each example reads
as an implementation of a stated contract rather than an ad-hoc class.

## Install

```bash
pip install -e .          # the runtime
pip install -e '.[all]'   # plus every provider SDK
```

## Quick start

```python
import asyncio
from src import Mind, texts

async def main():
    mind = Mind("demo", "echo:", system="Be terse.")
    mind.prompt("What is a digital mind?")
    await mind.process()
    print(texts(mind.get_replies()))

asyncio.run(main())
```

No wiring. The subject is the front door in both directions: `prompt` reaches it
because it is the entry, and its `reply` reaches you because the mind connected
it.

Driving a mind is three steps, kept apart on purpose.

| Call | What it does |
| --- | --- |
| `mind.prompt(text)` | Put something in. Delivers and returns; runs nothing. |
| `await mind.process()` | Tick until no module anywhere has work. |
| `await mind.process_one()` | Run exactly one tick, for stepping through a run. |
| `mind.get_replies()` | Read what came out. Reading drains. |

## The subject, the stages, and the ego

A mind is a pipeline with a fixed shape.

```
prompt -> subject -> [stages] -> ego -> world
```

| Part | Reads | Writes |
| --- | --- | --- |
| `subject` | `prompt` | `subject_context`, `reply`, `thought` |
| stage | `subject_context` | `ego_input` |
| `ego` | `ego_input` | `reply` |

**subject** is the target model, the thing an experiment is about. It thinks and
publishes what it thought. **stages** sit in between; each takes a context and
passes a context along, which is where interception lives. **ego** speaks from
whatever context reaches it, including an edited one, and cannot tell.

The ego is optional. Without one the subject's own reply goes straight to you, so
the simplest mind is a subject and nothing else.

Every channel is one-directional and means one thing. No module consumes what
it produces, so **the path cannot loop** and no stage has to know when to
stop.

```python
mind = Mind("study", "openai:gpt-5", ego="ollama:qwen3:8b")
mind.intercept(interceptor)     # prompt -> subject -> interceptor -> ego -> world
```

`mind.intercept` puts stages between the subject and whoever speaks. It is
shorthand for `register` calls and nothing else. That one line is exactly:

```python
mind.subject.register(interceptor, "subject_context")
interceptor.register(mind.ego, "ego_input")
mind.ego.register(mind.world, "reply")
```

The two window channels are named for the ends of the path, so a single stage
between them needs no renaming at all: the wiring reads the way it runs. It is
also what stops a module consuming what it produces, so the path cannot loop.

Chain two stages and the middle link is renamed, because the second still
reads `subject_context` when what reached it came from an editor. That is the
price of names this direct, and `mind.describe()` shows every rename.

Write those yourself when the shape is not a line. `Mind(..., autowire=False)`
lays out nothing, and `mind.describe()` marks the laid-out links `[auto]` so
nothing is hidden.

The mind offers this because subject, stages, and ego are its own anatomy. It
has no routing table. Everything else registers itself, and intercepting again
leaves those hand-made links alone.

```python
mind.subject.register(blackboard, "*")   # a monitor, wired by hand, survives
```

Put something else at either end:

```python
Mind("halves", "ollama:qwen3:8b", subject=MySubject)   # a subclass, or a factory
Mind("halves", "openai:gpt-5", ego=MyEgo("ego", get_llm("echo:")))
```

`echo:` is a fake model. It needs no key and always answers the same way, so
you can build wiring before spending a token.

## The five ideas

| Idea | What it is |
| --- | --- |
| `Module` | A queue, declared output channels, and a turn. |
| `Message` | One thing sent on a channel, from one module to another. |
| `Subject` | The target model at the centre of a mind. |
| `Mind` | Where modules live, and what you drive. |
| `LLM` | One chat interface. Providers are chosen by a string. |

Everything else is built from these.

## The scheduling rule

This is the one thing to understand. A tick has two phases.

1. **Act.** Every module with something to do takes a turn: `on_input` for each
   message that arrived, then one `on_process`. Modules take their turns
   concurrently, so two agents think at the same moment.
2. **Deliver.** Everything emitted during phase 1 lands in its target queue,
   all at once.

Nothing emitted at tick `t` is visible before tick `t+1`. A module can never
observe how far another module got inside the same tick. Concurrency therefore
cannot change the outcome. Each module writes to a private outbox, and those
outboxes are concatenated in registration order, so delivery order is fixed too.

An external input runs to **quiescence**. `mind.prompt(...)` injects one message
and ticks until every queue is empty, so when it returns the mind has finished
thinking.

This gives you strict lock-step for free:

```
t=0   target answers and exports its context.   interceptor idle.
t=1   target idle.                              interceptor rewrites, emits back.
t=2   target adopts the rewrite and reruns.     interceptor idle.
```

## Writing a module

A turn is two steps, run once per tick: `on_input` for every message that
arrived, in order, then one `on_process`. Splitting them means a module absorbs
everything that reached it before it acts, so it never decides on half the
picture.

```python
from src import Agent, user

class Critic(Agent):
    INPUTS = {"subject_context": "the subject's window"}
    OUTPUTS = {"ego_input": "what the ego gets fed, annotated"}

    async def on_process(self, ctx):
        for message in self.take_inputs():
            completion = await self.think(
                messages=[*self.transcript.messages,
                          user(f"Critique this: {message.payload.messages[-1].content}")],
                tag="critique",
            )
            revision = message.payload.copy()
            revision.messages.append(completion.as_message(stage="critique"))
            ctx.emit("ego_input", revision)
```

The default `on_input` buffers into `self.inputs`; `take_inputs()` drains that
buffer. There is no other dispatch. Override `on_input` only when a module
should react to each message as it is absorbed, the way a workspace records.

Return True from `wants_process` to take a turn with an empty queue, which is
how a module acts unprompted.

Modules never call each other. They emit on a channel, and the scheduler
delivers to whoever registered, at the start of the next tick.

Payloads are anything. `Text`, `Context`, and `Vector` are conveniences, and you
are free to ignore them. Send raw activations if that is your experiment.

## Channels and wiring

A module declares what it can emit. Emitting on anything else is an error, and
so is registering against a channel that does not exist, so a typo fails at
wiring time rather than silently dropping messages.

```python
class Critic(Agent):
    INPUTS  = {"subject_context": "the subject's window"}
    OUTPUTS = {"ego_input": "what the ego gets fed, annotated"}
```

Wiring lives on the modules. The mind has no routing table and no opinion about
who talks to whom.

```python
mind.subject.register(critic, "subject_context")
critic.register(monitor, "ego_input", as_channel="overheard")
mind.subject.register(blackboard, "*")   # a workspace hears everything
```

Renaming matters: the critic emits `"ego_input"` and the monitor hears
`"overheard"`, so neither module knows the other's vocabulary. `"*"` forwards
every channel under its real name, which is how a monitor attaches.

**Registering is the only verb.** It wires the channel *and* brings the module
into the mind, so there is no separate assembly step. Whichever of the two
modules is already in a mind pulls the other one in.

```python
mind = Mind("demo")
assistant = Agent("assistant", mind.model("openai:gpt-5"))
assistant.register(mind.world, "reply")     # wires it and adds it
```

`mind.add(...)` still exists, but only for a module wired to nothing, such as
one that runs on `wants_process` alone.

## Swapping models

One string is the only change.

```python
get_llm("echo:")                             # no keys, deterministic
get_llm("openai:gpt-5")
get_llm("anthropic:claude-opus-5")
get_llm("gemini:gemini-2.5-flash")
get_llm("ollama:qwen3:8b")                   # local Qwen
get_llm("ollama:gemma3:4b")                  # local Gemma
get_llm("hf:Qwen/Qwen3-4B-Instruct-2507")    # local, weights in this process
get_llm("vllm:Qwen/Qwen3-8B")                # any OpenAI-compatible server
```

Short aliases exist: `qwen`, `gemma`, `claude`, `gpt`, `gemini`, `echo`.

The `hf` backend keeps the torch module on the object, so interpretability work
has something to hook.

```python
llm = get_llm("hf:Qwen/Qwen3-4B-Instruct-2507")
llm.load()
llm.model_obj    # the torch module
llm.tokenizer
```

Add your own backend without touching the package:

```python
from src import register_provider
register_provider("mine", lambda model, spec, **kw: MyLLM(model, spec, **kw))
```

## Memory

There are three stores, and they share no base class. A transcript is a
sequence, a scratchpad is a mapping, and a journal is a searchable log.

```python
agent.transcript.replace_all(new_messages)   # what an interceptor does
agent.transcript.window(10)                  # last 10, system kept in front
agent.transcript.tagged("stage", "draft")    # find messages by meta

agent.scratch["draft"] = text

journal = Journal(path=paths.memory("octopuses"))
journal.remember("octopuses are curious")
journal.recall("octopuses", k=3)
```

Recall scores word overlap and breaks ties by recency. It uses no embeddings, so
there is nothing to install. Pass `scorer=` to swap in a real retriever.

## Logs

Every module and every model call is instrumented. There is no flag to turn it
on. A run writes:

```
out/
  runs/<mind>/<run-id>/       one directory per run, grouped by mind
    meta.json                 what the run was: models, wiring, counts
    trace.jsonl               every event, in order
    modules/<name>.jsonl      the same events, split per module
  memory/<name>.jsonl         journals, which outlive the run that wrote them
  tapes/<name>.jsonl          cassettes, likewise
```

Three buckets, divided by how long the thing lives. A run directory is
disposable: delete it and you lose a recording, not an experiment. Memory and
tapes are what you keep, so they sit outside any single run. `src.dminds.paths`
holds the layout, and `out/` is gitignored.

Each event carries a logical tick, a UTC timestamp, the module name, the event
kind, and a duration where one applies. Model calls record the provider spec,
token counts, latency, and the full text.

```
t=0  0.000s world     task.emit     world --prompt--> subject What is a digital mind?
t=0  0.001s subject   handle.start  1 in: prompt
t=0  0.001s subject   llm.request   -> demo:subject [answer] 2 msgs: What is a digital mind?
t=0  0.002s subject   llm.response  <- demo:subject [answer] <think>They asked about ...
t=0  0.002s subject   task.emit     subject --context--> interceptor <3 messages: after answer>
t=1  0.003s intercep. handle.start  1 in: context
```

Log from your own code with `ctx.log.note(...)`, and time a block with
`ctx.log.span(...)`. Subagents get a nested name through `ctx.log.child(...)`.

## Composition

`Mind` builds nothing itself. The scheduler, the tracer, and the model factory
are all arguments, defaulting to the shipped implementations.

```python
Mind("fast",  "echo:", scheduler=lambda host: MyScheduler(host))
Mind("taped", "echo:", model_factory=taped(paths.tape("study")))
```

The scheduler is internal machinery: subclass `src.dminds.Scheduler` only when
you want a different notion of time.

Channels are declared, so a mistyped one fails the moment you register or emit:

```
UndeclaredChannel: Target 'target' has no output channel 'thougth'.
Declared channels: reply, thought. Add it to OUTPUTS, or register on "*"
to receive everything.
```

`mind.validate()` catches what is left, and `process` calls it before the
first tick.

## Replay

The model call is the only place non-determinism enters. Capture it there and
the whole run reproduces.

```python
llm = Cassette(get_llm("openai:gpt-5"), paths.tape("study"))   # one model
```

For a whole mind, attach the factory instead of wrapping models one at a time.
Every model the mind builds goes through it:

```python
mind = Mind("study", "openai:gpt-5", model_factory=taped(paths.tape("study")))
watcher = Agent("watcher", mind.model("ollama:qwen3:8b"))
mind.subject.register(watcher, "context")
```

Both cassettes share one `Tape`, so the replay cursor is global. Two agents on
the same model asking the same question get their own recorded answers back, in
order.

Modes are `auto`, `replay`, and `record`. Two replayed runs produce the same
event sequence, so you can diff two traces and see what a change did.

## Examples

```bash
python examples/01_hello_agent.py         # one subject, and where the logs go
python examples/02_think_interceptor.py   # subject -> interceptor -> ego
python examples/03_bicameral.py           # subject -> voice -> ego, watched
python examples/04_memory_and_replay.py   # persistence, tapes, trace diffing
```

Each role picks its model as: the environment variable, then a local Qwen3 if
Ollama has one pulled, then a scripted stand-in so the examples always run.

```bash
MODEL=anthropic:claude-opus-5 python examples/01_hello_agent.py
SUBJECT_MODEL=anthropic:claude-opus-5 INTERCEPTOR_MODEL=ollama:qwen3:8b \
    python examples/02_think_interceptor.py
```

## Watching a mind think

```bash
python ui/server.py                    # subject -> interceptor -> ego
python ui/server.py --mind bicameral   # a voice speaks into the window
```

A browser opens on the conversation, the live trace grouped by tick, and an
inspector showing every module's channels and every context window. Click a
model call to read the exact prompt and completion. Tick *step mode* before
sending and `Step 1 tick` walks the mind forward one tick at a time.

The whole integration is one line, because `Sink` already receives everything
a mind emits:

```python
mind.tracer.add_sink(WebSink(broadcast))
```

See `ui/README.md`.

## Tests

```bash
python tests/run_tests.py    # no dependencies
pytest                       # also works
```

## Replacing a part

Implement the contract and pass your version in. Nothing else changes.

```python
from src.api import Sink

class SlackSink(Sink):                      # a new log destination
    def write(self, event): ...
    def close(self): ...

mind.tracer.add_sink(SlackSink())
```

```python
from src import Completion, register_provider
from src.dminds import BaseLLM             # a new backend

class MyLLM(BaseLLM):
    def _chat(self, messages, opts):
        return Completion(text="...", model=self.model)

register_provider("mine", lambda model, spec, **kw: MyLLM(model, spec, **kw))
```

`BaseLLM` handles timing and thread offload, so a provider writes one
synchronous method. Implement `api.LLM` directly when you need control over the
async call itself, as `Cassette` does.

The same holds for the rest: satisfy `api.EpisodicStore` to put memory in a
real vector database, or subclass `src.dminds.Scheduler` for a different
notion of time.

An agent is a module with a model and a memory. `api.Agent` is the contract:

```python
class Agent(Module):
    llm: LLM
    transcript: MessageStore
    scratch: KeyValueStore
    journal: EpisodicStore | None

    def prompt_messages(self) -> list[ChatMessage]: ...
    async def think(self, messages=None, opts=None, tag="") -> Completion: ...
    async def say(self, text, tag="say") -> Completion: ...
```

Override `prompt_messages` to change context assembly. Pass `messages=` to
`think` to reason over somebody else's context, which is how a monitor works.

## What is deliberately absent

There is no agent loop, no tool-calling protocol, and no planner. You are
experimenting on those, so shipping one would prejudge the experiment. The two
hemispheres in `examples/03` are built from the same public parts you have.

Streaming is also absent. Mid-generation interception needs it, and the tick
model would have to grow a sub-tick notion of a partial message. To add it, put
a `stream` method on `LLM` and emit one task per chunk.
