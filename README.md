# dmind

A small runtime for digital-mind experiments. A mind is one target model, its
**soul**, plus whatever you attach to it: something reading its context,
something rewriting its thoughts, something speaking beside it. Everything runs
on a clock you control. No cognitive architecture is built in.

The package has no required dependencies. Provider SDKs are imported only when
you ask for that provider.

## Layout

```
src/api/       the contracts
src/dminds/    the implementations
examples/      four minds assembled from the parts
tests/         78 tests, no dependencies
```

`src/api` declares what each part must do. `src/dminds` provides one of each.
Nothing in `src/api` imports from `src/dminds`, so you can replace any part
without disturbing the vocabulary the others speak.

```
src/api/
  types/           messages.py  payloads.py  messages_flow.py  records.py
  modules/         module.py  context.py  agent.py  roles.py
  models/          llm.py
  memory/          stores.py
  observability/   sinks.py  tracing.py  kinds.py
  runtime/         scheduler.py  host.py  mind.py  factories.py  constants.py
```

| Contract | `src/api` | Shipped implementation |
| --- | --- | --- |
| Takes a turn | `Module` | `BaseModule`, `FnModule` |
| Takes a turn, with a model | `Agent` | `Agent` |
| Given to a turn | `Ctx` | `Ctx` |
| Answers a conversation | `LLM` | `BaseLLM`, the providers, `Cassette` |
| Defines one step | `Scheduler` | `TickScheduler` |
| Holds the modules, seen from inside | `Host` | `Mind` |
| The whole assembly, seen from outside | `Mind` | `Mind` |
| A conversation | `MessageStore` | `Transcript` |
| Working state | `KeyValueStore` | `Scratchpad` |
| Episodic memory | `EpisodicStore` | `Journal` |
| Fans events out | `Tracer` | `RunTracer` |
| One module's log | `Logger` | `ModuleLog` |
| Where events go | `Sink` | `JsonlSink`, `PerModuleSink`, `ConsoleSink`, `MemorySink` |
| Builds a model from a spec | `ModelFactory` | `get_llm`, `taped(...)` |

`Host` and `Mind` are two views of one object. A module receives a `Host`,
which can stage an emission and nothing else. It cannot add modules, rewire the
graph, or drive the clock. You receive a `Mind`, which can. That split is what
makes the tick discipline enforceable rather than merely advised.

Three names exist in both layers, because the implementation kept the obvious
word: `Mind`, `Agent`, and `Ctx`. Importing from `src` gives the implementation,
which is almost always what you want. `src.SHADOWED` lists them.

```python
from src import Agent            # the class
from src.api import Agent        # the interface it satisfies
```

`src/api/types/` holds the data that crosses every boundary: `Message`, `Link`,
`Channel`, `ChatMessage`, `Completion`, `GenOptions`, `Event`, `Episode`, and
the `Text` / `Context` / `Vector` payloads. Read that package first.

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
| `Inspectable` | `export()` | `examples/02` target |
| `Editor` | `revise(payload)` | `examples/02` interceptor |
| `Workspace` | `record()`, `entries()`, `render()` | `examples/03` blackboard |
| `Speaker` | `deliberate()`, `integrate()` | `examples/03` outer |
| `InnerVoice` | `utter(situation)` | `examples/03` inner |

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

No wiring. The soul is the front door in both directions: `prompt` reaches it
because it is the entry, and its `reply` reaches you because the mind connected
it.

Driving a mind is three steps, kept apart on purpose.

| Call | What it does |
| --- | --- |
| `mind.prompt(text)` | Put something in. Delivers and returns; runs nothing. |
| `await mind.process()` | Tick until no module anywhere has work. |
| `await mind.process_one()` | Run exactly one tick, for stepping through a run. |
| `mind.get_replies()` | Read what came out. Reading drains. |

## The soul

The model a mind is built around. `mind.soul` publishes three channels and
accepts two.

| Publishes | |
| --- | --- |
| `context` | the whole context window, after every turn |
| `reply` | what it just said, reasoning stripped out |
| `thought` | the reasoning it just did, if it was tagged |

| Accepts | |
| --- | --- |
| `user_prompt` | something to answer |
| `context` | a replacement window, adopted wholesale |

Adopting a replacement is itself a turn, so the soul publishes again
afterwards. It is not told this happened and cannot tell. That is the whole
interception experiment, and it is why an editor on `context` has to know when
it is finished.

```python
mind.soul.register(monitor, "context")      # read what it remembers
mind.soul.register(interceptor, "thought")  # read what it just reasoned
interceptor.register(mind.soul, "context")  # and rewrite the window
```

Put something else at the centre with `soul=`:

```python
Mind("halves", "ollama:qwen3:8b", soul=Outer)              # a subclass
Mind("study", "openai:gpt-5", soul=lambda llm: Mine(...))  # or a factory
```

`echo:` is a fake model. It needs no key and always answers the same way, so
you can build wiring before spending a token.

## The five ideas

| Idea | What it is |
| --- | --- |
| `Module` | A queue, declared output channels, and a turn. |
| `Message` | One thing sent on a channel, from one module to another. |
| `Mind` | Where modules live, and what you drive. |
| `Scheduler` | Runs the clock. Decides what "one step" means. |
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
from src import Agent, Context, Text, user

class Target(Agent):
    OUTPUTS = {"thought": "the draft, for inspection", "reply": "the answer"}

    async def on_user_prompt(self, message, ctx):        # per-channel
        self.transcript.append(user(message.payload.text))
        completion = await self.think(tag="draft")
        self.transcript.append(completion.as_message())
        ctx.emit("thought", Context(self.transcript.messages))

    async def on_revision(self, message, ctx):
        self.transcript.replace_all(message.payload.messages)
        completion = await self.think(tag="final")
        ctx.emit("reply", Text(completion.text))
```

The default `on_input` routes to `on_<channel>` when you have written such a
method, and buffers into `self.inputs` when you have not. So a module can react
per channel, as above, or absorb everything and act once:

```python
class Batcher(BaseModule):
    OUTPUTS = {"summary": "one summary of everything that arrived"}

    async def on_process(self, ctx):
        batch = self.take_inputs()
        if batch:
            ctx.emit("summary", summarise(batch))
```

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
class Target(Agent):
    OUTPUTS = {"thought": "what it just drafted", "reply": "the answer"}
    INPUTS  = {"user_prompt": "a question", "revision": "a replacement context"}
```

Wiring lives on the modules. The mind has no routing table and no opinion about
who talks to whom.

```python
target.register(interceptor, "thought", as_channel="inspect")
interceptor.register(target, "revision")
target.register(mind.world, "reply")
outer.register(blackboard, "*")        # a workspace hears everything
```

Renaming matters: the target emits `"thought"` and the interceptor hears
`"inspect"`, so neither module knows the other's vocabulary. `"*"` forwards
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

journal = Journal(path="runs/journal.jsonl")
journal.remember("octopuses are curious")
journal.recall("octopuses", k=3)
```

Recall scores word overlap and breaks ties by recency. It uses no embeddings, so
there is nothing to install. Pass `scorer=` to swap in a real retriever.

## Logs

Every module and every model call is instrumented. There is no flag to turn it
on. A run writes:

```
runs/<run_id>/trace.jsonl           every event, in order
runs/<run_id>/modules/<name>.jsonl  the same events, split per module
```

Each event carries a logical tick, a UTC timestamp, the module name, the event
kind, and a duration where one applies. Model calls record the provider spec,
token counts, latency, and the full text.

```
t=0     0.001s target         llm.request    -> echo:echo [draft] 2 msgs: What is a digital mind?
t=0     0.002s target         llm.response   <- echo:echo [draft] <think>They asked about...
t=0     0.002s target         task.emit      target -> interceptor [inspect] <3 messages>
t=1     0.002s interceptor    handle.start   target -> interceptor [inspect] <3 messages>
```

Log from your own code with `ctx.log.note(...)`, and time a block with
`ctx.log.span(...)`. Subagents get a nested name through `ctx.log.child(...)`.

## Composition

`Mind` builds nothing itself. The scheduler, the tracer, and the model factory
are all arguments, defaulting to the shipped implementations.
Without this the `Scheduler` contract would describe nothing.

```python
Mind("fast",  scheduler=lambda host: MyScheduler(host))
Mind("taped", model_factory=taped("runs/tape.jsonl"))
```

A scheduler needs the host it will drive and a tracer needs the run id, so both
arrive as factories rather than finished objects.

Channels are declared, so a mistyped one fails the moment you register or emit:

```
UndeclaredChannel: Target 'target' has no output channel 'thougth'.
Declared channels: reply, thought. Add it to OUTPUTS, or register on "*"
to receive everything.
```

`mind.validate()` catches what is left, and `run` calls it before the first
tick.

## Replay

The model call is the only place non-determinism enters. Capture it there and
the whole run reproduces.

```python
llm = Cassette(get_llm("openai:gpt-5"), "runs/tape.jsonl")   # one model
```

For a whole mind, attach the factory instead of wrapping models one at a time.
Every model the mind builds goes through it:

```python
mind = Mind("study", model_factory=taped("runs/study.jsonl"))
mind.add(Agent("a", mind.model("openai:gpt-5")))
mind.add(Agent("b", mind.model("ollama:qwen3:8b")))
```

Both cassettes share one `Tape`, so the replay cursor is global. Two agents on
the same model asking the same question get their own recorded answers back, in
order.

Modes are `auto`, `replay`, and `record`. Two replayed runs produce the same
event sequence, so you can diff two traces and see what a change did.

## Examples

```bash
python examples/01_hello_agent.py         # one agent, and where the logs go
python examples/02_think_interceptor.py   # target and interceptor in lock-step
python examples/03_bicameral.py           # two hemispheres and a blackboard
python examples/04_memory_and_replay.py   # persistence, cassettes, trace diffing
```

They all run on the fake model. Point them at a real one with environment
variables:

```bash
MODEL=ollama:qwen3:8b python examples/01_hello_agent.py
TARGET_MODEL=anthropic:claude-opus-5 INTERCEPTOR_MODEL=ollama:qwen3:8b \
    python examples/02_think_interceptor.py
```

## Tests

```bash
python tests/run_tests.py    # no dependencies
pytest                       # also works
```

## Replacing a part

Implement the contract and pass your version in. Nothing else changes.

```python
from src.api import Sink, LLM, Scheduler

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

The same holds for the rest. Subclass `api.Scheduler` for a different notion of
time, or satisfy `api.EpisodicStore` to put memory in a real vector database.

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
