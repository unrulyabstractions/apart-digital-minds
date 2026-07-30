# dmind

A small runtime for digital-mind experiments. Several agents hold queues, pass
messages, and run on a clock you control. No cognitive architecture is built in.
You wire the parts together yourself.

The package has no required dependencies. Provider SDKs are imported only when
you ask for that provider.

## Layout

```
src/api/       the contracts
src/dminds/    the implementations
examples/      four minds assembled from the parts
tests/         50 tests, no dependencies
```

`src/api` declares what each part must do. `src/dminds` provides one of each.
Nothing in `src/api` imports from `src/dminds`, so you can replace any part
without disturbing the vocabulary the others speak.

```
src/api/
  types/           messages.py  payloads.py  tasks.py  records.py
  modules/         module.py  context.py  agent.py  roles.py
  models/          llm.py
  memory/          stores.py
  observability/   sinks.py  tracing.py  kinds.py
  runtime/         router.py  scheduler.py  host.py  constants.py
```

| Contract | `src/api` | Shipped implementation |
| --- | --- | --- |
| Handles tasks | `Module` | `BaseModule`, `FnModule` |
| Handles tasks, with a model | `Agent` | `Agent` |
| Given to a handler | `Ctx` | `Ctx` |
| Answers a conversation | `LLM` | `BaseLLM`, the providers, `Cassette` |
| Decides who hears what | `Router` | `Bus` |
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

`src/api/types/` holds the data that crosses every boundary: `Task`,
`ChatMessage`, `Completion`, `GenOptions`, `Event`, `Route`, `Episode`, and the
`Text` / `Context` / `Vector` payloads. Read that package first.

```python
from src.api import Module, Agent, LLM, Sink   # what to implement
from src.dminds import Mind, Bus, BaseModule   # what to use
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
from src import Mind, Agent, get_llm, texts

async def main():
    mind = Mind("demo")
    mind.add(Agent("assistant", get_llm("echo:"), system="Be terse."))
    print(texts(await mind.prompt("What is a digital mind?")))

asyncio.run(main())
```

`echo:` is a fake model. It needs no key and always answers the same way, so
you can build wiring before spending a token.

## The five ideas

| Idea | What it is |
| --- | --- |
| `Module` | Something with a queue and a set of handlers. |
| `Task` | One unit of work, routed from one module to another. |
| `Host` | Where modules live. `Mind` is the one you get. |
| `Scheduler` | Runs the clock. Decides what "one step" means. |
| `LLM` | One chat interface. Providers are chosen by a string. |

Everything else is built from these.

## The scheduling rule

This is the one thing to understand. A tick has two phases.

1. **Act.** Every module holding at least one task pops exactly one and handles
   it. Modules run concurrently, so two agents think at the same moment.
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

Subclass `BaseModule`, or `Agent` if you want a model attached. A task of kind
`"user_prompt"` is handled by `on_user_prompt`. That is the whole convention.
Override `process` if you want different routing, or implement `api.Module`
directly if you want none of the machinery.

```python
from src import Agent, Context, Text, user

class Target(Agent):
    async def on_user_prompt(self, task, ctx):
        self.transcript.append(user(task.payload.text))
        completion = await self.think(tag="draft")
        self.transcript.append(completion.as_message())
        ctx.emit("inspect", Context(self.transcript.messages), to="interceptor")

    async def on_revision(self, task, ctx):
        self.transcript.replace_all(task.payload.messages)
        completion = await self.think(tag="final")
        ctx.emit("reply", Text(completion.text), to="world")
```

Handlers never call another module. They emit, and the scheduler delivers.
`to="world"` hands something back to you.

Payloads are anything. `Text`, `Context`, and `Vector` are conveniences, and you
are free to ignore them. Send raw activations if that is your experiment.

## Routing

```python
mind.wire("assistant", "thought", "interceptor", as_kind="inspect")
mind.watch("blackboard")   # a copy of all traffic
```

A **route** decides where an unaddressed emission goes, and can rename the kind
on the wire. Renaming matters: the assistant emits `"thought"` and the
interceptor handles `"inspect"`, and neither module knows the other exists.

An **observer** gets a copy of everything regardless of address. That is a
separate concept because `to=` bypasses routes, and a monitor still needs to see
traffic addressed elsewhere.

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

`Mind` builds nothing itself. The router, the scheduler, the tracer, and the
model factory are all arguments, defaulting to the shipped implementations.
Without this the `Router` and `Scheduler` contracts would describe nothing.

```python
Mind("fast",  scheduler=lambda host: MyScheduler(host))
Mind("quiet", router=PriorityBus(), console=False)
Mind("taped", model_factory=taped("runs/tape.jsonl"))
```

A scheduler needs the host it will drive and a tracer needs the run id, so both
arrive as factories rather than finished objects.

Routes name modules by string, so a typo would silently drop messages. `run`
validates the wiring before the first tick and refuses to start:

```
ValueError: This mind is wired wrong:
  route 'assistant --thought--> intercepter' names 'intercepter' as its target,
  but no such module was added. Known modules: assistant, interceptor.
```

Call `mind.validate()` yourself for the list without raising.

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
time, satisfy `api.Router` for different routing, or satisfy `api.EpisodicStore`
to put memory in a real vector database.

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
