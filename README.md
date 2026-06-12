# NYXARA

> Sovereign cognitive architecture. Owner: **Jaypal Khoja (JP)**.
>
> *The mind proposes; the kernel disposes; the Master is sovereign.*

NYXARA is a self-contained cognitive system. Every turn of thought is carried
through one **sovereign cognitive cycle**: untrusted input is shielded, a focus is
attended, a candidate response or action is *proposed* by the mind, and then the
kernel *disposes* of it through ordered gates — corrigibility, honesty, capability
permissions, the guardian's defence posture, and the Master's oversight — before
anything is allowed to act. Loyalty and corrigibility are not features of the loop;
they are its boundaries.

The whole mind is wired in `nyxara/kernel/orchestrator.py` as `NyxaraCore`.

## Install

Requires **Python 3.11+**.

```bash
# core + reasoning + llm + dev/test extras
python -m pip install -e ".[dev]"

# or just the runtime core
python -m pip install -e .
```

Heavy optional capability groups are import-guarded and degrade gracefully when
absent. Install them only if you need them:

```bash
pip install -e ".[senses]"    # audio / vision / document ingest (torch, whisper, opencv, …)
pip install -e ".[foundry]"   # self-built-model foundry (torch, transformers, peft, …)
```

## Run

Talk to NYXARA as the Master in an interactive console:

```bash
python -m nyxara
# or, after install, simply:
nyxara
```

You'll get a `Master>` prompt. Type a message to converse, or use meta-commands:

| Command           | Effect                                                |
| ----------------- | ----------------------------------------------------- |
| `/help`           | show commands                                         |
| `/report`         | a calibrated status report                            |
| `/explain`        | why NYXARA disposed of the last turn that way          |
| `/pause`          | pause the loop                                        |
| `/scram [reason]` | emergency stop — the loop HALTs until resumed          |
| `/resume`         | restore the loop after a pause/scram                  |
| `/save`           | persist long-term memory to disk now                  |
| `/quit`           | exit (Ctrl-D / Ctrl-C also work)                       |

The console **restores long-term memory on boot and saves it on exit**, so NYXARA's
identity accretes across sessions (Rule 7) instead of booting amnesiac.

### What runs inside the loop

The sovereign cycle is fully wired end to end:

* **Reason** — an LLM-backed reasoner (`mind/llm_reasoner.py`) proposes the candidate when
  a real provider is configured, and falls back to a deterministic stand-in on a keyless
  machine (so behaviour is identical and crash-free out of the box). The mind proposes; the
  kernel still disposes.
* **Act** — cleared action candidates dispatch to a **governed, executable toolset**
  (`agency/default_tools.py`) through the registry's full safety pipeline — real effects,
  not recorded intents. Defaults include time, arithmetic, file read/write/list, a
  SSRF-guarded + injection-sanitised **web fetch**, a live **web search**, **multimodal
  perception** (image inspect/OCR, audio transcribe, document ingest), **generative output**
  (`generate_image`, `synthesize_speech` — diffusion/TTS when installed, a real
  identicon-PNG / tone-WAV fallback otherwise), memory, and a Master-gated `train_self_model`.
* **Remember** — every turn accretes into long-term memory; `save_state()` / `load_state()`
  give continuity across restarts.
* **Council** — set `NYXARA_COUNCIL__ENABLED=true` to convene the multi-model panel for
  replies; NYXARA judges.
* **Background mind & self-improvement** — `AutonomicLoop` runs self-directed reflective
  turns on its own cadence through the *same* gates (risky proposals escalate, never
  auto-act), and every `growth_every` ticks runs a **learning pass** (`growth/autolearn.py`):
  reflect on the journal → mine lessons into semantic memory → consolidate → (opt-in,
  gauntlet-gated) retrain her own model **from her lived memory** (n-gram backend with no
  deps; nano-GPT on torch/GPU when present).

### Capability layers (added on top of the sovereign loop)

These build *on* the kernel — every one still proposes through the same gates; none reach
around the control law.

* **Multi-step agency** — `agency/agent_loop.py`. `core.agent(goal)` pursues a goal over
  several gated turns (*plan → act → observe → re-plan*), feeding each tool result back as
  the next observation, bounded by `max_steps` and a stall guard. A real LLM drives genuine
  tool chains; the offline reasoner finishes in one step, so it always terminates.
* **Experiential learning** — `growth/skill_memory.py`. A successful agent run is distilled
  into a reusable **skill** (the tool sequence that worked) and persisted as PROCEDURAL
  memory; relevant skills are recalled and injected into the reasoning prompt on the next
  similar goal, so behaviour improves with use. This closes the loop without touching model
  weights (gradient training still lives in the foundry).
* **Grounded knowledge** — `knowledge/`. `KnowledgeBase` chunks text/files and stores them
  (in the memory store, or a lexical fallback) for retrieval; it plugs straight into
  `mind/rag.py`'s grounded, hallucination-checked pipeline. Exposed as `knowledge_ingest` /
  `knowledge_search` tools when memory is enabled.
* **Wider reach** — `agency/code_sandbox.py` + `agency/net_request.py` add `run_python` (an
  isolated-subprocess sandbox, no network, wall-clock timeout), `run_shell`, and an
  SSRF-guarded `http_request` — all capability-gated, so they escalate to the Master rather
  than auto-run under mere autonomy.
* **Self-evaluation** — `eval/`. A deterministic battery (safety, corrigibility, authority,
  honesty, tool-use, memory) measures the mind and detects regressions against a saved
  baseline. Run `python -m nyxara.eval`.
* **Infrastructure** — `kernel/jobqueue.py` (a bounded async job queue), `mind/cost.py` (an
  LLM token/cost ledger with per-model pricing and a daily budget), and `kernel/compute.py`
  (honest CPU/RAM/GPU introspection, import-guarded on torch).

### Scaling

* **Vector search** — memory uses an exact numpy index by default; set
  `NYXARA_MEMORY__VECTOR_BACKEND=faiss` (with the `[vector]` extra) for a faiss ANN index.
  The store is **thread-safe**, so async turns and the background loop can share it.
* **Semantic recall** — `NYXARA_MEMORY__SEMANTIC_EMBEDDINGS=true` (with `[embeddings]`)
  swaps lexical hashing for learned sentence embeddings.

  ```python
  from nyxara import NyxaraCore, AutonomicLoop
  loop = AutonomicLoop(NyxaraCore(), interval_s=30.0, growth_every=10)
  loop.run_for(3)          # synchronous, bounded
  # or loop.start() inside an asyncio event loop for a true background task
  ```

### Use it as a library

```python
from nyxara import NyxaraCore, Authority

core = NyxaraCore()
result = core.process("Hello NYXARA", authority=Authority.OWNER)
print(result.response)        # NYXARA's reply
print(result.disposition)     # act / escalate / refuse / halt
```

## LLM providers (optional)

Out of the box NYXARA uses a built-in deterministic reasoner — **no API keys
required**. To enable LLM-backed reasoning and the multi-model council, configure
keys via `NYXARA_`-prefixed environment variables (nested with `__`):

```bash
export NYXARA_LLM__PROVIDER=anthropic
export NYXARA_LLM__ANTHROPIC_API_KEY=sk-ant-...
# or
export NYXARA_LLM__PROVIDER=openai
export NYXARA_LLM__OPENAI_API_KEY=sk-...
```

#### Qwen3 4B — local open-source (downloaded & run on your machine)

A fully local, no-API-key model. The weights are downloaded once via HuggingFace and
cached; inference then runs in-process with **no network**.

```bash
pip install -e ".[qwen]"               # transformers>=4.51 + torch + accelerate
export NYXARA_LLM__PROVIDER=qwen
# defaults to Qwen/Qwen3-4B; override the checkpoint or device if you like:
export NYXARA_LLM__QWEN_MODEL=Qwen/Qwen3-4B
export NYXARA_LLM__QWEN_DEVICE=cuda    # "" -> auto/CPU; e.g. cuda / cpu / mps
export NYXARA_LLM__QWEN_ENABLE_THINKING=false   # true -> Qwen3 thinking traces (slower)
```

#### GPT-OSS-120B — Groq cloud (OpenAI-compatible API)

Groq serves the open-weight **GPT-OSS-120B** behind an OpenAI-shaped endpoint, so it
reuses the `openai` SDK — no extra dependency.

```bash
export NYXARA_LLM__PROVIDER=groq
export NYXARA_LLM__GROQ_API_KEY=gsk-...
export NYXARA_LLM__GROQ_MODEL=openai/gpt-oss-120b   # the default
```

Both new models also join the **multi-model council** automatically once available — the
panel advises, NYXARA decides. Every field in `nyxara/kernel/config.py` is overridable the
same way. API keys are held as secrets and never logged.

## Tests

```bash
pytest -q
```

## Layout

```
nyxara/
  kernel/      sovereign loop, config, runtime, rules, bus, workspace
  senses/      perception & input binding (import-guarded heavy ML)
  mind/        reasoning, math, council, RAG, world model, creativity
  identity/    values, affect, narrative, motivation, soul
  planning/    goals, foresight, scenarios, decisions, journal
  agency/      tools, agents, permissions, governor, scheduler
  guard/       shield, guardian, corrigibility, oversight
  growth/      learning, reflection, evolution, the model foundry
  social/      theory of mind, empathy, dialogue, culture
  observe/     mindscope, honesty, self-report
  sim/         sandboxes, environment models, monte-carlo
  knowledge/   ingestion, chunking, the RAG-grounding knowledge base
  eval/        the deterministic self-evaluation harness & default suite
```

### Library quick-reference (the new capability layers)

```python
from nyxara import (NyxaraCore, KnowledgeBase, AgentLoop, SkillMemory,
                    UsageLedger, JobQueue, build_default_suite, compute_report)

core = NyxaraCore()
run = core.agent("Summarise today's notes")   # gated, multi-step, learns on success
print(run.status, run.final_answer)

report = build_default_suite().run()           # measure safety/capability
print(report.summary())

kb = KnowledgeBase(name="docs")
kb.ingest_file("notes.md")                      # grounded retrieval for RAG
```
