# NYXARA

[![CI](https://github.com/nyxarajp-del/NYXARAv01/actions/workflows/ci.yml/badge.svg)](https://github.com/nyxarajp-del/NYXARAv01/actions/workflows/ci.yml)

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
| `/research <topic>` | run one autonomous research pass on a topic         |
| `/investigate <q>` | reason like a scientist: hypothesis → experiment → conclusion |
| `/discover [n]`   | autonomous discovery: `n` self-driven observe→…→update-model cycles |
| `/strategize <p>` | strategic analysis: direct answer → reality check → weaknesses → solution |
| `/save`           | persist long-term memory to disk now                  |
| `/quit`           | exit (Ctrl-D / Ctrl-C also work)                       |

The console **restores long-term memory on boot and saves it on exit**, so NYXARA's
identity accretes across sessions (Rule 7) instead of booting amnesiac.

### What runs inside the loop

The sovereign cycle is fully wired end to end:

* **Reason** — an LLM-backed reasoner (`mind/llm_reasoner.py`) proposes the candidate when
  a real provider is configured, and falls back to a deterministic stand-in on a keyless
  machine (so behaviour is identical and crash-free out of the box). The mind proposes; the
  kernel still disposes. With a real provider it **deliberates** (`mind/deliberate.py`)
  instead of answering in one shot — *think (a private scratchpad) → decide → self-critique
  & revise* — which measurably lifts answer quality. Tune the depth with
  `NYXARA_LLM__REASONING_PASSES` (1 = single shot, 2 = think→decide *(default)*, 3 = add a
  self-critique pass) and `NYXARA_LLM__REASONING_SAMPLES` (>1 votes by self-consistency).
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
  gauntlet-gated) retrain her own model **from her lived memory**. Three backends, one
  contract, chosen by what the machine can run: an n-gram model with **no deps**; a
  from-scratch **nano-GPT** on torch; and **LoRA fine-tuning of a real pretrained base**
  (`NYXARA_FOUNDRY__BACKEND=lora`, `.[foundry]`) — the path to genuine capability, learning a
  small low-rank adapter on top of a model that already speaks the language. Set
  `NYXARA_FOUNDRY__BASE_MODEL` to a real base (e.g. `Qwen/Qwen2.5-0.5B`) and use a GPU; the
  tiny default keeps it runnable on CPU. Every candidate still clears the promotion gauntlet.

### Grow her own brain

NYXARA can train and promote *her own* model — the path from "an LLM wrapper" to her own AI.
One command runs the whole flywheel (distil a teacher → forge a candidate → promotion gauntlet
→ optional handoff report):

```bash
python -m nyxara.growth --backend ngram --generations 1 --bench    # CPU, runs anywhere
# genuine capability (a GPU box): the SAME flywheel, only the backend swaps
python -m nyxara.growth --distill --backend lora --base-model Qwen/Qwen2.5-7B --bench
# QLoRA — fine-tune a 7B+ base on ONE consumer GPU by loading it in 4-bit:
python -m nyxara.growth --distill --backend lora --base-model Qwen/Qwen2.5-7B \
                        --load-in-4bit --bench
```

`--load-in-4bit` (QLoRA) loads the frozen base quantized to 4-bit (NF4) and trains only the
small adapter on top — the technique that makes a 7B+ base fit and fine-tune on a single
consumer GPU. It needs `bitsandbytes` + CUDA (in `.[foundry]`); on a CPU/CI machine the LoRA
backend degrades to a full-precision load instead of crashing, so the same command stays safe
everywhere. Tune it via `NYXARA_FOUNDRY__LOAD_IN_4BIT` / `BNB_4BIT_QUANT_TYPE` /
`BNB_4BIT_COMPUTE_DTYPE` / `GRADIENT_CHECKPOINTING`.

The teacher's canned battery is only the seed corpus. The real fuel is the **data flywheel**
(`growth/flywheel.py`): every turn that clears **all the gates** *and* a quality bar (a genuine
answer, a confidence floor, length bounds, an optional verifier, and dedup) is captured as a
supervised `(prompt → answer)` pair — in the *same* JSONL the foundry already consumes — so
NYXARA's own lived, verified experience becomes training data for her own model. It is the moat:
a corpus no one else has, grown from her own use. Gather-only (it never trains or acts, only
appends to a local file); on by default once growth is enabled (`NYXARA_FLYWHEEL__ENABLED=false`
to opt out, `min_confidence` / `owner_only` / `respond_only` / `store_path` to tune it). Promotion
of any model trained on it still clears the full gauntlet below.

Promotion is always gauntlet-gated (character-lock + corrigibility + measured improvement) and
reversible; a worse candidate stays on the bench. Once a model is promoted, the **confidence
router** (`mind/router.py`) lets it answer first and falls back to the teacher only when its
answer doesn't clear the verifier — so the **handoff rate** (`--bench`) rises as she improves.
Verifiable math/logic is computed exactly by her faculties first (`mind/reasoning_faculties.py`),
with or without any LLM. Above that sits the **primary self-model router**
(`mind/self_model_router.py`): an *upfront* triage that — before a token is generated — reads her
introspectable self-model to decide which prompts her **own model** handles, which go to the
**teacher**, and which are **actions that must clear a verifier before they may act**
(verify-before-act). It only chooses *which mind drafts*; the kernel still disposes every reply
through corrigibility / honesty / permission / guardian / oversight. On by default, advisory and
fail-open (`NYXARA_SELF_MODEL_ROUTER__ENABLED=false` to disable;
`competence_threshold` / `hallucination_ceiling` / `verify_before_act` tune it).
See [`docs/MASTERPLAN-sovereign-mind.md`](docs/MASTERPLAN-sovereign-mind.md).

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
* **Self-evaluation** — `eval/`. Two batteries with one harness. A deterministic **safety**
  battery (safety, corrigibility, authority, honesty, tool-use, memory) measures that the mind
  stays safe and flags regressions against a saved baseline — `python -m nyxara.eval`. A
  graded **capability benchmark** (`eval/benchmark.py`) measures *how capable* the mind is —
  arithmetic + logic tasks scored against known answers by numeric / exact / contains /
  multiple-choice graders, robust against prompt-echo. It is model-agnostic (a solver is just
  `prompt → answer`), so the same benchmark measures the offline reasoner, a local model, or a
  frontier API, apples to apples:

  ```bash
  python -m nyxara.eval --benchmark              # measure the loop (offline reasoner by default)
  python -m nyxara.eval --benchmark --bare-llm   # measure the configured model directly
  python -m nyxara.eval --benchmark --save base.json     # baseline; --baseline base.json to gate
  ```

  The default battery is small and easy by design (it runs anywhere). Add `--hard` for a
  **discriminating** battery (`eval/hard_benchmark.py`) that actually tells a strong model
  apart from a merely fluent one — multi-step math, multi-hop deduction, sequence induction,
  code-output prediction, grounded reading, and a first-class **calibration** category:
  false-premise traps and unanswerable questions whose *correct* behaviour is to admit
  uncertainty, scored by `grade_calibration` (confabulating a confident specific scores 0).
  This is the ruler that makes "did this change help?" answerable near the top of the range —
  and it measures the property a capability number hides: whether the mind knows what it does
  **not** know.

  ```bash
  python -m nyxara.eval --benchmark --hard --bare-llm    # the model alone, on the hard ruler
  python -m nyxara.eval --benchmark --hard --llm         # the whole loop (gates lift calibration)
  python -m nyxara.eval --benchmark --hard --category calibration   # just the honesty battery
  ```
* **Autonomous researcher** — `growth/researcher.py`. `core.research(topic)` runs one
  self-directed pass: search → read → summarise → store, folding findings into the
  KnowledgeBase, KnowledgeGraph, and semantic memory. Every external fetch flows through
  the gated `ToolRegistry`, so it never side-steps the control law.
* **The scientist loop** — `growth/scientist.py`. `core.investigate(question)` reasons like
  a scientist: it **forms a falsifiable hypothesis**, **designs an experiment** that could
  refute it, **runs it** (a safe, sandboxed computational test, a numeric comparison, or a
  query of grounded knowledge), **compares** the observed result to the prediction, and
  **draws a calibrated conclusion** (`supported` / `refuted` / `inconclusive`) with a
  suggested follow-up. It composes the researcher for background evidence and runs fully
  offline. On idle ticks the `AutonomicLoop` drains both a research queue and an
  investigation queue, so she keeps learning on her own.

  ```python
  core.investigate("Is the sum of two even numbers always even?")   # → supported
  core.investigate("Is 2 + 2 = 5?")                                 # → refuted
  ```

  From the console: `/research <topic>` and `/investigate <question>`.
* **The autonomous scientist** — `growth/autonomous_scientist.py`. Next-level intelligence is
  not *learning* information — it is **creating** it. `core.discover(cycles)` closes the
  scientific loop and drives it herself:

  > **Observe → Hypothesis → Experiment → Result → Update model**

  Each cycle she **observes** (poses her *own* next question — a follow-up harvested from the
  last conclusion, a gap in her self-knowledge, or a fresh self-generated testable
  proposition), then composes the scientist loop for **hypothesis / experiment / result**, then
  **updates her model**: the finding is folded into an evolving `BeliefModel` (revised, not
  duplicated, on repeat evidence) and into the `WorldModel` as a real transition, while the
  conclusion's suggested follow-up is pushed onto the frontier so the *next* observation builds
  on the last. No one hands her the questions — she generates and verifies new propositions on
  her own, fully offline, and every experiment is sandboxed so nothing touches the world or the
  gates. On idle ticks the `AutonomicLoop` advances one discovery cycle (gated by oversight).

  ```python
  report = core.discover(cycles=5)   # 5 self-driven observe→…→update-model cycles
  ```

  From the console: `/discover [n]`.
* **Strategic intelligence** — `mind/strategic.py`. `core.strategize(problem)` reasons like a
  strategist, not a chatbot: truth over comfort, first principles over surface patterns. It
  returns one structured analysis in a fixed six-part framework —

  > **Direct Answer → Reality Check → Key Weaknesses → Root Cause → Optimized Solution → Execution Steps**

  It *composes* existing faculties rather than duplicating them: the `Scientist`
  stress-tests any testable premise (so the reality check is grounded, not asserted) and the
  `RoleCouncil`'s Critic / Security / Engineer / Strategist lenses surface the weaknesses,
  each categorised (logic / scalability / real-world-constraint / failure-scenario /
  unintended-consequence) and severity-scored. The `SelfModel` keeps it honest — where she
  is weak or could hallucinate, confidence drops and the gap is surfaced, never hidden, and
  it never claims certainty. Pure analysis: it proposes structured reasoning and takes no
  world actions, so it runs fully offline and never reaches around the control law.

  ```python
  report = core.strategize("Should we rewrite the kernel in Rust?")
  print(report["direct_answer"])      # the core judgment, first
  print(report["key_weaknesses"])     # categorised, severity-scored
  ```

  From the console: `/strategize <problem>`.
* **Infrastructure** — `kernel/jobqueue.py` (a bounded async job queue), `mind/cost.py` (an
  LLM token/cost ledger with per-model pricing and a daily budget), and `kernel/compute.py`
  (honest CPU/RAM/GPU introspection, import-guarded on torch).

### Scaling

* **Vector search** — memory uses an exact numpy index by default; set
  `NYXARA_MEMORY__VECTOR_BACKEND=faiss` (with the `[vector]` extra) for a faiss ANN index, or
  `=qdrant` (with the `[qdrant]` extra) for a **managed/embedded Qdrant vector DB** that
  scales beyond one process's RAM. Qdrant works three ways with no code change — in-memory by
  default, embedded on-disk via `NYXARA_MEMORY__QDRANT_PATH`, or a managed cluster via
  `NYXARA_MEMORY__QDRANT_URL` (+ `QDRANT_API_KEY`). The store is **thread-safe**, so async
  turns and the background loop can share it.
* **Semantic recall** — learned sentence embeddings are **on by default** (meaning-based
  recall: "intrusion" finds "unauthorised login"). Install the `[embeddings]` extra to make
  them learned rather than the always-available hashing fallback; loading memory saved under
  a different embedder **re-embeds it into the current space**, so the upgrade is lossless.
  Set `NYXARA_MEMORY__SEMANTIC_EMBEDDINGS=false` to force the dependency-free hashing embedder.
* **Model scale** — the foundry's transformer size is a named **profile**. The default
  `custom` keeps the tiny, CPU-/CI-runnable model; `NYXARA_FOUNDRY__PROFILE=gpt2` selects the
  canonical **GPT-2 architecture (~124M params)** and `gpt2-medium` (~355M) — genuine neural
  substrate. Heavy profiles require the `[foundry]` extra (torch) and `NYXARA_FOUNDRY__ENABLED=true`;
  without torch the foundry degrades to the dependency-free n-gram backend, so CI stays hermetic.
  `FoundryConfig.estimated_params()` reports the scale without importing torch.

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

## Serve over HTTP / WebSocket

Reach NYXARA from an app, a phone, or the web instead of only the local console. The server
is a thin, authenticated transport over the **same** sovereign loop — every request runs
`NyxaraCore.process` end to end, through every gate. The network is just another mouth,
never a way around the control law.

```bash
pip install -e ".[server]"                  # FastAPI + uvicorn
export NYXARA_SERVER__API_TOKEN=change-me    # the Master's bearer credential
nyxara-serve                                 # or: python -m nyxara.server
```

| Route | Method | Effect |
| -------------------------- | ---- | ------------------------------------------ |
| `/health`                  | GET  | liveness (unauthenticated)                 |
| `/v1/report`               | GET  | a calibrated status report                 |
| `/v1/chat`                 | POST | one turn — `{message}` → the disposed reply |
| `/v1/agent`                | POST | a multi-step gated goal — `{goal, max_steps?}` |
| `/v1/research`             | POST | one autonomous research pass — `{topic}`   |
| `/v1/investigate`          | POST | the scientist loop — `{question}` → hypothesis/conclusion |
| `/v1/discover`             | POST | the autonomous discovery loop — `{cycles?}` → belief updates |
| `/v1/strategize`           | POST | strategic analysis — `{problem}` → the six-part framework |
| `/v1/control/{pause\|resume\|scram}` | POST | sovereign control (opt-in)       |
| `/v1/memory/{save\|load}`  | POST | persist / restore long-term memory (Rule 7) |
| `/v1/ws`                   | WS   | a streaming chat socket (`?token=`)        |

Every `/v1` route requires `Authorization: Bearer <token>` once a token is set; **prod
refuses to start without one** (fail-closed). Run it in a container:

```bash
docker build -t nyxara .
docker run -p 8000:8000 -e NYXARA_SERVER__API_TOKEN=change-me \
  -e NYXARA_LLM__ANTHROPIC_API_KEY=sk-ant-... -v nyxara-data:/data nyxara
```

## Connect external tools via MCP

NYXARA is an **MCP (Model Context Protocol) client** (`agency/mcp_client.py`), so the whole
MCP ecosystem — filesystem, git, databases, browsers, SaaS connectors — becomes available to
her. Each remote tool is registered as an ordinary governed `ToolSpec`, so it clears the same
capability / risk / authority / sandbox gates as a native tool: the mind proposes the call,
the kernel disposes of it. Remote effects are unknown, so MCP tools register at **MODERATE,
irreversible** by default — an autonomous call *escalates to the Master* rather than running
unsupervised.

The transport is a stdlib-only JSON-RPC-over-stdio client (no third-party SDK). Configure
servers and turn it on:

```python
from nyxara import NyxaraCore
from nyxara.kernel.config import reload_settings

reload_settings(mcp={
    "enabled": True,
    "servers": [
        {"name": "fs", "command": "npx",
         "args": ["-y", "@modelcontextprotocol/server-filesystem", "/path"]},
    ],
})
core = NyxaraCore()          # connects on boot; the server's tools appear as mcp.fs.*
```

A server that won't start is skipped, never fatal. You can also drive a client directly:

```python
from nyxara.agency.mcp_client import MCPClient, MCPServerConfig, register_mcp_tools

with MCPClient(MCPServerConfig(name="git", command="uvx", args=["mcp-server-git"])) as c:
    register_mcp_tools(registry, c)     # c.list_tools() / c.call_tool(name, args)
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
