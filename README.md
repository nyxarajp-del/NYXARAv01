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
