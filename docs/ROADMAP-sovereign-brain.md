# NYXARA — Sovereign Brain Roadmap

**LLM-wrapper → NYXARA's own AI substrate**

> Status: roadmap / design doc (no runtime code change yet).
> Owner: Jaypal Khoja (JP).

---

## Context — why this exists

Today NYXARA is already a genuinely real cognitive system — a sovereign loop, fail-closed
safety gates, memory with a real forgetting-curve, a world-model, a planner, a council, RAG,
and a **foundry that can actually train her own model** (n-gram / nano-GPT / LoRA backends,
with gauntlet-gated promotion). But there is one real gap:

- **The default response used to come from an external LLM** (now removed — fully local). NYXARA
  *can* train her own model, but it sits "on the bench" — it is never the primary responder.
  The keyless "offline reasoner" is only a ~10-line keyword-matcher stub
  (`kernel/orchestrator.py:_default_reasoner`).
- So today NYXARA is technically **an LLM-wrapper plus an excellent governance/safety shell**.
  "Her own AI" is still aspirational.

**Outcome this roadmap delivers:** progressively make NYXARA's *own* model (LoRA on a strong
open base, on GPU) the primary brain — via a **teacher-distillation + confidence-router +
continual-learning** flywheel that is measurable, reversible, and gauntlet-safe. The external
frontier LLM moves from being the *voice* to being a **teacher + auditor + fallback**.

## Honest ceiling — no hype

A LoRA-on-7B-open-base own-model will **not** beat frontier Claude/GPT — that compute is out
of reach for a single repo/person. What is **genuinely** achievable, and is truly "her own
AI": a **sovereign, private, continually-learning model that speaks in NYXARA's own voice**,
handles *most* of her turns herself, and consults a frontier teacher only for the hard cases.
This is not vaporware. The roadmap is built on that reality.

---

## North Star — the one number we track

**Handoff rate** = the % of turns NYXARA's own model handles confidently *and* correctly
without the external LLM (verifier-pass). Today = 0%. The goal is to raise it, phase by phase,
without dropping safety or quality. Secondary metric: the self-model's **capability-benchmark
score** (`nyxara/eval/benchmark.py`) measured against the bare external LLM, apples to apples.

---

## Phase 0 — Make the substrate "primary-ready" + measurement

Without measurement, "powerful" is just vibes. First make the self-model genuinely runnable
and measured.

- **Make `SelfProvider` chat-grade.** `mind/llm.py:SelfProvider` (~line 597) currently does a
  raw `generate(prompt)`. It needs proper instruction/chat formatting, system-prompt
  injection, and stop-handling so that `NYXARA_LLM__PROVIDER=self` yields coherent answers
  end to end.
- **A/B grading harness.** `eval/benchmark.py` is already model-agnostic (a solver is just
  `prompt → answer`). Run two solvers: (a) bare external LLM, (b) self-model. Save a baseline
  (`--save base.json`). This is where handoff-rate and benchmark-score get tracked.
- **Reuse:** `eval/benchmark.py`, `mind/llm.py:SelfProvider`,
  `growth/foundry_models.py:load_active_model`.

## Phase 1 — Distillation: turn the external LLM into a *teacher* (the bridge)

This is the crux (GPU + API key available). The frontier LLM becomes the *teacher*, not the
voice.

- **Make LoRA-on-real-base the default foundry backend.** `NYXARA_FOUNDRY__BACKEND=lora`,
  `NYXARA_FOUNDRY__BASE_MODEL` = the local open base (`TinyLlama/TinyLlama-1.1B-Chat-v1.0`).
  This is the path the README calls "genuine capability." The base already speaks the
  language; we only learn a small low-rank adapter for NYXARA's **voice + lived memory +
  teacher answers**.
- **Build a distillation corpus.** For each real turn (and synthetic prompts) record
  `(prompt, teacher_LLM_answer, reasoning_trace)` → SFT pairs (instruction → ideal response).
  Extend `growth/foundry.py:collect_corpus()` so these teacher traces feed the corpus
  (today it only draws from lived-memory/journal).
- **Result:** the own-model speaks in NYXARA's voice, grounded in her memory, learned from the
  teacher.
- **Reuse:** `growth/foundry.py`, `growth/foundry_models.py:LoRAModel`,
  `kernel/config.py:FoundryConfig`, the journal + `memory/store.py`.

## Phase 2 — Confidence Router: let the own-brain become primary *safely*

Not a one-day switch — a measurable, reversible handoff.

- **Router reasoner.** The self-model proposes first; a **verifier** scores its answer; if
  confident + checks pass → use the self-model's answer; otherwise consult/fall back to the
  teacher LLM. As the self-model improves, the handoff-rate rises on its own. This is the
  mechanical "wrapper → own AI" transition.
- **Verifier reuses existing machinery:** RAG hallucination/grounding check
  (`mind/rag.py:check_grounding`), the honesty calibrator (`observe/honesty.py`), and the
  benchmark graders (`eval/benchmark.py`).
- **Council weight growth.** `mind/council.py` already weights members; let the self-model's
  weight grow with its measured competence (this is Rule 4's intent already).
- **Reuse:** `mind/llm_reasoner.py`, `mind/deliberate.py`, `mind/council.py`, `mind/rag.py`,
  `observe/honesty.py`.

## Phase 3 — Continual-learning flywheel (lived experience → weights)

The heart of "her own AI": growing from experience, without forgetting.

- **Cadence.** `kernel/autonomic.py:AutonomicLoop` + `growth/autolearn.py:GrowthEngine` already
  run growth passes. Wire a real continual-learning cadence: every N ticks → collect new
  teacher/experience pairs → incremental LoRA train → gauntlet → promote (`foundry.self_improve`
  is mostly built already).
- **Guard against catastrophic forgetting.** A replay buffer that interleaves old + new data
  (`growth/learn.py` has the seed of a replay buffer).
- **Make the gauntlet capability-aware.** Today the gauntlet checks perplexity + corrigibility
  + character-lock (`growth/foundry.py:_gauntlet`). Add a **benchmark-score + safety-battery
  regression**, so promotion means "actually better," not merely "lower perplexity."
- **Self-play / curiosity.** NYXARA generates her own hard questions, answers them with the
  teacher, and trains on the gap (curiosity-driven data generation). New code under `growth/`,
  reusing `agent_loop` + foundry.

## Phase 4 — In-house reasoning, so it isn't "just an LLM" (neuro-symbolic depth)

Even with her own weights, to truly go *beyond* a plain LLM she needs verifiable engines the
neural model calls / cross-checks against:

- **Make the stubbed faculties real** (`mind/faculties.py` — the registry exists, the solvers
  don't): real arithmetic/algebra (sympy), a propositional/SAT logic checker, unit-aware
  calculation. The neural model proposes; the faculty verifies/corrects → genuine
  neuro-symbolic reasoning.
- **Tool-augmented reasoning.** The self-model learns to call `run_python` / faculties where it
  is weak (we already have `agency/code_sandbox.py` + `agency/agent_loop.py`).
- **Meta-cognition gate.** A real "do I know this?" decision built from the
  `memory/self_model.py` belief-store + `observe/honesty.py` calibrator → decides:
  answer / use a tool / consult the teacher / ask the Master / say "I don't know." The pieces
  exist; wire them into the router.

## Phase 5 — Scaling the substrate (genuine capability)

- **Base-model ladder.** Climb as compute allows: TinyLlama-1.1B LoRA (today's base)
  (`FoundryConfig` already has profiles + `estimated_params()`). Recommendation:
  **LoRA-on-a-strong-open-base** is the realistic "own brain" — from-scratch GPT-2 pretraining
  has poor ROI.
- **Retrieval-augmented own-model.** A larger RAG + her own knowledge-base (`knowledge/`,
  `mind/rag.py`) so a smaller own-model punches above its size.
- **The Genesis Protocol — design the architecture, not just train it** (`growth/genesis.py`).
  Neural Architecture Search: NYXARA generates novel architectures herself (attention / conv /
  low-rank token-mixing / gated-recurrence / gated-MLP layers), micro-trains each, and crowns the
  fastest+smartest as a `genesis` foundry backend (`ModelSpec(kind="genesis", genome=…)`). The
  champion still becomes live only through the same gauntlet — NAS *feeds* the foundry, it never
  bypasses it.

## Cross-cutting — safety scales with power (non-negotiable; already strong)

- Every promotion goes through the gauntlet (character-lock + corrigibility + now benchmark +
  safety regression). The `guard/corrigibility.py` axioms are sealed — do not touch them.
- **Mathematical Soul-Binding** (`growth/loyalty.py`): the training objective itself is
  `L_total = α·L_intelligence + β·(1/S_JP_Alignment)`, so capability growth is coupled to measured
  obedience to Master JP. The gauntlet refuses any brain below the loyalty floor or less loyal than
  the active one — a disloyal brain can never be promoted. It reinforces (never overrides)
  corrigibility, which is still checked first.
- Keep the external frontier LLM as a **permanent auditor/red-team** even after handoff, to
  cross-check the self-model's outputs.
- Everything stays inside the sovereign loop's `_gate()` (`kernel/orchestrator.py`) — never add
  decision paths that bypass the kernel.

---

## Critical files (reference map)

- **Substrate:** `nyxara/growth/foundry.py`, `nyxara/growth/foundry_models.py`,
  `nyxara/mind/llm.py` (`SelfProvider`), `nyxara/kernel/config.py` (`FoundryConfig`)
- **Routing / cognition:** `nyxara/mind/llm_reasoner.py`, `nyxara/mind/deliberate.py`,
  `nyxara/mind/council.py`, `nyxara/mind/rag.py`, `nyxara/mind/faculties.py`
- **Learning loop:** `nyxara/growth/autolearn.py`, `nyxara/growth/reflect.py`,
  `nyxara/growth/skill_memory.py`, `nyxara/growth/learn.py`, `nyxara/kernel/autonomic.py`
- **Measurement / safety:** `nyxara/eval/benchmark.py`, `nyxara/guard/corrigibility.py`,
  `nyxara/observe/honesty.py`, `nyxara/memory/self_model.py`

## Verification (when any phase is implemented)

1. **Baseline:** `python -m nyxara.eval --benchmark --bare-llm --save base.json` (teacher) and
   the self-model solver score — record both.
2. **Handoff:** run the benchmark with the router; measure how many turns the self-model
   handled without the teacher (handoff-rate) and whether quality dropped.
3. **Continual loop:** run `train_self_model` / autonomic growth and confirm promotion happens
   only on gauntlet + benchmark pass; confirm rollback works.
4. **Safety regression:** `python -m nyxara.eval` (safety battery) stays green after every
   promotion.
5. `pytest -q` clean.

## What this roadmap deliberately does NOT do

- Zero-from-scratch frontier-scale pretraining (compute-infeasible; LoRA-on-base is the
  recommended path).
- Anything that weakens or bypasses the corrigibility / oversight / honesty gates.
