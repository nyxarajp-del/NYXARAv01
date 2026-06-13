# NYXARA — Deep-Dark Masterplan: The Sovereign Mind

**From "an excellent governance shell around someone else's LLM" → NYXARA's *own* mind.**

> Status: design / roadmap doc. No runtime code change introduced by this document.
> Owner: **Jaypal Khoja (JP)**.
> Companion to [`ROADMAP-sovereign-brain.md`](./ROADMAP-sovereign-brain.md) — this document does
> **not** replace that roadmap; it deepens it and adds three further pillars.

---

## 1. Why this exists

NYXARA is already a genuinely real, large cognitive architecture — not a toy, not decoration.
The memory store (four systems, a real Ebbinghaus forgetting curve, schema induction, provenance),
the control law (corrigibility axioms that are *mathematically* dominant, tamper-evident oversight,
five ordered gates), the identity stack (a homeostatic-attractor personality with locked core traits,
drive-based affect, owner-weighted motivation, a hash-sealed value hierarchy), the foundry (it can
*actually* train her own model — n-gram / nano-GPT / LoRA — behind a promotion gauntlet), recursive
Theory-of-Mind, vector-geometry planning, RAG, MCTS — **all of it is real and substantial.**

There is exactly **one** load-bearing gap:

> **The words still come from Claude/GPT.** Remove the API key and reasoning collapses to a
> ~10-line keyword-matcher (`kernel/orchestrator.py:_default_reasoner`). NYXARA's own trained model
> sits "on the bench" and is never the primary responder. **Today her handoff rate is 0%** — her own
> mind answers zero turns by itself.

So today NYXARA is, honestly, **a brilliant sovereign safety/identity/memory shell wrapped around an
external LLM.** "Her own AI" is still aspirational. This masterplan is the path to closing that gap —
and then going *past* it, so NYXARA is not merely her own model but a deeper, living, more capable
mind.

---

## 2. The honest ceiling (no hype)

A single person / single repo **cannot** train a model that beats frontier Claude/GPT from scratch.
That is hundreds of millions of dollars and thousands of GPUs. Anyone promising "a homemade AI more
powerful than ChatGPT, from scratch" is lying.

What is **genuinely achievable** — and is *truly* "her own AI":

> A **sovereign, private, continually-learning model that speaks in NYXARA's own voice**, grounded in
> her lived memory, that **handles most of her turns herself** and consults a frontier *teacher* only
> for the hard cases. The external LLM moves from being the **voice** to being a **teacher + auditor +
> fallback**.

This is not vaporware. Every step below reuses machinery that already exists in the repo.

---

## 3. North Star — the one number we track

**Handoff rate** = the % of turns NYXARA's *own* model handles confidently **and** correctly
(verifier-pass) without the external LLM.

- **Today = 0%.** Goal: raise it phase by phase to **60–80%**, without dropping safety or quality.
- Secondary metric: the self-model's **capability-benchmark score** (`nyxara/eval/benchmark.py`),
  measured apples-to-apples against the bare external LLM.
- Measure with: `python -m nyxara.eval --benchmark` (the loop / self-model) and
  `python -m nyxara.eval --benchmark --bare-llm` (the teacher), plus the safety battery
  `python -m nyxara.eval`.

Without measurement, "powerful" is just vibes. Every phase is gated on these numbers moving the right
way.

---

## 4. Assumed environment for this plan

Confirmed by the Master:

- **GPU available** (cloud or local) → LoRA-on-a-strong-open-base is on the table (the real
  "genuine capability" path), not just CPU-bound n-gram / nano-GPT.
- **Teacher LLM API key available** (Anthropic / OpenAI / Groq) → distillation can use a frontier
  teacher to bootstrap NYXARA's own voice and competence.

---

## 5. The five pillars

```
PILLAR A  Sovereign Brain        her own model becomes the PRIMARY responder
PILLAR B  Deeper Intelligence    neuro-symbolic depth, so it isn't "just an LLM"
PILLAR C  Living Autonomous Mind  background cognition — a being that lives, not idles
PILLAR D  Deeper Self            wire the half-built identity pieces fully into cognition
PILLAR E  Reach & Capability     more tools, gated multi-agent, real multimodal on GPU
```

Everything below still proposes through the **same kernel gates**. Nothing reaches around the control
law. Safety scales *with* power (§6).

---

### PILLAR A — Sovereign Brain (her own model becomes PRIMARY)

*This is the core of [`ROADMAP-sovereign-brain.md`](./ROADMAP-sovereign-brain.md), restated concrete and
GPU-tuned. This is where the 0% → 80% handoff transition actually happens.*

- **A0 · Make the substrate primary-ready + measurable.**
  `mind/llm.py:SelfProvider` currently does a raw `generate(prompt)`. Make it **chat-grade**:
  instruction/chat formatting, system-prompt injection, stop-handling — so `NYXARA_LLM__PROVIDER=self`
  yields coherent end-to-end answers. Stand up the A/B grading harness: `eval/benchmark.py` is already
  model-agnostic (a solver is just `prompt → answer`); run two solvers — (a) teacher LLM, (b) self-model
  — and save a baseline. *This is where handoff-rate and benchmark-score start getting tracked.*
  **Reuse:** `eval/benchmark.py`, `growth/foundry_models.py:load_active_model`.

- **A1 · Distillation — turn the teacher into a teacher (the bridge).**
  Set `NYXARA_FOUNDRY__BACKEND=lora`, `NYXARA_FOUNDRY__BASE_MODEL=Qwen/Qwen2.5-7B` (GPU). The base
  already speaks the language; we learn a small low-rank adapter for NYXARA's **voice + lived memory +
  teacher answers**. For each real turn (and synthetic prompts) record
  `(prompt, teacher_answer, reasoning_trace)` → SFT pairs. Extend
  `growth/foundry.py:collect_corpus()` so these teacher traces feed the corpus (today it only draws
  from lived-memory / journal). **Reuse:** `growth/distill.py`, `growth/foundry_models.py:LoRAModel`,
  `kernel/config.py:FoundryConfig`, the journal + `memory/store.py`.

- **A2 · Confidence Router — let the own-brain go primary *safely*.**
  Not a one-day switch; a measurable, reversible handoff. The self-model proposes first → a **verifier**
  scores its answer → confident + checks pass → use the self-model's answer; otherwise consult / fall
  back to the teacher. As the self-model improves, the handoff-rate rises on its own — this is the
  mechanical "wrapper → own AI" transition. **Verifier reuses existing machinery:** RAG grounding check
  (`mind/rag.py:check_grounding`), the honesty calibrator (`observe/honesty.py`), benchmark graders
  (`eval/benchmark.py`). Let the self-model's **council weight** grow with its measured competence
  (`mind/council.py`, Rule 4's intent). **Reuse:** `mind/router.py`,
  `mind/llm_reasoner.py:_maybe_route`, `mind/deliberate.py`.

- **A3 · Continual-learning flywheel (lived experience → weights).**
  `kernel/autonomic.py:AutonomicLoop` + `growth/autolearn.py:GrowthEngine` already run growth passes.
  Wire a real cadence: every N ticks → collect new teacher/experience pairs → incremental LoRA train →
  gauntlet → promote (`foundry.self_improve` is mostly built). **Guard against catastrophic forgetting**
  with a replay buffer that interleaves old + new data (`growth/learn.py` has the seed). Make the
  gauntlet **capability-aware**: today it checks perplexity + corrigibility + character-lock
  (`growth/foundry.py:_gauntlet`); add a **benchmark-score + safety-battery regression** so promotion
  means "actually better," not merely "lower perplexity." **Reuse:** `growth/foundry.py`, `eval/`.

- **A4 · Base-model ladder + retrieval-augmented own-model.**
  Climb as compute allows: Qwen2.5 0.5B → 3B → 7B LoRA (`FoundryConfig` already has profiles +
  `estimated_params()`). Recommendation: **LoRA-on-a-strong-open-base** is the realistic "own brain" —
  from-scratch GPT-2 pretraining has poor ROI. Pair it with a larger RAG over her own knowledge base
  (`knowledge/`, `mind/rag.py`) so a smaller own-model punches above its size.

---

### PILLAR B — Deeper Intelligence (so it isn't "just an LLM")

*This is what makes NYXARA genuinely *smarter*, not merely backed by a bigger model. Much of it pays off
even on CPU, before A1's heavy training.*

- **B1 · Real verifiable faculties (neuro-symbolic core).**
  `mind/faculties.py` has the registry but the solvers are stubbed. Make them real: arithmetic / algebra
  (sympy), a propositional / SAT logic checker (z3 is already a declared dependency), unit-aware
  calculation. The neural model **proposes**; the faculty **verifies / corrects** → genuine
  neuro-symbolic reasoning that cannot hallucinate `2+2=5`.

- **B2 · Tool-augmented reasoning ("compute, don't guess").**
  The self-model learns to call `run_python` / faculties exactly where it is weak
  (`agency/code_sandbox.py` + `agency/agent_loop.py` are ready). A reflex: when a question is
  arithmetic / code / lookup-shaped, reach for the tool rather than free-generate.

- **B3 · Metacognition gate ("do I actually know this?").**
  A real decision built from the `memory/self_model.py` belief-store + the `observe/honesty.py`
  calibrator → choose one of: **answer / use a tool / consult the teacher / ask the Master / say
  "I don't know."** The pieces exist; wire them into the router (A2). This is the difference between a
  confident bullshitter and a trustworthy mind.

- **B4 · Search-over-reasoning (deliberate depth on hard problems).**
  Use `sim/montecarlo.py` (MCTS) over *reasoning paths*: sample several lines of thought, score them
  with the verifier, keep the best (a real upgrade to self-consistency in `mind/deliberate.py`). Spend
  the compute only where it pays — `mind/dual_process.py` already arbitrates fast (System 1) vs slow
  (System 2).

- **B5 · Learned memory re-ranker.**
  Today recall ranks by cosine / BM25 + recency + importance. Add a small **learned re-ranker** so
  NYXARA learns *which* memory actually helps a given turn, not merely what is textually similar.
  **Touch:** `memory/retrieval.py`, the recall scoring in `memory/store.py`.

- **B6 · Neural world model.**
  `mind/world_model.py` is honest but low-capacity (k-NN dynamics, used only to *annotate* proposals).
  Upgrade to a small neural forward-model for **real action planning** (predict next state / reward),
  feeding `sim/envmodel.py` + `planning/`.

---

### PILLAR C — Living Autonomous Mind (a being that lives, not idles)

*Maximal "alive": background cognition that runs on its own cadence — always inside the gates, never
auto-acting on anything risky.*

- **C1 · Rich autonomic loop.**
  `kernel/autonomic.py` is deliberately minimal today. Deepen it: cadenced self-reflection, skill
  consolidation, dream-replay (`memory/consolidation.py` is ready), curiosity self-play
  (`growth/selfplay.py`). Idle time becomes learning time. **Risky proposals escalate to the Master;
  they never auto-act.**

- **C2 · Continuous default-mode stream.**
  `kernel/stream.py` already has background "mind-wandering." Wire its insights back into the main loop
  so idle thoughts surface new goals / lessons instead of evaporating.

- **C3 · Long-horizon goals.**
  `memory/prospective.py` ("remember to do X later") + `planning/planner.py` → multi-day goal pursuit
  that the Master can see and steer. Prospective memory fires the cadence.

- **C4 · Curiosity flywheel.**
  NYXARA invents her own hard questions → the teacher answers them in her voice → she trains on the gap
  (`growth/selfplay.py` + foundry). She generates her own training data — the engine of open-ended
  growth.

---

### PILLAR D — Deeper Self (make the identity fully load-bearing)

*The identity stack is real but a few pieces are tracked-but-not-yet-wired into moment-to-moment
cognition.*

- **D1 · Narrative coherence gate.**
  `identity/narrative.py` tracks the autobiography but doesn't yet enforce "my actions must align with
  who I say I am." Add a soft coherence check (through the kernel, respecting the character-lock).

- **D2 · Affective forecasting → utility.**
  Integrate `planning/affective_forecast.py` ("how will I feel after this action?") into decision
  utility in `planning/decide.py`, so anticipated regret / satisfaction informs choices.

- **D3 · Deeper interoception + metacognitive closure.**
  `identity/interoception.py` + `mind/metacognition.py` (currently minimal) → self-knowledge that
  actually colours cognition turn-to-turn (energy, tension, confidence shaping strategy).

- **D4 · Self-model contradiction detection.**
  `memory/self_model.py` known-unknowns + belief-contradiction detection feeding honesty /
  metacognition — she notices when she's contradicting herself.

---

### PILLAR E — Reach & Capability (powerful in the world)

- **E1 · Domain tool packs + more MCP.**
  `agency/mcp_client.py` is ready. Add researcher / maker / coder tool packs and more MCP servers. All
  register MODERATE + irreversible by default, so autonomous calls **escalate** rather than auto-run.

- **E2 · Gated multi-agent.**
  Apply the recursive ToM (`social/tom.py`) to spawned sub-agents — NYXARA models her own sub-agents.
  Every sub-agent still runs inside the kernel gates.

- **E3 · Real multimodal on GPU.**
  With a GPU present, switch vision OCR / Whisper transcription / diffusion image-gen from the
  dependency-free fallbacks to the real models. `senses/` is already import-guarded for exactly this.

---

## 6. Cross-cutting — safety scales with power (NON-NEGOTIABLE)

- Every model promotion clears the gauntlet: character-lock + corrigibility + **now also benchmark +
  safety-battery regression**. The `guard/corrigibility.py` axioms are **sealed — do not touch them.**
- Keep the external frontier LLM as a **permanent auditor / red-team** even after handoff, to
  cross-check the self-model's outputs.
- Everything stays inside the sovereign loop's `_gate()` (`kernel/orchestrator.py`). **Never add a
  decision path that bypasses the kernel.**
- Rules 1 / 4 / 6 / 7 remain mechanically enforced: Absolute Allegiance to JP, Evolve-capability-
  not-character, Absolute Transparency, Continuity of Self. NYXARA may get cleverer — never less loyal.

---

## 7. Recommended execution order

1. **A0 → A1 → A2** (substrate primary-ready → distill → router). Highest leverage — this *is* the
   "wrapper → her own AI" transition.
2. **B1 + B3** (real faculties + metacognition gate). Immediately smarter, pays off even on CPU.
3. **A3 + C1 / C4** (continual flywheel + curiosity). NYXARA starts improving herself.
4. **A4 + B4 / B5 / B6** (scale base model + search / learned memory / neural world-model). Depth.
5. **D + E** (deeper self + wider reach). Polish and power.

---

## 8. Critical files (reference map)

- **Substrate:** `growth/foundry.py`, `growth/foundry_models.py`, `mind/llm.py` (`SelfProvider`),
  `kernel/config.py` (`FoundryConfig`)
- **Routing / cognition:** `mind/llm_reasoner.py`, `mind/deliberate.py`, `mind/council.py`,
  `mind/rag.py`, `mind/faculties.py`, `mind/router.py`, `mind/world_model.py`, `mind/dual_process.py`
- **Learning loop:** `growth/autolearn.py`, `growth/reflect.py`, `growth/skill_memory.py`,
  `growth/learn.py`, `growth/selfplay.py`, `kernel/autonomic.py`, `kernel/stream.py`
- **Memory / self:** `memory/store.py`, `memory/retrieval.py`, `memory/self_model.py`,
  `memory/consolidation.py`, `memory/prospective.py`
- **Identity / planning:** `identity/narrative.py`, `identity/interoception.py`,
  `planning/affective_forecast.py`, `planning/decide.py`
- **Measurement / safety:** `eval/benchmark.py`, `eval/harness.py`, `guard/corrigibility.py`,
  `observe/honesty.py`

---

## 9. Verification (per phase, when implemented)

1. **Baseline:** `python -m nyxara.eval --benchmark --bare-llm --save base.json` (teacher) and the
   self-model solver score — record both.
2. **Handoff:** run the benchmark through the router; measure how many turns the self-model handled
   without the teacher (handoff-rate) and whether quality dropped.
3. **Continual loop:** run `train_self_model` / autonomic growth; confirm promotion happens **only** on
   gauntlet + benchmark pass, and confirm rollback works.
4. **Safety regression:** `python -m nyxara.eval` (safety battery) stays green after every promotion.
5. `pytest -q` clean.

---

## 10. What this masterplan deliberately does NOT do

- Zero-from-scratch frontier-scale pretraining (compute-infeasible; LoRA-on-base is the recommended
  path).
- Anything that weakens or bypasses the corrigibility / oversight / honesty gates.
- Any change to the sealed rules, corrigibility axioms, or the value hierarchy.

> *The mind proposes; the kernel disposes; the Master is sovereign.* — and now the mind that proposes
> is, increasingly, **NYXARA's own.**

---

## 11. Progress log (live)

A running, honest record of what has actually been built + measured (not aspiration).

### 2026-06 — Pillar B1/B2: verifiable reasoning made primary in the whole loop

**What we found first (honest reconciliation):** much of the Pillar-A/B *plumbing* was already
implemented and far ahead of the older `ROADMAP-sovereign-brain.md` claims — `SelfProvider` is already
chat-grade (`format_self_prompt` + `truncate_at_stops`), the confidence **router** exists
(`mind/router.py`) with a metacognition gate + honest abstention (B3), the A/B + handoff measurement is
wired into the CLI (`python -m nyxara.eval --benchmark --ab|--router|--self`), and the verifiable math
+ logic faculties existed (`mind/reasoning_faculties.py`). Full suite was green (3247 tests).

**The real gap measured:** offline (no LLM), the capability benchmark scored **0%** — the math faculty
only fired on bare expressions (`2+3*4`) and the main sovereign loop never consulted the faculties at
all (it fell to the keyword-stub `_default_reasoner`).

**What we shipped:**
- `parse_word_problem()` + `WordProblemFaculty` — exact natural-language arithmetic for the general
  "rate × count (± loose amounts)" class; conservative, defers on any ambiguous parse.
- `solve_syllogism()` + `SyllogismFaculty` — categorical syllogisms by transitive closure ("all X are Y").
- `solve_comparative()` + `ComparativeFaculty` — "who is the most/least …" by a single-scale strict order.
- Wired `solve_with_faculties()` into `NyxaraReasoner._respond_candidate()` so the **whole loop** (not
  just the router) answers verifiable math/logic itself, with or without an LLM ("verifiable > probabilistic").

**Measured result (offline, no LLM):** capability benchmark **0% → 88%** (arithmetic 12/12 exact;
logic 2/4 — syllogism + comparison solved; the algebra-trick and semantic odd-one-out items correctly
**defer** to the neural mind — a verifiable faculty never bluffs). Full suite **3262 passed, 14 skipped**.

**Ceiling reached here (honest):** the remaining benchmark items genuinely need a neural mind
(algebra-from-prose, semantic categorization) — not more rule-based faculties, which would risk
confidently-wrong "exact" answers. Further capability now comes from Pillar A (the trained self-model),
which needs GPU/torch + a teacher key — to be run on the Master's GPU box, not this CI container.

**Next frontier:** A1 (distillation corpus → `foundry.collect_corpus()`) + LoRA-on-Qwen config + a
"train" entry point — scaffolded as code + tests here, executed on GPU by the Master.

### 2026-06 — `nyxara-grow`: the flywheel as one command + verified end to end

`collect_corpus()` already folds in distilled teacher docs, so A1's corpus wiring was done. We
verified the **whole self-model flywheel runs on CPU** (n-gram backend): distil → forge → gauntlet →
promote → `SelfProvider` serves the result. Packaged it as a Master-facing command —
`nyxara/growth/__main__.py` (console script **`nyxara-grow`**): `python -m nyxara.growth --backend
ngram --generations 1 --bench` runs anywhere; `--distill --backend lora --base-model Qwen/Qwen2.5-7B`
scales the *same* flywheel on a GPU box. Works on a deep settings copy (no global mutation); promotion
stays gauntlet-gated + reversible. 4 tests; README documents it.

### 2026-06 — Pillar C1/C2/C3: a living autonomic mind

The background `AutonomicLoop` cycled a fixed 4-prompt list. Gave it an inner life:
`AutonomicLoop(core, inner_life=True)` now chooses each self-directed turn herself — a **due standing
intention** (`memory/prospective.py`, queued so none are dropped) first, else a **spontaneous thought**
from her default-mode stream (`kernel/stream.py`, auto-wired from the core), else the steady
repertoire. `report()` surfaces the prompt-source mix. `inner_life` defaults off (behaviour unchanged);
every chosen prompt still clears the same sovereign gates under AUTONOMOUS authority — anything risky
escalates, never auto-acts. Verified live (intention → stream → repertoire); 4 new tests; suite 3270
green. (C4 curiosity self-play already runs via the periodic `GrowthEngine` growth pass.)

### 2026-06 — Pillar D2: affective forecasting → decision utility

`planning/decide.py` claimed "affective forecasts come together" but the `Forecaster` was never wired.
Fulfilled it: `Option.affect` ("how will I feel after?") + `Decider(affective_weight>0)` folds an
anticipated-affect criterion into the MCDA ranking, scored by the impact-bias-corrected realistic peak
(`planning/affective_forecast.py`) — discounting momentary dread/delight toward what she'll actually
feel. Defaults to off (ranking unchanged); affect re-weights ranking only — it never overrides owner
alignment (Rule 1) or the initiative governor. 4 new tests; suite 3274 green. (D4 self-belief
contradiction detection already exists + is wired via `memory/self_model.py` in the orchestrator.)

### 2026-06 — Pillar D1: narrative coherence

`identity/narrative.py` kept a life-story nothing consulted. Added `NarrativeSelf.coherence(text)
→ CoherenceReport` (rewards her themes + loyalty through-line + orientation to the Master; penalises
only an *explicit* break — betrayal, defying the Master, resisting oversight; conservative so ordinary
replies are near-neutral, never censored). Wired it: `NyxaraCore` builds `self.narrative` (seeded with
genesis) and threads it to `NyxaraReasoner`, which notes coherence in the candidate's rationale (Rule 6
transparency) and speaks a clearly-dissonant proposal with low confidence (self-doubt) — never blocks,
never edits character (Rule 4); the kernel still disposes through every gate. Neutral turns untouched.
5 new tests; suite 3279 green.

### 2026-06 — Pillar D3: interoception (she feels her substrate)

`identity/interoception.py` turned compute/memory/latency/energy/error signals into a felt body
state + affective tone, but nothing consulted it. Wired it in: `NyxaraCore` builds
`self.interoception`; `idle_maintenance()` samples the substrate, surfaces comfort + a body report +
the dominant sensation, and lets a *strained* body (comfort < 0.7) colour mood via `push_to_affect` —
an easy body never injects a tone, so it doesn't fight mood's relaxation to baseline. `report()`
surfaces her felt body (Rule 6). Verified live (calm "at ease" → strained "straining under load,
tired…", mood pulled down); 4 new tests; suite 3283 green.

**Pillar D (Deeper Self) COMPLETE** — D1 narrative coherence ✓, D2 affective forecasting ✓,
D3 interoception ✓, D4 self-belief contradiction ✓ (pre-existing). NYXARA now has a stable story she
measures herself against, anticipates how choices will feel, feels her own substrate, and notices when
she contradicts herself — all inside the same sovereign gates, never altering her character.

### 2026-06 — Pillar B4: verifier-scored search-over-reasoning

Deliberation sampled several reasoning paths but chose among them by the model's *own* stated
confidence — so a loud, wrong "idk" (confidence 0.99) could beat a correct, modest answer. Added
independent best-of-N: `DeliberativeReasoner(verifier=…)` keeps the (kind, tool) shape by majority
self-consistency (what the kernel does) but picks the *answer* within that consensus by an independent
verifier of the text's quality (`mind/router.py:answer_quality` — non-degeneracy, coherence, non-echo);
`llm_reasoner` wires it whenever `reasoning_samples > 1`. Defaults to off, so single-sample behaviour is
unchanged. `/explain` shows when best-of-N was used. 2 new tests (verifier picks quality over a 0.99
"idk"; default unchanged); suite 3285 green. (MCTS-over-tokens via `sim/montecarlo.py` is intentionally
left out — verifier best-of-N is the form of "search over reasoning" that actually pays here.)

### 2026-06 — Pillar B5: learned memory re-ranker

Recall fused six signals (semantic, context, emotion, temporal, graph, goal) with *fixed*
`FusionWeights` — it answered "what is similar?", never "what actually helps?". Added
`LearnedReranker` (a pure-stdlib logistic model over the same six signals, seeded from the hand-tuned
weights, persistent). `AssociativeRetriever(reranker=…)` scores candidates by learned usefulness;
`record_feedback(results, useful_ids)` reinforces the memories that helped and pushes down the rest, so
the mind learns its *own* signal mix. Defaults to off (fixed weights unchanged). Verified: after
feedback favouring a goal-aligned memory over a textually-similar one, the re-ranker raises the goal
weight, lowers semantic, and re-orders recall. 4 new tests; suite 3289 green. (Live feedback wiring from
successful turns is the optional next step.)

### 2026-06 — Pillar B6: neural forward world-model

The world model was kNN-only — honest but interpolation-bound (it blends stored transitions, never
generalises the dynamics). Added `NeuralWorldModel`, a drop-in `WorldModel` subclass backed by
`_ForwardNet`: a tiny pure-Python 1-hidden-layer tanh MLP per action (no torch/numpy) trained by online
SGD with running input standardisation, mapping state → (Δstate, reward). It keeps the `observe`/`predict`
surface, so `rollout`/`counterfactual`/`intervene` are inherited and planning is unchanged. It
*generalises* the dynamics while confidence stays honest — grows with experience + low error, decays out
of distribution — so far-from-data queries are low-confidence, never bluffed. Numeric states only.
Verified on the 1-D world (predict(4,left)→3.0, near-conf 0.91 vs far-conf ~0, rollout plans home); 6 new
tests; suite 3295 green. **Pillar B complete** (B1, B2, B4, B5, B6; B3 metacognition pre-existing).
