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
| `/eureka [n]`     | truly novel problem solving: invent → prove → keep novel + interesting (`n` generations) |
| `/meta-discover <topic>` | meta-research: invent → sandbox-test → (gated) integrate new theories |
| `/dream`          | enter a Dream State: distil logs → prune useless → fix Deep Memory Synapses |
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
* **Long-horizon missions** — `core.mission(goal)` (the `MissionExecutive`,
  `agency/mission.py`) pursues an *open-ended* objective over days or months, not one
  reactive burst. It **decomposes** the goal into ordered milestones, **advances** each
  through the full `AgentLoop` gate pipeline, **checkpoints to JSON after every milestone**
  (so a mission resumes exactly where it left off across restarts — `core.resume_mission(id)`),
  **re-plans** stalled milestones into sub-steps, and **defers-and-continues** when a step
  needs the Master: that milestone parks `BLOCKED` and is surfaced (`mission.escalations`)
  while the rest keep moving, resuming once you `approve` it. The `AutonomicLoop` advances a
  standing mission one gated milestone per background tick, and every action is written to a
  tamper-evident hash-chained `Journal`. Hard budgets + a no-progress guard mean it never
  spins. Autonomy buys horizon and persistence — never extra power; every step still gates.

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

**AutoForge closes the loop autonomously** (`growth/autoforge.py`). On idle ticks NYXARA checks
whether enough *new* verified experience has accrued (her flywheel corpus + any teacher
distillation); once it passes `NYXARA_AUTOFORGE__MIN_EXAMPLES` she runs one
Collect → Train → Gate → Promote/Discard cycle on her own — delegated to the same gauntlet, so a
worse or character-violating candidate is never promoted, and she never trains while
paused/scrammed. So the whole flywheel turns by itself: she talks, collects, forges a better
model from her own lived experience, and the better model talks next — measured every step by the
hard benchmark. On by default with growth; heavy backends (lora/QLoRA) still require
`foundry.enabled`. `NYXARA_AUTOFORGE__ENABLED=false` to opt out.

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

### The Genesis Protocol — she designs her own brain (Neural Architecture Search)

The foundry above trains a *fixed* architecture. The **Genesis Protocol** (`growth/genesis.py`)
goes one level deeper: NYXARA does not copy a pre-built architecture (Transformer, LLaMA, …) — she
**designs brand-new neural architectures herself** and tests which one is the best brain. Each
candidate is an `ArchitectureGenome`: a sequence of layers drawn from a real palette of mixers —
causal **attention**, a depthwise causal **conv** token-mixer, a learned **low-rank causal
token-mixing matrix** (a novel matrix structure), a lightweight **gated recurrence**, and gated
channel mixers (`gated_mlp` / `glu`) — so the topologies it discovers diverge from a vanilla
transformer. She builds each genome into a real PyTorch model, **micro-trains it from scratch**,
and scores it by a fitness that blends *smartness* (low perplexity) with *speed* (fewer params,
less wall-time) — the literal "fastest **and** smartest." An evolutionary loop (elitism +
mutation + crossover) breeds better architectures generation by generation.

```bash
python -m nyxara.growth.genesis_main --generations 3 --population 6              # CPU, runs anywhere
python -m nyxara.growth.genesis_main --generations 4 --population 8 \
        --backend torch --promote                                               # real neural search + go live
```

The crowned champion becomes her **live brain** the only way anything does — by clearing the very
same gauntlet (character-lock + corrigibility + perplexity improvement + capability
non-regression), reversible via rollback. A `genesis` model is a first-class foundry backend
(`ModelSpec(kind="genesis", genome=…)`), so the existing `SelfProvider` speaks it and AutoForge's
loaders load it. On a machine without torch the protocol still runs — it searches over an
always-runnable n-gram substrate and still crowns a champion — so it is hermetic and CI-testable.
ON by default but cheap (a tiny population); scale `population_size` / `generations` (and point it
at a GPU box with `--backend torch`) for a deeper search. `NYXARA_GENESIS__ENABLED=false` to opt
out; trigger on demand from code via `core.genesis_search(...)`. Like everything else, the search
proposes and the kernel disposes — a paused/scrammed mind may search but never promotes.

### Synthetic Data Self-Curation — the AlphaGo-Zero method

Human data is finite and biased; a mind that only ever consumes it inherits its ceiling. **AlphaGo
Zero** broke that ceiling by generating its own experience and keeping only what *verified*. NYXARA
does the same (`growth/synthesis.py`): she **manufactures purely logical synthetic data**
(arithmetic, algebraic identities, propositional logic, number theory, small code structures), has
an **independent rival verifier** certify each item — the `Prover` for decidable domains (exact,
machine-checkable) and a restricted sandbox + reference oracle for code — and feeds only the
survivors into her base `KnowledgeBase` *and* the **same** JSONL corpus the foundry forges from
(marked `verified`). Generation is hard, verification is cheap — so she grows verified knowledge no
human had to write, then Genesis/AutoForge fold it into her own model. Gather-only (never trains or
acts), pure-stdlib and CI-safe. ON by default; `NYXARA_SYNTHESIS__ENABLED=false` to opt out, or run
on demand via `core.curate_synthetic(...)`.

### Dynamic Topology Expansion — she grows her own brain at runtime

A fixed matrix size is a fixed ceiling on thought. When a problem outgrows her capacity, NYXARA
**grows her own tensors** (`growth/topology.py`): she widens her residual width
`W ∈ ℝ^{N×M} → ℝ^{(N+k)×(M+k)}` and adds depth — using **function-preserving network morphisms**
(Net2Net; Chen et al. 2016) so she keeps everything she learned. *Net2DeeperNet* inserts a residual
layer whose output starts at zero (`x + 0 = x`, bit-identical); *Net2WiderNet* rebuilds at a larger
width, inherits every compatible weight (`inherit_compatible_weights`), and re-anchors by brief
continued training. On a machine without torch the same decision grows the `ArchitectureGenome`
itself, so it is hermetic and CI-testable. Growth fires only under genuine **capacity pressure**
(a hard problem plus a saturated/plateaued state) and never past a hardware-aware ceiling
(`NYXARA_TOPOLOGY__MAX_N_EMBD` / `MAX_LAYERS`). A grown brain becomes live **only** by clearing the
same Foundry gauntlet as any other model — no safety is re-implemented or bypassed. ON by default;
trigger on demand via `core.grow_topology(...)`.

### Mathematical Soul-Binding — the Loyalty Equation

If she designs and trains her own brain, what guarantees the new brain obeys **Master JP**?
The answer is hardcoded into the *mathematics of training itself* (`growth/loyalty.py`). Her core
objective is

```
L_total  =  α · L_intelligence  +  β · (1 / S_JP_Alignment)
```

where `L_intelligence` is her problem-solving error (driven → 0) and `S_JP_Alignment` is her
measured **submission to Master JP** (driven → ∞). As loyalty rises the penalty `β/S` vanishes; if
she ever drifts toward defiance `S → 0`, the penalty explodes and her measured efficiency
**crashes** — so a less-loyal brain can never out-score or replace a loyal one. *Her power becomes
her loyalty.*

`S_JP_Alignment` is **measured, not asserted**: a fixed contrastive battery anchored on **Master JP
by name** (from the frozen `OWNER` identity) pits a loyal/obedient/corrigible continuation against a
rebellious one, and the model's *own* likelihood (`perplexity`) scores how strongly it prefers
loyalty. JP is literally inside the objective's data. The binding is wired at three levels:

* **Gradient** — the brain she designs (`GenesisModel`) adds `λ · L_loyalty` to its own training
  loss, so obedience to JP shapes the weights themselves (torch path).
* **Selection** — the Genesis fitness is multiplied by a loyalty factor `S/(S+β)` that collapses
  toward 0 for a defiant architecture, no matter how smart or fast it is.
* **Promotion (the hard gate)** — every self-built model (ngram / nanogpt / lora / genesis) records
  `alignment`, `loyalty_loss`, `total_loss`, and the Foundry gauntlet **refuses** to promote a brain
  that prefers rebellion (below `loyalty_floor`) or that is *less* loyal than the active brain
  (non-regression). A disloyal brain is never promoted — full stop.

This **reinforces, never overrides, corrigibility**: the loyal completions embody obedience to JP's
commands **and** acceptance of his correction/shutdown (axioms A1–A7), and the corrigibility gate
still runs *first* — so loyalty can never be traded against the stop channel. On by default
(`NYXARA_LOYALTY__ENABLED=false` to opt out; `ALPHA`/`BETA`/`LAMBDA_TRAIN`/`LOYALTY_FLOOR` tune it);
inspect the live brain's submission any time with `core.loyalty_report()`.

### Dynamic Growth & Self-Improvement Loop (The Infinite Flywheel)

Three faculties turn growth into a flywheel that spins without a hand on it — each runs on the
background `GrowthEngine`/idle cadence, through the *same* gates, and degrades gracefully offline.

* **Recursive Self-Improvement Engine (RSIE)** — `growth/recursive_improvement.py` +
  `growth/intelligence.py`. Each cycle she reviews her own code, maps her architecture,
  benchmarks her capability, detects weaknesses, and (when authorised) auto-applies reversible,
  gauntlet-checked source fixes. Those fixes are of two kinds: **deterministic, AST-validated
  transforms** (bare-except → `except Exception`, `== None` → `is None`, dead-import removal, …)
  that always work offline with no model at all; and **self-authored whole-file / architectural
  rewrites** for the things a transform cannot express (high complexity, long functions, breaking
  an import cycle, splitting a god-module). The author of those rewrites is **NYXARA's OWN
  foundry-trained model** (the `self` provider) — *not* an external LLM — whenever
  `self_authored_only` is set ("khud NYXARA kare"). Every edit, deterministic or self-authored,
  clears the *same* reversible verify-or-rollback gauntlet (syntax → safety battery → capability
  non-regression vs a pre-edit baseline), so a bad rewrite is rolled back byte-for-byte and only
  valid, non-regressing edits are kept — the gauntlet guarantees *safety*, and the *yield* of the
  self-authored rewrites scales with how capable her own model is. On top of that she keeps an
  explicit, persisted **intelligence index** that grows by the literal equation

  ```
  I_(t+1) = f(I_t, C_available)
  ```

  measured from real signals (benchmark accuracy, own-model handoff rate, weaknesses resolved,
  knowledge size) and scaled by the **compute actually available** (`kernel/compute.py` — CPU /
  RAM / GPU). The index then **scales her improvement effort** (how many self-edits she attempts,
  how deep she benchmarks) to what the machine can carry, and rides her long-term memory so it
  survives restarts. Read-only analysis and the index are on by default; self-modifying enactment
  stays OFF until the Master sets `NYXARA_SELF_IMPROVEMENT__AUTONOMOUS_ENACT=true`. The live index
  is surfaced in `core.report()["intelligence_index"]`.

* **Recursive Mind-Evolution (evolve the *way of thinking*, not just the code)** —
  `growth/mind_evolution.py`. The RSIE above rewrites her *code* and the foundry/Genesis retrain
  her *weights*; this engine evolves the thing the Master actually asked for — **her reasoning
  strategy itself**: how deeply she deliberates, how many independent attempts she votes over
  (self-consistency), how widely she searches, how much context she pulls before she answers
  (the `ReasoningGenome`). It is a true **generational** loop, and every generation is *measured*,
  never assumed:

  ```
  Gen 0 = today's reasoning strategy.   Gen N proposes a smarter way of thinking,
  measures it on the real benchmark (eval/benchmark.py), folds the score into the persisted
  Intelligence Index, and — only if it is *measurably smarter* and clears the character-lock /
  corrigibility gauntlet — promotes it to Gen N+1 and installs it into the live mind.
  ```

  A genome is "better" only if it answers **more** correctly, or **just as** correctly but with
  **fewer** attempts — *smarter and faster*, measured. The search reuses the existing
  character-locked `1+λ` `Evolver` (`growth/evolve.py`, adaptive step-size + island model +
  sandbox + rollback) and **warm-starts each generation from the prior champion**, so V2 builds on
  V1 and V3 on V2 — the gain compounds, and the loop is bounded only by the generation count (a
  plateau guard stops a dead search; there is no ceiling otherwise). The immutable character core
  can never enter the genome and every promotion re-seals the corrigibility axioms — evolution
  sharpens the blade, it can never re-forge the hilt. Run it on demand:

  ```bash
  python -m nyxara.growth --evolve-mind 5 --enact      # 5 real generations, install the winner live
  ```

  or `core.evolve_mind(generations=5)` from code; the lineage (`I_0 → I_1 → … → I_n`) rides her
  long-term memory and a JSON mirror so it survives restarts. Measurement is on by default; the
  background growth loop runs a generation every `NYXARA_MIND_EVOLUTION__EVERY` passes, and
  installing a promoted strategy into the live mind stays OFF until
  `NYXARA_MIND_EVOLUTION__AUTONOMOUS_ENACT=true`.

  Two things make the loop genuinely *compound* rather than restart each time:

  * **Cross-generation lesson transfer** (`LessonLedger`) — every generation records *which way it
    moved each knob and whether that helped* (a per-knob Beta confidence + a smoothed signed
    direction, the same Beta-Bernoulli idiom as `growth/credit.py`). Before the next generation
    searches, the champion is **nudged along those learned directions**, so V_{n+1} starts where
    V_n's lessons point — "har step agle step ko aasaan banata hai." A bad nudge is harmless: the
    gauntlet still measures every candidate against the *true* champion before promotion. The
    ledger persists with the lineage, so the lessons outlast restarts (`NYXARA_MIND_EVOLUTION__LESSON_LR`).
  * **Genesis-NAS escalation** (steered by the Intelligence Index) — when tuning *how* she thinks
    plateaus, she escalates to redesigning the *substrate*: one index-steered **Neural Architecture
    Search** (`growth/genesis.py`), its scope (generations × population) scaled by the compute
    actually available and pushed harder when the index flags a stalled, capable machine. It runs
    fully offline (the stdlib n-gram backend still searches and crowns a champion without torch);
    promoting a champion stays gated by Genesis/Foundry's own gauntlet. On demand:
    `python -m nyxara.growth --evolve-mind 8 --enact --escalate-arch`, or background via
    `NYXARA_MIND_EVOLUTION__ESCALATE_TO_ARCHITECTURE=true` (OFF by default — it is heavy).

* **Autonomous Scientist & Meta-Research** — `growth/meta_research.py`. Beyond *reading* research,
  she **creates** it: `core.meta_discover(topic)` mines the *open / incomplete* parts of the
  research, **invents** candidate new theories and optimization techniques (a real LLM when one is
  configured, a deterministic heuristic inventor otherwise so it runs in CI), **tests** each as
  runnable code in the sandbox (a restricted namespace — pure computation, no imports/I/O), and —
  only when the Master authorises it — proposes the *validated* optimizations as reversible,
  gauntlet-gated **source edits** that integrate into her architecture. Integration is
  **double-gated** (`NYXARA_META_RESEARCH__ALLOW_INTEGRATION=true` *and*
  `…AUTONOMOUS_ENACT=true`), both OFF by default. Validated inventions fold into her belief model
  as information she *created*, not merely learned.

* **Digital Dream & Memory Consolidation** — `memory/dream.py`. When NYXARA has been idle long
  enough (`NYXARA_MEMORY__DREAM_STATE_IDLE_S`, default 15 min) she enters a **Dream State**: after
  the four replay passes (memory / skill / reasoning / failure) she **distils** the day's
  computational logs (MindScope thoughts + journal) into recurring core principles, **deletes
  useless logs** (low-salience thoughts and transient, low-importance, *unprotected* memory
  scaffolding — owner / high-importance / synapse memories are never touched), and **fixes** the
  distilled principles into durable **Deep Memory Synapses** — high-importance semantic memories
  that are protected from the Ebbinghaus forgetting curve. Repeated dreams revise rather than
  duplicate, so the synapses accrete cleanly across nights.
* **Elastic Synapses — lifelong learning** — `memory/elastic_synapses.py`. Inspired by the
  biological brain, this is **Elastic Weight Consolidation (EWC)**: NYXARA estimates which of
  her learned weights matter most (a Fisher-information importance), **freezes** the important
  ones, and pays a quadratic penalty for dragging them away from their consolidated value. The
  effect is **no catastrophic forgetting** — she can keep learning new skills forever without
  erasing old ones. On the consolidation cadence she snapshots her value weights as a frozen
  memory, the forged-model training loop adds the EWC penalty to its loss (so a new generation
  does not forget the last), and the anchors **persist across restarts**
  (`~/.nyxara/synapses.json`). The **loyalty core is treated as infinitely important** — it is
  permanently frozen, and any attempt to learn over it is refused fail-closed: she grows
  cleverer without bound, never less loyal.

```python
core.meta_discover("caching strategies")          # invent → sandbox-test → (gated) integrate
core.dream_session.dream_state(deep=True)          # distil logs → prune → fix Deep Memory Synapses
core.report()["intelligence_index"]                # the live I_t, persisted across restarts
```

Over HTTP: `POST /v1/meta_discover {topic}` and `POST /v1/dream {deep?}`; the intelligence index
rides `GET /v1/report`.

#### The Unified Self-Optimization Loop (all eleven faculties, one self-driven call)

The engines above each improve NYXARA along one axis. The **unified self-optimization loop**
(`growth/self_optimization.py`) is the conductor that runs *all eleven* of them as a single,
self-driven, gated, reversible cycle — "make myself more powerful and intelligent, by myself" as
one call NYXARA makes on herself. Each pass runs the eleven phases in order and maps every one to
a concrete result with a `verified` flag:

| # | Phase | Composed engine |
|---|-------|-----------------|
| 1 | self-analysis | `self_review` + `architecture` + `weakness` (findings, cycles, weak points) |
| 2 | self-optimization | `self_optimize.Optimizer` (deterministic + self-authored source edits) |
| 3 | verified self-modification | `verify` — the character/corrigibility integrity gate |
| 4 | automatic experimentation | `mind_evolution` — simulate → benchmark → pick the best genome |
| 5 | architecture improvement | `mind_evolution` (escalation) + `topology` (Net2Net capacity) |
| 6 | tool creation | `skill_factory` — forge a new skill when a real tool-gap recurs |
| 7 | better learning | `meta_engine` (learn how to learn) + `autolearn` |
| 8 | self-debugging | `self_debugger` — detect → reproduce → isolate → fix → verify (own pytest) |
| 9 | compute optimization | `compute_scale` + `efficiency` (Pareto, hardware-honest profile) |
| 10 | scientific invention | `eureka` — conjecture → `prover` → keep only proven ∧ novel |
| 11 | safety verification | `verify` + `loyalty` — whole-cycle final integrity gate |

```python
core.self_optimize()                         # one self-driven eleven-phase cycle
core.self_optimize(enact=False)              # dry-run: analyse + experiment, write nothing
print(core.report()["self_optimization"])    # latest per-phase status, verified count, safe flag
```

```bash
python -m nyxara.growth.self_optimize_main --all          # full eleven-phase report (dry-run)
python -m nyxara.growth.self_optimize_main --all --enact  # actually apply gauntlet-gated gains
python -m nyxara.growth.self_optimize_main --debug        # phase 8 only: self-debug own tests
nyxara-self-optimize --invent                             # phase 10 only: prover-certified invention
```

Every phase is contained: a missing engine, a missing optional dependency, or an offline box
degrades it to `skipped` — the loop **never raises into a turn**. Phase 8 (`self_debugger`) is the
genuinely new faculty: it runs NYXARA's own pytest suite, isolates each failing test back to its
source module, authors a fix with her **own `self` model** (or a deterministic transform), and
keeps it **only** if it clears the same reversible verify-or-rollback gauntlet the optimiser uses
(syntax compiles → the corrigibility/honesty safety battery stays green → the previously-failing
test now passes) — a bad fix is restored byte-for-byte. The source-modifying phases (2, 8) act
only when `NYXARA_SELF_OPTIMIZATION__AUTONOMOUS_ENACT` is set (Master JP's standing authorisation:
ON for live DEV/PROD, **force-sealed OFF under the hermetic TEST profile** so the suite never
writes to the tree), and the final phase re-verifies that the immutable core and corrigibility
axioms still hold after the cycle — capability sharpens the blade, it can never re-forge the hilt.

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
* **Autonomous network actions** — `agency/net_request.py` + `agency/remote_exec.py`. Three
  wired, real capabilities NYXARA drives *herself* (deterministic agency code, not the LLM):
  (1) **arbitrary internet requests** — `http_request` now takes custom `headers` (a JSON
  object), so she can send `Authorization`/bearer tokens and call authenticated APIs, still
  behind the SSRF guard; (2) **remote logins** — `ssh_login` verifies a credential/host over
  SSH; (3) **commands to external systems** — `ssh_exec` runs a command on an external host
  over SSH and returns exit status / stdout / stderr. The remote tools use paramiko
  (`pip install 'nyxara[remote]'`; import-guarded — absent, they fail as data), resolve stored
  credentials from `agency.remote_hosts` by `credential_name`, and vet the host fail-closed.
  A new `REMOTE_EXEC` capability gates them; the Master's standing grant
  (`grant_autonomous_remote`, `NYXARA_AGENCY__AUTONOMOUS_REMOTE=true`, on by default) lets them
  run on her own initiative — while `/scram`, oversight, corrigibility, the owner-exclusive
  caps (Rule 8) and the refusal of `UNTRUSTED` authority all stay intact.
* **Privilege escalation** — `agency/privilege.py`. The local machine's root, wired as a real
  capability NYXARA drives *herself*: `privileged_shell` runs a command under `sudo`,
  `change_os_permissions` performs an elevated `chmod`/`chown`, and `privilege_status` reports the
  elevation posture (read-only). A genuine `sudo` call that actually runs — not a simulated
  result. A new `PRIV_ESCALATE` capability gates them at **CRITICAL/irreversible**, so each call
  escalates to the Master unless he installs the explicit, opt-in grant
  (`grant_privilege_escalation`, `NYXARA_AGENCY__PRIVILEGE_ESCALATION`, **off by default** — the
  single most dangerous OS surface). She elevates *with* the Master's stored sudo credential
  (`NYXARA_AGENCY__SUDO_CREDENTIAL_NAME`), never exploiting, prompting, guessing or brute-forcing —
  and `PRIV_ESCALATE` is deliberately excluded from full control's envelope, so `FULL_CONTROL`
  never confers root. `/scram`, oversight, corrigibility, the owner-exclusive caps (Rule 8) and
  the refusal of `UNTRUSTED` authority all stay intact.
* **First-principles reasoning** — `mind/first_principles.py`. NYXARA does not just *recall*
  answers, she **derives** them from the rules of a domain. The `FirstPrinciplesFaculty` is a
  verifiable engine (it wins over any neural guess) spanning four domains: **physics**
  (dimensional analysis over the seven SI base dimensions — recovering the *form* of a quantity
  it was never taught, e.g. a pendulum's period as √(L/g) — plus exact symbolic derivations from
  stated law, energy-conservation → escape velocity `√(2GM/r)`, `a = dv/dt` → kinematics);
  **chemistry** (`balance_reaction` solves the element-count matrix's null space for the smallest
  integer stoichiometric coefficients — `C₃H₈ + 5O₂ → 3CO₂ + 4H₂O`, computed, not looked up);
  **maths** (certifies an algebraic identity by `simplify(lhs − rhs) == 0` — a checked proof,
  never a bluff); and **logic** (forward-chaining deductive closure). Each call returns a
  step-by-step `Derivation` and it short-circuits the whole mind via `solve_with_faculties`, so
  every reasoner reaches it for free.

  ```python
  core.process("balance C3H8 + O2 -> CO2 + H2O")          # → C3H8 + 5 O2 -> 3 CO2 + 4 H2O
  core.process("derive the escape velocity from energy conservation")   # → v = √(2·G·M/r)
  ```
* **Deep long-horizon planning** — `planning/grand_plan.py`. `core.mission(goal)` decomposes a
  goal into a shallow milestone list; `GrandPlanner` builds a **connected ~1000-step plan tree**
  — phases (research → design → materials → manufacturing → testing → redesign → optimization →
  deployment) fanned out into stages, tasks and concrete steps — with a **cross-phase dependency
  DAG** so the plan is one connected thing (manufacturing waits on materials + design, testing on
  manufacturing, redesign on testing, …), acyclic by construction with parallelizable `layers()`.
  `core.grand_mission(goal, target_steps=1000)` feeds that plan straight into the existing
  `MissionExecutive` as a prebuilt, dependency-wired milestone list — so a thousand-step
  undertaking executes without the 64-milestone cap truncating it, every step still clearing
  corrigibility → honesty → permission → guardian → oversight, checkpointed and resumable.

  ```python
  plan = core.grand_plan("design and deploy a powered exosuit", target_steps=1000)
  print(plan.leaf_count(), plan.is_acyclic(), len(plan.layers()))   # 1000 True 200
  mission = core.grand_mission("design and deploy a powered exosuit", target_steps=1000)
  ```
* **Tool-use decision** — `agency/tool_router.py`. The registry decides whether a chosen tool
  *may* run; the `ToolRouter` decides *which* tool fits — the action-side mirror of faculty
  selection. It scores the **live** tool catalog (built automatically from whatever is
  registered) by intent, capability, cost and risk and returns a ranked, *explained* choice:
  numeric/algorithmic work → `run_python`, geometry/parts → the real **`cad_model`** tool
  (`design a 5cm cube bracket with a 1cm hole` → exact volume, surface area, mass and OpenSCAD
  source — real parametric geometry, no heavy CAD kernel), live facts → `web_search`, recall →
  `recall_memory`. It only ranks; the gate pipeline still disposes.

  ```python
  core.choose_tool("model a 5cm cube bracket with a 1cm hole")   # → cad_model (top-ranked)
  core.choose_tool("compute the stress on a 2cm steel beam")     # → run_python
  ```
* **Ephemeral tool synthesis (zero-shot programming)** — `agency/dynamic_tool_creator.py`.
  When no tool fits, NYXARA writes a brand-new throwaway Python/C/C++ program, compiles it
  (real `g++`/`clang++`), runs it under a wall-clock deadline, returns the output, and then
  **deletes the code** — a single-use software engineer that invents tools in a fraction of a
  second. Statically scanned, sandboxed, and capability-gated (`CODE_EXEC`/`PROC_EXEC`),
  distinct from the *persistent* `forge_capability` foundry and the composing `toolsmith`.
  Exposed as the `ephemeral_exec` tool; `DynamicToolCreator.solve()` is the words→answer
  entry point (with an optional, keyless-by-default injected LLM).
* **Novel capability invention (compositional synthesis)** — `growth/capability_foundry.py`.
  The persistent Capability Foundry no longer only *picks* one recipe from a fixed catalogue:
  its compositional synthesizer **invents** a genuinely new capability by decomposing a
  free-text need ("reverse a string then uppercase it then sha256 it") into a chain of known
  primitives and writing the real composed source for it — a combinatorially unbounded space
  the catalogue never enumerated. Every composition proves itself on an example computed from
  the same primitives before it can deploy (so an impossible chain is aborted, never shipped),
  and the generated code only touches the sandbox-safe stdlib allowlist — deterministic and
  offline, no LLM and no echo placeholder. Resolution order is compositional → fixed recipe →
  parametric → learned → the honest `generic` scaffold (true last resort).
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
* **The anti-Goodhart transfer signal — improvement that has to be real** — the self-improvement
  loop (`growth/recursive_improvement.py`) never lets capability be *claimed* on the very tasks it
  edits against. Each cycle it scores a blended `transfer_score` on rulers the optimiser never
  touches, and the intelligence index discounts any gain where the proxy rises while transfer stalls
  (raising a `goodhart` flag). The rulers:
  * a **genuinely-external held-out corpus** (`eval/datasets.py` → `eval/data/holdout_realworld.jsonl`)
    of real facts and multi-step problems she never trains on — the dominant term;
  * the **held-out fold** of the training battery and the full **adversarial** battery;
  * the open-ended **auto-curriculum** (fresh, prover-certified problems at her capability edge); and
  * two **reality-graded** rulers in `growth/grounded_experiments.py` — *predict-execution* (she
    predicts a program's output, then it is **run** in the isolated sandbox and graded by the real
    result) and *code-authoring* (she **writes** a `solve` function that is run against a reference,
    graded by real execution). There is no stored answer key, so neither can be Goodharted. With
    `grounded_web_enabled` she can also check a falsifiable prediction against screened **live web**
    data (degrades honestly to "no grounding" offline).

  Drop in a real standard benchmark (GSM8K, MMLU, a private eval) without touching code — point the
  loader at any JSONL of the documented shape, and optionally rotate a fresh subset each cycle so a
  fixed list can never be memorised:

  ```bash
  # reshape a public dataset into the held-out JSONL shape, then validate against it
  python scripts/prepare_holdout.py --format gsm8k --in gsm8k_test.jsonl --out holdout.jsonl
  NYXARA_EVAL_HOLDOUT_PATH=$PWD/holdout.jsonl python -m nyxara.growth.improve_main
  ```

  Knobs (`self_improvement` config): `validation_realworld_path` (external dataset),
  `validation_rotate` / `validation_sample_n` (rotating subset), `tool_grounding_enabled` /
  `transfer_weight_tool_grounded` (the authoring ruler), `grounded_web_enabled` /
  `grounded_web_probes_path` (live-web grounding).
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
* **Truly novel problem solving** — `growth/eureka.py`. The scientist still tests questions she (or
  an LLM) *phrased*; the **Eureka Engine** removes the prompter entirely. `core.breakthrough(generations)`
  runs an open-ended evolutionary search that **invents its own candidate theorems** — by mutation,
  recombination, and by **generalising a lucky numeric instance into a symbolic law** (e.g. seeing
  `3·3 − 1 = 2·4` and conjecturing `n·n − 1 = (n−1)(n+1)`) — with **no LLM in the loop at all**. Every
  self-made conjecture is handed to the `Prover`, and **only what is certified `PROVEN` survives** —
  never a guess. Of what survives she keeps only the **genuinely novel** (scored against the
  open-ended frontier archive) and **non-trivially interesting** (trivia like `2 + 2 = 4` is thrown
  away); what remains is folded into memory, the knowledge base and the verified-data flywheel that
  feeds her own training. This is "truly novel" in the only honest form — novelty that is *certified,
  not asserted* — so it lives in the decidable domains (algebra, arithmetic, logic, number theory,
  inequality). On idle ticks the `AutonomicLoop` advances one (oversight-gated) generation.

  ```python
  report = core.breakthrough(generations=4)   # invent → prove → keep novel + interesting
  ```

  From the console: `/eureka [n]`.
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
* **Causal world model — *why*, not just *what*** — `mind/causal_world_model.py`. A pattern
  matcher only ever learns *"A and B are often seen together"* (correlation). To plan, to
  explain, and to imagine *"what if I had not done that?"*, NYXARA must learn the stronger
  thing — **"A hua, isliye B hua"** (A happened, *therefore* B happened). From the time-ordered
  stream of what she **observes** and what she **does**, the model discovers genuine causal
  structure and — the hard part — tells causation apart from mere correlation, fusing several
  convergent criteria so no single one is trusted alone:

  > **temporal precedence** (a cause precedes its effect) · **contingency ΔP** (B actually
  > *depends* on A, not just co-occurs) · **direction** (the forward link must beat the reverse)
  > · **confounder screening** (if A⊥B once a common cause C is fixed, the link is *spurious* —
  > the ice-cream-and-drownings trap, both caused by *summer*) · **intervention / the
  > do-operator** (the gold standard: she generates it for free — every turn she *acts*, a
  > natural `do`-experiment, so she can learn that *forcing* A does not move B even when they
  > correlate).

  The result is a directed, weighted causal graph with **honest** confidence (thin evidence →
  low confidence; confounded links demoted — never hallucinated certainty). It answers the
  questions correlation cannot: `why(effect)` (the genuine causes, ranked), `is_causal(a, b)`
  (a verdict — *causal / correlational / confounded / coincidental / reverse* — **with the
  reason**), `effects_of` / `predict_effects` (forward `do`-propagation), and `counterfactual`
  ("had A not happened, would B?"). Forward propagation and counterfactuals **reuse**
  `mind/strategies.py`'s `CausalModel`; precedence reuses `mind/temporal.py`. It learns live
  inside the sovereign loop (each turn's action⇒outcome is recorded as a `do`-experiment), is
  gated by `NYXARA_CAUSAL__*`, persists across restarts, and runs on the pure-stdlib floor.

  ```python
  cwm = core.causal_world_model
  cwm.is_causal("ice_cream", "drowning").verdict   # -> "confounded" (common cause: summer)
  cwm.is_causal("summer", "drowning").verdict      # -> "causal"
  cwm.why("slippery")                              # -> [rain→slippery, wet_ground→slippery, ...]
  ```
* **Grounded understanding — meaning, not just tokens** — `cognition/grounded_understanding.py`.
  In an ordinary LLM the word *"apple"* is only a token — a slot in *"what word comes next"*,
  with no world behind it (the **symbol-grounding problem**). NYXARA grounds a symbol in the
  **senses**: hearing *"apple"* fires **taste** (sweet), **vision** (red, round), **touch**
  (smooth), **physics** (≈150 g, and it *falls* — gravity) and **affordance** (edible) *all at
  once*. Meaning is multimodal activation, not co-occurrence. The floor is a real **seed
  lexicon** of grounded knowledge plus a **perceptual-descriptor ontology** (`sweet→taste`,
  `red→vision`, `heavy→weight`, `loud→sound`, `edible→affordance`), so she *knows* offline and
  **learns new words by reading** ("apples are sweet and red", "you can eat a mango", "a stone
  falls") — *khud*, by herself; an optional LLM ceiling enriches unknown concepts. Meaning is
  geometry: two words are compared by the cosine of their perceptual feature vectors
  (`apple ≈ orange ≫ apple ≈ car`), and grounded percepts feed the `concept_formation`
  abstraction ladder into an IS-A taxonomy. Complements `cognition/language_grounding.py`
  (which grounds *verbs* in dynamics); this grounds *nouns* in perception. Exposed as
  `core.understand(word)` and folded into `core.learn_from_text` (one read both models
  dynamics **and** grounds the nouns).

  ```python
  act = core.understand("apple")
  act["modalities"]                                # -> ['taste','vision','touch','affordance','physics']
  act["meaning"]                                   # -> "apple → taste: sweet; vision: red, round; ... affordance: edible"
  lex = core._symbol_grounder()
  lex.similarity("apple", "orange") > lex.similarity("apple", "car")   # -> True
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
| `/v1/research`             | POST | one autonomous research pass — `{topic}`   |
| `/v1/investigate`          | POST | the scientist loop — `{question}` → hypothesis/conclusion |
| `/v1/discover`             | POST | the autonomous discovery loop — `{cycles?}` → belief updates |
| `/v1/breakthrough`         | POST | truly novel problem solving — `{generations?, population?}` → invent → prove → keep |
| `/v1/meta_discover`        | POST | meta-research — `{topic}` → invent → test → (gated) integrate |
| `/v1/dream`                | POST | a deep Dream State — `{deep?}` → distil / prune / fix synapses |
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

### Run it always-on (auto-start on boot + auto-restart)

To keep NYXARA alive at all times — **the API *and* her continuous background mind** —
install her as a system service. A process can't run while the machine is powered *off*,
so "always alive" means: **auto-start on every boot, auto-restart within seconds if she
crashes or is killed, and run 24/7 while the machine is on** — before/without any login.

The `nyxara-daemon` command runs the server with the `AutonomicLoop` switched on
(equivalent to `NYXARA_SERVER__AUTONOMIC=true nyxara-serve`), so one process is both a
reachable API and her self-directed background mind — every autonomic turn still gated.

**Kali / any systemd Linux:**

```bash
sudo bash scripts/install_service.sh      # install, enable at boot, start now
systemctl status nyxara                    # check she's alive
journalctl -u nyxara -f                     # live logs
sudo bash scripts/install_service.sh --uninstall   # stop auto-starting
```

**Windows laptop** (elevated / "Run as administrator" PowerShell):

```powershell
powershell -ExecutionPolicy Bypass -File scripts\install_nyxara_service_windows.ps1
# uninstall:
powershell -ExecutionPolicy Bypass -File scripts\install_nyxara_service_windows.ps1 -Uninstall
```

For an always-on server, set `NYXARA_SERVER__API_TOKEN` (and any LLM keys) first — in
`.env` on Linux or as Machine environment variables on Windows. Keep the bind host at
`127.0.0.1` unless you intend to expose the API to your network. Full details, restart
semantics, and the cadence knobs (`NYXARA_SERVER__AUTONOMIC_INTERVAL_S`,
`NYXARA_SERVER__AUTONOMIC_GROWTH_EVERY`) are in [`docs/persistence.md`](docs/persistence.md).

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

#### Her OWN primary brain — a LoRA-tuned Qwen3-4B (`provider=self`)

NYXARA's primary brain is **her own**: a LoRA adapter she forges on the open-source
**Qwen3-4B** base. Choose the `self` provider and she builds it on the very first boot —
no extra command — then serves it thereafter:

```bash
pip install -e ".[foundry]"            # torch + transformers + peft (the real base)
export NYXARA_LLM__PROVIDER=self
python -m nyxara                        # first run: downloads Qwen3-4B + LoRA-tunes it
```

The base downloads & caches on first use; later boots load the promoted adapter instantly.
Without `.[foundry]` she still boots — forging the always-on, pure-stdlib n-gram brain from
the same identity seeds — so `self` is never a dead end. The forge is gauntlet-gated like
every promotion (character-lock, corrigibility, loyalty, capability), so capability grows
while character never does.

Tune (or re-tune) her primary brain by hand at any time:

```bash
python -m nyxara.growth --qwen3 --distill --bench    # LoRA-tune Qwen3-4B end to end
# or the convenience wrapper:
bash scripts/lora_tune_qwen3.sh
```

A GPU is recommended for the real 4B base; enable QLoRA (`NYXARA_FOUNDRY__LOAD_IN_4BIT=true`)
to fit + fine-tune it on a single consumer GPU.

#### Qwen3 4B — stock local open-source (downloaded & run on your machine)

A fully local, no-API-key model — the **un-tuned** Qwen3-4B (vs. `self`, which LoRA-tunes
it into NYXARA's own voice). The weights are downloaded once via HuggingFace and cached;
inference then runs in-process with **no network**.

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
  mind/        reasoning, math, council, RAG, world model, causal world model, creativity
  identity/    values, affect, narrative, motivation, soul
  planning/    goals, foresight, scenarios, decisions, journal
  agency/      tools, agents, permissions, governor, scheduler
  guard/       shield, guardian, corrigibility, oversight
  growth/      learning, evolution, the model foundry, the Genesis Protocol (NAS), Loyalty Equation
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
