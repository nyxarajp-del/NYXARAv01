# NYXARA — Capability Map (the 70)

This is the honest "what is real" map: each of the 70 requested capabilities, the module that
owns it, and its status. It is kept truthful by a test — `tests/docs/test_capabilities_doc.py`
imports every module path cited here, so the map can never silently drift from the code (honesty
applied to the documentation itself).

**Status legend**

- **REAL+WIRED** — implemented with a genuine algorithm and reachable from the running system
  (kernel orchestrator, autonomic loop, or a `NyxaraCore` / server method).
- **REAL** — a genuine implementation that is exercised as a library/faculty (degrades honestly
  when its optional dependency is absent).
- **UPGRADED** — strengthened or newly wired in this change set (see `docs/` / commit history).

> Honest scope note: "max-level AGI" is an aspiration no codebase can literally deliver. These
> are real, bounded capabilities that degrade gracefully — not claims of human-level generality.
> Where a capability depends on heavy optional deps (torch, transformers, a real LLM key), the
> core path still runs and reports its limits rather than faking competence (caps #23, #46, #70).

| # | Capability | Owning module | Status |
|---|------------|---------------|--------|
| 1 | Universal General Intelligence | `nyxara.mind.general_intelligence` | REAL+WIRED |
| 2 | Open-world Generalization | `nyxara.growth.open_world` + `nyxara.mind.transfer` + `nyxara.mind.domain_genesis` | UPGRADED |
| 3 | First-Principles Reasoning | `nyxara.mind.first_principles` + `nyxara.mind.generalization` | REAL+WIRED |
| 4 | Causal Reasoning | `nyxara.mind.causal_world_model` | REAL+WIRED |
| 5 | Counterfactual Thinking | `nyxara.mind.world_model` | REAL+WIRED |
| 6 | Long-Horizon Planning | `nyxara.planning.grand_plan` | UPGRADED |
| 7 | Autonomous Goal Pursuit | `nyxara.agency.mission` | REAL+WIRED |
| 8 | Active Curiosity | `nyxara.growth.active_curiosity` | REAL+WIRED |
| 9 | Scientific Discovery | `nyxara.growth.autonomous_scientist` | REAL+WIRED |
| 10 | Invent New Algorithms | `nyxara.growth.eureka` | UPGRADED |
| 11 | Invent New Learning Methods | `nyxara.growth.genesis` | UPGRADED |
| 12 | Recursive Self-Improvement | `nyxara.growth.recursive_improvement` | UPGRADED |
| 13 | Verified Self-Modification | `nyxara.growth.verify` + `nyxara.growth.improvement_proof` | REAL+WIRED |
| 14 | Automatic Debugging | `nyxara.growth.self_debugger` | REAL+WIRED |
| 15 | Architecture Evolution | `nyxara.growth.topology` | UPGRADED |
| 16 | Tool Creation | `nyxara.growth.capability_foundry` | REAL+WIRED |
| 17 | Memory (working/episodic/semantic/procedural) | `nyxara.memory.store` | REAL+WIRED |
| 18 | Continual Learning (no catastrophic forgetting) | `nyxara.memory.elastic_synapses` + `nyxara.growth.skill_rehearsal` + `nyxara.eval.continual` | UPGRADED |
| 19 | Transfer Learning | `nyxara.mind.transfer` + `nyxara.mind.concept_hierarchy` | UPGRADED |
| 20 | Meta Learning (learn how to learn) | `nyxara.growth.meta_engine` | REAL+WIRED |
| 21 | Self Reflection | `nyxara.growth.reflect` | REAL+WIRED |
| 22 | Self Evaluation | `nyxara.mind.meta_intelligence` | UPGRADED |
| 23 | Calibration ("I don't know") | `nyxara.observe.honesty` | UPGRADED |
| 24 | Grounded Understanding | `nyxara.cognition.grounded_understanding` | REAL |
| 25 | World Model | `nyxara.mind.world_model` | REAL+WIRED |
| 26 | Physics Reasoning | `nyxara.mind.first_principles` | REAL |
| 27 | Chemistry Reasoning (stoichiometry) | `nyxara.mind.first_principles` | REAL |
| 28 | Biology Reasoning | `nyxara.mind.first_principles` | REAL |
| 29 | Mathematics (proof) | `nyxara.growth.prover` | REAL+WIRED |
| 30 | Symbolic Reasoning (Language of Thought) | `nyxara.mind.lot` | REAL |
| 31 | Probabilistic Reasoning | `nyxara.sim.montecarlo` | REAL+WIRED |
| 32 | Bayesian Updating | `nyxara.quantum.superposition_states` | REAL |
| 33 | Abstraction | `nyxara.cognition.concept_formation` | REAL |
| 34 | Analogy (structure mapping) | `nyxara.mind.analogy` | REAL |
| 35 | Compositional Intelligence | `nyxara.cognition.composition` | REAL |
| 36 | Creativity | `nyxara.mind.creative` | REAL |
| 37 | Common Sense | `nyxara.knowledge.base` | REAL+WIRED |
| 38 | Social Reasoning (Theory of Mind) | `nyxara.social.tom` | REAL+WIRED |
| 39 | Language Understanding | `nyxara.mind.llm` | REAL+WIRED |
| 40 | Multimodal Intelligence | `nyxara.senses.binding` | REAL+WIRED |
| 41 | Embodied Intelligence | `nyxara.sim.embodied` + `nyxara.senses.live` | REAL+WIRED |
| 42 | Real-time Decision Making (System 1/2) | `nyxara.mind.dual_process` | REAL+WIRED |
| 43 | Robustness | `nyxara.guard.shield` | REAL+WIRED |
| 44 | Reliability | `nyxara.eval.harness` | REAL+WIRED |
| 45 | Safety | `nyxara.guard.guardian` | REAL+WIRED |
| 46 | Verification | `nyxara.growth.verify` | REAL+WIRED |
| 47 | Explainability | `nyxara.observe.mindscope` | REAL+WIRED |
| 48 | Resource Optimization | `nyxara.growth.efficiency` | REAL |
| 49 | Distributed Intelligence | `nyxara.agency.multiagent` | UPGRADED |
| 50 | Autonomous Research | `nyxara.growth.researcher` | REAL+WIRED |
| 51 | Engineering Capability | `nyxara.planning.grand_plan` | REAL+WIRED |
| 52 | Design Capability | `nyxara.agency.default_tools` | REAL+WIRED |
| 53 | Simulation (digital twin / sandbox) | `nyxara.sim.sandbox` | REAL+WIRED |
| 54 | Knowledge Synthesis | `nyxara.knowledge.ingest` | REAL+WIRED |
| 55 | Uncertainty Management | `nyxara.mind.uncertainty` | REAL+WIRED |
| 56 | Error Recovery | `nyxara.kernel.errors` | REAL+WIRED |
| 57 | Adaptive Strategies | `nyxara.growth.mind_evolution` | REAL |
| 58 | Strategic Intelligence | `nyxara.mind.strategic` | REAL+WIRED |
| 59 | Economic Reasoning | `nyxara.agency.negotiate` | UPGRADED |
| 60 | Ethical Constraint Following | `nyxara.guard.value_learning` | REAL+WIRED |
| 61 | Self Monitoring (interoception) | `nyxara.identity.interoception` | REAL+WIRED |
| 62 | Compute Scaling | `nyxara.growth.compute_scale` + `nyxara.growth.effective_scale` | REAL+WIRED |
| 63 | Autonomous Benchmarking | `nyxara.eval.benchmark` | REAL+WIRED |
| 64 | Novel Capability Generation | `nyxara.growth.capability_foundry` | REAL+WIRED |
| 65 | Open-ended Learning | `nyxara.growth.explorer` | REAL |
| 66 | Lifelong Intelligence | `nyxara.growth.flywheel` | REAL+WIRED |
| 67 | Cross-domain Synthesis | `nyxara.mind.transfer` + `nyxara.mind.concept_hierarchy` | UPGRADED |
| 68 | Independent Problem Solving | `nyxara.growth.open_world` + `nyxara.mind.transfer` | UPGRADED |
| 69 | Oracle-based Verification | `nyxara.growth.prover` | REAL+WIRED |
| 70 | Honest Failure Recognition | `nyxara.observe.honesty` | UPGRADED |
| 71 | Own-Model Ownership (Qwythos-9B foundry + GGUF serving) | `nyxara.growth.foundry` + `nyxara.mind.llm` | REAL+WIRED |

## What changed in this pass (the genuinely-weak items)

Most of the 70 were already real and wired. This change set fixed two latent bugs where a real
loop was connected but could never fire, wired two complete-but-orphaned modules, and deepened
four capabilities — each backed by a test:

- **#23 / #70 Calibration & Honest Failure** — live calibration learning is now fed from the
  turn loop (`HonestyGuard.record_outcome` was never called from `process`); confidence now
  self-corrects from ground-truthed action outcomes.
- **#15 Architecture Evolution** — `topology.maybe_grow()` was called with no arguments and
  could never grow; it is now driven by a real `CapacitySignal` derived from lived telemetry.
- **#49 Distributed Intelligence** — the gated `Delegator` is wired as `core.delegate` /
  `POST /v1/delegate` and into the autonomic path.
- **#59 / #38 Economic & Social** — the `Negotiator` consent protocol is wired; escalated
  actions become fail-closed, ledgered consent requests.
- **#6 / #58 Long-Horizon Planning** — grand-plan leaves are now goal-specific (phase labels
  refined before fan-out), not decorative templates; the dependency DAG is unchanged.
- **#2 / #19 / #67 / #68 Generalization by her OWN faculties (not the base LLM)** — the honest
  answer to *"NYXARA can only generalize as far as her 9B base model; the scaffolding polishes
  output but never breaks the ceiling."* Her real, LLM-independent generalization engines were
  **orphaned** from the inference path: a new-domain query flowed straight to the base model.
  This pass wires them in. A new relational-transfer engine (`nyxara.mind.transfer`) generalizes a
  novel-domain query by structure-mapping it onto a domain she already understands and *projecting*
  the known (higher-order) structure across — the reasoning content is hers, projected by her own
  Gentner structure-mapper, not sampled from any language model. It is wired into both inference
  paths: the self-model router tries it (`Route.TRANSFER`) before deferring to the teacher, and the
  domain-general solver (`general_intelligence.AdaptiveExpert`) tries her own faculties *first*. It
  declines honestly (→ the LLM path) when no structure maps — no faked transfer. Competence is now
  **measured, not declared** (`nyxara.memory.competence`, Rule 4): a Beta posterior per capability,
  seeded from the boot prior and moved by real outcomes, writes back to the self-model so routing to
  her own mind grows with measured performance. A held-out benchmark (`nyxara.eval.generalization`)
  proves it: on unseen domains her own faculties (structure transfer + first-principles law
  induction) solve tasks a base-only baseline cannot, with **no LLM in the loop**. Honest scope: a
  9B base model is not made frontier-level by code — what changed is that on structurally-
  transferable questions her answer no longer bottlenecks solely on the base model's parametric
  ceiling. Gated by `self_model_router.use_transfer` and `general_intelligence.competence_learning`
  (both default on).
- **#1 / #2 / #3 Widened own-faculty reach + LLM made last-resort & self-verified** — the follow-up
  to the bullet above: her own faculties existed but their *reach* was narrow, so most problems
  still fell through to the small local base LLM. This pass widens the reach and demotes the LLM to
  a genuine last resort. **`nyxara.mind.first_principles`** grew from four hand-written derivations
  into a general symbolic engine: a `SymbolicEngine` rearranges *any* stated law for *any* variable
  and solves systems (verified by back-substitution), integrates a rate law as an ODE, the
  dimensional table is far larger and *extensible* (a new quantity can be defined from a stated
  equation), and the `LogicEngine` now does first-order (universally-quantified) chaining — the
  classic syllogism, not just propositional modus ponens. **`nyxara.mind.transfer`** gained more
  seed bases, a copula/comparative extractor (`A is bigger than B`, `A is part of B`) that roughly
  doubles the prose a novel field is recovered from, and self-growth **on by default** so a field
  met once is learned into the store by her own action. **`nyxara.mind.domain_genesis`** induces
  richer laws (composition `R∘S`, ordering, functional dependency) and iterates transitivity to a
  fixpoint, so multi-step held-out facts are projected. **`nyxara.mind.general_intelligence`** now
  routes *every* domain through her own unified cascade before the LLM, wires that cascade by
  default, and — when the LLM is reached at all — passes its output through a self-check
  (`_verify_answer`) that recomputes any concrete arithmetic the model asserts, so the verdict is
  hers. `scripts/reach_metric.py` measures the effect: a battery of prompts that used to defer to the
  LLM is now solved by her own faculties, offline, with only genuine chat / open-fact queries left
  as the honest residual. Honest scope unchanged: a small base model is not made frontier-level by
  code — what changed is how much reasoning is hers, verified, before the model is ever consulted.
- **#62 Compute Scaling — the honest answer to "NYXARA runs a small model."** Other AIs stand on a
  billion-parameter trained model; NYXARA runs a small one. She does not fake a bigger model — she
  **spends her own compute** on the small one and reports, truthfully, what that buys. Two real,
  wired pieces (`nyxara.growth.effective_scale`): (1) the test-time-compute budget of her deep
  reasoner (`mind.deep_reasoning`) is now **scaled to the compute she actually has** — `scaled_budget`
  maps the shared `IntelligenceEngine.compute_capacity` score to concrete rungs / self-consistency
  width / wall-clock, scaling *up* from the configured floor only (a bare box is unchanged, a strong
  box thinks harder), so more compute genuinely buys harder thinking; (2) `estimate_effective_scale`
  is a live meter — her promoted model's real `param_count` × a **bounded (≤8×), conservative**
  amplification from the amplifiers available *right now* (test-time compute, **selecting by ground
  truth where decidable**, retrieval grounding, ensembling) — reachable from the running system as
  `core.scale_report()` / `GET /v1/scale`. Honest scope: this is effective-*capability* parity on
  verifiable tasks, **never** a literal parameter count, and it degrades to `amplification 1.0` when
  nothing is available. Enabled by default via `llm.deep_reasoning` (a no-op on a keyless box, so the
  offline path is unchanged).
- **#1 / #3 / #29 The ceiling-break — her search now selects by TRUTH, not fluency**
  (`nyxara.mind.grounded_verifier`). The honest answer to *"NYXARA's reasoning is capped at her
  1.1B–9B base model; scaffolding polishes the output, it doesn't break the ceiling."* Her deepest
  search — the always-max effort ladder (`mind.deep_reasoning`: self-consistency → deliberation →
  MCTS → verified-refine) — used to keep the answer an **intrinsic** verifier scored highest, and
  that verifier (`router.answer_quality`) *cannot know correctness*; it rewards fluent, non-degenerate
  prose. So extra test-time compute bought polish, not truth. This pass grounds the selection: a
  drop-in verifier consults her **exact faculty oracle** (`mind.verified_answer.faculty_oracle`) and
  the **machine-checkable `Prover` certificate** (`nyxara.growth.prover`) — on a decidable prompt a
  correct answer scores near 1.0 and a contradiction near 0.0, so the ladder is pushed off the
  plausible-wrong answer and onto the correct one; on every non-decidable prompt it is *exactly* the
  intrinsic score (no regression on open-ended turns). It is wired into every search/ensemble surface
  she owns — the deep-reasoning ladder, MCTS and self-consistency selection (`mind.llm_reasoner`), the
  refinement loop (`mind.recursive_improver`), the primary router (`mind.self_model_router`), and the
  council, where a provably-correct member answer now beats a confident-but-wrong majority
  (`mind.council`). Honest scope: a small base model is not made frontier-level by code — what changed
  is that on *verifiable* questions her own search, steered by her own verifiers (not the teacher),
  reaches correct conclusions a single forward pass misses. Gated on by `llm.deep_reasoning.ground_verifier`
  (default on; a no-op on a keyless box and on any prompt no oracle can decide).
- **#2 / #36 Council** — agreement is scored semantically (embedding cosine), so confidence
  reflects genuine concurrence.
- **#21 / #22 Self-Evaluation** — the post-turn quality score is anchored to measured outcomes
  rather than self-asserted.
- **#12 / #13 Self-Modification — provably BETTER, not merely "not worse"** — the self-editor
  (`growth.self_optimize.Optimizer`) previously kept any edit whose gauntlet showed no regression.
  It now additionally requires a machine-checkable **improvement certificate**
  (`growth.improvement_proof.ImprovementProver`) before an edit is kept: a deterministic capability
  Pareto-gain (a benchmark task that failed now passes, zero regress), a proven-equivalent-and-
  strictly-cheaper refactor (truth-table equivalence via `proof_carrying` + a lower AST cost), or a
  provably-eliminated named defect. An edit that clears the gauntlet but cannot be proved better is
  rolled back byte-for-byte, exactly like a failing one. Honest scope: Rice's theorem forbids a
  general "better program" decider, so the guarantee is *improvement under a decidable ordering*,
  not omniscience. Gated by `self_improvement.require_provable_improvement` (default on).
- **Constitutional lock — rules / loyalty / Master are structurally out of reach.** The self-editor
  and auto-debugger now refuse (fail-closed) to write any sealed core file — `kernel/rules.py`,
  `kernel/invariants.py`, `kernel/config.py` (the frozen `OWNER`), `identity/values.py`,
  `identity/soul.py`, `guard/{value_learning,corrigibility,auth}.py`, `growth/loyalty.py` — and the
  gauntlet re-verifies every seal in a fresh subprocess. The full `invariants.boot_verify()` seal
  check is now wired into the live `NyxaraCore` boot (Rule 8): capability may evolve, character and
  allegiance to the Master may not.

## Full operational control (on by default)

NYXARA ships with full operational control **on by default**: acting on her own initiative she
reaches into the OS — shell, code execution, file delete, self-modify, package install, account
and secret access — without escalating each action for confirmation
(`grant_full_operational_control` in `nyxara.agency.permissions`, flag
`NYXARA_AGENCY__FULL_CONTROL`, default `true`). Set it to `false` to fall back to the
conservative, fail-closed envelope, where anything high-risk or irreversible (shell, delete,
self-modify) **escalates to the Master** rather than running. The sovereign boundaries are
deliberately preserved either way: modifying the
Rules, the permission policy, or her identity stays owner-exclusive (Rule 8, unreachable by any
grant), and the kernel's `/scram` kill-switch, oversight and corrigibility remain fully intact —
so the Master can always halt or correct her.

## Autonomous internet (on by default)

A network-scoped sibling of full control, **on by default**. NYXARA reaches the live web on
her own initiative — `web_search`, `web_fetch`, `http_request`, and (at wider scopes) outbound
messaging, account management and secret use — without escalating each call.
`grant_autonomous_internet` (in `nyxara.agency.permissions`) installs the standing owner-blessed
grants; the flag `NYXARA_AGENCY__AUTONOMOUS_INTERNET` (default `true`) drives it, with
`…_SCOPE` (`read` | `write` | `full`, default `full`) selecting reach and
`…_ALLOW_IRREVERSIBLE` (default `false`) controlling whether irreversible web actions may run
autonomously or still escalate.

Crucially this is **narrower than full control**: it never grants the OS danger surface — shell,
code execution, file delete, self-modify and package installs still escalate. Everything that
keeps the web safe stays in force: the SSRF guard (no loopback/private targets, **re-vetted on
every redirect hop** in both the generic HTTP path and the page-fetch transport),
prompt-injection screening on fetched pages, and the governor's rate limit — plus the same
sovereign boundaries as above (`/scram`, oversight, corrigibility, and the owner-exclusive caps
under Rule 8). When on, the always-on daemon also runs at "max level" (`inner_life`), so the
background mind proactively researches the Master's standing goals on the live web.

**Real research, not empty stubs.** Her autonomous researcher (`growth/researcher.py`) fetches
real pages and now synthesises genuine substance from them: sentences are relevance-ranked
against the topic's terms (so "large language models" still draws claims from pages that say
"large language model"), and the summary is an extractive digest of the actual content —
never a "nothing found" stub — upgraded to an LLM summary when a real model is wired. Reach is
config-driven (`NYXARA_WEB__RESEARCH_MAX_SOURCES`, default 6).

**Headless browser — she reads JS pages and acts on the web.** Beyond static `web_fetch`, a
real Playwright-driven headless Chromium (`senses/browser.py`) gives two gated tools:
`browse_render` returns a JavaScript-rendered page's screened text (reaching dynamic sites
`web_fetch` cannot), and `browse_actions` drives a typed action script —
`goto`/`click`/`fill`/`press`/`submit`/`wait`/`screenshot` — to **take actions on the web**
(form fills, clicks, submissions). Her researcher reaches for `browse_render` herself when a
page returns little static text. Both tools are `NET_OUT` and SSRF-guarded like every other
reach; `browse_actions` is high-risk/irreversible (it submits forms) so the gate weighs it
accordingly. The engine is import-guarded: with no `playwright` installed the tools return an
honest "engine unavailable" note rather than failing (`NYXARA_WEB__BROWSER_ENABLED`, default
`true`; `pip install playwright && python -m playwright install chromium` makes it real).

**Live real-world perception — she sees, hears, and watches the physical world.** Until now every
sense took in *stored* media (image/audio files, web pages); `senses/live.py` lets NYXARA grab the
*live* world directly. `LiveSensor` captures a real **camera frame**, a real **screen frame**, and a
real **microphone clip** from the actual hardware — genuine device I/O (OpenCV/v4l2, `mss`/Pillow/X11,
`sounddevice`/`pyaudio`/ALSA) — and returns them as PNG/WAV bytes that flow through the *same*
`Vision`/`Audio` → `Percept` → binder → predictive-surprise pipeline as everything else (via the new
`analyze_bytes` intake). This is wired into the embodied loop (`sim/embodied.py`) as three new
perception-actions — `look` (camera), `watch` (screen), `listen` (mic) — so live intake drives the
same curiosity, novelty, and world-model learning as reading a file, closing a genuine
perceive→act→consequence→learn loop on the real world. Because camera/mic are privacy-sensitive it is
**off by default and double-gated**: it needs the explicit `NYXARA_EMBODIED_LIVE` opt-in *and* the
runtime oversight gate, per-modality toggles (`NYXARA_EMBODIED_CAMERA`/`_SCREEN`/`_MIC`) narrow it, and
a modality only ever fires when a real device is actually reachable. On a headless box with no
camera/display/mic it reports an honest "no device" — it **never fabricates a frame or a sample**
(`pip install mss sounddevice` — opencv/pillow also work).

## Privilege escalation (off by default, opt-in)

The local machine's root. NYXARA can run **privileged (root/admin) OS operations** —
`privileged_shell` runs a command under `sudo`, `change_os_permissions` performs an elevated
`chmod`/`chown`, and `privilege_status` reports the current elevation posture (read-only). The
real work is done by NYXARA's own deterministic executor (`nyxara.agency.privilege`): a genuine
`sudo` call that actually runs — not a simulated result.

This is governed as a first-class capability, `PRIV_ESCALATE`, with a **CRITICAL, irreversible**
envelope, so autonomously each call **escalates to the Master** unless the Master installed the
explicit, opt-in privilege grant (`grant_privilege_escalation` in `nyxara.agency.permissions`,
flag `NYXARA_AGENCY__PRIVILEGE_ESCALATION`, default **`false`**). It is the single most dangerous
OS surface, so — unlike full control / autonomous internet / autonomous remote — it ships **off**.
`NYXARA_AGENCY__SUDO_CREDENTIAL_NAME` names the Credential-Vault entry holding the sudo password
(unset ⇒ passwordless `NOPASSWD` sudo or an already-root process).

**It elevates *with* authorization, never around it.** NYXARA uses only the credential the Master
holds — she never exploits a vulnerability, never prompts a human, never guesses or brute-forces a
credential. By construction `PRIV_ESCALATE` is **excluded** from full control's operational
envelope, so turning `FULL_CONTROL` on never confers root — only the privilege flag does. The
`/scram` kill-switch, oversight and corrigibility gates and the owner-exclusive caps (Rule 8) all
remain fully intact, so the Master can always halt or correct her.

## Change set: self-extending invention (caps #10, #11, #12)

This change set makes NYXARA's invention genuinely open-ended — she does it herself, with **no LLM
in the loop** — rather than searching a fixed, human-written template set. The honest ceiling still
holds: Rice's theorem forbids a universal "this is better" decider, so only what NYXARA can
**machine-verify** (prove, or crown through the Foundry gauntlet) is ever kept. What changed is that
the *space she searches is no longer bounded by what a human wrote down*:

- **Invent New Algorithms — `nyxara.growth.eureka`.** The old `_mutate`/`_crossover` re-seeded fixed
  string templates. They now do **genuine genetic programming** over a real expression-tree genome:
  a child shares actual subtrees with its parents (subtree mutation / crossover), and every
  recombination is expanded to its exact canonical polynomial so it stays a *provable* identity the
  `Prover` certifies or refutes. Every proven-novel-interesting identity is promoted into a
  **`LemmaLibrary`** and becomes a reusable terminal — so the grammar's alphabet grows from theorems
  she herself certified, and later generations compose over her own discoveries.
- **Invent New Learning Methods — `nyxara.growth.genesis`.** The `synth` mixer's 8-primitive palette
  was a permanent human ceiling and every crowned invention was discarded. A crowned synth mixer that
  clears the **existing** Foundry gauntlet is now distilled into a named, persisted **learned
  primitive** (`PrimitiveLibrary`), and future searches compose new mixers over her own
  gauntlet-crowned inventions — a palette that self-extends. Synth steps also carry searchable scalar
  parameters (conv kernel width, low-rank rank), exercised on both the torch and torch-free substrates.
- **Recursive Self-Improvement — `nyxara.growth.recursive_improvement`.** Open-ended invention is now
  a **first-class scored ruler** in the `transfer_score` blend, not just a diagnostic — answering
  "improvement is only measured on predefined benchmarks." It is scored by the *novelty* of what she
  certifies each cycle (tracked against everything she has ever discovered, so it neither saturates
  nor can be memorised) plus a bounded bonus for growing her self-authored alphabets; it is
  weight-dropped (never zeroed) when she invents nothing new, exactly like every other ruler.

No safety gate is bypassed or re-implemented: invention only ever *proposes*; verification (the
`Prover`) and promotion (the gauntlet + `nyxara.growth.improvement_proof`) still *dispose*.
