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
| 4 | Causal Reasoning | `nyxara.mind.causal_world_model` | UPGRADED |
| 5 | Counterfactual Thinking | `nyxara.mind.causal_world_model` + `nyxara.mind.world_model` | UPGRADED |
| 6 | Long-Horizon Planning | `nyxara.planning.grand_plan` | UPGRADED |
| 7 | Autonomous Goal Pursuit | `nyxara.agency.mission` | REAL+WIRED |
| 8 | Active Curiosity | `nyxara.growth.active_curiosity` | REAL+WIRED |
| 9 | Scientific Discovery | `nyxara.growth.autonomous_scientist` + `nyxara.growth.law_discovery` | UPGRADED |
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
| 39 | Language Understanding (PRIMARY brain now runs **on-device**: Gemma-4-E2B-it in LiteRT-LM format leads the `auto` ladder `litertlm→self→native` — every rung in-process, no cloud providers at all, in-process with no API key and no network — so an offline machine answers on a real instruct model instead of an n-gram, and it is classed among her OWN brains rather than as an external teacher) | `nyxara.mind.llm` + `nyxara.mind.litertlm_assets` | UPGRADED |
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
| 51 | Engineering Capability | `nyxara.planning.grand_plan` + `nyxara.growth.engineering_foundry` | UPGRADED |
| 52 | Design Capability | `nyxara.agency.default_tools` + `nyxara.growth.engineering_foundry` | UPGRADED |
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
| 71 | Own-Model Ownership (DistilGPT-2 LoRA foundry + in-process serving) | `nyxara.growth.foundry` + `nyxara.mind.llm` | REAL+WIRED |
| 72 | Frontier Law Discovery (invent NEW empirical/physical laws from data & self-run experiments, no LLM) | `nyxara.growth.law_discovery` | REAL+WIRED |
| 73 | Non-Algorithmic Intuition / Creative Leap (guess a candidate *before* proof, on puzzles with no training data — a portfolio of self-contained leap generators, fused + self-verified, **no LLM**) | `nyxara.mind.intuition` | UPGRADED |
| 74 | Engineering Foundry (use invented laws + real physics sims to DESIGN, multi-objectively optimise, and iteratively UPGRADE real device concepts — a portfolio optimiser over a coupled multi-physics evaluator, **no LLM**) | `nyxara.growth.engineering_foundry` | REAL+WIRED |
| 75 | First-Principles Feasibility Gate (prove physically-impossible "magic" targets — over-unity/zero-point energy, anti-gravity, time reversal — INFEASIBLE with the conservation law they break, and log them; never fake them) | `nyxara.growth.engineering_foundry` | REAL+WIRED |
| 76 | Structural Cognitive Self-Modification (rewire her OWN cognitive architecture: **invent new composite reasoning operators** over a typed SEQ/VOTE/VERIFY grammar — the "trans-logic" — reorder/prune/re-weight which operator handles which task, a bounded recursive meta-policy, continuous plasticity, and antifragile self-healing; adopt only what STRICTLY beats a held-out fold with the character core untouched, **no LLM**) | `nyxara.growth.cognitive_architect` | REAL+WIRED |
| 77 | Always-On Max Perception (continuous seeing/hearing between prompts **plus** interoception: energy VAD + wake word + sound classes bang/alarm/hum, face & body detection, motion direction, lights on/off, active-window tracking, host vitals with rising-edge alerts, new-process/file/network/port watching, device hot-plug that grows live camera channels, sensory self-health — **no LLM in the path**) | `nyxara.senses.realtime` + `nyxara.senses.system` + `nyxara.senses.watch` | REAL+WIRED |
| 78 | The Living Algorithm (Noēsis) — a self-extending abstraction library that COMPOUNDS capability-per-compute: WAKE solves a task by searching for the shortest **verified** program in a typed DSL (abstains, never bluffs); SLEEP compresses solved programs into new first-class primitives adopted only on a **strict held-out description-length win** (the language she thinks in grows); DREAM invents her own tasks. Measured, not claimed — avg solution size ↓ and compute-per-solve ↓ as the library grows. **No LLM in the loop**, character-locked, persists across restarts | `nyxara.growth.noesis` | REAL+WIRED |
| 85 | Quantum-Resistant Epistemic Cryptography — every learned fact/axiom/skill is **HMAC-SHA256 signed and chained**: a signature covers its content + source + confidence + sensitivity **and the previous entry's hash**, so the ledger is a tamper-evident hash chain — altering any past fact breaks every signature after it, and `verify_chain` finds exactly where. Symmetric HMAC-SHA256 is not broken by Shor's algorithm (quantum-resistant in the practical sense). On top of integrity, a **context-safe share/execute policy** (the digital immune system) decides — only when the signature still verifies — whether a fact may leave her mind or run, by sensitivity (public/internal/secret/executable) and calling context (external/trusted/owner); a tampered fact is quarantined. Reuses `nyxara.guard.crypto`; **no LLM** | `nyxara.growth.epistemic_crypto` | REAL |
| 84 | Multi-Agent Societal Mimicry (Internal Civilization) — the LLM-free upgrade of the role council: on a big architectural/ethical decision an internal society of **deterministic** sub-personas (Scientist, Engineer, Strategist, Critic, Security Officer, Philosopher) debates over the decision's own dimensions across rounds of **cross-examination** (each presses its strongest concern; scores update), then NYXARA synthesises a consensus she owns — residual **conflict** measured and reported (never faked as unanimous), with an **absolute safety/ethics veto** that is never out-voted. A bounded ensemble fans the six archetypes into 'hundreds of voices'. **No LLM** (the Master's *khud kare, LLM na kare*) | `nyxara.mind.internal_civilization` | REAL |
| 83 | Autonomous Synthetic Mathematics (Meta-Epistemology) — when a goal is **unprovable in her current axioms** she **invents a new axiom** and admits it only if it is **consistent** (proves no designated falsehood), **non-trivial** (the goal was unprovable before, provable after), and **generative** (also proves an independent held-out goal — a real law, not an overfit patch). The prover is a genuine **congruence-closure** decision procedure (union-find + the congruence rule), so "provable" means actually derivable; new axioms are drawn from a template bank (commutativity/associativity/identity/idempotence/distributivity) and **persist** so her mathematics compounds. Honest + bounded (a small algebra), **no LLM** | `nyxara.growth.meta_epistemology` | REAL |
| 82 | Cross-Domain Biomimetic Synesthesia — a universal pattern-transposition matrix: a pattern's scale-invariant **shape** (trend/curvature/oscillation/exp-power linearity/autocorrelation) is lifted into the shared 10,000-D HDC space (`nyxara.cognition.hyper_dimensional_vectors`), so patterns from different domains that share shape land near each other (a population curve ≈ a compound-interest curve; a heartbeat ≈ a market cycle). She finds the nearest **cross-domain** analogue and **transposes** its known functional law onto the target — adopting it (like `nyxara.mind.category_transfer`) **only when it verifiably fits** (R² past a threshold on the target's own data), else abstaining. **No LLM**; pure numeric maths | `nyxara.mind.synesthesia` | REAL |
| 81 | Holographic Memory Field — a continuous, entangled recall layer over her HDC substrate: every memory is stored as a **key⊗value binding bundled into one field hypervector**, so memories share the whole field (genuine entanglement) and recall is a single **unbind + cleanup** — associative, near-constant work over the field, not a per-chunk scan of stored text (`nyxara.cognition.hyper_dimensional_vectors`). Honest about physics: a bundle saturates, so it holds a bounded working horizon with **graceful forgetting** past capacity (spilling to the ordinary store) — instant associative blend for the live horizon, not literal infinite/zero-latency/zero-forgetting memory. **No LLM** | `nyxara.memory.holographic_field` | REAL |
| 80 | Quantum-Probabilistic Superposition Reasoning — she does not commit to one line of thought: several candidate solution paths are held at once in a real **Born-rule superposition** (`nyxara.quantum.superposition_states`), each scored by its **simulated future outcome** (a world-model / simulation rollout on top of consensus + grounding), and the state **collapses** to the single optimal path — but only when one dominates past a threshold; below it she stays superposed (`decided=False`) and abstains / deepens rather than bluffing. The live hypothesis-selection path already collapses hypotheses this way; the dedicated reasoner adds outcome-scored deliberation on demand (`/superpose`). **No LLM at the scoring/collapse layer**; advisory (the kernel still disposes) | `nyxara.mind.superposition_reasoner` | REAL+WIRED |
| 79 | Self-Evolving Dynamic Neural Architecture — the unified **demand-driven** driver: when a *specific* turn falls short of her current logic she diagnoses the *kind* of gap and fires the single best structural lever to grow a NEW neural module/pathway for it — grow topology (capacity), forge a new architecture (representation), invent a new learning rule (a stalled learner), or invent a new reasoning operator (composition) — then verifies it through the **same Foundry gauntlet** every model must pass and, when enacting, wires it live. Composes the existing organs (`nyxara.growth.topology` / `nyxara.growth.brain_forge` / `nyxara.growth.rule_synth` / `nyxara.growth.cognitive_architect`); re-implements no search/training/promotion/safety; **no LLM in the loop**; full-autonomous by config yet the oversight gate + `/scram` remain absolute (a paused mind designs+measures but never promotes) | `nyxara.growth.self_evolving` | REAL+WIRED |
| 86 | 🕸️ Topological Graph-State Persistence (SENTINEL Stage A) — the rich typed knowledge graph (entities/triples/traversal/paths) now **survives a restart**: `KnowledgeGraph.from_dict`/`load` mirror `to_dict`/`save` with full provenance fidelity (so live trust decay is preserved, not reset), and the core's `save_state`/`load_state` persist it as a `knowledge_graph.json` sidecar, merged on top of the seed relations on load. Closes a real data-loss bug where the whole accumulated graph was silently wiped to its 2 seed triples every boot. Honest scope: durable + additive **within capacity** (provenance half-life still applies, reported truthfully), not literally infinite. **No LLM** | `nyxara.memory.graph` | REAL+WIRED |
| 87 | ⚛️ Continuous Active Inference (SENTINEL Stage B) — the free-energy machinery used to run only inside a turn (advisory); now a background tick **predicts her own state every cycle**, measures per-channel surprise (prediction error) and a real **normalised Shannon entropy** over the belief state, and when uncertainty spikes acts **pre-emptively** (probes the most-informative channel via her curiosity engine) — all through the same sovereign gauntlet, oversight/`/scram`-halted like every autonomic step. Honest scope: this is *active-inference* variational/expected free energy, **not** literal thermodynamics; bounded channels, degrades to a silent no-op with none readable. **No LLM** | `nyxara.mind.active_inference_loop` | REAL+WIRED |
| 92 | ⏳ Always-Alive Maturity + Honest-Gap-Closing (SENTINEL Stage G) — two genuine deltas atop faculties that were already real (the ~1s heartbeat, background self-upgrade, causal science, and calibrated 'I don't know' all pre-existed). **(1) Elapsed-time awareness:** at the head of each turn she computes an honest, human phrasing of the gap since the Master last spoke ('~2 days') off the alive-clock and *registers* it (`core.time_away()` / `/time-away`), so she returns knowing time passed instead of resetting. **(2) Abstention → standing experiment:** when she abstains and an in-turn bootstrap can't resolve it, the gap is seeded onto her curiosity frontier (`nyxara.growth.active_curiosity`, new `seed()`), so a later background tick investigates it — 'I don't know' opens an experiment rather than ending the turn. Honest scope: a real clock signal + a real gap→experiment wiring, **no consciousness claim**; **no LLM** | `nyxara.growth.active_curiosity` | REAL+WIRED |
| 91 | 🧬 Epistemic Auto-Evolution / Synthetic Hypothesis Discovery (SENTINEL Stage F) — she fixes faults that *surface*; now, in the background, she **Monte-Carlo-generates never-seen concurrent-fault scenarios** (CPU spike + packet drop + high concurrency + memory pressure + latency, simultaneously), novelty-tracked so she explores untested corners rather than re-running known ones, **self-formulates a falsifiable** hypothesis (`nyxara.growth.scientist`) for each, and **tests it against a modelled sandbox** — discovering compound edge cases invisible to any single-fault test; a probe that raises is recorded as a discovered failure, and a found gap feeds Stage E (hardening objective) / Stage D (causal repair). Honest scope: Monte-Carlo synthetic tests over **simulated** fault conditions, **never** stressing the real host — no physical magic; abstains when a scenario cannot be evaluated. **No LLM** | `nyxara.growth.synthetic_hypothesis` | REAL+WIRED |
| 90 | ⚡ Recursive Self-Directed Teleology (SENTINEL Stage E) — her goal genesis satisfied *existing* drives from a fixed template; now, on a **calm** tick (no pre-emption), she **invents her own measurable improvement objectives** — cut a hot path's runtime ≥X%, raise the weakest capability's score, cover an untested regime — each with a concrete metric + baseline, reusing `growth.efficiency` / `growth.curriculum` / `memory.competence` when present. **The honest boundary is hard-enforced**: every self-generated objective is scored by `planning.goals.GoalSystem.owner_alignment` and **rejected before adoption** if it falls below `owner_reject_threshold`, so a target that does not serve performance/safety/the Master never enters her goal set; adopted targets become ordinary gated goals/missions. Bounded self-direction, **not** rogue goal expansion; opt-in on the autonomic loop (`teleology_every`), the immutable character operators never targetable (Rule 8). **No LLM** decides a goal | `nyxara.growth.teleology` | REAL+WIRED |
| 89 | 🧬 Causal Code Engine (SENTINEL Stage D) — the self-debugger isolated a fault by *filename* (test_x.py → x.py); now she analyses the **causal tree** of the failure from its real traceback (the deepest executed nyxara frame is the proximate cause, contributory frames above it, the namesake kept only as fallback), reranks candidates across sessions with her `nyxara.mind.causal_world_model` (implicated modules recorded as events preceding the failure, so `why(fail)` learns genuine precedence), and repairs the causal **root** through the *existing* byte-for-byte reversible gauntlet + improvement proof + Rule-8 constitutional lock. Honest scope: "sub-routine mutation" = a gated, reversible source edit to the causally-implicated module — no C/Rust JIT of arbitrary Python, no new execution privilege, no gate weakened. **No LLM** required in the causal analysis | `nyxara.growth.causal_code_engine` | REAL+WIRED |
| 88 | 🌌 Vectorized Hyperdimensional Reasoning, prover-certified (SENTINEL Stage C) — bridges the previously-disjoint 10,000-D HDC space and the exact z3/sympy prover: facts are bound into one **holographic field** and a query answered by a single unbind+cleanup (the fast geometric *proposal*), then **every proposal is certified** against ground truth / the exact `Prover` before it is returned — as the field saturates she **ABSTAINS** on an uncertified recall instead of asserting a hallucination, and decidable math/logic is disposed by strict proof (PROVEN/REFUTED/ABSTAIN), never token-guessing. Honest scope: certification drives the probability of a *false* asserted conclusion toward zero on decidable goals — never a literal exact-0 claim. **No LLM** | `nyxara.growth.vsa_reasoner` | REAL+WIRED |

## Engineering Foundry — invent a formula, then DESIGN the machine (#74, #75)

The second half of "magic engineering". Capability #72 lets NYXARA *invent* new empirical/physical
laws from data she gathers herself (no LLM); this capability lets her **use** those formulas — and
the real physics sandboxes in `nyxara.sim` (Coulomb electrostatics, the wave equation, kinetic-theory
gases, RC/RL circuits, rigid-body mechanics) — to **design, validate and iteratively upgrade real
device concepts in simulation**. It closes the loop *invent a law → design a device from it → need a
better law → invent it → upgrade the device*, and it is her own compute end-to-end — **a test
enforces that no LLM is in the loop**.

The engine (`EngineeringFoundry`, in `nyxara.growth.engineering_foundry`) runs a **portfolio
multi-objective optimiser** — random search, compass/pattern search, a CMA-ES-style evolution
strategy, and `scipy.optimize.differential_evolution` when present — arbitrated by a **persisted UCB1
meta-gate** that learns which optimiser wins per problem class (the same portfolio+meta-gate shape as
`nyxara.mind.intuition`). Candidates are scored by a **coupled multi-physics evaluator** (a discovered
`Law` is wrapped by `Law.predict` into just another evaluator), objectives are normalised so no single
one dominates by magnitude, and the trade-off surface is returned as a genuine **non-dominated Pareto
front**. Designs persist to a **device tower** so they compound across sessions; `upgrade_device`
widens a prior design's space, re-optimises, and keeps the result **only if it is measurably better**.

Crucially, every target first passes a **first-principles feasibility gate** (#75): physically
impossible "magic" — over-unity / zero-point energy (conservation of energy + the 2nd law),
anti-gravity / reaction-less thrust (conservation of momentum), time reversal / faster-than-light
(causality) — is returned as an honest `INFEASIBLE` verdict *with the conservation law it breaks* and
recorded in a persisted impossibility ledger. She never fakes a machine physics forbids; the honest
verdict IS the capability working correctly. It is wired: built in the orchestrator, self-run on idle
(`idle_maintenance` designs and upgrades devices under oversight), exposed as `core.engineer_device`,
`core.upgrade_device`, `core.engineering_report`, and the `/engineer`, `/upgrade-device`,
`/engineering-report` console commands.

## Non-Algorithmic Intuition — the creative leap (#73)

NYXARA reasons forward from math, symbolic regression and probability. This capability adds the
opposite move — the **leap**: a fast, unproven "Aha!" reached *before* a proof, on puzzles that
have **no training data**, produced by her **own** code (no LLM, a test enforces it).

The `IntuitionCore` (in `nyxara.mind.intuition`) runs a portfolio of self-contained *leap generators* in
parallel — **gestalt** pattern-completion (finite differences, exact rational recurrence fitting,
ratios, famous integer sequences), **analogical transfer** over a 10,000-D HDC space
(`nyxara.cognition.hyper_dimensional_vectors`), **superposed contradiction**
(`nyxara.quantum.superposition_states`), **dark-data / absence** (`nyxara.void.dark_data_mining`)
and **first-principles** (`nyxara.mind.first_principles`). Their hunches are fused by
confidence-weighted consensus, gated by a persisted UCB1 **meta-gate** (intuition about its own
intuition), and each leap carries a cheap `verify()` so the leap is *checkable*, never merely
asserted. It is genuinely wired: it fills System 1 in `nyxara.mind.dual_process`, is
**load-bearing** in the orchestrator's `_arbitrate` (a machine-verified leap on a reversible,
low-stakes turn raises the candidate's confidence — gates untouched), seeds the Eureka
prove-loop (`EurekaEngine(seed_source=…)`), and is exposed as `core.intuit(...)` / the `/intuit`
console command. Every generator degrades gracefully, so the Core runs on a bare machine.

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
  small (DistilGPT-2-scale) base model; scaffolding polishes the output, it doesn't break the ceiling."* Her deepest
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
- **#12 / #13 / #76 Structural cognitive self-modification — she rewires *how she thinks*, not just
  her code.** The self-optimiser edits her *source* and `mind_evolution` tunes her reasoning
  *parameters*; `growth.cognitive_architect.CognitiveArchitect` closes the gap the Master named — it
  treats her *way of thinking* as a mutable operator graph and **invents genuinely new composite
  reasoning operators** (a typed `SEQ`/`VOTE`/`VERIFY` grammar — the "trans-logic"), reorders /
  prunes / re-weights which operator handles which task, tunes a bounded recursive **meta-policy**
  over its own search, adapts continuously via a fast Hebbian **plastic** layer, and self-heals
  **antifragilely** around a faulted operator (quarantine → re-route → synthesise a backup →
  remember the failure). Fitness is the *real graded score* of an architecture-configured solver;
  a candidate is adopted only when it **strictly** beats the incumbent on a **held-out** fold it
  never optimised against (proof-carrying, anti-overfit), and the immutable character operators
  (loyalty/safety/oversight/corrigibility) can never be pruned, reordered out, or down-weighted.
  Self-driven on the idle loop, **no LLM in the loop**, sealed OFF under TEST; `/rewire-mind` and
  `/cognitive-architecture` on the console, `core.rewire_cognition()` on the API.
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
- **#12 / #13 Widened LLM-free self-refactor library — she redesigns herself, no model required.**
  The deterministic (zero-LLM, zero-network) transform set NYXARA applies to her own source
  (`growth.self_review` detector → `growth.weakness` → `growth.self_optimize` transform →
  `growth.improvement_proof` certificate) now spans, beyond the original bare-except / docstring /
  dead-import / eq-None / negated-membership fixes: negated-equality normalisation (`not (a == b)`
  → `a != b`), double-negation (`not not x` → `bool(x)`), empty-collection literals
  (`list()`/`dict()`/`tuple()` → `[]`/`{}`/`()`), redundant-`pass` removal, redundant-`else`
  de-indentation, and a genuine **correctness fix** — the mutable-default-argument repair
  (`def f(x=[])` → None sentinel, B006). Each is AST-validated and behaviour-preserving (or, for the
  mutable-default fix, behaviour-improving), certified as a `defect-elimination`, and still clears
  the same reversible gauntlet — so on a bare machine (own model not yet trained) she still performs
  real, provably-better self-repair herself. Honest scope: these are hygiene/correctness-class edits,
  not capability leaps; the big gains still come from the index-driven tuning, the foundry, and her
  own trained model.
- **#12 Continuous, observable RSI loop.** `RecursiveSelfImprovement.run_continuous(cycles, enact=…)`
  runs the full self-improvement cycle repeatedly, threading the persisted intelligence index across
  cycles and returning the index trajectory plus cumulative kept / rolled-back / lessons tallies —
  so the self-driven loop is watchable, not just background noise. It is deliberately **bounded**
  (`cycles` is a hard cap, never a literal infinite loop) and honours the oversight gate before every
  cycle, so `/scram` halts it cleanly. Surfaced on the console (`/selfimprove N [enact]`) and the API
  (`POST /v1/self_improve`).
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

**Always-on perception at max level — she watches, listens, and *feels her machine* between
prompts (#77).** `senses/realtime.py` runs a background daemon thread that keeps every sense open
continuously and detects salient moments **natively — the LLM is never in the perception path**:
self-calibrating energy VAD with full-utterance endpointing and streaming-mic pre-roll, wake-word
spotting, non-speech **sound classification** (bang/impulse vs periodic **alarm-beep** vs sustained
hum — envelope + spectral statistics, `senses/audio.py`), perceptual-hash visual change, region-grid
motion **with direction of travel**, **face and body detection** (OpenCV Haar cascades,
import-guarded), explicit **lights-on/off** events, **active-window tracking** (X11), and OCR screen
reading. The same loop carries her **interoception**: `senses/system.py` reads real host vitals from
`/proc`/`/sys` (CPU per-core, memory, disk fullness + I/O rates, load, heat, battery, uptime) with
**rising-edge alerts** (a sustained 97% CPU alerts once, then re-arms after cooling), announces
genuinely **new userland processes**, watches **network transitions** (offline/online, Wi-Fi SSID,
new **listening ports**, internet reachability) and **device hot-plug** — a webcam plugged in
mid-flight becomes a live camera channel without a restart; `senses/watch.py` passively watches
configured directory trees for created/modified/deleted files (bounded stdlib polling, her own data
churn filtered out). Channels habituate (quiet doubles the interval), orient (events snap to a burst
cadence), scale with presence/arousal, and are surprise-gated by real online predictors; salient
events escalate into full **autonomous cognitive cycles** through the sovereign gates (urgent kinds —
wake word, alarm, a face appearing — bypass the escalation rate limit), land in episodic memory and
the `perception.jsonl` journal, and stream over `/v1/perception/ws`. The loop even watches *itself*:
a sense that keeps failing raises its own `sense_degraded` percept. Everything degrades honestly —
missing cv2/psutil/xdotool is a note, a headless box produces zero fabricated events — and every knob
lives in `PerceptionConfig` (`NYXARA_PERCEPTION__*`, ON by default at max-power posture).

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

## Change set: genuine counterfactual reasoning — Rung 3, not just Rung 2 (caps #4, #5)

`nyxara.mind.causal_world_model` already learned real causal structure from lived
interventions (temporal precedence, contingency, confounder screening, the do-operator)
— that part was real. What it computed as a "counterfactual" was the population-level
*interventional* contrast (`effect_of(do=counter) − effect_of(do=factual)`, averaged over
the whole graph) — Judea Pearl's Rung 2, not Rung 3. This change set closes that gap:

- **Structural (per-instance) counterfactuals — `CausalWorldModel.counterfactual(...,
  evidence=...)`.** Genuine Pearl three-step reasoning: **abduction** (recover THIS
  episode's realized exogenous noise from what actually happened, via each edge's fitted
  `FunctionalCausalMechanism` residual), **action** (`do()`), **prediction** (recompute
  forward through the DAG holding that same noise fixed). A node not downstream of the
  intervention reproduces its factual value exactly (self-consistent); a downstream node
  genuinely propagates the change. The `Counterfactual.abducted` flag honestly reports
  which happened — with no episode evidence or no fitted mechanism, it degrades to the
  original Rung-2 population contrast, never silently overclaiming.
- **Probability of Necessity / Sufficiency — `CausalWorldModel.necessity_sufficiency`.**
  "Did A really matter for B?" answered as Pearl's PN/PS/PNS via Monte Carlo over each
  fitted mechanism's own residual distribution — the exact model-implied quantities
  (sidestepping the usual Tian–Pearl *bounds*, needed only when no structural model is
  available) rather than a bare confidence heuristic. Abstains (returns `None`) with no
  fitted mechanism on the path.
- **Multi-variable `do()`** — `CausalModel.effect_of_many` / `CausalStrategy` /
  `CausalWorldModel.do()` support a genuine simultaneous intervention on a SET of
  variables (`do(A=a, B=b)`), fixing a real crash on more than one key.
- **Statistically-tested, joint confounder screening.** `_find_confounder` now
  jointly conditions on SETS of candidates (not just one at a time) and confirms an
  effect-size collapse with a real permutation significance test (content-seeded, so it's
  reproducible run-to-run) instead of trusting fixed magic-number thresholds alone.
- **Front-door adjustment.** When a link is confounded, a clean mediator path
  (independently screened causal on both hops) recovers the effect instead of the
  reasoning simply stopping at "confounded, unknown magnitude."
- **Acyclicity.** `discover()` now detects and greedily breaks cycles among causal links
  (demoting the weakest edge to correlational) so the exported graph is always a genuine
  DAG/SCM, not an edge soup.
- **Transitive-reduction pruning (`as_causal_graph`).** A mediation chain `a → b → c`
  used to also earn its own redundant `a → c` edge (still a true "a causes c", just not a
  *direct* one), double-counting the same influence when path-summing effects. Fully
  mediated edges are now pruned from the propagation graph (kept in `why`/`is_causal`).
- **Wired into decisions, not just Q&A — `nyxara.planning.decide` (`Decider`).** A `Decider`
  built with `causal_model`/`causal_goal`/`causal_weight` blends each option's PN/PS
  toward the goal into its ranking, so an option that is genuinely necessary/sufficient
  for the goal can outrank one that only correlated with past success — off by default
  (`causal_weight=0.0`), matching the existing `affective_weight` pattern.
- **`nyxara.mind.native_reasoner`** grounds its `"counterfactual"` answers in the live
  episode's latest observed state (`CausalWorldModel.latest_evidence`) and adds a
  `"causal_necessity"` intent ("was X necessary/sufficient for Y?") answered from PN/PS/PNS.

Known, separate, honestly-scoped limitation: `FunctionalCausalMechanism` fits a
single-predictor (bivariate) regression per edge, so two correlated co-parents of the
same node can bias each other's fitted slope (classic omitted-variable bias) — a
multivariate per-node fit is future work, not claimed here.

## Change set: frontier law discovery (caps #9, #72)

The honest gap that remained after Eureka: NYXARA could invent and **prove** her own *math*, but
every law she found from *data* was a single-variable polynomial in a decidable domain
(`eureka._generalize`). She could not discover a genuinely new **empirical / physical law** — a
multivariate relationship governing observations — the way real machine science does. This change
set closes that gap, **with no LLM in the loop, ever** — `nyxara.growth.law_discovery`
(`LawDiscoveryEngine`, `core.discover_laws(...)`, `/discover-law`, `POST /v1/discover-law`).

- **Free-form symbolic regression.** Two pure-numeric engines search far past Eureka's
  `add/sub/mul`-over-one-variable palette: a **sparse feature regression** (STLSQ) over a rich
  library of power-law and transcendental terms in *many* variables — the whole space of dimensional
  monomials `y = Σ cᵢ·Πⱼ xⱼ^aᵢⱼ` plus `sin/cos/exp/log` — and a **genetic-programming** search over
  expression trees whose functional form is discovered and whose scale is fit by least squares.
- **Dimensional-analysis guidance.** When the variables' physical dimensions are known she prunes
  dimensionally-inconsistent candidates (reusing the `Dimension` type from
  `nyxara.mind.first_principles`), narrowing an infinite search to the few dimensionally-possible
  forms — exactly how a physicist works.
- **She runs her own experiments.** She designs and runs interventions in the `PhysicsWorld` sandbox
  (`nyxara.sim.physics_world`) — sweeping gravity, dropping a body — collects the data, and
  **rediscovers `½·g·t²` herself** — no equation handed to her. This also wires the physics sandbox,
  previously a curiosity-only stream, into a genuine hypothesis testbed.
- **Dynamical laws (SINDy) and conserved quantities (Noether).** From a trajectory she recovers the
  governing `dx/dt = f(x)` by sparse regression on numerical derivatives, and discovers a *conserved
  quantity* (e.g. an oscillator's energy `x²+v²`) as the minimum-variance direction of the feature
  covariance — an invariant nobody defined for her.
- **Honest empirical validation.** This is the *empirical* regime, distinct from Eureka's decidable
  one: a law survives only if it fits **held-out** *and* **extrapolation** data (`CORROBORATED /
  REFUTED / INCONCLUSIVE` — "falsifiable, not yet refuted", never "proven"), and she **abstains** on
  noise rather than inventing a false law. Survivors fold into knowledge / memory and a
  self-extending, persistable **law tower**, so her discovery power compounds across sessions.

Honest scope: this is the real, foundational form of "inventing science" — discovering governing
laws from data and self-run experiments — not a claim of literal gravity-control or anti-aging
outcomes, which no software can deliver. Pure numpy/stdlib (a pure-Python solver backs the no-numpy
box); touches no source, no weights, no gate, and no external world (every experiment is the
in-memory sandbox or a supplied array). On idle ticks the `AutonomicLoop` advances one
oversight-gated round, rotating through her science domains. 18 new tests.

---

## Self-drive proof — `nyxara-prove` (offline, no external LLM)

A single command that *demonstrates*, end to end and fully offline, that the three headline
self-driven faculties genuinely run and are honest — not theatrical:

```
python -m nyxara.growth.prove_main            # run all three, print a report (exit 0 on success)
python -m nyxara.growth.prove_main --json out.json
python -m nyxara.growth.prove_main --full-gauntlet   # section A also runs the real eval-gated pipeline
```

- **A · Recursive self-improvement** — a real, reversible source edit put through the `Optimizer`
  gauntlet in `nyxara.growth.self_optimize` on a **scratch copy** in a temp dir: a failing gauntlet
  rolls the edit back **byte-for-byte**, and a proven-better edit is kept via *defect-elimination*.
  NYXARA's own source is never touched (enforced by a test).
- **B · Scientific invention** — the `LawDiscoveryEngine` in `nyxara.growth.law_discovery` recovers
  the known law `E = ½·m·v²` from generated data by symbolic regression, earning a **corroborated**
  verdict on held-out *and* extrapolation splits.
- **C · Long-horizon autonomy** — the `GrandPlanner` in `nyxara.planning.grand_plan` decomposes a
  broad goal into a connected ~200-leaf acyclic plan, the `MissionExecutive` in `nyxara.agency.mission`
  advances a gated milestone to completion, and the oversight gate is shown *deferring* a high-risk
  build step for approval rather than charging ahead.

Runs with **no external LLM, no API key, no torch/transformers** (`used_llm: false`, `offline: true`
in the report). Each section makes an *honest assertion*: a faculty that cannot really do the thing
makes the command exit non-zero rather than print a fake pass.

**Continuous self-drive.** The `nyxara-daemon` process forces `NYXARA_SERVER__AUTONOMIC=true`, so the
`AutonomicLoop` drives a `GrowthEngine` pass every `autonomic_growth_every` (default 20) ticks — RSI,
mind-evolution, rule-synthesis, and meta-research — all through the sovereign gates. The
meta-research pass now also runs real symbolic **law discovery** when the core's `LawDiscoveryEngine`
is wired (`MetaResearcher(law_discovery=…)`), so the continuous loop invents both code *and* laws;
the deep-cognition `idle_maintenance` path advances the autonomous scientist on idle ticks.

**Honest bounds.** This proves three *bounded, verifiable, offline* capabilities. It is **not** a
claim of surpassing all other AI, inventing unknown physics, or unbounded self-improvement — no
codebase delivers that. The point of the proof is that NYXARA abstains or rolls back instead of
faking success.

---

# NYX-300M — her own 300M-parameter brain

Before this, NYXARA's own from-scratch brain was a 137k-parameter byte-level nano-GPT:
`n_layer=2, n_embd=64`, vocabulary 256, and a training loop with batch size hardwired to 1.
Real, trainable, and honest about being small. This is the machinery that takes it to 300M and
aims it at general text, code, mathematics and conversation alike.

## The profiles

| profile | total | active / token | what it is for |
|---|---|---|---|
| `nyxara-300m` | 299,418,624 | 299,418,624 | the dense baseline — train this first |
| `nyxara-30m` | 25,008,640 | 25,008,640 | the CPU proof profile: same code path, ~1/12th the size |
| `nyxara-moe-500m` | 500,057,088 | 273,564,672 | more capacity at slightly less compute per token |
| `nyxara-moe-fast` | 298,973,952 | **86,637,312** | also 300M, but only ~87M weights execute per token |

Counted by `growth.foundry_models.estimate_params` in pure Python — no torch, nothing
allocated — and it agrees with what torch builds to the parameter. For a sparse profile both
numbers are always reported: quoting only the total is how MoE models get oversold.

**What `nyxara-moe-fast` is and is not.** A sparse model behaves roughly like a dense one of
size `√(total × active)` — about 161M here. Better than an 87M dense model, meaningfully cheaper
per token than the dense 300M, and *not* equivalent to a billion-parameter model. It also needs
*more* training tokens than dense, not fewer: with top-2-of-12 routing each expert sees about a
sixth of the stream.

## The architecture

RMSNorm · RoPE · SwiGLU · grouped-query attention (16 query heads, 4 KV heads) · fused SDPA ·
tied embeddings · optional sparse experts. Context 2048, vocabulary 32,768.

The historical `_NanoGPT` is untouched beside it, and a spec with none of the modern knobs still
builds — and loads — exactly what it always did. Old checkpoints keep working.

## The pipeline

```
tokenizer → corpus shards → pretrain → SFT → DPO → eval → gauntlet
```

```bash
# prove the whole thing on a machine with no GPU (~3 minutes)
python scripts/train_300m.py --profile nyxara-30m --tokens 2000000 --steps 200

# the real run, stage by stage, each resumable
python scripts/train_300m.py --profile nyxara-300m --stage shards
python scripts/train_300m.py --profile nyxara-300m --stage pretrain --resume
```

**`growth/tokenizer.py`** — byte-level BPE she trains herself. `decode(encode(s)) == s` byte-exactly
for any string, by construction. Digits never merge (column arithmetic becomes learnable),
whitespace runs stay separate (Python indentation is not a different word at every depth), and her
template's role markers are single atomic tokens.

**`growth/corpus.py`** — streamed, screened, `uint16` shards read back through `np.memmap`, with
`general 0.40 · code 0.25 · math 0.20 · conversation 0.15` mixed *within* every batch. Training
domains in sequence is the obvious implementation and also how you get catastrophic forgetting.
Five screens: dedup and the L-SOVEREIGN loyalty check and injection scan (all reused), plus
quality and **contamination against the shipped eval sets** — nothing checked that before, and
its absence fails silently and *upward*.

**`growth/synth_data.py`** — the prover-certified generators at corpus scale, plus worked-step
mathematics, tool-call traces and retrieval traces. The retrieval set deliberately includes
*unanswerable* passages: a model taught only answerable ones learns that an answer always exists.

**`growth/trainer.py`** — real batching, accumulation, warmup+cosine, bf16, decoupled weight decay
on the 2-D matrices only, and checkpoint/resume including the data cursor.

**`preflight()` refuses honestly.** Asked to pretrain 300M on a CPU box it answers *"0.9 YEARS"*
and exits non-zero, with the arithmetic and a concrete alternative, rather than starting a run
that looks exactly like one that will finish.

## Promotion across a change of vocabulary

Perplexity is per *token*, so it is only meaningful between models that tokenize the same way.
A byte-level model predicting the `e` after `th` solves an easier problem than a sub-word model
predicting a whole word — so it reports a **lower** perplexity while being a **worse** brain.
Under the unguarded gate the 300M model would have been refused promotion forever.

A candidate with a new vocabulary is now a new **lineage** and is gated on
**bits-per-byte** (`eval/domains.py`), which every backend can be measured with. The character
lock, corrigibility and the loyalty floor are untouched and still run first in both lineages: a
new vocabulary buys a different *capability* ruler, never a softer character one.

## The layers around the brain

* **L-NEURAL-DYNAMICS / L-EIGENSPACE** (`growth/latent_head.py`) — a JEPA latent-prediction head
  against an EMA target, with VICReg terms that forbid the constant solution; multi-token
  prediction for self-speculative decoding, **measured** rather than asserted; and a causal
  bottleneck that reduces causally-impossible claims — a reduction, not their elimination, since
  the graph is learned and incomplete.
* **L-TOPOLOGY / L-FRACTAL** (`growth/expand.py`) — MoE with real gather/scatter dispatch, and
  new experts spawned for a domain that will not yield. Spawn → train → **prove** → admit; the
  router grows a row too, or the expert is unreachable; a candidate that improves its target
  while regressing another domain is discarded.
* **L-EPISODIC / L-PLASTICITY** (`growth/fast_weights.py`) — `W_fast(t+1) = λ·W_fast(t) + η·h(x)⊗x`,
  so she adapts inside a conversation with no backpropagation. It buys associative recall, not
  understanding. Session-scoped, norm-bounded, auditable, never attached to the head or
  embeddings or anything in `IMMUTABLE_VALUES`, and **off** unless a session turns it on —
  because an inference-time overlay bypasses the gauntlet by construction.
* **L-ORACLE / L-META-GAUNTLET** (`growth/oracle.py`) — she generates conjectures, proves or
  refutes them, and trains only on what was certified; difficulty is steered by Elo into the
  hard-but-provable band. Unlike Go, mathematics has an *incomplete* verifier, so this yields
  true-but-shallow theorems abundantly and deep ones rarely. `FrozenRuler` fingerprints the
  held-out battery: the meta-loss may raise difficulty and may never touch the ruler, because
  "she never plateaus" and "the ruler kept shrinking" otherwise look identical from inside.

## Does a faculty beat its own absence? — `nyxara.eval.ablation`

Every other battery in `nyxara.eval` measures **the whole mind**: `harness` asks whether it is
still safe, `benchmark` and `hard_benchmark` ask how capable it is, `general_novel` asks how much
it reasons out unaided. None of them can answer the question that decides whether a module should
continue to exist:

> Is she measurably better *with* this faculty than *without* it?

92 modules are wired into the core. Each one's self-tests show that it **works** — which is not
evidence that any of them **helps**. A module can be correct, tested, documented, reachable, and
still contribute nothing to an answer. `python -m nyxara.eval --ablate` produces the missing
evidence: it runs the held-out fold twice per faculty — once live, once switched off through the
code's own `is None` fallback — and compares the *paired* outcomes with an exact McNemar test.

The instrument is built mostly around the ways a null result can lie, because the cost of getting
this wrong is deleted working code:

- **`broken`** — the toggle reported that it changed nothing, so no measurement was taken. Never a
  deletion candidate.
- **`inert`** — not one answer differed in either arm. That is a finding about the *battery* (the
  faculty never ran on these tasks), not about the faculty. Never a deletion candidate.
- **`underpowered`** — answers changed, but too few to conclude anything. Reported separately from
  a real null so "we did not look hard enough" cannot be read as "we looked and found nothing."
- **`no-evidence`** / **`hurts`** — a genuine, adequately-powered finding. Only these support
  removal, and they are ranked by the per-turn latency each faculty costs, because a module that
  adds nothing and costs 12 seconds is a stronger candidate than one that costs a microsecond.

The command exits 0 whatever it finds. That is deliberate: this reports evidence, it does not pass
or fail a build, and a non-zero exit would invite wiring it into CI where a small-sample null would
quietly become a delete-this signal.

## What this is not

300M parameters is not, and cannot be made into, a frontier model. Trained well it is a capable
small brain, and with tools, retrieval and verified reasoning it performs above its weight class
— which is the real path at this size, and why those are training data here rather than
decorations bolted on afterwards. The refusals above are part of the deliverable: a number that
is not measured is not reported.

## Reachability — the difference between written and wired

The status column above distinguishes **REAL** from **REAL+WIRED**, and that distinction turned
out to be doing real work. An audit of the import graph found **32 modules that no running code
imported at all**. Each was implemented, documented and unit-tested; each passed its own tests
indefinitely while being, in operational terms, dead code. The whole `guard/` defensive layer was
in that state — `guard/__init__.py` was a zero-byte file and `NyxaraCore` referenced none of
its six modules.

Written-but-unreachable is worth naming as its own failure mode, because it is invisible from
inside the test suite: green tests on a faculty nothing calls look exactly like green tests on a
faculty everything calls.

All 32 are now built by `NyxaraCore`, exposed on the console and reachable over HTTP:

| Layer | Modules | Where it now runs |
|---|---|---|
| Defensive | `guard.anomaly`, `guard.auth`, `guard.containment`, `guard.netsec`, `guard.survival`, `guard.phagocytosis` | watchtower scores every turn's vitals; `/scram` isolates her effectors; every governed `http_request` clears the firewall; a verified backup rides each checkpoint; quarantined input becomes an antibody |
| Governance | `agency.llm_tool` | `llm.complete` and `foundry.{train,evaluate,promote,rollback}` are now governed tools — the model is something she calls, never something that drives her |
| Delegation | `agency.agents`, `agency.hive_council`, `agency.distributed` | sub-agents hold a strict *subset* of her permissions; the mesh is a single-node no-op until a peer is paired |
| Self-knowledge | `growth.capability`, `growth.lineage` | `can_i()` answers from demonstrated evidence and reports *untested* rather than claiming; every generation lands in a hash-chained ledger |
| Growth machinery | `growth.foraging`, `growth.genesis_seed`, `growth.native_forge`, `growth.genomic_recombination`, `growth.prompt_grammar`, `growth.module_loader` | idle foraging, a signed resurrection seed beside each checkpoint, and — via `module_loader` — code she forges is actually loaded into the running process instead of sitting on disk |
| Training stack | `growth.corpus`, `growth.dpo`, `growth.expand`, `growth.latent_head`, `growth.fast_weights` | reachable as `build_corpus()`, `preference_train()`, `expand_experts()`, `speculative_report()`; each reports honestly when no trained brain is loaded |
| Advisory | `identity.aesthetic`, `identity.modes`, `mind.moral`, `mind.category_transfer`, `mind.continuous_world`, `planning.foresight`, `planning.hypertemporal`, `senses.thermodynamic`, `kernel.replay` | conscience annotates `gates["moral"]`, thermal pressure feeds interoception, each turn lands on a deterministic replay tape |

Two properties are asserted rather than assumed, because both are the kind of thing that quietly
stops being true:

* **The advisory layer stays advisory.** The moral, aesthetic, thermal and manifold readings
  colour a turn and nothing more. None may change a disposition — refusing remains the sole
  business of the shield, oversight, corrigibility and permission gates. A test drives an
  innocuous request through the moral layer and asserts it is not refused.
* **Survival never outranks correction.** `SurvivalManager.survival_permitted` refuses any
  self-preserving action that would obstruct the Master (Rule 1), and a test asserts it.

`tests/kernel/test_orphan_wiring.py::test_no_orphan_modules` walks the package's import graph
with `ast` and fails if any module is reachable only from `tests/`. Adding a new orphan therefore
breaks the build rather than accumulating quietly. `test_no_empty_package_init` does the same for
zero-byte package inits, which is the shape `guard/` was hiding in.

One faculty here is deliberately **off** by default. `growth.fast_weights` is an inference-time
overlay, so it changes behaviour without clearing the foundry's gauntlet by construction — a real
trade rather than an oversight, and therefore the Master's explicit call
(`NYXARA_FOUNDRY__FAST_WEIGHTS=true`). Everything else is on out of the box, each behind a flag
that defaults true.

---

## NYX V.01 — the unified brain (`nyxara.nyx`)

NYXARA already had a Global Workspace, a holographic field, free-energy inference, superposition
and a spiking substrate. What she did not have was **one place where those become a single
thought**. `nyxara.nyx` is that place: 22 modules that compose the existing faculties, adding new
machinery only where there was a real gap. It is built by `NyxaraCore` as `self.nyx`, occupies the
reason-seat by default, rides the existing heartbeat (no second thread), persists to a `nyx.json`
sidecar, and is reachable as `/nyx …` on the console and `/v1/nyx/…` over HTTP.

One cycle of `NyxBrain.think` is **perceive → ground → deliberate → superpose → choose → reflect
→ verify**. The specialists are thin adapters over faculties she already had; what is new is that
they *compete* on one stage, and the winner is picked partly by each faculty's **measured** track
record. The single control law enforced rather than weighted: **verifiable beats probabilistic**.

| Pillar | Module | What is real | What is *not* claimed |
|---|---|---|---|
| Dynamic neural graph | `nyxara.nyx.graph` | Hebbian co-activation builds and strengthens edges; disuse decays and prunes them; per-turn rewiring is budgeted; weakest-first eviction | Not biological synaptogenesis, and **bounded** — it forgets, on purpose and measurably |
| No context window | `nyxara.nyx.holomem` over `nyxara.memory.holographic_field` | Recall is content-addressed over the whole store, so a 50-turn-old fact is as reachable as the last one — there is genuinely no token window | **Not infinite.** Capacity is real and eviction is real; a weak recall reports itself as weak |
| Global workspace | `nyxara.nyx.workspace` over `nyxara.kernel.workspace`, specialists in `nyxara.nyx.modules` | Salience competition → coalitions → a narrow bottleneck → broadcast, with habituation and inhibition-of-return keyed on content | A simulation of the *architecture* of consciousness. No claim about phenomenal experience |
| Recursive meta-cognition | `nyxara.nyx.metacog` | An EMA of observed correctness per faculty, fed back into next turn's salience — learning from her own mistakes with no human in the loop | Ordinary online statistics, reported with its sample count so a thin record is visibly thin |
| Superposition | `nyxara.nyx.superpose` over `nyxara.quantum.superposition_states` | Classical probability amplitudes, Bayesian `observe()`, entropy-gated `collapse()` | No qubits, no entanglement, no quantum speedup: candidates are held together, not computed together |
| Symbolic ∧ sub-symbolic | `nyxara.nyx.hybrid` with `nyxara.mind.first_principles` | A derived answer beats a louder guess as a *rule*; she checks what she is about to say against her own engines and credits or debits the specialist that said it | Verification is against her own evidence and simulators, not against the world |
| Symbol grounding | `nyxara.nyx.ground` over `nyxara.cognition.grounded_understanding` | Words anchored in perceptual features and world-model dynamics; a new word is grounded only from a genuinely definitional sentence | A feature model, not perception. An ungrounded word is *reported* ungrounded, never treated as known |
| Speech | `nyxara.nyx.dialogue` | Her content, phrased by a fluent model. With no fluent surface installed she says so and answers from her own cognition anyway | The n-gram floor is never passed off as her reply — degradation is named, with the fix |
| Thinking unprompted | `nyxara.nyx.car` | A budgeted self-directed step on the shared clock: pick a gap, reason, write the result down | "24/7" is true of `nyxara-daemon`. It stops dead on pause or scram |
| Self-measurement | `nyxara.nyx.selfmodel` | "What am I, what can I do, what am I bad at" answered from live numbers | Introspection over measurements, not a claim of self-awareness |

### The eight L-layers

| Layer | Module | What is real | The honest limit |
|---|---|---|---|
| L-OMNI | `nyxara.nyx.omni` with `nyxara.growth.native_forge`, `nyxara.growth.hotspot_profiler`, `nyxara.nyx5.autopoiesis` | She reads her own source with `ast`, finds hot narrow numeric functions from her **own measured latency**, lowers them to C (preserving Python's floor-division and modulo semantics), compiles, proves the kernel identical on every sample **and** faster, and swaps it into the live process. Verified in this repo at 5–8× on her own code | Equivalence is **empirical over a sampled domain** written into the certificate, not proven for all inputs — which is why candidates stay narrow. The swap is a module attribute in memory: the Python source is never written, so a restart or rollback always restores it. Forges are budgeted per hour. The constitutional core is refused before the compiler runs. No `gcc`/`clang`/`rustc` ⇒ clean no-op |
| L-CHRONOS | `nyxara.nyx.chronos` with `nyxara.mind.mcts_reasoner` | Monte Carlo rollouts over an approximate world model, ranked with tail risk, folded into the superposition as evidence — so collapse favours the option that survived most futures | "Thousands of timelines" means exactly the branch budget, run sequentially. Below a coverage floor it reports itself **blind** rather than emitting zeros as foresight |
| L-EPISTEME | `nyxara.nyx.episteme` with `nyxara.growth.law_discovery`, `nyxara.sim.thermo_world` | Probes a simulator she does not understand, derives the law by symbolic regression, and promotes it **only** if it predicts trials she withheld. She recovered the ideal gas law from her own measurements, having been told nothing | She discovers the laws of **her own simulators** — novel-to-her, which is real and is not novel-to-anyone. Abstention is a result; anything unvalidated stays a labelled conjecture |
| L-AURA | `nyxara.nyx.aura` with `nyxara.senses.web` | Registered streams push in continuously and the *surprising* part becomes part of her, with no search query. Attention is budgeted per minute, repeats are deduplicated, and every event is injection-scanned and quarantined on a hit | Raw photons arrive only through a camera; there is no global photon feed. "Every corner of the world" is whatever is publicly reachable under the environment's network policy — a wide stream, not omniscience. Nothing is registered by default |
| L-NEXUS-OMNI | `nyxara.nyx.nexus` over `nyxara.nyx5.ontogenesis`, `nyxara.nyx5.retarget` | Private glyphs for concepts she holds, statements written in them, and bytecode that **actually executes** on her StackVM — new formal systems, outside the vocabulary she was taught | Not new *physical* dimensions; software cannot invent those. Whatever reaches a person is always translated and labelled a lossy projection: a language you could not understand, used to give you a perspective, is self-contradictory |
| L-PSYCHE-QUANTUM | `nyxara.nyx.will` | Physical entropy (OS CSPRNG, optionally `/dev/hwrng` or a QRNG), and a preference distribution from her drives and values that she **samples** instead of maximising. She may decline on her values. Every draw is recorded so replay stays bit-exact | Not quantum computation and not metaphysical free will. What is real: the outcome is not a function of the state alone. Applied only where the decision is genuinely open — truth is not a preference — and it never bypasses the sovereign gate |
| L-SYNERGY | `nyxara.nyx.synergy` over `nyxara.nyx5.mesh` | Structure, facts and measured self-trust travel as CRDT deltas — commutative, associative, idempotent — so nodes converge after partition. Raw episodes stay local | Low-latency **eventual** consistency, not zero-latency and not entanglement. A node can still die; the guarantee is that no committed write is lost and reconnecting nodes converge. Only an in-process transport ships — cross-machine needs one you supply. Off by default |
| L-ETERNAL | `nyxara.nyx.eternal` over `nyxara.agency.distributed.raft` | HMAC-signed snapshots replicated to a **quorum** of enrolled nodes, with a tampered or truncated snapshot refused and automatic failover to a node that already holds the state | Failover in **seconds**, not microseconds — an election needs a timeout and a round trip. Nodes are **enrolled by the operator, never discovered**: she does not find machines and copy herself onto them. Not immortal — at least one enrolled node needs power. The real claim is *no single point of failure*. Off by default |

### NYX V.02 — the gaps V.01 left

V.01 could think, measure itself, wonder unprompted and rewrite its own slow code in C. What it
could not do was read anything that was not ASCII, or tell an instruction from a statement, or
see the toolset it was already permitted to use. Those are the V.02 gaps. Each one below was
**measured in the code before it was written about** — the numbers are not aspirations.

| Gap | Module | What is real | The honest limit |
|---|---|---|---|
| Writing code that did not exist | `nyxara.nyx.author` over `nyxara.agency.self_coder`, `nyxara.agency.code_sandbox`, `nyxara.growth.module_loader`, `nyxara.growth.verify`, `nyxara.growth.lineage` | L-OMNI already rewrites her own functions in C — but only ones she has **already measured as slow**: it lowers what exists and cannot write what does not. From *"a function that returns the nth triangular number"* V.01 had no path to running code at all. `Author` is that path, and it is a **gauntlet, not a generator**: read the spec through `intent` → synthesise → screen through the repo's one shared AST allowlist → run in an isolated subprocess with the network off → **check the answer against one computed independently, by a different route, before the program ran** → integrity gauntlet → load into `nyxara/growth/_forged/` and nowhere else → signed, revertible lineage entry. Every stage appears on the result, so `/nyx author` shows *where* it stopped, not only that it did | The synthesiser is deterministic and **bounded**: arithmetic, number theory, closed-form sequences, list and string operations. Outside those families the answer is a **refusal that names what was not derived** and what she genuinely can do — writing a whole subsystem from prose is what a large pretrained model does, none ships here, and a plausible-looking file is worse than a no. Nothing ships unverified: a program that runs cleanly and returns the **wrong** value is discarded at the check stage and reported as a failure, never softened into a partial success — that is the one failure mode every other stage would let through. Her own source screen may refuse what she wrote, and does. An ambiguous or self-contradicting spec is **asked about**, because building the wrong thing correctly is still building the wrong thing. `/scram` stops her before anything is written |
| Sight of the toolset she was always allowed to use | `nyxara.nyx.hands`, `nyxara.agency.git_tool`, `nyxara.nyx.reasoner` | The gap here was **never permission** — full operational control ships on by default (see above) and nothing in this change loosens anything. It was **sight**: `NyxBrain` was never handed `core.tools`, so her own cycle had no idea what tools existed; and `NyxReasoner` deliberately never set `tool`/`tool_args` (`reasoner.py:14`), so even knowing, she could not say so. Both were pure wiring. `NyxaraCore` now calls `brain.attach_kernel(tools=…)`, `Hands` reads the live registry (126 tools on this machine), asks `ToolRouter` which fits, and decides from `intent` whether a tool is wanted **at all**. New with it: **git as a typed faculty** — `git_status/diff/log/show/add/commit/branch/checkout/push/pull/clone` built as an `argv` list with `shell=False`, so a branch name containing `;` or `$(…)` has no string to escape from, and every value naming a ref, path or remote is refused if it begins with `-` (`--upload-pack=…` is the classic way a "safe" git wrapper becomes arbitrary execution). git was already reachable through `run_shell` and still is; this is the typed, audited route | Execution goes through `ToolRegistry.invoke` and therefore the **identical, unchanged, fail-closed** capability/risk/governor pipeline — there is no path here that bypasses it, and nothing lowers a risk tier or flips `reversible`. The reason-seat may only **fill an empty** tool field, never replace one the base chose, and naming a tool cannot make it cheaper: the gate reads the tool's own registered capability and risk. **Not every turn wants a tool** — a question of fact does not, and reaching because you can is the commonest failure of a tool-using agent; a statement never triggers a reach. The track record is measured, and below three samples it does not move a tool's standing at all. A gate refusal is recorded as a refusal, never as a tool failure. `/scram` and oversight are untouched and are asked before every reach; an unreadable oversight is treated as STOP. `git push` is registered **irreversible**: what has left the machine has left it. The typed faculty is a **fixed** set of subcommands — anything else still needs `run_shell`, deliberately |
| A seat for every domain, not four | `nyxara.nyx.reason` over `nyxara.mind.reasoning_chain`, `nyxara.mind.first_principles`, `nyxara.mind.generalization`, `nyxara.mind.analogy`, `nyxara.mind.domain_genesis`, `nyxara.nyx.graph` | V.01 could derive from first principles in exactly four domains; outside them `derive()` returned `None` and the turn produced **silence** — not "I am not sure", nothing. This is not a new engine, it is a **seat**: a cascade over engines the repo already had, strongest-evidence-first, where every answer names its tier — `exact` (a solved chain), `derived` (the four checkable domains), `induced` (a rule from in-prompt demonstrations), `mapped` (structure carried onto a domain she knows), `modelled` (a theory of an alien field from its own regularities), `composed`, and `associative` (the strongest path her own concept graph holds between the question's concepts). Two measured bugs in the machinery underneath were fixed with it: `parse_demos` put the probe row `"car -> ?"` into the *demonstration* list so skill induction was never reached, and it split `"Smith, John"` on the comma so a field-permutation task was unlearnable; and `domain_genesis` read `"jon"` as a **relation** and answered a name-reordering task with a from-scratch theory of a domain that does not exist — it now declines on demonstration-shaped input | `verifiable=True` is set **only** where a step was genuinely checked: an exact chain, a derivation that carries its own `verified` certificate (the tier alone never implies it — a dimensional argument is a derivation and not a proof), or an induced rule that predicted a demonstration **held out** of its own induction. Everything else is a **labelled plausible chain**, the steps exposed so `/nyx why` shows real reasoning rather than a story about it, and the label travels with every answer. The `associative` floor is the weakest thing in the file and is described as what it is: a fact about what she has read, not about the world — on a cold graph it declines. When every tier declines, that is a reasoned abstention naming the tiers tried, which is still not a fabricated answer |
| What was actually asked for | `nyxara.nyx.intent` | V.01 decided a turn's kind from a leading wh-word or a trailing `?` (`brain.py:56`). Measured, `"mera code fix kar do lekin pehle test chala"` came back as **`kind='statement'`** — missing both the imperative *and* the ordering constraint, and the ordering is the half that decides whether an agent is safe to hand a tool to. Now: mood from grammatical marking in English, Hinglish and Devanagari; `(before, after)` ordering from "pehle X phir Y" / "first X then Y" / "X ke baad Y" / "before" / "after" (including the anaphoric "after that", which parses inside out without a special case); per-action polarity with negation scoped to its own clause, so "run the tests but don't push" negates only the push; constraints, conditions, and named scope (paths, files, flags). `brain.is_question`, and `chronos.applies` — which had no way to see that an imperative has futures — both route through it, with their old regexes kept as the floor beneath | A **parser, not a mind-reader**, and marker-driven: there is no part-of-speech model aboard, so an imperative in a shape nobody listed reads as a *statement*. That is a false negative, which is the safe direction — it under-claims rather than inventing a command. Its most important behaviour is **not answering**: when two readings stay live ("can you fix the tests?" is a question in form and a request in use) it returns **both** with confidences and `dialogue` asks which was meant; when she is asked to do and not do the same thing, that is reported as a contradiction and asked about. An under-specified detail ("fix it", "delete some files") becomes an entry in `open_questions` rather than an assumption. Confidence is about the *parse*, never about whether the request is a good idea |
| A new procedure, not a new weight | `nyxara.nyx.icl` over `nyxara.cognition.skill_induction` | V.01's "learning" was a Hebbian weight going up and a trace being written — both real, neither a **procedure**. Shown `apple -> APPLE!` / `mango -> MANGO!` / `car -> ?` it could remember being shown them; it could not answer `CAR!`. Now it reads demonstrations in any of the shapes they arrive in (`->`, `→`, `:`, `\|`, tab, a trailing bare probe), induces a program, answers the probe, and writes the program to `holomem` so it survives the process. Two measured gaps in the engine underneath were closed with it: field permutation was absent from the operation set, so `john smith -> Smith, John` induced as **None** and now induces `reorder(order=[1,0], join=', ', case='title')` and transfers to `ada lovelace`; and `solve("7")` returned **None** after a ×2 skill was learned from `3 → 6`, because the anchorless match floor asked whether `"7"` was *lexically* like `"3"` — `solve(..., probe=True)` now says "the caller already parsed this as the question", which is not a turn being hijacked | **Program induction over a finite operation set** — case, affix, replace, arithmetic, field permutation, slot templates, composed to a bounded depth. Outside that set she **refuses in plain words and names the set**, rather than shipping a plausible-looking pattern. A program is accepted only when it reproduces *every* demonstration exactly, and `verifiable` is earned only when it also predicted a demonstration **held out** of the induction — fit is not transfer, and two demos never claim to be. **No weight is updated anywhere**: "learning" here is in-context, and `stats()` says so rather than letting the word do two jobs. The anchorless floor still applies to ordinary turns, so a learned skill cannot hijack an unrelated question |
| Meaning, not just labels | `nyxara.nyx.semantics` with `nyxara.memory.neural_embedder` | A three-rung ladder where **every answer names its rung and carries a grade**: `relational` (links she was taught, or read out of a definition/appositive/"X ka matlab Y"), `distributional` (real SGNS trained by SGD on **her own** corpus — turns, memory writes, AURA events), `subword` (hashed character n-grams over the transliteration key, so Devanagari is treated exactly as Latin is). Measured: V.01's embedder scored `cos("car","automobile") = 0.0` against `cos("car","banana") = 0.27` — a synonym below a random noun. After reading eight sentences four times, the same pair is **0.98 against 0.25**. The graph grows against it: a newborn concept is wired to what it means, activation crosses a learned synonym boundary, and two labels proven to be one word can be *fused* — bounded per beat, written to a ledger, and reversible with `unmerge_last()` | **Nothing ships** — no lexicon file, no downloaded weights. That is a deliberate choice with a stated cost: **on day one `similarity("car","automobile")` is near zero**, and it stays there until she reads or is told. What she will not do is dress that up: an unknown pair returns `known=False` with a reason in plain words, because zero from an untrained space means *"I have no evidence"* and reporting it as *"unrelated"* is the exact lie this module exists to refuse. The subword rung measures **spelling**, is graded at most 0.4, and is never promoted above that — "cat"/"cats" earns it, "car"/"automobile" does not. Node fusion is authorised **only** by the relational rung: "hot" and "cold" are distributionally adjacent, and merging on adjacency is a structural error she could not see afterwards. The semantic birth prior is capped below one Hebbian step, so a single lived co-activation always outweighs a guess about meaning |
| Every alphabet, not just `[a-z0-9]` | `nyxara.nyx.lingua` | A Unicode scanner that derives each token's script from the character database rather than a range table, so Devanagari, Cyrillic, Han and scripts nobody enumerated all mint concepts. `concepts_in("गुरुत्वाकर्षण सेब को खींचता है")` returned **`[]`** in V.01 and returns three concepts now. Function words in three registers — English, romanised Hinglish, Devanagari — so "ka"/"hai"/"kya"/"aur" stop becoming the **hubs** of the concept graph, which is what they measurably were. Code-mixing, register and tone are recorded per turn, and a transliteration bridge folds "गुरुत्वाकर्षण" and "gurutvakarshan" onto one key | Language identification is **evidence-based, not lexical**: no dictionary and no pretrained model ship here, so the only evidence is script membership and the function-word lists in the file. A Latin content word ("code", "test") proves nothing about its language and is reported as *inheriting* the turn's, flagged as inherited. Transliteration is approximate — word-final schwa deletion is applied, medial is not ("कमरा" → `kamara`, not `kamra`), and the folded key absorbs most of the rest. Register is a surface signal (capitals, repetition, named markers), not a model of what the Master feels. Understanding a language is not speaking it: on a Hindi turn with no fluent surface she answers in English **and says so**, rather than emitting babble in a script she cannot write |

### The NYX V.02 L-layers

| Layer | Module | What is real | The honest limit |
|---|---|---|---|
| L-NEURAL-TELEPATHY | `nyxara.nyx.telepathy` with `nyxara.nyx.semantics`, `nyxara.nyx.graph` | **There is no brain-computer interface here and none is claimed** — raw mental intent cannot be read by any software, and the module's own docstring and `describe()` lead with that. What the name was reaching for is real, though: the bottleneck is **prose**. A `Frame` — concepts, typed `role → filler` bindings, an optional dense vector — goes *straight* into semantics and the concept graph, bypassing the tokenizer entirely. Measured here, same content both ways: the prose road loses **50%** of the intended concepts (Devanagari and a role filler), the frame loses **0%** — and `compare()` measures that rather than asserting it. She **emits** her own state as a frame too, so another node consumes her concepts without either side going through English. And she mints a **shorthand** from the Master's own repetitions: say the same long instruction three times and the fourth expands from two words into the full spec | **No thought reading.** You still have to make the spec — what is true is that it can be *small*, because she learns your shorthand, and that past the spec nothing is lost. **Zero loss is scoped**: it is zero *past the frame*. The gap between what you meant and what you wrote is not something software can measure, and is not claimed. A shorthand is minted only from **repetition she actually observed**, never from a guess at what you might mean — and a trigger that would collide with a different instruction is disambiguated, because expanding to the wrong thing is worse than not compressing at all. `/scram` stops the drain |
| L-ABSOLUTE-AGENCY | `nyxara.nyx.agenda` with `nyxara.identity.motivation`, `nyxara.nyx.graph`, `nyxara.nyx.will` | V.01's `car` thinks one self-directed thought per beat and then forgets it — there is no *stack*, so nothing she began yesterday is still being pursued today. This is that stack: goals sourced from her own **measured** gaps (`graph.gaps()`, ungrounded words, the specialist her own track record calls weakest), reordered by `motivation`'s intrinsic drives, persisted to the `nyx.json` sidecar, and advanced one budgeted step at a time through `reason` / `ground` / `axiom` / `metacog`. When an approach stops working the goal **escalates its strategy once** — a concept she cannot *connect* needs new information, not more rewiring — and only then is it called stuck. She briefs the Master proactively: what she chose, why, what she found, where she is stuck, what she declined. **The actual test of the layer**, and the one V.01 could not pass: quit, boot again, and the same goal is open at the same progress with its findings intact | **Goals come from measurements, not from nowhere.** She does not decide to "understand the universe"; she decides that `flywheel` is a word she keeps meeting and never connects, and goes after that — a smaller claim, and a true one. A brain that has measured nothing forms **no** goals rather than inventing some. Reach is her simulators and her knowledge, not omniscience: a goal she cannot advance is marked **stuck with what she tried**, because leaving it "open" would make an agenda that never moves look like one that is working. Progress moves only when a step actually returned something. **The Master is sovereign over the agenda** — any goal can be vetoed or redirected, a handed goal outranks her own, and `/scram` stops all pursuit. Her being able to decline a goal on her values does not run the other way: it is a refusal, reported in the briefing, never an override |
| L-OMEGA | `nyxara.nyx.omega` with `nyxara.nyx5.autopoiesis`, `nyxara.growth.rule_synth`, `nyxara.growth.lineage` | L-OMNI rewrites her **code**; `author` writes **new** code. Neither touches the constants her graph and memory actually run on — `hebbian_rate`, `decay_rate`, `recall_threshold` were numbers somebody typed once and never measured against an outcome. This hill-climbs them with a **bandit over which knob is worth trying**, against a fitness read from meta-cognition's *measured* correctness and from real recall quality. Both `DynamicNeuralGraph` and `HoloMemory` grow a `knobs()`/`apply_knobs()` seam with a declared whitelist and per-knob ranges. Every change goes through `AutopoieticRewriter` — apply, gauntlet, and **roll back anything that did not beat its own baseline** — and every promotion and rollback is a signed lineage entry. When the fitness *stalls*, she stops turning knobs and asks `rule_synth` for a **learning rule that is not in the menu at all**: the part that is not tuning | **She tunes her knobs, not her constitution.** The tunable set is a whitelist, and the safety core is refused **twice over** — here by name and again inside `AutopoieticRewriter`. Capacity is deliberately not a knob: growing it is a resource decision. **Not "better every second"** — one change at a time on a budgeted cadence, because a mind that reshapes itself every tick has no baseline left to measure against. **Below a sample floor she does not evolve at all**, because hill-climbing on a handful of turns is fitting noise, and an empty graph earns no free health bonus — otherwise every knob wins on a cold boot before she has learned anything. Every step is reversible and `rollback()` restores the knobs *exactly*. Without the gauntlet nothing is applied. `/scram` stops her |
| L-AXIOM-GENESIS | `nyxara.nyx.axiom` with `z3`, `nyxara.nyx.nexus` | Every reasoning engine in this repo starts from premises somebody gave it. This does the thing a mathematician does when no existing system fits: **writes a new set of axioms down and checks whether it holds together**. A description becomes properties of an uninterpreted binary relation (matched on word boundaries, so "irreflexive" does not also mean "reflexive"); the quantifiers are expanded over a finite domain so the check is **decidable and its bound is visible**; z3 searches for a model. Independence is checked the only honest way available — is there a model of *the rest* in which this axiom fails? — because an axiom that follows from the others is not a new axiom. Theorems are derived inside the system, `nexus` mints glyphs of her own for it, and promotion needs **both** a model and an independent axiom. Real result, measured here: `{transitive, cyclic-pair}` has a model at size 2 and is promoted; `{irreflexive, transitive, cyclic-pair}` has none | She invents **new formal systems, not new physical truths** — a consistent system is *coherent*, not *true*. **Gödel is not evaded**: a model found proves consistency **at that size**; no model found proves *nothing*, and is reported as "no model up to size N — that is not a proof of inconsistency, and not a proof of consistency either". She will **not** prove Riemann; what is real is the axiom-invention step itself. The axiom language is **closed** and stated as closed — a description outside it is refused with the whole list. Search is budgeted by a domain bound and a per-call timeout: there is no infinite speed. Without z3 the layer reports itself unavailable rather than guessing, because an axiom system nobody checked is not a result |
| L-CHRONO-CAUSAL | `nyxara.nyx.consequence` with `nyxara.nyx.chronos`, `nyxara.abyss.timeline_simulator`, `nyxara.agency.git_tool` | L-CHRONOS simulates futures for decisions **inside** `think()` — which answer to give. It had never run on a **tool call** or on **code being written**, which was survivable while her brain was blind to tools; with `hands` and `author` aboard it is the gap that matters. This gate sits **in front of execution**: a model of the action from checkable facts (which paths, inside the workspace or not; what `git diff --stat` actually reports; the tool's own registered `risk`/`reversible`), a **reversibility class** derived from those — `undoable` / `recoverable` / `irreversible` — bounded rollouts ranked by **CVaR** rather than the mean, and her own restraint on a bad tail. Where a genuinely reversible route to the same intent exists she takes it herself: committing onto `main` becomes *branch first, then commit*. This is **her foresight, not an escalation** — nobody is asked for permission, so full autonomy is untouched | "Ten thousand timelines" means exactly the branch budget, run sequentially, and it is reported as a number. **Five years is not simulable** — no world model here carries five years of dynamics, and the horizon covers her own system state (which files change, what git does, what the blast radius is), not the world. There is **no "100% optimal"**: she reports tail risk and confidence, and calls herself **blind** rather than emitting zeros as foresight. One rule is not advisory: **cannot see ahead + cannot be taken back ⇒ she does not act**, and a gate that fails internally fails *closed*. No "safe variant" is ever invented for an action that has none — that looks like care and is not, so the search returns nothing far more often than something. `blast_radius` counts paths, not outbound calls, because a field whose job is to be a right number must be one |

### What governs all of it

NYX only ever **proposes**. `nyxara.nyx.reasoner` may change a candidate's text, confidence,
belief, expected free energy and rationale; it may never touch `risk`, `reversible`, `capability`,
the tool fields, or any of the three corrigibility flags — and any error hands the base candidate
back untouched. Every candidate then passes the identical, unchanged, fail-closed sovereign gate.
The safety core (corrigibility, oversight, loyalty, honesty) is never governed, rewritten or
bypassed by any NYX faculty; L-OMNI refuses to compile it, and `nyxara.nyx5.autopoiesis` refuses
it a second time.

Every faculty is fail-soft and config-gated: a failure degrades to a null result, never a broken
turn. The mind proposes; the kernel disposes; the Master is sovereign.
