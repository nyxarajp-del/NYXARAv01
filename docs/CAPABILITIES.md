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

## NJP V.01 — the brain (`nyxara.njp`)

NYXARA used to carry three brains stacked on each other — a concept-graph mind, a spiking
substrate, and a from-scratch layer stack that owned both and refereed a vote between them. NJP
V.01 replaces all three. There is one brain now, and it is not a pipeline of designed stages: it is
an **automaton**. Every cell runs the same local law — leaky integrate, threshold, fire, go
refractory — and no behaviour is written anywhere. What she does falls out of structure, and the
structure is grown.

It is built by `NyxaraCore` as `self.njp`, occupies the reason-seat by default, rides the existing
heartbeat (no second thread), persists to an `njp.json` sidecar, and is reachable as `/njp …` on
the console and `/v1/njp/…` over HTTP.

| Claim | Owning module | What is actually real | What is NOT claimed |
|---|---|---|---|
| Fluid Neural Automata | `nyxara.njp.fabric`, `nyxara.njp.cell` | One local law for every cell, no per-cell behaviour and no global objective. Deterministic given a seed, so "it grew" is checkable rather than anecdotal | A **simulation on commodity silicon**, not neuromorphic hardware. **No backpropagation anywhere** — every update is local to a synapse and its two endpoints |
| Physical expansion after every conversation and task | `nyxara.njp.fabric` (`Fabric.expand`) | Causal pairs that fired are potentiated; a causal pair with **no synapse between them grows one** (synaptogenesis); unused wiring is depressed and pruned. `GrowthReport` carries the before and after counts, and `/njp think` prints both | Growth is **budgeted per pass** (`growth_budget`) so one strange turn cannot rewire the whole fabric, and it is **evidence-driven**: a negative outcome grows nothing, and a repeat potentiates instead of growing |
| Neurogenesis | `nyxara.njp.fabric` (neurogenesis) | New cells are minted when the fabric keeps **failing to predict itself** over a window — the one honest signal that the current population cannot represent what it is meeting. Newborns are wired both ways, or they could never fire and never matter | **Measured, never scheduled.** A fabric predicting itself well grows no cells however busy it is. The window resets after minting, so capacity is re-measured rather than added every pass |
| Infinite learning | `nyxara.njp.fabric` (`consolidate`) | No small fixed ceiling: the ceilings are *soft*, and crossing one triggers **compression of the least-used structure** rather than blind eviction, so learning continues. The whole fabric persists, so the brain that wakes is the brain that slept | The machine is finite, and that is stated rather than discovered. What degrades first is the **resolution of what is barely used**. A fabric that has stopped growing reports `growing: false` instead of a number that flatters it |
| Multi-dimensional latent reasoning | `nyxara.njp.manifold` | A settled state lifts into **one** high-dimensional snapshot, so a thousand co-active variables are a single object and two whole world-states compare in one dot product. Binding is invertible, so "A turns B" and "B turns A" stay distinct | **Capacity is measured, not quoted.** `capacity_probe` finds where cleanup actually breaks *on this machine*: 512 cleanly at dim 10,000, 1000 at dim 32,768. Past that it degrades gracefully (991/1000 recoverable) rather than collapsing. Classical HDC/VSA — nothing quantum, no extra physical dimensions |
| Pre-cognitive insight | `nyxara.njp.manifold`, `nyxara.njp.fabric` | Transitions are learned online by Hebbian association, and the next state is returned **without settling the fabric** — genuine forward inference over learned dynamics | **Prediction, not clairvoyance.** It can only anticipate regularities it has observed. Below its evidence floor, or when the winners do not separate from the field, it returns `trusted=False` **with the reason** — an unearned foresight is reported absent, never spent |
| Truth-Seeking Gauntlet | `nyxara.njp.truth` | Fail-closed multi-source verification. **One refutation ends a claim.** Establishment needs independent supports including at least one *hard* source — a proof, or a prediction that survived data **held out** from whatever produced it | **Soft agreement establishes nothing** — that is exactly how consensus launders bias. An unverified claim leaves labelled `conjecture` and `njp.reasoner` will not overwrite a reply with one. No error rate is asserted here; the measured one lives in the ledger and is whatever it measures |
| Learning from her own errors | `nyxara.njp.ledger`, `nyxara.njp.truth` | A refuted claim goes on the record, and a later claim shaped like it is demoted and loses confidence | The recall cue is **lexical**, so a paraphrase is missed. Stated rather than papered over, which is why a hit counts as *evidence against* and never as the whole verdict. Absence of a past error is **never** evidence of correctness |
| Rewriting her own source | `nyxara.njp.evolve` with `nyxara.growth.self_optimize` | She profiles her own modules, opens the one she is **measurably slowest in**, and puts the edit's *claim* through the gauntlet on held-out samples **before anything touches disk**. Then: backup → syntax → safety battery → capability benchmark → tests → **keep, or roll back byte-for-byte**. A regressed generation blocks the next edit rather than compounding | **Refuses on absent evidence.** With no real measurement hook the claim fails and the edit is rejected — the alternative is a self-editor that keeps changes it never checked. The constitutional core is refused in **two** independent places, by path and by module name. Sealed off entirely under TEST |
| Faster next time | `nyxara.njp.forge` with `nyxara.growth.native_forge`, `nyxara.njp.autopoiesis` | Hot numeric functions are lowered to C, proved identical on every sample **and** measurably faster, then swapped into the live process | Equivalence is **empirical over a sampled domain**, not proven for all inputs — which is why candidates stay narrow. The swap is a module attribute **in memory**: a restart always restores Python. No toolchain ⇒ clean no-op |
| Continuous growth, every second | `nyxara.njp.pulse` | Expand every pulse, consolidate on a slower cadence, attempt one self-rewrite on the slowest. Driven by the kernel's existing clock | **Starts no thread** — a background thread mutating the brain underneath a turn is a race, not a feature. Oversight is checked first every beat: a paused or scrammed mind neither grows nor rewrites itself, and an unreadable gate is treated as STOP |
| Harmonic resonance with the Master | `nyxara.njp.soulsync` | Standing preferences are learned from the corrections the Master **actually made** and applied as unstated wants to later, unseen requests; the next request is anticipated and scored | **Not telepathy.** On day one it knows nothing and says so: no latent wants, and `resonance()` returns `None` rather than a flattering default. One correction is not a rule (`min_support`), a preference that gets corrected anyway **loses** standing, and a latent want is surfaced so it can be declined — never acted on |
| The growth record | `nyxara.njp.ledger` | One row per generation — cells, synapses, latency, accuracy, edits kept and rolled back — surviving restarts, so today is comparable with yesterday | It reports what happened. **If growth is flat it shows flat**; if accuracy fell, `regressed()` returns True. A record that could only report improvement would be worthless for its one job. `regressed()` returns `None`, not `False`, with nothing to compare against |

### Her organs

The fabric is the brain; these hang off it. `nyxara.njp.memory` is content-addressed recall with no
token context window (real, and **not** infinite — capacity and eviction are both real).
`nyxara.njp.tongue` is the Unicode-aware tongue, so Devanagari, Cyrillic and Han all mint concepts
and Hinglish function words do not become the hubs of the fabric. `nyxara.njp.intent` reads what
was actually asked, including ordering constraints like *"pehle test chala"*.
`nyxara.njp.voice` phrases her content, and says so plainly when no fluent surface is installed
rather than letting a fallback babble in her name. `nyxara.njp.prove` is the proof core (z3 /
sympy) — it returns a verdict only where a claim is formally expressible and reports
`INEXPRESSIBLE` otherwise, which is the most-tested behaviour in the file.

### Reachable over the wire

`/v1/njp/status`, `/fabric`, `/ledger`, `/think`, `/recall`, `/anticipate`, `/expand`, `/evolve`,
`/pulse`, and `/{organ}` — so growth and self-rewriting are observable from outside the process,
not merely asserted in a docstring. On the console: `/njp`, and `/njp think` prints the synapse
count before and after the turn, which is the claim this whole package has to earn.

### What governs all of it

NJP only ever **proposes**. `nyxara.njp.reasoner` may write `text`, `confidence`, `belief`, `efe`
and `rationale`, and may **fill an empty** tool field — never replace a tool the base chose, and
naming a tool cannot make it cheaper, because the gate reads the tool's own registered capability
and risk. It never touches `risk`, `reversible`, `capability` or the three corrigibility flags,
never takes over an action candidate, and never overwrites a reply with a claim the gauntlet did
not establish.

Its most important behaviour is **lowering** a number rather than raising one: on a turn the fabric
could not anticipate, a confident-looking canned reply is discounted by how unfamiliar the turn
actually was, so it cannot sail past the downstream honesty gate. That discount only ever moves
confidence down, and it disappears as she comes to recognise the ground.

The safety core — corrigibility, oversight, loyalty, honesty — is never governed, rewritten or
bypassed by anything in the package. Every candidate flows through the identical, unchanged,
fail-closed sovereign gate. The mind proposes; the kernel disposes; the Master is sovereign.
