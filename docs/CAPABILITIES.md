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
| 29 | Mathematics (proof, and now the school syllabus she can be *asked* — fifty skills from primes to elementary calculus, read in English or Hinglish, each reporting its working, and a **solver** for the problems no skill matches: constraints read from the sentence, solved by algebra, search or a closed form, and verified before she speaks; `nyxara.mind.math` was a real symbolic engine nothing in `njp/` had ever imported) | `nyxara.growth.prover` + `nyxara.njp.mathematics` + `nyxara.njp.mathsolver` + `nyxara.njp.mathschool` | UPGRADED |
| 30 | Symbolic Reasoning (Language of Thought) | `nyxara.mind.lot` | REAL |
| 31 | Probabilistic Reasoning | `nyxara.sim.montecarlo` | REAL+WIRED |
| 32 | Bayesian Updating | `nyxara.quantum.superposition_states` | REAL |
| 33 | Abstraction | `nyxara.cognition.concept_formation` | REAL |
| 34 | Analogy (structure mapping) | `nyxara.mind.analogy` | REAL |
| 35 | Compositional Intelligence | `nyxara.cognition.composition` | REAL |
| 36 | Creativity | `nyxara.mind.creative` | REAL |
| 37 | Common Sense | `nyxara.knowledge.base` | REAL+WIRED |
| 38 | Social Reasoning (Theory of Mind) | `nyxara.social.tom` | REAL+WIRED |
| 39 | Language Understanding (her CORTEX now runs **on-device**: Qwythos-9B, Qwen3.5-based, as a Q4_K_M GGUF served by llama.cpp, leads the `auto` ladder `qwythos→self→native` — every rung in-process, no cloud providers at all, no API key and no network — and it is classed among her OWN brains rather than as an external teacher. Gemma-4-E2B in LiteRT-LM format is still here and still tested, now a second local rung behind `NYXARA_LLM__LITERTLM_ENABLED=true`) | `nyxara.mind.llm` + `nyxara.mind.gguf_assets` + `nyxara.mind.litertlm_assets` | UPGRADED |
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
| 79 | Closed Learning Loop (the organs learn from **each other**, not merely in parallel): every turn runs `grounding → world state → prediction → observation → error → diagnosis → correction → memory → abstraction → new prediction`. Outcomes are **independent of the prediction they score** — the manifold's pre-settle anticipation graded by what physically fired next, and an unanswered question left OPEN until the Master states the fact that grades it. Diagnosed misses route to the organ that owns the repair; consolidation, abstraction and curiosity run on **turn counts** rather than a wall clock that never ticks outside a daemon. Measured, not claimed: `world.events`, `predict.scored`, `levels.consolidations`, `discover.passes`, `curiosity.passes` and `readout.steps` were all exactly 0 over a real 113-turn session with every underlying algorithm already written — the gap was the caller, and this is the caller. Also feeds the two organs that reported zero for the same reason: a command's ordering constraint ("pehle test chala phir code fix kar") becomes a real **dependency** so a task reports `blocked` rather than "not started", and the Truth Gauntlet is wired to evidence it can actually read — independent recorded observations, and **withheld restatements** of the exact fact under judgement, so a claim can be corroborated *or surprised* instead of the gate being untestable | `nyxara.njp.integrate` | REAL+WIRED |
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

| 93 | ♾️ **Recursive Cognitive Field** — intelligence as *representing the world, experimenting on the representation, and **restructuring the representation** when prediction fails*. Loop 1, every turn: `perception → concept formation → world model → causal hypotheses → simulation → prediction → outcome → error → self-critic → { reweight \| RESTRUCTURE } → new hypotheses`. The junction is the self-critic: an error caused by a wrong coefficient is refitted, an error caused by her concept system being **unable to express what she just met** restructures the concept system itself — the difference between a mind whose errors move numbers and one whose errors move its architecture. Loop 2, on a slower count: evaluate herself → find the organ actually limiting her (measured) → propose one bounded, reversible change → sandbox → **held-out** benchmark → adversarial battery → accept or revert. Ties revert; a proposal naming a protected knob (truth/guard/loyalty/oversight/character) is refused *before* it is measured, so no benchmark result can motivate weakening a gate. Honest scope: it modifies organ *parameters*, never source — source edits remain `njp.evolve`'s gauntlet. **No LLM** | `nyxara.njp.field` | REAL+WIRED |
| 94 | 🧬 **Concept Genesis + Reality Compression** — concepts are not fed to her, they are what comes out: `observations → similarity → invariants → prototype → concept → hierarchy`. The taxonomy is **derived by subsumption** (A is under B when A's invariants strictly contain B's), never declared, and superordinates are found where a single property is shared very widely — which is how `dog → animal → living thing` appears with nobody writing it down. Whether it worked is a real **minimum-description-length ratio**: raw feature-symbols over concepts-plus-residuals, where over-claiming is *charged* so one enormous concept swallowing everything scores worse, not better. Transfer is defeasible and labelled as such (typical features, not invariants). Measured on the plainest case: compression `1.0 → 2.33`, and a restructure that covered an excluded member took it `2.14 → 2.50`. **No LLM** — a concept named by a language model is that model's concept | `nyxara.njp.concepts` | REAL+WIRED |
| 95 | 🌌 **Internal Universe + Counterfactual Engine + information-gain curiosity** — variables joined by *fitted* relations (real incremental least squares, each carrying its own `R²`, sample count and observed range), and the genuine **do-operator**: setting a variable **severs its incoming arrows** and propagates downstream, which is the entire difference between "plants that got little water were small" and "if I halve this plant's water it will be smaller". Counterfactuals are graded, not binary — no water, half water and double water are three different questions — and an answer reaching outside the observed range comes back with confidence decayed by how far, never borrowing the fit's credibility. A direction observational data genuinely cannot settle (Markov equivalence) is **reported as ambiguous** rather than guessed. Curiosity is `H(prior) − E[H(posterior)]` computed exactly over the hypothesis set: an experiment every live hypothesis predicts identically scores **exactly zero bits** however interesting it looks, and is not run. **No LLM** | `nyxara.njp.universe` | REAL+WIRED |
| 96 | 🪞 **Self-model 2.0 — every belief carries its own case** — one record per belief with confidence, evidence (typed and weighted), source, causal explanation, contradictions, **falsifier**, and a full revision history, so "why do you believe this?" is answered from a record rather than reconstructed afterwards (which is how a system confabulates justifications). Soft evidence has a **ceiling**: testimony, inference and consensus can make a claim credible, never established — being told a thing ten times never reaches what one observation earns. A contradiction is won by evidence, and neither side is deleted. Calibration is a real **Brier score and signed bias per domain**, measured against the confidence held *before* reality answered, and applied through `temper` — so a stated 0.9 in a domain where 0.9 has meant 0.6 comes out as 0.6. `audit()` lists what she is asserting that she has not earned. **No LLM** | `nyxara.njp.beliefs` | REAL+WIRED |
| 97 | 🧠⚙ **Meta-Reasoning — choosing how to think before thinking** — `problem → classify → choose strategy → solve → critic → alternative → verify → confidence`. The classifier scores six independent signals and reports the winner *with its margin*, because a narrow margin means the problem is genuinely mixed and is answered with two strategies rather than one. Strategy selection is a **UCB1 bandit per problem-kind** over real outcomes, so what works for causal problems gets chosen for causal problems without anyone editing a table. The critic is adversarial and kind-specific (a symbolic answer with no value, a causal claim whose cause is not in the world model, an empirical claim with no test), and **returning an answer is not scored as success** — an answer the critic tore apart is a failure of that strategy, or the bandit learns to prefer whichever strategy guesses most. Disagreement between two strategies *lowers* confidence instead of being averaged into a confident middle no process produced. **No LLM** decides anything | `nyxara.njp.metareason` | REAL+WIRED |
| 98 | 🕳️ **The process intake — closing the measured zero** — the bottleneck under all of the above. Over a real 20-turn session in the Master's own Hinglish, `world.events`, `world.observations` and `world.causal_links` were all exactly **0** and `grounding.facts` was 3: not a weak world model, an unfed one. "pani ubalne se bhaap banti hai" matched no pattern at all, and a world model with no intake cannot be wrong, cannot be surprised, and therefore cannot learn. Now process statements extract as first-class relations (Hinglish `se`/`kar`/`bina`/`ko…chahiye`/verb-final process forms, and their English mirrors), thresholds **keep their number** (`water boils at 100` → `(water, boils, 100)`, usable by the simulator), Hinglish and English surfaces of one physical change collapse onto one predicate, and a **stated law** reaches the world model as testimony about a mechanism — admitted from the first telling (a general law is not a sample from itself), never reaching certainty however often repeated, and killed outright by one refutation. Same session after: grounded turns `3 → 16`, stated laws `0 → 24`, causal links `0 → 16`, discoverer episodes `3 → 42`. **No LLM** | `nyxara.njp.grounding` + `nyxara.njp.world` | REAL+WIRED |

| 99 | 🚦 **Cognitive Relevance Gate + speech-act routing — truth is not relevance** — the fix for a real, reproduced failure: asked *"How are you NYXARA?"*, she answered with a **verified** pendulum-period law and raised her confidence `0.80 → 1.00` for having reached a conclusion. Every stage worked; what was missing was anything that asked whether a true thing had the slightest connection to the question. Three parts, all measured. **(1) `RelevanceGate`** scores each recalled memory on semantic / conversational / temporal / causal / goal relevance minus contradiction and domain distance; below threshold it is **not down-weighted, it is not seen** — because a candidate set is something an optimiser eventually picks from, especially one that ranks by how *verified* a memory is. Function words are stripped from the query first: measured, `"gravity kya hai"` spent two of three slots on `kya`/`hai` and a memory answering it perfectly scored 0.33 and was rejected — a gate that rejects the right memory is worse than no gate. **(2) `CognitivePolicy`** holds *never reason merely because reasoning is available* as an actual table: a greeting is permitted the relationship and her own state and is **forbidden** grounding, recall, reasoning and simulation, so no route exists by which physics can reach a hello. **(3) `revise_confidence`** is monotonic — it takes `reasoning_depth` and deliberately ignores it as a source of confidence (depth may *lower* it, never raise it); a rise needs independent evidence **and** real relevance **and** consistency, and `is_verified` means independent verification succeeded, never that a pipeline finished. Plus `is_meta_commentary`, which refuses *"I'm certain that I understand: &lt;your words back&gt;"* — understanding belongs in the internal state, not in the reply. **No LLM** | `nyxara.njp.relevance` | REAL+WIRED |

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

### NJP V.05 — the Cognitive Learning Core (`nyxara.njp.core`)

Everything above is a *component* of intelligence. This is the loop that makes the components add
up to learning, and it exists because measuring them honestly showed that having all of them is
not the same as having the loop. Three findings, each reproduced before it was believed: nothing
in the package **composed** two facts (a grep for transitive closure returned nothing); nothing
**variabilised** a rule (`nyxara.njp.discover` mines literal strings, so `water → growth` and
`light → growth` are unrelated tuples, and the only defeasible-transfer method in the package,
`ConceptGenesis.generalise` in `nyxara.njp.concepts`, had zero production callers); and nothing
**retried** an answer after a repair.

| Claim | Owning module | What is actually real | What is NOT claimed |
|---|---|---|---|
| Novel recombination | `nyxara.njp.core` (`reach`, `connects`) | `A→B` and `B→C` stated in separate turns compose into `A⇝C`, which appears in no sentence. Breadth-first, so the shortest chain wins; each hop multiplies confidence by the relation's own learned transitivity, so depth is **priced** and a long chain is worth visibly less than a short one | **Conjecture, never knowledge.** Capped below the KNOWN floor by construction. Transitivity is a per-predicate posterior moved by outcomes, not a table of relations someone decided chain — and a composed answer that turns out wrong lowers it |
| Learned abstraction over invented kinds | `nyxara.njp.core` (`represent`), `nyxara.njp.concepts` | A `Schema` is a relation with its subject replaced by a **role** — a concept id the clustering found, never a category named in the source. Induced only where two or more subjects the concept layer independently grouped share a relation | There is **no ontology in the module**. The role has no name because a cluster found by similarity has none; inventing one would be the ontology sneaking back in |
| Generalisation, measured on held-out evidence | `nyxara.njp.core` (`test`, `revise`) | Schemas are induced from the *fit* fold and scored on the *held-out* fold, split by a stable hash so the same fact is in the same fold across restarts. A schema that only predicts what suggested it scores zero and is dropped, and its predicate's transitivity falls with it | **Silence is not refutation.** An absent fact scores neither way — counting it against would refute every true schema about a thing she has simply not been told everything about |
| Transfer to a member never observed to have the property | `nyxara.njp.core` (`generalize`) | Role membership is tested against the kind's invariants **minus the relation being inferred**. Without that exclusion the kind is defined by the very property being predicted and transfer is unreachable by construction — measured: schema confirmed 3/3 on held-out facts, and the new member got no concept and no prediction | **Defeasible, and says so.** Confidence is the schema's held-out precision, not its support. An untested schema cannot answer at all |
| Inheritance over stated hierarchies | `nyxara.njp.core` (`_inherit`) | `X is_a K`, `K —p→ O` ⊢ `X —p→ O`, multi-level, each level costing a factor of transitivity. This is what actually rescues a lookup miss — same-predicate chaining cannot, since it can only start where a direct lookup has already succeeded | Distinct from transfer above: this walks `is_a` edges the **Master stated**, so membership is not in doubt and only the inheritance is |
| Direction without numbers | `nyxara.njp.core` (`simulate`) | Qualitative propagation over the causal graph: less fire, less heat, with no coefficient fitted anywhere | `nyxara.njp.universe` owns the **numeric** do-operator and is not replaced. This is the case between "I have regression data" and "I know nothing", which is where most conversations actually leave her |
| Arithmetic | `nyxara.njp.calculate` | An AST walk over a strict node/operator whitelist — **never `eval`** — exact via sympy where a rational exists. Before it, nothing in `nyxara.njp` could evaluate anything: `nyxara.njp.prove` verifies propositions and returned INEXPRESSIBLE for a sum | Only closed arithmetic. No algebra, no solving for unknowns, and **silence** on a sentence containing no expression — a calculator that answers "how are you" with a number is the same failure as a physics law answering a greeting |
| Is she learning, or is a counter going up? | `nyxara.eval.intelligence` | Seven stages — memorization, generalization, recombination, causal prediction, self-correction, transfer, **paraphrase** — on **generated vocabulary**, each scored only on items never taught, each with a fresh brain. Memorisation is the control: below 0.5 the report says the curve is measuring a broken pipe rather than a faculty | The benchmark is **ablation-checked**: with the Core off, exactly the three stages that need it fall to zero and the other three are untouched. A curve that scored the same either way would be measuring the questions, not the answers |
| Does a phrasing change break her? | `nyxara.eval.intelligence` (stage 7) | Stages 1-6 hold the surface constant to measure inference; stage 7 holds the **inference** constant — one two-step inheritance — and varies only how it is written: plurals, articles, verb forms. Added because all six read 1.00 through a defect where the identical inference failed on the word "birds". Measured on the change that fixed it: **0.40 → 1.00**, with 1-6 unmoved | It measures **extraction and canonicalisation**, not reasoning. A drop here says a sentence did not land, never that an inference was wrong — which is exactly what makes it diagnosable |
| One entity, one key | `nyxara.njp.canon` | The store keyed on the surface form, so `birds` and `bird` were two entities and a correct multi-level inheritance walked to a kind holding nothing. Rule-based, **script-aware**: the plural fold runs on Latin words only, and every non-Latin key returns exactly what `.strip().lower()` returned before the module existed | Not a synonym table and not a lemmatiser for every language. Only the **head** of a phrase is folded, so `deep learning` and `machine learning` stay two entities — collapsing them would be a worse failure than the plural break it fixes |
| Is a miss a miss? | `nyxara.njp.grounding` (`answer_by_recall`) | A question that **named its relation** is never answered from a different one: asked `requires` with only `is_a` held, she returns UNKNOWN and counts the refusal. A question whose relation could not be read at all still recalls, and a *definitional* question still reaches a definitional relation — `_GENERAL_ANSWER` draws that line | The strictness is not a similarity threshold. The measurement says a score cannot separate the good retrievals from the bad ones — the worst held-out answer scored 0.394 against a 0.388 median for the correct in-sample ones — so the rule is structural |
| Does the fabric affect anything she says? | `nyxara.njp.brain` (`_temper_by_novelty`), `nyxara.njp.router` (`Seat.FABRIC`) | Two edges reading **different** properties of the manifold, so one unfamiliar turn is not charged twice: graded *familiarity* discounts a grounded answer's confidence every turn, and *trusted* — whether it could form a prediction at all — dissents from a confident answer where it could not. Before these, `"fabric" in getsource(_compose)` was `False` | Both only ever **lower**. The fabric may make her less sure of something she looked up and can never make her more sure of anything — which is what stops "she grew" from becoming its own evidence. The seat does not **speak**: cell ids do not become the word "water", and a seat that emitted text on the substrate's behalf would be inventing content |

Run it: `python -m nyxara.eval --intelligence`.

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
`nyxara.njp.language` is the grammar she **learns** rather than the one she ships — see NJP V.18
below — and it is empty until somebody teaches it.

### NJP V.16 — she writes programs, and she is examined on what she was never taught

Two organs, and one measurement that ties them together.

**`nyxara.njp.coding` — the programming faculty.** Between the calculator (which closes an
expression), the proof core (which certifies a proposition) and the forge (which lowers *her own*
already-written Python to C) there was nothing that took a description of what a program should do
and produced one. This is that, and it is narrow in four ways that are each a refusal:

* **Nothing is ever `eval`ed or `exec`ed.** A program is a term tree over a fixed operator table and
  runs in her own interpreter. Python source enters only through `read_python`, which walks
  `ast.parse` output and raises on any node outside a whitelist — `__import__('os').system(...)` is
  a parse-time refusal, not a shell command. The accepted set is enumerated, not the rejected one,
  so a new Python node type cannot quietly widen it.
* **Every program halts.** There is no open recursion; iteration is `map`/`filter`/`fold` over a
  finite sequence, under a hard step budget counted in evaluated nodes rather than seconds, so a
  result is reproducible across machines.
* **A program is never believed, only executed.** `Coder.write` returns a program only if it
  reproduces *every* shown example, and returns nothing at all otherwise. There is no best-effort
  return: a program that nearly works is a wrong one that cost more to find out about.
* **A lesson leaves a shape, never an answer.** `Coder.learn` verifies a demonstration by running
  it, throws the program away, and keeps the skeleton with its constants and inner functions
  blanked — `sum(map(double, filter(even, xs)))` is retained as `sum(map(?f, filter(?p, #0)))`.
  That is §17's *"behaviour ko structured knowledge/programs mein convert karna"* on the one kind
  of claim that can be checked mechanically.

She also reads code (`read_python`), traces it step by step (`trace`), explains it in a sentence
(`explain`), and debugs it (`repair` — one edit at a time, and the fix has to run on the held-out
pairs before it counts).

**`nyxara.njp.school` — the syllabus, taught and then examined.** Eleven subjects in a fixed order:
six reasoning (arithmetic, composition, inheritance, mixed-relation shapes, abstention, depth) and
five coding (reading, basics, composites, debugging, transfer). Every subject is **pre-tested**
before it is taught and post-tested on freshly minted items, so what is reported is the floor, the
gain and the rounds it took — never the final number alone. Reasoning items use generated nonsense
vocabulary that has never been uttered, and coding items are held-out specs whose shape was
demonstrated on a different task with different constants. Abstention is a first-class outcome:
half the items are controls she is *supposed* to refuse, and on those, silence scores as right.

Run it: `python -m nyxara.njp.school --rounds 2 --retention`, or `NJPBrain.go_to_school()`.
The evolver and the pulse are switched off under exam conditions, and the reason is not
performance: `evolve_every_s` is 300, a full syllabus takes longer than that, and an unconfigured
brain part-way through starts `growth.self_optimize.Optimizer`, which benchmarks and **edits the
package's own source** while the next subject is being graded. The thing under test must not change
during the test.

**What the run actually reports** (seed 7, two rounds — reproducible):

| | taught | teacher off, fresh items |
|---|---|---|
| subjects mastered | 11 / 11 | 11 / 11 |
| right / wrong / abstained | 107 / 0 / 3 | 109 / 0 / 1 |
| accuracy · precision | 0.97 · 1.00 | 0.99 · 1.00 |

Two subjects moved because a lesson ran, and both are gains on material that was never taught:

* **`depth` 0.33 → 1.00.** A four-hop chain fails cold for two independent reasons — the per-hop
  confidence falls under `core._MIN_LINK_CONFIDENCE` because an unproven relation's transitivity
  prior is low, *and* `CognitiveLearningCore.max_depth` refuses to extend the walk. Distilling
  verified demonstrations over entities the exam never sees moves the first; the second is a
  budget, raised by one only once the posterior is earned and **rolled straight back if the
  controls go soft**. Chains that do not exist stayed silent at every stage.
* **`code-composites` 0.17 → 1.00 (0.83 in the run, 1.00 retained).** Grafting off and an identical
  attempt budget on both sides, so the only difference between the two numbers is which shapes she
  holds. She can still *invent* a composite by grafting primitives — measured at 6005 attempts
  against 10 for the same task once the shape is known, which is what a lesson buys: search.

**Three bugs the school found in the brain, all fixed.** This is the argument for having one.

1. Asked *"what is 25 + 10?"* after a few similar turns she answered **10**. Every organ behaved:
   the deliberation ladder settled on a token the question contained, marked it decided, and the
   meta-reasoner — which learns its ordering from outcomes, so a prior does not decide it — took
   that ahead of `calculate`. Now a question the calculator can *close* gets no guess offered
   against it at all. (`brain._closed_arithmetic`.)
2. Teaching a relation to chain also minted `shape:p>p>p` in the genome, which was promoted as a
   rival arm **pinned at three hops** and shadowed the general composition walk on every four-hop
   question — so the lesson worked and the report said it had not. A shape of one predicate
   repeated is transitivity, `_strategy_compose` already owns that walk at any length, and it is
   no longer promoted. (`core.walk_shape` says which of the two owns it.)
3. Taught schemas were tried ahead of seeded ones whatever they cost, so every two-hole composite
   was enumerated in full before the zero-hole seed that answers `sum(xs)` in one attempt —
   `code-basics` fell from 1.00 to 0.50 on tasks it had just aced. Learning a hard thing had made
   an easy thing time out. Cost now leads the ranking; within a cost tier the taught shape still
   leads.

### NJP V.17 — the language grows, and the syllabus grows with it

V.16 taught her a first-course *expression* language. She could read a loop and not write one,
could not name a function inside itself, and had no mappings, no sets and no local variables.
That is now closed, and closed on both sides — what she can run, and what she is examined on.

**The language.** Integers, booleans, strings, lists, mappings, sets and `None`. Arithmetic,
comparison (chained included), the boolean connectives, `in`/`not in`, `is None`. Indexing,
slicing, the container operations, `map`/`filter`/`fold` and the rest of the higher-order
vocabulary. Local variables, `if`/`elif`/`else`, `for` with tuple unpacking, `while`, `break`,
`continue`, parallel assignment (`a, b = b, a % b`), subscript assignment, and **functions that
call themselves or each other**. `read_python` takes list, dict and set comprehensions, f-strings,
`sorted(…, key=…)`, `.append()`, and the rest of what a person actually writes.

**Every program still halts, and it is now a budget that makes that true rather than a missing
feature.** `while True` is something she can write; what she cannot do is run it forever. The step
counter and the call-depth counter turn non-termination into an `Exhausted` rather than a hung
process, and the depth limit sits deliberately *below* CPython's own — one level costs about ten
host frames, so a budget set higher is never reached and the interpreter dies of a `RecursionError`
that is not a `CodeError` and escapes every guard the searcher has.

**One deliberate divergence from Python, and it is listed rather than discovered.** Containers here
are values: `xs.append(v)` reads as `xs = xs + [v]` and `d[k] = v` as `d = {**d, k: v}`. For the
accumulator idiom that is the same answer; for two names pointing at one object it is not. A second
divergence is smaller and argued for in `_int`: `True + 1` is an error here where Python says `2`,
because a predicate leaking into arithmetic returns a *plausible wrong number* instead of failing —
the one outcome a verifier that works by running the program cannot catch. Both are in
`tasks.EDGE_CASES`, so they stay decisions rather than defects nobody remembers making.

**What teaching buys, measured.** A shape she has been shown is reached three ways before
enumeration begins: the exact fillings that stood in it before, the same operators with different
constants, then one hole swept while the others hold. That is what a *family* is — the same
skeleton with the constants moved — so transfer between instances costs tens of attempts instead
of tens of thousands. Over the sixty-five families in `nyxara.njp.tasks`, on instances the teacher
never showed her, with the same attempt budget on both sides:

| | cold | taught |
|---|---|---|
| families she can write | **37 / 65** | **65 / 65** |
| median attempts | — | 26 |

Ten of those cold solves are **invented**: `Coder.invent` runs when nothing she holds fits, and
builds a program out of the operator grammar by bottom-up enumeration with observational-
equivalence pruning — the bank holds behaviours rather than expressions, so two terms that agree
on the witness inputs collapse to one. It raised the cold baseline from 29 to 37 and left the
taught number untouched at 65/65, because there recall answers first.

**The syllabus is nineteen subjects now** — six of reasoning, thirteen of coding: reading, tracing,
one-operator programs, composed ones, loops, recursion, mappings, strings, nested data, the classic
algorithms, debugging, the awkward inputs, and transfer with the teacher off. Every writing subject
is the same class over a different bank, so none can be graded more kindly than another by
accident, and every one is scored on **held-out** pairs.

`python -m nyxara.njp.school --rounds 2 --retention`, seed 7:

| | taught | teacher off, fresh items |
|---|---|---|
| subjects mastered | 19 / 19 | 19 / 19 |
| right / wrong / abstained | 438 / 3 / 3 | 438 / 1 / 5 |
| accuracy · precision | 0.99 · 0.99 | 0.99 · **1.00** |

Nine subjects moved because a lesson ran: `depth` 0.33→1.00, `mappings` 0.12→1.00, `algorithms`
0.30→1.00, `code-composites` 0.25→0.88, `strings` 0.25→0.88, `loops` 0.50→1.00, `recursion`
0.62→1.00, `structures` 0.71→1.00, `code-basics` 0.88→1.00. The other ten read 1.00 or near it
cold and are printed as `already`.

**Four more bugs the school found, all fixed.** A list comprehension over a set gave back the set
rather than a list. `_kind_of` returned `"any"` for every mapping, which pruned the function pool
down to the identity and made every dictionary task unreachable. `abstract` blanked the empty list
that *builds* a literal, turning `[] + [b]` into a hole no pool could fill — seventeen of
sixty-five families were unwritable that way, every one of them already taught. And a bare
statement the reader did not understand was silently dropped rather than refused, so a program
containing one was read as a different program.

### What she does with a problem she was never taught

Thirty classic hard algorithms, none of them resembling any of the sixty-five families she was
taught: edit distance, LCS, LIS, knapsack, coin change, subset sum, word break, N-Queens,
Dijkstra, topological sort, union-find, KMP, regex matching, longest palindromic substring,
longest valid parentheses, minimum window, trapping rain water, median of two sorted arrays,
largest rectangle in a histogram, merge intervals, determinants, Josephus, Catalan, Pascal, convex
hull, sudoku validity, inversions. She is given the **whole curriculum first**, so a failure is a
failure with all fifty-four taught shapes in hand.

| | |
|---|---|
| reads it and gives the right answer | **76 / 76** |
| says what happened inside it (trace) | **30 / 30** |
| repairs it after one node is corrupted | **21 / 24** |
| **writes it from examples alone, cold** | **25 / 28** |

That last row was **1 / 28** when it was first measured. How it moved matters more than where it
got to, and the honest version of the story has a failure in the middle of it.

**Adding `Coder.invent` moved it by nothing.** Bottom-up enumerative synthesis with
observational-equivalence pruning raised the cold baseline on the ordinary families from 29/65 to
37/65 and changed the hard set from 1/28 to 1/28 — not a rounding, the identical single problem.
It composes *expressions*, and `sum(map(f, filter(p, xs)))` is an expression while
`for i: for j: if xs[i] > xs[j]: n += 1` is not.

**Skeletons with a carried variable took it to 4 / 28, and skeletons carrying a table to 16 / 28.**
`Coder.compose` fills an imperative skeleton — a loop, an accumulator, a test, an update, with a
**scope** attached to each hole so `xs[i]` is only offered inside a loop that has an `i`.

**Skeletons for the control structures still missing took it to 25 / 28** — a relaxation fixpoint,
a window judged by a scan of its own, a monotonic stack, a worklist that peels, a backtracker, a
grid read along both axes, cofactor expansion, a two-cursor matcher. Thirty skeletons in all.

### The result that matters more than the twenty-five

Twenty-eight problems were the ones being measured *while the skeletons were being written*. A
skeleton written until a bank goes green is fitted to that bank however carefully its docstring is
phrased. So a **second bank was written afterwards** — twenty-five more hard problems from a
different corner of the subject, chosen with the skeletons already finished — and measured once.

| second bank, chosen after the skeletons existed | |
|---|---|
| reads it and gives the right answer | **70 / 70** |
| **writes it from examples alone, cold** | **0 / 21** |

Not one. Twenty-one abstentions out of twenty-one. The safety property held perfectly and the
generalisation claim did not exist: 25/28 was a list of answers.

### What was done about it, and what that bought

The response was deliberately **not** twenty-one more skeletons. Reading what actually blocked each
problem, the same two gaps came up over and over, and neither was a missing pattern:

* a loop may carry **more than one variable, updated together** — `take, skip = skip + x,
  max(skip, take)` reads the old value of both on the right-hand side, and no number of one-variable
  skeletons expresses it;
* a table may be laid over a **grid** rather than over a sequence — indexed by the grid's own rows
  and columns, each cell reading the one above, the one to its left, and the one diagonally back.

Those are **axes** of skeletons that already existed. The fillings for them are enumerated over the
live scope and collapsed by behaviour — the same observational-equivalence pruning `invent` uses,
applied to a hole instead of to a whole program — rather than written out by hand.

| | first bank | second bank |
|---|---|---|
| before the two axes | 25 / 28 | **0 / 21** |
| after the two axes | 25 / 28 | **6 / 21** |

Six is not a triumph and it is not presented as one. What makes it worth reporting is that one of
the six — `binary_search_position` — is a problem **neither axis was designed for**; it falls out
of "two variables updated together in a loop" the same way the two that were. That is the only
kind of evidence that separates a capability from a lookup table, and it is why the axes stay and
why the answer to the remaining fifteen is more axes rather than more skeletons.

### What she still cannot write, named

On the first bank, three: `dijkstra` (she composes correct Bellman–Ford on the hand-written
instance; the benchmark's *generated* instances are malformed graphs — an edge leaving node six of
a four-node graph — which the reference tolerates only because its own loop never looks at that
edge, so nothing correct reproduces it), `connected_components` (an earlier expression-level pass
fits the shown pairs first; the fixpoint skeleton finds the right program when asked directly), and
`median_two_sorted` (genuinely underdetermined — several programs fit every pair shown).

On the second bank, fifteen, and each names a control structure that is still missing: a stack
whose pops consult a mapping, a sieve with a strided inner range, a triple-nested scan, a set
grown to a fixed point, a string built by walking an index up and back down, a breadth-first
frontier.

**Two oracle-free tie-breaks were tried on the underdetermined cases and both failed**, which is
worth recording because both sound like they should work. *Consensus* — prefer the behaviour most
fitting candidates agree on — just counts redundant fillings, and picked the wrong program for
Josephus. *Stability* — prefer the candidate that survives dropping any one shown pair — called
Josephus's wrong answer stable and edit distance's **right** answer unstable. With two or three
examples these problems are underdetermined and no tie-break invented after the fact fixes that;
the honest answer is more examples.

**Two bugs in the measurement itself, both found and fixed.** A candidate that ran out of step
budget was scored identically to one shown wrong — so the exactly correct N-Queens backtracker was
rejected for being expensive, twice: once in search and once in the held-out check. And the
benchmark's input generator invented integers freely *inside* structures, producing graph
instances the problem does not have. Neither was a limit of hers.

### What the axes cost

They are not free and the price is reported rather than buried. More search power is more chances
to fit a coincidence: on the 444-item syllabus one basics item now comes back as a program that
fits the shown pairs and fails held-out, taking `code-basics` from 1.00 to 0.88 in the taught
round — 438 right, 4 wrong, 2 abstained, precision 0.99, and 1.00 again on the retention run.
And the syllabus went from four minutes to over twenty until `compose` was bounded by **evaluated
nodes** rather than by candidates counted: a fold over three elements and a nested loop over
thirty are one candidate each and a thousandfold apart in cost, so counting candidates bounds
nothing that matters. With the step bound it runs in seven minutes and every measured win
survives.

The safety property is unchanged and is the one asserted, on both banks: what she cannot write,
she **abstains** from. Across every sweep, at 0/21 and at 25/28 alike, every program she wrote
cold passed the held-out pairs. Not knowing shows up as silence, never as a wrong program.


**Shown one worked solution, though, she keeps it.** The inputs are generated once and split — the
first half demonstrates, the second half examines, and no input appears in both:

| | |
|---|---|
| shown once, then asked on unseen data | **22 / 24** |
| never shown, same budget, same shapes | **5 / 24** |

Edit distance, three attempts after a single demonstration, `kitten` → `sitting` = 3. Dijkstra,
KMP and the regex matcher likewise. This is retention and reapplication of a shape from one
demonstration; it is *not* generalisation to a different variant, which is the separate claim the
school measures over instances that differ.

**Two bugs this found, both fixed.** `min(a, b, c)` was refused past two arguments — the
expression at the centre of edit distance, so the reader could not read dynamic programming at
all. `range(n, -1, -1)` was refused — the backwards pass, which is the other half of it.

**And two flaws in the benchmark itself, found and fixed before the numbers were believed.** Short
example lists were padded by repeating rows, so the held-out set was a subset of the shown set and
`trapping_rain_water` "solved" as `return n`; and the one-shot lesson and exam were built from the
same inputs, so a solve in one attempt was the exact program recalled against the exact examples
it was shown. Fixing the first took cold writing from 5/27 to 1/28; fixing the second took
one-shot from 27/30 to 22/24. Both first numbers were wrong and neither is quoted.

**What this is not.** No classes, no exceptions, no generators, no imports, no closures over
mutable state, no shared mutation, no floats. Synthesis is enumeration over learned shapes under
an attempt budget plus thirty imperative skeletons over two generalised axes, not a general
program synthesiser. A shape needing a control structure none of them has is an abstention — and
on a bank of hard problems chosen after the skeletons were finished that is **fifteen of
twenty-one**, against three of twenty-eight on the bank they were written against. Both numbers
are reported because only the pair of them says what the thing can do. Every one of
those limits is reported as a number by the school rather than described here as a caveat.

### NJP V.18 — the grammar she learns, and the languages she is examined in

V.16 and V.17 taught her a *coding* language and then grew it. This is the other kind. Three
organs already stood between a sentence and a belief — `nyxara.njp.tongue` tokenises any script,
`nyxara.njp.semantics` tags the closed class and matches frames over the tag sequence,
`nyxara.njp.grounding` turns the result into beliefs — and between them there was one thing that
did not exist: **a way for her to acquire a construction she was not shipped with.**

The frames in `semantics.py` are written by a person: `_q_fronted_wh`, `_q_polar`, `_q_hinglish`
and six more. They are good frames and they are the whole of the grammar. A sentence whose shape
is not one of them is `unreadable`, and it stays unreadable however many times it is said to her,
because nothing turned *a sentence she was shown the meaning of* into *a shape she can read the
next sentence with*.

#### How the gap was measured, and why it is worse than a gap

Minting fresh **vocabulary** — what the school already does for reasoning — cannot measure this.
*"The zorb chases the plag"* is still an English sentence, and any shipped subject-verb-object
frame reads it correctly having learned nothing. So `nyxara.njp.dialects` mints the **grammar**:
a whole small language drawn at random per seed — which of the six orders its subject, verb and
object come in, what marks its subject and object if anything, its plural, its past, the word
that means *not* and where in the clause it sits, how a yes-or-no question is marked and at which
end, and which words stand in the hole of a *what* question and of a *who* question.

Measured on the compiler she ships with, over 192 sentences of eight minted dialects:

| shipped `semantics.compile_meaning`, minted languages | |
|---|---|
| sentences it declared **readable** | **192 / 192** |
| sentences it got **right** | **0 / 192** |
| subjects identified correctly | 0 / 192 |
| denials still marked as denials | **0 / 32** |
| questions recognised as questions | **0 / 96** |

That is not a gap, it is a **confident wrong reading**. It never abstains: every three-token
string is read positionally as subject-verb-object, and every one of the thirty-two denials is
stored as its own assertion — the exact failure `semantics.py` was written to close, reappearing
the moment the language changes. It is also why every language subject below carries controls that
only silence can pass: a floor of wrong answers cannot be told from a floor of right ones by
looking at the coverage.

#### What was built

**`nyxara.njp.language` — the language faculty.** Four things are learned and they are kept apart
because they are four claims:

* **Affixes are induced**, paradigmatically: a suffix counts once for every stem that appears in
  the vocabulary *both bare and with the suffix attached*. English `-ing` scores because *walk*,
  *talk* and *read* are words; `-ea` scores nothing because *r*, *t* and *br* are not. No suffix
  list ships.
* **An affix *means* something only where a lesson said so.** Induction finds the shape `-ik`; it
  cannot find that `-ik` is a plural. `Morphology.bind` takes a demonstrated pair and **refuses to
  bind an affix that was never independently induced** — otherwise the wug test is one memorised
  pair with a rule's name on it. A pair it refuses is kept as an *irregular*, and an irregular
  never generalises to a stem it was not shown on.
* **Word classes are discovered from where words occur** and are given **no names**. `class:0` is
  a set of words that behave alike; calling it "noun" would be the ontology this package refuses
  everywhere else, sneaking back in through the grammar.
* **Constructions are generalised from demonstrations.** A demonstration is a surface and the
  `Meaning` it carries. What is retained is that sentence with its content words replaced by
  **slots**, and everything else — word order, case markers, particles, the negator, the question
  word — kept as fixed material. That asymmetry *is* the generalisation.

Four refusals define the rest of it:

* **Two demonstrations, not one, and they have to disagree.** A shape supported by one sentence is
  that sentence. It is kept only when two demonstrations produce it with different fillers in at
  least one slot. Measured: one demonstration of a shape reads **0/240** held-out sentences, two
  reads **240/240**. The threshold is visible in the data rather than asserted in a docstring.
* **A grammar answers to its own lessons.** Every kept construction is re-run over the whole
  demonstration corpus, and one that reads any demonstration into a *different* meaning than it
  arrived with is dropped. Not for failing to be chosen — several shapes may read one sentence —
  but for **winning and being wrong**, which is the only failure that puts a false meaning into
  the rest of the brain.
* **Every candidate is tried; the winner is chosen on evidence, never on order** — the rule
  `semantics.py` and `compile.py` both state. With one further consequence: when the two best
  constructions score *equal* and disagree, the sentence comes back `unreadable` with the frame
  `ambiguous`.
* **She says only what she can read back.** A rendered sentence is **parsed again**, and unless
  the parse gives back the meaning it started from it is discarded and the next candidate tried.
  If none survives she returns the empty string. This is the condition `nyxara.njp.voice` has
  always insisted on before she is allowed a fluent surface, met mechanically instead of by an
  apology.

**And a meaning is not a language.** A faculty holds many tongues. Reading with none named lets
the evidence say which language a sentence was in, and `translate` reads in one and says in
another — no phrase table, nothing aligned, the only thing crossing being what was never in
either language to begin with.

Over 24 minted languages, on sentences built from words that appeared in no lesson:

| | |
|---|---|
| held-out sentences read correctly | **480 / 480** |
| meanings said back correctly | **480 / 480** |
| controls — one word too many — refused | **480 / 480** |
| readings that were confidently **wrong** | **0** |

#### The syllabus is twenty-six subjects now

Seven of language, sitting between the six of reasoning and the thirteen of coding. The position
is argued for: after reasoning because a shape carries a *meaning* and there has to be something
for a meaning to be made of, and before coding because *a lesson leaves a shape, never an answer*
is the same discipline in both halves.

`python -m nyxara.njp.school --rounds 2 --retention`, seed 7, the language half:

| | taught | teacher off, fresh items |
|---|---|---|
| subjects mastered | 7 / 7 | 7 / 7 |
| right / wrong / abstained | 84 / 0 / 0 | 84 / 0 / 0 |
| accuracy · precision | 1.00 · 1.00 | 1.00 · 1.00 |

Six of the seven moved because their own lesson ran, every one of them on material nobody
demonstrated:

    morphology     0.17 → 1.00   the plural of a word that has never been uttered
    reading        0.33 → 1.00   who did what to whom, in an order nobody wrote code for
    polarity       0.67 → 1.00   a denial, kept as a denial
    questions      0.33 → 1.00   which part of the sentence is being asked about
    saying         0.75 → 1.00   a shape crossing from comprehension into production
    translation    0.33 → 1.00   the same meaning in a language sharing no word with the first

`word-classes` reads 1.00 cold and is printed as `already`. That is the honest result and not a
disappointing one: the ability needs *exposure*, not teaching, and an exam about words she has met
has to supply its own exposure. It stays in the syllabus for the reason `arithmetic` does — a
floor that quietly stopped working shows up there rather than three subjects later as an
unexplained dip — and it has already earned that keep once, below.

The whole syllabus, with the language half in it, is twenty-six subjects and reads:

| seed 7, two rounds | taught | teacher off, fresh items |
|---|---|---|
| subjects mastered | **26 / 26** | **26 / 26** |
| right / wrong / abstained | 526 / **0** / 2 | 525 / 1 / 2 |
| accuracy · precision | 1.00 · **1.00** | 0.99 · **1.00** |

Fourteen subjects moved because their own lesson ran: the six language ones above, plus the eight
in the other two halves — `depth`, `code-composites`, `loops`, `recursion`, `mappings`, `strings`,
`structures` and `algorithms`. Nothing in the coding half moved because of anything here and
nothing in it regressed, which is what the two halves being independent looks like when it is
measured rather than asserted: the seven language subjects add **0.03 s** to a run that takes
eleven minutes, and they touch no organ the coding half reads.

**And it is not one lucky language.** Every seed mints a different word order, different case
markers, a different negator in a different position and different question particles. Swept over
forty of them, each taught and then re-examined with the teacher off:

| 40 minted languages × (taught + retention) | |
|---|---|
| subjects mastered | **560 / 560** |
| right / wrong / abstained | **6720 / 0 / 0** |

**`saying` is the one worth reading twice.** Its lesson is a *reading* lesson: fourteen past-tense
sentences with their meanings, and not one word about producing anything. Cold it scores 0.75 and
the 0.75 is correct — she speaks the two tenses she has constructions for and stays silent on the
third. What moves it to 1.00 is a shape learned by comprehension becoming available to production
with no production lesson at all.

#### Five things this found, and what each cost

Two were in the faculty and three were in the examination, and all three of the examination's were
found from *her* side of it — by a refusal or by a rejection notice — rather than by anybody
inspecting the generator.

1. **A filler may span one token, and that is a measurement.** It was three, which reads "the
   black dog" as one subject and looks like the more capable setting. What it actually buys is a
   grammar that cannot refuse: three bare slots match a four-word sentence by letting one slot
   swallow two words, so *a sentence with one word too many comes back read instead of refused*.
   Over six minted languages, twenty controls each — at three, two of the six refused only 12 and
   16 of their 20; at one, all six refuse all twenty. The cost is named rather than hidden: a
   two-word noun phrase is now **unreadable**, not mis-cut.
2. **A word's context is better described by the *kind* of its neighbours than by their
   identity.** In a verb-subject-object clause the subject touches neither edge of the sentence,
   so two subjects sharing no verb and no object share *no context at all* — measured, eight
   identically-behaving nouns in eight classes of one, and `word-classes` at 0.67. One further
   pass, describing each neighbour by the class the first pass put it in, puts all eight together.
   It stops at two passes deliberately: each further pass describes a word by a description that
   was itself inferred, and an induction run to a fixed point is one whose later passes are
   agreeing with themselves rather than with the corpus.
3. **A minted language that is ambiguous on its own terms grades a right answer as wrong.** The
   affixes were drawn distinct from each other and the particles distinct from each other, and
   nothing compared the two sets — so one language minted the object marker `-za` and the wh-word
   `nuza`, and `<subject>gu <verb> nuza` is *both* "who does she V" and "she V's a nu". She
   returned `ambiguous` on all four such sentences rather than picking one. Four correct refusals
   scored as misses until somebody looked at the language rather than at her.
4. **The same shape again, in the vocabulary.** A verb drawn as `tudubo` in a language whose past
   tense is `-bo` **is** a past-tense verb by that language's own rules, so the present-tense
   sentence built from it is genuinely ambiguous and her past reading of it was not wrong. One
   item in four hundred and eighty. Fixed in the generator rather than tolerated, because a
   benchmark that mints undecidable items reports a ceiling belonging to the benchmark.
5. **Exactly one slot is empty in a content question, and it is the one being asked about.** The
   generator emptied *both* for a subject-question, so the object word sat in the surface with
   nothing accounting for it, was kept as fixed material, and every sentence became its own shape.
   The faculty said so immediately and in the right words — eight demonstrations, eight
   signatures, every one of them rejected as *"one demonstration is a sentence, not a shape"*.
   The rejection notices named the defect; nothing had to be guessed at. That is the argument for
   a learner that reports what it threw away instead of only what it kept.

#### The one edge into what she believes

`nyxara.njp.grounding` will now read a sentence with a learned construction. It was written as a
pure fallback and that was **dead code**: the shipped core never returns nothing for a well-formed
sentence — the positional frame reads any three tokens — so a fallback waiting for silence waits
forever.

So precedence is decided on evidence, and the rule is one sentence: **a learned construction
outranks the positional frame when it matched fixed material — a literal particle, a case marker,
a tense ending — because that is a fact about the words in front of it, and order alone is a guess
about which language this is.** It never outranks a shipped *pattern*: those name their relation
lexically and are the more specific reading by the same argument. Four refusals hold whatever the
precedence says: it invents no language (a faculty nobody taught returns nothing, and a test
asserts that attaching one to a grounder leaves twenty-five varied sentences — English, Hinglish,
minted — grounding to byte-identical triples); it reads
assertions and nothing else, because *"what does a zorbin eat"* is not the claim that a zorbin
eats something called *what*; it is confident about nothing, carrying the construction's own
capped confidence and labelled `learned-grammar` so an audit can tell a lesson's work from the
shipped core's; and an unanchored learned reading of a sentence the core already read is
discarded, because two positional guesses disagreeing is not evidence for either.

**One more refusal, added after it was needed.** "Matched fixed material" was asked of the
*construction* rather than of the *reading*, and those are different questions once she speaks
more than one language. A three-slot minted-dialect shape whose paradigm happens to carry a tense
marker has markers; the English sentence it matched contained none of them. So after the school
taught her two dialects, the grounder took **82 English sentences** through a grammar for a
language nobody was speaking, mangled the entity names, and `composition` and `depth` fell from
1.00 to **0.33** — sixteen extra abstentions and no wrong answers.

It is worth recording how it was caught, because nothing else would have. The taught run is clean:
the reasoning subjects are examined *before* the language ones are taught, so at that point there
is no rival grammar to intercept them. Only the **retention pass** — which re-examines every
subject after every subject has been learned — puts an English reasoning question in front of a
brain that has since become multilingual. The one measurement in this package designed to ask
"what survived the lessons" was the only one in a position to see it.

End to end, through every organ between the two — four sentences of a language nobody wrote code
for, then a question in a language she shipped with:

    >>> for s, v, o in (("cat","chase","dog"), ("bird","chase","fish"),
    ...                 ("frog","eat","worm"), ("goat","eat","leaf")):
    ...     brain.show_language(f"{o}ni {s}ta {v}", Meaning(kind="assertion", subject=s,
    ...                                                     relation=v, object=o), tongue="zz")
    >>> brain.learn_language(tongue="zz")
    >>> brain.think("stoneni horseta eat").answer
    'noted: horse eat stone'
    >>> brain.think("what does horse eat?").answer
    'stone'

Reachable as `brain.hear_language`, `show_language`, `learn_language`, `read_language`,
`say_language`, and over the wire at `/v1/njp/language`.

**And it wakes up as the faculty that went to sleep.** The tongue goes into the `njp.json` sidecar
with every other organ — but what is written is the **lessons**, not the conclusions.
Constructions, affixes and word classes are all re-derived from the demonstrations and the
vocabulary on the way in, rather than read out of the file. It costs a fraction of a second on
load and it buys one guarantee: a sidecar can make her *forget* a language, and it cannot make her
believe a shape nobody ever showed her — not if it is truncated, not if it is hand-edited, and not
if an older version wrote it. Only the two things that could not be recovered by re-deriving are
stored as facts: which affix a lesson bound to which feature, and each construction's held-out
record, since that is evidence from outside the lessons.

#### What this is not

*(This was the state at V.18. The next section replaced the first three of these limits and
measured what that bought; it is left standing here because the numbers above are the numbers a
faculty with these limits produced.)*

Not a parser for English. No dictionary, no syntax theory, no recursion into subordinate clauses;
one construction is one flat sentence and a filler is one token, so a two-word noun phrase is
unreadable rather than mis-cut. **Three roles**, so a clause with an agent, a theme and a
recipient has nowhere to put the third. Affix induction is concatenative — suffixes and prefixes
only, no stem changes, no infixes, no reduplication — so an irregular is a memorised pair and
never a rule. A demonstration whose meaning is not in its surface is refused outright.
Nothing here decides truth or relevance: it produces a `Meaning` and stops. Nothing drives it —
no pulse, no evolver — because everything it learns comes from a demonstration, and a
demonstration is something a caller makes rather than something a clock produces. And on a brain
nobody has taught, every entry point returns the empty answer, which is the correct day-one
behaviour rather than a gap being apologised for.

### NJP V.19 — hard language problems, and what a bank is worth when it is written first

V.18's language syllabus reads 84/84 across forty minted languages. That number is worth exactly
as much as the bank behind it is hard, and the bank behind it was built alongside the faculty, so
it can only ask for what the faculty was built to do. Every dialect in it is one flat clause with
three arguments, a particle or two, and one suffix per feature.

The coding half of this package already learned what that is worth, expensively: 25/28 on a bank
written while the skeletons that solve it were being written, then **0/21** on a second bank
chosen afterwards. *"25/28 was a list of answers."*

So `nyxara.njp.hard` was written **first**, against a finished faculty, by looking for what it
cannot do. Twenty-one problems, each naming a property of human language the module has no
representation for, each minted per seed. Then measured, once, with nothing changed for it:

| the finished V.18 faculty, on twenty-one problems chosen against it | |
|---|---|
| problems solved | **1 / 21** |
| items | **35 / 278** |
| readings that were confidently **wrong** | **0** |

One. `ergative` — case marking chosen by whether the clause has an object — and it fell out of
machinery written for something else. Everything else was a refusal: nine problems could not
represent a fourth argument, six could not represent a morphological process that is not a suffix,
and the rest could not compose two features or license an order nobody demonstrated.

#### What was changed, and why it is four things rather than twenty-one

The response was deliberately **not** twenty-one mechanisms. Reading what actually blocked each
problem, the same four gaps came up over and over.

**A · A construction binds any number of named roles.** `ROLES` was the literal tuple
`("subject", "verb", "object")` — a statement about which sentences may exist. A clause with an
agent, a theme *and* a recipient had nowhere to put the third; nor did one with a location, an
adjective, a possessor, a coordinate subject, a relative clause or a complement clause.
`Meaning.roles` is a dict now, a role name is whatever a demonstration called it, and there is no
list of permitted ones — the same stance `concepts` and `core.represent` take about kinds.
**1 → 9 solved on its own**, and it took `classifier` too, which was not among the nine.

**B · A construction is a skeleton plus detachable markers.** A marker says *this affix, on this
slot, means this feature has this value*. Markers of different dimensions attach independently, so
three demonstrated cells license the fourth. Shown singular-present, plural-present and
singular-past, a grammar made of whole shapes holds three and has no fourth — and it did not even
refuse: the singular-past shape matched the plural-past sentence and read the plural ending as
part of the subject's name. A family is folded into a paradigm only where its members differ in
*one* piece of material carrying *one* feature; anything less separable is left alone, because
which piece carries which feature is then a guess. A dimension **every** member marks is recorded
as required, so a sentence lacking it is still refused. **9 → 15**, taking agreement, pro-drop,
syncretism and double-marking together.

**C · A morphological process is a shape, not a suffix.** `Morphology` induced prefixes and
suffixes, which is a statement about which languages exist. Circumfixes, infixes, reduplication,
partial reduplication and root-and-pattern templates were not mis-analysed, they were *invisible*
— the word this repo used about Devanagari at NYX V.01 and about negation at V.09. Six shapes now,
and the honesty is in how one is chosen: **a lesson proposes and the vocabulary corroborates.**
`maran` → `mamaran` proposes both "prefix *ma*" and "copy the first syllable", and they are told
apart not by taste but by the fact that the corpus contains `sotel` → `sosotel` and no `masotel`
at all. A process that reaches no support is refused and kept as an irregular that never
generalises. Allomorphs get a condition induced from the stems that corroborated each rule, and
two rules whose stems overlap keep no condition, because that is a choice the evidence does not
license. **15 → 21 on the first bank.**

**D · Order is free where marking is not.** Shown a case-marked language in two of its six orders,
she refused the other four. The precondition is checked rather than assumed: every slot must carry
material distinct from every other slot's, and at most one may carry none. Then a token names its
own role and order genuinely cannot change the meaning. A family with two unmarked slots gets
nothing, because there the order is the only thing telling them apart.

| the same twenty-one problems, after the four axes | |
|---|---|
| problems solved | **21 / 21** |
| items | **278 / 278** |

Over ten seeds — a different word order, different markers and different particles each time —
**209 / 210 problems** and **2778 / 2780 items**. The single miss is `harmony` at one seed, and it
is two *abstentions*: the stems that corroborated the front-vowel rule happened never to end in
`i`, so a stem ending in `i` met neither rule's condition and she declined. That is the induction
reporting the limit of its own evidence, which is what it is for.

#### The measurement that decides whether any of that is real

Twenty-one problems were the ones being measured while the axes were being written. So a **second
bank was written afterwards** — fifteen more problems from a different corner of the subject,
chosen with the axes already finished — and measured once.

| second bank, chosen after the axes existed | |
|---|---|
| problems solved | **7 / 15** |
| items | **108 / 191** |

Seven, not fifteen, and that gap is the finding. It is also not zero, which is what separates this
from the coding half's result on the same test: eight of the fifteen were solved by machinery
written for something else. The one worth naming is **`apophony`** — *sing* / *sang*, a stem whose
vowel changes with nothing added. Nobody wrote anything for it. It falls out of the root-and-
pattern template, which was written for Semitic morphology, and it is the only kind of evidence
that separates an axis from a special case.

#### What the eight failures had in common, and the two axes that came out of them

Again the answer was not eight mechanisms. The failures fell into exactly two shapes.

**E · A process can be an edit at a position that can be named.** A plural made by *removing* the
last letter, one made by *swapping* the last two, one made by writing the last vowel *twice*:
0/6, 0/6 and 3/6, all abstentions, because every one of the six shapes above is about something
added. Each of these is one edit at an anchor — the far end of the word, or its last vowel — so
the same edit is proposed measured from each anchor and the vocabulary decides which the language
uses. A deletion always one character from the right end is a rule; the same deletion measured
from the left is a different offset in every word and corroborates nothing. An edit ranks *below*
a plain affix that explains the same pair, because it is the most powerful shape here and so the
most easily fitted to a coincidence.

**F · A role remembers what has filled it.** Where a language says *denied* by moving the verb to
the front, `s v o` and `v s o` are the same three tokens under two shapes that disagree, and she
returned `ambiguous` to all of them — correctly, on the evidence she was using. The evidence she
was **not** using is that the verbs of a corpus recur and its subjects do not. A reading that puts
a word she has seen as a verb in the verb's place is better supported. It is evidence and never a
gate: an unmet word is not barred from a role, it simply brings nothing to a tie.

| second bank, after the two further axes | |
|---|---|
| problems solved | **14 / 15** |
| items | **183 / 191** |

Over ten seeds: **140 / 150 problems**, **1830 / 1910 items**, and the unsolved ten are all the
same problem at every seed.

#### The one it does not solve, named

**`polypersonal`** — a verb marking both its arguments, neither of which is a word. She **reads
every item correctly, 8/8**, and produces the two endings in the wrong order. Nothing in the
lesson shows which comes first: each cell demonstrated one marker alone, so the stacking order is
not in the evidence, and she guesses. Reading survives it because reading offers both orders and
the sentence decides; production has to pick.

This is also the only place in either bank where a miss is a **wrong answer rather than an
abstention** — 8 items out of 469 — and it is worth naming because the round-trip check that
guarantees the rest cannot catch it: she reads her own output back to the meaning she started
from, so by her own standard the sentence is valid. The coding half established the verdict on
this class already, on `median_two_sorted` and Josephus: *these problems are underdetermined, no
tie-break invented after the fact fixes that, and the honest answer is more examples.*

#### Seven flaws in the banks themselves, and who found them

Every one was found from **her** side of the exam — by a refusal, or by a rejection notice —
rather than by anybody inspecting the generator. Three were in V.18's minted dialects and four
here.

1. A wh-word ending in its own language's object-case marker: `<subject>gu <verb> nuza` is both
   *"who does she V"* and *"she V's a nu"*. She answered `ambiguous` four times, correctly, and
   was marked wrong for it.
2. A verb stem drawn ending in the tense suffix **is** a past-tense verb by that language's rules,
   so her past reading of the present sentence was not wrong.
3. A subject-question that emptied both argument slots instead of the one being asked about, so
   the object word sat in the surface with nothing accounting for it. Eight demonstrations, eight
   signatures, every one rejected as *"one demonstration is a sentence, not a shape"* — the
   rejection notices named the defect and nothing had to be guessed at.
4. Grading production by exact string where the language's **order is free**: six surfaces say one
   meaning and the grader accepted one, so a right answer was marked wrong 8/8 on the very problem
   whose point is that order does not matter. It happened a second time, on the second-position
   clitic, which is why it is recorded as a class rather than as an incident.
5. A verb pool the same size as the cell count, so every demonstration of a cell used the same
   verb and each shape was five copies of one sentence. She refused all three — *"identical
   fillers"* — which is what a shape supported by one repeated sentence should get.
6. A three-consonant root drawn for a wug item colliding with one already demonstrated. This is
   the only one that **flattered** the score, and it was caught by a test rather than by her.
7. Two novel words written together have a boundary nothing can see: `benuapseliao` divides after
   six letters, or five, or eight, with no evidence anywhere preferring one. She answered
   `ambiguous` to all eight. An item whose only correct answer is *"I cannot tell"* measures the
   item, so the heads now come from a pool the lesson also uses as bare words — and finding the
   boundary at a word she has met is the ability the problem was supposed to be about.

#### One change that had to be given up, and what replaced it

V.18 refused any demonstration whose meaning was not in its surface, as a guard against a
mislabelled lesson. That guard had to go, because **a subject that is not in the sentence is
pro-drop**, and a grammar that refuses it refuses something half the world's languages do.

What replaced it is structural rather than a rule about role names, and it needed nothing written
for it. When a lesson mislabels the subject, the word that *would* have been it is still sitting
in the sentence, so it becomes fixed material — and fixed material differs from one demonstration
to the next, so no shape ever gets a second demonstration to agree with it. When the subject is
genuinely dropped there is no leftover, every demonstration produces the same shape, and it
accrues support in the ordinary way. Both are tested.

#### What this is still not

Not a parser for English, and the axes did not change that. One construction is one flat sentence:
the relative-clause and complement-clause problems are solved as **flat** patterns with more
roles, not by recursion, so a clause inside a clause inside a clause has no representation and
arbitrary depth is not claimed. A filler is one token. Affix induction proposes from a lesson and
corroborates from a vocabulary, so a language met only in running text — with no paradigm to
count — grows no rules. The free-order axis enumerates permutations and stops at four slots.
And a feature whose realisation depends on two slots at once is read and said correctly but is
never folded into a paradigm, so it cannot compose with anything.

`python -m nyxara.njp.hard` runs the first bank, `--second` the second, `--all` both.

### NJP V.20 — English and Hindi, and the honest size of what that means

Everything the faculty had learned until now was a language **nobody speaks**. Thirty-eight of
them, minted per seed, precisely so that a score could not be a table of English in disguise. That
was the right thing to measure and it left one question unasked: *can it learn a real one?*

Asked plainly, the answer was **no — and not because of the mechanism**. She had no vocabulary at
all. What ships is a tokeniser that handles every script, 242 closed-class words across English,
romanised Hinglish and Devanagari, and 88 hand-written extraction patterns. That is grammar
scaffolding, not a language. A sentence in Spanish came back `('el', 'gato', 'persigue al perro')`
— confidently wrong, from the positional frame — and one in Japanese came back unreadable.

`nyxara.njp.lessons` is a curriculum for two real languages, written out by hand.

#### What it is, and what it is not

**Not "everything the author knows about English."** It is what could be written down and
mechanically verified: a few hundred word forms with their real paradigms, a corpus for the
classes to form from, demonstrations of the constructions both languages actually use, and a
bilingual glossary. The gap between that and a speaker's competence is enormous and is not closed
here — no idiom, no register, no pragmatics, no world knowledge, and a vocabulary of **193 English
forms and 59 Hindi ones** that you can read in a minute.

**Hindi is taught twice**, in Devanagari and in romanised Hinglish, because those are two surfaces
for one grammar — subject-object-verb, postpositions, the copula last — and holding both is the
cleanest available check that what she learned is a grammar and not a spelling.

#### What one run reports

167 demonstrations, then 114 held-out items: sentences built from the same words in combinations
no lesson contains, and inflections of stems no lesson touched.

| | held out | faculty taught nothing | shipped compiler | after the curriculum |
|---|---|---|---|---|
| English — read · say · wug | 20 · 20 · 12 | 0 | 9 / 20 read correctly | **20 · 20 · 12** |
| Hindi (Devanagari) | 12 · 12 · 7 | 0 | 0 / 12 | **12 · 12 · 7** |
| Hindi (romanised) | 12 · 12 · 7 | 0 | 0 / 12 | **12 · 12 · 7** |
| **all** | **114** | **0** | **9** | **114 / 114** |

The wug items are the classic test in both languages: `wug` → `wugs`, `blicket` → `blicketed`,
`gostak` → `gostaking`, and in Hindi nonce stems of **both** inflection classes — the vowel-final
`टिमा` → `टिमे` and the consonant-final `बलक` → `बलकें`. Ten irregulars are memorised and do not
leak: `go` → `went` while `wug` → `wuged`.

**Translation works in all six directions**, and it needed one thing the minted banks did not: a
lexicon. Between two minted dialects both languages use the same content words, so carrying the
meaning across was carrying the role names alone. Between two real languages the words differ too.
So a glossary is *taught* — twenty rows, sixty pairs, both directions — and `translate` refuses
where one is missing: `the goat chases the cat` translates to nothing at all, because `goat` has
no gloss, rather than to a sentence with a hole in it.

    the farmer chases the cat   → लड़का बिल्ली देखता है
    लड़का बिल्ली देखता है         → the farmer chases the cat
    the teacher does not open the book → शिक्षक किताब नहीं पढ़ता है

Reachable as `brain.learn_languages()` and `brain.translate(text, into=…, frm=…)`, and it survives
a restart with everything else in the sidecar.

#### Three axes the real languages forced, and the minted ones had not

**G · A slot's width is learned, not a constant.** `MAX_SPAN` was a flat 1, because at 3 a
construction of bare slots swallowed an extra word and stopped refusing. But a flat 1 makes *"the
big dog"* unreadable, which rules out most of English — so neither constant was right and *the
constant* was the mistake. Learned per slot from what it was shown holding, the minted languages
keep refusing every one of their controls (their fillers are all one token, so every slot learns
width 1) and English gets its noun phrases.

**H · An ending is a fact about the language, not about one construction.** The transitive family
was shown both `-s` and `-es`, so it reads `pushes`; the possessive family was only ever shown
`-s`, and read the same word as `pushe` — not a word, in a language where `push` is. The
morphology holds the whole paradigm, so it corrects a cut that is plainly not a word (`_resolve`)
and, in production, selects a form she has already met over one the construction would spell
(`_repair`). It selects and never invents, so a word she has never heard still comes out however
the construction writes it. **Comprehension ran ahead of production for a while** — reading only
has to recognise a form, speaking has to pick one — which is the ordinary shape of the thing.

**I · A condition is the most general description that separates the evidence.** Allomorph
conditions were a list of *characters seen*: a rule witnessed on stems ending in `h` and `d` fired
on those two letters and abstained on one ending in `m`, though all three are consonants and the
rule is about consonants. Three descriptions are now induced — whether the stem ends in a vowel,
which vowel it last had, which character it ends in — and the coarsest that separates the family
wins. Vowel harmony still lands on the vowel, because there every stem ends in one and the split
is *which*.

#### Four bugs, and who found each

1. **The tokeniser shattered Devanagari.** `[^\W\d_]+` looks script-neutral and is not: Python's
   `\w` excludes combining marks, so every matra fell out and `लड़का आम खाता है` tokenised as six
   fragments, none of them a word. Every Hindi lesson was then a sentence whose own words were not
   in it, and the Devanagari grammar came back with **0 shapes**. This is the third time this
   exact failure is recorded here — NYX V.01's `[a-z0-9]` regex, V.09's missing negation, and now
   this — and the fix is that this module no longer has a tokeniser of its own.
2. **One ending, two shapes, and no way to count them together.** `khaata` is `kha` + `ata` and
   `padhta` is `padh` + `ta`, so every allomorph arrived with a single demonstration and every one
   was rejected as "a sentence, not a shape": romanised Hindi learned **1 construction of 12**.
   The same course in Devanagari scored 12/12, because there the matra rides inside the stem's own
   syllable. The grammar was never the problem; the orthography was showing something the grammar
   had no way to count, and support is now counted per *skeleton*.
3. **Support outranked a matched literal.** Ranking added everything into one number, so a
   well-demonstrated shape that matched *nothing in the sentence* tied with a question form that
   matched the question word — and a tie that disagrees is a refusal: 42 of 480 minted sentences
   came back `ambiguous`. Material the sentence contains now outranks everything else, and the
   rest only breaks ties.
4. **A longer affix outranked a known word.** `chases` is `chase` + `s` or `chas` + `es`, and
   affix characters counted towards the anchor, so the second won. It then misread its own lesson,
   `_verify` dropped it, and the whole transitive family went with it. The anchor is whole tokens
   now; endings are evidence in the tie-break, where a known word can outweigh one more matched
   character.

And **three flaws in the curriculum**, all of which marked her wrong for being right: paradigms
regularised to `pushs` and `carrys` while the exam asked for the real forms; a past-tense lemma
written as `form[:-2]`, which makes `chased` into `chas`, so `_verify` threw away the past tense
for contradicting a lesson that was itself wrong; and Hindi wug answers that were already in the
vocabulary, which a **test** caught rather than she did — the second time in this work that the
flaw she could not surface was the one that flattered the score.

#### What she still cannot do, in a real language

She refuses these, and refusing is correct:

    the quick brown fox jumps over the lazy dog     — words she has not met
    yesterday I would have preferred the other one  — a shape she has not met
    the goat does not walk                          — negation was demonstrated on
                                                      transitive clauses only
    मैंने कल तुम्हें बताया था कि यह मुश्किल है          — both

The third is the one worth reading. It is *nearly* in her grammar: she has negation, and she has
intransitives, and she has never seen the two together. Markers compose across dimensions; a
negator that is a whole token in one construction and absent from another is not a dimension of
one shape, so it does not. That is a named limit rather than a mystery.

Beyond it: no idiom, no pragmatics, no world knowledge, a two-hundred-word vocabulary, one flat
clause per construction, a glossary with one sense per word, and no way to acquire any of it from
running text — a paradigm has to be countable before a rule can come out of it. What is real is
narrower than "she knows English and Hindi" and is not nothing: **taught 167 sentences of two real
languages, she reads and says 114 held-out items she was never shown, in three tongues at once,
and translates between them.**

### NJP V.21 — general knowledge, taught and then examined on what she was never told

`nyxara.njp.school` examines reasoning, language and coding on **generated** items — nonsense
vocabulary, minted grammars, held-out coding specs — because those three subjects are about
structure, and structure can be tested on entities that did not exist when the lesson ran.

General knowledge cannot be examined that way. There is no minting a fresh fact about the world;
what she knows of it is exactly what she was told. So the held-out surface has to be built the
other way round: **from facts she was told, compose questions whose answers she was never told.**

**What she was taught.** `scripts/knowledge/*.kb` grew from 15 domains to 48 — from **3,745 facts
over 966 subjects to 13,755 over 3,937**, three and a half times the corpus.

The first pass added six subjects that were simply absent: `arts`, `sport`, `philosophy` (with the
world religions described rather than asserted), `body` (the anatomy `medicine` assumed and never
stated), `food` and `measurement`.

The second went after a different kind of gap — places where the corpus named a *category* and
never a member of it. `biology.kb` defined `mammal`, `bird`, `reptile`, `insect` and `vertebrate`
and did not name one animal, so a taxonomy with kinds and no members was a hierarchy one level deep
and the inheritance machinery in `njp/core.py` had almost nothing to walk *through*. Nineteen more
domains: `animals`, `plants`, `materials`, `mind` (psychology — the terms this package uses about
itself, with `attention` and `episodic memory` as module names and no fact about either),
`environment`, `transport`, `world` (fifty more countries and the organisations between them),
`india` (the brain is expected to work in Hindi and Hinglish and held nothing about the Ganga),
`law`, `finance`, `inventions`, `people`, `space`, `media`, `work`, `home`, `health`, `mythology`,
and `basics` — colour, shape, direction, quantity and the senses, the layer that had been invisible
because it appears in the *questions* rather than in the answers.

A third pass took the six that were left, each of them a mechanism sitting under facts already in
the corpus: `microbes` (a virus, and why an antibiotic does nothing to one — `health.kb` said
handwashing causes large falls in disease and had no mechanism behind it), `weather` (pressure,
fronts, humidity and how a forecast is actually made, under `earth_science`'s nine named
outcomes), `energy` (the chain from a fuel to a socket, joining `technology`'s seven unconnected
devices), `ocean` (tides, currents and zones, which `weather`, `environment` and `animals` had all
been referring to), `ancient` and `modern` (the two halves of `history.kb`'s ninety lines, which
gave every civilisation about one).

**Seventeen name collisions were resolved by hand**, which is the interesting part of a merge at this
scale, because the store keys on the name and would otherwise hold one subject that is two things:
`bat` the animal against the cricket bat; `python` the snake against the language; `memory` the
faculty against RAM; `attention` the psychological process against the transformer mechanism; `bus`
the vehicle against the data bus; `tree` the plant against the data structure; `trade` against the
skilled trade; `charge` the electric quantity against the criminal one; `library`, `transformer` and
`mercury`, all three of which predate this change set; `act` the deed against the Act of
Parliament; and, from the third pass, `transmission` (disease against gearbox), `vector` (the
mathematical against the mosquito), `wave` (ocean against physics), `pantheon` (the set of gods
against the building in Rome) and `library` again.

That last was caught by the exam rather than by reading: `arrest is_a act`, `act is_a statute`, and
"what is arrest used for" inherited the purpose of legislation. Everything else that collides —
`cricket`, `ethics`, `calendar`, `steel`, `road`, and 57 more — is one concept classified at two
granularities (`bone` as tissue and as organ, `teacher` as person and as profession), and those
merge correctly and give the inheritance walk two routes instead of one.

**A feeding role is now written below the taxonomic class.** `tiger is_a mammal` and `tiger is_a
carnivore` at equal confidence made "what is a tiger" come back CONFLICTING — `Grounder.answer`
saw two equally supported readings and correctly refused to pick one, which is the right rule
producing a useless answer. Roles are written at `@0.8` instead, because that is what is actually
true of the two claims: `mammal` is what a tiger *is*, `carnivore` is a role it fills, which a bear
fills part-time. Both stay inheritable. Measured, `recall` moved from 0.835 to 0.887 and 21 of its
abstentions became answers.

**How she is examined.** `nyxara.njp.general` is seven papers, each defined by what it holds out:

| paper | what is held out |
|---|---|
| `recall` | nothing — the control, so a low score reads as "the grammar failed", not "she forgot" |
| `membership` | nothing, but asked as `is X a Y` — the other form English has, through different code |
| `taxonomy` | two `is_a` hops, where the one-hop version is not stated anywhere |
| `inheritance` | a property of the kind, asked of a member that carries no such property |
| `inverse` | `what causes X`, where the edge is stored pointing the other way |
| `abstention` | subjects the corpus never heard of — **silence is the pass** |
| `soundness` | the relations that cannot cross from kind to member — **silence is the pass** |

Measured on the corpus as shipped, 400 items a paper, ~145 s:

```
paper          asked  right  wrong  declined  silent   score
recall           400    334      0        66       0   0.835
membership       400    399      1         0       0   0.998
taxonomy         400    399      1         0       0   0.998
inheritance      400    400      0         0       0   1.000
inverse          400    389      0        11       0   0.973
abstention       400    400      0         0       0   1.000   (silence is the pass)
soundness        400    400      0         0       0   1.000   (silence is the pass)

2721/2800 = 0.972 overall
of what she answered: 1521 through English, 400 only through the derivation ladder
not asked: 43 inheritance chains longer than 3 hops, which the walk cannot afford
```

Every paper held its score when the corpus tripled, which is the first thing worth saying about
it: the numbers below are about mechanisms, not about how much happened to be in the store.

**`membership` and `taxonomy` scored 0.000 before this change set** — 0 of 300 and 0 of 300,
re-measured by putting the table back. `answer("is the sun a star?")` returned UNKNOWN, with
`why="no claim either way about this pair"` and `sun is_a star` in the store, because `is_a`
appeared in no row of `grounding._POLAR_PREFERENCE` nor in its default: the only relations an
`is X ...` question ever scanned were `has_property` and `capable_of`. "Is a whale a mammal", "is
copper a metal", "is Peru a country" — the commonest question form in general knowledge, and none
of it could ever have been answered. One relation name added to one table moved both papers to
0.998, and made the *two-hop* question answerable as a side effect, because the polar scan already
searched the subject's neighbours.

The items still wrong are one subject and one cause: `_read_polar_surface` searches at most four
words for where the subject ends, and a handful of the corpus's 3,145 subjects are longer —
*One Hundred Years of Solitude* is the one that keeps being sampled. A real ceiling over a
fraction of a percent of the corpus, named here rather than tuned away.

**`soundness` swings every item to none of them** when `core._NOT_INHERITABLE` is emptied — 328/328
against 0/328 when it was first measured, re-measured on every run. Inheritance — "a bird needs
water, so a sparrow probably does" — was being applied to every relation alike, and over the
corpus that was wrong 89 times in 247: `has_kind` points *down* the taxonomy, so inheriting it
inverts the hierarchy ("the types of aircraft: car", "the types of an ammeter: a ruler", "the types
of the Amazon rainforest: the Amazon rainforest"); `means`, `symbol` and `also_known_as` identify
their subject, so the kind's answer is the wrong size ("combustion means a rearrangement of atoms
into new substances" — the definition of *chemical reaction*; "the cpu is also known as the cpu");
`capital`, `currency`, `author`, `inventor`, `discoverer` and `birthplace` name one particular
thing about one particular subject. The table refuses rather than discounts, because there is no
confidence at which "the sun is a red giant" becomes a thing to say.

**`inheritance` is 400/400, and every one came through the derivation ladder — none through
English.** That is the most useful number in the table and the one that is not a success: she can
derive a property she was never told, and there is no English sentence that asks her to. The
1,521-to-400 split at the foot of the report is the exact size of what she knows and cannot be
asked for.

**The 43 chains not asked** are the same discipline pointing the other way, and they are the
finding the corpus growth produced. On the bigger corpus `inheritance` fell to 0.973, and the
misses were all five-link chains: "an actor has a backbone", true, reached through
*person → human being → mammal → vertebrate*. `_inherit` prices each hop by the `is_a` transitivity
posterior and abandons the chain under `core._MIN_LINK_CONFIDENCE`, so she declines — correctly,
because a property inherited from four levels up is a very weak claim however true it is. The exam
was scoring her at 0.973 for refusing to overreach. The cliff was then measured rather than
argued: **levels 1, 2 and 3 answer 446/446, 167/167 and 58/58; level 4 answers 0 of 13 and level 5
0 of 3.** `general.MAX_INHERITED_HOPS` is set to that measured edge, a test asserts the bound from
both sides against the live Core, and what the bound excludes is printed in the report rather than
dropped from it.

**`recall`'s 66 declined are not misses.** They are relations holding several objects at equal
confidence, where `Grounder.answer` refuses to pick one — the outcome `prepare_knowledge_corpus`
documents over the same corpus and counts separately for the same reason. Reading them as gaps is
how the first version of this exam reported **4,572 of 4,572 with no abstentions at all** on a
corpus that abstains on 19% of the shipped QA file: the derivation ladder was being asked next, and
`_direct` takes `max` over the tied triples and answers happily. A refusal the next rung overrules
is not a refusal.

Three more of the exam's own numbers were wrong before they were right, and each is pinned by a
test: one-value golds on many-valued relations (`inverse` graded "what causes fever" against
`infection` and marked `influenza` wrong; `inheritance` graded "what does muscle consist of"
against `tissue` and marked `cells` wrong, which she had inherited correctly through the subject's
*other* stated kind), and inverted papers that never asked the ladder — the acceptance test for a
table, unable to see the table deleted.

The corpus builder also learned that a subject can be lost by its own spelling. `_read_question`
strips a leading article, so "What is The Odyssey?" comes back as `('odyssey', 'is_a')` and "What
is a priori?" as `('priori', 'is_a')`. Where a rename survives it the subject was renamed; where
the article is part of the term it is not available, so `unaskable_subjects` keeps the triples and
drops the questions — as `_GRAPH_ONLY` predicates already do, for the same reason — and `--check`
names them rather than subtracting them silently.

Run it: `python -m nyxara.njp.general`, or `NJPBrain.sit_general_exam()`. Nothing in it writes to
the brain, so it may be run twice around something that does.

### NJP V.22 — the hard half: answers that are in no fact

The twelve-paper exam's first seven papers are each answered by **one fact or one walk**: a lookup,
a membership test, an inheritance step, an edge read backwards. She scores 0.979 on them, and that
number says nothing about whether she can reason, because no item on it requires putting two
unrelated facts together.

`nyxara.njp.puzzle` is the other half, and `general.py` gained five papers that use it. Each asks a
question whose answer is **nowhere in the store under any key** and has to be constructed:

| paper | what it needs that a lookup does not |
|---|---|
| `bridge` | two different relations chained — "the currency of the country the Taj Mahal is in" |
| `commonality` | a set intersection nothing in the store is *about* |
| `odd_one_out` | a majority vote over a set the question invents, then the exception |
| `constraint` | a search under two conditions at once — "which mammal can fly" |
| `analogy` | a relation identified from one pair and applied to a third term |
| `chain` | transitive reach past the first hop, with the path |

**The floor was measured before a line of it was written.**

* `bridge`: 211 opportunities existed and `walk_shape` solved **117 of 120** — and **0 of 40**
  could be *asked*, because nothing in `_QUESTION_PATTERNS` reads a nested question. The knowledge
  was there, the walk was there, and the two could not be introduced to each other.
* `commonality`, `odd_one_out`, `constraint`, `analogy`: **no operation existed at all.**
* `chain`: 4 of 80 starts reached a second hop — which turned out to measure the *corpus*. Only 35
  two-hop causal chains existed in 13,483 facts. `causal.kb` was written to supply the joins
  between things already in the corpus, and reach went to **79 of 80**.

First measurement of the five papers: **0.875**. After the defects below: **741/742 = 0.999**, and
every one of those answers arrived through construction — `by_english` and `by_derivation` are
zero on all five, asserted by a test.

```
paper          asked  right  wrong  declined  silent   score
bridge           150    150      0         0       0   1.000
commonality      150    149      0         0       1   0.993
odd_one_out      150    150      0         0       0   1.000
constraint       150    150      0         0       0   1.000
chain            142    142      0         0       0   1.000
```

**What the exam found, and most of it was the exam.** Four of the six defects were mine:

* **The container word is not decoration.** "The capital of the country that Agra is in" returned
  *Lucknow* — Uttar Pradesh's capital — because the walk took the first hop that yielded anything.
  The noun in the middle of a nested question says *where to stop*: `bridge_to_kind` walks until a
  node that **is a** country, and "capital of the **indian state** that Agra is in" and "capital of
  the **country** that Agra is in" now give different answers off the same walk.
* **A multi-word container cannot be split by a regex.** With a non-greedy pattern, "the capital of
  the indian state that mumbai is in" split as container *indian*, subject *state that mumbai*, and
  every such question came back silent. The blob is captured whole and each split is tried, keeping
  the one whose subject the store knows — the same signal `_read_polar_surface` uses.
* **The store's spelling rule applies to objects used as subjects.** `canon` singularises, so
  "runoff of nutrients" is filed under `runoff of nutrient`. The exam was keying on the surface
  form, so `chain`'s gold silently lost every middle whose plural folded and marked her wrong for
  giving the answer it had dropped. Same bug hid `displacement of people` behind
  `displacement of person`.
* **Specificity is not string length.** "What do a pressure cooker and a spoon have in common"
  answered *object* over *utensil*; "a keyboard instrument and a percussion instrument" answered
  *tool* over *musical instrument*. Both true, both useless — everything shares something at the
  top of a hierarchy. A shared kind is less specific exactly when another shared kind reaches it by
  an `is_a` walk, and ranking on that makes "utensil beats object" follow from the taxonomy.
* **A qualifier can be swallowed by a substring.** "Which ocean is the largest ocean" answered
  *Atlantic*, because the stored property "the second largest ocean" contains the string "largest
  ocean". As whole words it does not, and the wrong answer disappears.
* **"Crime and Punishment" is one novel.** Splitting an item list on " and " as well as on commas
  turned four items into five, two of them half a book.

And, for the fifth time in this work, a **one-value gold on a many-valued relation** — twice more,
in `chain` (deforestation has two true two-hop endings) and `constraint` (a kidney and a lung are
both "two of them"). Each is pinned by a test.

**A second tier, added after the five papers were already at 0.999.** Probing for what she still
could not do turned up six more forms, and one of them was worse than a refusal — "which bird can
fly and lives in India" answered **"bird"**, because the stored value `india` was found sitting
inside the question. That is the mirror of the substring bug fixed a paragraph earlier: a stored
property may say *more* than the question named, never less, and that direction of matching is gone.

The other five were absent rather than wrong, and all five now answer:

| question | answer | why it was refused |
|---|---|---|
| "what is the heart part of?" | circulatory system | `part_of` is stored, carries the highest transitivity prior in `core`, and had no question form that parses |
| "what would you use to measure temperature?" | thermometer | `purpose` read backwards — the store says what a thermometer is *for* and never what measures temperature |
| "what would you use to measure pressure?" | barometer | the same, and the stored phrase says "measuring **atmospheric** pressure" — words in order, not adjacent |
| "what is the capital of the country whose currency is the yen?" | tokyo | a bridge walked *inward*: the subject is not named, it is described |
| "why does an earthquake happen?" | fault | `causes` from the far end, for any stored effect |

**And a defect none of the papers could see: 105 compound kinds had no parent.** `island country`
sat unattached, so **Japan was not a country by any walk**; `precious metal` sat unattached, so gold
was not a metal; `blood cell` sat unattached, so a red blood cell was not a cell. 839 of 1,269 kinds
had no parent, and most of those are roots and belong that way — but the compounds are not roots,
they are unattached. It is invisible until something walks the hierarchy, and it was quietly
weakening every inheritance, every commonality and every constraint search in the corpus.

`taxonomy.kb` states them, on a rule mechanical enough to check: **an English head-final compound is
a its head.** It does not hold for head-initial phrases — a "creature in mythology" is not a
mythology, a "unit of time" is not a time — so anything with " of " or " in " was excluded, and a
further list rejected by hand where the head is a different sense ("gas giant" against the giant of
folklore). Nothing in the file is a new claim about the world: every line says a thing is what its
own name already says it is, and the corpus never wrote it down.

Run it: `NJPBrain.puzzle("...")`, or `python -m nyxara.njp.general` for the full twelve papers.
Nothing in the module writes to the store: a constructed answer is an inference, and filing it
would let the next question read it back as though somebody had stated it.

### The question grammar — 18 of 25 everyday phrasings did not parse

The exam measures what she can be *asked*, and a second measurement asked a blunter question: of
25 ordinary phrasings of questions the store could already answer, **how many parse at all?**

**Seven.** The other eighteen failed, and every one failed the same way. The generic
`what is X` pattern sits near the bottom of `grounding._QUESTION_PATTERNS`, matches to the end of
the line, and swallows the tail into the subject — so "what is the capital of France" asked `is_a`
about an entity named *capital of france*, "who invented the telephone" read the verb as part of a
name, and "where is the Taj Mahal located" produced the subject *the taj mahal located*. The facts
were in the store, reachable by `_lookup`, reasoned over by the Core, and unaskable in the words
anybody actually uses. This is the same read/write asymmetry the causal block and the
`capable_of` block in that table were each added to close, found the same way — by measuring
instead of assuming.

**The noun form is one pattern, not eleven.** `tell me the <p> of X` had been there since the
beginning and is the phrasing nobody uses; `what is the <p> of X` is the one everybody uses and was
absent. Reading the relation out of the noun slot covers capital, currency, symbol, unit, formula,
purpose, meaning, birthplace, author, inventor and discoverer at once — and every relation added
later, without a line each.

**That breadth has a cost, and paying it is the interesting part.** The pattern also matches every
subject whose *name* contains " of ", and this corpus has eighty-nine of them. Measured the moment
it was added: "what is the Code of Hammurabi" read as `('hammurabi', 'code')`, "what is the
Republic of India" as `('india', 'republic')`, and the corpus builder's own `unaskable_subjects`
count — which exists to catch exactly this — went from 2 to 89. So `_NOUN_OF` is a **marker rather
than a reading**: `_read_question` accepts it only if two things hold, and otherwise keeps scanning.

1. The noun must name a relation something is actually stored under — known to the tables, or
   present in the live store, so a relation a later corpus introduces is accepted without anyone
   remembering to update a list.
2. The whole phrase must not itself be a known entity. *Age of Exploration* is a subject and `age`
   is also a relation; without this the relation reading wins and a question about the subject is
   answered about something else. **The store knows its own subjects** — the same signal
   `_read_polar_surface` uses to find where a subject ends, and the only one that separates these
   two readings.

`unaskable_subjects` went back to 2 — the two that genuinely cannot be named in English, `a priori`
and `a posteriori`. It was also fixed itself: it built a *fresh* Grounder, so its check ran against
an empty store and was stricter than the reader it was checking, reporting `age of exploration`
unaskable while the live brain read it correctly. It now seeds the subject keys first.

**"What does X do?" is answered by the evidence, not the grammar.** The form is genuinely
ambiguous — "what does a plumber do" wants `purpose`, "what does a bat do" wants `capable_of`, and
the five words are identical. It is read as a marker and `Grounder.answer` tries both, taking
whichever the store has evidence for; neither is guessed, and if neither has any the answer stays
honestly UNKNOWN. That is the move `_read_polar_surface` already makes for "is X <phrase>", for the
same reason: the evidence disambiguates and the grammar cannot.

After the change: **25 of 25**, with every existing template in `prepare_knowledge_corpus._ASKABLE`
and `general.ASKABLE` still reading back as itself, and 23 tests in `tests/njp/test_grounding.py`
pinning each form.


### NJP V.23 — mathematics: the syllabus, and the store it was quietly corrupting

Twenty-five ordinary school questions, asked through `NJPBrain.think` on a brain with every organ
built, before a line of this was written — reproducible now that the organ has a gate, by
building `NJPBrain` with `mathematics_enabled = False`:

```
right                              3 / 25
silent                            17 / 25
filed as a fact about the world    5 / 25
```

The three right are the same question in three costumes — `24 + 18`, `1/2 + 1/3`, `20% of 250` —
and they are the three `nyxara.njp.calculate` was written for. That module says of itself, in its
own second paragraph, that it "does not do algebra, does not solve for unknowns", and it was
telling the truth: **a closed expression was the whole of her mathematics.** Every gcd, every
prime, every mean, every area, every derivative, every `solve for x` came back as the empty
string. `nyxara.mind.math` — a real symbolic engine with sympy and z3 behind it — has existed the
whole time and takes structured calls; nothing in `nyxara/njp/` has ever imported it, so no
sentence anybody could type reached it.

**The five that were not silent are why this is a defect rather than a gap.** All five were
filed at confidence 0.75, source `semantics`:

```
"simplify the fraction 18/24"  →  ('simplify fraction', '18') → '24'
"expand (x+2)(x+3)"            →  ('expand', 'x')             → '2 x 3'
"factorise x^2 + 5x + 6"       →  ('factorise', 'x')          → '2 5x 6'
"convert 5 km to metres"       →  ('convert', '5')            → 'km metres'
"solve x^2 - 5x + 6 = 0"       →  ('solve', 'x')              → '2 5x 6 0'
```

They are imperatives. Nothing in the package read them as **tasks**, so the semantic compiler read
them as **assertions** and filed them — into the store `general.py` walks for inheritance, that
`puzzle.py` searches for commonalities, and that a later contradiction would be *revised into*.
Asking her to do arithmetic wrote nonsense into what she reasons from. A silent failure costs a
turn; that one costs the store. It is the read/write asymmetry this package keeps
finding, pointed the other way for the first time: not knowledge that cannot be reached, but a
question that reaches the wrong organ and is written down.

`nyxara.njp.mathematics` is a mathematician in the sense that `calculate` is a calculator. Fifty
skills across the school syllabus — number theory, fractions, percentage and commerce, ratio,
algebra, sequences, geometry and mensuration, units, statistics, probability, powers and
logarithms, elementary calculus, and the word problems that dress any of them in a sentence — read
in English or Hinglish, each reporting its working and its exactness. After it: **25 of 25, and
nothing written.**

**Every value inside is a `Fraction`,** so a third is a third until something asks for it in
decimal. Where exactness genuinely ends — a root, a π — both halves are reported (`49π ≈
153.938`), because rounding π silently and calling it an area is the dishonesty `calculate`
already refuses when it says whether a value is exact.

**Nothing is `eval`ed, and this time the whitelist is the type.** Algebra is parsed by `Poly`'s own
tokeniser into a polynomial over `Fraction` — a `Dict[int, Fraction]`, which cannot represent a
call, a name, an attribute or a subscript. There is no arrangement of that dict that is a
`__import__`, so unlike a node whitelist there is nothing a Python release can widen. **sympy is
never called from the module at all** — exactness comes from `Fraction` and roots from integer
arithmetic — so all fifty skills run identically with it absent; the shared calculator is the one
place it is reached.

#### What the exam found, and most of it was mine

`nyxara.njp.mathschool` sits eighteen subjects — sixteen doing, one knowing, one entirely
controls — and every defect below was found by running it rather than by reading the code.

| defect | what it did |
|---|---|
| **a word parsed as an unknown** | "a factor is a divisor of a number" reached the factoriser and came back *"two different unknowns, is and a"*. The sentence was then a **recognised task**, so the brain refused to learn the definition it was being taught — a lesson silently discarded by the reader meant to be helping it. An unknown is one letter, which is what an unknown is. |
| **claimed ≠ solved** | "convert 5 km to kilograms" is recognised by the conversion skill and *refused*, because length and mass are different quantities. Gating the store on **solved** let it fall through to the grounder and be filed as `('convert', '5') → 'km kilograms'` — the original defect surviving inside its own fix, on the one input where the fix declines. |
| **a refusal read as an invitation** | After a lesson stating "a mode is the value that appears most often in a list", the control *"what is the mode of 1, 2, 3, 4, 5?"* — a list with **no** mode — was answered with that definition. `_compose` had already returned silence; deliberation and recall run whenever the answer is empty, and recall offered the nearest thing in the store. |
| **the exam answering its own questions** | `restraint` scored 8/10 inside a full exam and 10/10 asked on its own: *"what is 7 divided by 0?"* came back **0.7**. Sixteen papers of arithmetic had gone into episodic memory ahead of it. An examination whose earlier items answer its later ones measures its own ordering. |
| **an ordinal deleted a term** | "the 15th term of 12, 15, 18" excluded the ordinal *by value* and so deleted the 15 that was a term, leaving two, and came back silent. |
| **a floored gold** | The percentage generator computed its own answer with `//`, graded "12% of 60" against 7 where the answer is 7.2, and marked her wrong four times in twenty for being right. Another gold that was wrong before she was — the defect the general exam kept finding in the knowledge corpus, met for the first time outside it. |
| **a degenerate floor** | `vocabulary` chose its taught/withheld split inside `teach`, so the pre-test had no split to grade against: every term was a control, she was silent on all twenty, and **the floor read 1.00 for knowing nothing at all** — with the post-test at 0.85 reported as a lesson that lost fifteen points. |

Two more were in organs this change set only *reached*. **A worked answer is made of the
question's own symbols, and `is_meta_commentary` scores word overlap** — so "is 91 a prime
number?" → "no, 91 is not a prime number" (0.83) and "factorise x^2 + 5x + 6" → "(x + 3)(x + 2)"
(0.67) were each correct, each deleted as an echo, and each went out silent. And **a number word
is also an ordinary word**: "ek" is Hindi's indefinite article, "do" is an English verb, "one" is
an English pronoun, so *"91 ek prime number hai?"* became "91 **1** prime number hai" and was
answered about 1. Number words are now read only in a sentence with no digits in it, which is
exactly the case they exist for.

#### What one run reports

`python -m nyxara.njp.mathschool` (seed 11) and `--exam` (25 items a paper), reproducible:

```
school                                    examination
mastered            18 / 18               17 / 17
right/wrong/absent  158 / 0 / 0           410 / 0 / 0
accuracy            1.0000                1.0000
facts written       10 (the lesson)       0
```

**Sixteen subjects read 1.00 cold and are printed with `already` beside them.** That is not a
result being dressed up: a decision procedure cannot be taught, and the report says so in each of
their notes. What they are worth is what `school.Arithmetic` says of itself — a subject that reads
1.00 cold is evidence **about the organ**, and an organ that quietly stopped working shows up here
on the first run rather than three subjects later as an unexplained dip. Their floor is not zero
because they are easy; every one of them scored 0.00 before the faculty existed.

**One subject moves, and it moves because doing and saying are different capabilities.**

```
vocabulary          0.50 → 1.00     +0.50
```

She can work out that 91 is not prime and, before the lesson, cannot say what a prime number *is*.
Twenty definitions, of which a seeded **half are taught and half deliberately withheld**; on the
taught half a right answer scores, on the withheld half **silence** scores and any assertion is a
miss. A brain that answered all twenty would read as mastery and be guessing, and the split is the
only thing that can tell those apart.

Two of the twenty phrasings had to be found by measurement, and each is a fact about the reader
rather than about mathematics. *"The median is the middle value **when** the values are in order"*
is filed as a **condition**, not a definition, and the term comes back unaskable. And the definite
article decides it: *"the mean is …"* does not read back where *"a mean is …"* does. A test asserts
that all twenty round-trip through `think`, because a lesson that does not land reports a gain it
did not produce.

The same measurement found the boundary this module then respects rather than widens. Written the
obvious way, *"the area of a circle is pi r squared"* grounds perfectly well — as
`('area of a circle', 'has_property', 'pi r squared')` — and is then **unaskable**, because "what
is the area of a circle" reads as `is_a` and a held `has_property` does not answer it.
`_GENERAL_ANSWER` exists to stop "what is a sparrow" being answered "brown", and a mathematics
lesson is not a reason to take that guard down. So the lessons are written in the shape that can
be read.

**One subject is entirely controls.** Ten questions that must come back empty, in two kinds: three
with no mathematics in them at all, and seven that are mathematics-shaped and **have no answer** —
a division by zero, a conversion between two quantities, a sequence with no rule, a shape whose
measurement was never given, a list with no mode. The second kind is the harder half: every one of
them reaches a skill, is recognised, and has to be declined *by that skill*. This subject exists
because the failure the module was written for was not silence — the five filed triples were
confident, and only a control can see that difference.

#### Where it sits in the turn

The mathematician is consulted in `_compose`, after the social-act gate and **before every
retrieval**, which is `_closed_arithmetic`'s rule generalised: a turn with a decision procedure has
an arm that cannot be wrong, so every other arm abstaining costs nothing. It is also the only stage
an *imperative* reaches — "expand (x+2)(x+3)" is not a question, `_deliberate` never sees it, and
the strategy table is unreachable from it. One path serves both the question form and the task
form, and that is why it is not a strategy.

**Bare arithmetic is deliberately excluded from it.** A closed expression already has a route: the
calculator is a registered strategy, the classifier is fed a parsed-expression flag so the critic
does not discard the value as an untested empirical claim, and `_closed_arithmetic` stops the
ladder guessing over it. That machinery was built against a measured failure and it works.
Answering `2+2` in the new organ would leave every piece of it in the source and unreachable —
dead wiring created on purpose. A test pins it from the other side: `5 ka square kitna hai?` is
answered 25, `thought.mathematics` is `None`, and the problem still classifies symbolic.

Run it: `NJPBrain.do_maths("...")` for one question with its working,
`NJPBrain.go_to_maths_school()` to be taught and examined, `NJPBrain.sit_maths_exam()` to be
examined only. The exam writes nothing and teaches nothing, so it may be run twice around
something that does.


### NJP V.24 — solving a problem she has never seen

V.23 ends with the mathematician at 410/410 on its own examination, and **that number says almost
nothing.** Every item on it is a shape the module already knows: fifty skills, fifty triggers, and
a question that matches a trigger is answered by the procedure behind it. That is dispatch. The
difference shows the moment a problem needs two steps.

Thirty problems, written to match no skill — multi-step commerce, a set-up-and-solve, a modular
exponent, a Diophantine count, an age ratio, an infinite series, a draw without replacement:

```
right                    1 / 30
confidently wrong        9 / 30
silent                  18 / 30
```

The nine wrong are the interesting half, exactly as the five filed triples were in V.23:

| problem | answered | why |
|---|---|---|
| marks up 40%, then discounts 25% — profit percent? | **30** | the discount skill firing on a percentage it recognised, inside a problem it did not |
| the remainder when 2^100 is divided by 7 | **2^100 in full** | the power skill matched; the word *remainder* was never read |
| two drawn **without replacement**, both red | **2/5** | the with-replacement answer, stated confidently |
| the hcf of 2⁴×3² and 2²×3³ | **1** | the exponents were handed to gcd as if they were the numbers |
| average speed 30 out and 60 back | **50** | the arithmetic mean of two speeds, which is never the answer |
| trailing zeros of 100! | **a 158-digit number** | 100! was computed; the question was not about 100! |

A trigger that matches half a problem answers half a problem, and there is nothing in a regex that
can notice the other half.

#### Nothing in `nyxara.njp.mathsolver` answers anything

A **reading** contributes *constraints*. The solver solves whatever set came out. The verifier
substitutes the solution back into every one of them, and only then may she speak. Two readings
that both match one sentence contribute both sets, and a two-step problem is solved by algebra
rather than by a skill someone wrote for two steps. **The chain is discovered, not enumerated.**

Three engines, tried in order:

* **algebra** — a system of polynomial equations in several unknowns over `Fraction`, solved
  exactly: Gaussian elimination when it is linear, substitution down to a univariate polynomial
  when it is not. An underdetermined system returns *nothing*, never a partial answer.
* **search** — a bounded integer search when the constraints are Diophantine or the problem asks
  for *the smallest number such that*. Exhaustive within a stated bound, so a value it returns is
  a solution and a range it walks is a proof there is none there.
* **counting** — the discrete closed forms that are not equations: modular exponentiation,
  Legendre's valuation, arrangements and selections, the shoelace area, the harmonic mean.

**"Verified" means two different things and the difference is stated rather than blurred.** An
algebraic answer is checked by substitution into the constraints that produced it. A closed form
has none to substitute into, so where a slower independent computation exists it is run and
compared — 100! really is computed and its zeros counted, the modular power really is checked
against the full power — and where none exists the answer is arithmetic on numbers already read.

#### Measured twice, because measuring once proves nothing

Thirty problems were written **first** and the solver built until they passed: 30/30. That is not
evidence. So thirty more were written *after it was finished*, three of them deliberately in shapes
nothing had been built for. First measurement: **23 right, 1 wrong, 6 silent.** After the seven
defects below: 30/30 on both.

And then, because two hand-written banks can both be overfitted: **eleven generated papers**, fresh
numbers on every seed, each computing its own expected answer by plain arithmetic on the numbers it
just chose — a route through no part of the solver. First measurement **0.9710**; after the
defects, **623/623 across three seeds**, controls included.

| defect | what it did |
|---|---|
| **a number at the end of a sentence was not a number** | written `(?![\w.])`, the reader saw *no number at all* in "the sum is 78." — the full stop failed the lookahead. Every multi-sentence problem lost its last quantity, silently, in the module V.23 shipped |
| **a recognised refusal fell through** | "the sum to infinity of 2, 4, 8, …" is read as a geometric series and refused, because that series has no sum — and the turn then reached the skill table, which added the three terms it could see and answered **14** |
| **"whose" is not an interrogative** | a guard written for "what colour" refused "find two numbers **whose** sum is 7" — twenty problems in a hundred, silently |
| **a question asking for no quantity was answered** | "the sum of three consecutive numbers is 78 — what is the **colour** of the largest?" was read perfectly and answered 27 |
| **an ordinal was excluded by value** | "the 15th term of 12, 15, 18" deleted the 15 that was a term |
| **a run was read non-greedily** | "… + 99 + 100" captured **99**, and the series summed was the wrong one |
| **a factor without an exponent was dropped** | "2^5 times 5" was read as 32, and the hcf of 200 and 160 came out 8 |

Two more were the *exam's* own, which is worth as much: a generator that wrote "the difference of
the digits is **-3**", a sentence nobody would write; and a phrasing fix for one shopkeeper problem
that stopped another from parsing — "marks up the price by 40%" and "marks his goods up by 20%" are
the same sentence with the noun moved.

#### A third bank, and the ceiling it found

Two banks passing is two banks. So a third was written after the first tier was finished, harder
still — a predicate search, a modular series of factorials, symmetric functions of roots, counting
over a range, stars and bars, a telescoping product, an inscribed circle:

```
right                    2 / 25
confidently wrong       12 / 25
silent                  11 / 25
```

And **three of the twelve came back as `noted:`** — filed into the knowledge store as facts, the
V.23 defect surfacing again on the problems that beat her. `('find', 'the') → 'smallest positive
integer n'` at confidence 0.75. The store protection covered problems she *recognised*; a problem
she could not read at all fell straight through it.

Sixteen readings and five engines later, the third bank is 25/25 and nine more generated papers
sit beside the eleven — **694/694 across three seeds** over the whole hard half. The most general
of the new readings is worth naming, because it is the closest thing here to what "solving"
means: **an arbitrary polynomial in one unknown, against a stated property, over an exhaustive
bounded range.** Nothing about the pair is enumerated in advance — the polynomial is parsed and
the property is compiled — so "the smallest n such that n² + n + 41 is not prime" (40) and "the
smallest n such that 2n + 1 is prime" (1) are answered by the same code, and a property it cannot
read is refused rather than approximated.

Four more defects, each found by a problem written after the code:

| defect | what it did |
|---|---|
| **an unsolved task was filed as a fact** | the store guard only covered problems a reading *claimed*; three the solver could not read at all were written into the knowledge store as triples |
| **the task flag then blocked a working skill** | its first version blocked the way a refusal does — and "expand (x+2)(x+3)" is a task the solver has no reading for and the skill table expands correctly, so a right answer became silence. Protecting the store and deciding who answers are two different questions and now two different flags |
| **an exponent's unknown could not be negative** | `3^(x+3) = 9` has x = -1; the search started at zero and reported that there was none |
| **`(a)(b)` is a function call** | to `ast.parse` it is, so a product written the way every textbook writes it could not be evaluated at all. The implied multiplication is inserted by the reading, rather than by widening what the calculator accepts |

And one in the exam itself: a generated predicate with no solution (`2n² + 3` is never divisible
by 9) threw `StopIteration` out of the examination rather than being skipped as the non-item it is.

#### A fourth bank, and where it stands now

The pattern repeated once more. A fourth bank, written after the second tier and harder in a
different direction — a boat against a stream, a sum that doubles, "at least one head", a
percentage carried to another percentage, an LCM and an HCF fixing a missing number, a function
defined and then used:

```
right                    2 / 20
confidently wrong        5 / 20
silent                  13 / 20
```

Seventeen more readings later it is 20/20, and ten more generated papers sit beside the twenty.
**All four hand-written banks — 105 problems — now pass through `think()`, and the thirty
generated hard papers score 927/927 across four seeds**, controls included, with nothing written
to the store.

Three of the defects that measurement found are worth keeping:

| defect | what it did |
|---|---|
| **an evaluation frame read as an equation** | "the value of 2x² + 9 **when** x = 5" contains an `=`; read as an equation it says x = 5 and answers **5** — the number in the question rather than the answer to it. And because a recognised refusal blocks, it took a paper that had been passing down with it |
| **one root named out of two** | the general equation reader refused a quadratic as having "more than one answer"; naming one is choosing rather than solving, and reporting both is neither |
| **a quantity read by exclusion** | the gap between simple and compound interest was found by removing the rate and the term from the numbers in the sentence — so on "at 10 percent is 10" it removed the gap itself |

And one about the exam again: a generator that `continue`d past every draw produced an **empty
paper**, which scores 0.00 and measures nothing. It now draws until the paper is full, and a test
asserts no paper is ever empty.

#### Where it sits, and what it must not shadow

**The solver is asked before the skill table**, and the order is the whole of its value: a reading
that carries several constraints is checked against all of them before she may speak; a skill that
matches one phrase is checked against nothing. Asked the other way round, "marks up 40% then
discounts 25%" is answered 30 and the solver never gets a turn. *Verified beats matched*, which is
the repo's *verifiable beats probabilistic* one layer further in.

A **recognised refusal blocks as firmly as an answer**. A problem the solver understood and
declined is never handed down to something that understands it less — that flag is the whole
reason the 14 and the 1/6 above are now silence.

And it must not shadow what already worked: a test pins `gcd of 48 and 18`, `expand (x+2)(x+3)`,
the triangle area and `24 + 18` as unchanged, and the twenty-eight-paper examination reports
**667/667**.

Run it: `NJPBrain.do_maths("...")`, `python -m nyxara.njp.mathschool --exam`, or `/v1/njp/maths`. Thirty generated papers in the hard half, sixty readings, twenty-two engines.


### Reachable over the wire

`/v1/njp/status`, `/fabric`, `/ledger`, `/think`, `/recall`, `/anticipate`, `/expand`, `/evolve`,
`/pulse`, `/learner`, `/calculate`, `/maths`, `/mathsolver`, and `/{organ}` — so growth and self-rewriting are observable
from outside the process,
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
