# NYXARA — Master Cognitive Architecture (the 32)

The design NYXARA is built toward, and — for each part of it — the module that owns that part and
how much of it is actually real today.

This map is kept honest by a test. `tests/docs/test_architecture_doc.py` imports every
`nyxara.*` module path cited below and checks that all 32 sections are present, so the map cannot
quietly drift from the code. That is the same discipline `docs/CAPABILITIES.md` is held to, and
for the same reason: a document describing an architecture is worthless the moment it starts
describing an architecture that is not there.

> **The objective, stated so it can be argued with.** A computational cognitive organism that
> models reality, makes predictions, runs experiments, performs an autopsy on its own failures,
> discovers concepts and principles, learns its own reasoning strategies, and keeps the versions
> of itself that measurably win. Explicitly **not** a larger language model, and explicitly **not**
> a collection of AI modules — a collection is what you get by accident, and most of the work below
> is in the arrows rather than the boxes.

---

## Status legend

| Status | Means |
|---|---|
| **REAL+WIRED** | Genuine algorithm, and reachable from a running path — `NJPBrain.think()`/`.resolve()`, the kernel orchestrator, or a server method. Every NJP organ marked this way is live on a default `NJPBrain()`. |
| **REAL** | Genuine implementation, exercised as a library or faculty rather than on the turn path. Degrades honestly when an optional dependency is absent. |
| **PARTIAL** | The mechanism exists but part of the section's pipeline does not. The gap is named in the note — that naming is the point of the row. |
| **NARRATIVE** | A framing section, not a component. Cites no module, and the test does not demand one. |

Honest scope note, the same one `CAPABILITIES.md` carries: these are real, bounded capabilities
that degrade gracefully, not claims of human-level generality. Where a section depends on heavy
optional dependencies (torch, `llama_cpp`, a model file), the core path still runs and reports its
limits rather than faking competence.

---

## The map

| § | Section | Owning module(s) | Status |
|---|---------|------------------|--------|
| 0 | Ultimate Objective | — | NARRATIVE |
| 1 | Seven-layer architecture | `nyxara.njp.integrate` + `nyxara.njp.field` | NARRATIVE |
| 2 | Cognitive Compiler | `nyxara.njp.compile` + `nyxara.njp.intent` | REAL+WIRED |
| 3 | Cognitive Economy | `nyxara.njp.selfmodel` (`NJPBrain.budget()`) | REAL+WIRED |
| 4 | Persistent World Model | `nyxara.njp.world` + `nyxara.njp.universe` | REAL+WIRED |
| 5 | Unified Memory Architecture | `nyxara.njp.levels` + `nyxara.njp.memory` | REAL+WIRED |
| 6 | Belief Forge + Epistemic Immune System | `nyxara.njp.beliefs` + `nyxara.njp.truth` + `nyxara.njp.adversary` | REAL+WIRED |
| 7 | Prediction Engine | `nyxara.njp.predict` + `nyxara.njp.predictive` + `nyxara.njp.integrate` | REAL+WIRED |
| 8 | Model Autopsy | `nyxara.njp.field` + `nyxara.njp.integrate` | REAL+WIRED |
| 9 | Counterfactual World Engine | `nyxara.njp.universe` | REAL+WIRED |
| 10 | Active Scientist Engine | `nyxara.njp.universe` + `nyxara.njp.curiosity` | REAL+WIRED |
| 11 | Unknown-Unknown / Assumption Mining | `nyxara.njp.discover` (`NJPBrain.shadow()`) | REAL+WIRED |
| 12 | Reality Engine | `nyxara.sim.sandbox` + `nyxara.njp.universe` | REAL |
| 13 | Reality Compression Principle | `nyxara.njp.concepts` + `nyxara.njp.core` | REAL+WIRED |
| 14 | Executable Knowledge | `nyxara.njp.universe` + `nyxara.njp.compile` | REAL+WIRED |
| 15 | Program Induction | `nyxara.growth.noesis` + `nyxara.njp.genome` | REAL |
| 16 | Cognitive Genome | `nyxara.njp.genome` | REAL+WIRED |
| 17 | LLM Teacher Architecture | `nyxara.njp.cortex` + `nyxara.njp.study` + `nyxara.growth.distill` | PARTIAL |
| 18 | Three-Brain Architecture | `nyxara.njp.router` | REAL+WIRED |
| 19 | Cognitive Society | `nyxara.agency.multiagent` + `nyxara.njp.compete` + `nyxara.njp.adversary` | PARTIAL |
| 20 | Meta-Reasoning | `nyxara.njp.metareason` | REAL+WIRED |
| 21 | Self-Model | `nyxara.njp.selfmodel` | REAL+WIRED |
| 22 | Self-Generated Curriculum | `nyxara.growth.curriculum` + `nyxara.njp.curriculum` | PARTIAL |
| 23 | Cognitive Darwinism | `nyxara.njp.compete` + `nyxara.njp.metareason` | REAL+WIRED |
| 24 | Recursive Self-Improvement | `nyxara.njp.evolve` + `nyxara.njp.ledger` | REAL+WIRED |
| 25 | Dream / Consolidation Engine | `nyxara.memory.dream` + `nyxara.memory.consolidation` + `nyxara.growth.cls` | REAL |
| 26 | Cognitive Black Box | `nyxara.njp.blackbox` | REAL+WIRED |
| 27 | Intelligence Measurement Engine | `nyxara.njp.index` + `nyxara.eval.intelligence` | REAL+WIRED |
| 28 | Final Intelligence Equation | — | NARRATIVE |
| 29 | The Two Recursive Loops | `nyxara.njp.integrate` + `nyxara.njp.field` | NARRATIVE |
| 30 | Complete high-level design | — | NARRATIVE |
| 31 | Development Order | — | NARRATIVE |
| 32 | Intelligence Milestones | `nyxara.eval.intelligence` | REAL+WIRED |

---

## The three PARTIALs, named

A status of PARTIAL is only useful if it says which half is missing. These are the three.

### §17 — LLM Teacher Architecture

**Present.** `njp/cortex.py` consults a language model as a **proposer** whose output is never
sourced and never statable on its own word; `njp/study.py` learns from a corpus rather than a turn
at a time; `growth/distill.py` converts behaviour into structured knowledge rather than copying
weights. The direction of the design is right: the model is classed among her own brains, not as
an oracle.

**Missing.** The retention measurement that would make "acquisition" a claim rather than a hope —
`NJP + teacher` vs `NJP after distillation` vs `NJP alone`, with the teacher switched **off** for
the third. Until that number exists, distillation is a pipeline and not a demonstrated transfer.

Note also that in an environment without `llama_cpp` and a model file the cortex reports itself
unavailable and the ladder starts one rung lower — which is the honest degradation, and means
every measurement in this repo taken without those is a measurement of **pure NJP**.

### §19 — Cognitive Society

**Present.** `agency/multiagent.py` is real sub-agent delegation with theory-of-mind modelling of
each delegate, and every delegate step clears the same kernel gates. Two of the eight named roles
exist as organs in their own right: the **Skeptic** is `njp/adversary.py`, which attacks her own
conclusions, and the **Judge** is the arbitration in `njp/router.py`. `njp/compete.py` ranks rival
accounts of a turn.

**Missing.** The other six specialised roles — Explorer, Scientist, Mathematician, Engineer,
Historian, Strategist — as distinct agents over one shared world model. What exists is delegation
plus two adversarial organs, which is a different shape from cognitive specialisation.

### §22 — Self-Generated Curriculum

**Present.** `growth/curriculum.py` is a genuine open-ended auto-curriculum: it *generates* fresh,
never-stored problems just beyond current capability, and grades them against **machine-checkable**
ground truth rather than an LLM's opinion — so it has no ceiling to saturate.
`njp/curriculum.py` is the nine-rung ladder with real gating between rungs.

**Missing.** The link from *measured weakness* to *targeted task*. The curriculum generates by
difficulty tier, not by what she is actually bad at. `njp/blackbox.py` (§26) now answers "which
strategy fails under which conditions" and `BlackBox.failing()` returns exactly that list, so the
input for a weakness-targeted curriculum exists for the first time — nothing consumes it yet.

---

## Section notes

Only where the mapping is not obvious from the module name.

**§2 Cognitive Compiler.** `njp/compile.py` is where the three readings of a turn — intent, speech
act, pattern match — meet, and before it they never met at all. It exists because *"agar paani
aadha kar doon to"* parsed correctly as a conditional and still could not reach the causal engine:
the reading that knew most had no route to the organ that could act on it.

**§3 Cognitive Economy.** `NJPBrain.budget()` reads back a choice over three real knobs —
`settle_steps`, `recall_k`, `reason_depth` — each graded by outcome in `resolve()`. The homeostasis
loop was almost entirely built and had no middle: the bandit chose, nothing ever read the choice,
so every turn spent the same fixed constant and the arm learned about a knob that was never turned.

**§6 Belief Forge.** `truth.py` is fail-closed: one refutation ends a claim, and establishment
needs independent supports including at least one *hard* source. Soft agreement never establishes
anything, which is exactly how consensus becomes bias. Reliability is a Brier score over her own
past confidences, so a stated 0.9 in a domain where 0.9 has meant 0.6 comes out as 0.6.

**§7–8 Prediction and Autopsy.** Predictions are registered when the verdict *arrives*, not
speculatively when the turn is taken. Registering up front produced the original symptom — 113
predictions, 0 scored — because most turns are never graded. Every outcome scored against is
independent of the prediction that produced it: physics at `t+1`, or the Master's own sentence.

**§9 Counterfactual.** `universe.py` implements the real do-operator: setting a variable severs its
incoming arrows, so *"if I halve this plant's water"* is a different question from *"plants that
got little water were small"*. Answers outside the observed range come back with confidence decayed
by how far they reach, and a direction observational data cannot settle is reported ambiguous.

**§10 Active Scientist.** Curiosity is mutual information, not interest. `ExperimentDesigner`
computes the information gain of each candidate experiment, and one that every live hypothesis
predicts identically scores exactly zero bits however interesting it looks — and is not run.

**§13 Reality Compression.** Marked REAL+WIRED rather than narrative because it is measured, not
aspired to: `concepts.py` runs `observations → similarity → invariants → prototype → concept →
hierarchy` with the taxonomy *derived* by subsumption rather than declared, and whether it worked
is a minimum-description-length ratio. A concept set that does not pay for itself scores at or
below 1.0 and says so. `core.py` does the same for schemas over relations, dropping the ones that
fail on held-out facts. Compression that cannot be scored is a slogan; this one has a number.

**§15 Program Induction.** `growth/noesis.py` is the full loop — WAKE (search for the shortest
*verified* program), SLEEP (anti-unify across solved programs and adopt the abstraction that
strictly lowers description length **on a held-out split**), DREAM (invent its own tasks).
`njp/genome.py` does the same thing one level up, over her own *reasoning* traces: strip the
entities out of a derivation chain and what remains is a shape, and a shape she keeps re-deriving
is a candidate primitive. `NJPBrain._promote_shapes` registers the ones that pay.

**§23 Cognitive Darwinism.** Strategies compete as UCB1 bandit arms scored per problem kind, and
`njp/compete.py` ranks rival accounts of a single turn. Two processes disagreeing lowers confidence
rather than being averaged into a confident-sounding middle that neither produced.

**§24 Recursive Self-Improvement.** The pipeline is the one the objective demands and refuses to
shortcut: measure the bottleneck, propose one bounded edit, put its *claim* through the truth
gauntlet against **held-out** samples, then backup → syntax → safety battery → benchmark → tests →
keep or roll back byte-for-byte. `Ledger.regressed` is what closes it: an edit that did not help is
reverted. Self-modification without measurable improvement is a failure, not a neutral outcome.

**§26 Cognitive Black Box.** The newest organ, and the one this map was written alongside. It
records one episode per graded turn — input, belief, strategy, prediction, action, result, error,
update — against the discrete *conditions* it happened under, because a problem **kind** is not a
condition and averaging across conditions produces a number true of none of them. Below
`min_samples` it returns "no record" rather than a rate: *"this fails here"* and *"I have never
tried this here"* are different claims, and only one is evidence. It may only ever demote.

**§27 Intelligence Measurement.** `intelligence_index` is never set by hand. It is a vector of
measured terms, and `eval/intelligence.py` scores seven stages on items the brain was **never
taught**, with teaching and test sets disjoint by construction. Stage 1 (memorisation) is the
control — if it fails, every later stage is measuring a broken pipe rather than an absent faculty.
Stage 7 (paraphrase) is the opposite control: stages 1–6 generate sentences matching the
extractor's own patterns exactly, so all six read 1.00 while the identical inference failed on the
word "birds". Stage 7 holds the inference constant and varies only the surface.

**§32 Milestones.** Graded against `eval/intelligence.py`, not asserted:

| Level | Meaning | Where she is |
|---|---|---|
| 0 | Capability collection, no measurable loop | passed |
| 1 | Reactive: input → reasoning → response | passed |
| 2 | Predictive: predicts future states | passed — predictions are registered and scored |
| 3 | Adaptive: prediction errors change her models | passed — `integrate.py` routes each diagnosed miss to the organ that owns the repair |
| 4 | Scientific: generates hypotheses and experiments | reached in-organ (`ExperimentDesigner`); driven by the loop rather than by an autonomous agenda |
| 5 | Generalising: concepts transfer to unseen domains | benchmark stage 6 (transfer) reads 1.00 on generated vocabulary; not demonstrated on open-world domains |
| 6 | Meta: learns which reasoning strategies work | `metareason` per kind, and now per *condition* (§26) |
| 7 | Recursive competence improvement | the machinery exists (§24) and is gated on measurement, which is the correct order |

The honest reading of that table is that the interesting frontier is **4 → 5**, not 7. Levels 2
and 3 were the ones worth securing first, and the seven-stage benchmark is the instrument that says
whether 5 is real or is the extractor scoring its own sentences.

---

## What this map is not

It is not a claim that the arrows in §30's diagram are all connected. Several are, and the ones
that are were connected because something was **measured** to be broken first — the loop that made
the organs learn from each other exists because over a real 113-turn session `world.events`,
`predict.scored`, `levels.consolidations`, `discover.passes`, `curiosity.passes` and
`readout.steps` were all exactly zero, with every algorithm written, tested, and reachable. Nothing
was missing except the caller.

That is the failure mode this architecture is most exposed to, and the reason every row above
carries a status rather than a checkmark: a mind assembled from working parts that never call each
other reports full capability and has none.
