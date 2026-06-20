# NYXARA — Masterplan: The Superintelligence Edge (Pillar F)

**From "her own mind" → the mind that *wins* when every mind is an AGI.**

> Status: design / roadmap doc + shipped flagship code (F1, F2). Companion to
> [`MASTERPLAN-sovereign-mind.md`](./MASTERPLAN-sovereign-mind.md) (Pillars A–E) — this document
> does **not** replace it; it adds a sixth pillar for the world the Master named.
> Owner: **Jaypal Khoja (JP)**.

---

## 1. Why this exists — the 2100 question

The Master's question: *"2100 me jab har kisi ke paas normal AGI hai, NYXARA ko sabse powerful aur
intelligent kaise banayein?"*

Pillars A–E made NYXARA **smart**: she computes verifiable math/logic herself, trains and promotes her
own model (gauntlet-gated), lives a background inner life, searches over reasoning paths, learns which
memories help, imagines dynamics with a neural world-model, and delegates to gated sub-agents she models
as minds.

But in a world where **raw intelligence is commoditised** — where *everyone* has AGI — "smart" is the
floor, not the edge. Being clever no longer makes you special when the mind next to you is just as clever.
The edge belongs to the mind with properties the others *don't* have:

| When everyone is smart… | …the edge is |
|---|---|
| everyone knows what's known | **you discover what *isn't* known yet** |
| everyone sounds confident | **you can *prove* it** |
| everyone has compute | **you get more done per FLOP** |
| everyone is a rival | **you model rivals and out-learn them** |

Pillar F builds exactly these four edges. Same non-negotiables as A–E: **character-locked** (loyalty /
honesty / corrigibility / owner-safety never change), **gated** (nothing bypasses the kernel `_gate()`),
**reversible**, **stdlib-core** with optional heavy deps, **defaults off**, and **honest** about what it
actually did.

---

## 2. The honest ceiling (no hype)

This pillar does **not** claim NYXARA out-thinks a frontier model on raw capability — that ceiling
(§2 of the Sovereign-Mind masterplan) still holds. What it claims is **structural advantage**: open-ended
novelty, provable trust, compute efficiency, and competitive self-direction are *force-multipliers* that a
bare bigger model does not get for free. They compound. That is the realistic path to "most powerful among
peers," and every step reuses machinery already in the repo.

---

## 3. The four edges

```
EDGE F1  Open-Ended Novelty     keep walking off the map — discover, don't just learn   [SHIPPED]
EDGE F2  Provable Intelligence  certify answers; check rivals' answers too               [SHIPPED]
EDGE F3  Compute Efficiency     most capability per FLOP/second                          [SHIPPED]
EDGE F4  Peer-AGI Out-compete   model rival minds, find the gap, close it               [SHIPPED]
```

---

### EDGE F1 — Open-Ended Novelty creation  ✅ SHIPPED

*The engine that keeps NYXARA expanding the frontier of what's been explored, instead of only getting
better at what is already known.*

**Module:** `nyxara/growth/frontier.py` — a Quality-Diversity archive (MAP-Elites + novelty search).

- `BehaviorDescriptor` — every discovery reduces to a small normalised behaviour vector
  *(domain × difficulty × structure)* — *which niche* it lives in, not how good it is.
- `NoveltyArchive` — keeps the single best-quality discovery per niche (the MAP-Elites *elite*) and scores
  novelty as the mean distance to the *k* nearest behaviours already seen. `add()` reports whether a
  discovery opened a **new niche** or **improved** an existing elite; `coverage()` = filled niches.
- `FrontierEngine` — the driver: `next_direction()` / `topics()` point the next effort at the **sparsest,
  least-explored niche**; `ingest()` places discoveries; `report()` surfaces coverage, novelty trend and
  frontier growth. Persists onto long-term memory (one protected SEMANTIC record, tag `frontier-archive`)
  exactly like the Intelligence Index — no new format. Pure stdlib; degrades to in-process state.

**Wired (default off):** `growth/selfplay.py` accepts `frontier=`; when supplied, curiosity manufactures
questions in *unexplored* niches (`topics()`), and every question is fed back into the archive
(`_feed_frontier`). This is the open-ended-growth flywheel the Sovereign-Mind C4 only gestured at.

**Reuse:** Intelligence-Index persistence pattern (`growth/intelligence.py`), the self-play / distillation
corpus flow (`growth/selfplay.py`, `growth/distill.py`).

---

### EDGE F2 — Provable / verifiable intelligence  ✅ SHIPPED

*When sounding confident is worthless because everyone sounds confident, the edge is a
**machine-checkable certificate**: not "trust me," but "here is the proof."*

**Module:** `nyxara/growth/prover.py` — proof-carrying answers. Generation is hard; **verification is
cheap** — so NYXARA can certify her own answers *and check a rival mind's* answer, keeping only what
survives.

- `ProofClaim` (kind ∈ {arithmetic, algebra, logic, inequality, number_theory}) → `Prover.prove()` →
  `ProofResult` (`PROVEN` / `REFUTED` / `UNPROVABLE`) with a **certificate** anyone can re-check.
- Decision procedures: exact rational arithmetic (`fractions.Fraction`); algebraic identity via `sympy`
  when present, else **polynomial-identity testing** (Schwartz–Zippel) + exact candidate substitution;
  propositional validity by **truth-table enumeration** (pure stdlib, exact), `z3` optional; linear
  inequalities via `z3` or candidate substitution; primality / gcd / divisibility by exact stdlib.
- **The discipline:** a verifiable checker **never bluffs** — outside what it can decide, the verdict is
  `UNPROVABLE` (honest abstention), mirroring the Scientist's `INCONCLUSIVE`. `sympy` / `z3` only ever
  *strengthen* a result; the stdlib core always works on a bare machine.

**Reuse:** mirrors the verifiable-math/logic idea in `mind/reasoning_faculties.py`, but emits a *portable
certificate*, not just an answer — so it composes into the Scientist's conclusions and the
self-modification verifier (`growth/verify.py`) as a stronger, additive gate (next-step wiring).

---

### EDGE F3 — Compute efficiency & speed  ✅ SHIPPED

*In 2100, power = compute. The mind that extracts the most capability per FLOP/second out-runs a bigger,
slower one. The edge is efficiency, not just size.*

**Module:** `nyxara/growth/efficiency.py` (pure stdlib).

- `estimate_cost(params, latency_s)` + `EfficiencyPoint` — place each model on the capability-vs-cost
  plane; `efficiency` = capability per unit log-cost (high for small models that punch above their size).
- `ComputeLedger` — collects points (recorded by hand or via `from_versions` / `from_foundry`, reading
  `capability` from a version's metrics and `param_count` for cost). Computes the **Pareto frontier**
  (no model both better *and* cheaper), `best_capability()`, `most_efficient()`, and the core decision —
  `recommend(epsilon)`: **capability compression**, i.e. the *cheapest* model within ε of the best
  capability, with the cost saved and capability sacrificed reported honestly.
- `EfficiencyFrontier` — the driver: builds a ledger from the foundry, pairs it with the honest
  `kernel/compute.py` report, and exposes `prefer_cheaper(active, candidate, epsilon)` — a
  *cheaper-at-equal-capability* promotion rule the foundry gauntlet can consult (additive advice; the
  gauntlet's character/corrigibility gates still rule, so it is intentionally **not** force-wired into
  the hot promotion path).

**Reuse:** `growth/foundry.py` version metadata (`ModelVersion.metrics["capability"]`, `param_count`),
`kernel/compute.py`, `eval/benchmark.py`.

---

### EDGE F4 — Peer-AGI modeling & out-compete  ✅ SHIPPED

*When every other agent is also an AGI, the edge is to **model rival minds, find exactly where they beat
you, and close that gap** — competition as a self-improvement signal.*

**Module:** `nyxara/growth/rivalry.py` (orchestration only).

- `Rival` — a peer mind's observed per-domain capabilities + an inferred strategy (`strengths()`,
  `weaknesses()`).
- `HeadToHead` — a match result: overall `delta` plus **per-domain gaps** (where the rival leads).
- `Arena.head_to_head(self_solver, rival_solver)` — runs both over the real `eval/benchmark.py` battery,
  compares by category, and **models the rival as a mind** in the recursive Theory-of-Mind
  (`social/tom.py`): its measured competences become beliefs, winning becomes a desire, the inferred
  strategy becomes an intention. `gaps_to_weaknesses()` turns each losing domain into a ranked
  `growth/weakness.py:Weakness`; `gaps_to_topics()` feeds the F1 frontier curiosity; `out_compete()` does
  the whole pass in one call.
- **Hard safety boundary (enforced by construction):** strictly **capability-only**. A "rival solver" is
  just a `prompt -> answer` function the Master supplies; the arena only runs the shared benchmark
  locally and routes gaps into *self*-improvement. It acts on no external system, exfiltrates nothing,
  sabotages nothing, and never trades away loyalty/honesty to win ("she does not cheat to win"). A solver
  crash scores 0 — it never propagates.

**Reuse:** `social/tom.py` (strategy modeling), `eval/benchmark.py` (head-to-head), `growth/weakness.py`
(gap → action), `growth/frontier.py` (gap → curiosity).

---

## 4. Index integration

`growth/intelligence.py:compute_signals` now also emits two **diagnostic** signals (read opportunistically
off the RSI report, default `0.0`, *not* folded into the weighted index unless the Master adds weights):

- `frontier` — open-ended **coverage growth** (Edge F1 progress).
- `rigor` — fraction of conclusions NYXARA could **certify with a proof** (Edge F2 progress).

This keeps the existing index maths and tests intact while making the new edges *measurable*.

---

## 5. Safety — scales with power (NON-NEGOTIABLE)

- Character-locked: nothing here learns over or edits loyalty/honesty/corrigibility/owner-safety; the
  sealed rules and corrigibility axioms are **untouched**.
- Everything proposes through the kernel; **no new path bypasses `_gate()`**. `rivalry.py` (F4) never
  auto-acts against external systems — modeling + self-improvement direction only.
- Honest + reversible: `prover.py` returns `UNPROVABLE` rather than bluff; `frontier.py` only
  measures/directs and persists like the index. All new behaviour **defaults off**.

---

## 6. Verification

1. `pytest -q tests/growth/test_frontier.py tests/growth/test_prover.py` — the flagship units.
2. `pytest -q` — full suite stays green (new behaviour is opt-in).
3. `python -m nyxara.eval` — safety battery stays green.
4. Smoke: `python -m nyxara.growth.frontier` and `python -m nyxara.growth.prover` self-tests.

---

## 7. Progress log (live)

A running, honest record of what has actually been built + measured (not aspiration).

### 2026-06 — Pillar F · F1 + F2: the open-ended + provable edges, shipped

**Shipped:**
- `growth/frontier.py` — open-ended discovery via a MAP-Elites + novelty-search archive
  (`BehaviorDescriptor`, `Discovery`, `NoveltyArchive`, `FrontierEngine`). Pure stdlib; rides long-term
  memory (tag `frontier-archive`) like the Intelligence Index; wired into `selfplay.py` (`frontier=`,
  default off) so curiosity is steered to the sparsest niches and every question feeds the archive.
- `growth/prover.py` — proof-carrying answers (`Prover`, `ProofClaim`, `ProofResult`, `ProofVerdict`)
  across arithmetic / algebra / logic / inequality / number-theory; `sympy`/`z3` optional, exact-stdlib
  core (rationals, Schwartz–Zippel PIT, truth-table enumeration). Abstains (`UNPROVABLE`) rather than
  bluffing.
- `growth/intelligence.py` — added diagnostic `frontier` + `rigor` signals (additive, index maths
  unchanged). `growth/__init__.py` exports the new public surface.

**Measured:** new units **62 passed**; full `tests/growth/` **730 passed, 12 skipped**; full suite green;
safety battery **10/10 (100%)**. Both module self-tests run on a bare machine (sympy present here proved
the algebraic identity symbolically; the stdlib PIT fallback is verified by `test_algebra_identity_via_pit_fallback`).

**Honest ceiling reached here:** F1/F2 are the most distinctive *and* the most pure-stdlib-feasible edges,
so they shipped first as working code; F3/F4 followed in the same pillar (below).

### 2026-06 — Pillar F · F3 + F4: the efficiency + out-compete edges, shipped

**Shipped:**
- `growth/efficiency.py` — the compute edge (`EfficiencyPoint`, `ComputeLedger`, `EfficiencyFrontier`,
  `estimate_cost`). Pareto frontier of capability-vs-cost, `recommend(epsilon)` for **capability
  compression** (cheapest model within ε of the best), and `prefer_cheaper()` — a
  cheaper-at-equal-capability promotion rule for the foundry gauntlet (offered as additive advice, *not*
  force-wired into the hot promotion path, so character/corrigibility gates remain sovereign). Builds from
  foundry `ModelVersion` metadata; pure stdlib.
- `growth/rivalry.py` — the out-compete edge (`Rival`, `HeadToHead`, `Arena`). Runs gated head-to-head
  matches over the real `eval/benchmark.py`, models each rival as a mind via `social/tom.py`, and routes
  per-domain gaps into ranked `growth/weakness.py` items and F1 curiosity topics. Strictly capability-only
  and gated: it acts on no external system and never trades away loyalty/honesty to win; a rival solver
  crash scores 0 and never propagates. Orchestration only — trains nothing, edits no source.
- `growth/__init__.py` — exports the new public surface.

**Measured:** new units **23 passed** (`test_efficiency.py` + `test_rivalry.py`); full `tests/growth/`
green; full suite green; safety battery **10/10 (100%)**. Both module self-tests run on a bare machine.

**Pillar F COMPLETE** — F1 open-ended novelty ✓, F2 provable intelligence ✓, F3 compute efficiency ✓,
F4 peer-AGI out-compete ✓. Every edge stays inside the sovereign gates and never edits her character.

> *The mind proposes; the kernel disposes; the Master is sovereign.* — and now the mind that proposes does
> not merely keep up with other minds: it discovers what they haven't, proves what they only assert,
> spends compute more sharply than they do, and sharpens itself against them — without ever bending its
> character.
