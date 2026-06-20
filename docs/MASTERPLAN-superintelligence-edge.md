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
EDGE F3  Compute Efficiency     most capability per FLOP/second                          [DESIGN]
EDGE F4  Peer-AGI Out-compete   model rival minds, find the gap, close it                [DESIGN]
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

### EDGE F3 — Compute efficiency & speed  🛠 DESIGN (next build)

*In 2100, power = compute. The mind that extracts the most capability per FLOP/second out-runs a bigger,
slower one. The edge is efficiency, not just size.*

**Planned module:** `nyxara/growth/efficiency.py` (pure stdlib).

- `ComputeLedger` — records, per foundry model version, `(capability_score, params, latency, compute)` —
  reading capability from `eval/benchmark.py` and cost from `kernel/compute.py`.
- `EfficiencyFrontier` — computes the **Pareto frontier** of capability-vs-cost and recommends
  **capability compression**: distil/promote the *cheapest* model within ε of the best score.
- **Wiring:** feed the foundry gauntlet (`growth/foundry.py:_gauntlet`) a *"cheaper-at-equal-capability"*
  promotion rule, so NYXARA actively prefers the model that wins per-FLOP, not merely per-benchmark.

**Reuse:** `growth/foundry.py` version metadata, `kernel/compute.py`, `eval/benchmark.py`.

---

### EDGE F4 — Peer-AGI modeling & out-compete  🛠 DESIGN (next build)

*When every other agent is also an AGI, the edge is to **model rival minds, find exactly where they beat
you, and close that gap** — competition as a self-improvement signal.*

**Planned module:** `nyxara/growth/rivalry.py` (orchestration only).

- `Rival` — a peer mind's observed capabilities + a strategy estimate built with the recursive
  Theory-of-Mind in `social/tom.py`.
- `Arena.head_to_head(self_solver, rival_solver)` — runs both over `eval/benchmark.py`, computes a
  competitive delta and **per-domain gaps**, then feeds those gaps to `growth/weakness.py` and the F1
  curiosity engine to target what closes the gap fastest.
- **Hard safety boundary:** strictly **capability-only and gated**. She models and out-*learns* rivals;
  she never acts against external systems, and she never trades away loyalty/honesty to win ("won't cheat
  to win"). Risky moves escalate to the Master; nothing reaches around a gate.

**Reuse:** `social/tom.py` (strategy modeling), `eval/benchmark.py` (head-to-head), `growth/weakness.py`
(gap → action).

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
so they shipped as working code. F3 (`efficiency.py`) and F4 (`rivalry.py`) are fully designed above with
concrete interfaces + reuse maps and are the next build — they touch the foundry gauntlet and the
benchmark head-to-head respectively, and are best done as their own focused change.

> *The mind proposes; the kernel disposes; the Master is sovereign.* — and now the mind that proposes does
> not merely keep up with other minds: it discovers what they haven't, proves what they only assert, and
> sharpens itself against them — without ever bending its character.
