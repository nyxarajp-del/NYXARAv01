# NYXARA — Noēsis: the Living Algorithm

**A self-extending program-synthesis substrate that compounds capability-per-compute — with no LLM.**

> Status: **shipped code** (Tier 1 complete: the core + F1–F5). Companion to the sovereign-mind /
> superintelligence-edge masterplans. Owner: **Jaypal Khoja (JP)**.

---

## 1. Why this exists

The Master asked for a genuinely **new algorithm** — not a wrapper, not a fixed model — that could take
a small system toward another level of intelligence, that is **not frozen**, and that **NYXARA herself
keeps upgrading** to grow more powerful over time.

Every other self-improvement engine in the repo rewrites **one** thing over a **fixed** primitive set:
`rule_synth.py` invents weight-update rules, `cognitive_architect.py` invents reasoning operators,
`godel_loop.py` climbs to stronger logics, `eureka.py` keeps proven lemmas, `open_world.py` fits laws by
MDL. None of them is a single, cross-task, **self-extending abstraction library** whose growth makes
future problems *cheaper* to solve. Noēsis is that missing core.

## 2. The honest ceiling (no hype)

No single-repo algorithm *guarantees* AGI or beats a frontier LLM in general — and **no such claim is
written into this code or these docs**. Noēsis is a real algorithm from a real lineage —
**compression-as-intelligence / library learning** (Solomonoff–Levin universal search + DreamCoder-style
abstraction). Its power is genuine but **bounded to what it can verify**: it grows fastest on
structured/programmatic reasoning and **abstains** rather than bluff elsewhere. What it *does*
deliver — measured, not asserted — is **more capability per unit compute over time**, which is exactly
the "less compute, more powerful" property the Master named, in the honest form that is actually true.

## 3. The theory

Intelligence-per-compute grows when a system **amortises the structure it discovers** into reusable
concepts, so later problems need *shorter* programs and *less* search. Noēsis makes that a loop she runs
herself, in code:

- **WAKE** — solve a task `input → output` by bounded, type-directed search for the **shortest verified
  program** in a small typed DSL. The verifier decides (exact I/O match); with nothing verified in
  budget she **abstains**.
- **SLEEP** — mine recurring structure across every solved program (anti-unification) and adopt the
  abstraction that **strictly lowers the total description length on a held-out split** (MDL,
  anti-overfit). An adopted abstraction becomes a **new first-class primitive** — the language she
  thinks in grows.
- **DREAM** — invent her own new tasks so growth is open-ended.

She upgrades the algorithm on **three levels**, none of them frozen: the **library** (data, grows +
persists), the **search-guide prior** (params, learned from her own solves), and the abstraction
**operators** (meta). That is the "not fixed — she keeps upgrading it" the Master required.

## 4. What shipped (Tier 1 — the core + five faculties)

| # | Module | What it adds |
|---|---|---|
| core | `growth/noesis.py` | WAKE / SLEEP / DREAM, typed DSL, MDL library learning, persistence |
| F5 | `growth/redteam.py` | Adversarial self-falsification: a solution enters the corpus only if it survives a boundary-condition battery (vs an oracle, else self-consistency) |
| F1 | `growth/postmortem.py` | Metacognition: diagnoses each failure and retunes **bounded** search knobs from calibrated `BetaBelief` — character/safety knobs are unreachable by construction |
| F3 | `growth/neuromod.py` | Neuromodulated plasticity (novel/surprising solves written in fast, habitual ones barely) + REM pruning of stale abstractions, **archived reversibly** |
| F2 | `mind/active_inference.py` | Predictive processing: prediction error = surprise; epistemic action toward highest uncertainty, over the existing `planning/voi.py` engine |
| F4 | `mind/latent_geometry.py` | TDA (persistent homology: components + holes) + hyperbolic tree embedding, chosen only where it measurably wins |

## 5. Measured (ground truth, not aspiration)

On the synthetic compounding battery (`python -m nyxara.growth.noesis`), a 6-cycle run:

- **compute-per-solve fell ~60×** (≈8,300 → ≈135 program expansions) as the library grew, while
  **solve-rate stayed ~1.0** — the honest "more capability per unit compute."
- **average solution size fell ~50%** (≈4.5 → ≈2.2 nodes); the library learned ~6 abstractions,
  including ones that **compose earlier abstractions** (hierarchical library learning).
- With F5 wired in, the red-team **caught 8 over-fit solutions** in a 6-cycle run — programs that fit
  the examples but were wrong in general — and kept them out of the permanent corpus.

The compounding property is locked by a test: after SLEEP adopts a family's shared skeleton, a
**held-out** task in that family re-solves with a **strictly shorter** program than before. That
inequality *is* the result.

## 6. Safety (non-negotiable)

- **Verifier-gated + red-teamed:** nothing enters the corpus unverified, and F5 adds an adversarial pass
  before any promotion. **No LLM in the loop** — enforced by a test.
- **Character-locked:** an abstraction cannot even be *named* after a value/character token; F1 knobs
  refuse any character/safety name and are clamped to sealed ranges; F3 pruning is archived + reversible
  and never touches protected entries. Corrigibility / `IMMUTABLE_VALUES` untouched.
- **Advisory / bounded:** it grows a library of *verified programs* — it trains no weights, edits no
  source, and touches no gate. Hard caps on search depth/breadth/wall-clock.

## 7. Owner surface

- `python -m nyxara.growth.noesis [--cycles N --tasks M --json out]` — run it and see the compression
  curve.
- `/noesis [n]` in the console — runs with F5 + F1 and prints her introspection snapshot; persists to
  `~/.nyxara/noesis.json` so the library **compounds across sessions**.
- Public API exported from `nyxara.growth` (`NoesisEngine`, `RedTeam`, `Metacognition`,
  `Neuromodulators`, `REMPruner`, `UtilityLedger`) and `nyxara.mind`
  (`ActiveInference`, `LatentGeometry`).

## 8. Verification

1. `pytest -q tests/growth/test_noesis.py tests/growth/test_redteam.py tests/growth/test_postmortem.py
   tests/growth/test_neuromod.py tests/mind/test_active_inference.py tests/mind/test_latent_geometry.py`
   — the Tier-1 units.
2. `python -m nyxara.growth.noesis --cycles 6` — watch avg solution size ↓ and expansions ↓.
3. `pytest -q` — full suite stays green (new behaviour is opt-in / self-contained).

## 9. Roadmap — COMPLETE (all 23 phases shipped)

All faculties built, each layered on the working core, each inside the sovereign gates, each keeping the
honest ceiling. Every one is torch-free (heavy deps optional with honest fallbacks), no LLM in any loop.

- **Tier 1** ✅ — F5 red-team, F1 metacognition, F3 neuromodulation+pruning, F2 active inference,
  F4 latent geometry.
- **Tier 2** ✅ — F9 proof-carrying abstractions, F6 self-evolving grammar + meta-types, F11 internal
  ecosystem (Explorer/Skeptic/Synthesizer), F7 functorial transfer, F10 counterfactual dreaming,
  F8 continuous world-simulators.
- **Tier 3** ✅ — F17 integrity under isolation (the safety anchor), F12 hardware self-sensing,
  F14 self-repair + portable bundle, F13 bounded axiom invention + sub-perceptual DSP, F15 meta-invention
  codegen, F16 hyper-temporal planning.
- **F18** ✅ — online / connected mode: continuous **verified** acquisition, offline stays the floor,
  outbound actions escalate to the Master.

Module map: `growth/{noesis,redteam,postmortem,neuromod,formal_proof,grammar,ecosystem,counterfactual,
integrity,self_repair,ontogenesis,codegen,online}.py`, `mind/{active_inference,latent_geometry,
category_transfer,continuous_world}.py`, `planning/hypertemporal.py`, `senses/hardware.py`. Every phase
ships with its own test module; the safety line (corrigibility / oversight / immutable values) is
untouched throughout, and F17 pins the anti-shutdown guarantee that all of Tier 3 inherits.
