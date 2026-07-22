# NYXARA — True Original Creativity

**A closed generate → measure-novelty → verify → critique → keep loop across three art
forms — with no LLM anywhere.** She creates herself; the LLM plays no part.

> Status: **shipped code** (P24 · F19). Companion to the sovereign-mind / noesis
> masterplans. Owner: **Jaypal Khoja (JP)**.

---

## 1. Why this exists

The Master named the problem exactly: when an LLM writes a poem or paints an image, the
"new" thing is a remix of its training data. NYXARA had a real structured-creativity
engine (`nyxara/mind/creative.py` — SCAMPER, lateral thinking, analogy, conceptual
blending) but it was **wired into nothing**: no core method, no faculty registration,
no console command. Meanwhile the ingredients of *measured* originality already existed
elsewhere in the repo — kNN novelty search + MAP-Elites (`growth/frontier.py`),
evolutionary invention (`growth/eureka.py`), hyperdimensional vector memory
(`cognition/hyper_dimensional_vectors.py`) — but nothing composed them into a creative
organism she runs herself.

P24 · F19 fixes the orphan and builds the organism. "Original" is made **operational**,
not rhetorical: a creation is kept only if its measured behaviour is far from
everything she has ever made, if it survives her own adversarial critic, and (for
inventions) if its mechanism provably carries its benefit inside the laws of physics.

## 2. The honest ceiling (no hype)

No algorithm manufactures "creation from zero" — and no such claim is written into
this code. What IS true, and measured: every piece here is produced by **her own
deterministic engines** (no learned weights, no LLM call, no training-data remix), is
**gated on novelty measured against her own archive** (kNN distance in behaviour
space — new niches, not near-duplicates), and must **survive adversarial critique and
reality auditing** before it exists at all. That is creativity in the falsifiable
form: search + measurement + selection pressure toward the genuinely-far, with
provenance stamped `SELF_REFLECTION` and `llm_used: False` on every output.

## 3. The organism — one core loop, fifteen organs

Core (`nyxara/mind/originality.py` — `OriginalityEngine`): a `Genome` per creation
(operator chains for inventions; form/device/meter genes for verse; generator or
cellular-automaton RULE genes for art) is **expressed** into real content, **described**
as a 4-dim behaviour vector (modality, lexical diversity, log-length, bigram entropy),
and **gated**: content-hash unseen ∧ archive-novel (new MAP-Elites niche or novelty ≥
floor) ∧ quality ≥ floor ∧ critic-survived. Elites breed the next round.

| # | Organ | Where | What is real about it |
|---|-------|-------|----------------------|
| 1 | Muse (autonomous motivation) | `mind/muse.py` | She opens her own projects from an ignorance map; each carries a falsifiable hypothesis, later judged corroborated/refuted against measured novelty |
| 2 | Concept graph | `mind/concept_graph.py` | Deterministic hypervector embeddings + learned co-occurrence edges; `distant_pair()` feeds genuine cross-domain blending |
| 3 | Strategy evolver | `originality.py` | Her algorithm's knobs as a clamped `StrategyGenome`; rewrites adopted **only** on a seeded, sandboxed benchmark win |
| 4 | Imagination tree | `originality.py` | Real MCTS (UCB1) over genome space; every rollout is a full express+verify+novelty evaluation |
| 5 | Aesthetic judge | `mind/aesthetic.py` | Computational aesthetics (alliteration, rhythm, symmetry, golden ratio, hue harmony…) with online-EMA learned weights — her taste evolves from `/rate` and from success |
| 6 | Causal sketch | `originality.py` | A DAG per invention; do-interventions prove the benefit dies without its mechanism |
| 7 | Inner critic | `mind/inner_critic.py` | Adversarial objections (clichés, vagueness, contradictions, redundancy, unsupported claims, degenerate geometry); one targeted revision, then rejection |
| 8 | Dream phase | `originality.py` | Near capacity she clusters, extracts recurring motifs into the concept graph, merges near-duplicates, THEN trims — generalization, not FIFO deletion |
| 9 | CA physics sandbox | `originality.py` | Automaton art genomes are RULES that get **simulated**; dead/frozen/saturating rules fail on behaviour, not syntax |
| 10 | Atelier | `mind/atelier.py` | Four algorithmic personas (Logician/Disruptor/Minimalist/Archivist) propose under distinct biases; influence-weighted Borda negotiation picks the master output; influence is earned and bounded |
| 11 | Reality anchor | `originality.py` | Energy ledger over the causal chain (output < input, every stage pays); perpetual-motion-class claims are hard-rejected with the violated law named |
| 12 | Ego narrative | `mind/muse.py` | Recency-weighted self-model: biases, strengths, repeated mistakes; exerts real corrective pressure on the Muse and the Atelier |
| 13 | Dialectic | `mind/inner_critic.py` | Thesis → antonym-mapped antithesis → synthesis; the synthesis is kept only if it *beats* its thesis on novelty×quality |
| 14 | Transmutation | `originality.py` | Structure-preserving maps between modalities (entropy histograms → meter; meter → lissajous params); transmuted pieces pass the SAME gates |
| 15 | Meta-genetic rewriter | `originality.py` | Genetic operators are compositions of a closed combinator vocabulary; on plateau she invents new compositions, trials them with credit assignment, retires them without measured merit |

## 4. The wiring (she does it herself)

- **Faculty registry** — `CreativeFaculty` registered in `build_default_faculties()`
  (`mind/reasoning_faculties.py`): GENERATION scores 0.85·0.75/0.5 = **1.275** vs the
  LLM's 0.9·0.6/3.0 = **0.18**. Creative tasks system-wide route to HER engine.
- **Kernel** — `NyxaraCore.create()` / `.imagine()` (the orphan fix) /
  `.originality_report()` / `.rate_creation()`; state persists at
  `~/.nyxara/originality.json` so taste, archive, self-model and atelier economy
  compound across restarts.
- **Idle** — `idle_maintenance()` block 4f.4: every 8th idle tick (oversight-gated)
  the Muse opens a project and she creates unprompted; kept pieces are recorded as
  INSIGHT thoughts.
- **Autonomic** — `_guaranteed_self_work()` gains a "creation" fallback: when a tick
  would otherwise do nothing, she makes something new instead.
- **Console** — `/create [idea|verse|art] <topic>`, `/imagine <topic> [~ blend]`,
  `/originals`, `/muse`, `/rate <1-10>`.

## 5. The no-LLM guarantee, surfaced

No engine in this feature takes an `llm` argument; the composed `CreativeEngine` is
constructed `llm=None`; the module never imports `nyxara.mind.llm` (test-enforced).
Every `Original` and `Proposal` carries `Provenance(SourceType.SELF_REFLECTION,
method="generated")` and `metadata["llm_used"] = False`; `report()` states it.

## 6. Verification

```
python -m nyxara.mind.originality      # engine self-test (all organs)
python -m nyxara.mind.aesthetic ; python -m nyxara.mind.concept_graph
python -m nyxara.mind.inner_critic ; python -m nyxara.mind.muse
python -m nyxara.mind.atelier
python -m pytest -q tests/mind/test_originality.py tests/mind/test_aesthetic.py \
  tests/mind/test_concept_graph.py tests/mind/test_inner_critic.py \
  tests/mind/test_muse.py tests/mind/test_atelier.py \
  tests/kernel/test_creativity_wiring.py tests/mind/test_creative.py
python -m nyxara                       # /imagine defense ~ chess
                                       # /create verse the night sky
                                       # /create art spiral galaxies
                                       # /originals ; /rate 9 ; /muse
```
