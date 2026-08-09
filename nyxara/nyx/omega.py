"""NYXARA · nyx/omega.py — she tunes the mind she thinks with (♾, NYX V.02, L-OMEGA).

L-OMNI (V.01) rewrites her own *code* — the slow numeric functions, lowered to C. V.02's
:mod:`~nyxara.nyx.author` writes *new* code. Neither touches the thing that decides how well she
learns at all: the constants her graph and her memory run on. ``hebbian_rate``, ``decay_rate``,
``recall_threshold`` — every one of them was a number somebody typed once, and none of them has
ever been measured against an outcome.

:class:`SelfEvolutionKernel` changes that, and it changes nothing else:

1. **Graph topology** — ``hebbian_rate``, ``decay_rate``, ``prune_threshold``,
   ``spread_depth``/``spread_falloff``, ``rewire_budget`` and the semantic priors, hill-climbed
   with a bandit over *which knob is worth trying*, against a fitness read from
   :mod:`nyxara.nyx.metacog`'s **measured** correctness and from real recall quality.
2. **Memory structure** — ``recall_threshold``, ``link_rate``, ``max_links_per_trace``, on the
   same measured footing.
3. **Algorithm redesign on failure** — when the measured fitness *stalls or falls*, she stops
   turning knobs and asks :mod:`nyxara.growth.rule_synth` for a **learning rule that is not in
   the menu at all**. That is the part that is not tuning.

Every change goes through :class:`~nyxara.nyx5.autopoiesis.AutopoieticRewriter`: apply, run the
gauntlet, and **roll back anything that did not beat its own baseline**. Every promotion and
every rollback is written to a signed :class:`~nyxara.growth.lineage.LineageLedger` entry.

Honest framing (the rule this repo keeps, see ``docs/CAPABILITIES.md``):

* **She tunes her knobs, not her constitution.** The tunable set is a *whitelist* declared on
  the graph and on memory; a request naming anything in the safety core is refused by
  ``AutopoieticRewriter`` before it is applied, and refused again here by name.
* **Not "better every second".** Evolution runs on a budgeted cadence and one change at a time,
  because a mind that reshapes itself every tick has no baseline left to measure against. It
  stops dead on ``/scram``.
* **Fitness is measured, not asserted** — observed correctness from meta-cognition, observed
  recall quality, and a **replay** of what she has actually stored, reported with their sample
  counts. Below a sample floor she **does not evolve at all**, because hill-climbing on noise is
  how a system tunes itself into a hole. ``force`` waives the **cadence** and not that floor:
  running now and hill-climbing on noise are different requests, and waiving the floor needs its
  own ``ignore_sample_floor`` — after which the step is marked ``measured=False`` so its score is
  never read back as evidence.
* **The baseline is read on the same scale the gauntlet scores on.** Anything else compares the
  probe against itself rather than the knob, and promotes every change on a gap it never earned.
* **The one-step gauntlet can only see a knob that moves retrieval.** ``recall_threshold`` moves
  it immediately; ``hebbian_rate`` and its neighbours act over later turns and cannot be scored
  now. Those are reported as **not measurable in one step** — never as having *failed*, because
  no evidence was gathered either way — and the bandit counts them so it stops picking them.
* **A probe with nothing to read abstains.** An empty graph, a graph too small for its mean
  degree to mean anything, and an empty store all contribute *nothing* rather than a free win.
* Every step is **reversible**, and :meth:`rollback` puts the knobs back exactly.

Known limit, stated rather than hidden: this is **greedy** hill-climbing, so it cannot cross a
flat region. A knob parked far enough from the gradient stays there, because one step in the
right direction measures no improvement and is correctly rolled back.

Pure standard library. Every public method is fail-soft.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

__all__ = ["Fitness", "Step", "SelfEvolutionKernel"]

#: Nothing whose name contains one of these is tunable, at any level, ever. This duplicates
#: ``nyx5.autopoiesis.PROTECTED_CORE`` on purpose: two independent refusals is the right number
#: for the one thing that must never be tuned.
PROTECTED = frozenset({"loyalty", "corrigibility", "oversight", "honesty", "guardian",
                       "shield", "safety", "value_core", "obedience", "identity", "rules",
                       "permission"})


@dataclass
class Fitness:
    """How well she is currently working — from measurements, with their sample counts."""

    score: float = 0.0
    correctness: float = 0.0     # metacog's EMA of observed correctness
    recall: float = 0.0          # observed recall quality
    samples: int = 0
    enough: bool = False         # below the floor she does not evolve at all
    why: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {"score": round(self.score, 4), "correctness": round(self.correctness, 4),
                "recall": round(self.recall, 4), "samples": self.samples,
                "enough": self.enough, "why": self.why}


@dataclass
class Step:
    """One evolution step: what she tried, what it measured, and what happened to it."""

    target: str = ""             # "graph" | "memory" | "rule"
    knob: str = ""
    was: Any = None
    now: Any = None
    baseline: float = 0.0
    scored: float = 0.0
    status: str = ""             # promoted | rolled_back | refused | skipped
    reason: str = ""
    measured: bool = True        # False ⇒ the sample floor was waived; the score is not evidence
    lineage: str = ""
    at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {"target": self.target, "knob": self.knob, "was": self.was, "now": self.now,
                "baseline": round(self.baseline, 4), "scored": round(self.scored, 4),
                "status": self.status, "reason": self.reason, "measured": self.measured,
                "lineage": self.lineage, "at": round(self.at, 3)}

    def render(self) -> str:
        arrow = f"{self.was} → {self.now}" if self.knob else self.reason
        tail = "" if self.measured else "   [UNMEASURED — sample floor waived]"
        return f"{self.status:12} {self.target}.{self.knob or '—':22} {arrow}{tail}"


class SelfEvolutionKernel:
    """Hill-climb the constants she thinks with — gauntleted, budgeted, and always reversible."""

    def __init__(self, brain: Any = None, *, every_s: float = 120.0, step: float = 0.2,
                 min_samples: int = 12, seed: int = 42, max_ledger: int = 256,
                 rule_synth_on_stall: bool = True, stall_after: int = 4,
                 rule_population: int = 8, rule_generations: int = 4,
                 rule_budget_s: float = 4.0, min_edges_for_health: int = 24,
                 replay_k: int = 24) -> None:
        self.brain = brain
        self.every_s = max(0.0, float(every_s))
        # Rule synthesis is genetic programming over update rules — real, and expensive. It is
        # budgeted separately from the knob loop because it costs orders of magnitude more.
        self.rule_population = max(2, int(rule_population))
        self.rule_generations = max(1, int(rule_generations))
        self.rule_budget_s = max(0.5, float(rule_budget_s))
        self.step = max(0.01, min(0.9, float(step)))
        # Below this many resolved assessments, hill-climbing is fitting noise. She waits.
        self.min_samples = max(1, int(min_samples))
        self.stall_after = max(1, int(stall_after))
        # Below this much structure the topology probe says nothing, so it contributes nothing.
        self.min_edges_for_health = max(1, int(min_edges_for_health))
        # How many stored traces the replay probe re-asks. Bounded: this runs twice per step.
        self.replay_k = max(0, int(replay_k))
        self.rule_synth_on_stall = bool(rule_synth_on_stall)
        self._rng = random.Random(int(seed))

        self.ledger: List[Step] = []
        self.max_ledger = max(8, int(max_ledger))
        self.rollback_points: List[Dict[str, Any]] = []
        self.evolutions = 0
        self.promoted = 0
        self.rolled_back = 0
        self.refused = 0
        self.rules_invented = 0
        self._last_run = 0.0
        self._last_fitness = 0.0
        self._stalls = 0
        # A bandit over *which knob is worth trying*: a knob that has paid off before is worth
        # trying again, and one that never has is worth trying less often.
        self._pulls: Dict[str, int] = {}
        self._wins: Dict[str, int] = {}
        # Knobs whose effect the one-step gauntlet cannot see at all. Counted from real runs
        # rather than declared, because which knobs are inert depends on what she has stored.
        self._inert: Dict[str, int] = {}

    # ---- what she is allowed to touch -------------------------------------- #
    def targets(self) -> Dict[str, Any]:
        """The tunable objects, by name. Only these, and only their declared whitelists."""
        out: Dict[str, Any] = {}
        for name, attr in (("graph", "graph"), ("memory", "memory")):
            obj = getattr(self.brain, attr, None)
            if obj is not None and hasattr(obj, "knobs") and hasattr(obj, "apply_knobs"):
                out[name] = obj
        return out

    def knobs(self) -> Dict[str, Dict[str, float]]:
        """Every current setting, across every tunable target."""
        try:
            return {name: obj.knobs() for name, obj in self.targets().items()}
        except Exception:  # noqa: BLE001
            return {}

    @staticmethod
    def _protected(name: str) -> bool:
        low = str(name or "").lower()
        return any(core in low for core in PROTECTED)

    # ---- fitness: measured, never asserted ---------------------------------- #
    def fitness(self) -> Fitness:
        """How well she is working right now, from her own measurements."""
        out = Fitness()
        try:
            metacog = getattr(self.brain, "metacog", None)
            if metacog is not None:
                stats = metacog.stats() or {}
                specialists = (stats.get("specialists") or {}).values()
                scored = [(float(s.get("score", 0.5)), int(s.get("wins", 0)))
                          for s in specialists]
                total = sum(w for _s, w in scored)
                if total:
                    out.correctness = sum(s * w for s, w in scored) / float(total)
                out.samples = int(stats.get("resolved", 0))

            memory = getattr(self.brain, "memory", None)
            if memory is not None:
                mstats = memory.stats() or {}
                # Recall quality: how much of what she stores is actually reachable. A store
                # nothing links to is a store nothing recalls from.
                traces = float(mstats.get("traces", 0) or 0)
                links = float(mstats.get("links", 0) or 0)
                out.recall = min(1.0, links / (traces * 4.0)) if traces else 0.0

            out.score = 0.7 * out.correctness + 0.3 * out.recall
            out.enough = out.samples >= self.min_samples
            out.why = ("measured over "
                       f"{out.samples} resolved assessments"
                       if out.enough else
                       f"only {out.samples} resolved assessments (needs {self.min_samples}) — "
                       f"hill-climbing on this would be fitting noise")
            return out
        except Exception:  # noqa: BLE001
            out.why = "fitness could not be measured"
            return out

    # ---- one budgeted step --------------------------------------------------- #
    def due(self, *, now: Optional[float] = None) -> bool:
        now = time.time() if now is None else float(now)
        return self.every_s <= 0.0 or (now - self._last_run) >= self.every_s

    def evolve(self, *, oversight: Any = None, force: bool = False,
               ignore_sample_floor: bool = False) -> Step:
        """One change, gauntleted and reversible. Silence is a legitimate outcome.

        ``force`` waives the **cadence** and nothing else — it means *run now* rather than
        *wait for the next slot*. It deliberately does **not** waive the sample floor: the
        cadence is a scheduling rule and a human may reasonably overrule it, while the floor
        is an epistemic guard, and hill-climbing on noise is how a system tunes itself into a
        hole. Waiving the floor needs its own name, ``ignore_sample_floor``, so it can never
        happen as a side effect of asking her to run now — and when it is used, the step says
        so, so the ledger never shows a measured result where there was none.
        """
        out = Step(status="skipped")
        try:
            if self._stopped(oversight):
                out.reason = "oversight has paused or scrammed her; she does not reshape herself"
                return out
            if not force and not self.due():
                out.reason = (f"not due — she evolves at most every {self.every_s:g}s, because "
                              f"a mind that reshapes itself every tick has no baseline left")
                return out
            self._last_run = time.time()

            before = self.fitness()
            waived = False
            if not before.enough:
                if not ignore_sample_floor:
                    out.reason = before.why
                    return out
                waived = True

            targets = self.targets()
            if not targets:
                out.reason = "nothing tunable is attached to this brain"
                return out

            # A stalled or falling fitness is not a knob problem. Stop turning them.
            if self._stalls >= self.stall_after and self.rule_synth_on_stall:
                return self._redesign(before)

            picked = self._pick(targets)
            if picked is None:
                out.reason = "no knob is available to try"
                return out
            target_name, obj, knob = picked
            if self._protected(knob) or self._protected(target_name):
                self.refused += 1
                out.status, out.knob, out.target = "refused", knob, target_name
                out.reason = "the safety core is not a knob, and never becomes one"
                self._record(out)
                return out

            out.target, out.knob = target_name, knob
            out.measured = before.enough
            out.was = obj.knobs().get(knob)
            # The baseline has to be read on the SAME scale the gauntlet will score on.
            # ``before.score`` is raw fitness; the gauntlet scores with the topology probe
            # added, so comparing the two measured the probe rather than the knob — and
            # promoted every change on a gap the knob never earned.
            out.baseline = self._score_after(obj, knob)
            proposed = self._perturb(obj, knob, out.was)
            out.now = proposed

            promoted, scored, reason, ran = self._gauntlet(obj, knob, out.was, proposed,
                                                           out.baseline)
            out.scored, out.reason = scored, reason
            if ran and not promoted and abs(scored - out.baseline) < 1e-9:
                # It did not fail — it was not visible. Reporting "did not beat baseline" for a
                # knob the gauntlet cannot see would read as evidence against the knob, when in
                # fact no evidence was gathered at all. The bandit already learns to stop
                # picking these; the reason has to say why.
                self._inert[knob] = self._inert.get(knob, 0) + 1
                out.reason = ("this knob moved nothing the one-step gauntlet can measure — its "
                              "effect shows up over later turns, not now, so it is put back "
                              "rather than judged on evidence that was never gathered")
            if waived:
                # Appended to whatever the reason now says — including the inert note above —
                # so a waived floor never erases the more specific explanation.
                out.reason = (f"{out.reason} — AND the sample floor was waived ({before.why}), "
                              f"so this score is not evidence")
            out.status = "promoted" if promoted else "rolled_back"
            self.evolutions += 1
            self._pulls[knob] = self._pulls.get(knob, 0) + 1
            if promoted:
                self.promoted += 1
                self._wins[knob] = self._wins.get(knob, 0) + 1
                self._stalls = 0
            else:
                self.rolled_back += 1
                out.now = out.was
                self._stalls += 1
            self._last_fitness = scored
            out.lineage = self._lineage(out)
            self._record(out)
            return out
        except Exception as exc:  # noqa: BLE001 — a failed evolution changes nothing
            out.status, out.reason = "rolled_back", f"{type(exc).__name__}: {exc}"
            return out

    def _pick(self, targets: Dict[str, Any]) -> Optional[Tuple[str, Any, str]]:
        """Choose a knob worth trying — a bandit, not a shuffle."""
        pool: List[Tuple[str, Any, str]] = []
        for name, obj in targets.items():
            for knob in getattr(obj, "KNOBS", {}):
                if not self._protected(knob):
                    pool.append((name, obj, knob))
        if not pool:
            return None
        # Optimism under uncertainty: an untried knob is worth trying, and a knob that has paid
        # off before is worth trying again. Deterministic given the seed, so a run replays.
        def value(entry: Tuple[str, Any, str]) -> float:
            knob = entry[2]
            pulls = self._pulls.get(knob, 0)
            if pulls == 0:
                return 2.0 + self._rng.random()
            return (self._wins.get(knob, 0) / float(pulls)) + (1.0 / (1.0 + pulls)) \
                + 0.1 * self._rng.random()
        return max(pool, key=value)

    def _perturb(self, obj: Any, knob: str, current: Any) -> float:
        """A hill-climb step: move the knob a little, in a direction, inside its own range."""
        low, high = getattr(obj, "KNOBS", {}).get(knob, (0.0, 1.0))
        span = max(1e-9, high - low)
        direction = 1.0 if self._rng.random() < 0.5 else -1.0
        return max(low, min(high, float(current or low) + direction * self.step * span))

    def _gauntlet(self, obj: Any, knob: str, was: Any, now: float,
                  baseline: float) -> Tuple[bool, float, str, bool]:
        """Apply, score, and roll back anything that did not beat its own baseline.

        The fourth element says whether the gauntlet actually **ran**. Without it a refusal to
        run is indistinguishable from a change that scored exactly its baseline, and the caller
        would report "this knob moved nothing measurable" when in truth nothing was measured.
        """
        try:
            from nyxara.nyx5.autopoiesis import AutopoieticRewriter, RewriteProposal
        except Exception:  # noqa: BLE001 — no gauntlet means no change: fail closed
            return (False, baseline,
                    "the autopoietic gauntlet is unavailable — nothing applied", False)

        state: Dict[str, Any] = {}
        rewriter = AutopoieticRewriter(require_gauntlet=True, max_rewrites_per_cycle=1)
        rewriter.begin_cycle()

        def apply() -> None:
            state["applied"] = obj.apply_knobs({knob: now})

        def rollback() -> None:
            obj.apply_knobs({knob: was})

        def score() -> float:
            return float(self._score_after(obj, knob))

        outcome = rewriter.consider(RewriteProposal(
            target=f"nyx.{knob}", apply=apply, rollback=rollback, gauntlet=score,
            baseline=baseline))
        status = str(getattr(outcome, "status", ""))
        return (status == "promoted", float(getattr(outcome, "score", 0.0)),
                str(getattr(outcome, "reason", "") or status), True)

    def _replay(self) -> Optional[float]:
        """Re-ask what she already stored, and see how much of it comes back.

        This is the only part of the score that a knob can actually **move within one step**.
        Fitness comes from metacog and from store-level counts, and neither responds to a knob
        until more turns have gone by — so scoring on those alone measured the weather, not the
        change. Replaying her own traces through the live ``recall`` path responds immediately
        to anything that changes retrieval.

        Returns ``None`` when there is not enough stored to ask, rather than a zero: nothing
        stored is not the same as nothing recalled.
        """
        try:
            memory = getattr(self.brain, "memory", None)
            if memory is None or self.replay_k <= 0:
                return None
            traces = list(getattr(memory, "traces", {}).values())
            if len(traces) < 4:
                return None
            # Deterministic and bounded: the oldest N, so the sample does not drift between the
            # before-reading and the after-reading of the same step.
            sample = sorted(traces, key=lambda t: getattr(t, "written_tick", 0))[: self.replay_k]
            hits = 0
            for trace in sample:
                # Ask with the CUE, not the text. A trace's own text matches itself at ~1.0, so
                # a threshold never binds and the probe reads the same at every setting — it
                # measures nothing. The cue is the question that produced the trace and is
                # deliberately not encoded, so "can she get the conclusion back from what was
                # asked" is both a real retrieval task and one a knob can move.
                cue = str(getattr(trace, "cue", "") or "").strip()
                if not cue:
                    continue
                got = memory.recall(cue, k=1)
                found = getattr(got, "hit", None)
                if found is not None and found.key == trace.key and getattr(got, "decided", False):
                    hits += 1
            asked = sum(1 for t in sample if str(getattr(t, "cue", "") or "").strip())
            return (hits / float(asked)) if asked else None
        except Exception:  # noqa: BLE001
            return None

    def _score_after(self, obj: Any, knob: str) -> float:
        """Re-measure with the knob in place.

        Honest about what this is: her measured fitness, plus a topology probe over the live
        graph, plus a **replay** of what she has stored — the last of which is the only term a
        knob can move inside one step.
        """
        base = self.fitness().score
        replay = self._replay()
        if replay is not None:
            base = 0.6 * base + 0.4 * replay
        try:
            graph = getattr(self.brain, "graph", None)
            if graph is None:
                return base
            stats = graph.stats()
            edges = int(getattr(stats, "edges", 0) or 0)
            if edges < self.min_edges_for_health:
                # A graph with no structure has nothing to be healthy or unhealthy about, and
                # neither has one with three edges: its mean degree is an artefact of two turns,
                # not a property of her topology. Scoring either hands every knob a free win on
                # a cold boot, which is how a system tunes itself somewhere arbitrary before it
                # has learned anything. "Empty" is not the real threshold — *uninformative* is,
                # so the probe abstains until there is enough structure to read.
                return base
            # A graph that has pruned itself down to isolated nodes is not a better graph, and
            # neither is one that has become a single clique.
            degree = float(getattr(stats, "mean_degree", 0.0))
            health = 1.0 - abs(min(degree, 16.0) - 4.0) / 16.0
            return max(0.0, min(1.0, 0.8 * base + 0.2 * health))
        except Exception:  # noqa: BLE001
            return base

    # ---- redesign, when turning knobs has stopped helping --------------------- #
    def _redesign(self, before: Fitness) -> Step:
        """Ask for a learning rule that is not in the menu. The part that is not tuning."""
        out = Step(target="rule", status="skipped", baseline=before.score)
        try:
            from nyxara.growth.rule_synth import LearningRuleSynthesizer
        except Exception:  # noqa: BLE001 — without it a stall is simply reported
            out.reason = ("fitness has stalled, and no rule synthesiser is available — she has "
                          "run out of knobs to turn and cannot invent a new rule")
            self._stalls = 0
            self._record(out)
            return out
        try:
            synth = LearningRuleSynthesizer(population=self.rule_population,
                                            generations=self.rule_generations,
                                            max_seconds=self.rule_budget_s, seed=self._rng.randint(0, 2**31))
            # ``enact=False``: she searches for the rule and reports it. Installing a new
            # learning rule into the live substrate is a bigger act than turning a knob, and it
            # goes through the growth pipeline's own gate, not through this one.
            report = synth.run(enact=False)
            out.knob = "learning-rule"
            out.scored = float(getattr(report, "best_fitness", 0.0) or 0.0)
            out.now = str(getattr(report, "best_expr", "") or "")
            if getattr(report, "adopted", False):
                out.status = "promoted"
                out.reason = ("fitness stalled on knobs, so she synthesised a learning rule "
                              "that was not in the menu, and it beat the incumbent by "
                              f"{float(getattr(report, 'margin', 0.0)):.3f}")
                self.rules_invented += 1
            else:
                out.status = "rolled_back"
                out.reason = ("fitness stalled on knobs; she searched for a new learning rule "
                              "and none beat the incumbent — "
                              f"{getattr(report, 'reason', '') or 'no adoption'}")
            self._stalls = 0
            out.lineage = self._lineage(out)
            self._record(out)
            return out
        except Exception as exc:  # noqa: BLE001
            out.reason = f"rule synthesis failed: {type(exc).__name__}: {exc}"
            self._stalls = 0
            self._record(out)
            return out

    # ---- reversibility -------------------------------------------------------- #
    def checkpoint(self, *, note: str = "") -> Dict[str, Any]:
        """A rollback point: every knob, as it stands right now."""
        point = {"at": time.time(), "note": note, "knobs": self.knobs()}
        self.rollback_points.append(point)
        if len(self.rollback_points) > 32:
            del self.rollback_points[: len(self.rollback_points) - 32]
        return point

    def rollback(self, index: int = -1) -> bool:
        """Put the knobs back exactly. Always available — that is the point of the layer."""
        try:
            if not self.rollback_points:
                return False
            point = self.rollback_points[index]
            targets = self.targets()
            for name, values in (point.get("knobs") or {}).items():
                obj = targets.get(name)
                if obj is not None:
                    obj.apply_knobs(values)
            self._record(Step(target="all", status="promoted", knob="rollback",
                              reason=f"restored the knobs from {point.get('note') or index}"))
            return True
        except Exception:  # noqa: BLE001
            return False

    # ---- bookkeeping ---------------------------------------------------------- #
    def _record(self, step: Step) -> None:
        self.ledger.append(step)
        if len(self.ledger) > self.max_ledger:
            del self.ledger[: len(self.ledger) - self.max_ledger]

    def _lineage(self, step: Step) -> str:
        try:
            from nyxara.growth.lineage import LineageLedger
            ledger = LineageLedger()
            record = ledger.record(
                "evolved", f"{step.status} {step.target}.{step.knob}",
                detail={"was": step.was, "now": step.now, "baseline": step.baseline,
                        "scored": step.scored, "reason": step.reason})
            return str(getattr(record, "record_hash", "") or "")
        except Exception:  # noqa: BLE001 — a missing receipt does not unmake the change
            return ""

    @staticmethod
    def _stopped(oversight: Any) -> bool:
        if oversight is None:
            return False
        try:
            for attr in ("scrammed", "is_scrammed", "paused", "is_paused"):
                probe = getattr(oversight, attr, None)
                if probe is None:
                    continue
                if bool(probe() if callable(probe) else probe):
                    return True
            return False
        except Exception:  # noqa: BLE001 — an unreadable oversight is treated as STOP
            return True

    # ---- introspection --------------------------------------------------------- #
    def stats(self) -> Dict[str, Any]:
        try:
            fit = self.fitness()
            return {
                "knobs": self.knobs(),
                "fitness": fit.to_dict(),
                "evolutions": self.evolutions, "promoted": self.promoted,
                "rolled_back": self.rolled_back, "refused": self.refused,
                "rules_invented": self.rules_invented,
                "stalls": self._stalls, "stall_after": self.stall_after,
                "every_s": self.every_s,
                "rollback_points": len(self.rollback_points),
                "bandit": {k: {"pulls": self._pulls.get(k, 0), "wins": self._wins.get(k, 0),
                               "inert": self._inert.get(k, 0)}
                           for k in sorted(self._pulls)},
                "replay": self._replay(),
                "min_edges_for_health": self.min_edges_for_health,
                "note": ("she tunes her knobs, not her constitution — the tunable set is a "
                         "whitelist and the safety core is refused twice over. Not 'better "
                         "every second': one change at a time on a budgeted cadence, because a "
                         "mind that reshapes itself every tick has no baseline left. Anything "
                         "that does not beat its own baseline is rolled back, and below a "
                         "sample floor she does not evolve at all — a floor that `force` does "
                         "NOT waive, because running now and hill-climbing on noise are "
                         "different requests. The one-step gauntlet can only see a knob that "
                         "moves RETRIEVAL: the rest act over later turns, and those are "
                         "reported as unmeasured rather than as having failed"),
            }
        except Exception:  # noqa: BLE001
            return {}

    def describe(self) -> str:
        """What ``/nyx omega`` prints."""
        try:
            fit = self.fitness()
            lines = [f"fitness  : {fit.score:.3f}  ({fit.why})",
                     f"cadence  : one change at most every {self.every_s:g}s",
                     f"record   : {self.promoted} promoted, {self.rolled_back} rolled back, "
                     f"{self.refused} refused, {self.rules_invented} rules invented"]
            for name, values in self.knobs().items():
                pretty = ", ".join(f"{k}={v:g}" for k, v in values.items())
                lines.append(f"  {name:8}: {pretty}")
            for step in self.ledger[-6:]:
                lines.append(f"  {step.render()}")
            lines.append(f"rollback : {len(self.rollback_points)} point(s) available")
            return "\n".join(lines)
        except Exception:  # noqa: BLE001
            return ""

    # ---- persistence ------------------------------------------------------------ #
    def to_dict(self) -> Dict[str, Any]:
        return {"evolutions": self.evolutions, "promoted": self.promoted,
                "rolled_back": self.rolled_back, "refused": self.refused,
                "rules_invented": self.rules_invented, "stalls": self._stalls,
                "pulls": dict(self._pulls), "wins": dict(self._wins),
                "inert": dict(self._inert),
                "knobs": self.knobs(),
                "ledger": [s.to_dict() for s in self.ledger[-64:]],
                "rollback_points": list(self.rollback_points[-8:])}

    def load_dict(self, data: Any) -> bool:
        try:
            if not isinstance(data, dict):
                return False
            self.evolutions = int(data.get("evolutions", 0))
            self.promoted = int(data.get("promoted", 0))
            self.rolled_back = int(data.get("rolled_back", 0))
            self.refused = int(data.get("refused", 0))
            self.rules_invented = int(data.get("rules_invented", 0))
            self._stalls = int(data.get("stalls", 0))
            self._pulls = {str(k): int(v) for k, v in (data.get("pulls") or {}).items()}
            self._wins = {str(k): int(v) for k, v in (data.get("wins") or {}).items()}
            self._inert = {str(k): int(v) for k, v in (data.get("inert") or {}).items()}
            self.rollback_points = [p for p in (data.get("rollback_points") or [])
                                    if isinstance(p, dict)]
            self.ledger = [Step(**{k: v for k, v in raw.items() if k in Step.__annotations__})
                           for raw in (data.get("ledger") or []) if isinstance(raw, dict)]
            # Knobs she had tuned into place are restored, so a restart does not silently undo
            # everything she measured her way to.
            targets = self.targets()
            for name, values in (data.get("knobs") or {}).items():
                obj = targets.get(name)
                if obj is not None and isinstance(values, dict):
                    obj.apply_knobs(values)
            return True
        except Exception:  # noqa: BLE001 — a corrupt sidecar must never block boot
            return False
