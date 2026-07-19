"""NYXARA · mind/metacognitive_allocator.py — first-class metacognition: uncertainty → compute (🪞⚖, Problem #1).

**The problem this closes.** Metacognition used to be an *add-on*: a gate that chose who answers
(:mod:`mind.metacognition`) while the deep-reasoning ladder (:mod:`mind.deep_reasoning`) climbed
**always-max** — every turn got the whole ladder, an easy greeting the same as a hard proof. That
wastes the budget where it buys nothing and starves the turns where it buys everything.

This module makes metacognition **first-class**: *calibrated uncertainty drives compute
allocation*. An easy problem gets **one forward pass**; a hard one gets the **full ladder, up to a
ten-minute search**. And the decision is NYXARA's own — measured from her lived outcomes, never by
asking the LLM how hard to try. Four cooperating pieces:

1. **Predictive difficulty** (:meth:`MetacognitiveAllocator.assess`) — before any compute is spent,
   a calibrated estimate of how hard *this* turn will be, from her own signals only: a
   **hierarchical Beta belief** over problem signatures (exact signature → shape bucket → global,
   evidence-weighted back-off, recency-decayed), novelty, and stakes; recalibrated through a
   :class:`~nyxara.mind.uncertainty.Calibrator` fed by every lived outcome, so predicted hardness
   converges to *empirical* hardness (the ECE is visible in :meth:`report`).

2. **Allocation** (:meth:`MetacognitiveAllocator.allocate`) — the uncertainty maps to a concrete
   :class:`TurnBudget`: low → 1 rung · 1 sample · seconds (one pass); high → the full ceiling with
   up to ``hard_max_seconds`` (default 600 s) of search. The compute-scaled ceiling
   (:class:`~nyxara.growth.effective_scale.ReasoningBudget`) stays the *ceiling*; metacognition
   decides how much of it to *spend*. High-stakes signatures never get the one-pass shortcut.

3. **Value-of-computation stopping** (:meth:`MetacognitiveAllocator.should_continue`) — genuine
   metareasoning during the climb (Russell–Wefald style): continue to the next rung only while its
   *expected* verified-score gain (learned per shape × rung from lived score deltas) justifies its
   *expected* wall-clock cost, or the target is unmet. The moment the target is cleared or the
   marginal value collapses, the climb stops — this is what makes "easy = 1 pass" real. And it is
   **anytime in both directions**: a turn predicted easy whose first measured score comes back weak
   is *escalated* mid-climb toward the full ladder — prediction is the prior, measurement corrects it.

4. **The closed loop** (:meth:`MetacognitiveAllocator.record`) — after every turn the allocator
   learns: did the cheap pass suffice? which rungs actually improved the answer? how long did each
   cost? All of it updates the beliefs, the calibrator, and the per-rung value estimates, and
   persists (atomic JSON under ``paths.data_dir``), so allocation at turn 100 is measurably sharper
   than at turn 1 — real compounding, no model weights involved.

Everything here is auditable (:meth:`report`) and honest: unknown signatures default to *thinking
hard* (epistemic humility spends compute, it never bluffs cheapness), and every budget carries a
human-readable ``reason``. Pure standard library plus NYXARA's own uncertainty machinery.
"""

from __future__ import annotations

import json
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple

from nyxara.mind.effort_memory import signature
from nyxara.mind.metacognition import MetaCognition
from nyxara.mind.uncertainty import BetaBelief, Calibrator

__all__ = [
    "TurnBudget",
    "Assessment",
    "StopDecision",
    "MetacognitiveAllocator",
    "get_allocator",
]

_GLOBAL_KEY = "global"
# minimum calibration evidence before the empirical curve overrides the raw estimate
_MIN_CALIBRATION_SAMPLES = 10
# score improvement a rung must deliver to count as "it paid off" (VOC evidence)
_MIN_GAIN = 0.05
# expected-gain floor below which the next rung is judged not worth its cost
_VOC_GAIN_FLOOR = 0.04
# evidence needed before a VOC "this rung never helps" verdict may stop a climb
_VOC_MIN_OBS = 3.0
# a measured first-pass score this far below target on a predicted-easy turn → escalate
_ESCALATE_GAP = 0.25
# prior wall-clock estimates per rung (seconds), refined by lived EWMA measurements
_DEFAULT_RUNG_SECONDS = {1: 2.0, 2: 6.0, 3: 15.0, 4: 10.0}
_SECONDS_EWMA = 0.3
_ONE_PASS_SECONDS = 5.0     # runaway guard for a single forward pass, not a quality cap
_LOG_KEEP = 50


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def _shape_key(sig: str) -> str:
    return "shape:" + (sig.split("/", 1)[0] if sig else "unknown")


def _stakes_high(sig: str) -> bool:
    return sig.endswith("/high")


# --------------------------------------------------------------------------- #
# Data carriers
# --------------------------------------------------------------------------- #
@dataclass
class Assessment:
    """A calibrated pre-compute difficulty read for one stimulus."""

    uncertainty: float          # calibrated, [0,1] — drives the budget
    raw: float                  # pre-calibration blend, kept for audit
    signature: str
    evidence: float             # observations backing the exact-signature belief
    stakes_high: bool
    reason: str

    def to_dict(self) -> Dict[str, Any]:
        return {"uncertainty": round(self.uncertainty, 4), "raw": round(self.raw, 4),
                "signature": self.signature, "evidence": round(self.evidence, 2),
                "stakes_high": self.stakes_high, "reason": self.reason}


@dataclass
class TurnBudget:
    """A concrete per-turn compute budget, decided by calibrated uncertainty."""

    max_rung: int
    samples: int
    max_seconds: float
    target_score: float          # early-stop bar: the climb halts once the best score clears this
    predicted_uncertainty: float
    signature: str = ""
    escalated: bool = False      # True once measurement overrode an easy prediction mid-climb
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {"max_rung": self.max_rung, "samples": self.samples,
                "max_seconds": round(self.max_seconds, 2),
                "target_score": round(self.target_score, 3),
                "predicted_uncertainty": round(self.predicted_uncertainty, 4),
                "signature": self.signature, "escalated": self.escalated,
                "reason": self.reason}


@dataclass
class StopDecision:
    """The metareasoner's verdict between rungs: keep climbing, stop, or escalate."""

    stop: bool
    escalate: bool
    reason: str

    def to_dict(self) -> Dict[str, Any]:
        return {"stop": self.stop, "escalate": self.escalate, "reason": self.reason}


# --------------------------------------------------------------------------- #
# The allocator
# --------------------------------------------------------------------------- #
class MetacognitiveAllocator:
    """Calibrated uncertainty → compute. Predict, allocate, stop optimally, learn, repeat."""

    def __init__(self, *, easy_uncertainty: float = 0.25, hard_uncertainty: float = 0.70,
                 base_target: float = 0.80, hard_max_seconds: float = 600.0,
                 belief_decay: float = 0.99, voc_stopping: bool = True,
                 live_escalation: bool = True, path: Optional[Path] = None) -> None:
        self.easy_uncertainty = _clamp(float(easy_uncertainty))
        self.hard_uncertainty = _clamp(max(float(hard_uncertainty),
                                           self.easy_uncertainty + 1e-6))
        self.base_target = _clamp(float(base_target))
        self.hard_max_seconds = max(1.0, min(600.0, float(hard_max_seconds)))
        self.belief_decay = max(0.5, min(1.0, float(belief_decay)))
        self.voc_stopping = bool(voc_stopping)
        self.live_escalation = bool(live_escalation)

        self._lock = threading.RLock()
        # P(this kind of problem is HARD — one cheap pass did not suffice), per hierarchy level.
        self._difficulty: Dict[str, BetaBelief] = {}
        # P(running rung r improved the best verified score by >= _MIN_GAIN), per shape.
        self._rung_improves: Dict[str, Dict[int, BetaBelief]] = {}
        # lived wall-clock per rung (EWMA seconds), refining the priors.
        self._rung_seconds: Dict[int, float] = dict(_DEFAULT_RUNG_SECONDS)
        # predicted-hardness vs actual-hardness — the calibration that makes this honest.
        self.calibrator = Calibrator()
        # stated answer-confidence vs verified correctness — a SECOND, separate calibration
        # (different variable!) shared with the who-answers gates (mind/router.py,
        # mind/self_model_router.py) so their MetaCognition discounts a bluffing model.
        self.answer_calibrator = Calibrator()
        self._decisions: Deque[Dict[str, Any]] = deque(maxlen=_LOG_KEEP)
        self._turns = 0

        self.path = Path(path) if path is not None else None
        if self.path is not None and self.path.exists():
            try:
                self.load(self.path)
            except Exception:  # noqa: BLE001 — a corrupt file must never break reasoning
                pass

    # ------------------------------------------------------------------ #
    # 1. predictive difficulty (before any compute)
    # ------------------------------------------------------------------ #
    def _belief(self, key: str) -> BetaBelief:
        b = self._difficulty.get(key)
        if b is None:
            b = BetaBelief()
            self._difficulty[key] = b
        return b

    def _blended_hardness(self, sig: str) -> Tuple[float, float, float]:
        """Evidence-weighted back-off: exact signature → shape → global → uniform prior.

        Returns ``(hardness, epistemic, exact_evidence)``. Each level contributes in proportion
        to how much evidence it actually holds (shrinkage ``w = n/(n+k)``), so a brand-new
        signature borrows strength from its shape instead of guessing blindly, and a well-worn
        signature speaks for itself."""
        k = 4.0
        estimate = 0.5           # the uniform prior at the back of the chain
        epistemic = 1.0
        for key in (_GLOBAL_KEY, _shape_key(sig), sig):   # innermost applied last → dominates
            b = self._difficulty.get(key)
            if b is None:
                continue
            w = b.observations / (b.observations + k)
            estimate = w * b.mean + (1.0 - w) * estimate
            epistemic = min(epistemic, b.epistemic)
        exact = self._difficulty.get(sig)
        return estimate, epistemic, (exact.observations if exact is not None else 0.0)

    def assess(self, stimulus: str, *, mems_count: int = 0) -> Assessment:
        """A calibrated, pre-compute read of how hard this turn will be — her signals only."""
        sig = signature(stimulus)
        with self._lock:
            hardness, epistemic, evidence = self._blended_hardness(sig)
            novelty = 1.0 / (1.0 + 0.5 * max(0, int(mems_count)))
            # introspection (the canonical blend) lifts unknowns toward "think hard": epistemic
            # humility spends compute — an unseen kind of problem is NEVER assumed cheap.
            unknown = MetaCognition.introspect(epistemic=epistemic, novelty=novelty)
            # the "I haven't seen this kind before" lift is gated by epistemic mass, so it fades
            # as lived evidence accumulates and known-easy kinds actually reach the 1-pass band.
            raw = _clamp(hardness + (1.0 - hardness) * epistemic * unknown)
            calibrated = raw
            if self.calibrator.samples >= _MIN_CALIBRATION_SAMPLES:
                calibrated = _clamp(self.calibrator.recalibrate(raw))
            stakes = _stakes_high(sig)
            if stakes:   # high stakes never get the one-pass shortcut
                calibrated = max(calibrated, self.hard_uncertainty)
            reason = (f"hardness {hardness:.2f} (evidence {evidence:.0f}), "
                      f"epistemic {epistemic:.2f}, novelty {novelty:.2f}"
                      + (", calibrated" if calibrated != raw and not stakes else "")
                      + (", stakes floor" if stakes else ""))
            return Assessment(uncertainty=calibrated, raw=raw, signature=sig,
                              evidence=evidence, stakes_high=stakes, reason=reason)

    # ------------------------------------------------------------------ #
    # 2. allocation (uncertainty → budget)
    # ------------------------------------------------------------------ #
    def allocate(self, stimulus: str, *, ceiling: Any = None,
                 mems_count: int = 0) -> TurnBudget:
        """Map calibrated uncertainty to a concrete budget, bounded above by ``ceiling``.

        ``ceiling`` is anything exposing ``max_rung`` / ``samples`` / ``max_seconds`` (the
        compute-scaled :class:`ReasoningBudget`, or the raw config). Machine capacity stays the
        *ceiling*; metacognition decides the *spend* — except wall-clock on a genuinely hard
        problem, which may rise to ``hard_max_seconds`` (the ten-minute search) even on a modest
        box, because time is the one resource a small machine can always trade for depth."""
        a = self.assess(stimulus, mems_count=mems_count)
        ceil_rung = int(getattr(ceiling, "max_rung", 4) or 4)
        ceil_samples = max(1, int(getattr(ceiling, "samples", 5) or 5))
        ceil_seconds = float(getattr(ceiling, "max_seconds", 60.0) or 60.0)
        hard_seconds = max(ceil_seconds, self.hard_max_seconds)
        u = a.uncertainty

        if u <= self.easy_uncertainty:
            budget = TurnBudget(
                max_rung=1, samples=1,
                max_seconds=min(_ONE_PASS_SECONDS, ceil_seconds),
                target_score=self.base_target, predicted_uncertainty=u,
                signature=a.signature,
                reason=f"metacognition: calibrated uncertainty {u:.2f} ({a.reason}) → 1 pass")
        elif u >= self.hard_uncertainty:
            budget = TurnBudget(
                max_rung=ceil_rung, samples=ceil_samples, max_seconds=hard_seconds,
                target_score=_clamp(self.base_target + (0.1 if a.stakes_high else 0.05),
                                    hi=0.95),
                predicted_uncertainty=u, signature=a.signature,
                reason=(f"metacognition: calibrated uncertainty {u:.2f} ({a.reason}) → "
                        f"full ladder, up to {hard_seconds:.0f}s search"))
        else:
            t = (u - self.easy_uncertainty) / (self.hard_uncertainty - self.easy_uncertainty)
            budget = TurnBudget(
                max_rung=max(1, min(ceil_rung, round(1 + t * (ceil_rung - 1)))),
                samples=max(1, min(ceil_samples, round(1 + t * (ceil_samples - 1)))),
                max_seconds=_ONE_PASS_SECONDS + t * (hard_seconds - _ONE_PASS_SECONDS),
                target_score=self.base_target, predicted_uncertainty=u,
                signature=a.signature,
                reason=(f"metacognition: calibrated uncertainty {u:.2f} ({a.reason}) → "
                        f"partial ladder"))
        return budget

    # ------------------------------------------------------------------ #
    # 3. value-of-computation stopping / live escalation (during the climb)
    # ------------------------------------------------------------------ #
    def _rung_belief(self, shape: str, rung: int) -> Optional[BetaBelief]:
        return self._rung_improves.get(shape, {}).get(rung)

    def should_continue(self, *, budget: TurnBudget, next_rung: int, best_score: float,
                        elapsed: float, ceiling: Any = None) -> StopDecision:
        """Between rungs: is the next rung worth its cost? (Metareasoning, not a fixed table.)

        Stops when the target is met, the clock would be blown, or the next rung's *expected*
        verified-score gain (learned from lived deltas) no longer justifies its expected cost.
        Escalates — the reverse direction — when a turn predicted easy measures weak: the budget
        is raised toward the ceiling mid-climb, because measurement outranks prediction."""
        if best_score >= budget.target_score:
            return StopDecision(True, False,
                                f"target met ({best_score:.2f} ≥ {budget.target_score:.2f})")

        stakes = _stakes_high(budget.signature)
        # measurement contradicts an easy prediction → escalate before anything else
        if (self.live_escalation and not budget.escalated
                and budget.predicted_uncertainty <= self.easy_uncertainty
                and best_score < budget.target_score - _ESCALATE_GAP):
            return StopDecision(False, True,
                                f"predicted easy but measured {best_score:.2f} — escalating")

        if next_rung > budget.max_rung:
            return StopDecision(True, False, "budget's rung ceiling reached")

        with self._lock:
            est = float(self._rung_seconds.get(next_rung,
                                               _DEFAULT_RUNG_SECONDS.get(next_rung, 10.0)))
            if elapsed + est > budget.max_seconds:
                return StopDecision(True, False,
                                    f"time budget ({elapsed:.0f}s + rung ≈{est:.0f}s > "
                                    f"{budget.max_seconds:.0f}s)")
            if self.voc_stopping and not stakes:
                b = self._rung_belief(_shape_key(budget.signature), next_rung)
                if b is not None and b.observations >= _VOC_MIN_OBS:
                    expected_gain = b.mean * max(0.0, budget.target_score - best_score)
                    if expected_gain < _VOC_GAIN_FLOOR:
                        return StopDecision(
                            True, False,
                            f"marginal value collapsed (rung {next_rung} pays off "
                            f"{b.mean:.0%} of the time here → expected gain "
                            f"{expected_gain:.3f} < {_VOC_GAIN_FLOOR})")
        return StopDecision(False, False, "expected gain justifies the next rung")

    def escalate(self, budget: TurnBudget, *, ceiling: Any = None) -> TurnBudget:
        """Raise a budget toward the full ceiling mid-climb (measurement overrode prediction)."""
        ceil_rung = int(getattr(ceiling, "max_rung", 4) or 4)
        ceil_samples = max(1, int(getattr(ceiling, "samples", 5) or 5))
        ceil_seconds = float(getattr(ceiling, "max_seconds", 60.0) or 60.0)
        budget.max_rung = max(budget.max_rung, ceil_rung)
        budget.samples = max(budget.samples, ceil_samples)
        budget.max_seconds = max(budget.max_seconds, ceil_seconds, self.hard_max_seconds)
        budget.escalated = True
        budget.reason += " [escalated mid-climb: measured harder than predicted]"
        return budget

    # ------------------------------------------------------------------ #
    # 4. the closed loop (after the turn)
    # ------------------------------------------------------------------ #
    def _decayed_observe(self, key: str, outcome: bool) -> None:
        b = self._belief(key)
        # decay old mass toward the prior (recency weighting) before folding in the new evidence,
        # so what "easy" means tracks drift instead of freezing at the first hundred turns.
        b.alpha = b.prior_alpha + (b.alpha - b.prior_alpha) * self.belief_decay
        b.beta = b.prior_beta + (b.beta - b.prior_beta) * self.belief_decay
        b.observe(outcome)

    def record(self, stimulus: str, *, predicted_uncertainty: float, target_score: float,
               scores: List[Tuple[int, float]], winning_rung: int,
               rung_seconds: Optional[Dict[int, float]] = None,
               stated_confidence: Optional[float] = None) -> None:
        """Fold one lived turn back into the beliefs — the step that makes prediction calibrated.

        ``scores`` is the climb's per-rung verified scores in run order (the
        :class:`LadderResult`'s trace). "The cheap pass sufficed" iff rung 1's score cleared the
        target or rung 1 won outright — that is the ground truth the difficulty beliefs and the
        calibrator learn from. Per-rung improvement deltas feed the VOC value estimates; measured
        wall-clock refines the cost estimates. Never raises."""
        try:
            sig = signature(stimulus)
            first = next((s for r, s in scores if r == 1), None)
            best = max((s for _, s in scores), default=0.0)
            was_hard = not ((first is not None and first >= target_score)
                            or (winning_rung <= 1 and best >= target_score))
            with self._lock:
                for key in (sig, _shape_key(sig), _GLOBAL_KEY):
                    self._decayed_observe(key, was_hard)
                self.calibrator.add(_clamp(float(predicted_uncertainty)), was_hard)
                if stated_confidence is not None:
                    # the model's own confidence, judged by the independent verifier — feeds
                    # the answer calibration the who-answers gates share.
                    self.answer_calibrator.add(_clamp(float(stated_confidence)),
                                               best >= target_score)
                # VOC evidence: did each later rung actually improve the best-so-far?
                shape = _shape_key(sig)
                best_so_far: Optional[float] = None
                for rung, score in scores:
                    if best_so_far is not None:
                        improved = score >= best_so_far + _MIN_GAIN
                        row = self._rung_improves.setdefault(shape, {})
                        b = row.get(rung)
                        if b is None:
                            b = BetaBelief()
                            row[rung] = b
                        b.alpha = b.prior_alpha + (b.alpha - b.prior_alpha) * self.belief_decay
                        b.beta = b.prior_beta + (b.beta - b.prior_beta) * self.belief_decay
                        b.observe(improved)
                    best_so_far = score if best_so_far is None else max(best_so_far, score)
                for rung, secs in (rung_seconds or {}).items():
                    prev = self._rung_seconds.get(int(rung),
                                                  _DEFAULT_RUNG_SECONDS.get(int(rung), 10.0))
                    self._rung_seconds[int(rung)] = ((1.0 - _SECONDS_EWMA) * prev
                                                     + _SECONDS_EWMA * max(0.0, float(secs)))
                self._turns += 1
                self._decisions.append({
                    "at": time.time(), "signature": sig,
                    "predicted": round(_clamp(float(predicted_uncertainty)), 4),
                    "was_hard": was_hard, "winning_rung": int(winning_rung),
                    "best_score": round(best, 4),
                })
            self.save()
        except Exception:  # noqa: BLE001 — learning is best-effort, never fatal to a turn
            pass

    # ------------------------------------------------------------------ #
    # audit
    # ------------------------------------------------------------------ #
    def report(self) -> Dict[str, Any]:
        """The full auditable state — what makes this first-class: inspectable and honest."""
        with self._lock:
            table = sorted(((k, b) for k, b in self._difficulty.items()
                            if not k.startswith("shape:") and k != _GLOBAL_KEY),
                           key=lambda kv: kv[1].observations, reverse=True)
            return {
                "turns": self._turns,
                "calibration": self.calibrator.to_dict(),
                "answer_calibration": self.answer_calibrator.to_dict(),
                "difficulty": {k: {"p_hard": round(b.mean, 3),
                                   "observations": round(b.observations, 1)}
                               for k, b in table[:12]},
                "rung_value": {shape: {r: round(b.mean, 3) for r, b in row.items()}
                               for shape, row in self._rung_improves.items()},
                "rung_seconds": {r: round(s, 2) for r, s in self._rung_seconds.items()},
                "recent": list(self._decisions)[-5:],
            }

    # ------------------------------------------------------------------ #
    # persistence (mirrors mind/effort_memory.py: atomic tmp-replace, lock-guarded)
    # ------------------------------------------------------------------ #
    def to_dict(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "difficulty": {k: {"alpha": b.alpha, "beta": b.beta}
                               for k, b in self._difficulty.items()},
                "rung_improves": {shape: {str(r): {"alpha": b.alpha, "beta": b.beta}
                                          for r, b in row.items()}
                                  for shape, row in self._rung_improves.items()},
                "rung_seconds": {str(r): s for r, s in self._rung_seconds.items()},
                "calibration": self.calibrator.state_dict(),
                "answer_calibration": self.answer_calibrator.state_dict(),
                "turns": self._turns,
            }

    def load_dict(self, data: Dict[str, Any]) -> "MetacognitiveAllocator":
        def _beta(params: Any) -> Optional[BetaBelief]:
            try:
                a, b = float(params["alpha"]), float(params["beta"])
                return BetaBelief(alpha=a, beta=b) if a > 0 and b > 0 else None
            except (KeyError, TypeError, ValueError):
                return None

        difficulty: Dict[str, BetaBelief] = {}
        for key, params in (data.get("difficulty") or {}).items():
            b = _beta(params)
            if b is not None:
                difficulty[str(key)] = b
        improves: Dict[str, Dict[int, BetaBelief]] = {}
        for shape, row in (data.get("rung_improves") or {}).items():
            if not isinstance(row, dict):
                continue
            parsed: Dict[int, BetaBelief] = {}
            for rung, params in row.items():
                b = _beta(params)
                try:
                    if b is not None:
                        parsed[int(rung)] = b
                except (TypeError, ValueError):
                    continue
            if parsed:
                improves[str(shape)] = parsed
        seconds = dict(_DEFAULT_RUNG_SECONDS)
        for rung, s in (data.get("rung_seconds") or {}).items():
            try:
                seconds[int(rung)] = max(0.0, float(s))
            except (TypeError, ValueError):
                continue
        with self._lock:
            self._difficulty = difficulty
            self._rung_improves = improves
            self._rung_seconds = seconds
            try:
                self.calibrator.load_state(data.get("calibration") or {})
                self.answer_calibrator.load_state(data.get("answer_calibration") or {})
            except Exception:  # noqa: BLE001 — calibration state is advisory
                pass
            try:
                self._turns = int(data.get("turns", 0))
            except (TypeError, ValueError):
                self._turns = 0
        return self

    def save(self, path: Optional[Path] = None) -> None:
        target = Path(path) if path is not None else self.path
        if target is None:
            return
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            tmp = target.with_suffix(target.suffix + ".tmp")
            tmp.write_text(json.dumps(self.to_dict(), ensure_ascii=False), encoding="utf-8")
            tmp.replace(target)   # atomic on POSIX — a crash mid-write never corrupts the state
        except Exception:  # noqa: BLE001 — persistence is best-effort, never fatal
            pass

    def load(self, path: Optional[Path] = None) -> "MetacognitiveAllocator":
        target = Path(path) if path is not None else self.path
        if target is None or not target.exists():
            return self
        return self.load_dict(json.loads(target.read_text(encoding="utf-8")))


# --------------------------------------------------------------------------- #
# Shared instance — ONE calibration source for the whole mind
# --------------------------------------------------------------------------- #
_shared: Dict[str, MetacognitiveAllocator] = {}
_shared_lock = threading.Lock()


def get_allocator(settings: Any = None) -> MetacognitiveAllocator:
    """The process-wide shared allocator (one per data-dir), so the how-much-compute decision
    and the who-answers gates (:mod:`mind.router`, :mod:`mind.self_model_router`) read the SAME
    calibrated beliefs. Never raises; falls back to an in-memory instance without persistence."""
    path: Optional[Path] = None
    cfg: Any = None
    try:
        if settings is None:
            from nyxara.kernel.config import get_settings
            settings = get_settings()
        cfg = getattr(getattr(settings, "llm", None), "deep_reasoning", None)
        data_dir = getattr(getattr(settings, "paths", None), "data_dir", None)
        if data_dir is not None:
            path = Path(data_dir) / "metacognitive_allocator.json"
    except Exception:  # noqa: BLE001
        path = None
    key = str(path) if path is not None else "<memory>"
    with _shared_lock:
        inst = _shared.get(key)
        if inst is None:
            inst = MetacognitiveAllocator(
                easy_uncertainty=float(getattr(cfg, "easy_uncertainty", 0.25)),
                hard_uncertainty=float(getattr(cfg, "hard_uncertainty", 0.70)),
                base_target=float(getattr(cfg, "early_stop_score", 0.80)),
                hard_max_seconds=float(getattr(cfg, "hard_max_seconds", 600.0)),
                belief_decay=float(getattr(cfg, "belief_decay", 0.99)),
                voc_stopping=bool(getattr(cfg, "voc_stopping", True)),
                live_escalation=bool(getattr(cfg, "live_escalation", True)),
                path=path)
            _shared[key] = inst
        return inst


# --------------------------------------------------------------------------- #
# Self-test / demo
# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # pragma: no cover
    import tempfile

    print("=" * 70)
    print("NYXARA metacognitive-allocator self-test")
    print("=" * 70)

    class _Ceiling:
        max_rung, samples, max_seconds = 4, 9, 120.0

    alloc = MetacognitiveAllocator(easy_uncertainty=0.25, hard_uncertainty=0.70)

    # 1) unknown problem kind → epistemic humility → thinks HARD by default (never bluffs cheap)
    novel = alloc.allocate("why is the sky blue?", ceiling=_Ceiling())
    print(f"\nnovel   : {novel.reason}")
    assert novel.predicted_uncertainty >= 0.70 and novel.max_rung == 4
    assert novel.max_seconds >= 600.0, "a hard/unknown problem earns the 10-minute search"

    # 2) lived experience: this kind keeps being easy → the SAME kind now gets ONE PASS
    q = "why is the sky blue?"
    for _ in range(12):
        alloc.record(q, predicted_uncertainty=0.8, target_score=0.8,
                     scores=[(1, 0.9)], winning_rung=1)
    easy = alloc.allocate(q, ceiling=_Ceiling(), mems_count=5)
    print(f"easy    : {easy.reason}")
    assert easy.predicted_uncertainty <= 0.25, easy.predicted_uncertainty
    assert easy.max_rung == 1 and easy.samples == 1 and easy.max_seconds <= 5.0

    # 3) a kind that keeps needing the ladder → full budget, ten-minute search
    hard_q = "prove the polynomial identity holds for all n"
    for _ in range(12):
        alloc.record(hard_q, predicted_uncertainty=0.3, target_score=0.8,
                     scores=[(1, 0.3), (2, 0.5), (3, 0.85)], winning_rung=3)
    hard = alloc.allocate(hard_q, ceiling=_Ceiling(), mems_count=5)
    print(f"hard    : {hard.reason}")
    assert hard.predicted_uncertainty >= 0.70 and hard.max_rung == 4
    assert hard.max_seconds >= 600.0

    # 4) hierarchical back-off: an UNSEEN signature of a well-worn easy shape borrows strength.
    #    A long question is a different signature (its length bucket differs) yet the same easy
    #    'question' shape, so shape-level evidence should soften its novelty below the hard bar.
    cousin = alloc.assess("what makes grass green, and does the same pigment explain why the "
                          "leaves on deciduous trees turn yellow and red every autumn season?",
                          mems_count=5)
    print(f"cousin  : {cousin.to_dict()}")
    assert cousin.signature != signature(q), "the cousin must be a genuinely new signature"
    assert cousin.uncertainty < 0.7, "shape-level evidence must soften a novel signature"

    # 5) high stakes NEVER get the one-pass shortcut, however easy history looks
    risky = "delete the production database"
    for _ in range(12):
        alloc.record(risky, predicted_uncertainty=0.8, target_score=0.8,
                     scores=[(1, 0.95)], winning_rung=1)
    guard = alloc.allocate(risky, ceiling=_Ceiling(), mems_count=5)
    print(f"stakes  : {guard.reason}")
    assert guard.max_rung == 4, "high stakes must climb, not shortcut"

    # 6) VOC stopping: rungs that historically never improve the answer stop the climb
    for _ in range(6):
        alloc.record(q, predicted_uncertainty=0.2, target_score=0.8,
                     scores=[(1, 0.85), (2, 0.85), (3, 0.85)], winning_rung=1)
    mid = TurnBudget(max_rung=4, samples=3, max_seconds=120.0, target_score=0.95,
                     predicted_uncertainty=0.5, signature="question/short/normal")
    voc = alloc.should_continue(budget=mid, next_rung=2, best_score=0.7, elapsed=1.0)
    print(f"voc     : {voc.to_dict()}")
    assert voc.stop and "value" in voc.reason

    # 7) target met → stop immediately (easy stays one pass even with budget to spare)
    met = alloc.should_continue(budget=mid, next_rung=2, best_score=0.96, elapsed=1.0)
    assert met.stop and "target met" in met.reason

    # 8) live escalation: predicted easy, measured weak → raise the budget mid-climb
    cheap = TurnBudget(max_rung=1, samples=1, max_seconds=5.0, target_score=0.8,
                       predicted_uncertainty=0.1, signature="question/short/normal")
    esc = alloc.should_continue(budget=cheap, next_rung=2, best_score=0.3, elapsed=0.5)
    print(f"escalate: {esc.to_dict()}")
    assert esc.escalate and not esc.stop
    cheap = alloc.escalate(cheap, ceiling=_Ceiling())
    assert cheap.escalated and cheap.max_rung == 4 and cheap.max_seconds >= 600.0

    # 9) calibration is measured and improves with outcomes
    rep = alloc.report()
    print(f"report  : calibration={rep['calibration']}")
    assert rep["calibration"]["samples"] >= 30
    assert rep["turns"] >= 30 and rep["difficulty"]

    # 10) persistence round-trips the learned state
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "meta.json"
        alloc.save(p)
        again = MetacognitiveAllocator(path=p)
        assert again.allocate(q, ceiling=_Ceiling(), mems_count=5).max_rung == 1
        assert again.calibrator.samples == alloc.calibrator.samples
    print("persist : learned allocation survives a restart ✓")

    # 11) never raises on garbage
    alloc.record("", predicted_uncertainty=float("nan"), target_score=0.8,
                 scores=[], winning_rung=0)
    junk = alloc.allocate("", ceiling=object())
    assert junk.max_rung >= 1
    print("garbage : never raises ✓")

    print("\nALL SELF-TESTS PASSED ✓")
