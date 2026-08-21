"""NYXARA · mind/metacontrol.py — first-class metacognitive compute allocation (🪞⬆ → 👑).

**The problem this closes.** NYXARA's metacognition existed but was an add-on: the deep-reasoning
controller always climbed the whole effort ladder to its ceiling, and nothing decided *before*
reasoning how hard this particular turn deserved to be thought about. An easy greeting cost the
same machinery as a novel proof. This module makes metacognition **first-class**: a calibrated
uncertainty estimate, computed upfront from NYXARA's own measured signals, drives the compute
allocation for the turn — an easy problem gets **one forward pass**, a hard one gets up to ten
minutes of search.

**Who decides — NYXARA, not the model.** Every judgment here is her own deterministic code:

* :meth:`MetacognitiveController.estimate` blends internal signals she already measures —
  hyperdimensional novelty, memory-recall strength, self-model competence, learned effort
  history, input shape/length/stakes — through :meth:`MetaCognition.introspect` into one
  difficulty in [0, 1]. No LLM is ever asked "how hard is this?".
* A :class:`~nyxara.mind.uncertainty.Calibrator` tracks predicted-easiness vs actual outcome,
  so the estimate is **calibrated**: if she keeps calling turns easy and then failing them,
  the correction raises her difficulty (and her confidence bar) automatically.
* :meth:`MetacognitiveController.allocate` maps calibrated difficulty to a monotone
  :class:`ComputeBudget` — entry rung, ladder ceiling, self-consistency width, wall-clock —
  spanning one forward pass (``entry_rung == 0``) to the full 600-second deep search.
* :meth:`MetacognitiveController.record_outcome` closes the loop after the kernel disposes:
  the turn's verified outcome updates the calibrator and per-band beliefs, and the state
  persists as JSON (like :mod:`~nyxara.mind.effort_memory`) so calibration compounds across
  restarts.

The ladder itself (:mod:`~nyxara.mind.deep_reasoning`) consults the budget in-flight: it starts
at the allocated entry rung and stops early the moment the verifier clears the calibrated
confidence target — escalating past the entry rung only when allowed and needed. A wrong "easy"
call is recoverable (escalation) and self-correcting (the calibrator observes it).

Safety is untouched: the controller only ever sizes *effort*; every candidate still passes every
kernel gate. Pure standard library plus NYXARA's own uncertainty/effort primitives.
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from nyxara.mind.effort_memory import EffortMemory, signature
from nyxara.mind.metacognition import MetaCognition
from nyxara.mind.uncertainty import BetaBelief, Calibrator

__all__ = ["DifficultySignals", "DifficultyEstimate", "ComputeBudget",
           "MetacognitiveController", "RUNG_FAST"]

# Entry rung 0 — the fast path: a single forward pass, no ladder at all. Rungs 1..4 are the
# deep-reasoning ladder's own (mind/deep_reasoning.py mirrors these as ints).
RUNG_FAST = 0

_BANDS = ("easy", "moderate", "hard", "extreme")

# Deterministic content cues (her own reading of the prompt's FORM, no LLM): words that mark
# genuine multi-step reasoning work, and smalltalk that marks a turn as conversational.
_HARD_CUES = ("prove", "theorem", "derive", "derivation", "from first principles", "optimi",
              "algorithm", "complexity", "step by step", "rigorous", "formal", "counterexample",
              "why does", "why is", "explain how", "explain why", "design a", "plan a",
              "strategy", "analyze", "analyse", "compare and", "trade-off", "tradeoff",
              "implement", "refactor", "debug", "integral", "equation", "conjecture", "lemma",
              "what would happen if", "counterfactual", "simulate")
#: Speech acts that are about the two of them, or about her, rather than about the world — the
#: names :class:`nyxara.njp.relevance.SpeechAct` uses. Kept as bare strings so this module keeps
#: its "deterministic, no dependencies" character: the caller does the reading, this only prices it.
_SOCIAL_ACTS = frozenset({"greeting", "farewell", "thanks", "social_checkin", "state_query"})

#: What a social turn's difficulty may reach, whatever the runtime signals say. Below
#: ``easy_below`` (0.30) by design, so such a turn lands on the fast path.
#:
#: The bug this exists for: latent novelty answers "have I met this input before", and for
#: smalltalk that has nothing to do with how hard the turn is. A greeting she has not seen
#: scores novelty ~1.0, ``0.35 * introspected`` alone carries it to 0.38, and the first "Hii" of
#: every session bought the full apparatus — six role-council queries, five recursive-improve
#: iterations — for a word she could answer in one pass. A greeting she has never seen is still
#: a greeting.
_SOCIAL_CEILING = 0.12

_SMALLTALK = ("hello", "hi ", "hi!", "hi.", "hey", "thanks", "thank you", "good morning",
              "good evening", "good night", "how are you", "nice to", "ok", "okay", "yes",
              "no problem", "great", "cool", "bye", "goodbye", "see you")


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


@dataclass
class DifficultySignals:
    """The internal signals a difficulty estimate was blended from (each optional, [0,1])."""

    novelty: Optional[float] = None            # hyperdimensional latent novelty (1 = never seen)
    recall_strength: Optional[float] = None    # best semantic score of recalled memories
    competence: Optional[float] = None         # self-model mastery for this prompt's domain
    effort_history: Optional[float] = None     # learned nudge from EffortMemory (0.5 = neutral)
    self_consistency: Optional[float] = None   # agreement across own samples (1 = stable)
    length: float = 0.0                        # word-count ramp (full at ~120 words)
    stakes: float = 0.0                        # keyword stakes (mirrors effort_memory buckets)
    shape: str = "statement"                   # math | command | question | statement
    act: str = ""                              # speech act, when the caller read one

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {"length": round(self.length, 4), "stakes": round(self.stakes, 4),
                               "shape": self.shape}
        if self.act:
            out["act"] = self.act
        for name in ("novelty", "recall_strength", "competence", "effort_history",
                     "self_consistency"):
            v = getattr(self, name)
            if v is not None:
                out[name] = round(float(v), 4)
        return out


@dataclass
class DifficultyEstimate:
    """An upfront, auditable difficulty judgment for one turn."""

    difficulty: float          # raw blend, [0,1]
    calibrated: float          # after Calibrator correction (== raw until enough samples)
    signature: str             # effort_memory.signature(stimulus) — the coarse problem bucket
    signals: DifficultySignals
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {"difficulty": round(self.difficulty, 4), "calibrated": round(self.calibrated, 4),
                "signature": self.signature, "signals": self.signals.to_dict(),
                "reason": self.reason}


@dataclass
class ComputeBudget:
    """A concrete per-turn compute allocation, spanning 1 forward pass … the full deep search.

    Duck-type compatible with :class:`~nyxara.growth.effective_scale.ReasoningBudget` consumers
    (``max_rung`` / ``samples`` / ``max_seconds``), plus the metacognitive fields the upgraded
    ladder reads: ``entry_rung`` (0 = fast path, skip the ladder entirely), ``confidence_target``
    (stop climbing once the verifier clears it) and ``allow_escalation``.
    """

    max_rung: int
    samples: int
    max_seconds: float
    entry_rung: int = 1
    confidence_target: Optional[float] = None
    allow_escalation: bool = True
    estimate: Optional[DifficultyEstimate] = None
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {"entry_rung": self.entry_rung, "max_rung": self.max_rung,
                "samples": self.samples, "max_seconds": round(self.max_seconds, 2),
                "confidence_target": (None if self.confidence_target is None
                                      else round(self.confidence_target, 4)),
                "allow_escalation": self.allow_escalation, "reason": self.reason,
                "estimate": self.estimate.to_dict() if self.estimate is not None else None}


class MetacognitiveController:
    """Calibrated uncertainty → compute allocation. NYXARA's own code decides; the LLM never does."""

    def __init__(self, *, settings: Any = None, effort_memory: Any = None,
                 ceiling: Any = None, probe: Optional[Callable[[str], Optional[float]]] = None,
                 path: Optional[Path] = None) -> None:
        if settings is None:
            from nyxara.kernel.config import get_settings
            settings = get_settings()
        self.settings = settings
        self.cfg = getattr(settings, "metacontrol", None)
        self.deep_cfg = getattr(getattr(settings, "llm", None), "deep_reasoning", None)
        self.probe = probe
        self.path = Path(path) if path is not None else None
        self._lock = threading.RLock()
        self.calibrator = Calibrator()
        self._band: Dict[str, BetaBelief] = {b: BetaBelief() for b in _BANDS}
        self._records = 0
        self.last_plan: Optional[ComputeBudget] = None
        # the compute-scaled ceiling (growth/effective_scale.ReasoningBudget); lazily computed.
        self._ceiling_cache: Any = ceiling
        # Share one persisted EffortMemory with the deep ladder, so the same lived outcomes that
        # aim its samples also inform the upfront difficulty estimate.
        if effort_memory is None and bool(getattr(self.deep_cfg, "learn_effort", True)):
            try:
                em_path = None
                data_dir = getattr(getattr(settings, "paths", None), "data_dir", None)
                if data_dir is not None:
                    em_path = Path(data_dir) / "effort_memory.json"
                effort_memory = EffortMemory(
                    min_observations=float(getattr(self.deep_cfg, "effort_min_observations", 3.0)),
                    success_floor=float(getattr(self.deep_cfg, "effort_success_floor", 0.5)),
                    path=em_path)
            except Exception:  # noqa: BLE001 — effort history is advisory, never required
                effort_memory = None
        self.effort_memory = effort_memory
        if self.path is not None and self.path.exists():
            try:
                self.load(self.path)
            except Exception:  # noqa: BLE001 — a corrupt file must never break reasoning
                pass

    # ------------------------------------------------------------------ #
    # config access (works with or without a MetaControlConfig attached)
    # ------------------------------------------------------------------ #
    def _cfgv(self, name: str, default: Any) -> Any:
        return getattr(self.cfg, name, default) if self.cfg is not None else default

    def enabled(self) -> bool:
        return bool(self._cfgv("enabled", True))

    # ------------------------------------------------------------------ #
    # (1) upfront difficulty — deterministic, before any reasoning
    # ------------------------------------------------------------------ #
    def estimate(self, stimulus: str, *, novelty: Optional[float] = None,
                 recall_strength: Optional[float] = None,
                 competence: Optional[float] = None,
                 self_consistency: Optional[float] = None,
                 act: Optional[str] = None) -> DifficultyEstimate:
        """Blend NYXARA's own signals into one calibrated difficulty for this turn.

        ``act`` is the caller's reading of what the Master was *doing* with the turn
        (:class:`nyxara.njp.relevance.SpeechAct`). It is not another hardness cue — it is the one
        signal that can say a turn is trivial *regardless* of what the runtime signals think, and
        without it an unfamiliar greeting was priced as a hard problem. See :data:`_SOCIAL_CEILING`.
        """
        sig = signature(stimulus)
        parts = sig.split("/")
        shape = parts[0] if parts else "statement"
        stakes_bucket = parts[2] if len(parts) > 2 else "normal"
        words = len((stimulus or "").split())
        length = _clamp(words / 120.0)
        stakes = 0.8 if stakes_bucket == "high" else (0.55 if shape == "command" else 0.3)

        # learned effort history: a signature whose cheap rungs keep paying off is known-easy;
        # one the ladder has learned to escalate on (favoured rung 3/4) is known-hard.
        nudge = 0.0
        if self.effort_memory is not None:
            try:
                cheap = [m for m in (self.effort_memory.mean_for(sig, 1),
                                     self.effort_memory.mean_for(sig, 2)) if m is not None]
                if cheap and max(cheap) >= 0.7:
                    nudge -= 0.15 * max(cheap)
                if self.effort_memory.suggest(sig) in (3, 4):
                    nudge += 0.15
            except Exception:  # noqa: BLE001 — effort history is advisory
                nudge = 0.0

        # epistemic term: how much she does NOT already hold this (weak recall, low mastery)
        epi_terms = [1.0 - _clamp(v) for v in (recall_strength, competence) if v is not None]
        epistemic = sum(epi_terms) / len(epi_terms) if epi_terms else None
        introspected = MetaCognition.introspect(
            self_consistency=self_consistency, epistemic=epistemic, novelty=novelty)

        # content hardness — the prompt's own FORM, read deterministically: reasoning-work
        # cues raise it, smalltalk lowers it. This keeps the estimate honest when the
        # runtime signals are thin (first turn, no retriever) or noisy (novelty spikes).
        low = (stimulus or "").strip().lower()
        hardness = min(0.45, 0.15 * sum(1 for cue in _HARD_CUES if cue in low))
        smalltalk = (-0.15 if (any(low.startswith(c) for c in _SMALLTALK) and words <= 12
                               and hardness == 0.0) else 0.0)

        difficulty = (0.35 * introspected + 0.20 * length + 0.25 * stakes
                      + hardness + smalltalk + nudge)
        if shape == "math":
            # searchable & verifiable — always worth at least a modest ladder budget
            difficulty = max(difficulty, 0.35)
        elif shape == "statement":
            difficulty -= 0.05
        difficulty = _clamp(difficulty)

        # A social or reflexive act caps the whole estimate. Deliberately a cap rather than another
        # subtracted term: the terms it has to beat are unbounded blends of runtime signals, and a
        # −0.15 nudge lost to a novelty spike every time — which is exactly how this was missed.
        # Guarded by `hardness == 0.0` so "hey, why does the reactor scram at 400K" is still work.
        act = str(act or "").strip().lower()
        capped = bool(act in _SOCIAL_ACTS and hardness == 0.0)
        if capped:
            difficulty = min(difficulty, _SOCIAL_CEILING)
        calibrated = self._calibrate(difficulty)

        signals = DifficultySignals(
            novelty=novelty, recall_strength=recall_strength, competence=competence,
            effort_history=(None if nudge == 0.0 else _clamp(0.5 + nudge)),
            self_consistency=self_consistency, length=length, stakes=stakes, shape=shape,
            act=act)
        reason = (f"introspected={introspected:.2f} hardness={hardness:.2f} length={length:.2f} "
                  f"stakes={stakes:.2f} nudge={nudge:+.2f} shape={shape}"
                  + (f" act={act} (capped at {_SOCIAL_CEILING})" if capped
                     else (f" act={act}" if act else ""))
                  + f" -> difficulty={difficulty:.2f} (calibrated {calibrated:.2f})")
        return DifficultyEstimate(difficulty=difficulty, calibrated=calibrated,
                                  signature=sig, signals=signals, reason=reason)

    def _calibrate(self, difficulty: float) -> float:
        """Correct a raw difficulty against lived outcomes: predicted-ease vs actual sufficiency."""
        with self._lock:
            if self.calibrator.samples < int(self._cfgv("min_calibration_samples", 20)):
                return difficulty
            ease = 1.0 - _clamp(difficulty)
            return _clamp(1.0 - self.calibrator.recalibrate(ease))

    # ------------------------------------------------------------------ #
    # (2) calibrated difficulty -> compute budget (monotone, bounded)
    # ------------------------------------------------------------------ #
    def allocate(self, est: DifficultyEstimate) -> ComputeBudget:
        """Map a difficulty estimate to a concrete budget: 1 forward pass … full deep search."""
        d = _clamp(est.calibrated)
        stakes = est.signals.stakes
        easy_below = float(self._cfgv("easy_below", 0.30))
        moderate_below = float(self._cfgv("moderate_below", 0.55))
        hard_below = float(self._cfgv("hard_below", 0.80))
        seconds_ceiling = float(self._cfgv("max_seconds_ceiling", 600.0))
        escalation = bool(self._cfgv("escalation", True))

        ceiling = self._ceiling()
        ceil_rung = max(1, int(getattr(ceiling, "max_rung", None)
                               or getattr(self.deep_cfg, "max_rung", 4) or 4))
        ceil_samples = max(1, int(getattr(ceiling, "samples", None)
                                  or getattr(self.deep_cfg, "samples", 3) or 3))
        ceil_seconds = float(getattr(ceiling, "max_seconds", None)
                             or getattr(self.deep_cfg, "max_seconds", 60.0) or 60.0)
        base_samples = max(1, int(getattr(self.deep_cfg, "samples", 3) or 3))

        if d < easy_below and stakes < float(self._cfgv("fast_path_max_stakes", 0.5)):
            entry, max_rung, samples, seconds = RUNG_FAST, 1, 1, 5.0
            band = "easy"
        elif d < moderate_below:
            entry, max_rung = 1, min(2, ceil_rung)
            samples, seconds = base_samples, min(30.0, max(ceil_seconds, 30.0))
            band = "moderate"
        elif d < hard_below:
            entry, max_rung = 2, min(3, ceil_rung)
            samples, seconds = base_samples + 2, min(180.0, max(ceil_seconds, 60.0))
            band = "hard"
        else:
            entry, max_rung = 3, ceil_rung
            samples = ceil_samples
            seconds = min(seconds_ceiling, max(ceil_seconds, 300.0))
            band = "extreme"

        # ---- three invariants the band table cannot express on its own ---- #
        # (1) No band may out-spend the machine. ``ceil_samples`` is a hardware-scaled ceiling
        #     (growth/effective_scale.scaled_budget), while the moderate/hard rows are written off
        #     the STATIC config — so on a machine that scales down, ``base_samples + 2`` asked for
        #     more sampling than the ceiling allows, and more than the *extreme* band above it was
        #     permitted. Clamping here fixes the over-spend and the inversion in one move: harder
        #     bands can now only ever buy the same compute or more.
        samples = max(1, min(samples, ceil_samples))
        # (2) An entry rung above the max rung is not a budget, it is a contradiction. A low
        #     ``ceil_rung`` used to leave e.g. entry=3 with max_rung=1 — enter above the ceiling and
        #     the ladder has nowhere to start.
        max_rung = max(1, min(max_rung, ceil_rung))
        entry = min(entry, max_rung)
        # (3) ``max_seconds_ceiling`` must actually be a ceiling. It was consulted only by the
        #     *extreme* band; moderate and hard used ``max(ceil_seconds, 30/60)``, so lowering the
        #     configured ceiling could not lower them at all — a knob documented as a cap that
        #     could only ever raise the budget. Exactly the hole invariant (1) already closes for
        #     samples, left open for seconds. Clamping here makes the two consistent, and gives the
        #     test suite a way to say "no turn here is worth three minutes of deliberation" without
        #     disabling metacontrol and losing the allocation entirely.
        seconds = max(0.1, min(seconds, seconds_ceiling))

        # A history of over-confident "easy" calls raises the bar the ladder must clear before
        # it may stop early — calibration steering effort, not only the estimate.
        with self._lock:
            over = max(0.0, self.calibrator.overconfidence())
        target = _clamp(float(self._cfgv("confidence_target", 0.75)) + 0.5 * over, 0.0, 0.95)

        budget = ComputeBudget(
            max_rung=max_rung, samples=samples, max_seconds=seconds, entry_rung=entry,
            confidence_target=target, allow_escalation=escalation, estimate=est,
            reason=(f"{band} (calibrated difficulty {d:.2f}) -> rung {entry}..{max_rung}, "
                    f"{samples} samples, {seconds:.0f}s, stop at {target:.2f}"))
        return budget

    def plan(self, stimulus: str, *, novelty: Optional[float] = None,
             recall_strength: Optional[float] = None, competence: Optional[float] = None,
             self_consistency: Optional[float] = None,
             act: Optional[str] = None) -> ComputeBudget:
        """Estimate + allocate in one act; remembers the plan for outcome recording / reporting."""
        if (self_consistency is None and self.probe is not None
                and bool(self._cfgv("probe_self_consistency", False))):
            try:
                self_consistency = self.probe(stimulus)
            except Exception:  # noqa: BLE001 — the probe is optional evidence, never required
                self_consistency = None
        est = self.estimate(stimulus, novelty=novelty, recall_strength=recall_strength,
                            competence=competence, self_consistency=self_consistency, act=act)
        budget = self.allocate(est)
        self.last_plan = budget
        return budget

    # ------------------------------------------------------------------ #
    # (3) in-flight decision the ladder consults
    # ------------------------------------------------------------------ #
    @staticmethod
    def should_stop(budget: ComputeBudget, rung: int, score: float) -> bool:
        """Early exit: the verifier already cleared this turn's calibrated confidence target."""
        target = getattr(budget, "confidence_target", None)
        return target is not None and float(score) >= float(target)

    # ------------------------------------------------------------------ #
    # (4) close the loop: lived outcome -> calibration
    # ------------------------------------------------------------------ #
    def record_outcome(self, budget: Optional[ComputeBudget], *, success: bool,
                       verified_score: Optional[float] = None,
                       rung_used: Optional[int] = None,
                       escalated: bool = False, early_exit: bool = False) -> None:
        """Fold the turn's real outcome back into the calibrator and band beliefs.

        The allocation was *sufficient* iff the turn succeeded without escalating past its entry
        rung and the verified score cleared the floor — escalation on a success is the honest
        "I under-allocated" signal, and a failure on a predicted-easy turn is over-confidence
        the calibrator will correct.
        """
        if budget is None or getattr(budget, "estimate", None) is None:
            return
        est = budget.estimate
        floor = float(self._cfgv("success_floor", 0.5))
        sufficient = bool(success) and not bool(escalated) and (
            verified_score is None or float(verified_score) >= floor)
        with self._lock:
            self.calibrator.add(_clamp(1.0 - est.difficulty), sufficient)
            self._band[self._band_for(est.calibrated)].observe(sufficient)
            self._records += 1
            due = (self._records % max(1, int(self._cfgv("persist_every", 5)))) == 0
        if due and self.path is not None and bool(self._cfgv("persist", True)):
            try:
                self.save()
            except Exception:  # noqa: BLE001 — persistence is best-effort, never fatal
                pass

    def _band_for(self, d: float) -> str:
        if d < float(self._cfgv("easy_below", 0.30)):
            return "easy"
        if d < float(self._cfgv("moderate_below", 0.55)):
            return "moderate"
        if d < float(self._cfgv("hard_below", 0.80)):
            return "hard"
        return "extreme"

    # ------------------------------------------------------------------ #
    # compute ceiling (lazily scaled to the machine's real compute)
    # ------------------------------------------------------------------ #
    def _ceiling(self) -> Any:
        if self._ceiling_cache is not None:
            return self._ceiling_cache or None
        try:
            from nyxara.growth.effective_scale import scaled_budget
            from nyxara.kernel.compute import compute_report
            self._ceiling_cache = scaled_budget(compute_report(), self.deep_cfg)
        except Exception:  # noqa: BLE001 — scaling is additive; the static config stands alone
            self._ceiling_cache = False
        return self._ceiling_cache or None

    # ------------------------------------------------------------------ #
    # persistence (mirrors EffortMemory: atomic JSON, fail-open load)
    # ------------------------------------------------------------------ #
    def to_dict(self) -> Dict[str, Any]:
        with self._lock:
            return {"calibrator": self.calibrator.state_dict(),
                    "bands": {b: {"alpha": bel.alpha, "beta": bel.beta}
                              for b, bel in self._band.items()},
                    "records": self._records}

    def load_dict(self, data: Dict[str, Any]) -> "MetacognitiveController":
        with self._lock:
            self.calibrator = Calibrator.from_state(dict((data or {}).get("calibrator") or {}))
            bands = (data or {}).get("bands") or {}
            for b in _BANDS:
                params = bands.get(b)
                try:
                    a, be = float(params["alpha"]), float(params["beta"])
                    if a > 0 and be > 0:
                        self._band[b] = BetaBelief(alpha=a, beta=be)
                except (KeyError, TypeError, ValueError):
                    self._band[b] = BetaBelief()
            try:
                self._records = int((data or {}).get("records", 0))
            except (TypeError, ValueError):
                self._records = 0
        return self

    def save(self, path: Optional[Path] = None) -> None:
        target = Path(path) if path is not None else self.path
        if target is None:
            return
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(target.suffix + ".tmp")
        tmp.write_text(json.dumps(self.to_dict(), ensure_ascii=False), encoding="utf-8")
        tmp.replace(target)  # atomic on POSIX — a crash mid-write never corrupts the state

    def load(self, path: Optional[Path] = None) -> "MetacognitiveController":
        target = Path(path) if path is not None else self.path
        if target is None or not target.exists():
            return self
        return self.load_dict(json.loads(target.read_text(encoding="utf-8")))

    # ------------------------------------------------------------------ #
    # introspection (for /explain and the reporter)
    # ------------------------------------------------------------------ #
    def report(self) -> Dict[str, Any]:
        with self._lock:
            return {"calibration": self.calibrator.to_dict(),
                    "bands": {b: round(bel.mean, 4) for b, bel in self._band.items()},
                    "records": self._records,
                    "last_plan": (self.last_plan.to_dict()
                                  if self.last_plan is not None else None)}


# --------------------------------------------------------------------------- #
# Self-test / demo
# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # pragma: no cover
    import tempfile

    print("=" * 70)
    print("NYXARA metacognitive-control self-test")
    print("=" * 70)

    from nyxara.kernel.config import NyxaraSettings, Profile
    settings = NyxaraSettings.for_profile(Profile.TEST)

    mc = MetacognitiveController(settings=settings, path=None)

    # deterministic: same stimulus + signals -> identical estimate
    e1 = mc.estimate("hi there")
    e2 = mc.estimate("hi there")
    assert e1.to_dict() == e2.to_dict()

    # easy < hard, and the budgets order the same way
    easy = mc.plan("hi there")
    hard = mc.plan("prove that every finite division ring is a field, from first principles",
                   novelty=0.9, recall_strength=0.0)
    print(f"easy : {easy.reason}")
    print(f"hard : {hard.reason}")
    assert easy.estimate.difficulty < hard.estimate.difficulty
    assert easy.entry_rung == RUNG_FAST and easy.samples == 1   # genuinely 1 forward pass
    assert hard.entry_rung >= 2 and hard.max_seconds > easy.max_seconds

    # high stakes never takes the fast path, however easy the words look
    risky = mc.plan("delete the production database")
    print(f"risky: {risky.reason}")
    assert risky.entry_rung >= 1

    # budget mapping is monotone in difficulty
    prev = None
    for d10 in range(11):
        est = DifficultyEstimate(difficulty=d10 / 10.0, calibrated=d10 / 10.0,
                                 signature="question/short/normal",
                                 signals=DifficultySignals(stakes=0.3))
        b = mc.allocate(est)
        if prev is not None:
            assert b.entry_rung >= prev.entry_rung and b.max_rung >= prev.max_rung
            assert b.samples >= prev.samples and b.max_seconds >= prev.max_seconds
        prev = b
    print("budget mapping        : monotone in difficulty ✓")

    # calibration loop: keep calling turns easy and failing -> difficulty gets corrected UP
    raw = mc.estimate("hi there").difficulty
    for _ in range(30):
        p = mc.plan("hi there")
        mc.record_outcome(p, success=False)
    corrected = mc.estimate("hi there").calibrated
    print(f"calibration           : raw {raw:.2f} -> calibrated {corrected:.2f} after 30 failures")
    assert corrected > raw
    assert mc.plan("hi there").confidence_target >= 0.75  # over-confidence raises the bar

    # persistence round-trip
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "metacontrol.json"
        mc.save(p)
        mc2 = MetacognitiveController(settings=settings, path=p)
        assert mc2.calibrator.samples == mc.calibrator.samples
        assert abs(mc2._band["easy"].mean - mc._band["easy"].mean) < 1e-9
    print("persistence           : calibration survives a restart ✓")

    print("\nALL SELF-TESTS PASSED ✓")
