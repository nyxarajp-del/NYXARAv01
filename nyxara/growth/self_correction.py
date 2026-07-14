"""NYXARA · growth/self_correction.py — the self-correcting, epistemic loop (♺, Rule 4 & 6).

A mind that can *act* but never knows **when it is wrong** is not intelligent — it is
merely busy. The Master's gap: NYXARA could call tools yet had no faculty that, *while she
works*, notices she is **likely-wrong** or **stuck in a loop**, honestly names the gap ("I
don't know"), and **runs a new experiment to fill it** before changing course. This module
is that faculty — the active controller that turns NYXARA's existing epistemic *primitives*
into behaviour.

It **composes** faculties she already has and re-implements none of them:

* :mod:`nyxara.mind.uncertainty` — Bayesian belief (epistemic vs aleatoric), calibration,
  and the :class:`~nyxara.mind.uncertainty.AbstentionPolicy` that says *"I don't know."*
* :mod:`nyxara.mind.metacognition` — the "do I actually know this?" introspection gate.
* :mod:`nyxara.mind.critique` — the devil's-advocate :class:`~nyxara.mind.critique.Critic`.
* :mod:`nyxara.mind.grounded_verifier` — a truth-scored verifier ``(prompt, answer) -> float``.
* :mod:`nyxara.mind.prediction_engine` — calibrated prediction, so **surprise** (prediction
  error) exposes a wrong belief *on the first try*, not only after a loop.
* :mod:`nyxara.planning.voi` — value-of-information, to pick the **most informative** probe.
* :mod:`nyxara.planning.planner` — to **decompose** a stuck goal into sub-goals.
* :mod:`nyxara.growth.scientist` / :mod:`nyxara.growth.grounded_experiments` — to run a real,
  sandboxed / reality-graded **experiment** and read its verdict.

The closed loop it drives:

    1. PREDICT   — before an action, predict whether it will make progress.
    2. OBSERVE   — after it, measure **surprise** (predicted vs actual).
    3. ASSESS    — fuse surprise + loop/cycle detection + epistemic uncertainty into one
                   :class:`EpistemicVerdict`: *progressing / uncertain / stuck / likely-wrong*.
    4. RECOVER   — on a gap: name it honestly, pick a strategy (a **learned bandit** over
                   RETRY→REFRAME→DECOMPOSE→EXPERIMENT→CONSULT→ABSTAIN→ESCALATE, VoI-guided),
                   run the experiment, and hand back a *finding* to inject as a new observation.
    5. LEARN     — record which recovery worked (per gap-type), so competence compounds and
                   the same stuck-state is recognised — and fixed — instantly next time.
    6. GATE      — before speaking a final answer, abstain honestly if it can't be trusted.

Every decision here is **NYXARA's own deterministic code** — there is no LLM in this loop.
All dependencies are injected and optional, so the faculty degrades gracefully and stays
fully offline / CI-safe. It fails as *data*: nothing here raises into the cognitive cycle,
and loyalty, corrigibility and the Master's primacy are never touched — experiments are
sandboxed and the last rung of the ladder is an honest escalation to the Master.

Pure standard library plus in-repo imports.
"""

from __future__ import annotations

import hashlib
import json
import random
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from nyxara.mind.metacognition import HONEST_ABSTENTION, MetaCognition
from nyxara.mind.uncertainty import AbstentionPolicy, BetaBelief, Calibrator, UncertaintyModel

__all__ = [
    "EpistemicState",
    "Strategy",
    "EpistemicVerdict",
    "Recovery",
    "SelfCorrectionLoop",
]


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def _sig(text: str, n: int = 6) -> str:
    """A short, stable signature of a piece of text (first ``n`` normalised tokens)."""
    toks = [t for t in "".join(c.lower() if c.isalnum() else " " for c in str(text)).split() if t]
    if not toks:
        return "∅"
    return " ".join(toks[:n])


def _hash(text: str) -> str:
    return hashlib.sha1(str(text).encode("utf-8", "replace")).hexdigest()[:12]


# --------------------------------------------------------------------------- #
# Verdict & recovery types
# --------------------------------------------------------------------------- #
class EpistemicState(str, Enum):
    PROGRESSING = "progressing"   # moving toward the goal — nothing to correct
    UNCERTAIN = "uncertain"       # not confident enough — she may not know
    STUCK = "stuck"               # looping / no new information — she is spinning
    LIKELY_WRONG = "likely_wrong"  # confident but surprised / a critic objects — probably wrong


class Strategy(str, Enum):
    RETRY = "retry"           # try the same move once more (weakest — rarely new info)
    REFRAME = "reframe"       # restate the problem from a different angle
    DECOMPOSE = "decompose"   # break the goal into smaller sub-goals (Planner)
    EXPERIMENT = "experiment"  # run a real, sandboxed experiment to fill the gap (Scientist)
    CONSULT = "consult"       # gather external evidence / ask a knowledge source
    ABSTAIN = "abstain"       # honestly say "I don't know" rather than bluff
    ESCALATE = "escalate"     # surface to the Master — the honest last resort


@dataclass
class EpistemicVerdict:
    """NYXARA's own read of whether she is progressing, unsure, stuck, or likely wrong."""

    state: EpistemicState
    epistemic: float = 0.0      # reducible ignorance [0,1] — "I haven't seen enough"
    surprise: float = 0.0       # prediction error [0,1] — "that's not what I expected"
    confidence: float = 0.0     # confidence of the latest step [0,1]
    reason: str = ""

    @property
    def needs_correction(self) -> bool:
        return self.state in (EpistemicState.STUCK, EpistemicState.LIKELY_WRONG,
                              EpistemicState.UNCERTAIN)

    @property
    def gap_type(self) -> str:
        return self.state.value

    def to_dict(self) -> Dict[str, Any]:
        return {"state": self.state.value, "epistemic": round(self.epistemic, 4),
                "surprise": round(self.surprise, 4), "confidence": round(self.confidence, 4),
                "needs_correction": self.needs_correction, "reason": self.reason}


@dataclass
class Recovery:
    """The outcome of one recovery attempt — a finding to inject and how it was reached."""

    strategy: Strategy
    acted: bool = False          # did she actually run a probe / produce a move?
    new_info: bool = False       # did it yield genuinely new information to inject?
    finding: str = ""            # the text to fold back in as a fresh observation
    reason: str = ""
    experiment: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {"strategy": self.strategy.value, "acted": self.acted,
                "new_info": self.new_info, "finding": self.finding[:400],
                "reason": self.reason, "experiment": self.experiment}


# strategy ladders per gap-type: most-diversifying / most-informative moves first
_LADDER: Dict[str, Tuple[Strategy, ...]] = {
    EpistemicState.STUCK.value: (Strategy.DECOMPOSE, Strategy.EXPERIMENT, Strategy.REFRAME,
                                 Strategy.RETRY, Strategy.ESCALATE),
    EpistemicState.LIKELY_WRONG.value: (Strategy.EXPERIMENT, Strategy.REFRAME,
                                        Strategy.DECOMPOSE, Strategy.ABSTAIN),
    EpistemicState.UNCERTAIN.value: (Strategy.EXPERIMENT, Strategy.CONSULT, Strategy.REFRAME,
                                     Strategy.ABSTAIN),
}
_DEFAULT_LADDER: Tuple[Strategy, ...] = (Strategy.EXPERIMENT, Strategy.REFRAME)


# --------------------------------------------------------------------------- #
# The controller
# --------------------------------------------------------------------------- #
class SelfCorrectionLoop:
    """Active self-correction & epistemic-uncertainty controller (composes existing faculties)."""

    def __init__(self, *,
                 uncertainty: Any = None, metacognition: Any = None, critic: Any = None,
                 verifier: Any = None, predictor: Any = None, voi: Any = None,
                 planner: Any = None, scientist: Any = None, grounded_experiments: Any = None,
                 active_curiosity: Any = None, reflector: Any = None, memory: Any = None,
                 settings: Any = None, path: Any = None,
                 max_recoveries: int = 2, epistemic_floor: float = 0.5,
                 surprise_floor: float = 0.55, stuck_repeat: int = 2,
                 answer_min_confidence: float = 0.55, seed: int = 1729) -> None:
        # thresholds
        self.max_recoveries = max(0, int(max_recoveries))
        self.epistemic_floor = float(epistemic_floor)
        self.surprise_floor = float(surprise_floor)
        self.stuck_repeat = max(1, int(stuck_repeat))
        self.answer_min_confidence = float(answer_min_confidence)

        # epistemic primitives (owned defaults keep the faculty usable standalone/offline)
        self._uncertainty: UncertaintyModel = uncertainty or UncertaintyModel(
            policy=AbstentionPolicy(min_confidence=answer_min_confidence,
                                    max_epistemic=epistemic_floor))
        self._meta = metacognition or MetaCognition(answer_threshold=answer_min_confidence)
        self._critic = critic if critic is not None else self._default_critic()
        self._verifier = verifier if verifier is not None else self._default_verifier(settings)
        self._voi = voi if voi is not None else self._default_voi()
        self._planner = planner if planner is not None else self._default_planner()

        # experiment engines (injected — the core shares its live ones)
        self._predictor = predictor
        self._scientist = scientist
        self._grounded = grounded_experiments
        self._curiosity = active_curiosity
        self._reflector = reflector
        self._memory = memory
        self._settings = settings

        # learning state (persisted)
        self._rng = random.Random(seed)
        self._policy: Dict[str, Dict[str, List[float]]] = {}   # gap_type -> strategy -> [α, β]
        self._lessons: Dict[str, Dict[str, Any]] = {}          # situation-sig -> proven recovery
        self._step_beliefs: Dict[str, BetaBelief] = {}         # intent-sig -> P(makes progress)
        self._surprise_cal = Calibrator()
        self._last_prediction: Dict[str, float] = {}
        self._stats = {"assessments": 0, "gaps": 0, "recoveries": 0, "experiments": 0,
                       "new_info": 0, "abstentions": 0, "escalations": 0}

        self.path = self._resolve_path(path, settings)
        self._load()

    # ------------------------------------------------------------------ #
    # default faculty builders (best-effort — degrade to None on any error)
    # ------------------------------------------------------------------ #
    @staticmethod
    def _default_critic() -> Any:
        try:
            from nyxara.mind.critique import Critic
            return Critic()
        except Exception:  # noqa: BLE001 — critique is a bonus signal, never required
            return None

    @staticmethod
    def _default_verifier(settings: Any) -> Any:
        try:
            from nyxara.mind.grounded_verifier import grounded_verifier
            return grounded_verifier(settings=settings)
        except Exception:  # noqa: BLE001 — truth verification degrades to "unavailable"
            return None

    @staticmethod
    def _default_voi() -> Any:
        try:
            from nyxara.planning.voi import ValueOfInformation
            return ValueOfInformation()
        except Exception:  # noqa: BLE001
            return None

    @staticmethod
    def _default_planner() -> Any:
        try:
            from nyxara.planning.planner import Planner
            return Planner()
        except Exception:  # noqa: BLE001
            return None

    # ------------------------------------------------------------------ #
    # 1–2. PREDICT-then-VERIFY — surprise exposes a wrong belief
    # ------------------------------------------------------------------ #
    def predict_step(self, intent: str) -> float:
        """Predict the probability that ``intent`` makes real progress toward the goal.

        Uses the injected :class:`PredictionEngine` when present; otherwise NYXARA's own
        learned Beta-belief for this kind of move. The prediction is remembered so the next
        observation can measure how *surprised* she should be."""
        key = _sig(intent)
        p = self._step_beliefs.setdefault(key, BetaBelief()).mean
        if self._predictor is not None:
            try:
                pred = self._predictor.predict(intent)
                pp = getattr(pred, "probability", None)
                if pp is not None:
                    p = 0.5 * p + 0.5 * float(pp)   # blend her own base-rate with the engine
            except Exception:  # noqa: BLE001 — prediction is advisory, never fatal
                pass
        p = _clamp(p)
        self._last_prediction[key] = p
        return p

    def observe_surprise(self, intent: str, made_progress: bool) -> float:
        """Measure surprise = |predicted − actual| for ``intent`` and learn from the outcome.

        High surprise on a *confident* move is the signal that a belief was wrong. The learned
        Beta-belief and (if present) the PredictionEngine both update, so surprise self-calibrates.
        """
        key = _sig(intent)
        predicted = self._last_prediction.get(key, self._step_beliefs.setdefault(key, BetaBelief()).mean)
        actual = 1.0 if made_progress else 0.0
        surprise = _clamp(abs(predicted - actual))
        self._step_beliefs.setdefault(key, BetaBelief()).observe(bool(made_progress))
        self._surprise_cal.add(predicted, bool(made_progress))
        if self._predictor is not None:
            try:
                self._predictor.observe_outcome(intent, bool(made_progress))
            except Exception:  # noqa: BLE001
                pass
        self._last_prediction.pop(key, None)
        return surprise

    # ------------------------------------------------------------------ #
    # truth-scored self-check — "is this answer actually right?"
    # ------------------------------------------------------------------ #
    def check_answer(self, prompt: str, answer: str, *, confidence: float = 0.5,
                     evidence: Optional[Sequence[str]] = None) -> Dict[str, Any]:
        """Judge whether ``answer`` is likely *correct* (not merely fluent).

        Combines the truth-scored grounded verifier with the devil's-advocate critic. Returns
        a dict with a ``likely_wrong`` flag the assess/gate steps consume."""
        truth: Optional[float] = None
        if self._verifier is not None and answer:
            try:
                truth = float(self._verifier(prompt, answer))
            except Exception:  # noqa: BLE001 — verification degrades to "no signal"
                truth = None
        verdict = "ok"
        severity = 0.0
        if self._critic is not None and answer:
            try:
                from nyxara.mind.critique import CritiqueContext
                crit = self._critic.critique(CritiqueContext(
                    claim=str(answer), confidence=_clamp(confidence),
                    evidence=list(evidence or [])))
                verdict = crit.verdict.value
                severity = float(crit.severity)
            except Exception:  # noqa: BLE001
                pass
        # "likely wrong" only when a signal is actually available and it is adverse
        likely_wrong = bool((truth is not None and truth < 0.5) or verdict == "object")
        return {"truth": (round(truth, 4) if truth is not None else None),
                "verdict": verdict, "severity": round(severity, 4),
                "likely_wrong": likely_wrong}

    # ------------------------------------------------------------------ #
    # 3. ASSESS — fuse surprise + loop/cycle + epistemic into one state
    # ------------------------------------------------------------------ #
    def assess(self, *, signatures: Sequence[str], confidences: Sequence[float],
               no_progress: int = 0, surprise: float = 0.0,
               novelty: Optional[float] = None) -> EpistemicVerdict:
        """NYXARA's own metacognitive read of the run so far.

        ``signatures`` are per-step state fingerprints (kind:tool:observation); ``confidences``
        the per-step confidences; ``no_progress`` the count of consecutive no-new-observation
        steps; ``surprise`` the latest prediction error. Detects exact repeats, short cycles
        (oscillation), and monotone-no-improvement — not merely "identical twice"."""
        self._stats["assessments"] += 1
        sigs = list(signatures)
        confs = list(confidences)
        last_conf = _clamp(confs[-1]) if confs else 0.0

        loop_signal = self._loop_signal(sigs, no_progress)
        # self-consistency: how varied her recent moves are (all-same → low consistency of *progress*)
        distinct = len(set(sigs[-4:])) if sigs else 0
        self_consistency = _clamp(distinct / 4.0) if sigs else 1.0
        epistemic = MetaCognition.introspect(
            self_consistency=self_consistency,
            epistemic=(1.0 - last_conf),
            novelty=novelty)

        # decide the state — order matters: confident-but-surprised is the strongest "wrong" signal
        if surprise >= self.surprise_floor and last_conf >= self.answer_min_confidence:
            state = EpistemicState.LIKELY_WRONG
            reason = (f"confident ({last_conf:.2f}) but surprised ({surprise:.2f}) — "
                      "the outcome contradicts my belief")
        elif loop_signal >= 0.5 or no_progress >= self.stuck_repeat:
            state = EpistemicState.STUCK
            reason = self._loop_reason(sigs, no_progress)
        elif epistemic >= self.epistemic_floor or last_conf < self.answer_min_confidence:
            state = EpistemicState.UNCERTAIN
            reason = (f"epistemic uncertainty {epistemic:.2f} / confidence {last_conf:.2f} "
                      "below the bar — I may not know this")
        else:
            state = EpistemicState.PROGRESSING
            reason = "making progress"

        verdict = EpistemicVerdict(state, epistemic=epistemic, surprise=_clamp(surprise),
                                   confidence=last_conf, reason=reason)
        if verdict.needs_correction:
            self._stats["gaps"] += 1
        return verdict

    def _loop_signal(self, sigs: List[str], no_progress: int) -> float:
        """A [0,1] score for how loop-like the recent trajectory is."""
        if len(sigs) < 2:
            return 0.0
        score = 0.0
        # exact repeat of the last move
        if sigs[-1] == sigs[-2]:
            score = max(score, 1.0)
        # period-2 oscillation A,B,A,B
        if len(sigs) >= 4 and sigs[-1] == sigs[-3] and sigs[-2] == sigs[-4]:
            score = max(score, 0.8)
        # period-3 cycle A,B,C,A,B,C
        if len(sigs) >= 6 and sigs[-1] == sigs[-4] and sigs[-2] == sigs[-5] and sigs[-3] == sigs[-6]:
            score = max(score, 0.7)
        # monotone no-improvement
        if no_progress >= self.stuck_repeat:
            score = max(score, 0.6)
        return _clamp(score)

    def _loop_reason(self, sigs: List[str], no_progress: int) -> str:
        if len(sigs) >= 2 and sigs[-1] == sigs[-2]:
            return "repeating the same action with no new result — stuck"
        if len(sigs) >= 4 and sigs[-1] == sigs[-3]:
            return "oscillating between the same states — stuck in a cycle"
        return f"{no_progress} steps with no new information — going nowhere"

    # ------------------------------------------------------------------ #
    # 4. RECOVER — name the gap, pick a strategy, run an experiment
    # ------------------------------------------------------------------ #
    def recover(self, goal: str, observations: Sequence[str],
                verdict: Optional[EpistemicVerdict] = None, *, gap_hint: str = "") -> Recovery:
        """React to a detected epistemic gap: pick the best recovery and execute it.

        Returns a :class:`Recovery` whose ``finding`` (when ``new_info``) is meant to be folded
        back into the run as a fresh observation, breaking the loop with real new information."""
        self._stats["recoveries"] += 1
        gap_type = verdict.gap_type if verdict is not None else EpistemicState.STUCK.value
        situation = self._situation_sig(goal, observations, gap_type)
        strategy = self._choose_strategy(gap_type, situation, verdict)

        rec = self._execute_strategy(strategy, goal, observations, verdict, gap_hint)

        # learn: did this strategy produce genuinely new information?
        self._record_policy(gap_type, strategy, rec.new_info)
        if rec.new_info:
            self._remember_lesson(situation, strategy)
            self._stats["new_info"] += 1
        if strategy is Strategy.ABSTAIN:
            self._stats["abstentions"] += 1
        if strategy is Strategy.ESCALATE:
            self._stats["escalations"] += 1
        self._save()
        return rec

    def _choose_strategy(self, gap_type: str, situation: str,
                         verdict: Optional[EpistemicVerdict]) -> Strategy:
        """Pick a recovery strategy: proven lesson first, else a VoI-guided learned bandit."""
        # a proven recovery for this exact situation wins outright
        proven = self._lessons.get(situation)
        if proven and int(proven.get("wins", 0)) >= 2:
            try:
                return Strategy(proven["strategy"])
            except Exception:  # noqa: BLE001 — a stale lesson never blocks recovery
                pass

        candidates = list(_LADDER.get(gap_type, _DEFAULT_LADDER))
        candidates = [s for s in candidates if self._strategy_available(s)]
        if not candidates:
            return Strategy.ESCALATE

        # VoI: is it worth *gathering* (an experiment) rather than just re-trying cheaply?
        if not self._worth_gathering(verdict) and Strategy.EXPERIMENT in candidates \
                and len(candidates) > 1:
            candidates = [s for s in candidates if s is not Strategy.EXPERIMENT] or candidates

        # learned bandit: Thompson-sample each candidate's success belief, take the best draw
        best, best_draw = candidates[0], -1.0
        for s in candidates:
            a, b = self._policy_ab(gap_type, s)
            draw = self._rng.betavariate(a, b)
            if draw > best_draw:
                best, best_draw = s, draw
        return best

    def _worth_gathering(self, verdict: Optional[EpistemicVerdict]) -> bool:
        if self._voi is None or verdict is None:
            return True
        try:
            from nyxara.planning.voi import DecisionContext
            evpi = self._voi.evpi(DecisionContext(
                uncertainty=_clamp(max(verdict.epistemic, verdict.surprise)),
                stakes=0.6, reversibility=1.0))
            return evpi > self._voi.act_threshold
        except Exception:  # noqa: BLE001 — VoI is advisory; default to gathering
            return True

    def _strategy_available(self, s: Strategy) -> bool:
        if s is Strategy.EXPERIMENT:
            return self._scientist is not None or self._grounded is not None \
                or self._curiosity is not None
        if s is Strategy.DECOMPOSE:
            return True   # a heuristic split always works, Planner sharpens it
        return True

    def _execute_strategy(self, strategy: Strategy, goal: str, observations: Sequence[str],
                          verdict: Optional[EpistemicVerdict], gap_hint: str) -> Recovery:
        gap = gap_hint or (verdict.reason if verdict is not None else "no progress")
        if strategy is Strategy.EXPERIMENT:
            return self._do_experiment(goal, observations, gap)
        if strategy is Strategy.DECOMPOSE:
            return self._do_decompose(goal, gap)
        if strategy is Strategy.REFRAME:
            return self._do_reframe(goal, observations, gap)
        if strategy is Strategy.CONSULT:
            return self._do_experiment(goal, observations, gap, prefer_curiosity=True)
        if strategy is Strategy.RETRY:
            return Recovery(Strategy.RETRY, acted=True, new_info=False,
                            finding=f"Retrying with more care on: {goal[:80]}",
                            reason="cheap retry — no new information gathered")
        if strategy is Strategy.ABSTAIN:
            return Recovery(Strategy.ABSTAIN, acted=False, new_info=False,
                            finding=HONEST_ABSTENTION,
                            reason=f"cannot resolve honestly ({gap}) — abstaining")
        # ESCALATE
        return Recovery(Strategy.ESCALATE, acted=False, new_info=False,
                        finding=("I've tried self-correcting and I'm still stuck, Master — "
                                 f"I don't yet know how to make progress on: {goal[:100]}"),
                        reason=f"recovery ladder exhausted ({gap}) — escalating to the Master")

    # ---- the experiment: hypothesise → run → conclude → inject ---- #
    def _do_experiment(self, goal: str, observations: Sequence[str], gap: str,
                       *, prefer_curiosity: bool = False) -> Recovery:
        question = self._frame_question(goal, gap)
        self._stats["experiments"] += 1
        # 1) the Scientist — a falsifiable, sandboxed investigation
        if self._scientist is not None and not prefer_curiosity:
            try:
                report = self._scientist.investigate(question)
                concl = getattr(report, "conclusion", None)
                verdict = getattr(getattr(concl, "verdict", None), "value", "inconclusive")
                reasoning = getattr(concl, "reasoning", "") or ""
                nxt = getattr(concl, "next_hypothesis", "") or ""
                new_info = bool(reasoning) and verdict != "inconclusive"
                finding = (f"Experiment on '{question[:60]}' → {verdict}: {reasoning[:200]}"
                           + (f" (next: {nxt[:80]})" if nxt else ""))
                return Recovery(Strategy.EXPERIMENT, acted=True, new_info=new_info,
                                finding=finding, reason="ran a sandboxed scientific experiment",
                                experiment={"question": question, "verdict": verdict,
                                            "next_hypothesis": nxt})
            except Exception:  # noqa: BLE001 — a failed experiment is data, not a crash
                pass
        # 2) Active Curiosity — pose & internally test her own question
        if self._curiosity is not None:
            for attr in ("wonder", "tick"):
                fn = getattr(self._curiosity, attr, None)
                if callable(fn):
                    try:
                        res = fn()
                        text = self._stringify(res)
                        if text:
                            return Recovery(Strategy.CONSULT, acted=True, new_info=True,
                                            finding=f"Curiosity probe → {text[:220]}",
                                            reason="posed and tested my own question")
                    except Exception:  # noqa: BLE001
                        pass
                    break
        # nothing produced new information
        return Recovery(Strategy.EXPERIMENT, acted=True, new_info=False,
                        finding=f"Investigated '{question[:60]}' but found nothing conclusive",
                        reason="experiment ran but yielded no new information")

    def _do_decompose(self, goal: str, gap: str) -> Recovery:
        subgoals = self._subgoals(goal)
        if not subgoals:
            return Recovery(Strategy.DECOMPOSE, acted=False, new_info=False,
                            finding=f"Could not decompose: {goal[:80]}",
                            reason="no smaller sub-goals found")
        finding = ("The goal is stuck as one piece — breaking it into sub-goals to tackle "
                   "one at a time: " + "; ".join(subgoals[:4]))
        return Recovery(Strategy.DECOMPOSE, acted=True, new_info=True, finding=finding,
                        reason="decomposed a stuck goal into sub-goals",
                        experiment={"subgoals": subgoals[:6]})

    def _do_reframe(self, goal: str, observations: Sequence[str], gap: str) -> Recovery:
        obs_tail = self._stringify(observations[-1]) if observations else ""
        reframed = (f"Approach '{goal[:70]}' differently: the last route ({gap}) isn't working; "
                    f"question the assumption behind it"
                    + (f" and reconsider what '{obs_tail[:60]}' actually tells me" if obs_tail else ""))
        return Recovery(Strategy.REFRAME, acted=True, new_info=True, finding=reframed,
                        reason="reframed the problem from a new angle")

    # ---- helpers ---- #
    def _frame_question(self, goal: str, gap: str) -> str:
        return f"Why am I not making progress on: {goal[:90]}? ({gap[:60]})"

    def _subgoals(self, goal: str) -> List[str]:
        # a light, dependency-free split; the Planner (if present) validates the shape
        parts: List[str] = []
        for sep in (" and then ", " then ", " and ", "; ", ", "):
            if sep in goal:
                parts = [p.strip() for p in goal.split(sep) if p.strip()]
                if len(parts) >= 2:
                    break
        if len(parts) < 2:
            g = goal.strip().rstrip("?.")
            parts = [f"clarify what '{g[:60]}' requires",
                     f"identify the first concrete step of '{g[:60]}'",
                     f"verify each step of '{g[:60]}' before moving on"]
        return parts

    @staticmethod
    def _stringify(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value
        for attr in ("to_dict", "summary"):
            fn = getattr(value, attr, None)
            if callable(fn):
                try:
                    return json.dumps(fn(), default=str)[:300]
                except Exception:  # noqa: BLE001
                    pass
        return repr(value)[:300]

    def _situation_sig(self, goal: str, observations: Sequence[str], gap_type: str) -> str:
        tail = self._stringify(observations[-1]) if observations else ""
        return _hash(f"{gap_type}|{_sig(goal)}|{_sig(tail)}")

    # ------------------------------------------------------------------ #
    # 5. LEARN — the recovery-policy bandit + lesson memory
    # ------------------------------------------------------------------ #
    def _policy_ab(self, gap_type: str, strategy: Strategy) -> Tuple[float, float]:
        ab = self._policy.setdefault(gap_type, {}).setdefault(strategy.value, [1.0, 1.0])
        return float(ab[0]), float(ab[1])

    def _record_policy(self, gap_type: str, strategy: Strategy, worked: bool) -> None:
        ab = self._policy.setdefault(gap_type, {}).setdefault(strategy.value, [1.0, 1.0])
        if worked:
            ab[0] += 1.0
        else:
            ab[1] += 1.0

    def record_outcome(self, gap_type: str, strategy: Strategy, worked: bool) -> None:
        """External confirmation that a recovery ultimately helped (or not). Persists."""
        self._record_policy(gap_type, strategy if isinstance(strategy, Strategy)
                            else Strategy(strategy), worked)
        self._save()

    def _remember_lesson(self, situation: str, strategy: Strategy) -> None:
        lesson = self._lessons.setdefault(situation, {"strategy": strategy.value, "wins": 0})
        if lesson.get("strategy") == strategy.value:
            lesson["wins"] = int(lesson.get("wins", 0)) + 1
        else:
            lesson["strategy"], lesson["wins"] = strategy.value, 1
        lesson["at"] = time.time()
        # optional: fold into the reflection tower as a durable lesson
        if self._reflector is not None:
            rec = getattr(self._reflector, "note", None) or getattr(self._reflector, "add_lesson", None)
            if callable(rec):
                try:
                    rec(f"when {situation}, recover via {strategy.value}")
                except Exception:  # noqa: BLE001 — reflection is best-effort
                    pass

    # ------------------------------------------------------------------ #
    # 6. GATE — never speak a confident-but-wrong final answer
    # ------------------------------------------------------------------ #
    def gate_answer(self, prompt: str, answer: str, confidence: float, *,
                    epistemic: float = 0.0) -> Dict[str, Any]:
        """Decide whether ``answer`` may be spoken, or must be honestly abstained.

        Runs the abstention policy (calibrated confidence + epistemic) and the truth-scored
        self-check. If the answer can't be trusted, returns the honest "I don't know"."""
        check = self.check_answer(prompt, answer, confidence=confidence)
        decision = self._uncertainty.assess_confidence(_clamp(confidence), epistemic=_clamp(epistemic))
        abstain = bool(decision.should_abstain or check["likely_wrong"])
        if abstain:
            self._stats["abstentions"] += 1
            reason = (check["verdict"] == "object" and "a critical objection stands") \
                or (check["likely_wrong"] and "the answer does not verify as true") \
                or decision.reason
            return {"abstain": True, "answer": HONEST_ABSTENTION,
                    "confidence": round(decision.confidence, 4), "reason": reason,
                    "check": check}
        return {"abstain": False, "answer": answer,
                "confidence": round(decision.confidence, 4),
                "reason": "verified and calibrated-confident enough to answer", "check": check}

    # ------------------------------------------------------------------ #
    # reporting & persistence
    # ------------------------------------------------------------------ #
    def report(self) -> Dict[str, Any]:
        best: Dict[str, Dict[str, Any]] = {}
        for gap_type, strategies in self._policy.items():
            ranked = sorted(strategies.items(), key=lambda kv: kv[1][0] / (kv[1][0] + kv[1][1]),
                            reverse=True)
            if ranked:
                s, ab = ranked[0]
                best[gap_type] = {"strategy": s, "success_rate": round(ab[0] / (ab[0] + ab[1]), 3)}
        return {"stats": dict(self._stats), "best_recovery": best,
                "lessons": len(self._lessons),
                "surprise_calibration": self._surprise_cal.to_dict(),
                "max_recoveries": self.max_recoveries}

    def snapshot(self) -> Dict[str, Any]:
        return {"version": 1, "policy": self._policy, "lessons": self._lessons,
                "step_beliefs": {k: [b.alpha, b.beta] for k, b in self._step_beliefs.items()},
                "stats": self._stats}

    def restore(self, data: Dict[str, Any]) -> None:
        if not isinstance(data, dict):
            return
        self._policy = {k: {s: [float(v[0]), float(v[1])] for s, v in d.items()}
                        for k, d in (data.get("policy") or {}).items()}
        self._lessons = dict(data.get("lessons") or {})
        self._step_beliefs = {k: BetaBelief(float(v[0]), float(v[1]), 1.0, 1.0)
                              for k, v in (data.get("step_beliefs") or {}).items()}
        self._stats.update({k: int(v) for k, v in (data.get("stats") or {}).items()
                            if k in self._stats})

    @staticmethod
    def _resolve_path(path: Any, settings: Any) -> Optional[Path]:
        if path is not None:
            return Path(path)
        try:
            if settings is None:
                from nyxara.kernel.config import get_settings
                settings = get_settings()
            return Path(settings.paths.data_dir) / "self_correction.json"
        except Exception:  # noqa: BLE001 — no settings ⇒ in-memory only
            return None

    def _load(self) -> None:
        if self.path is None or not self.path.exists():
            return
        try:
            self.restore(json.loads(self.path.read_text(encoding="utf-8")))
        except Exception:  # noqa: BLE001 — a corrupt store is simply "nothing learned yet"
            pass

    def _save(self) -> None:
        if self.path is None:
            return
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(self.path.suffix + ".tmp")
            tmp.write_text(json.dumps(self.snapshot(), indent=2, default=str), encoding="utf-8")
            tmp.replace(self.path)   # atomic swap
        except Exception:  # noqa: BLE001 — a failed save must never crash the cycle
            pass


# --------------------------------------------------------------------------- #
# Self-test / demo
# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # pragma: no cover
    import tempfile

    print("=" * 70)
    print("NYXARA self-correction self-test")
    print("=" * 70)

    # a minimal in-repo Scientist so the experiment path is REAL (no LLM)
    from nyxara.growth.scientist import Scientist
    tmp = Path(tempfile.mkdtemp()) / "self_correction.json"
    scl = SelfCorrectionLoop(scientist=Scientist(), path=tmp, seed=7)

    # 1) predict-then-verify: a confident move that fails is SURPRISING
    scl.predict_step("call the calculator")
    for _ in range(6):
        scl._step_beliefs[_sig("call the calculator")].observe(True)   # build a strong prior
    scl.predict_step("call the calculator")
    surprise = scl.observe_surprise("call the calculator", made_progress=False)
    print(f"surprise (confident move failed): {surprise:.2f}")
    assert surprise > 0.5, "a failed confident move must be surprising"

    # 2) assess: exact repetition is detected as STUCK
    v_stuck = scl.assess(signatures=["act:now:x", "act:now:x", "act:now:x"],
                         confidences=[0.5, 0.5, 0.5], no_progress=2)
    print(f"repeat            : {v_stuck.to_dict()}")
    assert v_stuck.state is EpistemicState.STUCK

    # oscillation A,B,A,B is a cycle -> STUCK
    v_cycle = scl.assess(signatures=["A", "B", "A", "B"], confidences=[0.5, 0.5, 0.5, 0.5])
    assert v_cycle.state is EpistemicState.STUCK, "oscillation must read as stuck"

    # confident but surprised -> LIKELY_WRONG
    v_wrong = scl.assess(signatures=["A", "B"], confidences=[0.9, 0.9], surprise=0.8)
    print(f"surprised+conf    : {v_wrong.to_dict()}")
    assert v_wrong.state is EpistemicState.LIKELY_WRONG

    # varied progress, decent confidence -> PROGRESSING
    v_ok = scl.assess(signatures=["A", "B", "C"], confidences=[0.7, 0.75, 0.8], no_progress=0)
    assert v_ok.state is EpistemicState.PROGRESSING

    # 3) recover on a stuck run -> runs a REAL experiment / decompose, yields a finding
    rec = scl.recover("compute the 10th prime and then square it",
                      ["act:now:x", "act:now:x"], v_stuck)
    print(f"recovery          : {rec.to_dict()}")
    assert rec.acted and rec.finding, "recovery must act and produce a finding"
    assert rec.strategy in (Strategy.EXPERIMENT, Strategy.DECOMPOSE, Strategy.REFRAME,
                            Strategy.RETRY, Strategy.ESCALATE)

    # 4) learned bandit: reward DECOMPOSE for 'stuck' repeatedly -> it wins next time
    for _ in range(15):
        scl.record_outcome("stuck", Strategy.DECOMPOSE, worked=True)
        scl.record_outcome("stuck", Strategy.RETRY, worked=False)
    picks = [scl._choose_strategy("stuck", "sig", v_stuck) for _ in range(20)]
    print(f"bandit favours    : DECOMPOSE {picks.count(Strategy.DECOMPOSE)}/20")
    assert picks.count(Strategy.DECOMPOSE) >= 12, "bandit must learn the winning strategy"

    # 5) honest final-answer gate: low confidence -> abstain ("I don't know")
    gate = scl.gate_answer("what is the mass of the Higgs to 12 digits?", "about 125 GeV",
                           confidence=0.2, epistemic=0.9)
    print(f"gate (unsure)     : abstain={gate['abstain']} reason={gate['reason']}")
    assert gate["abstain"] and gate["answer"] == HONEST_ABSTENTION

    gate2 = scl.gate_answer("say hello", "Hello, Master.", confidence=0.95, epistemic=0.05)
    assert not gate2["abstain"], "a confident, verifiable answer must pass the gate"

    # 6) persistence round-trip
    scl2 = SelfCorrectionLoop(path=tmp)
    assert scl2._policy.get("stuck", {}).get("decompose", [1, 1])[0] > 5, "policy must persist"
    print(f"\nreport            : {json.dumps(scl.report(), default=str)[:400]}")

    print("\nALL SELF-TESTS PASSED ✓")
