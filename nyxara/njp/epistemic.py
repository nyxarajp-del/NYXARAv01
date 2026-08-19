"""NYXARA · njp/epistemic.py — "I don't know" → "what would let me know?" (🧪, NJP V.06).

:attr:`~nyxara.njp.adversary.Stance.UNDECIDED` is the honest verdict when the record cannot settle
an attack, and it was a dead end. Nothing consumed it. The claim kept its confidence, the attack
counted itself, and the turn moved on — so an undecided belief stayed undecided for exactly as long
as it took for nobody to look at it again.

That was survivable while every claim came from something the Master actually said. It stops being
survivable the moment :mod:`nyxara.njp.cortex` exists, because a hypothesis generator produces
undecided claims *faster than evidence arrives*: the natural end state is a ledger full of
plausible, unfalsified, unusable propositions. A mind in that state is not being careful. It is
accumulating.

This module is the consumer that was missing::

    UNDECIDED belief / competing hypotheses
      ↓  what would we observe under each?          ← the hypotheses' own predictions
      ↓  which observation SPLITS them?             ← the discrimination matrix below
      ↓  what is that worth, against what it costs? ← planning/voi.ValueOfInformation (EVPI)
      ↓  best experiment
      ↓  act | ask the Master | gather              ← curiosity.Question, which already retires them
      ↓  observation → update

**Almost none of this is new machinery, and that is the design.**
:class:`~nyxara.njp.adversary.SelfAttacker` already produces the undecided verdicts,
:meth:`~nyxara.njp.universe.InternalUniverse.intervene` already implements the real do-operator, and
:class:`~nyxara.planning.voi.ValueOfInformation` already prices information against the cost of
getting it. Three finished organs and no wire between them. The one genuinely new thing here is
**discriminating-experiment selection**, and it is one rule:

    An observation is worth making only if the candidates DISAGREE about it.

:meth:`EpistemicCompiler.discriminate` scores each candidate observation by how evenly it splits the
hypothesis set, and an observation that every surviving hypothesis predicts scores **zero and is
refused**. That refusal is the whole point. Testing a hypothesis against a prediction only it makes
— or that all of them make — is the failure :mod:`nyxara.eval.intelligence` names exactly: *a mind
that grades its guesses against its own later guesses is not learning, it is agreeing with itself.*
Confirmation feels like inquiry and costs the same. Only the split is inquiry.

**A bare causal claim gets its rivals synthesised.** When the only thing on the table is
``A causes B`` with no stated alternatives, :meth:`EpistemicCompiler.rivals_for` builds the standing
four — cause, reverse, confound, coincidence — and derives what each predicts under
:func:`do(A)` and :func:`do(¬A)`. Those four are not a guess about this claim; they are the four
ways a correlation can arise, and the do-operator is what tells them apart.

Pure standard library, and every entry point is fail-soft: nothing to discriminate returns no
experiment rather than an arbitrary one.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

log = logging.getLogger("nyxara.njp.epistemic")

__all__ = [
    "ObservationKind",
    "Observation",
    "Discrimination",
    "Experiment",
    "EpistemicCompiler",
]


class ObservationKind:
    """How an observation would be obtained. The cost differs, so the choice differs."""

    OBSERVE = "observe"        # wait for it / look it up in what she already records
    INTERVENE = "intervene"    # do(X) — the only thing that separates causation from correlation
    ASK = "ask"                # put it to the Master. Deliberately expensive.

    ALL = (OBSERVE, INTERVENE, ASK)


#: What each kind costs, in the [0,1] currency ``planning/voi`` prices against. Asking the Master is
#: the dearest on purpose: a mind that asks about everything has simply moved its work onto him.
_COST = {ObservationKind.OBSERVE: 0.15,
         ObservationKind.INTERVENE: 0.35,
         ObservationKind.ASK: 0.6}

#: Below this an "experiment" is not discriminating enough to be worth its cost — it is a
#: confirmation wearing the shape of a test.
_MIN_POWER = 1e-9


def _norm(text: Any) -> str:
    return " ".join(str(text or "").strip().lower().split())


@dataclass
class Observation:
    """One thing that could be observed or done, and what it would cost to get it."""

    text: str = ""
    kind: str = ObservationKind.OBSERVE

    @property
    def cost(self) -> float:
        return _COST.get(self.kind, _COST[ObservationKind.OBSERVE])

    def key(self) -> str:
        return _norm(self.text)

    def to_dict(self) -> Dict[str, Any]:
        return {"text": self.text, "kind": self.kind, "cost": round(self.cost, 3)}


@dataclass
class Discrimination:
    """One observation scored against the whole candidate set.

    :attr:`power` is the expected fraction of candidates eliminated, under the prior that the
    observation occurs in proportion to how many candidates predict it. It peaks when the set is
    split evenly and is **zero** when every candidate agrees — which is the case this module exists
    to refuse.
    """

    observation: Observation = field(default_factory=Observation)
    predicted_by: List[str] = field(default_factory=list)
    forbidden_by: List[str] = field(default_factory=list)
    silent: List[str] = field(default_factory=list)
    power: float = 0.0

    @property
    def total(self) -> int:
        return len(self.predicted_by) + len(self.forbidden_by) + len(self.silent)

    @property
    def discriminating(self) -> bool:
        """Do the candidates actually disagree about this? The only question that matters."""
        return bool(self.predicted_by) and bool(self.forbidden_by)

    @property
    def why(self) -> str:
        if self.discriminating:
            return (f"{len(self.predicted_by)} candidate(s) predict it and "
                    f"{len(self.forbidden_by)} rule it out — observing it eliminates one side")
        if not self.predicted_by and not self.forbidden_by:
            return "no candidate says anything about it — it could not eliminate any of them"
        side = "predict it" if self.predicted_by else "rule it out"
        return (f"every candidate that speaks to it says the same thing ({side}) — "
                "observing it would confirm, not discriminate")

    def to_dict(self) -> Dict[str, Any]:
        return {"observation": self.observation.to_dict(),
                "predicted_by": list(self.predicted_by),
                "forbidden_by": list(self.forbidden_by),
                "silent": list(self.silent),
                "power": round(self.power, 4),
                "discriminating": self.discriminating, "why": self.why}


@dataclass
class Experiment:
    """The observation worth making, why it is worth it, and how to go and get it."""

    observation: Observation = field(default_factory=Observation)
    discrimination: Discrimination = field(default_factory=Discrimination)
    candidates: List[str] = field(default_factory=list)
    evpi: float = 0.0
    net_value: float = 0.0
    action: str = "gather"               # act | ask | gather, from ValueOfInformation
    question: str = ""
    rationale: str = ""
    at: float = field(default_factory=time.time)

    @property
    def worth_running(self) -> bool:
        return self.discrimination.discriminating and self.net_value > 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {"observation": self.observation.to_dict(),
                "discrimination": self.discrimination.to_dict(),
                "candidates": list(self.candidates),
                "evpi": round(self.evpi, 4), "net_value": round(self.net_value, 4),
                "action": self.action, "question": self.question,
                "rationale": self.rationale, "worth_running": self.worth_running}


class EpistemicCompiler:
    """Turns an unresolved question into the observation that would resolve it."""

    def __init__(self, *, universe: Any = None, curiosity: Any = None, voi: Any = None,
                 stakes: float = 0.5) -> None:
        self.universe = universe
        self.curiosity = curiosity
        self._voi = voi
        self._voi_tried = voi is not None
        #: How much it costs to stay wrong about this class of question. Feeds EVPI directly.
        self.stakes = max(0.0, min(1.0, float(stakes)))
        self.compiled = 0
        self.refused = 0
        self.experiments = 0

    def _get_voi(self) -> Any:
        if not self._voi_tried:
            self._voi_tried = True
            try:
                from nyxara.planning.voi import ValueOfInformation
                self._voi = ValueOfInformation()
            except Exception:  # noqa: BLE001
                self._voi = None
        return self._voi

    # ---- candidate sets ----------------------------------------------------- #
    @staticmethod
    def rivals_for(cause: str, effect: str) -> List[Any]:
        """The standing four rivals for a bare ``cause → effect`` claim, with their predictions.

        Not a guess about this particular claim: these are the four ways a correlation between two
        things can arise, and the do-operator is what separates them. Building them means an
        undecided claim with no stated alternatives still has something to be discriminated against
        — which is the difference between "we cannot test this" and "nobody wrote the rivals down".
        """
        from nyxara.njp.cortex import CortexHypothesis, HypothesisKind
        do_cause = f"do({cause}): {effect} follows"
        remove_cause = f"do(not {cause}): {effect} happens anyway"
        do_effect = f"do({effect}): {cause} follows"
        return [
            CortexHypothesis(
                claim=f"{cause} causes {effect}", kind=HypothesisKind.CAUSE,
                predicts=[do_cause], forbids=[remove_cause, do_effect]),
            CortexHypothesis(
                claim=f"{effect} causes {cause}", kind=HypothesisKind.REVERSE,
                predicts=[do_effect], forbids=[do_cause]),
            CortexHypothesis(
                claim=f"something else causes both {cause} and {effect}",
                kind=HypothesisKind.CONFOUND,
                predicts=[remove_cause], forbids=[do_cause]),
            CortexHypothesis(
                claim=f"{cause} and {effect} merely co-occur",
                kind=HypothesisKind.COINCIDENCE,
                predicts=[remove_cause], forbids=[do_cause]),
        ]

    @staticmethod
    def _observations(candidates: Sequence[Any]) -> List[Observation]:
        """Every distinct thing the candidates commit to, deduplicated by text."""
        seen: Dict[str, Observation] = {}
        for candidate in candidates:
            for text in list(getattr(candidate, "predicts", ()) or ()) + \
                    list(getattr(candidate, "forbids", ()) or ()):
                key = _norm(text)
                if not key or key in seen:
                    continue
                kind = (ObservationKind.INTERVENE if key.startswith("do(")
                        else ObservationKind.OBSERVE)
                seen[key] = Observation(text=str(text).strip(), kind=kind)
        return list(seen.values())

    # ---- the one rule ------------------------------------------------------- #
    def discriminate(self, observation: Observation,
                     candidates: Sequence[Any]) -> Discrimination:
        """Score one observation by how evenly it splits the candidate set.

        The power is the expected fraction eliminated: with ``p`` candidates predicting it and ``f``
        forbidding it, observing it kills ``f`` and not observing it kills ``p``, and under the
        proportional prior ``P(observed) = p/(p+f)`` the expectation is ``2pf/(p+f)``, normalised by
        the size of the set. It is maximal at ``p == f`` and **zero whenever either side is empty**
        — which is the refusal this module is for.
        """
        out = Discrimination(observation=observation)
        key = observation.key()
        for candidate in candidates:
            name = str(getattr(candidate, "claim", "") or getattr(candidate, "text", "") or candidate)
            predicts = {_norm(t) for t in (getattr(candidate, "predicts", ()) or ())}
            forbids = {_norm(t) for t in (getattr(candidate, "forbids", ()) or ())}
            if key in predicts and key in forbids:
                # A candidate that both predicts and forbids the same observation has said nothing
                # about it; it cannot be eliminated either way and counting it on both sides would
                # inflate the power of an observation that discriminates less, not more.
                out.silent.append(name)
            elif key in predicts:
                out.predicted_by.append(name)
            elif key in forbids:
                out.forbidden_by.append(name)
            else:
                out.silent.append(name)
        p, f = len(out.predicted_by), len(out.forbidden_by)
        out.power = 0.0 if (p == 0 or f == 0) else (2.0 * p * f) / ((p + f) * max(1, out.total))
        return out

    # ---- compilation --------------------------------------------------------- #
    def compile(self, question: str, candidates: Sequence[Any], *,
                stakes: Optional[float] = None,
                reversibility: float = 1.0) -> Optional[Experiment]:
        """The best discriminating experiment for these candidates, or ``None``.

        ``None`` is a real answer and is returned whenever no observation splits the set. Returning
        the least-bad confirmation instead would spend real effort to learn nothing, and — worse —
        would come back looking like evidence.
        """
        self.compiled += 1
        candidates = [c for c in (candidates or ()) if c is not None]
        if len(candidates) < 2:
            # One candidate is not a set. There is nothing to tell apart, and an "experiment"
            # against a lone hypothesis can only ever confirm it.
            self.refused += 1
            return None

        scored = [self.discriminate(obs, candidates) for obs in self._observations(candidates)]
        usable = [d for d in scored if d.discriminating and d.power > _MIN_POWER]
        if not usable:
            self.refused += 1
            log.debug("no discriminating observation for %r among %d candidates",
                      question, len(candidates))
            return None
        best = max(usable, key=lambda d: (d.power, -d.observation.cost))

        names = [str(getattr(c, "claim", "") or getattr(c, "text", "") or c) for c in candidates]
        experiment = Experiment(observation=best.observation, discrimination=best,
                                candidates=names)

        voi = self._get_voi()
        if voi is None:
            experiment.evpi = best.power
            experiment.net_value = best.power - best.observation.cost
            experiment.action = ("ask" if best.observation.kind == ObservationKind.ASK
                                 else "gather")
            experiment.rationale = "no VOI engine attached — priced on discriminating power alone"
        else:
            try:
                from nyxara.planning.voi import DecisionContext, InfoSource
                ctx = DecisionContext(
                    # Uncertainty IS the spread of the candidate set: two live hypotheses is real
                    # doubt, and five is more of it. Reading it off the set rather than taking it
                    # from a caller is what keeps this from being a number someone chose.
                    uncertainty=max(0.0, min(1.0, 1.0 - 1.0 / max(1, len(candidates)))),
                    stakes=self.stakes if stakes is None else max(0.0, min(1.0, float(stakes))),
                    reversibility=max(0.0, min(1.0, float(reversibility))))
                source = InfoSource(name=best.observation.text,
                                    kind=("ask" if best.observation.kind == ObservationKind.ASK
                                          else "gather"),
                                    reliability=min(1.0, best.power * 2.0),
                                    cost=best.observation.cost)
                recommendation = voi.recommend(ctx, [source])
                experiment.evpi = float(recommendation.evpi)
                experiment.net_value = float(voi.net_value(source, ctx))
                experiment.action = str(getattr(recommendation.action, "value",
                                                recommendation.action))
                experiment.rationale = recommendation.rationale
            except Exception as exc:  # noqa: BLE001 — a failed pricing still leaves a real experiment
                experiment.evpi = best.power
                experiment.net_value = best.power - best.observation.cost
                experiment.action = "gather"
                experiment.rationale = f"VOI pricing failed ({exc}); priced on power alone"

        experiment.question = self._question(question, best)
        self.experiments += 1
        return experiment

    def from_attack(self, attack: Any, *, question: str = "") -> Optional[Experiment]:
        """Compile an experiment out of an UNDECIDED attack report.

        This is the join the adversary never had. It produces a report whose verdict is *"nothing
        could be settled from the record"* and stops — which is correct, and is also the exact point
        at which something should go and get more record.
        """
        cause = str(getattr(attack, "cause", "") or "")
        effect = str(getattr(attack, "effect", "") or "")
        if not cause or not effect:
            return None
        if getattr(attack, "refuted_by", None):
            return None                  # already settled: it lost
        asked = question or f"does {cause} cause {effect}?"
        return self.compile(asked, self.rivals_for(cause, effect))

    # ---- handing it to the organ that already retires questions --------------- #
    def to_question(self, experiment: Experiment, *, subject: str = "",
                    predicate: str = "") -> Any:
        """Register the experiment as a :class:`~nyxara.njp.curiosity.Question`.

        Curiosity already carries what would close a question and already retires it on
        :meth:`~nyxara.njp.curiosity.Curiosity.resolve`, so an experiment that lands there cannot
        become a permanent open loop. ``None`` when no curiosity organ is attached.
        """
        if self.curiosity is None or experiment is None:
            return None
        try:
            from nyxara.njp.curiosity import Gap, Question
            question = Question(
                text=experiment.question,
                gap=Gap.UNKNOWN,
                subject=subject, predicate=predicate,
                uncertainty=max(0.0, min(1.0, experiment.discrimination.power * 2.0)),
                stakes=self.stakes,
                cost=experiment.observation.cost,
                value=experiment.net_value,
                action=experiment.action)
            questions = getattr(self.curiosity, "questions", None)
            if isinstance(questions, dict):
                questions.setdefault(question.key(), question)
            elif isinstance(questions, list):
                questions.append(question)
            return question
        except Exception as exc:  # noqa: BLE001 — a question that cannot be filed is not a crash
            log.debug("could not file the experiment as a question: %s", exc)
            return None

    @staticmethod
    def _question(asked: str, best: Discrimination) -> str:
        """Phrased as what to go and observe, and which candidates it kills either way."""
        head = f"{asked.strip()} " if asked else ""
        kills_if = ", ".join(best.forbidden_by) or "none"
        kills_unless = ", ".join(best.predicted_by) or "none"
        verb = "run" if best.observation.kind == ObservationKind.INTERVENE else "check whether"
        return (f"{head}→ {verb} '{best.observation.text}'. "
                f"If it holds, that rules out: {kills_if}. If it does not, that rules out: "
                f"{kills_unless}.")

    def stats(self) -> Dict[str, Any]:
        return {"compiled": self.compiled, "experiments": self.experiments,
                "refused": self.refused,
                "refusal_rate": (self.refused / self.compiled) if self.compiled else 0.0}
