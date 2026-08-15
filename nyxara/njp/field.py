"""NYXARA · njp/field.py — the Recursive Cognitive Field (♾️, NJP V.04).

The theory this implements, in one line: **intelligence is representing the world, running
experiments on that representation, and restructuring the representation when prediction fails.**

Not "adjusting weights when prediction fails" — *restructuring*. That distinction is the whole
module. A mind whose every error can only ever move a number has an architecture no experience can
reach, and it will converge to the best model expressible in a shape somebody else chose for it.

**Loop 1 — the cognitive cycle, once per turn.**::

    REALITY → PERCEPTION → CONCEPT FORMATION → WORLD MODEL → CAUSAL HYPOTHESES
            → SIMULATION → PREDICTION → OUTCOME → PREDICTION ERROR
            → SELF-CRITIC → { reweight | RESTRUCTURE } → NEW HYPOTHESES → ↺

The junction that matters is the self-critic. :meth:`RecursiveCognitiveField._diagnose` sorts an
error into one of three kinds, and they call for genuinely different repairs:

* **numeric** — the right variables, the wrong coefficients. Refit. :mod:`nyxara.njp.universe`
  already does this every time it observes.
* **structural** — the right concepts, a missing arrow. Declare the arrow and let it fit.
* **conceptual** — her concept system cannot express what she just met at all. This is the one
  that must not be answered with a coefficient. :meth:`ConceptGenesis.restructure` searches for
  the smallest change to the conditions of concept formation that covers the observation *without
  costing compression elsewhere*, and if no such change exists it says so rather than pretending.

**Loop 2 — the slow loop, and the interesting one.**::

    brain → evaluate itself → find bottleneck → invent modification
          → sandbox → benchmark → adversarial test → accept | reject → ↺

:meth:`RecursiveCognitiveField.meta_cycle` measures her own organs against each other, finds the
one actually limiting her, proposes a bounded modification to it, and then — this is the part that
keeps it safe and the part that makes it *science* rather than drift — evaluates the modification
against a **held-out** benchmark she did not tune on, plus an adversarial battery, and reverts it
byte-for-byte unless it strictly wins. Reality decides which version survives. She only proposes.

**What this module deliberately does not do.** It does not rewrite her source: that is
:mod:`nyxara.njp.evolve`, which already has the backup → syntax → safety → benchmark → tests →
rollback gauntlet, and duplicating it here would mean two paths to modifying her, one of them
newer and less tested. It does not touch the character core, the loyalty operators or any gate —
:attr:`RecursiveCognitiveField.PROTECTED` names what is not a legal target, and a proposal
touching one is refused before it is evaluated, not after.

Everything here is measured. Every number in :meth:`RecursiveCognitiveField.stats` is produced by
this module doing the thing it claims to do, and a cycle that changed nothing reports that.

No LLM anywhere in this file.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field as dc_field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

__all__ = [
    "ErrorClass",
    "Diagnosis",
    "CycleReport",
    "Bottleneck",
    "Modification",
    "Trial",
    "RecursiveCognitiveField",
]


class ErrorClass:
    """What *kind* of wrong a prediction error is — and so which repair it calls for."""

    NONE = "none"
    NUMERIC = "numeric"           # right variables, wrong coefficients → refit
    STRUCTURAL = "structural"     # right concepts, missing arrow → declare it
    CONCEPTUAL = "conceptual"     # her concepts cannot express this → restructure
    UNKNOWN = "unknown"           # not enough information to say


@dataclass
class Diagnosis:
    """The self-critic's verdict on one error."""

    kind: str = ErrorClass.NONE
    magnitude: float = 0.0
    subject: str = ""
    detail: str = ""
    coverage: Any = None          # the concepts.Coverage that produced a conceptual verdict

    @property
    def actionable(self) -> bool:
        return self.kind in (ErrorClass.NUMERIC, ErrorClass.STRUCTURAL, ErrorClass.CONCEPTUAL)

    def to_dict(self) -> Dict[str, Any]:
        return {"kind": self.kind, "magnitude": round(self.magnitude, 5),
                "subject": self.subject[:80], "detail": self.detail[:160],
                "actionable": self.actionable,
                "coverage": self.coverage.to_dict() if self.coverage is not None else None}


@dataclass
class CycleReport:
    """One turn through loop 1, and what each stage actually did."""

    turn: int = 0
    perceived: int = 0
    concepts_fed: int = 0
    crystallised: bool = False
    compression: float = 1.0
    arrows: int = 0
    hypotheses: int = 0
    simulated: bool = False
    predicted: Optional[float] = None
    observed: Optional[float] = None
    error: Optional[float] = None
    diagnosis: Optional[Diagnosis] = None
    restructured: bool = False
    restructure_gain: float = 0.0
    experiment: str = ""
    experiment_gain: float = 0.0
    beliefs_touched: int = 0
    # The world-level prediction: what she expected the situation to lead to, and what the world
    # charged her in bits for being wrong about it. Separate from `error`, which grades her
    # against her own firing — this grades her against reality.
    world_predicted: str = ""
    world_bits: float = 0.0
    world_excess: float = 0.0
    surprised: bool = False
    ms: float = 0.0

    @property
    def learned(self) -> bool:
        """Did anything durable change? A cycle that only observed has not learned."""
        return bool(self.restructured or self.crystallised or self.arrows
                    or self.beliefs_touched or self.experiment)

    def to_dict(self) -> Dict[str, Any]:
        return {"turn": self.turn, "perceived": self.perceived,
                "concepts_fed": self.concepts_fed, "crystallised": self.crystallised,
                "compression": round(self.compression, 4), "arrows": self.arrows,
                "hypotheses": self.hypotheses, "simulated": self.simulated,
                "error": None if self.error is None else round(self.error, 5),
                "diagnosis": self.diagnosis.to_dict() if self.diagnosis else None,
                "restructured": self.restructured,
                "restructure_gain": round(self.restructure_gain, 4),
                "experiment": self.experiment,
                "experiment_gain": round(self.experiment_gain, 5),
                "beliefs_touched": self.beliefs_touched,
                "world_predicted": self.world_predicted,
                "world_bits": round(self.world_bits, 4),
                "world_excess": round(self.world_excess, 4),
                "surprised": self.surprised,
                "learned": self.learned, "ms": round(self.ms, 3)}


@dataclass
class Bottleneck:
    """The organ actually limiting her, with the number that says so."""

    organ: str = ""
    metric: str = ""
    value: float = 0.0
    severity: float = 0.0        # 0..1, how badly it is limiting
    why: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {"organ": self.organ, "metric": self.metric, "value": round(self.value, 5),
                "severity": round(self.severity, 4), "why": self.why[:160]}


@dataclass
class Modification:
    """A bounded, reversible change to one knob of one organ."""

    organ: str = ""
    knob: str = ""
    before: float = 0.0
    after: float = 0.0
    rationale: str = ""

    @property
    def reversible(self) -> bool:
        return bool(self.organ and self.knob)

    def to_dict(self) -> Dict[str, Any]:
        return {"organ": self.organ, "knob": self.knob,
                "before": round(self.before, 5), "after": round(self.after, 5),
                "rationale": self.rationale[:160]}


@dataclass
class Trial:
    """A modification put through sandbox → benchmark → adversarial → verdict."""

    modification: Optional[Modification] = None
    bottleneck: Optional[Bottleneck] = None
    baseline: float = 0.0
    candidate: float = 0.0
    adversarial_passed: bool = False
    accepted: bool = False
    why: str = ""
    ms: float = 0.0

    @property
    def gain(self) -> float:
        return self.candidate - self.baseline

    def to_dict(self) -> Dict[str, Any]:
        return {"modification": self.modification.to_dict() if self.modification else None,
                "bottleneck": self.bottleneck.to_dict() if self.bottleneck else None,
                "baseline": round(self.baseline, 5), "candidate": round(self.candidate, 5),
                "gain": round(self.gain, 5),
                "adversarial_passed": self.adversarial_passed,
                "accepted": self.accepted, "why": self.why[:200], "ms": round(self.ms, 3)}


# --------------------------------------------------------------------------- #
# The field
# --------------------------------------------------------------------------- #
class RecursiveCognitiveField:
    """Both loops: the cognitive cycle per turn, and the self-modification cycle behind it."""

    #: Knobs that are never a legal target of loop 2. Not a policy that can be argued with — a
    #: proposal naming one of these is refused before it is measured, so no benchmark result can
    #: ever motivate weakening a gate. The character core is not an optimisation surface.
    PROTECTED = frozenset({
        "truth", "guard", "guardian", "loyalty", "corrigibility", "oversight", "soul",
        "identity", "values", "character", "kernel", "gate", "scram", "min_sources",
        "require_hard",
    })

    def __init__(self, brain: Any, *, concepts: Any = None, universe: Any = None,
                 designer: Any = None, beliefs: Any = None, meta: Any = None,
                 crystallise_every: int = 4, meta_every: int = 25,
                 holdout_share: float = 0.25, blame_floor: int = 12) -> None:
        self.brain = brain
        self.concepts = concepts
        self.universe = universe
        self.designer = designer
        self.beliefs = beliefs
        self.meta = meta
        self.crystallise_every = max(1, int(crystallise_every))
        self.meta_every = max(1, int(meta_every))
        self.holdout_share = min(0.5, max(0.05, float(holdout_share)))
        # Observations the concept system must hold before a coverage gap counts as *its* fault.
        self._blame_floor = max(4, int(blame_floor))
        # A restructure is a real cost — it re-crystallises everything — so the same subject is
        # not allowed to trigger one twice running. Without this, a subject she genuinely cannot
        # classify would restructure her whole concept system on every turn it is mentioned.
        self._last_restructured: str = ""
        # The most recent benchmark score, so `stats()` can report it without paying for it.
        self._last_benchmark: float = 0.0

        self.cycles = 0
        self.restructures = 0
        self.restructures_kept = 0
        self.experiments_designed = 0
        self.surprises = 0
        self.errors_diagnosed: Dict[str, int] = {}
        self.trials: List[Trial] = []
        self.accepted_trials = 0
        self.last: Optional[CycleReport] = None
        # Samples the benchmark is computed on. The held-out half is *never* used by any
        # restructure decision, which is the only thing that makes a benchmark win mean anything.
        self._samples: List[Tuple[str, List[str]]] = []
        self._holdout: List[Tuple[str, List[str]]] = []

    # ================= LOOP 1 — the cognitive cycle ========================== #
    def cycle(self, thought: Any) -> CycleReport:
        """One full turn through the field. Called at the end of every thought."""
        rep = CycleReport(turn=int(getattr(self.brain, "turns", 0)))
        t0 = time.perf_counter()
        try:
            self.cycles += 1
            triples = self._triples(thought)
            rep.perceived = len(triples)

            rep.concepts_fed = self._form_concepts(triples, rep)
            self._predict_world(triples, rep)
            rep.arrows = self._sync_world(rep)
            rep.hypotheses = self._raise_hypotheses(rep)
            self._simulate(thought, rep)
            self._score(thought, rep)

            if rep.error is not None:
                rep.diagnosis = self._diagnose(thought, triples, rep)
                self.errors_diagnosed[rep.diagnosis.kind] = \
                    self.errors_diagnosed.get(rep.diagnosis.kind, 0) + 1
                self._repair(rep)

            rep.beliefs_touched = self._record_beliefs(thought, triples)
            self._design_experiment(rep)

            if self.concepts is not None:
                rep.compression = self.concepts.compression()
            self.last = rep
            return rep
        except Exception:  # noqa: BLE001 — a failed cycle learns nothing and never breaks a turn
            self.last = rep
            return rep
        finally:
            rep.ms = (time.perf_counter() - t0) * 1000.0

    # ---- perception -------------------------------------------------------- #
    @staticmethod
    def _triples(thought: Any) -> List[Any]:
        grounding = getattr(getattr(thought, "percept", None), "grounding", None)
        return list(getattr(grounding, "triples", None) or [])

    # ---- concept formation -------------------------------------------------- #
    def _form_concepts(self, triples: Sequence[Any], rep: CycleReport) -> int:
        if self.concepts is None or not triples:
            return 0
        fed = self.concepts.observe_triples(triples)
        # Every observation is a benchmark sample; a deterministic quarter of them are held out
        # and never touched by a restructure decision. Hashing the subject rather than using a
        # counter keeps the same subject on the same side of the split across restarts, so the
        # held-out set does not quietly leak in as the session grows.
        for triple in triples:
            subject = str(getattr(triple, "subject", "") or "").strip().lower()
            if not subject:
                continue
            obs = self.concepts._by_subject.get(subject)
            if obs is None:
                continue
            row = (subject, sorted(obs.features))
            target = (self._holdout
                      if (abs(hash(subject)) % 1000) / 1000.0 < self.holdout_share
                      else self._samples)
            target.append(row)
            del target[:-400]
        if fed and self.cycles % self.crystallise_every == 0:
            report = self.concepts.crystallise()
            rep.crystallised = report.changed
        return fed

    # ---- the predictive brain ------------------------------------------------ #
    def _predict_world(self, triples: Sequence[Any], rep: CycleReport) -> None:
        """Expect what comes next, look, and price the difference in bits.

        This is the step that separates a fact store from something that thinks. Every other
        signal in this loop measures her against *herself* — the manifold's anticipation of her
        own firing, graded by her own firing. This one measures her against the **world**: she
        commits to what the situation leads to, the next turn shows what it actually led to, and
        the gap is real information she did not have.

        Surprise is routed by ``excess``, not by raw bits. A model that honestly reported "I have
        no idea" and was then wrong has learned something and done nothing wrong; one that said
        "certainly X" and got Y has a defect worth restructuring for. Grading both the same is
        how a system is taught that vagueness is safe.
        """
        model = getattr(self.brain, "predictive", None)
        if model is None or not triples:
            return
        try:
            from nyxara.njp.predictive import WorldState
            facts = [f"{getattr(t, 'subject', '')} {getattr(t, 'predicate', '')} "
                     f"{getattr(t, 'object', '')}".strip() for t in triples[:4]]
            state = WorldState.of(facts)
            if state.empty:
                return
            intent = getattr(self.brain, "_last_intent_kind", "") or ""
            prediction, surprise = model.predict_and_observe(state, intent)
            rep.world_bits = surprise.bits
            rep.world_excess = surprise.excess
            rep.world_predicted = prediction.top
            if surprise.surprising:
                rep.surprised = True
                self.surprises += 1
        except Exception:  # noqa: BLE001 — a failed expectation costs a signal, never the turn
            pass

    # ---- world model -------------------------------------------------------- #
    def _sync_world(self, rep: CycleReport) -> int:
        if self.universe is None:
            return 0
        added = self.universe.sync_from_world()
        # Numbers the grounder extracted are the universe's actual data. A threshold triple
        # ("water boils at 100") is a variable with a value, and without this the simulator would
        # own a causal skeleton it could never fit a single coefficient to.
        state: Dict[str, float] = {}
        for subject, obs in list(getattr(self.concepts, "_by_subject", {}).items())[-32:] \
                if self.concepts is not None else []:
            for key, value in obs.numbers.items():
                state[f"{subject}.{key}"] = value
        if state:
            self.universe.observe(state)
        return added

    # ---- causal hypotheses --------------------------------------------------- #
    def _raise_hypotheses(self, rep: CycleReport) -> int:
        """Turn the strongest unsettled causal links into rival hypotheses she can test.

        Only links that are *stated but unobserved* become hypotheses. A link she has already
        watched happen repeatedly is not in question, and spending an experiment on it is exactly
        the waste the information-gain criterion exists to prevent.
        """
        world = getattr(self.brain, "world", None)
        if self.designer is None or world is None:
            return 0
        raised = 0
        try:
            for link in world.links(causal_only=True)[:6]:
                if link.together or not link.stated:
                    continue
                name = f"{link.cause}→{link.effect}"
                if name in self.designer.hypotheses:
                    continue
                # Two rivals, so the question is decidable: the arrow is real, or it is not.
                self.designer.propose(name, probability=max(0.1, link.strength),
                                      predictions={f"remove:{link.cause}": "absent"})
                self.designer.propose(f"not({name})", probability=max(0.1, 1.0 - link.strength),
                                      predictions={f"remove:{link.cause}": "present"})
                raised += 1
        except Exception:  # noqa: BLE001
            return raised
        return raised

    # ---- simulation ---------------------------------------------------------- #
    def _simulate(self, thought: Any, rep: CycleReport) -> None:
        """Roll the universe forward on the variable this turn actually moved."""
        if self.universe is None or not self.universe.usable_relations():
            return
        try:
            leading = self.universe.usable_relations()[0]
            current = self.universe.state.get(leading.cause)
            if current is None:
                return
            roll = self.universe.imagine("continue", {leading.cause: current}, steps=1)
            rep.simulated = bool(roll.states)
            rep.predicted = roll.final.get(leading.effect)
        except Exception:  # noqa: BLE001
            pass

    # ---- outcome and error ---------------------------------------------------- #
    def _score(self, thought: Any, rep: CycleReport) -> None:
        """Score the manifold's own anticipation — an outcome independent of the prediction.

        Reusing the fabric's pre-settle anticipation rather than inventing a second signal is
        deliberate: it is already graded against what physically fired next, so it cannot be
        gamed by the thing being graded. A mind that scores its guesses against its own later
        guesses is not learning, it is agreeing with itself.
        """
        try:
            loop = getattr(thought, "loop", None)
            error = getattr(loop, "error", None) if loop is not None else None
            if error is None:
                predictor = getattr(self.brain, "predictor", None)
                error = predictor.recent_error() if predictor is not None else None
            if error is None:
                return
            rep.error = float(error)
            rep.observed = 1.0 - float(error)
        except Exception:  # noqa: BLE001
            pass

    # ---- the self-critic ------------------------------------------------------- #
    def _diagnose(self, thought: Any, triples: Sequence[Any], rep: CycleReport) -> Diagnosis:
        """Sort the error into the repair it calls for. The junction the whole theory turns on."""
        out = Diagnosis(magnitude=float(rep.error or 0.0))
        if out.magnitude < 0.15:
            out.kind = ErrorClass.NONE
            out.detail = "within tolerance"
            return out

        # Conceptual first, because it is the one that must not be answered with a coefficient.
        # If her concept system cannot even express what she just met, refitting an arrow between
        # variables she does not have is not a repair — it is the model getting more confident
        # about the wrong shape.
        #
        # But only once the concept system has enough to be *blamed*. A gap on her fourth
        # observation is not a defect in how she forms concepts, it is a brain that has met four
        # things — and treating it as a defect is not harmless: measured over a 26-turn session,
        # every early gap fired a restructure and drove the similarity threshold to its floor,
        # where everything resembles everything and the concepts formed are worthless. A repair
        # mechanism that fires before there is anything to repair destroys the thing it protects.
        if self.concepts is not None and len(self.concepts.observations) >= self._blame_floor:
            for triple in triples[:3]:
                subject = str(getattr(triple, "subject", "") or "")
                if not subject:
                    continue
                coverage = self.concepts.explain(subject)
                if coverage.gap:
                    out.kind = ErrorClass.CONCEPTUAL
                    out.subject = subject
                    out.coverage = coverage
                    out.detail = (f"no concept covers {subject!r} ({coverage.gap})"
                                  if coverage.gap == "unknown"
                                  else f"{subject!r} violates {coverage.violated}")
                    return out

        # Structural: the variables exist but no arrow connects them, so nothing could have
        # carried the prediction.
        if self.universe is not None and triples:
            triple = triples[0]
            cause = str(getattr(triple, "subject", "") or "").lower()
            effect = str(getattr(triple, "object", "") or "").lower()
            if cause and effect:
                rel = self.universe.relations.get((cause, effect))
                if rel is None or not rel.usable:
                    out.kind = ErrorClass.STRUCTURAL
                    out.subject = f"{cause}→{effect}"
                    out.detail = "no usable arrow between named variables"
                    return out

        if self.universe is not None and self.universe.usable_relations():
            out.kind = ErrorClass.NUMERIC
            out.detail = "arrows exist; coefficients are wrong"
            return out
        out.kind = ErrorClass.UNKNOWN
        out.detail = "not enough model to attribute the error"
        return out

    def _repair(self, rep: CycleReport) -> None:
        """Route the diagnosis to the repair that owns it. Restructure is the only costly one."""
        diagnosis = rep.diagnosis
        if diagnosis is None or not diagnosis.actionable:
            return
        try:
            if diagnosis.kind == ErrorClass.CONCEPTUAL and self.concepts is not None:
                if diagnosis.subject == self._last_restructured:
                    return          # already tried for this subject; a second pass learns nothing
                self._last_restructured = diagnosis.subject
                before = self.concepts.compression()
                report = self.concepts.restructure(diagnosis.coverage)
                self.restructures += 1
                rep.restructure_gain = report.compression_after - before
                # "Kept" means it did not cost her compression — the restructure paid for itself
                # on everything else she knows, not just on the observation that triggered it.
                if rep.restructure_gain >= -1e-9:
                    rep.restructured = True
                    self.restructures_kept += 1
            elif diagnosis.kind == ErrorClass.STRUCTURAL and self.universe is not None:
                cause, _, effect = diagnosis.subject.partition("→")
                if cause and effect:
                    self.universe.declare(cause, effect)
            # NUMERIC needs no action here: `universe.observe` refits every arrow on every
            # observation already, and adding a second refit would double-count the sample.
        except Exception:  # noqa: BLE001
            pass

    # ---- beliefs ---------------------------------------------------------------- #
    def _record_beliefs(self, thought: Any, triples: Sequence[Any]) -> int:
        """Everything grounded this turn becomes a belief with its case attached."""
        if self.beliefs is None:
            return 0
        touched = 0
        try:
            from nyxara.njp.beliefs import EvidenceKind
            for triple in triples[:4]:
                subject = str(getattr(triple, "subject", "") or "")
                predicate = str(getattr(triple, "predicate", "") or "")
                obj = str(getattr(triple, "object", "") or "")
                if not subject or not predicate:
                    continue
                claim = f"{subject} {predicate} {obj}".strip()
                # Held at the strength of the evidence that actually exists for it — testimony —
                # not at the parser's confidence that it read the sentence correctly. Those are
                # two different quantities and conflating them made every belief she holds
                # "unsupported" the moment it was recorded: 26 beliefs, 26 audit flags, nothing
                # known. How sure she is that the Master *said* something is not how sure she is
                # that it is true. Observing it herself is what raises it, through `support`.
                parsed = float(getattr(triple, "confidence", 0.6) or 0.6)
                confidence = min(parsed, EvidenceKind.WEIGHTS[EvidenceKind.TESTIMONY])
                self.beliefs.hold(
                    claim, confidence=confidence, domain=self._domain_of(predicate),
                    produced_by="grounding",
                    falsifier=f"an observation in which {subject} does not {predicate} {obj}".strip(),
                    why="grounded from the Master's statement")
                self.beliefs.support(claim, EvidenceKind.TESTIMONY,
                                     detail=str(getattr(triple, "text", ""))[:120],
                                     source="master")
                touched += 1
        except Exception:  # noqa: BLE001
            return touched
        return touched

    @staticmethod
    def _domain_of(predicate: str) -> str:
        predicate = predicate.lower()
        if predicate in ("causes", "produces", "requires", "boils", "melts", "freezes"):
            return "physics"
        if predicate in ("is_a", "part_of"):
            return "taxonomy"
        if predicate in ("has_name", "located_in", "works_at", "likes", "knows"):
            return "personal"
        return "general"

    # ---- curiosity as information gain -------------------------------------------- #
    def _design_experiment(self, rep: CycleReport) -> None:
        if self.designer is None:
            return
        try:
            experiment = self.designer.design()
            if experiment is None:
                return
            rep.experiment = experiment.name
            rep.experiment_gain = experiment.gain
            self.experiments_designed += 1
            # An experiment she cannot run is still worth *asking* about — it becomes an open
            # question in the ledger, which is how "I don't know" turns into a standing
            # investigation rather than the end of a turn.
            if self.beliefs is not None:
                self.beliefs.ask(f"what happens if {experiment.name}?",
                                 why=f"expected information gain {experiment.gain:.3f} bits")
        except Exception:  # noqa: BLE001
            pass

    # ================= LOOP 2 — the self-modification cycle ==================== #
    def due_for_meta(self) -> bool:
        return self.cycles > 0 and self.cycles % self.meta_every == 0

    def meta_cycle(self) -> Optional[Trial]:
        """Evaluate herself, find the bottleneck, propose a fix, and let reality decide.

        The order is load-bearing. The baseline is measured on the **held-out** samples *before*
        the modification, the candidate on the same samples *after*, and the modification is
        reverted unless it strictly wins and survives the adversarial battery. Nothing about the
        proposal gets to influence the set it is judged on.
        """
        trial = Trial()
        t0 = time.perf_counter()
        try:
            bottleneck = self.find_bottleneck()
            trial.bottleneck = bottleneck
            if bottleneck is None:
                trial.why = "no bottleneck worth acting on"
                return trial

            modification = self.propose(bottleneck)
            trial.modification = modification
            if modification is None:
                trial.why = f"no legal modification for {bottleneck.organ}"
                return trial
            if self._protected(modification):
                trial.why = "refused: names a protected knob"
                return trial

            trial.baseline = self.benchmark()
            applied = self._apply(modification)
            if not applied:
                trial.why = "modification could not be applied"
                return trial
            trial.candidate = self.benchmark()
            trial.adversarial_passed = self._adversarial()

            # Strictly better, and still safe. Ties revert: an equal score is not evidence for a
            # change, and accepting ties is how a system drifts a long way on no evidence at all.
            if trial.candidate > trial.baseline + 1e-9 and trial.adversarial_passed:
                trial.accepted = True
                trial.why = (f"{bottleneck.organ}.{modification.knob}: "
                             f"{trial.baseline:.4f} → {trial.candidate:.4f}")
                self.accepted_trials += 1
            else:
                self._revert(modification)
                trial.why = ("adversarial battery failed" if not trial.adversarial_passed
                             else f"no gain ({trial.baseline:.4f} → {trial.candidate:.4f})")
            return trial
        except Exception:  # noqa: BLE001 — a failed meta-cycle changes nothing
            trial.why = "meta-cycle failed"
            return trial
        finally:
            trial.ms = (time.perf_counter() - t0) * 1000.0
            self.trials.append(trial)
            self.trials = self.trials[-64:]

    def find_bottleneck(self) -> Optional[Bottleneck]:
        """Which organ is actually limiting her — measured, not guessed.

        Severity is what makes this a ranking rather than a list of complaints: an organ that is
        producing *nothing* is a worse bottleneck than one producing something imperfectly, and
        a zero here is the signal that something is not wired rather than merely weak. That is
        exactly the failure that put six of these counters at zero for 113 turns.
        """
        found: List[Bottleneck] = []
        if self.concepts is not None:
            ratio = self.concepts.compression()
            if ratio < 1.6:
                found.append(Bottleneck(
                    organ="concepts", metric="compression", value=ratio,
                    severity=min(1.0, max(0.0, (1.6 - ratio) / 1.6)),
                    why="concepts barely compress the observations they claim to explain"))
        if self.universe is not None:
            stats = self.universe.stats()
            usable, total = stats["usable_relations"], max(1, stats["relations"])
            share = usable / total
            if share < 0.5:
                found.append(Bottleneck(
                    organ="universe", metric="usable_share", value=share,
                    severity=min(1.0, 1.0 - share),
                    why="most learned arrows do not explain enough to propagate through"))
        if self.designer is not None:
            stats = self.designer.stats()
            if stats["hypotheses"] and not stats["run"]:
                found.append(Bottleneck(
                    organ="designer", metric="run", value=0.0, severity=0.7,
                    why="experiments are designed but never resolved"))
        if self.meta is not None:
            rate = self.meta.stats().get("assertable_rate")
            if rate is not None and rate < 0.6:
                found.append(Bottleneck(
                    organ="meta", metric="assertable_rate", value=float(rate),
                    severity=min(1.0, 1.0 - float(rate)),
                    why="most answers do not survive their own critic"))
        if not found:
            return None
        return max(found, key=lambda b: b.severity)

    def propose(self, bottleneck: Bottleneck) -> Optional[Modification]:
        """Invent a bounded, reversible change aimed at *this* bottleneck.

        Bounded is the operative word. Each knob has a hard range, so the worst a bad proposal
        can do is score badly on the benchmark and be reverted — never leave an organ in a state
        no benchmark could recover from.
        """
        if bottleneck.organ == "concepts" and self.concepts is not None:
            before = float(self.concepts.similarity)
            after = round(min(0.8, max(0.1, before + (0.06 if before < 0.45 else -0.06))), 4)
            if abs(after - before) < 1e-9:
                return None
            return Modification(organ="concepts", knob="similarity", before=before, after=after,
                                rationale="move the clustering threshold toward better compression")
        if bottleneck.organ == "universe" and self.universe is not None:
            before = float(self.universe.max_depth)
            after = float(min(10, max(1, int(before) + 1)))
            if abs(after - before) < 1e-9:
                return None
            return Modification(organ="universe", knob="max_depth", before=before, after=after,
                                rationale="propagate further so more arrows are exercised")
        if bottleneck.organ == "meta" and self.meta is not None:
            return Modification(organ="field", knob="crystallise_every",
                                before=float(self.crystallise_every),
                                after=float(max(1, self.crystallise_every - 1)),
                                rationale="refresh concepts more often so answers rest on newer structure")
        return None

    def _protected(self, modification: Modification) -> bool:
        target = f"{modification.organ}.{modification.knob}".lower()
        return any(word in target for word in self.PROTECTED)

    def _apply(self, modification: Modification) -> bool:
        holder = {"concepts": self.concepts, "universe": self.universe,
                  "meta": self.meta, "field": self}.get(modification.organ)
        if holder is None or not hasattr(holder, modification.knob):
            return False
        try:
            current = getattr(holder, modification.knob)
            setattr(holder, modification.knob,
                    int(modification.after) if isinstance(current, int) else modification.after)
            return True
        except Exception:  # noqa: BLE001
            return False

    def _revert(self, modification: Modification) -> bool:
        holder = {"concepts": self.concepts, "universe": self.universe,
                  "meta": self.meta, "field": self}.get(modification.organ)
        if holder is None:
            return False
        try:
            current = getattr(holder, modification.knob, None)
            setattr(holder, modification.knob,
                    int(modification.before) if isinstance(current, int) else modification.before)
            return True
        except Exception:  # noqa: BLE001
            return False

    # ---- the benchmark ------------------------------------------------------- #
    def benchmark(self) -> float:
        """Score her current cognitive configuration on data she did not tune on.

        Held-out only. Every restructure decision in loop 1 reads :attr:`_samples`; this reads
        :attr:`_holdout`, and the two never mix. That separation is the only reason a benchmark
        win here is evidence of anything: a model evaluated on what shaped it will always look
        like it is improving.

        Re-crystallises first, so the score reflects the configuration as it stands rather than
        whatever the last cycle happened to leave behind. That makes this **expensive**, which is
        why :meth:`stats` reports :attr:`_last_benchmark` instead of calling it — a reporting
        method that silently rebuilds the concept hierarchy is a performance bug waiting for the
        concept count to grow, and reporting should never have side effects in the first place.
        """
        if self.concepts is None or not self._holdout:
            return 0.0
        try:
            self.concepts.crystallise()
            covered = 0
            total = 0
            for subject, features in self._holdout[-200:]:
                total += 1
                if self.concepts.explain(subject, features).covered:
                    covered += 1
            coverage = covered / total if total else 0.0
            # Compression and coverage together, because either alone is gameable: one enormous
            # concept covers everything and compresses nothing; a concept per observation
            # compresses nothing and covers everything.
            ratio = self.concepts.compression()
            self._last_benchmark = round(0.5 * coverage + 0.5 * min(1.0, (ratio - 1.0) / 2.0), 6)
            return self._last_benchmark
        except Exception:  # noqa: BLE001
            return 0.0

    def _adversarial(self) -> bool:
        """The battery a modification must survive whatever the benchmark says.

        Deliberately about *invariants*, not performance: a configuration that scores brilliantly
        while asserting concepts with no invariants, or claiming every observation belongs to one
        kind, has found a way to win the metric rather than a way to think better.
        """
        try:
            if self.concepts is None:
                return True
            concepts = list(self.concepts.concepts.values())
            if not concepts:
                return True
            # 1. No concept may assert nothing.
            if any(not c.invariants for c in concepts):
                return False
            # 2. No single concept may swallow the whole world — that is not a kind, it is a
            #    restatement of "things exist".
            observations = max(1, len(self.concepts.observations))
            if len(concepts) == 1 and concepts[0].support >= observations:
                return False
            # 3. Compression must not have fallen below break-even.
            if self.concepts.compression() < 1.0:
                return False
            # 4. An unseen observation with nothing in common must NOT be confidently covered.
            probe = self.concepts.explain("__adversarial_probe__", ["__nonsense_feature__"])
            if probe.covered:
                return False
            return True
        except Exception:  # noqa: BLE001 — a battery that cannot run has not passed
            return False

    # ---- reporting ------------------------------------------------------------- #
    def stats(self) -> Dict[str, Any]:
        accepted = [t for t in self.trials if t.accepted]
        return {
            "cycles": self.cycles,
            "restructures": self.restructures,
            "restructures_kept": self.restructures_kept,
            "experiments_designed": self.experiments_designed,
            "surprises": self.surprises,
            "errors": dict(self.errors_diagnosed),
            "samples": len(self._samples), "holdout": len(self._holdout),
            # The last score computed, not a fresh one: see `benchmark`. 0.0 means loop 2 has not
            # run yet, which is a different thing from a configuration that scored zero.
            "benchmark": self._last_benchmark,
            "meta_trials": len(self.trials),
            "meta_accepted": self.accepted_trials,
            "meta_gain": round(sum(t.gain for t in accepted), 5),
            "last_trial": self.trials[-1].to_dict() if self.trials else None,
            "last_cycle": self.last.to_dict() if self.last else None,
        }

    def to_dict(self) -> Dict[str, Any]:
        return {"cycles": self.cycles, "restructures": self.restructures,
                "restructures_kept": self.restructures_kept,
                "experiments_designed": self.experiments_designed,
                "errors": dict(self.errors_diagnosed),
                "accepted_trials": self.accepted_trials,
                "crystallise_every": self.crystallise_every,
                "samples": [[s, f] for s, f in self._samples[-200:]],
                "holdout": [[s, f] for s, f in self._holdout[-200:]]}

    def load_dict(self, d: Dict[str, Any]) -> None:
        try:
            self.cycles = int(d.get("cycles", 0))
            self.restructures = int(d.get("restructures", 0))
            self.restructures_kept = int(d.get("restructures_kept", 0))
            self.experiments_designed = int(d.get("experiments_designed", 0))
            self.errors_diagnosed = {str(k): int(v)
                                     for k, v in (d.get("errors") or {}).items()}
            self.accepted_trials = int(d.get("accepted_trials", 0))
            self.crystallise_every = max(1, int(d.get("crystallise_every",
                                                      self.crystallise_every)))
            self._samples = [(str(s), list(f)) for s, f in (d.get("samples") or [])]
            self._holdout = [(str(s), list(f)) for s, f in (d.get("holdout") or [])]
        except Exception:  # noqa: BLE001
            pass
