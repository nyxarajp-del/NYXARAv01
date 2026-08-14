"""NYXARA · njp/integrate.py — the loop that makes the organs learn from each other (🔄, NJP V.04).

Every organ NJP needs already existed before this module. The world model could take events, the
predictor could score an expectation, the levels could consolidate, the discoverer could test an
abstraction, the self-model could take an observation, the readout could take a gradient step.
Measured over a 113-turn session, here is what they had actually done::

    world:     events = 0        observations = 0
    predict:   predictions = 113 scored = 0        correct = 0
    levels:    consolidations = 0 promoted = 0     forgotten = 0
    discover:  passes = 0        proposed = 0      confirmed = 0
    curiosity: passes = 0        raised = 0
    selfmodel: every capability untested, observations = 0
    readout:   steps = 0

Not one of those is a missing algorithm. Every one of them is a **method that nothing ever
called**. The organs were installed and wired to a caller that did not exist: the slow half of
cognition hung off :class:`~nyxara.njp.pulse.PulseEngine`, whose beats are driven by the kernel's
clock, so a brain used the way a brain is actually used — ``think()``, turn after turn — ran
perception and grew the fabric and never once closed the loop behind it.

This module is that caller. It is the difference between *"the organs are present"* and *"the
organs are learning from each other"*, and the only evidence that it works is that the counters
above stop being zero on a real session.

**The cycle, in the order it runs**::

    INPUT → GROUNDING → WORLD STATE → PREDICTION → OBSERVATION
          → ERROR → DIAGNOSIS → CORRECTION → MEMORY → ABSTRACTION → NEW PREDICTION

**Where the outcomes come from, and why they are not circular.** Scoring a prediction against
your own later opinion is not learning, it is self-congratulation, and it is the easiest way to
make these counters move dishonestly. So every outcome here is independent of the prediction it
scores:

* **next-state** — the manifold anticipates which cells will fire *before* the fabric settles, and
  the settle then says which actually did. The prediction is computed from ``W`` at time ``t`` and
  the observation is physics at ``t+1``; nothing about the answer she gives can influence it. This
  is the self-supervised signal that runs on every single turn.
* **deferred answers** — when she is asked something and does not know, that "<unknown>" is left
  **open**. It is scored later, if and only if the Master *states* the fact, and the truth then
  comes from his sentence rather than from her inference. A question she was never told the answer
  to stays open forever and is never counted, which is why :attr:`LoopReport.deferred_open` is
  reported alongside the resolved ones.
* **corrections** — a fact that supersedes an earlier one means the earlier answer was wrong, and
  the correction is the Master's, not hers.

**Cadence, and why it is counted in turns.** Consolidation, abstraction discovery and curiosity
are slow organs: running them every turn is waste, and running them on a wall clock means they
never run at all in the batch and CLI paths where nothing beats. They run here on **turn counts**,
so a session of ``n`` turns does a predictable amount of slow work whether it took a second or a
week. The pulse keeps its own wall-clock cadence for experience arriving *outside* a turn; the two
do not double-count, because the pulse's queue is fed by ``remember``, not by ``think``.

Honest, as everywhere in this repo:

* **This adds no new intelligence.** Every algorithm it invokes was already written and already
  tested. What was missing was the wiring, and wiring is what this is. Anyone reading it hoping to
  find a new kind of reasoning will find a scheduler.
* **A closed loop is not a smart loop.** ``scored > 0`` means she is now measuring herself. It does
  not mean she is right, and :attr:`LoopReport.correct` is reported next to it precisely so the
  difference stays visible.
* **Fail-soft throughout.** Every step is independently guarded: an organ that raises is skipped
  for the turn and recorded as having contributed nothing, because a learning loop that can break
  a reply is worse than one that occasionally learns nothing.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

__all__ = ["LoopReport", "LearningLoop"]

# How often the slow organs run, counted in turns. Chosen so a short session still exercises each
# of them at least once — a cadence that never fires inside a normal conversation is the bug this
# module exists to fix, and setting these to "every N minutes" would reintroduce it exactly.
_CONSOLIDATE_EVERY = 8
_DISCOVER_EVERY = 12
_WONDER_EVERY = 16

# Width of the numeric state handed to the dynamics model. Small on purpose: this is a coarse
# summary of which regions of the fabric were active, not a reconstruction of it, and a kNN model
# over a 10,000-D one-hot would be a distance computation with no neighbours.
_STATE_WIDTH = 16

# A prediction is "correct enough" for a capability observation at this similarity. Not the same
# number as `Outcome.correct` uses — that one gates the accuracy statistic, this one grades a
# capability, and a capability graded pass/fail on exact match learns nothing from a near miss.
_PARTIAL_CREDIT = 0.5


@dataclass
class LoopReport:
    """What one closing of the loop actually did. Every field is a real count."""

    turn: int = 0
    # world
    events: int = 0
    transitions: int = 0
    # prediction
    scored: int = 0
    correct: int = 0
    deferred_opened: int = 0
    deferred_resolved: int = 0
    deferred_open: int = 0
    corrections: int = 0
    # error → repair
    diagnoses: Dict[str, int] = field(default_factory=dict)
    repaired: int = 0
    # learning
    capabilities: int = 0
    trained: bool = False
    loss_before: Optional[float] = None
    loss_after: Optional[float] = None
    # slow organs
    consolidated: bool = False
    promoted: int = 0
    forgotten: int = 0
    discovered: bool = False
    proposed: int = 0
    confirmed: int = 0
    refuted: int = 0
    wondered: int = 0
    ms: float = 0.0

    @property
    def learned(self) -> bool:
        """Did anything at all change as a result of this turn?

        The single claim this module has to earn. False is a legitimate answer — a turn that
        grounded nothing and predicted nothing has nothing to learn from, and reporting that
        honestly is the point of having the flag.
        """
        return bool(self.events or self.transitions or self.scored or self.deferred_resolved
                    or self.capabilities or self.trained or self.consolidated
                    or self.discovered or self.wondered)

    def to_dict(self) -> Dict[str, Any]:
        return {"turn": self.turn, "events": self.events, "transitions": self.transitions,
                "scored": self.scored, "correct": self.correct,
                "deferred": {"opened": self.deferred_opened, "resolved": self.deferred_resolved,
                             "open": self.deferred_open, "corrections": self.corrections},
                "diagnoses": dict(self.diagnoses), "repaired": self.repaired,
                "capabilities": self.capabilities,
                "trained": self.trained,
                "loss": ([round(self.loss_before, 6), round(self.loss_after, 6)]
                         if self.loss_before is not None and self.loss_after is not None
                         else None),
                "consolidated": self.consolidated, "promoted": self.promoted,
                "forgotten": self.forgotten,
                "discovered": self.discovered, "proposed": self.proposed,
                "confirmed": self.confirmed, "refuted": self.refuted,
                "wondered": self.wondered, "learned": self.learned,
                "ms": round(self.ms, 3)}


@dataclass
class _Deferred:
    """A question she could not answer, kept open until the world says what the answer was."""

    key: str = ""                 # the predictor key — the question as asked
    subject: str = ""
    predicate: str = ""
    asked_at: int = 0
    answered: str = ""            # what she said at the time ("" when she abstained)


class LearningLoop:
    """Runs the closing half of a turn: score, diagnose, repair, consolidate, abstract.

    Holds no cognitive state of its own beyond what is needed to connect one turn to the next —
    the previous turn's firing pattern, and the questions still waiting on an answer. Everything
    it learns is written into the organs that own it, so removing this class loses the *loop* and
    not the knowledge.
    """

    def __init__(self, brain: Any, *, consolidate_every: int = _CONSOLIDATE_EVERY,
                 discover_every: int = _DISCOVER_EVERY, wonder_every: int = _WONDER_EVERY,
                 train: bool = True, defer_capacity: int = 256) -> None:
        self.brain = brain
        self.consolidate_every = max(1, int(consolidate_every))
        self.discover_every = max(1, int(discover_every))
        self.wonder_every = max(1, int(wonder_every))
        self.train_enabled = bool(train)
        self.defer_capacity = max(8, int(defer_capacity))

        self.closes = 0
        self.last: Optional[LoopReport] = None
        self.totals: Dict[str, int] = {
            "events": 0, "transitions": 0, "scored": 0, "correct": 0,
            "deferred_opened": 0, "deferred_resolved": 0, "corrections": 0,
            "repaired": 0, "capabilities": 0, "train_steps": 0,
            "consolidations": 0, "discoveries": 0, "wonders": 0,
        }

        # Turn-to-turn carry. The readout learns "what fired then → what fires now", which needs
        # exactly one turn of history and no more.
        self._prev_fired: Tuple[int, ...] = ()
        self._prev_state: List[float] = []
        # The manifold snapshots either side of the current turn. Kept here rather than read off
        # the fabric because the fabric overwrites its own "previous" during the settle.
        self._prev_snapshot: Any = None
        self._snapshot: Any = None
        self._deferred: Dict[str, _Deferred] = {}

        self._install_repairs()

    # ---- repairs ---------------------------------------------------------- #
    def _install_repairs(self) -> None:
        """Route each kind of error to the organ that owns it.

        :mod:`nyxara.njp.predict` already classifies a miss by kind and already looks for a repair
        registered against that kind — and nothing had ever registered one, so every diagnosis
        ended at ``repaired = False``. These are the updates that make a diagnosis worth making.
        """
        try:
            predictor = getattr(self.brain, "predictor", None)
            if predictor is None:
                return
            from nyxara.njp.predict import ErrorKind

            predictor.register_repair(ErrorKind.PERCEPTION, self._repair_perception)
            predictor.register_repair(ErrorKind.GROUNDING, self._repair_grounding)
            predictor.register_repair(ErrorKind.MEMORY, self._repair_memory)
            predictor.register_repair(ErrorKind.WORLD_MODEL, self._repair_world)
            predictor.register_repair(ErrorKind.REASONING, self._repair_reasoning)
        except Exception:  # noqa: BLE001 — an unrepairable brain still runs, it just learns less
            pass

    def _repair_perception(self, outcome: Any) -> bool:
        """Nothing reached her. Widen the fabric's capacity to represent this kind of input."""
        try:
            fabric = getattr(self.brain, "fabric", None)
            if fabric is None:
                return False
            # Neurogenesis is the fabric's own answer to "I cannot represent what I am meeting",
            # and a perception miss is exactly that evidence. Recording it is enough — `expand`
            # reads the error window and mints cells when it stays high.
            fabric.note_error(float(getattr(outcome, "error", 1.0) or 1.0))
            return True
        except Exception:  # noqa: BLE001
            return False

    def _repair_grounding(self, outcome: Any) -> bool:
        """A statement of fact extracted nothing. Re-parse it with the fluent surface allowed."""
        try:
            grounder = getattr(self.brain, "grounder", None)
            text = str((getattr(outcome, "context", None) or {}).get("stimulus", "") or "")
            if grounder is None or not text:
                return False
            # `deep` is the slow path and this is the one place it is worth paying for: the
            # deterministic core has already failed on this exact sentence, so asking again
            # cheaply would only fail again. A successful deep parse also teaches a pattern.
            result = grounder.ground(text, deep=True)
            return bool(getattr(result, "triples", None))
        except Exception:  # noqa: BLE001
            return False

    def _repair_memory(self, outcome: Any) -> bool:
        """The answer was on record and was not retrieved. Strengthen that entry's reachability."""
        try:
            levels = getattr(self.brain, "levels", None)
            if levels is None:
                return False
            expected = str(getattr(outcome, "expected", "") or "")
            hit = False
            for entry in levels.at("semantic") + levels.at("episodic"):
                if expected and expected.lower() in str(getattr(entry, "text", "")).lower():
                    levels.touch(getattr(entry, "key", ""))
                    hit = True
            return hit
        except Exception:  # noqa: BLE001
            return False

    def _repair_world(self, outcome: Any) -> bool:
        """A miss on what-follows-what. Replay the missed transition into the manifold.

        Deliberately **not** ``world.observe_transition``: :meth:`_observe_transition` already
        hands the dynamics model this exact ``(before, action, after)`` on every turn, so doing it
        again here would record one real transition twice and report the duplicate as a repair.
        Measured when it did: 202 transitions logged for 113 turns, 90 of them the same pairs
        relabelled ``correction``. A repair that inflates a counter is worse than no repair,
        because it looks like learning.

        What is genuinely additive is *error-weighted replay*. The settle taught the manifold this
        pair once, along with every other pair. Teaching it again — and only when the prediction
        actually missed — spends the extra Hebbian step on the transitions she got wrong instead
        of spreading it evenly over the ones she already had right.

        The two snapshots are the loop's own, not the fabric's. ``Fabric.settle`` assigns
        ``self._prev_snapshot = out.snapshot`` as its last act, so by the time a repair runs the
        fabric's "previous" and "current" are the same object and every replay was silently a
        no-op — the repair reported success on 90 misses and taught the manifold nothing.
        """
        try:
            manifold = getattr(getattr(self.brain, "fabric", None), "manifold", None)
            before, after = self._prev_snapshot, self._snapshot
            if manifold is None or before is None or after is None or before is after:
                return False
            manifold.learn_transition(before, after)
            return True
        except Exception:  # noqa: BLE001
            return False

    def _repair_reasoning(self, outcome: Any) -> bool:
        """Facts right, conclusion wrong. Open it as a problem so the next pass goes deeper."""
        try:
            reasoner = getattr(self.brain, "reasoner", None)
            if reasoner is None:
                return False
            key = str(getattr(outcome, "key", "") or "")
            if not key:
                return False
            state = reasoner.problems.get(key)
            if state is None:
                return False
            # What she concluded was wrong, so record it as a dead end. Without this she
            # re-proposes the same conclusion every time the question comes round again.
            expected = str(getattr(outcome, "expected", "") or "")
            if not expected or state.was_rejected(expected):
                return False
            from nyxara.njp.reason import Hypothesis
            state.reject(Hypothesis(claim=expected, source="outcome"),
                         why="contradicted by what actually happened")
            return True
        except Exception:  # noqa: BLE001
            return False

    # ---- the loop --------------------------------------------------------- #
    def close(self, thought: Any) -> LoopReport:
        """Close the loop behind one turn. Called at the end of :meth:`NJPBrain._think`."""
        rep = LoopReport(turn=int(getattr(self.brain, "turns", 0)))
        t0 = time.perf_counter()
        try:
            self.closes += 1
            percept = getattr(thought, "percept", None)
            settled = getattr(percept, "settled", None)
            fired = tuple(getattr(settled, "fired", ()) or ())
            # Taken before any repair can run, so `_repair_world` sees a genuine before/after pair.
            self._snapshot = getattr(settled, "snapshot", None)

            self._count_events(thought, rep)
            self._observe_transition(thought, fired, rep)
            self._score_next_state(thought, fired, rep)
            self._deferred_answers(thought, rep)
            self._observe_capabilities(thought, rep)
            self._train_readout(fired, rep)
            self._slow(rep)

            self._prev_fired = fired
            self._prev_state = self._encode_state(fired)
            self._prev_snapshot = self._snapshot
            self._roll_up(rep)
            self.last = rep
            return rep
        except Exception:  # noqa: BLE001 — a failed close learns nothing and never breaks a turn
            self.last = rep
            return rep
        finally:
            rep.ms = (time.perf_counter() - t0) * 1000.0

    # ---- world ------------------------------------------------------------ #
    def _count_events(self, thought: Any, rep: LoopReport) -> None:
        """How many events this turn put on the timeline.

        The extraction itself happens in :meth:`NJPBrain.perceive`, which already calls
        ``world.from_grounding``. This only counts, so the number is reported by the loop that
        claims to be running rather than inferred from a total that something else moved.
        """
        try:
            grounding = getattr(getattr(thought, "percept", None), "grounding", None)
            if grounding is None:
                return
            from nyxara.njp.world import _EVENT_PREDICATES
            rep.events = sum(1 for t in (getattr(grounding, "triples", None) or [])
                             if t.predicate in _EVENT_PREDICATES)
        except Exception:  # noqa: BLE001
            pass

    def _observe_transition(self, thought: Any, fired: Sequence[int], rep: LoopReport) -> None:
        """Hand ``(before, action, after)`` to the dynamics model, once per turn.

        The action label is the *kind* of turn — question, command, statement — because that is
        the intervention she actually made on her own state. Labelling every transition with the
        raw text instead would give the kNN a distinct action per turn and no neighbours at all.
        """
        try:
            world = getattr(self.brain, "world", None)
            if world is None or not self._prev_state or not fired:
                return
            intent = getattr(thought, "intent", None)
            action = str(getattr(intent, "kind", "") or "turn")
            world.observe_transition(self._prev_state, action, self._encode_state(fired))
            rep.transitions = 1
        except Exception:  # noqa: BLE001
            pass

    @staticmethod
    def _encode_state(cells: Sequence[int]) -> List[float]:
        """Fired cells → a fixed-width normalised histogram.

        Coarse and deliberately so: this is "which regions were active, in what proportion", which
        is what a kNN dynamics model can actually use. It is stable under neurogenesis — new cells
        land in existing buckets rather than changing the width — which matters because the state
        encoding must not change shape every time the fabric grows.
        """
        vec = [0.0] * _STATE_WIDTH
        try:
            for cell in cells:
                vec[int(cell) % _STATE_WIDTH] += 1.0
            total = sum(vec)
            if total > 0:
                vec = [v / total for v in vec]
        except Exception:  # noqa: BLE001
            return [0.0] * _STATE_WIDTH
        return vec

    # ---- prediction: the self-supervised signal ---------------------------- #
    def _score_next_state(self, thought: Any, fired: Sequence[int], rep: LoopReport) -> None:
        """Score the manifold's pre-settle anticipation against what actually fired.

        This is the one signal that is available on **every** turn, and the one that makes
        ``scored`` stop being zero. It is honest because of *when* the two halves are computed:
        :meth:`NJPBrain.perceive` asks the manifold what will fire before it stimulates anything,
        and the settle that produces the answer runs afterwards. There is no path by which the
        prediction can see the observation.

        An untrusted anticipation is still scored. A prediction she had no confidence in and got
        wrong is a real miss — declining to count it would quietly restrict the accuracy statistic
        to the cases she already felt good about, which is how a calibration number becomes a lie.
        """
        try:
            predictor = getattr(self.brain, "predictor", None)
            percept = getattr(thought, "percept", None)
            anticipated = getattr(percept, "anticipated", None)
            if predictor is None or anticipated is None or not fired:
                return
            expected = tuple(getattr(anticipated, "cells", ()) or ())
            if not expected:
                return

            key = f"{getattr(thought, 'cycle_id', '')}:next-state"
            grounding = getattr(percept, "grounding", None)
            predictor.predict(
                key, expected, confidence=float(getattr(anticipated, "margin", 0.0) or 0.0),
                organ="manifold",
                stimulus=str(getattr(thought, "stimulus", "")),
                concepts=list(getattr(percept, "concepts", []) or []),
                triples=list(getattr(grounding, "triples", None) or []))
            outcome = predictor.observe(key, fired)
            if outcome is None:
                return
            rep.scored = 1
            if outcome.correct:
                rep.correct = 1
            diagnosis = getattr(outcome, "diagnosis", None)
            if diagnosis is not None and getattr(diagnosis, "attributed", False):
                rep.diagnoses[diagnosis.kind] = rep.diagnoses.get(diagnosis.kind, 0) + 1
                if getattr(diagnosis, "repaired", False):
                    rep.repaired += 1
                self._tell_self_model(diagnosis)
        except Exception:  # noqa: BLE001
            pass

    # ---- prediction: deferred answers -------------------------------------- #
    def _deferred_answers(self, thought: Any, rep: LoopReport) -> None:
        """Open an expectation for what she could not answer; close it when the Master says.

        The asymmetry is the point. Opening is cheap and happens whenever a question goes
        unanswered. Closing requires the fact to arrive from outside her — a *statement*, grounded
        into a triple — so what scores the guess is the Master's sentence and never her own later
        guess at the same question.
        """
        try:
            grounder = getattr(self.brain, "grounder", None)
            predictor = getattr(self.brain, "predictor", None)
            percept = getattr(thought, "percept", None)
            grounding = getattr(percept, "grounding", None)
            if grounder is None or predictor is None or grounding is None:
                return

            if getattr(grounding, "is_question", False):
                self._open_deferred(thought, grounder, predictor, grounding, rep)
            else:
                self._resolve_deferred(grounding, predictor, rep)
            rep.deferred_open = len(self._deferred)
        except Exception:  # noqa: BLE001
            pass

    def _open_deferred(self, thought: Any, grounder: Any, predictor: Any,
                       grounding: Any, rep: LoopReport) -> None:
        """Record what she said about a question, so a later fact can grade it."""
        try:
            subject, predicate = grounder._read_question(
                str(getattr(thought, "stimulus", "")).lower())
            if not predicate:
                return
            answer = getattr(grounding, "answer", None)
            said = str(getattr(answer, "text", "") or "")
            key = f"deferred:{subject.lower()}:{predicate}"
            if key in self._deferred:
                return
            predictor.predict(key, said or "<unknown>",
                              confidence=float(getattr(answer, "confidence", 0.0) or 0.0),
                              organ="grounding",
                              stimulus=str(getattr(thought, "stimulus", "")),
                              concepts=list(getattr(grounding, "concepts", []) or []),
                              expected_triples=True, triples=[])
            self._deferred[key] = _Deferred(
                key=key, subject=subject, predicate=predicate,
                asked_at=int(getattr(self.brain, "turns", 0)), answered=said)
            rep.deferred_opened = 1
            if len(self._deferred) > self.defer_capacity:
                for stale in list(self._deferred)[:len(self._deferred) - self.defer_capacity]:
                    self._deferred.pop(stale, None)
        except Exception:  # noqa: BLE001
            pass

    def _resolve_deferred(self, grounding: Any, predictor: Any, rep: LoopReport) -> None:
        """A fact arrived. Grade any open question it answers, and any answer it corrects."""
        try:
            for triple in (getattr(grounding, "triples", None) or []):
                key = f"deferred:{triple.subject.lower()}:{triple.predicate}"
                pending = self._deferred.pop(key, None)
                if pending is None:
                    continue
                outcome = predictor.observe(
                    key, triple.object,
                    evidence={"stimulus": triple.text, "concepts": [triple.object],
                              "triples": [triple], "expected_triples": True,
                              "in_memory": bool(pending.answered)})
                if outcome is None:
                    continue
                rep.deferred_resolved += 1
                if outcome.correct:
                    rep.correct += 1
                else:
                    # She had said something and the Master's statement says otherwise. That is a
                    # correction, and it is worth counting separately from a plain "she did not
                    # know" — the two call for different repairs.
                    if pending.answered:
                        rep.corrections += 1
                    diagnosis = getattr(outcome, "diagnosis", None)
                    if diagnosis is not None and getattr(diagnosis, "attributed", False):
                        rep.diagnoses[diagnosis.kind] = rep.diagnoses.get(diagnosis.kind, 0) + 1
                        if getattr(diagnosis, "repaired", False):
                            rep.repaired += 1
                        self._tell_self_model(diagnosis)
                rep.scored += 1
        except Exception:  # noqa: BLE001
            pass

    # ---- the self-model ---------------------------------------------------- #
    def _observe_capabilities(self, thought: Any, rep: LoopReport) -> None:
        """Grade the faculties this turn actually exercised, and only those.

        A capability is observed when the turn produced evidence about it, never on a schedule.
        Grading ``planning`` on a turn that planned nothing would fill a Beta posterior with
        counts that mean nothing, and the whole value of that posterior is that its counts mean
        something.
        """
        try:
            model = getattr(self.brain, "self_model", None)
            percept = getattr(thought, "percept", None)
            if model is None or percept is None:
                return
            grounding = getattr(percept, "grounding", None)

            # grounding — did words become structure, or a question find its answer?
            if grounding is not None and str(getattr(thought, "stimulus", "")).strip():
                if getattr(grounding, "is_question", False):
                    answer = getattr(grounding, "answer", None)
                    if answer is not None:
                        model.observe("grounding", 1.0 if answer.answered else 0.0)
                        rep.capabilities += 1
                elif getattr(grounding, "concepts", None):
                    # Only a turn that carried content is graded. "ok" grounding to nothing is
                    # correct behaviour, not a grounding failure.
                    model.observe("grounding", 1.0 if grounding.grounded else 0.0)
                    rep.capabilities += 1

            # recall — did content-addressed memory bring anything decided back?
            recall = getattr(percept, "recall", None)
            if recall is not None and hasattr(recall, "decided"):
                model.observe("recall", 1.0 if recall.decided else 0.0)
                rep.capabilities += 1

            # prediction — graded on the real next-state error, with partial credit.
            if rep.scored:
                predictor = getattr(self.brain, "predictor", None)
                outcomes = list(getattr(predictor, "outcomes", []) or [])
                if outcomes:
                    model.observe("prediction", max(0.0, 1.0 - float(outcomes[-1].error)))
                    rep.capabilities += 1

            # language — she had content and either put it into words or did not.
            if getattr(thought, "epistemic", "") != "unknown":
                model.observe("language", 1.0 if str(getattr(thought, "answer", "")) else 0.0)
                rep.capabilities += 1
        except Exception:  # noqa: BLE001
            pass

    def _tell_self_model(self, diagnosis: Any) -> None:
        """A diagnosed miss is direct evidence about the organ it was attributed to."""
        try:
            model = getattr(self.brain, "self_model", None)
            if model is not None:
                model.observe_diagnosis(diagnosis)
        except Exception:  # noqa: BLE001
            pass

    # ---- gradients --------------------------------------------------------- #
    def _train_readout(self, fired: Sequence[int], rep: LoopReport) -> None:
        """One real gradient step on ``what fired last turn → what fired this turn``.

        The readout had ``steps = 0`` because it was only ever trained inside a dream, and dreams
        run on the pulse. Training it here means the gradient learner sees every turn the local
        learner does, which is the whole premise of "two learners over one substrate".
        """
        try:
            if not self.train_enabled:
                return
            readout = getattr(self.brain, "readout", None)
            if readout is None or not self._prev_fired or not fired:
                return
            step = readout.train_step([(list(self._prev_fired), list(fired))])
            if step is None or not getattr(step, "samples", 0):
                return
            rep.trained = True
            rep.loss_before = float(step.loss_before)
            rep.loss_after = float(step.loss_after)
        except Exception:  # noqa: BLE001
            pass

    # ---- the slow organs ---------------------------------------------------- #
    def _slow(self, rep: LoopReport) -> None:
        """Consolidation, abstraction and curiosity, on a turn-counted cadence."""
        turn = int(getattr(self.brain, "turns", 0))

        if turn % self.consolidate_every == 0:
            try:
                levels = getattr(self.brain, "levels", None)
                if levels is not None:
                    got = levels.consolidate()
                    rep.consolidated = True
                    rep.promoted = int(getattr(got, "promoted", 0))
                    rep.forgotten = int(getattr(got, "forgotten", 0))
            except Exception:  # noqa: BLE001
                pass

        if turn % self.discover_every == 0:
            try:
                discoverer = getattr(self.brain, "discoverer", None)
                if discoverer is not None:
                    got = discoverer.discover()
                    rep.discovered = True
                    rep.proposed = int(getattr(got, "proposed", 0))
                    rep.confirmed = int(getattr(got, "confirmed", 0))
                    rep.refuted = int(getattr(got, "refuted", 0))
            except Exception:  # noqa: BLE001
                pass

        if turn % self.wonder_every == 0:
            try:
                curiosity = getattr(self.brain, "curiosity", None)
                if curiosity is not None:
                    rep.wondered = len(curiosity.wonder())
            except Exception:  # noqa: BLE001
                pass

    # ---- bookkeeping --------------------------------------------------------- #
    def _roll_up(self, rep: LoopReport) -> None:
        """Session totals, so ``stats()`` can answer 'is she learning' without a replay."""
        try:
            self.totals["events"] += rep.events
            self.totals["transitions"] += rep.transitions
            self.totals["scored"] += rep.scored
            self.totals["correct"] += rep.correct
            self.totals["deferred_opened"] += rep.deferred_opened
            self.totals["deferred_resolved"] += rep.deferred_resolved
            self.totals["corrections"] += rep.corrections
            self.totals["repaired"] += rep.repaired
            self.totals["capabilities"] += rep.capabilities
            self.totals["train_steps"] += int(rep.trained)
            self.totals["consolidations"] += int(rep.consolidated)
            self.totals["discoveries"] += int(rep.discovered)
            self.totals["wonders"] += int(bool(rep.wondered))
        except Exception:  # noqa: BLE001
            pass

    def stats(self) -> Dict[str, Any]:
        """Is the loop actually closed? Every number here was zero before this module existed."""
        scored = self.totals["scored"]
        return {
            "closes": self.closes,
            "totals": dict(self.totals),
            "accuracy": (round(self.totals["correct"] / scored, 4) if scored else None),
            "deferred_open": len(self._deferred),
            "cadence": {"consolidate": self.consolidate_every,
                        "discover": self.discover_every,
                        "wonder": self.wonder_every},
            "training": self.train_enabled,
            "last": self.last.to_dict() if self.last is not None else None,
        }
