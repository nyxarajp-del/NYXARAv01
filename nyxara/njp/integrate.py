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
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

__all__ = ["LoopReport", "LearningLoop"]

# How often the slow organs run, counted in turns. Chosen so a short session still exercises each
# of them at least once — a cadence that never fires inside a normal conversation is the bug this
# module exists to fix, and setting these to "every N minutes" would reintroduce it exactly.
_CONSOLIDATE_EVERY = 8
_DISCOVER_EVERY = 12
_WONDER_EVERY = 16
# How often she goes after her own strongest causal claim. Slowest of the four, because an attack
# reads the whole event record and because a belief does not need challenging every turn — but
# on a *turn* count like the others, since a wall-clock cadence is exactly what left the slow
# organs at zero across a 113-turn session.
_ATTACK_EVERY = 20
# The capability ladder, and the growth ledger. Both were reachable only from a wall clock or a
# reporting method nobody calls in a turn — `ledger.record` from `pulse` alone, and
# `curriculum.assess` from `report_card` alone — which is precisely the failure this module was
# written to fix, in two organs it missed. Measured over 1,200 corpus pairs: `curriculum
# .assessments` 0 and `ledger.generations` 0, so "which stage has she reached" and "is she more
# than she was" both had no answer at all.
_ASSESS_EVERY = 24
_LEDGER_EVERY = 32

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
    # what an independently-established outcome was allowed to change. Every one of these was
    # structurally pinned at zero before the return edge existed, however well the organ behind
    # it worked when called by hand.
    strategies_graded: int = 0
    beliefs_settled: int = 0
    beliefs_retracted: int = 0
    questions_closed: int = 0
    experiments_run: int = 0
    bits_gained: float = 0.0
    # goals
    goals_added: int = 0
    goals_blocked: int = 0
    goals_completed: int = 0
    goals_open: int = 0
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
    # What she did to her own conclusions this turn. Kept on the report rather than only in the
    # attacker's counters so a caller can see, per turn, that a belief was challenged at all.
    attacked: int = 0
    attacks_refuted: int = 0
    attack_verdict: str = ""
    # Where she stands on the ladder, and which generation this turn closed. Both are measurements
    # rather than work, and both were unreachable from a turn.
    stage: str = ""
    stages_mastered: int = 0
    generation: int = 0
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
                    or self.discovered or self.wondered or self.attacked
                    or self.generation)

    def to_dict(self) -> Dict[str, Any]:
        return {"turn": self.turn, "events": self.events, "transitions": self.transitions,
                "scored": self.scored, "correct": self.correct,
                "deferred": {"opened": self.deferred_opened, "resolved": self.deferred_resolved,
                             "open": self.deferred_open, "corrections": self.corrections},
                "diagnoses": dict(self.diagnoses), "repaired": self.repaired,
                "graded": {"strategies": self.strategies_graded,
                           "beliefs_settled": self.beliefs_settled,
                           "beliefs_retracted": self.beliefs_retracted,
                           "questions_closed": self.questions_closed,
                           "experiments_run": self.experiments_run,
                           "bits_gained": round(self.bits_gained, 4)},
                "goals": {"added": self.goals_added, "blocked": self.goals_blocked,
                          "completed": self.goals_completed, "open": self.goals_open},
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
                "attacked": self.attacked, "attacks_refuted": self.attacks_refuted,
                "attack_verdict": self.attack_verdict,
                "stage": self.stage, "stages_mastered": self.stages_mastered,
                "generation": self.generation,
                "ms": round(self.ms, 3)}


@dataclass
class _Deferred:
    """A question she could not answer, kept open until the world says what the answer was."""

    key: str = ""                 # the predictor key — the question as asked
    subject: str = ""
    predicate: str = ""
    asked_at: int = 0
    answered: str = ""            # what she said at the time ("" when she abstained)
    # How she reached it, kept so the credit can find its way back to the strategy that produced
    # the answer once reality says whether it was right. A grade that cannot be attributed is a
    # grade nothing learns from.
    solution: Any = None
    question: str = ""            # the turn as the Master asked it


class LearningLoop:
    """Runs the closing half of a turn: score, diagnose, repair, consolidate, abstract.

    Holds no cognitive state of its own beyond what is needed to connect one turn to the next —
    the previous turn's firing pattern, and the questions still waiting on an answer. Everything
    it learns is written into the organs that own it, so removing this class loses the *loop* and
    not the knowledge.
    """

    def __init__(self, brain: Any, *, consolidate_every: int = _CONSOLIDATE_EVERY,
                 discover_every: int = _DISCOVER_EVERY, wonder_every: int = _WONDER_EVERY,
                 attack_every: int = _ATTACK_EVERY,
                 assess_every: int = _ASSESS_EVERY, ledger_every: int = _LEDGER_EVERY,
                 train: bool = True, defer_capacity: int = 256) -> None:
        self.brain = brain
        self.consolidate_every = max(1, int(consolidate_every))
        self.discover_every = max(1, int(discover_every))
        self.wonder_every = max(1, int(wonder_every))
        self.attack_every = max(1, int(attack_every))
        self.assess_every = max(1, int(assess_every))
        self.ledger_every = max(1, int(ledger_every))
        self.train_enabled = bool(train)
        self.defer_capacity = max(8, int(defer_capacity))

        self.closes = 0
        self.last: Optional[LoopReport] = None
        self.totals: Dict[str, int] = {
            "events": 0, "transitions": 0, "scored": 0, "correct": 0,
            "deferred_opened": 0, "deferred_resolved": 0, "corrections": 0,
            "repaired": 0, "capabilities": 0, "train_steps": 0,
            "consolidations": 0, "discoveries": 0, "wonders": 0,
            "goals_added": 0, "goals_completed": 0,
            "strategies_graded": 0, "beliefs_settled": 0,
            "beliefs_retracted": 0, "questions_closed": 0,
            "experiments_run": 0,
            "attacked": 0, "attacks_refuted": 0,
        }

        # Experiments already settled against the record. Kept here rather than read off the
        # designer, whose `run` is a count and whose `resolved` names hypotheses rather than
        # experiments — so neither can answer "have I settled this one already".
        self._experiments_run: Set[str] = set()

        # Turn-to-turn carry. The readout learns "what fired then → what fires now", which needs
        # exactly one turn of history and no more.
        self._prev_fired: Tuple[int, ...] = ()
        self._prev_state: List[float] = []
        # The manifold snapshots either side of the current turn. Kept here rather than read off
        # the fabric because the fabric overwrites its own "previous" during the settle.
        self._prev_snapshot: Any = None
        self._snapshot: Any = None
        self._deferred: Dict[str, _Deferred] = {}
        # Which goal node stands for which of her own questions, so one question is one task and
        # answering it completes that task rather than leaving a duplicate behind.
        self._question_nodes: Dict[str, str] = {}
        self._mission_nid: str = ""

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
            self._close_curiosity(thought, rep)
            self._track_goals(thought, rep)
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
                asked_at=int(getattr(self.brain, "turns", 0)), answered=said,
                solution=getattr(thought, "solution", None),
                question=str(getattr(thought, "stimulus", "")))
            self._stake_a_belief(thought, said, subject, predicate)
            rep.deferred_opened = 1
            if len(self._deferred) > self.defer_capacity:
                for stale in list(self._deferred)[:len(self._deferred) - self.defer_capacity]:
                    self._deferred.pop(stale, None)
        except Exception:  # noqa: BLE001
            pass

    def _close_curiosity(self, thought: Any, rep: LoopReport) -> None:
        """A fact arrived; any question it answers stops being open.

        Deliberately not tied to the deferral path. A deferral only exists where *she* was asked
        something and could not answer, while curiosity's questions are ones she raised herself —
        "what is Amit's employer?" is answered by the Master mentioning it in passing, not by him
        replying to her. Matching on subject and predicate is what makes that work: both sides
        already carry the pair, and the question was generated from exactly the relation the
        incoming triple now supplies.

        Without this, ``Curiosity.resolve`` had no caller, ``resolved`` never moved off zero, and
        the goal-completion branch in :meth:`_goals_from_curiosity` was unreachable — so a
        question she raised could only ever age out, never be answered.
        """
        try:
            curiosity = getattr(self.brain, "curiosity", None)
            grounding = getattr(getattr(thought, "percept", None), "grounding", None)
            if curiosity is None or grounding is None:
                return
            triples = list(getattr(grounding, "triples", None) or [])
            if not triples:
                return
            open_questions = list(curiosity.open_questions())
            if not open_questions:
                return
            for triple in triples:
                subject = str(getattr(triple, "subject", "") or "").strip().lower()
                predicate = str(getattr(triple, "predicate", "") or "").strip().lower()
                if not subject:
                    continue
                for question in open_questions:
                    if getattr(question, "resolved", False):
                        continue
                    if str(getattr(question, "subject", "") or "").strip().lower() != subject:
                        continue
                    wanted = str(getattr(question, "predicate", "") or "").strip().lower()
                    # A question naming no particular relation is answered by anything about its
                    # subject; one that names a relation waits for that relation specifically.
                    if wanted and wanted != predicate:
                        continue
                    if curiosity.resolve(question, str(getattr(triple, "object", "") or "")):
                        rep.questions_closed += 1
        except Exception:  # noqa: BLE001
            pass

    def _stake_a_belief(self, thought: Any, said: str, subject: str, predicate: str) -> None:
        """Record what she just asserted as a belief that can later be found wrong.

        The ledger only ever held what the *Master* said — ``field._record_beliefs`` writes his
        testimony and nothing writes hers — so the one class of claim that could be graded against
        an independent later fact was the one class never entered. Settling had nothing to settle,
        which is why ``retract``, ``settle`` and the whole calibration path behind them sat unused
        while working perfectly.

        Every belief goes in with the falsifier that would kill it, stated in advance and in the
        world's terms rather than hers. A claim recorded without one cannot take part in
        prediction-against-reality at all; it can only accumulate.
        """
        try:
            beliefs = getattr(self.brain, "beliefs", None)
            if beliefs is None or not said or said == "<unknown>":
                return
            solution = getattr(thought, "solution", None)
            beliefs.hold(
                said,
                confidence=float(getattr(solution, "confidence", 0.0) or 0.0) or 0.5,
                domain=predicate or "general",
                produced_by=str(getattr(solution, "strategy", "") or "grounding"),
                falsifier=f"{subject} {predicate} is stated to be something other than {said}",
                why="asserted in answer to a question")
        except Exception:  # noqa: BLE001 — an unrecorded belief loses the grade, never the turn
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
                self._grade_by_reality(pending, bool(outcome.correct), triple, rep)
                rep.scored += 1
        except Exception:  # noqa: BLE001
            pass

    def _grade_by_reality(self, pending: _Deferred, correct: bool,
                          triple: Any, rep: LoopReport) -> None:
        """Send one independently-established outcome to everything that staked a claim on it.

        This is the return edge the rest of NJP was missing. Every organ below could already be
        told it was right or wrong and none of them ever was, so each was left grading itself:
        strategy credit came from the critic that had just approved the answer, and the belief
        ledger recorded what it had been told without ever finding out whether it held.

        What makes the signal worth propagating is *where it comes from*. It is the Master's own
        later statement, grounded into a triple on a turn after the one being graded — so the
        thing doing the grading is independent of the thing being graded. A loop closed against
        her own later opinion of the same question would move every counter here and teach
        nothing, which is the failure this is built to avoid rather than the one it risks.
        """
        # Strategy credit. `outcome` explicitly overrides the provisional credit the critic
        # awarded, which is the whole reason it exists and the reason its absence mattered.
        try:
            metareason = getattr(self.brain, "metareason", None)
            if metareason is not None and pending.solution is not None:
                metareason.outcome(pending.solution, correct=correct)
                rep.strategies_graded += 1
        except Exception:  # noqa: BLE001
            pass

        # Belief settlement, and with it the calibration stack. Nothing called `settle` or
        # `retract`, so `_outcomes` stayed empty, `reliability()` always reported zero samples,
        # and `temper()` was a guaranteed no-op — an entire calibration path written, tested and
        # inert. A belief is retracted rather than deleted: driven to zero and kept, so the
        # tombstone survives and "this has failed before" stays answerable.
        try:
            beliefs = getattr(self.brain, "beliefs", None)
            if beliefs is not None and pending.answered:
                settled = beliefs.settle(pending.answered, true=correct,
                                         why=f"the Master stated {triple.object}")
                if settled is not None:
                    rep.beliefs_settled += 1
                elif not correct:
                    if beliefs.retract(pending.answered, why="contradicted by observation"):
                        rep.beliefs_retracted += 1
        except Exception:  # noqa: BLE001
            pass


    # ---- goals ------------------------------------------------------------- #
    def _track_goals(self, thought: Any, rep: LoopReport) -> None:
        """Turn what she was *asked to do* into structure she can be held to.

        :class:`~nyxara.njp.goals.GoalTree` was empty on every session — ``nodes = 0`` — and not
        because goal-tracking was unimplemented. :class:`~nyxara.njp.intent.IntentReader` already
        extracted ``goal``, ``actions``, ``ordering`` and ``polarity`` from every command, and
        nothing consumed any of it. The ordering constraint is the one that matters most and was
        being dropped: "pehle test chala phir code fix kar" is a *dependency*, and acting on the
        second half before the first is the failure mode a tool-using agent must not have.

        Two sources, both real, neither invented:

        * a **command** from the Master becomes a goal with one task per action, and its ordering
          pairs become genuine dependencies, so a task whose predecessor is unfinished reports
          ``blocked`` rather than "not started";
        * a **question she chose to investigate** becomes a task under her own standing mission,
          because deciding to go and find something out is a commitment, and one she can then be
          measured against.

        A negated action ("mat karo", "don't") is deliberately **not** added. Recording something
        she was told not to do as a thing to do is the worst possible reading of an instruction.
        """
        try:
            tree = getattr(self.brain, "goals", None)
            if tree is None:
                return
            self._goals_from_command(thought, tree, rep)
            self._goals_from_curiosity(tree, rep)
            rep.goals_open = len([n for n in tree.nodes.values()
                                  if n.state in ("open", "active", "blocked")])
        except Exception:  # noqa: BLE001
            pass

    def _goals_from_command(self, thought: Any, tree: Any, rep: LoopReport) -> None:
        """A command becomes a goal; its actions become tasks; its ordering becomes dependencies."""
        try:
            intent = getattr(thought, "intent", None)
            if intent is None or str(getattr(intent, "kind", "")) != "command":
                return
            actions = [a for a in (getattr(intent, "actions", None) or [])
                       if (getattr(intent, "polarity", None) or {}).get(a, True)]
            goal_name = str(getattr(intent, "goal", "") or "").strip() or \
                str(getattr(thought, "stimulus", ""))[:80]
            if not goal_name:
                return

            goal = tree.add(goal_name, kind="goal",
                            priority=float(getattr(intent, "confidence", 0.5) or 0.5),
                            confidence=float(getattr(intent, "confidence", 0.5) or 0.5))
            if goal is None:
                return
            rep.goals_added += 1

            # Ordering pairs name actions, and dependencies are by node id, so the tasks have to
            # exist before the constraint can be attached to any of them.
            by_action: Dict[str, Any] = {}
            for action in actions[:8]:
                task = tree.add(str(action)[:80], parent=goal.nid, kind="task")
                if task is not None:
                    by_action[str(action)] = task
                    rep.goals_added += 1

            for before, after in (getattr(intent, "ordering", None) or [])[:8]:
                first = self._task_for(before, by_action)
                second = self._task_for(after, by_action)
                if first is None or second is None or first.nid == second.nid:
                    continue
                if first.nid in second.dependencies:
                    continue
                second.dependencies.append(first.nid)
                tree._refresh_blocked(second)
                rep.goals_blocked += 1
        except Exception:  # noqa: BLE001
            pass

    @staticmethod
    def _task_for(phrase: Any, by_action: Dict[str, Any]) -> Any:
        """Which task an ordering endpoint refers to.

        The two halves of an ordering constraint are *phrases* — "pehle **test chala** phir
        **code fix kar**" yields ``("test chala", "code fix kar")`` — while the actions are the
        verbs pulled out of them, ``["test", "chala", "fix", "kar"]``. Matching the two by string
        equality therefore never matched anything, and every ordering constraint was silently
        dropped: the tasks existed, none of them depended on any other, and "run the tests first"
        was recorded as two unrelated jobs.

        Longest action contained in the phrase wins, so "code fix kar" resolves to ``fix`` rather
        than to a short accidental substring of it.
        """
        try:
            low = str(phrase or "").lower()
            if not low:
                return None
            best, best_len = None, 0
            for action, task in by_action.items():
                token = action.lower()
                if token and token in low and len(token) > best_len:
                    best, best_len = task, len(token)
            return best
        except Exception:  # noqa: BLE001
            return None

    def _goals_from_curiosity(self, tree: Any, rep: LoopReport) -> None:
        """A question she decided to investigate is a commitment, so it gets a node.

        Only questions whose value-of-information decision was to **go and look** become tasks.
        One she decided to act on what she already has, or to put to the Master, is not something
        she has undertaken to do, and filling the tree with those would make ``nodes`` a count of
        thoughts rather than of work.
        """
        try:
            curiosity = getattr(self.brain, "curiosity", None)
            if curiosity is None:
                return
            mission = self._own_mission(tree)
            if mission is None:
                return
            for question in curiosity.open_questions()[:4]:
                if str(getattr(question, "action", "")) not in ("gather", "investigate"):
                    continue
                name = str(getattr(question, "text", ""))[:80]
                if not name or name in self._question_nodes:
                    continue
                task = tree.add(name, parent=mission.nid, kind="task",
                                expected_value=float(getattr(question, "value", 0.5) or 0.5),
                                cost=float(getattr(question, "cost", 0.2) or 0.2))
                if task is not None:
                    self._question_nodes[name] = task.nid
                    rep.goals_added += 1

            # A question that has since been answered is work that is genuinely finished.
            for question in curiosity.questions.values():
                name = str(getattr(question, "text", ""))[:80]
                nid = self._question_nodes.get(name)
                if nid and getattr(question, "resolved", False):
                    if tree.complete(nid) is not None:
                        rep.goals_completed += 1
                    self._question_nodes.pop(name, None)
        except Exception:  # noqa: BLE001
            pass

    def _own_mission(self, tree: Any) -> Any:
        """Her one standing mission — understanding what she does not yet understand.

        Created lazily and once. A mission node that exists before she has ever wondered anything
        would be a slogan in the tree rather than a record of work.
        """
        try:
            if self._mission_nid and self._mission_nid in tree.nodes:
                return tree.nodes[self._mission_nid]
            node = tree.mission("close the gaps in what I know", priority=0.6)
            if node is not None:
                self._mission_nid = node.nid
            return node
        except Exception:  # noqa: BLE001
            return None

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

        if turn % self.discover_every == 0:
            self._run_experiments(rep)

        if turn % self.wonder_every == 0:
            try:
                curiosity = getattr(self.brain, "curiosity", None)
                if curiosity is not None:
                    rep.wondered = len(curiosity.wonder())
            except Exception:  # noqa: BLE001
                pass

        if turn % self.assess_every == 0:
            try:
                curriculum = getattr(self.brain, "curriculum", None)
                if curriculum is not None:
                    got = curriculum.assess(self.brain)
                    rep.stage = str(getattr(got, "current", "") or "")
                    rep.stages_mastered = len(getattr(got, "mastered", None) or ())
            except Exception:  # noqa: BLE001
                pass

        if turn % self.ledger_every == 0:
            try:
                ledger = getattr(self.brain, "ledger", None)
                fabric = getattr(self.brain, "fabric", None)
                if ledger is not None and fabric is not None:
                    evolver = getattr(self.brain, "evolver", None)
                    gen = ledger.record(
                        fabric_stats=fabric.stats(),
                        edits_kept=int(getattr(evolver, "kept", 0) or 0),
                        edits_rolled_back=int(getattr(evolver, "rolled_back", 0) or 0),
                        note="turn")
                    rep.generation = int(getattr(gen, "n", 0) or 0)
            except Exception:  # noqa: BLE001
                pass

        if turn % self.attack_every == 0:
            try:
                adversary = getattr(self.brain, "adversary", None)
                if adversary is not None:
                    for got in adversary.attack_strongest(limit=1):
                        rep.attacked += 1
                        rep.attacks_refuted += int(bool(got.refuted_by))
                        rep.attack_verdict = got.verdict
            except Exception:  # noqa: BLE001
                pass

    @staticmethod
    def _as_recorded(world: Any, effect: str) -> str:
        """The name the record actually files this effect under, where it differs from the law's.

        ``world`` carries two namings for one thing and they were never reconciled. A *stated*
        law names its effect with the noun the Master used — "water causes growth" — while an
        *observed* occurrence is filed under the canonical event predicate — "the plant grew"
        becomes ``grows``. So a hypothesis raised from the law asked the record about ``growth``,
        which the record has never heard of, and every experiment stayed unresolvable no matter
        how much evidence accumulated.

        Matched on a shared stem, and only when **exactly one** recorded event matches. An
        ambiguous stem returns the original name, which leaves the experiment open — the right
        outcome, because settling a hypothesis against the wrong event is worse than not settling
        it at all.
        """
        try:
            counts = getattr(world, "_counts", None) or {}
            if effect in counts:
                return effect
            stem = str(effect or "")[:4].lower()
            if len(stem) < 4:
                return effect
            matches = [name for name in counts if str(name)[:4].lower() == stem]
            return matches[0] if len(matches) == 1 else effect
        except Exception:  # noqa: BLE001
            return effect

    def _run_experiments(self, rep: LoopReport) -> None:
        """Settle a designed experiment against the record, and let the losing hypothesis die.

        :class:`~nyxara.njp.universe.ExperimentDesigner` designed an experiment on every single
        turn — measured at 8 designed, 0 run — and ``observe_result`` had no caller anywhere, so
        ``bits_gained`` was permanently 0.0 and no hypothesis was ever eliminated. Curiosity that
        computes the informative experiment and never performs it is not curiosity; it is a
        report about curiosity.

        The outcome comes from :meth:`~nyxara.njp.world.WorldView.counterfactual`, which counts,
        over her own event record, how often the effect happened *without* the cause. That matters
        more than the wiring: the alternative — asking the fitted universe what removing the cause
        would do — grades a hypothesis with the model the hypothesis is about, and every
        probability would move while nothing was learned. ``counterfactual`` returns
        ``still_happens=None`` when the record is too thin or too even to call, and an unresolved
        experiment is left open rather than settled on a guess.
        """
        designer = getattr(self.brain, "designer", None)
        world = getattr(self.brain, "world", None)
        if designer is None or world is None:
            return
        try:
            for name in list(getattr(designer, "hypotheses", None) or {}):
                if "→" not in name or name.startswith("not("):
                    continue
                cause, _, effect = name.partition("→")
                cause, effect = cause.strip(), effect.strip()
                experiment = f"remove:{cause}"
                if not cause or not effect or experiment in self._experiments_run:
                    continue
                verdict = world.counterfactual(cause, self._as_recorded(world, effect))
                if not verdict.answerable:
                    continue
                # "still happens without it" refutes the arrow; "never happens without it"
                # supports it. These are the two labels `_raise_hypotheses` predicted with.
                before = designer.prior_entropy()
                designer.observe_result(
                    experiment, "present" if verdict.still_happens else "absent")
                self._experiments_run.add(experiment)
                rep.experiments_run += 1
                rep.bits_gained += max(0.0, before - designer.prior_entropy())
        except Exception:  # noqa: BLE001 — an unresolved experiment stays open, never fatal
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
            self.totals["strategies_graded"] += rep.strategies_graded
            self.totals["beliefs_settled"] += rep.beliefs_settled
            self.totals["beliefs_retracted"] += rep.beliefs_retracted
            self.totals["questions_closed"] += rep.questions_closed
            self.totals["experiments_run"] += rep.experiments_run
            self.totals["capabilities"] += rep.capabilities
            self.totals["train_steps"] += int(rep.trained)
            self.totals["consolidations"] += int(rep.consolidated)
            self.totals["discoveries"] += int(rep.discovered)
            self.totals["wonders"] += int(bool(rep.wondered))
            self.totals["goals_added"] += rep.goals_added
            self.totals["goals_completed"] += rep.goals_completed
            self.totals["attacked"] += rep.attacked
            self.totals["attacks_refuted"] += rep.attacks_refuted
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
