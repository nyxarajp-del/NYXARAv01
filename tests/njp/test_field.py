"""The Recursive Cognitive Field: both loops, and the guards that keep loop 2 honest.

Loop 1's claim is that a prediction error can *restructure what she can represent*, not only
adjust a number. Loop 2's claim is that a self-modification survives only if a held-out benchmark
says so. Both claims are the kind that are easy to assert and easy to fake, so the tests below go
after the fakes: a restructure that fires before there is anything to restructure, a benchmark
scored on the data that shaped it, and a modification accepted on a tie.
"""

from __future__ import annotations

from nyxara.njp.beliefs import BeliefLedger
from nyxara.njp.concepts import ConceptGenesis
from nyxara.njp.field import Bottleneck, ErrorClass, Modification, RecursiveCognitiveField
from nyxara.njp.metareason import MetaReasoner
from nyxara.njp.universe import ExperimentDesigner, InternalUniverse


class _Triple:
    def __init__(self, subject, predicate, obj="", confidence=0.9):
        self.subject, self.predicate, self.object = subject, predicate, obj
        self.confidence, self.text = confidence, f"{subject} {predicate} {obj}"


class _Grounding:
    def __init__(self, triples):
        self.triples = triples
        self.is_question = False


class _Percept:
    def __init__(self, triples):
        self.grounding = _Grounding(triples)
        self.concepts = [t.subject for t in triples]
        self.settled = None


class _Thought:
    def __init__(self, triples):
        self.percept = _Percept(triples)
        self.loop = None
        self.intent = None


class _Brain:
    """Only what the field actually reads, so the test is about the field."""

    def __init__(self, world=None):
        self.turns = 0
        self.world = world
        self.predictor = None


def _field(**kwargs) -> RecursiveCognitiveField:
    return RecursiveCognitiveField(
        _Brain(), concepts=ConceptGenesis(), universe=InternalUniverse(),
        designer=ExperimentDesigner(), beliefs=BeliefLedger(), meta=MetaReasoner(),
        **kwargs)


def test_a_cycle_forms_concepts_and_records_beliefs():
    field = _field()
    report = field.cycle(_Thought([_Triple("dog", "is_a", "animal")]))
    assert report.perceived == 1 and report.concepts_fed >= 1
    assert report.beliefs_touched == 1
    assert field.beliefs.beliefs, "nothing was believed from a grounded turn"


def test_a_belief_is_held_at_the_strength_of_its_evidence_not_the_parsers_confidence():
    """How sure she is the Master *said* it is not how sure she is that it is true."""
    field = _field()
    field.cycle(_Thought([_Triple("aag", "causes", "garmi", confidence=0.99)]))
    belief = next(iter(field.beliefs.beliefs.values()))
    assert belief.confidence <= 0.6 + 1e-9
    assert not belief.unsupported, "a freshly grounded belief must not audit as over-claimed"
    assert belief.falsifiable


def test_a_conceptual_error_restructures_rather_than_reweighting():
    field = _field(blame_floor=4)
    for name in ("dog", "cat", "wolf"):
        field.concepts.observe(name, ["living", "is_a:animal", "has:legs", "makes:sound"])
    field.concepts.observe("horse", ["living", "is_a:animal", "has:legs"])
    field.concepts.crystallise()

    thought = _Thought([_Triple("horse", "is_a", "animal")])
    report = field.cycle(thought)
    report.error = 0.9                                  # force the critic to run
    diagnosis = field._diagnose(thought, field._triples(thought), report)

    assert diagnosis.kind == ErrorClass.CONCEPTUAL
    assert diagnosis.subject == "horse"
    before = field.concepts.compression()
    report.diagnosis = diagnosis
    field._repair(report)
    assert field.concepts.compression() >= before


def test_a_repair_does_not_fire_before_there_is_anything_to_repair():
    """The measured failure: early gaps drove the clustering threshold to its floor."""
    field = _field(blame_floor=12)
    thought = _Thought([_Triple("quasar", "emits", "radio")])
    report = field.cycle(thought)
    report.error = 0.9
    diagnosis = field._diagnose(thought, field._triples(thought), report)
    assert diagnosis.kind != ErrorClass.CONCEPTUAL, "blamed a concept system with 1 observation"


def test_the_same_subject_does_not_restructure_twice_running():
    field = _field(blame_floor=2)
    for i in range(6):
        field.concepts.observe(f"thing{i}", ["living", "has:legs"])
    field.concepts.crystallise()

    thought = _Thought([_Triple("alien", "emits", "radio")])
    report = field.cycle(thought)
    report.error = 0.9
    report.diagnosis = field._diagnose(thought, field._triples(thought), report)
    field._repair(report)
    first = field.restructures
    field._repair(report)                                # same subject again
    assert field.restructures == first


def test_a_structural_error_declares_the_missing_arrow():
    field = _field(blame_floor=1000)                     # keep the conceptual branch out of it
    thought = _Thought([_Triple("aag", "causes", "garmi")])
    report = field.cycle(thought)
    report.error = 0.9
    diagnosis = field._diagnose(thought, field._triples(thought), report)
    assert diagnosis.kind == ErrorClass.STRUCTURAL
    report.diagnosis = diagnosis
    field._repair(report)
    assert ("aag", "garmi") in field.universe.relations


def test_an_experiment_is_designed_and_becomes_an_open_question():
    field = _field()
    field.designer.propose("a", predictions={"probe": "yes"})
    field.designer.propose("b", predictions={"probe": "no"})
    report = field.cycle(_Thought([_Triple("dog", "is_a", "animal")]))
    assert report.experiment == "probe" and report.experiment_gain > 0
    assert any("probe" in q for q in field.beliefs.open_questions)


# --------------------------------------------------------------------------- #
# Loop 2
# --------------------------------------------------------------------------- #
def test_the_benchmark_is_scored_on_held_out_samples_only():
    field = _field()
    for i in range(40):
        field.cycle(_Thought([_Triple(f"thing{i}", "is_a", "kind")]))
    assert field._holdout, "nothing was held out"
    assert not (set(s for s, _ in field._samples) & set(s for s, _ in field._holdout)), \
        "the benchmark set overlaps the set that shaped the model"


def test_a_protected_knob_is_refused_before_it_is_measured():
    field = _field()
    assert field._protected(Modification(organ="truth", knob="min_sources"))
    assert field._protected(Modification(organ="guard", knob="oversight"))
    assert not field._protected(Modification(organ="concepts", knob="similarity"))


def test_a_modification_that_does_not_win_is_reverted():
    field = _field()
    before = field.concepts.similarity
    modification = Modification(organ="concepts", knob="similarity",
                                before=before, after=before + 0.2)
    assert field._apply(modification)
    assert field.concepts.similarity != before
    assert field._revert(modification)
    assert field.concepts.similarity == before


def test_a_tie_is_not_accepted():
    """An equal score is not evidence for a change; accepting ties is how a system drifts."""
    field = _field()
    for i in range(30):
        field.cycle(_Thought([_Triple(f"thing{i}", "is_a", "kind")]))
    field.benchmark = lambda: 0.5                       # identical before and after
    field.find_bottleneck = lambda: Bottleneck(organ="concepts", metric="compression",
                                               value=1.0, severity=0.9, why="test")
    trial = field.meta_cycle()
    assert not trial.accepted
    # A benchmark that returns the same number whatever changed is, by construction, one that
    # cannot see the knob — which is what this stub makes it. Reported as the stated gap it is
    # rather than as a failed experiment; see `meta_cycle`. The property under test is unchanged:
    # a tie is never accepted.
    assert "cannot see this knob" in trial.why, trial.why
    assert field.stats()["meta_unmeasurable"] >= 1


def test_a_measurably_worse_change_is_reported_as_no_gain():
    """The other half of the same classification: the measure saw it, and it lost."""
    field = _field()
    for i in range(30):
        field.cycle(_Thought([_Triple(f"thing{i}", "is_a", "kind")]))
    scores = iter((0.6, 0.4))                           # baseline, then a worse candidate
    field.benchmark = lambda: next(scores, 0.4)
    field.find_bottleneck = lambda: Bottleneck(organ="concepts", metric="compression",
                                               value=1.0, severity=0.9, why="test")
    trial = field.meta_cycle()
    assert not trial.accepted
    assert "no gain" in trial.why, trial.why


def test_the_adversarial_battery_rejects_a_concept_that_asserts_nothing():
    field = _field()
    for i in range(6):
        field.concepts.observe(f"thing{i}", [f"unique:{i}"])
    field.concepts.crystallise()
    # An unseen observation with nothing in common must not come back "covered".
    assert not field.concepts.explain("__probe__", ["__nonsense__"]).covered


def test_the_meta_cycle_runs_on_its_own_count():
    field = _field(meta_every=5)
    for i in range(4):
        field.cycle(_Thought([_Triple(f"thing{i}", "is_a", "kind")]))
    assert not field.due_for_meta()
    field.cycle(_Thought([_Triple("thing4", "is_a", "kind")]))
    assert field.due_for_meta()


def test_state_survives_a_round_trip():
    field = _field()
    for i in range(12):
        field.cycle(_Thought([_Triple(f"thing{i}", "is_a", "kind")]))
    revived = _field()
    revived.load_dict(field.to_dict())
    assert revived.cycles == field.cycles
    assert len(revived._holdout) == len(field._holdout)
