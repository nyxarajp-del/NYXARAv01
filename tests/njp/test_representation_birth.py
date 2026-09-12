"""Phase 5 ⑩ — noticing that no concept explains a thing, and letting a new kind form.

``ConceptGenesis.explain`` could answer *"what is this, and if nothing, why not"* since the module
was written, and ``restructure`` could act on the answer. The only thing that ever produced one was
a prediction error diagnosed ``CONCEPTUAL`` — a real route, and a rare and incidental one. Nothing
ever asked the plain question: *is there anything here right now that none of my kinds covers?*
Measured on a brain fresh off the corpus, **7 of 44 subjects were uncovered**, including a concept
actively over-claiming about a platypus, and not one of them had ever been looked at.

The falsifier is :func:`test_the_detector_closes_a_gap_the_error_path_never_sees`: detector off
leaves the subject uncovered, detector on covers it, and compression does not fall. If those three
do not hold together the mechanism is either inert or a net loss, and either way it comes out.
"""

from __future__ import annotations

import os


from nyxara.njp.concepts import ConceptGenesis

os.environ.setdefault("NYXARA_NJP__EVOLVE_ENABLED", "false")


def _overclaiming() -> ConceptGenesis:
    """Four mammals that agree, and one that breaks the invariant the concept promised."""
    genesis = ConceptGenesis()
    for name in ("dog", "cat", "horse", "cow"):
        genesis.observe(name, {"is_a:mammal", "has_property:fur", "has_property:live young"})
    genesis.observe("platypus", {"is_a:mammal", "has_property:fur", "has_property:lays eggs"})
    genesis.crystallise()
    return genesis


# --------------------------------------------------------------------------- #
# The detector
# --------------------------------------------------------------------------- #
def test_the_detector_finds_what_no_concept_covers():
    genesis = _overclaiming()
    gaps = genesis.uncovered()
    assert [c.subject for c in gaps] == ["platypus"]
    assert gaps[0].gap == "violates"
    assert gaps[0].violated == ["has_property:live young"]


def test_a_covered_store_reports_no_gaps():
    """The control. A detector that always finds something is not detecting anything."""
    genesis = ConceptGenesis()
    for name in ("dog", "cat", "horse", "cow"):
        genesis.observe(name, {"is_a:mammal", "has_property:fur"})
    genesis.crystallise()
    assert genesis.uncovered() == []


def test_an_over_claiming_boundary_outranks_a_thing_nothing_resembles():
    """Acting on the wrong gap wastes the one restructure a cycle can afford."""
    genesis = _overclaiming()
    genesis.observe("quasar", {"has_property:distant", "has_property:bright"})
    genesis.crystallise()
    kinds = [c.gap for c in genesis.uncovered()]
    assert kinds[0] == "violates", kinds
    assert "unknown" in kinds, "the stranger is still found, it is just not acted on first"


def test_a_node_with_no_content_is_not_a_gap():
    """A subject recorded only because a relation points at it has nothing to be covered."""
    genesis = _overclaiming()
    genesis.observe("mammal", set())
    assert all(c.subject != "mammal" for c in genesis.uncovered())


# --------------------------------------------------------------------------- #
# The falsifier
# --------------------------------------------------------------------------- #
def _run(detector: bool):
    from nyxara.njp.brain import NJPBrain

    brain = NJPBrain()
    if not detector:
        brain.field._detect_gap = lambda rep: None
    lines = []
    for animal in ("a dog", "a cat", "a horse", "a cow"):
        lines += [f"{animal} is a mammal", f"{animal} has fur", f"{animal} has live young"]
    lines += ["a platypus is a mammal", "a platypus has fur", "a platypus lays eggs",
              "what is a platypus", "what is a dog", "what is a cow",
              "what is a horse", "what is a cat", "what is a mammal"]
    for line in lines:
        brain.think(line)
    return brain


def test_the_detector_closes_a_gap_the_error_path_never_sees():
    """Off leaves it uncovered, on covers it, and the description length does not fall."""
    off, on = _run(False), _run(True)

    assert off.stats()["field"]["gaps_detected"] == 0
    assert on.stats()["field"]["gaps_detected"] >= 1

    assert len(off.genesis.uncovered(limit=99)) >= 1, "the gap is there to be found"
    assert len(on.genesis.uncovered(limit=99)) == 0, "and the detector closed it"

    # A repair that explains one observation by making the model of everything else worse is a
    # trade she should refuse, so the whole point is that this costs nothing.
    assert on.genesis.compression() >= off.genesis.compression() - 1e-9

    # A kind was born rather than a fact learned.
    assert on.genesis.stats()["concepts"] > off.genesis.stats()["concepts"]


def test_a_repair_that_costs_compression_is_put_back():
    """`gaps_reverted` is why `gaps_detected` is not just a count of things tried."""
    from nyxara.njp.brain import NJPBrain
    from nyxara.njp.concepts import Coverage
    from nyxara.njp.field import CycleReport, _GAP_EVERY

    # A fresh brain, not `_run(False)` — that one has `_detect_gap` stubbed out, so calling it
    # here would be exercising the stub.
    brain = NJPBrain()
    field = brain.field
    genesis = brain.genesis
    for name in ("dog", "cat", "horse", "cow"):
        genesis.observe(name, {"is_a:mammal", "has_property:fur", "has_property:live young"})
    genesis.crystallise()
    knobs = (genesis.similarity, genesis.invariant_share)

    # A restructure that always makes things worse, so the keep-test is the only thing deciding.
    def _ruin(_coverage):
        from nyxara.njp.concepts import GenesisReport

        genesis.similarity = 0.01
        genesis.crystallise()
        return GenesisReport(compression_after=0.0)

    genesis.uncovered = lambda **kw: [Coverage(subject="platypus", gap="violates",
                                               concept="c:x", score=0.5)]
    genesis.restructure = _ruin
    field._last_restructured = ""
    field.cycles = _GAP_EVERY
    report = CycleReport()
    field._detect_gap(report)

    assert field.gaps_detected == 1
    assert field.gaps_reverted == 1
    assert not report.restructured
    assert (genesis.similarity, genesis.invariant_share) == knobs, "the knobs were put back"


def test_unknown_gaps_are_found_but_not_acted_on():
    """Measured: acting on them cost compression and closed nothing. Stated, not hidden."""
    from nyxara.njp.brain import NJPBrain
    from nyxara.njp.field import CycleReport, _GAP_EVERY

    brain = NJPBrain()
    genesis = brain.genesis
    for name in ("dog", "cat", "horse"):
        genesis.observe(name, {"is_a:mammal", "has_property:fur"})
    genesis.observe("quasar", {"has_property:distant", "has_property:bright"})
    genesis.crystallise()
    assert [c.gap for c in genesis.uncovered()] == ["unknown"]

    brain.field.cycles = _GAP_EVERY
    brain.field._last_restructured = ""
    brain.field._detect_gap(CycleReport())
    assert brain.field.gaps_detected == 0, "found, reported, and deliberately not acted on"


def test_the_error_path_keeps_priority():
    """A turn whose error was diagnosed conceptual already names the observation."""
    from nyxara.njp.brain import NJPBrain
    from nyxara.njp.field import CycleReport, ErrorClass, _GAP_EVERY

    brain = NJPBrain()
    field = brain.field
    field.cycles = _GAP_EVERY
    field._last_restructured = ""
    before = field.gaps_detected

    report = CycleReport()
    report.diagnosis = type("D", (), {"kind": ErrorClass.CONCEPTUAL, "subject": "x",
                                      "actionable": True, "coverage": None})()
    field._detect_gap(report)
    assert field.gaps_detected == before, "the scan yields to better evidence"
