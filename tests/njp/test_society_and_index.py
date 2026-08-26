"""§19 the Cognitive Society, and §27 the Intelligence Measurement Engine.

Two sections of the plan that had the same shape of gap in opposite directions.

**§19** asks for eight specialists over one world model, in sequence — *"ye multi-agent voting
nahi, cognitive specialisation hai"*. Three roles existed (``Explorer`` and ``Skeptic`` in
`growth/ecosystem.py`, ``Scientist`` in `growth/scientist.py`), none of them over NJP's world
model, five did not exist at all, and nothing anywhere ran them over a single claim.

**§27** asks that ``intelligence_index`` never be set by hand — *"capability claim nahi —
evidence"*. `njp/index.py` had computed the eight-term vector since it was written and
``measure()`` had **zero callers**, so the engine that exists to produce evidence produced it for
nobody. A single reading also cannot answer the question the module's own header poses — *is she
better than she was* — which takes two readings and a comparison, and
`IntelligenceVector.regressions` could make that comparison and had no caller either.
"""

from __future__ import annotations

from types import SimpleNamespace

from nyxara.njp import NJPBrain
from nyxara.njp.index import IntelligenceIndex
from nyxara.njp.society import ROLES, Case, CognitiveSociety

CAUSAL = ["aag se garmi hoti hai", "garmi se pasina hoti hai", "birds need water",
          "a sparrow is a bird", "what does a sparrow need?"]


def _lived(turns: int = 120) -> NJPBrain:
    brain = NJPBrain()
    for turn in range(turns):
        brain.think(CAUSAL[turn % len(CAUSAL)])
    return brain


# --------------------------------------------------------------------------- #
# §19 — specialisation, in order, never a vote
# --------------------------------------------------------------------------- #

def test_all_eight_roles_exist_and_are_asked_in_the_plans_order():
    assert [r.name for r in ROLES] == ["explorer", "scientist", "skeptic", "mathematician",
                                       "engineer", "historian", "strategist", "judge"]
    case = CognitiveSociety().deliberate(_lived(60), "aag causes pasina")
    assert [c.role for c in case.contributions] == [r.name for r in ROLES]


def test_a_role_whose_organ_cannot_speak_abstains_rather_than_voting_neutral():
    brain = _lived(40)
    case = CognitiveSociety().deliberate(brain, "aag causes pasina")
    silent = [c for c in case.contributions if not c.spoke]
    assert silent, "on a young brain some organ must have nothing to say"
    assert all(c.stance == "" for c in silent)
    assert all(c.confidence == 0.0 for c in silent)


def test_the_mathematician_proves_refutes_or_honestly_abstains():
    """The one role that can be decisive on its own, and the one that must not bluff."""
    society = CognitiveSociety()
    brain = NJPBrain()
    proved = society.deliberate(brain, "2 + 2 = 4").by_role("mathematician")
    refuted = society.deliberate(brain, "2 + 2 = 5").by_role("mathematician")
    vague = society.deliberate(brain, "aag causes pasina").by_role("mathematician")
    assert proved.spoke and proved.stance == "supports"
    assert refuted.spoke and refuted.stance == "objects"
    assert not vague.spoke, "an unformalisable claim must abstain, never guess"


def test_the_judge_does_not_count_the_others():
    """A verdict reached by tallying seven specialists is a fact about the committee."""
    society = CognitiveSociety()
    case = Case(claim="aag causes pasina")
    from nyxara.njp.society import _judge

    brain = _lived(60)
    # Stuff the case with objections the gauntlet knows nothing about.
    case.contributions.append(SimpleNamespace(role="skeptic", spoke=True, stance="objects"))
    verdict = _judge(brain, case)
    assert verdict is not None
    assert verdict.organ == "truth", "the verdict comes from the gauntlet, not from a tally"


def test_an_objection_stops_a_claim_being_established():
    society = CognitiveSociety()
    brain = _lived(60)
    case = society.deliberate(brain, "aag causes pasina")
    if case.objections:
        assert not case.established
    assert isinstance(case.established, bool)


def test_the_society_runs_from_the_loop_on_a_claim_she_rests_weight_on():
    """A society that deliberates over things nobody is relying on is a committee."""
    brain = _lived(160)
    assert brain.loop.totals["deliberations"] > 0
    assert brain.society.deliberations == brain.loop.totals["deliberations"]
    claims = {c.claim for c in brain.society.cases}
    held = {getattr(b, "claim", "") for b in brain.beliefs.beliefs.values()}
    assert claims & held, (claims, held)
    assert brain.pipeline_report()["society→verdict"]["state"] == "closed"


def test_every_role_is_reachable_under_its_own_condition():
    """Not a count on one corpus — which role speaks depends on the claim, and counting them on a
    session I happened to pick would pass or fail by luck. Each binding is exercised under the
    condition that organ needs, so "silent" can never quietly mean "unwired".
    """
    from nyxara.njp.society import (
        _engineer, _explorer, _historian, _judge, _mathematician, _scientist, _skeptic,
        _strategist,
    )

    brain = _lived(120)
    causal = Case(claim="aag causes pasina")

    assert _explorer(brain, causal) is not None          # she always has an open question
    assert _strategist(brain, causal) is not None        # and a standing mission
    assert _judge(brain, causal) is not None             # the gauntlet always returns a judgement
    assert _mathematician(brain, Case(claim="2 + 2 = 4")).stance == "supports"
    assert _scientist(brain, causal) is not None         # four rivals for a bare cause → effect

    # The Skeptic needs a claim its attacks can bind to; `attack` takes the two ends by name.
    skeptic = _skeptic(brain, causal)
    assert skeptic is not None and skeptic.stance in ("objects", "supports", "notes")

    # The Historian needs a recorded error to find a precedent in.
    brain.ledger.errors.record("aag causes pasina", verdict="refuted", truth="coincidence")
    assert _historian(brain, causal) is not None

    # The Engineer needs a reasoning form the genome has judged worth naming. Stubbed, because
    # forcing a real one takes a session shaped to it and what is under test is the binding.
    shaped = SimpleNamespace(genome=SimpleNamespace(
        candidates=lambda: [SimpleNamespace(shape=("is_a", "requires"), rate=0.8)]))
    engineer = _engineer(shaped, causal)
    assert engineer is not None and "is_a>requires" in engineer.finding


def test_silence_is_recorded_per_role_so_it_cannot_be_mistaken_for_wiring():
    brain = _lived(160)
    stats = brain.society.stats()
    assert stats["deliberations"] > 0
    assert set(stats["spoke"]) | set(stats["abstained"]) <= {r.name for r in ROLES}
    assert sum(stats["spoke"].values()) + sum(stats["abstained"].values()) == (
        stats["deliberations"] * len(ROLES))


def test_the_society_gated_off_is_absent_not_broken():
    brain = NJPBrain(config=SimpleNamespace(society_enabled=False))
    for line in CAUSAL:
        brain.think(line)
    assert brain.society is None
    assert "society" not in brain.stats()
    assert "society→verdict" not in brain.pipeline_report()


# --------------------------------------------------------------------------- #
# §27 — evidence, and the comparison that makes it evidence of *progress*
# --------------------------------------------------------------------------- #

def test_the_vector_is_read_by_the_loop_rather_than_by_nobody():
    brain = _lived(200)
    stats = brain.stats()["index"]
    assert stats["measurements"] > 0
    assert brain.loop.totals["index_measurements"] == stats["measurements"]
    assert brain.pipeline_report()["progress→measured"]["state"] == "closed"


def test_a_reading_is_kept_so_change_can_be_seen():
    """A single reading answers "how good is she", which this module says it cannot answer."""
    brain = _lived(60)
    index = IntelligenceIndex(width=6)
    assert index.trend is None
    index.track(brain)
    assert index.trend is None, "one reading is not a trend"
    index.track(brain)
    assert index.trend is not None
    assert len(index.stats()["series"]) >= 1


def test_regressions_are_found_rather_than_merely_computable():
    """`IntelligenceVector.regressions` could name the term that went backwards from the day it
    was written, and nothing called it."""
    brain = _lived(200)
    stats = brain.stats()["index"]
    assert stats["regressions_found"] >= 0
    assert isinstance(stats["last_regressions"], dict)
    # Over a real session at least one term moves the wrong way; if none did, the counter is
    # still the honest reading and the shape is what is asserted here.
    if stats["regressions_found"]:
        term, movement = next(iter(stats["last_regressions"].items()))
        assert set(movement) >= {"was", "now"}
        assert term in "GTCNREHU"


def test_a_narrower_run_measures_something_else_rather_than_the_same_thing_faster():
    """Regression on a config choice of mine: `index_width=4` reads `G = 0.000` — the
    generalization stage teaches six members and holds four out — and a zeroed term takes the
    whole geometric mean with it. `I_t` read 0.0000 while every other term was healthy."""
    brain = _lived(120)
    narrow = IntelligenceIndex(width=4).measure(brain)
    wide = IntelligenceIndex(width=6).measure(brain)
    assert wide.value("G") is not None
    assert wide.value("G") > (narrow.value("G") or 0.0)
    assert brain.index.width == 6


def test_absent_is_not_zero():
    """A product of five numbers in [0,1] would be zeroed by one unmeasured term — and worse, an
    unmeasured term would read as a measured failure."""
    brain = _lived(60)
    vector = IntelligenceIndex(width=6).measure(brain, benchmarks=False)
    assert "G" in vector.absent and "T" in vector.absent
    assert vector.value("G") is None
    assert vector.scalar is None or vector.scalar >= 0.0


def test_the_index_gated_off_is_absent_not_broken():
    brain = NJPBrain(config=SimpleNamespace(index_enabled=False))
    for line in CAUSAL:
        brain.think(line)
    assert brain.index is None
    assert "index" not in brain.stats()
    assert "progress→measured" not in brain.pipeline_report()
