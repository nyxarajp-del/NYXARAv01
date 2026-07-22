"""Tests for the Autonomous Motivation Engine + Ego Narrative (P24 · F19, organs 1 & 12)."""

import random

from nyxara.mind.muse import EgoNarrative, MuseEngine


def _summary(idea=9, verse=1, art=4):
    return {"per_modality": {"idea": {"count": idea, "mean_novelty": 0.3},
                             "verse": {"count": verse, "mean_novelty": 0.2},
                             "art": {"count": art, "mean_novelty": 0.4}}}


def test_project_targets_the_sparsest_modality():
    muse = MuseEngine(rng=random.Random(11))
    p = muse.propose_project(_summary())
    assert p.modality == "verse"
    assert p.status == "open"
    assert "will beat my rolling mean novelty" in p.hypothesis


def test_outcome_judges_hypothesis_honestly():
    muse = MuseEngine(rng=random.Random(11))
    win = muse.propose_project(_summary())
    muse.record_outcome(win, kept=2, best_novelty=0.55, mean_novelty=0.2)
    assert win.status == "corroborated"
    lose = muse.propose_project(_summary())
    muse.record_outcome(lose, kept=0, best_novelty=0.0, mean_novelty=0.2)
    assert lose.status == "refuted"


def test_refuted_fusion_still_grows_the_graph():
    muse = MuseEngine(rng=random.Random(1))
    before = muse.graph.associations_seen
    p = muse.propose_project()
    muse.record_outcome(p, kept=0, best_novelty=0.0, mean_novelty=0.5)
    assert muse.graph.associations_seen == before + 1


def test_ego_detects_planted_bias_and_muse_corrects_course():
    muse = MuseEngine(rng=random.Random(11))
    for _ in range(10):
        muse.ego.observe(modality="art", generator="lissajous", kept=True)
    bias = muse.ego.bias()
    assert bias is not None and bias[:2] == ("modality", "art")
    assert "lean on art" in muse.ego.narrative()
    p = muse.propose_project()
    assert p.modality != "art"         # corrective pressure is real


def test_ego_remembers_mistakes():
    ego = EgoNarrative()
    ego.observe(modality="idea", kept=False, rejection="cliche")
    ego.observe(modality="idea", kept=False, rejection="cliche")
    ego.observe(modality="idea", kept=False, rejection="quality")
    assert ego.worst_mistake() == "cliche"


def test_deterministic_under_seed():
    a = MuseEngine(rng=random.Random(5)).propose_project()
    b = MuseEngine(rng=random.Random(5)).propose_project()
    assert a.question == b.question and a.hypothesis == b.hypothesis


def test_persistence_round_trip():
    muse = MuseEngine(rng=random.Random(2))
    p = muse.propose_project(_summary())
    muse.record_outcome(p, kept=1, best_novelty=0.9, mean_novelty=0.1)
    muse.ego.observe(modality="verse", kept=True)
    clone = MuseEngine.from_dict(muse.to_dict(), rng=random.Random(2))
    assert len(clone.projects) == len(muse.projects)
    assert clone.projects[0].status == "corroborated"
    assert clone.ego.observations == muse.ego.observations
    assert clone.report()["by_status"] == muse.report()["by_status"]
