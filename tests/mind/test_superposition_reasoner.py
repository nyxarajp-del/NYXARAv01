"""Tests for nyxara.mind.superposition_reasoner — Quantum-Probabilistic Superposition Reasoning.

Many candidate solution paths are held in a real Born-rule superposition, scored by a simulated
future outcome (world-model / sim rollout + grounded verification + confidence), and collapsed to the
optimal path — or left honestly superposed when none dominates. These tests assert the collapse
decisions, the abstain-on-ambiguity behaviour, and that a simulated outcome can overturn a confident
prior.
"""

from __future__ import annotations

from nyxara.mind.superposition_reasoner import (
    SuperpositionReasoner,
    SuperpositionResult,
    ThoughtPath,
)


class _GV:
    """A stub grounded verifier: only an answer containing '42' checks out."""

    def verify(self, stimulus, text):
        return 1.0 if "42" in str(text) else 0.05


class _Sim:
    """A stub simulator: rolling a path forward reveals 'safe' good, 'risky' bad."""

    def rollout(self, text, depth=2):
        return 0.95 if "safe" in str(text) else 0.1


def test_empty_input_is_noop():
    res = SuperpositionReasoner().deliberate("q", [])
    assert isinstance(res, SuperpositionResult)
    assert res.label is None and not res.decided and res.n_paths == 0


def test_verified_path_collapses_out():
    r = SuperpositionReasoner(collapse_threshold=0.55, grounded_verifier=_GV())
    hyps = [("a", "the answer is 7", 0.5), ("b", "the answer is 42", 0.5),
            ("c", "the answer is 13", 0.5)]
    res = r.deliberate("what is 6*7?", hyps)
    assert res.label == "b" and res.decided
    assert res.payload == "the answer is 42"


def test_ambiguous_stays_superposed():
    r = SuperpositionReasoner(collapse_threshold=0.7)
    amb = [("a", "maybe X", 0.5), ("b", "maybe Y", 0.5), ("c", "maybe Z", 0.5)]
    res = r.deliberate("ambiguous", amb)
    assert not res.decided                # three tied paths must not collapse
    assert res.entropy > 1.0              # high entropy ⇒ genuinely undecided


def test_equal_priors_stay_equal():
    """Regression: seeding N equal-quality paths must yield equal probabilities (add_many, not add)."""
    r = SuperpositionReasoner(collapse_threshold=0.9)
    res = r.deliberate("q", [("a", "x", 0.5), ("b", "y", 0.5), ("c", "z", 0.5)])
    probs = {p["label"]: p for p in res.paths}
    # with no distinguishing evidence all three outcomes are equal
    outs = {round(p["outcome"], 6) for p in res.paths}
    assert len(outs) == 1, res.paths


def test_prior_tilt_leads():
    r = SuperpositionReasoner(collapse_threshold=0.5)
    res = r.deliberate("q", [("strong", "well-supported", 0.9), ("weak", "shaky", 0.2)])
    assert res.label == "strong"


def test_simulated_outcome_overturns_confident_prior():
    r = SuperpositionReasoner(collapse_threshold=0.55, simulator=_Sim(), rollout_depth=2)
    paths = [("risky", "take the risky path", 0.8), ("safe", "take the safe path", 0.55)]
    res = r.deliberate("which action?", paths)
    assert res.label == "safe", res.to_dict()


def test_accepts_duck_typed_candidates():
    class _Cand:
        def __init__(self, text, conf):
            self.text = text
            self.confidence = conf
            self.kind = "respond"

    r = SuperpositionReasoner(collapse_threshold=0.5, grounded_verifier=_GV())
    res = r.deliberate("q", [_Cand("the answer is 1", 0.5), _Cand("the answer is 42", 0.5)])
    assert res.decided
    assert getattr(res.payload, "text", "") == "the answer is 42"


def test_scorer_failure_degrades_to_prior():
    def _bad_scorer(path, stimulus):
        raise RuntimeError("boom")

    r = SuperpositionReasoner(collapse_threshold=0.5)
    res = r.deliberate("q", [("a", "x", 0.9), ("b", "y", 0.1)], scorer=_bad_scorer)
    # a scorer that always raises must not crash; the prior quality still ranks a over b
    assert res.label == "a"


def test_max_paths_capped():
    r = SuperpositionReasoner(max_paths=3)
    res = r.deliberate("q", [(str(i), f"p{i}", 0.5) for i in range(10)])
    assert res.n_paths == 3
