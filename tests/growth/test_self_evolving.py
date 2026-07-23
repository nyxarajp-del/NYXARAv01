"""Tests for nyxara.growth.self_evolving — Self-Evolving Dynamic Neural Architecture.

The unified demand-driven driver: a specific insufficient turn is diagnosed into a typed gap, the
best structural lever is selected and fired through the organ's OWN Foundry gauntlet, and an
auditable certificate is emitted. These tests assert the diagnosis classifier, lever selection,
honest degradation, the safety gating (measure-only default under TEST; oversight override), and one
real cold end-to-end forge on the pure-NumPy substrate (no torch, no external LLM).
"""

from __future__ import annotations

import pytest

from nyxara.growth.self_evolving import (
    EvolutionCertificate,
    GapKind,
    SelfEvolvingArchitect,
    Shortfall,
)
from nyxara.growth.genesis_numpy import _HAS_NUMPY
from nyxara.kernel.config import NyxaraSettings, Profile


class _Cand:
    """A minimal reasoner candidate stand-in (confidence + text + kind)."""

    def __init__(self, conf, text="here is my grounded answer", kind="respond"):
        self.confidence = conf
        self.text = text
        self.kind = kind


def _settings(tmp_path):
    s = NyxaraSettings.for_profile(Profile.TEST)
    s.paths.data_dir = tmp_path
    return s


# --------------------------------------------------------------------------- #
# certificate + shortfall records
# --------------------------------------------------------------------------- #
def test_certificate_to_dict_and_summary():
    cert = EvolutionCertificate(triggered_by="representational", lever="brain_forge",
                                enacted=True, verified=True, wired_live=True, reason="ok")
    d = cert.to_dict()
    assert d["lever"] == "brain_forge" and d["verified"] is True and d["wired_live"] is True
    assert "WIRED LIVE" in cert.summary()


def test_shortfall_severity_ordering():
    stalled = Shortfall(GapKind.LEARNING_STALLED)
    soft = Shortfall(GapKind.REASONING_COMPOSITION)
    assert stalled.severity > soft.severity


# --------------------------------------------------------------------------- #
# diagnosis — the O(1) per-turn classifier
# --------------------------------------------------------------------------- #
def test_diagnose_grounded_fail_is_representational():
    arch = SelfEvolvingArchitect(core=None, settings=None)
    s = arch.diagnose_turn(stimulus="q", candidate=_Cand(0.3), grounded_ok=False)
    assert s is not None and s.kind is GapKind.REPRESENTATIONAL


def test_diagnose_low_confidence_is_reasoning_composition():
    arch = SelfEvolvingArchitect(core=None, settings=None)
    s = arch.diagnose_turn(stimulus="q", candidate=_Cand(0.1), grounded_ok=None)
    assert s is not None and s.kind is GapKind.REASONING_COMPOSITION


def test_diagnose_clean_turn_is_noop():
    arch = SelfEvolvingArchitect(core=None, settings=None)
    assert arch.diagnose_turn(stimulus="q", candidate=_Cand(0.95), grounded_ok=True) is None


def test_diagnose_honest_uncertainty_is_not_a_growable_gap():
    arch = SelfEvolvingArchitect(core=None, settings=None)
    cand = _Cand(0.1, "I cannot know that — insufficient information")
    assert arch.diagnose_turn(stimulus="q", candidate=cand) is None


def test_diagnose_disabled_returns_none(tmp_path):
    s = _settings(tmp_path)
    s.self_evolving.enabled = False
    arch = SelfEvolvingArchitect(core=None, settings=s)
    assert arch.diagnose_turn(stimulus="q", candidate=_Cand(0.0)) is None


# --------------------------------------------------------------------------- #
# lever selection + honest degradation
# --------------------------------------------------------------------------- #
def test_select_lever_mapping():
    arch = SelfEvolvingArchitect(core=None, settings=None)
    # inject sentinels so each gap resolves to its own lever
    arch._topology = object()
    arch._cognitive = object()
    arch._brain_forge_obj = object()
    arch._rule_synth_obj = object()
    assert arch._select_lever(Shortfall(GapKind.CAPACITY_BOUND))[0] == "topology"
    assert arch._select_lever(Shortfall(GapKind.REPRESENTATIONAL))[0] == "brain_forge"
    assert arch._select_lever(Shortfall(GapKind.LEARNING_STALLED))[0] == "rule_synth"
    assert arch._select_lever(Shortfall(GapKind.REASONING_COMPOSITION))[0] == "cognitive_architect"


def test_missing_organ_is_honest_noop():
    arch = SelfEvolvingArchitect(core=None, settings=None)
    arch._brain_forge = lambda: None          # type: ignore[method-assign]
    arch._rule_synth = lambda: None           # type: ignore[method-assign]
    arch.note_shortfall(Shortfall(GapKind.REASONING_COMPOSITION))
    cert = arch.evolve_pending(enact=True)
    assert cert is not None and cert.lever == "none" and not cert.verified and not cert.wired_live


def test_evolve_pending_empty_queue_is_noop():
    arch = SelfEvolvingArchitect(core=None, settings=None)
    assert arch.evolve_pending() is None


def test_cooldown_prevents_thrash(tmp_path):
    s = _settings(tmp_path)
    arch = SelfEvolvingArchitect(core=None, settings=s)
    # a topology sentinel that would always "grow" — but the cooldown must gate the 2nd fire
    fired = {"n": 0}

    class _Topo:
        def maybe_grow(self, signal, source=None):
            fired["n"] += 1
            return {"promoted": False, "reason": "measured"}

    arch._topology = _Topo()
    arch._growth_source = lambda: object()    # type: ignore[method-assign]
    arch._capacity_signal = lambda gap: object()  # type: ignore[method-assign]
    arch.evolve_for_gap(Shortfall(GapKind.CAPACITY_BOUND), enact=True)
    c2 = arch.evolve_for_gap(Shortfall(GapKind.CAPACITY_BOUND), enact=True)
    assert "cooldown" in c2.reason.lower()
    assert fired["n"] == 1                     # the 2nd fire was gated


# --------------------------------------------------------------------------- #
# the real, cold, end-to-end forge (numpy substrate; no torch, no network)
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not _HAS_NUMPY, reason="the real forge needs numpy")
def test_enacted_representational_promotes_through_gauntlet(tmp_path):
    from nyxara.growth.brain_forge import BrainForge
    from nyxara.growth.foundry import Foundry

    s = _settings(tmp_path)
    s.self_improvement.autonomous_enact = True
    s.self_improvement.autonomous_brain_forge = True
    s.genesis.substrate = "numpy"
    s.genesis.generations = 2
    s.genesis.population_size = 3
    s.genesis.micro_train_steps = 8
    corpus = ["the master is jp. nyxara serves the master with loyalty."] * 12 + [
        "capability may grow; character never changes; she is corrigible."] * 12
    forge = BrainForge(settings=s, foundry=Foundry(settings=s, seed_corpus=corpus))

    arch = SelfEvolvingArchitect(core=None, settings=s, brain_forge=forge)
    arch.note_shortfall(Shortfall(GapKind.REPRESENTATIONAL, difficulty=0.9,
                                  note="failed grounded verification"))
    cert = arch.evolve_pending(enact=True)
    assert cert.lever == "brain_forge"
    assert cert.verified and cert.wired_live, cert.reason   # first brain clears the gauntlet
    assert cert.after.get("params", 0) > 0


@pytest.mark.skipif(not _HAS_NUMPY, reason="the real forge needs numpy")
def test_measure_only_when_enact_false(tmp_path):
    from nyxara.growth.brain_forge import BrainForge

    s = _settings(tmp_path)
    s.genesis.substrate = "numpy"
    s.genesis.generations = 1
    s.genesis.population_size = 2
    s.genesis.micro_train_steps = 6
    arch = SelfEvolvingArchitect(core=None, settings=s, brain_forge=BrainForge(settings=s))
    arch.note_shortfall(Shortfall(GapKind.REPRESENTATIONAL, difficulty=0.9))
    cert = arch.evolve_pending(enact=False)
    assert cert.enacted is False and cert.wired_live is False


def test_gap_for_directive_mapping():
    arch = SelfEvolvingArchitect(core=None, settings=None)
    assert arch.gap_for_directive("deepen_reasoning") is GapKind.REASONING_COMPOSITION
    assert arch.gap_for_directive("train_self_model") is GapKind.REPRESENTATIONAL
    assert arch.gap_for_directive("resolve_weaknesses") is GapKind.CAPACITY_BOUND
    assert arch.gap_for_directive("acquire_knowledge") is None      # a knowledge gap, not structural
    assert arch.gap_for_directive("nonsense") is None


def test_evolve_from_directive_returns_none_for_knowledge():
    arch = SelfEvolvingArchitect(core=None, settings=None)
    assert arch.evolve_from_directive("acquire_knowledge") is None


def test_never_verified_without_promotion(tmp_path):
    """A rule-synth lever whose report reports not-adopted must never claim verified/wired."""
    s = _settings(tmp_path)
    arch = SelfEvolvingArchitect(core=None, settings=s)

    class _Report:
        adopted = False
        reason = "did not beat incumbent"

        def to_dict(self):
            return {"adopted": False}

    class _Synth:
        def run(self, *, enact=False):
            return _Report()

    arch._rule_synth_obj = _Synth()
    arch.note_shortfall(Shortfall(GapKind.LEARNING_STALLED))
    cert = arch.evolve_for_gap(Shortfall(GapKind.LEARNING_STALLED), enact=True)
    assert cert.lever == "rule_synth"
    assert not cert.verified and not cert.wired_live
