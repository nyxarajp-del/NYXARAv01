"""Tests for nyxara.growth.meta_research — invent → test → (gated) integrate.

Hermetic and offline: no LLM, no network. The deterministic heuristic inventor poses
genuinely checkable propositions verified in the sandbox; integration is double-gated OFF by
default, so no source is ever touched in CI.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from nyxara.growth.meta_research import CandidateTheory, MetaResearcher, MetaResearchReport
from nyxara.memory.store import MemoryStore
from nyxara.sim.sandbox import Sandbox


def _mr(**kw) -> MetaResearcher:
    kw.setdefault("sandbox", Sandbox())
    return MetaResearcher(**kw)


# --------------------------------------------------------------------------- #
# inventing + verifying, fully offline
# --------------------------------------------------------------------------- #
def test_runs_offline_with_heuristic_fallback():
    report = _mr().run("caching strategies")
    assert isinstance(report, MetaResearchReport)
    assert report.candidates
    assert all(c.source == "heuristic" for c in report.candidates)
    assert report.validated >= 1


def test_candidates_are_actually_sandbox_tested():
    report = _mr().run("algorithm optimization")
    assert all(c.tested for c in report.candidates)
    # at least one validated candidate carries the proof-of-equality result
    validated = [c for c in report.candidates if c.validated]
    assert validated and all("validated" in c.experiment_result for c in validated)


def test_false_theory_is_refuted():
    mr = _mr()
    bad = CandidateTheory(title="2+2=5", kind="theory",
                          code="result = 2 + 2\nexpected = 5\n")
    mr._test(bad)
    assert bad.tested and not bad.validated


def test_open_questions_are_gathered():
    report = _mr().run("memory consolidation")
    assert report.open_questions          # always poses at least the offline fallback questions


# --------------------------------------------------------------------------- #
# recursive meta tower over the meta-research SEARCH (invention breadth) — wired end to end
# --------------------------------------------------------------------------- #
def test_meta_meta_tower_sealed_by_default():
    # default (test-profile, env-sealed) settings → the tower never builds; behaviour is unchanged
    assert _mr()._meta_meta() is None
    assert _mr().run("caching strategies").meta_meta is None


def test_meta_meta_tower_wires_and_surfaces_on_report(tmp_path):
    from nyxara.kernel.config import NyxaraSettings, Profile

    s = NyxaraSettings.for_profile(Profile.DEV)
    s.meta_research.meta_meta_enabled = True     # flag on (env seal only affects get_settings())
    s.meta_research.use_llm = False              # deterministic offline heuristic inventor
    s.meta_research.meta_levels = 2
    s.paths.data_dir = str(tmp_path)             # isolate tower persistence to tmp

    mr = MetaResearcher(sandbox=Sandbox(), settings=s)
    assert mr._meta_meta() is not None           # builds under a non-test profile with the flag on
    report = mr.run("caching strategies")
    # the recursive tower's status is surfaced on the report — wired end to end
    assert report.meta_meta is not None
    assert "champion" in report.meta_meta and "levels" in report.meta_meta
    # and this pass bound a bounded invention breadth from the tower's active genome into the config
    assert 1 <= s.meta_research.max_candidates <= 12


# --------------------------------------------------------------------------- #
# safety: restricted execution, no integration by default
# --------------------------------------------------------------------------- #
def test_candidate_code_cannot_import():
    mr = _mr()
    evil = CandidateTheory(title="escape", kind="theory",
                           code="import os\nresult = 1\nexpected = 1\n")
    mr._test(evil)
    # import is blocked in the restricted namespace -> the test fails, never escapes
    assert evil.tested and not evil.validated


def test_does_not_integrate_when_gates_off():
    pkg = Path(__import__("nyxara").__file__).resolve().parent
    before = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in pkg.rglob("*.py")}

    report = _mr(memory=MemoryStore()).run("caching strategies")
    assert report.integrated == 0
    assert all(not c.integrated for c in report.candidates)

    after = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in pkg.rglob("*.py")}
    assert before == after, "no source file may change while integration is gated off"
    # the grown-code file must not have been created
    assert not (pkg / "growth" / "_meta_grown.py").exists()


def test_validated_inventions_are_remembered():
    store = MemoryStore()
    _mr(memory=store).run("caching strategies")
    hits = [r for r in store._kv.values() if "invention" in r.tags]
    assert hits


# --------------------------------------------------------------------------- #
# bookkeeping
# --------------------------------------------------------------------------- #
def test_total_validated_and_reports_accumulate():
    mr = _mr()
    mr.run("a")
    mr.run("b")
    assert len(mr.all_reports()) == 2
    assert mr.total_validated() >= 2


def test_report_to_dict_roundtrip():
    d = _mr().run("optimization").to_dict()
    assert d["topic"] == "optimization"
    assert "candidates" in d and isinstance(d["candidates"], list)
    assert "validated" in d and "integrated" in d


# --------------------------------------------------------------------------- #
# growth-engine + core wiring
# --------------------------------------------------------------------------- #
def test_growth_engine_meta_pass():
    from nyxara.growth.autolearn import GrowthEngine

    engine = GrowthEngine(memory=MemoryStore(), enable_meta_research=True,
                          meta_research_every=1)
    rep = engine.run()
    assert rep.meta_research is not None
    assert rep.meta_research["validated"] >= 1


def test_core_meta_discover_is_wired():
    from nyxara.kernel.orchestrator import NyxaraCore

    core = NyxaraCore(enable_tools=False, enable_skills=False)
    out = core.meta_discover("algorithm optimization")
    assert "error" not in out
    assert out["validated"] >= 1
