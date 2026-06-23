"""Tests for nyxara.growth.synthesis — Synthetic Data Self-Curation (the AlphaGo-Zero method).

Pure-stdlib, no torch, no network: the rival verifier is the deterministic Prover + a restricted
code sandbox, so these run anywhere CI runs.
"""
from __future__ import annotations

from pathlib import Path

from nyxara.growth.flywheel import DataFlywheel
from nyxara.growth.synthesis import (
    DEFAULT_DOMAINS,
    Domain,
    RivalVerifier,
    SyntheticCurator,
    SyntheticGenerators,
    SyntheticItem,
)
from nyxara.kernel.config import SynthesisConfig
from nyxara.knowledge.base import KnowledgeBase


def test_every_domain_generates_a_verifiable_item():
    gens = SyntheticGenerators(seed=11)
    verifier = RivalVerifier()
    for dom in DEFAULT_DOMAINS:
        item = gens.one(dom)
        assert item is not None and item.domain == dom
        ok, why = verifier.verify(item)
        assert ok, f"{dom} item should verify: {why}"


def test_rival_refutes_false_arithmetic():
    verifier = RivalVerifier()
    bad = SyntheticItem(domain="arithmetic", prompt="?", answer="5",
                        proof_kind="arithmetic", statement="2 + 2 = 5")
    ok, why = verifier.verify(bad)
    assert ok is False and "REFUTED" in why.upper()


def test_rival_refutes_buggy_code_in_sandbox():
    verifier = RivalVerifier()
    bad = SyntheticItem(
        domain="code", prompt="?", answer="def f(x):\n    return x + 1", proof_kind="code",
        code_spec={"fn": "f", "reference": "def f(x):\n    return x * x",
                   "candidate": "def f(x):\n    return x + 1", "tests": [(2,), (5,)]})
    ok, _ = verifier.verify(bad)
    assert ok is False


def test_sandbox_blocks_imports():
    """Generated code cannot reach the filesystem/network — __import__ is unavailable."""
    verifier = RivalVerifier()
    malicious = SyntheticItem(
        domain="code", prompt="?", answer="x", proof_kind="code",
        code_spec={"fn": "f",
                   "reference": "def f(x):\n    return x",
                   "candidate": "def f(x):\n    import os\n    return os.getpid()",
                   "tests": [(1,)]})
    ok, _ = verifier.verify(malicious)
    assert ok is False  # the import raises inside the sandbox → refuted, not executed


def test_curate_feeds_knowledge_and_flywheel(tmp_path: Path):
    kb = KnowledgeBase(name="synthetic")
    fw = DataFlywheel(store_path=tmp_path / "synthetic.jsonl", min_chars=1)
    cur = SyntheticCurator(knowledge=kb, flywheel=fw,
                           cfg=SynthesisConfig(batch_size=25, seed=3))
    report = cur.curate()
    assert report.accepted >= 1
    assert report.fed_knowledge >= 1
    assert report.fed_flywheel >= 1
    assert len(kb) >= 1
    # everything captured by the flywheel is marked verified (AlphaGo-Zero discipline)
    assert all(ex.source == "flywheel-verified" for ex in fw.examples())


def test_curator_keeps_producing_new_items(tmp_path: Path):
    fw = DataFlywheel(store_path=tmp_path / "s.jsonl", min_chars=1)
    cur = SyntheticCurator(flywheel=fw, cfg=SynthesisConfig(batch_size=20, seed=5))
    cur.curate()
    first = fw.count()
    cur.curate()
    # the generator RNG advances, so a second pass adds genuinely new (non-duplicate) items
    assert fw.count() > first


def test_maybe_run_disabled_returns_none(tmp_path: Path):
    cur = SyntheticCurator(cfg=SynthesisConfig(enabled=False))
    assert cur.maybe_run() is None


def test_synthetic_item_round_trips():
    item = SyntheticGenerators(seed=1).one(Domain.ARITHMETIC.value)
    assert item is not None
    d = item.to_dict()
    assert d["domain"] == "arithmetic" and "answer" in d
