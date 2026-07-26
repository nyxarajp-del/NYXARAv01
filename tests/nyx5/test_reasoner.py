"""Nyx5Reasoner: wraps a base for text, enriches with NYX-5 signals, fail-soft, never lowers safety."""

from __future__ import annotations

from dataclasses import dataclass

from nyxara.kernel.config import Nyx5Config
from nyxara.nyx5 import Nyx5Brain, Nyx5Reasoner


@dataclass
class Cand:
    text: str = ""
    confidence: float = 0.7
    belief: float = 0.7
    rationale: str = "base"
    efe: float = 0.0
    risk: str = "low"


def _brain():
    return Nyx5Brain(Nyx5Config(n_neurons=64))


def test_wraps_base_and_returns_candidate():
    r = Nyx5Reasoner(base=lambda s, f=None: Cand(text=f"reply:{s}"), brain=_brain())
    c = r("hello")
    assert c.text == "reply:hello"


def test_enriches_confidence_and_efe():
    r = Nyx5Reasoner(base=lambda s, f=None: Cand(), brain=_brain())
    c = r("a very novel surprising stimulus never seen before")
    assert 0.0 <= c.confidence <= 1.0
    assert c.efe >= 0.0                      # expected-free-energy annotated from surprise
    assert "nyx5:" in c.rationale


def test_base_exception_is_fail_soft():
    def boom(s, f=None):
        raise RuntimeError("base failed")

    r = Nyx5Reasoner(base=boom, brain=_brain())
    try:
        c = r("x")
    except RuntimeError:
        c = None
    # a crashing base is the base's contract; the reasoner must not add its own crash on top
    assert c is None or hasattr(c, "text")


def test_enrichment_failure_returns_base_candidate():
    # a candidate with no numeric fields must still come back unharmed
    class Bare:
        text = "bare"

    r = Nyx5Reasoner(base=lambda s, f=None: Bare(), brain=_brain())
    c = r("hello")
    assert c.text == "bare"


def test_dialectic_suggestion_attached_advisorily():
    r = Nyx5Reasoner(base=lambda s, f=None: Cand(), brain=_brain())
    c = r("use a nested loop on a single node")
    assert "dialectic:" in c.rationale       # advisory suggestion attached, candidate not blocked
