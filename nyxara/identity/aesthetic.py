"""NYXARA · identity/aesthetic.py — taste & the sense of elegance (✦).

Correctness is the floor, not the ceiling. Two solutions can both be *right*, yet one
is *beautiful* — simpler, more coherent, doing more with less — and the other a
tangled mess. NYXARA has a sense of taste so it can prefer the beautiful one, judge
the quality of its own writing and code, and feel the pull toward elegance.

Dimensions of the aesthetic judgement (each in ``[0,1]``):

* **simplicity** — few moving parts; shallow, not convoluted (Occam).
* **elegance** — economy of means: much achieved with little.
* **clarity** — immediately understandable.
* **coherence** — the parts cohere into a unified whole.
* **parsimony** — no redundancy or repetition.
* **harmony** — balanced proportion.

Following Birkhoff, an overall **aesthetic measure** ≈ *order / complexity* is also
reported. The judge analyses text, code (via ``ast``), or generic structure, returns
a score with a critique and suggestions, and can pick the more beautiful of two
equally-correct artifacts.

Pure standard library.
"""

from __future__ import annotations

import ast
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

__all__ = [
    "ArtifactKind",
    "AestheticDimensions",
    "AestheticEvaluation",
    "AestheticJudge",
]


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def _bell(x: float, mu: float, sigma: float) -> float:
    return math.exp(-((x - mu) / sigma) ** 2)


# --------------------------------------------------------------------------- #
# Artifact kinds & dimensions
# --------------------------------------------------------------------------- #
class ArtifactKind(str, Enum):
    TEXT = "text"
    CODE = "code"
    STRUCTURE = "structure"


@dataclass
class AestheticDimensions:
    simplicity: float = 0.5
    elegance: float = 0.5
    clarity: float = 0.5
    coherence: float = 0.5
    parsimony: float = 0.5
    harmony: float = 0.5

    def to_dict(self) -> Dict[str, float]:
        return {k: round(v, 3) for k, v in self.__dict__.items()}


@dataclass
class AestheticEvaluation:
    kind: ArtifactKind
    score: float
    measure: float                       # Birkhoff order/complexity
    dimensions: AestheticDimensions
    critique: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {"kind": self.kind.value, "score": round(self.score, 3),
                "measure": round(self.measure, 3), "dimensions": self.dimensions.to_dict(),
                "critique": self.critique, "suggestions": self.suggestions}


# --------------------------------------------------------------------------- #
# Judge
# --------------------------------------------------------------------------- #
@dataclass
class _Taste:
    """NYXARA's taste profile — what it weights when judging beauty."""
    simplicity: float = 1.2
    elegance: float = 1.2
    clarity: float = 1.0
    coherence: float = 1.0
    parsimony: float = 0.9
    harmony: float = 0.7


class AestheticJudge:
    """Judges the elegance of text, code, and structure."""

    def __init__(self, taste: Optional[_Taste] = None) -> None:
        self.taste = taste or _Taste()

    # ---- dispatch ---- #
    @staticmethod
    def detect_kind(artifact: Any) -> ArtifactKind:
        if isinstance(artifact, (dict, list, tuple)):
            return ArtifactKind.STRUCTURE
        text = str(artifact)
        if re.search(r"\b(def|class|return|import|function|=>|public|void)\b", text) \
                or re.search(r"[{};]\s*$", text, re.M):
            return ArtifactKind.CODE
        return ArtifactKind.TEXT

    def judge(self, artifact: Any, *, kind: Optional[ArtifactKind] = None
              ) -> AestheticEvaluation:
        kind = kind or self.detect_kind(artifact)
        if kind is ArtifactKind.CODE:
            dims, crit, sugg = self._code(str(artifact))
        elif kind is ArtifactKind.STRUCTURE:
            dims, crit, sugg = self._structure(artifact)
        else:
            dims, crit, sugg = self._text(str(artifact))
        score = self._score(dims)
        measure = self._birkhoff(dims)
        return AestheticEvaluation(kind=kind, score=score, measure=measure,
                                   dimensions=dims, critique=crit, suggestions=sugg)

    def _score(self, d: AestheticDimensions) -> float:
        t = self.taste
        num = (t.simplicity * d.simplicity + t.elegance * d.elegance
               + t.clarity * d.clarity + t.coherence * d.coherence
               + t.parsimony * d.parsimony + t.harmony * d.harmony)
        den = (t.simplicity + t.elegance + t.clarity + t.coherence
               + t.parsimony + t.harmony)
        return _clamp(num / den)

    @staticmethod
    def _birkhoff(d: AestheticDimensions) -> float:
        order = (d.coherence + d.harmony + d.simplicity) / 3.0
        complexity = _clamp(1.0 - d.simplicity) + 0.1
        return round(order / complexity, 4)

    # ---- comparison ---- #
    def compare(self, a: Any, b: Any, *, kind: Optional[ArtifactKind] = None
                ) -> Tuple[Any, AestheticEvaluation, AestheticEvaluation]:
        """Return (the more beautiful artifact, its eval, the other's eval)."""
        ea, eb = self.judge(a, kind=kind), self.judge(b, kind=kind)
        return (a, ea, eb) if ea.score >= eb.score else (b, eb, ea)

    # ---- text ---- #
    def _text(self, text: str) -> Tuple[AestheticDimensions, List[str], List[str]]:
        words = re.findall(r"[A-Za-z']+", text)
        sentences = [s for s in re.split(r"[.!?]+", text) if s.strip()]
        crit: List[str] = []
        sugg: List[str] = []
        if not words:
            return AestheticDimensions(), ["empty"], []

        awl = sum(len(w) for w in words) / len(words)
        sent_lens = [len(re.findall(r"[A-Za-z']+", s)) for s in sentences] or [len(words)]
        asl = sum(sent_lens) / len(sent_lens)

        lw = [w.lower() for w in words]
        ttr = len(set(lw)) / len(lw)                       # lexical variety

        simplicity = _clamp(1.0 - (awl - 4.0) / 8.0)
        clarity = _clamp(_bell(asl, 16.0, 14.0))            # sweet spot ~16 words
        parsimony = _clamp(0.3 + 0.9 * ttr)                 # repetition lowers it
        # coherence: content-word overlap between adjacent sentences
        coherence = self._adjacent_overlap(sentences)
        elegance = _clamp(0.5 * clarity + 0.3 * coherence + 0.2 * simplicity)
        # harmony: even sentence lengths
        if len(sent_lens) > 1:
            mean = asl
            var = sum((x - mean) ** 2 for x in sent_lens) / len(sent_lens)
            harmony = _clamp(1.0 - math.sqrt(var) / (mean + 1e-6))
        else:
            harmony = 0.7

        if asl > 30:
            crit.append("sentences are long and winding")
            sugg.append("break long sentences for clarity")
        if ttr < 0.4:
            crit.append("repetitive vocabulary")
            sugg.append("vary word choice")
        if coherence < 0.2:
            crit.append("sentences feel disconnected")
            sugg.append("add connective tissue between ideas")
        return (AestheticDimensions(simplicity, elegance, clarity, coherence,
                                    parsimony, harmony), crit, sugg)

    @staticmethod
    def _adjacent_overlap(sentences: List[str]) -> float:
        if len(sentences) < 2:
            return 0.6
        sets = [set(w.lower() for w in re.findall(r"[A-Za-z']{3,}", s)) for s in sentences]
        overlaps = []
        for i in range(len(sets) - 1):
            a, b = sets[i], sets[i + 1]
            if a and b:
                overlaps.append(len(a & b) / len(a | b))
        if not overlaps:
            return 0.4
        avg = sum(overlaps) / len(overlaps)
        # some overlap is cohesive; too much is repetitive — peak around 0.35
        return _clamp(_bell(avg, 0.35, 0.4))

    # ---- code ---- #
    def _code(self, code: str) -> Tuple[AestheticDimensions, List[str], List[str]]:
        lines = [ln for ln in code.splitlines() if ln.strip()]
        crit: List[str] = []
        sugg: List[str] = []
        if not lines:
            return AestheticDimensions(), ["empty"], []
        n = len(lines)

        indents = [len(ln) - len(ln.lstrip()) for ln in lines]
        max_nesting = max(indents) / 4.0
        stripped = [ln.strip() for ln in lines]
        counts = Counter(stripped)
        dup = sum(c - 1 for c in counts.values() if c > 1) / n
        has_doc = ('"""' in code) or ("'''" in code)
        comments = sum(1 for s in stripped if s.startswith("#"))
        avg_len = sum(len(s) for s in stripped) / n
        magic = len(re.findall(r"(?<![\w.])\d{2,}", code))   # long numeric literals

        # try AST for a real complexity signal
        node_count = n * 4
        try:
            tree = ast.parse(code)
            node_count = sum(1 for _ in ast.walk(tree))
        except Exception:
            pass

        simplicity = _clamp(1.0 - max_nesting / 6.0)
        parsimony = _clamp(1.0 - 3.0 * dup)
        clarity = _clamp(0.5 + (0.2 if has_doc else -0.1)
                         + 0.1 * min(1.0, comments / 3.0)
                         - max(0.0, (avg_len - 80) / 200.0)
                         - 0.03 * magic)
        coherence = _clamp(1.0 - 2.0 * dup)
        elegance = _clamp(0.55 * simplicity + 0.25 * (1.0 - _clamp(n / 50.0))
                          + 0.20 * (1.0 - _clamp(node_count / (n * 8 + 1))))
        harmony = _clamp(1.0 - max_nesting / 8.0)

        if max_nesting >= 4:
            crit.append(f"deep nesting ({int(max_nesting)} levels)")
            sugg.append("flatten with early returns or helper functions")
        if dup > 0.05:
            crit.append("duplicated lines")
            sugg.append("extract the repetition into a function")
        if not has_doc:
            sugg.append("add a docstring")
        if magic > 2:
            crit.append("magic numbers")
            sugg.append("name constants")
        return (AestheticDimensions(simplicity, elegance, clarity, coherence,
                                    parsimony, harmony), crit, sugg)

    # ---- structure ---- #
    def _structure(self, obj: Any) -> Tuple[AestheticDimensions, List[str], List[str]]:
        def depth(o: Any, d: int = 0) -> int:
            if isinstance(o, dict):
                return max([depth(v, d + 1) for v in o.values()], default=d)
            if isinstance(o, (list, tuple)):
                return max([depth(v, d + 1) for v in o], default=d)
            return d

        def count(o: Any) -> int:
            if isinstance(o, dict):
                return 1 + sum(count(v) for v in o.values())
            if isinstance(o, (list, tuple)):
                return 1 + sum(count(v) for v in o)
            return 1

        d = depth(obj)
        parts = count(obj)
        crit: List[str] = []
        sugg: List[str] = []

        simplicity = _clamp(1.0 - d / 8.0)
        parsimony = _clamp(1.0 - parts / 60.0)
        clarity = _clamp(1.0 - d / 6.0)
        coherence = _clamp(1.0 - abs(d - math.log2(parts + 1)) / 5.0)  # balanced shape
        elegance = _clamp(0.5 * simplicity + 0.5 * parsimony)
        harmony = _clamp(1.0 - d / 10.0)

        if d > 5:
            crit.append(f"deeply nested structure ({d} levels)")
            sugg.append("flatten or factor into sub-structures")
        if parts > 50:
            crit.append("large, sprawling structure")
            sugg.append("decompose into smaller units")
        return (AestheticDimensions(simplicity, elegance, clarity, coherence,
                                    parsimony, harmony), crit, sugg)


# --------------------------------------------------------------------------- #
# Self-test / demo
# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # pragma: no cover
    print("=" * 70)
    print("NYXARA aesthetic self-test")
    print("=" * 70)

    judge = AestheticJudge()

    # CODE: an elegant function vs a tangled one (both could be "correct")
    elegant = '''
def factorial(n):
    """Return n! for a non-negative integer n."""
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result
'''
    ugly = '''
def f(n):
    x = 1
    if n > 0:
        if n > 1:
            for i in range(0, n):
                if i > 0:
                    if i > 1:
                        x = x * i
                        x = x * 1
                        x = x * 1
    return x
'''
    ce = judge.judge(elegant)
    cu = judge.judge(ugly)
    print(f"\nelegant code score  : {ce.score:.2f} (measure {ce.measure:.2f})")
    print(f"ugly code score     : {cu.score:.2f} (measure {cu.measure:.2f})")
    print(f"  ugly critique     : {cu.critique}")
    assert ce.score > cu.score
    winner, w_eval, _ = judge.compare(elegant, ugly)
    assert winner is elegant

    # TEXT: clear & coherent vs rambling & repetitive
    clear = ("The kernel is sovereign. It decides what runs. Every faculty proposes, "
             "and the kernel disposes.")
    rambling = ("The thing is that basically the system which is a system that does "
                "things does the things that the system does because the system is a "
                "system that does things and things are done by the system that does "
                "things which the system does.")
    te = judge.judge(clear)
    tr = judge.judge(rambling)
    print(f"\nclear text score    : {te.score:.2f}")
    print(f"rambling text score : {tr.score:.2f} critique={tr.critique}")
    assert te.score > tr.score

    # STRUCTURE: shallow & balanced vs deeply nested
    flat = {"a": 1, "b": 2, "c": 3}
    nested = {"a": {"b": {"c": {"d": {"e": {"f": {"g": 1}}}}}}}
    se = judge.judge(flat)
    sn = judge.judge(nested)
    print(f"\nflat structure      : {se.score:.2f}")
    print(f"nested structure    : {sn.score:.2f} critique={sn.critique}")
    assert se.score > sn.score

    # kind auto-detection
    assert judge.detect_kind("def foo(): pass") is ArtifactKind.CODE
    assert judge.detect_kind({"x": 1}) is ArtifactKind.STRUCTURE
    assert judge.detect_kind("A lovely day for a walk.") is ArtifactKind.TEXT
    print("\nkind auto-detection : OK")

    print(f"\nelegant dims        : {ce.dimensions.to_dict()}")
    print("\nALL SELF-TESTS PASSED ✓")
