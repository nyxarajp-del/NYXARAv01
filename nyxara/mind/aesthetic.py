"""NYXARA · mind/aesthetic.py — the Aesthetic Feedback Loop (P24 · F19, organ 5).

Computational aesthetics with a *learnable* taste. Every creation (idea / verse / art)
is measured on real, deterministic features — alliteration and rhythm for verse,
symmetry / golden-ratio proximity / hue harmony for SVG art, concreteness and tension
for ideas — and folded into one **resonance** score in ``[0, 1]``.

The judge is not frozen: :meth:`AestheticJudge.learn` performs an online
exponential-moving-average update of the per-feature weights from a reward signal —
implicit (a kept creation's measured novelty) or explicit (the Master's ``/rate``).
Her taste therefore *evolves* with experience, and the weights persist.

**No LLM anywhere in this module.** Pure standard library, seed-free and deterministic:
the same content always earns the same features.
"""

from __future__ import annotations

import math
import re
from typing import Any, Dict, List, Tuple

__all__ = [
    "AestheticJudge",
    "GOLDEN_RATIO",
]

GOLDEN_RATIO = (1.0 + math.sqrt(5.0)) / 2.0

_VOWELS = set("aeiou")

# words that signal groundedness (concreteness) vs abstraction, and creative tension
_CONCRETE = {
    "stone", "water", "light", "wire", "gear", "seed", "river", "mirror", "clock",
    "bridge", "thread", "storm", "lens", "key", "fire", "shadow", "echo", "compass",
    "mask", "magnet", "swarm", "engine", "valve", "spring", "glass", "iron", "root",
    "wing", "blade", "coil", "wheel", "lattice", "membrane", "circuit", "crystal",
}
_TENSION_PAIRS = [
    ("light", "dark"), ("order", "chaos"), ("silence", "storm"), ("rise", "fall"),
    ("open", "closed"), ("fast", "slow"), ("hot", "cold"), ("near", "far"),
    ("build", "break"), ("bind", "free"), ("remember", "forget"), ("begin", "end"),
]


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def _words(text: str) -> List[str]:
    return re.findall(r"[a-z]+", text.lower())


def _shannon(counts: Dict[Any, int]) -> float:
    total = sum(counts.values())
    if total <= 0:
        return 0.0
    h = 0.0
    for c in counts.values():
        if c > 0:
            p = c / total
            h -= p * math.log2(p)
    return h


class AestheticJudge:
    """Deterministic aesthetic features + an online-learned weighting = evolving taste."""

    #: the feature vocabulary, one weight per feature; weights stay in [0.2, 3.0]
    FEATURES: Tuple[str, ...] = (
        "alliteration", "assonance", "rhythm", "surprise", "imagery",
        "symmetry", "golden", "harmony", "balance",
        "concreteness", "tension", "coherence",
    )

    def __init__(self, *, learning_rate: float = 0.15) -> None:
        self.learning_rate = float(learning_rate)
        self.weights: Dict[str, float] = {f: 1.0 for f in self.FEATURES}
        self.ratings_seen: int = 0

    # ------------------------------------------------------------------ #
    # features per modality (all deterministic, all in [0,1])
    # ------------------------------------------------------------------ #
    def features(self, content: str, modality: str) -> Dict[str, float]:
        if modality == "verse":
            return self._verse_features(content)
        if modality == "art":
            return self._art_features(content)
        return self._idea_features(content)

    # ---- verse ---- #
    def _verse_features(self, text: str) -> Dict[str, float]:
        lines = [ln for ln in text.splitlines() if _words(ln)]
        ws = _words(text)
        if not ws:
            return {"alliteration": 0.0, "assonance": 0.0, "rhythm": 0.0,
                    "surprise": 0.0, "imagery": 0.0}
        # alliteration: *different* adjacent words sharing an initial consonant
        # (repeating the same word is not craft — identical pairs never count)
        allit = sum(1 for a, b in zip(ws, ws[1:])
                    if a != b and a[0] == b[0] and a[0] not in _VOWELS)
        # assonance: different adjacent words sharing a dominant vowel
        def dom_vowel(w: str) -> str:
            for ch in w:
                if ch in _VOWELS:
                    return ch
            return ""
        asson = sum(1 for a, b in zip(ws, ws[1:])
                    if a != b and dom_vowel(a) and dom_vowel(a) == dom_vowel(b))
        # rhythm: regularity of line lengths (low variance == steady meter)
        lens = [len(_words(ln)) for ln in lines] or [len(ws)]
        mean = sum(lens) / len(lens)
        var = sum((x - mean) ** 2 for x in lens) / len(lens)
        rhythm = 1.0 / (1.0 + var)
        # lexical surprise: type/token ratio (repetition is cheap; variety surprises)
        surprise = len(set(ws)) / len(ws)
        imagery = len([w for w in ws if w in _CONCRETE]) / len(ws)
        return {
            "alliteration": _clamp(allit / max(1, len(ws) - 1) * 3.0),
            "assonance": _clamp(asson / max(1, len(ws) - 1) * 2.5),
            "rhythm": _clamp(rhythm),
            "surprise": _clamp(surprise),
            "imagery": _clamp(imagery * 4.0),
        }

    # ---- art (SVG source) ---- #
    def _art_features(self, svg: str) -> Dict[str, float]:
        # numeric coordinates drawn on the canvas
        nums = [float(x) for x in re.findall(r"-?\d+\.?\d*", svg)[:4000]]
        xs = nums[0::2][:1000]
        ys = nums[1::2][:1000]
        symmetry = 0.0
        golden = 0.0
        balance = 0.0
        if xs and ys:
            cx = sum(xs) / len(xs)
            cy = sum(ys) / len(ys)
            # mirror symmetry: how well the point cloud folds onto itself about the centre
            spread_x = sum(abs(x - cx) for x in xs) / len(xs)
            fold = [x for x in xs if x <= cx]
            unfold = [2 * cx - x for x in xs if x > cx]
            if fold and unfold and spread_x > 1e-9:
                fm = sum(fold) / len(fold)
                um = sum(unfold) / len(unfold)
                symmetry = _clamp(1.0 - abs(fm - um) / (spread_x + 1e-9))
            # golden-ratio proximity of the drawing's bounding proportions
            w = max(xs) - min(xs)
            h = max(ys) - min(ys)
            if w > 1e-9 and h > 1e-9:
                ratio = max(w, h) / min(w, h)
                golden = _clamp(1.0 - abs(ratio - GOLDEN_RATIO) / GOLDEN_RATIO)
            # balance: the centroid should not be pinned to a corner of the canvas
            span_x = (max(xs) - min(xs)) or 1.0
            span_y = (max(ys) - min(ys)) or 1.0
            off = math.hypot((cx - (min(xs) + span_x / 2)) / span_x,
                             (cy - (min(ys) + span_y / 2)) / span_y)
            balance = _clamp(1.0 - off)
        # hue harmony: hues present should be well-spaced, not one flat colour
        hues = [int(h) % 360 for h in re.findall(r"hsl\((\d+)", svg)]
        if len(set(hues)) > 1:
            uniq = sorted(set(hues))
            gaps = [(b - a) for a, b in zip(uniq, uniq[1:])] + [360 - uniq[-1] + uniq[0]]
            ideal = 360.0 / len(uniq)
            harmony = _clamp(1.0 - sum(abs(g - ideal) for g in gaps) / 360.0)
        elif hues:
            harmony = 0.35   # monochrome: coherent but flat
        else:
            harmony = 0.0
        return {"symmetry": symmetry, "golden": golden,
                "harmony": harmony, "balance": balance}

    # ---- idea / invention ---- #
    def _idea_features(self, text: str) -> Dict[str, float]:
        ws = _words(text)
        if not ws:
            return {"concreteness": 0.0, "tension": 0.0, "coherence": 0.0, "surprise": 0.0}
        wset = set(ws)
        concreteness = len([w for w in ws if w in _CONCRETE]) / len(ws)
        tension = sum(1 for a, b in _TENSION_PAIRS if a in wset and b in wset)
        # coherence: sentences share vocabulary with their neighbours (a thread runs through)
        sents = [set(_words(s)) for s in re.split(r"[.!?\n]+", text) if _words(s)]
        overlaps = [len(a & b) / max(1, min(len(a), len(b)))
                    for a, b in zip(sents, sents[1:])]
        coherence = sum(overlaps) / len(overlaps) if overlaps else 0.5
        surprise = len(wset) / len(ws)
        return {"concreteness": _clamp(concreteness * 5.0),
                "tension": _clamp(tension / 2.0),
                "coherence": _clamp(coherence),
                "surprise": _clamp(surprise)}

    # ------------------------------------------------------------------ #
    # the score, and the taste that learns
    # ------------------------------------------------------------------ #
    def resonance(self, content: str, modality: str) -> float:
        """Weighted aesthetic resonance in ``[0, 1]`` under the *current* taste."""
        feats = self.features(content, modality)
        if not feats:
            return 0.0
        num = sum(self.weights.get(f, 1.0) * v for f, v in feats.items())
        den = sum(self.weights.get(f, 1.0) for f in feats)
        return _clamp(num / den if den > 0 else 0.0)

    def learn(self, content: str, modality: str, signal: float) -> Dict[str, float]:
        """Move the taste toward what earned ``signal`` (reward in ``[0, 1]``).

        Features present in a *well*-received piece gain weight; features present in a
        *poorly*-received piece lose it — a real online EMA update, clamped to sane
        bounds so no single rating can capsize her taste. Returns the updated weights."""
        signal = _clamp(float(signal))
        feats = self.features(content, modality)
        for f, v in feats.items():
            # centred reward: >0.5 praises the features present, <0.5 penalises them
            delta = self.learning_rate * (signal - 0.5) * v
            self.weights[f] = _clamp(self.weights.get(f, 1.0) + delta, 0.2, 3.0)
        self.ratings_seen += 1
        return dict(self.weights)

    # ------------------------------------------------------------------ #
    # persistence
    # ------------------------------------------------------------------ #
    def to_dict(self) -> Dict[str, Any]:
        return {"weights": {k: round(v, 6) for k, v in self.weights.items()},
                "ratings_seen": self.ratings_seen,
                "learning_rate": self.learning_rate}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "AestheticJudge":
        judge = cls(learning_rate=float(d.get("learning_rate", 0.15) or 0.15))
        for k, v in (d.get("weights", {}) or {}).items():
            if k in judge.weights:
                judge.weights[k] = _clamp(float(v), 0.2, 3.0)
        judge.ratings_seen = int(d.get("ratings_seen", 0) or 0)
        return judge


# --------------------------------------------------------------------------- #
# Self-test / demo
# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # pragma: no cover
    print("=" * 70)
    print("NYXARA aesthetic-judge self-test")
    print("=" * 70)

    judge = AestheticJudge()

    verse = ("Silver river, silver stone —\n"
             "the storm remembers light,\n"
             "the shadow keeps the fire's key,\n"
             "and silence ends the night.")
    flat = "the the the the\nthe the the the"
    rv = judge.resonance(verse, "verse")
    rf = judge.resonance(flat, "verse")
    print(f"\nverse resonance      : crafted={rv:.3f}  flat={rf:.3f}")
    assert rv > rf

    sym = ('<svg><polyline points="100,100 200,150 300,100 200,50 100,100" '
           'stroke="hsl(0,70%,50%)"/><polyline points="100,300 200,350 300,300" '
           'stroke="hsl(120,70%,50%)"/><path stroke="hsl(240,70%,50%)" d="M 150 200 L 250 200"/></svg>')
    lop = ('<svg><polyline points="10,10 11,12 10,14 12,10 11,11" stroke="hsl(7,70%,50%)"/></svg>')
    sa = judge.resonance(sym, "art")
    sl = judge.resonance(lop, "art")
    print(f"art resonance        : balanced={sa:.3f}  lopsided={sl:.3f}")
    assert sa > sl

    idea = ("A mirror-lattice engine: the circuit binds light and dark through a "
            "crystal membrane. The membrane filters the storm; the filtered storm "
            "drives the wheel.")
    vague = "A thing that does stuff in some way that might be good somehow."
    ia = judge.resonance(idea, "idea")
    iv = judge.resonance(vague, "idea")
    print(f"idea resonance       : grounded={ia:.3f}  vague={iv:.3f}")
    assert ia > iv

    # determinism
    assert judge.resonance(verse, "verse") == rv

    # learning: praising imagery-rich verse raises the imagery weight
    w0 = judge.weights["imagery"]
    for _ in range(5):
        judge.learn(verse, "verse", 1.0)
    assert judge.weights["imagery"] > w0
    # panning it pulls the weight back down
    w1 = judge.weights["imagery"]
    for _ in range(5):
        judge.learn(verse, "verse", 0.0)
    assert judge.weights["imagery"] < w1
    print(f"taste evolves        : imagery weight {w0:.3f} -> {w1:.3f} -> "
          f"{judge.weights['imagery']:.3f}")

    # weights stay bounded
    for _ in range(200):
        judge.learn(verse, "verse", 1.0)
    assert all(0.2 <= w <= 3.0 for w in judge.weights.values())

    # persistence round-trip
    j2 = AestheticJudge.from_dict(judge.to_dict())
    assert j2.weights == judge.weights and j2.ratings_seen == judge.ratings_seen
    print("persistence          : OK")

    print("\nALL SELF-TESTS PASSED ✓")
