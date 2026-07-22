"""NYXARA · mind/inner_critic.py — Adversarial Self-Red-Teaming + the Dialectic
(P24 · F19, organs 7 & 13).

Two engines that make her creations *fight for their lives* before release:

* :class:`InnerCritic` — the Internal Opponent. Every finished creation is attacked:
  cliché lexicon hits, vagueness density, internal contradictions, redundancy
  (repeated n-grams), unsupported claims (benefit words with no mechanism words),
  degenerate geometry for art. A creation is released only if the summed objection
  severity stays under the critic's threshold — otherwise the strongest objection is
  handed back so the engine can revise the genome *against* it.

* :class:`Dialectic` — the Hegelian loop. A passed creation is a **Thesis**; the
  Dialectic constructs its structured opposite (**Antithesis**) via an antonym map,
  then fuses both into a **Synthesis** candidate that must resolve the tension.

Mirrors the adversarial stance of :mod:`nyxara.growth.redteam` and the
generator↔discriminator pattern, specialised for creative output.
**No LLM anywhere in this module.** Pure standard library, deterministic.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

__all__ = [
    "Objection",
    "CritiqueReport",
    "InnerCritic",
    "Dialectic",
]

_CLICHES = [
    "think outside the box", "game changer", "game-changer", "paradigm shift",
    "at the end of the day", "low-hanging fruit", "synergy", "win-win",
    "next level", "cutting edge", "cutting-edge", "state of the art",
    "revolutionary breakthrough", "best of both worlds", "push the envelope",
    "seamless", "disruptive innovation", "world-class", "the possibilities are endless",
]
_VAGUE = {
    "stuff", "things", "somehow", "someway", "whatever", "various", "certain",
    "some", "maybe", "possibly", "perhaps", "generally", "basically", "very",
    "really", "quite", "nice", "good", "great", "amazing", "interesting",
}
_CONTRADICTIONS = [
    ("always", "never"), ("everything", "nothing"), ("infinite", "finite"),
    ("silent", "loud"), ("frozen", "boiling"), ("impossible", "guaranteed"),
    ("effortless", "gruelling"),
]
# benefit-claims demand mechanism words somewhere in the text
_BENEFIT = {"faster", "cheaper", "stronger", "safer", "better", "improves",
            "solves", "eliminates", "guarantees", "boosts", "unlocks"}
_MECHANISM = {"because", "by", "through", "via", "using", "drives", "converts",
              "binds", "filters", "routes", "stores", "transfers", "mechanism",
              "so", "thus", "therefore", "causes"}

# a small antonym map for building a real antithesis (both directions are registered)
_ANTONYMS = {
    "light": "dark", "dark": "light", "order": "chaos", "chaos": "order",
    "silence": "noise", "noise": "silence", "rise": "fall", "fall": "rise",
    "open": "closed", "closed": "open", "fast": "slow", "slow": "fast",
    "hot": "cold", "cold": "hot", "near": "far", "far": "near",
    "build": "break", "break": "build", "bind": "free", "free": "bind",
    "remember": "forget", "forget": "remember", "begin": "end", "end": "begin",
    "many": "few", "few": "many", "strong": "weak", "weak": "strong",
    "large": "small", "small": "large", "inside": "outside", "outside": "inside",
    "future": "past", "past": "future", "create": "destroy", "destroy": "create",
    "connect": "sever", "sever": "connect", "expand": "contract", "contract": "expand",
}


def _words(text: str) -> List[str]:
    return re.findall(r"[a-z]+", text.lower())


@dataclass
class Objection:
    """One weakness the internal opponent found."""

    kind: str          # cliche | vagueness | contradiction | redundancy | unsupported | degenerate
    detail: str
    severity: float    # in [0, 1]

    def to_dict(self) -> Dict[str, Any]:
        return {"kind": self.kind, "detail": self.detail,
                "severity": round(self.severity, 3)}


@dataclass
class CritiqueReport:
    """The full adversarial verdict on one creation."""

    objections: List[Objection] = field(default_factory=list)
    threshold: float = 0.6

    @property
    def total_severity(self) -> float:
        return sum(o.severity for o in self.objections)

    @property
    def survived(self) -> bool:
        return self.total_severity < self.threshold

    def strongest(self) -> Optional[Objection]:
        return max(self.objections, key=lambda o: o.severity, default=None)

    def to_dict(self) -> Dict[str, Any]:
        return {"survived": self.survived,
                "total_severity": round(self.total_severity, 3),
                "threshold": self.threshold,
                "objections": [o.to_dict() for o in self.objections]}


class InnerCritic:
    """The Devil's Advocate: attack every creation; only the strong are released."""

    def __init__(self, *, threshold: float = 0.6) -> None:
        self.threshold = max(0.05, min(2.0, float(threshold)))
        self.attacks = 0
        self.rejections = 0

    # ------------------------------------------------------------------ #
    def attack(self, content: str, modality: str = "idea") -> CritiqueReport:
        """Aggressively enumerate the creation's weaknesses. Deterministic."""
        self.attacks += 1
        obs: List[Objection] = []
        low = content.lower()
        ws = _words(content)

        # 1) clichés — instant, heavy hits
        for c in _CLICHES:
            if c in low:
                obs.append(Objection("cliche", f"tired phrase: {c!r}", 0.35))

        # 2) vagueness density
        if ws:
            vague = [w for w in ws if w in _VAGUE]
            density = len(vague) / len(ws)
            if density > 0.08:
                obs.append(Objection(
                    "vagueness",
                    f"{len(vague)} vague words ({', '.join(sorted(set(vague))[:4])}…) "
                    f"— {density:.0%} of the text says nothing",
                    min(1.0, density * 4.0)))

        # 3) internal contradictions (both poles asserted with no resolving connective)
        wset = set(ws)
        resolving = bool(wset & {"yet", "paradox", "tension", "between", "balances",
                                 "resolves", "reconciles"})
        for a, b in _CONTRADICTIONS:
            if a in wset and b in wset and not resolving:
                obs.append(Objection(
                    "contradiction", f"asserts both {a!r} and {b!r} without resolving them",
                    0.3))

        # 4) redundancy — repeated 3-grams (the text circles instead of advancing).
        #    Only for textual work: SVG/markup is structurally repetitive by nature.
        if modality != "art" and len(ws) >= 6:
            grams: Dict[Tuple[str, ...], int] = {}
            for i in range(len(ws) - 2):
                g = tuple(ws[i:i + 3])
                grams[g] = grams.get(g, 0) + 1
            rep = sum(c - 1 for c in grams.values() if c > 1)
            if rep > 0:
                obs.append(Objection(
                    "redundancy", f"{rep} repeated 3-gram(s) — the piece circles",
                    min(0.8, 0.2 * rep)))

        # 5) unsupported claims — benefits promised, no mechanism anywhere
        if modality == "idea" and wset & _BENEFIT and not (wset & _MECHANISM):
            claimed = ", ".join(sorted(wset & _BENEFIT)[:3])
            obs.append(Objection(
                "unsupported", f"claims [{claimed}] with no mechanism word backing it",
                0.5))

        # 6) degenerate art — collapsed geometry, single colour, off-canvas drawing
        if modality == "art":
            nums = [float(x) for x in re.findall(r"-?\d+\.?\d*", content)[:2000]]
            xs, ys = nums[0::2][:500], nums[1::2][:500]
            if len(xs) < 4:
                obs.append(Objection("degenerate", "almost no geometry on the canvas", 0.8))
            else:
                if (max(xs) - min(xs)) < 4.0 and (max(ys) - min(ys)) < 4.0:
                    obs.append(Objection("degenerate", "geometry collapsed to a point", 0.8))
                if all(v < 0 or v > 1000 for v in xs):
                    obs.append(Objection("degenerate", "drawing entirely off-canvas", 0.7))
            hues = set(re.findall(r"hsl\((\d+)", content))
            if len(hues) <= 1 and "fill" not in content.lower():
                obs.append(Objection("degenerate", "single flat colour, no palette", 0.25))

        # 7) emptiness — the cheapest trick of all
        if not ws:
            obs.append(Objection("degenerate", "no content at all", 1.0))

        report = CritiqueReport(objections=obs, threshold=self.threshold)
        if not report.survived:
            self.rejections += 1
        return report

    # ------------------------------------------------------------------ #
    def to_dict(self) -> Dict[str, Any]:
        return {"threshold": self.threshold, "attacks": self.attacks,
                "rejections": self.rejections}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "InnerCritic":
        c = cls(threshold=float(d.get("threshold", 0.6) or 0.6))
        c.attacks = int(d.get("attacks", 0) or 0)
        c.rejections = int(d.get("rejections", 0) or 0)
        return c


class Dialectic:
    """Thesis → Antithesis → Synthesis, on real text structure."""

    def antithesis(self, content: str) -> str:
        """The structured opposite: antonym-map every mappable word, invert the frame."""
        def flip(m: "re.Match[str]") -> str:
            w = m.group(0)
            anti = _ANTONYMS.get(w.lower())
            if anti is None:
                return w
            return anti.capitalize() if w[0].isupper() else anti

        flipped = re.sub(r"[A-Za-z]+", flip, content)
        return f"Counter-position: {flipped}"

    def synthesis(self, thesis: str, antithesis: str) -> str:
        """Fuse both poles into a candidate that names and resolves the tension."""
        t_ws = [w for w in _words(thesis) if len(w) > 3]
        a_ws = [w for w in _words(antithesis) if len(w) > 3]
        t_key = t_ws[0] if t_ws else "the thesis"
        a_key = next((w for w in a_ws if w not in set(t_ws)), a_ws[0] if a_ws else "its opposite")
        return (f"Synthesis — the tension between «{t_key}» and «{a_key}» resolves: "
                f"{thesis.strip().rstrip('.')}. Yet its counter-face holds too — "
                f"{antithesis.strip().rstrip('.')}. The reconciled whole keeps "
                f"{t_key} as engine and {a_key} as governor, each bounding the other.")

    def run(self, thesis: str) -> Dict[str, str]:
        anti = self.antithesis(thesis)
        return {"thesis": thesis, "antithesis": anti,
                "synthesis": self.synthesis(thesis, anti)}


# --------------------------------------------------------------------------- #
# Self-test / demo
# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # pragma: no cover
    print("=" * 70)
    print("NYXARA inner-critic + dialectic self-test")
    print("=" * 70)

    critic = InnerCritic()

    # planted flaws are caught
    bad = ("A game changer that basically improves stuff somehow. "
           "A game changer that basically improves stuff somehow.")
    rep = critic.attack(bad, "idea")
    kinds = {o.kind for o in rep.objections}
    print(f"\nflawed idea         : survived={rep.survived} kinds={sorted(kinds)}")
    assert not rep.survived
    assert {"cliche", "vagueness", "redundancy"} <= kinds
    assert "unsupported" in kinds   # 'improves' promised, no mechanism

    # clean, mechanised content passes
    good = ("A membrane valve routes heat through a copper lattice because the "
            "lattice stores charge; the stored charge drives the pump.")
    rep2 = critic.attack(good, "idea")
    print(f"clean idea          : survived={rep2.survived} "
          f"severity={rep2.total_severity:.2f}")
    assert rep2.survived

    # degenerate art is caught; real art passes
    flat = '<svg><circle cx="1" cy="1" r="0"/></svg>'
    rich = ('<svg><polyline points="100,100 180,220 300,90 210,40" stroke="hsl(10,70%,50%)"/>'
            '<polyline points="120,300 200,260 310,330" stroke="hsl(190,70%,50%)"/></svg>')
    assert not critic.attack(flat, "art").survived
    assert critic.attack(rich, "art").survived
    print("art gate            : degenerate rejected, rich released")

    # the strongest objection is actionable
    strongest = rep.strongest()
    assert strongest is not None and strongest.severity > 0
    print(f"strongest objection : [{strongest.kind}] {strongest.detail[:50]}")

    # determinism
    assert critic.attack(bad, "idea").to_dict() == critic.attack(bad, "idea").to_dict()

    # dialectic: thesis -> antithesis -> synthesis
    d = Dialectic()
    triad = d.run("The open lattice binds light and lets memory rise.")
    print(f"\nthesis              : {triad['thesis']}")
    print(f"antithesis          : {triad['antithesis']}")
    print(f"synthesis           : {triad['synthesis'][:90]}…")
    assert "closed" in triad["antithesis"] and "dark" in triad["antithesis"]
    assert "fall" in triad["antithesis"]
    # synthesis names the tension and passes the critic's contradiction probe
    srep = critic.attack(triad["synthesis"], "idea")
    assert "resolves" in triad["synthesis"]
    print(f"synthesis survives  : {srep.survived}")

    # persistence
    c2 = InnerCritic.from_dict(critic.to_dict())
    assert c2.threshold == critic.threshold and c2.attacks == critic.attacks
    print("persistence          : OK")

    print("\nALL SELF-TESTS PASSED ✓")
