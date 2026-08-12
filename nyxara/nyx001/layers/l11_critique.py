"""NYXARA · nyx001/layers/l11_critique.py — Layer 11, self-critique (✂️, output → finalize).

The charter: "Output → Critique → Find Error → Reconstruct → Finalize."

The cycle is implemented literally, with one addition that keeps it from being theatre: a critique
must **name a defect that can be checked**, and reconstruction must **measurably improve** the
named defect or be discarded. A critic that always finds something and a reconstructor that always
produces something make a loop that always runs and never improves anything. So:

* :meth:`critique` returns :class:`Finding` objects, each with a ``kind`` and a ``severity``
  derived from a measurement — not from an opinion.
* :meth:`reconstruct` applies a repair for the finding, then **re-critiques**. If the severity did
  not fall, the reconstruction is rejected and the original is kept. :attr:`Critique.improved`
  records which actually happened.
* :meth:`finalize` returns whichever version survived, and the record of why.

The four defects it can actually check are: overconfidence (stated confidence above measured
calibration), unsupported claim (no evidence), contradiction (the reasoning engine holds evidence
against it), and vacuity (an answer carrying no content). Those are checkable. It does not critique
style, tone, or "quality", because it has no measurement for them and inventing a score would make
the loop dishonest.

Honest: this improves what it can measure and is blind to everything else. A confidently-worded,
well-supported, self-consistent, entirely wrong answer passes this layer cleanly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

__all__ = ["Finding", "Critique", "SelfCritique"]


@dataclass
class Finding:
    """One checkable defect."""

    kind: str = ""
    severity: float = 0.0
    detail: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {"kind": self.kind, "severity": round(self.severity, 4), "detail": self.detail[:160]}


@dataclass
class Critique:
    """The full cycle's record: what was found, what was tried, and whether it helped."""

    original: str = ""
    final: str = ""
    findings: List[Finding] = field(default_factory=list)
    improved: bool = False
    severity_before: float = 0.0
    severity_after: float = 0.0
    rejected: int = 0

    @property
    def clean(self) -> bool:
        return not self.findings

    def to_dict(self) -> Dict[str, Any]:
        return {"improved": self.improved, "clean": self.clean,
                "severity_before": round(self.severity_before, 4),
                "severity_after": round(self.severity_after, 4),
                "rejected": self.rejected,
                "findings": [f.to_dict() for f in self.findings[:8]],
                "changed": self.original != self.final}


class SelfCritique:
    """Layer 11. Critiques an output against checkable defects, and keeps repairs that work."""

    def __init__(self, substrate: Any, *, overconfidence_slack: float = 0.15,
                 min_content_chars: int = 12, max_rounds: int = 2) -> None:
        self.substrate = substrate
        self.overconfidence_slack = float(overconfidence_slack)
        self.min_content_chars = int(min_content_chars)
        self.max_rounds = int(max_rounds)
        self.cycles = 0
        self.improvements = 0
        self.rejections = 0

    # ---- critique ---- #
    def critique(self, text: str, *, confidence: float = 0.5,
                 calibration: Optional[float] = None,
                 support: Optional[List[str]] = None,
                 contradictions: Optional[List[str]] = None) -> List[Finding]:
        """Find checkable defects. Returns an empty list when there is genuinely nothing wrong."""
        out: List[Finding] = []
        try:
            body = str(text or "").strip()
            if len(body) < self.min_content_chars:
                out.append(Finding(kind="vacuity", severity=1.0,
                                   detail=f"answer carries {len(body)} characters of content"))
            if calibration is not None and confidence > calibration + self.overconfidence_slack:
                gap = float(confidence - calibration)
                out.append(Finding(kind="overconfidence", severity=min(1.0, gap * 2.0),
                                   detail=f"stated {confidence:.2f} against measured "
                                          f"calibration {calibration:.2f}"))
            if not support:
                out.append(Finding(kind="unsupported", severity=0.5,
                                   detail="no evidence attached to the claim"))
            if contradictions:
                out.append(Finding(kind="contradicted", severity=min(1.0, 0.3 * len(contradictions)),
                                   detail=f"{len(contradictions)} pieces of evidence against"))
            return out
        except Exception:  # noqa: BLE001
            return out

    @staticmethod
    def severity(findings: List[Finding]) -> float:
        """Total severity — what a reconstruction has to actually lower."""
        return float(sum(f.severity for f in findings or []))

    # ---- reconstruct ---- #
    def reconstruct(self, text: str, findings: List[Finding], *,
                    confidence: float = 0.5,
                    calibration: Optional[float] = None) -> str:
        """Repair the named defects. Only the defects — this does not rewrite for taste."""
        body = str(text or "")
        kinds = {f.kind for f in findings or []}
        try:
            if "overconfidence" in kinds and calibration is not None:
                body = (f"{body.rstrip()}\n\n(Confidence lowered to {calibration:.2f}: my measured "
                        f"calibration does not support {confidence:.2f}.)")
            if "unsupported" in kinds:
                body = f"{body.rstrip()}\n\n(No supporting evidence — this is a conjecture.)"
            if "contradicted" in kinds:
                body = f"{body.rstrip()}\n\n(I hold evidence against this; treat it as contested.)"
            if "vacuity" in kinds and not body.strip():
                body = "I have nothing substantive to say here."
            return body
        except Exception:  # noqa: BLE001
            return str(text or "")

    # ---- the cycle ---- #
    def run(self, text: str, *, confidence: float = 0.5, calibration: Optional[float] = None,
            support: Optional[List[str]] = None,
            contradictions: Optional[List[str]] = None) -> Critique:
        """Output → critique → reconstruct → re-critique → keep only if it improved."""
        self.cycles += 1
        out = Critique(original=str(text or ""), final=str(text or ""))
        try:
            findings = self.critique(text, confidence=confidence, calibration=calibration,
                                     support=support, contradictions=contradictions)
            out.findings = findings
            out.severity_before = out.severity_after = self.severity(findings)
            if not findings:
                return out
            current = out.original
            for _round in range(max(1, self.max_rounds)):
                candidate = self.reconstruct(current, findings, confidence=confidence,
                                             calibration=calibration)
                after = self.critique(candidate, confidence=calibration or confidence,
                                      calibration=calibration,
                                      support=support, contradictions=contradictions)
                sev_after = self.severity(after)
                if sev_after < out.severity_after:
                    current = candidate
                    out.severity_after = sev_after
                    out.improved = True
                    findings = after
                    if not after:
                        break
                else:
                    # The repair did not lower a measured severity, so it is discarded. This is
                    # the branch that stops the loop being theatre: a reconstruction that only
                    # *looks* better is not kept.
                    out.rejected += 1
                    self.rejections += 1
                    break
            out.final = current
            if out.improved:
                self.improvements += 1
            return out
        except Exception:  # noqa: BLE001 — a failed critique leaves the original untouched
            return out

    def stats(self) -> Dict[str, Any]:
        return {"cycles": self.cycles, "improvements": self.improvements,
                "rejections": self.rejections,
                "improvement_rate": round(self.improvements / self.cycles, 4)
                if self.cycles else None}

    def to_dict(self) -> Dict[str, Any]:
        return {"cycles": self.cycles, "improvements": self.improvements,
                "rejections": self.rejections}

    def load_dict(self, d: Dict[str, Any]) -> None:
        try:
            self.cycles = int(d.get("cycles", 0))
            self.improvements = int(d.get("improvements", 0))
            self.rejections = int(d.get("rejections", 0))
        except Exception:  # noqa: BLE001
            pass
