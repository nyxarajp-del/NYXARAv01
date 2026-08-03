"""NYXARA · growth/epistemic_drive.py — L-AGENCY: one hunger out of five numbers (🜂).

NYXARA already measures her own ignorance in five separate places, and every one of them is real:

* :meth:`~nyxara.planning.intent.IntentSystem._epistemic_for` — expected-free-energy epistemic
  value, computed over the world model's own actions.
* ``Question.uncertainty`` in :mod:`~nyxara.growth.active_curiosity` — how unsure she is of an
  answer she is actively chasing.
* ``world_model.mean_epistemic()`` — the predictive model's own uncertainty.
* :class:`~nyxara.growth.curiosity_nets.RNDNovelty` — random-network-distillation novelty, with
  real numpy backprop.
* ``Weakness.severity`` in :mod:`~nyxara.growth.weakness` — normalised, ranked, with a
  plain-language remediation attached.

The problem was never that these are weak. It is that they are five **unrelated numbers on five
different scales with no arbiter**, so nothing could ever ask the one question that matters: *what
am I most ignorant about right now, and what should I go read?*

This module is that arbiter. It normalises each signal to ``[0, 1]``, fuses them into a single
:class:`DrivePressure`, and — more usefully — turns them into a **ranked list of subjects**.

It also closes the gap that made the whole acquisition loop epistemically blind.
:func:`~nyxara.growth.acquire.gap_topics` lists ``weakness_report`` as its highest-priority input,
but neither call site ever passed one (``autolearn.py`` and ``foundry.py`` both omit it), so the
strongest signal she had was dead on arrival and topic selection fell through to memory regexes and
a static default battery. With a drive attached, "go learn something" is finally driven by measured
ignorance instead of a fallback list.

Every reader is individually fail-soft: a missing faculty contributes nothing rather than raising,
so the drive works on a bare install and gets sharper as more of her is wired up.

Depends on ``growth/weakness``, ``growth/signal_bus``, ``growth/curriculum`` and (optionally) a
live core. Nothing imports back — :mod:`~nyxara.growth.acquire` reaches *here* lazily.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

__all__ = ["Gap", "DrivePressure", "EpistemicDrive"]

_SOURCE_WEIGHTS: Dict[str, float] = {
    # A measured weakness is the sharpest signal she has: it names the thing and rates it.
    "weakness": 1.0,
    # An open question she is already chasing — she chose it, so it carries her own priority.
    "curiosity": 0.9,
    # The predictive model saying it cannot predict a region.
    "world_model": 0.7,
    # Ambient focus from the signal bus — what the rest of her has been talking about.
    "signal": 0.5,
    # Raw novelty: real, but the least *directed* of the five.
    "novelty": 0.4,
}


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


# Weakness sources that describe HER OWN CODE rather than something she could go read about.
# "layering violation: nyxara.agency.llm_tool → nyxara.growth.foundry" is a real, correctly-ranked
# weakness — and a useless web-search query. Acquisition needs knowledge gaps; self-improvement
# needs engineering ones. Conflating them sends the acquirer looking for her own module names.
_ENGINEERING_SOURCES = frozenset({"architecture", "code", "review", "lint", "profiler"})


@dataclass
class Gap:
    """One thing she does not know, with how much it presses and where that came from."""

    subject: str
    pressure: float = 0.0
    source: str = ""
    detail: str = ""
    kind: str = "knowledge"          # "knowledge" (go read about it) | "engineering" (go fix it)

    @property
    def searchable(self) -> bool:
        """Is this a subject worth sending to a web search, as opposed to a repair task?"""
        return self.kind == "knowledge"

    def to_dict(self) -> Dict[str, Any]:
        return {"subject": self.subject, "pressure": round(self.pressure, 4),
                "source": self.source, "detail": self.detail, "kind": self.kind}


@dataclass
class DrivePressure:
    """The fused hunger, and every component that produced it — never just the total."""

    weakness: float = 0.0
    curiosity: float = 0.0
    world_model: float = 0.0
    novelty: float = 0.0
    signal: float = 0.0
    contributors: List[str] = field(default_factory=list)

    @property
    def total(self) -> float:
        """A weighted mean over the signals that actually reported. 0.0 when none did.

        A mean over *reporting* signals rather than over all five, so a bare install with one live
        faculty is not diluted to near-zero and reported as contentment."""
        parts: List[Tuple[float, float]] = []
        for name in ("weakness", "curiosity", "world_model", "novelty", "signal"):
            if name in self.contributors:
                parts.append((getattr(self, name), _SOURCE_WEIGHTS.get(name, 0.5)))
        if not parts:
            return 0.0
        total_weight = sum(w for _v, w in parts)
        return _clamp01(sum(v * w for v, w in parts) / max(total_weight, 1e-9))

    def summary(self) -> str:
        if not self.contributors:
            return "· epistemic drive: no signal available (nothing wired up yet)"
        detail = ", ".join(f"{n}={getattr(self, n):.2f}" for n in self.contributors)
        return f"· epistemic drive: pressure {self.total:.2f} ({detail})"

    def to_dict(self) -> Dict[str, Any]:
        return {"total": round(self.total, 4), "weakness": round(self.weakness, 4),
                "curiosity": round(self.curiosity, 4), "world_model": round(self.world_model, 4),
                "novelty": round(self.novelty, 4), "signal": round(self.signal, 4),
                "contributors": list(self.contributors)}


class EpistemicDrive:
    """Reads every uncertainty signal NYXARA has and answers "what should I go learn?"."""

    def __init__(self, *, core: Any = None, settings: Any = None, memory: Any = None,
                 world_model: Any = None, curiosity: Any = None,
                 weakness_report: Any = None) -> None:
        self.core = core
        self.settings = settings
        self.memory = memory if memory is not None else getattr(core, "memory", None)
        self.world_model = (world_model if world_model is not None
                            else getattr(core, "world_model", None))
        self.curiosity = (curiosity if curiosity is not None
                          else getattr(core, "active_curiosity", None))
        self._weakness_report = weakness_report

    # ------------------------------------------------------------------ #
    # individual readers — each returns (score, gaps) and never raises
    # ------------------------------------------------------------------ #
    def weakness_report(self) -> Any:
        """Her ranked weaknesses. Built from the architecture analysis when nothing is supplied.

        This is the input :func:`~nyxara.growth.acquire.gap_topics` always wanted and never got."""
        if self._weakness_report is not None:
            return self._weakness_report
        try:
            from nyxara.growth.architecture import ArchitectureAnalyzer
            from nyxara.growth.weakness import WeaknessSynthesizer
            arch = ArchitectureAnalyzer().analyze()
            self._weakness_report = WeaknessSynthesizer().synthesize(arch=arch)
        except Exception:  # noqa: BLE001 — no analysis available is simply no weakness signal
            self._weakness_report = None
        return self._weakness_report

    def _from_weakness(self, limit: int) -> Tuple[float, List[Gap]]:
        report = self.weakness_report()
        if report is None:
            return 0.0, []
        try:
            top = report.top(limit)
        except Exception:  # noqa: BLE001
            return 0.0, []
        gaps: List[Gap] = []
        for w in top:
            title = (getattr(w, "title", "") or "").strip()
            if not title:
                continue
            origin = str(getattr(w, "source", "") or "")
            gaps.append(Gap(subject=title, pressure=_clamp01(getattr(w, "severity", 0.0)),
                            source="weakness", detail=origin,
                            kind=("engineering" if origin in _ENGINEERING_SOURCES
                                  else "knowledge")))
        score = max((g.pressure for g in gaps), default=0.0)
        return score, gaps

    def _from_curiosity(self, limit: int) -> Tuple[float, List[Gap]]:
        curiosity = self.curiosity
        if curiosity is None:
            return 0.0, []
        gaps: List[Gap] = []
        try:
            for question in (getattr(curiosity, "frontier", lambda *_a, **_k: [])() or [])[:limit]:
                subject = (getattr(question, "subject", "")
                           or getattr(question, "text", "") or "").strip()
                if subject:
                    gaps.append(Gap(subject=subject,
                                    pressure=_clamp01(getattr(question, "uncertainty", 0.5)),
                                    source="curiosity",
                                    detail=str(getattr(question, "kind", ""))))
        except Exception:  # noqa: BLE001
            return 0.0, []
        return max((g.pressure for g in gaps), default=0.0), gaps

    def _from_world_model(self) -> Tuple[float, List[Gap]]:
        model = self.world_model
        if model is None:
            return 0.0, []
        for attribute in ("mean_epistemic", "epistemic", "uncertainty"):
            reader = getattr(model, attribute, None)
            if not callable(reader):
                continue
            try:
                value = _clamp01(float(reader()))
            except Exception:  # noqa: BLE001
                continue
            gap = [Gap(subject="world dynamics she cannot yet predict", pressure=value,
                       source="world_model", detail=attribute)] if value > 0.4 else []
            return value, gap
        return 0.0, []

    def _from_signals(self, limit: int) -> Tuple[float, List[Gap]]:
        """Ambient focus — what the rest of her has been posting about lately.

        Reads the signals' **payloads**, not ``focus_terms()``. focus_terms tokenises, which is
        exactly right for steering which file to edit and exactly wrong here: "protein folding
        dynamics" would reach the acquirer as three separate one-word searches."""
        try:
            from nyxara.growth.signal_bus import _FOCUS_KINDS, get_signal_bus
            bus = get_signal_bus()
        except Exception:  # noqa: BLE001
            return 0.0, []

        gaps: List[Gap] = []
        peak = 0.0
        for kind in _FOCUS_KINDS:
            try:
                signals = bus.recent(kind, limit)
            except Exception:  # noqa: BLE001
                continue
            for signal in signals:
                payload = getattr(signal, "payload", None)
                subject = payload.strip() if isinstance(payload, str) else ""
                if not subject:
                    continue
                weight = _clamp01(float(getattr(signal, "weight", 1.0)))
                peak = max(peak, weight)
                gaps.append(Gap(subject=subject, pressure=weight, source="signal", detail=kind))
        if not gaps:
            return 0.0, []
        return peak, gaps[:limit]

    def _from_novelty(self) -> Tuple[float, List[Gap]]:
        """RND novelty, when a curiosity net is live. A scalar drive with no subject of its own."""
        net = getattr(self.core, "novelty_net", None) or getattr(self.curiosity, "novelty", None)
        scorer = getattr(net, "score", None)
        if not callable(scorer):
            return 0.0, []
        try:
            return _clamp01(float(scorer([0.0]))), []
        except Exception:  # noqa: BLE001
            return 0.0, []

    def _competence(self) -> float:
        """Her mastery tier, 0..1 — used to temper how hard the drive pushes."""
        try:
            from nyxara.growth.curriculum import AutoCurriculum
            return _clamp01(int(AutoCurriculum(memory=self.memory).frontier_tier()) / 8.0)
        except Exception:  # noqa: BLE001
            return 0.5

    # ------------------------------------------------------------------ #
    # the fused view
    # ------------------------------------------------------------------ #
    def pressure(self, *, limit: int = 8) -> DrivePressure:
        """The single normalised "epistemic hunger", with its components kept visible."""
        out = DrivePressure()
        for name, (score, _gaps) in (
                ("weakness", self._from_weakness(limit)),
                ("curiosity", self._from_curiosity(limit)),
                ("world_model", self._from_world_model()),
                ("signal", self._from_signals(limit)),
                ("novelty", self._from_novelty())):
            if score > 0.0:
                setattr(out, name, score)
                out.contributors.append(name)
        return out

    def _ranked_gaps(self, *, pool: int = 64) -> List[Gap]:
        """Every gap every signal reported, deduplicated and ranked. Filtering happens *after*.

        Ranking is ``pressure × source weight``, so a severe measured weakness outranks an ambient
        signal-bus term even when both report 1.0 on their own scale. The pool is generous on
        purpose: a repo with fifty architectural weaknesses would otherwise fill a short list
        entirely and every knowledge gap would be truncated away before anything could select it."""
        collected: List[Gap] = []
        for _score, gaps in (self._from_weakness(pool), self._from_curiosity(pool),
                             self._from_world_model(), self._from_signals(pool),
                             self._from_novelty()):
            collected.extend(gaps)

        best: Dict[str, Gap] = {}
        for gap in collected:
            key = gap.subject.strip().lower()
            if not key:
                continue
            weighted = gap.pressure * _SOURCE_WEIGHTS.get(gap.source, 0.5)
            existing = best.get(key)
            if existing is None or weighted > existing.pressure:
                best[key] = Gap(subject=gap.subject.strip(), pressure=weighted,
                                source=gap.source, detail=gap.detail, kind=gap.kind)
        return sorted(best.values(), key=lambda g: g.pressure, reverse=True)

    def top_gaps(self, k: int = 8) -> List[Gap]:
        """The k things she is most ignorant about, ranked across every signal (both kinds)."""
        return self._ranked_gaps()[:max(0, k)]

    def topics(self, *, limit: int = 8) -> List[str]:
        """Searchable subjects only — the shape :func:`~nyxara.growth.acquire.gap_topics` wants.

        Engineering gaps are excluded even though they rank highest. They belong to the
        self-improvement loop, not the acquirer: no amount of web reading fixes an import cycle,
        and searching for her own module names burns the acquisition budget on nothing. Use
        :meth:`top_gaps` when you want both kinds."""
        return [g.subject for g in self._ranked_gaps() if g.searchable][:max(0, limit)]

    def engineering_gaps(self, k: int = 8) -> List[Gap]:
        """The repair-shaped half — what the self-improvement loop should aim at."""
        return [g for g in self._ranked_gaps() if not g.searchable][:max(0, k)]

    def report(self, *, k: int = 8) -> Dict[str, Any]:
        return {"pressure": self.pressure().to_dict(),
                "gaps": [g.to_dict() for g in self.top_gaps(k)],
                "competence": round(self._competence(), 4)}

    def summary(self, *, k: int = 5) -> str:
        lines = [self.pressure().summary()]
        for gap in self.top_gaps(k):
            lines.append(f"    [{gap.pressure:.2f}] ({gap.source}) {gap.subject}")
        if len(lines) == 1:
            lines.append("    (no ranked gaps — nothing measured her ignorance yet)")
        return "\n".join(lines)
