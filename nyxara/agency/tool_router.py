"""NYXARA · agency/tool_router.py — decide *which* tool fits a sub-task (🧭, tool selection).

The :class:`~nyxara.agency.tools.ToolRegistry` decides whether a chosen tool *may* run; it does
not decide *which* tool to choose. Until now that choice was left entirely to the LLM proposing a
``Candidate(tool=…)``. This module gives NYXARA an explicit, deterministic tool-selection
reasoner — the action-side mirror of :class:`~nyxara.mind.faculties.FacultySelector` (which picks
the right *cognitive engine*): given a sub-task in natural language, it scores the live tool
catalog and returns a ranked, *explained* choice. Numeric/algorithmic work routes to ``run_code``
(Python); geometry/parts to ``cad_model``; live facts to ``web_search`` / ``web_fetch``; recall to
``recall_memory``; and so on.

The catalog is built automatically from whatever is registered — every
:class:`~nyxara.agency.tools.ToolSpec` already carries a name, description, capability and risk —
so the router stays in sync with the deployment's real tools and never invents one that is not
present. Selection is pure and side-effect-free: it ranks, it does not run. The kernel's gate
pipeline still decides whether the top-ranked tool actually executes, so this only *informs* the
sovereign loop, it never bypasses it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

__all__ = ["ToolProfile", "ToolChoice", "ToolRouter", "INTENT_CUES"]


# --------------------------------------------------------------------------- #
# Intent cues — keyword evidence that a sub-task wants a given *kind* of tool.
# Keyed by tool name (matched against the live registry); unknown tools fall back
# to their own name/description tokens, so a deployment's custom tools still rank.
# --------------------------------------------------------------------------- #
INTENT_CUES: Dict[str, Tuple[str, ...]] = {
    "run_python": ("compute", "calculate", "simulate", "algorithm", "script", "python", "numeric",
                   "solve", "stress", "matrix", "optimi", "data", "plot", "iterate", "analyze"),
    "ephemeral_exec": ("throwaway", "one-off program", "scratch script", "quick program"),
    "cad_model": ("cad", "model", "geometry", "part", "bracket", "cube", "cylinder", "sphere",
                  "box", "mesh", "dimension", "volume", "mechanical", "hole",
                  "assembly", "fabricat", "mount", "enclosure", "flange"),
    "web_search": ("search", "news", "latest", "google", "current", "today", "recent",
                   "web", "internet", "price"),
    "web_fetch": ("fetch", "url", "scrape", "website", "webpage"),
    "http_request": ("api", "endpoint", "rest", "service"),
    "recall_memory": ("remember", "recall", "earlier", "memory", "discussed",
                      "previously", "you said"),
    "remember_fact": ("note", "store", "save"),
    "knowledge_search": ("knowledge", "documents", "docs", "manual", "reference"),
    "read_document": ("pdf", "document", "read the report", "parse document"),
    "read_file": ("file", "contents", "open file"),
    "now": ("time", "date", "what day", "clock"),
    "calculate": ("arithmetic", "add", "multiply", "percentage", "divide"),
    "run_shell": ("shell", "bash", "terminal", "command"),
    "inspect_image": ("image", "photo", "picture", "look at the image"),
    "transcribe_audio": ("audio", "transcribe", "speech to text"),
    "generate_image": ("draw", "render an image", "generate an image"),
}

# friendly capability-class label for the rationale
_CLASS = {
    "run_python": "Python / code execution", "ephemeral_exec": "throwaway code",
    "cad_model": "CAD / parametric geometry",
    "web_search": "Internet search", "web_fetch": "Internet fetch",
    "http_request": "Internet API", "recall_memory": "memory recall",
    "knowledge_search": "knowledge retrieval", "read_document": "document reader",
    "now": "clock", "calculate": "arithmetic", "run_shell": "shell",
    "inspect_image": "vision", "transcribe_audio": "audio", "generate_image": "image generation",
}

# tokens too generic to be evidence when a tool has no explicit cue list
_STOPWORDS = frozenset((
    "the", "and", "for", "with", "this", "that", "from", "into", "call", "make", "get",
    "use", "via", "tool", "run", "set", "all", "any", "out", "its", "via", "txt",
))


@dataclass
class ToolProfile:
    """A tool's selection-relevant summary, derived from its live :class:`ToolSpec`."""

    name: str
    description: str = ""
    capability: str = ""
    risk: int = 1                       # RiskTier int (LOW=1 … CRITICAL=4)
    cost: float = 0.0
    cues: Tuple[str, ...] = ()          # intent keywords that favour this tool

    def cue_set(self) -> Tuple[str, ...]:
        if self.cues:
            return self.cues
        # fall back to the tool's own name/description tokens (minus generic stopwords)
        toks = [t for t in re.findall(r"[a-z]{3,}", f"{self.name} {self.description}".lower())
                if t not in _STOPWORDS]
        return tuple(dict.fromkeys(toks))


@dataclass
class ToolChoice:
    """One ranked candidate: a tool, a score, and a human-readable reason."""

    tool: str
    score: float
    rationale: str
    profile: Optional[ToolProfile] = None

    def to_dict(self) -> Dict[str, Any]:
        return {"tool": self.tool, "score": round(self.score, 4), "rationale": self.rationale}


class ToolRouter:
    """Score the live tool catalog for a sub-task and return a ranked, explained choice."""

    def __init__(self, registry: Any = None, *, profiles: Optional[List[ToolProfile]] = None) -> None:
        self.registry = registry
        self._profiles = profiles if profiles is not None else self._profiles_from_registry(registry)

    # ---- catalog construction ---- #
    @staticmethod
    def _profiles_from_registry(registry: Any) -> List[ToolProfile]:
        out: List[ToolProfile] = []
        if registry is None:
            return out
        try:
            names = registry.names()
        except Exception:  # noqa: BLE001
            return out
        for name in names:
            spec = None
            try:
                spec = registry.get(name)
            except Exception:  # noqa: BLE001
                spec = None
            risk = 1
            cap = ""
            cost = 0.0
            desc = ""
            if spec is not None:
                desc = getattr(spec, "description", "") or ""
                cost = float(getattr(spec, "cost", 0.0) or 0.0)
                cap_obj = getattr(spec, "capability", None)
                cap = getattr(cap_obj, "value", str(cap_obj or ""))
                risk_obj = getattr(spec, "risk", None)
                risk = int(getattr(risk_obj, "value", risk_obj) or 1)
            out.append(ToolProfile(name=name, description=desc, capability=cap, risk=risk,
                                   cost=cost, cues=INTENT_CUES.get(name, ())))
        return out

    def refresh(self) -> None:
        """Rebuild the catalog from the registry (call after tools are added/removed)."""
        self._profiles = self._profiles_from_registry(self.registry)

    def catalog(self) -> List[str]:
        return [p.name for p in self._profiles]

    # ---- scoring ---- #
    def _score(self, profile: ToolProfile, subtask: str) -> Tuple[float, List[str]]:
        low = subtask.lower()
        hits = [cue for cue in profile.cue_set() if cue in low]
        # intent overlap dominates; whole-word matches count double
        overlap = 0.0
        for cue in hits:
            overlap += 2.0 if re.search(rf"\b{re.escape(cue)}\b", low) else 1.0
        score = overlap
        # mild preference for cheaper, lower-risk, more-reversible tools when intent ties
        score -= 0.15 * max(0, profile.risk - 1)
        score -= 0.05 * profile.cost
        return score, hits

    def rank(self, subtask: str) -> List[ToolChoice]:
        """Every tool, scored and ordered best-first (ties broken by lower risk then name)."""
        subtask = subtask or ""
        scored: List[ToolChoice] = []
        for p in self._profiles:
            s, hits = self._score(p, subtask)
            label = _CLASS.get(p.name, p.capability or p.name)
            if hits:
                why = f"{label}: matched {', '.join(sorted(set(hits))[:4])}"
            else:
                why = f"{label}: no direct intent match"
            scored.append(ToolChoice(tool=p.name, score=s, rationale=why, profile=p))
        scored.sort(key=lambda c: (-c.score, c.profile.risk if c.profile else 1, c.tool))
        return scored

    def select(self, subtask: str, *, top_k: int = 3, min_score: float = 0.5
               ) -> List[ToolChoice]:
        """The top ``top_k`` candidates that clear ``min_score`` (best-first)."""
        ranked = [c for c in self.rank(subtask) if c.score >= min_score]
        return ranked[:max(1, top_k)] if ranked else []

    def choose(self, subtask: str) -> Optional[ToolChoice]:
        """The single best tool for ``subtask`` (``None`` when nothing clears the bar)."""
        top = self.select(subtask, top_k=1)
        return top[0] if top else None
