"""NYXARA · mind/muse.py — the Autonomous Motivation Engine + the Ego Narrative
(P24 · F19, organs 1 & 12).

Two engines that give her creativity *its own will*:

* :class:`MuseEngine` — curiosity-driven goal generation. From an **ignorance map**
  (the thinnest regions of her :class:`~nyxara.mind.concept_graph.ConceptGraph` plus
  the sparsest modalities of her novelty archive) she opens her OWN creative research
  projects: a question, a falsifiable hypothesis ("blending A with B will beat my mean
  novelty"), a topic, a modality — no user prompt anywhere. Each project is later
  judged **corroborated or refuted** against the measured outcome, so motivation is a
  real hypothesize → test → update loop, mirroring :mod:`nyxara.growth.active_curiosity`.

* :class:`EgoNarrative` — the autobiographical self-model. She tracks her own recent
  choices (modality / generator / persona), detects her biases with recency-weighted
  frequencies, names her strengths and mistakes, and emits corrective pressure: the
  next Muse project deliberately steers AWAY from an over-used groove. Honest
  self-knowledge, no consciousness theatre.

**No LLM anywhere in this module.** Pure standard library, seed-deterministic.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from nyxara.mind.concept_graph import ConceptGraph

__all__ = [
    "CreativeProject",
    "MuseEngine",
    "EgoNarrative",
]

MODALITIES: Tuple[str, ...] = ("idea", "verse", "art")

# seed concepts so the very first projects have raw material before any creation exists
_SEED_CONCEPTS = [
    "gravity", "sonnet", "swarm", "mirror", "entropy", "harvest", "circuit",
    "tide", "myth", "lattice", "compass", "ember", "archive", "chorus",
    "membrane", "orbit", "loom", "signal", "glacier", "hive",
]


@dataclass
class CreativeProject:
    """One self-chosen creative research project with a falsifiable hypothesis."""

    project_id: int
    question: str
    hypothesis: str
    topic: str
    modality: str
    blend_with: Optional[str] = None
    status: str = "open"           # open | corroborated | refuted
    outcome: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {"project_id": self.project_id, "question": self.question,
                "hypothesis": self.hypothesis, "topic": self.topic,
                "modality": self.modality, "blend_with": self.blend_with,
                "status": self.status, "outcome": self.outcome,
                "created_at": self.created_at}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "CreativeProject":
        return cls(project_id=int(d.get("project_id", 0) or 0),
                   question=str(d.get("question", "")),
                   hypothesis=str(d.get("hypothesis", "")),
                   topic=str(d.get("topic", "")),
                   modality=str(d.get("modality", "idea")),
                   blend_with=d.get("blend_with"),
                   status=str(d.get("status", "open")),
                   outcome=dict(d.get("outcome", {}) or {}),
                   created_at=float(d.get("created_at", time.time()) or time.time()))


class EgoNarrative:
    """Recency-weighted self-model over her own creative choices."""

    DECAY = 0.9          # how fast old choices fade from the self-image
    BIAS_LEVEL = 0.5     # a share above this in any dimension counts as a groove

    def __init__(self) -> None:
        self._freq: Dict[str, Dict[str, float]] = {"modality": {}, "generator": {},
                                                   "persona": {}}
        self._kept: float = 0.0
        self._tried: float = 0.0
        self._mistakes: Dict[str, float] = {}   # rejection reason -> decayed count
        self.observations: int = 0

    # ---- observation ---- #
    def observe(self, *, modality: str, generator: str = "", persona: str = "",
                kept: bool = False, rejection: str = "") -> None:
        for dim in self._freq.values():
            for k in dim:
                dim[k] *= self.DECAY
        for dim, value in (("modality", modality), ("generator", generator),
                           ("persona", persona)):
            if value:
                self._freq[dim][value] = self._freq[dim].get(value, 0.0) + 1.0
        self._kept = self._kept * self.DECAY + (1.0 if kept else 0.0)
        self._tried = self._tried * self.DECAY + 1.0
        if rejection:
            for k in self._mistakes:
                self._mistakes[k] *= self.DECAY
            self._mistakes[rejection] = self._mistakes.get(rejection, 0.0) + 1.0
        self.observations += 1

    # ---- self-knowledge ---- #
    def bias(self) -> Optional[Tuple[str, str, float]]:
        """The strongest current groove: (dimension, value, share) — or None if balanced."""
        best: Optional[Tuple[str, str, float]] = None
        for dim, counts in self._freq.items():
            total = sum(counts.values())
            if total <= 0:
                continue
            value, mass = max(counts.items(), key=lambda kv: kv[1])
            share = mass / total
            if share > self.BIAS_LEVEL and (best is None or share > best[2]):
                best = (dim, value, share)
        return best

    def underused(self, dimension: str, options: Tuple[str, ...]) -> str:
        counts = self._freq.get(dimension, {})
        return min(options, key=lambda o: counts.get(o, 0.0))

    def kept_rate(self) -> float:
        return self._kept / self._tried if self._tried > 0 else 0.0

    def worst_mistake(self) -> Optional[str]:
        return max(self._mistakes, key=lambda k: self._mistakes[k], default=None)

    def narrative(self) -> str:
        """A plain-language autobiographical sentence — honest, not theatrical."""
        parts: List[str] = [f"I have made {self.observations} recent creative choices; "
                            f"my kept-rate is {self.kept_rate():.0%}."]
        b = self.bias()
        if b is not None:
            dim, value, share = b
            away = self.underused(dim, MODALITIES) if dim == "modality" else None
            parts.append(f"I notice I lean on {value} for {share:.0%} of my {dim} "
                         f"choices" + (f"; I should shift toward {away}." if away else "."))
        else:
            parts.append("My choices are currently balanced across my range.")
        m = self.worst_mistake()
        if m is not None:
            parts.append(f"My most repeated mistake lately: {m}.")
        return " ".join(parts)

    # ---- persistence ---- #
    def to_dict(self) -> Dict[str, Any]:
        return {"freq": {d: {k: round(v, 5) for k, v in c.items()}
                         for d, c in self._freq.items()},
                "kept": round(self._kept, 5), "tried": round(self._tried, 5),
                "mistakes": {k: round(v, 5) for k, v in self._mistakes.items()},
                "observations": self.observations}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "EgoNarrative":
        ego = cls()
        for dim, counts in (d.get("freq", {}) or {}).items():
            if dim in ego._freq:
                ego._freq[dim] = {str(k): float(v) for k, v in (counts or {}).items()}
        ego._kept = float(d.get("kept", 0.0) or 0.0)
        ego._tried = float(d.get("tried", 0.0) or 0.0)
        ego._mistakes = {str(k): float(v)
                         for k, v in (d.get("mistakes", {}) or {}).items()}
        ego.observations = int(d.get("observations", 0) or 0)
        return ego


class MuseEngine:
    """She decides for herself what to create next, and states why — falsifiably."""

    MAX_PROJECTS = 64

    def __init__(self, graph: Optional[ConceptGraph] = None,
                 ego: Optional[EgoNarrative] = None, *,
                 rng: Optional[random.Random] = None) -> None:
        self.graph = graph if graph is not None else ConceptGraph()
        self.ego = ego if ego is not None else EgoNarrative()
        self.rng = rng or random.Random()
        self.projects: List[CreativeProject] = []
        self._next_id = 1
        if not self.graph.concepts():
            for c in _SEED_CONCEPTS:
                self.graph.add_concept(c, 0.5)

    # ------------------------------------------------------------------ #
    # the ignorance map → a self-chosen project
    # ------------------------------------------------------------------ #
    def propose_project(self, archive_summary: Optional[Dict[str, Any]] = None
                        ) -> CreativeProject:
        """Open a new creative research project from her own gaps.

        ``archive_summary`` (optional) is the originality archive's per-modality view:
        ``{"per_modality": {"idea": {"count": n, "mean_novelty": x}, ...}}``. The Muse
        combines it with the concept graph's thinnest regions and the Ego's corrective
        pressure to pick WHAT to try and WHY — no prompt from anyone."""
        summary = archive_summary or {}
        per_mod = summary.get("per_modality", {}) or {}

        # 1) modality: prefer the Ego's corrective target, else the archive's sparsest
        bias = self.ego.bias()
        if bias is not None and bias[0] == "modality":
            modality = self.ego.underused("modality", MODALITIES)
            why_mod = f"I over-use {bias[1]} ({bias[2]:.0%}); correcting course"
        else:
            modality = min(MODALITIES,
                           key=lambda m: per_mod.get(m, {}).get("count", 0))
            why_mod = "my least-explored modality"

        # 2) material: a semantically distant pair (cross-domain), else a thin region
        pair = self.graph.distant_pair(rng=self.rng)
        weak = self.graph.weak_regions(k=3)
        if pair is not None:
            a, b = pair
            topic, blend = a, b
            question = (f"What emerges when «{a}» is fused with the foreign domain "
                        f"«{b}» in {modality}?")
        else:
            a = weak[0] if weak else "novelty"
            topic, blend = a, None
            question = f"What lives in my thinnest region of meaning, «{a}», as {modality}?"

        mean_nov = float(per_mod.get(modality, {}).get("mean_novelty", 0.0) or 0.0)
        hypothesis = (f"Creating {modality} from «{topic}»"
                      + (f" × «{blend}»" if blend else "")
                      + f" will beat my rolling mean novelty ({mean_nov:.2f})"
                      + f" — chosen because: {why_mod}.")

        project = CreativeProject(
            project_id=self._next_id, question=question, hypothesis=hypothesis,
            topic=(f"{topic} {blend}" if blend else topic), modality=modality,
            blend_with=blend)
        self._next_id += 1
        self.projects.append(project)
        if len(self.projects) > self.MAX_PROJECTS:
            # keep every open project; trim the oldest CLOSED ones first
            closed = [p for p in self.projects if p.status != "open"]
            for old in closed[: len(self.projects) - self.MAX_PROJECTS]:
                self.projects.remove(old)
        return project

    # ------------------------------------------------------------------ #
    # closing the loop: the hypothesis meets the measurement
    # ------------------------------------------------------------------ #
    def record_outcome(self, project: CreativeProject, *,
                       kept: int, best_novelty: float,
                       mean_novelty: float) -> CreativeProject:
        """Judge the project's hypothesis against what actually happened, and learn."""
        corroborated = kept > 0 and best_novelty > mean_novelty
        project.status = "corroborated" if corroborated else "refuted"
        project.outcome = {"kept": int(kept),
                           "best_novelty": round(float(best_novelty), 4),
                           "mean_novelty_at_test": round(float(mean_novelty), 4)}
        # fold the tested combination back into the concept graph — win or lose,
        # the map of meaning grows (a refuted fusion is knowledge too)
        self.graph.associate(project.topic, weight=1.0 if corroborated else 0.4,
                             extra_concepts=[project.modality])
        return project

    def open_projects(self) -> List[CreativeProject]:
        return [p for p in self.projects if p.status == "open"]

    # ------------------------------------------------------------------ #
    # persistence & report
    # ------------------------------------------------------------------ #
    def to_dict(self) -> Dict[str, Any]:
        return {"projects": [p.to_dict() for p in self.projects[-self.MAX_PROJECTS:]],
                "next_id": self._next_id,
                "ego": self.ego.to_dict(),
                "graph": self.graph.to_dict()}

    @classmethod
    def from_dict(cls, d: Dict[str, Any], *,
                  rng: Optional[random.Random] = None) -> "MuseEngine":
        muse = cls(graph=ConceptGraph.from_dict(d.get("graph", {}) or {}),
                   ego=EgoNarrative.from_dict(d.get("ego", {}) or {}), rng=rng)
        muse.projects = [CreativeProject.from_dict(p)
                         for p in (d.get("projects", []) or [])]
        muse._next_id = int(d.get("next_id", len(muse.projects) + 1) or 1)
        return muse

    def report(self) -> Dict[str, Any]:
        by_status: Dict[str, int] = {}
        for p in self.projects:
            by_status[p.status] = by_status.get(p.status, 0) + 1
        return {"projects": len(self.projects), "by_status": by_status,
                "open": [p.question for p in self.open_projects()[:3]],
                "self_model": self.ego.narrative(),
                "graph": self.graph.report()}


# --------------------------------------------------------------------------- #
# Self-test / demo
# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # pragma: no cover
    print("=" * 70)
    print("NYXARA muse + ego-narrative self-test")
    print("=" * 70)

    muse = MuseEngine(rng=random.Random(11))

    # she opens her own project — question, falsifiable hypothesis, no prompt
    p = muse.propose_project({"per_modality": {"idea": {"count": 9, "mean_novelty": 0.3},
                                               "verse": {"count": 1, "mean_novelty": 0.2},
                                               "art": {"count": 4, "mean_novelty": 0.4}}})
    print(f"\nquestion            : {p.question}")
    print(f"hypothesis          : {p.hypothesis}")
    assert p.status == "open" and p.modality == "verse"   # sparsest modality
    assert "will beat my rolling mean novelty" in p.hypothesis

    # the outcome judges the hypothesis honestly
    muse.record_outcome(p, kept=2, best_novelty=0.55, mean_novelty=0.2)
    assert p.status == "corroborated"
    p2 = muse.propose_project()
    muse.record_outcome(p2, kept=0, best_novelty=0.0, mean_novelty=0.2)
    assert p2.status == "refuted"
    print(f"verdicts            : #{p.project_id} {p.status}, #{p2.project_id} {p2.status}")

    # a refuted fusion still grows the map of meaning
    assert muse.graph.associations_seen >= 2

    # the ego notices a planted groove and pushes away from it
    ego = muse.ego
    for _ in range(10):
        ego.observe(modality="art", generator="lissajous", kept=True)
    b = ego.bias()
    assert b is not None and b[0] == "modality" and b[1] == "art"
    p3 = muse.propose_project()
    assert p3.modality != "art"     # corrective pressure is real
    print(f"self-model          : {ego.narrative()}")
    assert "lean on art" in ego.narrative()

    # mistakes are remembered
    ego.observe(modality="idea", kept=False, rejection="cliche")
    ego.observe(modality="idea", kept=False, rejection="cliche")
    assert ego.worst_mistake() == "cliche"

    # determinism under a seed
    m1 = MuseEngine(rng=random.Random(5)).propose_project()
    m2 = MuseEngine(rng=random.Random(5)).propose_project()
    assert m1.question == m2.question and m1.hypothesis == m2.hypothesis
    print("deterministic w/seed: OK")

    # persistence round-trip
    muse2 = MuseEngine.from_dict(muse.to_dict(), rng=random.Random(1))
    assert len(muse2.projects) == len(muse.projects)
    assert muse2.ego.observations == muse.ego.observations
    assert muse2.report()["by_status"] == muse.report()["by_status"]
    print("persistence          : OK")

    print(f"\nreport              : {muse.report()['by_status']}")
    print("\nALL SELF-TESTS PASSED ✓")
