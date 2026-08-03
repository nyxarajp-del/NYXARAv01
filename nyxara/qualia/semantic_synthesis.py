"""NYXARA · qualia/semantic_synthesis.py — concepts with no word for them (🜛).

A language model's concepts are its tokens: whatever it can think, it can already name, because
naming is all it ever did. This module works the other way round. Text enters, becomes geometry,
and **stays** geometry — clustered, bound, compared and reasoned over in the fitted Poincaré space
of :mod:`~nyxara.mind.concept_hyperspace` — and only converts back to words at the boundary where
something has to be said to the Master.

**What "a concept with no word" means here, precisely.** Cluster the latent points. For each
cluster, find the nearest *labelled* anchor. If even the nearest anchor is further than
``label_threshold``, no vocabulary item covers that region: the structure is real, and unnamed.
That is a measurable test with a number in it, not a metaphor — and it is falsifiable, since
lowering the threshold makes fewer concepts qualify.

**Why an unnamed cluster still has to earn its place.** It is not automatically a truth. Clustering
finds structure in noise as happily as in signal, and an embedding artefact looks exactly like a
discovery from the inside. So a concept may sit in the workspace freely, but it may not enter an
*answer* until it holds a :class:`ConceptCertificate`, and there are three ways to earn one, each
delegating to machinery that already exists:

* **predictive** — including the concept measurably improves held-out prediction
  (:meth:`~nyxara.growth.foundry.Foundry.evaluate`'s perplexity, or any supplied scorer).
* **symbolic** — it expresses as a term in :mod:`~nyxara.mind.lot` that
  :class:`~nyxara.growth.prover.Prover` certifies.
* **causal** — it has a real fitted mechanism in the causal graph
  (:mod:`~nyxara.mind.causal_world_model`).

This is the same propose-then-certify discipline :mod:`~nyxara.growth.vsa_reasoner` already runs on:
the geometry proposes, something outside the geometry confirms. Without that gate this layer would
be an excellent machine for producing confident-sounding nonsense, and the gate is what makes it
real instead.

**The Babel codec** converts a latent concept back to language: nearest labelled anchors as
grounding, then her own brain phrasing it. When nothing is near enough, it **says so** — "I do not
have a word for this; it is the structure between X and Y" — rather than inventing a name. A
fabricated label is the one output that would make an unnamed concept indistinguishable from a
named one, which is exactly the distinction this package exists to keep.

Pure standard library. Depends on ``mind/concept_hyperspace`` and, optionally,
``cognition/hyper_dimensional_vectors``, ``growth/prover`` and ``mind/causal_world_model``.
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from nyxara.mind.latent_geometry import poincare_distance

__all__ = ["LatentConcept", "ConceptCertificate", "QualiaWorkspace", "BabelCodec", "QualiaReport"]

_METHODS = ("predictive", "symbolic", "causal")


@dataclass
class ConceptCertificate:
    """Proof that a discovered concept is worth using, and by which route it was earned."""

    method: str = ""                    # "predictive" | "symbolic" | "causal"
    score: float = 0.0                  # method-specific strength, normalised to [0, 1]
    detail: str = ""
    at: float = field(default_factory=time.time)

    @property
    def valid(self) -> bool:
        return self.method in _METHODS and self.score > 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {"method": self.method, "score": round(self.score, 4), "detail": self.detail,
                "at": self.at}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ConceptCertificate":
        return cls(method=str(d.get("method", "")), score=float(d.get("score", 0.0)),
                   detail=str(d.get("detail", "")), at=float(d.get("at", 0.0)))


@dataclass
class LatentConcept:
    """A region of the latent space. ``label is None`` means no word covers it."""

    key: str                                        # stable internal handle
    centroid: List[float] = field(default_factory=list)
    members: List[str] = field(default_factory=list)
    label: Optional[str] = None                     # None = alien: no vocabulary anchor is close
    nearest_anchor: str = ""                        # the closest labelled concept, whatever it is
    anchor_distance: float = float("inf")
    certificate: Optional[ConceptCertificate] = None

    @property
    def alien(self) -> bool:
        """True when no existing word covers this region — the property this module is about."""
        return self.label is None

    @property
    def certified(self) -> bool:
        return self.certificate is not None and self.certificate.valid

    def to_dict(self) -> Dict[str, Any]:
        return {"key": self.key, "centroid": [round(c, 12) for c in self.centroid],
                "members": list(self.members), "label": self.label,
                "nearest_anchor": self.nearest_anchor,
                "anchor_distance": (None if self.anchor_distance == float("inf")
                                    else round(self.anchor_distance, 6)),
                "certificate": self.certificate.to_dict() if self.certificate else None}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "LatentConcept":
        distance = d.get("anchor_distance")
        certificate = d.get("certificate")
        return cls(key=str(d.get("key", "")),
                   centroid=[float(c) for c in d.get("centroid", [])],
                   members=[str(m) for m in d.get("members", [])],
                   label=d.get("label"),
                   nearest_anchor=str(d.get("nearest_anchor", "")),
                   anchor_distance=float("inf") if distance is None else float(distance),
                   certificate=(ConceptCertificate.from_dict(certificate)
                                if isinstance(certificate, dict) else None))


@dataclass
class QualiaReport:
    """What a synthesis pass found — and, deliberately, how much of it is not usable yet."""

    clusters: int = 0
    named: int = 0
    alien: int = 0
    certified: int = 0
    notes: List[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [f"· qualia: {self.clusters} concept region(s) — {self.named} named, "
                 f"{self.alien} with no word for them; {self.certified} certified for use"]
        if self.alien and not self.certified:
            lines.append("    (uncertified concepts stay in the workspace and never enter an "
                         "answer — an unnamed cluster is not yet a truth)")
        lines.extend(f"    · {n}" for n in self.notes)
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {"clusters": self.clusters, "named": self.named, "alien": self.alien,
                "certified": self.certified, "notes": list(self.notes)}


class QualiaWorkspace:
    """Discovers latent concepts in a fitted hyperspace, and gates them on certification."""

    def __init__(self, space: Any, *, min_cluster: int = 3,
                 link_quantile: float = 0.15, label_quantile: float = 0.25,
                 require_certificate: bool = True) -> None:
        self.space = space
        self.min_cluster = max(2, int(min_cluster))
        # Both thresholds are QUANTILES of the space's own pairwise geodesic distances, never
        # absolute numbers. Hyperbolic distance has no natural scale — a space fitted on ten
        # concepts has a median pair around 8, one fitted on ten thousand has a very different
        # one — so any fixed radius is correct for exactly one space and silently wrong for
        # every other. A quantile means "close relative to how spread out this space actually is".
        self.link_quantile = max(0.01, min(0.9, float(link_quantile)))
        self.label_quantile = max(0.01, min(0.9, float(label_quantile)))
        self.require_certificate = bool(require_certificate)
        self.concepts: Dict[str, LatentConcept] = {}
        self._distances: List[float] = []
        self.label_threshold = 0.0      # resolved from the distance distribution on synthesise()

    # ------------------------------------------------------------------ #
    # discovery
    # ------------------------------------------------------------------ #
    def _points(self) -> List[Tuple[str, List[float]]]:
        return [(name, concept.point) for name, concept in
                getattr(self.space, "concepts", {}).items() if concept.point]

    def _cluster(self, points: Sequence[Tuple[str, List[float]]],
                 radius: float) -> List[List[str]]:
        """Single-linkage clustering under the geodesic metric.

        Single-linkage on purpose: it finds *connected structure* rather than round blobs, and a
        concept that has no name is far more likely to be an elongated relation between known
        things than a tidy sphere sitting on its own."""
        names = [n for n, _p in points]
        coordinates = {n: p for n, p in points}
        parent = {n: n for n in names}

        def find(x: str) -> str:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                a, b = names[i], names[j]
                if poincare_distance(coordinates[a], coordinates[b]) <= radius:
                    ra, rb = find(a), find(b)
                    if ra != rb:
                        parent[ra] = rb

        groups: Dict[str, List[str]] = {}
        for name in names:
            groups.setdefault(find(name), []).append(name)
        return [sorted(members) for members in groups.values()]

    def _centroid(self, members: Sequence[str]) -> List[float]:
        """The hyperbolic centroid — the Fréchet mean, never a Euclidean average of coordinates."""
        from nyxara.mind.hyperbolic_manifold import frechet_mean
        points = [self.space.concepts[m].point for m in members if m in self.space.concepts]
        return frechet_mean(points) if points else []

    def _pairwise(self, points: Sequence[Tuple[str, List[float]]]) -> List[float]:
        out: List[float] = []
        for i in range(len(points)):
            for j in range(i + 1, len(points)):
                out.append(poincare_distance(points[i][1], points[j][1]))
        out.sort()
        return out

    @staticmethod
    def _quantile(sorted_values: Sequence[float], q: float) -> float:
        if not sorted_values:
            return 0.0
        index = min(len(sorted_values) - 1, max(0, int(q * (len(sorted_values) - 1))))
        return sorted_values[index]

    def synthesise(self, *, radius: Optional[float] = None,
                   label_threshold: Optional[float] = None,
                   anchors: Optional[Sequence[str]] = None) -> QualiaReport:
        """Find concept regions and decide which of them any existing word actually covers.

        ``radius`` and ``label_threshold`` default to quantiles of this space's own distance
        distribution (see the constructor), so the same call behaves sensibly on a ten-concept
        space and a ten-thousand-concept one. Pass either explicitly to override.

        ``anchors`` is the *named* vocabulary; it defaults to every concept in the space, which
        makes the test strict — a region only counts as unnamed when it is far from everything she
        already has a word for."""
        report = QualiaReport()
        points = self._points()
        if len(points) < self.min_cluster:
            report.notes.append("not enough fitted concepts to cluster")
            return report

        self._distances = self._pairwise(points)
        radius = (self._quantile(self._distances, self.link_quantile) if radius is None
                  else float(radius))
        self.label_threshold = (self._quantile(self._distances, self.label_quantile)
                                if label_threshold is None else float(label_threshold))
        report.notes.append(f"link radius {radius:.2f}, label threshold "
                            f"{self.label_threshold:.2f} (from this space's own distances)")

        anchor_names = list(anchors) if anchors else [n for n, _p in points]
        anchor_points = {n: self.space.concepts[n].point for n in anchor_names
                         if n in getattr(self.space, "concepts", {})}

        for members in self._cluster(points, radius):
            if len(members) < self.min_cluster:
                continue
            centroid = self._centroid(members)
            if not centroid:
                continue
            key = "region:" + "|".join(members[:4])
            # nearest labelled anchor that is NOT simply one of this region's own members —
            # otherwise every cluster trivially names itself and nothing is ever alien
            outside = {n: p for n, p in anchor_points.items() if n not in members}
            nearest, distance = "", float("inf")
            for name, point in outside.items():
                d = poincare_distance(centroid, point)
                if d < distance:
                    nearest, distance = name, d

            concept = LatentConcept(key=key, centroid=centroid, members=members,
                                    nearest_anchor=nearest, anchor_distance=distance)
            concept.label = nearest if distance <= self.label_threshold else None
            self.concepts[key] = concept
            report.clusters += 1
            report.named += int(not concept.alien)
            report.alien += int(concept.alien)

        report.certified = sum(1 for c in self.concepts.values() if c.certified)
        return report

    # ------------------------------------------------------------------ #
    # certification — the gate between "found" and "usable"
    # ------------------------------------------------------------------ #
    def certify_predictive(self, key: str, scorer: Callable[[Sequence[str]], float], *,
                           baseline: float) -> Optional[ConceptCertificate]:
        """Earn a certificate by improving prediction on held-out data.

        ``scorer`` returns an error (perplexity, loss — lower is better) for the concept's members;
        beating ``baseline`` is the evidence. Never raises."""
        concept = self.concepts.get(key)
        if concept is None:
            return None
        try:
            score = float(scorer(concept.members))
        except Exception:  # noqa: BLE001 — a scorer that fails certifies nothing, it never crashes
            return None
        if not math.isfinite(score) or score >= baseline or baseline <= 0:
            return None
        improvement = max(0.0, min(1.0, (baseline - score) / baseline))
        concept.certificate = ConceptCertificate(
            method="predictive", score=improvement,
            detail=f"error {score:.4f} vs baseline {baseline:.4f}")
        return concept.certificate

    def certify_symbolic(self, key: str, claim: Any, *, prover: Any = None
                         ) -> Optional[ConceptCertificate]:
        """Earn a certificate by having the prover certify a claim about the concept."""
        concept = self.concepts.get(key)
        if concept is None:
            return None
        try:
            if prover is None:
                from nyxara.growth.prover import Prover
                prover = Prover()
            result = prover.prove(claim)
        except Exception:  # noqa: BLE001
            return None
        proven = bool(getattr(result, "proven", False) or getattr(result, "ok", False))
        if not proven:
            return None
        concept.certificate = ConceptCertificate(
            method="symbolic", score=1.0, detail=str(claim)[:120])
        return concept.certificate

    def certify_causal(self, key: str, model: Any, *, min_confidence: float = 0.25
                       ) -> Optional[ConceptCertificate]:
        """Earn a certificate by having a real fitted mechanism in the causal graph."""
        concept = self.concepts.get(key)
        if concept is None:
            return None
        try:
            links = getattr(model, "_links", {}) or {}
            members = set(concept.members)
            best = 0.0
            detail = ""
            for (cause, effect), link in links.items():
                if cause in members or effect in members:
                    confidence = float(getattr(link, "confidence", 0.0))
                    if confidence > best:
                        best, detail = confidence, f"{cause} → {effect}"
        except Exception:  # noqa: BLE001
            return None
        if best < min_confidence:
            return None
        concept.certificate = ConceptCertificate(method="causal", score=best, detail=detail)
        return concept.certificate

    # ------------------------------------------------------------------ #
    # what may actually be used
    # ------------------------------------------------------------------ #
    def usable(self) -> List[LatentConcept]:
        """The concepts allowed into an answer.

        With ``require_certificate`` on — the default — an uncertified concept is withheld no
        matter how striking it looks. That withholding is the feature."""
        if not self.require_certificate:
            return list(self.concepts.values())
        return [c for c in self.concepts.values() if c.certified]

    def alien_concepts(self, *, certified_only: bool = False) -> List[LatentConcept]:
        """Regions no existing word covers — optionally only the ones that earned their place."""
        found = [c for c in self.concepts.values() if c.alien]
        return [c for c in found if c.certified] if certified_only else found

    # ------------------------------------------------------------------ #
    # persistence
    # ------------------------------------------------------------------ #
    def to_dict(self) -> Dict[str, Any]:
        return {"min_cluster": self.min_cluster, "link_quantile": self.link_quantile,
                "label_quantile": self.label_quantile,
                "label_threshold": self.label_threshold,
                "require_certificate": self.require_certificate,
                "concepts": [c.to_dict() for c in self.concepts.values()]}

    def load_dict(self, data: Dict[str, Any]) -> bool:
        try:
            self.min_cluster = max(2, int(data.get("min_cluster", self.min_cluster)))
            self.link_quantile = float(data.get("link_quantile", self.link_quantile))
            self.label_quantile = float(data.get("label_quantile", self.label_quantile))
            self.label_threshold = float(data.get("label_threshold", self.label_threshold))
            self.require_certificate = bool(data.get("require_certificate",
                                                     self.require_certificate))
            restored: Dict[str, LatentConcept] = {}
            for raw in data.get("concepts", []):
                try:
                    concept = LatentConcept.from_dict(raw)
                except Exception:  # noqa: BLE001 — one bad entry never costs the rest
                    continue
                if concept.key:
                    restored[concept.key] = concept
            self.concepts = restored
            return True
        except Exception:  # noqa: BLE001 — a corrupt workspace is an empty one, never a crash
            return False

    def save(self, path: Any) -> bool:
        try:
            target = Path(path)
            target.parent.mkdir(parents=True, exist_ok=True)
            tmp = target.with_suffix(target.suffix + ".tmp")
            tmp.write_text(json.dumps(self.to_dict()), encoding="utf-8")
            os.replace(tmp, target)
            return True
        except Exception:  # noqa: BLE001
            return False

    def load(self, path: Any) -> bool:
        try:
            return self.load_dict(json.loads(Path(path).read_text(encoding="utf-8")))
        except Exception:  # noqa: BLE001
            return False


class BabelCodec:
    """Latent ⇄ language, and honest about the gap between them."""

    def __init__(self, workspace: QualiaWorkspace, *, brain: Any = None) -> None:
        self.workspace = workspace
        self.brain = brain          # anything with .generate(); optional

    def encode(self, name: str) -> Optional[List[float]]:
        """A word's position in the latent space, if she has fitted one for it."""
        concept = getattr(self.workspace.space, "concepts", {}).get(name)
        return list(concept.point) if concept is not None and concept.point else None

    def _grounding(self, concept: LatentConcept, k: int = 3) -> List[Tuple[str, float]]:
        """The nearest named anchors — the vocabulary a description has to be built out of."""
        space_concepts = getattr(self.workspace.space, "concepts", {})
        scored = [(name, poincare_distance(concept.centroid, c.point))
                  for name, c in space_concepts.items()
                  if c.point and name not in concept.members]
        scored.sort(key=lambda t: t[1])
        return scored[:max(1, k)]

    def decode(self, key: str) -> str:
        """Put a latent concept into words — or say plainly that there are none.

        When the region has no anchor close enough, this does **not** invent a name. A fabricated
        label would make an unnamed concept indistinguishable from a named one in every downstream
        use, which is precisely the distinction worth keeping."""
        concept = self.workspace.concepts.get(key)
        if concept is None:
            return "(no such concept)"

        members = ", ".join(concept.members[:5])
        if not concept.alien:
            return f"{concept.label} — the region covering {members}"

        grounding = self._grounding(concept)
        between = " and ".join(name for name, _d in grounding[:2]) or "anything she has a word for"
        description = (f"I do not have a word for this. It is the structure holding {members} "
                       f"together, sitting between {between}")
        if self.brain is not None:
            try:
                phrased = self.brain.generate(
                    f"Describe, in one sentence, what these have in common: {members}.",
                    max_tokens=48)
                if phrased and phrased.strip():
                    description += f" — {phrased.strip()}"
            except Exception:  # noqa: BLE001 — phrasing is a garnish; the honest statement stands
                pass
        if not concept.certified:
            description += " (uncertified: found, not yet shown to be real)"
        return description

    def describe_all(self, *, certified_only: bool = True) -> List[str]:
        source = (self.workspace.usable() if certified_only
                  else list(self.workspace.concepts.values()))
        return [self.decode(concept.key) for concept in source]
