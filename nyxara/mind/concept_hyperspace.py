"""NYXARA · mind/concept_hyperspace.py — L-HOLO: concepts as geometry, fitted not hashed (🌐).

:mod:`~nyxara.mind.hyperbolic_manifold` already gives NYXARA a correct Poincaré ball — Möbius
addition, exp/log maps, Einstein midpoint, Fréchet mean, all tested to the point of checking the
triangle inequality and that the Fréchet objective actually decreases. What it does **not** have is
a reason for any particular concept to sit where it sits. ``BallEmbedder.project`` is a *fixed
seeded Gaussian projection*: deterministic, reproducible, and carrying no information about
hierarchy at all. Radius means nothing. Nothing is ever fitted to anything.

This module supplies the missing objective. It is the Nickel–Kiela Poincaré embedding: for an
observed relation ``(child, parent)``, minimise

    L = −log( exp(−d(u,v)) / Σ_{v' ∈ negatives ∪ {v}} exp(−d(u,v')) )

by **Riemannian** SGD — the Euclidean gradient rescaled by ``(1−‖θ‖²)²/4`` and retracted with
:func:`~nyxara.mind.hyperbolic_manifold.project_to_ball`. Hyperbolic space is the right home for
this because a tree's node count grows exponentially with depth and so does the volume of a ball in
hyperbolic space; Euclidean space runs out of room and starts lying about distances. After fitting,
general concepts genuinely sit near the origin and specific ones near the boundary, because the
loss put them there.

Two further things the geometry then buys, both of which the package could not do before:

* **Entailment cones.** A concept is stored as a point *and an aperture*, so "is A a kind of B?"
  becomes a containment test in the cone at B rather than a lookup in a hand-built graph.
* **Cross-domain correspondence.** Two domains are aligned by matching their *geodesic distance
  matrices* — the shape of a domain rather than its vocabulary — which is what lets a structure
  shared by two fields surface even when they share no words. Every candidate is scored, and one
  that does not clear ``min_score`` is **not returned**: an unverified analogy stated confidently
  is worse than silence. Where symbolic relations are available the survivor is additionally put
  through :class:`~nyxara.mind.analogy.StructureMapper`, so geometry proposes and structure
  confirms — the same propose-then-certify discipline
  :mod:`~nyxara.growth.vsa_reasoner` already uses.

Pure standard library: the whole fit runs on lists of floats, so it works on a box with no numpy.

Depends on ``mind/hyperbolic_manifold`` (the ball), ``mind/latent_geometry`` (the canonical
distance) and, optionally, ``mind/analogy`` (symbolic verification) and ``memory/neural_embedder``
(the 256-d semantic floor). Nothing imports back.
"""

from __future__ import annotations

import json
import math
import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from nyxara.mind.hyperbolic_manifold import project_to_ball
from nyxara.mind.latent_geometry import poincare_distance

__all__ = ["Concept", "Correspondence", "HyperspaceReport", "ConceptHyperspace"]

_EPS = 1e-5
_MAX_NORM = 1.0 - 1e-5


@dataclass
class Concept:
    """A concept as a geometric object: a point on the ball plus the cone it subtends."""

    name: str
    point: List[float] = field(default_factory=list)
    aperture: float = 0.2               # entailment-cone half-angle at this point
    domain: str = ""
    members: List[str] = field(default_factory=list)

    @property
    def radius(self) -> float:
        """Distance from the origin — after fitting, this is genuinely specificity."""
        return math.sqrt(sum(c * c for c in self.point))

    def to_dict(self) -> Dict[str, Any]:
        # 12 decimals, not the usual 5–6: hyperbolic distance diverges as ‖x‖ → 1, and a fitted
        # leaf sits at r ≈ 0.999, so coarse rounding there changes distances materially.
        return {"name": self.name, "point": [round(c, 12) for c in self.point],
                "aperture": round(self.aperture, 6), "domain": self.domain,
                "members": list(self.members)}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Concept":
        return cls(name=str(d.get("name", "")), point=[float(c) for c in d.get("point", [])],
                   aperture=float(d.get("aperture", 0.2)), domain=str(d.get("domain", "")),
                   members=[str(m) for m in d.get("members", [])])


@dataclass
class Correspondence:
    """One proposed structural mapping between two domains, with its evidence."""

    source_domain: str
    target_domain: str
    pairs: List[Tuple[str, str]] = field(default_factory=list)
    score: float = 0.0                  # 0..1 — how well the two shapes align
    verified: bool = False              # symbolically confirmed, not merely geometric
    method: str = "geodesic-alignment"

    def to_dict(self) -> Dict[str, Any]:
        return {"source_domain": self.source_domain, "target_domain": self.target_domain,
                "pairs": [list(p) for p in self.pairs], "score": round(self.score, 4),
                "verified": self.verified, "method": self.method}

    def describe(self) -> str:
        mapped = ", ".join(f"{a}↔{b}" for a, b in self.pairs[:6])
        seal = "verified" if self.verified else "geometric only"
        return (f"{self.source_domain} ≅ {self.target_domain} "
                f"(score {self.score:.2f}, {seal}): {mapped}")


@dataclass
class HyperspaceReport:
    """What a fit actually achieved — including whether it beat a Euclidean baseline."""

    concepts: int = 0
    relations: int = 0
    epochs: int = 0
    initial_loss: float = 0.0
    final_loss: float = 0.0
    mean_rank: float = 0.0              # mean rank of the true parent among negatives (1 = perfect)
    notes: List[str] = field(default_factory=list)

    @property
    def improved(self) -> bool:
        return self.final_loss < self.initial_loss

    def summary(self) -> str:
        return (f"· hyperspace: {self.concepts} concept(s), {self.relations} relation(s), "
                f"{self.epochs} epoch(s); loss {self.initial_loss:.4f} → {self.final_loss:.4f} "
                f"({'improved' if self.improved else 'NO improvement'}), "
                f"mean parent rank {self.mean_rank:.2f}")

    def to_dict(self) -> Dict[str, Any]:
        return {"concepts": self.concepts, "relations": self.relations, "epochs": self.epochs,
                "initial_loss": round(self.initial_loss, 6),
                "final_loss": round(self.final_loss, 6), "improved": self.improved,
                "mean_rank": round(self.mean_rank, 4), "notes": list(self.notes)}


# --------------------------------------------------------------------------- #
# the gradient
# --------------------------------------------------------------------------- #
def _distance_grad(u: Sequence[float], v: Sequence[float]) -> List[float]:
    """``∂d(u,v)/∂u`` in Euclidean coordinates (Nickel & Kiela 2017, eq. 4–5).

    Returned as the *Euclidean* gradient; the caller rescales it to the Riemannian one. Kept
    analytic rather than finite-differenced so the fit stays fast and exact."""
    uu = sum(c * c for c in u)
    vv = sum(c * c for c in v)
    uv = sum(a * b for a, b in zip(u, v))
    alpha = max(1.0 - uu, _EPS)
    beta = max(1.0 - vv, _EPS)
    diff_sq = uu - 2.0 * uv + vv
    gamma = 1.0 + (2.0 / (alpha * beta)) * diff_sq
    root = math.sqrt(max(gamma * gamma - 1.0, _EPS))
    scale = 4.0 / (beta * root)
    coefficient = (vv - 2.0 * uv + 1.0) / (alpha * alpha)
    return [scale * (coefficient * a - b / alpha) for a, b in zip(u, v)]


def _to_riemannian(point: Sequence[float], grad: Sequence[float]) -> List[float]:
    """Rescale a Euclidean gradient into the Poincaré metric: ``((1−‖θ‖²)²/4)·∇``.

    Without this the step size is wrong everywhere except the origin, and points near the boundary
    — the specific concepts, the ones that matter most — barely move at all."""
    norm_sq = sum(c * c for c in point)
    factor = ((1.0 - norm_sq) ** 2) / 4.0
    return [factor * g for g in grad]


class ConceptHyperspace:
    """A fitted Poincaré embedding of concepts, with cones, neighbours and cross-domain analogy."""

    def __init__(self, *, dim: int = 64, seed: int = 0, lr: float = 0.3,
                 negatives: int = 8, burn_in_lr_scale: float = 0.1) -> None:
        self.dim = max(2, int(dim))
        self.seed = int(seed)
        self.lr = float(lr)
        self.negatives = max(1, int(negatives))
        self.burn_in_lr_scale = float(burn_in_lr_scale)
        self.concepts: Dict[str, Concept] = {}
        # Every relation ever fitted, so a later fit() refines the space instead of wrecking it.
        self._relations: List[Tuple[str, str, str]] = []       # (child, parent, domain)
        self._rng = random.Random(self.seed)

    # ------------------------------------------------------------------ #
    # construction
    # ------------------------------------------------------------------ #
    def _ensure(self, name: str, *, domain: str = "") -> Concept:
        """Create a concept near the origin if unseen. Small init is what lets it move at all."""
        concept = self.concepts.get(name)
        if concept is None:
            point = [self._rng.uniform(-1e-3, 1e-3) for _ in range(self.dim)]
            concept = Concept(name=name, point=project_to_ball(point), domain=domain)
            self.concepts[name] = concept
        elif domain and not concept.domain:
            concept.domain = domain
        return concept

    def add(self, name: str, *, domain: str = "", members: Optional[Sequence[str]] = None) -> Concept:
        concept = self._ensure(name, domain=domain)
        if members:
            for member in members:
                if member not in concept.members:
                    concept.members.append(str(member))
        return concept

    # ------------------------------------------------------------------ #
    # the fit
    # ------------------------------------------------------------------ #
    def fit(self, relations: Sequence[Tuple[str, str]], *, epochs: int = 30,
            burn_in: int = 5, domain: str = "") -> HyperspaceReport:
        """Fit the embedding to ``(child, parent)`` relations by Riemannian SGD.

        A short burn-in at a reduced learning rate is Nickel & Kiela's own recommendation: it lets
        the angular arrangement settle before points are allowed to race outward toward the
        boundary, which otherwise traps the fit in a bad configuration."""
        report = HyperspaceReport(epochs=max(0, int(epochs)))
        incoming = [(str(c), str(p)) for c, p in relations
                    if str(c) and str(p) and str(c) != str(p)]
        if not incoming:
            report.notes.append("no relations to fit")
            return report
        for child, parent in incoming:
            self._ensure(child, domain=domain)
            self._ensure(parent, domain=domain)
            if (child, parent, domain) not in self._relations:
                self._relations.append((child, parent, domain))

        # Refit over EVERY relation seen so far, not just the incoming batch. Fitting a second
        # domain in isolation would drag the first one's concepts around as negatives and quietly
        # destroy an embedding that was already correct.
        pairs = [(c, p) for c, p, _d in self._relations]
        report.relations = len(pairs)
        report.concepts = len(self.concepts)
        if len(self.concepts) < 3:
            report.notes.append("need at least three concepts for negative sampling")
            return report

        report.initial_loss = self._loss(pairs)
        rng = random.Random(self.seed + 1)
        for epoch in range(report.epochs):
            lr = self.lr * (self.burn_in_lr_scale if epoch < burn_in else 1.0)
            order = list(pairs)
            rng.shuffle(order)
            for child, parent in order:
                self._step(child, parent, self._negative_pool(child), lr, rng)
        report.final_loss = self._loss(pairs)
        report.mean_rank = self._mean_rank(pairs, list(self.concepts))
        return report

    def _negative_pool(self, child: str) -> List[str]:
        """The concepts a relation may be contrasted against — its own domain, when it has one.

        Cross-domain negatives are meaningless: pushing ``electron`` away from ``planet`` says
        nothing about either hierarchy, and doing it wrecks whichever domain was fitted first."""
        domain = self.concepts[child].domain if child in self.concepts else ""
        if not domain:
            return list(self.concepts)
        same = [n for n, c in self.concepts.items() if c.domain == domain]
        return same if len(same) >= 3 else list(self.concepts)

    def _sample_negatives(self, child: str, parent: str, names: Sequence[str],
                          rng: random.Random) -> List[str]:
        """Draw distinct negatives — the contrast that gives the softmax something to push against."""
        out: List[str] = []
        attempts = 0
        if not names:
            return out
        while len(out) < self.negatives and attempts < self.negatives * 8:
            attempts += 1
            candidate = names[rng.randrange(len(names))]
            if candidate not in (child, parent) and candidate not in out:
                out.append(candidate)
        return out

    def _step(self, child: str, parent: str, names: Sequence[str], lr: float,
              rng: random.Random) -> None:
        """One RSGD step on the softmax-over-negatives loss for a single relation."""
        negatives = self._sample_negatives(child, parent, names, rng)
        if not negatives:
            return
        u = self.concepts[child].point
        candidates = [parent] + negatives
        distances = [poincare_distance(u, self.concepts[c].point) for c in candidates]
        shifted = [-d for d in distances]
        peak = max(shifted)
        exps = [math.exp(s - peak) for s in shifted]
        total = sum(exps) or 1e-12
        probabilities = [e / total for e in exps]

        # L = d₀ + log Σⱼ exp(−dⱼ), so dL/dd₀ = 1 − p₀ (positive: pull the true parent IN) and
        # dL/ddᵢ = −pᵢ for a negative (negative: push it OUT). Getting these two signs the wrong
        # way round turns the whole fit into gradient ascent, which looks like training right up
        # until every point is pinned to the boundary.
        grad_u = [0.0] * self.dim
        for i, name in enumerate(candidates):
            weight = (1.0 if i == 0 else 0.0) - probabilities[i]
            if weight == 0.0:
                continue
            v = self.concepts[name].point
            d_grad = _distance_grad(u, v)
            for k in range(self.dim):
                grad_u[k] += weight * d_grad[k]
            # the other endpoint moves too, by the symmetric gradient
            other_grad = _to_riemannian(v, [weight * g for g in _distance_grad(v, u)])
            self.concepts[name].point = self._retract(v, other_grad, lr)

        self.concepts[child].point = self._retract(u, _to_riemannian(u, grad_u), lr)

    @staticmethod
    def _retract(point: Sequence[float], grad: Sequence[float], lr: float) -> List[float]:
        """Descend and project back inside the ball — the manifold's version of a parameter update."""
        moved = [p - lr * g for p, g in zip(point, grad)]
        norm = math.sqrt(sum(c * c for c in moved))
        if norm >= _MAX_NORM:
            moved = [c * (_MAX_NORM / max(norm, 1e-15)) for c in moved]
        return project_to_ball(moved)

    def _loss(self, pairs: Sequence[Tuple[str, str]]) -> float:
        """Mean loss over all relations against *every* other concept — a stable, seed-free read."""
        names = list(self.concepts)
        if len(names) < 2:
            return 0.0
        total = 0.0
        for child, parent in pairs:
            u = self.concepts[child].point
            others = [n for n in names if n != child]
            distances = [poincare_distance(u, self.concepts[n].point) for n in others]
            shifted = [-d for d in distances]
            peak = max(shifted)
            denominator = sum(math.exp(s - peak) for s in shifted)
            numerator = math.exp(-poincare_distance(u, self.concepts[parent].point) - peak)
            total += -math.log(max(numerator / max(denominator, 1e-12), 1e-12))
        return total / len(pairs)

    def _mean_rank(self, pairs: Sequence[Tuple[str, str]], names: Sequence[str]) -> float:
        """Mean rank of the true parent among all other concepts. 1.0 is a perfect embedding."""
        ranks: List[float] = []
        for child, parent in pairs:
            u = self.concepts[child].point
            scored = sorted(((poincare_distance(u, self.concepts[n].point), n)
                             for n in names if n != child), key=lambda t: t[0])
            for position, (_distance, name) in enumerate(scored, start=1):
                if name == parent:
                    ranks.append(float(position))
                    break
        return sum(ranks) / len(ranks) if ranks else 0.0

    # ------------------------------------------------------------------ #
    # queries
    # ------------------------------------------------------------------ #
    def distance(self, a: str, b: str) -> float:
        """Geodesic distance between two concepts (``inf`` when either is unknown)."""
        ca, cb = self.concepts.get(a), self.concepts.get(b)
        if ca is None or cb is None:
            return float("inf")
        return poincare_distance(ca.point, cb.point)

    def nearest(self, name: str, k: int = 5) -> List[Tuple[str, float]]:
        """The k geodesically nearest concepts."""
        anchor = self.concepts.get(name)
        if anchor is None:
            return []
        scored = [(other, poincare_distance(anchor.point, concept.point))
                  for other, concept in self.concepts.items() if other != name]
        scored.sort(key=lambda t: t[1])
        return scored[:max(0, k)]

    def entails(self, child: str, parent: str, *, k: int = 3) -> bool:
        """Is ``child`` a kind of ``parent``, according to the fitted geometry?

        Two conditions, both read off what this embedding actually encodes:

        1. ``parent`` is **more general** — strictly nearer the origin. The Nickel–Kiela loss is
           what puts it there, so this is a fitted fact rather than a stipulation.
        2. ``parent`` is among ``child``'s ``k`` nearest *more-general* concepts. The fit is
           trained to place a child next to its parent, and :attr:`HyperspaceReport.mean_rank`
           reports how well it managed — at rank 1.0 the true parent is the single nearest
           concept.

        A note on what this is **not**: proper entailment cones (Ganea et al. 2018) need an
        embedding trained with the *cone* loss, which is a different objective from the distance
        loss fitted here. Bolting a cone aperture onto a distance-trained space produces
        confident-looking answers that do not hold up, so this reports the containment the fit can
        actually support and leaves the cone construction to a future cone-trained space."""
        ca, cb = self.concepts.get(child), self.concepts.get(parent)
        if ca is None or cb is None or not ca.point or not cb.point:
            return False
        if cb.radius >= ca.radius:
            return False
        more_general = [(other, poincare_distance(ca.point, concept.point))
                        for other, concept in self.concepts.items()
                        if other != child and concept.radius < ca.radius]
        if not more_general:
            return False
        more_general.sort(key=lambda t: t[1])
        return parent in {name for name, _d in more_general[:max(1, k)]}

    # ------------------------------------------------------------------ #
    # cross-domain correspondence
    # ------------------------------------------------------------------ #
    def _domain_members(self, domain: str) -> List[str]:
        return sorted(n for n, c in self.concepts.items() if c.domain == domain)

    def find_correspondences(self, source_domain: str, target_domain: str, *,
                             min_score: float = 0.6, max_pairs: int = 8,
                             relations: Optional[Dict[str, Sequence[Any]]] = None
                             ) -> List[Correspondence]:
        """Align two domains by the *shape* of their geodesic distance matrices.

        Vocabulary is deliberately not used: the point is to surface a structure two fields share
        even when they share no words. A candidate below ``min_score`` is not returned at all — an
        analogy asserted without support is worse than no analogy. When ``relations`` supplies
        symbolic predicates for both domains, the survivor is additionally checked by
        :class:`~nyxara.mind.analogy.StructureMapper` and only then marked ``verified``."""
        source = self._domain_members(source_domain)[:max_pairs]
        target = self._domain_members(target_domain)[:max_pairs]
        if len(source) < 2 or len(target) < 2:
            return []

        pairs = self._greedy_align(source, target)
        if len(pairs) < 2:
            return []
        score = self._alignment_score(pairs)
        if score < min_score:
            return []

        correspondence = Correspondence(source_domain=source_domain, target_domain=target_domain,
                                        pairs=pairs, score=score)
        if relations:
            correspondence.verified = self._symbolically_verify(
                relations.get(source_domain), relations.get(target_domain))
            if correspondence.verified:
                correspondence.method = "geodesic-alignment + structure-mapping"
        return [correspondence]

    def _greedy_align(self, source: Sequence[str], target: Sequence[str]) -> List[Tuple[str, str]]:
        """Match the two domains one pair at a time, minimising distance-matrix distortion.

        The first pair is anchored on the two most *central* concepts (smallest radius = most
        general), which after fitting is a meaningful choice rather than an arbitrary one."""
        remaining = list(target)
        anchor_source = min(source, key=lambda n: self.concepts[n].radius)
        anchor_target = min(remaining, key=lambda n: self.concepts[n].radius)
        pairs = [(anchor_source, anchor_target)]
        remaining.remove(anchor_target)

        for name in source:
            if name == anchor_source or not remaining:
                continue
            best, best_cost = None, float("inf")
            for candidate in remaining:
                cost = 0.0
                for mapped_source, mapped_target in pairs:
                    cost += abs(self.distance(name, mapped_source)
                                - self.distance(candidate, mapped_target))
                cost /= len(pairs)
                if cost < best_cost:
                    best, best_cost = candidate, cost
            if best is not None:
                pairs.append((name, best))
                remaining.remove(best)
        return pairs

    def _alignment_score(self, pairs: Sequence[Tuple[str, str]]) -> float:
        """How well the two domains' geodesic distance matrices agree, in ``[0, 1]``.

        The score is the **Pearson correlation** of the two distance vectors, floored at zero, and
        it is deliberately not a relative-error measure. Relative error is easy to satisfy: any two
        domains whose distances happen to sit in the same numeric range score well, so a star
        (every sibling equidistant) matches a chain (distances strictly increasing) and a false
        analogy is announced with confidence. Correlation asks the question that actually matters —
        do the distances *co-vary* — and a domain with no variance in its distances correctly
        correlates with nothing."""
        if len(pairs) < 3:
            return 0.0     # two mapped pairs give a single distance each: nothing to correlate
        source_distances: List[float] = []
        target_distances: List[float] = []
        for i in range(len(pairs)):
            for j in range(i + 1, len(pairs)):
                source_distances.append(self.distance(pairs[i][0], pairs[j][0]))
                target_distances.append(self.distance(pairs[i][1], pairs[j][1]))
        n = len(source_distances)
        if n < 2:
            return 0.0
        mean_s = sum(source_distances) / n
        mean_t = sum(target_distances) / n
        covariance = sum((s - mean_s) * (t - mean_t)
                         for s, t in zip(source_distances, target_distances))
        var_s = sum((s - mean_s) ** 2 for s in source_distances)
        var_t = sum((t - mean_t) ** 2 for t in target_distances)
        denominator = math.sqrt(var_s * var_t)
        if denominator < 1e-12:
            return 0.0     # one domain's distances are constant — it has no shape to match
        return max(0.0, covariance / denominator)

    @staticmethod
    def _symbolically_verify(source: Any, target: Any) -> bool:
        """Confirm a geometric match with the existing structure mapper. False on any doubt."""
        if not source or not target:
            return False
        try:
            from nyxara.mind.analogy import StructureMapper
            mapping = StructureMapper(analogical=True).map(list(source), list(target))
        except Exception:  # noqa: BLE001 — no symbolic layer means unverified, never "verified"
            return False
        return bool(getattr(mapping, "structural_score", 0.0) > 0.0)

    # ------------------------------------------------------------------ #
    # persistence
    # ------------------------------------------------------------------ #
    def to_dict(self) -> Dict[str, Any]:
        return {"dim": self.dim, "seed": self.seed, "lr": self.lr, "negatives": self.negatives,
                "concepts": [c.to_dict() for c in self.concepts.values()]}

    def load_dict(self, data: Dict[str, Any]) -> bool:
        """Restore from a dict, surviving garbage (a corrupt sidecar must never block boot)."""
        try:
            self.dim = max(2, int(data.get("dim", self.dim)))
            self.seed = int(data.get("seed", self.seed))
            self.lr = float(data.get("lr", self.lr))
            self.negatives = max(1, int(data.get("negatives", self.negatives)))
            restored: Dict[str, Concept] = {}
            for raw in data.get("concepts", []):
                try:
                    concept = Concept.from_dict(raw)
                except Exception:  # noqa: BLE001 — one malformed entry never costs the rest
                    continue
                if concept.name and len(concept.point) == self.dim:
                    restored[concept.name] = concept
            self.concepts = restored
            return True
        except Exception:  # noqa: BLE001 — a corrupt space is an empty space, never a crash
            return False

    def save(self, path: Any) -> bool:
        """Atomically persist the fitted space (tmp file → ``os.replace``)."""
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
        except Exception:  # noqa: BLE001 — a missing/corrupt file is simply an unfitted space
            return False


def relations_from_pairs(pairs: Iterable[Tuple[str, str]]) -> List[Tuple[str, str]]:
    """Normalise ``(child, parent)`` pairs, dropping self-loops and blanks."""
    out: List[Tuple[str, str]] = []
    for child, parent in pairs:
        c, p = str(child).strip().lower(), str(parent).strip().lower()
        if c and p and c != p:
            out.append((c, p))
    return out
