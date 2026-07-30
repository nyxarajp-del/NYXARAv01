"""NYXARA · mind/hyperbolic_manifold.py — self-mutating hyperbolic concept manifold (🌀, Rule 4, F4).

Concepts live as points in the **Poincaré ball** (‖v‖ < 1), where hierarchical structure fits
naturally: hyperbolic volume grows exponentially with radius, so a taxonomy that would crush
itself in flat space spreads out toward the boundary. Distance between concepts is the true
Riemannian geodesic ``d(u,v) = arcosh(1 + 2‖u−v‖² / ((1−‖u‖²)(1−‖v‖²)))`` — reused from
:mod:`nyxara.mind.latent_geometry`, the single source of that formula.

Unlike a table of static weights, this layer **mutates during live inference**:

* **Node genesis.** When a turn lands farther than a threshold ``τ`` (geodesic) from every
  concept she knows, a new node is born — placed at the **hyperbolic barycenter** of the active
  context cluster (Einstein midpoint in the Klein model, refined by a few Riemannian-gradient
  steps toward the weighted Fréchet mean). Never a Euclidean average: means are taken on the
  manifold, not in the ambient coordinates.

* **Hebbian edge adaptation.** After each turn's real outcome is known, edges among the concepts
  that turn touched update by ``Δe_ij = η · utility · (1 − d(v_i,v_j)/d_norm)`` — the classic
  proximity-gated Hebbian rule, with the (unbounded) Poincaré distance honestly normalised by
  ``d_norm`` so the proximity factor stays in [0, 1]. Edges decay each cycle and are pruned
  below a floor; nodes beyond a cap are evicted weakest-and-stalest first.

* **Assimilation.** A familiar turn does not spawn a node: the nearest concept drifts a step
  along the geodesic toward the observation (``v ← exp_v(lr · log_v(q))``), so positions are
  *learned* from the lived stream, not laid out once.

Pure standard library, deterministic given a seed, no wall-clock in the math. Advisory colour
for the rest of the mind — it never gates.
"""

from __future__ import annotations

import math
import random
import zlib
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Sequence, Tuple

from nyxara.mind.latent_geometry import poincare_distance

__all__ = [
    "BALL_EPS",
    "project_to_ball",
    "mobius_add",
    "exp_map",
    "log_map",
    "einstein_midpoint",
    "frechet_mean",
    "BallEmbedder",
    "ObserveResult",
    "HyperbolicConceptManifold",
]

Vec = List[float]

BALL_EPS = 1e-5


# --------------------------------------------------------------------------- #
# Poincaré-ball operations (gyrovector calculus, curvature −1)
# --------------------------------------------------------------------------- #
def _norm(v: Sequence[float]) -> float:
    return math.sqrt(sum(c * c for c in v))


def _dot(u: Sequence[float], v: Sequence[float]) -> float:
    return sum(a * b for a, b in zip(u, v))


def project_to_ball(v: Sequence[float], eps: float = BALL_EPS) -> Vec:
    """Clip a point back inside the open unit ball (norm ≤ 1−eps). Every op ends here."""
    n = _norm(v)
    limit = 1.0 - eps
    if n >= limit:
        scale = limit / max(n, 1e-15)
        return [c * scale for c in v]
    return list(v)


def mobius_add(u: Sequence[float], v: Sequence[float]) -> Vec:
    """Möbius addition u ⊕ v — the hyperbolic analogue of vector addition in the ball."""
    uu = _dot(u, u)
    vv = _dot(v, v)
    uv = _dot(u, v)
    cu = 1.0 + 2.0 * uv + vv
    cv = 1.0 - uu
    denom = 1.0 + 2.0 * uv + uu * vv
    if abs(denom) < 1e-12:
        denom = 1e-12 if denom >= 0 else -1e-12
    return project_to_ball([(cu * a + cv * b) / denom for a, b in zip(u, v)])


def _lambda_x(x: Sequence[float]) -> float:
    """Conformal factor λ_x = 2 / (1 − ‖x‖²)."""
    return 2.0 / max(1.0 - _dot(x, x), 1e-12)


def exp_map(x: Sequence[float], t: Sequence[float]) -> Vec:
    """Exponential map: shoot from x along tangent vector t, landing on the manifold."""
    tn = _norm(t)
    if tn < 1e-15:
        return list(x)
    arg = math.tanh(_lambda_x(x) * tn / 2.0)
    scaled = [arg * c / tn for c in t]
    return mobius_add(x, scaled)


def log_map(x: Sequence[float], y: Sequence[float]) -> Vec:
    """Logarithm map: the tangent vector at x that exp_map shoots to reach y (inverse of exp)."""
    w = mobius_add([-c for c in x], y)
    wn = _norm(w)
    if wn < 1e-15:
        return [0.0] * len(x)
    wn_clamped = min(wn, 1.0 - 1e-12)
    factor = (2.0 / _lambda_x(x)) * math.atanh(wn_clamped) / wn
    return [factor * c for c in w]


def einstein_midpoint(points: Sequence[Sequence[float]],
                      weights: Optional[Sequence[float]] = None) -> Vec:
    """Weighted Einstein midpoint: Lorentz-factor-weighted mean in the Klein model.

    A closed-form hyperbolic centroid — the proper initialisation for the Fréchet mean,
    never a Euclidean average of ball coordinates.
    """
    if not points:
        return []
    dim = len(points[0])
    if weights is None:
        weights = [1.0] * len(points)
    num = [0.0] * dim
    den = 0.0
    for p, w in zip(points, weights):
        pp = _dot(p, p)
        k = [2.0 * c / (1.0 + pp) for c in p]  # Poincaré → Klein
        gamma = 1.0 / math.sqrt(max(1.0 - _dot(k, k), 1e-12))
        wg = w * gamma
        for i in range(dim):
            num[i] += wg * k[i]
        den += wg
    m_k = [c / max(den, 1e-12) for c in num]
    kk = _dot(m_k, m_k)
    back = 1.0 + math.sqrt(max(1.0 - kk, 0.0))  # Klein → Poincaré
    return project_to_ball([c / back for c in m_k])


def frechet_mean(points: Sequence[Sequence[float]],
                 weights: Optional[Sequence[float]] = None,
                 *, iters: int = 3, lr: float = 0.5) -> Vec:
    """Weighted Fréchet mean: Einstein-midpoint init + Riemannian gradient descent.

    Each step moves along ``m ← exp_m(lr · Σ wᵢ log_m(xᵢ) / Σ wᵢ)`` — gradient descent on the
    weighted sum of squared geodesic distances, entirely on the manifold.
    """
    if not points:
        return []
    if weights is None:
        weights = [1.0] * len(points)
    m = einstein_midpoint(points, weights)
    total = max(sum(weights), 1e-12)
    for _ in range(max(0, iters)):
        g = [0.0] * len(m)
        for p, w in zip(points, weights):
            t = log_map(m, p)
            for i in range(len(g)):
                g[i] += w * t[i]
        g = [c / total for c in g]
        if _norm(g) < 1e-12:
            break
        m = exp_map(m, [lr * c for c in g])
    return project_to_ball(m)


# --------------------------------------------------------------------------- #
# text / vector → point in the ball (deterministic given (dim, seed))
# --------------------------------------------------------------------------- #
class BallEmbedder:
    """Deterministic text/vector → Poincaré-ball point.

    Feature floor is a stdlib hashing bag (word tokens + character trigrams); an external
    embedder's vector can be passed instead via :meth:`embed_vector`, so the manifold inherits
    the strongest semantics available without ever *requiring* one. A fixed seeded Gaussian
    matrix projects features to ``dim``; ``tanh`` shrinks the result inside the ball.
    """

    def __init__(self, dim: int = 8, seed: int = 42, alpha: float = 0.5,
                 feature_dim: int = 128) -> None:
        self.dim = int(dim)
        self.seed = int(seed)
        self.alpha = float(alpha)
        self.feature_dim = int(feature_dim)
        self._matrices: Dict[int, List[List[float]]] = {}

    def _matrix(self, in_dim: int) -> List[List[float]]:
        """Fixed Gaussian projection (in_dim × dim), drawn lazily per source width."""
        mat = self._matrices.get(in_dim)
        if mat is None:
            rng = random.Random(self.seed * 1_000_003 + in_dim)
            mat = [[rng.gauss(0.0, 1.0 / math.sqrt(self.dim)) for _ in range(self.dim)]
                   for _ in range(in_dim)]
            self._matrices[in_dim] = mat
        return mat

    def featurize(self, text: str) -> Vec:
        """Hash-bucket bag of lowercase word tokens + char trigrams, L2-normalised."""
        feats = [0.0] * self.feature_dim
        low = (text or "").lower()
        tokens = low.split()
        grams = [low[i:i + 3] for i in range(max(0, len(low) - 2))]
        for tok in tokens + grams:
            # crc32, not hash(): stable across processes so persisted points stay comparable
            h = zlib.crc32(f"{self.seed}\x00{tok}".encode("utf-8", "ignore"))
            idx = h % self.feature_dim
            feats[idx] += 1.0 if (h >> 16) % 2 == 0 else -1.0
        n = _norm(feats)
        if n > 0:
            feats = [c / n for c in feats]
        return feats

    def project(self, feats: Sequence[float]) -> Vec:
        """Project features to dim and shrink into the ball via tanh radial scaling."""
        mat = self._matrix(len(feats))
        z = [0.0] * self.dim
        for f, row in zip(feats, mat):
            if f == 0.0:
                continue
            for i in range(self.dim):
                z[i] += f * row[i]
        zn = _norm(z)
        if zn < 1e-15:
            return [0.0] * self.dim
        r = math.tanh(self.alpha * zn)
        return project_to_ball([r * c / zn for c in z])

    def embed_text(self, text: str) -> Vec:
        return self.project(self.featurize(text))

    def embed_vector(self, vec: Sequence[float]) -> Vec:
        """Embed an external embedder's output (any width) — normalised, then projected."""
        n = _norm(vec)
        v = [c / n for c in vec] if n > 0 else list(vec)
        return self.project(v)


# --------------------------------------------------------------------------- #
# the self-mutating layer
# --------------------------------------------------------------------------- #
@dataclass
class ObserveResult:
    """What one observation did to the manifold."""
    node_id: str
    point: Vec = field(default_factory=list)
    created: bool = False
    min_dist: float = math.inf
    nearest_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "point": list(self.point),
            "created": self.created,
            "min_dist": (self.min_dist if math.isfinite(self.min_dist) else None),
            "nearest_id": self.nearest_id,
        }


class HyperbolicConceptManifold:
    """Concept nodes in the Poincaré ball with live genesis and Hebbian edge plasticity.

    ``observe`` is the mutation operator (genesis beyond ``tau``, geodesic assimilation
    otherwise); ``reinforce_active`` applies the Hebbian rule to the recent working set once
    the turn's utility is known. Bounded: edges decay/prune, nodes evict past ``max_nodes``.
    Deterministic given ``seed``; fully serialisable via ``to_dict``/``load_dict``.
    """

    def __init__(self, *, dim: int = 8, tau: float = 1.2, eta: float = 0.1,
                 decay: float = 0.02, edge_floor: float = 0.05, max_nodes: int = 512,
                 seed: int = 42, place_lr: float = 0.05, frechet_iters: int = 3,
                 genesis_edge_weight: float = 0.3, d_norm: float = 5.0,
                 prune_every: int = 25, active_window: int = 8,
                 embedder: Optional[Any] = None) -> None:
        self.dim = int(dim)
        self.tau = float(tau)
        self.eta = float(eta)
        self.decay = float(decay)
        self.edge_floor = float(edge_floor)
        self.max_nodes = int(max_nodes)
        self.seed = int(seed)
        self.place_lr = float(place_lr)
        self.frechet_iters = int(frechet_iters)
        self.genesis_edge_weight = float(genesis_edge_weight)
        self.d_norm = float(d_norm)
        self.prune_every = int(prune_every)
        self.active_window = int(active_window)
        self.embedder = embedder if embedder is not None else BallEmbedder(dim=dim, seed=seed)
        self._nodes: Dict[str, Vec] = {}
        self._edges: Dict[Tuple[str, str], float] = {}
        self._last_used: Dict[str, int] = {}
        self._active: Deque[str] = deque(maxlen=self.active_window)
        self._tick: int = 0
        self._genesis_count: int = 0

    # ------------------------------------------------------------------ #
    # observation: genesis or assimilation
    # ------------------------------------------------------------------ #
    def observe(self, text: Optional[str] = None, *, vector: Optional[Sequence[float]] = None,
                concept_id: Optional[str] = None,
                context_ids: Optional[Sequence[str]] = None) -> ObserveResult:
        """The dynamic mutation operator, called live on each turn.

        A query farther than ``tau`` (geodesic) from everything known births a new node at the
        weighted Fréchet mean of the active context cluster (the query itself included, weight
        2.0); a familiar query pulls its nearest node one geodesic step toward the observation.
        """
        if vector is not None:
            q = self.embedder.embed_vector(vector)
        elif text is not None:
            q = self.embedder.embed_text(text)
        else:
            raise ValueError("observe() needs text or vector")

        min_dist = math.inf
        nearest: Optional[str] = None
        for nid, p in self._nodes.items():
            d = poincare_distance(q, p)
            if d < min_dist:
                min_dist, nearest = d, nid

        if nearest is None or min_dist > self.tau:
            node_id = self._genesis(q, text, concept_id, context_ids)
            result = ObserveResult(node_id=node_id, point=list(self._nodes[node_id]),
                                   created=True, min_dist=min_dist, nearest_id=nearest)
        else:
            # assimilation: the nearest concept drifts along the geodesic toward the query
            v = self._nodes[nearest]
            self._nodes[nearest] = exp_map(v, [self.place_lr * c for c in log_map(v, q)])
            node_id = nearest
            result = ObserveResult(node_id=node_id, point=list(self._nodes[node_id]),
                                   created=False, min_dist=min_dist, nearest_id=nearest)

        self._active.append(node_id)
        self._tick += 1
        self._last_used[node_id] = self._tick
        if self.prune_every > 0 and self._tick % self.prune_every == 0:
            self.decay_and_prune()
        if len(self._nodes) > self.max_nodes:
            self._evict()
        return result

    def _genesis(self, q: Vec, text: Optional[str], concept_id: Optional[str],
                 context_ids: Optional[Sequence[str]]) -> str:
        """Birth a node at the hyperbolic barycenter of {query (w=2)} ∪ active context cluster."""
        ctx: List[str] = []
        pool = context_ids if context_ids else list(self._active)
        for cid in pool:
            if cid in self._nodes and cid not in ctx:
                ctx.append(cid)
        points: List[Vec] = [q]
        weights: List[float] = [2.0]
        for cid in ctx:
            points.append(self._nodes[cid])
            weights.append(max(0.5, self._mean_incident_weight(cid)))
        v_new = frechet_mean(points, weights, iters=self.frechet_iters)
        node_id = concept_id if concept_id else f"c{self._tick}:{(text or '')[:24]}"
        if node_id in self._nodes:  # ids must stay unique across genesis events
            node_id = f"{node_id}#{self._tick}"
        self._nodes[node_id] = v_new
        for cid in ctx:
            self._edges[self._ekey(node_id, cid)] = self.genesis_edge_weight
        self._genesis_count += 1
        return node_id

    # ------------------------------------------------------------------ #
    # Hebbian plasticity
    # ------------------------------------------------------------------ #
    @staticmethod
    def _ekey(i: str, j: str) -> Tuple[str, str]:
        return (i, j) if i <= j else (j, i)

    def _mean_incident_weight(self, nid: str) -> float:
        ws = [w for (a, b), w in self._edges.items() if nid in (a, b)]
        return sum(ws) / len(ws) if ws else 0.0

    def reinforce(self, i: str, j: str, utility: float) -> Optional[float]:
        """Hebbian edge adaptation: Δe = η · clamp(utility) · max(0, 1 − d/d_norm).

        The Poincaré distance is unbounded, so the proximity factor ``(1 − d)`` of the classic
        rule is honestly normalised by ``d_norm``; weights stay clamped to [0, 1].
        """
        if i == j or i not in self._nodes or j not in self._nodes:
            return None
        u = max(-1.0, min(1.0, float(utility)))
        d = poincare_distance(self._nodes[i], self._nodes[j])
        delta = self.eta * u * max(0.0, 1.0 - d / self.d_norm)
        key = self._ekey(i, j)
        if key not in self._edges:
            if delta <= self.edge_floor:
                return None
            self._edges[key] = 0.0
        w = max(0.0, min(1.0, self._edges[key] + delta))
        self._edges[key] = w
        return w

    def reinforce_active(self, utility: float) -> int:
        """Apply the Hebbian rule to every pair in the active window; returns pairs updated."""
        ids = list(dict.fromkeys(self._active))
        count = 0
        for a in range(len(ids)):
            for b in range(a + 1, len(ids)):
                if self.reinforce(ids[a], ids[b], utility) is not None:
                    count += 1
        return count

    def decay_and_prune(self) -> Tuple[int, int]:
        """Decay every edge by (1 − decay); prune edges below the floor. → (decayed, pruned)."""
        decayed = 0
        dead: List[Tuple[str, str]] = []
        for key in self._edges:
            self._edges[key] *= (1.0 - self.decay)
            decayed += 1
            if self._edges[key] < self.edge_floor:
                dead.append(key)
        for key in dead:
            del self._edges[key]
        return decayed, len(dead)

    def _evict(self) -> None:
        """Drop weakest-and-stalest nodes (never the active window) until under max_nodes."""
        protected = set(self._active)
        while len(self._nodes) > self.max_nodes:
            candidates = [nid for nid in self._nodes if nid not in protected]
            if not candidates:
                break
            victim = min(candidates,
                         key=lambda nid: (sum(w for (a, b), w in self._edges.items()
                                              if nid in (a, b)),
                                          self._last_used.get(nid, 0)))
            del self._nodes[victim]
            self._last_used.pop(victim, None)
            for key in [k for k in self._edges if victim in k]:
                del self._edges[key]

    # ------------------------------------------------------------------ #
    # queries & introspection
    # ------------------------------------------------------------------ #
    def nearest(self, query: Any, k: int = 5) -> List[Tuple[str, float]]:
        """k nearest concepts (id, geodesic distance) to a text or a ball point."""
        if isinstance(query, str):
            q = self.embedder.embed_text(query)
        else:
            q = list(query)
        scored = [(nid, poincare_distance(q, p)) for nid, p in self._nodes.items()]
        scored.sort(key=lambda t: t[1])
        return scored[:max(1, int(k))]

    def distance(self, id_a: str, id_b: str) -> Optional[float]:
        """Geodesic distance between two known concepts (None if either is unknown)."""
        a = self._nodes.get(id_a)
        b = self._nodes.get(id_b)
        if a is None or b is None:
            return None
        return poincare_distance(a, b)

    def edge_weight(self, id_a: str, id_b: str) -> Optional[float]:
        return self._edges.get(self._ekey(id_a, id_b))

    def stats(self) -> Dict[str, Any]:
        radii = [_norm(p) for p in self._nodes.values()]
        weights = list(self._edges.values())
        return {
            "n_nodes": len(self._nodes),
            "n_edges": len(self._edges),
            "genesis_count": self._genesis_count,
            "tick": self._tick,
            "mean_edge_weight": (sum(weights) / len(weights)) if weights else 0.0,
            "mean_radius": (sum(radii) / len(radii)) if radii else 0.0,
            "max_radius": max(radii) if radii else 0.0,
        }

    # ------------------------------------------------------------------ #
    # persistence
    # ------------------------------------------------------------------ #
    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": 1,
            "dim": self.dim,
            "tau": self.tau,
            "nodes": {nid: list(p) for nid, p in self._nodes.items()},
            "edges": {f"{a}||{b}": w for (a, b), w in self._edges.items()},
            "last_used": dict(self._last_used),
            "active": list(self._active),
            "tick": self._tick,
            "genesis_count": self._genesis_count,
        }

    def load_dict(self, data: Dict[str, Any]) -> None:
        try:
            nodes = data.get("nodes", {})
            self._nodes = {str(nid): project_to_ball([float(c) for c in p])
                           for nid, p in nodes.items()}
            self._edges = {}
            for key, w in data.get("edges", {}).items():
                a, _, b = str(key).partition("||")
                if a in self._nodes and b in self._nodes:
                    self._edges[self._ekey(a, b)] = max(0.0, min(1.0, float(w)))
            self._last_used = {str(k): int(v) for k, v in data.get("last_used", {}).items()
                               if str(k) in self._nodes}
            self._active = deque((nid for nid in data.get("active", []) if nid in self._nodes),
                                 maxlen=self.active_window)
            self._tick = int(data.get("tick", 0))
            self._genesis_count = int(data.get("genesis_count", 0))
        except Exception:  # noqa: BLE001 — a corrupt sidecar must never break the mind
            pass


# --------------------------------------------------------------------------- #
# Self-test / demo
# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # pragma: no cover
    print("=" * 70)
    print("NYXARA hyperbolic-manifold self-test (pure stdlib)")
    print("=" * 70)

    rng = random.Random(7)
    zero = [0.0, 0.0, 0.0]
    v = [0.3, -0.2, 0.1]

    # 1) Möbius identity and inverse
    assert max(abs(a - b) for a, b in zip(mobius_add(zero, v), v)) < 1e-12
    inv = mobius_add([-c for c in v], v)
    print(f"möbius identity/inverse  : ‖(−v)⊕v‖ = {_norm(inv):.2e} (≈0 expected)")
    assert _norm(inv) < 1e-9

    # 2) exp/log round trip
    x = [0.2, 0.1, -0.3]
    y = [-0.4, 0.25, 0.1]
    y2 = exp_map(x, log_map(x, y))
    err = max(abs(a - b) for a, b in zip(y, y2))
    print(f"exp∘log round trip       : max err = {err:.2e}")
    assert err < 1e-6

    # 3) midpoint stays in the ball, roughly equidistant
    m = einstein_midpoint([x, y])
    print(f"einstein midpoint        : ‖m‖ = {_norm(m):.3f}  "
          f"d(x,m)={poincare_distance(x, m):.3f} d(m,y)={poincare_distance(m, y):.3f}")
    assert _norm(m) < 1.0

    # 4) genesis demo: three near concepts, one far → node born at the barycenter
    mm = HyperbolicConceptManifold(dim=8, tau=2.2, seed=11,
                                   embedder=BallEmbedder(dim=8, seed=11, alpha=1.0))
    for t in ("the moon orbits the earth", "lunar orbital mechanics", "orbits of the moon"):
        r = mm.observe(t)
        print(f"observe {t!r:34s} → created={r.created} d_min="
              f"{'∞' if not math.isfinite(r.min_dist) else f'{r.min_dist:.2f}'}")
    r = mm.observe("sourdough bread recipe")
    print(f"observe 'sourdough...'   → created={r.created} (genesis expected)")
    assert r.created

    # 5) Hebbian reinforcement + decay
    before = mm.stats()["mean_edge_weight"]
    n = mm.reinforce_active(1.0)
    after = mm.stats()["mean_edge_weight"]
    mm.decay_and_prune()
    decayed = mm.stats()["mean_edge_weight"]
    print(f"hebbian reinforcement    : pairs={n} mean edge {before:.3f} → {after:.3f} "
          f"→ decay {decayed:.3f}")
    assert after >= before

    print("\nstats:", mm.stats())
    print("OK — all self-tests passed.")
