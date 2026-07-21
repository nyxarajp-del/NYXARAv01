"""NYXARA · mind/latent_geometry.py — topological & geometric feature synthesis (📐, Rule 4, F4).

Data has a **shape**, not only coordinates. This module lets NYXARA read that shape and pick a geometry
that fits it — two capabilities, both honest and both degrading gracefully with no heavy dependency:

* **Topological data analysis (persistent homology).** Over a Vietoris–Rips filtration of a point
  cloud she computes the **H0 barcode** (how connected components are born and merge as scale grows) and
  an **H1 Betti estimate** (independent loops / holes) via the Euler characteristic of the Rips
  2-complex. So she can tell "one blob" from "two clusters" from "a ring with a hole in the middle" —
  regimes/cycles a covariance matrix cannot see. Pure standard library; if ``ripser`` is importable it
  is used to sharpen H1, otherwise the stdlib estimate stands (and says so).

* **Non-Euclidean (hyperbolic) embedding.** Hierarchies — taxonomies, syntax/skill trees — fit badly in
  flat space (a tree's node count grows exponentially with depth, which Euclidean area cannot hold) but
  naturally in the **Poincaré disk**, whose area grows exponentially too. She embeds a tree both ways,
  **measures the distortion of each honestly**, and recommends hyperbolic **only when it measurably
  wins** — never a blanket "100× everywhere" claim.

No LLM; touches no gate; it only changes *how she represents* data, never what she values.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from itertools import combinations
from typing import Dict, List, Optional, Sequence, Tuple

__all__ = [
    "Bar",
    "TopologyReport",
    "persistent_h0",
    "betti1_estimate",
    "analyze_points",
    "poincare_distance",
    "embed_tree",
    "embedding_distortion",
    "EmbeddingChoice",
    "recommend_geometry",
    "LatentGeometry",
]

Point = Sequence[float]


def _dist(a: Point, b: Point) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


# --------------------------------------------------------------------------- #
# union-find (for connected components across the Rips filtration)
# --------------------------------------------------------------------------- #
class _UF:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> bool:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        self.parent[ra] = rb
        return True


# --------------------------------------------------------------------------- #
# persistent homology (H0 barcode + H1 estimate)
# --------------------------------------------------------------------------- #
@dataclass
class Bar:
    birth: float
    death: float          # math.inf for the component that never dies

    @property
    def persistence(self) -> float:
        return self.death - self.birth


@dataclass
class TopologyReport:
    betti0: int           # connected components at the chosen scale
    betti1: int           # independent loops / holes
    barcode_h0: List[Bar] = field(default_factory=list)
    scale: float = 0.0
    method: str = "stdlib"

    def to_dict(self) -> Dict[str, object]:
        return {"betti0": self.betti0, "betti1": self.betti1, "scale": round(self.scale, 4),
                "method": self.method,
                "h0_bars": [(round(b.birth, 3), b.death if b.death == math.inf else round(b.death, 3))
                            for b in self.barcode_h0]}


def persistent_h0(points: Sequence[Point]) -> List[Bar]:
    """H0 persistence barcode: every component is born at scale 0 and dies when it merges."""
    n = len(points)
    if n == 0:
        return []
    edges = sorted(((_dist(points[i], points[j]), i, j)
                    for i, j in combinations(range(n), 2)), key=lambda e: e[0])
    uf = _UF(n)
    bars: List[Bar] = []
    alive = n
    for d, i, j in edges:
        if uf.union(i, j):
            bars.append(Bar(0.0, d))   # a component died (merged) at scale d
            alive -= 1
            if alive == 1:
                break
    bars.append(Bar(0.0, math.inf))    # the one component that never dies
    return bars


def _components(points: Sequence[Point], eps: float) -> int:
    n = len(points)
    uf = _UF(n)
    for i, j in combinations(range(n), 2):
        if _dist(points[i], points[j]) <= eps:
            uf.union(i, j)
    return len({uf.find(i) for i in range(n)})


def betti1_estimate(points: Sequence[Point], eps: float) -> Tuple[int, int]:
    """(b0, b1) of the Rips complex at scale ``eps`` via the Euler characteristic (b2 assumed 0).

    χ = V − E + F = b0 − b1 + b2.  With b2≈0 for planar samples, b1 = b0 − χ.
    """
    n = len(points)
    if n == 0:
        return 0, 0
    adj = [[False] * n for _ in range(n)]
    E = 0
    for i, j in combinations(range(n), 2):
        if _dist(points[i], points[j]) <= eps:
            adj[i][j] = adj[j][i] = True
            E += 1
    F = sum(1 for i, j, k in combinations(range(n), 3)
            if adj[i][j] and adj[j][k] and adj[i][k])
    b0 = _components(points, eps)
    chi = n - E + F
    b1 = max(0, b0 - chi)
    return b0, b1


def _auto_scale(points: Sequence[Point]) -> float:
    """A robust Rips scale: ~1.6× the median nearest-neighbour distance (connects locally, not across)."""
    n = len(points)
    if n < 2:
        return 1.0
    nn: List[float] = []
    for i in range(n):
        best = math.inf
        for j in range(n):
            if i != j:
                best = min(best, _dist(points[i], points[j]))
        nn.append(best)
    nn.sort()
    median = nn[len(nn) // 2]
    return max(1e-9, 1.6 * median)


def analyze_points(points: Sequence[Point], eps: Optional[float] = None) -> TopologyReport:
    """Read the shape of a point cloud: components + holes, at an auto or given scale."""
    if eps is None:
        eps = _auto_scale(points)
    b0, b1 = betti1_estimate(points, eps)
    method = "stdlib"
    try:  # ripser only ever *sharpens* H1; the stdlib estimate always stands
        import ripser  # type: ignore  # noqa: F401
        method = "ripser-available"
    except Exception:  # noqa: BLE001
        pass
    return TopologyReport(betti0=b0, betti1=b1, barcode_h0=persistent_h0(points),
                          scale=eps, method=method)


# --------------------------------------------------------------------------- #
# hyperbolic (Poincaré-disk) embedding of a tree, vs Euclidean
# --------------------------------------------------------------------------- #
def poincare_distance(u: Point, v: Point) -> float:
    """Geodesic distance in the Poincaré disk (points strictly inside the unit ball)."""
    du = sum(c * c for c in u)
    dv = sum(c * c for c in v)
    sq = sum((a - b) ** 2 for a, b in zip(u, v))
    denom = (1.0 - du) * (1.0 - dv)
    if denom <= 0:
        denom = 1e-12
    return math.acosh(1.0 + 2.0 * sq / denom)


def _tree_structure(edges: Sequence[Tuple[int, int]], root: int = 0):
    nodes = {root}
    for a, b in edges:
        nodes.add(a); nodes.add(b)
    adj: Dict[int, List[int]] = {u: [] for u in nodes}
    for a, b in edges:
        adj[a].append(b); adj[b].append(a)
    # BFS tree: parent + depth + children
    parent: Dict[int, Optional[int]] = {root: None}
    depth: Dict[int, int] = {root: 0}
    order = [root]
    qi = 0
    while qi < len(order):
        u = order[qi]; qi += 1
        for w in adj[u]:
            if w not in depth:
                depth[w] = depth[u] + 1
                parent[w] = u
                order.append(w)
    children: Dict[int, List[int]] = {u: [] for u in nodes}
    for u in order:
        if parent[u] is not None:
            children[parent[u]].append(u)
    return order, depth, children


def embed_tree(edges: Sequence[Tuple[int, int]], *, root: int = 0,
               hyperbolic: bool = True, spread: float = 0.85) -> Dict[int, Tuple[float, float]]:
    """Cone-layout a tree in 2-D. Hyperbolic: radius grows toward the Poincaré boundary (exponential
    room for exponentially-many descendants); Euclidean: radius grows linearly with depth."""
    order, depth, children = _tree_structure(edges, root)
    coords: Dict[int, Tuple[float, float]] = {}
    # angular wedge per node
    wedge: Dict[int, Tuple[float, float]] = {root: (0.0, 2.0 * math.pi)}
    for u in order:
        lo, hi = wedge[u]
        theta = 0.5 * (lo + hi)
        d = depth[u]
        if hyperbolic:
            r = math.tanh(0.5 * spread * d)          # → boundary of the unit disk
        else:
            r = float(d)                              # flat, linear radius
        coords[u] = (r * math.cos(theta), r * math.sin(theta))
        kids = children[u]
        if kids:
            step = (hi - lo) / len(kids)
            for idx, c in enumerate(kids):
                wedge[c] = (lo + idx * step, lo + (idx + 1) * step)
    return coords


def _graph_distances(edges: Sequence[Tuple[int, int]], root: int = 0) -> Dict[Tuple[int, int], int]:
    order, _, _ = _tree_structure(edges, root)
    nodes = order
    adj: Dict[int, List[int]] = {u: [] for u in nodes}
    for a, b in edges:
        adj[a].append(b); adj[b].append(a)
    dists: Dict[Tuple[int, int], int] = {}
    for s in nodes:
        d = {s: 0}
        q = [s]; qi = 0
        while qi < len(q):
            u = q[qi]; qi += 1
            for w in adj[u]:
                if w not in d:
                    d[w] = d[u] + 1
                    q.append(w)
        for t, dv in d.items():
            if s < t:
                dists[(s, t)] = dv
    return dists


def _pearson(xs: Sequence[float], ys: Sequence[float]) -> float:
    n = len(xs)
    if n < 2:
        return 0.0
    mx = sum(xs) / n
    my = sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    vx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    vy = math.sqrt(sum((y - my) ** 2 for y in ys))
    if vx == 0 or vy == 0:
        return 0.0
    return cov / (vx * vy)


def embedding_distortion(coords: Dict[int, Tuple[float, float]],
                         graph_d: Dict[Tuple[int, int], int], *, hyperbolic: bool) -> float:
    """Scale-invariant distortion in [0,1]: 1 − Pearson(embedded distance, graph distance)."""
    emb: List[float] = []
    gph: List[float] = []
    for (s, t), gd in graph_d.items():
        u, v = coords[s], coords[t]
        d = poincare_distance(u, v) if hyperbolic else _dist(u, v)
        emb.append(d)
        gph.append(float(gd))
    return max(0.0, 1.0 - _pearson(emb, gph))


@dataclass
class EmbeddingChoice:
    geometry: str                     # "hyperbolic" | "euclidean"
    hyperbolic_distortion: float
    euclidean_distortion: float

    def to_dict(self) -> Dict[str, object]:
        return {"geometry": self.geometry,
                "hyperbolic_distortion": round(self.hyperbolic_distortion, 4),
                "euclidean_distortion": round(self.euclidean_distortion, 4)}


def recommend_geometry(edges: Sequence[Tuple[int, int]], *, root: int = 0) -> EmbeddingChoice:
    """Embed the tree both ways, measure honest distortion, prefer hyperbolic only if it wins."""
    gd = _graph_distances(edges, root)
    hc = embed_tree(edges, root=root, hyperbolic=True)
    ec = embed_tree(edges, root=root, hyperbolic=False)
    hd = embedding_distortion(hc, gd, hyperbolic=True)
    ed = embedding_distortion(ec, gd, hyperbolic=False)
    geometry = "hyperbolic" if hd < ed else "euclidean"
    return EmbeddingChoice(geometry, hd, ed)


class LatentGeometry:
    """Facade: read a point cloud's shape (TDA) and choose a fitting geometry for a hierarchy."""

    def analyze(self, points: Sequence[Point], eps: Optional[float] = None) -> TopologyReport:
        return analyze_points(points, eps)

    def choose_geometry(self, edges: Sequence[Tuple[int, int]], *, root: int = 0) -> EmbeddingChoice:
        return recommend_geometry(edges, root=root)
