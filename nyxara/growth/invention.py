"""NYXARA · growth/invention.py — genuine invention (🛠️, Pillar F · Edge 3, Rule 4).

Her divergent faculty (:mod:`nyxara.mind.creative`) *remixes*: SCAMPER, analogy and conceptual
blending recombine a concept into fresh framings. Her :class:`~nyxara.growth.eureka.EurekaEngine`
*invents* — but only over **decidable mathematics**, where a theorem prover can certify truth. Master
JP asked for the next thing: *"sirf remix nahi — bilkul nayi invention"* — genuinely new **engineering
and algorithmic artifacts**: a new battery chemistry, a new aircraft structure, a new algorithm.

This module is that faculty. The **Invention Engine** lives by the same asymmetry Eureka trusts —
*generation is cheap and unbounded; evaluation is rigorous and honest* — but where Eureka's verifier is
a theorem prover, here it is a **quantitative, first-principles evaluator** per domain plus a **library
of the real, known inventions**. The loop is the same open-ended evolutionary one: *propose → evaluate
→ score → keep → feed back*. She invents candidate designs by combinatorial / evolutionary search over
a real design space (no LLM in the loop at all), evaluates each against a grounded physical /
computational model, and **keeps only what is feasible ∧ genuinely novel ∧ better than the best known
baseline**. Everything else is discarded — an infeasible design like a refuted conjecture, a re-derived
textbook design like a trivial identity.

The three domains, exactly as Master JP named them:

* **battery** — a new *chemistry*. The genome is an (anode, cathode, electrolyte) triple drawn from
  real material-property tables; the evaluator computes cell voltage, gravimetric capacity (series
  electrode-mass model ``1/C = 1/Cₐ + 1/C𝒸``) and **active-material specific energy** (Wh/kg = V × C),
  applies hard feasibility constraints (the cell voltage must sit inside the electrolyte's
  stability window; dendrite-prone metal anodes need a solid electrolyte), and ranks designs by a
  multi-objective figure of merit (energy · safety-margin · earth-abundance · cycle-life).
* **aircraft** — a new *structure*. The genome is (topology, aspect-ratio, material, taper); the
  evaluator estimates the lift-to-drag ratio ``L/D = ½·√(π·e·AR / C_D₀)`` and the structural mass
  fraction from a beam-bending / buckling bound, with feasibility limits on aspect ratio (aeroelastic)
  and mass, and ranks by ``L/D × payload-fraction``.
* **algorithm** — a new *algorithm*. The most rigorous, Eureka-grade domain: the genome composes
  algorithmic primitives (divide-and-conquer, an insertion base-case, a pivot rule, a heap fallback)
  into a concrete sorting strategy. The engine **synthesises and actually executes** the candidate on
  a fixed instance suite, **verifies correctness against a brute-force oracle** (a wrong algorithm is
  discarded — never bluffed, exactly as Eureka discards ``REFUTED``), and measures empirical operation
  counts against a baseline. *Kept* iff ``correct ∧ faster ∧ structurally novel`` — which is how real
  hybrids like introsort (quicksort + heapsort fallback + insertion base-case) were actually invented.

Novelty — the answer to *"not a remix"* — reuses the quality-diversity machinery NYXARA already trusts
(:class:`~nyxara.growth.frontier.NoveltyArchive` / :class:`~nyxara.growth.frontier.BehaviorDescriptor`):
every **known** invention is seeded into a per-domain archive at construction, so a candidate's novelty
is its design-space distance from *both* the known art *and* everything she has already invented. A
design below the novelty floor is rejected as a remix; a kept one is ingested so the next generation
must push past it (open-endedness). What survives is folded back through the same sinks Eureka uses —
long-term memory, the grounded knowledge base, and the verified-data flywheel that feeds her training.

Design rules (the growth/ contract): pure standard library (``random`` / ``math``); deterministic and
CI-reproducible under a fixed seed. It is gather-only — it invents and evaluates *in silico*, never
acts on the world, never reaches the network, touches no source, no weights, and no gate. The kept
results are honest in the only form that matters: feasibility is *computed*, correctness is *executed
and checked*, novelty is *measured against the real known art* — asserted by nothing, certified by the
model. Character-locked like all of growth/.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from random import Random
from typing import Any, Callable, Dict, List, Optional, Tuple

__all__ = [
    "DesignCandidate",
    "Invention",
    "InventionReport",
    "InventionEngine",
    "DOMAINS",
    "synthesize_sort",
    "algorithm_metrics",
]

# The engineering / algorithmic domains over which she invents and certifies-by-evaluation.
DOMAINS: Tuple[str, ...] = ("battery", "aircraft", "algorithm")

_MEM_TAG = "invention"


def _clamp01(x: Any) -> float:
    try:
        return max(0.0, min(1.0, float(x)))
    except Exception:  # noqa: BLE001
        return 0.0


# =========================================================================== #
# Data model — mirrors growth/eureka.py (Conjecture / Breakthrough / Report).
# =========================================================================== #
@dataclass
class DesignCandidate:
    """One design the engine *invented* (not prompted). The genome is a plain, JSON-able dict."""

    domain: str                                  # battery | aircraft | algorithm
    genome: Dict[str, Any]
    origin: str = "seed"                         # seed | mutate | crossover
    lineage: Tuple[str, ...] = ()

    def key(self) -> str:
        """A canonical de-duplication key over the genome's sorted items."""
        items = ";".join(f"{k}={self.genome[k]}" for k in sorted(self.genome))
        return f"{self.domain}::{items}"

    def summary(self) -> str:
        return _SUMMARY.get(self.domain, lambda g: str(g))(self.genome)

    def to_dict(self) -> Dict[str, Any]:
        return {"domain": self.domain, "genome": dict(self.genome), "origin": self.origin,
                "lineage": list(self.lineage), "summary": self.summary()}


@dataclass
class Invention:
    """A candidate that survived: feasible, far from the known art, and beating the best baseline."""

    candidate: DesignCandidate
    metrics: Dict[str, Any]
    feasible: bool = True
    novelty: float = 0.0
    improvement: float = 1.0          # candidate figure-of-merit ÷ best known baseline (>1 == better)
    at: float = field(default_factory=time.time)

    @property
    def domain(self) -> str:
        return self.candidate.domain

    @property
    def summary(self) -> str:
        return self.candidate.summary()

    def score(self) -> float:
        """Selection score: novelty × bounded gain. gain = 1 − 1/improvement ∈ [0,1) for imp ≥ 1,
        so a huge algorithmic speed-up and a modest battery gain stay cross-domain comparable."""
        gain = 1.0 - (1.0 / self.improvement) if self.improvement > 0 else 0.0
        return round(self.novelty * max(0.0, gain), 6)

    def to_dict(self) -> Dict[str, Any]:
        return {"domain": self.domain, "summary": self.summary,
                "genome": dict(self.candidate.genome), "origin": self.candidate.origin,
                "lineage": list(self.candidate.lineage), "feasible": self.feasible,
                "metrics": self.metrics,
                "novelty": round(float(self.novelty), 6),
                "improvement": round(float(self.improvement), 6),
                "score": self.score(), "at": self.at}


@dataclass
class InventionReport:
    """Everything one ``invent`` run produced — nothing kept silently (mirrors BreakthroughReport)."""

    generations: int = 0
    population: int = 0
    generated: int = 0
    feasible: int = 0
    infeasible: int = 0
    novel_kept: int = 0
    by_domain: Dict[str, int] = field(default_factory=dict)
    inventions: List[Invention] = field(default_factory=list)
    fed_knowledge: int = 0
    fed_flywheel: int = 0
    frontier: Dict[str, Any] = field(default_factory=dict)
    reason: str = ""
    elapsed_ms: float = 0.0

    def best(self) -> Optional[Invention]:
        return max(self.inventions, key=Invention.score) if self.inventions else None

    def to_dict(self) -> Dict[str, Any]:
        return {"generations": self.generations, "population": self.population,
                "generated": self.generated, "feasible": self.feasible,
                "infeasible": self.infeasible, "novel_kept": self.novel_kept,
                "by_domain": dict(self.by_domain),
                "inventions": [iv.to_dict() for iv in self.inventions[:12]],
                "best": self.best().to_dict() if self.best() else None,
                "fed_knowledge": self.fed_knowledge, "fed_flywheel": self.fed_flywheel,
                "frontier": self.frontier, "reason": self.reason,
                "elapsed_ms": round(self.elapsed_ms, 3)}


# =========================================================================== #
# Domain 1 — BATTERY chemistry. Real first-order electrochemistry over property tables.
# =========================================================================== #
# cathode -> (potential_vs_Li [V], gravimetric_capacity [mAh/g], earth_abundance, cyclelife_base)
_CATHODES: Dict[str, Tuple[float, float, float, float]] = {
    "LiCoO2": (3.9, 140.0, 0.40, 0.70),
    "LiFePO4": (3.4, 160.0, 0.92, 0.95),
    "NMC811": (3.8, 200.0, 0.55, 0.80),
    "LiNiO2": (3.7, 240.0, 0.50, 0.60),
    "LiMn2O4": (4.0, 120.0, 0.85, 0.70),
    "sulfur": (2.1, 1000.0, 0.90, 0.45),
    "FeF3": (3.0, 420.0, 0.95, 0.50),            # conversion cathode
    "air_O2": (2.9, 1200.0, 0.99, 0.30),         # lithium/metal-air
    "Na3V2(PO4)3": (3.4, 110.0, 0.90, 0.85),
    "organic_quinone": (2.6, 300.0, 0.99, 0.60),  # sustainable organic
}
# anode -> (potential_vs_Li [V], gravimetric_capacity [mAh/g], abundance, dendrite_risk, cyclelife_base)
_ANODES: Dict[str, Tuple[float, float, float, float, float]] = {
    "graphite": (0.10, 372.0, 0.95, 0.10, 0.90),
    "hard_carbon": (0.20, 300.0, 0.97, 0.10, 0.85),
    "silicon": (0.40, 2200.0, 0.98, 0.25, 0.55),
    "lithium_metal": (0.00, 3860.0, 0.70, 0.92, 0.45),
    "Li4Ti5O12": (1.55, 175.0, 0.80, 0.02, 0.97),
    "tin": (0.60, 990.0, 0.80, 0.30, 0.60),
    "sodium_metal": (0.00, 1160.0, 0.92, 0.88, 0.40),
}
# electrolyte -> (oxidative_stability_window_top [V], abundance, is_solid)
_ELECTROLYTES: Dict[str, Tuple[float, float, bool]] = {
    "carbonate_liquid": (4.3, 0.90, False),
    "hv_carbonate": (4.6, 0.85, False),
    "aqueous": (1.5, 0.99, False),
    "sulfide_solid": (5.0, 0.70, True),
    "polymer_solid": (4.2, 0.85, True),
    "ionic_liquid": (5.5, 0.60, False),
}

# energy normaliser — Li-S active-material ceiling (2.1 V × 1000 mAh/g ≈ 2100 Wh/kg) sits near 1.0.
_ENERGY_REF = 2200.0


def _battery_eval(genome: Dict[str, Any]) -> Dict[str, Any]:
    """Compute feasibility + a multi-objective figure of merit for one chemistry. Deterministic."""
    a = _ANODES.get(genome.get("anode", ""))
    c = _CATHODES.get(genome.get("cathode", ""))
    e = _ELECTROLYTES.get(genome.get("electrolyte", ""))
    if a is None or c is None or e is None:
        return {"feasible": False, "reason": "unknown material", "fom": 0.0}
    a_v, a_cap, a_ab, a_dend, a_cyc = a
    c_v, c_cap, c_ab, c_cyc = c
    win_top, e_ab, e_solid = e

    voltage = round(c_v - a_v, 4)                       # nominal full-cell voltage
    # series electrode-mass model: total gravimetric capacity of the paired electrodes.
    cap = (a_cap * c_cap) / (a_cap + c_cap) if (a_cap + c_cap) > 0 else 0.0
    specific_energy = voltage * cap                     # active-material Wh/kg  (mAh/g · V)

    # ---- hard feasibility ----
    if voltage <= 0.3:
        return {"feasible": False, "reason": "non-positive / trivial cell voltage", "fom": 0.0,
                "voltage": voltage}
    if voltage > win_top:
        return {"feasible": False, "reason": f"voltage {voltage}V exceeds electrolyte window {win_top}V",
                "fom": 0.0, "voltage": voltage}
    # a dendrite-prone metal anode in a liquid electrolyte is a known safety failure mode.
    if a_dend >= 0.5 and not e_solid:
        return {"feasible": False, "reason": "dendrite-prone metal anode needs a solid electrolyte",
                "fom": 0.0, "voltage": voltage}

    # ---- multi-objective figure of merit ----
    energy_norm = _clamp01(specific_energy / _ENERGY_REF)
    safety = _clamp01((win_top - voltage) / 0.8)        # margin below the oxidative limit
    abundance = min(a_ab, c_ab, e_ab)
    cyclelife = a_cyc * c_cyc * (1.0 if (a_dend < 0.5 or e_solid) else 0.3)
    fom = 0.5 * energy_norm + 0.2 * safety + 0.2 * abundance + 0.1 * cyclelife
    return {"feasible": True, "voltage": voltage,
            "specific_energy_Wh_per_kg": round(specific_energy, 1),
            "capacity_mAh_per_g": round(cap, 1),
            "earth_abundance": round(abundance, 3), "cycle_life": round(cyclelife, 3),
            "safety_margin": round(safety, 3), "fom": round(fom, 5)}


_BATTERY_BASELINES: Tuple[Dict[str, str], ...] = (
    {"anode": "graphite", "cathode": "LiCoO2", "electrolyte": "carbonate_liquid"},     # classic Li-ion
    {"anode": "graphite", "cathode": "LiFePO4", "electrolyte": "carbonate_liquid"},    # LFP
    {"anode": "graphite", "cathode": "NMC811", "electrolyte": "carbonate_liquid"},     # NMC
    {"anode": "hard_carbon", "cathode": "Na3V2(PO4)3", "electrolyte": "carbonate_liquid"},  # Na-ion
    {"anode": "lithium_metal", "cathode": "sulfur", "electrolyte": "polymer_solid"},   # Li-S (solid)
    {"anode": "lithium_metal", "cathode": "NMC811", "electrolyte": "sulfide_solid"},   # solid-state
)


# =========================================================================== #
# Domain 2 — AIRCRAFT structure. Aerodynamic + structural first-order model.
# =========================================================================== #
# topology -> (parasitic_drag_C_D0, oswald_e, struct_mass_base, max_aspect_ratio, bending_relief)
_TOPOLOGIES: Dict[str, Tuple[float, float, float, float, float]] = {
    "tube_and_wing": (0.022, 0.80, 0.28, 12.0, 0.00),
    "flying_wing": (0.013, 0.90, 0.25, 14.0, 0.00),
    "box_wing": (0.020, 1.40, 0.30, 16.0, 0.05),       # Prandtl best-wing-system: e > 1
    "truss_braced": (0.021, 0.85, 0.22, 28.0, 0.40),   # strut relieves bending → very high AR
    "canard": (0.023, 0.82, 0.27, 12.0, 0.00),
    "blended_wing_body": (0.015, 0.88, 0.24, 13.0, 0.05),
}
# material -> (structural_mass_factor, max_AR_bonus)
_MATERIALS: Dict[str, Tuple[float, float]] = {
    "aluminum": (1.00, 1.00),
    "al_li_alloy": (0.88, 1.05),
    "titanium": (0.90, 1.05),
    "cfrp_composite": (0.72, 1.20),
}
_TAPERS: Tuple[float, ...] = (0.2, 0.3, 0.4, 0.5, 0.7, 1.0)


def _aircraft_eval(genome: Dict[str, Any]) -> Dict[str, Any]:
    """Estimate L/D and structural mass fraction; feasibility on aeroelastic AR + mass. Deterministic."""
    top = _TOPOLOGIES.get(genome.get("topology", ""))
    mat = _MATERIALS.get(genome.get("material", ""))
    if top is None or mat is None:
        return {"feasible": False, "reason": "unknown topology/material", "fom": 0.0}
    cd0, e, struct_base, max_ar, relief = top
    mat_factor, ar_bonus = mat
    ar = float(genome.get("aspect_ratio", 9.0))
    taper = float(genome.get("taper", 0.4))

    # Oswald efficiency peaks near the optimal taper ≈ 0.4 (elliptical-ish loading).
    e_eff = e * (1.0 - 0.10 * abs(taper - 0.4))
    ld = 0.5 * math.sqrt(math.pi * e_eff * ar / cd0)

    # beam-bending / buckling: structural mass grows with AR^1.5, relieved by a strut, scaled by material.
    struct_frac = (struct_base + 2.0 * (ar ** 1.5) / 1000.0 * (1.0 - relief)) * mat_factor
    payload_frac = 1.0 - struct_frac - 0.35            # 0.35 ≈ fixed systems + fuel reserve

    ar_ceiling = max_ar * ar_bonus
    if ar < 4.0 or ar > ar_ceiling:
        return {"feasible": False, "reason": f"aspect ratio {ar} outside aeroelastic limit {ar_ceiling:.0f}",
                "fom": 0.0, "L_over_D": round(ld, 2)}
    if struct_frac >= 0.50 or payload_frac <= 0.05:
        return {"feasible": False, "reason": "structurally too heavy to carry useful payload",
                "fom": 0.0, "L_over_D": round(ld, 2), "structural_mass_fraction": round(struct_frac, 3)}

    fom = ld * payload_frac
    return {"feasible": True, "L_over_D": round(ld, 2),
            "structural_mass_fraction": round(struct_frac, 3),
            "payload_fraction": round(payload_frac, 3),
            "aspect_ratio": ar, "fom": round(fom, 4)}


_AIRCRAFT_BASELINES: Tuple[Dict[str, Any], ...] = (
    {"topology": "tube_and_wing", "aspect_ratio": 9, "material": "aluminum", "taper": 0.3},     # 737-class
    {"topology": "tube_and_wing", "aspect_ratio": 10, "material": "cfrp_composite", "taper": 0.3},  # A350-class
    {"topology": "flying_wing", "aspect_ratio": 6, "material": "cfrp_composite", "taper": 0.3},  # B-2-class
    {"topology": "truss_braced", "aspect_ratio": 19, "material": "cfrp_composite", "taper": 0.4},  # TTBW/SUGAR
    {"topology": "box_wing", "aspect_ratio": 12, "material": "cfrp_composite", "taper": 0.4},    # box-wing concept
    {"topology": "canard", "aspect_ratio": 11, "material": "cfrp_composite", "taper": 0.4},      # Starship-class
)


# =========================================================================== #
# Domain 3 — ALGORITHM. Synthesise → execute → verify against an oracle (Eureka-grade).
# =========================================================================== #
class _Counter:
    """A primitive-operation counter threaded through the synthesised algorithm."""

    __slots__ = ("ops",)

    def __init__(self) -> None:
        self.ops = 0


def _insertion(a: List[int], c: _Counter) -> List[int]:
    a = list(a)
    for i in range(1, len(a)):
        key, j = a[i], i - 1
        while j >= 0:
            c.ops += 1
            if a[j] > key:
                a[j + 1] = a[j]
                c.ops += 1
                j -= 1
            else:
                break
        a[j + 1] = key
        c.ops += 1
    return a


def _selection(a: List[int], c: _Counter) -> List[int]:
    a = list(a)
    n = len(a)
    for i in range(n):
        m = i
        for j in range(i + 1, n):
            c.ops += 1
            if a[j] < a[m]:
                m = j
        if m != i:
            a[i], a[m] = a[m], a[i]
            c.ops += 1
    return a


def _heap(a: List[int], c: _Counter) -> List[int]:
    a = list(a)
    n = len(a)

    def sift(root: int, end: int) -> None:
        while True:
            child = 2 * root + 1
            if child > end:
                break
            if child + 1 <= end:
                c.ops += 1
                if a[child] < a[child + 1]:
                    child += 1
            c.ops += 1
            if a[root] < a[child]:
                a[root], a[child] = a[child], a[root]
                c.ops += 1
                root = child
            else:
                break

    for start in range(n // 2 - 1, -1, -1):
        sift(start, n - 1)
    for end in range(n - 1, 0, -1):
        a[0], a[end] = a[end], a[0]
        c.ops += 1
        sift(0, end - 1)
    return a


def _merge(a: List[int], c: _Counter, threshold: int) -> List[int]:
    if len(a) <= 1:
        return list(a)
    if threshold > 0 and len(a) <= threshold:          # insertion base-case (the hybrid composition)
        return _insertion(a, c)
    mid = len(a) // 2
    left = _merge(a[:mid], c, threshold)
    right = _merge(a[mid:], c, threshold)
    out: List[int] = []
    i = j = 0
    while i < len(left) and j < len(right):
        c.ops += 1
        if left[i] <= right[j]:
            out.append(left[i]); i += 1
        else:
            out.append(right[j]); j += 1
    out.extend(left[i:]); out.extend(right[j:])
    c.ops += len(left) - i + len(right) - j
    return out


def _quick(a: List[int], c: _Counter, pivot: str, threshold: int,
           depth_limit: int, depth: int, fallback: bool) -> List[int]:
    if len(a) <= 1:
        return list(a)
    if threshold > 0 and len(a) <= threshold:          # insertion base-case
        return _insertion(a, c)
    if fallback and depth >= depth_limit:              # introsort heap fallback on bad recursion depth
        return _heap(a, c)
    if pivot == "first":
        p = a[0]
    elif pivot == "last":
        p = a[-1]
    elif pivot == "median3":
        c.ops += 3
        p = sorted((a[0], a[len(a) // 2], a[-1]))[1]
    else:  # "mid"
        p = a[len(a) // 2]
    less: List[int] = []
    eq: List[int] = []
    gt: List[int] = []
    for x in a:
        c.ops += 1
        if x < p:
            less.append(x)
        elif x > p:
            gt.append(x)
        else:
            eq.append(x)
    return (_quick(less, c, pivot, threshold, depth_limit, depth + 1, fallback) + eq
            + _quick(gt, c, pivot, threshold, depth_limit, depth + 1, fallback))


def _counting(a: List[int], c: _Counter) -> List[int]:
    if not a:
        return []
    lo, hi = min(a), max(a)
    width = hi - lo + 1
    buckets = [0] * width
    for x in a:
        buckets[x - lo] += 1
        c.ops += 1
    out: List[int] = []
    for v, cnt in enumerate(buckets):
        c.ops += 1
        out.extend([v + lo] * cnt)
    return out


_STRATEGIES: Tuple[str, ...] = ("insertion", "selection", "merge", "quick", "heap", "counting")
_PIVOTS: Tuple[str, ...] = ("first", "mid", "last", "median3")


def synthesize_sort(genome: Dict[str, Any]) -> Callable[[List[int]], Tuple[List[int], int]]:
    """Compile a genome into a real, runnable sort returning ``(sorted_list, operation_count)``."""
    strat = genome.get("strategy", "insertion")
    thr = int(genome.get("base_threshold", 0))
    pivot = genome.get("pivot", "mid")
    fallback = bool(genome.get("introsort_fallback", False))

    def run(arr: List[int]) -> Tuple[List[int], int]:
        c = _Counter()
        data = list(arr)
        if strat == "insertion":
            out = _insertion(data, c)
        elif strat == "selection":
            out = _selection(data, c)
        elif strat == "merge":
            out = _merge(data, c, thr)
        elif strat == "heap":
            out = _heap(data, c)
        elif strat == "counting":
            out = _counting(data, c)
        elif strat == "quick":
            n = max(2, len(data))
            depth_limit = 2 * max(1, int(math.log2(n)))
            out = _quick(data, c, pivot, thr, depth_limit, 0, fallback)
        else:
            out = _insertion(data, c)
        return out, c.ops

    return run


def _sort_instances() -> List[List[int]]:
    """A fixed, deterministic instance suite (random / reverse / nearly-sorted), 0..255 integers."""
    rng = Random(20250627)
    sizes = (16, 32, 48, 64, 80)
    suite: List[List[int]] = []
    for n in sizes:
        suite.append([rng.randint(0, 255) for _ in range(n)])           # random
        suite.append(list(range(n, 0, -1)))                             # reverse-sorted (worst case)
        near = list(range(n))
        for _ in range(max(1, n // 10)):                                # nearly sorted
            i, j = rng.randrange(n), rng.randrange(n)
            near[i], near[j] = near[j], near[i]
        suite.append(near)
    return suite


_BASELINE_ALG = {"strategy": "insertion"}


def algorithm_metrics(genome: Dict[str, Any]) -> Dict[str, Any]:
    """Run the synthesised algorithm on the suite; verify against ``sorted`` and measure speed-up.

    Returns ``correct`` (passes the oracle on *every* instance), total ``ops``, the insertion
    ``baseline_ops``, and the ``speedup``. A wrong algorithm is reported ``correct=False`` and is
    discarded upstream — never kept, never bluffed."""
    suite = _sort_instances()
    algo = synthesize_sort(genome)
    base = synthesize_sort(_BASELINE_ALG)
    correct = True
    ops = 0
    base_ops = 0
    for inst in suite:
        out, used = algo(inst)
        if out != sorted(inst):
            correct = False
        ops += used
        _, b = base(inst)
        base_ops += b
    speedup = (base_ops / ops) if ops > 0 else 0.0
    return {"feasible": bool(correct), "correct": bool(correct), "ops": ops,
            "baseline_ops": base_ops, "speedup": round(speedup, 4),
            "fom": round(speedup, 4)}


_ALGORITHM_BASELINES: Tuple[Dict[str, Any], ...] = (
    {"strategy": "insertion"},
    {"strategy": "selection"},
    {"strategy": "merge", "base_threshold": 0},
    {"strategy": "quick", "base_threshold": 0, "pivot": "first", "introsort_fallback": False},
    {"strategy": "quick", "base_threshold": 0, "pivot": "mid", "introsort_fallback": False},
    {"strategy": "heap"},
)


# =========================================================================== #
# Human-readable one-line summaries per domain.
# =========================================================================== #
_SUMMARY: Dict[str, Callable[[Dict[str, Any]], str]] = {
    "battery": lambda g: f"{g.get('anode')}‖{g.get('electrolyte')}‖{g.get('cathode')} cell",
    "aircraft": lambda g: (f"{g.get('topology')} · AR{g.get('aspect_ratio')} · "
                           f"{g.get('material')} · taper {g.get('taper')}"),
    "algorithm": lambda g: ("sort: " + g.get("strategy", "?")
                            + (f"+insertion≤{g['base_threshold']}" if g.get("base_threshold") else "")
                            + (f" pivot={g['pivot']}" if g.get("pivot") and g.get("strategy") == "quick" else "")
                            + (" +heap-fallback" if g.get("introsort_fallback") else "")),
}


# =========================================================================== #
# The engine
# =========================================================================== #
class InventionEngine:
    """Invent candidate designs by search, evaluate each against a real model, keep the feasible +
    novel + better-than-baseline. Mirrors :class:`~nyxara.growth.eureka.EurekaEngine`.

    Parameters
    ----------
    memory / knowledge / flywheel:
        optional sinks; each kept invention is remembered, grounded, and fed to the verified-data
        flywheel when present. All best-effort — a missing sink never raises.
    settings:        :class:`~nyxara.kernel.config.NyxaraSettings` (unused beyond future hooks).
    seed:            makes the whole search deterministic and CI-reproducible.
    novelty_floor:   minimum design-space distance from the known art to count as "not a remix".
    """

    def __init__(self, *, memory: Any = None, knowledge: Any = None, flywheel: Any = None,
                 settings: Any = None, seed: int = 1337, novelty_floor: float = 0.06) -> None:
        self.memory = memory
        self.knowledge = knowledge
        self.flywheel = flywheel
        self.settings = settings
        self.rng = Random(seed)
        self.novelty_floor = float(novelty_floor)
        self._evaluators: Dict[str, Callable[[Dict[str, Any]], Dict[str, Any]]] = {
            "battery": _battery_eval, "aircraft": _aircraft_eval, "algorithm": algorithm_metrics,
        }
        self._baselines: Dict[str, Tuple[Dict[str, Any], ...]] = {
            "battery": _BATTERY_BASELINES, "aircraft": _AIRCRAFT_BASELINES,
            "algorithm": _ALGORITHM_BASELINES,
        }
        # per-domain novelty archive, seeded with the known art so a re-derived classic scores 0 novelty.
        self._archives: Dict[str, Any] = {}
        self._best_baseline_fom: Dict[str, float] = {}
        self._seed_archives()
        self._elites: Dict[str, List[DesignCandidate]] = {d: [] for d in DOMAINS}
        self._seen: set[str] = set()
        self.total_inventions: int = 0

    # ------------------------------------------------------------------ #
    # Seed the known art: novelty archive + best-baseline figure of merit.
    # ------------------------------------------------------------------ #
    def _seed_archives(self) -> None:
        try:
            from nyxara.growth.frontier import BehaviorDescriptor, Discovery, NoveltyArchive
        except Exception:  # noqa: BLE001 — without the QD module, novelty degrades to "always novel"
            self._archives = {d: None for d in DOMAINS}
            self._best_baseline_fom = {d: 0.0 for d in DOMAINS}
            return
        # k=1 (nearest-neighbour novelty): a candidate identical to *any* known or already-invented
        # design reads as distance 0 and is rejected as a remix — the exact anti-remix guarantee.
        for domain in DOMAINS:
            arc = NoveltyArchive(resolution=8, k=1, ndims=3)
            best = 0.0
            for genome in self._baselines[domain]:
                metrics = self._evaluators[domain](genome)
                if metrics.get("feasible"):
                    best = max(best, float(metrics.get("fom", 0.0)))
                dims = _DESCRIPTORS[domain](genome)
                arc.add(Discovery(content=_SUMMARY[domain](genome),
                                  descriptor=BehaviorDescriptor(dims=dims, labels=(domain, "", "")),
                                  quality=float(metrics.get("fom", 0.0)), provenance="known-art"))
            self._archives[domain] = arc
            self._best_baseline_fom[domain] = best

    # ================================================================== #
    # Public entry point — the open-ended evolutionary invention loop.
    # ================================================================== #
    def invent(self, generations: int = 4, population: int = 24,
               domains: Optional[Tuple[str, ...]] = None) -> InventionReport:
        """Run ``generations`` rounds of invent → evaluate → score → select. Always returns a report."""
        t0 = time.perf_counter()
        active = tuple(d for d in (domains or DOMAINS) if d in DOMAINS) or DOMAINS
        generations = max(1, int(generations))
        population = max(len(active), int(population))
        per_domain = max(1, population // len(active))
        report = InventionReport(generations=generations, population=population)
        for _ in range(generations):
            batch: List[DesignCandidate] = []
            for domain in active:
                batch.extend(self._propose(domain, per_domain))
            for cand in batch:
                report.generated += 1
                if cand.key() in self._seen:
                    continue
                self._seen.add(cand.key())
                metrics = self._evaluate(cand)
                if not metrics.get("feasible"):
                    report.infeasible += 1
                    continue
                report.feasible += 1
                self._elites.setdefault(cand.domain, []).append(cand)
                iv = self._consider(cand, metrics)
                if iv is not None:
                    report.inventions.append(iv)
                    report.novel_kept += 1
                    report.by_domain[cand.domain] = report.by_domain.get(cand.domain, 0) + 1
                    report.fed_knowledge += self._feed_knowledge(iv)
                    report.fed_flywheel += self._feed_flywheel(iv)
                    self._remember(iv)
            self._prune_elites()
        report.inventions.sort(key=Invention.score, reverse=True)
        self.total_inventions += report.novel_kept
        report.frontier = self._frontier_snapshot(active)
        report.reason = (f"invented {report.generated}, feasible {report.feasible}, "
                         f"kept {report.novel_kept} novel+better across {generations} gen(s)")
        report.elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return report

    # convenience alias, mirroring the rest of growth/.
    def run(self, generations: int = 4, population: int = 24,
            domains: Optional[Tuple[str, ...]] = None) -> InventionReport:
        return self.invent(generations=generations, population=population, domains=domains)

    # ================================================================== #
    # Step 1 — INVENT a population of candidates (no LLM, ever).
    # ================================================================== #
    def _propose(self, domain: str, n: int) -> List[DesignCandidate]:
        out: List[DesignCandidate] = []
        elites = self._elites.get(domain, [])
        for _ in range(n):
            r = self.rng.random()
            if elites and r < 0.30:
                cand = self._mutate(self.rng.choice(elites))
            elif len(elites) >= 2 and r < 0.45:
                cand = self._crossover(self.rng.choice(elites), self.rng.choice(elites))
            else:
                cand = self._seed(domain)
            if cand is not None and cand.genome:
                out.append(cand)
        return out

    def _seed(self, domain: str) -> DesignCandidate:
        return DesignCandidate(domain=domain, genome=_SEEDERS[domain](self.rng), origin="seed")

    def _mutate(self, parent: DesignCandidate) -> DesignCandidate:
        genome = _MUTATORS[parent.domain](self.rng, dict(parent.genome))
        return DesignCandidate(domain=parent.domain, genome=genome, origin="mutate",
                               lineage=(parent.summary()[:60],))

    def _crossover(self, a: DesignCandidate, b: DesignCandidate) -> DesignCandidate:
        genome: Dict[str, Any] = {}
        for key in a.genome:
            src = a if self.rng.random() < 0.5 else b
            genome[key] = src.genome.get(key, a.genome[key])
        return DesignCandidate(domain=a.domain, genome=genome, origin="crossover",
                               lineage=(a.summary()[:48], b.summary()[:48]))

    # ================================================================== #
    # Step 2 — EVALUATE (the model decides; the engine never asserts merit).
    # ================================================================== #
    def _evaluate(self, cand: DesignCandidate) -> Dict[str, Any]:
        try:
            return self._evaluators[cand.domain](cand.genome)
        except Exception:  # noqa: BLE001 — a malformed genome is simply infeasible, never fatal
            return {"feasible": False, "reason": "evaluation error", "fom": 0.0}

    # ================================================================== #
    # Step 3 — KEEP iff novel ∧ better-than-baseline.
    # ================================================================== #
    def _consider(self, cand: DesignCandidate, metrics: Dict[str, Any]) -> Optional[Invention]:
        best = self._best_baseline_fom.get(cand.domain, 0.0)
        fom = float(metrics.get("fom", 0.0))
        improvement = (fom / best) if best > 0 else (fom if fom > 0 else 0.0)
        if improvement <= 1.0:                              # not better than the best known design
            return None
        novelty = self._novelty(cand)
        if novelty < self.novelty_floor:                    # a remix of the known art — reject
            return None
        self._ingest(cand, quality=fom)                     # push the frontier past it (open-endedness)
        return Invention(candidate=cand, metrics=metrics, feasible=True,
                         novelty=novelty, improvement=improvement)

    def _descriptor(self, cand: DesignCandidate) -> Any:
        from nyxara.growth.frontier import BehaviorDescriptor
        dims = _DESCRIPTORS[cand.domain](cand.genome)
        return BehaviorDescriptor(dims=dims, labels=(cand.domain, "", ""))

    def _novelty(self, cand: DesignCandidate) -> float:
        arc = self._archives.get(cand.domain)
        if arc is None:
            return 1.0
        try:
            return float(arc.novelty(self._descriptor(cand)))
        except Exception:  # noqa: BLE001
            return 1.0

    def _ingest(self, cand: DesignCandidate, *, quality: float) -> None:
        arc = self._archives.get(cand.domain)
        if arc is None:
            return
        try:
            from nyxara.growth.frontier import Discovery
            arc.add(Discovery(content=cand.summary(), descriptor=self._descriptor(cand),
                              quality=float(quality), provenance="invention"))
        except Exception:  # noqa: BLE001
            pass

    # ================================================================== #
    # Step 4 — FEED what survived back into memory / knowledge / training.
    # ================================================================== #
    def _feed_knowledge(self, iv: Invention) -> int:
        if self.knowledge is None:
            return 0
        try:
            text = (f"Self-invented {iv.domain} design: {iv.summary}\n"
                    f"Metrics: {iv.metrics}\n"
                    f"Novelty {iv.novelty:.3f}, {iv.improvement:.3f}× the best known baseline.")
            self.knowledge.ingest_text(text, source=f"invention:{iv.domain}")
            return 1
        except Exception:  # noqa: BLE001
            return 0

    def _feed_flywheel(self, iv: Invention) -> int:
        if self.flywheel is None:
            return 0
        try:
            dec = self.flywheel.consider(
                prompt=f"Invent a better {iv.domain} design than the known art.",
                answer=f"{iv.summary} — {iv.improvement:.3f}× best baseline. {iv.metrics}",
                confidence=float(min(1.0, iv.score() + 0.5)), verified=True)
            return 1 if getattr(dec, "collected", False) else 0
        except Exception:  # noqa: BLE001
            return 0

    def _remember(self, iv: Invention) -> bool:
        if self.memory is None:
            return False
        try:
            from nyxara.memory.provenance import Provenance, SourceType
            from nyxara.memory.store import MemoryType
        except Exception:  # noqa: BLE001
            return False
        try:
            content = f"[invention] {iv.domain}: {iv.summary}  ({iv.improvement:.2f}× best known)"[:300]
            self.memory.remember(
                content, mem_type=MemoryType.SEMANTIC,
                provenance=Provenance(SourceType.SELF_REFLECTION, confidence=0.9),
                importance=0.7, tags=[_MEM_TAG, iv.domain, "discovery"],
                metadata={"invention": iv.to_dict()})
            return True
        except Exception:  # noqa: BLE001
            return False

    # ================================================================== #
    # housekeeping
    # ================================================================== #
    def _prune_elites(self, keep: int = 16) -> None:
        for dom, lst in self._elites.items():
            if len(lst) > keep:
                self._elites[dom] = lst[-keep:]

    def _frontier_snapshot(self, active: Tuple[str, ...]) -> Dict[str, Any]:
        snap: Dict[str, Any] = {}
        for domain in active:
            arc = self._archives.get(domain)
            if arc is not None:
                try:
                    snap[domain] = {"coverage": arc.coverage(), "total_seen": arc.total_seen,
                                    "best_baseline_fom": round(self._best_baseline_fom.get(domain, 0.0), 5)}
                except Exception:  # noqa: BLE001
                    pass
        return snap


# =========================================================================== #
# Per-domain genome operators (seed / mutate / descriptor). Pure stdlib.
# =========================================================================== #
def _seed_battery(rng: Random) -> Dict[str, Any]:
    return {"anode": rng.choice(list(_ANODES)),
            "cathode": rng.choice(list(_CATHODES)),
            "electrolyte": rng.choice(list(_ELECTROLYTES))}


def _mutate_battery(rng: Random, g: Dict[str, Any]) -> Dict[str, Any]:
    field_ = rng.choice(("anode", "cathode", "electrolyte"))
    pool = {"anode": list(_ANODES), "cathode": list(_CATHODES), "electrolyte": list(_ELECTROLYTES)}[field_]
    g[field_] = rng.choice(pool)
    return g


def _descr_battery(g: Dict[str, Any]) -> Tuple[float, float, float]:
    an = list(_ANODES); ca = list(_CATHODES); el = list(_ELECTROLYTES)
    ai = an.index(g["anode"]) if g.get("anode") in an else 0
    ci = ca.index(g["cathode"]) if g.get("cathode") in ca else 0
    ei = el.index(g["electrolyte"]) if g.get("electrolyte") in el else 0
    return (ai / max(1, len(an) - 1), ci / max(1, len(ca) - 1), ei / max(1, len(el) - 1))


def _seed_aircraft(rng: Random) -> Dict[str, Any]:
    top = rng.choice(list(_TOPOLOGIES))
    max_ar = int(_TOPOLOGIES[top][3] * 1.2)
    return {"topology": top, "aspect_ratio": rng.randint(5, max(6, max_ar)),
            "material": rng.choice(list(_MATERIALS)), "taper": rng.choice(_TAPERS)}


def _mutate_aircraft(rng: Random, g: Dict[str, Any]) -> Dict[str, Any]:
    field_ = rng.choice(("topology", "aspect_ratio", "material", "taper"))
    if field_ == "aspect_ratio":
        g["aspect_ratio"] = max(4, int(g.get("aspect_ratio", 9)) + rng.choice((-3, -2, -1, 1, 2, 3, 4)))
    elif field_ == "topology":
        g["topology"] = rng.choice(list(_TOPOLOGIES))
    elif field_ == "material":
        g["material"] = rng.choice(list(_MATERIALS))
    else:
        g["taper"] = rng.choice(_TAPERS)
    return g


def _descr_aircraft(g: Dict[str, Any]) -> Tuple[float, float, float]:
    tops = list(_TOPOLOGIES); mats = list(_MATERIALS)
    ti = tops.index(g["topology"]) if g.get("topology") in tops else 0
    mi = mats.index(g["material"]) if g.get("material") in mats else 0
    ar = _clamp01(float(g.get("aspect_ratio", 9)) / 34.0)
    return (ti / max(1, len(tops) - 1), ar, mi / max(1, len(mats) - 1))


def _seed_algorithm(rng: Random) -> Dict[str, Any]:
    strat = rng.choice(_STRATEGIES)
    g: Dict[str, Any] = {"strategy": strat}
    if strat in ("merge", "quick"):
        g["base_threshold"] = rng.choice((0, 0, 4, 8, 12, 16, 24, 32))
    if strat == "quick":
        g["pivot"] = rng.choice(_PIVOTS)
        g["introsort_fallback"] = rng.random() < 0.5
    return g


def _mutate_algorithm(rng: Random, g: Dict[str, Any]) -> Dict[str, Any]:
    strat = g.get("strategy", "insertion")
    knobs = ["strategy"]
    if strat in ("merge", "quick"):
        knobs.append("base_threshold")
    if strat == "quick":
        knobs += ["pivot", "introsort_fallback"]
    knob = rng.choice(knobs)
    if knob == "strategy":
        return _seed_algorithm(rng)
    if knob == "base_threshold":
        g["base_threshold"] = rng.choice((0, 4, 8, 12, 16, 24, 32))
    elif knob == "pivot":
        g["pivot"] = rng.choice(_PIVOTS)
    else:
        g["introsort_fallback"] = not g.get("introsort_fallback", False)
    return g


def _descr_algorithm(g: Dict[str, Any]) -> Tuple[float, float, float]:
    si = _STRATEGIES.index(g["strategy"]) if g.get("strategy") in _STRATEGIES else 0
    thr = _clamp01(float(g.get("base_threshold", 0)) / 32.0)
    pi = _PIVOTS.index(g["pivot"]) if g.get("pivot") in _PIVOTS else 0
    flag = 0.5 if g.get("introsort_fallback") else 0.0
    d2 = _clamp01((pi / max(1, len(_PIVOTS) - 1)) * 0.5 + flag)
    return (si / max(1, len(_STRATEGIES) - 1), thr, d2)


_SEEDERS: Dict[str, Callable[[Random], Dict[str, Any]]] = {
    "battery": _seed_battery, "aircraft": _seed_aircraft, "algorithm": _seed_algorithm,
}
_MUTATORS: Dict[str, Callable[[Random, Dict[str, Any]], Dict[str, Any]]] = {
    "battery": _mutate_battery, "aircraft": _mutate_aircraft, "algorithm": _mutate_algorithm,
}
_DESCRIPTORS: Dict[str, Callable[[Dict[str, Any]], Tuple[float, float, float]]] = {
    "battery": _descr_battery, "aircraft": _descr_aircraft, "algorithm": _descr_algorithm,
}


# --------------------------------------------------------------------------- #
# Self-test / demo
# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # pragma: no cover
    print("=" * 78)
    print("NYXARA invention (genuine invention — battery / aircraft / algorithm) self-test")
    print("=" * 78)

    eng = InventionEngine(seed=7)
    rep = eng.invent(generations=4, population=30)
    print(f"\ninvented={rep.generated}  feasible={rep.feasible}  infeasible={rep.infeasible}  "
          f"kept(novel+better)={rep.novel_kept}")
    print(f"by domain: {rep.by_domain}")
    print("\ntop self-invented designs (beating the known art):")
    for iv in rep.inventions[:8]:
        print(f"  [{iv.domain:9s}|{iv.candidate.origin:9s}] {iv.summary}")
        print(f"        ↑ {iv.improvement:.3f}× best baseline   novelty={iv.novelty:.3f}   "
              f"score={iv.score():.4f}")
    print(f"\nfrontier: {rep.frontier}")

    # (a) determinism under a fixed seed
    a = [iv.summary for iv in InventionEngine(seed=42).invent(3, 24).inventions]
    b = [iv.summary for iv in InventionEngine(seed=42).invent(3, 24).inventions]
    assert a == b, "runs must be deterministic under a fixed seed"
    print("\ndeterministic w/ seed : OK")

    # (b) infeasible designs are dropped — an aqueous electrolyte can't hold a 3.8 V cathode.
    bad = _battery_eval({"anode": "graphite", "cathode": "NMC811", "electrolyte": "aqueous"})
    assert bad["feasible"] is False, "voltage outside the electrolyte window must be infeasible"
    print("infeasible dropped    : OK")

    # (c) a re-derived textbook design scores ~0 novelty (a remix, not an invention).
    classic = DesignCandidate("battery", dict(_BATTERY_BASELINES[0]))
    assert eng._novelty(classic) < eng.novelty_floor, "the known art must read as non-novel"
    print("remix rejected        : OK")

    # (d) EVERY kept algorithm invention is correct vs the brute-force oracle AND faster than baseline.
    alg = [iv for iv in rep.inventions if iv.domain == "algorithm"]
    for iv in alg:
        m = algorithm_metrics(iv.candidate.genome)
        assert m["correct"] is True, f"kept algorithm is incorrect: {iv.candidate.genome}"
        assert m["speedup"] > 1.0, f"kept algorithm is not faster: {iv.candidate.genome}"
    print(f"algorithms verified   : OK ({len(alg)} correct & faster than insertion baseline)")

    # (e) report serialises to a plain JSON-able dict
    import json
    json.dumps(rep.to_dict())
    print("report serialises     : OK")

    print("\nALL SELF-TESTS PASSED ✓")
