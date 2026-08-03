"""NYXARA · growth/dataset_causal.py — L-CAUSAL: a dataset's table becomes a causal graph (⇢).

:class:`~nyxara.mind.causal_world_model.CausalWorldModel` already implements Pearl's whole ladder —
association with a stratified permutation test and confounder-set screening, ``do()`` by genuine
structural surgery on the DAG, and counterfactuals by the textbook abduction → action → prediction
procedure with PN/PS/PNS on top. What it could not do is **read a table**: every entry point it has
(:meth:`~nyxara.mind.causal_world_model.CausalWorldModel.observe`, ``observe_many``,
``observe_snapshot``) wants a *time-ordered stream of events*, and there is no ``from_rows``
anywhere in the package. So the engine and the Master's CSV had no way to meet. This module is the
meeting point.

**Why this does not fake an event stream.** The obvious shortcut is to replay each row as
co-occurring events with synthetic timestamps and let ``discover()`` run. That shortcut is wrong,
and quietly so: the association layer measures ``P(B|A) − P(B|¬A)`` over time windows, and if every
column "occurs" in every row then ``¬A`` never happens, the contingency degenerates, and the engine
reports verdicts about evidence it never actually had. Rather than launder a modelling assumption
through a machine that thinks it measured something, the structure for tabular data is computed
where it belongs — NOTEARS for orientation, multivariate least squares for effect sizes — and then
*registered* on the model with provenance saying exactly how each edge was derived. The engine's own
temporal discovery is left untouched for the streams it was built for.

**Orientation is the hard part, and abstention is a real answer.** A static table carries no arrow
of time, so direction has to come from somewhere:

* ``order=`` — the Master declares the ordering (treatment before outcome). Deterministic, honest,
  and always available.
* **NOTEARS** (:class:`~nyxara.mind.neural_causal.DifferentiableCausalStructure`) — gradient-based
  structure learning that needs no timestamps at all. Requires numpy.
* Neither → the report says ``orientation="unknown"`` and returns *no* causal edges. Guessing a
  direction and calling it causal would be worse than saying nothing.

**Effect sizes are multivariate on purpose.** Every mechanism in ``causal_world_model`` is a
single-predictor ridge fit, and the repo's own tests admit what that costs
(``tests/mind/test_causal_counterfactuals.py:36-41``): with correlated co-parents the slope absorbs
the other parent's influence and ``do()`` inherits the bias. Here each node is regressed on **all**
its parents at once through the existing :func:`~nyxara.growth.law_discovery._solve_lstsq`, so a
diamond gives the right coefficients instead of plausible ones.

Depends on ``mind/causal_world_model`` (the ladder), ``mind/neural_causal`` (NOTEARS, numpy-optional)
and ``growth/law_discovery`` (the least-squares solver). Nothing imports back.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

__all__ = [
    "CausalDatasetReport",
    "learn_causal_graph",
    "learn_from_dataset_store",
    "simulate_intervention",
    "counterfactual_on_dataset",
]

_MIN_ROWS = 8          # below this nothing can be honestly fit, so nothing is claimed


@dataclass
class CausalDatasetReport:
    """What was learned from a table — including, deliberately, what was refused."""

    rows: int = 0
    columns: List[str] = field(default_factory=list)
    orientation: str = "unknown"                 # "declared" | "notears" | "unknown"
    edges: List[Tuple[str, str, float]] = field(default_factory=list)
    r2: Dict[str, float] = field(default_factory=dict)          # node -> fit quality
    mechanisms: Dict[str, Dict[str, float]] = field(default_factory=dict)
    notears_edges: List[Tuple[str, str, float]] = field(default_factory=list)
    registered: int = 0                          # edges written onto the CausalWorldModel
    notes: List[str] = field(default_factory=list)

    def note(self, message: str) -> None:
        if message and message not in self.notes:
            self.notes.append(message)

    def summary(self) -> str:
        if self.orientation == "unknown":
            head = (f"· causal: {self.rows} row(s), {len(self.columns)} column(s) — "
                    f"direction unknown, so NO causal edge is claimed")
        else:
            head = (f"· causal ({self.orientation}): {len(self.edges)} edge(s) from {self.rows} "
                    f"row(s) over {len(self.columns)} column(s)")
        lines = [head]
        for cause, effect, weight in self.edges:
            fit = self.r2.get(effect)
            quality = f", R²={fit:.2f}" if fit is not None else ""
            lines.append(f"    {cause} → {effect}  (coef {weight:+.4g}{quality})")
        lines.extend(f"    · {n}" for n in self.notes)
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {"rows": self.rows, "columns": list(self.columns),
                "orientation": self.orientation,
                "edges": [[c, e, round(w, 6)] for c, e, w in self.edges],
                "r2": {k: round(v, 5) for k, v in self.r2.items()},
                "mechanisms": {k: {p: round(c, 6) for p, c in v.items()}
                               for k, v in self.mechanisms.items()},
                "notears_edges": [[c, e, round(w, 6)] for c, e, w in self.notears_edges],
                "registered": self.registered, "notes": list(self.notes)}


# --------------------------------------------------------------------------- #
# matrix preparation
# --------------------------------------------------------------------------- #
def _clean_matrix(columns: Sequence[str], rows: Sequence[Dict[str, float]]
                  ) -> Tuple[List[str], List[List[float]]]:
    """Keep the columns present-and-finite in every retained row. Never raises."""
    cols = [str(c) for c in columns]
    if not cols:
        return [], []
    matrix: List[List[float]] = []
    for row in rows:
        try:
            values = [float(row[c]) for c in cols]
        except (KeyError, TypeError, ValueError):
            continue          # a row missing a column cannot contribute to a joint fit
        if all(math.isfinite(v) for v in values):
            matrix.append(values)
    return cols, matrix


def _variance(values: Sequence[float]) -> float:
    n = len(values)
    if n < 2:
        return 0.0
    mean = sum(values) / n
    return sum((v - mean) ** 2 for v in values) / (n - 1)


# --------------------------------------------------------------------------- #
# orientation
# --------------------------------------------------------------------------- #
def _notears_edges(cols: Sequence[str], matrix: Sequence[Sequence[float]], *,
                   seed: int, threshold: float) -> List[Tuple[str, str, float]]:
    """Gradient-learned directed edges, or ``[]`` without numpy / on any failure."""
    try:
        from nyxara.mind.neural_causal import DifferentiableCausalStructure, has_numpy
        if not has_numpy():
            return []
        structure = DifferentiableCausalStructure(list(cols), seed=seed).fit(list(matrix))
        return structure.edges(threshold=threshold) if structure.fitted else []
    except Exception:  # noqa: BLE001 — NOTEARS is one opinion; its absence is a note, not a crash
        return []


def _order_from_edges(cols: Sequence[str],
                      edges: Sequence[Tuple[str, str, float]]) -> Optional[List[str]]:
    """A topological order over the learned edges (Kahn). ``None`` when the graph has a cycle."""
    parents: Dict[str, set] = {c: set() for c in cols}
    children: Dict[str, set] = {c: set() for c in cols}
    for cause, effect, _weight in edges:
        if cause in parents and effect in parents and cause != effect:
            parents[effect].add(cause)
            children[cause].add(effect)
    ready = sorted(c for c in cols if not parents[c])
    order: List[str] = []
    while ready:
        node = ready.pop(0)
        order.append(node)
        for child in sorted(children[node]):
            parents[child].discard(node)
            if not parents[child] and child not in order and child not in ready:
                ready.append(child)
        ready.sort()
    return order if len(order) == len(cols) else None


# --------------------------------------------------------------------------- #
# mechanisms
# --------------------------------------------------------------------------- #
def _fit_node(matrix: Sequence[Sequence[float]], index: Dict[str, int], node: str,
              parents: Sequence[str]) -> Tuple[Dict[str, float], Dict[str, float], float]:
    """Regress ``node`` on **all** its parents at once.

    Returns ``({parent: raw_coef}, {parent: standardized_coef}, R²)``. Multivariate on purpose: a
    single-predictor fit lets one parent's slope absorb a correlated co-parent's influence, and
    every downstream ``do()`` then inherits that bias. On a chain ``x→m→y`` this is the difference
    between reporting a direct ``x→y`` effect of ~6 and correctly reporting ~0.

    The standardized coefficient (``coef · σ(parent) / σ(node)``) is what makes pruning scale-free:
    whether a coefficient of 0.03 is negligible or dominant depends entirely on the units, and raw
    magnitude cannot tell the difference."""
    if not parents:
        return {}, {}, 0.0
    from nyxara.growth.law_discovery import _solve_lstsq

    target = [row[index[node]] for row in matrix]
    design = [[row[index[p]] for p in parents] + [1.0] for row in matrix]   # +1 = intercept
    solved = _solve_lstsq(design, target)
    if not solved or len(solved) != len(parents) + 1:
        return {}, {}, 0.0

    coefficients = {p: float(solved[i]) for i, p in enumerate(parents)}
    intercept = float(solved[-1])
    predicted = [sum(coefficients[p] * row[index[p]] for p in parents) + intercept
                 for row in matrix]
    residual = sum((t - p) ** 2 for t, p in zip(target, predicted))
    total = _variance(target) * (len(target) - 1)
    r2 = 1.0 - (residual / total) if total > 0 else 0.0

    node_sd = math.sqrt(_variance(target))
    standardized: Dict[str, float] = {}
    for parent, coefficient in coefficients.items():
        parent_sd = math.sqrt(_variance([row[index[parent]] for row in matrix]))
        standardized[parent] = (coefficient * parent_sd / node_sd) if node_sd > 0 else 0.0
    return coefficients, standardized, max(0.0, min(1.0, r2))


# --------------------------------------------------------------------------- #
# the entry point
# --------------------------------------------------------------------------- #
def learn_causal_graph(columns: Sequence[str], rows: Sequence[Dict[str, float]], *,
                       model: Any = None, order: Optional[Sequence[str]] = None,
                       threshold: float = 0.05, min_r2: float = 0.05,
                       min_effect: float = 0.05, seed: int = 0,
                       register: bool = True) -> CausalDatasetReport:
    """Learn a causal graph from a dataset's numeric columns.

    ``order`` declares the causal ordering (each column may only be caused by earlier ones). Absent
    it, NOTEARS supplies the direction; absent numpy too, the report abstains with
    ``orientation="unknown"`` and claims no causal edge at all.

    When ``model`` is a :class:`~nyxara.mind.causal_world_model.CausalWorldModel` and ``register``
    is set, the learned edges are written onto it so ``do()``, ``why()`` and ``counterfactual()``
    work over the dataset — each carrying provenance in ``CausalLink.evidence`` recording how it was
    derived. Never raises."""
    report = CausalDatasetReport()
    cols, matrix = _clean_matrix(columns, rows)
    report.columns, report.rows = cols, len(matrix)
    if len(cols) < 2:
        report.note("need at least two numeric columns to relate")
        return report
    if len(matrix) < _MIN_ROWS:
        report.note(f"only {len(matrix)} usable row(s); need {_MIN_ROWS} to fit honestly")
        return report

    report.notears_edges = _notears_edges(cols, matrix, seed=seed, threshold=threshold)

    # ---- orientation ---- #
    allowed: Optional[Dict[str, set]] = None      # node -> the parents NOTEARS actually supports
    if order:
        ordering = [c for c in order if c in cols]
        missing = [c for c in cols if c not in ordering]
        ordering.extend(missing)                  # undeclared columns sort last, never dropped
        report.orientation = "declared"
        if missing:
            report.note(f"columns not in --order, placed last: {', '.join(missing)}")
    elif report.notears_edges:
        ordering = _order_from_edges(cols, report.notears_edges)
        if ordering is None:
            report.note("NOTEARS returned a cyclic graph; no acyclic ordering — abstaining")
            return report
        report.orientation = "notears"
        # NOTEARS minimises ‖X − XW‖² — a LINEAR structural model. On a nonlinear law the reverse
        # direction can fit better than the true one, and the arrows come back inverted with full
        # confidence. Measured against sim/ ground truth (growth/causal_validation.py): it recovers
        # F ~ n·T exactly, abstains on τ = R·C, and gets v = √(T/μ) backwards. So a gradient-derived
        # direction is reported as what it is, and `order=` remains the reliable path.
        report.note("direction inferred by NOTEARS (linear structural fit) — it can invert on a "
                    "nonlinear law; pass order= when the ordering is known")
        allowed = {c: set() for c in cols}
        for cause, effect, _w in report.notears_edges:
            allowed[effect].add(cause)
    else:
        report.note("no --order given and NOTEARS unavailable (needs numpy) — "
                    "direction unknown, so no causal edge is claimed")
        return report

    # ---- mechanisms + edges ---- #
    index = {c: i for i, c in enumerate(cols)}
    for position, node in enumerate(ordering):
        candidates = [p for p in ordering[:position]
                      if allowed is None or p in allowed.get(node, set())]
        if not candidates:
            continue
        coefficients, standardized, r2 = _fit_node(matrix, index, node, candidates)
        if not coefficients or r2 < min_r2:
            continue
        report.r2[node] = r2
        # Prune on the STANDARDIZED effect, not the raw coefficient: raw magnitude is a statement
        # about units, not about influence. This is what drops the spurious direct edge on a
        # mediated chain — the multivariate fit already found it to be ~0, and this notices.
        kept = {p: c for p, c in coefficients.items() if abs(standardized.get(p, 0.0)) >= min_effect}
        if not kept:
            continue
        report.mechanisms[node] = kept
        for parent, coefficient in kept.items():
            report.edges.append((parent, node, coefficient))

    if register and model is not None:
        report.registered = _register(model, report, n_rows=len(matrix))
    return report


def _register(model: Any, report: CausalDatasetReport, *, n_rows: int) -> int:
    """Write the learned edges onto a CausalWorldModel with full provenance.

    Reaches into ``model._links`` deliberately: there is no public "assert this edge" API, and the
    alternative — synthesising an event stream so ``discover()`` re-derives them — would make the
    engine report measurements it never took. Confidence is the fit's own R² damped by sample size,
    never a fabricated contingency, and ``evidence`` records how the edge was derived so an audit
    can tell a table-learned edge from a stream-measured one."""
    try:
        from nyxara.mind.causal_world_model import CAUSAL, CausalLink
    except Exception:  # noqa: BLE001
        return 0
    links = getattr(model, "_links", None)
    if links is None:
        return 0

    written = 0
    # a small-sample damper: 30 rows is where the fit stops being mostly luck
    support = min(1.0, n_rows / 30.0)
    for cause, effect, coefficient in report.edges:
        r2 = report.r2.get(effect, 0.0)
        try:
            links[(cause, effect)] = CausalLink(
                cause=cause, effect=effect, strength=float(coefficient),
                confidence=max(0.0, min(1.0, r2 * support)), verdict=CAUSAL, lag=0.0,
                evidence={"source": "dataset_table", "orientation": report.orientation,
                          "r2": round(r2, 5), "n_rows": n_rows,
                          "parents": sorted(report.mechanisms.get(effect, {}))})
            written += 1
        except Exception:  # noqa: BLE001 — one unwritable edge never costs the rest
            continue
    return written


def learn_from_dataset_store(store_path: Any, *, model: Any = None,
                             order: Optional[Sequence[str]] = None,
                             seed: int = 0) -> CausalDatasetReport:
    """Read the numeric rows out of an ingested dataset store and learn their causal graph."""
    from nyxara.growth.dataset import load_dataset_table

    columns, rows = load_dataset_table(store_path)
    return learn_causal_graph(columns, rows, model=model, order=order, seed=seed)


# --------------------------------------------------------------------------- #
# asking the questions a table alone cannot answer
# --------------------------------------------------------------------------- #
def simulate_intervention(model: Any, interventions: Dict[str, float],
                          target: Optional[str] = None) -> Dict[str, float]:
    """``do(X=x)`` over the learned graph — what happens downstream if we *set* a variable.

    Structural surgery, not conditioning: the intervened variable is cut from its parents before
    the effect propagates. Returns ``{}`` when nothing is learned yet (never raises)."""
    try:
        result = model.do(dict(interventions), target=target)
    except Exception:  # noqa: BLE001 — an unlearned model answers with silence, not a traceback
        return {}
    if isinstance(result, dict):
        return {str(k): float(v) for k, v in result.items()}
    return {str(target or ""): float(result)} if result is not None else {}


def counterfactual_on_dataset(model: Any, cause: str, target: str, *,
                              factual: float = 1.0, counter: float = 0.0,
                              evidence: Optional[Dict[str, float]] = None) -> Optional[Any]:
    """"Would ``target`` still have happened had ``cause`` been different?"

    Delegates to the engine's three-step procedure (abduction → action → prediction). ``None`` when
    the model cannot answer — which it says honestly rather than guessing."""
    try:
        return model.counterfactual(cause, target, factual=factual, counter=counter,
                                    evidence=evidence)
    except Exception:  # noqa: BLE001
        return None
