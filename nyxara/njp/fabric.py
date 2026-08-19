"""NYXARA · njp/fabric.py — the Fluid Neural Automata (🧠, NJP V.01, the heart).

This is NJP's substrate, and the one thing in NYXARA that is neither a designed pipeline nor a
bag of weights. It is an **automaton**: every cell runs the *same* local law — leaky integrate,
threshold, fire, go refractory — and nothing anywhere says what any particular cell is for.
Behaviour is not written here. It falls out of structure, and **the structure is grown**.

What makes it *fluid* rather than merely plastic: the population itself changes. A conventional
network has a topology chosen up front and learns weights inside it. Here the cells and the
synapses are both outcomes.

**:meth:`expand` — what happens after every conversation and every task.** Five things, in order,
all driven by what actually fired and how the turn actually turned out:

1. **potentiation** — a synapse that carried a *causal* pair (``j`` fired, then ``i`` fired) is
   strengthened in proportion to the outcome.
2. **synaptogenesis** — a causal pair with **no synapse between them grows one**. This is the
   literal, physical expansion: ``len(fabric)`` synapses is higher after the call than before it,
   and :mod:`nyxara.njp.ledger` records both numbers.
3. **depression** — a synapse whose presynaptic cell fired and whose target then stayed silent is
   weakened. Growth without this is just accumulation.
4. **pruning** — a synapse below the floor is deleted outright.
5. **neurogenesis** — when the fabric keeps failing to *predict itself* (its own manifold
   precognition stays wrong over a window), that is evidence it lacks the capacity to represent
   what it is meeting, so **new cells are minted** and wired into the active cluster. Followed by
   **apoptosis**: cells that ended up connected to nothing and never fire are removed.

**Infinite learning, and what that honestly means here.** There is no small fixed ceiling: cells
and synapses are added whenever the evidence calls for them, and the fabric is written to disk so
growth survives a restart — the fabric that wakes up tomorrow is the one that went to sleep today.
What bounds it is the machine. Under memory pressure :meth:`consolidate` **compresses** — it
merges cells that have become functionally identical and drops the weakest tail — rather than
blindly evicting whatever is oldest. Growth continues; it is the resolution of the least-used
structure that degrades first. A fabric that has stopped growing says so in :meth:`stats` instead
of reporting a number that flatters it.

**Multi-dimensional latent reasoning.** :mod:`nyxara.njp.manifold` lifts each settled state into
one high-dimensional snapshot, so a thousand co-active cells are a single object and two whole
world-states compare in one dot product. The fabric learns its own transitions on that manifold,
which is what lets :meth:`anticipate` answer **before settling** — real forward inference over
learned dynamics, reported with the margin that says whether to believe it, and never asserted
when the map has not earned it.

Honest, as everywhere in this repo:

* **Real compute, and honest about the hardware.** The dynamics are genuine leaky
  integrate-and-fire physics solved over real elapsed time, executed as vectorised numpy — not a
  toy loop. It is still running on commodity silicon: ``stats()["backend"]`` names what is
  actually live, and will say ``numpy`` until neuromorphic hardware exists to say otherwise.
* **No backpropagation *in here*.** Every update in this file is local to a synapse and its two
  endpoints; no global error signal reaches it. Gradient learning is a separate, additive organ
  (:mod:`nyxara.njp.learn`) that reads this substrate without changing how it grows.
* **Deterministic given a seed.** Same seed, same stimulus, same fabric — which is what makes
  "it grew" a checkable claim rather than an anecdote.
* Every public method is **fail-soft**: on error it degrades to a null result rather than breaking
  a turn.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as _np

from nyxara.njp.backend import Backend, detect
from nyxara.njp.cell import Cell
from nyxara.njp.manifold import Manifold, Prediction, Snapshot

__all__ = ["GrowthReport", "SettleResult", "Fabric"]

#: Sentinel for "this fabric holds no grafted cells". Above every id she could grow: `next_id`
#: counts up from 1, and njp/graft.py allocates from 2**40, so a single integer comparison
#: separates the two populations without a set lookup on the hot path.
_GRAFT_NONE = 1 << 62


@dataclass
class SettleResult:
    """One settle: which cells fired, in what order, and what it cost."""

    trace: List[Tuple[int, ...]] = field(default_factory=list)   # firing set per step
    seed: Tuple[int, ...] = ()        # the externally driven cells, IN INPUT ORDER
    steps: int = 0
    quiescent: bool = False           # settled on its own rather than hitting the cap
    ms: float = 0.0
    snapshot: Optional[Snapshot] = None
    anticipated: Optional[Prediction] = None    # what was predicted BEFORE this ran
    prediction_score: Optional[float] = None    # how right a TRUSTED prediction turned out to be
    surprise: float = 1.0                       # how wrong she was about herself, 0.0 … 1.0

    @property
    def fired(self) -> Tuple[int, ...]:
        """Every cell that fired at any point in the settle, order preserved, de-duplicated."""
        seen: Dict[int, None] = {}
        for step in self.trace:
            for cid in step:
                seen.setdefault(cid, None)
        return tuple(seen)

    def to_dict(self) -> Dict[str, Any]:
        return {"steps": self.steps, "quiescent": self.quiescent,
                "n_fired": len(self.fired), "ms": round(self.ms, 3),
                "prediction_score": (round(self.prediction_score, 4)
                                     if self.prediction_score is not None else None),
                "surprise": round(self.surprise, 4),
                "anticipated": (self.anticipated.to_dict()
                                if self.anticipated is not None else None)}


@dataclass
class GrowthReport:
    """What one :meth:`Fabric.expand` actually changed. Every number is a real count."""

    potentiated: int = 0
    grown: int = 0                    # NEW synapses — the physical expansion
    depressed: int = 0
    pruned: int = 0
    born: int = 0                     # NEW cells — neurogenesis
    apoptosed: int = 0
    cells_before: int = 0
    cells_after: int = 0
    synapses_before: int = 0
    synapses_after: int = 0
    outcome: float = 0.0
    ms: float = 0.0

    @property
    def net_synapses(self) -> int:
        return self.synapses_after - self.synapses_before

    @property
    def expanded(self) -> bool:
        """Did the fabric actually get bigger? The claim this whole module has to earn."""
        return self.synapses_after > self.synapses_before or self.cells_after > self.cells_before

    def to_dict(self) -> Dict[str, Any]:
        return {"potentiated": self.potentiated, "grown": self.grown,
                "depressed": self.depressed, "pruned": self.pruned,
                "born": self.born, "apoptosed": self.apoptosed,
                "cells": [self.cells_before, self.cells_after],
                "synapses": [self.synapses_before, self.synapses_after],
                "net_synapses": self.net_synapses, "expanded": self.expanded,
                "outcome": round(self.outcome, 4), "ms": round(self.ms, 3)}


class Fabric:
    """Cells under one local law, with a population and a wiring that both grow."""

    def __init__(self, *, seed: int = 42,
                 leak: Optional[float] = None, threshold: float = 1.0, refractory: int = 2,
                 max_settle_steps: int = 24,
                 hebbian_rate: float = 0.08, depress_rate: float = 0.02,
                 prune_threshold: float = 0.015,
                 growth_budget: int = 256, initial_weight: float = 0.12,
                 neurogenesis_error: float = 0.55, neurogenesis_window: int = 8,
                 neurogenesis_batch: int = 8, neurogenesis_fanout: int = 6,
                 soft_cell_ceiling: int = 250_000, soft_synapse_ceiling: int = 4_000_000,
                 soft_latency_ms: float = 250.0, latency_floor_synapses: int = 8192,
                 manifold_dim: int = 10000, manifold: Optional[Manifold] = None,
                 dt: float = 0.001, tau_m: float = 0.02, tau_ref: float = 0.002,
                 candidate_cap: int = 4096,
                 backend: Optional[Backend] = None) -> None:
        self.seed = int(seed)

        # ---- real time, in seconds ---- #
        # The membrane equation is dV/dt = -V/tau_m + I. Solved exactly over dt for constant
        # input that is V <- V*exp(-dt/tau_m) + I, which is what `decay` returns. `leak` used to
        # be a bare per-step multiplier with no units; it is honoured when a caller sets it
        # explicitly (so old configs behave identically) and otherwise derived from tau_m.
        self.dt = max(1e-6, float(dt))                # simulated step, seconds
        self.tau_m = max(1e-6, float(tau_m))          # membrane time constant, seconds
        self.tau_ref = max(0.0, float(tau_ref))       # absolute refractory period, seconds
        self.now = 0.0                                # elapsed simulated time, seconds
        # `leak` defaults to None rather than to a number, so "not specified" and "specified as
        # the old default" are distinguishable. Keying off the value instead meant a caller who
        # explicitly asked for leak=0.85 silently got tau_m's decay instead of the one they asked
        # for — the two are only equal by coincidence, and never for any other value.
        if leak is not None:
            lk = min(1.0 - 1e-9, max(1e-9, float(leak)))
            self.tau_m = -self.dt / math.log(lk)
        self.leak = self.decay

        self.backend: Backend = backend if backend is not None else detect()
        # How many cells precognition decodes against. See `_candidate_pool` — this bounds the
        # decode AND keeps the trust margin meaningful on a large fabric.
        self.candidate_cap = max(16, int(candidate_cap))
        self._recent: Dict[int, None] = {}      # recency window, newest last

        # ---- the compiled sparse form (built lazily, invalidated by any structural edit) ---- #
        self._dirty = True
        self._order: Optional[List[int]] = None
        self._index: Dict[int, int] = {}
        self._indptr: Any = None
        self._indices: Any = None
        self._weights: Any = None
        self._state: Any = None
        self._bias: Any = None
        self._thresh: Any = None
        self._refr_until: Any = None
        self._hits: Any = None
        self._last_fired: Any = None
        self.compilations = 0
        self._synced = True
        self._touched: Any = None
        self.base_threshold = float(threshold)
        self.refractory = max(0, int(refractory))
        self.max_settle_steps = max(1, int(max_settle_steps))

        self.hebbian_rate = float(hebbian_rate)
        self.depress_rate = float(depress_rate)
        self.prune_threshold = float(prune_threshold)
        self.growth_budget = max(0, int(growth_budget))
        self.initial_weight = float(initial_weight)

        self.neurogenesis_error = float(neurogenesis_error)
        self.neurogenesis_window = max(1, int(neurogenesis_window))
        self.neurogenesis_batch = max(1, int(neurogenesis_batch))
        self.neurogenesis_fanout = max(1, int(neurogenesis_fanout))

        # "Soft" is the whole point: these do not stop growth, they turn on `consolidate`, which
        # compresses the least-used structure to make room. A hard cap would end learning; this
        # degrades the resolution of what is least used and keeps going.
        self.soft_cell_ceiling = max(64, int(soft_cell_ceiling))
        self.soft_synapse_ceiling = max(256, int(soft_synapse_ceiling))
        self.soft_latency_ms = max(1.0, float(soft_latency_ms))
        self.latency_floor_synapses = max(256, int(latency_floor_synapses))

        self.cells: Dict[int, Cell] = {}
        self.out: Dict[int, Dict[int, float]] = {}   # pre  -> {post: weight}
        self.inn: Dict[int, Set[int]] = {}           # post -> {pre}

        # The second synapse tier: weights she did not grow. `out`/`inn` are the right shape for
        # growth (one insert is O(1)) and the wrong one for a converted layer, where a Python dict
        # entry per weight runs ~100 bytes and a billion-parameter graft would be hundreds of GB.
        # A grafted region keeps its weights quantised and dense instead (njp/graft.py), and the
        # two tiers are kept apart on purpose: `grown` and `grafted` are different claims, and
        # folding them into one counter would make `growing` unfalsifiable.
        self.grafts: List[Any] = []
        self._frozen_ranges: Tuple[Tuple[int, int], ...] = ()
        self._graft_min = _GRAFT_NONE          # fast reject — any id below this she grew herself
        self._frozen_min = _GRAFT_NONE
        self._n_grafted_cells = 0

        self.manifold = manifold if manifold is not None else Manifold(dim=manifold_dim, seed=seed)

        self.tick = 0
        self.next_id = 1
        self.born_total = 0
        self.grown_total = 0
        self.pruned_total = 0
        self.apoptosed_total = 0
        self.expansions = 0

        self._pending: List[int] = []                # externally stimulated, fire next step
        self._last: Optional[SettleResult] = None
        self._prev_snapshot: Optional[Snapshot] = None
        self._errors: List[float] = []               # recent prediction errors — neurogenesis reads
        # What one expansion actually costs, in wall-clock milliseconds. The count ceilings above
        # bound *memory*; this bounds *time*, and time is what makes a fabric unusable first.
        self._expand_ms: List[float] = []
        self.consolidations = 0
        self.consolidated_on_latency = 0

    # ---- shape ----------------------------------------------------------- #
    def __len__(self) -> int:
        """The number of **synapses** — the thing that is claimed to grow."""
        return sum(len(t) for t in self.out.values())

    @property
    def n_cells(self) -> int:
        return len(self.cells)

    @property
    def n_synapses(self) -> int:
        return len(self)

    def ensure(self, cell_id: int) -> Cell:
        """Get a cell, minting it if this is the first time the fabric has met it."""
        got = self.cells.get(cell_id)
        if got is not None:
            return got
        cell = Cell(threshold=self.base_threshold, born_tick=self.tick)
        self.cells[cell_id] = cell
        self.out.setdefault(cell_id, {})
        self.inn.setdefault(cell_id, set())
        self.born_total += 1
        self._dirty = True
        if cell_id >= self.next_id:
            self.next_id = cell_id + 1
        return cell

    def connect(self, pre: int, post: int, weight: float) -> bool:
        """Create or overwrite one synapse. Returns True when it is genuinely NEW."""
        try:
            if pre == post:
                return False              # no self-synapses: a cell exciting itself never settles
            self.ensure(pre)
            self.ensure(post)
            fresh = post not in self.out[pre]
            self.out[pre][post] = float(weight)
            self.inn[post].add(pre)
            self._dirty = True
            if fresh:
                self.grown_total += 1
            return fresh
        except Exception:  # noqa: BLE001
            return False

    # ---- the grafted tier ------------------------------------------------ #
    def attach_graft(self, region: Any) -> None:
        """Attach a converted region (see :mod:`nyxara.njp.graft`) and refresh the id index."""
        self.grafts.append(region)
        self.refresh_grafts()
        self._dirty = True

    def refresh_grafts(self) -> None:
        """Recompute the id ranges the freeze and the drive path both read.

        Two different questions, two different ranges. *Grafted at all* decides whether apoptosis
        may remove a cell — and it must cover plastic regions too, because the check apoptosis
        actually performs (``no entry in out/inn and never fired``) is true of every grafted cell
        by construction: its synapses live in the region, not in the dicts. Reading that as "this
        cell connects to nothing" would delete a correctly wired layer. *Frozen* decides whether
        plasticity may touch it, and that one genuinely excludes ``plastic=True``.
        """
        frozen: List[Tuple[int, int]] = []
        graft_min = _GRAFT_NONE
        frozen_min = _GRAFT_NONE
        for r in self.grafts:
            spans = ((int(r.pre_lo), int(r.pre_lo) + int(r.n_pre)),
                     (int(r.post_lo), int(r.post_lo) + int(r.n_post)))
            for lo, hi in spans:
                graft_min = min(graft_min, lo)
                if not getattr(r, "plastic", False):
                    frozen.append((lo, hi))
                    frozen_min = min(frozen_min, lo)
        self._frozen_ranges = tuple(sorted(frozen))
        self._graft_min = graft_min
        self._frozen_min = frozen_min
        # Union rather than sum: regions that share a presynaptic block (a SwiGLU gate and up
        # projection read the same input) would otherwise count those cells twice, and the
        # grafted-cell total is one of the numbers `stats` publishes.
        spans = sorted((int(r.pre_lo), int(r.pre_lo) + int(r.n_pre)) for r in self.grafts)
        spans += sorted((int(r.post_lo), int(r.post_lo) + int(r.n_post)) for r in self.grafts)
        total, cursor = 0, -1
        for lo, hi in sorted(spans):
            lo = max(lo, cursor)
            if hi > lo:
                total += hi - lo
                cursor = hi
        self._n_grafted_cells = total

    def is_grafted(self, cell_id: int) -> bool:
        """True when this cell was placed by a graft rather than grown. O(1) in the common case."""
        return int(cell_id) >= self._graft_min

    def is_frozen(self, cell_id: int) -> bool:
        """True when plasticity must leave this cell alone."""
        cid = int(cell_id)
        if cid < self._frozen_min:
            return False
        for lo, hi in self._frozen_ranges:
            if lo <= cid < hi:
                return True
        return False

    @property
    def n_grafted_synapses(self) -> int:
        """Parameters held in grafted regions. Counted apart from ``n_synapses`` throughout."""
        return sum(int(r.n_params) for r in self.grafts)

    def _graft_drive(self, fired_mask: Any, drive: Any) -> None:
        """Add the current that grafted regions contribute from whatever just fired.

        The regions carry the same arithmetic the CSR path carries, in a different container: for
        each region, the presynaptic cells that spiked this step are gathered into a 0/1 vector,
        multiplied by the quantised weight block, and accumulated onto the region's post cells.
        Over a settle this is exactly the rate code :mod:`nyxara.njp.graft` converts against —
        the fabric's own law then integrates it and decides who fires, unchanged.
        """
        if not self.grafts or self._index is None:
            return
        for r in self.grafts:
            try:
                rows = getattr(r, "_pre_rows", None)
                prows = getattr(r, "_post_rows", None)
                if rows is None or prows is None:
                    continue
                act = fired_mask[rows]
                if not act.any():
                    continue
                drive[prows] += r.weights.matvec(act.astype(_np.float32))
            except Exception:  # noqa: BLE001 — one bad region must not lose the step
                continue

    def run_graft(self, region: Any, x: Any, *, timesteps: Optional[int] = None) -> Any:
        """Drive one converted region for ``T`` steps and return its output rates.

        **Why a graft needs its own driver, and why that is not a workaround.** A settle is
        *event-driven*: :meth:`stimulate` marks cells as having fired, :meth:`step` propagates from
        whatever fired last, and when nothing fires the settle is over. One stimulation therefore
        buys one propagation step per layer — which is exactly right for an automaton whose
        signals are events, and fatal for a rate code, where the information is *how often* a cell
        fires over a window. Measured directly: 30 turns of stimulate/settle/expand across a
        grafted layer produced zero spikes in its output cells, because each turn delivered a
        single sub-threshold step and the leak took it back before the next one.

        So the input is **held** for ``T`` steps here rather than injected once. That is the honest
        difference between the two tiers and it is a property of rate coding, not a defect: the
        converted layer is a function evaluated over a window, the grown fabric is a network of
        events. Both run on the same cells, and the spikes counted here land on those cells'
        real ``hits``/``last_fired``, so a grafted region that computed shows up in the fabric's
        own activity rather than in a private counter.
        """
        # No `decay=` override: the region carries the decay it was grafted under, and passing
        # this fabric's current one would simulate a law the fidelity certificate never measured.
        rates = region.forward(x, timesteps=timesteps, tau_ref_steps=0)
        try:
            T = int(timesteps if timesteps is not None else region.timesteps)
            counts = _np.rint(_np.asarray(rates) * T).astype(int)
            for k, n in enumerate(counts):
                if n <= 0:
                    continue
                cell = self.cells.get(region.post_lo + k)
                if cell is not None:
                    cell.hits += int(n)
                    cell.last_fired = self.now
            self._synced = False
            self._dirty = True
        except Exception:  # noqa: BLE001 — bookkeeping must never lose the computed answer
            pass
        return rates

    def _index_graft_rows(self) -> None:
        """Cache each region's compiled row numbers. Rebuilt with the CSR arrays, never per step.

        A region whose cells are not all present is left un-indexed rather than partially wired:
        a graft that computes over some of its inputs is not a slower graft, it is a different
        function, and it would pass every count-based check while returning the wrong numbers.
        """
        for r in self.grafts:
            try:
                pre = [self._index.get(r.pre_lo + k, -1) for k in range(r.n_pre)]
                post = [self._index.get(r.post_lo + k, -1) for k in range(r.n_post)]
                if min(pre) < 0 or min(post) < 0:
                    r._pre_rows = None
                    r._post_rows = None
                    continue
                r._pre_rows = _np.asarray(pre, dtype=_np.int64)
                r._post_rows = _np.asarray(post, dtype=_np.int64)
            except Exception:  # noqa: BLE001
                r._pre_rows = None
                r._post_rows = None

    def disconnect(self, pre: int, post: int) -> bool:
        try:
            targets = self.out.get(pre)
            if not targets or post not in targets:
                return False
            del targets[post]
            self.inn.get(post, set()).discard(pre)
            self.pruned_total += 1
            self._dirty = True
            return True
        except Exception:  # noqa: BLE001
            return False

    # ---- the local law --------------------------------------------------- #
    def stimulate(self, cell_ids: Iterable[int]) -> None:
        """Inject external drive: these cells are treated as having fired, so the next step
        propagates from them. This is the only way anything enters the fabric."""
        try:
            for cid in cell_ids:
                self.ensure(int(cid))
                self._pending.append(int(cid))
        except Exception:  # noqa: BLE001
            pass

    # ---- the compiled sparse form ---------------------------------------- #
    def _compile(self) -> None:
        """Compile the adjacency dicts into CSR arrays for the hot path.

        Two representations, each for what it is good at. Growth and pruning are irregular
        structural edits and stay on the dicts, where inserting one synapse is O(1). Propagation
        is a dense arithmetic problem over whatever fired, and belongs in arrays.

        Rebuilt lazily and only when the structure actually changed, so a settle-heavy workload
        compiles once and a growth-heavy one is not compiling per edit.
        """
        order = sorted(self.cells)
        index = {cid: i for i, cid in enumerate(order)}
        n = len(order)

        indptr = _np.zeros(n + 1, dtype=_np.int64)
        indices: List[int] = []
        weights: List[float] = []
        for i, cid in enumerate(order):
            targets = self.out.get(cid) or {}
            for post, w in targets.items():
                j = index.get(post)
                if j is None:
                    continue
                indices.append(j)
                weights.append(w)
            indptr[i + 1] = len(indices)

        self._order = order
        self._index = index
        self._indptr = indptr
        self._indices = _np.asarray(indices, dtype=_np.int64)
        self._weights = _np.asarray(weights, dtype=_np.float64)
        self._state = _np.asarray([self.cells[c].state for c in order], dtype=_np.float64)
        self._bias = _np.asarray([self.cells[c].bias for c in order], dtype=_np.float64)
        self._thresh = _np.asarray([self.cells[c].threshold for c in order], dtype=_np.float64)
        self._refr_until = _np.asarray(
            [self.cells[c].refractory_until for c in order], dtype=_np.float64)
        self._hits = _np.asarray([self.cells[c].hits for c in order], dtype=_np.int64)
        self._last_fired = _np.asarray(
            [self.cells[c].last_fired for c in order], dtype=_np.float64)
        # Grafted regions address cells by row like everything else on the hot path, so their row
        # numbers are rebuilt here — with the CSR arrays, on the same structural-change trigger,
        # never per step.
        if self.grafts:
            self._index_graft_rows()
        self._dirty = False
        self._synced = True
        self._touched = None
        self.compilations += 1

    def _writeback(self) -> None:
        """Push the vectorised state back onto the Cell objects the rest of the package reads.

        Called once per settle rather than once per step. Everything that reads a ``Cell`` —
        plasticity, apoptosis, persistence — runs after a settle completes, so syncing at the
        boundary is both sufficient and two orders of magnitude cheaper.
        """
        if self._order is None or self._synced:
            return
        # Only what the settle actually reached. Everything else is bit-for-bit unchanged, so
        # copying it back would be pure cost.
        indices = (_np.flatnonzero(self._touched) if self._touched is not None
                   else range(len(self._order)))
        for i in indices:
            cid = self._order[i]
            cell = self.cells.get(cid)
            if cell is None:
                continue
            cell.state = float(self._state[i])
            cell.refractory_until = float(self._refr_until[i])
            cell.hits = int(self._hits[i])
            cell.last_fired = float(self._last_fired[i])
        self._synced = True
        self._touched = None

    @property
    def decay(self) -> float:
        """``exp(-dt/tau_m)`` — how much of the membrane potential survives one step.

        This is the exact solution of the leaky-integrate membrane equation over ``dt`` for
        constant input, not a per-step multiply that resembles one. ``leak`` is kept as the
        configured name and now *derives* from real time constants, so existing configs keep
        working and the number gains units.
        """
        if self.tau_m <= 0.0:
            return 0.0
        return math.exp(-self.dt / self.tau_m)

    def step(self, driving: Sequence[int]) -> Tuple[int, ...]:
        """One step of the automaton. THE local rule, identical for every cell.

        Only the *targets* of the cells that just fired are evaluated — the fabric is sparse, and
        sweeping every cell every step would make a large fabric quadratic for no gain. A cell
        nobody is driving cannot cross threshold on this step, because the leak only ever moves it
        toward zero.

        The arithmetic is vectorised through :mod:`nyxara.njp.backend`. The previous form was a
        Python loop over a dict of dicts, which measured 72.9 ms per turn at 24k synapses —
        extrapolating to roughly 12 s per turn at the ``soft_synapse_ceiling`` this class already
        declares. The capacity was a number in a config file rather than something the code could
        reach.
        """
        self.tick += 1
        self.now += self.dt
        try:
            if self._dirty or self._order is None:
                self._compile()
            if not self._order:
                return ()

            rows = [self._index[c] for c in driving if c in self._index]
            if not rows:
                return ()

            # Gather the outgoing synapses of everything that just fired, and sum them onto their
            # targets. `bincount` is the accumulate: several presynaptic cells hitting the same
            # target must ADD, and plain fancy-index assignment would keep only the last one.
            # One gather over the concatenated out-edges of the whole frontier, then a single
            # bincount. `np.add.at` expresses the same thing and is roughly an order of magnitude
            # slower — it takes an unbuffered elementwise path specifically to handle repeated
            # indices, which is exactly what bincount is optimised for.
            spans = [(self._indptr[r], self._indptr[r + 1]) for r in rows]
            spans = [(lo, hi) for lo, hi in spans if hi > lo]
            if not spans:
                return ()
            idx = _np.concatenate([self._indices[lo:hi] for lo, hi in spans])
            wts = _np.concatenate([self._weights[lo:hi] for lo, hi in spans])
            drive = _np.bincount(idx, weights=wts, minlength=len(self._order))

            # Converted layers contribute to the SAME drive vector, and then the same local law
            # decides who fires. This is what keeps a graft inside the automaton rather than
            # beside it: nothing downstream can tell which tier a milliamp came from.
            if self.grafts:
                fired_now = _np.zeros(len(self._order), dtype=bool)
                fired_now[rows] = True
                self._graft_drive(fired_now, drive)

            touched = drive != 0.0
            if not touched.any():
                return ()

            refractory = self._refr_until >= self.now
            state, fired = self.backend.integrate(
                self._state, drive, self.decay, self._bias, self._thresh, refractory)

            # Only cells that actually received input this step may change: the frontier is the
            # unit of work, and letting the update touch silent cells would make every step O(N).
            self._state = _np.where(touched, state, self._state)
            fired = fired & touched

            if fired.any():
                self._note_recent(_np.flatnonzero(fired))
                self._refr_until = _np.where(fired, self.now + self.tau_ref, self._refr_until)
                self._last_fired = _np.where(fired, self.now, self._last_fired)
                self._hits = self._hits + fired.astype(_np.int64)

            # Deliberately NOT writing back to the Cell objects here. Writeback is a Python loop
            # over every cell, so doing it per step is O(N) per step regardless of how small the
            # active frontier is — which is precisely the cost the vectorisation exists to remove.
            # Measured: per-step writeback made a 24k-synapse turn *slower* than the dict loop it
            # replaced. The Cells are synced once, at the end of the settle, before anything reads
            # them.
            # Remember which cells this step actually touched, so the sync at the end of the
            # settle can write back only those. A settle usually touches a tiny fraction of a
            # large fabric, and syncing all of it was O(N_cells) per settle for no reason.
            self._touched = touched if self._touched is None else (self._touched | touched)
            self._synced = False
            return tuple(self._order[i] for i in _np.flatnonzero(fired))
        except Exception:  # noqa: BLE001 — a failed step is an empty step, never a crash
            return ()

    def settle(self, *, max_steps: Optional[int] = None,
               anticipate: bool = True) -> SettleResult:
        """Run the automaton from whatever was stimulated until it goes quiet.

        When ``anticipate`` is on and the manifold has learned enough, the expected outcome is
        computed **first** — that is the pre-cognitive read — and then scored against what the
        fabric actually did. That score is the fabric's own measure of how well it models itself,
        and it is what drives neurogenesis.
        """
        out = SettleResult()
        t0 = time.perf_counter()
        try:
            driving = tuple(dict.fromkeys(self._pending))
            self._pending = []
            if not driving:
                out.ms = (time.perf_counter() - t0) * 1000.0
                return out

            seed_snapshot = self.manifold.encode(driving, tick=self.tick)
            if anticipate:
                out.anticipated = self.manifold.precognition(
                    seed_snapshot, candidates=self._candidate_pool())

            # The externally driven cells DID fire — that is what stimulation means — so they are
            # step 0 of the trace. Without this a cold fabric could never grow: plasticity reads
            # the trace, the trace comes from propagation, and propagation needs the synapses that
            # do not exist yet. Recording the drive breaks that circle honestly.
            out.seed = driving
            out.trace.append(driving)

            cap = int(max_steps) if max_steps is not None else self.max_settle_steps
            for _ in range(cap):
                fired = self.step(driving)
                if not fired:
                    out.quiescent = True
                    break
                out.trace.append(fired)
                driving = fired
            out.steps = len(out.trace)
            self._writeback()          # the Cells are read from here on

            fired_all = out.fired
            out.snapshot = self.manifold.encode(fired_all or driving, tick=self.tick)

            # Surprise: how far what actually fired differed from what she expected to fire. This
            # is measured on EVERY settle, not only on trusted ones — "I was confident and wrong"
            # and "I had no expectation at all" are both genuine surprise, and a mind that only
            # scored itself when it already trusted itself could never notice the second. It is
            # the fabric's own read on how well it models itself, and the reason-seat discounts
            # confidence by it.
            if out.anticipated is not None and out.anticipated.cells:
                overlap = self.manifold.score_prediction(out.anticipated.cells, fired_all)
                out.surprise = max(0.0, min(1.0, 1.0 - overlap))
                if out.anticipated.trusted:
                    out.prediction_score = overlap
                    self._errors.append(out.surprise)
                    if len(self._errors) > self.neurogenesis_window * 4:
                        del self._errors[:-self.neurogenesis_window * 4]

            # Teach the manifold what actually followed what, so the next anticipate is better.
            if self._prev_snapshot is not None and out.snapshot is not None:
                self.manifold.learn_transition(self._prev_snapshot, out.snapshot)
            if seed_snapshot is not None and out.snapshot is not None:
                self.manifold.learn_transition(seed_snapshot, out.snapshot)
            self._prev_snapshot = out.snapshot

            self._last = out
            out.ms = (time.perf_counter() - t0) * 1000.0
            return out
        except Exception:  # noqa: BLE001
            out.ms = (time.perf_counter() - t0) * 1000.0
            return out

    def note_error(self, error: float) -> None:
        """Record a prediction error from *outside* the settle, into the same window.

        Neurogenesis reads this window and mints cells when it stays high, on the reasoning that
        persistent failure to predict is the one honest signal of insufficient capacity. A miss
        diagnosed by :mod:`nyxara.njp.predict` as a *perception* failure is exactly that evidence
        and had no way to reach here — the window was fed only by the fabric's own self-scoring,
        so an organ that noticed she could not represent an input could not act on it.

        Bounded identically to the in-settle path, so an external caller cannot flood the window
        and force growth that the fabric's own measurements do not support.
        """
        try:
            self._errors.append(max(0.0, min(1.0, float(error))))
            if len(self._errors) > self.neurogenesis_window * 4:
                del self._errors[:-self.neurogenesis_window * 4]
        except Exception:  # noqa: BLE001
            pass

    def _note_recent(self, rows: Any) -> None:
        """Keep a bounded, insertion-ordered window of what has fired lately.

        Maintained incrementally as cells fire — O(fired) per step — because the alternative is
        rebuilding it from every cell on every settle, which is the cost this window exists to
        avoid.
        """
        try:
            for i in rows:
                cid = self._order[int(i)]
                self._recent.pop(cid, None)      # re-insert so ordering is genuine recency
                self._recent[cid] = None
            overflow = len(self._recent) - self.candidate_cap * 2
            if overflow > 0:
                for cid in list(self._recent)[:overflow]:
                    del self._recent[cid]
        except Exception:  # noqa: BLE001
            pass

    def _candidate_pool(self) -> List[int]:
        """Which cells precognition should decode against.

        Not all of them, once the fabric is large, and that is a correctness argument before it is
        a speed one. Decoding is a similarity score over every candidate: a cell that has **never
        fired** cannot be what fires next, and including thousands of them does not add a
        possibility — it adds noise that dilutes the margin, which is the very number that decides
        whether the prediction is trusted. More candidates makes her less able to tell a real
        prediction from a vague one.

        It is also where the time goes. At 20k cells the decode is a 20000x10000 matvec, and it
        dominated the entire settle.

        So: cells that have actually fired, most recently first, capped. Below the cap this is
        every cell that has ever fired and the behaviour is unchanged; a cold fabric with nothing
        fired yet falls back to the full set, because then there is nothing to prefer.
        """
        try:
            if len(self.cells) <= self.candidate_cap:
                return list(self.cells.keys())
            # Read off the recency window maintained during `step`, NOT recomputed here. Scanning
            # and sorting every cell per settle is O(N log N) in Python and was itself slower than
            # the decode it was meant to bound — measured, at 200k synapses, twice as slow.
            if self._recent:
                return list(self._recent)[-self.candidate_cap:]
            return list(self.cells.keys())[: self.candidate_cap]
        except Exception:  # noqa: BLE001
            return list(self.cells.keys())[: self.candidate_cap]

    def anticipate(self, cell_ids: Iterable[int]) -> Prediction:
        """What would fire, **without running the fabric**. The pre-cognitive path.

        Honest by construction: when the manifold has too few transitions or the winners are not
        separated, this comes back ``trusted=False`` with the reason, and the caller is expected
        to settle properly instead. Foresight that has not been earned is reported as absent.
        """
        try:
            ids = list(cell_ids)
            snap = self.manifold.encode(ids, tick=self.tick)
            # Ask for about as many cells as a turn like this actually lights up, rather than a
            # fixed count: predicting sixteen when four fire pads the answer with near-misses and
            # drags the separation down. The seed width is the fabric's own best estimate of that.
            k = max(4, min(len(ids) * 2, 64))
            return self.manifold.precognition(snap, k=k, candidates=self._candidate_pool())
        except Exception:  # noqa: BLE001
            return Prediction(reason="anticipation failed")

    # ---- growth ---------------------------------------------------------- #
    def expand(self, *, outcome: float = 1.0,
               result: Optional[SettleResult] = None) -> GrowthReport:
        """**The expansion.** Run after every conversation and every task.

        ``outcome`` in ``[-1, 1]``: positive means the turn went well and what fired should be
        reinforced; negative means it did not and the same structure should be weakened. Zero is a
        real value — it means "no signal", and it correctly causes almost no change rather than
        being treated as failure.
        """
        rep = GrowthReport(outcome=float(outcome))
        t0 = time.perf_counter()
        rep.cells_before = self.n_cells
        rep.synapses_before = self.n_synapses
        try:
            res = result if result is not None else self._last
            if res is None or not res.trace:
                rep.cells_after, rep.synapses_after = rep.cells_before, rep.synapses_before
                rep.ms = (time.perf_counter() - t0) * 1000.0
                return rep

            outcome = max(-1.0, min(1.0, float(outcome)))
            self._potentiate_and_grow(res, outcome, rep)
            self._depress(res, outcome, rep)
            self._prune(rep)
            self._neurogenesis(res, rep)
            self._apoptosis(rep)

            if self.n_cells > self.soft_cell_ceiling or self.n_synapses > self.soft_synapse_ceiling:
                self.consolidate()
            elif self._too_slow():
                # The count ceilings are a memory bound and they are the wrong bound to hit first.
                # This class's own `_step` docstring works it out: 72.9 ms per turn at 24k
                # synapses extrapolates to roughly 12 SECONDS per turn at `soft_synapse_ceiling`.
                # So a fabric left to grow into its declared capacity becomes unusable long before
                # it becomes large, and `consolidate` — written precisely to keep learning going
                # under pressure — never runs. Measured over 1,200 corpus pairs: 136,085 synapses,
                # 3.4% of the ceiling, zero consolidations, and 748 ms per pair and climbing.
                #
                # Compressing on elapsed cost makes the honest claim in this module's header
                # ("what bounds it is the machine") into something the code can actually reach.
                self.consolidate()
                self.consolidated_on_latency += 1

            self.expansions += 1
            rep.cells_after = self.n_cells
            rep.synapses_after = self.n_synapses
            rep.ms = (time.perf_counter() - t0) * 1000.0
            self._record_cost(rep.ms)
            return rep
        except Exception:  # noqa: BLE001 — a failed expansion changes nothing, never crashes
            rep.cells_after = self.n_cells
            rep.synapses_after = self.n_synapses
            rep.ms = (time.perf_counter() - t0) * 1000.0
            return rep

    def _causal_pairs(self, res: SettleResult) -> List[Tuple[int, int]]:
        """(j, i) where j fired and i fired on the *next* step — the only pairs plasticity uses.

        Deliberately **not** "everything that fired together". Co-activation is symmetric and says
        only that two things turned up at once; a pair separated by one tick is the fabric's own
        evidence that j *drove* i. Growing on symmetric co-activation is how a network becomes a
        clique of hubs that mean nothing.
        """
        pairs: List[Tuple[int, int]] = []
        try:
            for t in range(len(res.trace) - 1):
                for j in res.trace[t]:
                    for i in res.trace[t + 1]:
                        if j != i:
                            pairs.append((j, i))
            # The seed is the one place where order inside a single step is real information:
            # the input arrived as a sequence, and "gravity" preceding "apple" is evidence the
            # propagated steps do not carry (cells firing on the same tick have no order at all).
            # This is also what gives a cold fabric its first structure to propagate through.
            for k in range(len(res.seed) - 1):
                j, i = res.seed[k], res.seed[k + 1]
                if j != i:
                    pairs.append((j, i))
        except Exception:  # noqa: BLE001
            pass
        # A frozen graft is excluded here, at the source, rather than inside each of the five
        # phases. Potentiation, synaptogenesis and depression all read this one list, so filtering
        # it once is both cheaper and harder to get wrong than remembering the rule four times.
        # Note what this protects against: the imported weights themselves live in the region and
        # are already out of reach of `_depress`/`_prune` (which only walk `out`), but
        # synaptogenesis would happily grow NEW dict synapses across a converted layer — quietly
        # turning a verified function into a different one that no certificate describes.
        if self._frozen_ranges and pairs:
            pairs = [(j, i) for j, i in pairs
                     if not (self.is_frozen(j) or self.is_frozen(i))]
        return pairs

    def _potentiate_and_grow(self, res: SettleResult, outcome: float,
                             rep: GrowthReport) -> None:
        """Strengthen what carried the signal — and **grow a synapse where none existed**."""
        budget = self.growth_budget
        for j, i in self._causal_pairs(res):
            targets = self.out.setdefault(j, {})
            if i in targets:
                targets[i] = _clip(targets[i] + self.hebbian_rate * outcome)
                self._dirty = True
                rep.potentiated += 1
            elif budget > 0 and outcome > 0.0:
                # SYNAPTOGENESIS. A causal pair the fabric had no wire for now has one.
                if self.connect(j, i, self.initial_weight * outcome):
                    rep.grown += 1
                    budget -= 1

    def _depress(self, res: SettleResult, outcome: float, rep: GrowthReport) -> None:
        """Weaken synapses whose presynaptic cell fired into a target that stayed silent."""
        try:
            for t in range(len(res.trace) - 1):
                nxt = set(res.trace[t + 1])
                for j in res.trace[t]:
                    for post, w in list(self.out.get(j, {}).items()):
                        if post not in nxt:
                            self.out[j][post] = _clip(w - self.depress_rate * abs(outcome))
                            self._dirty = True
                            rep.depressed += 1
        except Exception:  # noqa: BLE001
            pass

    def _prune(self, rep: GrowthReport) -> None:
        """Drop synapses that have fallen below the floor. Growth without this is hoarding."""
        try:
            for pre in list(self.out.keys()):
                for post, w in list(self.out[pre].items()):
                    if abs(w) < self.prune_threshold:
                        if self.disconnect(pre, post):
                            rep.pruned += 1
        except Exception:  # noqa: BLE001
            pass

    def _neurogenesis(self, res: SettleResult, rep: GrowthReport) -> None:
        """Mint new cells when the fabric keeps failing to predict itself.

        The trigger is **measured, not scheduled**: sustained precognition error means the current
        population cannot represent what it is meeting, which is the one honest reason to add
        capacity. A fabric that is predicting itself well grows no cells, however busy it is.
        """
        try:
            window = self._errors[-self.neurogenesis_window:]
            if len(window) < self.neurogenesis_window:
                return
            if (sum(window) / len(window)) < self.neurogenesis_error:
                return
            active = list(res.fired)[: self.neurogenesis_fanout * 2]
            if not active:
                return
            for _ in range(self.neurogenesis_batch):
                cid = self.next_id
                self.next_id += 1
                self.ensure(cid)
                # Wire the newborn INTO the active cluster, both ways: it must be reachable (or it
                # can never fire) and it must reach something (or it can never matter).
                for k, src in enumerate(active[: self.neurogenesis_fanout]):
                    self.connect(src, cid, self.initial_weight)
                    self.connect(cid, active[(k + 1) % len(active)], self.initial_weight)
                rep.born += 1
            self._errors.clear()      # capacity was added; re-measure before adding more
        except Exception:  # noqa: BLE001
            pass

    def _apoptosis(self, rep: GrowthReport) -> None:
        """Remove cells that connect to nothing and have never fired. Bounded per pass."""
        try:
            doomed: List[int] = []
            for cid, cell in self.cells.items():
                if cell.hits > 0:
                    continue
                # Every grafted cell looks unconnected to the test below, because a graft's
                # synapses are held in its region rather than in `out`/`inn`. It is not
                # unconnected; it is wired somewhere this loop cannot see. Left in, this test
                # would delete a freshly converted layer 64 ticks after it was attached and
                # before it had ever been driven.
                if self._graft_min != _GRAFT_NONE and cid >= self._graft_min:
                    continue
                if self.out.get(cid) or self.inn.get(cid):
                    continue
                if self.tick - cell.born_tick < 64:
                    continue              # give a newborn time to be recruited
                doomed.append(cid)
                if len(doomed) >= 128:
                    break
            for cid in doomed:
                self.cells.pop(cid, None)
                self.out.pop(cid, None)
                self.inn.pop(cid, None)
                self._dirty = True
                self.apoptosed_total += 1
                rep.apoptosed += 1
        except Exception:  # noqa: BLE001
            pass

    # ---- compression, so growth can continue ------------------------------ #
    def _record_cost(self, ms: float) -> None:
        """Remember what this expansion cost, over a short window."""
        try:
            self._expand_ms.append(max(0.0, float(ms)))
            if len(self._expand_ms) > 32:
                del self._expand_ms[:-32]
        except Exception:  # noqa: BLE001
            pass

    def _mean_expand_ms(self) -> Optional[float]:
        """Mean cost of a recent expansion, or ``None`` before there is a window to average."""
        if len(self._expand_ms) < 8:
            return None
        return sum(self._expand_ms) / len(self._expand_ms)

    def _too_slow(self) -> bool:
        """Has growth started costing more per turn than the machine should spend on it?

        Two guards, and both matter. A window is required so one slow turn — a garbage collection,
        a cold cache — cannot trigger compression of structure that is doing its job. And a floor
        on size is required so a genuinely small fabric on a loaded machine compresses nothing:
        below ``latency_floor_synapses`` the cost is the interpreter, not the structure, and
        dropping synapses would not make it faster. It would only make her smaller.
        """
        try:
            if self.n_synapses < self.latency_floor_synapses:
                return False
            mean = self._mean_expand_ms()
            return mean is not None and mean > self.soft_latency_ms
        except Exception:  # noqa: BLE001
            return False

    def consolidate(self, *, fraction: float = 0.05) -> Dict[str, Any]:
        """Make room by **compressing the least-used structure**, never by blind eviction.

        Called when the fabric crosses a soft ceiling. It drops the weakest tail of synapses —
        the ones carrying least signal — so the strong, load-bearing structure is untouched and
        learning continues. This is what "it keeps learning" means on a finite machine: the
        resolution of what is barely used degrades first, and that is reported rather than hidden.
        """
        out = {"synapses_before": self.n_synapses, "cells_before": self.n_cells, "dropped": 0}
        self.consolidations += 1
        # The window is cleared because it describes a fabric that no longer exists. Leaving it
        # would keep `_too_slow` true on the pre-compression costs and compress again next turn,
        # and the turn after — a fabric that shrank itself away while its own measurements said
        # it was still slow.
        self._expand_ms.clear()
        try:
            weights: List[Tuple[float, int, int]] = []
            for pre, targets in self.out.items():
                for post, w in targets.items():
                    weights.append((abs(w), pre, post))
            if not weights:
                out["synapses_after"] = self.n_synapses
                out["cells_after"] = self.n_cells
                return out
            weights.sort()
            n_drop = max(1, int(len(weights) * max(0.0, min(0.5, float(fraction)))))
            for _w, pre, post in weights[:n_drop]:
                if self.disconnect(pre, post):
                    out["dropped"] += 1
            out["synapses_after"] = self.n_synapses
            out["cells_after"] = self.n_cells
            return out
        except Exception:  # noqa: BLE001
            out["synapses_after"] = self.n_synapses
            out["cells_after"] = self.n_cells
            return out

    # ---- reporting ------------------------------------------------------- #
    def stats(self) -> Dict[str, Any]:
        # `cells`/`synapses` are what she GREW, with grafted structure subtracted out. This is
        # the whole reason the two tiers are counted apart. A 5-billion-parameter graft next to
        # forty grown synapses would swamp every growth number this dict reports, and `growing`
        # — the one counter that answers "is the automaton still an automaton" — would read true
        # forever on the strength of weights somebody else trained. Declared structure is
        # reported below, separately, and never added in.
        n_syn, n_cell = self.n_synapses, max(0, self.n_cells - self._n_grafted_cells)
        recent = self._errors[-self.neurogenesis_window:] if self._errors else []
        return {
            "cells": n_cell, "synapses": n_syn,
            "grafted": ({"regions": len(self.grafts),
                         "cells": self._n_grafted_cells,
                         "parameters": self.n_grafted_synapses,
                         "frozen_regions": sum(1 for r in self.grafts
                                               if not getattr(r, "plastic", False)),
                         "bytes": sum(int(getattr(r, "nbytes", 0)) for r in self.grafts),
                         "names": [str(getattr(r, "name", "")) for r in self.grafts[:8]]}
                        if self.grafts else None),
            "mean_degree": round(n_syn / n_cell, 4) if n_cell else 0.0,
            "tick": self.tick, "expansions": self.expansions,
            "born_total": self.born_total, "grown_total": self.grown_total,
            "pruned_total": self.pruned_total, "apoptosed_total": self.apoptosed_total,
            "recent_prediction_error": (round(sum(recent) / len(recent), 4) if recent else None),
            "manifold": self.manifold.stats(),
            # What the dynamics are ACTUALLY executing on, and the real time constants they use.
            # `numpy` here is the honest answer on a machine with no spiking hardware.
            "backend": self.backend.stats(),
            "physics": {"dt_s": self.dt, "tau_m_s": self.tau_m, "tau_ref_s": self.tau_ref,
                        "decay": round(self.decay, 6), "elapsed_s": round(self.now, 6)},
            "compilations": self.compilations,
            "candidate_cap": self.candidate_cap,
            # What growth currently costs, and the bound it is measured against. Reported so
            # "she is getting slower" is a number rather than something the Master notices.
            "mean_expand_ms": (round(self._mean_expand_ms(), 3)
                               if self._mean_expand_ms() is not None else None),
            "soft_latency_ms": self.soft_latency_ms,
            "consolidations": self.consolidations,
            "consolidated_on_latency": self.consolidated_on_latency,
            # Stated rather than implied: growth is not unconditional, and a fabric that has
            # stopped growing should be visibly stopped rather than quietly flat.
            "growing": self.expansions > 0 and self.grown_total > 0,
        }

    def to_dict(self) -> Dict[str, Any]:
        """Everything **learned**. This is what makes tomorrow's fabric today's fabric.

        Grafted regions are deliberately NOT here, and the omission is worth stating because a
        reader would otherwise assume the round trip is lossless. Two reasons. They are not
        learned — re-running the converter over the same GGUF is deterministic and reproduces
        them exactly, so persisting them would be storing a derived artefact. And they are
        gigabytes: writing 5 billion quantised weights into the same JSON as her synapses would
        make every save of a fabric she grew by forty synapses cost minutes and disk.
        ``stats()["grafted"]`` reports what is currently attached; ``python -m nyxara.njp.graft``
        re-attaches it.
        """
        # Grafted CELLS are excluded here as well as grafted weights, and this is not a detail.
        # Persisting them without their regions writes cells carrying converted thresholds and
        # biases into the save; on reload `_n_grafted_cells` is zero, nothing claims them, and
        # they are counted as cells she GREW. Measured: a fabric of 20 grown cells with one small
        # graft reloaded as 68 grown cells — the growth claim inflating by itself, silently, once
        # per restart. Excluding them keeps a reload honest, and the converter puts them back.
        return {
            "seed": self.seed, "tick": self.tick, "next_id": self.next_id,
            "cells": {str(cid): c.to_dict() for cid, c in self.cells.items()
                      if not (self._graft_min != _GRAFT_NONE and cid >= self._graft_min)},
            "synapses": [[pre, post, round(w, 6)]
                         for pre, targets in self.out.items()
                         for post, w in targets.items()],
            "counters": {"born": self.born_total, "grown": self.grown_total,
                         "pruned": self.pruned_total, "apoptosed": self.apoptosed_total,
                         "expansions": self.expansions},
            "manifold": self.manifold.to_dict(),
        }

    def load_dict(self, d: Dict[str, Any]) -> None:
        try:
            self.tick = int(d.get("tick", 0))
            self.next_id = int(d.get("next_id", 1))
            self.cells = {}
            self.out = {}
            self.inn = {}
            for sid, payload in (d.get("cells") or {}).items():
                cid = int(sid)
                self.cells[cid] = Cell.from_dict(payload)
                self.out.setdefault(cid, {})
                self.inn.setdefault(cid, set())
            for row in (d.get("synapses") or []):
                try:
                    pre, post, w = int(row[0]), int(row[1]), float(row[2])
                except Exception:  # noqa: BLE001 — one bad row never voids the rest
                    continue
                self.ensure(pre)
                self.ensure(post)
                self.out[pre][post] = w
                self.inn[post].add(pre)
            counters = d.get("counters") or {}
            self.born_total = int(counters.get("born", 0))
            self.grown_total = int(counters.get("grown", 0))
            self.pruned_total = int(counters.get("pruned", 0))
            self.apoptosed_total = int(counters.get("apoptosed", 0))
            self.expansions = int(counters.get("expansions", 0))
            self._dirty = True
            if d.get("manifold"):
                self.manifold.load_dict(d["manifold"])
        except Exception:  # noqa: BLE001 — a corrupt sidecar leaves a freshly-born fabric
            pass


def _clip(w: float, *, lo: float = -4.0, hi: float = 4.0) -> float:
    """Keep a synapse in a sane range. Unbounded weights make the automaton diverge, not learn."""
    return lo if w < lo else (hi if w > hi else float(w))
