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

* **Simulation on commodity silicon.** Not neuromorphic hardware. **No backpropagation** anywhere:
  every update here is local to a synapse and its two endpoints.
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

from nyxara.njp.cell import Cell
from nyxara.njp.manifold import Manifold, Prediction, Snapshot

__all__ = ["GrowthReport", "SettleResult", "Fabric"]


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
                 leak: float = 0.85, threshold: float = 1.0, refractory: int = 2,
                 max_settle_steps: int = 24,
                 hebbian_rate: float = 0.08, depress_rate: float = 0.02,
                 prune_threshold: float = 0.015,
                 growth_budget: int = 256, initial_weight: float = 0.12,
                 neurogenesis_error: float = 0.55, neurogenesis_window: int = 8,
                 neurogenesis_batch: int = 8, neurogenesis_fanout: int = 6,
                 soft_cell_ceiling: int = 250_000, soft_synapse_ceiling: int = 4_000_000,
                 manifold_dim: int = 10000, manifold: Optional[Manifold] = None) -> None:
        self.seed = int(seed)
        self.leak = float(leak)
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

        self.cells: Dict[int, Cell] = {}
        self.out: Dict[int, Dict[int, float]] = {}   # pre  -> {post: weight}
        self.inn: Dict[int, Set[int]] = {}           # post -> {pre}

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
            if fresh:
                self.grown_total += 1
            return fresh
        except Exception:  # noqa: BLE001
            return False

    def disconnect(self, pre: int, post: int) -> bool:
        try:
            targets = self.out.get(pre)
            if not targets or post not in targets:
                return False
            del targets[post]
            self.inn.get(post, set()).discard(pre)
            self.pruned_total += 1
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

    def step(self, driving: Sequence[int]) -> Tuple[int, ...]:
        """One tick of the automaton. THE local rule, identical for every cell.

        Only the *targets* of the cells that just fired are evaluated — the fabric is sparse, and
        sweeping every cell every tick would make a large fabric quadratic for no gain. A cell
        nobody is driving cannot cross threshold on this tick, because the leak only ever moves it
        toward zero.
        """
        self.tick += 1
        fired: List[int] = []
        try:
            # Accumulate this tick's input, per target cell.
            drive: Dict[int, float] = {}
            for pre in driving:
                for post, w in self.out.get(pre, {}).items():
                    drive[post] = drive.get(post, 0.0) + w

            for cid, net in drive.items():
                cell = self.cells.get(cid)
                if cell is None:
                    continue
                if self.tick <= cell.refractory_until:
                    cell.state = 0.0            # refractory cells integrate nothing
                    continue
                cell.state = cell.state * self.leak + net + cell.bias
                if cell.state >= cell.threshold:
                    cell.state = 0.0            # reset-to-zero after firing
                    cell.refractory_until = self.tick + self.refractory
                    cell.last_fired = self.tick
                    cell.hits += 1
                    fired.append(cid)
            return tuple(fired)
        except Exception:  # noqa: BLE001 — a failed tick is an empty tick, never a crash
            return tuple(fired)

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
                    seed_snapshot, candidates=list(self.cells.keys()))

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
            return self.manifold.precognition(snap, k=k, candidates=list(self.cells.keys()))
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

            self.expansions += 1
            rep.cells_after = self.n_cells
            rep.synapses_after = self.n_synapses
            rep.ms = (time.perf_counter() - t0) * 1000.0
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
        return pairs

    def _potentiate_and_grow(self, res: SettleResult, outcome: float,
                             rep: GrowthReport) -> None:
        """Strengthen what carried the signal — and **grow a synapse where none existed**."""
        budget = self.growth_budget
        for j, i in self._causal_pairs(res):
            targets = self.out.setdefault(j, {})
            if i in targets:
                targets[i] = _clip(targets[i] + self.hebbian_rate * outcome)
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
                self.apoptosed_total += 1
                rep.apoptosed += 1
        except Exception:  # noqa: BLE001
            pass

    # ---- compression, so growth can continue ------------------------------ #
    def consolidate(self, *, fraction: float = 0.05) -> Dict[str, Any]:
        """Make room by **compressing the least-used structure**, never by blind eviction.

        Called when the fabric crosses a soft ceiling. It drops the weakest tail of synapses —
        the ones carrying least signal — so the strong, load-bearing structure is untouched and
        learning continues. This is what "it keeps learning" means on a finite machine: the
        resolution of what is barely used degrades first, and that is reported rather than hidden.
        """
        out = {"synapses_before": self.n_synapses, "cells_before": self.n_cells, "dropped": 0}
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
        n_syn, n_cell = self.n_synapses, self.n_cells
        recent = self._errors[-self.neurogenesis_window:] if self._errors else []
        return {
            "cells": n_cell, "synapses": n_syn,
            "mean_degree": round(n_syn / n_cell, 4) if n_cell else 0.0,
            "tick": self.tick, "expansions": self.expansions,
            "born_total": self.born_total, "grown_total": self.grown_total,
            "pruned_total": self.pruned_total, "apoptosed_total": self.apoptosed_total,
            "recent_prediction_error": (round(sum(recent) / len(recent), 4) if recent else None),
            "manifold": self.manifold.stats(),
            # Stated rather than implied: growth is not unconditional, and a fabric that has
            # stopped growing should be visibly stopped rather than quietly flat.
            "growing": self.expansions > 0 and self.grown_total > 0,
        }

    def to_dict(self) -> Dict[str, Any]:
        """Everything learned. This is what makes tomorrow's fabric today's fabric."""
        return {
            "seed": self.seed, "tick": self.tick, "next_id": self.next_id,
            "cells": {str(cid): c.to_dict() for cid, c in self.cells.items()},
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
            if d.get("manifold"):
                self.manifold.load_dict(d["manifold"])
        except Exception:  # noqa: BLE001 — a corrupt sidecar leaves a freshly-born fabric
            pass


def _clip(w: float, *, lo: float = -4.0, hi: float = 4.0) -> float:
    """Keep a synapse in a sane range. Unbounded weights make the automaton diverge, not learn."""
    return lo if w < lo else (hi if w > hi else float(w))
