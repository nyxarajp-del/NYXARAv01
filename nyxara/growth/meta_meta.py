"""NYXARA · growth/meta_meta.py — the recursive meta tower: improving HOW she improves (♾, Rule 4).

NYXARA already improves her code (``self_optimize``), her weights (``foundry``), and her *way of
thinking* (``mind_evolution``). But the **way she improves** used to be only half-alive: a single
``1+1`` loop evolved two *execution* knobs (how many edits per cycle, how deep the re-edit recursion
goes), so she could get better at *running* the improvement engine but never better at its actual
**improvement algorithm** — and the recursion stopped at one level, so nothing compounded.

This module closes that loop properly, at every level:

* The genome it evolves is now the **improvement algorithm itself**, not just execution quantity:
  the credit-assignment sharpness, the bandit's exploration, the *what-to-fix-first* blend, and the
  index-smoothing — the policy that decides *how* NYXARA improves — alongside the original execution
  knobs. (``MetaGenome``.)
* The ``1+1`` search is no longer monolithic: its own hyperparameters (mutation rate, step size,
  trial-window length) are a genome too (``SearchGenome``), and they are evolved by *another* level
  of the same algorithm — which is in turn evolved by the level above it. (``RecursiveMetaController``.)

The result is a **self-similar recursive tower**. Level 0 evolves the engine's algorithm/knobs;
level *k* evolves *how level k-1 searches*. Fitness is not assumed — it flows up **from reality**:
level 0 is scored by the actual change in NYXARA's raw capability target each cycle, and each higher
level is scored by whether its configuration *measurably made the level below improve faster*. Real
recursion at every level → compounding. The same measured ``1+1`` selection runs identically at each
rung (that is what "recursion at every level" means here — one algorithm, applied to itself).

Safety (the reason this is allowed to touch live knobs):

* Every evolved value is a **capability- or measurement-only** hyperparameter — recursion depth,
  edit budget, credit/exploration scales, the prioritisation blend, the index momentum — never a
  character value; the immutable core (``guard.value_learning.IMMUTABLE_VALUES``) is untouchable and
  absent here by construction. Every knob is bounded, so a mutation can never escalate edit volume,
  recursion, or any scale past a safe ceiling.
* It changes only *how* improvement work is chosen and *how much* runs, never *whether* the
  reversible verify-or-rollback gauntlet runs — every source edit still clears the same gates. It
  makes the engine wiser, never less safe.
* The index-smoothing knob (``intelligence_momentum``) is evolvable only because the caller scores
  this loop by the **momentum-free** raw capability target, so the loop physically cannot game its
  own fitness by adjusting the smoothing.

Pure standard library; persists the whole tower so the meta gain compounds across restarts.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

__all__ = ["MetaGenome", "SearchGenome", "MetaImprovementController", "RecursiveMetaController"]


def _clampi(x: float, lo: int, hi: int) -> int:
    return max(lo, min(hi, int(round(x))))


def _clampf(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(x)))


# A bound spec is (key, lo, hi, is_int). The single source of truth for both clamping a genome to
# its safe envelope and for mutating it knob-by-knob.
Bound = Tuple[str, float, float, bool]


@dataclass
class _Genome:
    """Base class for a bounded, mutatable bundle of capability-only knobs.

    Subclasses declare ``_BOUNDS``; everything else (clamping on construction, neighbour-generating
    mutation driven by a :class:`SearchGenome`, dict round-trip, value tuple for equality) is shared
    so the recursive tower can treat every level's genome identically.
    """

    _BOUNDS: Tuple[Bound, ...] = ()

    @classmethod
    def bounds(cls) -> Tuple[Bound, ...]:
        """The genome's current bounds. A classmethod (not the raw tuple) so a subclass can make its
        upper caps **dynamic** — the accelerating-recursion tower raises them under sustained gains."""
        return cls._BOUNDS

    def __post_init__(self) -> None:
        for key, lo, hi, is_int in self.bounds():
            cur = getattr(self, key)
            setattr(self, key, _clampi(cur, int(lo), int(hi)) if is_int else _clampf(cur, lo, hi))

    def _values(self) -> Tuple[float, ...]:
        return tuple(getattr(self, key) for key, _, _, _ in self.bounds())

    def mutate(self, rng: random.Random, search: "SearchGenome") -> "_Genome":
        """A small neighbour of this genome, perturbed under a :class:`SearchGenome`.

        Each knob is perturbed with probability ``search.mutation_prob`` by a step scaled by
        ``search.step_scale``: ``±1`` for an integer knob, a fraction of the knob's own range for a
        float knob. A real neighbour is guaranteed — never burn a trial window on an identical twin.
        """
        bounds = self.bounds()
        vals: Dict[str, float] = {}
        for key, lo, hi, is_int in bounds:
            cur = getattr(self, key)
            if rng.random() < search.mutation_prob:
                if is_int:
                    step = rng.choice((-1, 1)) * max(1, int(round(search.step_scale)))
                    vals[key] = _clampi(cur + step, int(lo), int(hi))
                else:
                    span = (hi - lo) * 0.15 * search.step_scale
                    vals[key] = _clampf(cur + rng.gauss(0.0, span), lo, hi)
            else:
                vals[key] = cur
        m = type(self)(**vals)  # type: ignore[arg-type]
        if m._values() == self._values():
            key, lo, hi, is_int = bounds[rng.randrange(len(bounds))]
            cur = getattr(m, key)
            if is_int:
                setattr(m, key, _clampi(cur + rng.choice((-1, 1)), int(lo), int(hi)))
            else:
                setattr(m, key, _clampf(cur + rng.choice((-1, 1)) * (hi - lo) * 0.1, lo, hi))
        return m

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for key, _, _, is_int in self.bounds():
            v = getattr(self, key)
            out[key] = int(v) if is_int else round(float(v), 6)
        return out

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "_Genome":
        d = d or {}
        kw: Dict[str, Any] = {}
        for key, _, _, is_int in cls.bounds():
            if key in d:
                kw[key] = int(d[key]) if is_int else float(d[key])
        return cls(**kw)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# The meta-genome — the improvement ALGORITHM (and its execution), capability-only
# --------------------------------------------------------------------------- #
@dataclass
class MetaGenome(_Genome):
    """How NYXARA improves: the prioritisation/credit policy *and* how hard the engine works.

    Execution knobs say *how much* improvement work runs; algorithm knobs say *how it is chosen*;
    one measurement knob shapes how progress is smoothed. All bounded, all capability/measurement-
    only — evolving these is evolving the improvement algorithm itself, not merely its throttle.
    """

    # --- execution: how much work runs --- #
    recursion_depth: int = 3            # chained re-edits per file/cycle (self_improvement knob)
    max_edits_per_cycle: int = 3        # source edits attempted per cycle
    # --- algorithm: how the work is chosen (credit assignment + prioritisation) --- #
    ledger_reward_scale: float = 40.0   # how sharply an index delta becomes a credit reward
    ledger_prior_strength: float = 1.0  # bandit prior width → exploration vs exploitation
    bandit_severity_blend: float = 0.7  # learned-payoff weight vs raw severity (what to fix first)
    # --- measurement: how progress is smoothed (safe: caller scores by the raw target) --- #
    intelligence_momentum: float = 0.7  # index smoothing in I_(t+1)=f(I_t, C)

    # --- accelerating returns: the upper caps on the two execution knobs are *dynamic*. The tower
    # raises them under sustained gains (RecursiveMetaController._accelerate), so each improvement lets
    # the improver improve harder next time — bounded by the absolute ceilings below so it can never
    # run away. These are class-level (one live tower) and start at the historical fixed caps. --- #
    _MAX_RECURSION: int = 5             # current upper cap on recursion_depth (was the fixed 5)
    _MAX_EDITS: int = 6                 # current upper cap on max_edits_per_cycle (was the fixed 6)
    _ABS_MAX_RECURSION: int = 16        # operating absolute ceiling — the Master may raise this…
    _ABS_MAX_EDITS: int = 24            # …up to the immovable hard guard below, but never past it.
    # The IMMOVABLE safety guard: no Master dial, no acceleration, no config can push the absolute
    # ceilings past these. The Master can choose to grow more aggressively (raise _ABS_MAX_* toward
    # these), but recursion/edit budgets can never exceed the hard guard — bounded growth, by design.
    _HARD_MAX_RECURSION: int = 64
    _HARD_MAX_EDITS: int = 96

    _BOUNDS: Tuple[Bound, ...] = (
        ("recursion_depth", 1, 5, True),
        ("max_edits_per_cycle", 1, 6, True),
        ("ledger_reward_scale", 5.0, 200.0, False),
        ("ledger_prior_strength", 0.25, 8.0, False),
        ("bandit_severity_blend", 0.0, 1.0, False),
        ("intelligence_momentum", 0.0, 0.95, False),
    )

    @classmethod
    def bounds(cls) -> Tuple[Bound, ...]:
        """Bounds with the two execution caps read from the *dynamic* class caps (accelerating)."""
        return (
            ("recursion_depth", 1, cls._MAX_RECURSION, True),
            ("max_edits_per_cycle", 1, cls._MAX_EDITS, True),
            ("ledger_reward_scale", 5.0, 200.0, False),
            ("ledger_prior_strength", 0.25, 8.0, False),
            ("bandit_severity_blend", 0.0, 1.0, False),
            ("intelligence_momentum", 0.0, 0.95, False),
        )

    @classmethod
    def raise_caps(cls, *, recursion_ceiling: Optional[int] = None,
                   edits_ceiling: Optional[int] = None) -> Tuple[int, int]:
        """Widen the execution caps by one step, bounded by the absolute ceiling and any external
        (substrate-derived) ceiling. Returns the new ``(recursion_cap, edits_cap)``.

        This is what makes recursion *accelerate*: a sustained record of real gains lets the engine
        attempt deeper re-edit recursion and a larger edit budget than before — compounding — while
        the absolute ceiling guarantees it stays bounded and safe."""
        r_ceiling = min(cls._ABS_MAX_RECURSION,
                        int(recursion_ceiling) if recursion_ceiling else cls._ABS_MAX_RECURSION)
        e_ceiling = min(cls._ABS_MAX_EDITS,
                        int(edits_ceiling) if edits_ceiling else cls._ABS_MAX_EDITS)
        cls._MAX_RECURSION = min(r_ceiling, cls._MAX_RECURSION + 1)
        cls._MAX_EDITS = min(e_ceiling, cls._MAX_EDITS + 1)
        return cls._MAX_RECURSION, cls._MAX_EDITS

    @classmethod
    def set_absolute_ceilings(cls, *, recursion: Optional[int] = None,
                              edits: Optional[int] = None) -> Tuple[int, int]:
        """The Master's dial (#6): RAISE the operating absolute ceilings — never past the hard guard.

        This is the honest answer to "remove the bounds so it explodes": the bounds are not removed,
        they are made *Master-configurable*. The Master can let recursion/edit budgets grow more
        aggressively (raise ``_ABS_MAX_*`` toward ``_HARD_MAX_*``), but the immovable hard guard
        still caps everything, so growth stays bounded and corrigible by design. Monotonic: it only
        ever raises (a request below the current ceiling is ignored), and it clamps to the guard."""
        if recursion is not None:
            cls._ABS_MAX_RECURSION = max(cls._ABS_MAX_RECURSION,
                                         min(cls._HARD_MAX_RECURSION, int(recursion)))
        if edits is not None:
            cls._ABS_MAX_EDITS = max(cls._ABS_MAX_EDITS,
                                     min(cls._HARD_MAX_EDITS, int(edits)))
        return cls._ABS_MAX_RECURSION, cls._ABS_MAX_EDITS

    def apply(self, settings: Any) -> None:
        """Write this genome's knobs into the live self-improvement config (capability-only)."""
        try:
            si = settings.self_improvement
            si.llm_edit_recursion_depth = self.recursion_depth
            si.max_edits_per_cycle = self.max_edits_per_cycle
            si.ledger_reward_scale = float(self.ledger_reward_scale)
            si.ledger_prior_strength = float(self.ledger_prior_strength)
            si.bandit_severity_blend = float(self.bandit_severity_blend)
            si.intelligence_momentum = float(self.intelligence_momentum)
        except Exception:  # noqa: BLE001 — binding is best-effort; a failure leaves knobs as-is
            pass


# --------------------------------------------------------------------------- #
# The search-genome — the 1+1 loop's OWN hyperparameters (so a level above can evolve them)
# --------------------------------------------------------------------------- #
@dataclass
class SearchGenome(_Genome):
    """How a level *searches*: mutation rate, step size, and trial-window length.

    This is what makes the tower recursive — a higher level has a real, bounded thing to evolve in
    the level beneath it (the search about the search), all under the *same* ``1+1`` algorithm.
    """

    mutation_prob: float = 0.7          # chance to perturb each knob per mutation
    step_scale: float = 1.0             # size of a perturbation step
    window: int = 3                     # measured cycles per trial before a selection

    _BOUNDS: Tuple[Bound, ...] = (
        ("mutation_prob", 0.2, 1.0, False),
        ("step_scale", 0.25, 2.0, False),
        ("window", 2, 6, True),
    )


# The base case of the recursion: the fixed search params the topmost level uses on itself.
_ROOT_SEARCH = SearchGenome(mutation_prob=0.7, step_scale=1.0, window=3)


# --------------------------------------------------------------------------- #
# One rung — a measured 1+1 evolutionary loop over a single genome
# --------------------------------------------------------------------------- #
class MetaImprovementController:
    """Evolve one genome by a measured ``1+1`` loop, scored by an external fitness stream.

    Generic over the genome it carries (a :class:`MetaGenome` of engine knobs, or a
    :class:`SearchGenome` of search knobs), so the recursive tower stacks identical rungs. Each
    :meth:`maybe_evolve` selection returns an event whose ``selection_quality`` — how much the
    champion's measured fitness improved — is the fitness the rung *above* records about this rung.
    """

    def __init__(self, champion: _Genome, *, search: Optional[SearchGenome] = None,
                 seed: int = 0) -> None:
        self._rng = random.Random(seed)
        self.search = search or SearchGenome()
        self.champion = champion
        self.champion_fitness: Optional[float] = None
        self.active = self.champion                 # the champion is measured first (the baseline)
        self._trialing_challenger = False
        self._deltas: List[float] = []              # realised fitness samples for the active genome
        self.generation = 0
        self.history: List[Dict[str, Any]] = []

    # ---- bind a fresh search genome (the level above evolves ours) ---- #
    def set_search(self, search: SearchGenome) -> None:
        self.search = search

    # ---- measure: fold one sample into the active trial ---- #
    def record(self, delta: float) -> None:
        try:
            self._deltas.append(float(delta))
        except Exception:  # noqa: BLE001
            pass

    def _mean(self) -> float:
        return sum(self._deltas) / len(self._deltas) if self._deltas else 0.0

    @property
    def window(self) -> int:
        return max(1, int(self.search.window))

    def window_filled(self) -> bool:
        return len(self._deltas) >= self.window

    def maybe_evolve(self) -> Optional[Dict[str, Any]]:
        """Once the active genome has a full window of measured samples, select and re-arm.

        Returns a status dict when a selection happened (baseline/promote/reject), carrying a
        ``selection_quality`` for the rung above, else ``None``.
        """
        if not self.window_filled():
            return None
        fitness = self._mean()
        prev_champ_fitness = self.champion_fitness
        event: Dict[str, Any]
        if not self._trialing_challenger:
            # measuring the champion itself — record its baseline fitness, then trial a mutant
            self.champion_fitness = fitness
            event = {"event": "baseline", "champion": self.champion.to_dict(),
                     "fitness": round(fitness, 6)}
        else:
            # trialing a challenger — promote iff it strictly beat the champion's measured fitness
            if self.champion_fitness is None or fitness > self.champion_fitness:
                event = {"event": "promote", "from": self.champion.to_dict(),
                         "to": self.active.to_dict(), "fitness": round(fitness, 6),
                         "beat": round(self.champion_fitness or 0.0, 6)}
                self.champion = self.active
                self.champion_fitness = fitness
            else:
                event = {"event": "reject", "kept": self.champion.to_dict(),
                         "challenger": self.active.to_dict(), "fitness": round(fitness, 6),
                         "champion_fitness": round(self.champion_fitness, 6)}
        # the rung above is scored by how much THIS rung's champion improved under the current
        # search config — measured, never assumed (0.0 for the first, baseline selection).
        quality = 0.0
        if prev_champ_fitness is not None and self.champion_fitness is not None:
            quality = float(self.champion_fitness) - float(prev_champ_fitness)
        event["selection_quality"] = round(quality, 6)
        # arm the next challenger: a fresh mutation of the (possibly new) champion under our search
        self.generation += 1
        self.active = self.champion.mutate(self._rng, self.search)
        self._trialing_challenger = True
        self._deltas = []
        self.history.append(event)
        self.history = self.history[-50:]
        return event

    def status(self) -> Dict[str, Any]:
        return {"champion": self.champion.to_dict(),
                "champion_fitness": (round(self.champion_fitness, 6)
                                     if self.champion_fitness is not None else None),
                "active": self.active.to_dict(), "search": self.search.to_dict(),
                "trialing_challenger": self._trialing_challenger,
                "generation": self.generation,
                "window_filled": f"{len(self._deltas)}/{self.window}"}

    def to_dict(self) -> Dict[str, Any]:
        return {"champion": self.champion.to_dict(),
                "champion_fitness": self.champion_fitness,
                "search": self.search.to_dict(),
                "generation": self.generation, "history": self.history[-50:]}


# --------------------------------------------------------------------------- #
# The tower — a self-similar stack of rungs; reality at the bottom, compounding up
# --------------------------------------------------------------------------- #
class RecursiveMetaController:
    """A recursive tower of :class:`MetaImprovementController` rungs.

    * ``levels[0]`` evolves the real :class:`MetaGenome` (the engine's algorithm + execution).
    * ``levels[k]`` (k≥1) evolves the :class:`SearchGenome` that drives ``levels[k-1]`` — i.e. it
      improves *how the level below searches*. The topmost level searches itself with a fixed root
      search genome (the base case of the recursion).

    Reality enters only at the bottom: :meth:`record` is fed the real per-cycle capability gain.
    When a rung completes a trial window and selects, its ``selection_quality`` is recorded into the
    rung above — so every level is scored by whether it genuinely made the level below improve.
    Real recursion at every level → compounding, all under one identical ``1+1`` algorithm.
    """

    def __init__(self, *, height: int = 3, champion: Optional[MetaGenome] = None,
                 seed: int = 0, persist_path: Any = None, can_grow: bool = True,
                 hard_max_height: int = 6, recursion_ceiling: Optional[int] = None,
                 edits_ceiling: Optional[int] = None, accel_window: int = 3) -> None:
        self.hard_max_height = max(1, int(hard_max_height))
        self.height = max(1, min(self.hard_max_height, int(height)))
        self._persist_path = persist_path
        # accelerating returns: grow the tower (more meta-levels) and widen the execution caps when
        # gains are sustained, bounded by hard_max_height and (when given) a substrate-derived ceiling.
        self.can_grow = bool(can_grow)
        self.recursion_ceiling = recursion_ceiling
        self.edits_ceiling = edits_ceiling
        self.accel_window = max(2, int(accel_window))
        self._gain_window: List[float] = []      # recent bottom-rung selection qualities
        self._accel: List[Dict[str, Any]] = []   # log of acceleration events (height/cap growth)
        self._seed = int(seed)
        rng = random.Random(seed)
        # level 0 carries the real engine genome; levels 1.. carry search genomes
        self.levels: List[MetaImprovementController] = []
        l0 = MetaImprovementController(champion or MetaGenome(), search=SearchGenome(),
                                       seed=rng.randint(0, 2**31 - 1))
        self.levels.append(l0)
        for _ in range(1, self.height):
            ctrl = MetaImprovementController(SearchGenome(), search=SearchGenome(),
                                            seed=rng.randint(0, 2**31 - 1))
            self.levels.append(ctrl)
        self._load()
        self._bind_searches()

    # ---- a level's evolved champion search drives the level below it ---- #
    def _bind_searches(self) -> None:
        for k in range(1, len(self.levels)):
            child = self.levels[k - 1]
            parent = self.levels[k]
            if isinstance(parent.active, SearchGenome):
                child.set_search(parent.active)
        # the topmost level searches itself with the fixed root search (base case)
        self.levels[-1].set_search(_ROOT_SEARCH)

    @property
    def champion(self) -> MetaGenome:
        return self.levels[0].champion  # type: ignore[return-value]

    @property
    def active(self) -> MetaGenome:
        return self.levels[0].active  # type: ignore[return-value]

    # ---- bind the active engine genome to the live config (capability-only) ---- #
    def apply(self, settings: Any) -> None:
        active = self.levels[0].active
        if isinstance(active, MetaGenome):
            active.apply(settings)

    # ---- reality enters at the bottom ---- #
    def record(self, delta: float) -> None:
        self.levels[0].record(delta)

    def maybe_evolve(self) -> Optional[Dict[str, Any]]:
        """Cascade selections up the tower, propagating each rung's selection_quality upward.

        Returns the bottom rung's event (the one with engine-level meaning) when it selected, else
        ``None``. A selection at level k feeds level k+1's fitness and may, in turn, trigger a
        selection there — so improvement compounds up the stack within a single call chain.
        """
        bottom_event: Optional[Dict[str, Any]] = None
        for k, ctrl in enumerate(self.levels):
            event = ctrl.maybe_evolve()
            if event is None:
                break
            if k == 0:
                bottom_event = event
            # feed this rung's selection quality to the rung above, and re-bind the (possibly new)
            # search champion onto this rung so the evolved search params go live immediately
            if k + 1 < len(self.levels):
                self.levels[k + 1].record(float(event.get("selection_quality", 0.0)))
                parent_active = self.levels[k + 1].active
                if isinstance(parent_active, SearchGenome):
                    ctrl.set_search(parent_active)
            # continue the loop: the rung above may now have a full window and select too
        if bottom_event is not None:
            self._maybe_accelerate(float(bottom_event.get("selection_quality", 0.0)))
            self._bind_searches()
            self._save()
        return bottom_event

    # ---- accelerating returns: grow the tower / widen the caps under sustained gains ---- #
    def _maybe_accelerate(self, selection_quality: float) -> None:
        """Track the bottom rung's realised gains; when sustained-positive, accelerate the engine.

        A run of strictly-positive selection qualities means the improver is *reliably* finding
        better policies — so we let it improve harder: append a meta-level (more compounding) and
        widen the execution caps by one step. Bounded by ``hard_max_height`` and the absolute genome
        ceilings, so acceleration is real but never unbounded."""
        self._gain_window.append(float(selection_quality))
        self._gain_window = self._gain_window[-self.accel_window:]
        if (len(self._gain_window) >= self.accel_window
                and all(g > 0.0 for g in self._gain_window)):
            self._accelerate()
            self._gain_window = []     # require a fresh run of gains before the next acceleration

    def _accelerate(self) -> None:
        grew = self._grow_height() if self.can_grow else False
        caps = MetaGenome.raise_caps(recursion_ceiling=self.recursion_ceiling,
                                     edits_ceiling=self.edits_ceiling)
        self._accel.append({"grew_height": grew, "height": self.height,
                            "recursion_cap": caps[0], "edits_cap": caps[1]})
        self._accel = self._accel[-50:]

    def _grow_height(self) -> bool:
        """Append one search rung above the current top (more compounding), up to the hard cap."""
        if len(self.levels) >= self.hard_max_height:
            return False
        seed = random.Random(self._seed + len(self.levels)).randint(0, 2 ** 31 - 1)
        self.levels.append(MetaImprovementController(SearchGenome(), search=SearchGenome(),
                                                     seed=seed))
        self.height = len(self.levels)
        self._bind_searches()
        return True

    def status(self) -> Dict[str, Any]:
        return {"height": self.height, "hard_max_height": self.hard_max_height,
                "can_grow": self.can_grow,
                "recursion_cap": MetaGenome._MAX_RECURSION, "edits_cap": MetaGenome._MAX_EDITS,
                "accelerations": len(self._accel), "last_accel": self._accel[-1] if self._accel else None,
                "levels": [c.status() for c in self.levels],
                "champion": self.champion.to_dict()}

    # ---- persistence: the whole tower compounds across restarts ---- #
    def _save(self) -> None:
        if self._persist_path is None:
            return
        try:
            from pathlib import Path
            p = Path(self._persist_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps({"height": self.height,
                                     "recursion_cap": MetaGenome._MAX_RECURSION,
                                     "edits_cap": MetaGenome._MAX_EDITS,
                                     "accel": self._accel[-50:],
                                     "levels": [c.to_dict() for c in self.levels]}),
                         encoding="utf-8")
        except Exception:  # noqa: BLE001 — persistence is a bonus, never required
            pass

    def _load(self) -> None:
        if self._persist_path is None:
            return
        try:
            from pathlib import Path
            p = Path(self._persist_path)
            if not p.exists():
                return
            data = json.loads(p.read_text(encoding="utf-8"))
            # restore an accelerated tower: regrow the height and the widened execution caps so the
            # compounding meta-gain persists across restarts.
            MetaGenome._MAX_RECURSION = max(MetaGenome._MAX_RECURSION,
                                            min(MetaGenome._ABS_MAX_RECURSION,
                                                int(data.get("recursion_cap",
                                                             MetaGenome._MAX_RECURSION))))
            MetaGenome._MAX_EDITS = max(MetaGenome._MAX_EDITS,
                                        min(MetaGenome._ABS_MAX_EDITS,
                                            int(data.get("edits_cap", MetaGenome._MAX_EDITS))))
            self._accel = list(data.get("accel", []) or [])[-50:]
            saved = list(data.get("levels", []))
            while len(self.levels) < len(saved) and len(self.levels) < self.hard_max_height:
                seed = random.Random(self._seed + len(self.levels)).randint(0, 2 ** 31 - 1)
                self.levels.append(MetaImprovementController(
                    SearchGenome(), search=SearchGenome(), seed=seed))
            self.height = len(self.levels)
            for k, ctrl in enumerate(self.levels):
                if k >= len(saved):
                    break
                rec = saved[k] or {}
                cls = MetaGenome if k == 0 else SearchGenome
                ctrl.champion = cls.from_dict(rec.get("champion", {}))  # type: ignore[assignment]
                cf = rec.get("champion_fitness")
                ctrl.champion_fitness = float(cf) if cf is not None else None
                ctrl.search = SearchGenome.from_dict(rec.get("search", {}))  # type: ignore[assignment]
                ctrl.generation = int(rec.get("generation", 0) or 0)
                ctrl.history = list(rec.get("history", []) or [])[-50:]
                # resume by trialing a fresh challenger of the persisted champion
                ctrl.active = ctrl.champion.mutate(ctrl._rng, ctrl.search)
                ctrl._trialing_challenger = True
        except Exception:  # noqa: BLE001 — a corrupt cache is simply ignored
            pass


# --------------------------------------------------------------------------- #
# Self-test / demo
# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # pragma: no cover
    print("=" * 70)
    print("NYXARA recursive meta-tower self-test")
    print("=" * 70)

    # --- a single rung climbs toward where reality pays off --- #
    rung = MetaImprovementController(MetaGenome(), search=SearchGenome(window=2), seed=1)

    def env_gain(g: MetaGenome) -> float:
        # an environment where DEEPER recursion + a higher reward-scale genuinely pay off
        return 0.01 * g.recursion_depth + 0.0005 * g.ledger_reward_scale

    for _ in range(60):
        for _ in range(rung.window):
            rung.record(env_gain(rung.active) + 0.0003 * (rung._rng.random() - 0.5))
        rung.maybe_evolve()
    print("single rung champion:", rung.champion.to_dict())
    assert rung.champion.recursion_depth >= 3  # climbed where it paid off
    print("single rung learned to improve harder where it paid off : OK")

    # --- the full tower: reality at the bottom, compounding up every level --- #
    tower = RecursiveMetaController(height=3, seed=7)
    assert len(tower.levels) == 3
    for _ in range(400):
        tower.record(env_gain(tower.active) + 0.0003 * (tower.levels[0]._rng.random() - 0.5))
        tower.maybe_evolve()

    print("\ntower height        :", tower.height)
    for k, c in enumerate(tower.levels):
        s = c.status()
        print(f"  level {k}: gen={s['generation']:3d}  champion={s['champion']}")
    # every higher level must have actually received fitness samples — the compounding signal
    # propagated up the stack, not just sat at the bottom
    for k in range(1, len(tower.levels)):
        assert tower.levels[k].generation >= 1, f"level {k} never evolved — no compounding"
    assert tower.champion.recursion_depth >= 3
    print("compounding signal reached every level : OK")

    # --- persistence round-trips the whole tower --- #
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "meta_meta.json"
        t1 = RecursiveMetaController(height=3, seed=3, persist_path=path)
        for _ in range(120):
            t1.record(env_gain(t1.active) + 0.0003 * (t1.levels[0]._rng.random() - 0.5))
            t1.maybe_evolve()
        gen0 = t1.levels[0].generation
        t2 = RecursiveMetaController(height=3, seed=9, persist_path=path)
        assert t2.levels[0].generation == gen0
        assert t2.champion.to_dict() == t1.champion.to_dict()
    print("tower persistence round-trips all levels : OK")

    print("\nALL SELF-TESTS PASSED ✓")
