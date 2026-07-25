"""NYXARA · causal/engine.py — the CAUSAL / OMNISCIENCE core coordinator.

One object that wires the Master's eight asks into a single subsystem NYXARA runs herself. It owns
no new cleverness of its own — it is a *composer* (like :class:`~nyxara.growth.self_evolving.
SelfEvolvingArchitect`): it holds the eight engines and sequences them.

Per conversational turn (the cheap hot path, ``O(bank)`` diagnose, no LLM):

    1. :meth:`resonate` — field-interference retrieval of the concepts most relevant to the query;
    2. :meth:`verify_knots` — a post-reasoning gate: the answer's causal claims must tie into the
       lattice or a **Knot Mutation Failure** flags a probable hallucination;
    3. :meth:`phase_shift_if_gap` — on a resonance gap, re-synthesise a new field + knot live (gated).

Continuously, off the hot path:

    * :meth:`beat` — the thermodynamic monitor reads surprise every heartbeat and pre-emptively heals;
    * :meth:`discover_axioms`, :meth:`hot_swap`, :meth:`compress_memory`,
      :meth:`simulate_and_collapse` — the OMNISCIENCE maintenance/decision capabilities, run as
      governed passes (meta-commands or the autonomic loop), never inline on a user's turn.

Everything degrades gracefully: with no ``core`` and default settings the engine is fully usable and
testable on its own.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from nyxara.causal.axiom_synth import Axiom, AxiomSynthesizer
from nyxara.causal.causal_knots import KnotCheck, KnotLattice
from nyxara.causal.field_resonance import ResonanceField, ResonanceHit
from nyxara.causal.hypergraph_compress import CompressionReport, HyperGraphCompressor
from nyxara.causal.phase_shift import PhaseShift, PhaseShiftMutator
from nyxara.causal.polymorph_runtime import PolymorphRuntime, SwapOutcome
from nyxara.causal.self_simulation import SelfSimulationRace, SimulationVerdict
from nyxara.causal.thermo_inference import ThermodynamicMonitor, ThermoReading

__all__ = ["EngineTurn", "CausalEngine"]


# --------------------------------------------------------------------------- #
# Per-turn record
# --------------------------------------------------------------------------- #
@dataclass
class EngineTurn:
    """What the engine contributed to one turn."""

    query: str
    resonance: List[Dict[str, Any]] = field(default_factory=list)
    knot_check: Optional[Dict[str, Any]] = None
    phase_shift: Optional[Dict[str, Any]] = None
    gap: bool = False
    consistent: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {"query": self.query[:120], "resonance": self.resonance,
                "knot_check": self.knot_check, "phase_shift": self.phase_shift,
                "gap": self.gap, "consistent": self.consistent}


# --------------------------------------------------------------------------- #
# Coordinator
# --------------------------------------------------------------------------- #
class CausalEngine:
    """The OMNISCIENCE core: eight engines, one governed subsystem."""

    def __init__(self, *, core: Any = None, settings: Any = None) -> None:
        self.core = core
        s = settings
        self.field = ResonanceField(
            dim=_get(s, "field_dim", 512),
            capacity=_get(s, "field_capacity", 4096),
            resonance_floor=_get(s, "resonance_floor", 0.15))
        self.lattice = KnotLattice()
        self.thermo = ThermodynamicMonitor(
            window=_get(s, "thermo_window", 64),
            spike_z=_get(s, "spike_z", 2.5),
            heal=self._default_heal)
        self.phase = PhaseShiftMutator(
            core=core, max_new_per_min=_get(s, "max_phase_shifts_per_min", 30))
        self.axioms = AxiomSynthesizer()
        self.runtime = PolymorphRuntime()
        self.compressor = HyperGraphCompressor()
        self.simulator = SelfSimulationRace(
            max_branches=_get(s, "max_sim_branches", 16),
            max_workers=_get(s, "max_sim_workers", 8))
        self.resonate_top = _get(s, "resonate_top", 3)
        self.knot_gate_abstains = _get(s, "knot_gate_abstains", False)
        self._beat_thermo = _get(s, "beat_thermo", True)
        self._turns = 0

    # -- hot path ----------------------------------------------------------- #
    def resonate(self, query: str, *, top: int = 3) -> List[ResonanceHit]:
        """Field-interference retrieval of the most relevant concepts."""
        return self.field.resonate(query, top=top)

    def imprint(self, label: str, text: Optional[str] = None, **kw: Any) -> None:
        """Teach the field a concept (so future queries can resonate with it)."""
        self.field.imprint(label, text, **kw)

    def verify_knots(self, claims: Sequence[Tuple[str, str, Any]]) -> KnotCheck:
        """Post-reasoning hallucination gate over the answer's causal claims."""
        return self.lattice.check(claims)

    def commit_knots(self, claims: Sequence[Tuple[str, str, Any]]) -> KnotCheck:
        """Verify and, only if consistent, tie the claims into the live lattice."""
        return self.lattice.commit(claims)

    def phase_shift_if_gap(self, query: str, *,
                           reason: str = "no resonance for this query") -> Optional[PhaseShift]:
        """If nothing resonates, re-synthesise new structure live (gated)."""
        if not self.field.is_gap(query):
            return None
        return self.phase.shift(query, self.field, self.lattice, reason=reason)

    def turn(self, query: str, *,
             claims: Optional[Sequence[Tuple[str, str, Any]]] = None,
             top: int = 3) -> EngineTurn:
        """Run the whole cheap per-turn flow and return what the engine contributed."""
        self._turns += 1
        hits = self.resonate(query, top=top)
        et = EngineTurn(query=query, resonance=[h.to_dict() for h in hits])
        if claims:
            check = self.verify_knots(claims)
            et.knot_check = check.to_dict()
            et.consistent = check.consistent
        if not hits:
            et.gap = True
            shift = self.phase_shift_if_gap(query, reason="resonance gap on this turn")
            et.phase_shift = None if shift is None else shift.to_dict()
        return et

    # -- continuous (heartbeat) -------------------------------------------- #
    def beat(self, core: Any = None, *, signal: Any = None) -> ThermoReading:
        """One thermodynamic beat — read surprise, pre-emptively heal on a spike."""
        return self.thermo.beat(core if core is not None else self.core, signal=signal)

    def _default_heal(self, reading: ThermoReading) -> bool:
        """Pre-emptive heal: nudge the field toward stability and ask the core to self-repair."""
        healed = False
        core = self.core
        if core is not None:
            for path, method in (("health", "check"), ("self_repair", "repair"),
                                 ("self_repair", "run"), ("guardian", "stabilize")):
                obj = getattr(core, path, None)
                fn = getattr(obj, method, None) if obj is not None else None
                if callable(fn):
                    try:
                        fn()
                        healed = True
                        break
                    except Exception:  # noqa: BLE001 — healing is best-effort
                        continue
        return healed

    # -- OMNISCIENCE maintenance / decisions (governed, off hot path) ------- #
    def discover_axioms(self, name: str, samples: Sequence[Tuple[Dict[str, int], int]],
                        variables: Sequence[str], **kw: Any) -> Optional[Axiom]:
        """Discover + prove a new axiom; adopted only if a prover certifies it."""
        return self.axioms.synthesize(name, samples, variables, **kw)

    def hot_swap(self, key: str, replacement: Any = None, **kw: Any) -> SwapOutcome:
        """Zero-downtime hot-swap of a registered live component (rolls back on failure)."""
        return self.runtime.hot_swap(key, replacement, **kw)

    def register_component(self, key: str, target: Any) -> Any:
        """Register a live component so it can later be hot-swapped."""
        return self.runtime.register(key, target)

    def compress_memory(self, triples: Iterable[Tuple[str, str, str]],
                        history: Optional[Sequence[str]] = None
                        ) -> Tuple[Dict[str, Any], CompressionReport]:
        """Fold a knowledge graph + history into a dense lossless hyper-graph."""
        return self.compressor.compress(triples, history)

    def simulate_and_collapse(self, hypotheses: Sequence[Any],
                              rollout: Callable[[Any], Tuple[Any, float]], *,
                              verify: Optional[Callable[[Any], float]] = None
                              ) -> SimulationVerdict:
        """Race bounded parallel rollouts and collapse to the best-verified / least-surprising."""
        return self.simulator.race(hypotheses, rollout, verify=verify)

    # -- introspection ------------------------------------------------------ #
    def status(self) -> Dict[str, Any]:
        return {
            "turns": self._turns,
            "field": self.field.status(),
            "lattice": self.lattice.status(),
            "thermo": self.thermo.status(),
            "phase_shift": self.phase.status(),
            "axioms": self.axioms.status(),
            "runtime": self.runtime.status(),
            "simulator": self.simulator.status(),
        }


def _get(settings: Any, name: str, default: Any) -> Any:
    """Read a setting by attribute name, falling back to a default (works with None)."""
    if settings is None:
        return default
    return getattr(settings, name, default)
