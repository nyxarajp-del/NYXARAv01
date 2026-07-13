"""NYXARA · growth/adaptation.py — structurally re-organize herself in a brand-new environment (🜁).

The critique this module answers: *"Other minds, dropped into a wholly new environment, re-organize
themselves structurally. NYXARA's adaptation is bounded — she models a new system, saves a profile,
and carries on, but the shape of her own brain never changes."* This closes the loop: when NYXARA is
put in front of an unfamiliar environment she does the whole thing **herself**, deterministically, with
no language model anywhere in the path:

    SURVEY   →   MODEL (own faculty)   →   PRESSURE?   →   RE-ORGANIZE (grow her brain)   →   REMEMBER
    (enumerate    (open_world cracks       (how much of      (topology grows width/depth,        (registry
     the boxes)    each unknown box)        it defeated her)  gauntlet-gated, to the real HW cap) persists it)

Concretely, :meth:`EnvironmentAdapter.adapt`:

1. **Surveys** the environment — a collection of black boxes she can only poke (callables, objects
   with ``.interact``, or declarative ``{family, params}`` specs that :func:`nyxara.growth.open_world.
   build_system` turns into boxes).
2. **Models** each unknown system from first principles with her OWN
   :class:`~nyxara.growth.open_world.OpenWorldGeneralizer` — recognising ones she has cracked before
   (via the persistent :class:`~nyxara.growth.env_registry.EnvironmentRegistry`) and inducing laws for
   the rest. No LLM: the reasoning is a deterministic symbolic law search.
3. **Measures the pressure** the environment put her under — the fraction of its systems she could not
   model or modelled only weakly — and, if that clears the threshold, **re-organizes her own brain**
   through :class:`~nyxara.growth.topology.DynamicTopology`: widening/deepening her genesis net up to
   the *real hardware ceiling* (:func:`nyxara.growth.topology.hardware_ceiling`), function-preservingly
   and still gated by the exact same Foundry gauntlet as any other model — never a safety bypass.
4. **Remembers** every cracked system so the same environment is recognised instantly next time.

It **composes** the existing faculties; it re-implements none of them. Everything degrades gracefully:
with no topology engine it models-and-remembers only; with no registry it models fresh; with nothing
knowable it returns an honest report saying so. Pure standard library at this layer.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from nyxara.growth.open_world import (DomainSpec, OpenWorldGeneralizer, Verdict, build_system)

__all__ = ["EnvironmentAdapter", "AdaptationReport", "SystemOutcome"]


# --------------------------------------------------------------------------- #
# Records
# --------------------------------------------------------------------------- #
@dataclass
class SystemOutcome:
    """What became of one system in the environment: modelled, recognised, or defeated her."""
    label: str
    verdict: str
    law: Optional[str] = None
    law_family: Optional[str] = None
    confidence: float = 0.0
    recognized: bool = False
    probes_used: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {"label": self.label, "verdict": self.verdict, "law": self.law,
                "law_family": self.law_family, "confidence": round(float(self.confidence), 4),
                "recognized": self.recognized, "probes_used": self.probes_used}


@dataclass
class AdaptationReport:
    """An auditable record of one adaptation to an environment."""
    environment: str = "unknown-environment"
    n_systems: int = 0
    modelled: int = 0
    recognized: int = 0
    unmodelled: int = 0
    pressure: float = 0.0
    grew: bool = False
    growth: Optional[Dict[str, Any]] = None
    ceiling: Optional[Dict[str, int]] = None
    outcomes: List[SystemOutcome] = field(default_factory=list)
    seconds: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {"environment": self.environment, "n_systems": self.n_systems,
                "modelled": self.modelled, "recognized": self.recognized,
                "unmodelled": self.unmodelled, "pressure": round(float(self.pressure), 4),
                "grew": self.grew, "growth": self.growth, "ceiling": self.ceiling,
                "outcomes": [o.to_dict() for o in self.outcomes],
                "seconds": round(self.seconds, 4)}


# --------------------------------------------------------------------------- #
# The adapter
# --------------------------------------------------------------------------- #
class EnvironmentAdapter:
    """Model an unfamiliar environment and, under pressure, re-organize her own brain to meet it.

    Parameters
    ----------
    open_world :
        The :class:`~nyxara.growth.open_world.OpenWorldGeneralizer` used to model each system. One is
        created on demand (wired to ``registry`` when given).
    topology :
        The :class:`~nyxara.growth.topology.DynamicTopology` engine used to grow her brain under
        pressure. Optional — without it she models-and-remembers only.
    registry :
        The persistent :class:`~nyxara.growth.env_registry.EnvironmentRegistry`. Optional.
    growth_source :
        The brain to grow from (a genome / model), or a zero-arg callable returning it. Resolved
        lazily at adapt time so a champion forged later is picked up.
    cfg :
        An ``EnvironmentAdaptationConfig`` (probe budget, pressure threshold, toggles). Defaults are
        used when omitted.
    """

    def __init__(self, *, open_world: Any = None, topology: Any = None, registry: Any = None,
                 growth_source: Any = None, cfg: Any = None, settings: Any = None,
                 seed: int = 0) -> None:
        self.cfg = cfg or getattr(settings, "environment_adaptation", None)
        if self.cfg is None:
            from nyxara.kernel.config import EnvironmentAdaptationConfig
            self.cfg = EnvironmentAdaptationConfig()
        self.registry = registry
        self.open_world = open_world or OpenWorldGeneralizer(registry=registry, seed=seed)
        self.topology = topology
        self.growth_source = growth_source
        self.seed = int(seed)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def adapt(self, environment: Any, *, budget: Optional[int] = None,
              label: str = "unknown-environment") -> AdaptationReport:
        """Adapt to ``environment`` — model its systems, and re-organize her brain if it strains her.

        ``environment`` may be a mapping ``{name: system}``, a list of systems, a list of
        ``(name, system[, DomainSpec])`` tuples, or declarative ``{"family": ..., "params": ...}``
        specs (turned into boxes via :func:`~nyxara.growth.open_world.build_system`). Always returns
        an :class:`AdaptationReport`; never raises on a single bad system."""
        t0 = time.monotonic()
        budget = int(budget if budget is not None else getattr(self.cfg, "probe_budget", 48))
        systems = self._normalize(environment)
        report = AdaptationReport(environment=label, n_systems=len(systems))
        if not systems:
            report.seconds = time.monotonic() - t0
            return report

        weak = 0
        for name, system, domain in systems:
            outcome = self._model_one(name, system, domain, budget)
            report.outcomes.append(outcome)
            if outcome.recognized:
                report.recognized += 1
                report.modelled += 1
            elif outcome.verdict == Verdict.MODELLED.value:
                report.modelled += 1
            else:
                report.unmodelled += 1
                weak += 1

        report.pressure = weak / max(1, len(systems))
        # Under real pressure, restructure her own brain (gauntlet-gated, hardware-bounded).
        if (bool(getattr(self.cfg, "grow_under_pressure", True))
                and self.topology is not None
                and report.pressure >= float(getattr(self.cfg, "pressure_threshold", 0.5))):
            report.growth, report.grew = self._grow(report.pressure)
        report.ceiling = self._ceiling()
        report.seconds = time.monotonic() - t0
        return report

    # ------------------------------------------------------------------ #
    # Model one system with her own generalizer
    # ------------------------------------------------------------------ #
    def _model_one(self, name: str, system: Any, domain: Optional[DomainSpec],
                   budget: int) -> SystemOutcome:
        try:
            rep = self.open_world.understand(system, domain=domain, budget=budget, label=name)
        except Exception:  # noqa: BLE001 — a broken box is an honest 'unmodelled', not fatal
            return SystemOutcome(label=name, verdict=Verdict.UNMODELLED.value,
                                 law=None, law_family=None, confidence=0.0,
                                 recognized=False, probes_used=0)
        return SystemOutcome(label=rep.system, verdict=rep.verdict.value, law=rep.law,
                             law_family=rep.law_family, confidence=rep.confidence,
                             recognized=bool(rep.recognized), probes_used=rep.probes_used)

    # ------------------------------------------------------------------ #
    # Re-organize the brain under environmental pressure
    # ------------------------------------------------------------------ #
    def _grow(self, pressure: float) -> Tuple[Optional[Dict[str, Any]], bool]:
        source = self._resolve_source()
        if source is None:
            return {"grew": False, "reason": "no brain to grow from yet"}, False
        try:
            from nyxara.growth.topology import CapacitySignal
            signal = CapacitySignal(problem_difficulty=pressure, saturation=pressure,
                                    loss_plateau=pressure,
                                    note="environmental pressure: systems left unmodelled")
            result = self.topology.maybe_grow(signal, source=source)
        except Exception as exc:  # noqa: BLE001 — growth is a capability, never required
            return {"grew": False, "reason": f"growth failed: {exc}"}, False
        if not result:
            return {"grew": False, "reason": "no capacity pressure past the monitor / at ceiling"}, \
                False
        return result, bool(result.get("grew") or result.get("promoted"))

    def _resolve_source(self) -> Any:
        src = self.growth_source
        if callable(src):
            try:
                return src()
            except Exception:  # noqa: BLE001
                return None
        return src

    def _ceiling(self) -> Optional[Dict[str, int]]:
        if self.topology is None:
            return None
        try:
            return {"max_n_embd": int(self.topology.max_n_embd),
                    "max_layers": int(self.topology.max_layers)}
        except Exception:  # noqa: BLE001
            return None

    # ------------------------------------------------------------------ #
    # Normalise many environment shapes into (label, system, domain) triples
    # ------------------------------------------------------------------ #
    def _normalize(self, environment: Any) -> List[Tuple[str, Any, Optional[DomainSpec]]]:
        out: List[Tuple[str, Any, Optional[DomainSpec]]] = []
        items: List[Tuple[str, Any]] = []
        if isinstance(environment, dict):
            items = [(str(k), v) for k, v in environment.items()]
        elif isinstance(environment, (list, tuple)):
            for i, item in enumerate(environment):
                if isinstance(item, (list, tuple)) and len(item) >= 2 and isinstance(item[0], str):
                    items.append((item[0], item[1:] if len(item) > 2 else item[1]))
                else:
                    items.append((f"system-{i}", item))
        else:
            items = [("system-0", environment)]

        for name, spec in items:
            system, domain = self._as_box(spec)
            if system is not None:
                out.append((name, system, domain))
        return out

    @staticmethod
    def _as_box(spec: Any) -> Tuple[Any, Optional[DomainSpec]]:
        """Resolve one item to ``(system, domain)`` — a callable/interactor, a (system, domain) pair,
        or a declarative ``{family, params, ...}`` spec built via ``build_system``."""
        # (system, DomainSpec) pair carried through the list form
        domain: Optional[DomainSpec] = None
        if isinstance(spec, (list, tuple)) and spec:
            if len(spec) >= 2 and isinstance(spec[1], DomainSpec):
                return spec[0], spec[1]
            spec = spec[0]
        if callable(spec) or hasattr(spec, "interact"):
            return spec, domain
        if isinstance(spec, dict) and "family" in spec:
            system, dspec = build_system(
                spec["family"], spec.get("params", {}),
                dims=int(spec.get("dims", 1)), kind=str(spec.get("kind", "real")),
                low=float(spec.get("low", -6.0)), high=float(spec.get("high", 6.0)),
                scalar=spec.get("scalar"))
            return system, dspec
        return None, domain


# --------------------------------------------------------------------------- #
# Self-test / demo
# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # pragma: no cover
    import math
    import tempfile
    from pathlib import Path

    print("=" * 70)
    print("NYXARA Environment Adaptation self-test")
    print("=" * 70)

    from nyxara.growth.env_registry import EnvironmentRegistry

    with tempfile.TemporaryDirectory() as tmp:
        reg = EnvironmentRegistry(path=Path(tmp) / "env.json", min_checks=3)
        adapter = EnvironmentAdapter(registry=reg, seed=7)

        # an environment of several unknown boxes — some cleanly modellable, one hostile
        rng = __import__("random").Random(0)
        env = {
            "linear": lambda x: 3 * x - 2,
            "quadratic": lambda x: 2 * x * x + x - 5,
            "sine": lambda x: 3.0 * math.sin(1.0 * x) + 2.0,
            "noise": lambda x: rng.random() * 100,          # she should NOT bluff this
        }
        rep = adapter.adapt(env, label="test-env")
        d = rep.to_dict()
        print(f"\n  systems          : {d['n_systems']}  modelled={d['modelled']} "
              f"recognized={d['recognized']} unmodelled={d['unmodelled']}")
        for o in d["outcomes"]:
            print(f"    - {o['label']:10s} {o['verdict']:10s} {o['law']}")
        assert d["modelled"] >= 3 and d["unmodelled"] >= 1
        assert d["pressure"] > 0.0

        # re-adapting the SAME environment recognises the ones already cracked (no re-probing)
        rep2 = adapter.adapt(env, label="test-env")
        assert rep2.recognized >= 3
        print(f"\n  re-entry         : recognised {rep2.recognized} systems instantly ✓")

        # declarative environment (systems that crossed an 'API boundary' as specs)
        rep3 = adapter.adapt([{"family": "affine", "params": {"w": [1.0, 2.0]}, "dims": 1}],
                             label="declared")
        assert rep3.modelled == 1
        print(f"  declarative      : modelled {rep3.modelled} spec-declared system ✓")

    print("\nALL SELF-TESTS PASSED ✓")
