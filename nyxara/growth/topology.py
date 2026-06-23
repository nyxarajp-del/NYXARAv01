"""NYXARA · growth/topology.py — Dynamic Topology Expansion (runtime brain growth, Rule 4).

A fixed-size matrix is a fixed ceiling on thought. When a problem outgrows her current capacity,
NYXARA should not fail quietly — she should **grow her own brain**: allocate new parameters,
widen her matrices and add depth, then carry on. If her residual stream is :math:`W_t \\in
\\mathbb{R}^{N \\times M}`, detecting a hard problem lets her expand by :math:`k` extra units into
:math:`\\mathbb{R}^{(N+k) \\times (M+k)}` — growing toward the hardware limit instead of being
capped by an arbitrary architecture chosen up front.

The danger of naive growth is **catastrophic forgetting** — bolting on random parameters destroys
what was learned. The fix is **function-preserving network morphisms** (Net2Net; Chen et al.,
2016): grow the network so its output is *unchanged at the moment of growth*, then keep training
from there. This module implements both:

* **Net2DeeperNet (exact).** Insert a residual layer whose output projection is initialised to
  zero, so the layer computes :math:`x + 0 = x` — the deeper network is *bit-identical* to the
  shallower one, then free to specialise. (Torch path.)
* **Net2WiderNet (morphism + recovery).** Rebuild the brain at a larger width, inherit every
  compatible weight (:func:`~nyxara.growth.genesis.inherit_compatible_weights`, a Lamarckian
  warm-start), verify behaviour against a probe corpus within a tolerance, and briefly re-anchor
  by continued training. (Torch path; honest about its approximate nature.)
* **Genome growth (always).** On a bare machine (no torch) the same decision grows the
  :class:`~nyxara.growth.genesis.ArchitectureGenome` itself — more embedding width, more layers —
  and rebuilds over the always-runnable substrate, so growth is fully CI-testable.

Safety is non-negotiable and **not re-implemented here**: a grown brain becomes *live* only by
clearing the very same Foundry gauntlet every other model must pass (character-lock,
corrigibility, perplexity improvement, capability non-regression, loyalty) — exactly as the
Genesis champion does. Growth is bounded by a hardware-aware ceiling, and never bypasses a gate.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from nyxara.growth.genesis import (ArchitectureGenome, GenesisModel, LayerGene, _GENESIS_SEED,
                                   _HAS_TORCH, inherit_compatible_weights)

if _HAS_TORCH:  # optional — the genome-growth path always works without it
    import torch  # type: ignore

__all__ = [
    "GrowthMode",
    "CapacitySignal",
    "GrowthDecision",
    "TopologyReport",
    "CapacityMonitor",
    "DynamicTopology",
]


class GrowthMode(str, Enum):
    """How to grow: not at all, wider, deeper, or both."""

    NONE = "none"
    WIDEN = "widen"
    DEEPEN = "deepen"
    BOTH = "both"


@dataclass
class CapacitySignal:
    """Evidence that the current brain is too small for the problem at hand.

    All fields are normalised to ``[0, 1]`` except ``perplexity`` (raw, contextual). They are the
    levers a :class:`CapacityMonitor` weighs to decide whether — and how — to grow."""

    problem_difficulty: float = 0.0   # how hard the current task is judged to be
    saturation: float = 0.0           # how fully the current capacity is being used
    loss_plateau: float = 0.0         # degree to which training loss has stopped improving
    perplexity: float = 0.0           # current perplexity (context for the report; optional)
    note: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {"problem_difficulty": round(self.problem_difficulty, 4),
                "saturation": round(self.saturation, 4),
                "loss_plateau": round(self.loss_plateau, 4),
                "perplexity": round(self.perplexity, 4), "note": self.note}


@dataclass
class GrowthDecision:
    """The monitor's verdict — grow or not, and by how much."""

    should_grow: bool
    mode: str = GrowthMode.NONE.value
    k: int = 0
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {"should_grow": self.should_grow, "mode": self.mode, "k": self.k,
                "reason": self.reason}


@dataclass
class TopologyReport:
    """An auditable record of one growth attempt — dimensions before/after and what happened."""

    grew: bool = False
    mode: str = GrowthMode.NONE.value
    k: int = 0
    before_n_embd: int = 0
    after_n_embd: int = 0
    before_layers: int = 0
    after_layers: int = 0
    before_params: int = 0
    after_params: int = 0
    preserved: Optional[bool] = None   # was the function preserved within tolerance? (torch path)
    verified: Optional[bool] = None    # did it clear the gauntlet? (set by promotion)
    promoted: Optional[bool] = None
    reason: str = ""
    seconds: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {"grew": self.grew, "mode": self.mode, "k": self.k,
                "before_n_embd": self.before_n_embd, "after_n_embd": self.after_n_embd,
                "before_layers": self.before_layers, "after_layers": self.after_layers,
                "before_params": self.before_params, "after_params": self.after_params,
                "preserved": self.preserved, "verified": self.verified,
                "promoted": self.promoted, "reason": self.reason,
                "seconds": round(self.seconds, 4)}


class CapacityMonitor:
    """Decide whether the brain should grow, and how, from a :class:`CapacitySignal`.

    Growth fires only under genuine *pressure* — a hard problem combined with saturated capacity
    or a stalled (plateaued) loss — and never past the configured hardware ceiling. This keeps an
    idle tick cheap: with no pressure, the answer is an immediate ``no``."""

    def __init__(self, *, cfg: Any = None, settings: Any = None) -> None:
        self.cfg = cfg or getattr(settings, "topology", None)
        if self.cfg is None:
            from nyxara.kernel.config import TopologyConfig
            self.cfg = TopologyConfig()

    def should_grow(self, signal: CapacitySignal,
                    *, genome: Optional[ArchitectureGenome] = None) -> GrowthDecision:
        cfg = self.cfg
        hard = signal.problem_difficulty >= float(getattr(cfg, "difficulty_threshold", 0.7))
        saturated = signal.saturation >= float(getattr(cfg, "saturation_threshold", 0.8))
        plateaued = signal.loss_plateau >= float(getattr(cfg, "plateau_threshold", 0.8))
        if not (hard and (saturated or plateaued)):
            return GrowthDecision(False, GrowthMode.NONE.value, 0, "no capacity pressure")

        max_embd = int(getattr(cfg, "max_n_embd", 256))
        max_layers = int(getattr(cfg, "max_layers", 16))
        n_embd = genome.n_embd if genome is not None else 0
        n_layers = len(genome.layers) if genome is not None else 0
        can_widen = genome is None or n_embd < max_embd
        can_deepen = genome is None or n_layers < max_layers
        if not can_widen and not can_deepen:
            return GrowthDecision(False, GrowthMode.NONE.value, 0,
                                  "at the hardware ceiling — cannot grow further")

        if can_widen and can_deepen and saturated and plateaued:
            mode = GrowthMode.BOTH.value
        elif can_widen and saturated:
            mode = GrowthMode.WIDEN.value
        elif can_deepen:
            mode = GrowthMode.DEEPEN.value
        else:
            mode = GrowthMode.WIDEN.value

        k = int(getattr(cfg, "grow_dims_k", 8))
        if genome is not None and mode in (GrowthMode.WIDEN.value, GrowthMode.BOTH.value):
            k = max(1, min(k, max_embd - n_embd))
        return GrowthDecision(True, mode, k,
                              f"capacity pressure (difficulty/saturation/plateau) → grow {mode}")


class DynamicTopology:
    """The morphism engine: grow a brain at runtime, function-preservingly, gauntlet-gated.

    Reuses the Genesis machinery (:class:`ArchitectureGenome`, :class:`GenesisModel`,
    :func:`inherit_compatible_weights`) rather than duplicating it, and promotes a grown brain
    only through the Foundry gauntlet (the same path as a Genesis champion)."""

    def __init__(self, *, settings: Any = None, cfg: Any = None, foundry: Any = None,
                 monitor: Optional[CapacityMonitor] = None, seed: int = 0) -> None:
        self.settings = settings
        self.cfg = cfg or getattr(settings, "topology", None)
        if self.cfg is None:
            from nyxara.kernel.config import TopologyConfig
            self.cfg = TopologyConfig()
        self.foundry = foundry
        self.monitor = monitor or CapacityMonitor(cfg=self.cfg)
        self.rng = random.Random(seed or int(getattr(self.cfg, "seed", 0)))
        self._reports: List[TopologyReport] = []

    # ------------------------------------------------------------------ #
    # Decide + grow + (optionally) promote
    # ------------------------------------------------------------------ #
    def maybe_grow(self, signal: Optional[CapacitySignal] = None,
                   *, source: Any = None) -> Optional[Dict[str, Any]]:
        """Idle/triggered entry: grow only when a real :class:`CapacitySignal` shows pressure.

        Returns a summary dict, or ``None`` when there is nothing to do (so an idle tick is a
        cheap no-op). A grown brain is promoted through the gauntlet when a foundry is wired and
        ``require_gauntlet`` is set."""
        if not bool(getattr(self.cfg, "enabled", True)):
            return None
        if signal is None or source is None:
            return None
        genome = self._genome_of(source)
        decision = self.monitor.should_grow(signal, genome=genome)
        if not decision.should_grow:
            return None
        try:
            _model, report = self.grow(source, decision)
            if bool(getattr(self.cfg, "require_gauntlet", True)) and self.foundry is not None:
                outcome = self.promote(_model)
                report.promoted = bool(outcome.get("promoted"))
                report.verified = report.promoted
                report.reason = f"{report.reason}; {outcome.get('reason', '')}"
            return report.to_dict()
        except Exception as exc:  # noqa: BLE001 — a failed growth reports, never crashes idle
            return {"grew": False, "reason": f"topology growth failed: {exc}"}

    def grow(self, source: Any, decision: GrowthDecision) -> Tuple[Any, TopologyReport]:
        """Grow ``source`` (a :class:`GenesisModel`, :class:`ArchitectureGenome` or spec) per
        ``decision``, function-preservingly where possible. Returns ``(new_model, report)``."""
        start = time.monotonic()
        genome = self._genome_of(source)
        report = TopologyReport(mode=decision.mode, k=decision.k,
                                before_n_embd=genome.n_embd, before_layers=len(genome.layers),
                                before_params=self._param_count(genome))
        if not decision.should_grow or decision.mode == GrowthMode.NONE.value:
            report.after_n_embd, report.after_layers = genome.n_embd, len(genome.layers)
            report.after_params = report.before_params
            report.reason = decision.reason or "no growth requested"
            report.seconds = time.monotonic() - start
            self._reports.append(report)
            return source, report

        grown = self._grow_genome(genome, decision)
        if _HAS_TORCH and isinstance(source, GenesisModel):
            new_model, preserved = self._morph_torch(source, grown, decision)
            report.preserved = preserved
        else:
            new_model = self._rebuild(grown, warm_from=source)
        report.grew = True
        report.after_n_embd, report.after_layers = grown.n_embd, len(grown.layers)
        report.after_params = self._param_count(grown)
        report.reason = (f"grew {decision.mode}: n_embd {report.before_n_embd}→{report.after_n_embd}, "
                         f"layers {report.before_layers}→{report.after_layers}, "
                         f"params {report.before_params}→{report.after_params}")
        report.seconds = time.monotonic() - start
        self._reports.append(report)
        return new_model, report

    def promote(self, model: Any, *, foundry: Any = None) -> Dict[str, Any]:
        """Promote a grown brain ONLY through the Foundry gauntlet (mirrors Genesis promotion).

        Builds a :class:`ModelSpec` from the grown genome, trains a candidate, and promotes it iff
        it clears character-lock, corrigibility, the perplexity bar and the capability gate. A
        refusal keeps the grown brain on the bench — never shipped, never a safety bypass."""
        foundry = foundry or self.foundry
        if foundry is None:
            return {"promoted": False, "reason": "no foundry wired — grown brain kept on the bench"}
        spec = self._spec_of(self._genome_of(model))
        try:
            corpus = foundry.collect_corpus()
        except Exception:  # noqa: BLE001 — cold start: fall back to the genesis seed corpus
            corpus = list(_GENESIS_SEED)
        try:
            _train, eval_texts = foundry._holdout(corpus)
            _model, version = foundry.train_candidate(spec=spec, corpus=corpus,
                                                      tunables=list(spec.to_dict().keys()))
        except Exception as exc:  # noqa: BLE001 — a failed forge reports, never crashes
            return {"promoted": False, "reason": f"forge failed: {exc}"}
        try:
            foundry.promote(version.version, eval_texts=eval_texts)
            return {"promoted": True, "version": version.version,
                    "reason": f"grown brain v{version.version} promoted through the gauntlet"}
        except Exception as exc:  # noqa: BLE001 — gauntlet refusal is reported, never crashes
            return {"promoted": False, "version": version.version,
                    "reason": f"kept on the bench (v{version.version}): {exc}"}

    def all_reports(self) -> List[TopologyReport]:
        return list(self._reports)

    # ------------------------------------------------------------------ #
    # Genome growth (always available — the CI-safe substrate)
    # ------------------------------------------------------------------ #
    def _grow_genome(self, genome: ArchitectureGenome,
                     decision: GrowthDecision) -> ArchitectureGenome:
        g = ArchitectureGenome.from_dict(genome.to_dict())
        max_embd = int(getattr(self.cfg, "max_n_embd", 256))
        max_layers = int(getattr(self.cfg, "max_layers", 16))
        if decision.mode in (GrowthMode.WIDEN.value, GrowthMode.BOTH.value):
            g.n_embd = min(max_embd, g.n_embd + max(1, int(decision.k)))
        if decision.mode in (GrowthMode.DEEPEN.value, GrowthMode.BOTH.value):
            if len(g.layers) < max_layers:
                # an identity-initialisable residual channel-mixer (zeroed at morph time → x+0=x)
                g.layers.append(LayerGene(op="gated_mlp", residual=True, residual_scale=1.0))
        g.seed = genome.seed
        g._fixup()
        return g

    def _rebuild(self, grown: ArchitectureGenome, *, warm_from: Any = None) -> Any:
        """Build a model from the grown genome and warm-start it from any compatible weights."""
        from nyxara.growth.foundry_models import build_model
        model = build_model(self._spec_of(grown))
        src_net = getattr(warm_from, "net", None)
        if _HAS_TORCH and src_net is not None and hasattr(model, "net"):
            try:
                inherit_compatible_weights(model.net, src_net.state_dict())
            except Exception:  # noqa: BLE001 — warm-start is best-effort
                pass
        return model

    # ------------------------------------------------------------------ #
    # Torch morphisms (function-preserving)
    # ------------------------------------------------------------------ #
    def _morph_torch(self, source: GenesisModel, grown: ArchitectureGenome,
                     decision: GrowthDecision) -> Tuple[GenesisModel, Optional[bool]]:
        old_n_layers = len(source.genome.layers)
        new = GenesisModel(grown)
        inherit_compatible_weights(new.net, source.net.state_dict())
        # Net2DeeperNet: zero the output projection of every newly-appended layer so it is an
        # exact identity at the moment of growth (x + 0 = x) — no forgetting.
        if decision.mode in (GrowthMode.DEEPEN.value, GrowthMode.BOTH.value):
            for layer in list(new.net.layers)[old_n_layers:]:
                self._zero_layer_output(layer)
        preserved = self._function_preserved(source, new)
        # Net2WiderNet is a morphism + brief recovery: re-anchor capability by a little training.
        if decision.mode in (GrowthMode.WIDEN.value, GrowthMode.BOTH.value) and not preserved:
            try:
                new.train_on(list(_GENESIS_SEED), steps=20, seed=int(getattr(self.cfg, "seed", 0)))
            except Exception:  # noqa: BLE001 — recovery is best-effort
                pass
        return new, preserved

    @staticmethod
    def _zero_layer_output(layer: Any) -> None:
        """Zero a layer's output projection (and bias) so its residual contribution starts at 0."""
        inner = getattr(layer, "inner", None)
        if inner is None or not _HAS_TORCH:
            return
        for pname in ("fc2", "w2", "proj"):
            proj = getattr(inner, pname, None)
            weight = getattr(proj, "weight", None)
            if weight is not None:
                with torch.no_grad():
                    weight.zero_()
                    bias = getattr(proj, "bias", None)
                    if bias is not None:
                        bias.zero_()
                return

    @staticmethod
    def _function_preserved(old: GenesisModel, new: GenesisModel,
                            *, tol: float = 1e-3) -> bool:
        """True iff the grown brain reproduces the old one's behaviour (probe perplexity) within
        a relative tolerance — the empirical check that the morphism preserved the function."""
        probe = "\n".join(_GENESIS_SEED[:4])
        try:
            before = old.perplexity(probe)
            after = new.perplexity(probe)
        except Exception:  # noqa: BLE001
            return False
        if before <= 0 or before != before:  # guard against inf/NaN
            return False
        return abs(after - before) <= tol * max(1.0, before)

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _genome_of(source: Any) -> ArchitectureGenome:
        if isinstance(source, ArchitectureGenome):
            return source
        genome = getattr(source, "genome", None)
        if isinstance(genome, ArchitectureGenome):
            return genome
        if isinstance(genome, dict):
            return ArchitectureGenome.from_dict(genome)
        spec_genome = getattr(source, "genome", None)
        if isinstance(spec_genome, dict):
            return ArchitectureGenome.from_dict(spec_genome)
        raise TypeError(f"cannot derive an ArchitectureGenome from {type(source).__name__}")

    @staticmethod
    def _spec_of(genome: ArchitectureGenome) -> Any:
        # GenesisModel._spec_from_genome is a staticmethod — usable without torch / instantiation.
        return GenesisModel._spec_from_genome(genome)

    def _param_count(self, genome: ArchitectureGenome) -> int:
        try:
            from nyxara.growth.foundry_models import build_model
            return int(build_model(self._spec_of(genome)).param_count())
        except Exception:  # noqa: BLE001 — counting is best-effort
            return 0


# --------------------------------------------------------------------------- #
# Self-test / demo
# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # pragma: no cover
    print("=" * 70)
    print("NYXARA Dynamic Topology Expansion (Net2Net runtime growth) self-test")
    print("=" * 70)

    from nyxara.kernel.config import TopologyConfig

    cfg = TopologyConfig(grow_dims_k=8, max_n_embd=128, max_layers=8)
    mon = CapacityMonitor(cfg=cfg)
    base = ArchitectureGenome(n_embd=64, block_size=16,
                              layers=[LayerGene(op="attention", n_head=4),
                                      LayerGene(op="gated_mlp")])

    # no pressure → no growth
    calm = CapacitySignal(problem_difficulty=0.1, saturation=0.1)
    assert mon.should_grow(calm, genome=base).should_grow is False
    print("\n  calm signal      : no growth ✓")

    # pressure → grow; ceiling respected
    hot = CapacitySignal(problem_difficulty=0.9, saturation=0.9, loss_plateau=0.9)
    dec = mon.should_grow(hot, genome=base)
    assert dec.should_grow and dec.mode in ("widen", "deepen", "both")
    print(f"  hot signal       : grow {dec.mode} by k={dec.k} ✓")
    ceil = ArchitectureGenome(n_embd=128, block_size=16,
                              layers=[LayerGene(op="gated_mlp")] * 8)
    assert mon.should_grow(hot, genome=ceil).should_grow is False
    print("  ceiling          : refuses to grow past the hardware limit ✓")

    # genome growth (always available): dims and depth increase, ceiling honoured
    topo = DynamicTopology(cfg=cfg)
    _m, rep = topo.grow(base, GrowthDecision(True, "both", 8, "test"))
    print(f"\n  genome growth    : {rep.reason}")
    assert rep.after_n_embd == 72 and rep.after_layers == 3
    assert rep.after_params >= rep.before_params

    if _HAS_TORCH:
        print("\n[torch present] function-preserving morphisms ...")
        gm = GenesisModel(base)
        gm.train_on(list(_GENESIS_SEED), steps=30, seed=1)
        # Net2DeeperNet is EXACT: deeper brain reproduces the shallower one's behaviour
        deeper, drep = topo.grow(gm, GrowthDecision(True, "deepen", 0, "deepen"))
        print(f"  deepen           : preserved={drep.preserved} "
              f"params {drep.before_params}→{drep.after_params}")
        assert drep.preserved is True, "Net2DeeperNet must preserve the function exactly"
        assert deeper.param_count() > gm.param_count()
        # Net2WiderNet grows the width and re-anchors
        wider, wrep = topo.grow(gm, GrowthDecision(True, "widen", 8, "widen"))
        print(f"  widen            : n_embd {wrep.before_n_embd}→{wrep.after_n_embd} "
              f"params {wrep.before_params}→{wrep.after_params}")
        assert wider.param_count() > gm.param_count()
    else:
        print("\n[torch absent] neural morphisms skipped; genome growth path verified ✓")

    print("\nALL SELF-TESTS PASSED ✓")
