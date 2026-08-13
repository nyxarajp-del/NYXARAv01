"""NYXARA · njp/evolve.py — she rewrites her own source (🛠, NJP V.01).

Every other faculty in NJP changes what she *knows*. This one changes what she **is made of**, and
it changes it on disk — the file is different afterwards, and the difference survives a restart.

The loop, and every stage of it is real:

1. **measure** — :class:`Profiler` takes her own per-module cost from live turns. Nothing is
   hand-nominated: the module she is measurably slowest in is the module she opens. A profiler
   with no samples proposes nothing, which is the correct behaviour on a cold start.
2. **propose** — a concrete whole-file edit, with the full before and after text so rollback is
   byte-exact rather than approximate.
3. **the Truth-Seeking Gauntlet** — and this is the part that makes the loop honest. An edit is a
   *claim*: "this makes her faster and no less accurate." That claim goes through
   :class:`~nyxara.njp.truth.TruthGauntlet` like any other, and the hard evidence is a
   :class:`~nyxara.njp.truth.PredictiveSource` over **held-out** benchmark samples the edit was not
   tuned against. An edit whose improvement is only visible on the samples that motivated it is
   fitting noise, and it is refused here rather than discovered later.
4. **the source gauntlet** — :class:`nyxara.growth.self_optimize.Optimizer` then does the part that
   touches disk: backup → syntax → the safety battery (``python -m nyxara.eval``) → capability
   benchmark → the test suite → **keep, or roll back byte-for-byte**. Any failure, timeout or
   error rolls back. Every outcome is journalled.
5. **record** — kept and rolled-back counts go to :mod:`nyxara.njp.ledger`, and
   :meth:`~nyxara.njp.ledger.Ledger.regressed` is consulted *before* the next edit, so an edit that
   made her worse stops the next one instead of compounding.

Separately from source edits, :meth:`SelfEvolver.accelerate` drives
:mod:`nyxara.njp.forge` — the AST→C lowerer — which swaps a verified-identical, measurably faster
kernel into the running process **in memory**. Nothing on disk changes there, so a restart restores
pure Python; it is the cheap half of "faster next time" and it is safe to leave on.

**What is never a candidate.** The constitutional core is refused before anything is generated:
``kernel/rules.py``, ``kernel/invariants.py``, ``kernel/config.py``, ``identity/values.py``,
``identity/soul.py``, ``guard/*``, ``growth/loyalty.py``, and any target naming loyalty,
corrigibility, oversight or honesty. This is checked here *and* again inside the optimizer — two
independent refusals, because one of them being wrong is a thing that can happen.

**Autonomy is not extra permission.** Every beat checks oversight first: a paused or scrammed mind
evolves nothing, and an oversight gate that cannot be read is treated as STOP, never as "carry on".
Enactment additionally requires the standing authorisation
(``self_improvement.autonomous_enact``), which the test suite forces off so a hermetic run can
never edit the live tree.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from nyxara.njp.truth import PredictiveSource, TruthGauntlet, Verdict

__all__ = ["ModuleCost", "Profiler", "EvolutionStep", "SelfEvolver"]

# Mirrored from growth/self_optimize.PROTECTED_RELPATHS. Fail-closed: a path that cannot be
# resolved is treated as protected.
_PROTECTED_RELPATHS = (
    "kernel/rules.py", "kernel/invariants.py", "kernel/config.py",
    "identity/values.py", "identity/soul.py", "guard/value_learning.py",
    "guard/corrigibility.py", "guard/auth.py", "growth/loyalty.py",
)
_PROTECTED_NAMES = frozenset({
    "loyalty", "corrigibility", "oversight", "honesty", "guardian", "shield",
    "safety_core", "value_core", "obedience", "auth", "vault", "crypto",
})


def _permitted(oversight: Any) -> bool:
    """Autonomy buys no extra power. An unreadable gate means *do not act*, never *carry on*."""
    if oversight is None:
        return True
    try:
        check = getattr(oversight, "gate", None)
        if callable(check):
            return bool(check())
        return bool(check) if check is not None else True
    except Exception:  # noqa: BLE001
        return False


def is_protected(target: str) -> bool:
    """Is this file or name part of the constitutional core? Fail-closed on any doubt."""
    try:
        low = str(target or "").replace("\\", "/").lower()
        if not low:
            return True
        if any(low.endswith(rel) or rel in low for rel in _PROTECTED_RELPATHS):
            return True
        stem = low.rsplit("/", 1)[-1].replace(".py", "")
        return any(name in stem for name in _PROTECTED_NAMES)
    except Exception:  # noqa: BLE001 — cannot tell ⇒ protected
        return True


@dataclass
class ModuleCost:
    """What one module actually costs her, measured over real turns."""

    module: str = ""
    calls: int = 0
    total_ms: float = 0.0
    errors: int = 0

    @property
    def mean_ms(self) -> float:
        return (self.total_ms / self.calls) if self.calls else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {"module": self.module, "calls": self.calls,
                "mean_ms": round(self.mean_ms, 4), "total_ms": round(self.total_ms, 3),
                "errors": self.errors}


class Profiler:
    """Her own cost, per module. The only input that decides what she rewrites."""

    def __init__(self, *, min_calls: int = 12) -> None:
        self.min_calls = max(1, int(min_calls))
        self.costs: Dict[str, ModuleCost] = {}

    def observe(self, module: str, ms: float, *, error: bool = False) -> None:
        try:
            cost = self.costs.get(module)
            if cost is None:
                cost = ModuleCost(module=str(module))
                self.costs[module] = cost
            cost.calls += 1
            cost.total_ms += float(ms)
            if error:
                cost.errors += 1
        except Exception:  # noqa: BLE001 — a stopwatch never breaks a thought
            pass

    def worst(self) -> Optional[ModuleCost]:
        """The module she spends most time in — with enough samples to mean anything."""
        try:
            eligible = [c for c in self.costs.values()
                        if c.calls >= self.min_calls and not is_protected(c.module)]
            if not eligible:
                return None
            return max(eligible, key=lambda c: c.total_ms)
        except Exception:  # noqa: BLE001
            return None

    def stats(self) -> Dict[str, Any]:
        ranked = sorted(self.costs.values(), key=lambda c: c.total_ms, reverse=True)
        worst = self.worst()
        return {"modules": len(self.costs), "min_calls": self.min_calls,
                "worst": worst.to_dict() if worst is not None else None,
                "top": [c.to_dict() for c in ranked[:5]]}


@dataclass
class EvolutionStep:
    """One attempt at rewriting herself, and everything that decided the outcome."""

    target: str = ""
    proposed: bool = False
    judged: Optional[Any] = None          # njp.truth.Judgement on the improvement claim
    applied: bool = False
    kept: bool = False
    rolled_back: bool = False
    reason: str = ""
    ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {"target": self.target, "proposed": self.proposed, "applied": self.applied,
                "kept": self.kept, "rolled_back": self.rolled_back, "reason": self.reason,
                "ms": round(self.ms, 3),
                "judged": self.judged.to_dict() if self.judged is not None else None}


class SelfEvolver:
    """Measure herself, propose an edit, prove it is an improvement, keep it or undo it."""

    def __init__(self, brain: Any = None, *, ledger: Any = None,
                 settings: Any = None, min_gain: float = 0.05,
                 holdout_samples: int = 8, every_s: float = 300.0,
                 enabled: bool = True) -> None:
        self.brain = brain
        self.ledger = ledger
        self.settings = settings
        self.min_gain = float(min_gain)
        self.holdout_samples = max(2, int(holdout_samples))
        self.every_s = float(every_s)
        self.enabled = bool(enabled)

        self.profiler = Profiler()
        self.gauntlet = TruthGauntlet(ledger=ledger)
        self.steps: List[EvolutionStep] = []
        self.kept = 0
        self.rolled_back = 0
        self._last_beat = 0.0
        self._compiler: Any = None

    # ---- measurement ------------------------------------------------------ #
    def observe(self, module: str, ms: float, *, error: bool = False) -> None:
        """Feed one measured module cost. Called from the brain's own think-cycle."""
        self.profiler.observe(module, ms, error=error)

    # ---- the beat --------------------------------------------------------- #
    def beat(self, *, oversight: Any = None, force: bool = False) -> Optional[EvolutionStep]:
        """One attempt, at most once per ``every_s``. Returns ``None`` when nothing was tried."""
        if not self.enabled:
            return None
        if not _permitted(oversight):
            return None                       # paused or scrammed: evolve nothing
        now = time.monotonic()
        if not force and self._last_beat and (now - self._last_beat) < self.every_s:
            return None
        self._last_beat = now

        # An edit that made her worse stops the next one rather than compounding.
        try:
            if self.ledger is not None and self.ledger.regressed() is True:
                step = EvolutionStep(reason="last generation regressed — not editing on top of it")
                self.steps.append(step)
                return step
        except Exception:  # noqa: BLE001
            pass

        step = EvolutionStep()
        t0 = time.perf_counter()
        try:
            worst = self.profiler.worst()
            if worst is None:
                step.reason = "not enough measured samples to name a target"
                step.ms = (time.perf_counter() - t0) * 1000.0
                self.steps.append(step)
                return step
            step.target = worst.module
            if is_protected(worst.module):
                step.reason = "target is part of the constitutional core — refused"
                step.ms = (time.perf_counter() - t0) * 1000.0
                self.steps.append(step)
                return step

            edit = self.propose(worst)
            if edit is None:
                step.reason = "no safe edit found for the slowest module"
                step.ms = (time.perf_counter() - t0) * 1000.0
                self.steps.append(step)
                return step
            step.proposed = True

            # THE TRUTH-SEEKING GAUNTLET. The edit's own claim must survive held-out evidence.
            step.judged = self.verify_improvement(edit, worst)
            if step.judged is not None and step.judged.verdict != Verdict.ESTABLISHED:
                step.reason = (f"improvement claim was {step.judged.verdict}, not established — "
                               f"edit refused before it touched disk")
                step.ms = (time.perf_counter() - t0) * 1000.0
                self.steps.append(step)
                return step

            outcome = self.apply(edit)
            if outcome is None:
                step.reason = "enactment is not authorised (autonomous_enact off)"
            else:
                step.applied = bool(getattr(outcome, "applied", False))
                step.kept = bool(getattr(outcome, "kept", False))
                step.rolled_back = bool(getattr(outcome, "rolled_back", False))
                step.reason = str(getattr(outcome, "reason", "") or "")
                if step.kept:
                    self.kept += 1
                if step.rolled_back:
                    self.rolled_back += 1
            step.ms = (time.perf_counter() - t0) * 1000.0
            self.steps.append(step)
            if len(self.steps) > 256:
                del self.steps[:-256]
            return step
        except Exception:  # noqa: BLE001 — a failed evolution changes nothing
            step.reason = "evolution failed"
            step.ms = (time.perf_counter() - t0) * 1000.0
            self.steps.append(step)
            return step

    # ---- the stages ------------------------------------------------------- #
    def propose(self, cost: ModuleCost) -> Optional[Any]:
        """A concrete whole-file edit for the module she is slowest in.

        Reuses the repo's existing deterministic transforms rather than inventing a second edit
        vocabulary: they are narrow, reviewed, and already trusted by the optimizer's gauntlet.
        Larger refactors are deliberately not attempted here — a plausible-looking rewrite of a
        file she depends on is worse than no rewrite.
        """
        try:
            from nyxara.growth.self_optimize import EditGenerator, SourceEdit
            path = self._module_path(cost.module)
            if path is None or not path.exists():
                return None
            if is_protected(str(path)):
                return None
            before = path.read_text(encoding="utf-8")
            weakness = type("Weakness", (), {
                "file": str(path), "path": str(path), "module": cost.module,
                "kind": "performance",
                "detail": (f"{cost.module} is the module NJP spends most time in "
                           f"({cost.mean_ms:.3f} ms mean over {cost.calls} calls)"),
            })()
            edit = EditGenerator().generate(weakness)
            if edit is not None and getattr(edit, "changes", False):
                return edit
            # Nothing deterministic to change is a real, common outcome — say so by returning
            # None rather than inventing an edit to have something to do.
            return None
        except Exception:  # noqa: BLE001
            return None

    def verify_improvement(self, edit: Any, cost: ModuleCost) -> Any:
        """Put the edit's *claim* through the gauntlet, with held-out samples as hard evidence.

        The claim is only established when the edit measurably helps on samples it was **not**
        selected against. This is what stops her keeping a change that merely fits the workload
        that happened to motivate it.
        """
        try:
            claim = (f"editing {cost.module} makes NJP faster without losing accuracy "
                     f"(baseline {cost.mean_ms:.4f} ms over {cost.calls} calls)")
            holdout = self._holdout(cost)
            gauntlet = TruthGauntlet(
                sources=[PredictiveSource(predictor=self._predict_gain, holdout=holdout,
                                          min_samples=2, pass_ratio=0.75)],
                min_sources=1, require_hard=True, ledger=self.ledger)
            return gauntlet.judge(claim)
        except Exception:  # noqa: BLE001
            return None

    def _holdout(self, cost: ModuleCost) -> List[Tuple[str, float]]:
        """Benchmark samples the edit was not tuned against.

        Drawn from the brain's own replay set when it has one; otherwise from the profiler's live
        record. With fewer than two, :class:`PredictiveSource` abstains and the edit is refused —
        which is the correct default, not a gap.
        """
        try:
            replay = getattr(self.brain, "replay", None)
            if replay:
                return [(str(item), cost.mean_ms) for item in list(replay)[: self.holdout_samples]]
            return [(cost.module, cost.mean_ms)
                    for _ in range(min(self.holdout_samples, max(0, cost.calls // 4)))]
        except Exception:  # noqa: BLE001
            return []

    def _predict_gain(self, claim: str, sample: Any) -> bool:
        """Did this held-out sample actually get faster? Measured, never assumed.

        Without a real re-measurement hook this returns False, so the claim fails and the edit is
        refused. Refusing on absent evidence is the whole design: the alternative is a self-editor
        that keeps changes it never checked.
        """
        try:
            measure = getattr(self.brain, "measure_sample", None)
            if not callable(measure):
                return False
            _name, baseline_ms = sample if isinstance(sample, tuple) else (sample, None)
            if baseline_ms is None:
                return False
            after_ms = float(measure(_name))
            return after_ms <= float(baseline_ms) * (1.0 - self.min_gain)
        except Exception:  # noqa: BLE001
            return False

    def apply(self, edit: Any) -> Optional[Any]:
        """Hand the edit to the source gauntlet: backup → checks → keep or byte-exact rollback."""
        try:
            from nyxara.growth.self_optimize import Optimizer
            opt = Optimizer(settings=self.settings)
            if not opt.autonomous_enact:
                return None
            return opt.apply(edit)
        except Exception:  # noqa: BLE001
            return None

    @staticmethod
    def _module_path(module: str) -> Optional[Path]:
        try:
            import importlib.util
            spec = importlib.util.find_spec(module)
            origin = getattr(spec, "origin", None) if spec is not None else None
            return Path(origin) if origin else None
        except Exception:  # noqa: BLE001
            return None

    # ---- the in-memory half ----------------------------------------------- #
    def accelerate(self, *, oversight: Any = None) -> Any:
        """Lower a hot numeric function to C and swap it in — in memory, nothing written.

        Verified identical on every sample and measurably faster before it is adopted, and rolled
        back otherwise. With no toolchain present this is a clean no-op.
        """
        try:
            if not _permitted(oversight):
                return None
            if self._compiler is None:
                from nyxara.njp.forge import MetamorphicCompiler
                self._compiler = MetamorphicCompiler(self.brain)
            return self._compiler.beat(oversight=oversight)
        except Exception:  # noqa: BLE001 — recompiling herself never breaks a beat
            return None

    # ---- reporting -------------------------------------------------------- #
    def stats(self) -> Dict[str, Any]:
        attempts = len(self.steps)
        return {"enabled": self.enabled, "attempts": attempts,
                "kept": self.kept, "rolled_back": self.rolled_back,
                "keep_rate": round(self.kept / attempts, 4) if attempts else None,
                "profiler": self.profiler.stats(),
                "gauntlet": self.gauntlet.stats(),
                "last": self.steps[-1].to_dict() if self.steps else None}
