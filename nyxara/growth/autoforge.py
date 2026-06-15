"""NYXARA · growth/autoforge.py — Self-Built Models (Level 11).

Fully automated training pipeline:

    Collect → Distill → Train → Benchmark → Gate → Promote / Rollback

AutoForge is a thin coordinator that reuses Foundry, Distiller, and Eval exactly
as-is. It adds the automation layer: checking data thresholds, sequencing the pipeline,
interpreting the evaluation result, and promoting or rolling back the model.

Triggered by the autonomic loop when the distillation corpus reaches a configurable
threshold (default: 10 new examples since the last forge cycle). Idempotent: if data
hasn't grown, the cycle is a no-op.

Gauntlet gates remain (character-lock, corrigibility, capability). AutoForge never
bypasses safety checks — it uses the same Foundry/Distiller APIs that already
enforce them.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List

__all__ = ["AutoForge", "ForgeResult"]


# --------------------------------------------------------------------------- #
# ForgeResult
# --------------------------------------------------------------------------- #
@dataclass
class ForgeResult:
    """Outcome of one AutoForge cycle."""
    trained: bool = False
    promoted: bool = False
    rolled_back: bool = False
    benchmark_scores: Dict[str, float] = field(default_factory=dict)
    examples_used: int = 0
    reason: str = ""
    elapsed_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trained": self.trained,
            "promoted": self.promoted,
            "rolled_back": self.rolled_back,
            "benchmark_scores": self.benchmark_scores,
            "examples_used": self.examples_used,
            "reason": self.reason,
            "elapsed_ms": round(self.elapsed_ms, 1),
        }


# --------------------------------------------------------------------------- #
# AutoForge
# --------------------------------------------------------------------------- #
class AutoForge:
    """Coordinate the automated model training and promotion pipeline.

    Parameters
    ----------
    foundry:        Foundry instance for training and promotion.
    distiller:      Distiller instance for collecting training examples.
    min_examples:   minimum new examples required to trigger a cycle.
    eval_threshold: minimum benchmark score to promote the trained model.
    """

    def __init__(self, foundry: Any = None, distiller: Any = None,
                 flywheel: Any = None, min_examples: int = 10,
                 eval_threshold: float = 0.6) -> None:
        self.foundry = foundry
        self.distiller = distiller
        # her OWN lived, verified experience (growth/flywheel.py) — counted toward the trigger
        # so growth in her own data forges a new model, not only teacher distillation.
        self.flywheel = flywheel
        self.min_examples = max(1, int(min_examples))
        self.eval_threshold = max(0.0, min(1.0, float(eval_threshold)))
        self._last_example_count: int = 0
        self._cycles: List[ForgeResult] = []

    # ---------------------------------------------------------------------- #
    def run_cycle(self) -> ForgeResult:
        """Run one cycle: check the data threshold, then forge through the Foundry's gauntlet.

        Idempotent: if not enough *new* verified data has accrued since the last forge, this is
        a no-op. Otherwise it delegates train + gauntlet + promote/discard to
        :meth:`Foundry.self_improve` — the proven path that enforces character-lock,
        corrigibility, perplexity improvement and the capability gate — so AutoForge never
        re-implements (or weakens) a safety check. The cycle only ever runs on her own
        gate-cleared data, and a worse or character-violating candidate is never promoted.
        """
        t0 = time.monotonic()
        result = ForgeResult()

        # 1. trigger: enough NEW verified examples since the last forge?
        current_count = self._example_count()
        new_examples = current_count - self._last_example_count
        if new_examples < self.min_examples:
            result.reason = (f"insufficient new data: {new_examples} new "
                             f"(need {self.min_examples})")
            result.elapsed_ms = (time.monotonic() - t0) * 1000
            self._cycles.append(result)
            return result

        if self.foundry is None:
            result.reason = "no foundry wired"
            result.elapsed_ms = (time.monotonic() - t0) * 1000
            self._cycles.append(result)
            return result

        # 2. forge through the Foundry's own gauntlet (train -> gates -> promote/discard)
        try:
            forged = self.foundry.self_improve(generations=1)
        except Exception as exc:  # noqa: BLE001 — a failed forge is reported, never crashes idle
            result.reason = f"forge failed: {exc}"
            result.elapsed_ms = (time.monotonic() - t0) * 1000
            self._cycles.append(result)
            return result

        # 3. interpret the gauntlet's verdict
        if forged:
            fr = forged[-1]
            result.trained = True
            result.promoted = bool(fr.promoted)
            result.rolled_back = not result.promoted
            result.examples_used = current_count
            result.benchmark_scores = self._scores_of(fr)
            result.reason = (f"promoted v{fr.version}: {fr.reason}" if fr.promoted
                             else f"kept active (candidate v{fr.version} not promoted): {fr.reason}")
        else:
            result.reason = "self_improve returned no result"

        # advance the watermark only after a real forge attempt, so the threshold measures
        # data that arrived *since* this cycle (idempotent until new experience accrues)
        self._last_example_count = current_count
        result.elapsed_ms = (time.monotonic() - t0) * 1000
        self._cycles.append(result)
        return result

    def all_cycles(self) -> List[ForgeResult]:
        return list(self._cycles)

    # ---------------------------------------------------------------------- #
    def _example_count(self) -> int:
        """Total available verified examples — teacher distillation + her own flywheel.

        Growth in *either* source can trigger a forge; a missing/erroring source counts 0,
        never crashes the check."""
        total = 0
        for source in (self.distiller, self.flywheel):
            if source is None:
                continue
            try:
                total += int(source.count())
            except Exception:  # noqa: BLE001 — a flaky counter contributes 0, never raises
                pass
        return total

    @staticmethod
    def _scores_of(forged: Any) -> Dict[str, float]:
        """Pull a small, honest score summary from a FoundryResult for the report."""
        after = getattr(forged, "eval_after", None)
        scores: Dict[str, float] = {}
        if after is not None:
            for k in ("perplexity", "task_score"):
                v = getattr(after, k, None)
                if isinstance(v, (int, float)):
                    scores[k] = float(v)
        return scores


# --------------------------------------------------------------------------- #
# Self-test / demo
# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # pragma: no cover
    print("=" * 70)
    print("NYXARA autoforge self-test")
    print("=" * 70)

    # Test without real foundry/distiller — just the coordination logic
    forge = AutoForge(foundry=None, distiller=None, min_examples=5)

    # No data → no-op
    r1 = forge.run_cycle()
    print(f"\nno data: trained={r1.trained}, reason={r1.reason!r}")
    assert not r1.trained

    # Test threshold check
    forge._last_example_count = 0
    forge.min_examples = 1000  # huge threshold
    r2 = forge.run_cycle()
    print(f"high threshold: reason={r2.reason!r}")
    assert not r2.trained

    print(f"\nall cycles: {len(forge.all_cycles())}")
    assert len(forge.all_cycles()) == 2

    print("\nALL SELF-TESTS PASSED ✓")
