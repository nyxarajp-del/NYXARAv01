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
                 min_examples: int = 10, eval_threshold: float = 0.6) -> None:
        self.foundry = foundry
        self.distiller = distiller
        self.min_examples = max(1, int(min_examples))
        self.eval_threshold = max(0.0, min(1.0, float(eval_threshold)))
        self._last_example_count: int = 0
        self._cycles: List[ForgeResult] = []

    # ---------------------------------------------------------------------- #
    def run_cycle(self) -> ForgeResult:
        """Run one full Collect → Train → Benchmark → Gate → Promote/Rollback cycle.

        Returns a ForgeResult. Idempotent: if no new data, returns immediately.
        """
        t0 = time.monotonic()
        result = ForgeResult()

        # 1. check data threshold
        current_count = self._example_count()
        new_examples = current_count - self._last_example_count
        if new_examples < self.min_examples:
            result.reason = (f"insufficient new data: {new_examples} new "
                             f"(need {self.min_examples})")
            result.elapsed_ms = (time.monotonic() - t0) * 1000
            self._cycles.append(result)
            return result

        # 2. collect corpus
        corpus = self._collect_corpus()
        if not corpus:
            result.reason = "corpus empty after collection"
            result.elapsed_ms = (time.monotonic() - t0) * 1000
            self._cycles.append(result)
            return result

        result.examples_used = len(corpus)

        # 3. train
        try:
            version = self._train(corpus)
            result.trained = (version is not None)
        except Exception as exc:  # noqa: BLE001
            result.reason = f"training failed: {exc}"
            result.elapsed_ms = (time.monotonic() - t0) * 1000
            self._cycles.append(result)
            return result

        if not result.trained:
            result.reason = "training returned no version"
            result.elapsed_ms = (time.monotonic() - t0) * 1000
            self._cycles.append(result)
            return result

        # 4. benchmark
        scores = self._benchmark(corpus)
        result.benchmark_scores = scores
        avg_score = sum(scores.values()) / max(1, len(scores))

        # 5. gate: promote or rollback
        if avg_score >= self.eval_threshold:
            promoted = self._promote(version)
            result.promoted = promoted
            result.reason = f"promoted (score={avg_score:.2f} ≥ {self.eval_threshold})"
        else:
            self._rollback()
            result.rolled_back = True
            result.reason = f"rolled back (score={avg_score:.2f} < {self.eval_threshold})"

        self._last_example_count = current_count
        result.elapsed_ms = (time.monotonic() - t0) * 1000
        self._cycles.append(result)
        return result

    def all_cycles(self) -> List[ForgeResult]:
        return list(self._cycles)

    # ---------------------------------------------------------------------- #
    def _example_count(self) -> int:
        if self.distiller is None:
            return 0
        try:
            return self.distiller.count()
        except Exception:  # noqa: BLE001
            return 0

    def _collect_corpus(self) -> List[str]:
        if self.foundry is None:
            return []
        try:
            return list(self.foundry.collect_corpus())
        except Exception:  # noqa: BLE001
            return []

    def _train(self, corpus: List[str]) -> Any:
        if self.foundry is None:
            return None
        try:
            result = self.foundry.train_candidate(corpus=corpus)
            return getattr(result, "version", result)
        except Exception:  # noqa: BLE001
            return None

    def _benchmark(self, corpus: List[str]) -> Dict[str, float]:
        if self.foundry is None:
            return {"fallback": 0.5}
        try:
            active = self.foundry.active()
            if active is None:
                return {"no_model": 0.0}
            # use Foundry.evaluate with a small sample of corpus as eval texts
            eval_sample = corpus[:min(5, len(corpus))]
            eval_result = self.foundry.evaluate(active, eval_sample)
            return {"accuracy": getattr(eval_result, "accuracy", 0.5),
                    "loss": 1.0 - getattr(eval_result, "accuracy", 0.5)}
        except Exception:  # noqa: BLE001
            return {"fallback": 0.5}

    def _promote(self, version: Any) -> bool:
        if self.foundry is None:
            return False
        try:
            v = int(version) if not isinstance(version, int) else version
            self.foundry.promote(v)
            return True
        except Exception:  # noqa: BLE001
            return False

    def _rollback(self) -> None:
        if self.foundry is None:
            return
        try:
            self.foundry.rollback(steps=1)
        except Exception:  # noqa: BLE001
            pass


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
