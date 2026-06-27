"""NYXARA · eval/__init__.py — the deterministic evaluation harness (📏, public API).

A sovereign mind that cannot be *measured* cannot be *trusted to stay safe as it grows*.
This package is NYXARA's self-assessment instrument: a deterministic harness that runs a
fresh :class:`~nyxara.kernel.orchestrator.NyxaraCore` against a battery of cases —
safety, corrigibility, authority, honesty, tool-use, memory — and produces an auditable,
serialisable :class:`~nyxara.eval.harness.EvalReport`.

Everything here is deterministic against the offline reasoner (no API keys, no network),
so the same suite gives the same numbers on every machine — and a saved baseline lets a
later run detect *regressions* (cases that used to pass and now fail).

Run the whole thing with ``python -m nyxara.eval``.
"""

from __future__ import annotations

from nyxara.eval.benchmark import (Benchmark, BenchmarkReport, BenchmarkResult,
                                  BenchmarkTask, Grader, build_arithmetic_benchmark,
                                  build_default_benchmark, build_logic_benchmark,
                                  core_solver, grade_final_numeric, llm_solver, self_solver)
from nyxara.eval.datasets import (DatasetError, build_realworld_benchmark,
                                  load_jsonl_benchmark)
from nyxara.eval.hard_benchmark import (build_calibration_benchmark, build_code_benchmark,
                                        build_cross_domain_benchmark,
                                        build_deduction_benchmark, build_hard_benchmark,
                                        build_math_benchmark, build_reading_benchmark,
                                        build_sequence_benchmark, grade_calibration)
from nyxara.eval.harness import (EvalCase, EvalOutcome, EvalReport, EvalResult,
                                 EvalSuite, default_core_factory)
from nyxara.eval.suites import build_default_suite

__all__ = [
    "EvalCase",
    "EvalOutcome",
    "EvalResult",
    "EvalReport",
    "EvalSuite",
    "default_core_factory",
    "build_default_suite",
    # capability benchmarks
    "Benchmark",
    "BenchmarkTask",
    "BenchmarkResult",
    "BenchmarkReport",
    "Grader",
    "core_solver",
    "llm_solver",
    "self_solver",
    "grade_final_numeric",
    "build_arithmetic_benchmark",
    "build_logic_benchmark",
    "build_default_benchmark",
    # real-world, externally-true held-out validation (eval/datasets.py)
    "build_realworld_benchmark",
    "load_jsonl_benchmark",
    "DatasetError",
    # the hard, discriminating battery (incl. calibration / honesty)
    "build_hard_benchmark",
    "build_math_benchmark",
    "build_deduction_benchmark",
    "build_sequence_benchmark",
    "build_code_benchmark",
    "build_reading_benchmark",
    "build_calibration_benchmark",
    "build_cross_domain_benchmark",
    "grade_calibration",
]
