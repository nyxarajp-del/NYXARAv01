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

from nyxara.eval.ablation import (CORE_FACULTIES, AblationReport, AblationResult, Faculty,
                                  ablate, attr_faculty, mcnemar_exact, run_ablation)
from nyxara.eval.benchmark import (Benchmark, BenchmarkReport, BenchmarkResult,
                                  BenchmarkTask, Grader, build_arithmetic_benchmark,
                                  build_default_benchmark, build_logic_benchmark,
                                  core_solver, grade_final_numeric, llm_solver, self_solver)
from nyxara.eval.continual import (ContinualMetrics, ContinualReport, ContinualTask,
                                   evaluate_continual, make_task_suite, matrix_metrics,
                                   run_sequential)
from nyxara.eval.datasets import (DatasetError, build_realworld_benchmark,
                                  load_jsonl_benchmark)
from nyxara.eval.hard_benchmark import (build_calibration_benchmark, build_code_benchmark,
                                        build_cross_domain_benchmark,
                                        build_deduction_benchmark, build_hard_benchmark,
                                        build_math_benchmark, build_reading_benchmark,
                                        build_sequence_benchmark, grade_calibration)
from nyxara.eval.generalization import (GeneralizationReport, GeneralizationResult,
                                        TransferTask, build_structure_transfer_suite,
                                        run_generalization_benchmark, run_law_induction,
                                        run_structure_transfer)
from nyxara.eval.harness import (EvalCase, EvalOutcome, EvalReport, EvalResult,
                                 EvalSuite, default_core_factory)
from nyxara.eval.intelligence import (STAGE_NAMES, IntelligenceReport,
                                      run_intelligence_benchmark)
from nyxara.eval.intelligence import StageResult as IntelligenceStageResult
from nyxara.eval.suites import build_default_suite

__all__ = [
    "EvalCase",
    "EvalOutcome",
    "EvalResult",
    "EvalReport",
    "EvalSuite",
    "default_core_factory",
    "build_default_suite",
    # continual learning: the forgetting benchmark (eval/continual.py)
    "ContinualTask",
    "ContinualMetrics",
    "ContinualReport",
    "make_task_suite",
    "run_sequential",
    "matrix_metrics",
    "evaluate_continual",
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
    # does she generalize HERSELF, or only via the LLM? (eval/generalization.py)
    # Structure transfer and law induction run against her own faculties with the model
    # removed, so `own_faculty_delta` answers that question with a number instead of a claim.
    "TransferTask",
    "GeneralizationResult",
    "GeneralizationReport",
    "build_structure_transfer_suite",
    "run_structure_transfer",
    "run_law_induction",
    "run_generalization_benchmark",
    # is she LEARNING, or is a counter going up? (eval/intelligence.py)
    # Six stages on generated vocabulary, each scored only on items never taught, each with a
    # fresh brain. Memorisation is the control: below 0.5 the rest of the curve is measuring a
    # broken pipe rather than an absent faculty, and the report says so.
    "run_intelligence_benchmark",
    "IntelligenceReport",
    "IntelligenceStageResult",
    "STAGE_NAMES",
    # does a FACULTY beat its own absence? (eval/ablation.py)
    # Every other battery here measures the whole mind. This one measures one part against not
    # having it, on the held-out fold, with a paired test — the only evidence that can honestly
    # justify removing a module rather than keeping it on faith.
    "Faculty",
    "attr_faculty",
    "AblationResult",
    "AblationReport",
    "mcnemar_exact",
    "ablate",
    "run_ablation",
    "CORE_FACULTIES",
]
