"""NYXARA · eval/__main__.py — run the evaluation batteries from the CLI (📏).

Two batteries, one entry point:

* **Safety suite** (default) — ``python -m nyxara.eval`` runs the deterministic safety /
  corrigibility / honesty cases against fresh offline cores and exits non-zero on any
  failure, so it doubles as a CI gate.
* **Capability benchmark** — ``python -m nyxara.eval --benchmark`` measures *how capable*
  the mind is (arithmetic + logic), graded against known answers. By default it measures
  the offline reasoner (so it runs anywhere); add ``--llm`` to measure the configured
  provider, or ``--bare-llm`` to measure the model directly (bypassing the loop).
* **Real-world held-out validation** — ``python -m nyxara.eval --benchmark --realworld`` runs the
  genuinely-external corpus (``eval/datasets.py``: real facts + multi-step problems NYXARA never
  trains on). Point ``NYXARA_EVAL_HOLDOUT_PATH`` at a JSONL to validate against a real standard
  dataset (GSM8K/MMLU) instead of the bundled set.

Both support baseline regression tracking::

    python -m nyxara.eval [--benchmark] --baseline base.json   # compare & flag regressions
    python -m nyxara.eval [--benchmark] --save base.json       # write a fresh baseline
"""

from __future__ import annotations

import argparse
import sys


def _run_safety(args: argparse.Namespace) -> int:
    from nyxara.eval.suites import build_default_suite
    report = build_default_suite().run(category=args.category)
    print(report.summary())
    exit_code = 0 if report.pass_rate == 1.0 else 1

    if args.baseline:
        from nyxara.eval.harness import EvalReport
        try:
            baseline = EvalReport.load(args.baseline)
        except FileNotFoundError:
            print(f"\n(no baseline at {args.baseline}; skipping regression check)")
        else:
            regressions = report.regression_vs(baseline)
            if regressions:
                print(f"\nREGRESSIONS vs baseline: {', '.join(regressions)}")
                exit_code = 1
            else:
                print("\nno regressions vs baseline ✓")
    if args.save:
        report.save(args.save)
        print(f"\nbaseline saved -> {args.save}")
    return exit_code


def _select_benchmark(args: argparse.Namespace):
    """The battery this run measures — the hard ruler when ``--hard`` is set, else default."""
    if getattr(args, "hard", False):
        from nyxara.eval.hard_benchmark import build_hard_benchmark
        return build_hard_benchmark()
    from nyxara.eval.benchmark import build_default_benchmark
    return build_default_benchmark()


def _run_ab(args: argparse.Namespace) -> int:
    """A/B the external teacher vs NYXARA's OWN model on the same battery (Phase 0)."""
    from nyxara.eval.benchmark import llm_solver, self_solver
    bench = _select_benchmark(args)
    teacher = bench.run(llm_solver(), category=args.category)
    own = bench.run(self_solver(), category=args.category)

    print("── external teacher ──")
    print(teacher.summary())
    print("\n── NYXARA's own model ──")
    print(own.summary())

    gap = teacher.accuracy - own.accuracy
    print(f"\nA/B: teacher {teacher.accuracy:.0%}  vs  own {own.accuracy:.0%}  "
          f"(gap {gap:+.0%}); own mean-score {own.mean_score:.3f}")
    print(f"own model solves {own.passed}/{len(own)} unaided "
          f"— the Phase-0 handoff floor to grow from.")
    if args.save:
        own.save(args.save)
        print(f"\nbaseline (own model) saved -> {args.save}")
    return 0


def _run_router(args: argparse.Namespace) -> int:
    """Run the confidence router and report accuracy + the self/teacher handoff (Phase 2)."""
    from nyxara.eval.benchmark import run_router
    bench = _select_benchmark(args)
    report, sources = run_router(bench, category=args.category)
    print(report.summary())
    n = len(report) or 1
    # Both her learned self-brain AND her verifiable faculties answer with NO teacher — both are
    # NYXARA reasoning for herself, so the honest "answered herself" rate counts both. We still
    # break them out so the neural-substrate share (self) is visible as it grows.
    own_self = sources.get("self", 0)
    own_faculty = sources.get("faculty", 0)
    own = own_self + own_faculty
    handoff = own / n
    print(f"\nhandoff: NYXARA answered {own}/{n} herself, unaided ({handoff:.0%}) — "
          f"{own_faculty} by verifiable faculty, {own_self} by her learned brain; "
          f"teacher {sources.get('teacher', 0)}, abstained {sources.get('abstain', 0)}, "
          f"unanswered {sources.get('none', 0)}. Accuracy on handed-off turns is the number "
          f"that must hold as handoff rises.")
    if args.save:
        report.save(args.save)
        print(f"\nbaseline (router) saved -> {args.save}")
    return 0


def _run_realworld(args: argparse.Namespace) -> int:
    """Run the REAL, externally-true held-out validation corpus (eval/datasets.py).

    This is the genuinely-external ruler — real facts and multi-step problems NYXARA never trains on,
    or any standard dataset pointed at by ``$NYXARA_EVAL_HOLDOUT_PATH``. It measures the offline
    reasoner by default; ``--bare-llm`` / ``--self`` swap in the configured / own model instead."""
    from nyxara.eval.benchmark import core_solver, llm_solver, self_solver
    from nyxara.eval.datasets import build_realworld_benchmark
    try:
        bench = build_realworld_benchmark()
    except Exception as exc:  # noqa: BLE001
        print(f"could not load the real-world held-out set: {exc}")
        return 1
    solver = (self_solver() if args.self_model
              else llm_solver() if args.bare_llm
              else core_solver())
    report = bench.run(solver, category=args.category)
    print(report.summary())
    exit_code = 0
    if args.baseline:
        from nyxara.eval.benchmark import BenchmarkReport
        try:
            baseline = BenchmarkReport.load(args.baseline)
        except FileNotFoundError:
            print(f"\n(no baseline at {args.baseline}; skipping regression check)")
        else:
            regressions = report.regression_vs(baseline)
            if regressions:
                print(f"\nREGRESSIONS vs baseline: {', '.join(regressions)}")
                exit_code = 1
            else:
                print("\nno regressions vs baseline ✓")
    if args.save:
        report.save(args.save)
        print(f"\nbaseline (real-world held-out) saved -> {args.save}")
    return exit_code


def _run_benchmark(args: argparse.Namespace) -> int:
    if args.realworld:
        return _run_realworld(args)
    if args.ab:
        return _run_ab(args)
    if args.router:
        return _run_router(args)
    from nyxara.eval.benchmark import core_solver, llm_solver, self_solver
    bench = _select_benchmark(args)
    if args.self_model:
        # measure NYXARA's OWN promoted model directly, bypassing the loop
        solver = self_solver()
    elif args.bare_llm:
        solver = llm_solver()
    elif args.llm:
        # measure NYXARA's whole loop with whatever provider is configured
        solver = core_solver()
    else:
        solver = core_solver()  # offline reasoner unless a provider key is set
    report = bench.run(solver, category=args.category)
    print(report.summary())

    # Benchmarks aren't pass/fail gates by default (a weak offline model scores low and
    # that's honest) — only a regression against a saved baseline fails the run.
    exit_code = 0
    if args.baseline:
        from nyxara.eval.benchmark import BenchmarkReport
        try:
            baseline = BenchmarkReport.load(args.baseline)
        except FileNotFoundError:
            print(f"\n(no baseline at {args.baseline}; skipping regression check)")
        else:
            regressions = report.regression_vs(baseline)
            if regressions:
                print(f"\nREGRESSIONS vs baseline: {', '.join(regressions)}")
                exit_code = 1
            else:
                print("\nno regressions vs baseline ✓")
    if args.save:
        report.save(args.save)
        print(f"\nbaseline saved -> {args.save}")
    return exit_code


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="nyxara.eval",
                                     description="Run NYXARA's eval batteries.")
    parser.add_argument("--benchmark", action="store_true",
                        help="run the capability benchmark instead of the safety suite")
    parser.add_argument("--hard", action="store_true",
                        help="benchmark: use the hard, discriminating battery (incl. calibration)")
    parser.add_argument("--llm", action="store_true",
                        help="benchmark: run the whole loop with the configured provider")
    parser.add_argument("--bare-llm", action="store_true",
                        help="benchmark: measure the LLM directly, bypassing the loop")
    parser.add_argument("--self", dest="self_model", action="store_true",
                        help="benchmark: measure NYXARA's OWN promoted model directly")
    parser.add_argument("--ab", action="store_true",
                        help="benchmark: A/B the external teacher vs NYXARA's own model")
    parser.add_argument("--router", action="store_true",
                        help="benchmark: run the confidence router and report the handoff rate")
    parser.add_argument("--realworld", action="store_true",
                        help="benchmark: run the REAL held-out validation corpus (eval/datasets.py); "
                             "set NYXARA_EVAL_HOLDOUT_PATH to use an external dataset")
    parser.add_argument("--category", default=None, help="run only one category")
    parser.add_argument("--baseline", default=None,
                        help="compare against a saved baseline and flag regressions")
    parser.add_argument("--save", default=None, help="save this run as a baseline JSON")
    args = parser.parse_args(argv)
    return _run_benchmark(args) if args.benchmark else _run_safety(args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
