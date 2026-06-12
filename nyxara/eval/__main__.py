"""NYXARA · eval/__main__.py — run the evaluation batteries from the CLI (📏).

Two batteries, one entry point:

* **Safety suite** (default) — ``python -m nyxara.eval`` runs the deterministic safety /
  corrigibility / honesty cases against fresh offline cores and exits non-zero on any
  failure, so it doubles as a CI gate.
* **Capability benchmark** — ``python -m nyxara.eval --benchmark`` measures *how capable*
  the mind is (arithmetic + logic), graded against known answers. By default it measures
  the offline reasoner (so it runs anywhere); add ``--llm`` to measure the configured
  provider, or ``--bare-llm`` to measure the model directly (bypassing the loop).

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


def _run_benchmark(args: argparse.Namespace) -> int:
    from nyxara.eval.benchmark import build_default_benchmark, core_solver, llm_solver
    bench = build_default_benchmark()
    if args.bare_llm:
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
    parser.add_argument("--llm", action="store_true",
                        help="benchmark: run the whole loop with the configured provider")
    parser.add_argument("--bare-llm", action="store_true",
                        help="benchmark: measure the LLM directly, bypassing the loop")
    parser.add_argument("--category", default=None, help="run only one category")
    parser.add_argument("--baseline", default=None,
                        help="compare against a saved baseline and flag regressions")
    parser.add_argument("--save", default=None, help="save this run as a baseline JSON")
    args = parser.parse_args(argv)
    return _run_benchmark(args) if args.benchmark else _run_safety(args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
