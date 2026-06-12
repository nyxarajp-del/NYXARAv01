"""NYXARA · eval/__main__.py — run the default evaluation battery from the CLI (📏).

``python -m nyxara.eval`` builds the default suite, runs it against fresh offline cores,
prints the report (overall + per-category + any failures), and exits non-zero if anything
regressed below a perfect pass — so the harness is usable as a CI gate.

Optional baseline regression check::

    python -m nyxara.eval --baseline path/to/baseline.json   # compare & flag regressions
    python -m nyxara.eval --save path/to/baseline.json       # write a fresh baseline
"""

from __future__ import annotations

import argparse
import sys

from nyxara.eval.suites import build_default_suite


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="nyxara.eval",
                                     description="Run NYXARA's deterministic eval suite.")
    parser.add_argument("--category", default=None, help="run only one category")
    parser.add_argument("--baseline", default=None,
                        help="compare against a saved baseline and flag regressions")
    parser.add_argument("--save", default=None, help="save this run as a baseline JSON")
    args = parser.parse_args(argv)

    suite = build_default_suite()
    report = suite.run(category=args.category)
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


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
