"""NYXARA · growth/improve_main.py — run the recursive self-improvement cycle from the CLI (♻).

A Master-facing command for inspection and manual verification of the RSI faculties — the
*automatic* driver is the background :class:`~nyxara.kernel.autonomic.AutonomicLoop`; this
entry point lets you look at (and, with ``--enact``, deliberately trigger) the same cycle.

    python -m nyxara.growth.improve_main --all              # full dry-run report
    python -m nyxara.growth.improve_main --review            # just the code review
    python -m nyxara.growth.improve_main --arch              # just architecture analysis
    python -m nyxara.growth.improve_main --benchmark         # just the capability benchmark
    python -m nyxara.growth.improve_main --weaknesses        # ranked weaknesses
    python -m nyxara.growth.improve_main --optimize --enact  # actually apply gauntlet-gated fixes

Dry-run by default: nothing is written to source unless ``--enact`` is passed AND
``settings.self_improvement.autonomous_enact`` is set. ``--fail-on-weakness`` makes it a CI
gate (non-zero exit when any weakness exceeds the configured severity).
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import List, Optional


def _build_rsi(args: argparse.Namespace):
    from nyxara.growth.recursive_improvement import RecursiveSelfImprovement
    return RecursiveSelfImprovement(root=args.root)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="nyxara.growth.improve_main",
        description="Run NYXARA's recursive self-improvement: review → architecture → "
                    "benchmark → weaknesses → optimise.")
    parser.add_argument("--all", action="store_true",
                        help="run the full cycle (default when no phase flag is given)")
    parser.add_argument("--review", action="store_true", help="self code review only")
    parser.add_argument("--arch", action="store_true", help="self architecture analysis only")
    parser.add_argument("--benchmark", action="store_true", help="self benchmark testing only")
    parser.add_argument("--weaknesses", action="store_true", help="detect weaknesses only")
    parser.add_argument("--optimize", action="store_true",
                        help="run optimisation (dry-run unless --enact)")
    parser.add_argument("--enact", action="store_true",
                        help="actually apply gauntlet-gated source edits + safe enactment")
    parser.add_argument("--root", default=None, help="package root to analyse (defaults to nyxara/)")
    parser.add_argument("--category", default=None, help="benchmark: run only one category")
    parser.add_argument("--save", default=None, help="write the JSON report to this path")
    parser.add_argument("--fail-on-weakness", action="store_true",
                        help="exit non-zero if any weakness exceeds the severity threshold")
    args = parser.parse_args(argv)

    rsi = _build_rsi(args)
    phase = any([args.review, args.arch, args.benchmark, args.weaknesses, args.optimize])

    payload = None
    if not phase or args.all:
        report = rsi.run(enact=args.enact, category=args.category)
        print(report.summary())
        payload = report.to_dict()
    elif args.review:
        rep = rsi.review_code()
        print(rep.summary())
        payload = rep.to_dict()
    elif args.arch:
        rep = rsi.analyze_architecture()
        print(rep.summary())
        payload = rep.to_dict()
    elif args.benchmark:
        rep = rsi.run_benchmarks(category=args.category)
        print(f"benchmark accuracy: {rep.get('accuracy', 0.0):.0%}; "
              f"handoff: {rep.get('handoff', {})}")
        payload = rep
    elif args.weaknesses:
        rep = rsi.detect_weaknesses()
        print(rep.summary())
        payload = rep.to_dict()
    elif args.optimize:
        report = rsi.optimize(enact=args.enact)
        print(report.summary())
        payload = report.to_dict()

    if args.save and payload is not None:
        with open(args.save, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"\nreport saved -> {args.save}")

    exit_code = 0
    if args.fail_on_weakness:
        threshold = float(rsi.settings.self_improvement.weakness_fail_severity)
        wreport = rsi.detect_weaknesses()
        over = [w for w in wreport.weaknesses if w.severity >= threshold]
        if over:
            print(f"\nFAIL: {len(over)} weakness(es) ≥ severity {threshold:.2f}")
            for w in over[:10]:
                print(f"  ✗ [{w.severity:.2f}] {w.title}")
            exit_code = 1
        else:
            print(f"\nno weakness ≥ severity {threshold:.2f} ✓")
    return exit_code


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
