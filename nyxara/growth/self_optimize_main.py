"""NYXARA · growth/self_optimize_main.py — run the unified self-optimization loop from the CLI (♾).

A Master-facing command for inspection and manual triggering of NYXARA's eleven-phase
self-optimization cycle. The *automatic* driver is the background
:class:`~nyxara.kernel.autonomic.AutonomicLoop`; this entry point lets you look at (and, with
``--enact``, deliberately apply) the same self-driven cycle.

    python -m nyxara.growth.self_optimize_main --all            # full eleven-phase report
    python -m nyxara.growth.self_optimize_main --analyze        # phases 1–3 only (analyse/verify)
    python -m nyxara.growth.self_optimize_main --experiment     # phases 4–5 only
    python -m nyxara.growth.self_optimize_main --debug          # phase 8 only (self-debugging)
    python -m nyxara.growth.self_optimize_main --invent         # phase 10 only
    python -m nyxara.growth.self_optimize_main --verify         # phase 11 only (safety)
    python -m nyxara.growth.self_optimize_main --all --enact    # actually apply gauntlet-gated gains

Dry-run by default: nothing is written to source unless ``--enact`` is passed AND
``settings.self_optimization.autonomous_enact`` is set. ``--no-debug`` skips the slow
pytest-driven self-debug phase.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import List, Optional


def _build_loop(args: argparse.Namespace):
    from nyxara.growth.self_optimization import SelfOptimizationLoop
    return SelfOptimizationLoop(root=args.root)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="nyxara.growth.self_optimize_main",
        description="Run NYXARA's unified eleven-phase self-optimization loop "
                    "(analyse → optimise → verify → experiment → architecture → tools → learn "
                    "→ debug → compute → invent → safety).")
    parser.add_argument("--all", action="store_true",
                        help="run the full eleven-phase cycle (default when no phase flag given)")
    parser.add_argument("--analyze", action="store_true",
                        help="phases 1–3 only: self-analysis + optimise + verified modification")
    parser.add_argument("--experiment", action="store_true",
                        help="phases 4–5 only: automatic experimentation + architecture")
    parser.add_argument("--debug", action="store_true",
                        help="phase 8 only: self-debugging (run own tests, isolate, fix, verify)")
    parser.add_argument("--invent", action="store_true",
                        help="phase 10 only: scientific invention (conjecture → prover)")
    parser.add_argument("--verify", action="store_true",
                        help="phase 11 only: whole-cycle safety verification")
    parser.add_argument("--enact", action="store_true",
                        help="actually apply gauntlet-gated gains (else dry-run)")
    parser.add_argument("--no-debug", action="store_true",
                        help="skip the slow pytest-driven self-debug phase in --all")
    parser.add_argument("--generations", type=int, default=None,
                        help="override experiment/architecture generations")
    parser.add_argument("--root", default=None, help="repo root to operate on (defaults to cwd)")
    parser.add_argument("--save", default=None, help="write the JSON report to this path")
    args = parser.parse_args(argv)

    enact = bool(args.enact)
    phase = any([args.analyze, args.experiment, args.debug, args.invent, args.verify])

    try:
        loop = _build_loop(args)
    except Exception as exc:  # noqa: BLE001
        print(f"· could not build the self-optimization loop: {exc}")
        return 1

    if not phase or args.all:
        report = loop.run(enact=enact, generations=args.generations,
                          include_debug=not args.no_debug)
        print(report.summary())
        payload = report.to_dict()
    else:
        # run a single phase (or the small phase-group) for inspection
        payload = _run_single(loop, args, enact)

    if args.save and payload is not None:
        try:
            with open(args.save, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, default=str)
            print(f"\nreport saved -> {args.save}")
        except Exception as exc:  # noqa: BLE001
            print(f"· could not save report: {exc}")

    return 0


def _run_single(loop, args: argparse.Namespace, enact: bool):
    """Run one requested phase (or small group) and print its result."""
    results = []
    try:
        if args.analyze:
            p1, weak = loop._run_phase(1, loop._phase_analyze)         # noqa: SLF001
            results.append(p1)
            p2, opt = loop._run_phase(2, lambda: loop._phase_optimize(weak, enact))  # noqa: SLF001
            results.append(p2)
            results.append(loop._run_phase(                            # noqa: SLF001
                3, lambda: loop._phase_verify_modification(opt))[0])
        if args.experiment:
            results.append(loop._run_phase(                            # noqa: SLF001
                4, lambda: loop._phase_experiment(enact, args.generations))[0])
            results.append(loop._run_phase(                            # noqa: SLF001
                5, lambda: loop._phase_architecture(enact, args.generations))[0])
        if args.debug:
            results.append(loop._run_phase(8, loop._phase_debug)[0])   # noqa: SLF001
        if args.invent:
            results.append(loop._run_phase(10, loop._phase_invent)[0])  # noqa: SLF001
        if args.verify:
            results.append(loop._run_phase(11, loop._phase_safety)[0])  # noqa: SLF001
    except Exception as exc:  # noqa: BLE001
        print(f"· phase run failed: {exc}")
        return None

    glyph = {"ok": "✓", "skipped": "·", "failed": "✗", "withheld": "⊘"}
    for p in results:
        mark = glyph.get(p.status, "?")
        v = " [verified]" if p.verified else ""
        print(f"  {mark} {p.n:>2}. {p.name}{v} — {p.detail}")
    return {"phases": [p.to_dict() for p in results]}


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
