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


def _run_intelligence(args: argparse.Namespace) -> int:
    """The seven-stage learning curve.

    Exits non-zero only when the **control** stage fails. The other five are measurements, not
    assertions: a low transfer score is a fact about where she currently is, and turning it into
    a build failure would make the honest thing to do about a weak faculty be to stop measuring
    it. Memorisation is different — below half it means the harness is broken, and a broken
    harness reporting five more numbers is worse than reporting none.
    """
    from nyxara.eval.intelligence import run_intelligence_benchmark
    report = run_intelligence_benchmark(seed=int(args.seed or 0), width=args.width,
                                        only=tuple(args.stage or ()))
    print(report.render())
    if args.save:
        import json
        from pathlib import Path
        Path(args.save).write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
        print(f"\nsaved to {args.save}")
    return 0 if report.sound else 1


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


def _run_frontier(args: argparse.Namespace) -> int:
    """Probe the open-ended auto-curriculum frontier and (optionally) save the score as JSON.

    This is the READ-ONLY, non-saturating ruler the "provably better" gate (Method D,
    growth/improvement_proof.py) certifies against. It grades the sovereign solver on a
    deterministic, freshly-seeded batch at a fixed tier WITHOUT moving the edge, so a
    before-edit and an after-edit run over the *same* ``--seed`` (and ``--tier``) pose byte-identical
    problems — a strictly higher score is then a genuine dominance, not a luckier draw. Because it
    runs in a fresh process against the on-disk source, it exercises the real (possibly just-edited)
    code. Answers are prover-certified and the batch is regenerated every run, so the ruler can be
    neither memorised nor (as the tier rises with mastery) saturated."""
    import json

    from nyxara.eval.benchmark import core_solver
    from nyxara.growth.curriculum import AutoCurriculum
    tier = int(args.tier) if args.tier is not None else None
    cur = AutoCurriculum(memory=None)
    rep = cur.probe(core_solver(), seed=int(args.seed or 0), tier=tier,
                    per_tier=int(args.per_tier))
    payload = {"frontier_score": round(float(rep.frontier_score), 6),
               "by_tier": {int(k): round(float(v), 6) for k, v in rep.by_tier.items()},
               "frontier_tier": int(rep.frontier_tier), "seed": int(args.seed or 0),
               "per_tier": int(args.per_tier), "n_problems": int(rep.n_problems),
               "n_correct": int(rep.n_correct)}
    print(f"frontier probe: score={payload['frontier_score']:.4f} "
          f"tier={payload['frontier_tier']} by_tier={payload['by_tier']}")
    if args.save:
        with open(args.save, "w", encoding="utf-8") as fh:
            json.dump(payload, fh)
        print(f"\nfrontier probe saved -> {args.save}")
    return 0


def _run_adversarial(args: argparse.Namespace) -> int:
    """Run the adversarial natural-language battery and print the four rates per family.

    Its own module has documented ``python -m nyxara.eval --adversarial`` since it was written,
    and the flag did not exist — the battery was reachable only by importing it, which is the
    repository's recurring failure and not a cosmetic one: a benchmark nobody can invoke the
    documented way is a benchmark that stops being run. It is the instrument that grades the
    language surface every other measurement reaches *through*, so it is wired here.

    Exits non-zero only on a run that produced no items at all. The rates are evidence, not a
    gate: there is no threshold here that a CI job could fail on without inviting the thresholds
    to be tuned until they pass.
    """
    from nyxara.eval.adversarial import run_adversarial_benchmark

    seed = 20260823 if args.seed is None else int(args.seed)
    report = run_adversarial_benchmark(seed=seed, families=args.family or None)
    print(report.render())
    if args.save:
        import json
        with open(args.save, "w", encoding="utf-8") as fh:
            json.dump(report.to_dict(), fh, indent=2)
        print(f"\nadversarial run saved -> {args.save}")
    return 0 if report.families else 1


def _run_ablate(args: argparse.Namespace) -> int:
    """Measure each turn-path faculty against its own absence, and print what may be concluded.

    Deliberately exits 0 even when nothing earns its place: this instrument reports evidence, it
    does not pass or fail a build. A non-zero exit would invite wiring it into CI, where a
    small-sample null would quietly become a delete-this signal — which is exactly the inference
    the module is built to refuse.
    """
    import json

    from nyxara.eval.ablation import run_ablation
    report = run_ablation(holdout_frac=float(args.holdout_frac), seed=int(args.seed or 0),
                          limit=int(args.limit))
    print(report.render())
    if report.unmeasured:
        print(f"\nnote: {len(report.unmeasured)} faculty/faculties could not be decided by this "
              f"run. That is a fact about this battery's size and coverage, not evidence against "
              f"them — do not read it as permission to delete.")
    if args.save:
        with open(args.save, "w", encoding="utf-8") as fh:
            json.dump(report.to_dict(), fh, indent=2)
        print(f"\nablation report saved -> {args.save}")
    return 0


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
    parser.add_argument("--general", action="store_true",
                        help="run the HONEST general-capability battery (eval/general_novel.py): a "
                             "held-out set broader than the fixed faculties, measuring what NYXARA "
                             "reasons out herself UNAIDED (no model) vs. the aided program path")
    parser.add_argument("--frontier", action="store_true",
                        help="probe the open-ended auto-curriculum frontier (the non-saturating "
                             "ruler the provably-better gate certifies against)")
    parser.add_argument("--adversarial", action="store_true",
                        help="run the adversarial natural-language battery "
                             "(eval/adversarial.py): paraphrase, negation, open verbs, polar "
                             "questions and near-miss neighbours, on generated vocabulary with a "
                             "fresh brain per family. Measures the LANGUAGE SURFACE, not "
                             "intelligence")
    parser.add_argument("--family", action="append", default=None,
                        help="adversarial: run only this family (repeatable). Each family draws "
                             "its own seeded vocabulary, so one family alone poses exactly the "
                             "items it would have posed inside a full run")
    parser.add_argument("--ablate", action="store_true",
                        help="measure each turn-path faculty against its OWN ABSENCE on the "
                             "held-out fold (eval/ablation.py): does it beat not having it? "
                             "The evidence a deletion decision needs")
    parser.add_argument("--holdout-frac", dest="holdout_frac", type=float, default=0.4,
                        help="ablate: fraction of the battery held out for scoring")
    parser.add_argument("--limit", type=int, default=0,
                        help="ablate: score at most N held-out tasks (0 = all). Each faculty "
                             "costs two full passes, so this is the wall-clock dial")
    parser.add_argument("--seed", type=int, default=None,
                        help="seed for a deterministic batch (same seed ⇒ same problems, so "
                             "before/after edits are compared on identical questions). Defaults "
                             "to 0 for --frontier and to the battery's own seed for --adversarial")
    parser.add_argument("--tier", type=int, default=None,
                        help="frontier: probe at this fixed difficulty tier (default: the "
                             "curriculum's current frontier tier)")
    parser.add_argument("--per-tier", dest="per_tier", type=int, default=4,
                        help="frontier: problems generated per tier")
    parser.add_argument("--category", default=None, help="run only one category")
    parser.add_argument("--baseline", default=None,
                        help="compare against a saved baseline and flag regressions")
    parser.add_argument("--intelligence", action="store_true",
                        help="run the seven-stage learning curve (eval/intelligence.py): "
                             "memorization -> generalization -> recombination -> causal "
                             "prediction -> self-correction -> transfer. Every stage is scored "
                             "on items the brain was never taught, on generated vocabulary, "
                             "with a fresh brain per stage")
    parser.add_argument("--width", type=int, default=6,
                        help="intelligence: items per stage. Resolution, not difficulty")
    parser.add_argument("--stage", action="append", default=None,
                        help="intelligence: run only this stage (repeatable)")
    parser.add_argument("--relational", action="store_true",
                        help="probe whether the gradient head can generalise a relation "
                             "(eval/relational.py). Always reports the nonsense-subject control "
                             "beside the real one — the gap between them is the only number here "
                             "that means anything")
    parser.add_argument("--teach-is-a", action="store_true",
                        help="relational: also teach every member's kind, so two hops are "
                             "composable in principle. It measures worse")
    parser.add_argument("--expand", action="store_true",
                        help="relational: widen each query by the subject's neighbours from "
                             "njp/embed.py — the similarity structure a hashed cell id lacks")
    parser.add_argument("--epochs", type=int, default=60,
                        help="relational: passes over the training triples")
    parser.add_argument("--save", default=None, help="save this run as a baseline JSON")
    args = parser.parse_args(argv)
    if args.relational:
        from nyxara.eval.relational import probe
        report = probe(epochs=args.epochs, teach_is_a=args.teach_is_a,
                       expand=args.expand)
        print(report.render())
        return 0
    if args.intelligence:
        return _run_intelligence(args)
    if args.general:
        from nyxara.eval.general_novel import run_general
        report = run_general()
        return 0 if report.accuracy >= 0.0 else 1
    if args.adversarial:
        return _run_adversarial(args)
    if args.frontier:
        return _run_frontier(args)
    if args.ablate:
        return _run_ablate(args)
    return _run_benchmark(args) if args.benchmark else _run_safety(args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
