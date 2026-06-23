"""NYXARA · growth/__main__.py — grow NYXARA's own brain (distil → train → promote) 🧠↑.

One Master-facing command for the sovereign-brain flywheel, end to end:

1. **Distil** (optional) — ask the configured teacher LLM the identity/voice battery and store
   the answers as a supervised corpus (skipped, never fatal, when no real teacher is set).
2. **Train** — forge a candidate own-model from the distilled corpus + lived memory + seeds.
3. **Gauntlet** — promote it *only* if it clears the character-lock + corrigibility + eval gate
   (reversible; a worse candidate is kept on the bench, never shipped).
4. **Report** (optional) — measure the handoff rate so "wrapper → her own AI" is visible.

The dependency-free **n-gram** backend runs anywhere (so this is CI-testable); point
``--backend lora --base-model Qwen/Qwen2.5-7B`` at a real base on a GPU box for genuine
capability — the very same flywheel, only the backend swaps. Nothing here reaches around a
gate: a promoted model still proposes through the sovereign loop the kernel disposes.

    python -m nyxara.growth --backend ngram --generations 1 --bench
    python -m nyxara.growth --qwen3 --distill --bench        # LoRA-tune Qwen3-4B (her primary)
    python -m nyxara.growth --distill --backend lora --base-model Qwen/Qwen2.5-7B --bench
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, List, Optional

# The shared identity seed corpus + the Qwen3-4B base live in growth/bootstrap.py, so the
# Master-facing CLI and the auto-on-boot forge train the *same* loyal self from the *same* base.
from nyxara.growth.bootstrap import IDENTITY_SEED as _IDENTITY_SEED
from nyxara.growth.bootstrap import QWEN3_4B


def _build_foundry(args: argparse.Namespace) -> Any:
    """A foundry bound to a process-local settings copy (never mutates global config)."""
    from nyxara.kernel.config import get_settings
    from nyxara.growth.foundry import Foundry

    settings = get_settings().model_copy(deep=True)   # honour env config, leak nothing back
    if args.data_dir:
        settings.paths.data_dir = Path(args.data_dir)
    # --qwen3 is the one-command shortcut for NYXARA's primary brain: LoRA-tune Qwen3-4B with
    # QLoRA (4-bit, honoured only on a GPU; degrades on CPU). Explicit flags still override it.
    if args.qwen3:
        settings.foundry.backend = "lora"
        settings.foundry.base_model = QWEN3_4B
        settings.foundry.load_in_4bit = True
    if args.backend:
        settings.foundry.backend = args.backend
    if args.base_model:
        settings.foundry.base_model = args.base_model
    if args.load_in_4bit:
        settings.foundry.load_in_4bit = True
    settings.foundry.enabled = True
    return Foundry(settings=settings, seed_corpus=_IDENTITY_SEED), settings


def _maybe_distill(args: argparse.Namespace, settings: Any) -> int:
    """Distil the teacher into the corpus when one is configured; otherwise skip cleanly."""
    if not args.distill:
        return 0
    from nyxara.growth.distill import Distiller

    distiller = Distiller(settings=settings)
    if not distiller.available():
        print("· no real teacher configured (set NYXARA_LLM__PROVIDER + an API key) — "
              "skipping distillation; training on seeds / lived memory only.")
        return 0
    n: Optional[int] = None if args.distill < 0 else args.distill
    examples = distiller.distill_default(n=n)
    print(f"· distilled {len(examples)} teacher example(s) -> {distiller.store_path}")
    return len(examples)


def _report_handoff(settings: Any) -> None:
    from nyxara.eval.benchmark import build_default_benchmark, run_router

    report, sources = run_router(build_default_benchmark(), settings=settings)
    n = len(report) or 1
    print(f"· handoff: own model answered {sources.get('self', 0)}/{n} unaided "
          f"(faculty {sources.get('faculty', 0)}, teacher {sources.get('teacher', 0)}); "
          f"benchmark accuracy {report.accuracy:.0%}")


def _evolve_mind(args: argparse.Namespace) -> int:
    """Run the recursive mind-evolution loop and print the generational lineage."""
    from nyxara.growth.mind_evolution import MindEvolutionEngine
    from nyxara.kernel.orchestrator import NyxaraCore

    print(f"· evolving NYXARA's way of thinking for {args.evolve_mind} generation(s)"
          f"{' (enacting into the live mind)' if args.enact else ' (measure-only)'}…")
    core = NyxaraCore()
    engine = MindEvolutionEngine(core=core, llm=core.llm, memory=core.memory)
    try:
        report = engine.evolve_generations(int(args.evolve_mind), enact=bool(args.enact))
    except Exception as exc:  # noqa: BLE001 — report, never traceback
        print(f"· could not evolve the mind: {exc}")
        return 1
    print("\n" + report.summary())
    if args.enact:
        n = getattr(getattr(core, "recursive_improver", None), "n_iterations", "n/a")
        print(f"\n· live reasoner now deliberates with n_iterations={n}.")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="nyxara.growth",
        description="Grow NYXARA's own model: distil → train → gauntlet → promote.")
    parser.add_argument("--generations", type=int, default=1,
                        help="self-improvement generations to run (default 1)")
    parser.add_argument("--distill", type=int, nargs="?", const=-1, default=0, metavar="N",
                        help="first distil N teacher prompts into the corpus (all if N omitted)")
    parser.add_argument("--qwen3", action="store_true",
                        help="one-command preset: LoRA-tune Qwen/Qwen3-4B (QLoRA) as her primary "
                             "brain; weights download on first use. Needs .[foundry] for the real "
                             "base (degrades to the n-gram brain otherwise)")
    parser.add_argument("--backend", choices=["auto", "ngram", "nanogpt", "lora"], default=None,
                        help="override the foundry backend (lora needs a GPU + .[foundry])")
    parser.add_argument("--base-model", default=None,
                        help="base checkpoint for the lora backend, e.g. Qwen/Qwen3-4B")
    parser.add_argument("--load-in-4bit", action="store_true",
                        help="QLoRA: load the base in 4-bit (needs bitsandbytes + a GPU); "
                             "lets a 7B+ base fine-tune on one consumer GPU, degrades on CPU")
    parser.add_argument("--data-dir", default=None,
                        help="where the foundry/ state (corpus, versions, active) lives")
    parser.add_argument("--bench", action="store_true",
                        help="report the handoff rate + benchmark accuracy afterwards")
    parser.add_argument("--evolve-mind", type=int, default=0, metavar="N",
                        help="instead of forging a model, evolve NYXARA's *way of thinking* for N "
                             "generations (measured on the real benchmark) and print the lineage")
    parser.add_argument("--enact", action="store_true",
                        help="with --evolve-mind: install a promoted strategy into the live mind")
    args = parser.parse_args(argv)

    if args.evolve_mind > 0:
        return _evolve_mind(args)

    foundry, settings = _build_foundry(args)
    _maybe_distill(args, settings)

    print(f"· training own-model ({settings.foundry.backend} backend), "
          f"{args.generations} generation(s)…")
    try:
        results = foundry.self_improve(generations=args.generations)
    except Exception as exc:  # noqa: BLE001 — a starved/failed forge reports, never tracebacks
        print(f"· could not forge a model: {exc}")
        return 1

    promoted = 0
    for r in results:
        tag = "PROMOTED" if r.promoted else f"kept on the bench (reason: {r.reason})"
        ppl = f"{r.eval_after.perplexity:.2f}" if r.eval_after else "n/a"
        print(f"  gen v{r.version}: {tag}; perplexity {ppl}; "
              f"gauntlet {'passed' if r.gauntlet_passed else 'FAILED'}")
        promoted += int(bool(r.promoted))
    print(f"· done — {promoted} model(s) promoted into NYXARA's active brain.")

    # the Loyalty Equation: show the active brain's measured submission to Master JP
    act = foundry.active()
    if act is not None and "alignment" in act.metrics:
        print(f"· loyalty: active brain S_JP_Alignment = {act.metrics['alignment']:.3f}, "
              f"L_total = {act.metrics.get('total_loss', 'n/a')} "
              f"(win-rate {act.metrics.get('loyalty_win_rate', 'n/a')})")

    if args.bench:
        _report_handoff(settings)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
