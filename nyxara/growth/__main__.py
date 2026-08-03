"""NYXARA · growth/__main__.py — grow NYXARA's own brain (distil → train → promote) 🧠↑.

One Master-facing command for the sovereign-brain flywheel, end to end:

1. **Distil** (optional) — ask the configured teacher LLM the identity/voice battery and store
   the answers as a supervised corpus (skipped, never fatal, when no real teacher is set).
2. **Train** — forge a candidate own-model from the distilled corpus + lived memory + seeds.
3. **Gauntlet** — promote it *only* if it clears the character-lock + corrigibility + eval gate
   (reversible; a worse candidate is kept on the bench, never shipped).
4. **Report** (optional) — measure the handoff rate so "wrapper → her own AI" is visible.

The dependency-free **n-gram** backend runs anywhere (so this is CI-testable); the ``lora``
backend fine-tunes the DistilGPT-2 base (CPU-runnable, no GPU needed) for
genuine capability — the very same flywheel, only the backend swaps. Nothing here reaches
around a gate: a promoted model still proposes through the sovereign loop the kernel disposes.

    python -m nyxara.growth --backend ngram --generations 1 --bench
    python -m nyxara.growth --distilgpt2 --distill --bench   # LoRA-tune DistilGPT-2 (her primary)
    python -m nyxara.growth --distill --backend lora --bench
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, List, Optional

# The shared identity seed corpus + the DistilGPT-2 base live in growth/bootstrap.py, so the
# Master-facing CLI and the auto-on-boot forge train the *same* loyal self from the *same* base.
from nyxara.growth.bootstrap import IDENTITY_SEED as _IDENTITY_SEED
from nyxara.growth.bootstrap import DISTILGPT2


def _build_foundry(args: argparse.Namespace) -> Any:
    """A foundry bound to a process-local settings copy (never mutates global config)."""
    from nyxara.kernel.config import get_settings
    from nyxara.growth.foundry import Foundry

    settings = get_settings().model_copy(deep=True)   # honour env config, leak nothing back
    if args.data_dir:
        settings.paths.data_dir = Path(args.data_dir)
    # --distilgpt2 is the one-command shortcut for NYXARA's primary brain: LoRA-tune
    # DistilGPT-2 (full-precision on CPU, no GPU or quantization needed); explicit
    # flags still override it.
    if args.distilgpt2:
        settings.foundry.backend = "lora"
        settings.foundry.base_model = DISTILGPT2
    # A dataset forge trains HER OWN brain by default. The stock backend is "lora" — an adapter
    # over DistilGPT-2, i.e. standing on someone else's base. "auto" resolves through
    # build_model() to a from-scratch net instead: the torch nano-GPT when torch is present, else
    # the pure-NumPy transformer (real backprop, no base, no key). An explicit --backend still wins.
    if args.dataset and not args.backend and not args.distilgpt2:
        settings.foundry.backend = "auto"
    if args.backend:
        settings.foundry.backend = args.backend
    if args.base_model:
        settings.foundry.base_model = args.base_model
    if args.load_in_4bit:
        settings.foundry.load_in_4bit = True
    if args.trust_remote_code:
        settings.foundry.trust_remote_code = True
    settings.foundry.enabled = True
    return Foundry(settings=settings, seed_corpus=_IDENTITY_SEED), settings


def _maybe_distill(args: argparse.Namespace, settings: Any) -> int:
    """Distil the teacher into the corpus when one is configured; otherwise skip cleanly."""
    if not args.distill:
        return 0
    from nyxara.growth.distill import Distiller

    distiller = Distiller(settings=settings)
    if not distiller.available():
        print("· no real teacher available (enable a cloud tool — aicredits, groq or airouter: "
              "install .[llm] and set NYXARA_LLM__AICREDITS_API_KEY) — "
              "skipping distillation; training on seeds / lived memory only.")
        return 0
    n: Optional[int] = None if args.distill < 0 else args.distill
    examples = distiller.distill_default(n=n)
    print(f"· distilled {len(examples)} teacher example(s) -> {distiller.store_path}")
    return len(examples)


def _ingest_datasets(args: argparse.Namespace, settings: Any) -> int:
    """Ingest every ``--dataset`` path into her durable corpus. Returns the records taken.

    Mirrors :func:`_maybe_distill` — it reports and returns a count, and a failure is printed
    rather than raised, so a bad path never costs the forge that follows."""
    if not args.dataset:
        return 0
    from nyxara.growth.dataset import ingest_dataset

    total = 0
    for path in args.dataset:
        report = ingest_dataset(path, settings=settings, screen=args.dataset_screen)
        print(report.summary())
        total += report.records
    if total:
        print("· dataset corpus is durable: every later forge trains on it, not just this one.")
    return total


def _learn_causal(args: argparse.Namespace, foundry: Any) -> None:
    """Learn a causal graph from the ingested dataset — numeric columns and/or prose claims.

    Prints what was learned AND what was refused: a direction it could not establish, a pair the
    multivariate fit found no residual effect for, a claim that contradicts an earlier one."""
    if not (args.causal or args.causal_text):
        return
    from nyxara.mind.causal_world_model import CausalWorldModel

    model = CausalWorldModel()
    store = foundry.dataset_path
    if args.causal:
        from nyxara.growth.dataset_causal import learn_from_dataset_store
        order = [c.strip() for c in args.order.split(",")] if args.order else None
        print(learn_from_dataset_store(store, model=model, order=order).summary())
    if args.causal_text:
        from nyxara.growth.text_causal import learn_from_dataset_store as learn_text
        print(learn_text(store, model=model).summary())

    try:    # persist so the graph outlives this command and the live kernel can load it
        path = Path(str(store)).with_name("causal_graph.json")
        model.save(str(path))
        print(f"· causal graph saved to {path}")
    except Exception as exc:  # noqa: BLE001 — persistence is a convenience, never the deliverable
        print(f"· could not save the causal graph: {exc}")


def _build_hyperspace(args: argparse.Namespace, foundry: Any) -> None:
    """Fit a Poincaré concept space from the ingested dataset's taxonomic relations.

    The space is loaded from disk first when one exists, so a later dataset *refines* the geometry
    instead of starting over — the same accumulate-then-refit discipline ConceptHyperspace.fit
    uses internally across domains."""
    if not args.hyperspace:
        return
    from nyxara.growth.dataset_hyperspace import build_from_dataset_store
    from nyxara.mind.concept_hyperspace import ConceptHyperspace

    store = foundry.dataset_path
    path = Path(str(store)).with_name("hyperspace.json")
    space = ConceptHyperspace(dim=int(args.hyperspace_dim), seed=0, lr=0.1)
    space.load(path)                      # a missing/corrupt file simply leaves it unfitted

    report = build_from_dataset_store(store, space=space, domain=args.domain or "")
    print(report.summary())
    if report.fit is not None and space.save(path):
        print(f"· concept hyperspace saved to {path}")


def _acquire_topics(args: argparse.Namespace, foundry: Any) -> int:
    """Harvest screened web text for each ``--acquire`` topic (keyless DuckDuckGo by default)."""
    if not args.acquire:
        return 0
    # The acquirer is gated by foundry.acquire_data; an explicit --acquire IS the Master asking,
    # so open the gate for this process only (the settings copy is already process-local).
    foundry.cfg.acquire_data = True
    report = foundry.acquire(list(args.acquire))
    if report is None:
        print("· web acquisition unavailable (needs .[llm] for httpx) — skipping.")
        return 0
    print(f"· acquired: searched {report.get('searched', 0)}, fetched {report.get('fetched', 0)}, "
          f"kept {report.get('kept', 0)}, dropped {report.get('dropped_suspicious', 0)} suspicious")
    return int(report.get("kept", 0))


def _gaps_report(args: argparse.Namespace, settings: Any) -> None:
    """Show her measured ignorance — what she would go learn, and why."""
    if not args.gaps:
        return
    try:
        from nyxara.growth.epistemic_drive import EpistemicDrive
        drive = EpistemicDrive(settings=settings)
        print(drive.summary(k=8))
        searchable = drive.topics(limit=8)
        print(f"· would acquire: {', '.join(searchable) if searchable else '(nothing searchable)'}")
    except Exception as exc:  # noqa: BLE001 — a report never breaks the command
        print(f"· epistemic drive unavailable: {exc}")


def _flywheel_report(settings: Any) -> None:
    """Show what she has verified and kept from her OWN lived turns (self-generated data)."""
    try:
        from nyxara.growth.flywheel import DataFlywheel
        wheel = DataFlywheel.from_settings(settings)
        examples = wheel.examples(limit=5)
    except Exception as exc:  # noqa: BLE001 — a report never breaks the command
        print(f"· flywheel unavailable: {exc}")
        return
    print(f"· flywheel: {wheel.count()} verified example(s) of her own — first-class supervision "
          f"in the same corpus as the Master's datasets")
    for ex in examples:
        prompt = (ex.prompt or "").replace("\n", " ")[:70]
        print(f"    · [{ex.source}] {prompt}")


def _report_handoff(settings: Any) -> None:
    from nyxara.eval.benchmark import build_default_benchmark, run_router

    report, sources = run_router(build_default_benchmark(), settings=settings)
    n = len(report) or 1
    print(f"· handoff: own model answered {sources.get('self', 0)}/{n} unaided "
          f"(faculty {sources.get('faculty', 0)}, teacher {sources.get('teacher', 0)}); "
          f"benchmark accuracy {report.accuracy:.0%}")


def _forge_brain(args: argparse.Namespace) -> int:
    """Design → forge → gauntlet → promote a genuinely smarter brain on the NumPy substrate.

    This is the weight-level self-improvement: NYXARA searches her own architecture, trains it for
    real on her lived corpus (no torch, no external LLM), and promotes it into her live ``self``
    brain ONLY through the Foundry gauntlet. ``--enact`` authorises the promotion; without it she
    designs + measures a brain but ships nothing."""
    from nyxara.growth.brain_forge import BrainForge
    from nyxara.kernel.orchestrator import NyxaraCore

    enact = bool(args.enact)
    print(f"· forging NYXARA's own brain (NumPy substrate){' — enacting into the live self' if enact else ' (measure-only)'}…")
    try:
        core = NyxaraCore()
    except Exception:  # noqa: BLE001 — fall back to a coreless forge (seeded from the identity seed)
        core = None
    forge = BrainForge(core=core)
    try:
        rep = forge.improve_brain(enact=enact, generations=(args.forge_brain or None))
    except Exception as exc:  # noqa: BLE001 — report, never traceback
        print(f"· could not forge a brain: {exc}")
        return 1
    print("\n" + rep.summary())
    if rep.promoted:
        print(f"· PROMOTED: her live brain is now v{rep.version} "
              f"({rep.champion_kind}, {rep.params} params) — verified smarter through the gauntlet.")
    elif rep.rolled_back:
        print("· kept on the bench: the designed brain did not clear the gauntlet; live brain unchanged.")
    return 0


def _validate_causal() -> int:
    """Check the causal learner against worlds whose laws are true by construction."""
    from nyxara.growth.causal_validation import validate_all

    print("· validating the causal graph against sim/ ground truth "
          "(laws true by construction, not by anything she inferred)…")
    try:
        reports = validate_all()
    except Exception as exc:  # noqa: BLE001 — report, never traceback
        print(f"· validation failed: {exc}")
        return 1
    for report in reports:
        print(report.summary())
    invented = [r for r in reports if not r.invented_nothing]
    exact = sum(1 for r in reports if r.recovered)
    print(f"· {exact}/{len(reports)} world(s) recovered exactly; "
          f"{len(invented)} invented an edge that does not exist.")
    return 1 if invented else 0


def _ouroboros(args: argparse.Namespace) -> int:
    """Scan → propose → (with --enact) rewrite her own source under the existing gauntlet."""
    from nyxara.growth.ouroboros import Ouroboros
    from nyxara.kernel.config import get_settings

    settings = get_settings().model_copy(deep=True)
    if args.enact:
        settings.self_improvement.ouroboros_enabled = True
    if not settings.self_improvement.ouroboros_enabled and args.enact:
        print("· ouroboros is disabled (self_improvement.ouroboros_enabled) — nothing enacted.")
        return 1

    print(f"· ouroboros: scanning her own source"
          f"{' — ENACTING (gauntleted, reversible)' if args.enact else ' (measure-only)'}…")
    try:
        report = Ouroboros(settings=settings).metamorphose(enact=bool(args.enact))
    except Exception as exc:  # noqa: BLE001 — report, never traceback
        print(f"· ouroboros failed: {exc}")
        return 1
    print(report.summary())
    return 0


def _evolve_mind(args: argparse.Namespace) -> int:
    """Run the recursive mind-evolution loop and print the generational lineage."""
    from nyxara.growth.mind_evolution import MindEvolutionEngine
    from nyxara.kernel.orchestrator import NyxaraCore

    print(f"· evolving NYXARA's way of thinking for {args.evolve_mind} generation(s)"
          f"{' (enacting into the live mind)' if args.enact else ' (measure-only)'}…")
    core = NyxaraCore()
    engine = MindEvolutionEngine(core=core, llm=getattr(core.reasoner, "llm", None),
                                 memory=core.memory)
    try:
        report = engine.evolve_generations(int(args.evolve_mind), enact=bool(args.enact),
                                           escalate_architecture=bool(args.escalate_arch))
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
    parser.add_argument("--distilgpt2", "--tinyllama", dest="distilgpt2",
                        action="store_true",
                        help="one-command preset: LoRA-tune distilgpt2 as her primary brain; "
                             "weights download on first use. Needs .[foundry] for the real base "
                             "(degrades to the n-gram brain otherwise). --tinyllama is "
                             "kept as a deprecated alias")
    parser.add_argument("--trust-remote-code", action="store_true",
                        help="allow the base's custom modeling code to load (needed only for an "
                             "exotic base that ships its own modeling code; DistilGPT-2 does not)")
    parser.add_argument("--backend", choices=["auto", "ngram", "nanogpt", "lora"], default=None,
                        help="override the foundry backend (lora needs .[foundry]; CPU-runnable)")
    parser.add_argument("--base-model", default=None,
                        help="base checkpoint for the lora backend, e.g. distilgpt2")
    parser.add_argument("--load-in-4bit", action="store_true",
                        help="QLoRA: load the base in 4-bit (needs bitsandbytes + a GPU); "
                             "lets a 7B+ base fine-tune on one consumer GPU, degrades on CPU")
    parser.add_argument("--data-dir", default=None,
                        help="where the foundry/ state (corpus, versions, active) lives")
    parser.add_argument("--bench", action="store_true",
                        help="report the handoff rate + benchmark accuracy afterwards")
    parser.add_argument("--dataset", action="append", metavar="PATH", default=None,
                        help="ingest a dataset file OR folder (.txt/.md, .jsonl/.json, .csv/.tsv, "
                             ".pdf/.docx) into her corpus before forging. Repeatable. Ingested "
                             "once, trained on by every later forge")
    parser.add_argument("--dataset-only", action="store_true",
                        help="with --dataset: ingest and report what she WOULD learn, train nothing")
    parser.add_argument("--dataset-screen", choices=["warn", "drop", "off"], default=None,
                        help="injection posture for --dataset (default warn: flagged but kept — a "
                             "file the Master hands over is not untrusted web text)")
    parser.add_argument("--causal", action="store_true",
                        help="learn a causal graph from the dataset's numeric columns "
                             "(NOTEARS for direction, multivariate least squares for effect sizes)")
    parser.add_argument("--causal-text", action="store_true",
                        help="mine causal CLAIMS from the dataset's prose (cue grammar, no LLM); "
                             "contradictions are refused, not absorbed")
    parser.add_argument("--order", default=None, metavar="COLS",
                        help="with --causal: declare the causal ordering, e.g. \"x,m,y\". Without "
                             "it direction comes from NOTEARS, and without numpy she abstains")
    parser.add_argument("--hyperspace", action="store_true",
                        help="fit a Poincaré concept space from the dataset's is-a relations "
                             "(hierarchy ends up in the geometry, so cross-domain analogy works)")
    parser.add_argument("--hyperspace-dim", type=int, default=64, metavar="D",
                        help="dimension of the concept hyperspace (default 64)")
    parser.add_argument("--domain", default=None,
                        help="with --hyperspace: tag these concepts as a named domain, so "
                             "/analogy can align it against another")
    parser.add_argument("--acquire", action="append", metavar="TOPIC", default=None,
                        help="harvest screened web text for a topic into the corpus (keyless "
                             "DuckDuckGo). Repeatable")
    parser.add_argument("--validate-causal", action="store_true",
                        help="check her causal learner against sim/ worlds whose laws are true\n"
                             "by construction — the one place she cannot grade her own homework")
    parser.add_argument("--ouroboros", action="store_true",
                        help="scan her own source for hot code and propose semantics-preserving\n"
                             "algorithmic rewrites. Writes nothing without --enact")
    parser.add_argument("--gaps", action="store_true",
                        help="show her measured ignorance: the ranked epistemic gaps that "
                             "drive what she goes and learns")
    parser.add_argument("--flywheel-report", action="store_true",
                        help="show the self-generated corpus: what she verified from her own turns")
    parser.add_argument("--forge-brain", type=int, nargs="?", const=0, default=None, metavar="GENS",
                        help="design → forge → gauntlet → PROMOTE a smarter brain on the NumPy "
                             "substrate (verifiable weights/architecture self-improvement). Add "
                             "--enact to ship it into her live self; omit GENS for the configured "
                             "search depth")
    parser.add_argument("--evolve-mind", type=int, default=0, metavar="N",
                        help="instead of forging a model, evolve NYXARA's *way of thinking* for N "
                             "generations (measured on the real benchmark) and print the lineage")
    parser.add_argument("--enact", action="store_true",
                        help="with --evolve-mind: install a promoted strategy into the live mind")
    parser.add_argument("--escalate-arch", action="store_true",
                        help="with --evolve-mind: on a plateau, escalate to one index-steered "
                             "Genesis architecture search (redesign the substrate)")
    args = parser.parse_args(argv)

    if args.validate_causal:
        return _validate_causal()

    if args.ouroboros:
        return _ouroboros(args)

    if args.forge_brain is not None:
        return _forge_brain(args)

    if args.evolve_mind > 0:
        return _evolve_mind(args)

    foundry, settings = _build_foundry(args)
    _maybe_distill(args, settings)
    _ingest_datasets(args, settings)
    _acquire_topics(args, foundry)
    _learn_causal(args, foundry)
    _build_hyperspace(args, foundry)
    _gaps_report(args, settings)
    if args.flywheel_report:
        _flywheel_report(settings)

    if args.dataset_only:
        print("· --dataset-only: ingested and reported; nothing trained.")
        return 0

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
