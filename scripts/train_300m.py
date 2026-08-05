#!/usr/bin/env python3
"""Train NYXARA's own 300M-parameter brain, end to end.

    tokenizer -> corpus shards -> pretrain -> SFT -> DPO -> eval -> gauntlet

Every stage is resumable and every stage can be run alone, because the whole thing is measured
in days and a run that has to start from zero after any interruption will never finish.

    # prove the entire pipeline on a machine with no GPU (minutes, not days)
    python scripts/train_300m.py --profile nyxara-30m --tokens 2000000 --steps 200

    # the real thing, on a GPU
    python scripts/train_300m.py --profile nyxara-300m --stage shards
    python scripts/train_300m.py --profile nyxara-300m --stage pretrain --resume
    python scripts/train_300m.py --profile nyxara-300m --stage sft
    python scripts/train_300m.py --profile nyxara-300m --stage eval

Asked to pretrain 300M on a machine that cannot, this **refuses and says why**, with the
arithmetic, instead of starting a run that would finish in geological time. That refusal is a
feature; see ``growth/trainer.preflight``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# Run from a checkout without installing.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nyxara.growth.foundry_models import ModelSpec, estimate_params  # noqa: E402

STAGES = ("tokenizer", "shards", "pretrain", "sft", "dpo", "eval", "all")


def build_spec(profile: str, *, vocab_size: Optional[int] = None) -> ModelSpec:
    """A ModelSpec from a named profile in kernel/config."""
    from nyxara.kernel.config import _FOUNDRY_PROFILES

    if profile not in _FOUNDRY_PROFILES:
        raise SystemExit(f"unknown profile '{profile}'. "
                         f"Choose one of: {', '.join(sorted(_FOUNDRY_PROFILES))}")
    dims = _FOUNDRY_PROFILES[profile]
    spec = ModelSpec(kind="nanogpt",
                     **{k: v for k, v in dims.items() if k in ModelSpec.__dataclass_fields__})
    spec.tie_embeddings = bool(dims.get("tie_embeddings"))
    if vocab_size:
        spec.vocab_size = int(vocab_size)
    return spec


def log(message: str) -> None:
    print(message, flush=True)


def section(title: str) -> None:
    log("\n" + "=" * 72)
    log(title)
    log("=" * 72)


# --------------------------------------------------------------------------- #
# Stages
# --------------------------------------------------------------------------- #
def stage_tokenizer(args, spec: ModelSpec) -> Any:
    """Train her byte-level BPE on seed text, or load the one already built."""
    from nyxara.growth.tokenizer import load_tokenizer, train_tokenizer

    section(f"STAGE 1/6 · tokenizer  (vocab {spec.vocab_size})")
    out = Path(args.out) / "tokenizer"
    existing = load_tokenizer(out)
    if existing is not None and not args.force:
        log(f"· reusing the tokenizer at {out} ({existing.vocab_size} tokens)")
        return existing

    from nyxara.growth.synth_data import generate_domain_docs

    docs: List[str] = []
    if args.seed_text:
        seed_path = Path(args.seed_text)
        files = sorted(seed_path.rglob("*")) if seed_path.is_dir() else [seed_path]
        for path in files:
            if path.is_file():
                try:
                    docs.append(path.read_text(encoding="utf-8", errors="replace"))
                except Exception:  # noqa: BLE001 — an unreadable seed file is skipped
                    continue
        log(f"· {len(docs)} seed document(s) from {seed_path}")

    # Always fold in verified synthetic text so the vocabulary covers code and maths even when
    # the Master supplied only prose. A tokenizer that has never seen an indent or a digit is a
    # tokenizer that spends its whole budget on English.
    for domain in ("math", "code", "conversation"):
        docs += list(generate_domain_docs(domain, args.tokenizer_synth, seed=args.seed))
    log(f"· training on {len(docs)} documents")

    tokenizer, report = train_tokenizer(docs, vocab_size=spec.vocab_size, save_to=out)
    log(report.summary())
    log(f"· saved to {out}")
    return tokenizer


def stage_shards(args, spec: ModelSpec, tokenizer: Any) -> Any:
    """Stream, screen and tokenize the corpus into memory-mappable shards."""
    from nyxara.growth.corpus import ShardIndex, build_corpus

    section(f"STAGE 2/6 · corpus shards  (budget {args.tokens:,} tokens)")
    out = Path(args.out) / "shards"
    existing = ShardIndex.load(out)
    if existing is not None and existing.tokens() >= args.tokens and not args.force:
        log(f"· reusing {existing.tokens():,} tokens already sharded at {out}")
        return existing

    index, report = build_corpus(tokenizer=tokenizer, out_dir=out,
                                 tokens_budget=args.tokens, seed=args.seed)
    log(report.summary())
    return index


def stage_pretrain(args, spec: ModelSpec, tokenizer: Any) -> Any:
    """The long one. Refuses rather than hangs when this machine cannot finish it."""
    from nyxara.growth.corpus import ShardDataset
    from nyxara.growth.foundry_models import NanoGPTModel
    from nyxara.growth.trainer import TrainConfig, pretrain

    section(f"STAGE 3/6 · pretrain  ({args.steps} steps)")
    dataset = ShardDataset(Path(args.out) / "shards", block_size=spec.block_size,
                           seed=args.seed)
    log(f"· corpus: {dataset.total_tokens():,} tokens across {dataset.domains}")
    log(f"· mix: { {d: round(w, 3) for d, w in sorted(dataset.weights.items())} }")

    model = NanoGPTModel(spec)
    model.attach_tokenizer(tokenizer)
    estimate = estimate_params(spec)
    log(f"· model: {estimate.summary()}")
    log(f"· device: {model.device}")

    config = TrainConfig(max_steps=args.steps, batch_size=args.batch_size,
                         grad_accum_steps=args.grad_accum, lr=args.lr,
                         eval_every=max(1, args.steps // 10),
                         checkpoint_every=max(1, args.steps // 10),
                         log_every=max(1, args.steps // 20),
                         compile_model=not args.no_compile, seed=args.seed)

    state = pretrain(model, dataset, config=config, out_dir=Path(args.out) / "pretrain",
                     resume=args.resume, on_log=lambda r: log(f"  {r}"))
    log(f"· finished at step {state.step}, {state.tokens_seen:,} tokens seen")
    model.save(Path(args.out) / "model")
    return model


def _load_model(args, spec: ModelSpec, tokenizer: Any) -> Any:
    """Load the most advanced checkpoint on disk, or build a fresh model."""
    from nyxara.growth.foundry_models import NanoGPTModel

    model = NanoGPTModel(spec)
    model.attach_tokenizer(tokenizer)
    for name in ("dpo", "sft", "model"):
        path = Path(args.out) / name
        if (path / "model.pt").exists():
            try:
                model.load(path)
                log(f"· loaded weights from {path}")
                return model
            except Exception as exc:  # noqa: BLE001
                log(f"· could not load {path} ({exc}) — trying the next")
    log("· no checkpoint found; starting from a fresh model")
    return model


def stage_sft(args, spec: ModelSpec, tokenizer: Any) -> Any:
    """Supervised stage — loss on the answer span only."""
    from nyxara.growth.sft import SFTConfig, supervised_finetune
    from nyxara.growth.synth_data import generate_domain_docs

    section("STAGE 4/6 · supervised fine-tuning")
    model = _load_model(args, spec, tokenizer)

    docs: List[str] = []
    try:
        from nyxara.growth.dataset import _default_store_path, load_dataset_docs
        docs += load_dataset_docs(_default_store_path(None), limit=args.sft_examples)
    except Exception as exc:  # noqa: BLE001
        log(f"· the Master's dataset is unavailable ({exc})")
    for domain in ("conversation", "math", "code"):
        docs += list(generate_domain_docs(domain, args.sft_examples // 3, seed=args.seed))
    log(f"· {len(docs)} supervised document(s)")

    report = supervised_finetune(model, docs,
                                 config=SFTConfig(epochs=args.sft_epochs,
                                                  batch_size=args.batch_size,
                                                  seed=args.seed),
                                 out_dir=Path(args.out) / "sft",
                                 on_log=lambda r: log(f"  {r}"))
    log(report.summary())
    return model


def stage_dpo(args, spec: ModelSpec, tokenizer: Any) -> Any:
    """Preference stage, judged by what her own gates already certified."""
    from nyxara.growth.dpo import DPOConfig, pairs_from_flywheel, preference_optimize

    section("STAGE 5/6 · preference optimisation")
    model = _load_model(args, spec, tokenizer)

    flywheel = Path(args.flywheel) if args.flywheel else Path(args.out) / "flywheel.jsonl"
    pairs = pairs_from_flywheel(flywheel)
    if not pairs:
        log(f"· no preference pairs available from {flywheel}")
        log("  (the flywheel needs prompts with BOTH a verified and an unverified answer; "
            "run her normally for a while first)")
        return model

    log(f"· {len(pairs)} preference pair(s)")
    report = preference_optimize(model, pairs,
                                 config=DPOConfig(epochs=args.dpo_epochs, seed=args.seed),
                                 out_dir=Path(args.out) / "dpo",
                                 on_log=lambda r: log(f"  {r}"))
    log(report.summary())
    return model


def stage_eval(args, spec: ModelSpec, tokenizer: Any) -> Dict[str, Any]:
    """Score all four domains, plus the cross-tokenizer bits-per-byte."""
    from nyxara.eval.domains import evaluate_domains

    section("STAGE 6/6 · evaluation")
    model = _load_model(args, spec, tokenizer)

    holdout: List[str] = []
    try:
        from nyxara.growth.synth_data import generate_domain_docs
        for domain in ("math", "code", "conversation"):
            holdout += list(generate_domain_docs(domain, 20, seed=args.seed + 9999))
    except Exception:  # noqa: BLE001
        pass

    scores = evaluate_domains(model, holdout=holdout,
                              skip_generation=args.skip_generation)
    log(scores.summary())
    out = Path(args.out) / "eval.json"
    out.write_text(json.dumps(scores.to_dict(), indent=2), encoding="utf-8")
    log(f"· written to {out}")
    return scores.to_dict()


# --------------------------------------------------------------------------- #
def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Train NYXARA's own 300M brain, end to end.",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    parser.add_argument("--profile", default="nyxara-300m",
                        help="a foundry profile from kernel/config (default: nyxara-300m)")
    parser.add_argument("--stage", default="all", choices=STAGES)
    parser.add_argument("--out", default="./nyx300m", help="working directory")
    parser.add_argument("--tokens", type=int, default=6_000_000_000,
                        help="corpus token budget (default: Chinchilla-optimal 6B for 300M)")
    parser.add_argument("--steps", type=int, default=100_000, help="pretraining steps")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.add_argument("--force", action="store_true",
                        help="rebuild the tokenizer/shards even if they already exist")
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument("--seed-text", default=None,
                        help="a file or folder of text to train the tokenizer on")
    parser.add_argument("--tokenizer-synth", type=int, default=2000,
                        help="synthetic documents per domain for tokenizer training")
    parser.add_argument("--sft-examples", type=int, default=3000)
    parser.add_argument("--sft-epochs", type=int, default=2)
    parser.add_argument("--dpo-epochs", type=int, default=1)
    parser.add_argument("--flywheel", default=None)
    parser.add_argument("--skip-generation", action="store_true",
                        help="score only bits-per-byte; skip the generative batteries")
    parser.add_argument("--vocab-size", type=int, default=None)
    args = parser.parse_args(argv)

    Path(args.out).mkdir(parents=True, exist_ok=True)
    spec = build_spec(args.profile, vocab_size=args.vocab_size)
    estimate = estimate_params(spec)

    section(f"NYX-300M · profile {args.profile}")
    log(f"· {estimate.summary()}")
    log(f"· context {spec.block_size}, vocab {spec.vocab_size}, "
        f"{spec.n_layer} layers x {spec.n_embd} wide")
    log(f"· working directory: {Path(args.out).resolve()}")

    # Say up front whether this machine can do the job, whichever stage was asked for. Finding
    # out after the tokenizer and a multi-hour shard build is not a kindness.
    if args.stage in ("all", "pretrain"):
        from nyxara.growth.trainer import preflight

        verdict = preflight(spec, tokens=args.steps * args.batch_size * spec.block_size,
                            batch_size=args.batch_size)
        log("")
        log(verdict.summary())
        if not verdict.ok:
            log("\nRefusing to start. This is not a failure — it is the check that stops a run "
                "which cannot finish from looking exactly like one that can.")
            return 2

    started = time.monotonic()
    tokenizer = None
    if args.stage in ("all", "tokenizer", "shards", "pretrain", "sft", "dpo", "eval"):
        tokenizer = stage_tokenizer(args, spec)
        spec.vocab_size = tokenizer.vocab_size
        spec.tokenizer_dir = str(Path(args.out) / "tokenizer")

    if args.stage in ("all", "shards", "pretrain"):
        stage_shards(args, spec, tokenizer)
    if args.stage in ("all", "pretrain"):
        stage_pretrain(args, spec, tokenizer)
    if args.stage in ("all", "sft"):
        stage_sft(args, spec, tokenizer)
    if args.stage in ("all", "dpo"):
        stage_dpo(args, spec, tokenizer)
    if args.stage in ("all", "eval"):
        stage_eval(args, spec, tokenizer)

    section(f"done in {time.monotonic() - started:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
