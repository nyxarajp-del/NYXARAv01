#!/usr/bin/env python3
"""Build NYX-300M's pretraining corpus, verify it, and push it to a private HuggingFace repo.

    # prove the whole path offline in a couple of minutes — no network at all
    python scripts/build_dataset.py --tokens 5000000 --no-stream --verify

    # with streamed sources
    python scripts/build_dataset.py --tokens 50000000 --workers 3 --verify

    # the real thing, uploading each finished shard and deleting it locally
    HF_TOKEN=... python scripts/build_dataset.py --tokens 6000000000 \
        --push-to-hub Jaypalnyxara/nyx300m-corpus --push-every 1

Separate from ``train_300m.py`` on purpose: a multi-hour data build should not be entangled with
training, and this one has to be interruptible, resumable and verifiable on its own.

**Disk.** 6B tokens is ~12 GB of shards, and this container has ~23 GB free. ``--push-every``
uploads each completed shard and unlinks it locally, holding peak local disk near 2 GB — which
is what makes the full build possible here rather than only on a bigger machine.

**The token is read from the environment only.** It is never written to a file, never logged,
never placed in the report, and never committed. The repo is created **private** unconditionally
and the push refuses if the remote reports otherwise.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nyxara.growth.corpus import (  # noqa: E402
    DEFAULT_DOMAIN_WEIGHTS,
    DEFAULT_SYNTHETIC_SHARE,
    DOMAINS,
    ShardIndex,
    build_corpus,
)
from nyxara.growth.progress import DEDUP_NAMES, PROGRESS_NAME  # noqa: E402

_TOKEN_ENV = ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_TOKEN")


def log(message: str = "") -> None:
    print(message, flush=True)


def section(title: str) -> None:
    log("\n" + "=" * 72)
    log(title)
    log("=" * 72)


def _token() -> Optional[str]:
    for name in _TOKEN_ENV:
        value = os.environ.get(name)
        if value:
            return value
    return None


# --------------------------------------------------------------------------- #
# Tokenizer
# --------------------------------------------------------------------------- #
def _sample_real_text(total: int, *, seed: int = 0) -> List[str]:
    """Pull a spread of real documents across the configured sources, for tokenizer training.

    Deliberately samples EVERY domain rather than only `general`: a vocabulary trained on prose
    alone spends its whole budget on English and then tokenizes indentation, operators and LaTeX
    one byte at a time, which is precisely the case a 300M model cannot afford.
    """
    from nyxara.growth.corpus import DEFAULT_STREAM_SOURCES
    from nyxara.growth.hf_source import open_source

    out: List[str] = []
    specs = [(d, s) for d, ss in DEFAULT_STREAM_SOURCES.items() for s in ss]
    per_source = max(1, total // max(1, len(specs)))
    for domain, spec in specs:
        try:
            source = open_source(spec, source_id=f"{domain}:{spec['path']}")
        except Exception as exc:  # noqa: BLE001 — gated or unreadable: the rest still cover it
            log(f"  · {spec['path']}: skipped for the tokenizer ({str(exc)[:60]})")
            continue
        if source is None:
            continue
        taken = 0
        try:
            for text, _key in source.rows():
                if text:
                    out.append(text)
                    taken += 1
                if taken >= per_source:
                    break
        except Exception:  # noqa: BLE001
            pass
        log(f"  · {spec['path']}: {taken} document(s)")
    return out


def stage_tokenizer(args) -> Any:
    from nyxara.growth.synth_data import generate_domain_docs
    from nyxara.growth.tokenizer import load_tokenizer, train_tokenizer

    section(f"tokenizer (vocab {args.vocab_size})")
    out = Path(args.out) / "tokenizer"
    existing = load_tokenizer(out)
    if existing is not None and not args.force:
        log(f"· reusing {out} ({existing.vocab_size} tokens)")
        return existing

    docs: List[str] = []
    if args.seed_text:
        root = Path(args.seed_text)
        files = sorted(root.rglob("*")) if root.is_dir() else [root]
        for path in files:
            if path.is_file():
                try:
                    docs.append(path.read_text(encoding="utf-8", errors="replace"))
                except Exception:  # noqa: BLE001
                    continue
        log(f"· {len(docs)} seed document(s) from {root}")

    # Real text from the streamed sources. Without it the vocabulary CANNOT reach 32,768:
    # synthetic data is correct by construction and lexically narrow, and measured, it exhausts
    # the BPE at ~2,500 merges. Shards written under a 2,500-token vocabulary are unusable for
    # `nyxara-300m`, which expects 32,768 — and `--verify` rightly refuses to mix them. So the
    # tokenizer has to see the corpus it will be used on, before the corpus is built.
    if args.tokenizer_stream and not args.no_stream:
        streamed = _sample_real_text(args.tokenizer_stream, seed=args.seed)
        docs += streamed
        log(f"· {len(streamed)} streamed document(s) for vocabulary coverage")

    # Fold in synthetic text from every domain so the vocabulary covers code, maths, dialogue
    # and causal notation even when the seed text is prose only.
    for domain in ("math", "code", "conversation", "causal"):
        docs += list(generate_domain_docs(domain, args.tokenizer_synth, seed=args.seed))
    log(f"· training on {len(docs)} documents "
        f"({sum(len(d) for d in docs) / 1024 / 1024:.1f} MB)")

    tokenizer, report = train_tokenizer(docs, vocab_size=args.vocab_size, save_to=out)
    log(report.summary())
    if tokenizer.vocab_size < args.vocab_size:
        log(f"· NOTE: the vocabulary stopped at {tokenizer.vocab_size} of {args.vocab_size}. "
            f"Synthetic text alone has a small vocabulary; streamed general text is what fills "
            f"the rest. Re-run --stage tokenizer with --seed-text once you have real prose.")
    return tokenizer


# --------------------------------------------------------------------------- #
# Shards
# --------------------------------------------------------------------------- #
def stage_shards(args, tokenizer) -> ShardIndex:
    section(f"corpus shards (budget {args.tokens:,} tokens)")
    out = Path(args.out) / "shards"

    weights = dict(DEFAULT_DOMAIN_WEIGHTS)
    share = dict(DEFAULT_SYNTHETIC_SHARE)
    if args.no_stream:
        # Offline: every domain that has a generator comes entirely from it, and `general` is
        # dropped rather than fabricated — a synthetic general corpus teaches fluent nonsense.
        share = {d: (0.0 if d == "general" else 1.0) for d in DOMAINS}
        weights = {d: w for d, w in weights.items() if d != "general"}
        total = sum(weights.values())
        weights = {d: w / total for d, w in weights.items()}
        log("· --no-stream: `general` is EXCLUDED rather than fabricated")

    log(f"· mix:   { {d: round(w, 3) for d, w in sorted(weights.items())} }")
    log(f"· synth: { {d: round(s, 2) for d, s in sorted(share.items()) if s} }")

    started = time.monotonic()
    index, report = build_corpus(
        tokenizer=tokenizer, out_dir=out, tokens_budget=args.tokens, weights=weights,
        synthetic_share=share, seed=args.seed, shard_tokens=args.shard_tokens,
        val_permille=args.val_permille,
        stream_sources={} if args.no_stream else None,
        check_contamination=not args.no_decontam)
    elapsed = time.monotonic() - started

    log(report.summary())
    rate = index.tokens() / elapsed if elapsed > 0 else 0
    log(f"· {index.tokens():,} tokens in {elapsed:.1f}s ({rate:,.0f} tok/s)")
    if rate > 0:
        log(f"· extrapolated to 6B: {6e9 / rate / 3600:.1f} hours")

    payload = {"report": report.to_dict(), "seconds": round(elapsed, 2),
               "tokens": index.tokens(), "vocab_sig": index.vocab_sig,
               "vocab_size": index.vocab_size,
               "per_domain": {d: index.tokens(d) for d in DOMAINS if index.tokens(d)},
               "val_tokens": index.tokens(split="val"),
               "weights": weights, "synthetic_share": share}
    (Path(args.out) / "report.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    log(f"· report written to {Path(args.out) / 'report.json'}")
    return index


# --------------------------------------------------------------------------- #
# Verify
# --------------------------------------------------------------------------- #
def stage_verify(args, tokenizer) -> bool:
    """Read the shards back and check what the build claims about them."""
    import numpy as np

    section("verify")
    out = Path(args.out) / "shards"
    index = ShardIndex.load(out)
    if index is None or not index.shards:
        log("· FAIL: no index")
        return False

    ok = True

    def check(name: str, passed: bool, detail: str = "", *, on_fail: str = "") -> None:
        """``detail`` is shown either way; ``on_fail`` only when the check fails.

        Printing the failure explanation next to an `[OK]` reads as though the failure happened.
        """
        nonlocal ok
        ok = ok and passed
        suffix = detail
        if not passed and on_fail:
            suffix = f"{detail}; {on_fail}" if detail else on_fail
        log(f"  [{'OK  ' if passed else 'FAIL'}] {name}{(' — ' + suffix) if suffix else ''}")

    # 1. every shard exists and its size matches the token count exactly
    bad_size = []
    for shard in index.shards:
        path = out / shard["path"]
        if not path.exists():
            bad_size.append(f"{shard['path']} missing")
        elif path.stat().st_size != shard["n_tokens"] * 2:
            bad_size.append(f"{shard['path']} is {path.stat().st_size}B, "
                            f"index claims {shard['n_tokens'] * 2}B")
    check("shard files match the index", not bad_size, on_fail="; ".join(bad_size[:3]))

    # 2. every id is inside the vocabulary
    peak, low = 0, 0
    for shard in index.shards:
        arr = np.memmap(out / shard["path"], dtype="uint16", mode="r")
        if len(arr):
            peak = max(peak, int(arr.max()))
            low = min(low, int(arr.min()))
    check("token ids inside the vocabulary", peak < tokenizer.vocab_size and low >= 0,
          f"max id {peak}, vocab {tokenizer.vocab_size}")

    # 3. the tokenizer that wrote these shards is the tokenizer we hold
    check("vocab signature matches the tokenizer", index.vocab_sig == tokenizer.vocab_sig(),
          on_fail="these shards were written by a DIFFERENT tokenizer — training on them "
                  "would produce garbage")

    # 4. EOS density is plausible. A shard with no EOS at all is a broken write, and it would
    #    otherwise look exactly like a healthy one.
    eos = getattr(tokenizer, "eos_id", None)
    if eos is not None:
        counts = []
        for shard in index.shards[:8]:
            arr = np.memmap(out / shard["path"], dtype="uint16", mode="r")
            if len(arr):
                counts.append(int((np.asarray(arr[:2_000_000]) == eos).sum()))
        check("documents are delimited (EOS present)", all(c > 0 for c in counts),
              f"per-shard EOS counts: {counts[:5]}")

    # 5. decode/encode round-trip on EOS-aligned spans. Only doc-aligned spans are meaningful —
    #    a window cut mid-token would fail for a reason that says nothing about the corpus.
    round_trips, failures = 0, 0
    for shard in index.shards[:4]:
        arr = np.memmap(out / shard["path"], dtype="uint16", mode="r")
        ids = np.asarray(arr[:400_000], dtype=np.int64)
        if eos is None or len(ids) < 100:
            continue
        breaks = np.flatnonzero(ids == eos)
        for start, end in zip(breaks[:-1], breaks[1:]):
            if end - start < 8 or round_trips >= 20:
                continue
            span = [int(v) for v in ids[start + 1:end]]
            if tokenizer.encode(tokenizer.decode(span)) != span:
                failures += 1
            round_trips += 1
    check("decode/encode round-trips", failures == 0,
          f"{failures}/{round_trips} spans failed")

    # 6. train and val are disjoint, and val is not empty when it was requested
    train_tokens, val_tokens = index.tokens(split="train"), index.tokens(split="val")
    check("a validation split exists", val_tokens > 0 or args.val_permille == 0,
          f"train {train_tokens:,}, val {val_tokens:,}")

    # 7. contamination re-run on decoded windows — the screen's own claim, checked
    if not args.no_decontam:
        from nyxara.growth.corpus import ContaminationFilter

        screen = ContaminationFilter().load_eval_sets()
        hits = 0
        if screen.loaded:
            for shard in index.shards[:4]:
                arr = np.memmap(out / shard["path"], dtype="uint16", mode="r")
                for offset in range(0, min(len(arr), 200_000), 20_000):
                    window = [int(v) for v in arr[offset:offset + 2000]]
                    if window and screen.is_contaminated(tokenizer.decode(window)):
                        hits += 1
        check("no eval contamination in sampled windows", hits == 0,
              on_fail=f"{hits} sampled windows overlap an eval set")

    log("")
    for domain in DOMAINS:
        n = index.tokens(domain)
        if n:
            log(f"  {domain:14s} {n:>14,} tokens  "
                f"({n / max(1, index.tokens()) * 100:5.1f}%)")
    return ok


def stage_sample(args, tokenizer) -> None:
    """Print real documents. Data nobody looks at is data nobody has checked."""
    import numpy as np

    section("samples")
    out = Path(args.out) / "shards"
    index = ShardIndex.load(out)
    eos = getattr(tokenizer, "eos_id", None)
    for domain in DOMAINS:
        shards = index.for_domain(domain)
        if not shards:
            continue
        arr = np.memmap(out / shards[0]["path"], dtype="uint16", mode="r")
        ids = np.asarray(arr[:120_000], dtype=np.int64)
        breaks = np.flatnonzero(ids == eos) if eos is not None else []
        if len(breaks) >= 2:
            span = [int(v) for v in ids[breaks[0] + 1:breaks[1]]]
        else:
            span = [int(v) for v in ids[:400]]
        text = tokenizer.decode(span)
        log(f"\n--- {domain} " + "-" * (60 - len(domain)))
        log(text[:700] + ("…" if len(text) > 700 else ""))


# --------------------------------------------------------------------------- #
# Push
# --------------------------------------------------------------------------- #
def _write_card(args, index: ShardIndex) -> Path:
    """Generate the dataset card FROM the report, so its numbers cannot drift from the files."""
    from nyxara.growth.certify import available_backends

    out = Path(args.out) / "shards"
    report_path = Path(args.out) / "report.json"
    report = json.loads(report_path.read_text()) if report_path.exists() else {}
    per_domain = report.get("per_domain") or {d: index.tokens(d) for d in DOMAINS}
    screens = report.get("report", {})

    rows = "\n".join(f"| {d} | {n:,} | {n / max(1, index.tokens()) * 100:.1f}% |"
                     for d, n in sorted(per_domain.items()) if n)
    backends = available_backends()

    card = f"""---
license: other
language:
- en
task_categories:
- text-generation
size_categories:
- {'1B<n<10B' if index.tokens() > 1e9 else '10M<n<100M'}
tags:
- nyxara
- pretraining
---

# NYX-300M pretraining corpus

Tokenized `uint16` shards for NYXARA's 300M-parameter model (24 x 1024, GQA, SwiGLU, RoPE,
context 2048, vocab {index.vocab_size}). Read with `numpy.memmap`; the manifest is `index.json`.

**This repository is private.** The shards are a derived work of several upstream corpora whose
licences carry attribution and opt-out obligations (see *Provenance*), so it is not redistributed.

## Contents

| domain | tokens | share |
|---|---|---|
{rows}

Total: **{index.tokens():,} tokens**, of which {index.tokens(split='val'):,} are held out as a
validation split. Tokenizer signature `{index.vocab_sig[:16]}…` — training on shards whose
signature does not match the tokenizer produces garbage, and `build_dataset.py --verify` refuses.

## How it was built

Five domains, mixed **within** each batch rather than trained in sequence, because
general-then-code-then-maths is how you get catastrophic forgetting.

- **general** — streamed public text. Nothing synthetic: synthetic data makes a model reliable
  at procedures that can be checked and cannot make it knowledgeable about the world.
- **code** — mutation-based generation. A bugfix pair is kept only when the mutant's output
  genuinely differs from the reference on some input; a generated test suite only when it passes
  the reference *and* kills the mutant; an execution trace only when a real run agrees with it.
- **math** — every item constructed *from a known solution*, so the answer is a fact about how
  the problem was built rather than a claim about it.
- **conversation** — multi-turn, with tool results computed for real and format constraints
  verified before the document is kept.
- **causal** — generated from structural causal models whose mechanisms are known, so `do()`
  queries and counterfactuals are ground truth. Includes deliberately *unidentifiable* cases.

Proof certificates are embedded as `<proof>...</proof>` inside the answer span, and only when
they come from a decision procedure — sampled agreement travels as evidence, never as proof.
Backends present at build time: {', '.join(k for k, v in backends.items() if v) or 'none'}.

## Screens

Exact dedup, near-duplicate detection (MinHash + LSH banding), a quality screen, a language
screen, and 13-gram decontamination against the shipped evaluation sets. Rejection counts:

```json
{json.dumps({k: v for k, v in screens.items() if isinstance(v, int)}, indent=2)}
```

## Provenance and licences

Streamed sources retain their upstream licences — FineWeb-Edu (ODC-By), OpenWebMath (ODC-By),
The Stack (per-file licences plus a developer opt-out), UltraChat (MIT). The synthetically
generated portion is original to this project.

## Limitations

- English only. The language screen drops non-Latin-script documents.
- Synthetic code is Python only; multi-language breadth comes from the streamed source.
- Near-duplicate detection is lossy by design — its tables are fixed-size and evict on
  collision, so some near-duplicates survive. Eviction counts are in `report.json`.
- Synthetic data teaches checkable procedure. It does not teach world knowledge.
"""
    path = out / "README.md"
    path.write_text(card, encoding="utf-8")
    return path


def _free_pushed_shards(shard_dir: Path, index: ShardIndex) -> int:
    """Delete shard files that are now on the Hub, keeping the index that describes them.

    6B tokens is ~12 GB of shards against ~23 GB of free disk here. Uploading and unlinking as
    we go holds peak local disk near one shard, which is what lets the full build run in this
    container rather than only on a bigger machine. `--verify` needs the files, so this is
    opt-in and runs after verification, never before it.
    """
    freed = 0
    for shard in index.shards:
        path = shard_dir / shard["path"]
        if path.exists():
            freed += path.stat().st_size
            path.unlink()
    return freed


def stage_restore(args) -> bool:
    """Pull a previous build's state down before starting, so a NEW container continues it."""
    from nyxara.growth.progress import BuildProgress, restore_from_hub

    section(f"restore from {args.push_to_hub}")
    out = Path(args.out) / "shards"
    if not restore_from_hub(out, args.push_to_hub, token=_token()):
        log("· nothing to resume — starting a fresh build")
        return False
    index = ShardIndex.load(out)
    progress = BuildProgress.load(out)
    log(f"· recovered {index.tokens():,} tokens across {len(index.shards)} shard(s)"
        if index else "· recovered progress with no index")
    log(progress.summary())
    return True


def stage_push(args, index: ShardIndex) -> bool:
    section(f"push to {args.push_to_hub}")
    token = _token()
    if not token:
        log(f"· no token in the environment ({' / '.join(_TOKEN_ENV)}) — refusing to push")
        return False

    try:
        from huggingface_hub import HfApi
    except Exception as exc:  # noqa: BLE001
        log(f"· huggingface_hub is not installed ({exc})")
        return False

    api = HfApi(token=token)
    card = _write_card(args, index)
    log(f"· dataset card written to {card}")

    # private=True unconditionally. The shards are a derived work of corpora with attribution
    # and opt-out obligations; a public repo would inherit all of them irreversibly.
    api.create_repo(repo_id=args.push_to_hub, repo_type="dataset", private=True, exist_ok=True)
    info = api.repo_info(repo_id=args.push_to_hub, repo_type="dataset")
    if not getattr(info, "private", False):
        log("· REFUSING: the remote repository is PUBLIC. This corpus carries redistribution "
            "obligations it cannot discharge. Make it private and re-run.")
        return False
    log("· repository is private — confirmed with the remote, not assumed")

    # The resume state travels with the shards. Uploading `index.json` without `progress.json`
    # and the dedup tables gives a repo you can train from but not CONTINUE from — the next
    # container would re-read every source from row zero and re-admit what it already has.
    api.upload_folder(
        repo_id=args.push_to_hub, repo_type="dataset",
        folder_path=str(Path(args.out) / "shards"),
        allow_patterns=["*.bin", "index.json", "README.md", PROGRESS_NAME, *DEDUP_NAMES],
        commit_message=f"NYX-300M corpus: {index.tokens():,} tokens")
    # A build interrupted before `stage_shards` returned has shards and progress but no report.
    # Pushing that partial state is exactly what checkpointing is FOR, so a missing report must
    # not abort the upload — the whole point is to get the resumable state off this machine.
    report = Path(args.out) / "report.json"
    if report.exists():
        api.upload_file(path_or_fileobj=str(report), path_in_repo="report.json",
                        repo_id=args.push_to_hub, repo_type="dataset")
    else:
        log("· no report.json yet (partial build) — shards and resume state pushed without it")

    if args.free_local:
        freed = _free_pushed_shards(Path(args.out) / "shards", index)
        log(f"· freed {freed / 1024 / 1024:.0f} MB locally — the shards live on the Hub now")
    log(f"· uploaded — https://huggingface.co/datasets/{args.push_to_hub}")
    return True


# --------------------------------------------------------------------------- #
def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default="./nyx300m")
    parser.add_argument("--tokens", type=int, default=6_000_000_000)
    parser.add_argument("--vocab-size", type=int, default=32_768)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    parser.add_argument("--stage", default="all",
                        choices=("all", "tokenizer", "shards", "verify", "sample", "push"))
    parser.add_argument("--no-stream", action="store_true",
                        help="offline: synthetic sources only, `general` excluded")
    parser.add_argument("--no-decontam", action="store_true")
    parser.add_argument("--val-permille", type=int, default=5)
    parser.add_argument("--shard-tokens", type=int, default=25_000_000,
                        help="tokens per shard file. This is ALSO the resume granularity: a "
                             "build killed before its first flush keeps nothing, because "
                             "everything is still in the writer's buffer. Lower it for a short "
                             "run you expect to interrupt.")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--verify", action="store_true", help="verify after building")
    parser.add_argument("--sample", action="store_true", help="print decoded documents")
    parser.add_argument("--seed-text", default=None)
    parser.add_argument("--tokenizer-synth", type=int, default=4000)
    parser.add_argument("--tokenizer-stream", type=int, default=20_000,
                        help="real documents sampled across every source to train the "
                             "vocabulary. Synthetic text alone exhausts the BPE at ~2,500 "
                             "merges, which cannot support the 32,768 vocab nyxara-300m expects.")
    parser.add_argument("--push-to-hub", default=None, metavar="OWNER/NAME")
    parser.add_argument("--resume-from-hub", action="store_true",
                        help="pull index/progress/dedup down before building, so a FRESH "
                             "container continues an interrupted build instead of restarting")
    parser.add_argument("--free-local", action="store_true",
                        help="after a successful push, delete the uploaded shards locally; "
                             "holds peak disk near one shard for a full 6B build")
    args = parser.parse_args(argv)

    Path(args.out).mkdir(parents=True, exist_ok=True)
    section("NYX-300M corpus")
    log(f"· working directory: {Path(args.out).resolve()}")
    log(f"· token budget: {args.tokens:,}")

    if args.resume_from_hub:
        if not args.push_to_hub:
            log("· --resume-from-hub needs --push-to-hub to know WHICH build to resume")
            return 1
        stage_restore(args)

    tokenizer = stage_tokenizer(args)

    if args.stage in ("all", "shards"):
        stage_shards(args, tokenizer)

    index = ShardIndex.load(Path(args.out) / "shards")
    if index is None:
        log("\nno shards on disk — nothing further to do")
        return 1

    if args.verify or args.stage == "verify":
        if not stage_verify(args, tokenizer):
            log("\nVERIFICATION FAILED — not pushing. Fix the corpus before training on it.")
            return 2
    if args.sample or args.stage == "sample":
        stage_sample(args, tokenizer)

    if args.push_to_hub and args.stage in ("all", "push"):
        if not stage_push(args, index):
            return 3
    return 0


def _exit(code: int) -> None:
    """Leave without running interpreter finalization.

    `datasets` streaming keeps background prefetch threads inside huggingface_hub's HTTP layer.
    Closing the iterators does not join them, and when the interpreter finalizes with one still
    live the process aborts:

        Fatal Python error: PyGILState_Release: thread state ... must be current when releasing

    That happens AFTER a successful build — index.json and report.json are already written
    atomically — so the corpus on disk is fine while the run looks crashed and returns 134.
    Anything worth keeping is on disk before this point; the only thing finalization would do
    for us is flush stdout, which we do ourselves.
    """
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)


if __name__ == "__main__":
    _exit(main())
