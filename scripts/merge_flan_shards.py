#!/usr/bin/env python3
"""Fold the per-file shards of a full FLAN read into one corpus per kind, deduplicated.

`stream_flan.py` runs a process per submix and each writes its own shard, because four readers
saturate the wire where one does not — measured, 9.6 MB/s alone against 26 MB/s together. The
shards overlap: the same premise turns up in `flan2021` and in `t0`, phrased by different
templates. This is where that is settled, once, on the whole set.

    python3 scripts/merge_flan_shards.py --shards <dir> --out nyxara/njp/data

The existing corpora are folded in as well rather than replaced, so nothing already extracted from
the chain-of-thought submix is lost when the wider read supersedes it.
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Set, Tuple

KINDS = ("inference", "maths", "instruction", "qa")

#: What makes two rows the same thing. Not the whole row: the same pair reached by two templates
#: differs in wording and is one pair.
KEY: Dict[str, Tuple[str, ...]] = {
    "inference": ("premise", "hypothesis"),
    "maths": ("question",),
    "instruction": ("instruction",),
    "qa": ("question",),
}


def rows_of(path: Path) -> Iterator[Dict[str, Any]]:
    try:
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    yield json.loads(line)
                except Exception:  # noqa: BLE001
                    continue
    except Exception:  # noqa: BLE001
        return


_WH = ("what", "who", "when", "where", "which", "why", "how", "whose", "whom", "name the",
       "name a")


def _ascii_share(text: str) -> float:
    return sum(1 for c in text if ord(c) < 128) / max(1, len(text))


def usable(kind: str, row: Dict[str, Any]) -> bool:
    """Whether this row is what its kind claims to be. The extractor is broad; this is not.

    Written after a cold reading was taken against the raw ``qa`` extraction and came back 0.000.
    The number would have been about the extractor, not the reader: "What are the software testers
    aware of?" had been paired with the answer ``yes``, "How many people migrated to Thuringia?"
    with ``(2).``, and a Portuguese sentence with its Galician translation. Those are a
    classification task, a broken span and a translation task, and none of them is a question with
    an answer.

    So a question-and-answer row has to look like one: a real question word, an answer short enough
    to be an answer, ``yes``/``no`` only where the question invites it, and both halves in the same
    script — which is what rules the translation pairs out.
    """
    if kind != "qa":
        return True
    question = " ".join(str(row.get("question") or "").split())
    answer = " ".join(str(row.get("answer") or "").split())
    if not question or not answer or len(answer) > 60:
        return False
    low = question.lower()
    if not (any(low.startswith(w) for w in _WH) or low.split(" ", 1)[0] in
            ("is", "are", "was", "were", "do", "does", "did", "can", "could", "has", "have")):
        return False
    if answer.lower() in ("yes", "no", "true", "false") and low.startswith(_WH):
        return False
    if abs(_ascii_share(question) - _ascii_share(answer)) > 0.15:
        return False
    if answer.strip("().,\'\" ") == "":
        return False
    return True


def _key(kind: str, row: Dict[str, Any]) -> int:
    return hash(tuple(" ".join(str(row.get(f) or "").lower().split())[:300]
                      for f in KEY[kind]))


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shards", required=True)
    parser.add_argument("--out", default="nyxara/njp/data")
    parser.add_argument("--keep", type=int, default=0, help="cap per kind after merging")
    args = parser.parse_args(argv)

    shards = Path(args.shards)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for kind in KINDS:
        sources: List[Path] = sorted(shards.glob(f"flan_{kind}.*.jsonl.gz"))
        existing = out / f"flan_{kind}.jsonl.gz"
        if existing.exists():
            sources.append(existing)
        # The chain-of-thought extraction from V.52 lives under different names; fold it in too.
        if kind == "inference" and (out / "flan_pairs.jsonl.gz").exists():
            sources.append(out / "flan_pairs.jsonl.gz")
        if kind == "maths" and (out / "flan_maths.jsonl.gz").exists():
            sources.append(out / "flan_maths.jsonl.gz")
        if not sources:
            continue
        seen: Set[int] = set()
        kept = 0
        read = 0
        dropped = 0
        target = out / f"flan_{kind}.merged.jsonl.gz"
        with gzip.open(target, "wt", encoding="utf-8") as handle:
            for source in sources:
                for row in rows_of(source):
                    read += 1
                    if not usable(kind, row):
                        dropped += 1
                        continue
                    fingerprint = _key(kind, row)
                    if fingerprint in seen:
                        continue
                    seen.add(fingerprint)
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                    kept += 1
                    if args.keep and kept >= args.keep:
                        break
                if args.keep and kept >= args.keep:
                    break
        target.replace(out / f"flan_{kind}.jsonl.gz")
        size = (out / f"flan_{kind}.jsonl.gz").stat().st_size / 1024 / 1024
        print(f"  {kind:<12}{kept:>9,} of {read:>9,} read from {len(sources)} shards"
              f"  ({dropped:,} were not what the kind claims)  {size:>7.1f} MB",
              file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
