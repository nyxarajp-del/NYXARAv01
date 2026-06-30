#!/usr/bin/env python3
"""Convert a standard benchmark (GSM8K / MMLU) into NYXARA's held-out validation JSONL.

NYXARA's self-improvement loop measures REAL, transferable capability on a genuinely-external
held-out corpus (``nyxara/eval/datasets.build_realworld_benchmark``). By default it ships a small
curated set (``nyxara/eval/data/holdout_realworld.jsonl``); pointing it at a real standard benchmark
makes the anti-Goodhart signal much stronger. The loader already accepts any JSONL on disk via
``$NYXARA_EVAL_HOLDOUT_PATH`` or ``settings.self_improvement.validation_realworld_path`` — this
script just reshapes a public dataset into the exact record shape that loader expects::

    {"id": ..., "category": ..., "grader": ..., "answer": ..., "prompt": ...}

Dependency-free (standard library only). Examples
-------------------------------------------------
    # GSM8K (each line: {"question": ..., "answer": "...\n#### 42"})
    python scripts/prepare_holdout.py --format gsm8k --in gsm8k_test.jsonl --out holdout.jsonl

    # MMLU (each line: {"question", "choices": [...], "answer": 0..3})
    python scripts/prepare_holdout.py --format mmlu --in mmlu_test.jsonl --out holdout.jsonl --limit 200

Then run NYXARA against it:
    NYXARA_EVAL_HOLDOUT_PATH=$PWD/holdout.jsonl python -m nyxara.growth.improve_main
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from typing import Any, Dict, Iterator, List

_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def _read_jsonl(path: str) -> Iterator[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for lineno, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                sys.exit(f"{path}:{lineno}: invalid JSON ({exc})")


def _gsm8k_record(rec: Dict[str, Any], idx: int) -> Dict[str, Any]:
    """GSM8K: 'answer' ends with a '#### <final>' marker; we keep the final numeric answer."""
    question = str(rec.get("question", "")).strip()
    raw_answer = str(rec.get("answer", ""))
    m = re.search(r"####\s*(-?[\d,\.]+)", raw_answer)
    final = (m.group(1) if m else raw_answer).replace(",", "").strip()
    return {
        "id": f"gsm8k-{idx:04d}",
        "category": "math",
        "grader": "numeric_final",
        "answer": final,
        "prompt": question + "\nGive just the final number.",
    }


def _mmlu_record(rec: Dict[str, Any], idx: int) -> Dict[str, Any]:
    """MMLU: a question, a list of choices, and an integer answer index (0-based)."""
    question = str(rec.get("question", "")).strip()
    choices: List[str] = [str(c) for c in (rec.get("choices") or [])]
    ans = rec.get("answer")
    if not choices or not isinstance(ans, int) or not (0 <= ans < len(choices)):
        raise ValueError(f"mmlu record {idx} missing valid choices/answer")
    rendered = "  ".join(f"{_LETTERS[i]}) {c}" for i, c in enumerate(choices))
    return {
        "id": f"mmlu-{idx:04d}",
        "category": str(rec.get("subject", "knowledge")),
        "grader": "multiple_choice",
        "answer": _LETTERS[ans],
        "choices": choices,
        "prompt": f"{question}\n{rendered}",
    }


_CONVERTERS = {"gsm8k": _gsm8k_record, "mmlu": _mmlu_record}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--format", required=True, choices=sorted(_CONVERTERS),
                    help="source dataset format")
    ap.add_argument("--in", dest="infile", required=True, help="source JSONL path")
    ap.add_argument("--out", dest="outfile", required=True, help="destination JSONL path")
    ap.add_argument("--limit", type=int, default=0, help="keep at most N records (0 = all)")
    args = ap.parse_args()

    convert = _CONVERTERS[args.format]
    written = 0
    with open(args.outfile, "w", encoding="utf-8") as out:
        for idx, rec in enumerate(_read_jsonl(args.infile)):
            if args.limit and written >= args.limit:
                break
            try:
                shaped = convert(rec, idx)
            except ValueError as exc:
                print(f"skip: {exc}", file=sys.stderr)
                continue
            out.write(json.dumps(shaped, ensure_ascii=False) + "\n")
            written += 1
    print(f"wrote {written} task(s) → {args.outfile}")
    print("use it:  NYXARA_EVAL_HOLDOUT_PATH=$PWD/{0} python -m nyxara.growth.improve_main"
          .format(args.outfile))


if __name__ == "__main__":
    main()
