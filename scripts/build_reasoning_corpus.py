#!/usr/bin/env python3
"""Sample the chain-of-thought part of Open-Orca/FLAN into a corpus of reasoning items.

FLAN is 377,759,274 rows and 317 GB. Nothing here downloads it. What this takes is a **spread
sample** — batches of a hundred rows at random offsets, so the sample crosses the tasks the
dataset is ordered by instead of reading the first few thousand of one of them.

What comes out is not knowledge and not an answer key. It is **worked reasoning**: a premise, a
hypothesis, the options offered, the answer, and the one line of rationale that says why. Turning
that into something she can do is `njp.entail`'s job and is measured separately; a downloader that
also decided what made an answer right would make that measurement impossible to take.

    python3 scripts/build_reasoning_corpus.py --batches 40

Licence: CC BY 4.0, recorded on every row.
"""

from __future__ import annotations

import argparse
import gzip
import json
import random
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

ROWS = "https://datasets-server.huggingface.co/rows"
DATASET = "Open-Orca/FLAN"
UA = "NYXARA-research/0.1 (https://github.com/nyxarajp-del/NYXARAv01) reasoning-sample"
LICENCE = "CC BY 4.0"
TOTAL = 377_759_274

#: Where the chain-of-thought rows are. FLAN is ordered by task source and CoT is a small block at
#: the very front — measured by probing: offsets 0 and 100,000 are CoT, 300,000 is already Dialog.
#: Sampling uniformly over all 377 million rows returned **zero** CoT rows in two batches, which is
#: what a fraction of that size does to uniform sampling. Named as a range rather than hidden in a
#: filter, because it is a claim about the file that a future dump could falsify.
COT_LOW = 0
COT_HIGH = 200_000

PACE = 1.5
BACKOFF = 10.0


def _get(offset: int, length: int, *, tries: int = 5) -> Dict[str, Any]:
    query = urllib.parse.urlencode({"dataset": DATASET, "config": "default", "split": "train",
                                    "offset": offset, "length": length})
    wait = BACKOFF
    for attempt in range(tries):
        try:
            request = urllib.request.Request(f"{ROWS}?{query}", headers={"User-Agent": UA})
            with urllib.request.urlopen(request, timeout=60) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            if error.code not in (429, 500, 502, 503) or attempt == tries - 1:
                raise
        except Exception:  # noqa: BLE001
            if attempt == tries - 1:
                raise
        time.sleep(wait)
        wait *= 1.8
    return {}


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batches", type=int, default=40)
    parser.add_argument("--length", type=int, default=100)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--low", type=int, default=COT_LOW)
    parser.add_argument("--high", type=int, default=COT_HIGH)
    parser.add_argument("--source", default="CoT", help="_task_source to keep; '' keeps all")
    parser.add_argument("--out", default="nyxara/njp/data/flan_cot.jsonl.gz")
    args = parser.parse_args(argv)

    rng = random.Random(args.seed)
    kept: List[Dict[str, Any]] = []
    seen: set = set()
    tasks: Dict[str, int] = {}
    for batch in range(args.batches):
        offset = rng.randrange(args.low, max(args.low + 1, args.high - args.length))
        try:
            page = _get(offset, args.length)
        except Exception as error:  # noqa: BLE001
            print(f"  offset {offset}: FAILED {error}", file=sys.stderr, flush=True)
            continue
        for entry in page.get("rows", []):
            row = entry.get("row") or {}
            source = str(row.get("_task_source") or "")
            if args.source and source != args.source:
                continue
            key = (str(row.get("inputs"))[:200], str(row.get("targets"))[:200])
            if key in seen:
                continue
            seen.add(key)
            task = str(row.get("_task_name") or "")
            tasks[task] = tasks.get(task, 0) + 1
            kept.append({"inputs": row.get("inputs"), "targets": row.get("targets"),
                         "task": task, "source": source,
                         "template": str(row.get("_template_type") or ""),
                         "offset": offset, "dataset": DATASET, "licence": LICENCE})
        print(f"  batch {batch + 1}/{args.batches} @ {offset}: {len(kept)} kept",
              file=sys.stderr, flush=True)
        time.sleep(PACE)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(out, "wt", encoding="utf-8") as handle:
        for row in kept:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"{len(kept)} rows across {len(tasks)} tasks -> {out} "
          f"({out.stat().st_size / 1024:.0f} KB)", file=sys.stderr)
    for task, count in sorted(tasks.items(), key=lambda kv: -kv[1]):
        print(f"  {task:<28}{count:>5}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
