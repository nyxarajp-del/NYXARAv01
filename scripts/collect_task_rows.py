#!/usr/bin/env python3
"""Keep a handful of rows of every task in FLAN, so the shapes can be induced from them.

The first read of this dataset extracted 747,897 rows out of 83,271,754 — **0.9%** — and threw the
rest away. Not because the rest held nothing, but because the extractor decided in advance what a
piece of knowledge looks like: a premise and a hypothesis, a ``Q:``, an ``In this task``, a chain of
sums. Five hand-written shapes, and anything not in one of them was invisible. `dialog` is 10.7 GB
and yielded four items.

This does the opposite. It reads every row and keeps almost none of them — but it keeps a *few of
each kind*, and the kind is not something anybody decided: FLAN stamps every row with the task it
came from and the index of the template that rendered it, so rows sharing both were produced by
**one string with holes in it**. Six of them are enough to find that string by alignment, and the
string is the task's shape: what is constant is the instruction, what varies is the data.

    python3 scripts/collect_task_rows.py --out <dir> --per-group 6

What comes out is small — a few rows per group rather than a share of 83 million — and it covers
*every* task in the dataset rather than the ones that matched a pattern. What is done with it is
:mod:`nyxara.njp.shapes`, which aligns each group and induces the template without being told what
any of the fields mean.
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))

from stream_flan import BASE, FILES, objects  # noqa: E402

#: How much of a prompt is kept, from the front and only from the front.
#:
#: Keeping both ends and dropping the middle was the obvious thing and it is wrong. Rows of one
#: group differ in length, so "the last four hundred characters" starts at a different point in
#: each of them, and an anchor found in one tail matches the wrong place in another. Measured:
#: groups whose rows were clipped reconstructed at **0.373**, against **0.966** for groups short
#: enough to be kept whole. Aligning the two halves separately did not fix it either — 0.903
#: overall against 0.951 — because the tails were still cut at unrelated points.
#:
#: A head is a prefix. Every clipped row is then a prefix of its own original, all of them start
#: at the same place, and alignment is exact whether or not a row was cut. The instruction and the
#: field markers live at the front; what is lost is the far end of a long passage, which is data
#: rather than shape.
HEAD = 2000

#: How many rows of one group are enough to see which parts vary. Two would find the varying
#: regions; six makes it unlikely that a constant-looking span is a coincidence of two rows.
PER_GROUP = 6

#: A ceiling, so a pathological file cannot exhaust memory. FLAN has on the order of ten thousand
#: (task, template) pairs; this is well above that and is reported if it is ever reached.
MAX_GROUPS = 120_000


def _clip(text: str) -> str:
    return str(text or "")[:HEAD]


def one_file(name: str, out: str, per_group: int, limit_bytes: int) -> Dict[str, Any]:
    """One submix, in its own process, writing the groups it saw."""
    kept: Dict[str, list] = {}
    rows = 0
    full = 0
    began = time.time()
    try:
        for row in objects(BASE + name, limit_bytes=limit_bytes):
            if not row:
                continue
            rows += 1
            task = str(row.get("_task_name") or "")
            index = str(row.get("_template_idx") or "")
            if not task:
                continue
            key = f"{task}\t{index}"
            here = kept.get(key)
            if here is None:
                if len(kept) >= MAX_GROUPS:
                    full += 1
                    continue
                here = kept[key] = []
            if len(here) >= per_group:
                continue
            here.append({"inputs": _clip(row.get("inputs")),
                         "targets": _clip(row.get("targets")),
                         "kind": str(row.get("_template_type") or ""),
                         "source": str(row.get("_task_source") or "")})
            if rows % 1_000_000 == 0:
                print(f"  {name}: {rows:,} rows, {len(kept):,} groups",
                      file=sys.stderr, flush=True)
    except Exception as error:  # noqa: BLE001 — one file failing is not the run failing
        print(f"  {name}: stopped after {rows:,} rows — {error}", file=sys.stderr, flush=True)

    target = Path(out) / f"tasks.{name.split('_submix')[0]}.jsonl.gz"
    target.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(target, "wt", encoding="utf-8") as handle:
        for key, group in kept.items():
            task, _, index = key.partition("\t")
            handle.write(json.dumps({"task": task, "template": index, "rows": group},
                                    ensure_ascii=False) + "\n")
    print(f"  {name}: {rows:,} rows in {time.time() - began:.0f}s; "
          f"{len(kept):,} groups kept, {full:,} refused at the ceiling",
          file=sys.stderr, flush=True)
    return {"rows": rows, "groups": len(kept), "refused": full}


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", default="")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--out", required=True)
    parser.add_argument("--per-group", type=int, default=PER_GROUP)
    parser.add_argument("--bytes", type=int, default=0)
    args = parser.parse_args(argv)

    files = (args.only,) if args.only else FILES
    began = time.time()
    totals = {"rows": 0, "groups": 0, "refused": 0}
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = [pool.submit(one_file, name, args.out, args.per_group, args.bytes)
                   for name in files]
        for future in futures:
            try:
                got = future.result()
            except Exception as error:  # noqa: BLE001
                print(f"  a file failed: {error}", file=sys.stderr, flush=True)
                continue
            for key in totals:
                totals[key] += got.get(key, 0)
    print(f"read {len(files)} files in {time.time() - began:.0f}s; {totals['rows']:,} rows, "
          f"{totals['groups']:,} groups", file=sys.stderr, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
