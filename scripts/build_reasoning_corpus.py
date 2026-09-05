#!/usr/bin/env python3
"""Extract the whole chain-of-thought submix of Open-Orca/FLAN into what can be learned from.

An earlier version of this read the rows API and sampled. It worked and it was tiny: 4,980 rows,
**0.00132%** of the dataset, and two batches drawn uniformly over all 377,759,274 rows returned
zero chain-of-thought rows at all, because FLAN is ordered by task and CoT is a small band at the
front. The submix is also published whole, as one 240 MB file, so there is no reason to sample it:

    curl -L -O https://huggingface.co/datasets/Open-Orca/FLAN/resolve/main/cot_submix_data.jsonl
    python3 scripts/build_reasoning_corpus.py cot_submix_data.jsonl

That is **192,696 rows — every one there is**. What cannot be taken is said plainly rather than
quietly skipped: the other submixes are `dialog` 10.7 GB, `flan2021` 12.6 + 13.3 GB, `niv2` 14.0 +
5.6 GB and `t0` 18.6 GB, about 75 GB against 20 GB of writable disk here. Those are not reasoning
data — they are dialogue, translation, summarisation and task instructions — and none of them is
what "learn to reason" asks for.

Two files come out, because the submix holds two different kinds of reasoning and one organ cannot
read both:

* **inference pairs** — a premise, a hypothesis, one of three answers, and the line of rationale
  that says why. From `cot_esnli` mostly. 36,293 of them, unique.
* **worked arithmetic** — a word problem and a rationale that is a chain of stated sums, each of
  which can be **recomputed**. From `cot_gsm8k` and `stream_aqua`. This is the half where an answer
  is checkable rather than merely usual.

Both are extractions, not answers: the rationale is carried through verbatim and nothing here
decides what makes any of it right.
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Set, Tuple

DATASET = "Open-Orca/FLAN"
LICENCE = "CC BY 4.0"

_ANSWER = re.compile(r"[Tt]he (?:final )?answer is[:\s]+([^\n]+)|[Tt]he final answer:\s*([^\n]+)")
_QUOTED = re.compile(r'"([^"]{6,400})"')
#: A stated sum is a whole arithmetic expression and the value claimed for it — not two operands
#: and a result. Written the narrow way it cut ``3/10 * 20/11 = 6/11`` into ``20 / 11 = 6`` and
#: then reported the corpus as wrong about it, which would have been a finding about this regex
#: dressed as a finding about the data. The expression must run from a digit to a digit or a
#: closing bracket, and must not have another operator or digit pressed against either end.
_SUM = re.compile(
    r"(?<![\d.)])(\d[\d,.\s()+\-*/×÷]*[\d)])\s*=\s*"
    r"(-?\d[\d,]*(?:\.\d+)?(?:\s*/\s*\d+)?)(?![\d.]*\s*[-+*/×÷=])")
_NUMBER = re.compile(r"-?\d[\d,]*(?:\.\d+)?")
LABELS = ("yes", "no", "it is not possible to tell")
MATHS = ("cot_gsm8k", "cot_gsm8k_ii", "stream_aqua", "stream_aqua_ii")


def rows_of(path: Path) -> Iterator[Dict[str, Any]]:
    with open(path, encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:  # noqa: BLE001 — a malformed row is skipped, not fatal
                continue


def blocks(text: str) -> List[str]:
    """A worked example ends where its answer line ends. The prompt holds several."""
    out, start = [], 0
    for match in _ANSWER.finditer(text):
        end = text.find("\n", match.end())
        end = len(text) if end < 0 else end
        out.append(text[start:end])
        start = end
    return out


def said_answer(block: str) -> Optional[str]:
    match = _ANSWER.search(block)
    if not match:
        return None
    said = match.group(1) or match.group(2) or ""
    said = re.split(r"\s*(?:--+|\*\*)", said.strip())[0]
    return said.strip().rstrip(".").strip() or None


def rationale_of(block: str) -> str:
    match = _ANSWER.search(block)
    if not match:
        return ""
    before = block[:match.start()].strip().splitlines()
    return before[-1].strip() if before else ""


#: ``30 x 5 / 6 = 25`` writes its multiplication with a letter. Left unhandled the expression
#: pattern started at ``5 / 6`` instead and the audit reported the corpus wrong about a sum it had
#: got right — one of five residual disagreements that were all this parser's, checked one by one
#: against their source text rather than assumed.
_TIMES = re.compile(r"(?<=\d)\s*[x×]\s*(?=\d)")


def steps_of(text: str) -> List[Tuple[str, str]]:
    """Every stated sum in a rationale, as it was written. Nothing is evaluated here."""
    out: List[Tuple[str, str]] = []
    for expression, said in _SUM.findall(_TIMES.sub(" * ", text)):
        expression = " ".join(expression.split())
        if not any(op in expression for op in "+-*/×÷"):
            continue                       # "= 40" with nothing computed is not a step
        out.append((expression, " ".join(said.split())))
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dump", help="cot_submix_data.jsonl")
    parser.add_argument("--pairs", default="nyxara/njp/data/flan_pairs.jsonl.gz")
    parser.add_argument("--maths", default="nyxara/njp/data/flan_maths.jsonl.gz")
    args = parser.parse_args(argv)

    seen_pairs: Set[Tuple[str, str]] = set()
    seen_maths: Set[str] = set()
    pairs: List[Dict[str, Any]] = []
    maths: List[Dict[str, Any]] = []
    rows = 0
    for row in rows_of(Path(args.dump)):
        rows += 1
        task = str(row.get("_task_name") or "")
        whole = f"{row.get('inputs') or ''}\n{row.get('targets') or ''}"
        for block in blocks(whole):
            answer = said_answer(block)
            if answer is None:
                continue
            low = answer.lower()
            if low in LABELS:
                quoted = _QUOTED.findall(block)
                if len(quoted) < 2:
                    continue
                key = (quoted[-2].strip().lower(), quoted[-1].strip().lower())
                if key in seen_pairs:
                    continue
                seen_pairs.add(key)
                pairs.append({"premise": quoted[-2].strip(), "hypothesis": quoted[-1].strip(),
                              "label": low, "rationale": rationale_of(block), "task": task,
                              "dataset": DATASET, "licence": LICENCE})
                continue
            if task in MATHS:
                worked = block[:_ANSWER.search(block).start()].strip()
                steps = steps_of(worked)
                if not steps:
                    continue
                question = _question_of(block)
                if not question or question in seen_maths:
                    continue
                seen_maths.add(question)
                maths.append({"question": question, "worked": worked, "answer": answer,
                              "steps": [list(s) for s in steps], "task": task,
                              "numbers": _NUMBER.findall(question),
                              "dataset": DATASET, "licence": LICENCE})

    _write(Path(args.pairs), pairs)
    _write(Path(args.maths), maths)
    print(f"{rows} rows -> {len(pairs)} inference pairs, {len(maths)} worked problems",
          file=sys.stderr)
    return 0


def _question_of(block: str) -> str:
    """The problem, which is whatever the prompt asked just before the working began."""
    text = block.strip()
    for marker in ("[Question]", "My question is:", "Question:", "q:", "Q:"):
        if marker in text:
            text = text.rsplit(marker, 1)[-1]
    for marker in ("[Answer]", "Your thoughts:", "Stream of consciousness:", "a:", "A:"):
        if marker in text:
            text = text.split(marker, 1)[0]
    return " ".join(text.split())[:600]


def _write(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"  {len(rows):>6} -> {path} ({path.stat().st_size / 1024 / 1024:.1f} MB)",
          file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())
