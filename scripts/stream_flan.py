#!/usr/bin/env python3
"""Read every submix of Open-Orca/FLAN over the wire, and keep only the structure.

FLAN is 94.6 GB across nine files and this container has 20 GB of writable disk. That was treated
as a reason to sample, and it was not one: **the files never have to be stored.** They are streamed,
parsed as they arrive, and thrown away a chunk at a time. Measured at 27.5 MB/s, the whole dataset
is about an hour of reading and no disk at all.

    python3 scripts/stream_flan.py                 # every file
    python3 scripts/stream_flan.py --only niv2_zs_submix_data.json

Two file shapes and one parser. ``.jsonl`` is one object per line; ``.json`` is a single array
19 GB long, which no ordinary loader will take, so objects are cut out of the byte stream by
tracking brace depth and string state. Neither is ever held whole in memory.

**What is kept is not the rows.** A FLAN row is a prompt and a completion, and storing those would
be storing the dataset's answers — the thing V.51 refused to do. What comes out is the structure
underneath them, in four kinds, and each one already has an organ waiting for it:

* **inference** — a premise, a hypothesis and one of three answers. Feeds :mod:`nyxara.njp.entail`,
  which had 36,302 of these and moved 0.9 points on five times the data, so the question of whether
  it is the data or the reader that is short is one this can finally settle.
* **worked arithmetic** — a chain of stated sums that can be recomputed. Feeds
  :mod:`nyxara.njp.arithmetic`.
* **instructions** — niv2 says what its task *is* before giving an instance: "In this task, you are
  given a question and a context passage." That is procedural knowledge, written down, at a scale
  nothing else in this package has.
* **question and answer** — a short question with a short answer, which is what
  :mod:`nyxara.njp.grounding` can actually hold.

Caps are per kind and are stated in the output, because a corpus that silently stopped early is a
corpus nobody can reason about.
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
import sys
import time
import urllib.request
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterator, Optional, Sequence, Set, Tuple

BASE = "https://huggingface.co/datasets/Open-Orca/FLAN/resolve/main/"
UA = "NYXARA-research/0.1 (https://github.com/nyxarajp-del/NYXARAv01) full-read"
LICENCE = "CC BY 4.0"

#: Every file, largest last so a short run still crosses several submixes.
FILES: Tuple[str, ...] = (
    "cot_zs_submix_data.json",
    "niv2_zs_submix_data.json",
    "dialog_submix_data.jsonl",
    "flan2021_zsnoopt_submix_data.json",
    "flan2021_zsopt_submix_data.json",
    "niv2_submix_data.jsonl",
    "t0_zsnoopt_submix_data.json",
    "t0_zsopt_submix_data.json",
)

CHUNK = 1 << 20

#: Yielded in place of a row the pre-check ruled out, so the caller still counts it as read. A
#: reader that silently dropped them could not say how much of the dataset it had been through.
_SKIPPED: Dict[str, Any] = {}
LABELS = ("yes", "no", "it is not possible to tell", "entailment", "neutral", "contradiction")
_QUOTED = re.compile(r'"([^"]{6,400})"')
_ANSWER = re.compile(r"[Tt]he (?:final )?answer is[:\s]+([^\n]+)")
_SUM = re.compile(r"(?<![\d.)])(\d[\d,.\s()+\-*/×÷]*[\d)])\s*=\s*"
                  r"(-?\d[\d,]*(?:\.\d+)?(?:\s*/\s*\d+)?)(?![\d.]*\s*[-+*/×÷=])")
_TIMES = re.compile(r"(?<=\d)\s*[x×]\s*(?=\d)")
_PREMISE = re.compile(r"[Pp]remise:\s*(.+?)\n+\s*[Hh]ypothesis:\s*(.+?)(?:\n|$)", re.S)
_INSTRUCTION = re.compile(r"^(In this task[^\n]{0,400}\.)", re.M)
_QUESTION = re.compile(r"(?:^|\n)\s*(?:Q|Question|question):\s*(.{8,200}\?)\s*(?:\n|$)")


#: One whole quoted string, or a single brace. Iterating this instead of every character is what
#: makes the read keep up with the wire: a character loop managed 0.87 MB/s, which would have been
#: **thirty hours** for the dataset, and almost every one of those characters was the inside of a
#: prompt that the parser only had to skip. A string match jumps the whole value in one step, and
#: because the pattern consumes escapes itself, a `}` inside a prompt still cannot end an object.
# The "unrolled loop" form of a quoted string. Written as `"(?:[^"\\]|\\.)*"` it backtracks a
# character at a time and the read managed 2.59 MB/s; written this way the engine consumes whole
# runs between escapes and it does not. Same language, different cost.
_TOKEN = re.compile(r'"[^"\\]*(?:\\.[^"\\]*)*"|[{}]', re.S)

#: A row worth parsing says one of these somewhere. Checked on the raw slice before `json.loads`,
#: because most of FLAN is translation, summarisation and dialogue — nothing this extractor can
#: use — and decoding twenty kilobytes of JSON to discover that is the bulk of the work.
#: How long a passage may be and still be one somebody could read to answer a question. The floor
#: keeps out one-line prompts where "the answer is in the passage" is trivially true; the ceiling
#: keeps the corpus a size the repository can hold.
MIN_PASSAGE = 120
MAX_PASSAGE = 2500

#: How long an answer span may be. Beyond this it is a summary of the passage, not a span of it.
MAX_SPAN = 120

#: And a floor, because a one- or two-character answer matches somewhere in any passage by
#: accident. `C` matched the option letter in `(associated with "A", "B", "C", "D")` and `no`
#: matched the word inside *"classify your answers into yes or no"* — both of which are in the
#: task's own instructions rather than in anything anybody read.
MIN_SPAN = 4

#: FLAN's prompt furniture: the wrapper a row is dressed in before its actual content. This is
#: knowledge about how one dataset formats a string, not about language, and it is the same kind
#: of thing `_PREMISE` and `_QUESTION` already are. Everything up to the last of these goes.
_SCAFFOLD = re.compile(
    r"(?:^|\n)\s*(?:Teacher\s*:\s*Now,? understand the problem\?[^\n]*"
    r"|Detailed Instructions?\s*:"
    r"|Given the task definition and input, reply with output\."
    r"|You will be given a definition of a task first[^\n]*"
    r"|Solve this instance\s*:"
    r"|Teacher\s*:"
    r"|Student\s*:"
    r"|Input\s*:"
    r"|Problem\s*:)", re.I)

#: Corpora whose answers are spans of the passage **by construction**. This is a statement about
#: provenance, not about language: SQuAD, Quoref, DROP, ROPES, MRQA, NewsQA, DuoRC, CoQA, QuAC and
#: the ViquiQuAD family are built by asking annotators to mark a span, so a row from one of them
#: that also passes the verbatim test is extractive twice over.
#:
#: The alternative was tried first and does not work. Requiring only that the answer appear once
#: in the prompt admits every *classification* row in the dataset, because a classification task
#: spells its label vocabulary out in its own instructions — "classify into yes or no" contains
#: `no`, "(i) Hope speech ... (ii) Not Hope Speech" contains `Not Hope Speech`, and the reader
#: would have learned to find its answers in the task's directions rather than in anything anybody
#: read. Stripping the scaffolding helped and did not fix it, because for those tasks the
#: instruction *is* the prompt.
_EXTRACTIVE = re.compile(
    r"squad|quoref|mrqa|drop|ropes|duorc|newsqa|coqa|quac|adversarial_qa|viquiquad|"
    r"triviaqa|trivia_qa|natural_questions|hotpot|narrativeqa|record", re.I)

#: The `In this task ...` sentence itself, which `_INSTRUCTION` already knows how to find.
_TASK_SENTENCE = re.compile(r"In this task[^\n]{0,400}?\.\s", re.I)


#: Where the content starts, when the prompt says so. Same class of thing as `_SCAFFOLD`: how one
#: dataset labels the part of a string it wants read.
_CONTENT = re.compile(r"(?:^|\s)(?:Passage|Context|Paragraph|Article|Story|Text|Sentence)\s*:\s*",
                      re.I)


def _passage(text: str) -> str:
    """The prompt with its task instructions taken off, so what is left is what somebody read."""
    cut = 0
    for match in _SCAFFOLD.finditer(text):
        cut = max(cut, match.end())
    body = text[cut:]
    marked = list(_CONTENT.finditer(body))
    if marked:
        body = body[marked[-1].end():]
    body = _TASK_SENTENCE.sub(" ", body)
    return " ".join(body.split())


def _occurs_once(passage: str, answer: str) -> bool:
    """The answer is in the passage, at word boundaries, and in exactly one place.

    Once, and that is the load-bearing half. A span that appears twice may be either occurrence
    and the row cannot say which; more to the point, an answer that appears many times is usually
    a common word that the passage was never claiming as its answer.
    """
    found = re.findall(r"(?<!\w)" + re.escape(answer.lower()) + r"(?!\w)", passage.lower())
    return len(found) == 1

_WORTH = ("answer is", "In this task", "Premise:", "premise:", "Q:", "Question:", "question:")


def objects(url: str, *, limit_bytes: int = 0) -> Iterator[Dict[str, Any]]:
    """Every JSON object in the file, cut out of the byte stream as it arrives.

    Works for both shapes: one object per line, or a single array nineteen gigabytes deep. Neither
    is ever held whole — the buffer keeps only the object currently being read.

    **The chunk boundary is the whole difficulty**, and getting it wrong was silent. A megabyte
    ends wherever it ends, which is usually in the middle of a prompt. The pattern then cannot
    match that half-string as a string, so it matches the ``{`` and ``}`` characters *inside* the
    prompt as structure instead: depth went 1, 2, 3, 4, 5 across successive chunks, not one further
    object ever closed, and the buffer grew without bound. The reader had emitted 994 objects out
    of the first megabyte and then nothing at all — and reported no error, because from its own
    point of view it was still patiently reading one very large object.

    So the scan stops at the first quote it cannot close. Matches are taken as a list, the earliest
    unpaired ``"`` is found, everything at or after it is discarded as unscanned, and the buffer
    carries from there into the next chunk.
    """
    request = urllib.request.Request(url, headers={"User-Agent": UA})
    read = 0
    buffer = ""
    depth = 0
    start = -1
    with urllib.request.urlopen(request, timeout=120) as response:
        while True:
            block = response.read(CHUNK)
            if not block:
                break
            read += len(block)
            buffer += block.decode("utf-8", "replace")

            matches = list(_TOKEN.finditer(buffer))
            # Where the last complete string ended; a quote after that opens one this chunk cannot
            # close, and every brace beyond it is inside a prompt rather than in the structure.
            after_strings = 0
            for match in matches:
                if match.group(0)[0] == '"':
                    after_strings = match.end()
            unpaired = buffer.find('"', after_strings)
            edge = unpaired if unpaired >= 0 else len(buffer)

            cut = 0
            for match in matches:
                if match.start() >= edge:
                    break
                token = match.group(0)
                if token == "{":
                    if depth == 0:
                        start = match.start()
                    depth += 1
                elif token == "}":
                    depth -= 1
                    if depth == 0 and start >= 0:
                        raw = buffer[start:match.end()]
                        if any(mark in raw for mark in _WORTH):
                            try:
                                yield json.loads(raw)
                            except Exception:  # noqa: BLE001 — a malformed object is skipped
                                pass
                        else:
                            yield _SKIPPED
                        cut = match.end()
                        start = -1

            # Carry the object in progress, and **forget the depth that counted its opening
            # brace** — the next pass re-reads that brace from the start of the carried buffer, so
            # keeping the count meant counting it twice. That was the other half of the same
            # silence: depth climbed by one per chunk, never returned to zero, and the reader spent
            # the rest of a nineteen-gigabyte file convinced it was inside one object.
            keep_from = start if depth > 0 and start >= 0 else min(max(cut, 0), edge)
            buffer = buffer[keep_from:]
            depth = 0
            start = -1
            if limit_bytes and read >= limit_bytes:
                return


class Keep:
    """Gzip writers with a cap each, and a count of what was dropped once full."""

    def __init__(self, directory: Path, caps: Dict[str, int], *, tag: str = "") -> None:
        self.caps = caps
        self.kept: Dict[str, int] = {k: 0 for k in caps}
        #: How many candidates of each kind the file actually held, counted whether or not the cap
        #: had room for them. A run that stopped writing and reported only what it wrote would
        #: leave nobody able to say how much of the dataset was there.
        self.saw: Dict[str, int] = {k: 0 for k in caps}
        self.full: Set[str] = set()
        self.seen: Dict[str, Set[int]] = {k: set() for k in caps}
        directory.mkdir(parents=True, exist_ok=True)
        suffix = f".{tag}" if tag else ""
        self.handles = {k: gzip.open(directory / f"flan_{k}{suffix}.jsonl.gz", "wt",
                                     encoding="utf-8") for k in caps}

    def add(self, kind: str, row: Dict[str, Any], key: str) -> None:
        self.saw[kind] += 1
        if kind in self.full:
            return
        fingerprint = hash(key)
        if fingerprint in self.seen[kind]:
            return
        self.seen[kind].add(fingerprint)
        self.handles[kind].write(json.dumps(row, ensure_ascii=False) + "\n")
        self.kept[kind] += 1
        if self.kept[kind] >= self.caps[kind]:
            self.full.add(kind)

    def done(self) -> bool:
        return len(self.full) == len(self.caps)

    def close(self) -> None:
        for handle in self.handles.values():
            handle.close()


def harvest(row: Dict[str, Any], keep: Keep) -> None:
    """Pull every kind of structure this row happens to contain. Most rows contain none."""
    prompt = str(row.get("inputs") or "")
    reply = str(row.get("targets") or "")
    task = str(row.get("_task_name") or "")
    source = str(row.get("_task_source") or "")
    whole = f"{prompt}\n{reply}"

    said = _ANSWER.search(reply)
    answer = (said.group(1).strip().rstrip(".").lower() if said else reply.strip().lower())

    # inference — the explicit Premise:/Hypothesis: form, or a quoted pair with a three-way answer
    if answer[:30] in LABELS or answer in LABELS:
        found = _PREMISE.search(prompt)
        if found:
            premise, hypothesis = found.group(1).strip(), found.group(2).strip()
        else:
            quoted = _QUOTED.findall(whole)
            premise, hypothesis = (quoted[-2].strip(), quoted[-1].strip()) if len(quoted) >= 2 \
                else ("", "")
        if premise and hypothesis:
            keep.add("inference", {"premise": premise, "hypothesis": hypothesis,
                                   "label": answer, "task": task, "source": source,
                                   "licence": LICENCE},
                     f"{premise.lower()}|{hypothesis.lower()}")

    # worked arithmetic — a chain of sums that can be recomputed
    if said:
        steps = [(" ".join(a.split()), " ".join(b.split()))
                 for a, b in _SUM.findall(_TIMES.sub(" * ", reply))
                 if any(op in a for op in "+-*/×÷")]
        if steps:
            keep.add("maths", {"question": " ".join(prompt.split())[-500:],
                               "worked": " ".join(reply.split())[:900],
                               "answer": said.group(1).strip().rstrip("."),
                               "steps": [list(s) for s in steps], "task": task,
                               "licence": LICENCE},
                     " ".join(prompt.split())[-300:])

    # instructions — what the task is, said before an instance of it
    told = _INSTRUCTION.search(prompt)
    if told:
        keep.add("instruction", {"instruction": " ".join(told.group(1).split()),
                                 "task": task, "source": source, "licence": LICENCE},
                 told.group(1)[:200])

    # question and answer — short enough for the fact store to hold
    asked = _QUESTION.search(prompt)
    if asked and reply and len(reply.strip()) <= 90 and "\n" not in reply.strip():
        keep.add("qa", {"question": " ".join(asked.group(1).split()),
                        "answer": " ".join(reply.split()), "task": task,
                        "source": source, "licence": LICENCE},
                 asked.group(1)[:200])

    # a question with the passage that answers it — and the answer verifiably *inside* the passage
    #
    # This is the one kind here that validates itself. `qa` keeps a question and a reply and has
    # to trust that the reply answers it; five separate filters were needed after the fact to
    # throw out the rows where it did not. Here the test is arithmetic: the answer is kept only
    # if it appears, character for character, in the text the prompt supplied. A row that passes
    # cannot be a classification label, an option index, or another question, because none of
    # those is a substring of the passage.
    if asked and reply and _EXTRACTIVE.search(task):
        answered = " ".join(reply.split())
        context = _passage(prompt[:asked.start()])
        if (MIN_PASSAGE <= len(context) <= MAX_PASSAGE
                and MIN_SPAN <= len(answered) <= MAX_SPAN
                and _occurs_once(context, answered)):
            keep.add("reading", {"passage": context,
                                 "question": " ".join(asked.group(1).split()),
                                 "answer": answered, "task": task,
                                 "source": source, "licence": LICENCE},
                     f"{context[:150]}|{asked.group(1)[:120]}")


def one_file(name: str, out: str, caps: Dict[str, int], limit_bytes: int,
             tag: str) -> Dict[str, int]:
    """One submix, in its own process, writing its own shard. Merged afterwards."""
    keep = Keep(Path(out), caps, tag=tag)
    rows = 0
    began = time.time()
    try:
        for row in objects(BASE + name, limit_bytes=limit_bytes):
            rows += 1
            if row:
                harvest(row, keep)
            if rows % 500_000 == 0:
                print(f"  {name}: {rows:,} rows, {keep.kept}", file=sys.stderr, flush=True)
    except Exception as error:  # noqa: BLE001 — one file failing is not the run failing
        print(f"  {name}: stopped after {rows:,} rows — {error}", file=sys.stderr, flush=True)
    keep.close()
    print(f"  {name}: {rows:,} rows in {time.time() - began:.0f}s; kept {keep.kept}, "
          f"saw {keep.saw}", file=sys.stderr, flush=True)
    return {"rows": rows, **{f"kept_{k}": v for k, v in keep.kept.items()},
            **{f"saw_{k}": v for k, v in keep.saw.items()}}


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", default="", help="one file name, for a quick check")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--out", default="nyxara/njp/data")
    parser.add_argument("--bytes", type=int, default=0, help="stop each file after N bytes")
    parser.add_argument("--inference", type=int, default=400_000)
    parser.add_argument("--maths", type=int, default=120_000)
    parser.add_argument("--instruction", type=int, default=60_000)
    parser.add_argument("--qa", type=int, default=400_000)
    parser.add_argument("--reading", type=int, default=400_000)
    args = parser.parse_args(argv)

    files = (args.only,) if args.only else FILES
    share = max(1, args.workers)
    caps = {"inference": args.inference // share, "maths": args.maths // share,
            "instruction": args.instruction // share, "qa": args.qa // share,
            "reading": args.reading // share}
    started = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(one_file, name, args.out, caps, args.bytes, f"{index:02d}"): name
                   for index, name in enumerate(files)}
        totals: Dict[str, int] = {}
        for future in as_completed(futures):
            try:
                got = future.result()
            except Exception as error:  # noqa: BLE001
                print(f"  {futures[future]}: failed — {error}", file=sys.stderr)
                continue
            for key, value in got.items():
                totals[key] = totals.get(key, 0) + value
    print(f"\nread {len(files)} files in {time.time() - started:.0f}s; "
          f"{totals.get('rows', 0):,} rows", file=sys.stderr)
    for kind in ("inference", "maths", "instruction", "qa", "reading"):
        print(f"  {kind:<12}kept {totals.get('kept_' + kind, 0):>9,} "
              f"of {totals.get('saw_' + kind, 0):>9,} seen", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
