"""NYXARA · njp/shapes.py — she works out the dataset's shapes instead of being handed them (🧬).

The first read of FLAN kept 747,897 rows of 83,271,754. **Nine tenths of one percent.** Not because
the rest held nothing — because the extractor decided in advance what knowledge looks like. Five
hand-written patterns: a ``Premise:``, a ``Q:``, an ``In this task``, a chain of sums, a quoted
pair. Anything outside them was invisible, and ``dialog`` — 10.7 GB, five and a half million rows —
yielded **four items**.

The fault is not that the patterns were bad. It is that they were *mine*. Somebody sat down and
wrote what a piece of knowledge looks like, and the reader could then only ever find what that
person already knew to look for. That is the thing this package exists not to do.

So nothing here says what a task looks like. What it uses instead is a fact about the data:

    FLAN stamps every row with the task it came from and the index of the template that rendered
    it. Rows sharing both were produced by **one string with holes punched in it**.

Given six such rows, the string can be recovered by alignment — what every row has in common, in
the same order, is the template; what differs is what was poured into it. From the outside, with
no idea what any field means:

    "In this task, you are given a question and a context passage. You have to answer the
     question based on the given passage.\\nQ: ⟨1⟩, Context: ⟨2⟩"

Two slots, discovered. Nobody wrote ``Q:`` or ``Context:`` down; they are simply what did not
vary. The same procedure runs on a translation task, a summarisation task, a dialogue task, a task
in Tamil — on anything the dataset contains, including the shapes nobody anticipated, because it
never asks what the shape means.

What that buys is the difference between reading 0.9% of a dataset and reading the **form of all of
it**. It is not the same as knowing the 83 million rows: what is kept is one shape per task, with a
handful of examples. But a shape is the thing that generalises — the rows are instances of it — and
17,000 shapes is something that fits in a repository where 83 million rows never could.

Pure standard library.
"""

from __future__ import annotations

import gzip
import json
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

__all__ = ["Slot", "Shape", "Group", "align", "induce", "read_groups", "CORPUS",
           "MIN_ANCHOR", "TRIM"]

CORPUS = Path(__file__).with_name("data") / "flan_shapes.jsonl.gz"

#: How long a run of shared characters must be before it counts as part of the template rather
#: than a coincidence. Set by measurement, not taste: at four, ordinary English words shared by
#: two unrelated passages (`" the "`, `" of "`) are read as template and the slots shatter into
#: dozens of fragments; at sixteen, short real markers like ``"\\nQ: "`` are lost and two fields
#: merge into one. Twelve sits between, and :func:`nyxara.njp.shapeschool.sweep` prints the table.
MIN_ANCHOR = 12

#: A template with more slots than this is not a template — it is two passages that happen to
#: share some words. Seen when a group's rows are near-identical for a reason other than sharing
#: a template, which the alignment cannot tell apart from the real thing on its own.
MAX_SLOTS = 12

#: The mark a slot is rendered with when the shape is shown to a person.
HOLE = "⟨{}⟩"

#: Whether an anchor that a later row rejects is shortened to the part every row does hold, rather
#: than dropped whole. On by measurement, and the switch stays so the claim can be taken away
#: again. Over 13,113 groups of the collection, four rows aligned and the rest reconstructed:
#:
#:     trim    shaped   reconstructs   slots   template
#:     False   0.847    0.925          1.41    336 chars
#:     True    0.966    0.908          1.52    352 chars
#:
#: Read the two columns together rather than separately. Trimming shapes about **1,560 more
#: groups** and gives back 0.017 of exactness on the larger set it is then judged on — so of all
#: held-out rows, the share returned character-for-character goes from 0.783 to 0.877. Groups that
#: were previously refused outright are the ones being added, and they come in slightly harder
#: than the ones already there, which is what the reconstruction column is showing.
TRIM = True

#: There is deliberately no clip marker here any more, and the reason is worth keeping. The
#: collector used to drop the middle of a long prompt and join the ends with an ellipsis; this
#: module then treated that ellipsis as a hard boundary, which was a fix for a real problem and
#: not the right one — groups of clipped rows reconstructed at 0.373 either way. The collector now
#: keeps a **prefix**, so every clipped row starts where its original starts and alignment needs
#: no special case at all. Keeping the special case afterwards was worse than useless: `…` occurs
#: in ordinary text, so it went on firing on rows nobody had clipped.


@dataclass(frozen=True)
class Slot:
    """One varying region of a template, and what was found in it."""

    index: int = 0
    examples: Tuple[str, ...] = ()

    @property
    def widest(self) -> int:
        return max((len(e) for e in self.examples), default=0)

    @property
    def narrowest(self) -> int:
        return min((len(e) for e in self.examples), default=0)

    @property
    def steady(self) -> bool:
        """Every filler the same length — a code, a label, a letter, rather than free text."""
        return bool(self.examples) and self.widest == self.narrowest

    def to_dict(self) -> Dict[str, Any]:
        return {"index": self.index, "examples": list(self.examples[:3]),
                "narrowest": self.narrowest, "widest": self.widest}


@dataclass
class Shape:
    """One task's template: the constant text, the holes in it, and what it answers with.

    ``parts`` alternates literal text and slot indices, so the template can be printed, matched
    against a row it was not induced from, and used to pull the fields out of one.
    """

    task: str = ""
    template: str = ""
    source: str = ""
    parts: Tuple[Any, ...] = ()
    slots: Tuple[Slot, ...] = ()
    answers: Tuple[str, ...] = ()
    rows: int = 0

    @property
    def constant(self) -> str:
        """Everything the template says on its own, with the holes closed up."""
        return " ".join(p for p in self.parts if isinstance(p, str)).strip()

    @property
    def answer_space(self) -> Tuple[str, ...]:
        """The answers, when they are a *set* rather than merely few.

        The test is repetition, not count, and the difference matters: six rows of a
        question-answering task give six different answers, and "six distinct values, each short"
        is exactly what a small answer space looks like too. `task1295` was duly reported as
        answering from ``('Antarctica', 'Barrio Sur', 'mediums', 'the existence of Ishvara')`` —
        four free-text spans mistaken for a vocabulary because nothing had repeated yet.

        An answer space is a space because rows *reuse* it. If every row's answer is different,
        what has been seen is free text and too few samples to know otherwise.
        """
        said = [a.strip() for a in self.answers if a.strip()]
        seen = sorted(set(said))
        if len(said) < 3 or len(seen) >= len(said):
            return ()
        return tuple(seen) if 1 < len(seen) <= 6 and all(len(a) <= 40 for a in seen) else ()

    def render(self) -> str:
        out: List[str] = []
        for part in self.parts:
            out.append(part if isinstance(part, str) else HOLE.format(part + 1))
        return "".join(out)

    def read(self, prompt: str) -> Optional[List[str]]:
        """Pull this shape's fields out of a row. ``None`` when the row is not of this shape.

        This is what makes the induction checkable rather than decorative: a template induced from
        six rows either parses a seventh or it does not, and :mod:`nyxara.njp.shapeschool` asks
        exactly that of rows the alignment never saw.
        """
        raw = str(prompt or "")
        at = 0
        found: List[str] = []
        for index, part in enumerate(self.parts):
            if isinstance(part, int):
                continue
            if not part:
                continue
            where = raw.find(part, at)
            if where < 0:
                return None
            if index and isinstance(self.parts[index - 1], int):
                found.append(raw[at:where])
            at = where + len(part)
        if self.parts and isinstance(self.parts[-1], int):
            found.append(raw[at:])
        return found

    def to_dict(self) -> Dict[str, Any]:
        return {"task": self.task, "template": self.template, "source": self.source,
                "shape": self.render(), "slots": [s.to_dict() for s in self.slots],
                "answers": list(self.answers[:4]), "answer_space": list(self.answer_space),
                "rows": self.rows}


@dataclass
class Group:
    """The rows of one task-and-template, as the collector kept them."""

    task: str = ""
    template: str = ""
    source: str = ""
    prompts: Tuple[str, ...] = ()
    targets: Tuple[str, ...] = ()


# --------------------------------------------------------------------------------------------- #
#  alignment
# --------------------------------------------------------------------------------------------- #
def _shared_runs(a: str, b: str, least: int) -> List[str]:
    """The runs of characters these two have in common, in order, longest-first per position."""
    matcher = SequenceMatcher(None, a, b, autojunk=False)
    return [a[block.a:block.a + block.size]
            for block in matcher.get_matching_blocks() if block.size >= least]


def _survives(run: str, rows: Sequence[str], at: Sequence[int]) -> Optional[List[int]]:
    """Where this run sits in every row, after what has already been consumed — or nothing."""
    where = [row.find(run, at[i]) for i, row in enumerate(rows)]
    return where if all(w >= 0 for w in where) else None


def _longest(run: str, rows: Sequence[str], at: Sequence[int], least: int,
             end: bool = False) -> str:
    """The longest prefix (or suffix) of this run that every row holds.

    Survival is monotone — a shorter prefix of a surviving prefix survives, being a substring of it
    — so this is a binary search and costs a handful of finds rather than one per length.
    """
    low, high, best = least, len(run), ""
    while low <= high:
        mid = (low + high) // 2
        piece = run[-mid:] if end else run[:mid]
        if _survives(piece, rows, at) is not None:
            best, low = piece, mid + 1
        else:
            high = mid - 1
    return best


def _trimmed(run: str, rows: Sequence[str], at: Sequence[int], least: int) -> str:
    """The part of an over-proposed run that is actually template.

    A run is proposed from the **first pair**, so it can reach past the template at either end into
    whatever those two rows happened to share. Two questions that both open ``wh`` propose
    ``"\nQ: wh"``, and those two characters are enough for row five to reject the entire anchor —
    the instruction is then lost, not shortened. Trimming recovers what the rows do agree on.

    Both ends are tried, and the trimmed result is trimmed again from the other end, because a run
    spanning ``[tail of a field][template][head of a field]`` is over-proposed twice over.
    """
    front = _longest(run, rows, at, least)
    if front:
        front = _longest(front, rows, at, least, end=True) or front
    back = _longest(run, rows, at, least, end=True)
    if back:
        back = _longest(back, rows, at, least) or back
    return front if len(front) >= len(back) else back


def align(prompts: Sequence[str], least: int = MIN_ANCHOR, *, trim: bool = TRIM) -> Tuple[Any, ...]:
    """The parts every one of these strings shares, in order, with the gaps between them as slots.

    Two rows are enough to *propose* an anchor and are not enough to trust one: any two English
    passages share ``" and the "`` somewhere. So a run is proposed from the first pair and then
    has to survive **every** remaining row, in order. What happens when it does not is the
    ``trim`` switch: dropped whole, or shortened to the part the rows do share.
    """
    rows = [str(p or "") for p in prompts if str(p or "")]
    if len(rows) < 2:
        return (rows[0],) if rows else ()
    proposed = _shared_runs(rows[0], rows[1], least)
    if not proposed:
        return ()
    # Each anchor must appear in every row, and after the previous anchor — an anchor that only
    # appears earlier is not the same part of the template.
    kept: List[str] = []
    at = [0] * len(rows)
    for run in proposed:
        where = _survives(run, rows, at)
        if where is None and trim:
            run = _trimmed(run, rows, at, least)
            where = _survives(run, rows, at) if run else None
        if where is None:
            continue
        kept.append(run)
        at = [w + len(run) for w in where]
    if not kept:
        return ()
    parts: List[Any] = []
    slot = 0
    first = rows[0]
    cursor = 0
    for run in kept:
        where = first.find(run, cursor)
        if where > cursor:
            parts.append(slot)
            slot += 1
        elif where == cursor and not parts:
            pass                                  # the template opens with its own text
        parts.append(run)
        cursor = where + len(run)
    if cursor < len(first):
        parts.append(slot)
    return tuple(parts)


def induce(group: Group, least: int = MIN_ANCHOR, *, trim: bool = TRIM) -> Optional[Shape]:
    """One group of rows into one shape, or ``None`` when they share no template worth the name."""
    parts = align(group.prompts, least, trim=trim)
    if not parts:
        return None
    holes = [p for p in parts if isinstance(p, int)]
    if not holes or len(holes) > MAX_SLOTS:
        return None
    shape = Shape(task=group.task, template=group.template, source=group.source,
                  parts=parts, answers=tuple(group.targets), rows=len(group.prompts))
    # What went into each hole, taken from the rows the shape was induced from. A shape that
    # cannot read its own rows back is not a shape, and is refused rather than reported.
    filled: Dict[int, List[str]] = {}
    read_back = 0
    for prompt in group.prompts:
        got = shape.read(prompt)
        if got is None or len(got) != len(holes):
            continue
        read_back += 1
        for index, value in enumerate(got):
            filled.setdefault(index, []).append(value.strip())
    if read_back < max(2, len(group.prompts) // 2):
        return None
    shape.slots = tuple(Slot(index=i, examples=tuple(filled.get(i, [])[:4]))
                        for i in range(len(holes)))
    return shape


# --------------------------------------------------------------------------------------------- #
#  the corpus
# --------------------------------------------------------------------------------------------- #
def _lines(path: Path) -> Iterator[str]:
    """The lines of a gzipped shard, stopping where a truncated one stops.

    A collection killed part-way leaves a file with no end-of-stream marker, and reading it raises
    rather than returning the rows it does hold. Those rows are perfectly good — the run that
    wrote them read that much of the dataset — so the truncation ends the file instead of losing
    it. A shard nobody interrupted is unaffected.
    """
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        while True:
            try:
                line = handle.readline()
            except (EOFError, OSError):
                return
            if not line:
                return
            yield line


def read_groups(path: Path) -> Iterator[Group]:
    """The collector's output, as groups."""
    if not Path(path).exists():
        return
    for line in _lines(Path(path)):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except Exception:  # noqa: BLE001
            continue
        rows = row.get("rows") or []
        yield Group(task=str(row.get("task") or ""),
                    template=str(row.get("template") or ""),
                    source=str((rows[0] if rows else {}).get("source") or ""),
                    prompts=tuple(str(r.get("inputs") or "") for r in rows),
                    targets=tuple(str(r.get("targets") or "") for r in rows))


def read_shapes(path: Optional[Path] = None) -> List[Shape]:
    """The induced shapes, as shipped."""
    source = Path(path) if path is not None else CORPUS
    out: List[Shape] = []
    if not source.exists():
        return out
    with gzip.open(source, "rt", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except Exception:  # noqa: BLE001
                continue
            out.append(Shape(task=str(row.get("task") or ""),
                             template=str(row.get("template") or ""),
                             source=str(row.get("source") or ""),
                             parts=tuple(row.get("parts") or []),
                             slots=tuple(Slot(index=s.get("index", i),
                                              examples=tuple(s.get("examples") or []))
                                         for i, s in enumerate(row.get("slots") or [])),
                             answers=tuple(row.get("answers") or []),
                             rows=int(row.get("rows") or 0)))
    return out
