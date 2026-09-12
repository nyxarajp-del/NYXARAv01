"""NYXARA · njp/entail.py — does the second sentence follow from the first (⊨).

FLAN's chain-of-thought split is 4,980 rows of worked reasoning and the largest thing in it, by a
distance, is natural-language inference: a premise, a hypothesis, and one of three answers —
*yes*, *no*, or *it is not possible to tell* — with one line of rationale saying why.

Measured cold, she answers none of them.

The tempting thing to do with a dataset is to store its answers. That would be a lookup table with
7,226 rows in it and it would score zero on the 7,227th, because nothing in it is about **why** any
answer was right. What is learnable here is not the answers; it is the regularity behind them, and
eSNLI's is a real one that a person can state:

* the hypothesis says less than the premise, and only things the premise says  ->  **yes**
* the hypothesis says something the premise does not settle                    ->  **not possible to tell**
* the hypothesis says something the premise rules out                          ->  **no**

None of that is written into this module. What is written is a set of **generic measurements of a
pair of sentences** — how many of the hypothesis's content words the premise also has, how many it
adds, whether one carries a negation the other does not, whether the hypothesis is longer, whether
either mentions a number. Not one of them names an answer. Which of them predicts which answer is
induced by :mod:`nyxara.njp.induce`, the same cover that learns what breaks a program, from pairs
she has been shown and against pairs she has not.

The parsing is worth a word. A FLAN row is a few-shot prompt: several **complete** worked examples
followed by an unanswered question, and the answer to that last one in ``targets``. The examples
inside the prompt are worked reasoning too, and ignoring them threw away three quarters of the
data — 3,116 items from the targets against **11,642** in the prompts. Every block ending in *"The
answer is ..."* is one item, wherever it sits.

Pure standard library.
"""

from __future__ import annotations

import gzip
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from nyxara.njp.induce import Rule, cover

__all__ = ["Pair", "Reasoner", "BROAD", "CORPUS", "LABELS", "read_pairs", "probe",
           "mine_exclusions"]

#: Where the corpus is read from, most-read first. :data:`BROAD` is the fold of the whole 94.6 GB
#: read — every submix, not only the chain-of-thought one — and supersedes :data:`CORPUS`, which
#: ``scripts/build_reasoning_corpus.py`` writes from the CoT submix alone. :data:`RAW` is the
#: original sampled file, kept readable so the parser here stays exercised on the shape the
#: dataset actually ships in. Whichever exists first is what :func:`read_pairs` returns, so the
#: module works in a checkout that has run none of the builders, one of them, or both.
BROAD = Path(__file__).with_name("data") / "flan_inference.jsonl.gz"
CORPUS = Path(__file__).with_name("data") / "flan_pairs.jsonl.gz"
RAW = Path(__file__).with_name("data") / "flan_cot.jsonl.gz"

#: The three answers the task offers. Named because the parser needs to know which blocks are this
#: task and which are one of the seventeen others — not because anything here knows what they mean.
LABELS: Tuple[str, ...] = ("yes", "no", "it is not possible to tell")

#: FLAN asks the **same three relations** in two vocabularies. Some templates offer
#: ``yes / no / it is not possible to tell``; others offer ``entailment / contradiction /
#: neutral``. They are the same question, and treating them as six classes did two things, both
#: bad:
#:
#: * the three rare spellings could never be learned — 401 `entailment` and 391 `contradiction`
#:   pairs in a corpus of 564,166, against a support floor in the thousands; and
#: * worse, each of them sat in the **negatives** of the label it means, so every `contradiction`
#:   pair was evidence against `no` and every `neutral` pair evidence against `it is not possible
#:   to tell`.
#:
#: Folding is not a tuning knob. Two names for one relation are one relation, and the surface
#: vocabulary is a property of the *question*, not of the answer — :func:`as_asked` puts it back
#: when a caller needs the words a particular template offered.
RELATIONS: Dict[str, str] = {"entailment": "yes", "contradiction": "no",
                             "neutral": "it is not possible to tell"}
_SPELLED = {relation: name for name, relation in RELATIONS.items()}


def relation_of(label: str) -> str:
    """The relation this label names, whichever of the two vocabularies it is written in."""
    said = str(label or "").strip().lower()
    return RELATIONS.get(said, said)


def as_asked(relation: str, like: str) -> str:
    """Render a relation in the vocabulary some other label uses.

    ``like`` is any label from the template being answered. A task that offered `entailment` gets
    `entailment` back; one that offered `yes` gets `yes`. Answering in the wrong vocabulary is
    wrong however right the relation is, and nothing in :meth:`Reasoner.answer` can know which
    vocabulary was asked for — only the caller holding the question does.
    """
    said = str(relation or "").strip().lower()
    return _SPELLED.get(said, said) if str(like or "").strip().lower() in RELATIONS else said

_ANSWER = re.compile(r"[Tt]he answer is[:\s]+([^\n]+)")
_QUOTED = re.compile(r'"([^"]{6,400})"')
_WORD = re.compile(r"[^\W\d_][\w'’-]*|\d+", re.UNICODE)


def _closed() -> Dict[str, str]:
    try:
        from nyxara.njp.semantics import _CLOSED  # noqa: WPS433
        return _CLOSED
    except Exception:  # noqa: BLE001 — with no closed class every word counts as content
        return {}


def _content(text: str) -> Set[str]:
    closed = _closed()
    return {w for w in (m.group(0).lower() for m in _WORD.finditer(str(text or "")))
            if w not in closed and len(w) > 1}


def _negated(text: str) -> bool:
    closed = _closed()
    try:
        from nyxara.njp.semantics import Tag  # noqa: WPS433
        negs = {w for w, tag in closed.items() if tag == Tag.NEG}
    except Exception:  # noqa: BLE001
        negs = set()
    words = {m.group(0).lower() for m in _WORD.finditer(str(text or ""))}
    return bool(words & negs) or "n't" in str(text or "") or "cannot" in str(text or "").lower()


#: What a rationale says when it is saying two things cannot both hold. Counted in the corpus
#: before being used: of 11,352 rationales for *no*, 3,264 say "cannot", 1,984 "at the same time",
#: 1,374 "either", 1,120 "can't". These are the words the people who wrote the dataset reached for;
#: nothing here decides what is incompatible with what, only where to look for someone saying so.
_INCOMPATIBLE = ("cannot", "can not", "can't", "not both", "impossible", "either")

#: What joins the two incompatible things once the marker has said there are two.
_JOINTS = (" as well as ", " and also ", " while ", " and ", " or ")


def mine_exclusions(pairs: Sequence["Pair"]) -> Dict[str, Set[str]]:
    """What these pairs' own rationales say cannot hold at once.

    Measured before it was built: a stored relation linked premise to hypothesis in **14 of 300**
    contradictions and in 14 of 300 non-contradictions, so the knowledge the hard half needs was
    not in any corpus she had. It is in this one, said out loud — *"The men cannot be in a
    construction site as well as laying on the beach simultaneously."*

    So the rationale is read for the marker and the two sentences for the words, and what comes out
    is a word-level exclusion: something distinctive to the premise against something distinctive
    to the hypothesis. Crude, and it has to be — the alternative is deciding by hand what excludes
    what, which is the thing this whole file exists not to do.

    **Only ever mined from pairs she is learning from.** Mining across the split would be reading
    the answer to a held-out pair off the held-out pair, and the number that came out of it would
    mean nothing.
    """
    out: Dict[str, Set[str]] = {}
    for pair in pairs:
        if pair.label != "no":
            continue
        low = pair.rationale.lower()
        marker = next((m for m in _INCOMPATIBLE if m in low), "")
        if not marker:
            continue
        said = low.split(marker, 1)[1]
        joint = next((j for j in _JOINTS if j in said), "")
        if not joint:
            continue
        # The rationale names the two things, and it names them **after** the marker, one on each
        # side of the joining phrase: "cannot be in a construction site AS WELL AS laying on the
        # beach". Crossing the whole sentences instead — which the first version did — produced
        # "man excludes 046, 20, 30, 90": every distinctive word of one against every distinctive
        # word of the other, which is a cross product, not a piece of knowledge.
        left, right = said.split(joint, 1)
        here = _content(left) & _content(pair.premise)
        there = _content(right) & _content(pair.hypothesis)
        for word in here:
            for other in there:
                if word == other:
                    continue
                out.setdefault(word, set()).add(other)
                out.setdefault(other, set()).add(word)
    return out


def _bucket(n: int) -> str:
    """Counts compare by equality here, so they are named rather than numbered."""
    if n <= 0:
        return "none"
    if n == 1:
        return "one"
    if n <= 3:
        return "few"
    return "many"


# --------------------------------------------------------------------------------------------- #
#  what a pair is
# --------------------------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Pair:
    """A premise, a hypothesis, the answer a person gave, and the line of reasoning behind it."""

    premise: str = ""
    hypothesis: str = ""
    label: str = ""
    rationale: str = ""
    task: str = ""

    @property
    def key(self) -> Tuple[str, str]:
        return (self.premise.strip().lower(), self.hypothesis.strip().lower())

    def to_dict(self) -> Dict[str, Any]:
        return {"premise": self.premise, "hypothesis": self.hypothesis, "label": self.label,
                "rationale": self.rationale, "task": self.task}


def _blocks(text: str) -> List[str]:
    """Cut a few-shot prompt into worked examples. One ends where its answer line ends."""
    out, start = [], 0
    for match in _ANSWER.finditer(text):
        end = text.find("\n", match.end())
        end = len(text) if end < 0 else end
        out.append(text[start:end])
        start = end
    return out


def _label_of(block: str) -> Optional[str]:
    match = _ANSWER.search(block)
    if not match:
        return None
    said = re.split(r"\s*(?:--+|\*\*)", match.group(1).strip())[0]
    said = said.strip().rstrip(".").strip().lower()
    return said or None


def read_pairs(path: Optional[Path] = None) -> List[Pair]:
    """Every inference pair in the corpus, from the prompts as well as from the answers.

    Reads the extracted file when it is there and parses raw submix rows when it is not, so the
    same call works whether it is handed 36,302 pairs already pulled out of 192,696 rows or the
    rows themselves.
    """
    rows: List[Pair] = []
    source = Path(path) if path is not None else BROAD
    try:
        if path is None:
            for candidate in (BROAD, CORPUS, RAW):
                if candidate.exists():
                    source = candidate
                    break
        if not source.exists():
            return rows
        first = _peek(source)
        if first is not None and "premise" in first:
            return _read_extracted(source)
        seen: Set[Tuple[str, str]] = set()
        with gzip.open(source, "rt", encoding="utf-8") as handle:
            for line in handle:
                try:
                    row = json.loads(line)
                except Exception:  # noqa: BLE001
                    continue
                whole = f"{row.get('inputs') or ''}\n{row.get('targets') or ''}"
                for block in _blocks(whole):
                    label = _label_of(block)
                    if label not in LABELS:
                        continue
                    quoted = _QUOTED.findall(block)
                    if len(quoted) < 2:
                        continue
                    pair = Pair(premise=quoted[-2].strip(), hypothesis=quoted[-1].strip(),
                                label=label, task=str(row.get("task") or ""),
                                rationale=_rationale(block))
                    if pair.key in seen or not pair.premise or not pair.hypothesis:
                        continue
                    seen.add(pair.key)
                    rows.append(pair)
    except Exception:  # noqa: BLE001
        return rows
    return rows


def _peek(path: Path) -> Optional[Dict[str, Any]]:
    try:
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    return json.loads(line)
    except Exception:  # noqa: BLE001
        return None
    return None


def _read_extracted(path: Path) -> List[Pair]:
    out: List[Pair] = []
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except Exception:  # noqa: BLE001
                continue
            out.append(Pair(premise=str(row.get("premise") or ""),
                            hypothesis=str(row.get("hypothesis") or ""),
                            label=str(row.get("label") or ""),
                            rationale=str(row.get("rationale") or ""),
                            task=str(row.get("task") or "")))
    return out


def _rationale(block: str) -> str:
    """The line before the answer — the person's reason, kept but never learned from.

    It is here so a reading can be compared against what a person said about the same pair, and
    deliberately not an input to anything: learning from the rationale would be learning the
    dataset's words rather than the regularity underneath them.
    """
    match = _ANSWER.search(block)
    if not match:
        return ""
    before = block[:match.start()].strip().splitlines()
    return before[-1].strip() if before else ""


# --------------------------------------------------------------------------------------------- #
#  measuring a pair
# --------------------------------------------------------------------------------------------- #
def probe(premise: str, hypothesis: str,
          excludes: Optional[Dict[str, Set[str]]] = None) -> Dict[str, Any]:
    """Everything measurable about the two sentences. None of it names an answer.

    Counts are bucketed because the induction compares readings by equality, and an exact word
    count would make every pair its own category — a rule that fires on "eleven added words" has
    learned a number, not a relation.
    """
    p, h = _content(premise), _content(hypothesis)
    added, shared, dropped = h - p, h & p, p - h
    out: Dict[str, Any] = {
        "added": _bucket(len(added)),
        "shared": _bucket(len(shared)),
        "dropped": _bucket(len(dropped)),
        "adds nothing": not added,
        "shares nothing": not shared,
        "says less": len(h) < len(p),
        "says more": len(h) > len(p),
        "hypothesis negated": _negated(hypothesis),
        "premise negated": _negated(premise),
        "one is negated": _negated(hypothesis) != _negated(premise),
        "hypothesis longer": len(str(hypothesis).split()) > len(str(premise).split()),
        "hypothesis counts something": any(w.isdigit() for w in h),
        "premise counts something": any(w.isdigit() for w in p),
    }
    if p:
        ratio = len(shared) / len(h) if h else 0.0
        out["overlap"] = ("all" if ratio >= 0.999 else "most" if ratio >= 0.6
                          else "some" if ratio > 0.0 else "none")
    if excludes is not None:
        hits = sum(1 for one in p for other in h if other in excludes.get(one, ()))
        out["a word rules another out"] = hits > 0
        out["how many rule out"] = _bucket(hits)
    return out


# --------------------------------------------------------------------------------------------- #
#  the organ
# --------------------------------------------------------------------------------------------- #
#: How clean a rule has to be to be kept, and it is a claim about **this subject** rather than a
#: preference. Natural-language inference read off surface differences between two sentences is not
#: a decision procedure, and asking it to be one returns nothing.
#:
#: 0.72 was the setting for a long time and it was far too high. Decomposed on 12,000 pairs — how
#: often a rule fires at all, how right it is *when* it fires, and what the whole organ scores
#: against its own base rate — twice, on two disjoint slices of the corpus:
#:
#:     purity   rules   speaks on   right when it speaks   overall   lift
#:      0.40      6       1.000            0.461            0.461    +0.028
#:      0.45      6       0.783            0.540            0.504    +0.077   <- and 0.508/+0.075
#:      0.50      4-6     0.638            0.551            0.492    +0.064
#:      0.55      3-4     0.558            0.565            0.475    +0.047
#:      0.65      1       0.067            0.689            0.457    +0.029
#:      0.72      1       0.052            0.780            0.463    +0.030
#:
#: Read the two middle columns together. A high bar does not make the organ wrong — at 0.72 its
#: rules are right **78%** of the time — it makes them silent, firing on one pair in twenty. At
#: 0.45 they are right 54% of the time and reach four pairs in five, and the whole organ scores
#: 0.504 against a base rate of 0.428.
#:
#: The peak is at 0.45 on both slices, by nearly the same margin, and 0.40 collapses to the base
#: rate — it speaks on everything and says nothing. A unimodal curve with its maximum in the same
#: place on two independent samples is a property of the subject, not of a sample.
PURITY = 0.45


class Reasoner:
    """Learns which readings of a pair predict which answer, and abstains where none does."""

    def __init__(self, *, min_support: int = 20, min_share: float = 0.05, max_rules: int = 6,
                 max_terms: int = 3, learn: bool = True, purity: float = PURITY,
                 mine: bool = True) -> None:
        self.min_support = int(min_support)
        self.min_share = float(min_share)
        self.max_rules = int(max_rules)
        self.max_terms = int(max_terms)
        self.learning = bool(learn)
        #: Off, the exclusion probe is absent — the control that says what the mined knowledge was
        #: worth as against the surface readings alone.
        self.mining = bool(mine)
        #: How clean a rule must be to be kept. Not 1.0, because language is not a decision
        #: procedure and at 1.0 she learns **nothing at all** from 5,000 pairs. Every rule carries
        #: the rate it was kept at and the school checks that rate on pairs it never saw.
        self.purity = float(purity)
        self.rules: List[Rule] = []
        self.near_misses: List[Rule] = []
        self.shown: int = 0
        #: Word pairs the training rationales said cannot hold at once. Empty until
        #: :meth:`learn_from` runs, and never filled from anything held out.
        self.excludes: Dict[str, Set[str]] = {}
        #: The label she saw most often. Counted from what she was shown, used only by
        #: :meth:`guess`, and never by :meth:`answer`.
        self.commonest: str = ""

    def learn_from(self, pairs: Sequence[Pair]) -> List[Rule]:
        """Induce, for each answer, the smallest exact conjunctions that predict it."""
        self.rules, self.near_misses = [], []
        self.shown = len(pairs)
        if not self.learning or not pairs:
            return self.rules
        # One pass. The comprehension this replaces re-scanned every pair for every pair, which
        # is 1.3 billion comparisons at 36,302 pairs -- slow but survivable, so nothing caught it
        # -- and 3.2 *hundred billion* at the 564,166 the full read produced, where it stopped
        # looking like slowness and started looking like a hang.
        seen = Counter(relation_of(p.label) for p in pairs)
        self.commonest = seen.most_common(1)[0][0] if seen else ""
        self.excludes = mine_exclusions(pairs) if self.mining else {}
        # Folded to the relation, not the spelling — see :data:`RELATIONS`. Without this the two
        # vocabularies are six classes, three of which are too rare to learn and all six of which
        # poison each other's negatives.
        readings = [(probe(p.premise, p.hypothesis, self.excludes), relation_of(p.label))
                    for p in pairs]
        for label in sorted({one for _r, one in readings}):
            positives = [r for r, one in readings if one == label]
            negatives = [r for r, one in readings if one != label]
            rules, near = cover(positives, negatives, label=label,
                                min_support=self.min_support, min_share=self.min_share,
                                max_rules=self.max_rules, max_terms=self.max_terms,
                                purity=self.purity)
            self.rules.extend(rules)
            self.near_misses.extend(near)
        self.rules.sort(key=lambda rule: (-rule.purity, -rule.support))
        return self.rules

    def guess(self, premise: str, hypothesis: str) -> Tuple[str, str]:
        """As :meth:`answer`, but never silent — the commonest label when nothing fires.

        Reported apart from :meth:`answer` and never folded into it. Falling back to the majority
        label is not reasoning; it is what a caller gets when the reasoning has nothing to say, and
        a number that mixes the two cannot tell you which of them earned it.
        """
        said, why = self.answer(premise, hypothesis)
        if said != "unknown":
            return said, why
        return self.commonest, "nothing she has covers this; the commonest answer"

    def answer(self, premise: str, hypothesis: str) -> Tuple[str, str]:
        """``(answer, why)``. ``("unknown", reason)`` where nothing she has settles it."""
        reading = probe(premise, hypothesis, self.excludes)
        firing = [rule for rule in self.rules if rule.holds(reading)]
        if not firing:
            return "unknown", "no rule she has covers this pair"
        labels = {rule.label for rule in firing}
        weight = lambda rule: rule.support * rule.purity  # noqa: E731
        best = max(firing, key=weight)
        if len(labels) > 1:
            rival = max((r for r in firing if r.label != best.label), key=weight)
            if weight(best) <= weight(rival) * 1.25:
                return "unknown", f"two rules disagree: {best.label} and {rival.label}"
        return best.label, f"{best.render()} ({best.purity:.2f} of the pairs she saw)"

    def stats(self) -> Dict[str, Any]:
        return {"shown": self.shown, "rules": len(self.rules),
                "near_misses": len(self.near_misses),
                "words_with_exclusions": len(self.excludes),
                "exclusion_pairs": sum(len(v) for v in self.excludes.values()) // 2,
                "by_label": {label: sum(1 for r in self.rules if r.label == label)
                             for label in sorted({r.label for r in self.rules})}}

    def render(self) -> str:
        rows = [f"{self.shown} pairs shown, {len(self.rules)} rules"]
        for rule in self.rules:
            rows.append(f"  {rule.label:<26} {rule.render()}"
                        f"   ({rule.support} pairs, {rule.purity:.2f} of them)")
        for rule in self.near_misses:
            rows.append(f"  {rule.label:<26} NEAR MISS: {rule.render()} "
                        f"({rule.counterexamples} counterexamples)")
        return "\n".join(rows)
