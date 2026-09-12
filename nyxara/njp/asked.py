"""NYXARA · njp/asked.py — what kind of thing an answer has to be (❓).

Asked a thousand real questions out of FLAN, her fact store answers seventy of them and gets
**none** right. The three that follow are not near misses:

    "When was the Battle of the Coral Sea fought?"      -> coastal protection
    "what type of government does japan currently have?" -> island country
    "what is the latest operating system for android?"   -> system software

Look at what is wrong with them. It is not that she does not know the date of the Coral Sea — she
does not, and saying so would have been a perfectly good answer. It is that *"coastal protection"
cannot be an answer to "when"*. Nothing in the package could see that, because nothing in it held
any idea of what a question is **asking for**. The grounder matches a subject and a predicate and
returns whatever object scored highest, and an object is an object.

So this module learns one thing: **the shape of the answer a question wants.** Not the answer —
the shape. A ``when`` question wants a time. A ``how many`` wants a number. A ``did`` wants a yes
or a no. Knowing that turns a confidently wrong answer into an abstention, which is the difference
between being unreliable and being honest about a gap.

**It is not a table of question words.** The temptation is obvious and it is refused twice over.
Two sets of *generic surface measurements* are taken — one of the question, one of the answer —
and **not one of them names a category**:

* of a question: its first word, its first two words, whether a copula follows the wh-word,
  whether the word after it is a noun or an adjective, whether it holds a number, how long it is;
* of an answer: how many tokens, whether every token is a digit, whether it holds a four-digit
  number in the range a year falls in, whether it holds a unit-like token, whether it is one of
  the words a polar question is answered with, whether it opens capitalised mid-sentence.

Which question reading predicts which answer reading is then induced by :mod:`nyxara.njp.induce`
— the same greedy cover that learned what breaks a program and what makes one sentence follow
from another — from questions she has been shown, and measured against questions she has not.
``how many -> a number`` is a *finding* here, not a line of code.

The answer kinds themselves are the one place a name appears, and they are named from the answer's
own surface rather than from what anybody thinks a question means: ``polar``, ``count``, ``year``,
``span``, ``phrase``. A row is filed under the first that fits, and the fit is arithmetic.

What this is for is :meth:`Asked.contradicts`, which is a *veto* and not an answerer. It never
supplies a fact and it never raises a confidence. Handed a question and a candidate answer, it
says whether the candidate is the wrong kind of thing — and only when the rule that says so was
induced, so a question it has no rule for is one it says nothing about.

Pure standard library.
"""

from __future__ import annotations

import gzip
import json
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from nyxara.njp.induce import Rule, cover

__all__ = ["Question", "Asked", "KINDS", "CORPUS", "read_questions", "shape_of",
           "satisfies", "probe"]

#: Where the questions come from: the fold of the full FLAN read, filtered by
#: ``scripts/merge_flan_shards.usable`` — which, among other things, drops the 14,485 rows whose
#: source task exists to produce a *wrong* answer.
CORPUS = Path(__file__).with_name("data") / "flan_qa.jsonl.gz"

#: The kinds an answer's surface can fall into. Named from the surface and nothing else: a row is
#: a ``count`` because every token in it is a digit, not because somebody decided the question was
#: a counting question. Ordered, and the first fit wins — ``1994`` is a year before it is a count.
KINDS: Tuple[str, ...] = ("polar", "year", "count", "span", "phrase")

_WORD = re.compile(r"[^\W\d_][\w'’\-]*|\d+(?:[.,]\d+)*", re.UNICODE)
_YEAR = re.compile(r"\b(?:1[0-9]{3}|20[0-9]{2})\b")
_DIGIT = re.compile(r"^[\d,.]+$")

#: The words a polar question is answered with. Counted off the corpus rather than chosen: these
#: are the only single-token answers that occur more than a thousand times each and never occur as
#: the answer to a ``what``/``who``/``when``/``where`` question in the training half.
_POLAR = ("yes", "no", "true", "false")


def shape_of(answer: str) -> str:
    """Which kind this answer's surface falls into. Arithmetic, and about the answer alone."""
    text = " ".join(str(answer or "").split()).strip(" .")
    if not text:
        return ""
    if text.lower() in _POLAR:
        return "polar"
    tokens = _WORD.findall(text)
    if not tokens:
        return ""
    if _YEAR.search(text) and len(tokens) <= 3:
        return "year"
    if all(_DIGIT.match(t) for t in tokens):
        return "count"
    if len(tokens) <= 3:
        return "span"
    return "phrase"


#: What counts as short: the kinds :func:`shape_of` reaches only for an answer of three tokens or
#: fewer. They are not rivals — ``1947`` is a ``year`` rather than a ``span`` because the year test
#: runs first, and it is a one-token answer either way.
_SHORT = frozenset(("year", "count", "span"))

#: Numeric: the kinds whose surface is a number. A counting question is answered by either.
_NUMERIC = frozenset(("count", "year"))


def satisfies(got: str, wanted: str) -> bool:
    """Whether an answer of kind ``got`` answers a question that wants ``wanted``.

    Not every difference between two kinds is a contradiction, and treating it as one is how a
    veto starts refusing correct answers. The kinds come off **one ordered run of surface tests**,
    so the later ones are the coarser ones: an answer is a ``year`` only because the year test is
    tried before the span test, and it is a three-token answer either way. Refusing ``1947`` as an
    answer to *"what year was the film released?"* — which this did, until the itemised list of
    its own mistakes was read — is refusing on the order the tests happen to run in.

    So: anything short answers a question that wants something short; a number answers a question
    that wants a count; a span answers a question that wants a phrase and the other way round,
    because that pair differs only in wordiness. A polar question is the one real divide, and it
    is a divide in both directions.
    """
    if not wanted or got == wanted:
        return True
    if wanted == "count":
        return got in _NUMERIC
    if wanted == "year":
        return got in _NUMERIC
    if wanted == "span":
        return got in _SHORT or got == "phrase"
    if wanted == "phrase":
        return got in ("span", "phrase")
    return False


@dataclass(frozen=True)
class Question:
    """One question, its answer, and where both came from."""

    question: str = ""
    answer: str = ""
    task: str = ""
    source: str = ""
    licence: str = ""

    @property
    def kind(self) -> str:
        return shape_of(self.answer)


def _closed() -> Dict[str, str]:
    try:
        from nyxara.njp.semantics import _CLOSED  # noqa: WPS433
        return _CLOSED
    except Exception:  # noqa: BLE001
        return {}


def probe(question: str) -> Dict[str, Any]:
    """Generic measurements of a question. Not one of them names an answer kind.

    Every entry is something a person could take off any sentence without knowing what a question
    is for: which word opens it, whether the next word is closed class, whether a copula or an
    auxiliary follows, whether a number appears, how long it runs. What any of it *predicts* is
    :meth:`Asked.learn_from`'s to find out.
    """
    closed = _closed()
    text = " ".join(str(question or "").split())
    tokens = [t.lower() for t in _WORD.findall(text)]
    if not tokens:
        return {}
    first = tokens[0]
    second = tokens[1] if len(tokens) > 1 else ""
    return {
        "opens": first,
        "opens_two": f"{first} {second}".strip(),
        "second_class": closed.get(second, "WORD"),
        "third_class": closed.get(tokens[2], "WORD") if len(tokens) > 2 else "",
        "has_number": bool(_DIGIT.search(text)) or any(_DIGIT.match(t) for t in tokens),
        "length": _bucket(len(tokens)),
        "ends_word": tokens[-1],
        "closed_share": _share(sum(1 for t in tokens if t in closed), len(tokens)),
    }


def _bucket(n: int) -> str:
    """Lengths compare by equality here, so they are named rather than numbered."""
    if n <= 5:
        return "short"
    if n <= 10:
        return "medium"
    if n <= 20:
        return "long"
    return "very long"


def _share(part: int, whole: int) -> str:
    if not whole:
        return "none"
    value = part / whole
    if value < 0.25:
        return "few"
    if value < 0.5:
        return "some"
    return "many"


def read_questions(path: Optional[Path] = None, limit: int = 0) -> List[Question]:
    source = Path(path) if path is not None else CORPUS
    out: List[Question] = []
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
            out.append(Question(question=str(row.get("question") or ""),
                                answer=str(row.get("answer") or ""),
                                task=str(row.get("task") or ""),
                                source=str(row.get("source") or ""),
                                licence=str(row.get("licence") or "")))
            if limit and len(out) >= limit:
                break
    return out


@dataclass
class Asked:
    """What kind of answer a question wants, induced from questions that had answers.

    Constructed and never taught it predicts nothing and vetoes nothing, which is the honest
    floor: a thing that has been shown no questions has no expectations about answers.

    Both thresholds come off tables in :mod:`nyxara.njp.askedschool`, on 20,000 questions learned
    from and 5,000 held out, and they are different numbers because they are different bets.

    ``purity`` is the bar for *predicting*, where reach is what matters:

        purity   right when it fires   fires on   rules
          0.60          0.792            0.912      19
        **0.70**      **0.802**        **0.847**    13
          0.75          0.833            0.661      10
          0.80          0.839            0.645      11
          0.85          0.888            0.341      11
          0.90          0.910            0.201       7

    ``veto_purity`` is the bar for *refusing*, and the column that decides it is the share of
    held-out questions whose **own correct answer** the veto would reject. Every one of those is a
    right answer suppressed, which is the failure this module exists to reduce:

        bar     veto fires on   rejects a correct answer
        0.70        0.847            0.0492  (246 of 5000)
        0.75        0.418            0.0378  (189 of 5000)
        0.80        0.376            0.0292  (146 of 5000)
        0.85        0.253            0.0040  ( 20 of 5000)
        **0.90**  **0.126**        **0.0004** (  2 of 5000)

    Half the reach of 0.85 for a tenth of the damage, so 0.90 is the setting. It was not always a
    setting: measured against the corpus **before** its multiple-choice labels and category-name
    answers were taken out, no rule reached 0.90 at all — the purest was 0.855 — so the veto never
    fired and this same column read a flawless ``0.0000``. A mechanism that does nothing is never
    wrong, and for about an hour that zero was reported as a result. What caught it was printing
    reach beside cost; :func:`~nyxara.njp.askedschool.veto_bar` prints both and a test refuses a
    row that has one without the other.

    Always guessing the commonest kind scores 0.590.
    """

    purity: float = 0.70
    #: The bar a rule must clear before it may *veto* rather than merely predict. Higher than
    #: :attr:`purity` because the two uses are not the same bet: predicting wants reach, vetoing
    #: wants to be right. Set by the false-veto column above and nothing else.
    veto_purity: float = 0.90
    min_support: int = 20
    min_share: float = 0.02
    max_rules: int = 6
    max_terms: int = 2
    learning: bool = True
    rules: List[Rule] = field(default_factory=list)
    near_misses: List[Rule] = field(default_factory=list)
    shown: int = 0
    #: The commonest kind overall, used only by :meth:`guess` and never by :meth:`expects`.
    commonest: str = ""

    def learn_from(self, questions: Sequence[Question]) -> List[Rule]:
        self.rules, self.near_misses = [], []
        self.shown = len(questions)
        if not self.learning or not questions:
            return self.rules
        readings = [(probe(q.question), q.kind) for q in questions]
        readings = [(r, k) for r, k in readings if r and k]
        if not readings:
            return self.rules
        self.commonest = Counter(k for _r, k in readings).most_common(1)[0][0]
        for kind in KINDS:
            positives = [r for r, k in readings if k == kind]
            negatives = [r for r, k in readings if k != kind]
            if not positives:
                continue
            rules, near = cover(positives, negatives, label=kind,
                                min_support=self.min_support, min_share=self.min_share,
                                max_rules=self.max_rules, max_terms=self.max_terms,
                                purity=self.purity)
            self.rules.extend(rules)
            self.near_misses.extend(near)
        return self.rules

    # -- using it ------------------------------------------------------------------------- #
    def expects(self, question: str) -> Tuple[str, str]:
        """The kind of answer this question wants, and why — or ``("", "")`` for no opinion.

        Silence is the default and is meant to be. A question no induced rule fires on is one this
        module knows nothing about, and a veto issued on no evidence would be worse than the wrong
        answer it was meant to stop.
        """
        label, why, _purity = self._best(question)
        return label, why

    def _best(self, question: str) -> Tuple[str, str, float]:
        """The purest rule that fires here, and how pure it was on the half it was induced from."""
        reading = probe(question)
        if not reading:
            return "", "", 0.0
        best: Optional[Rule] = None
        for rule in self.rules:
            if not rule.holds(reading):
                continue
            if best is None or (rule.purity, rule.support) > (best.purity, best.support):
                best = rule
        if best is None:
            return "", "", 0.0
        return best.label, best.render(), best.purity

    def guess(self, question: str) -> str:
        """As :meth:`expects`, falling back to the commonest kind. Never used by the veto."""
        kind, _why = self.expects(question)
        return kind or self.commonest

    def contradicts(self, question: str, answer: str) -> Tuple[bool, str]:
        """Is this candidate the wrong *kind* of thing for this question?

        A veto and nothing else. It supplies no fact, it raises no confidence, and it abstains
        wherever :meth:`expects` abstains. ``(True, why)`` means the candidate's own surface says
        it cannot be what was asked for — *"coastal protection"* offered for a ``when``.
        """
        wanted, why, purity = self._best(question)
        # A rule good enough to predict with is not automatically good enough to refuse with.
        if not wanted or purity < self.veto_purity:
            return False, ""
        got = shape_of(answer)
        if not got or satisfies(got, wanted):
            return False, ""
        return True, f"{wanted} expected ({why}), but {answer.strip()[:40]!r} is a {got}"

    def learned(self) -> Dict[str, Any]:
        return {"shown": self.shown, "rules": len(self.rules),
                "near_misses": len(self.near_misses), "commonest": self.commonest,
                "by_kind": {kind: [r.render() for r in self.rules if r.label == kind]
                            for kind in KINDS}}
