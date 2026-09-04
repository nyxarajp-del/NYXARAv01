"""NYXARA · njp/explainread.py — reading a *why* or a *how* off the surface (🧭, NJP V.36).

This file is a **question grammar**, not a knowledge source, and the distinction is the point of
separating it from :mod:`nyxara.njp.explain`. Nothing here knows anything about the world. It
turns English into a pair — *which walk*, and *what to walk from* — and every answer is then
produced by the walk out of the fact store. Delete every fact she has and this module reads
exactly as well and answers exactly nothing, which a test asserts.

It exists because of a number. :mod:`nyxara.njp.general` measures ``inheritance`` at 400/400 and
records that **every one of those arrived through the derivation ladder and none through English**
— a real capability with no sentence that reaches it. Building a why-walk and leaving it reachable
only from Python would repeat that exactly, so the walk and the grammar were written together and
the exam asks both channels.

Four questions, and telling them apart is most of the work
----------------------------------------------------------
=================================  =========  ====================================================
Surface                            Walk       Why it is its own case
=================================  =========  ====================================================
why does X happen / why is X Y     because    Backwards along production. The default reading of
                                              *why*, and the one that must not answer a purpose.
what is X for / why do we have X   for        Forwards along ``purpose``. English says *why* for
                                              this too, and the two are different questions with
                                              different right answers about the same X.
what does X need                   needs      An enabling condition. Already had a one-hop answer
                                              through ``requires``; it is listed here so that a
                                              *why* is never allowed to return one silently.
how does X work                    mechanism  Parts and what they do. Not *why*, reversed.
how do you X / steps to X          procedure  Steps and a derived order.
=================================  =========  ====================================================

Two of those are ambiguous in English and are resolved here rather than in the walk. *"Why do we
have a fuse"* is a purpose question wearing a *why*, and ``we have`` is what says so — nobody asks
what brought a fuse about with that sentence. *"How does a heart work"* and *"how do you start a
heart"* differ by the pronoun and by nothing else.

Candidates, not a subject
-------------------------
:func:`read_explanation_question` returns **several spellings of the topic**, best first, because
the store's key and the question's phrasing are written by different hands and one of them
inflects. *"Why does water boil"* names an event the corpus files under ``boiling``; *"how does a
plant make food"* names one it files under ``photosynthesis`` and also under ``plant``. The reader
does not know which, and guessing one is how a question with an answer in the store returns
silence.

So it offers the phrase, the phrase without its subject, the verb nominalised, and the bare
subject, and :meth:`~nyxara.njp.explain.Explainer.ask` takes the first that the graph actually
reaches. That ordering is the claim: the most specific spelling that has an answer wins, and a
bare-subject answer is only reached when nothing more specific had one. Measured before the
nominalisation existed, *why does water boil* returned silence with ``boiling occurs_when the
temperature reaches the boiling point`` in the store.

The nominaliser is four suffix rules and it is deliberately not more. It is not a morphology — the
package has one of those in :mod:`nyxara.njp.language` — and it is not a verb list. It turns
``boil`` into ``boiling`` and ``freeze`` into ``freezing`` and it gets ``go`` wrong, and a wrong
candidate costs one failed lookup, which is why guessing cheaply is affordable here and would not
be if a candidate could produce a wrong answer rather than none.

Pure standard library, no state, deterministic.
"""

from __future__ import annotations

import re
from typing import Any, List, Optional, Pattern, Tuple

__all__ = ["read_explanation_question", "candidates_for", "nominalise", "STOPPED"]

#: Words that are never the topic on their own. A question that reduces to one of these has named
#: nothing, and returning it as a candidate sends the walk at a hub with thousands of edges.
STOPPED = frozenset({
    "it", "this", "that", "they", "them", "we", "you", "he", "she", "i", "one", "thing",
    "things", "something", "anything", "there", "here", "the", "a", "an", "some", "any",
})

#: Auxiliaries a wh-question inverts. Stripped after the wh-word and before the subject.
_AUX = r"(?:does|do|did|is|are|was|were|has|have|had|will|would|can|could|should|must)"

#: The verbs that mean *"nothing beyond the subject"* — ``why does X happen`` names X, not an
#: event called "X happen". Everything else after the subject is part of what is being asked
#: about, which is why this set is small and closed rather than a list of common verbs.
_EMPTY_VERBS = frozenset({"happen", "happens", "happened", "occur", "occurs", "occurred",
                          "exist", "exists", "take place", "takes place"})

# The purpose reading of *why*. All three say "what is this for" in English and none of them is
# asking what brought the thing about.
_PURPOSE_WHY = (
    re.compile(rf"^why\s+(?:{_AUX}\s+)?(?:we|you|people|they|one)\s+"
               r"(?:have|need|use|want|keep)\s+(?P<s>.+?)\s*[?.!]*$", re.I),
    re.compile(r"^why\s+(?:is|are|was|were)\s+(?P<s>.+?)\s+(?:used|needed|there|useful)\b"
               r".*?[?.!]*$", re.I),
)

_PURPOSE_WHAT = (
    re.compile(r"^what\s+(?:is|are|was|were)\s+(?:the\s+)?(?:purpose|point|use|function|role)\s+"
               r"of\s+(?P<s>.+?)\s*[?.!]*$", re.I),
    re.compile(r"^what\s+(?:is|are)\s+(?P<s>.+?)\s+(?:used\s+)?for\s*[?.!]*$", re.I),
    re.compile(r"^what\s+(?:does|do)\s+(?P<s>.+?)\s+(?:do|achieve)\s*[?.!]*$", re.I),
)

_NEEDS = (
    re.compile(r"^what\s+(?:does|do|did)\s+(?P<s>.+?)\s+(?:need|require)\s*[?.!]*$", re.I),
    re.compile(r"^what\s+(?:is|are)\s+(?:needed|required|necessary)\s+for\s+(?P<s>.+?)"
               r"\s*[?.!]*$", re.I),
)

_MECHANISM = (
    re.compile(r"^how\s+(?:does|do|did)\s+(?P<s>.+?)\s+(?:work|function|operate)\s*[?.!]*$", re.I),
    re.compile(r"^how\s+(?:is|are|was|were)\s+(?P<s>.+?)\s+(?:produced|made|formed|caused)"
               r"\s*[?.!]*$", re.I),
    re.compile(r"^what\s+(?:is|are)\s+(?:the\s+)?(?:mechanism|workings)\s+of\s+(?P<s>.+?)"
               r"\s*[?.!]*$", re.I),
    # The general case, and it is last for a reason: *how does a plant make food* is a mechanism
    # question with no mechanism word in it, and the three patterns above would each rather have
    # it than let it fall to *procedure*. It matches almost any `how does X …`, so anything more
    # specific has to be written above it. Measured: without this line the cold run's *how does a
    # plant make food* and *how does an engine work* parsed as nothing at all.
    re.compile(r"^how\s+(?:does|do|did|is|are|was|were)\s+(?P<s>.+?)\s*[?.!]*$", re.I),
)

# A procedure is asked with a second person or an infinitive, and that is the whole difference
# from a mechanism. "how does bread rise" is a mechanism; "how do you make bread" is a procedure,
# and the corpus files them under different subjects.
_PROCEDURE = (
    re.compile(rf"^how\s+(?:{_AUX}\s+)?(?:you|i|we|one|people)\s+(?P<s>.+?)\s*[?.!]*$", re.I),
    re.compile(r"^how\s+to\s+(?P<s>.+?)\s*[?.!]*$", re.I),
    re.compile(r"^what\s+(?:are\s+)?(?:the\s+)?steps\s+(?:of|for|to|in)\s+(?P<s>.+?)"
               r"\s*[?.!]*$", re.I),
    re.compile(r"^(?:what\s+is\s+)?(?:the\s+)?(?:procedure|process)\s+for\s+(?P<s>.+?)"
               r"\s*[?.!]*$", re.I),
)

# Plain *why*, last, so the purpose readings above get first refusal at the same sentence.
_BECAUSE = (
    re.compile(rf"^why\s+{_AUX}\s+(?P<s>.+?)\s*[?.!]*$", re.I),
    re.compile(r"^what\s+(?:is|are|was|were)\s+(?:the\s+)?(?:cause|causes|reason|reasons)\s+"
               r"(?:of|for|behind)\s+(?P<s>.+?)\s*[?.!]*$", re.I),
    re.compile(r"^why\s+(?P<s>.+?)\s*[?.!]*$", re.I),
)

#: Order matters and is the resolution of the two ambiguities named in the docstring. Procedures
#: before mechanisms so *how do you* is never read as *how does*; purposes before causes so
#: *why do we have* is never read as *why does*.
_TABLE: Tuple[Tuple[str, Tuple[Any, ...]], ...] = (
    ("procedure", _PROCEDURE),
    ("mechanism", _MECHANISM),
    ("for", _PURPOSE_WHY + _PURPOSE_WHAT),
    ("needs", _NEEDS),
    ("because", _BECAUSE),
)


def nominalisations(verb: str) -> List[str]:
    """Every noun the verb might be filed under, cheapest first.

    ``accelerate`` is filed as ``acceleration`` and ``boil`` as ``boiling``, and there is no rule
    that gets both from one guess. Since a candidate that reaches nothing costs a dictionary miss,
    the honest move is to offer both rather than to pick — the store decides which exists.
    Measured: *why does an object accelerate* returned silence with ``force causes acceleration``
    in the store, because ``accelerating`` is not a subject anywhere.
    """
    word = str(verb or "").strip().lower()
    if not word or " " in word:
        return []
    out = [nominalise(word)]
    if word.endswith("ate"):
        out.append(word[:-1] + "ion")       # accelerate -> acceleration
    elif word.endswith("e"):
        out.append(word[:-1] + "ion")       # dissolve -> dissolution (wrong, and free)
    elif word.endswith(("t", "d")):
        out.append(word + "ion")            # erupt -> eruption
    out.append(word + "ion")
    seen: List[str] = []
    for got in out:
        if got and got != word and got not in seen:
            seen.append(got)
    return seen


def nominalise(verb: str) -> str:
    """``boil`` → ``boiling``. Four suffix rules, and it is allowed to be wrong.

    See the module docstring: a wrong candidate costs a lookup that finds nothing, never a wrong
    answer, so the cheap rule is the right size of machinery here.
    """
    word = str(verb or "").strip().lower()
    if not word or " " in word or word.endswith("ing"):
        return word
    if word.endswith("ie"):                 # die -> dying
        return word[:-2] + "ying"
    if word.endswith("e") and not word.endswith(("ee", "oe", "ye")):
        return word[:-1] + "ing"            # freeze -> freezing
    if (len(word) > 2 and word[-1] not in "aeiouwxy"
            and word[-2] in "aeiou" and word[-3] not in "aeiou"):
        return word + word[-1] + "ing"      # stop -> stopping
    return word + "ing"


def _strip(phrase: str) -> str:
    out = " ".join(str(phrase or "").split())
    out = re.sub(r"^(?:the|a|an)\s+", "", out, flags=re.I)
    return out.strip(" ?.!,;:").strip()


def candidates_for(phrase: str, *, verbal: bool = False) -> List[str]:
    """Every spelling of the topic worth a lookup, most specific first.

    *"water boil"* yields ``water boil``, ``boiling``, ``boil``, ``water`` — the phrase as said,
    the last word nominalised, the last word bare, and the first word. A caller takes the first
    that the graph reaches, which is why the ordering is by specificity and not by likelihood.

    ``verbal`` says the phrase **begins** with its verb rather than ending in one, which is the
    shape a procedure is asked in: *"boil an egg"*, not *"water boil"*. The corpus files a
    procedure under its gerund — ``boiling an egg`` — so nominalising the head is what reaches it,
    and nominalising the tail, which is what the default does, reaches ``egging``.
    """
    base = _strip(phrase)
    if not base:
        return []
    out: List[str] = []

    def add(text: str) -> None:
        got = _strip(text)
        if got and got.lower() not in STOPPED and got not in out:
            out.append(got)

    words = base.split()
    add(base)
    if verbal and len(words) > 1:
        add(" ".join([nominalise(words[0])] + words[1:]))
        add(nominalise(words[0]))
        add(words[0])
        add(" ".join(words[1:]))
        return out
    if len(words) > 1:
        tail = words[-1]
        # "why does water boil" — the event is what the last word names, and the corpus files
        # events under a noun form far more often than under the bare verb.
        for form in nominalisations(tail):
            add(form)
        # Then the phrase's own subject, and only after it the trailing word on its own. That
        # order is a defect fixed rather than a preference. *How does a plant make food* has the
        # trailing word "food", which is a subject in the store with edges of its own, so the
        # bare-tail candidate answered a question about food — `food -> nutrients` — for a
        # question about plants. A less specific candidate that reaches *something* is the one
        # failure mode a candidate ladder has, and the guard is that the phrase's head is tried
        # before any word taken out of its middle or end.
        add(words[0])
        add(tail)
        add(" ".join(words[1:]))
    return out


#: The induced reader, or ``None`` while nothing has been taught. Set by
#: :func:`nyxara.njp.asking.install`, never by this module — a table does not get to appoint its
#: own replacement, and a process that has been shown nothing must not be consulted.
LEARNED: Any = None


def read_explanation_question(text: str) -> Optional[Tuple[str, Tuple[str, ...]]]:
    """``(walk, candidate topics)`` or ``None`` when this is not a why/how question.

    ``None`` rather than a guess is what lets a caller fall through to the ordinary answer path.
    A grammar that claimed every sentence would put a why-walk in front of *what is a mammal*.
    """
    raw = " ".join(str(text or "").split())
    if not raw:
        return None
    # The table first, the induced reader second, and that order is a decision.
    #
    # The table is a dozen phrasings somebody wrote down and it is *exactly right* about those
    # dozen — `what is X used for` is a purpose question and no amount of induction will make it
    # more so. What it cannot do is reach the thirteenth, which the gauntlet measured at 4 of 40.
    # So the induced reader is not a replacement, it is the fallback that has no edge: what the
    # table knows, it answers; what it has never seen, `Asking` may have been taught.
    #
    # Putting induction first would be worse and it is worth saying why. The table's readings were
    # each measured against `Grounder._read_question`; a learned cue is corroborated by two
    # demonstrations. Two demonstrations is enough to be worth consulting and not enough to
    # overrule something that was measured.
    for kind, patterns in _TABLE:
        for pattern in patterns:
            match = pattern.match(raw)
            if not match:
                continue
            phrase = _strip(match.group("s"))
            if not phrase:
                continue
            words = phrase.split()
            # "why does rain happen" is about rain. The trailing verb carries nothing and leaving
            # it on makes the topic a string no subject key will ever equal.
            if len(words) > 1 and words[-1].lower() in _EMPTY_VERBS:
                phrase = " ".join(words[:-1])
            got = candidates_for(phrase, verbal=(kind == "procedure"))
            if not got:
                continue
            return kind, tuple(got)
    if LEARNED is not None:
        try:
            return LEARNED.read(raw)
        except Exception:  # noqa: BLE001 — a learner that raises must not break the reader
            return None
    return None
