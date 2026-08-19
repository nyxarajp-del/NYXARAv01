"""NYXARA · njp/corpora.py — the ten sources, and what each one can actually become (🗺).

The Master named ten datasets in priority order and asked for NJP to be trained on them. This
module is the part of that which is a *decision* rather than a download: for each source, what
shape of NJP-readable row it converts into, and why the other shapes were refused.

**There are exactly three shapes, because there are exactly three ways into this brain.**

* ``triples`` — ``{"subject", "predicate", "object", "confidence"}``, loaded by
  :func:`nyxara.njp.ingest.ingest_triples` straight into the fact store.
* ``text`` — ``{"text": "..."}``, loaded by :func:`nyxara.njp.ingest.ingest_text`, which runs
  the grounder's own extractor over each statement.
* ``pairs`` — ``{"question", "answer"}``, taught turn-by-turn by
  :class:`nyxara.njp.study.Tutor`, which is the only path that can hold out a test set.

A source is assigned exactly one of them. That is a constraint on purpose: a source that fanned
out into all three would have its contribution to a held-out score spread across three loaders and
become unattributable, and attribution between stages is the whole reason
:mod:`nyxara.njp.train` exists.

**Why ``text`` exists at all, rather than a second extractor here.** Wikipedia, arXiv,
OpenWebMath and Gutenberg are prose. The obvious move is to write subject/predicate/object regexes
in this file and emit triples. That would be a second extractor, drifting against
:meth:`~nyxara.njp.grounding.Grounder._extract` from the day it was written, and the drift would
be silent — facts stored under predicate names her question grammar never forms. Measured, her own
extractor is already good at exactly this shape::

    _extract("Anarchism is a political philosophy.")
      -> ("Anarchism", "is_a", "political philosophy"), ("Anarchism", "is_a", "philosophy")

including the head-noun backoff. So the honest job for a prose source is **selection, not
extraction**: pick the sentences that state something, trim them to the clause that states it, and
hand them to the one extractor this package has.

**Selection is aggressive, and that is the point.** :meth:`Grounder.stats` reports ``unparsed`` —
54% of the bundled corpus's answers extract no triple at all. Feeding a million unfiltered
sentences would not raise what she knows, it would raise that number and the wall-clock. So
:func:`definitional_sentences` keeps only sentences with a copula or one of the relation verbs the
extractor reads, under a length bound, and drops the rest without apology. A prose corpus is
mostly narration; the fact store wants the definitions.

**No network, no third-party package, no brain.** Every function here takes rows that somebody
else fetched and returns rows somebody else writes. That is what makes the converters testable
offline and what keeps ``scripts/prepare_datasets.py`` a thin shell around them — the fetching is
the replaceable part, the conversion is not.

Pure standard library.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

__all__ = [
    "Source", "SOURCES", "FORMS", "by_key", "in_priority_order",
    "convert", "converter_for", "definitional_sentences", "sentences", "trim_clause",
]

#: The three ways into this brain. Anything else is a shape no loader reads.
FORMS = ("triples", "pairs", "text")


# --------------------------------------------------------------------------- #
# Prose → the sentences her extractor can actually use
# --------------------------------------------------------------------------- #

#: Sentence boundary. Deliberately crude — an abbreviation splitting early costs one truncated
#: sentence, which the definitional filter then drops, where a clever splitter costs a dependency.
_SENTENCE = re.compile(r"(?<=[.!?])\s+")

#: Where a definition stops being a definition. "Anarchism is a political philosophy and movement
#: **that is skeptical of all justifications for authority**" extracts an ``is_a`` whose object is
#: that entire tail — a string no question will ever match. Cutting at the first subordinator keeps
#: the head noun phrase, which is the part that is a kind.
_SUBORDINATE = re.compile(
    r"\s+(?:that|which|who|whom|whose|where|when|while|although|because|whereas)\s+"
    r"|\s*[;:—–]\s*|\s*,\s+(?:and|or|but|a|an|the)\s+", re.IGNORECASE)

#: A parenthetical in a lead sentence is a pronunciation guide, a date range or a native spelling.
#: None of them are the claim, and all of them wreck the object when they land inside it.
_PARENTHETICAL = re.compile(r"\s*[(\[][^)\]]{0,200}[)\]]")

#: Inline references and footnote markers, which arrive attached to the word before them.
_CITATION = re.compile(r"\[\d+\]|\{\{[^}]{0,80}\}\}")

#: What a definition looks like: a short subject, then one of the verbs
#: :meth:`Grounder._extract` actually reads. Two decisions in here were measured, not guessed.
#:
#: **The head constraint is the filter**, not the verb list. Searching for a relation verb anywhere
#: in the sentence lets narration straight through — "Bayes and his Theorem My earlier post on
#: Bayesian probability seems to *have* generated quite a lot of readers" matches on an auxiliary
#: eleven words in, and "As a historically left-wing movement, this reading of anarchism *is*
#: placed on the farthest left" matches ten words in with a subject the extractor would read as
#: that entire opening clause. Requiring the verb within the first six words drops both and keeps
#: "Anarchism is a political philosophy" and "Trials like this are usually called Bernoulli
#: trials". At least one word must precede it, so a fragment whose subject was lost to the
#: sentence splitter cannot be admitted headless.
#:
#: **The list is what extraction returns, not what English offers.** Run against
#: ``Grounder._extract`` one verb at a time, ``contains``, ``includes``, ``results in``,
#: ``belongs to``, ``composed of``, ``was caused by`` and ``have`` all return **no triple at all**
#: — there is no pattern for them. A sentence admitted on one of those is not a fact that got
#: stored badly, it is a row that costs the extractor a call and lands in ``stats()["unparsed"]``.
#: They are left out until the patterns exist. ``has`` stays because it extracts (as ``owns``),
#: and the copulas stay because they are the highest-yield shape in every prose source here.
_DEFINITIONAL = re.compile(
    r"^(?:\S+\s+){1,6}"
    r"(?:is|are|was|were|means|refers\s+to|consists\s+of|made\s+of|"
    r"causes?|requires?|needs?|leads\s+to|produces?|occurs?\s+when|has)\b",
    re.IGNORECASE)

#: A sentence that opens on a pronoun has a subject this file cannot resolve, and the extractor
#: would faithfully store a fact about "it". Anaphora is a real problem and guessing is worse than
#: dropping — a corpus this size can afford to keep only the sentences that name their subject.
_PRONOUN_START = re.compile(r"^(?:it|he|she|they|this|that|these|those|there|we|i|you)\b",
                            re.IGNORECASE)

#: Mathematical and markup noise. Any dollar sign or backslash at all, rather than a balanced
#: ``$…$`` pair: OpenWebMath's own text is full of half-open fragments — "!$ is the number of
#: distinct combinations of x objects" got through a balanced-pair test — and a fact whose object
#: contains a stray delimiter is a fact no question reaches.
_MATHY = re.compile(r"[$\\^]|\b_\{|={2,}")

#: Leading debris left by a crude sentence split: list bullets, chapter letters, the tail of an
#: ellipsis. Stripped rather than dropped, then the sentence must still begin with a letter.
_LEADING_JUNK = re.compile(r"^[^0-9A-Za-z\u0900-\u097F]+")

#: Bounds on a usable statement. The floor drops fragments and headings. Two ceilings, because
#: they answer different questions: ``_MAX_RAW`` asks whether this is a sentence or a paragraph,
#: and ``_MAX_CHARS`` asks whether what survived trimming is a claim or an essay. One ceiling
#: applied before trimming threw away every long Wikipedia lead — including "Anarchism is a
#: political philosophy and movement that is skeptical of all justifications for authority and
#: seeks to abolish the institutions it claims maintain unnecessary coercion and hierarchy,
#: typically including nation-states, and capitalism", whose first clause is the best sentence in
#: the article.
_MIN_CHARS, _MAX_CHARS, _MAX_RAW = 20, 200, 700


def sentences(text: Any, *, limit: int = 0) -> List[str]:
    """Split prose into sentences, with the markup that never belongs in a fact removed first."""
    body = _CITATION.sub(" ", str(text or ""))
    body = _PARENTHETICAL.sub(" ", body)
    body = " ".join(body.replace("\r", " ").replace("\n", " ").split())
    out = [s.strip() for s in _SENTENCE.split(body) if s.strip()]
    return out[:limit] if limit and limit > 0 else out


def trim_clause(sentence: Any) -> str:
    """Cut a sentence at its first subordinator, keeping the clause that makes the claim.

    Returned with a full stop, because the extractor's patterns are written against sentences and
    a bare clause is not one.
    """
    text = " ".join(str(sentence or "").split())
    if not text:
        return ""
    head = _SUBORDINATE.split(text, maxsplit=1)[0].strip()
    head = head.rstrip(" ,;:—–")
    if not head:
        return ""
    return head if head.endswith((".", "!", "?")) else head + "."


def definitional_sentences(text: Any, *, limit: int = 3,
                           allow_mathy: bool = False) -> List[str]:
    """The sentences of ``text`` that state something her extractor can hold.

    ``limit`` is per document and small on purpose. A Wikipedia article states its definition in
    its first sentence and then narrates for forty paragraphs; taking three is most of the yield
    for a fortieth of the rows, and the rows are what the wall-clock is made of.

    The order of the checks is load-bearing. Trimming happens **between** the two length ceilings
    so that a long sentence with a short claim survives, and the definitional shape is re-tested
    **after** trimming so that a cut which removed the verb removes the sentence too.
    """
    out: List[str] = []
    for raw in sentences(text):
        raw = _LEADING_JUNK.sub("", raw).strip()
        if len(raw) < _MIN_CHARS or len(raw) > _MAX_RAW:
            continue
        if not raw[:1].isalpha():
            continue
        if _PRONOUN_START.match(raw):
            continue
        if not allow_mathy and _MATHY.search(raw):
            continue
        if not _DEFINITIONAL.match(raw):
            continue
        trimmed = trim_clause(raw)
        if not (_MIN_CHARS <= len(trimmed) <= _MAX_CHARS):
            continue
        if not _DEFINITIONAL.match(trimmed):
            continue
        out.append(trimmed)
        if limit and len(out) >= limit:
            break
    return out


def _clean(value: Any) -> str:
    return " ".join(str(value or "").replace("\r", " ").replace("\n", " ").split())


def _meta(value: Any) -> Dict[str, Any]:
    """A row's JSON-in-a-string metadata column, or an empty mapping."""
    if isinstance(value, dict):
        return value
    try:
        got = json.loads(str(value or "{}"))
        return got if isinstance(got, dict) else {}
    except ValueError:
        return {}


# --------------------------------------------------------------------------- #
# The converters — one per source, each a stream in and a stream out
# --------------------------------------------------------------------------- #
# Every converter takes an *iterable of source rows* rather than one row, because two of them
# genuinely cannot work a row at a time: OASST1 arrives as flat messages that only become
# question/answer once a parent is matched to its best-ranked child, and any of them may want to
# deduplicate across the stream. A uniform signature is worth more than a per-row shortcut.


def _from_wikipedia(rows: Iterable[Dict[str, Any]]) -> Iterator[Dict[str, Any]]:
    """Articles → the definitional sentences of their leads.

    The title is not asserted as a subject in its own right. It is already the subject of the lead
    sentence — "Anarchism is a political philosophy" — and stating it twice would put a second
    unfolded copy of every article's ``is_a`` in the store under a slightly different surface form.
    """
    for row in rows:
        body = row.get("text")
        if not body:
            continue
        for line in definitional_sentences(body, limit=3):
            yield {"text": line, "source": _clean(row.get("url")) or "wikipedia"}


def _from_openwebmath(rows: Iterable[Dict[str, Any]]) -> Iterator[Dict[str, Any]]:
    """Math web pages → their definitional prose, with the LaTeX left behind.

    ``allow_mathy`` stays off here even though the corpus is mathematical, and that is the whole
    judgement in this converter. A sentence carrying ``$\\mathcal{F}$`` would be stored with that
    string as its object, and no question she can form contains it. What this corpus is worth to a
    fact store is its *prose* definitions — "a group is a set with an associative operation" — and
    those are written in words.
    """
    for row in rows:
        for line in definitional_sentences(row.get("text"), limit=4):
            yield {"text": line, "source": _clean(row.get("url")) or "openwebmath"}


def _from_proofnet(rows: Iterable[Dict[str, Any]]) -> Iterator[Dict[str, Any]]:
    """Theorem statements → question, natural-language proofs → answer.

    Pairs rather than triples, and the reason is the held-out set. A proof is the one thing in
    these ten sources that is *checkably* reasoning rather than recall, so it belongs on the path
    that can hold ten percent of it back and ask her for it later. As triples it would be a
    string in a store and nothing would ever be measured about it.

    Rows with no natural-language proof are dropped rather than paired with their Lean source. The
    formal statement is not an answer to the English question; it is the same claim in a language
    she does not read, and pairing them would teach the readout to emit Lean.
    """
    for row in rows:
        question = _clean(row.get("nl_statement"))
        answer = _clean(row.get("nl_proof"))
        answer = re.sub(r"\\(?:begin|end)\{proof\}", " ", answer).strip()
        if len(question) < 12 or len(answer) < 12:
            continue
        yield {"question": question, "answer": answer}


def _from_oasst1(rows: Iterable[Dict[str, Any]]) -> Iterator[Dict[str, Any]]:
    """Message trees → the prompter's turn and the best-ranked assistant reply to it.

    Flat messages in, conversation out. ``rank`` 0 is the reply the human annotators put first;
    taking every reply would give the same question several contradictory answers, and the fact
    store has no way to prefer one, so the disagreement would land as noise.

    Filtered to reviewed, undeleted, non-synthetic English and Hindi. ``lang`` is the source's own
    label rather than a guess, and both are kept because :mod:`nyxara.njp.tongue` is built for the
    two of them at once.
    """
    prompts: Dict[str, str] = {}
    best: Dict[str, Tuple[float, str]] = {}
    for row in rows:
        if row.get("deleted") or row.get("synthetic"):
            continue
        if row.get("review_result") is False:
            continue
        if str(row.get("lang") or "") not in ("en", "hi"):
            continue
        text = _clean(row.get("text"))
        if len(text) < 8:
            continue
        role = str(row.get("role") or "")
        if role == "prompter":
            prompts[str(row.get("message_id") or "")] = text
        elif role == "assistant":
            parent = str(row.get("parent_id") or "")
            if not parent:
                continue
            try:
                rank = float(row.get("rank") if row.get("rank") is not None else 0.0)
            except (TypeError, ValueError):
                rank = 0.0
            prior = best.get(parent)
            if prior is None or rank < prior[0]:
                best[parent] = (rank, text)
    for parent, (_, answer) in best.items():
        question = prompts.get(parent)
        if question:
            yield {"question": question, "answer": answer}


def _from_code(rows: Iterable[Dict[str, Any]]) -> Iterator[Dict[str, Any]]:
    """Functions → what they are for, as facts rather than as source.

    Triples rather than pairs, and the choice is about which of the two she can answer from. A
    docstring's first sentence *is* a purpose statement — "Estimate discontinuity in basis of low
    resolution image segmentation" — and ``purpose`` is the predicate
    ``grounding._GENERAL_ANSWER`` reads for "what is X used for". So the conversion that lands is
    the one that drops the body: a fact store cannot execute code, and storing it as an answer
    would only teach her to quote functions at people.

    ``part_of`` carries the highest transitivity prior in ``core._compose`` (0.70), so a function
    filed under its repository is a fact the reasoner can chain, not just retrieve.
    """
    for row in rows:
        name = _clean(row.get("func_name")).split(".")[-1]
        doc = _clean(row.get("func_documentation_string"))
        if not name or len(name) < 3 or len(doc) < 12:
            continue
        purpose = trim_clause(sentences(doc, limit=1)[0]) if sentences(doc, limit=1) else ""
        purpose = purpose.rstrip(".").strip()
        if len(purpose) < 8 or len(purpose) > 160:
            continue
        subject = name.replace("_", " ").strip()
        if not subject:
            continue
        yield {"subject": subject, "predicate": "purpose", "object": purpose,
               "confidence": 0.55}
        yield {"subject": subject, "predicate": "is_a", "object": "function",
               "confidence": 0.7}
        repo = _clean(row.get("repository_name"))
        if repo:
            yield {"subject": subject, "predicate": "part_of", "object": repo,
                   "confidence": 0.6}


def _from_arxiv(rows: Iterable[Dict[str, Any]]) -> Iterator[Dict[str, Any]]:
    """Abstracts → their definitional sentences.

    An abstract's first sentences are where a paper says what its object *is* before it says what
    it did with it, which is the half a fact store can hold. The title is not used as a subject:
    a paper title is a name, not a kind, and ``is_a`` between two paper titles would build a graph
    of bibliography rather than of the world.
    """
    for row in rows:
        for line in definitional_sentences(row.get("abstract"), limit=2):
            yield {"text": line, "source": "arxiv:" + (_clean(row.get("id")) or "?")}




#: GSM8K writes its arithmetic as ``<<48/2=24>>`` inline calculator annotations. They are a
#: training artefact of the dataset's own tooling, not part of the reasoning, and every one that
#: survives is punctuation the content-word scorer would count as a word.
_CALCULATOR = re.compile(r"<<[^>]{0,60}>>")

#: The final-answer marker. Kept as a sentence rather than stripped: "the answer is 72" is a claim,
#: where a bare "72" at the end of a paragraph is a token the exam's F1 cannot distinguish from a
#: quantity mentioned in passing.
_FINAL = re.compile(r"####\s*(.+)\s*$")


def _from_gsm8k(rows: Iterable[Dict[str, Any]]) -> Iterator[Dict[str, Any]]:
    """Word problems → question, worked solution → answer, with the calculator marks removed."""
    for row in rows:
        question = _clean(row.get("question") or row.get("problem"))
        raw = str(row.get("answer") or row.get("solution") or "")
        final = _FINAL.search(raw.strip())
        body = _clean(_CALCULATOR.sub("", _FINAL.sub("", raw)))
        if final:
            body = f"{body} The answer is {_clean(final.group(1))}.".strip()
        if len(question) < 12 or len(body) < 4:
            continue
        yield {"question": question, "answer": body}


def _from_math(rows: Iterable[Dict[str, Any]]) -> Iterator[Dict[str, Any]]:
    """Competition problems → question, solution → answer, with ``\\boxed{}`` restated in words.

    Split from :func:`_from_gsm8k` even though both emit pairs, because the two disagree about
    where the answer is: GSM8K marks it with ``####`` and MATH wraps it in ``\\boxed{}``. Reading
    one file with the other's rule silently yields pairs whose answer is the entire derivation and
    whose final number is nowhere the scorer can see it.
    """
    for row in rows:
        question = _clean(row.get("problem") or row.get("question"))
        raw = _clean(row.get("solution") or row.get("answer"))
        boxed = re.search(r"\\boxed\{([^{}]{1,80})\}", raw)
        if boxed:
            raw = f"{raw} The answer is {_clean(boxed.group(1))}."
        if len(question) < 12 or len(raw) < 4:
            continue
        yield {"question": question, "answer": raw}


#: Where a Project Gutenberg text stops being front matter and starts being the book, and where it
#: stops being the book. Both markers are part of the archive's own boilerplate, and the licence
#: text between them states hundreds of confident facts about the Project Gutenberg Foundation
#: that would otherwise become the most heavily corroborated knowledge she has.
_PG_START = re.compile(r"\*\*\*\s*START OF (?:THE|THIS) PROJECT GUTENBERG[^*]*\*\*\*", re.I)
_PG_END = re.compile(r"\*\*\*\s*END OF (?:THE|THIS) PROJECT GUTENBERG[^*]*\*\*\*", re.I)


def _from_gutenberg(rows: Iterable[Dict[str, Any]]) -> Iterator[Dict[str, Any]]:
    """Public-domain books → their definitional sentences, boilerplate stripped first.

    The per-document limit is higher than Wikipedia's because a book is not an article: it has no
    lead sentence, and what is worth having is scattered. It is still a limit, because a novel's
    hundredth "she was tired" is not knowledge and this corpus is mostly novels.
    """
    for row in rows:
        body = str(row.get("TEXT") or row.get("text") or "")
        start = _PG_START.search(body)
        if start:
            body = body[start.end():]
        end = _PG_END.search(body)
        if end:
            body = body[:end.start()]
        title = _clean(_meta(row.get("METADATA")).get("title")) or "gutenberg"
        for line in definitional_sentences(body, limit=12):
            yield {"text": line, "source": title[:80]}


#: How many content words a translation row may have before it stops being a term. A bilingual
#: *sentence* pair is not a synonym pair — asserting one would tell ``grounding._neighbours`` that
#: two whole sentences are the same node, and every word in either would inherit that.
_LEXICON_WORDS = 4


def _from_multilingual(rows: Iterable[Dict[str, Any]]) -> Iterator[Dict[str, Any]]:
    """Aligned en/hi text → a bilingual lexicon, and nothing longer.

    :mod:`nyxara.njp.tongue` already folds "गुरुत्वाकर्षण" and "gurutvakarshan" to one key and says
    in its own docstring that what it cannot do is learn that either of them is *gravity* — it
    ships no dictionary, by design. This is that dictionary, entered as facts rather than as a
    hard-coded table: ``also_known_as`` is what ``grounding._neighbours`` reads, so a question
    asked in one language reaches what she was told in the other.

    Only the short rows survive the filter, and most of a translation corpus is long rows. That is
    the correct yield: the corpus is a corpus of sentences and this wants the fraction of it that
    is a glossary.
    """
    for row in rows:
        pair = row.get("translation") if isinstance(row.get("translation"), dict) else row
        english = _clean(pair.get("en"))
        other = _clean(pair.get("hi") or pair.get("target"))
        if not english or not other:
            continue
        if english.endswith((".", "?", "!")) or other.endswith(("।", ".", "?", "!")):
            continue
        if len(english.split()) > _LEXICON_WORDS or len(other.split()) > _LEXICON_WORDS:
            continue
        if len(english) < 3 or len(other) < 2:
            continue
        yield {"subject": english, "predicate": "also_known_as", "object": other,
               "confidence": 0.7}


def _from_triples(rows: Iterable[Dict[str, Any]]) -> Iterator[Dict[str, Any]]:
    """Rows that are already triples, passed through.

    ConceptNet's converter lives in ``scripts/prepare_conceptnet.py`` because its dump is a
    five-column TSV rather than the row stream every other source arrives as, and because it was
    written and tested before this module existed. Re-implementing it here to make the table look
    uniform would give the project two ConceptNet relation maps to keep in agreement.
    """
    for row in rows:
        if row.get("subject") and row.get("predicate") and row.get("object"):
            yield dict(row)


# --------------------------------------------------------------------------- #
# The table
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Source:
    """One dataset, and the decision about what it becomes."""

    key: str
    priority: int
    tier: str
    title: str
    gets: str
    form: str
    dataset: str = ""
    config: str = ""
    split: str = "train"
    confidence: float = 0.6
    note: str = ""
    script: str = ""

    @property
    def filename(self) -> str:
        """What ``prepare_datasets.py`` writes and :mod:`nyxara.njp.train` looks for."""
        return f"{self.key}.{self.form}.jsonl.gz"

    def to_dict(self) -> Dict[str, Any]:
        return {"key": self.key, "priority": self.priority, "tier": self.tier,
                "title": self.title, "gets": self.gets, "form": self.form,
                "dataset": self.dataset, "config": self.config, "split": self.split,
                "confidence": self.confidence, "file": self.filename}


#: The Master's ten priorities, in his order, with the shape each one converts into.
#:
#: Eleven rows for ten priorities: #8 is "GSM8K + MATH", and they stay two rows because they are
#: two different measurements. Grade-school word problems and competition algebra fail in
#: different ways, and a single merged corpus could not say which of them moved a score.
#:
#: The order is not decoration and it is not merely his preference — it is the order the loaders
#: have to run in for the numbers between them to mean anything. Facts first, turns last: a corpus
#: of triples mechanically converts abstentions into answers, so running it *after* the conversational
#: sources would fold its effect into theirs and no exam between the two could separate them.
#: :mod:`nyxara.njp.train` exams between every stage on that basis.
SOURCES: Tuple[Source, ...] = (
    Source(key="conceptnet", priority=1, tier="red",
           title="ConceptNet 5.7",
           gets="concepts, relations, commonsense",
           form="triples",
           dataset="(offline dump)", config="", split="",
           confidence=0.6,
           note="prepared by scripts/prepare_conceptnet.py — a TSV dump, not a row stream"),
    Source(key="wikipedia", priority=2, tier="red",
           title="Wikipedia (English, 2023-11-01)",
           gets="broad world knowledge",
           form="text",
           dataset="wikimedia/wikipedia", config="20231101.en",
           note="lead-paragraph definitions only; the narration is dropped"),
    Source(key="openwebmath", priority=3, tier="red",
           title="OpenWebMath",
           gets="mathematical reasoning",
           form="text",
           dataset="open-web-math/open-web-math", config="default",
           note="prose definitions only; sentences carrying LaTeX are dropped"),
    Source(key="proofnet", priority=4, tier="red",
           title="ProofNet",
           gets="formal reasoning",
           form="pairs",
           dataset="hoskinson-center/proofnet", config="plain_text", split="validation",
           note="natural-language statement and proof; the Lean source is not an answer"),
    Source(key="oasst1", priority=5, tier="amber",
           title="OpenAssistant OASST1",
           gets="human-style conversation",
           form="pairs",
           dataset="OpenAssistant/oasst1", config="default",
           note="best-ranked reply per prompt, English and Hindi"),
    Source(key="code", priority=6, tier="amber",
           title="CodeSearchNet (Python)",
           gets="programming and algorithms",
           form="triples",
           dataset="code-search-net/code_search_net", config="python",
           confidence=0.55,
           note="docstring purpose, not source text — a fact store cannot execute code"),
    Source(key="arxiv", priority=7, tier="amber",
           title="arXiv abstracts",
           gets="scientific concepts and models",
           form="text",
           dataset="gfissore/arxiv-abstracts-2021", config="default",
           note="definitional sentences of abstracts; titles are names, not kinds"),
    Source(key="gsm8k", priority=8, tier="yellow",
           title="GSM8K",
           gets="arithmetic and problem solving",
           form="pairs",
           dataset="openai/gsm8k", config="main",
           note="calculator annotations stripped, final answer restated as a sentence"),
    Source(key="math", priority=8, tier="yellow",
           title="Hendrycks MATH",
           gets="arithmetic and problem solving",
           form="pairs",
           dataset="EleutherAI/hendrycks_math", config="algebra",
           note="shares priority 8 with GSM8K; \\boxed{} restated in words"),
    Source(key="gutenberg", priority=9, tier="yellow",
           title="Project Gutenberg (English)",
           gets="language and long-context reasoning",
           form="text",
           dataset="sedthh/gutenberg_english", config="default",
           note="archive boilerplate stripped before anything is read"),
    Source(key="multilingual", priority=10, tier="yellow",
           title="IIT Bombay English–Hindi",
           gets="Hindi/English and other languages",
           form="triples",
           dataset="cfilt/iitb-english-hindi", config="default",
           confidence=0.7, script="devanagari",
           note="short rows only — a sentence pair is not a synonym pair"),
)

#: Which converter reads which source. Separate from :data:`SOURCES` so that a source can be
#: renamed or re-pointed at a mirror without touching a function, and so a converter shared by two
#: sources is shared rather than copied.
_CONVERTERS: Dict[str, Callable[[Iterable[Dict[str, Any]]], Iterator[Dict[str, Any]]]] = {
    "conceptnet": _from_triples,
    "wikipedia": _from_wikipedia,
    "openwebmath": _from_openwebmath,
    "proofnet": _from_proofnet,
    "oasst1": _from_oasst1,
    "code": _from_code,
    "arxiv": _from_arxiv,
    "gsm8k": _from_gsm8k,
    "math": _from_math,
    "gutenberg": _from_gutenberg,
    "multilingual": _from_multilingual,
}


def by_key(key: str) -> Optional[Source]:
    """The source named ``key``, or ``None``."""
    wanted = str(key or "").strip().lower()
    for source in SOURCES:
        if source.key == wanted:
            return source
    return None


def in_priority_order(keys: Optional[Sequence[str]] = None) -> List[Source]:
    """The requested sources in the Master's order, or all of them.

    Sorted here rather than trusted from the caller: the whole attribution argument depends on
    facts landing before turns, and a caller that listed its sources alphabetically would get a
    run whose per-stage deltas silently mean something else.
    """
    if keys is None:
        chosen = list(SOURCES)
    else:
        wanted = {str(k).strip().lower() for k in keys}
        chosen = [s for s in SOURCES if s.key in wanted]
    return sorted(chosen, key=lambda s: (s.priority, s.key))


def converter_for(key: str) -> Optional[Callable[[Iterable[Dict[str, Any]]], Iterator[Dict[str, Any]]]]:
    """The converter for ``key``, or ``None`` if nothing here reads that source."""
    return _CONVERTERS.get(str(key or "").strip().lower())


def convert(key: str, rows: Iterable[Dict[str, Any]], *,
            limit: int = 0) -> Iterator[Dict[str, Any]]:
    """Source rows → NJP rows, in whichever of the three shapes that source converts into.

    Unknown keys yield nothing rather than raising: a preparation run over ten sources should
    report that one of them is unrecognised and finish the other nine, not die on the first.
    """
    fn = converter_for(key)
    if fn is None:
        return
    source = by_key(key)
    emitted = 0
    for row in fn(rows):
        if source is not None and source.form == "triples" and "confidence" not in row:
            row["confidence"] = source.confidence
        yield row
        emitted += 1
        if limit and emitted >= limit:
            return
