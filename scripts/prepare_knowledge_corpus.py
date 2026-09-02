#!/usr/bin/env python3
"""Turn the hand-written knowledge base in ``scripts/knowledge/*.kb`` into the two files she reads.

    scripts/knowledge/*.kb  ->  world_knowledge.jsonl.gz   (triples, for nyxara-study --ingest)
                            ->  world_qa.jsonl.gz          (question/answer, for Tutor.study)

Both outputs come from the *same* source rows, so the exam and the fact store cannot drift apart:
every question in the QA file is answerable from a triple in the triple file, by construction.

Why this exists at all
----------------------
:data:`nyxara.njp.study.DEFAULT_CORPUS` ships 35,693 pairs and its own docstring measures what
they are: **26.6% factual**, the rest generative ("Design a poster") or transformational ("Find
the volume of a cone"). Three-quarters of that corpus is unanswerable by a fact store *at any
quality of extraction*, so a coverage number over it has a denominator that is mostly unreachable.
This file produces the opposite kind of corpus — one where the answerable fraction is 100% by
construction — so that an exam score means "she did not know it" rather than "the question had no
fact of the matter".

And it is the other half of what ``prepare_conceptnet.py`` brings in. ConceptNet is broad, noisy
and crowd-sourced; this is narrow, curated and written down deliberately. Neither substitutes for
the other, in the same way ``growth/synth_data`` says verified synthesis cannot substitute for a
general corpus.

What it measures out at
-----------------------
Built from the shipped KB and loaded into a bare :class:`~nyxara.njp.brain.NJPBrain`, then asked
its own 2,930 questions back:

* **3,745 facts over 966 subjects**, ingested in ~160 ms. Nothing skipped, nothing capped.
* **2,432 answered** (83%), and every one of them is contained in the gold answer. **Zero wrong.**
* **498 abstentions**, all of them ``CONFLICTING`` — a relation with several objects, where
  ``Grounder.answer`` declines to pick one of two equally supported readings rather than
  answering arbitrarily. :mod:`nyxara.njp.study` counts that separately from being wrong, and it
  is the right outcome: the gold answer names all of them and she will not name one.
* **Zero UNKNOWN**, after a defect this corpus turned up. Twenty-four "when does X occur"
  questions came back UNKNOWN with the fact sitting in the store, because
  ``semantics.compile_meaning`` read a multi-word subject as a polar question
  (``polar(solar, eclipse, occur)``) and ``_answer_polar`` runs before the ordinary path. A
  wh-opener guard in :func:`nyxara.njp.grounding.Grounder._answer_polar` fixed it; one-word
  subjects had always worked, which is why nothing else had caught it.

Use both files together — the triples are the graph, the QA is the exam over it::

    nyxara-study --corpus nyxara/njp/data/world_qa.jsonl.gz \
                 --ingest nyxara/njp/data/world_knowledge.jsonl.gz

The KB format
-------------
One subject per line, pipe-separated ``predicate=object`` pairs after it, ``#`` starts a comment::

    Copper | is_a=metal | has_property=ductile | has_property=conductive | symbol=Cu
    photosynthesis | occurs_when=a plant is exposed to light | requires=sunlight@0.95

Repeat a predicate to give it several objects. ``@0.72`` after an object overrides the default
confidence for that one claim — for anything defeasible ("a bird can fly", and a penguin is still
a bird). Write subjects and objects in **natural case**: the triple side lowercases them the way
the Grounder does, and the prose side keeps "Paris" and "DNA" looking like themselves.

Three decisions below are worth reading before changing the tables.

**Every predicate here is one a question can reach, and that was measured, not assumed.** The
same failure ``prepare_conceptnet.py`` documents applies with more force here, because a QA
corpus whose questions do not parse teaches the exam to score zero for a reason that has nothing
to do with what she knows. So :data:`_ASKABLE` carries, per predicate, the question template
whose ``Grounder._read_question`` output is *exactly* ``(subject, predicate)`` — and
``tests/njp/test_knowledge_corpus.py`` re-measures every row of it against the live Grounder,
so a future addition cannot quietly emit a question that reads as something else.

**``known_for`` is deliberately absent, and it is the case that proves the point.** The pattern
table has a line for it — ``what is X known for`` — and it is dead: the ``purpose`` pattern
``what is X (used for|for)`` sits above it and matches the trailing " for" first. Measured,
``_read_question("what is marie curie known for?")`` returns ``('marie curie known', 'purpose')``:
wrong subject, wrong predicate. So facts about what someone is known for are written as ``is_a``
and ``discoverer``/``inventor``/``author`` instead, which do parse.

**``part_of`` and ``also_known_as`` are emitted as triples and carry no question.** ``part_of``
has the highest transitivity prior in ``njp/core.py`` (0.70) so the Core can chain it, and
``also_known_as`` feeds ``grounding._neighbours`` — both earn their place in the graph. Neither
has a question form that parses: "what is a wheel part of?" reads as ``('wheel part of', 'is_a')``.
Emitting a question anyway would put a guaranteed-wrong item in the exam. The split is the honest
one: the triple file is the whole graph, the QA file is the part of it English can ask for.

Standard library only, and nothing here opens a socket.

Examples
--------
    # both artefacts, into the package data directory she loads from
    python scripts/prepare_knowledge_corpus.py \\
        --triples nyxara/njp/data/world_knowledge.jsonl.gz \\
        --qa nyxara/njp/data/world_qa.jsonl.gz

    # validate the KB without writing anything (what CI wants)
    python scripts/prepare_knowledge_corpus.py --check

    # one domain, to plain .jsonl, to look at it
    python scripts/prepare_knowledge_corpus.py --domain chemistry --triples /dev/stdout

Then feed them to her:
    nyxara-study --ingest nyxara/njp/data/world_knowledge.jsonl.gz
"""
from __future__ import annotations

import argparse
import gzip
import io
import json
import os
import re
import sys
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

#: Where the hand-written sources live, relative to this file.
_KB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "knowledge")

#: Curated encyclopedic testimony, above ConceptNet's crowd-sourced ceiling of 0.9 nowhere, and
#: above ``study._SEED_CONFIDENCE`` (0.8) because these are claims about named things rather than
#: defeasible generalisations over a whole kind. Anything that *is* such a generalisation carries
#: its own ``@`` override in the KB and lands below this.
_DEFAULT_CONFIDENCE = 0.85

#: predicate -> (question template, which end of the triple the question is about)
#:
#: Every template's ``Grounder._read_question`` output was measured to be exactly
#: ``(<the named end>, <this predicate>)``. ``"o"`` means the question names the object and the
#: answer is found by scanning ``causes`` edges backwards — ``grounding._lookup_inverse``, reached
#: through the reserved ``_CAUSE_OF`` marker. Order here is the order questions are emitted in.
_ASKABLE: Dict[str, Tuple[str, str]] = {
    "is_a":         ("What is {s}?", "s"),
    "means":        ("What does {s} mean?", "s"),
    "has_kind":     ("What are the types of {s}?", "s"),
    "has_part":     ("What are the parts of {s}?", "s"),
    "consists_of":  ("What does {s} consist of?", "s"),
    "involves":     ("What does {s} involve?", "s"),
    "occurs_when":  ("When does {s} occur?", "s"),
    "purpose":      ("What is {s} used for?", "s"),
    "capable_of":   ("What can {s} do?", "s"),
    "has_property": ("What are the properties of {s}?", "s"),
    "causes":       ("What does {s} cause?", "s"),
    "requires":     ("What does {s} require?", "s"),
    "located_in":   ("Where is {s}?", "s"),
    # The noun-form relations. All of them reach the store through one pattern —
    # ``tell me the (?P<p>\w+) of (?P<s>.+)`` — which is why each name is a single word that
    # reads as English in that slot. ``discovered_by`` would match the regex just as well and
    # would put "tell me the discovered_by of penicillin" in a training corpus; ``discoverer``
    # is the same edge, askable in a sentence someone might actually write.
    "capital":      ("Tell me the capital of {s}.", "s"),
    "symbol":       ("Tell me the symbol of {s}.", "s"),
    "formula":      ("Tell me the formula of {s}.", "s"),
    "unit":         ("Tell me the unit of {s}.", "s"),
    "currency":     ("Tell me the currency of {s}.", "s"),
    "discoverer":   ("Tell me the discoverer of {s}.", "s"),
    "inventor":     ("Tell me the inventor of {s}.", "s"),
    "author":       ("Tell me the author of {s}.", "s"),
    "birthplace":   ("Tell me the birthplace of {s}.", "s"),
}

#: Emitted into the graph, never into the exam. See the module docstring for why each is here and
#: why neither gets a question.
_GRAPH_ONLY = ("part_of", "also_known_as")

#: Relations that hold exactly one value, where a second is a contradiction rather than an
#: addition. ``grounding._FUNCTIONAL`` names the first two; the rest are functional by meaning —
#: a country has one capital, an element one symbol — and ``_revise`` would fire on a second.
#:
#: ``prepare_conceptnet.py`` responds to this by dropping ``/r/AtLocation`` entirely, and that is
#: the right call *there*: ConceptNet lists dozens of locations per concept and the converter
#: cannot tell which one is meant. Here the KB is written by hand and :func:`parse` **rejects the
#: file** if any subject carries two of these, so the hazard is checked rather than avoided, and
#: geography keeps its capitals.
_FUNCTIONAL = frozenset({
    "located_in", "birthplace", "capital", "symbol", "formula", "unit", "currency",
    "discoverer", "inventor", "author",
})

#: How each relation reads back as a sentence. ``{s}`` is the subject in its KB casing, ``{o}``
#: the objects joined by :func:`_join`. The answer is prose because
#: :meth:`nyxara.njp.study.Tutor.study` grounds the *answer* to get the fact into the store —
#: a bare object would give the extractor nothing to parse.
_ANSWER: Dict[str, str] = {
    "is_a":         "{S} is {a}.",
    "means":        "{S} means {o}.",
    "has_kind":     "The main types of {s} are {o}.",
    "has_part":     "The parts of {s} are {o}.",
    "consists_of":  "{S} consists of {o}.",
    "involves":     "{S} involves {o}.",
    "occurs_when":  "{S} occurs when {o}.",
    "purpose":      "{S} is used for {o}.",
    "capable_of":   "{S} can {o}.",
    "has_property": "{S} is {o}.",
    "causes":       "{S} causes {o}.",
    "requires":     "{S} requires {o}.",
    "located_in":   "{S} is in {o}.",
    "capital":      "The capital of {s} is {o}.",
    "symbol":       "The symbol of {s} is {o}.",
    "formula":      "The formula of {s} is {o}.",
    "unit":         "The unit of {s} is {o}.",
    "currency":     "The currency of {s} is {o}.",
    "discoverer":   "{S} was discovered by {o}.",
    "inventor":     "{S} was invented by {o}.",
    "author":       "{S} was written by {o}.",
    "birthplace":   "{S} was born in {o}.",
}

#: Uncountable and already-determined objects, where ``is_a=X`` must not become "a X". The list is
#: short on purpose: it holds only what the shipped KB actually uses, and :func:`parse` warns on a
#: bare mass noun it has not been told about rather than silently writing "a water".
_MASS = frozenset({
    "water", "energy", "matter", "light", "heat", "sound", "electricity", "air", "rock", "sand",
    "wood", "metal", "gas", "money", "information", "data", "software", "hardware", "music",
    "mathematics", "physics", "chemistry", "biology", "history", "geography", "medicine", "art",
    "knowledge", "work", "power", "pressure", "mass", "time", "space", "blood", "oxygen",
    "hydrogen", "carbon", "nitrogen", "iron", "gold", "silver", "copper", "salt", "sugar",
    "protein", "starch", "glucose", "steel", "glass", "plastic", "paper", "cotton", "silk",
    "weather", "climate", "rain", "snow", "ice", "steam", "fire", "soil", "grass", "food",
    "fuel", "radiation", "gravity", "friction", "electricity", "language", "grammar", "code",
})

#: Written with a vowel, pronounced with a consonant: "a unit", "a European state". Getting this
#: wrong is not cosmetic — "an unit of force" would be written into every training document that
#: mentions a unit, and a 300M model learns the spelling it is shown.
_CONSONANT_SOUNDED = ("uni", "use", "usu", "usa", "uti", "ubi", "eu", "one", "ewe")

#: The mirror case: written with a consonant, pronounced with a vowel.
_VOWEL_SOUNDED = ("hour", "honest", "honour", "honor", "heir")

#: Singular nouns that end in an s. Without them the plural test below reads "noble gas" as a
#: plural and writes "helium is noble gas".
_SINGULAR_IN_S = frozenset({
    "gas", "bus", "lens", "species", "series", "news", "atlas", "canvas", "iris", "plus",
    "apparatus", "census", "chaos", "corps", "lens", "means", "kudos",
})

_LINE = re.compile(r"^(?P<subject>[^|#]+?)\s*\|\s*(?P<rest>.+)$")
_PAIR = re.compile(r"^(?P<predicate>[a-z_]+)\s*=\s*(?P<object>[^@]+?)(?:@(?P<conf>[01](?:\.\d+)?))?$")


class KBError(ValueError):
    """A source file that would produce a fact nothing can read, with the line that did it."""


# --------------------------------------------------------------------------- #
# Reading the KB
# --------------------------------------------------------------------------- #
def _norm(text: str) -> str:
    """The form the fact store holds: lowercase, single-spaced. What the Grounder would produce."""
    return " ".join(str(text).split()).lower()


def domains(kb_dir: str = _KB_DIR) -> List[str]:
    """The domain names available, from the filenames. Sorted, so output is deterministic."""
    if not os.path.isdir(kb_dir):
        return []
    return sorted(name[:-3] for name in os.listdir(kb_dir) if name.endswith(".kb"))


def parse_file(path: str, *, domain: str = "",
               warnings: Optional[List[str]] = None) -> Iterator[Dict[str, Any]]:
    """Yield one row per ``predicate=object`` pair in one ``.kb`` file.

    Rows keep both surfaces: ``subject``/``object`` are normalised for the store, ``subject_text``
    and ``object_text`` keep the KB's own casing for the prose side.
    """
    known = set(_ASKABLE) | set(_GRAPH_ONLY)
    with open(path, "rt", encoding="utf-8") as handle:
        for number, raw in enumerate(handle, start=1):
            line = raw.split("#", 1)[0].strip()
            if not line:
                continue
            match = _LINE.match(line)
            if match is None:
                raise KBError(f"{path}:{number}: no '|' after the subject: {line!r}")
            subject_text = " ".join(match.group("subject").split())
            for chunk in match.group("rest").split("|"):
                chunk = chunk.strip()
                if not chunk:
                    continue
                pair = _PAIR.match(chunk)
                if pair is None:
                    raise KBError(f"{path}:{number}: not 'predicate=object': {chunk!r}")
                predicate = pair.group("predicate")
                if predicate not in known:
                    raise KBError(f"{path}:{number}: unreachable predicate {predicate!r}. "
                                  f"Reachable: {', '.join(sorted(known))}")
                object_text = " ".join(pair.group("object").split())
                subject, obj = _norm(subject_text), _norm(object_text)
                if not subject or not obj:
                    raise KBError(f"{path}:{number}: empty subject or object: {chunk!r}")
                if subject == obj:
                    raise KBError(f"{path}:{number}: self-edge {subject!r} {predicate}")
                if warnings is not None and predicate == "is_a" and _article(object_text) == "":
                    head = obj.split()[0]
                    if (obj not in _MASS and not obj.endswith("s")
                            and not object_text[:1].isupper()
                            and head not in {"a", "an", "the", "one", "any"}
                            and not head.startswith(_CONSONANT_SOUNDED)):
                        warnings.append(f"{path}:{number}: '{obj}' takes no article — "
                                        f"add it to _MASS if that is right")
                yield {"domain": domain or os.path.basename(path)[:-3],
                       "subject": subject, "predicate": predicate, "object": obj,
                       "subject_text": subject_text, "object_text": object_text,
                       "confidence": float(pair.group("conf")) if pair.group("conf") else None,
                       "line": f"{os.path.basename(path)}:{number}"}


def parse(kb_dir: str = _KB_DIR, *, wanted: Optional[Sequence[str]] = None,
          warnings: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    """Every row of every requested domain, deduplicated, with the two invariants enforced.

    The invariants are the whole reason this is a function rather than a generator chain:

    * **A functional relation holds one value.** Two capitals for one country is not extra
      knowledge, it is a contradiction ``_revise`` will fire on, and it is a typo the author can
      fix in a second and never sees if this stays quiet.
    * **A triple appears once.** The ingest path dedups anyway; catching it here means the count
      this prints is the count she loads.
    """
    names = list(wanted) if wanted else domains(kb_dir)
    if not names:
        raise KBError(f"no .kb files in {kb_dir}")
    rows: List[Dict[str, Any]] = []
    seen: Dict[Tuple[str, str, str], str] = {}
    functional: Dict[Tuple[str, str], Tuple[str, str]] = {}
    for name in names:
        path = os.path.join(kb_dir, f"{name}.kb")
        if not os.path.exists(path):
            raise KBError(f"no such domain: {name} (have: {', '.join(domains(kb_dir))})")
        for row in parse_file(path, domain=name, warnings=warnings):
            key = (row["subject"], row["predicate"], row["object"])
            if key in seen:
                raise KBError(f"{row['line']}: duplicate of {seen[key]}: {' '.join(key)}")
            seen[key] = row["line"]
            if row["predicate"] in _FUNCTIONAL:
                fkey = (row["subject"], row["predicate"])
                if fkey in functional:
                    held, where = functional[fkey]
                    raise KBError(f"{row['line']}: {row['subject']!r} already has "
                                  f"{row['predicate']}={held!r} at {where}. That relation holds "
                                  f"one value; a second is a contradiction, not an addition.")
                functional[fkey] = (row["object"], row["line"])
            rows.append(row)
    return rows


# --------------------------------------------------------------------------- #
# The two outputs
# --------------------------------------------------------------------------- #
def triples(rows: Iterable[Dict[str, Any]], *,
            confidence: float = _DEFAULT_CONFIDENCE) -> Iterator[Dict[str, Any]]:
    """Ingest-shaped rows: exactly the four keys ``njp.ingest.ingest_triples`` reads."""
    for row in rows:
        yield {"subject": row["subject"], "predicate": row["predicate"], "object": row["object"],
               "confidence": round(float(row["confidence"] if row["confidence"] is not None
                                         else confidence), 4)}


def _head_noun(phrase: str) -> str:
    """The word that decides whether the phrase is plural.

    English noun phrases are head-final up to the first preposition: the head of "gas giant" is
    "giant" and the head of "unit of force" is "unit". Taking the *first* word for this — which is
    what the first version did — read "gas giant" as plural, because "gas" ends in an s, and
    emitted "Jupiter is gas giant".
    """
    chunk = re.split(r"\s+(?:of|in|for|from|with|to|between)\s+", phrase.strip(), maxsplit=1)[0]
    words = chunk.split()
    return words[-1].lower() if words else ""


def _article(text: str) -> str:
    """"a", "an", or "" for something that takes no article. Applied to ``is_a`` objects only."""
    word = text.strip()
    if not word:
        return ""
    head = word.split()[0].lower()
    if head in {"a", "an", "the", "one", "any"}:
        return ""
    if word[:1].isupper():            # a proper noun is already determined
        return ""
    # Whole-phrase, not head-word: a bare mass noun takes no article ("ice is water"), a modified
    # one used as a classifier does ("sodium is an alkali metal").
    if _norm(word) in _MASS:
        return ""
    noun = _head_noun(word)
    if (noun.endswith("s") and not noun.endswith(("ss", "us", "is"))
            and noun not in _SINGULAR_IN_S):
        return ""                     # already plural
    # Sound, not spelling: "a unit", "an hour". The two lists hold only the prefixes the shipped
    # KB actually reaches, and they are prefixes rather than whole words so that "university"
    # and "unit" need one entry between them.
    if head.startswith(_CONSONANT_SOUNDED):
        return "a"
    if head.startswith(_VOWEL_SOUNDED):
        return "an"
    return "an" if head[0] in "aeiou" else "a"


def _join(parts: Sequence[str], *, conjunction: str = "and") -> str:
    """Oxford-free list: "a", "a and b", "a, b and c"."""
    items = list(parts)
    if len(items) <= 1:
        return items[0] if items else ""
    return f"{', '.join(items[:-1])} {conjunction} {items[-1]}"


def _capitalise(sentence: str) -> str:
    """Uppercase the first letter and nothing else.

    Only the first character, deliberately: the KB writes objects in their own casing, and a
    ``.capitalize()`` here would turn "DNA is a nucleic acid" into "Dna is a nucleic acid" and put
    that spelling into the training corpus.
    """
    for index, character in enumerate(sentence):
        if character.isalpha():
            return sentence[:index] + character.upper() + sentence[index + 1:]
    return sentence


def _sentence(predicate: str, subject_text: str, objects: Sequence[str]) -> str:
    """One prose answer for one subject and one relation, over all of its objects."""
    template = _ANSWER[predicate]
    if predicate == "is_a":
        rendered = _join([f"{_article(o)} {o}".strip() for o in objects])
        return _capitalise(template.format(S=subject_text, s=subject_text,
                                           a=rendered, o=rendered))
    joined = _join(list(objects))
    return _capitalise(template.format(S=subject_text, s=subject_text, o=joined, a=joined))


def unaskable_subjects(rows: Iterable[Dict[str, Any]]) -> List[str]:
    """Subjects whose *name* the question reader cannot give back, whatever the relation.

    :data:`_ASKABLE` guarantees each template reads back as its own predicate, and that is only
    half of the pair. The other half is the subject, and a subject can be lost by its own spelling:
    ``Grounder._read_question`` strips a leading article, so ``What is a priori?`` comes back as
    ``('priori', 'is_a')`` and there is no fact under that key. ``The Odyssey`` failed the same way
    — the first two subjects in this KB that began with an article, and both arrived in one change
    set, which is how long the trap sat unsprung.

    Renaming is the right fix where a name survives it (``The Odyssey`` is fine as ``Odyssey``) and
    it is not available for a term whose article is part of it: *a priori* is not "priori". So
    those subjects keep their triples — the graph is the whole of what she knows — and carry no
    question, exactly as ``_GRAPH_ONLY`` predicates do, for exactly the same reason: a question
    guaranteed to be unanswerable is a guaranteed-wrong item in the exam, and the score would be
    measuring the spelling.

    The list is *reported* rather than silently applied. ``--check`` prints it, so a subject that
    drops out of the exam does so visibly.
    """
    from nyxara.njp.grounding import Grounder, _clean       # noqa: PLC0415 — optional at import

    grounder = Grounder()
    # The reader consults its own store to tell "the capital of France" (a relation about a
    # subject) from "the Age of Exploration" (a subject whose name contains " of "), so a check
    # run against an *empty* Grounder is stricter than the thing it is checking: it reported
    # `age of exploration` and `unit of measurement` unaskable while the live brain reads both
    # correctly. Seeding the subject keys — the only thing `_known_entity` looks at — makes this
    # measure the reader that actually runs.
    for subject in {row["subject"] for row in rows}:
        grounder.facts.setdefault((grounder._key(subject), "is_a"), [])
    lost: List[str] = []
    for subject in sorted({row["subject"] for row in rows}):
        text = next(r["subject_text"] for r in rows if r["subject"] == subject)
        read, predicate = grounder._read_question(_clean(f"What is {text}?").lower())
        if predicate != "is_a" or read != subject:
            lost.append(subject)
    return lost


def qa_pairs(rows: Iterable[Dict[str, Any]]) -> Iterator[Dict[str, str]]:
    """One question per (subject, relation), with every object of that relation in the answer.

    Per *relation*, not per triple, and that is the point: asking "what are the properties of
    copper?" three times with a third of the answer each time would teach the exam that a complete
    answer is wrong twice. The Grounder extracts each object back out of the list on the way in.

    Subjects the reader cannot give back are skipped — see :func:`unaskable_subjects`.
    """
    rows = list(rows)
    lost = set(unaskable_subjects(rows))
    grouped: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for row in rows:
        if row["predicate"] not in _ASKABLE or row["subject"] in lost:
            continue
        key = (row["subject"], row["predicate"])
        bucket = grouped.setdefault(key, {"subject_text": row["subject_text"], "objects": []})
        bucket["objects"].append(row["object_text"])
    for (subject, predicate), bucket in grouped.items():
        template, _end = _ASKABLE[predicate]
        yield {"question": template.format(s=bucket["subject_text"]),
               "answer": _sentence(predicate, bucket["subject_text"], bucket["objects"])}


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _write(rows: Iterable[Dict[str, Any]], path: str) -> int:
    """Write JSONL, gzipped if the name says so. Returns the number of lines."""
    written = 0
    if path == "/dev/stdout" or path == "-":
        for row in rows:
            sys.stdout.write(json.dumps(row, ensure_ascii=False) + "\n")
            written += 1
        return written
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    # ``gzip.GzipFile`` rather than ``gzip.open`` for one reason: ``mtime=0``. Without it the
    # current clock goes into the gzip header, so rebuilding an unchanged KB produces a different
    # file, and every run of this script shows up as a diff in a repository that stores the
    # artefact. ``gzip.open`` has no way to pass it.
    if path.endswith(".gz"):
        handle: Any = io.TextIOWrapper(
            gzip.GzipFile(path, "wb", compresslevel=9, mtime=0), encoding="utf-8")
    else:
        handle = open(path, "wt", encoding="utf-8")
    with handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            written += 1
    return written


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kb", default=_KB_DIR, help=f"knowledge base directory (default: {_KB_DIR})")
    ap.add_argument("--domain", action="append", default=None, metavar="NAME",
                    help="restrict to this domain (repeatable; default: all)")
    ap.add_argument("--triples", metavar="PATH", help="write ingest triples here (.jsonl/.jsonl.gz)")
    ap.add_argument("--qa", metavar="PATH", help="write question/answer pairs here")
    ap.add_argument("--confidence", type=float, default=_DEFAULT_CONFIDENCE,
                    help=f"confidence for claims with no @override (default: {_DEFAULT_CONFIDENCE})")
    ap.add_argument("--check", action="store_true",
                    help="parse and report, write nothing (exit 1 on any warning)")
    args = ap.parse_args(argv)

    warnings: List[str] = []
    try:
        rows = parse(args.kb, wanted=args.domain, warnings=warnings)
    except KBError as exc:
        sys.stderr.write(f"{exc}\n")
        return 2

    by_domain: Dict[str, int] = {}
    by_predicate: Dict[str, int] = {}
    for row in rows:
        by_domain[row["domain"]] = by_domain.get(row["domain"], 0) + 1
        by_predicate[row["predicate"]] = by_predicate.get(row["predicate"], 0) + 1

    pairs = list(qa_pairs(rows))
    report: Dict[str, Any] = {
        "facts": len(rows),
        "subjects": len({row["subject"] for row in rows}),
        "qa_pairs": len(pairs),
        # Named, not merely subtracted. A subject that leaves the exam because English cannot
        # give its name back should be visible in the report that says how big the exam is.
        "unaskable_subjects": unaskable_subjects(rows),
        "domains": dict(sorted(by_domain.items())),
        "predicates": dict(sorted(by_predicate.items(), key=lambda kv: (-kv[1], kv[0]))),
    }

    if not args.check:
        if args.triples:
            report["triples_written"] = _write(triples(rows, confidence=args.confidence),
                                               args.triples)
        if args.qa:
            report["qa_written"] = _write(pairs, args.qa)
        if not args.triples and not args.qa:
            sys.stderr.write("nothing to do: pass --triples PATH and/or --qa PATH, or --check\n")
            return 2

    print(json.dumps(report, indent=1, ensure_ascii=False))
    for warning in warnings:
        sys.stderr.write(f"warning: {warning}\n")
    if args.triples:
        print(f"\nuse it:  nyxara-study --ingest {args.triples}", file=sys.stderr)
    return 1 if (warnings and args.check) else 0


if __name__ == "__main__":
    raise SystemExit(main())
