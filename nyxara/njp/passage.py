"""NYXARA · njp/passage.py — a passage is not a triple, and reading one is not parsing a sentence (📖).

Every fact this package holds arrived as ``subject | relation=object``. The 62 files under
``scripts/knowledge/`` were typed that way by hand; ConceptNet arrived that way already. Neither
is reading. Put a real paragraph in front of her and measure what survives — this is
:mod:`nyxara.njp.grounding`, today, on five sentences of ordinary prose:

    "Photosynthesis is the process by which plants convert light energy into chemical energy."
        -> photosynthesis is_a process
        LOST: occurs_in=plants, uses=light energy, produces=chemical energy

    "It reduces the purchasing power of money."
        -> **It** decreases purchasing power of money

    "Photosynthesis occurs in the chloroplasts of plant cells and requires chlorophyll, water
     and carbon dioxide."
        -> subject = "photosynthesis occurs in the chloroplasts of plant cells and"
           object  = "chlorophyll, water and carbon dioxide"
        LOST: occurs_in entirely; three requirements fused into one

Three separate failures, and they have one cause: **the grounder looks for a subject inside each
sentence.** A relative clause hides its relations behind ``by which``; a second sentence names its
subject with a pronoun; a coordination puts two predicates in one sentence and the subject-finder
swallows the first predicate whole. A reader that hunts a subject per sentence cannot win any of
the three, and no amount of extra corpus fixes it, because the corpus is written in the form the
reader already handles.

A passage is *about* something. That is the architectural difference and the whole of this module:
the concept comes from the discourse, not from each clause, so a clause only has to supply a
relation and a filler. The coordination problem then disappears rather than being solved --
``and requires chlorophyll`` is a perfectly good clause once nobody demands a subject from it.

**Nothing here is an extraction table.** The shapes that read a clause are induced from
demonstrations, at two levels of abstraction, on the discipline :mod:`nyxara.njp.asking` set:

* a **frame** keeps the demonstration's own words as anchors, so it reads sentences written like
  the one it was shown;
* a **cued** shape replaces every open-class anchor with a hole and keeps only the closed class --
  ``into <SLOT>`` survives from *"converts light into chemical energy"* and reads *"transforms
  nitrogen into ammonia"*, a sentence with no content word in common with it.

A cued shape with no closed-class anchor left is refused rather than kept, because a pattern of
three wildcards matches every clause in the language and would read anything as anything. And a
cued shape is not trusted until **two different demonstrations** have produced it: one
demonstration showing ``into`` means ``produces`` is a coincidence, two are a shape.
:attr:`Shape.generalises` records the thing that actually matters -- whether it has since read a
passage no demonstration contained.

What the reader returns is a :class:`KnowledgeObject`, and its fields are the layers the plan
names: the concept, its definition, the kind it falls under, the entities the passage put on the
table, the relations with the condition each was stated under, **and the passage itself**. Prose
is not thrown away once it has been mined; a definition read out of a clause and the clause it
came from are different things and she keeps both. Every relation carries the sentence and the
shape that read it, so a claim can always say what produced it -- the discipline
:mod:`nyxara.njp.provenance` set, applied at the point of entry rather than after it.

Nothing here writes to the fact store. Reading a passage produces a candidate structure; what
happens to it is :mod:`nyxara.njp.immune`'s decision and the caller's, and a module that both
extracted and asserted would make that decision unreachable.

Pure standard library.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

__all__ = [
    "Relation", "KnowledgeObject", "Demonstration", "Shape", "PassageReader",
    "LESSONS", "taught_reader",
]

#: How many tokens of context on each side of a filler become the shape's anchors. Three on the
#: left because ``by which plants convert`` needs three to say which slot it is; two on the right
#: because the token after a filler is nearly always the preposition that ends it.
LEFT_CONTEXT = 3
RIGHT_CONTEXT = 2

#: The longest filler a shape will accept, before any demonstration has been seen. Without a cap
#: the "to the end of the clause" case reads half a paragraph into one object. It is a floor, not
#: a constant: :meth:`PassageReader.teach` raises it to the longest filler a teacher has actually
#: pointed at, because a definition is longer than a noun and a reader that capped both at the
#: same length would read every definition truncated. Left at 8 this module silently returned no
#: definition at all for its own first lesson.
MAX_FILLER = 8

#: How many distinct demonstrations must produce a cued shape before it is used, *when the shape
#: kept no relation word*. One is a coincidence; a shape anchored on the word that names one
#: relation and nothing else is not.
#:
#: A stated trade rather than a tuned constant, and the school prints both sides of it: at 1 the
#: reader recalls 0.915 of the held-out relations at 0.818 precision, at 2 it recalls 0.780 at
#: 0.979. The default prefers precision, on the standing rule that she may be silent but must not
#: be confidently wrong.
MIN_WITNESSES = 2

_TOKEN_RX = re.compile(r"[A-Za-z0-9][A-Za-z0-9'’\-]*|[,;]")
_SENTENCE_RX = re.compile(r"(?<=[.!?])\s+")

_PUNCT = "PUNCT"
_WORD = "WORD"
_HOLE = "*"


def _table() -> Tuple[Dict[str, str], Any]:
    """The closed class, from :mod:`nyxara.njp.semantics` — the half of language that anchors."""
    try:
        from nyxara.njp.semantics import _CLOSED, Tag  # noqa: WPS433
        return _CLOSED, Tag
    except Exception:  # noqa: BLE001 — a reader with no closed class tags everything open
        class _Tag:  # pragma: no cover - only reached if semantics is unimportable
            WORD = "WORD"
            DET = "DET"
            PREP = "PREP"
            CONJ = "CONJ"
            SUB = "SUB"
            AUX = "AUX"
            PRON = "PRON"
            WH = "WH"
            MODAL = "MODAL"
            NEG = "NEG"
        return {}, _Tag


@dataclass(frozen=True)
class Tok:
    """One token and its closed-class tag."""

    text: str = ""
    tag: str = _WORD

    @property
    def open_class(self) -> bool:
        return self.tag == _WORD


def _toks(text: str) -> List[Tok]:
    closed, _ = _table()
    out: List[Tok] = []
    for match in _TOKEN_RX.finditer(str(text or "").lower()):
        word = match.group(0)
        out.append(Tok(word, _PUNCT if word in (",", ";") else closed.get(word, _WORD)))
    return out


def _bare(toks: Sequence["Tok"]) -> List["Tok"]:
    """The token stream a pattern matches against, with determiners removed.

    ``a`` and ``the`` are closed class, so an anchor built literally keeps whichever one the
    demonstration happened to use — and every lesson here happens to use ``the``, which made
    ``Inflation is **a** general rise in prices`` unreadable by a shape induced from
    ``Photosynthesis is **the** process ...``. A determiner is the one closed class that carries
    no relation and varies freely with its noun, so it is removed from both sides rather than
    matched. Fillers are built from this stream too, which is why a read object is ``mitochondria``
    and not ``the mitochondria``.
    """
    _, Tag = _table()
    return [t for t in toks if t.tag != Tag.DET]


def _sentences(text: str) -> List[str]:
    return [s.strip() for s in _SENTENCE_RX.split(str(text or "").strip()) if s.strip()]


def _words(toks: Sequence[Tok]) -> List[str]:
    return [t.text for t in toks]


def _find(haystack: Sequence[str], needle: Sequence[str]) -> Optional[Tuple[int, int]]:
    """The first index range where ``needle`` appears in ``haystack``."""
    n = len(needle)
    if not n or n > len(haystack):
        return None
    for i in range(len(haystack) - n + 1):
        if list(haystack[i:i + n]) == list(needle):
            return i, i + n
    return None


# --------------------------------------------------------------------------------------------- #
#  what a reading is made of
# --------------------------------------------------------------------------------------------- #
@dataclass
class Relation:
    """One relation read out of a passage, with the sentence and the shape that produced it."""

    predicate: str = ""
    object: str = ""
    subject: str = ""
    #: The clause this was stated under, when the sentence stated one — ``"when stress is
    #: released"``. A relation with a condition is not the same claim as one without, and the plan
    #: is explicit that a conditional fact stored as a fact is the error to avoid.
    condition: str = ""
    sentence: str = ""
    shape: str = ""
    confidence: float = 0.0

    @property
    def key(self) -> Tuple[str, str, str]:
        return (self.subject, self.predicate, self.object)

    def to_dict(self) -> Dict[str, Any]:
        out = {"subject": self.subject, "predicate": self.predicate, "object": self.object,
               "confidence": round(self.confidence, 4), "shape": self.shape}
        if self.condition:
            out["condition"] = self.condition
        if self.sentence:
            out["sentence"] = self.sentence
        return out


@dataclass
class KnowledgeObject:
    """A passage read into layers, with the passage kept.

    ``concept`` / ``definition`` / ``kind`` are three different things and collapsing them is the
    defect this separates: today *"inflation is a general rise in prices"* is stored as
    ``is_a=general rise in prices``, so the question *"what is inflation"* and the question
    *"inflation is a kind of what"* land in the same slot and neither is answered well. Here the
    definition is the gloss, the kind is its structural head (``rise``), and both are kept.
    """

    concept: str = ""
    definition: str = ""
    kind: str = ""
    entities: Tuple[str, ...] = ()
    relations: Tuple[Relation, ...] = ()
    #: The passage, verbatim. Reading it does not consume it.
    text: str = ""
    sentences: Tuple[str, ...] = ()
    source: str = ""
    domain: str = ""
    confidence: float = 0.0
    #: ``sentence -> the shapes that read it``. Empty for a sentence nothing read, which is how
    #: the school finds what the induced grammar still cannot see.
    provenance: Dict[str, Tuple[str, ...]] = field(default_factory=dict)
    #: Pronoun subjects that could not be settled, with the reason. Not silently dropped.
    unresolved: Tuple[str, ...] = ()

    @property
    def conditions(self) -> Tuple[str, ...]:
        return tuple(sorted({r.condition for r in self.relations if r.condition}))

    def triples(self) -> Tuple[Tuple[str, str, str], ...]:
        """The flat form, for the fact store — everything this module exists to say is more."""
        return tuple(r.key for r in self.relations)

    def to_dict(self) -> Dict[str, Any]:
        return {"concept": self.concept, "definition": self.definition, "kind": self.kind,
                "entities": list(self.entities),
                "relations": [r.to_dict() for r in self.relations],
                "conditions": list(self.conditions),
                "source": self.source, "domain": self.domain,
                "confidence": round(self.confidence, 4),
                "sentences": list(self.sentences),
                "unresolved": list(self.unresolved),
                "text": self.text}


@dataclass
class Demonstration:
    """A passage and what a teacher says is in it. The only input the reader learns from."""

    name: str = ""
    text: str = ""
    concept: str = ""
    #: ``predicate -> objects``. ``definition`` is a predicate like any other here; the reader has
    #: no privileged copular rule, and the school's ``definition`` subject is a real measurement
    #: because of that.
    expect: Dict[str, Tuple[str, ...]] = field(default_factory=dict)
    #: ``sentence index -> the name a pronoun there settles on``, and ``""`` where the teacher
    #: says it settles on nothing. Both kinds are needed and for the reason
    #: :meth:`~nyxara.njp.discourse.Reference.fit` gives: on demonstrations that all resolve, a
    #: setting that never abstains fits perfectly and nothing rules it out.
    pronouns: Dict[int, str] = field(default_factory=dict)

    def pairs(self) -> List[Tuple[str, str]]:
        out: List[Tuple[str, str]] = []
        for predicate, objects in (self.expect or {}).items():
            for obj in objects:
                out.append((str(predicate), str(obj)))
        return out


@dataclass
class Shape:
    """A clause-reading pattern: anchors, a hole, and the relation the hole fills.

    ``left`` and ``right`` are token patterns. An element is either a literal word or
    :data:`_HOLE`, which matches exactly one open-class token — that is what makes a cued shape
    read a verb it has never seen while still requiring the preposition that carries the meaning.
    """

    predicate: str = ""
    left: Tuple[str, ...] = ()
    right: Tuple[str, ...] = ()
    level: str = "frame"
    witnesses: Set[str] = field(default_factory=set)
    #: The pattern still contains an open-class word that anchors **this predicate and no other**
    #: anywhere in the demonstrations. Set by :meth:`PassageReader._consolidate`, never by hand.
    relation_word: bool = False
    #: Passages this shape has read that were not the ones it was induced from.
    unseen: int = 0

    @property
    def key(self) -> Tuple[str, Tuple[str, ...], Tuple[str, ...], str]:
        return (self.predicate, self.left, self.right, self.level)

    @property
    def anchors(self) -> int:
        """Literal words in the pattern. A cued shape with none of these is refused."""
        return sum(1 for w in self.left + self.right if w != _HOLE)

    @property
    def trusted(self) -> bool:
        return (self.level == "frame" or self.relation_word
                or len(self.witnesses) >= MIN_WITNESSES)

    @property
    def generalises(self) -> bool:
        """It has read something nobody demonstrated. The only evidence that it is a shape."""
        return self.unseen > 0

    def render(self) -> str:
        return f"{' '.join(self.left)} <{self.predicate}> {' '.join(self.right)}".strip()

    def to_dict(self) -> Dict[str, Any]:
        return {"predicate": self.predicate, "pattern": self.render(), "level": self.level,
                "witnesses": sorted(self.witnesses), "unseen": self.unseen,
                "relation_word": self.relation_word,
                "trusted": self.trusted, "generalises": self.generalises}


# --------------------------------------------------------------------------------------------- #
#  the reader
# --------------------------------------------------------------------------------------------- #
class PassageReader:
    """Induces clause shapes from demonstrations, then reads passages with them.

    Three mechanisms, and the school removes each one separately to show what it was worth:

    * **the topic supplies the subject.** A clause only has to give a relation and a filler.
    * **a coordinated filler is a list**, when a demonstration has shown one. Until then
      ``chlorophyll, water and carbon dioxide`` stays one object, which is the honest reading for
      a reader that has never been told otherwise.
    * **a pronoun subject is resolved against the passage**, through
      :class:`~nyxara.njp.discourse.Reference`, which already does this on evidence and abstains
      where the evidence is thin. Nothing here re-implements it.
    """

    def __init__(self, *, min_witnesses: int = MIN_WITNESSES,
                 expand_lists: bool = True, resolve_pronouns: bool = True,
                 topic_subject: bool = True, use_cued: bool = True) -> None:
        self.shapes: Dict[Tuple[str, Tuple[str, ...], Tuple[str, ...], str], Shape] = {}
        self.min_witnesses = int(min_witnesses)
        #: Controls, for the falsification runs. Each defaults to on; the school turns one off at
        #: a time and reports the drop.
        self.expand_lists = bool(expand_lists)
        self.resolve_pronouns = bool(resolve_pronouns)
        self.topic_subject = bool(topic_subject)
        #: Off, the reader keeps only the literal frames — the run that says what abstraction was
        #: worth. Setting ``min_witnesses`` high does not do this, because a shape that kept a
        #: relation word is trusted on that evidence and not on its witness count.
        self.use_cued = bool(use_cued)
        #: Set by a demonstration that actually showed a coordinated filler expanding into
        #: several objects. Not a constant: a reader taught only from single-object passages
        #: never learns to split a list, and the school measures that.
        self._saw_list = False
        #: The longest filler any demonstration pointed at, in tokens. Learned, not set.
        self._max_filler = MAX_FILLER
        self._taught: Set[str] = set()
        self._demo_texts: Set[str] = set()
        #: Every word a teacher pointed at as an object, and every concept a lesson was about.
        #: Both are entities, and an entity that happens to sit next to a hole is not the word
        #: that names the relation. Learned from the lessons; nothing here is a stop-list.
        self._filler_words: Set[str] = set()
        self._concept_words: Set[str] = set()
        #: ``word -> the predicate it names``, filled by :meth:`_consolidate`. Read at match time
        #: as well as at induction time: a filler that contains the word naming *another*
        #: relation is not a filler, it is the next clause. Without this,
        #: ``requires an expanding money supply and produces a fall in real wages`` came back as
        #: ``requires = produces fall in real wages``.
        self._relation_words: Dict[str, str] = {}
        #: Reference cases the lessons themselves demonstrate, and what fitting them found.
        #: Constructing a bare :class:`~nyxara.njp.discourse.Reference` was a real defect and the
        #: school caught it by the only means that could: the ``no_pronouns`` ablation changed
        #: **no score at all**, because an unfitted resolver scores role parallelism at zero and
        #: abstained on every passage, leaving the topic fallback to do all the work and take all
        #: the credit. The cues are not tuned here — they are fitted to resolutions a teacher
        #: demonstrated, exactly as :mod:`nyxara.njp.discourseschool` fits them to its own.
        self._cases: List[Tuple[Any, str, str, str, str]] = []
        self._cues: Dict[str, float] = {}
        self._margin: Optional[float] = None

    # -- learning ------------------------------------------------------------ #
    def teach(self, demo: Demonstration) -> List[Shape]:
        """Induce every shape this demonstration supports. Returns the shapes it added or fed."""
        added: List[Shape] = []
        try:
            name = demo.name or demo.text[:32]
            self._taught.add(name)
            self._demo_texts.add(_norm(demo.text))
            for _predicate, obj in demo.pairs():
                self._filler_words.update(_words(_bare(_toks(obj))))
            self._concept_words.update(_words(_bare(_toks(demo.concept))))
            for sentence in _sentences(demo.text):
                toks = _bare(_toks(sentence))
                words = _words(toks)
                for predicate, obj in demo.pairs():
                    span = _find(words, _words(_bare(_toks(obj))))
                    if span is None:
                        continue
                    i, j = span
                    self._max_filler = max(self._max_filler, j - i)
                    for shape in self._induce(predicate, toks, i, j):
                        added.append(self._file(shape, name))
                self._learn_lists(demo, toks, words)
            self._reference_cases(demo)
            self._consolidate()
            self._fit_reference()
        except Exception:  # noqa: BLE001 — a demonstration that cannot be read teaches nothing
            return added
        return added

    def _consolidate(self) -> None:
        """Rebuild every cued shape from the frames, once the whole lesson set is in view.

        A cued shape cannot be induced from one demonstration in isolation, and the first version
        of this module tried: it replaced **every** open-class anchor with a hole, which turned
        ``and requires <SLOT>`` into ``<*> and <*> <SLOT>`` — a pattern that reads any coordinated
        clause as any relation, and did. The word that carries the relation has to survive.

        Which word that is, is not a judgement call and not a list: an open-class anchor that
        appears in the frames of **exactly one** predicate across every demonstration is naming
        that predicate, and one that appears under several is doing something else. ``requires``,
        ``occurs`` and ``produces`` come out as relation words; ``convert`` does not, because the
        lessons use it for ``uses`` and for ``occurs_in`` both. Exclusivity rather than frequency
        — the same test :mod:`nyxara.njp.provenance` ranks blame by, for the same reason: the
        anchor that only ever appears with one relation is the one that identifies it.
        """
        frames = [s for s in self.shapes.values() if s.level == "frame"]
        anchors: Dict[str, Dict[str, Set[str]]] = {}
        for frame in frames:
            for word in frame.left + frame.right:
                if not _open_class(word):
                    continue
                if word in self._filler_words or word in self._concept_words:
                    continue                  # an entity, not a relation word
                anchors.setdefault(word, {}).setdefault(frame.predicate, set())
                anchors[word][frame.predicate] |= frame.witnesses
        #: ``word -> the one predicate it anchors in more lessons than any other``. A word that
        #: anchors two relations in the same number of lessons names neither and is holed.
        dominant: Dict[str, str] = {}
        for word, spread in anchors.items():
            ranked = sorted(spread.items(), key=lambda kv: (-len(kv[1]), kv[0]))
            if len(ranked) == 1 or len(ranked[0][1]) > len(ranked[1][1]):
                dominant[word] = ranked[0][0]
        rebuilt: Dict[Tuple[str, Tuple[str, ...], Tuple[str, ...], str], Shape] = {
            f.key: f for f in frames}
        for frame in frames:
            def hole(word: str, predicate: str = frame.predicate) -> str:
                # A pronoun in an anchor is holed like a content word. It stands in for one, the
                # matching side already accepts a pronoun in a hole, and without the symmetry the
                # lesson "**It** produces sediment" abstracted to nothing at all: the cued shape
                # came out identical to its frame, was dropped as adding nothing, and every
                # ``produces`` list in the exam stopped splitting.
                if not (_open_class(word) or _pronoun_word(word)):
                    return word
                return word if dominant.get(word) == predicate else _HOLE
            left = tuple(hole(w) for w in frame.left)
            right = tuple(hole(w) for w in frame.right)
            if (left, right) == (frame.left, frame.right):
                continue                      # nothing was abstracted; the frame already is it
            cued = Shape(predicate=frame.predicate, left=left, right=right, level="cued")
            if not cued.anchors:
                continue                      # all holes — matches every clause, refused
            cued.relation_word = any(dominant.get(w) == frame.predicate for w in left + right)
            held = rebuilt.get(cued.key)
            if held is None:
                rebuilt[cued.key] = cued
                held = cued
            held.witnesses |= frame.witnesses
            held.relation_word = held.relation_word or cued.relation_word
        self._relation_words = dict(dominant)
        for key, shape in rebuilt.items():
            was = self.shapes.get(key)
            if was is not None and was.level == "cued":
                shape.unseen = max(shape.unseen, was.unseen)
        self.shapes = rebuilt

    def _induce(self, predicate: str, toks: Sequence[Tok], i: int, j: int) -> List[Shape]:
        """One filler span becomes a literal frame and, when it can, a cued generalisation."""
        n = len(toks)
        lo, hi = max(0, i - LEFT_CONTEXT), min(n, j + RIGHT_CONTEXT)
        left = tuple(toks[k].text for k in range(lo, i))
        right = tuple(toks[k].text for k in range(j, hi))
        out: List[Shape] = []
        if left or right:
            out.append(Shape(predicate=predicate, left=left, right=right, level="frame"))
        cued_left = tuple(_HOLE if toks[k].open_class else toks[k].text for k in range(lo, i))
        cued_right = tuple(_HOLE if toks[k].open_class else toks[k].text for k in range(j, hi))
        cued = Shape(predicate=predicate, left=cued_left, right=cued_right, level="cued")
        # A pattern of nothing but holes matches every clause in the language. Refusing it is the
        # difference between a generalisation and a machine that reads anything as anything.
        if cued.anchors:
            out.append(cued)
        return out

    def _file(self, shape: Shape, witness: str) -> Shape:
        held = self.shapes.get(shape.key)
        if held is None:
            shape.witnesses.add(witness)
            self.shapes[shape.key] = shape
            return shape
        held.witnesses.add(witness)
        return held

    def _reference_cases(self, demo: Demonstration) -> None:
        """Turn ``pronouns`` into the case shape :meth:`Reference.fit` reads."""
        if not demo.pronouns:
            return
        try:
            from nyxara.njp.discourse import Referent  # noqa: WPS433
        except Exception:  # noqa: BLE001
            return
        sentences = _sentences(demo.text)
        fillers = [o for _p, o in demo.pairs()]
        for index, expected in demo.pronouns.items():
            if index <= 0 or index >= len(sentences):
                continue
            turn = index - 1
            # The determiner-free form on both sides. Comparing a stripped name against the raw
            # sentence silently found nothing: "loss of water from leaf" is not a substring of
            # "the loss of water from a leaf", so two of the three demonstrations were dropped
            # and the fit ran on one case.
            said = _key_name(sentences[turn])
            referents = [Referent(name=_key_name(demo.concept), turn=turn, role="subject")]
            for filler in fillers:
                name = _key_name(filler)
                if name and name in said:
                    referents.append(Referent(name=name, turn=turn, role="object"))
            pronoun = _words(_bare(_toks(sentences[index])))[:1]
            if not pronoun or len(referents) < 2:
                continue
            self._cases.append((referents, pronoun[0], "subject", "",
                                _key_name(expected)))

    def _fit_reference(self) -> None:
        if not self._cases:
            return
        try:
            from nyxara.njp.discourse import Reference  # noqa: WPS433
            scratch = Reference()
            fitted = scratch.fit(self._cases)
            self._cues = dict(fitted.get("cues") or scratch.cues)
            self._margin = float(fitted.get("margin", scratch.margin))
        except Exception:  # noqa: BLE001
            return

    def _learn_lists(self, demo: Demonstration, toks: Sequence[Tok],
                     words: Sequence[str]) -> None:
        """Did this demonstration show one coordinated span standing for several objects?"""
        _, Tag = _table()
        for _predicate, objects in (demo.expect or {}).items():
            if len(objects) < 2:
                continue
            spans = [_find(words, _words(_toks(o))) for o in objects]
            if any(s is None for s in spans):
                continue
            lo = min(s[0] for s in spans if s)
            hi = max(s[1] for s in spans if s)
            joined = [toks[k] for k in range(lo, hi)]
            if any(t.tag in (_PUNCT, Tag.CONJ) for t in joined):
                self._saw_list = True
                return

    def taught(self) -> Tuple[str, ...]:
        return tuple(sorted(self._taught))

    def usable(self) -> List[Shape]:
        """Every frame, plus a cued shape that either kept a relation word or has two witnesses.

        The witness count alone was the first gate and it was too strict to be useful: it threw
        away ``<*> occurs in <SLOT> and requires``, which only one lesson happened to produce and
        which reads a chloroplast sentence perfectly. A shape anchored on a word that names one
        relation and nothing else is not a coincidence — the exclusivity across the whole lesson
        set is the second piece of evidence. The school runs both gates and reports both.
        """
        return [s for s in self.shapes.values()
                if s.level == "frame"
                or (self.use_cued and (s.relation_word
                                       or len(s.witnesses) >= self.min_witnesses))]

    def generalised(self) -> List[Shape]:
        return [s for s in self.shapes.values() if s.generalises]

    # -- reading ------------------------------------------------------------- #
    def read(self, text: str, *, concept: str = "", source: str = "", domain: str = "",
             confidence: float = 0.7) -> KnowledgeObject:
        """Turn a passage into layers. Never writes anything anywhere."""
        obj = KnowledgeObject(text=str(text or ""), source=str(source), domain=str(domain),
                              confidence=float(confidence))
        try:
            sentences = _sentences(text)
            obj.sentences = tuple(sentences)
            if not sentences:
                return obj
            reference, entities = self._discourse(), []
            topic = str(concept or "").strip()
            relations: List[Relation] = []
            unresolved: List[str] = []
            seen: Set[Tuple[str, str, str]] = set()
            unseen_passage = _norm(text) not in self._demo_texts
            for turn, sentence in enumerate(sentences):
                toks = _bare(_toks(sentence))
                if not toks:
                    continue
                condition, body = self._condition(toks)
                subject, note = self._subject(body, topic, reference, turn)
                if not topic and subject:
                    topic = subject
                if note:
                    unresolved.append(note)
                if not subject and self.topic_subject:
                    subject = topic
                read_by: List[str] = []
                for predicate, filler, shape in self._scan(body):
                    for one in self._split(filler):
                        relation = Relation(predicate=predicate, object=one, subject=subject,
                                            condition=condition, sentence=sentence,
                                            shape=shape.render(),
                                            confidence=self._worth(shape, confidence))
                        if relation.key in seen or not one:
                            continue
                        seen.add(relation.key)
                        relations.append(relation)
                        read_by.append(shape.render())
                        entities.append(one)
                        if unseen_passage:
                            shape.unseen += 1
                obj.provenance[sentence] = tuple(dict.fromkeys(read_by))
                self._introduce(reference, subject,
                                [r.object for r in relations if r.sentence == sentence], turn)
                if subject:
                    entities.append(subject)
            obj.concept = topic
            obj.relations = tuple(relations)
            obj.unresolved = tuple(unresolved)
            obj.definition = self._definition(relations)
            obj.kind = self._head(obj.definition) or self._kind(relations)
            obj.entities = tuple(dict.fromkeys(e for e in entities if e))
        except Exception:  # noqa: BLE001 — an unreadable passage is read as far as it got
            return obj
        return obj

    # -- the pieces reading is made of --------------------------------------- #
    def _scan(self, toks: Sequence[Tok]) -> List[Tuple[str, str, Shape]]:
        """Every filler every usable shape finds in this clause, shortest filler per shape."""
        out: List[Tuple[str, str, Shape]] = []
        for shape in self.usable():
            for filler in self._match(shape, toks):
                out.append((shape.predicate, filler, shape))
        return out

    def _match(self, shape: Shape, toks: Sequence[Tok]) -> List[str]:
        n = len(toks)
        found: List[str] = []
        width = len(shape.left)
        for i in range(0, n):
            if not self._pattern(shape.left, toks, i - width):
                continue
            # The cap exists to stop an unbounded rightward scan. Where the shape has no right
            # anchor the scan is already bounded by the end of the sentence, so applying the cap
            # there only truncates long definitions — which is exactly what it did.
            reach = n if not shape.right else min(n, i + self._max_filler)
            for j in range(i + 1, reach + 1):
                if shape.right:
                    if not self._pattern(shape.right, toks, j):
                        continue
                elif j != n and toks[j].tag != _PUNCT:
                    continue
                filler = self._clean(toks[i:j], shape.predicate)
                if filler:
                    found.append(filler)
                break
        return found

    @staticmethod
    def _pattern(pattern: Sequence[str], toks: Sequence[Tok], at: int) -> bool:
        """A hole stands for one nominal: a content word, or a pronoun standing in for one.

        Requiring the hole to be open-class alone was a silent recall hole. Every lesson that
        produced ``<*> requires <SLOT>`` had a noun in the hole, so ``Osmosis ... **It** requires
        a partially permeable membrane`` matched nothing at all — a pronoun is closed class, and
        the shape that was supposed to read the sentence refused it on a technicality about what
        a subject is spelled like.
        """
        _, Tag = _table()
        if not pattern:
            return True
        if at < 0 or at + len(pattern) > len(toks):
            return False
        for offset, want in enumerate(pattern):
            token = toks[at + offset]
            if want == _HOLE:
                if not (token.open_class or token.tag == Tag.PRON):
                    return False
            elif token.text != want:
                return False
        return True

    def _clean(self, toks: Sequence[Tok], predicate: str = "") -> str:
        """A filler is a noun phrase, and two things end one: a verb, and another relation.

        ``a temperature to be meaningful`` is a noun phrase followed by a clause, and reading the
        whole span as the thing required produced a claim the passage does not make. A noun phrase
        ends where a verb begins, so the span is cut at the first auxiliary rather than thrown
        away — which turns that confabulation into the right answer instead of into a silence.

        The second cut is what the lessons taught: a word this reader has induced as the name of
        some *other* relation cannot be inside this relation's object. ``requires an expanding
        money supply and produces a fall in real wages`` stops at ``produces``.
        """
        _, Tag = _table()
        span = list(toks)
        for index, token in enumerate(span):
            if token.tag in (Tag.AUX, Tag.MODAL):
                span = span[:index]
                break
            named = self._relation_words.get(token.text)
            if named and named != predicate and index > 0:
                span = span[:index]
                break
        while span and span[0].tag in (Tag.DET,):
            span.pop(0)
        while span and (span[-1].tag in (_PUNCT, Tag.CONJ, Tag.DET, Tag.PREP)):
            span.pop()
        if not span or not span[-1].open_class:
            return ""
        if all(not t.open_class for t in span):
            return ""
        # A filler that begins on an auxiliary or a subordinator is the rest of a clause, not a
        # thing. Reading it as an object is how "occurs in ... and" became a subject.
        if span[0].tag in (Tag.AUX, Tag.SUB, Tag.MODAL, Tag.WH):
            return ""
        return " ".join(t.text for t in span)

    def _split(self, filler: str) -> List[str]:
        """``chlorophyll, water and carbon dioxide`` -> three, once a demonstration showed one.

        A coordination is not always a list. ``the process by which current splits water into
        hydrogen and oxygen`` is one definition containing an ``and``, and splitting it produced
        two definitions, neither of them true. What separates the two cases is structural and
        needs no per-predicate exception: **every** piece of a list is a bare noun phrase, and a
        piece carrying a preposition, a relative or a subordinator is a clause, so the span is a
        clause and stays whole.
        """
        if not self.expand_lists or not self._saw_list:
            return [filler]
        _, Tag = _table()
        toks = _bare(_toks(filler))
        if not any(t.tag in (_PUNCT, Tag.CONJ) for t in toks):
            return [filler]
        if any(t.tag in (Tag.PREP, Tag.WH, Tag.SUB, Tag.AUX, Tag.MODAL) for t in toks):
            return [filler]
        items, current = [], []
        for token in toks:
            if token.tag in (_PUNCT, Tag.CONJ):
                if current:
                    items.append(" ".join(t.text for t in current))
                current = []
                continue
            current.append(token)
        if current:
            items.append(" ".join(t.text for t in current))
        items = [i for i in items if i]
        return items or [filler]

    def _condition(self, toks: Sequence[Tok]) -> Tuple[str, List[Tok]]:
        """Split off a ``when`` / ``if`` clause. What it states is a condition, not a fact.

        Only when the subordinator does not open the sentence: *"An earthquake occurs when X"*
        has ``occurs when`` as the relation itself, and stripping X would throw the claim away.
        """
        _, Tag = _table()
        for index, token in enumerate(toks):
            if token.tag == Tag.SUB and token.text in ("if", "unless") and index > 0:
                return " ".join(t.text for t in toks[index + 1:]), list(toks[:index])
        return "", list(toks)

    def _subject(self, toks: Sequence[Tok], topic: str, reference: Any,
                 turn: int) -> Tuple[str, str]:
        """The concept this clause is about: the topic, unless the clause names someone else."""
        _, Tag = _table()
        if not toks:
            return (topic if self.topic_subject else ""), ""
        first = toks[0]
        if first.tag == Tag.PRON:
            if not self.resolve_pronouns or reference is None:
                return "", f"{first.text}: pronoun resolution off"
            try:
                resolution = reference.resolve(first.text, role="subject", turn=turn)
            except Exception:  # noqa: BLE001
                return topic, f"{first.text}: resolver unavailable"
            if resolution.settled and resolution.referent:
                return str(resolution.referent), ""
            return (topic if self.topic_subject else ""), \
                f"{first.text}: {resolution.why or 'ambiguous'}"
        if not self.topic_subject:
            return "", ""
        stop = len(toks)
        for index, token in enumerate(toks):
            if token.tag in (Tag.AUX, Tag.MODAL):
                stop = index
                break
        head = [t for t in toks[:stop] if t.tag not in (Tag.DET,)]
        named = " ".join(t.text for t in head).strip()
        # A clause whose own subject the topic already names is about the topic; anything else
        # names itself. Both are structural, neither consults a list of entities.
        if topic and (not named or named in topic or topic in named):
            return topic, ""
        if named and len(head) <= 4 and any(t.open_class for t in head):
            return named, ""
        return (topic if self.topic_subject else ""), ""

    def _discourse(self) -> Any:
        if not self.resolve_pronouns:
            return None
        try:
            from nyxara.njp.discourse import Reference  # noqa: WPS433
            reference = Reference(margin=self._margin) if self._margin is not None else Reference()
            if self._cues:
                reference.cues = dict(self._cues)
            return reference
        except Exception:  # noqa: BLE001
            return None

    @staticmethod
    def _introduce(reference: Any, subject: str, objects: Sequence[str], turn: int) -> None:
        if reference is None:
            return
        try:
            from types import SimpleNamespace
            roles = {f"filler{n}": name for n, name in enumerate(objects[:-1])}
            meaning = SimpleNamespace(subject=subject, object=(objects[-1] if objects else ""),
                                      roles=roles)
            reference.introduce(meaning, turn=turn)
            reference.turn = turn + 1
        except Exception:  # noqa: BLE001
            return

    @staticmethod
    def _worth(shape: Shape, base: float) -> float:
        """A cued shape read on an anchor is worth less than a frame that matched literally."""
        return round(base * (1.0 if shape.level == "frame" else 0.9), 4)

    @staticmethod
    def _definition(relations: Sequence[Relation]) -> str:
        for relation in relations:
            if relation.predicate == "definition":
                return relation.object
        return ""

    @staticmethod
    def _kind(relations: Sequence[Relation]) -> str:
        for relation in relations:
            if relation.predicate in ("is_a", "kind"):
                return relation.object
        return ""

    @staticmethod
    def _head(phrase: str) -> str:
        """The head noun of a gloss: the last content word before its first modifier opens.

        ``a general rise in prices`` -> ``rise``. Structural, and it is what separates the kind
        from the definition without a list of nouns that are allowed to be kinds.
        """
        _, Tag = _table()
        toks = _toks(phrase)
        head: List[str] = []
        for token in toks:
            if token.tag in (Tag.PREP, Tag.WH, Tag.SUB, Tag.CONJ, _PUNCT):
                break
            if token.tag == Tag.DET:
                continue
            head.append(token.text)
        return head[-1] if head else ""

    # -- reporting ----------------------------------------------------------- #
    def stats(self) -> Dict[str, Any]:
        usable = self.usable()
        return {"shapes": len(self.shapes), "usable": len(usable),
                "max_filler": self._max_filler,
                "reference_cases": len(self._cases), "cues": self._cues,
                "margin": self._margin,
                "frames": sum(1 for s in usable if s.level == "frame"),
                "cued": sum(1 for s in usable if s.level == "cued"),
                "generalised": len(self.generalised()),
                "demonstrations": len(self._taught),
                "lists": self._saw_list}


def _key_name(text: str) -> str:
    """The form :class:`~nyxara.njp.discourse.Referent` stores a name in."""
    return " ".join(_words(_bare(_toks(text))))


def _pronoun_word(word: str) -> bool:
    closed, Tag = _table()
    return closed.get(word) == Tag.PRON


def _open_class(word: str) -> bool:
    closed, _ = _table()
    return word not in closed and word != _HOLE


def _norm(text: str) -> str:
    return " ".join(str(text or "").lower().split())


# --------------------------------------------------------------------------------------------- #
#  the lessons
# --------------------------------------------------------------------------------------------- #
#: What a teacher shows her. Six passages, from two subjects, and that is deliberately narrow:
#: the school's held-out passages are from subjects that appear nowhere here, so a shape that
#: reads them read something nobody demonstrated.
LESSONS: Tuple[Demonstration, ...] = (
    Demonstration(
        name="photosynthesis",
        concept="photosynthesis",
        text=("Photosynthesis is the process by which plants convert light energy into "
              "chemical energy."),
        expect={"definition": ("process by which plants convert light energy into chemical "
                              "energy",),
                "occurs_in": ("plants",),
                "uses": ("light energy",),
                "produces": ("chemical energy",)},
    ),
    Demonstration(
        name="respiration",
        concept="respiration",
        text=("Respiration is the reaction by which cells release energy from glucose. "
              "Respiration occurs in the mitochondria and requires oxygen."),
        expect={"definition": ("reaction by which cells release energy from glucose",),
                "occurs_in": ("cells", "the mitochondria"),
                "produces": ("energy",),
                "requires": ("oxygen",)},
    ),
    Demonstration(
        name="transpiration",
        concept="transpiration",
        text=("Transpiration is the loss of water from a leaf. It requires stomata, warmth "
              "and moving air."),
        expect={"definition": ("loss of water from a leaf",),
                "requires": ("stomata", "warmth", "moving air")},
        pronouns={1: "transpiration"},
    ),
    Demonstration(
        name="erosion",
        concept="erosion",
        text=("Erosion is the process by which wind removes soil from a surface. "
              "It produces sediment."),
        expect={"definition": ("process by which wind removes soil from a surface",),
                "occurs_in": ("wind",),
                "uses": ("soil",),
                "produces": ("sediment",)},
        pronouns={1: "erosion"},
    ),
    Demonstration(
        name="condensation",
        concept="condensation",
        text=("Condensation is the change by which vapour becomes water. "
              "Condensation requires cooling."),
        expect={"definition": ("change by which vapour becomes water",),
                "occurs_in": ("vapour",),
                "requires": ("cooling",)},
    ),
    Demonstration(
        name="weathering",
        concept="weathering",
        text=("Weathering is the breakdown of rock in place. "
              "Weathering requires water, temperature change and time."),
        expect={"definition": ("breakdown of rock in place",),
                "requires": ("water", "temperature change", "time")},
    ),
)


#: The lesson that teaches the resolver when to say nothing. Its second sentence could be about
#: either thing the first put on the table, and the teacher says so — a demonstration of an
#: abstention is as much a demonstration as one of an answer, and without it the fit would find
#: the weights that never abstain and call them the best explanation of the data.
_AMBIGUOUS: Demonstration = Demonstration(
    name="ambiguity",
    concept="silting",
    text=("Silting is the settling of sand behind a barrage. It requires slow water."),
    expect={"definition": ("settling of sand behind a barrage",)},
    pronouns={1: ""},
)

LESSONS = LESSONS + (_AMBIGUOUS,)


def taught_reader(**kwargs: Any) -> PassageReader:
    """A reader that has seen :data:`LESSONS` and nothing else."""
    reader = PassageReader(**kwargs)
    for lesson in LESSONS:
        reader.teach(lesson)
    return reader
