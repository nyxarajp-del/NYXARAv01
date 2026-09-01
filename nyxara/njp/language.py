"""NYXARA · njp/language.py — the grammar she learns, not the one she shipped (🗣, NJP V.18).

:mod:`nyxara.njp.tongue` tokenises any script. :mod:`nyxara.njp.semantics` tags the closed class
and matches **frames over the tag sequence**, which is what made an open verb list unnecessary.
:mod:`nyxara.njp.grounding` turns what came back into beliefs. Three organs, and between them one
thing that does not exist: a way for her to **acquire a construction she was not shipped with**.

The frames in ``semantics.py`` are written by a person — ``_q_fronted_wh``, ``_q_polar``,
``_q_hinglish`` and six more. They are good frames, and they are the whole of the grammar. A
sentence whose shape is not one of them is ``unreadable``, and it stays unreadable however many
times it is said to her, because nothing in the package turns *a sentence she was shown the
meaning of* into *a shape she can read the next sentence with*. Measured, that is not a subtle
gap: over a minted language whose words come in the order object–subject–verb, the shipped
compiler reads **0 of 24** held-out sentences, and would still read 0 after a thousand lessons.

This module is the missing half, and it is deliberately the same discipline
:class:`~nyxara.njp.coding.Coder` applies to a demonstrated program:

    a lesson leaves a **shape**, never an answer — and a shape is only kept once it has read a
    sentence nobody demonstrated.

**Four things are learned here, and they are kept apart because they are four claims.**

* **Affixes are induced.** :class:`Morphology` finds them from a vocabulary in which many stems
  appear both bare and extended. No suffix list ships. A suffix carried by three stems whose bare
  forms are also words is evidence; one seen once is a coincidence and is not kept.
* **An affix *means* something only where a lesson said so.** Induction finds the shape ``-ik``;
  it cannot find that ``-ik`` is a plural. :meth:`Morphology.bind` takes a demonstrated pair and
  binds the feature, and it **refuses to bind an affix that was never independently induced** —
  otherwise the wug test is one memorised pair with a rule's name on it.
* **Word classes are discovered from where words occur**, by :class:`Lexicon`, and are given no
  names. A cluster found by distribution has no name, and inventing one ("noun") would be the
  ontology this package refuses everywhere else, sneaking back in through the grammar.
* **Constructions are generalised from demonstrations**, by :class:`Grammar`. A demonstration is
  a surface and the :class:`~nyxara.njp.semantics.Meaning` it carries. What is retained is that
  sentence with its content words replaced by **slots**, and everything else — word order, case
  markers, particles, the negator, the question word — kept as fixed material.

**Two demonstrations, not one, and they have to disagree.** A shape supported by a single sentence
is that sentence. A shape is kept only when at least two demonstrations produce it *with different
fillers in at least one slot*, which is the smallest evidence that a slot is a position rather
than a word. Then every kept construction is re-run over the whole demonstration corpus, and one
that reads any demonstration into a **different** meaning than the one it came with is dropped —
a grammar that contradicts its own lessons is not a grammar.

**Every candidate is tried; the winner is chosen on evidence, never on order.** That is the rule
:mod:`nyxara.njp.semantics` and :mod:`nyxara.njp.compile` both state, and for the same reason:
``or`` short-circuits, and a short-circuit is how a sentence gets read by the construction written
for a different sentence. Here it has one more consequence worth having — when the two best
constructions score **equal** and disagree about what was said, the sentence comes back
``unreadable`` with the frame ``ambiguous``. A genuinely ambiguous sentence read confidently one
way is worse than one refused.

**She speaks only what she can read back.** :meth:`Grammar.say` renders a meaning through a
construction and then **parses its own output**; unless what comes back is the meaning it started
from, the sentence is thrown away and the next candidate is tried, and if none survives she
returns the empty string. This is what :mod:`nyxara.njp.voice` has always refused to do without —
a fluent surface she cannot verify is babble with her name on it. Here the verification is
mechanical, so the surface is allowed.

**And a meaning is not a language.** A faculty holds many :class:`Tongue`\\ s. Reading with no
tongue named tries all of them and lets the evidence say which language a sentence was in;
:meth:`LanguageFaculty.translate` reads in one and says in another, which is the only honest test
that meaning is stored apart from surface — the same fact, twice, in two grammars that share no
word.

**What is NOT claimed.** This is not a parser for English. It has no dictionary, no syntax theory,
no recursion into subordinate clauses, and one construction is one flat sentence. A filler is
**one token** (see :data:`MAX_SPAN`), so a two-word noun phrase is *unreadable* rather than
mis-cut. Affix induction is concatenative — suffixes and prefixes only, no stem changes, no
infixes, no reduplication — so an irregular is a memorised pair and never a rule. Nothing here
decides truth or relevance: it produces a :class:`~nyxara.njp.semantics.Meaning` and stops, and
where it cannot, it produces an unreadable one rather than a guess.

Pure standard library. No LLM, no corpus on disk, no network. Every public entry point is
fail-soft: on any internal error it degrades to a null result rather than breaking a turn.
"""

from __future__ import annotations

import math
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from nyxara.njp.semantics import Meaning

__all__ = [
    "MAX_SPAN", "ROLES", "Segment", "Affix", "Morphology", "Lexicon",
    "Slot", "Construction", "Demonstration", "Grammar", "Tongue",
    "LanguageFaculty", "Learned", "Reading", "tokenize_surface",
]

#: The most tokens one filler may span. **One**, and it is a measurement rather than a
#: simplification.
#:
#: It was three, which reads "the black dog" as one subject and looks like the more capable
#: setting. What it actually buys is a grammar that cannot refuse: a construction of three bare
#: slots matches a four-word sentence by letting one slot swallow two words, so *a sentence with
#: one word too many comes back read instead of refused*. Measured over six minted languages,
#: twenty controls each: at three, two of the six refused only 12 and 16 of their 20; at one, all
#: six refuse all twenty. Abstention is the property this package spends everything else to keep,
#: and a phrase-sized filler is not worth buying a confident misreading with.
#:
#: The cost is named rather than hidden: a two-word noun phrase is **unreadable** here, not
#: mis-cut. Raising this constant is the one edit that widens what a construction accepts, which
#: is why the rule in :meth:`Construction._walk` — an unanchored slot stays one token whatever
#: this says — is written to survive somebody raising it.
MAX_SPAN = 1

#: The roles a construction can bind — deliberately the three that
#: :class:`~nyxara.njp.semantics.Meaning` already carries. This module adds no representation of
#: its own, so everything above language keeps reading exactly the object it always read.
ROLES: Tuple[str, ...] = ("subject", "verb", "object")

_TOKEN_RX = re.compile(r"[^\W\d_]+|\d+|[?!]", re.UNICODE)


def tokenize_surface(text: str) -> List[str]:
    """Lower-cased tokens, script-agnostic, with ``?`` and ``!`` kept as tokens of their own.

    Terminal punctuation is kept because in several languages it is the *only* mark a question
    carries, and dropping it would make a question and its assertion the same string. The full
    stop is dropped, because nothing distinguishes it from the end of the sentence.
    """
    try:
        folded = unicodedata.normalize("NFKC", str(text or "")).strip().lower()
        return [match.group(0) for match in _TOKEN_RX.finditer(folded)]
    except Exception:  # noqa: BLE001
        return []


# --------------------------------------------------------------------------- #
# morphology — affixes induced from a vocabulary, features bound by demonstration
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Affix:
    """One induced affix: its form, which end of the word it sits on, and its evidence."""

    form: str = ""
    side: str = "suffix"          # "suffix" | "prefix"
    stems: int = 0                #: distinct stems appearing both bare and with this affix
    feature: str = ""             #: bound by a lesson, never by induction

    @property
    def bound(self) -> bool:
        return bool(self.feature)

    def apply(self, stem: str) -> str:
        return f"{stem}{self.form}" if self.side == "suffix" else f"{self.form}{stem}"

    def strip(self, word: str) -> str:
        """The stem, or ``""`` where this affix is not on this word."""
        if self.side == "suffix":
            if len(word) > len(self.form) and word.endswith(self.form):
                return word[: len(word) - len(self.form)]
        elif len(word) > len(self.form) and word.startswith(self.form):
            return word[len(self.form):]
        return ""

    def to_dict(self) -> Dict[str, Any]:
        return {"form": self.form, "side": self.side, "stems": self.stems,
                "feature": self.feature}


@dataclass(frozen=True)
class Segment:
    """A word taken apart. ``analysed`` is false when no induced affix explains it."""

    word: str = ""
    stem: str = ""
    affix: str = ""
    side: str = ""
    feature: str = ""

    @property
    def analysed(self) -> bool:
        return bool(self.affix)

    def to_dict(self) -> Dict[str, Any]:
        return {"word": self.word, "stem": self.stem, "affix": self.affix,
                "side": self.side, "feature": self.feature}


class Morphology:
    """Affixes found in a vocabulary, and what — if anything — a lesson said they mean.

    The induction is paradigmatic rather than statistical over characters: a suffix counts once
    for every stem that appears in the vocabulary **both bare and with the suffix attached**. That
    is the one piece of evidence separating a morpheme from a coincidence of spelling. English
    ``-ing`` scores because *walk*, *talk* and *read* are all words; ``-ea`` scores nothing,
    because *r*, *t* and *br* are not.

    Two thresholds, both deliberately blunt: an affix has to be carried by ``min_stems`` distinct
    stems, and it may not be longer than the stems that carry it. Everything else about morphology
    — allomorphy, stem changes, infixes, whether an affix is inflectional or derivational — is
    outside what a concatenative induction can see, and is reported absent rather than
    approximated.
    """

    def __init__(self, *, min_stems: int = 3, max_affix: int = 4) -> None:
        self.min_stems = max(2, int(min_stems))
        self.max_affix = max(1, int(max_affix))
        self.vocabulary: Dict[str, int] = {}
        self.affixes: List[Affix] = []
        #: Pairs a lesson demonstrated that induction could not support, kept so the refusal is
        #: inspectable rather than silent.
        self.refused: List[Tuple[str, str, str]] = []
        self.irregular: Dict[Tuple[str, str], str] = {}

    # -- observation -------------------------------------------------------- #
    def observe(self, words: Iterable[str]) -> int:
        """Add words to the vocabulary. Returns how many were new."""
        new = 0
        for raw in words:
            word = str(raw or "").strip().lower()
            if not word or not word[0].isalpha():
                continue
            if word not in self.vocabulary:
                new += 1
            self.vocabulary[word] = self.vocabulary.get(word, 0) + 1
        return new

    def observe_text(self, text: str) -> int:
        return self.observe(tokenize_surface(text))

    # -- induction ---------------------------------------------------------- #
    def induce(self) -> List[Affix]:
        """Find the affixes the vocabulary supports. Idempotent, and keeps bound features."""
        bound = {(a.form, a.side): a.feature for a in self.affixes if a.feature}
        words = set(self.vocabulary)
        suffixes: Dict[str, Set[str]] = {}
        prefixes: Dict[str, Set[str]] = {}
        for word in words:
            for cut in range(1, len(word)):
                head, tail = word[:cut], word[cut:]
                # One cut, read twice: `head` is a stem with `tail` suffixed to it, or `head` is a
                # prefix on the stem `tail`. Which of the two the evidence supports is decided by
                # whichever of the halves is itself a word.
                if len(tail) <= self.max_affix and len(tail) <= len(head) and head in words:
                    suffixes.setdefault(tail, set()).add(head)
                if len(head) <= self.max_affix and len(head) <= len(tail) and tail in words:
                    prefixes.setdefault(head, set()).add(tail)
        found: List[Affix] = []
        for form, stems in suffixes.items():
            if len(stems) >= self.min_stems:
                found.append(Affix(form=form, side="suffix", stems=len(stems),
                                   feature=bound.get((form, "suffix"), "")))
        for form, stems in prefixes.items():
            if len(stems) >= self.min_stems:
                found.append(Affix(form=form, side="prefix", stems=len(stems),
                                   feature=bound.get((form, "prefix"), "")))
        # Longest first, then best supported: :meth:`analyse` takes the first that fits, and a
        # longer affix is the more specific claim about the same word.
        found.sort(key=lambda a: (-len(a.form), -a.stems, a.form))
        self.affixes = found
        return found

    # -- binding: what an affix means, and only where it was demonstrated ---- #
    def bind(self, base: str, inflected: str, feature: str) -> bool:
        """Bind the affix relating ``base`` to ``inflected`` to ``feature``.

        Refuses — and records the refusal — when the affix relating the two was never induced.
        The alternative is a "rule" whose entire evidence is the pair that named it, which is
        exactly what the wug test exists to tell apart from a rule.
        """
        base, inflected = str(base or "").strip().lower(), str(inflected or "").strip().lower()
        feature = str(feature or "").strip()
        if not base or not inflected or not feature or inflected == base:
            return False
        side, form = "", ""
        if inflected.startswith(base):
            side, form = "suffix", inflected[len(base):]
        elif inflected.endswith(base):
            side, form = "prefix", inflected[: len(inflected) - len(base)]
        if not form:
            # Not concatenative at all. Kept as a memorised pair — which is what an irregular is —
            # and never generalised to a stem it was not shown on.
            self.irregular[(base, feature)] = inflected
            self.refused.append((base, inflected, "not concatenative — kept as an irregular"))
            return False
        for index, affix in enumerate(self.affixes):
            if affix.form == form and affix.side == side:
                self.affixes[index] = Affix(form=form, side=side, stems=affix.stems,
                                            feature=feature)
                return True
        self.irregular[(base, feature)] = inflected
        self.refused.append((base, inflected, f"{side} {form!r} was never induced"))
        return False

    # -- use ---------------------------------------------------------------- #
    def analyse(self, word: str) -> Segment:
        """Split a word into stem and affix, or report it unanalysed."""
        word = str(word or "").strip().lower()
        if not word:
            return Segment()
        for affix in self.affixes:
            stem = affix.strip(word)
            if stem and len(stem) >= 2:
                return Segment(word=word, stem=stem, affix=affix.form, side=affix.side,
                               feature=affix.feature)
        return Segment(word=word, stem=word)

    def stem_of(self, word: str) -> str:
        """The stem, but only where the affix carries a **bound** feature.

        Stripping an affix she has induced and does not understand would silently merge two
        entities whose difference she cannot name, so an unbound affix leaves the word alone.
        """
        segment = self.analyse(word)
        return segment.stem if segment.feature else segment.word

    def feature_of(self, word: str) -> str:
        return self.analyse(word).feature

    def inflect(self, stem: str, feature: str) -> str:
        """The surface for ``stem`` under ``feature``, or ``""`` where nothing binds it.

        The empty string is an abstention and is meant to be scored as one. A morphology that
        answered every request would have nothing to say about the difference between a rule it
        learned and a rule nobody taught it.
        """
        stem = str(stem or "").strip().lower()
        feature = str(feature or "").strip()
        if not stem or not feature:
            return ""
        memorised = self.irregular.get((stem, feature))
        if memorised:
            return memorised
        for affix in self.affixes:
            if affix.feature == feature:
                return affix.apply(stem)
        return ""

    def features(self) -> List[str]:
        return sorted({a.feature for a in self.affixes if a.feature})

    def stats(self) -> Dict[str, Any]:
        return {"vocabulary": len(self.vocabulary), "affixes": len(self.affixes),
                "bound": sum(1 for a in self.affixes if a.feature),
                "features": self.features(), "irregular": len(self.irregular),
                "refused": len(self.refused)}

    # -- persistence -------------------------------------------------------- #
    #: The most word types carried across a restart, most-heard first. A vocabulary is evidence
    #: for the affixes and it is not the affixes; past this many, one more hapax changes no
    #: induction and costs a sidecar that has to be read on every wake.
    max_kept = 8192

    def to_dict(self) -> Dict[str, Any]:
        """The vocabulary and the **bound** features, and nothing derived from them.

        The affixes themselves are not stored — :meth:`induce` re-runs on load and finds exactly
        what the vocabulary supports, which is the point: a stored affix could outlive the
        evidence for it, and an affix with no evidence is the one thing this class exists not to
        have. Only the feature *bindings* are written out, because those came from a lesson rather
        than from the corpus and re-counting would not recover them.
        """
        ranked = sorted(self.vocabulary.items(), key=lambda kv: (-kv[1], kv[0]))[:self.max_kept]
        return {"vocabulary": dict(ranked),
                "bound": [[a.form, a.side, a.feature] for a in self.affixes if a.feature],
                "irregular": [[base, feature, form]
                              for (base, feature), form in self.irregular.items()]}

    def load_dict(self, data: Dict[str, Any]) -> None:
        data = data or {}
        self.vocabulary = {str(word): int(count)
                           for word, count in (data.get("vocabulary") or {}).items()}
        self.irregular = {}
        for row in data.get("irregular") or []:
            if isinstance(row, (list, tuple)) and len(row) == 3:
                self.irregular[(str(row[0]), str(row[1]))] = str(row[2])
        self.induce()
        bound = {(str(row[0]), str(row[1])): str(row[2])
                 for row in (data.get("bound") or [])
                 if isinstance(row, (list, tuple)) and len(row) == 3}
        for index, affix in enumerate(self.affixes):
            feature = bound.get((affix.form, affix.side))
            if feature:
                self.affixes[index] = Affix(form=affix.form, side=affix.side,
                                            stems=affix.stems, feature=feature)


# --------------------------------------------------------------------------- #
# word classes — discovered from where words occur, and named nothing
# --------------------------------------------------------------------------- #

class Lexicon:
    """Distributional word classes: words occurring in the same places are the same kind.

    The context of a word is the multiset of ``(side, neighbour)`` pairs it was seen with, with
    ``#`` standing for the sentence boundary. Two words fall in one class when the cosine between
    those vectors clears ``threshold``. The clustering is greedy and single-pass over words sorted
    by how much evidence each carries, so the classes are deterministic given the same corpus.

    **A neighbour seen once is not a context, it is a word.** Every content word in a fresh corpus
    is unique, so contexts built from raw neighbours share nothing and every word lands in a class
    of its own — measured, one class per word and nothing learned. So a neighbour occurring fewer
    than ``context_min`` times is bucketed as ``*``, and what is left as a feature is the material
    that actually recurs: the boundary, the particles, and whichever words the corpus repeats.
    That is the frequent/open split :mod:`nyxara.njp.semantics` makes by hand, arrived at here by
    counting instead of by a table — which is why it works in a language nobody wrote a table for.

    **The classes have no names, and that is the claim.** ``class:0`` is a set of words that
    behave alike. Calling it "noun" would assert a part-of-speech theory the evidence does not
    contain, and every other discovery layer here (``concepts``, ``core.represent``) refuses the
    same temptation for the same reason.
    """

    def __init__(self, *, threshold: float = 0.35, min_count: int = 2,
                 context_min: int = 2, passes: int = 2) -> None:
        self.threshold = float(threshold)
        self.min_count = max(1, int(min_count))
        self.context_min = max(1, int(context_min))
        self.passes = max(1, int(passes))
        self.counts: Dict[str, int] = {}
        self.contexts: Dict[str, Dict[str, int]] = {}
        self.classes: Dict[str, str] = {}
        self.members: Dict[str, List[str]] = {}

    def observe(self, tokens: Sequence[str]) -> None:
        """One sentence, already tokenised."""
        padded = ["#"] + [str(token).lower() for token in tokens] + ["#"]
        for index in range(1, len(padded) - 1):
            word = padded[index]
            self.counts[word] = self.counts.get(word, 0) + 1
            bag = self.contexts.setdefault(word, {})
            for key in (f"L:{padded[index - 1]}", f"R:{padded[index + 1]}"):
                bag[key] = bag.get(key, 0) + 1

    def observe_text(self, text: str) -> None:
        self.observe(tokenize_surface(text))

    @staticmethod
    def _cosine(left: Dict[str, int], right: Dict[str, int]) -> float:
        if not left or not right:
            return 0.0
        shared = set(left) & set(right)
        if not shared:
            return 0.0
        dot = sum(left[key] * right[key] for key in shared)
        norm = (math.sqrt(sum(value * value for value in left.values()))
                * math.sqrt(sum(value * value for value in right.values())))
        return dot / norm if norm else 0.0

    def _vectors(self, classes: Optional[Dict[str, str]] = None) -> Dict[str, Dict[str, int]]:
        """Contexts, with each neighbour named either by its word or by its **kind**.

        The first pass names it by its word, with the rare ones collapsed to ``*``. Every later
        pass names it by the class the previous pass put it in, which is the step that makes the
        induction bootstrap: two words whose neighbours are different words but the *same kind of
        word* have nothing in common in the first description and everything in common in the
        second.

        That is not a refinement, it is the difference between working and not working, and the
        measurement says so. In a clause ordered verb-subject-object the subject is the one
        position touching neither edge of the sentence, so two subjects sharing no verb and no
        object share **no context at all** — measured on one minted language, eight subject nouns
        in eight classes of one, and two words that behave identically judged different kinds. One
        further pass over kinds puts all eight together.
        """
        frequent = {word for word, count in self.counts.items() if count >= self.context_min}
        out: Dict[str, Dict[str, int]] = {}
        for word, bag in self.contexts.items():
            vector: Dict[str, int] = {}
            for key, count in bag.items():
                side, neighbour = key.split(":", 1)
                if neighbour == "#":
                    shown = "#"
                elif classes is not None:
                    shown = classes.get(neighbour, "*")
                else:
                    shown = neighbour if neighbour in frequent else "*"
                folded = f"{side}:{shown}"
                vector[folded] = vector.get(folded, 0) + count
            out[word] = vector
        return out

    def _cluster(self, vectors: Dict[str, Dict[str, int]]) -> Dict[str, str]:
        """Greedy single pass over words ordered by evidence. Deterministic given the corpus."""
        ranked = sorted((word for word, count in self.counts.items() if count >= self.min_count),
                        key=lambda word: (-self.counts[word], word))
        centroids: Dict[str, Dict[str, int]] = {}
        members: Dict[str, List[str]] = {}
        assigned: Dict[str, str] = {}
        for word in ranked:
            vector = vectors.get(word, {})
            best, best_score = "", 0.0
            for name, centroid in centroids.items():
                score = self._cosine(vector, centroid)
                if score > best_score:
                    best, best_score = name, score
            if best and best_score >= self.threshold:
                assigned[word] = best
                members[best].append(word)
                for key, value in vector.items():
                    centroids[best][key] = centroids[best].get(key, 0) + value
            else:
                name = f"class:{len(centroids)}"
                centroids[name] = dict(vector)
                assigned[word] = name
                members[name] = [word]
        self.members = members
        return assigned

    def induce(self) -> int:
        """Cluster, then re-cluster over neighbour kinds. Returns the number of classes found.

        ``passes`` is two and stops there deliberately. Each further pass describes a word by a
        description that was itself inferred, so the evidence gets thinner while the numbers look
        the same, and an induction run to a fixed point is one whose later passes are agreeing
        with themselves rather than with the corpus.
        """
        self.classes, self.members = {}, {}
        assigned = self._cluster(self._vectors())
        for _ in range(max(0, self.passes - 1)):
            assigned = self._cluster(self._vectors(assigned))
        self.classes = assigned
        return len(self.members)

    def class_of(self, word: str) -> str:
        return self.classes.get(str(word or "").strip().lower(), "")

    def same_class(self, left: str, right: str) -> Optional[bool]:
        """``True`` / ``False``, or ``None`` where she has no grounds — an unseen word.

        Three outcomes rather than two, for the reason every other organ here keeps three: a word
        she has never met is not a word that belongs to a different class, and answering "no"
        would be an assertion made out of an absence.
        """
        first, second = self.class_of(left), self.class_of(right)
        if not first or not second:
            return None
        return first == second

    def stats(self) -> Dict[str, Any]:
        return {"words": len(self.counts), "classified": len(self.classes),
                "classes": len(self.members)}

    # -- persistence -------------------------------------------------------- #
    #: The most words whose contexts are carried across a restart, most-heard first. The tail of
    #: a corpus is words met once, and :attr:`min_count` already refuses to classify those.
    max_kept = 4096

    def to_dict(self) -> Dict[str, Any]:
        """The contexts, which are the evidence. The classes are re-derived on load.

        Same discipline as :meth:`Morphology.to_dict` and for the same reason: a stored class
        assignment could disagree with the contexts stored beside it, and nothing downstream would
        be able to tell which of the two was right.
        """
        keep = [word for word, _count in
                sorted(self.counts.items(), key=lambda kv: (-kv[1], kv[0]))[:self.max_kept]]
        return {"counts": {word: self.counts[word] for word in keep},
                "contexts": {word: self.contexts.get(word, {}) for word in keep}}

    def load_dict(self, data: Dict[str, Any]) -> None:
        data = data or {}
        self.counts = {str(word): int(count)
                       for word, count in (data.get("counts") or {}).items()}
        self.contexts = {}
        for word, bag in (data.get("contexts") or {}).items():
            if isinstance(bag, dict):
                self.contexts[str(word)] = {str(key): int(value) for key, value in bag.items()}
        self.induce()


# --------------------------------------------------------------------------- #
# constructions — a sentence shape generalised from a demonstration
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Slot:
    """A position in a construction, with whatever fixed material clings to its filler.

    ``prefix`` and ``suffix`` are the case marker, the plural, the tense — anything the
    demonstration attached to the content word. Keeping them **on the slot** rather than
    discarding them is what lets two constructions differing only by a verb ending stay two
    constructions, instead of colliding into one that reads both of them wrong.
    """

    role: str = ""
    prefix: str = ""
    suffix: str = ""

    @property
    def bound_chars(self) -> int:
        return len(self.prefix) + len(self.suffix)

    def wrap(self, filler: str) -> str:
        return f"{self.prefix}{filler}{self.suffix}"

    def unwrap(self, token: str) -> str:
        """The filler inside ``token``, or ``""`` if the fixed material is not there."""
        if len(token) <= self.bound_chars:
            return ""
        if not token.startswith(self.prefix) or not token.endswith(self.suffix):
            return ""
        inner = token[len(self.prefix): len(token) - len(self.suffix)] \
            if self.suffix else token[len(self.prefix):]
        return inner if len(inner) >= 2 else ""


@dataclass
class Construction:
    """One learned sentence shape, and the record of how it has done.

    ``pattern`` is a tuple whose members are either a literal string or a :class:`Slot`. It
    matches a whole sentence and never part of one: a construction matching a *prefix* of a
    sentence would read "the cat sat on the mat" with the shape for "the cat sat" and lose the
    half that changes what it says.
    """

    tongue: str = ""
    pattern: Tuple[Any, ...] = ()
    kind: str = "assertion"
    negated: bool = False
    focus: str = ""
    modality: str = ""
    temporal: str = ""
    support: int = 0
    hits: int = 0
    misses: int = 0
    source: str = ""              #: one demonstration, kept for the report card only

    # -- shape -------------------------------------------------------------- #
    @property
    def signature(self) -> Tuple[Any, ...]:
        """What makes two demonstrations the same shape."""
        return (self.tongue, self.kind, self.negated, self.focus, self.modality, self.temporal,
                tuple(item if isinstance(item, str)
                      else ("slot", item.role, item.prefix, item.suffix)
                      for item in self.pattern))

    @property
    def shape(self) -> str:
        """The pattern as one readable line: ``<subject>gu <verb> nuza``."""
        return " ".join(item if isinstance(item, str)
                        else f"{item.prefix}<{item.role}>{item.suffix}"
                        for item in self.pattern)

    @property
    def key(self) -> str:
        """:attr:`signature` as a string — the same identity, in something JSON can hold."""
        return (f"{self.kind}|{int(self.negated)}|{self.focus}|{self.temporal}|"
                f"{self.modality}|{self.shape}")

    @property
    def roles(self) -> Tuple[str, ...]:
        return tuple(item.role for item in self.pattern if isinstance(item, Slot))

    @property
    def literals(self) -> int:
        return sum(1 for item in self.pattern if isinstance(item, str))

    @property
    def bound_chars(self) -> int:
        return sum(item.bound_chars for item in self.pattern if isinstance(item, Slot))

    @property
    def record(self) -> float:
        """Held-out precision, ``0.5`` before there is any. Never a default that flatters."""
        seen = self.hits + self.misses
        return self.hits / seen if seen else 0.5

    # -- matching ----------------------------------------------------------- #
    def match(self, tokens: Sequence[str]) -> List[Dict[str, str]]:
        """Every way this construction reads ``tokens``. Empty where it does not.

        All readings are returned rather than the first, because *which* one is right is a
        question about evidence and the caller is what holds the evidence. A construction
        returning two readings of one sentence has found an ambiguity, and saying so is the
        honest outcome.
        """
        out: List[Dict[str, str]] = []
        try:
            self._walk(list(tokens), 0, 0, {}, out)
        except Exception:  # noqa: BLE001
            return []
        return out

    def _walk(self, tokens: List[str], at: int, index: int,
              fills: Dict[str, str], out: List[Dict[str, str]]) -> None:
        if len(out) >= 8:                       # an ambiguity this deep is already reported
            return
        if index == len(self.pattern):
            if at == len(tokens):
                out.append(dict(fills))
            return
        item = self.pattern[index]
        if isinstance(item, str):
            if at < len(tokens) and tokens[at] == item:
                self._walk(tokens, at + 1, index + 1, fills, out)
            return
        remaining = len(self.pattern) - index - 1
        # Two reasons a slot is held to exactly one token, and they are different reasons.
        #
        # Fixed material: case marking a whole noun phrase is a real thing in real languages and
        # it is *not* something this matcher can see, so it is refused rather than guessed at.
        #
        # No literals anywhere in the pattern: an unanchored multi-token slot is unfalsifiable.
        # A construction of three bare slots would otherwise match a sentence of four words by
        # letting one slot swallow two of them — so a sentence with a word too many, which is
        # exactly the sentence her grammar should refuse, would come back read instead. With at
        # least one literal in the pattern the phrase has something to end against, and then a
        # span is a reading rather than a shrug.
        spans = (1,) if (item.bound_chars or not self.literals) else range(1, MAX_SPAN + 1)
        for span in spans:
            if at + span > len(tokens) - remaining:
                break
            if span == 1:
                filler = item.unwrap(tokens[at])
            else:
                filler = " ".join(tokens[at:at + span])
            if not filler:
                continue
            fills[item.role] = filler
            self._walk(tokens, at + span, index + 1, fills, out)
            fills.pop(item.role, None)

    # -- rendering ---------------------------------------------------------- #
    def render(self, fills: Dict[str, str]) -> str:
        """Instantiate. Returns ``""`` when a slot has nothing to put in it."""
        words: List[str] = []
        for item in self.pattern:
            if isinstance(item, str):
                words.append(item)
                continue
            filler = str(fills.get(item.role, "") or "").strip().lower()
            if not filler:
                return ""
            words.append(item.wrap(filler))
        return re.sub(r"\s+([?!])", r"\1", " ".join(words))

    def to_dict(self) -> Dict[str, Any]:
        return {"tongue": self.tongue, "shape": self.shape, "kind": self.kind,
                "negated": self.negated, "focus": self.focus, "temporal": self.temporal,
                "modality": self.modality, "support": self.support,
                "hits": self.hits, "misses": self.misses, "record": round(self.record, 3)}


@dataclass(frozen=True)
class Demonstration:
    """A surface and what it meant. The only thing a teacher ever hands this module."""

    surface: str = ""
    meaning: Meaning = field(default_factory=Meaning)


@dataclass
class Reading:
    """A candidate parse and the evidence behind it, so the choice between two is inspectable."""

    meaning: Meaning = field(default_factory=Meaning)
    construction: Optional[Construction] = None
    score: float = 0.0
    anchor: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {"meaning": self.meaning.to_dict(), "score": round(self.score, 4),
                "anchor": round(self.anchor, 4),
                "construction": self.construction.to_dict() if self.construction else None}


@dataclass
class Learned:
    """What one round of learning did, including everything it threw away and why."""

    tongue: str = ""
    kept: int = 0
    rejected: int = 0
    reasons: List[str] = field(default_factory=list)
    affixes: int = 0
    bound: int = 0
    classes: int = 0
    demonstrations: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {"tongue": self.tongue, "kept": self.kept, "rejected": self.rejected,
                "affixes": self.affixes, "bound": self.bound, "classes": self.classes,
                "demonstrations": self.demonstrations, "reasons": self.reasons[:6]}


# --------------------------------------------------------------------------- #
# the grammar — generalise, verify, read, and say only what reads back
# --------------------------------------------------------------------------- #

#: The fields of a :class:`~nyxara.njp.semantics.Meaning` a construction is answerable for, and
#: therefore the only ones a demonstration needs to carry across a restart. Confidence, frame and
#: tokens are all *outputs* of a reading and are deliberately not stored: a persisted confidence
#: would be re-loaded as evidence for the very grammar that produced it.
_MEANING_FIELDS = ("kind", "subject", "relation", "object", "verb", "focus", "temporal",
                   "modality")


def _meaning_to_dict(meaning: Meaning) -> Dict[str, Any]:
    out: Dict[str, Any] = {"negated": bool(meaning.negated)}
    for name in _MEANING_FIELDS:
        value = getattr(meaning, name, "")
        if value:
            out[name] = str(value)
    return out


def _meaning_from_dict(data: Dict[str, Any]) -> Meaning:
    meaning = Meaning(negated=bool(data.get("negated")))
    for name in _MEANING_FIELDS:
        setattr(meaning, name, str(data.get(name) or ""))
    meaning.kind = meaning.kind or "assertion"
    return meaning


def _fills_of(meaning: Meaning) -> Dict[str, str]:
    """What a construction has to be able to put back, given the meaning it came with.

    The **verb** role is filled from ``relation`` — the lemma — rather than from ``verb``, the
    surface. That one choice is what makes tense visible to this module: whatever the surface
    carries beyond the lemma becomes fixed material on the slot, so *"glim"* and *"glimta"* end
    up in two constructions rather than in one that cannot tell a past from a present.
    """
    out: Dict[str, str] = {}
    for role in ROLES:
        value = getattr(meaning, "relation" if role == "verb" else role, "")
        value = " ".join(str(value or "").strip().lower().split())
        if value:
            out[role] = value
    return out


def _affixes_around(token: str, filler: str) -> Optional[Tuple[str, str]]:
    """Where ``filler`` sits inside ``token``, as ``(prefix, suffix)``. ``None`` if it does not."""
    at = token.find(filler)
    if at < 0:
        return None
    return token[:at], token[at + len(filler):]


class Grammar:
    """The constructions of one language: how they are learned, and how one is chosen.

    Learning is three steps and each of them can reject:

    #. **generalise** — one demonstration becomes one candidate shape, or nothing at all when the
       meaning it came with cannot be located in the surface it came with;
    #. **support** — a shape is kept only where two or more demonstrations produced it with
       different fillers, so a sentence cannot masquerade as a rule;
    #. **verify** — every kept shape is run back over the whole corpus, and one that reads any
       demonstration into a different meaning than it arrived with is dropped.
    """

    #: Demonstrations needed before a shape counts as a shape.
    min_support = 2

    def __init__(self, tongue: str = "en") -> None:
        self.tongue = str(tongue or "en")
        self.constructions: List[Construction] = []
        self.demonstrations: List[Demonstration] = []
        self.rejected: List[str] = []

    # -- teaching ----------------------------------------------------------- #
    def show(self, surface: str, meaning: Meaning) -> bool:
        """Record one demonstration. Nothing is generalised until :meth:`learn` runs."""
        if not str(surface or "").strip() or meaning is None:
            return False
        self.demonstrations.append(Demonstration(surface=str(surface), meaning=meaning))
        return True

    def generalise(self, surface: str, meaning: Meaning) -> Optional[Construction]:
        """One demonstration → one candidate shape, or ``None``.

        Content words become slots; **everything else stays**. That asymmetry is the whole of the
        generalisation: word order, particles, the negator and the case markers are not noise
        around the meaning, they are how the meaning was carried, and a shape that dropped them
        would match sentences that mean something else.
        """
        tokens = tokenize_surface(surface)
        fills = _fills_of(meaning)
        if not tokens or not fills:
            return None
        placed = self._place(tokens, fills)
        if placed is None:
            return None
        pattern: List[Any] = []
        index = 0
        while index < len(tokens):
            here = placed.get(index)
            if here is None:
                pattern.append(tokens[index])
                index += 1
                continue
            role, span, prefix, suffix = here
            pattern.append(Slot(role=role, prefix=prefix, suffix=suffix))
            index += span
        return Construction(
            tongue=self.tongue, pattern=tuple(pattern),
            kind=meaning.kind or "assertion", negated=bool(meaning.negated),
            focus=meaning.focus or "", modality=meaning.modality or "",
            temporal=meaning.temporal or "", support=1, source=" ".join(tokens))

    def _place(self, tokens: Sequence[str],
               fills: Dict[str, str]) -> Optional[Dict[int, Tuple[str, int, str, str]]]:
        """Assign every role to a non-overlapping span of ``tokens``, or refuse.

        Refusal is the right answer more often than it looks. A demonstration whose subject does
        not appear in its own surface is not a hard sentence, it is a **mislabelled** one, and
        generalising it would mint a shape that reads every sentence of that form into whatever
        the mislabelling said.
        """
        options: Dict[str, List[Tuple[int, int, str, str]]] = {}
        for role, filler in fills.items():
            found: List[Tuple[int, int, str, str]] = []
            for start in range(len(tokens)):
                for span in range(1, MAX_SPAN + 1):
                    if start + span > len(tokens):
                        break
                    if span == 1:
                        around = _affixes_around(tokens[start], filler)
                        if around is not None:
                            found.append((start, 1, around[0], around[1]))
                    elif " ".join(tokens[start:start + span]) == filler:
                        found.append((start, span, "", ""))
            if not found:
                return None
            # The tightest wrapping first: a filler that *is* the token beats one that had to
            # explain away four characters of what it did not cover.
            found.sort(key=lambda item: (len(item[2]) + len(item[3]), item[1], item[0]))
            options[role] = found
        roles = sorted(options, key=lambda role: len(options[role]))
        chosen: Dict[int, Tuple[str, int, str, str]] = {}
        used: Set[int] = set()

        def assign(depth: int) -> bool:
            if depth == len(roles):
                return True
            role = roles[depth]
            for start, span, prefix, suffix in options[role]:
                positions = set(range(start, start + span))
                if positions & used:
                    continue
                used.update(positions)
                chosen[start] = (role, span, prefix, suffix)
                if assign(depth + 1):
                    return True
                used.difference_update(positions)
                chosen.pop(start, None)
            return False

        return chosen if assign(0) else None

    def learn(self) -> Learned:
        """Generalise every demonstration, keep what is supported, drop what contradicts."""
        report = Learned(tongue=self.tongue, demonstrations=len(self.demonstrations))
        candidates: Dict[Tuple[Any, ...], List[Tuple[Construction, Dict[str, str]]]] = {}
        for shown in self.demonstrations:
            candidate = self.generalise(shown.surface, shown.meaning)
            if candidate is None:
                report.rejected += 1
                report.reasons.append(f"{shown.surface!r}: its meaning is not in its surface")
                continue
            candidates.setdefault(candidate.signature, []).append(
                (candidate, _fills_of(shown.meaning)))
        kept: List[Construction] = []
        for signature, group in candidates.items():
            constructions = [item[0] for item in group]
            fillings = [item[1] for item in group]
            if len(group) < self.min_support:
                report.rejected += 1
                report.reasons.append(
                    f"{constructions[0].source!r}: one demonstration is a sentence, not a shape")
                continue
            if not self._varies(fillings):
                report.rejected += 1
                report.reasons.append(
                    f"{constructions[0].source!r}: {len(group)} demonstrations, identical fillers")
                continue
            head = constructions[0]
            head.support = len(group)
            kept.append(head)
        previous = {c.signature: c for c in self.constructions}
        for construction in kept:
            older = previous.get(construction.signature)
            if older is not None:
                construction.support = max(construction.support, older.support)
                construction.hits, construction.misses = older.hits, older.misses
        self.constructions = kept
        self._verify(report)
        report.kept = len(self.constructions)
        self.rejected = list(report.reasons)
        return report

    @staticmethod
    def _varies(fillings: Sequence[Dict[str, str]]) -> bool:
        """Did any slot ever hold two different words? If not, this is one sentence twice."""
        for role in ROLES:
            seen = {filling.get(role, "") for filling in fillings}
            if len(seen - {""}) > 1:
                return True
        return False

    def _verify(self, report: Learned) -> None:
        """Re-read every demonstration. Drop any construction that changes what one of them said.

        This is the step that makes the grammar answerable to its own lessons. A construction is
        *not* dropped for failing to be chosen — several shapes can read one sentence and only one
        can win — it is dropped for **winning and being wrong**, which is the only failure that
        puts a false meaning into the rest of the brain.
        """
        for _pass in range(3):
            guilty: Dict[Tuple[Any, ...], str] = {}
            for shown in self.demonstrations:
                got = self.read(shown.surface)
                if not got.readable:
                    continue
                if self._agrees(got, shown.meaning):
                    continue
                winner = self._winner_signature(shown.surface)
                if winner is not None:
                    guilty[winner] = shown.surface
            if not guilty:
                return
            before = len(self.constructions)
            self.constructions = [c for c in self.constructions if c.signature not in guilty]
            report.rejected += before - len(self.constructions)
            for surface in list(guilty.values())[:3]:
                report.reasons.append(f"{surface!r}: a shape read its own lesson differently")

    def _winner_signature(self, surface: str) -> Optional[Tuple[Any, ...]]:
        readings = self.candidates(surface)
        return readings[0].construction.signature if readings and readings[0].construction \
            else None

    @staticmethod
    def _agrees(got: Meaning, want: Meaning) -> bool:
        """Same sentence, same reading — on the fields a construction is responsible for."""
        def norm(value: Any) -> str:
            return " ".join(str(value or "").strip().lower().split())
        return (got.kind == (want.kind or "assertion")
                and bool(got.negated) == bool(want.negated)
                and norm(got.focus) == norm(want.focus)
                and norm(got.subject) == norm(want.subject)
                and norm(got.object) == norm(want.object)
                and norm(got.relation) == norm(want.relation))

    # -- reading ------------------------------------------------------------ #
    def candidates(self, surface: str) -> List[Reading]:
        """Every reading every construction offers, best evidence first."""
        tokens = tokenize_surface(surface)
        if not tokens:
            return []
        out: List[Reading] = []
        for construction in self.constructions:
            for fills in construction.match(tokens):
                anchor = float(construction.literals) + construction.bound_chars / 2.0
                slack = sum(len(value.split()) - 1 for value in fills.values())
                score = (anchor - 0.5 * slack + 0.25 * min(construction.support, 8)
                         + construction.record)
                out.append(Reading(meaning=self._meaning(construction, fills, anchor, tokens),
                                   construction=construction, score=score, anchor=anchor))
        out.sort(key=lambda reading: -reading.score)
        return out

    def read(self, surface: str) -> Meaning:
        """The best-supported reading, or an unreadable Meaning.

        Two ways to come back unreadable, and they are different facts about the sentence:
        ``frame=""`` means no construction matched at all, and ``frame="ambiguous"`` means two
        matched equally well and disagreed. The second is the one worth having — it is a sentence
        her grammar genuinely cannot resolve, and reporting it as a confident reading of one of
        the two would be inventing a decision the evidence does not contain.
        """
        try:
            readings = self.candidates(surface)
        except Exception:  # noqa: BLE001
            return Meaning(language=self.tongue)
        if not readings:
            return Meaning(language=self.tongue)
        best = readings[0]
        for other in readings[1:]:
            if other.score < best.score - 1e-9:
                break
            if not self._agrees(other.meaning, best.meaning):
                return Meaning(kind="unreadable", frame="ambiguous", language=self.tongue)
        return best.meaning

    def anchored(self, surface: str) -> bool:
        """Did the construction that won this sentence match any **fixed material**?

        A literal token, or a case marker, or a tense ending — anything that is a fact about the
        words in front of it rather than about their number and order. It is the one property that
        separates a reading with lexical evidence from a reading that would have fired on any
        three tokens whatsoever, and :mod:`nyxara.njp.grounding` uses exactly that distinction to
        decide whether a learned construction may outrank the shipped positional frame.
        """
        try:
            readings = self.candidates(surface)
        except Exception:  # noqa: BLE001
            return False
        if not readings or readings[0].construction is None:
            return False
        return readings[0].anchor > 0.0

    def _meaning(self, construction: Construction, fills: Dict[str, str],
                 anchor: float, tokens: Sequence[str]) -> Meaning:
        meaning = Meaning(
            kind=construction.kind, negated=construction.negated, focus=construction.focus,
            modality=construction.modality, temporal=construction.temporal,
            frame=f"learned:{construction.tongue}", language=construction.tongue,
        )
        meaning.subject = fills.get("subject", "")
        meaning.object = fills.get("object", "")
        meaning.relation = fills.get("verb", "")
        meaning.verb = meaning.relation
        meaning.confidence = round(min(0.95, 0.55 + 0.05 * anchor
                                       + 0.3 * (construction.record - 0.5)), 4)
        return meaning

    # -- saying ------------------------------------------------------------- #
    def say(self, meaning: Meaning) -> str:
        """A sentence for ``meaning``, or ``""``.

        Rendered, then **read back**. A construction whose output does not parse into the meaning
        it was given is not used, however well it scored, and when none survives that check she
        says nothing at all. There is no best-effort surface here for the same reason there is no
        best-effort program in :mod:`nyxara.njp.coding`: a sentence that nearly says what was
        meant says something else.
        """
        try:
            fills = _fills_of(meaning)
            if not fills:
                return ""
            wanted = set(fills)
            ranked = sorted(
                (c for c in self.constructions
                 if set(c.roles) == wanted
                 and c.kind == (meaning.kind or "assertion")
                 and bool(c.negated) == bool(meaning.negated)
                 and (c.focus or "") == (meaning.focus or "")
                 and (c.temporal or "") == (meaning.temporal or "")
                 and (c.modality or "") == (meaning.modality or "")),
                key=lambda c: (-c.record, -c.support, -c.literals))
            for construction in ranked:
                surface = construction.render(fills)
                if not surface:
                    continue
                if self._agrees(self.read(surface), meaning):
                    return surface
        except Exception:  # noqa: BLE001
            return ""
        return ""

    # -- the record --------------------------------------------------------- #
    def score(self, surface: str, meaning: Meaning) -> bool:
        """Grade one held-out sentence against the construction that won it.

        This is what turns :attr:`Construction.record` into something other than a default. It is
        called by whoever holds the answer — the school, or a later turn in which the Master said
        what he meant — and never by :meth:`read`, which has no way of knowing.
        """
        readings = self.candidates(surface)
        if not readings or readings[0].construction is None:
            return False
        construction = readings[0].construction
        if self._agrees(readings[0].meaning, meaning):
            construction.hits += 1
            return True
        construction.misses += 1
        return False

    def stats(self) -> Dict[str, Any]:
        return {"tongue": self.tongue, "constructions": len(self.constructions),
                "demonstrations": len(self.demonstrations),
                "shapes": [c.to_dict() for c in self.constructions[:8]]}

    # -- persistence -------------------------------------------------------- #
    #: The most demonstrations carried across a restart. A grammar's demonstrations are its
    #: evidence and they do not expire, but a sidecar is not an archive: the newest are kept
    #: because a shape supported by five hundred sentences is not supported any better by a
    #: thousand, and one whose only support is older than that is a shape she has not used.
    max_kept = 512

    def to_dict(self) -> Dict[str, Any]:
        """The **lessons**, not the conclusions.

        Constructions are re-derived on load by re-running :meth:`learn`, rather than written out
        and read back. It is a little slower and it makes one thing impossible: a persisted
        grammar cannot contain a construction that its own demonstrations do not support. A
        sidecar that stored the conclusions could be edited, truncated or written by an older
        version into exactly that state, and nothing downstream would ever find out.

        What *is* stored beside them is each construction's held-out record, keyed by
        :attr:`Construction.key`, because that is evidence from outside the lessons and re-running
        them would not recover it.
        """
        return {
            "tongue": self.tongue,
            "demonstrations": [{"surface": shown.surface,
                                "meaning": _meaning_to_dict(shown.meaning)}
                               for shown in self.demonstrations[-self.max_kept:]],
            "records": {c.key: [c.hits, c.misses] for c in self.constructions},
        }

    def load_dict(self, data: Dict[str, Any]) -> None:
        self.demonstrations = []
        for row in (data or {}).get("demonstrations") or []:
            try:
                self.demonstrations.append(
                    Demonstration(surface=str(row.get("surface") or ""),
                                  meaning=_meaning_from_dict(row.get("meaning") or {})))
            except Exception:  # noqa: BLE001 — one unreadable row is not a lost grammar
                continue
        self.learn()
        records = (data or {}).get("records") or {}
        for construction in self.constructions:
            got = records.get(construction.key)
            if isinstance(got, (list, tuple)) and len(got) == 2:
                construction.hits, construction.misses = int(got[0]), int(got[1])


# --------------------------------------------------------------------------- #
# a tongue, and the faculty that holds several of them
# --------------------------------------------------------------------------- #

class Tongue:
    """One language: its morphology, its word classes, and its constructions.

    Kept as one object because the three are not independent — the grammar reads a noun's stem
    through the morphology, and the morphology's vocabulary is whatever the grammar was shown.
    Keeping them per-language is the other half of it: a suffix is a fact about a language, and
    a faculty that pooled them would let an English plural cut a Hindi word in half.
    """

    def __init__(self, name: str = "en") -> None:
        self.name = str(name or "en")
        self.morphology = Morphology()
        self.lexicon = Lexicon()
        self.grammar = Grammar(self.name)
        self.heard = 0

    def hear(self, text: str) -> None:
        """Surface only, with no meaning attached — which is most of what anyone ever hears."""
        tokens = tokenize_surface(text)
        if not tokens:
            return
        self.heard += 1
        self.morphology.observe(tokens)
        self.lexicon.observe(tokens)

    def hear_words(self, words: Iterable[str]) -> int:
        """A word list, which is **not** a corpus, and is kept out of the class induction.

        The distinction is not fussiness. A wug test is administered on isolated forms — *this is
        a wug; now there are two of them* — and feeding those forms to :class:`Lexicon` as though
        they were sentences would invent adjacencies nobody uttered and put the two halves of one
        paradigm in one distributional class for a reason that is an artefact of how the lesson
        was delivered.
        """
        return self.morphology.observe(words)

    def show(self, surface: str, meaning: Meaning) -> bool:
        self.hear(surface)
        return self.grammar.show(surface, meaning)

    def learn(self) -> Learned:
        self.morphology.induce()
        classes = self.lexicon.induce()
        report = self.grammar.learn()
        report.affixes = len(self.morphology.affixes)
        report.bound = sum(1 for affix in self.morphology.affixes if affix.feature)
        report.classes = classes
        return report

    def read(self, surface: str) -> Meaning:
        """Parse, then let the morphology have the last word on the content words.

        The stem is substituted **only** where the affix carries a feature a lesson bound, so
        ``zorbik`` becomes ``zorb`` once she knows what ``-ik`` does and stays ``zorbik`` while
        she does not. An organ that stripped what it could not name would be merging two entities
        on the strength of a spelling.
        """
        meaning = self.grammar.read(surface)
        if not meaning.readable:
            return meaning
        for role in ("subject", "object"):
            value = getattr(meaning, role, "")
            if value and " " not in value:
                setattr(meaning, role, self.morphology.stem_of(value))
        return meaning

    def say(self, meaning: Meaning) -> str:
        return self.grammar.say(meaning)

    def stats(self) -> Dict[str, Any]:
        return {"name": self.name, "heard": self.heard,
                "morphology": self.morphology.stats(), "lexicon": self.lexicon.stats(),
                "grammar": self.grammar.stats()}

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "heard": self.heard,
                "morphology": self.morphology.to_dict(), "lexicon": self.lexicon.to_dict(),
                "grammar": self.grammar.to_dict()}

    def load_dict(self, data: Dict[str, Any]) -> None:
        data = data or {}
        self.heard = int(data.get("heard") or 0)
        self.morphology.load_dict(data.get("morphology") or {})
        self.lexicon.load_dict(data.get("lexicon") or {})
        self.grammar.load_dict(data.get("grammar") or {})


class LanguageFaculty:
    """Every language she has been taught, and the one thing that is not in any of them.

    A meaning belongs to no language. That is not a slogan here — it is the only reason
    :meth:`translate` can exist without a phrase table, and the reason a
    :class:`~nyxara.njp.semantics.Meaning` produced by one tongue can be handed to another and
    come back as a sentence sharing not one character with the first.

    On day one she holds nothing: :meth:`read` returns an unreadable Meaning, :meth:`say` returns
    the empty string, and neither is a failure. Everything this faculty can do, somebody showed
    her.
    """

    def __init__(self, *, default: str = "en") -> None:
        self.default = str(default or "en")
        self.tongues: Dict[str, Tongue] = {}
        self.readings = 0
        self.refusals = 0
        self.said = 0
        self.mute = 0

    # -- the tongues -------------------------------------------------------- #
    def tongue(self, name: Optional[str] = None) -> Tongue:
        key = str(name or self.default)
        if key not in self.tongues:
            self.tongues[key] = Tongue(key)
        return self.tongues[key]

    def known(self) -> List[str]:
        return sorted(self.tongues)

    # -- learning ----------------------------------------------------------- #
    def hear(self, text: str, *, tongue: Optional[str] = None) -> None:
        try:
            self.tongue(tongue).hear(text)
        except Exception:  # noqa: BLE001
            pass

    def hear_words(self, words: Iterable[str], *, tongue: Optional[str] = None) -> int:
        """A list of forms, for the morphology only. See :meth:`Tongue.hear_words`."""
        try:
            return self.tongue(tongue).hear_words(words)
        except Exception:  # noqa: BLE001
            return 0

    def classify(self, *, tongue: Optional[str] = None) -> int:
        """Re-run the class induction alone. Returns how many classes the corpus supports."""
        try:
            return self.tongue(tongue).lexicon.induce()
        except Exception:  # noqa: BLE001
            return 0

    def show(self, surface: str, meaning: Meaning, *, tongue: Optional[str] = None) -> bool:
        try:
            return self.tongue(tongue).show(surface, meaning)
        except Exception:  # noqa: BLE001
            return False

    def bind(self, base: str, inflected: str, feature: str, *,
             tongue: Optional[str] = None) -> bool:
        """Demonstrate one inflection. Returns whether an induced affix could carry it."""
        try:
            spoken = self.tongue(tongue)
            spoken.morphology.observe((base, inflected))
            spoken.morphology.induce()
            return spoken.morphology.bind(base, inflected, feature)
        except Exception:  # noqa: BLE001
            return False

    def learn(self, *, tongue: Optional[str] = None) -> Learned:
        try:
            return self.tongue(tongue).learn()
        except Exception as exc:  # noqa: BLE001
            return Learned(tongue=str(tongue or self.default),
                           reasons=[f"{type(exc).__name__}: {exc}"])

    def learn_all(self) -> List[Learned]:
        return [self.learn(tongue=name) for name in self.known()]

    # -- use ---------------------------------------------------------------- #
    def read(self, surface: str, *, tongue: Optional[str] = None) -> Meaning:
        """Read, in a named language or in whichever of hers reads it best.

        With no tongue named this is language identification done the only honest way available
        here: not by guessing from the alphabet, but by which grammar can actually account for the
        sentence. A tie between two languages that disagree about what was said comes back
        unreadable, exactly as a tie inside one language does.
        """
        try:
            if tongue is not None:
                meaning = self.tongue(tongue).read(surface)
            else:
                best: Optional[Tuple[float, Meaning]] = None
                rivals: List[Tuple[float, Meaning]] = []
                for name in self.known():
                    spoken = self.tongues[name]
                    readings = spoken.grammar.candidates(surface)
                    if not readings:
                        continue
                    scored = (readings[0].score, spoken.read(surface))
                    rivals.append(scored)
                    if best is None or scored[0] > best[0]:
                        best = scored
                if best is None:
                    meaning = Meaning()
                else:
                    contested = [meaning for score, meaning in rivals
                                 if score >= best[0] - 1e-9
                                 and not Grammar._agrees(meaning, best[1])]
                    meaning = Meaning(kind="unreadable", frame="ambiguous") if contested \
                        else best[1]
        except Exception:  # noqa: BLE001
            return Meaning()
        if meaning.readable:
            self.readings += 1
        else:
            self.refusals += 1
        return meaning

    def say(self, meaning: Meaning, *, tongue: Optional[str] = None) -> str:
        try:
            surface = self.tongue(tongue).say(meaning)
        except Exception:  # noqa: BLE001
            surface = ""
        if surface:
            self.said += 1
        else:
            self.mute += 1
        return surface

    def translate(self, surface: str, *, into: str, frm: Optional[str] = None) -> str:
        """Read in one language, say in another. ``""`` when either half cannot be done.

        No phrase table and no alignment: the only thing carried across is the
        :class:`~nyxara.njp.semantics.Meaning`, so a translation she produces is one she could
        also have read back, in the target language, into the meaning she started from.
        """
        meaning = self.read(surface, tongue=frm)
        if not meaning.readable:
            return ""
        return self.say(meaning, tongue=into)

    # -- morphology, straight through --------------------------------------- #
    def inflect(self, stem: str, feature: str, *, tongue: Optional[str] = None) -> str:
        try:
            return self.tongue(tongue).morphology.inflect(stem, feature)
        except Exception:  # noqa: BLE001
            return ""

    def analyse(self, word: str, *, tongue: Optional[str] = None) -> Segment:
        try:
            return self.tongue(tongue).morphology.analyse(word)
        except Exception:  # noqa: BLE001
            return Segment()

    def anchored(self, surface: str, *, tongue: Optional[str] = None) -> bool:
        """Whether the winning construction matched fixed material. See :meth:`Grammar.anchored`.

        With no tongue named this is true where **any** of her grammars reads the sentence with
        lexical evidence, which is the right question for a caller asking "does she have a real
        account of this sentence" rather than "which language is it".
        """
        try:
            if tongue is not None:
                return self.tongue(tongue).grammar.anchored(surface)
            return any(self.tongues[name].grammar.anchored(surface) for name in self.known())
        except Exception:  # noqa: BLE001
            return False

    def same_class(self, left: str, right: str, *,
                   tongue: Optional[str] = None) -> Optional[bool]:
        try:
            return self.tongue(tongue).lexicon.same_class(left, right)
        except Exception:  # noqa: BLE001
            return None

    def stats(self) -> Dict[str, Any]:
        return {"tongues": self.known(), "readings": self.readings, "refusals": self.refusals,
                "said": self.said, "mute": self.mute,
                "detail": {name: spoken.stats() for name, spoken in self.tongues.items()}}

    # -- persistence -------------------------------------------------------- #
    def to_dict(self) -> Dict[str, Any]:
        """Every tongue she holds, as the lessons that produced it.

        A grammar that did not survive a restart would be the one organ in this brain that has to
        be taught again every morning — and this package's whole claim about persistence is that
        the brain that wakes is the brain that slept. What comes back is re-derived from the
        demonstrations rather than read out of the file, so a sidecar can make her forget and it
        cannot make her believe a shape nobody showed her.
        """
        out: Dict[str, Any] = {"default": self.default, "tongues": {}}
        for name, spoken in self.tongues.items():
            try:
                out["tongues"][name] = spoken.to_dict()
            except Exception:  # noqa: BLE001 — one unwritable tongue is not a lost faculty
                continue
        return out

    def load_dict(self, data: Dict[str, Any]) -> None:
        data = data or {}
        self.default = str(data.get("default") or self.default)
        for name, blob in (data.get("tongues") or {}).items():
            try:
                self.tongue(str(name)).load_dict(blob or {})
            except Exception:  # noqa: BLE001
                continue
