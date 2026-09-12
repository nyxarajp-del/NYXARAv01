"""NYXARA · njp/puzzle.py — hard general-knowledge problems (🧠, NJP V.22).

:mod:`nyxara.njp.general` examines what she knows. Every one of its seven papers is answered by
**one fact or one walk**: a lookup, a membership test, an inheritance step, an edge read backwards.
She scores 0.979 on it, and that number says nothing at all about whether she can *reason*, because
no item on it requires putting two unrelated facts together.

This module is the other half. It asks the questions a hard general-knowledge exam asks — the ones
whose answer is nowhere in the store under any key, and has to be **constructed**:

===========  ==================================================================================
Form         What it needs that a lookup does not
===========  ==================================================================================
bridge       Two *different* relations chained. "What is the currency of the country the Taj
             Mahal is in?" — ``located_in`` then ``located_in`` then ``currency``, three hops
             across three domains, and no fact anywhere says what the Taj Mahal's currency is.
commonality  A set intersection over two subjects' relations. "What do a whale and a bat have in
             common?" is not stored about either of them.
odd one out   A majority vote over a set, then the exception. Nothing in the store is *about* the
             set; the set is made up by the question.
constraint   A search under two simultaneous conditions. "Which metal is liquid at room
             temperature?" — the store can list metals and can list what is liquid, and the
             answer is the intersection, which is not a fact.
analogy      A relation identified from one pair and applied to a third term. "Sparrow is to bird
             as tiger is to what?" — the relation is not named in the question.
chain        Transitive reach beyond one hop, with the path. "What does deforestation eventually
             lead to?" wants the end of a walk, not the first step.
===========  ==================================================================================

**The floor was measured before any of this was written**, on the shipped corpus:

* ``bridge``: 211 opportunities exist, :meth:`CognitiveLearningCore.walk_shape` solves **117 of
  120** sampled — and **0 of 40** could be *asked*, because no pattern in
  ``grounding._QUESTION_PATTERNS`` reads a nested question. The knowledge was there, the walk was
  there, and the two could not be introduced to each other.
* ``commonality``, ``odd one out``, ``constraint`` and ``analogy``: **no operation existed at
  all** — not weak, absent.
* ``chain``: 4 of 80 starts reached a second hop, which turned out to measure the *corpus* rather
  than the walk. Only 35 two-hop causal chains existed to be found; `reach` handles them
  correctly when they are there.

Three rules this module is built on.

**Every answer carries its derivation, and an answer with no path is not returned.** A
:class:`Solution` holds the steps that produced it. This is not decoration: a bridge answer that
cannot say which three edges it crossed is indistinguishable from a guess, and the whole claim of
this module is that these answers are *constructed* rather than recalled.

**Nothing here writes to the store.** A constructed answer is an inference, not a fact, and filing
it would let a second question read it back as though somebody had said it. That is the failure
:data:`~nyxara.njp.core._COMPOSED_CEILING` exists to price and this module must not route around.

**A question it cannot read is refused, not guessed.** :meth:`PuzzleSolver.solve` returns an empty
:class:`Solution` when no form matches, and the exam counts that as silence. A solver that answers
every string is a solver whose score means nothing.
"""
from __future__ import annotations

import collections
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

#: How many hops a constructed answer may cross. Beyond this the confidence a chain carries is
#: below anything worth stating — the same argument :data:`~nyxara.njp.general.MAX_INHERITED_HOPS`
#: makes about inheritance, and measured the same way rather than chosen. Four rather than three
#: because a real nested question needs it: the Taj Mahal is in Agra, in Uttar Pradesh, in
#: India, and only then is there a currency to report — three locating hops before the fourth
#: that answers.
MAX_BRIDGE_HOPS = 4

#: Relations that can stand in the middle of a bridge — the ones that genuinely locate a subject
#: inside something else, so that a fact about the container is *about* the subject's situation.
#:
#: Deliberately narrow. ``causes`` chains are `chain`'s business and mean something different, and
#: ``has_property`` in the middle would let "the currency of the property of X" be constructed,
#: which is not a question anybody asks and would produce fluent nonsense.
BRIDGE_LINKS: Tuple[str, ...] = ("located_in", "part_of", "is_a")

#: Words a question uses about itself rather than about its subject; stripped before a term is
#: looked up so "the country India" and "India" reach the same key.
_HEDGE = re.compile(
    r"^(?:the|a|an|that|which|what|whose)\s+|"
    r"\s+(?:is|are|was|were|located|situated|found|placed|based)\s*$")

#: "the <relation> of the <relation> of <X>" — the nested form, innermost subject last.
_NESTED = re.compile(
    r"^what(?:'s| is|\s+are)\s+(?:the\s+)?(?P<outer>[\w ]+?)\s+of\s+"
    r"(?:the\s+)?(?P<inner>[\w ]+?)\s+(?:of|in|where)\s+(?P<subject>.+?)\s*(?:is|are)?\s*\??$")

#: "the <relation> of the <container> <X> is in" — the same walk, said the way people say it.
#: The container and the subject are **not** split by the regex, and that is deliberate. Both are
#: multi-word phrases with nothing between them but a space, so any fixed split is a guess: with a
#: non-greedy container, "the capital of the indian state that mumbai is in" split as container
#: *indian* and subject *state that mumbai*, and every such question came back silent. The blob is
#: captured whole and :meth:`PuzzleSolver._read_where_in` tries each split, keeping the one whose
#: subject the store actually knows — the same signal `grounding._read_polar_surface` uses to find
#: where a subject ends, and for the same reason: only the store knows its own subjects.
_WHERE_IN = re.compile(
    r"^what(?:'s| is|\s+are)\s+(?:the\s+)?(?P<outer>[\w ]+?)\s+of\s+(?:the\s+)?"
    r"(?P<middle>.+?)\s+is\s+(?:in|located\s+in|part\s+of)\s*\??$")

_COMMON = re.compile(
    r"^what\s+do(?:es)?\s+(?P<a>.+?)\s+and\s+(?P<b>.+?)\s+have\s+in\s+common\s*\??$")

_ODD = re.compile(
    r"^which\s+(?:one\s+)?(?:of\s+(?:these|them)\s+)?(?:does\s+not\s+belong|is\s+the\s+odd\s+"
    r"one\s+out|is\s+not\s+like\s+the\s+others)\s*[:\-]?\s*(?P<items>.+?)\s*\??$")

_ODD_KIND = re.compile(
    r"^which\s+of\s+(?:these\s+)?(?P<items>.+?)\s+is\s+not\s+an?\s+(?P<kind>[\w ]+?)\s*\??$")

#: The same question with the list at the end — "which of these is not a mammal: tiger, whale, …".
#: Both orders are ordinary English and only one of them was read.
_ODD_KIND_TAIL = re.compile(
    r"^which\s+(?:one\s+)?(?:of\s+(?:these|them|the\s+following)\s+)?is\s+not\s+an?\s+"
    r"(?P<kind>[\w ]+?)\s*[:\-]\s*(?P<items>.+?)\s*\??$")

_CONSTRAINT = re.compile(
    r"^(?:which|name\s+a|name\s+an|what)\s+(?P<kind>[\w ]+?)\s+"
    r"(?:is|are|has|have|can|could|needs?|requires?|contains?|uses?)\s+"
    r"(?P<constraint>.+?)\s*\??$")

_ANALOGY = re.compile(
    r"^(?P<a>.+?)\s+is\s+to\s+(?P<b>.+?)\s+as\s+(?P<c>.+?)\s+is\s+to\s+(?:what|which)\s*\??$")

#: "what is X part of" / "what is X a part of" — `part_of` is stored and, as the corpus builder
#: records, has no question form that parses through the ordinary grammar. It has one here.
_PART_OF = re.compile(
    r"^what\s+(?:is|are)\s+(?P<s>.+?)\s+(?:a\s+)?part\s+of\s*\??$")

#: "what would you use to measure temperature" — `purpose` read backwards. The store says a
#: thermometer's purpose is measuring temperature; nothing says what measures temperature, and the
#: question is asked far more often in that direction.
_FOR_PURPOSE = re.compile(
    r"^what\s+(?:would\s+you\s+use|do\s+you\s+use|is\s+used)\s+(?:to|for)\s+"
    r"(?P<purpose>.+?)\s*\??$")

#: "what is the capital of the country whose currency is the yen" — a bridge whose *start* is
#: unknown and has to be found from a property first. The nested form walks outward from a named
#: subject; this walks inward to one.
_WHOSE = re.compile(
    r"^what(?:'s| is|\s+are)\s+(?:the\s+)?(?P<outer>[\w ]+?)\s+of\s+(?:the\s+)?"
    r"(?P<kind>[\w ]+?)\s+whose\s+(?P<relation>[\w ]+?)\s+is\s+(?P<value>.+?)\s*\??$")

#: "why does rain fall" / "why does X happen" — the causes of a thing, which `grounding` reads
#: only for the exact form "why does X happen". Here the subject may be any stored effect.
_WHY = re.compile(r"^why\s+(?:does|do|is|are)\s+(?P<s>.+?)(?:\s+(?:happen|occur|fall|form))?\s*\??$")

_CHAIN = re.compile(
    r"^what\s+does\s+(?P<s>.+?)\s+(?:eventually|ultimately|in\s+the\s+end)\s+"
    r"(?:cause|lead\s+to|result\s+in)\s*\??$")


# --------------------------------------------------------------------------- #
# Result
# --------------------------------------------------------------------------- #
@dataclass
class Solution:
    """One constructed answer, and the path that constructed it."""

    #: Which form solved it — "bridge", "commonality", … — or "" when nothing did.
    form: str = ""
    #: The answer, or "" for an honest refusal.
    answer: str = ""
    #: Everything that would have been an equally good answer, the given one included.
    answers: Tuple[str, ...] = ()
    #: The edges crossed, as readable lines. **An answer with no steps is never returned.**
    steps: Tuple[str, ...] = ()
    confidence: float = 0.0
    why: str = ""

    @property
    def ok(self) -> bool:
        return bool(self.answer) and bool(self.steps)

    def to_dict(self) -> Dict[str, Any]:
        return {"form": self.form, "answer": self.answer, "answers": list(self.answers),
                "steps": list(self.steps), "confidence": round(self.confidence, 3),
                "why": self.why}


# --------------------------------------------------------------------------- #
# The solver
# --------------------------------------------------------------------------- #
class PuzzleSolver:
    """Constructs answers that are in no single fact, from facts that are.

    Reads the store once at construction. Nothing here writes to the brain, so a solver may be
    built, used and discarded around anything else without disturbing it.
    """

    def __init__(self, brain: Any) -> None:
        self.brain = brain
        self.grounder = getattr(brain, "grounder", None)
        self.core = getattr(brain, "learner", None)
        #: ``(subject, predicate) -> [object, ...]``
        self.by_sp: Dict[Tuple[str, str], List[str]] = collections.defaultdict(list)
        #: ``predicate -> {object -> [subject, ...]}`` — the store read backwards, which is what
        #: every search form here needs and no existing index provides.
        self.inverted: Dict[str, Dict[str, List[str]]] = collections.defaultdict(
            lambda: collections.defaultdict(list))
        self.kinds: Dict[str, List[str]] = collections.defaultdict(list)
        self._index()

    # ---- the store, read once -------------------------------------------- #
    def _index(self) -> None:
        facts = getattr(self.grounder, "facts", None)
        if not isinstance(facts, dict):
            return
        for key, triples in facts.items():
            try:
                subject, predicate = key
            except Exception:  # noqa: BLE001
                continue
            for triple in triples:
                if getattr(triple, "superseded", False):
                    continue
                obj = str(getattr(triple, "object", "") or "").strip()
                if not obj:
                    continue
                self.by_sp[(subject, predicate)].append(obj)
                # Indexed under **both** spellings of the object, and both are needed. A question
                # arrives as free text and is keyed through `_term`, which applies `canon`'s
                # singularising rule; the store wrote the object as it was given. Indexing only
                # one of them means every lookup whose plural folds silently misses — measured, it
                # took the `constraint` paper from 0.993 to 0.800 the moment `_term` started
                # canonicalising, on a store that had not changed at all.
                for spelling in {obj.strip().lower(), self._term(obj)}:
                    if spelling:
                        self.inverted[predicate][spelling].append(subject)
                if predicate == "is_a":
                    self.kinds[subject].append(obj)

    # ---- helpers ----------------------------------------------------------- #
    def _term(self, text: str) -> str:
        """A phrase as the store would key it: hedges off, then `canon`'s own spelling rule.

        The last step is not cosmetic. `canonical_entity` singularises, so "runoff of nutrients"
        is filed under ``runoff of nutrient``; a solver that lowercased and stopped would look up
        a key the store never wrote and report no answer about a fact it holds.
        """
        low = str(text or "").strip().lower().rstrip("?.,;")
        for _ in range(3):
            stripped = _HEDGE.sub("", low).strip()
            if stripped == low:
                break
            low = stripped
        if self.grounder is None:
            return low
        try:
            return self.grounder._key(low) or low
        except Exception:  # noqa: BLE001
            return low

    def _relation(self, word: str) -> str:
        """Fold a noun to the relation it names, or "" if it names none."""
        term = self._term(word).replace(" ", "_")
        if self.grounder is None:
            return ""
        try:
            folded = self.grounder._predicate(term)
        except Exception:  # noqa: BLE001
            return ""
        if not folded:
            return ""
        return folded if any(p == folded for _s, p in self.by_sp) else ""

    @staticmethod
    def _stem(word: str) -> str:
        """Enough of a stem to see that "flying" and "fly" are the same claim.

        Not a stemmer, and not trying to be one. The store writes a capability as "flying" and a
        question asks "can fly"; without this the two never meet, and with anything cleverer the
        matching starts inventing relationships nobody stated.
        """
        for suffix in ("ing", "ies", "es", "ed", "s"):
            if len(word) > len(suffix) + 2 and word.endswith(suffix):
                trimmed = word[: -len(suffix)]
                if suffix == "ies":
                    return trimmed[:-1]
                # English doubles the final consonant before -ing and -ed: "swimming" trims to
                # "swimm", which matches nothing. Measured — "which bird can swim" was answered
                # correctly, then refused, the moment stemming was introduced without this line.
                if (suffix in ("ing", "ed") and len(trimmed) > 2
                        and trimmed[-1] == trimmed[-2] and trimmed[-1] not in "aeiou"):
                    return trimmed[:-1]
                return trimmed
        # A trailing silent "e" is the other half of the same problem: the store writes "measuring
        # temperature" and the question asks "measure temperature", and "measur" against "measure"
        # is one character from matching. Only on words long enough that dropping it cannot make
        # two unrelated short words collide.
        if len(word) > 3 and word.endswith("e"):
            return word[:-1]
        return word

    @classmethod
    def _in_order(cls, haystack: str, needle: str) -> bool:
        """Are ``needle``'s words all in ``haystack``, in order, not necessarily adjacent?

        Weaker than :meth:`_contains` and used in exactly one place — see :meth:`for_purpose` —
        because on identifying values this looseness is what let "the largest ocean" match "the
        second largest ocean". On descriptive purpose phrases it is what lets "measure pressure"
        reach "measuring atmospheric pressure".
        """
        big = [cls._stem(w) for w in haystack.split()]
        small = [cls._stem(w) for w in needle.split()]
        if not small:
            return False
        index = 0
        for word in big:
            if index < len(small) and word == small[index]:
                index += 1
        return index == len(small)

    @classmethod
    def _contains(cls, haystack: str, needle: str) -> bool:
        """Is ``needle`` a **contiguous run of whole words** inside ``haystack``, stems compared?

        Two failures shaped this, in opposite directions.

        Substring containment let a qualifier through: "which ocean is the largest ocean" answered
        *Atlantic*, because the stored property "the second largest ocean" contains the string
        "largest ocean". As whole words it does not.

        And matching the other way round — asking whether a *stored* phrase appears inside the
        *question* — let a short object match a long question about something else entirely:
        "which bird can fly and lives in India" found the stored value ``india`` sitting inside the
        question and answered from it. That direction is gone; a stored property may be longer than
        what the question named, never shorter.
        """
        big = [cls._stem(w) for w in haystack.split()]
        small = [cls._stem(w) for w in needle.split()]
        if not small or len(small) > len(big):
            return False
        return any(big[i:i + len(small)] == small for i in range(len(big) - len(small) + 1))

    def _known(self, term: str) -> bool:
        key = self._term(term)
        return any(s == key for s, _p in self.by_sp)

    def _ancestors(self, subject: str, *, depth: int = 4) -> List[str]:
        seen, out = {subject}, []
        frontier = [self._term(k) for k in self.kinds.get(subject, ())]
        while frontier and depth > 0:
            nxt = []
            for kind in frontier:
                if kind and kind not in seen:
                    seen.add(kind)
                    out.append(kind)
                    nxt.extend(self._term(k) for k in self.kinds.get(kind, ()))
            frontier, depth = nxt, depth - 1
        return out

    # ---- the operations ---------------------------------------------------- #
    def bridge(self, subject: str, shape: Sequence[str]) -> Solution:
        """Walk a sequence of *different* relations and report where it lands.

        The walk itself is :meth:`~nyxara.njp.core.CognitiveLearningCore.walk_shape`, which has
        existed and had no caller that could reach it from a question. What is added here is the
        gold-standard part: **every** object the final hop could reach is reported, not only the
        strongest, because a bridge whose last relation holds several values has several right
        answers and naming one of them is not more correct than naming another.
        """
        out = Solution(form="bridge")
        steps = [str(p).strip().lower() for p in shape if str(p).strip()]
        start = self._term(subject)
        if not steps or not start or len(steps) > MAX_BRIDGE_HOPS or self.core is None:
            return out
        walked = self.core.walk_shape(start, steps)
        if not getattr(walked, "ok", False):
            return out
        # Re-derive the final hop's alternatives from the path the core actually took, so the
        # answer set is the store's and not a second guess at it.
        support = list(getattr(walked, "support", ()) or ())
        if not support:
            return out
        last_subject, last_predicate, _last_object = support[-1]
        alternatives = self.by_sp.get((self._term(last_subject), last_predicate), [])
        out.answer = str(getattr(walked, "answer", "") or "")
        out.answers = tuple(dict.fromkeys(alternatives)) or (out.answer,)
        out.steps = tuple(getattr(walked, "steps", ()) or ())
        out.confidence = float(getattr(walked, "confidence", 0.0) or 0.0)
        out.why = f"walked {' → '.join(steps)} from {start}"
        return out

    def bridge_to_kind(self, subject: str, link: str, kind: str, outer: str) -> Solution:
        """Walk ``link`` until a node that **is a** ``kind``, then read ``outer`` off that node.

        This is the difference between answering the question and answering a shorter one. "The
        capital of the country that Agra is in" names its container, and a walk that simply takes
        the first hop that yields anything returns *Lucknow* — the capital of Uttar Pradesh, which
        is a state. Measured exactly that way before this method existed. The noun in the middle of
        a nested question is not decoration; it says where to stop.
        """
        out = Solution(form="bridge")
        node, want = self._term(subject), self._term(kind)
        if not node or not want or not outer:
            return out
        steps: List[str] = []
        for _hop in range(MAX_BRIDGE_HOPS):
            reached = self.by_sp.get((node, link), [])
            if not reached:
                return out
            nxt = self._term(reached[0])
            steps.append(f"{node} —{link}→ {nxt}")
            node = nxt
            if want in set(self._ancestors(node)) | {node}:
                objects = self.by_sp.get((node, outer), [])
                if not objects:
                    return out
                out.answer = objects[0]
                out.answers = tuple(dict.fromkeys(objects))
                out.steps = tuple(steps + [f"{node} is_a* {want}",
                                           f"{node} —{outer}→ {out.answer}"])
                out.confidence = max(0.3, 0.85 ** (len(steps) + 1))
                out.why = f"walked {link} to the {want}, then read its {outer}"
                return out
        return out

    def commonality(self, first: str, second: str) -> Solution:
        """What two subjects share: a kind, a property, a requirement, a purpose.

        Nothing in the store is about the *pair*. The answer is a set intersection over what each
        of them holds, taken relation by relation so that "both are mammals" and "both need water"
        stay distinguishable rather than collapsing into one bag of strings.

        Inherited kinds count. A whale and a bat share ``mammal`` directly; a sparrow and a tiger
        share ``animal`` only several levels up, and refusing that would be refusing the more
        interesting half of the question.
        """
        out = Solution(form="commonality")
        a, b = self._term(first), self._term(second)
        if not a or not b or a == b or not (self._known(a) and self._known(b)):
            return out
        shared: List[Tuple[str, str]] = []
        for predicate in ("is_a", "has_property", "requires", "capable_of", "purpose",
                          "consists_of", "located_in", "has_part"):
            left = {o.strip().lower(): o for o in self.by_sp.get((a, predicate), ())}
            right = {o.strip().lower(): o for o in self.by_sp.get((b, predicate), ())}
            for key in left.keys() & right.keys():
                shared.append((predicate, left[key]))
        # The kinds each reaches, not only the kinds each was given.
        up_a, up_b = set(self._ancestors(a)), set(self._ancestors(b))
        for kind in sorted(up_a & up_b):
            if not any(p == "is_a" and o.strip().lower() == kind for p, o in shared):
                shared.append(("is_a", kind))
        if not shared:
            return out
        # **Most specific first**, and the first version got this badly wrong by sorting on string
        # length. Measured: "what do a pressure cooker and a spoon have in common" answered
        # *object* over *utensil*, and "a keyboard instrument and a percussion instrument"
        # answered *tool* over *musical instrument*. Both answers are true and both are useless,
        # which is the whole difficulty of this question: everything shares something at the top
        # of the hierarchy, so the interesting answer is the one furthest from it.
        #
        # A shared kind is *less* specific exactly when another shared kind reaches it by an
        # ``is_a`` walk. Ranking on that rather than on a proxy makes "utensil beats object"
        # follow from the taxonomy instead of from a coincidence of spelling.
        kinds_shared = {o.strip().lower() for p, o in shared if p == "is_a"}
        generic = {anc for k in kinds_shared for anc in self._ancestors(k)} & kinds_shared
        shared.sort(key=lambda pair: (pair[0] != "is_a",
                                      pair[1].strip().lower() in generic,
                                      len(pair[1])))
        out.answer = shared[0][1]
        out.answers = tuple(dict.fromkeys(o for _p, o in shared))
        out.steps = tuple(f"{a} —{p}→ {o} ; {b} —{p}→ {o}" for p, o in shared[:4])
        out.confidence = 0.7
        out.why = f"{len(shared)} shared claim(s)"
        return out

    def odd_one_out(self, items: Sequence[str], kind: str = "") -> Solution:
        """The member of a set that does not share what the rest share.

        Two readings, and the difference matters. With a ``kind`` named — "which of these is not a
        bird" — it is a membership test on each item and the answer is whichever fails. Without
        one, the shared property has to be *found* first: the kind the majority reach, and then the
        one that does not. The second is the harder question and the one a quiz actually asks.
        """
        out = Solution(form="odd_one_out")
        terms = [self._term(x) for x in items if self._term(x)]
        if len(terms) < 3:
            return out
        if kind:
            wanted = self._term(kind)
            missing = [t for t in terms if wanted not in set(self._ancestors(t)) | {t}]
            if len(missing) != 1:
                return out
            out.answer = missing[0]
            out.answers = (missing[0],)
            out.steps = tuple(f"{t} is_a* {wanted}" for t in terms if t != missing[0]) + (
                f"{missing[0]} reaches no {wanted}",)
            out.confidence = 0.75
            out.why = f"the only one that is not a {wanted}"
            return out

        reach = {t: set(self._ancestors(t)) | {t} for t in terms}
        tally: Dict[str, int] = collections.Counter()
        for kinds in reach.values():
            for k in kinds:
                tally[k] += 1
        # A kind held by all but exactly one is what makes this question well posed at all.
        candidates = [k for k, n in tally.items() if n == len(terms) - 1]
        if not candidates:
            return out
        # The most specific such kind: shared by the fewest things overall is the most informative.
        candidates.sort(key=lambda k: (-len(k), k))
        for shared_kind in candidates:
            missing = [t for t in terms if shared_kind not in reach[t]]
            if len(missing) == 1:
                out.answer = missing[0]
                out.answers = (missing[0],)
                out.steps = tuple(f"{t} reaches {shared_kind}" for t in terms if t != missing[0])
                out.steps += (f"{missing[0]} does not",)
                out.confidence = 0.7
                out.why = f"the others are all {shared_kind}"
                return out
        return out

    def constraint(self, kind: str, predicate: str, value: str) -> Solution:
        """A member of ``kind`` for which ``predicate`` holds ``value``.

        The store can list what is a metal and can list what is liquid; nothing lists metals that
        are liquid, and the answer is the intersection. Members are taken through the *reached*
        kinds rather than the stated ones, so "which mammal can fly" finds a bat even though the
        corpus files it under ``mammal`` and the flying under ``capable_of``.
        """
        out = Solution(form="constraint")
        want_kind, want_value = self._term(kind), self._term(value)
        if not want_kind or not want_value or not predicate:
            return out
        # Exact first, then a looser match on the value — and the widening is driven by whether a
        # member of the *kind* was found, not by whether any holder was. That distinction is the
        # whole of a measured bug: "which mammal can fly" matched ``fly`` exactly and found `bird`
        # and `aircraft`, neither of which is a mammal, so the search reported no answer while
        # `bat capable_of flying` sat one character away in the same index.
        exact = list(self.inverted.get(predicate, {}).get(want_value, ()))
        hits = [s for s in sorted(set(exact)) if want_kind in set(self._ancestors(s)) | {s}]
        if not hits:
            loose: Set[str] = set(exact)
            for obj, subjects in self.inverted.get(predicate, {}).items():
                # One direction only: the stored property may say more than the question asked
                # ("liquid" against "liquid at room temperature"), never less. See :meth:`_contains`.
                if self._contains(obj, want_value):
                    loose.update(subjects)
            hits = [s for s in sorted(loose) if want_kind in set(self._ancestors(s)) | {s}]
        if not hits:
            return out
        out.answer = hits[0]
        out.answers = tuple(hits)
        out.steps = tuple(f"{s} is_a* {want_kind} ; {s} —{predicate}→ {want_value}"
                          for s in hits[:4])
        out.confidence = 0.75
        out.why = f"{len(hits)} member(s) of {want_kind} with that property"
        return out

    def analogy(self, first: str, second: str, third: str) -> Solution:
        """``a`` is to ``b`` as ``c`` is to what — the relation is not named and has to be found.

        Two steps, and both can fail honestly. Find every relation under which ``a`` holds ``b``;
        then apply each to ``c``, nearest relation first. Where ``a`` reaches ``b`` only by an
        inheritance walk rather than a stated edge, the walk is used, because "sparrow is to
        animal" is a perfectly good left-hand side.
        """
        out = Solution(form="analogy")
        a, b, c = self._term(first), self._term(second), self._term(third)
        if not (a and b and c) or not (self._known(a) and self._known(c)):
            return out
        relations = [p for (s, p), objs in self.by_sp.items()
                     if s == a and any(o.strip().lower() == b for o in objs)]
        if not relations and b in set(self._ancestors(a)):
            relations = ["is_a"]
        for predicate in relations:
            direct = self.by_sp.get((c, predicate), [])
            if direct:
                out.answer = direct[0]
                out.answers = tuple(dict.fromkeys(direct))
                out.steps = (f"{a} —{predicate}→ {b}", f"so {c} —{predicate}→ {out.answer}")
                out.confidence = 0.7
                out.why = f"the relation between {a} and {b} is {predicate}"
                return out
            if predicate == "is_a":
                up = self._ancestors(c)
                if up:
                    out.answer = up[0]
                    out.answers = tuple(up[:4])
                    out.steps = (f"{a} —is_a*→ {b}", f"so {c} —is_a*→ {out.answer}")
                    out.confidence = 0.6
                    out.why = "the relation is membership of a kind"
                    return out
        return out

    def chain(self, subject: str, predicate: str = "causes") -> Solution:
        """Where a transitive relation *ends up*, not where its first hop goes.

        :meth:`~nyxara.njp.core.CognitiveLearningCore.reach` does the walk. What this adds is the
        refusal: a `reach` that found only one hop has answered the ordinary question, not this
        one, so it is reported as no solution rather than as a shallow success.
        """
        out = Solution(form="chain")
        start = self._term(subject)
        if not start or self.core is None:
            return out
        walked = self.core.reach(start, predicate)
        if not getattr(walked, "ok", False):
            return out
        support = list(getattr(walked, "support", ()) or ())
        if len(support) < 2:
            return out                       # one hop is a lookup, not a chain
        out.answer = str(getattr(walked, "answer", "") or "")
        out.answers = (out.answer,)
        out.steps = tuple(getattr(walked, "steps", ()) or ())
        out.confidence = float(getattr(walked, "confidence", 0.0) or 0.0)
        out.why = f"{len(support)} hops along {predicate}"
        return out

    # ---- reading a question ------------------------------------------------- #
    def solve(self, question: str) -> Solution:
        """Route an English question to the operation that can construct its answer.

        Returns an empty :class:`Solution` when no form reads — a refusal, counted as silence by
        the exam. A solver that answers every string is a solver whose score means nothing.
        """
        low = " ".join(str(question or "").strip().lower().split())
        if not low:
            return Solution()
        for reader in (self._read_nested, self._read_whose, self._read_where_in,
                       self._read_common, self._read_odd, self._read_constraint,
                       self._read_analogy, self._read_chain, self._read_part_of,
                       self._read_for_purpose, self._read_why):
            try:
                got = reader(low)
            except Exception:  # noqa: BLE001
                got = None
            if got is not None and got.ok:
                return got
        return Solution()

    def _read_nested(self, low: str) -> Optional[Solution]:
        match = _NESTED.match(low)
        if match is None:
            return None
        outer = self._relation(match.group("outer"))
        inner = self._relation(match.group("inner"))
        if not outer:
            return None
        subject = self._term(match.group("subject"))
        if inner:
            return self.bridge(subject, (inner, outer))
        # "the currency of the country where X is" — the middle noun names a *kind* rather than a
        # relation, so it says where the walk should stop. How many containers lie between a
        # monument and its country is a property of the corpus and the question says none of it:
        # the Taj Mahal is in Agra, in Uttar Pradesh, in India.
        for link in BRIDGE_LINKS:
            got = self.bridge_to_kind(subject, link, match.group("inner"), outer)
            if got.ok:
                return got
        # No node on the walk is that kind. Fall back to depth, which answers a slightly different
        # question and says so in its steps rather than silently.
        for depth in range(1, MAX_BRIDGE_HOPS):
            for link in BRIDGE_LINKS:
                got = self.bridge(subject, (link,) * depth + (outer,))
                if got.ok:
                    return got
        return None

    def _read_where_in(self, low: str) -> Optional[Solution]:
        match = _WHERE_IN.match(low)
        if match is None:
            return None
        outer = self._relation(match.group("outer"))
        if not outer:
            return None
        # Try every way of cutting "<container> that <subject>" and keep the one the store
        # recognises. Longest subject first, so "west bengal" beats "bengal" where both are known.
        words = re.sub(r"\bthat\b", " ", match.group("middle")).split()
        splits: List[Tuple[str, str]] = []
        for cut in range(1, len(words)):
            container, subject = " ".join(words[:cut]), " ".join(words[cut:])
            if self._known(subject):
                splits.append((container, subject))
        splits.sort(key=lambda pair: -len(pair[1]))
        for container, subject in splits:
            for link in BRIDGE_LINKS:
                got = self.bridge_to_kind(self._term(subject), link, container, outer)
                if got.ok:
                    return got
        for _container, subject in splits:
            for link in BRIDGE_LINKS:
                for depth in (1, 2):
                    got = self.bridge(self._term(subject), (link,) * depth + (outer,))
                    if got.ok:
                        return got
        return None

    def _read_common(self, low: str) -> Optional[Solution]:
        match = _COMMON.match(low)
        return None if match is None else self.commonality(match.group("a"), match.group("b"))

    @staticmethod
    def _items(raw: str) -> List[str]:
        """Split a list, and prefer the separator that cannot be part of an item.

        "crime and punishment, don quixote, great expectation, fixed cost" is four items, and
        splitting on " and " as well made it five — two of which were half a novel. A comma is
        unambiguous where "and" is not, so where there are commas they alone are the separator.
        """
        if "," in raw:
            return [p.strip() for p in raw.split(",") if p.strip()]
        return [p.strip() for p in re.split(r"\s+or\s+|\s+and\s+", raw) if p.strip()]

    def _read_odd(self, low: str) -> Optional[Solution]:
        match = _ODD_KIND_TAIL.match(low)
        if match is not None:
            return self.odd_one_out(self._items(match.group("items")), match.group("kind"))
        match = _ODD_KIND.match(low)
        if match is not None:
            return self.odd_one_out(self._items(match.group("items")), match.group("kind"))
        match = _ODD.match(low)
        if match is None:
            return None
        return self.odd_one_out(self._items(match.group("items")))

    def _read_constraint(self, low: str) -> Optional[Solution]:
        match = _CONSTRAINT.match(low)
        if match is None:
            return None
        kind = match.group("kind")
        rest = match.group("constraint")
        # "which metal is liquid" is `has_property`; "which mammal can fly" is `capable_of`;
        # "which bird needs water" is `requires`. The verb decides where it can, the store where
        # it cannot — the same "try both and let the evidence choose" the polar reader uses.
        head, _, tail = rest.partition(" ")
        candidates: List[Tuple[str, str]] = []
        folded = self._relation(head)
        if folded and tail:
            candidates.append((folded, tail))
        candidates.extend((p, rest) for p in ("has_property", "capable_of", "requires",
                                              "purpose", "consists_of", "located_in"))
        for predicate, value in candidates:
            got = self.constraint(kind, predicate, value)
            if got.ok:
                return got
        return None

    def _read_analogy(self, low: str) -> Optional[Solution]:
        match = _ANALOGY.match(low)
        if match is None:
            return None
        return self.analogy(match.group("a"), match.group("b"), match.group("c"))

    def _read_chain(self, low: str) -> Optional[Solution]:
        match = _CHAIN.match(low)
        return None if match is None else self.chain(match.group("s"))

    def part_of(self, subject: str) -> Solution:
        """What a thing belongs to — stated, or reached through what it is stated to be part of.

        ``part_of`` carries the highest transitivity prior in :mod:`nyxara.njp.core` and, as
        ``prepare_knowledge_corpus`` records, no question form that parses: "what is a wheel part
        of" reads as ``('wheel part of', 'is_a')`` through the ordinary grammar. So the relation
        was written into the graph deliberately and could never be asked for. It can be now.
        """
        out = Solution(form="part_of")
        node = self._term(subject)
        if not node:
            return out
        for predicate in ("part_of", "located_in"):
            held = self.by_sp.get((node, predicate), [])
            if held:
                out.answer = held[0]
                out.answers = tuple(dict.fromkeys(held))
                out.steps = (f"{node} —{predicate}→ {out.answer}",)
                out.confidence = 0.8
                out.why = f"stated {predicate}"
                return out
        # Not stated of the thing itself: inherited from what it is. A prokaryotic cell is a cell,
        # and a cell is part of an organism.
        for kind in self._ancestors(node):
            for predicate in ("part_of", "located_in"):
                held = self.by_sp.get((kind, predicate), [])
                if held:
                    out.answer = held[0]
                    out.answers = tuple(dict.fromkeys(held))
                    out.steps = (f"{node} is_a* {kind}", f"{kind} —{predicate}→ {held[0]}")
                    out.confidence = 0.55
                    out.why = f"inherited {predicate} from {kind}"
                    return out
        return out

    def for_purpose(self, purpose: str) -> Solution:
        """What is used *for* something — ``purpose`` read backwards.

        The store says a thermometer's purpose is measuring temperature. Nothing says what measures
        temperature, and that is the direction the question is nearly always asked in. Exactly the
        read/write asymmetry ``_CAUSE_OF`` exists to close for causation, one relation over.
        """
        out = Solution(form="for_purpose")
        want = self._term(purpose)
        if not want:
            return out
        hits: List[Tuple[str, str]] = []
        for obj, subjects in self.inverted.get("purpose", {}).items():
            if self._contains(obj, want):
                hits.extend((s, obj) for s in subjects)
        if not hits:
            # A stated purpose is a descriptive phrase and a question names only part of it: the
            # store says "measuring atmospheric pressure" and the question asks "measure pressure",
            # which is not a contiguous run. So the words are required in order but not adjacent —
            # loosened *only here*, where the phrases are descriptions rather than the identifying
            # values a `constraint` search compares, and where the alternative is refusing a
            # question whose answer she plainly holds.
            for obj, subjects in self.inverted.get("purpose", {}).items():
                if self._in_order(obj, want):
                    hits.extend((s, obj) for s in subjects)
        if not hits:
            return out
        hits.sort(key=lambda pair: (len(pair[1]), pair[0]))
        out.answer = hits[0][0]
        out.answers = tuple(dict.fromkeys(s for s, _o in hits))
        out.steps = tuple(f"{s} —purpose→ {o}" for s, o in hits[:4])
        out.confidence = 0.7
        out.why = f"{len(hits)} thing(s) with that purpose"
        return out

    def whose(self, kind: str, relation: str, value: str, outer: str) -> Solution:
        """Find a subject by one of its properties, then read another off it.

        "The capital of the country whose currency is the yen" is a bridge walked inward: the
        subject is not named, it is *described*, and finding it is the first hop.
        """
        out = Solution(form="whose")
        found = self.constraint(kind, relation, value)
        if not found.ok:
            return out
        objects = self.by_sp.get((self._term(found.answer), outer), [])
        if not objects:
            return out
        out.answer = objects[0]
        out.answers = tuple(dict.fromkeys(objects))
        out.steps = found.steps + (f"{found.answer} —{outer}→ {out.answer}",)
        out.confidence = 0.65
        out.why = f"found {found.answer} by its {relation}, then read its {outer}"
        return out

    def _read_part_of(self, low: str) -> Optional[Solution]:
        match = _PART_OF.match(low)
        return None if match is None else self.part_of(match.group("s"))

    def _read_for_purpose(self, low: str) -> Optional[Solution]:
        match = _FOR_PURPOSE.match(low)
        return None if match is None else self.for_purpose(match.group("purpose"))

    def _read_whose(self, low: str) -> Optional[Solution]:
        match = _WHOSE.match(low)
        if match is None:
            return None
        outer = self._relation(match.group("outer"))
        relation = self._relation(match.group("relation"))
        if not outer or not relation:
            return None
        return self.whose(match.group("kind"), relation, match.group("value"), outer)

    def _read_why(self, low: str) -> Optional[Solution]:
        """"Why does X happen" — the causes of X, which is `causes` read from the far end."""
        match = _WHY.match(low)
        if match is None:
            return None
        effect = self._term(match.group("s"))
        causes = sorted({s for s, objects in self.inverted.get("causes", {}).items()
                         for s in ([] if s != effect else objects)})
        if not causes:
            return None
        out = Solution(form="why")
        out.answer = causes[0]
        out.answers = tuple(causes)
        out.steps = tuple(f"{c} —causes→ {effect}" for c in causes[:4])
        out.confidence = 0.7
        out.why = f"{len(causes)} stated cause(s)"
        return out


def solver(brain: Any) -> PuzzleSolver:
    """The one call a caller needs."""
    return PuzzleSolver(brain)
