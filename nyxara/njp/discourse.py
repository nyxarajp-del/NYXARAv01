"""NYXARA · njp/discourse.py — what was meant, and by whom (🗣, NJP V.26).

:mod:`nyxara.njp.semantics` reads a sentence into a :class:`~nyxara.njp.semantics.Meaning`.
:mod:`nyxara.njp.language` learns the grammar that reading needs. Both stop at the sentence, and
that boundary is where this package's language work has ended since V.09: a turn is compiled,
grounded, and forgotten as an utterance. What is missing is everything that makes a sentence a
*communication* — who said it, what they were doing by saying it, what "it" refers to, whether it
agrees with what they said twenty turns ago, and what they believe that is not so.

**Measured, before this module existed.** Eight conversations put to ``NJPBrain.think()``:

* ``"I never visited Delhi."`` → **``noted: Master visit delhi``**. The compiler read
  ``negated=True`` and the grounder stored ``negated=True``; the line that tells the Master what
  was understood dropped the polarity, so the only window she has into her own uptake showed the
  **opposite of what was said**. This is the ``"Zorbins don't"`` defect of V.09 surviving in the
  one place nobody measured: not in the parse, in the confirmation.
* ``"Can you open the window?"`` → ``0 beliefs held with evidence, 0 open or unsupported``. Not a
  request, not a question, not readable at all.
* ``"Ravi met Arun."`` / ``"He was tired."`` → the second turn came back with the **first turn
  handed back** at confidence 0.75. No referent was resolved and none was refused.
* ``"I never visited Delhi."`` then ``"When I visited Delhi last year…"`` → the denial and its
  flat contradiction were both filed, in silence, and ``"Have I visited Delhi?"`` then answered
  with nothing at all.
* ``"The market swallowed the shock."`` → ``noted: market swallow shock``, filed into the
  knowledge store as a fact about the world at the same confidence as a fact she was taught.

Six organs, one discipline, and it is the discipline of :mod:`nyxara.njp.language` rather than a
new one: **a lesson leaves a shape, never an answer; a shape is kept only once it has read
something nobody demonstrated; where two readings survive the answer is that there are two.**

**1 · Speech acts are induced, not listed** (:class:`ActLearner`). *"Can you open the window?"* is
literally a question about ability and actually a request, and no amount of parsing the sentence
finds that — it is a convention, so it has to be learned from being shown. What a demonstration
leaves is the **tag skeleton** of the surface at four levels of abstraction (:func:`shapes_of`),
with the open class collapsed to positions, so a demonstration of *"Can you open the window?"*
reads *"Would you send the file?"* — a sentence with no word in common with it. Two
demonstrations at minimum and **they must differ in a filler**, or the shape is one sentence with
a rule's name on it. A shape two demonstrations disagree about is **contested** and stops being
read, which is the correct outcome for *"Can you swim?"*: that really is an ability question, and
a module that answered confidently either way would be inventing a convention rather than
learning one.

**And a misreading teaches the shape, not the sentence** (:meth:`ActLearner.misread`). That is
the mechanism the Master specified as *communication failure → learning*, and it is the reason
the levels exist: correcting *"Can you pass the salt?"* generalises to *"Could you close the
door?"* because what the correction touched was ``MODAL PRON … DET …?`` and not six words.

**2 · A pronoun is resolved or it is refused** (:class:`Reference`). Candidates are scored on
recency and on role parallelism, filtered by number agreement and by the one syntactic constraint
that is genuinely closed-class — a pronoun in object position cannot name the subject of its own
clause. When the top two scores are within :attr:`Reference.margin` the reading is **ambiguous**
and comes back with both, ranked, and the question that would settle it. *"Ravi met Arun. He was
tired."* is ambiguous and says so; *"Ravi met Arun. Arun gave him the book."* is not, and the
thing that separates them is structure rather than a guess about who is likelier to be tired.

**3 · A contradiction is not an update** (:class:`Ledger`). The distinction is the whole subject.
*"The key is in the drawer"* then *"The key is now on the table"* is a world that moved; *"I never
visited Delhi"* then *"when I visited Delhi last year"* is a speaker who cannot be right twice,
and a store that silently overwrites the first with the second has destroyed the evidence that
anything was ever wrong. What separates them is not the topic: it is a **change licence** — a
closed set of markers by which a speaker signals that time has moved — and a **universal**, which
is what ``never`` and ``always`` are. No later instance licenses away a universal, so the second
Delhi sentence is a contradiction whatever temporal it carries. Both claims are kept, both are
marked contested, and the question that would settle it is produced instead of an answer.

**Which relations are stateful is learned rather than declared.** A relation a speaker has used a
change marker with is one where later differs from wrong, and that is evidence sitting in the
transcript. Before any such evidence, a second object for a relation is **new information**, not a
contradiction — the honest reading of a relation nobody has shown her the behaviour of.

**4 · Bounded recursive belief, driven by a sentence** (:class:`Minds`).
:mod:`nyxara.social.tom` has been a real recursive belief–desire–intention engine for many
versions and **nothing in this package has ever put a sentence into it**: it had to be driven by
hand. Here a sentence drives it. Attitude verbs are a closed class in the sense modals are, so
they are listed and it is honest to list them, and the clause after one is read recursively —
*"Ravi thinks I don't know the box has the red ball"* is depth two, with the inner level carrying
attributed **ignorance** rather than a negated belief, because those are different states and
collapsing them is how a false-belief test gets passed for the wrong reason. Depth is bounded at
:attr:`Minds.max_depth`, and a level whose content is identical to the level below it is not
stored at all — which is the Master's *stop when additional depth has negligible value*, made
mechanical.

**5 · She says as much as the hearer needs, and never more than she holds** (:class:`Register`).
Four audiences, four surfaces, one meaning — and the surface is **parsed back** before it is
returned, exactly as :meth:`nyxara.njp.language.Grammar.say` does: unless what comes back carries
the same subject, relation, object and polarity, the rendering is thrown away. What varies across
registers is *how much of what she actually holds is said* — the bare claim, then its condition,
then its time and modality, then what she is entitled to claim it on. **This is not vocabulary
simplification and it cannot be**: she has no dictionary and no thesaurus, and a module that
claimed to make words simpler would be claiming a resource that does not exist in this package.

**6 · A sentence can be true without being literal** (:class:`Figure`). *"The market swallowed the
shock"* is not a fact about a market's digestion, and the only honest way to know that is to have
seen what has swallowed things before. So the judgement is made **from the store's own
evidence**: where a relation's witnessed subjects all share a kind the new subject demonstrably
lacks, the reading is flagged figurative and nothing is filed. Where there are no witnesses, or
the new subject's kinds are unknown, **nothing is claimed** — with an empty store no sentence is
figurative, and that is the correct answer rather than a limitation to apologise for.

**7 · And where one sentence has two readings, it has two** (:func:`attachment`). *"I saw the man
with the telescope."* The evidence that there is an ambiguity at all is in
:mod:`nyxara.njp.semantics`'s own output: it has no frame for a prepositional phrase after an
object, so it fuses the two noun phrases into one object string — ``man telescope``, with ``with
the`` silently dropped. Read that fusion back out and both attachment sites are there, without a
parser this package does not have. They come back at **0.50 each**, with the question, and nothing
picks a winner. Narrow on purpose: **one** preposition, because a module that flagged every
trailing prepositional phrase would turn abstention into a tic.

**What is NOT claimed.** There is no discourse parser here and no rhetorical structure theory. A
referent is one token, inheriting :data:`nyxara.njp.language.MAX_SPAN`'s problem. Attitude verbs
are a list, and a sentence that embeds a belief without one is read as a flat assertion. The
ledger keys on ``subject``/``relation`` as the compiler produced them, so two wordings of one
relation that the grounder's alias table would unify are two relations here. Nothing in this
module decides truth: it reports that two claims cannot both hold and stops.

Pure standard library. No LLM, no corpus on disk, no network. Every public entry point is
fail-soft: on any internal error it degrades to a null result rather than breaking a turn.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import (Any, Callable, Dict, List, Optional, Sequence, Set, Tuple)

from nyxara.njp.semantics import (Meaning, Tag, compile_meaning,  # noqa: F401
                                  tag_tokens)
from nyxara.njp.semantics import _lemma as _lemma_of

__all__ = [
    "GAP", "MORE", "LEVELS", "REGISTERS", "Readings", "attachment",
    "shapes_of", "fillers_of", "literal_act",
    "Act", "Interpretation", "ActLearner",
    "Referent", "Resolution", "Reference",
    "Claim", "Verdict", "Ledger",
    "Minds", "Said", "Register", "Figurative", "Figure",
    "Uptake", "Communicator",
]

#: One or more open-class tokens, in the shape that does not care how many.
GAP = "…"

#: The rest of the sentence, whatever it is. Only :data:`LEVELS`' loosest shape uses it.
MORE = "+"

#: The four abstraction levels a demonstration is recorded at, **most specific first**. A reading
#: uses the first level that has support, which is why the order is the mechanism rather than a
#: preference: a convention demonstrated for ``can`` specifically beats one generalised over every
#: modal, and a shape nobody has demonstrated at any level is read as its literal mood.
LEVELS: Tuple[str, ...] = ("form", "tags", "frame", "opening")

#: Who a rendering is for, in increasing order of how much is said. Not a scale of intelligence —
#: a scale of *how much of what she holds the hearer has asked to carry*.
REGISTERS: Tuple[str, ...] = ("child", "student", "engineer", "expert")

#: The speaker, the hearer, and the third parties. Read off the closed class, so this is a table
#: of pronouns rather than of people.
_SPEAKER: frozenset = frozenset((
    "i", "me", "my", "mine", "myself", "main", "mera", "meri", "mujhe", "मैं",
    "मेरा"))
_ADDRESSEE: frozenset = frozenset((
    "you", "your", "yours", "yourself", "tum", "aap", "tumhara", "tera", "tujhe",
    "तुम", "आप"))
#: Third person, singular. ``it`` is here and carries no gender, which is right: this module has
#: no ontology and cannot know that a drawer is not a person.
_THIRD_SG: frozenset = frozenset((
    "he", "him", "his", "she", "her", "hers", "it", "its", "uska", "uski", "usko", "usne",
    "उसका"))
_THIRD_PL: frozenset = frozenset((
    "they", "them", "their", "theirs", "unka", "unki", "unhone", "unko"))

#: The markers by which a speaker signals that time has moved — the **change licence**. Closed,
#: small, and the single thing that separates :class:`Ledger`'s ``updates`` from its
#: ``contradicts``. A bare re-statement with a different object carries none of these and is
#: therefore not licensed to overwrite anything.
_CHANGE: frozenset = frozenset((
    "now", "currently", "anymore", "ab", "abhi", "longer", "since", "moved", "became",
    "अब"))

#: Negations and adverbs that quantify over **all** times. A universal cannot be updated away by
#: a later instance: *"I never visited Delhi"* and *"I visited Delhi last year"* are not a world
#: that changed, they are a speaker who is wrong once.
_UNIVERSAL_NEG: frozenset = frozenset(("never", "कभी"))
_UNIVERSAL_POS: frozenset = frozenset(("always", "hamesha", "हमेशा"))

#: Attitude verbs. A closed class in the sense modals are — a few dozen words in any language,
#: changing over centuries — and listing them is the same move :mod:`nyxara.njp.semantics` makes
#: for determiners and auxiliaries, not the open-verb list that module exists to avoid.
_MENTAL: Dict[str, str] = {}


def _mental(forms: Sequence[str], strength: str) -> None:
    """Load one attitude verb's forms, exactly as :mod:`nyxara.njp.semantics` loads a tag."""
    for form in forms:
        _MENTAL[form] = strength


_mental(("think", "thinks", "thought", "thinking"), "believe")
_mental(("believe", "believes", "believed"), "believe")
_mental(("assume", "assumes", "assumed"), "believe")
_mental(("suppose", "supposes", "supposed"), "believe")
_mental(("expect", "expects", "expected"), "believe")
_mental(("reckon", "reckons", "reckoned"), "believe")
_mental(("know", "knows", "knew"), "know")
_mental(("realise", "realises", "realised", "realize", "realizes", "realized"), "know")
_mental(("sochta", "sochti", "sochte", "maanta", "maanti", "maante"), "believe")
_mental(("jaanta", "jaanti", "jaante", "pata"), "know")
_mental(("सोचता", "मानता"), "believe")
_mental(("जानता",), "know")

#: The value stored for *attributed ignorance* — "Ravi thinks I do not know". It is not the same
#: object as a belief whose content is false, and storing it as one is how a false-belief test
#: gets passed for the wrong reason.
UNKNOWN = "∅"


# --------------------------------------------------------------------------- #
# shapes — what a demonstration leaves behind
# --------------------------------------------------------------------------- #

def _terminal(surface: str) -> str:
    """``?``, ``!`` or ``.`` — the mood the punctuation carries.

    Kept as a shape element because :func:`~nyxara.njp.semantics.tag_tokens` discards
    punctuation, and *"you can open the window"* and *"can you open the window?"* would otherwise
    differ only in an order this module is deliberately abstracting over.
    """
    text = str(surface or "").strip()
    if text.endswith("?"):
        return "?"
    if text.endswith("!"):
        return "!"
    return "."


def shapes_of(surface: str) -> Tuple[Tuple[str, ...], ...]:
    """The four skeletons one surface leaves, most specific first — see :data:`LEVELS`.

    ``form`` keeps the closed-class **words** and collapses the open class to positions, so it
    generalises over content and not over the convention's own vocabulary. ``tags`` replaces those
    words with their tags, which is what lets ``could`` and ``would`` inherit what ``can`` was
    demonstrated with. ``frame`` collapses each run of open-class tokens to one :data:`GAP`, so a
    two-word object stops mattering. ``opening`` is the first two tags and then anything — the
    over-general shape, kept deliberately, because it is the one *"Can you swim?"* and *"Can you
    open the window?"* share, and a level that gets contested is telling you something true.
    """
    try:
        tokens = tag_tokens(surface)
        if not tokens:
            return ()
        end = _terminal(surface)
        form = tuple([t.text if t.tag != Tag.WORD else Tag.WORD for t in tokens] + [end])
        tags = tuple([t.tag for t in tokens] + [end])
        frame: List[str] = []
        for token in tokens:
            if token.tag == Tag.WORD:
                if frame and frame[-1] == GAP:
                    continue
                frame.append(GAP)
            else:
                frame.append(token.tag)
        opening = tuple([t.tag for t in tokens[:2]] + [MORE, end])
        out: List[Tuple[str, ...]] = []
        for shape in (form, tags, tuple(frame + [end]), opening):
            out.append(shape)
        return tuple(out)
    except Exception:  # noqa: BLE001
        return ()


def fillers_of(surface: str) -> Tuple[str, ...]:
    """The open-class tokens, in order. Two demonstrations that share these are one demonstration
    said twice, and a shape supported only by those has been shown a sentence rather than a
    position."""
    try:
        return tuple(t.text for t in tag_tokens(surface) if t.tag == Tag.WORD)
    except Exception:  # noqa: BLE001
        return ()


def literal_act(surface: str, meaning: Optional[Meaning] = None) -> str:
    """The surface's **mood**, and nothing more.

    This is the half of an utterance that is in the sentence. ``ability-question`` and
    ``willingness-question`` are named separately from ``polar-question`` because they are what an
    indirect request is literally *asking*, and a report that called both "question" could not
    show the gap this module exists to close.
    """
    try:
        tokens = tag_tokens(surface)
        if not tokens:
            return ""
        tags = [t.tag for t in tokens]
        words = [t.text for t in tokens]
        end = _terminal(surface)
        interrogative = end == "?" or tags[0] in (Tag.WH, Tag.AUX, Tag.MODAL)
        if interrogative:
            if (tags[0] == Tag.MODAL and len(tags) > 1 and tags[1] == Tag.PRON
                    and words[1] in _ADDRESSEE):
                if words[0] in ("can", "could", "sakta", "sakte", "sakti"):
                    return "ability-question"
                return "willingness-question"
            if Tag.WH in tags:
                return "question"
            return "polar-question"
        if meaning is not None and meaning.kind == "command":
            return "command"
        # An imperative, read off position rather than off a verb list: the sentence opens on the
        # open class (after an optional adverb) and the next token begins a noun phrase, so there
        # is no subject in front of the verb. *"Open the window."* and *"Water the plants."* match;
        # *"Dogs eat meat."* and *"London is the capital."* do not, because their second token is
        # open class and an auxiliary respectively. The compiler reads most imperatives as
        # ``unreadable``, so without this the whole mood was invisible here.
        head = 1 if (tags and tags[0] == Tag.ADV) else 0
        if (len(tags) > head + 1 and tags[head] == Tag.WORD
                and tags[head + 1] in (Tag.DET, Tag.PRON)):
            return "command"
        if end == "!":
            return "exclamation"
        return "assertion"
    except Exception:  # noqa: BLE001
        return ""


# --------------------------------------------------------------------------- #
# 1 · speech acts, induced
# --------------------------------------------------------------------------- #

@dataclass
class Act:
    """One kept convention: a shape, the act it carries, and the demonstrations behind it."""

    shape: Tuple[str, ...] = ()
    level: str = ""
    act: str = ""
    #: The filler tuples of the demonstrations that produced it. Its *size as a set* is the
    #: support, so the same sentence shown twice supports nothing.
    demonstrations: Tuple[Tuple[str, ...], ...] = ()

    @property
    def support(self) -> int:
        return len(set(self.demonstrations))

    def to_dict(self) -> Dict[str, Any]:
        return {"shape": list(self.shape), "level": self.level, "act": self.act,
                "support": self.support}


@dataclass
class Interpretation:
    """What was said and what was done by saying it, with the evidence for the second."""

    surface: str = ""
    literal: str = ""
    intended: str = ""
    confidence: float = 0.0
    level: str = ""
    support: int = 0
    ambiguous: bool = False
    #: The competing acts, when a shape has been demonstrated two ways. Ranked is the wrong word
    #: for it: they are demonstrated equally and this module refuses to break the tie.
    alternatives: Tuple[str, ...] = ()
    why: str = ""

    @property
    def indirect(self) -> bool:
        """Did the convention say something the sentence's own mood did not?"""
        return bool(self.intended and self.literal and self.intended != self.literal)

    def to_dict(self) -> Dict[str, Any]:
        out = {"literal": self.literal, "intended": self.intended,
               "confidence": round(self.confidence, 4), "level": self.level,
               "support": self.support, "indirect": self.indirect}
        if self.ambiguous:
            out["ambiguous"] = True
            out["alternatives"] = list(self.alternatives)
        if self.why:
            out["why"] = self.why
        return out


class ActLearner:
    """Indirect speech acts, generalised from demonstrations at four levels of abstraction.

    Nothing here knows that *"can you"* introduces a request. What it knows is what it was shown,
    and the only reason it reads a sentence it was never shown is that what a demonstration leaves
    is a **shape** — see :func:`shapes_of`.
    """

    #: Two, and they must differ in a filler. One demonstration is a sentence; two identical ones
    #: are the same sentence counted twice.
    min_support = 2

    def __init__(self, *, min_support: Optional[int] = None) -> None:
        if min_support is not None:
            self.min_support = max(1, int(min_support))
        #: ``(level, shape) → act → [filler tuple, …]``, everything ever demonstrated.
        self.seen: Dict[Tuple[str, Tuple[str, ...]], Dict[str, List[Tuple[str, ...]]]] = {}
        #: The surfaces, kept so a kept shape can be re-run over the whole corpus.
        self.corpus: List[Tuple[str, str]] = []
        self.kept: Dict[Tuple[str, Tuple[str, ...]], Act] = {}
        self.contested: Set[Tuple[str, Tuple[str, ...]]] = set()
        self.corrections = 0
        self.generalised = 0

    # -- being taught -------------------------------------------------------- #
    def show(self, surface: str, act: str) -> None:
        """One demonstration: this surface, doing this. Recorded at every level at once."""
        try:
            act = str(act or "").strip().lower()
            if not act:
                return
            fillers = fillers_of(surface)
            for level, shape in zip(LEVELS, shapes_of(surface)):
                bucket = self.seen.setdefault((level, shape), {})
                bucket.setdefault(act, []).append(fillers)
            self.corpus.append((str(surface), act))
            self._settle()
        except Exception:  # noqa: BLE001
            return

    def misread(self, surface: str, *, took_as: str, meant: str) -> bool:
        """A communication that failed, handed back as a lesson.

        The Master's mechanism, and the reason this is not :meth:`show` under another name is the
        return value: it says whether the correction **generalised** — whether what was learned
        was a shape that reads sentences other than this one, or merely this one. A correction
        that only ever fixes its own sentence is a patch, and this reports the difference rather
        than assuming it.
        """
        try:
            self.corrections += 1
            before = self.read(surface).intended
            self.show(surface, meant)
            after = self.read(surface)
            # A correction has generalised when what now reads the sentence is a **shape with
            # support**, not the sentence. Even the most specific level collapses the open class
            # to positions, so a kept shape at any level reads sentences that share no content
            # word with the one that was corrected — and support ≥ ``min_support`` is exactly the
            # guarantee that two demonstrations with different fillers stood behind it.
            generalised = (after.intended == meant and bool(after.level)
                           and after.support >= self.min_support)
            if generalised:
                self.generalised += 1
            _ = before, took_as
            return bool(generalised)
        except Exception:  # noqa: BLE001
            return False

    # -- generalisation ------------------------------------------------------ #
    def _settle(self) -> None:
        """Rebuild the kept shapes from every demonstration.

        Three filters, and each is a claim being refused. A shape two demonstrations disagree
        about is **contested** and never read. A shape whose demonstrations are all the same
        sentence has support 1 however many times it was shown. And a kept shape is re-run over
        the whole corpus: one that reads a demonstration into an act other than the one it came
        with is dropped, because a convention that contradicts its own lessons is not one.
        """
        kept: Dict[Tuple[str, Tuple[str, ...]], Act] = {}
        contested: Set[Tuple[str, Tuple[str, ...]]] = set()
        for key, bucket in self.seen.items():
            level, shape = key
            acts = [a for a, demos in bucket.items() if demos]
            if len(acts) > 1:
                contested.add(key)
                continue
            act = acts[0]
            demos = tuple(bucket[act])
            if len(set(demos)) < self.min_support:
                continue
            kept[key] = Act(shape=shape, level=level, act=act, demonstrations=demos)
        # Re-run every kept shape over the whole corpus. A shape that claims a demonstration it
        # disagrees with is dropped rather than down-weighted.
        for key in list(kept):
            level, shape = key
            for surface, act in self.corpus:
                found = shapes_of(surface)
                if len(found) != len(LEVELS):
                    continue
                if found[LEVELS.index(level)] == shape and act != kept[key].act:
                    del kept[key]
                    contested.add(key)
                    break
        self.kept, self.contested = kept, contested

    # -- reading ------------------------------------------------------------- #
    def read(self, surface: str, *, meaning: Optional[Meaning] = None) -> Interpretation:
        """The literal mood, and the convention if one has been learned for this shape."""
        out = Interpretation(surface=str(surface or ""))
        try:
            out.literal = literal_act(surface, meaning)
            out.intended = out.literal
            out.confidence = 0.5 if out.literal else 0.0
            shapes = shapes_of(surface)
            if len(shapes) != len(LEVELS):
                return out
            for level, shape in zip(LEVELS, shapes):
                key = (level, shape)
                if key in self.contested:
                    bucket = self.seen.get(key, {})
                    out.ambiguous = True
                    out.alternatives = tuple(sorted(bucket))
                    out.level = level
                    out.confidence = 0.0
                    out.why = (f"two conventions demonstrated for the same {level} shape; "
                               f"the sentence does not choose between them")
                    return out
                found = self.kept.get(key)
                if found is None:
                    continue
                out.intended = found.act
                out.level = level
                out.support = found.support
                # Support raises confidence; a looser level lowers it. Both are evidence about the
                # *shape*, never about whether the request is a good idea.
                penalty = 0.06 * LEVELS.index(level)
                out.confidence = max(0.3, min(0.95, 0.6 + 0.08 * found.support - penalty))
                out.why = f"{found.support} demonstrations of this {level} shape"
                return out
            out.why = "no convention demonstrated for this shape; read as its own mood"
            return out
        except Exception:  # noqa: BLE001
            return out

    def stats(self) -> Dict[str, Any]:
        return {"demonstrations": len(self.corpus), "kept": len(self.kept),
                "contested": len(self.contested), "corrections": self.corrections,
                "generalised": self.generalised,
                "by_level": {level: sum(1 for (lv, _s) in self.kept if lv == level)
                             for level in LEVELS}}


# --------------------------------------------------------------------------- #
# 2 · reference — resolved, or refused
# --------------------------------------------------------------------------- #

def _bare(name: str) -> str:
    """A referent's name with the closed-class words trimmed off its front.

    The compiler hands back what the sentence had — ``in the drawer`` for *"the key is in the
    drawer"* — and a discourse referent called *"in the drawer"* is one no later pronoun can
    match. Trimming is done from the closed class only, so nothing decides what a noun is.
    """
    try:
        tokens = tag_tokens(name)
        while tokens and tokens[0].tag in (Tag.DET, Tag.PREP, Tag.ADV):
            tokens = tokens[1:]
        return " ".join(t.text for t in tokens).strip()
    except Exception:  # noqa: BLE001
        return str(name or "").strip()


def _pronoun(word: str) -> str:
    """``speaker`` · ``addressee`` · ``third-sg`` · ``third-pl`` · ``""``."""
    word = str(word or "").strip().lower()
    if word in _SPEAKER:
        return "speaker"
    if word in _ADDRESSEE:
        return "addressee"
    if word in _THIRD_SG:
        return "third-sg"
    if word in _THIRD_PL:
        return "third-pl"
    return ""


@dataclass
class Referent:
    """One thing the conversation has introduced, and where it was introduced."""

    name: str = ""
    turn: int = 0
    role: str = ""
    plural: bool = False
    mentions: int = 1

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "turn": self.turn, "role": self.role,
                "plural": self.plural, "mentions": self.mentions}


@dataclass
class Resolution:
    """What a pronoun was taken to name, or the reason it was not taken to name anything."""

    pronoun: str = ""
    role: str = ""
    referent: str = ""
    confidence: float = 0.0
    ambiguous: bool = False
    #: Every surviving candidate with its share of the evidence, best first. This is the field the
    #: Master's specification is about: two readings at 0.61 and 0.39 is a different answer from
    #: one reading at 0.61, and a module that returned only the winner could not tell them apart.
    ranked: Tuple[Tuple[str, float], ...] = ()
    question: str = ""
    why: str = ""

    @property
    def settled(self) -> bool:
        return bool(self.referent) and not self.ambiguous

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {"pronoun": self.pronoun, "role": self.role,
                               "referent": self.referent,
                               "confidence": round(self.confidence, 4),
                               "ambiguous": self.ambiguous,
                               "ranked": [[n, round(s, 4)] for n, s in self.ranked]}
        if self.question:
            out["question"] = self.question
        if self.why:
            out["why"] = self.why
        return out


class Reference:
    """Discourse referents, and pronouns resolved against them on evidence or not at all.

    Four sources of evidence, and every one of them is either structural or countable — none is a
    guess about which entity is likelier to be tired:

    * **recency** — how many turns back the candidate was introduced;
    * **role parallelism** — a subject pronoun prefers a subject antecedent;
    * **topicality** — a candidate mentioned in more than one turn is more available than one
      mentioned once;
    * **number agreement**, and the one syntactic constraint that is genuinely closed-class: a
      pronoun in object position cannot name the subject of its own clause.

    The first three are scores and the last two are filters, which is the right shape for them: a
    preference that outvotes a grammatical impossibility is not a preference, it is a bug.
    """

    #: The most referents kept. Recency already makes an old one nearly unreachable; this stops a
    #: long-lived brain from carrying every noun it has ever heard as a live candidate.
    max_referents = 256

    #: How far apart the top two candidates must be before a pronoun is treated as settled. At
    #: 0.30, *"Ravi met Arun. He was tired."* comes back ambiguous with both ranked — which is the
    #: honest reading of that discourse, and the reading a confident resolver would destroy.
    margin = 0.30

    def __init__(self, *, speaker: str = "Master", addressee: str = "NYXARA",
                 margin: Optional[float] = None) -> None:
        self.speaker = str(speaker)
        self.addressee = str(addressee)
        if margin is not None:
            self.margin = float(margin)
        self.referents: List[Referent] = []
        self.turn = 0
        #: ``(name, relation) -> bool``, set by :class:`Communicator` to
        #: :meth:`Figure.plausible`. A **filter on evidence she actually has**: a candidate the
        #: store says cannot literally fill this predicate is struck out, and where the store
        #: knows nothing every candidate survives and the pronoun goes unresolved. That is the
        #: difference between *"Ravi walked home. He was tired."* and *"Ravi met Arun. He was
        #: tired."* — two structurally identical discourses that only a world model separates.
        self.plausible: Optional[Callable[[str, str], bool]] = None

    # -- what the conversation has put on the table -------------------------- #
    def introduce(self, meaning: Meaning, *, turn: Optional[int] = None) -> List[Referent]:
        """Record every non-pronoun filler this sentence introduced, with the role it held."""
        added: List[Referent] = []
        try:
            at = self.turn if turn is None else int(turn)
            pairs: List[Tuple[str, str]] = [("subject", meaning.subject),
                                            ("object", meaning.object)]
            for role, filler in (meaning.roles or {}).items():
                pairs.append((str(role), str(filler)))
            for role, filler in pairs:
                name = _bare(filler)
                if not name or _pronoun(name):
                    continue
                if all(t.tag != Tag.WORD for t in tag_tokens(name)):
                    continue
                existing = next((r for r in self.referents if r.name == name), None)
                if existing is not None:
                    existing.mentions += 1
                    existing.turn = at
                    existing.role = role
                    continue
                referent = Referent(name=name, turn=at, role=role,
                                    plural=_is_plural(name))
                self.referents.append(referent)
                added.append(referent)
            if len(self.referents) > self.max_referents:
                self.referents = self.referents[-self.max_referents:]
        except Exception:  # noqa: BLE001
            return added
        return added

    # -- resolution ---------------------------------------------------------- #
    def resolve(self, pronoun: str, *, role: str = "subject",
                turn: Optional[int] = None, clause_subject: str = "",
                relation: str = "") -> Resolution:
        """Score every candidate; return the winner only when the field is not close."""
        out = Resolution(pronoun=str(pronoun or "").lower(), role=str(role or ""))
        try:
            person = _pronoun(out.pronoun)
            if not person:
                out.why = "not a pronoun"
                return out
            if person == "speaker":
                out.referent, out.confidence = self.speaker, 0.95
                out.ranked = ((self.speaker, 1.0),)
                out.why = "first person names the speaker"
                return out
            if person == "addressee":
                out.referent, out.confidence = self.addressee, 0.95
                out.ranked = ((self.addressee, 1.0),)
                out.why = "second person names the hearer"
                return out
            at = self.turn if turn is None else int(turn)
            plural = person == "third-pl"
            excluded = _bare(clause_subject) if role == "object" else ""
            scored: List[Tuple[str, float]] = []
            for referent in self.referents:
                if referent.turn > at:
                    continue
                if referent.plural != plural and (referent.plural or plural):
                    continue                      # number disagreement — a filter, not a penalty
                if excluded and referent.name == excluded:
                    continue                      # a pronoun does not name its own clause subject
                if (relation and role == "subject" and self.plausible is not None
                        and not self.plausible(referent.name, relation)):
                    continue                      # struck out by what the store has witnessed
                back = max(0, at - referent.turn)
                score = 1.0 / (1.0 + back)
                if referent.role == role:
                    score += 0.35
                if referent.mentions > 1:
                    score += 0.15
                scored.append((referent.name, score))
            if not scored:
                out.why = "no candidate referent has been introduced"
                return out
            scored.sort(key=lambda pair: (-pair[1], pair[0]))
            total = sum(s for _n, s in scored) or 1.0
            out.ranked = tuple((n, s / total) for n, s in scored)
            best = out.ranked[0]
            second = out.ranked[1][1] if len(out.ranked) > 1 else 0.0
            if best[1] - second < self.margin:
                out.ambiguous = True
                out.confidence = float(best[1])
                names = " or ".join(n for n, _s in out.ranked[:3])
                out.question = f"when you say {out.pronoun!r}, do you mean {names}?"
                out.why = (f"top two within {self.margin:.2f} "
                           f"({best[1]:.2f} vs {second:.2f}) — the discourse does not choose")
                return out
            out.referent, out.confidence = best[0], float(best[1])
            out.why = f"clear by {best[1] - second:.2f} over the next candidate"
            return out
        except Exception:  # noqa: BLE001
            return out

    # -- the sentence, with its pronouns filled in --------------------------- #
    def rewrite(self, surface: str, *, meaning: Optional[Meaning] = None,
                turn: Optional[int] = None) -> Tuple[str, List[Resolution]]:
        """The surface with every **settled** pronoun replaced by what it names.

        An ambiguous pronoun is left exactly where it is, which is what makes this safe to hand to
        the grounder: an unresolved pronoun produces an unreadable sentence and therefore an
        abstention, and an abstention is the right answer to a reference nothing in the discourse
        settles. Resolving it on a coin flip would write a wrong fact into the store, and a wrong
        fact outlives the turn that made it.
        """
        resolutions: List[Resolution] = []
        text = str(surface or "")
        try:
            tokens = tag_tokens(text)
            subject = ""
            if meaning is not None and meaning.readable:
                subject = meaning.subject
            # The predicate the pronoun is an argument of, so the selectional filter has something
            # to be about. Taken from the reading where there is one, and otherwise from the last
            # open-class token — *"He was tired."* has no frame at all, and ``tired`` is the whole
            # of what the sentence predicates.
            relation = ""
            if meaning is not None and meaning.readable and meaning.relation:
                relation = meaning.relation
            else:
                opens = [t.text for t in tokens if t.tag == Tag.WORD]
                # Lemmatised, because that is what the compiler stores as a relation and
                # therefore what the witnesses are keyed on. Unlemmatised, ``tired`` never met
                # ``tir`` and the selectional filter silently had no evidence to work from —
                # a filter that is never consulted looks exactly like a filter that passes.
                relation = _lemma_of(opens[-1]) if opens else ""
            for position, token in enumerate(tokens):
                if token.tag != Tag.PRON or not _pronoun(token.text):
                    continue
                # Subject if nothing that could *be* a subject stands in front of it. Determiners,
                # subordinators, adverbs and conjunctions cannot, so a pronoun after them is still
                # the clause's subject — which is what *"When I visited…"* needs.
                before = tokens[:position]
                role = ("subject"
                        if all(t.tag in (Tag.DET, Tag.SUB, Tag.ADV, Tag.CONJ) for t in before)
                        else "object")
                resolved = self.resolve(token.text, role=role, turn=turn,
                                        clause_subject=subject if role == "object" else "",
                                        relation=relation)
                resolutions.append(resolved)
                # **Third person only.** A first- or second-person pronoun is resolved and
                # reported, and it is *not* substituted into the surface: every reader downstream
                # already maps ``I``/``mera`` to the speaker itself, and rewriting them changes a
                # sentence those readers have frames for into one they have not.
                #
                # Measured: substituting them turned *"mera naam kya hai?"* into *"Master naam kya
                # hai?"*, which :mod:`nyxara.njp.grounding`'s Hinglish question frames do not
                # match, and a question she had answered correctly for many versions came back
                # empty. Nothing about reference resolution needed it — the gain here is entirely
                # in the third person, where the antecedent is genuinely somewhere else in the
                # conversation.
                # And ``it`` is never substituted, however confidently it resolved. English
                # writes its expletive with the same word — *"It rained today"*, *"it is cold in
                # here"*, *"it turned out that…"* — and nothing available here tells one from a
                # referential ``it`` without a world model. Measured: with ``it`` substituted, the
                # discourse *"The key is in the drawer. It rained today."* filed **key rain** as a
                # fact about the world, which is a new confident error introduced by a fix for
                # confident errors. ``he``, ``she`` and ``they`` have no expletive use.
                if (resolved.settled and resolved.referent
                        and _pronoun(token.text) in ("third-sg", "third-pl")
                        and token.text not in ("it", "its")):
                    text = _replace_word(text, token.text, resolved.referent)
        except Exception:  # noqa: BLE001
            return text, resolutions
        return text, resolutions

    def stats(self) -> Dict[str, Any]:
        return {"referents": len(self.referents), "turn": self.turn,
                "names": [r.name for r in self.referents[-8:]]}


def _is_plural(name: str) -> bool:
    """The ``-s`` English puts on plural nouns, read by shape. Wrong on irregulars, and it is a
    **filter** rather than a score, so a wrong answer here costs a refusal and never a wrong
    referent."""
    word = str(name or "").strip().lower().split()[-1] if str(name or "").strip() else ""
    return len(word) > 3 and word.endswith("s") and not word.endswith(("ss", "us", "is"))


def _replace_word(text: str, word: str, replacement: str) -> str:
    """Replace the first whole-word occurrence, case-insensitively, keeping the rest untouched."""
    import re as _re
    try:
        return _re.sub(rf"\b{_re.escape(word)}\b", replacement, text, count=1,
                       flags=_re.IGNORECASE)
    except Exception:  # noqa: BLE001
        return text


# --------------------------------------------------------------------------- #
# 3 · the ledger — a contradiction is not an update
# --------------------------------------------------------------------------- #

@dataclass
class Claim:
    """One thing a speaker asserted, with the two flags that decide what a later one means."""

    subject: str = ""
    relation: str = ""
    object: str = ""
    negated: bool = False
    #: The claim quantifies over **all** times — ``never``, ``always``. A universal is not an
    #: instance and cannot be superseded by one.
    universal: bool = False
    #: The speaker signalled that time has moved — see :data:`_CHANGE`. This is the licence that
    #: makes a later claim an update rather than a contradiction, and it has to be **in the
    #: sentence**: inferring it from the topic is how a store silently overwrites its evidence.
    change: bool = False
    temporal: str = ""
    turn: int = 0
    surface: str = ""
    superseded: bool = False
    contested: bool = False

    @property
    def key(self) -> Tuple[str, str]:
        return (self.subject, self.relation)

    def to_dict(self) -> Dict[str, Any]:
        out = {"subject": self.subject, "relation": self.relation, "object": self.object,
               "negated": self.negated, "turn": self.turn}
        for name in ("universal", "change", "superseded", "contested"):
            if getattr(self, name):
                out[name] = True
        if self.temporal:
            out["temporal"] = self.temporal
        return out


@dataclass
class Verdict:
    """What a new claim did to the ones already on the ledger."""

    #: ``new`` · ``corroborates`` · ``updates`` · ``contradicts``
    kind: str = "new"
    claim: Optional[Claim] = None
    prior: Optional[Claim] = None
    why: str = ""
    question: str = ""

    @property
    def conflict(self) -> bool:
        return self.kind == "contradicts"

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {"kind": self.kind, "why": self.why}
        if self.claim is not None:
            out["claim"] = self.claim.to_dict()
        if self.prior is not None:
            out["prior"] = self.prior.to_dict()
        if self.question:
            out["question"] = self.question
        return out


#: Time words, and the two-word tails English builds out of them. The compiler carries its own
#: temporal table and it does not cover *"last year"*, so ``visited delhi last year`` and
#: ``visited delhi`` are two different claims to a ledger keyed on the object — which is how a
#: flat contradiction reads as two unrelated facts.
_TIME_WORDS: frozenset = frozenset((
    "now", "currently", "today", "yesterday", "tomorrow", "recently", "already", "still",
    "soon", "later", "then", "once", "aaj", "kal", "abhi", "ab", "pehle", "baad"))
_TIME_HEADS: frozenset = frozenset(("last", "next", "this", "every", "coming", "past", "each"))
_TIME_UNITS: frozenset = frozenset((
    "year", "years", "month", "months", "week", "weeks", "day", "days", "night", "nights",
    "morning", "evening", "time", "times", "summer", "winter", "saal", "mahina", "hafta"))


def _untimed(phrase: str) -> str:
    """A phrase with its time expression trimmed off either end.

    Trimmed rather than read: what the time *was* is :attr:`Claim.temporal`'s business, and what
    this needs is only that two mentions of one thing key the same way.
    """
    try:
        words = [t.text for t in tag_tokens(phrase)]
        while words and words[0] in _TIME_WORDS:
            words = words[1:]
        while words and words[-1] in _TIME_WORDS:
            words = words[:-1]
        if len(words) > 2 and words[-2] in _TIME_HEADS and words[-1] in _TIME_UNITS:
            words = words[:-2]
        return " ".join(words).strip()
    except Exception:  # noqa: BLE001
        return str(phrase or "").strip()


#: Subordinators after which the clause is still **asserted**. *"When I visited Delhi"* commits
#: the speaker to having visited Delhi; *"if I visited Delhi"* does not, and putting a
#: hypothetical on the ledger as a claim is how a question's presupposition becomes a fact.
_FACTIVE_SUB: frozenset = frozenset((
    "when", "whenever", "while", "because", "since", "although", "though", "after",
    "jab", "kyunki", "kyonki"))

#: The copulas a locative frame may hinge on. Closed class, read as a *shape* — ``NP be PREP NP``
#: — rather than as a verb.
_COPULA: frozenset = frozenset(("is", "are", "was", "were", "am", "be", "hai", "hain", "tha",
                                "thi", "thay", "है", "हैं", "था"))


def _locative(surface: str) -> Optional[Tuple[str, str, str]]:
    """``NP be PREP NP`` read as ``(subject, "is_at", place)``, or ``None``.

    The compiler reads *"the key is in the drawer"* as a bare ``np-verb`` clause whose verb is
    ``drawer``, so *"the key is now on the table"* is a claim about a **different relation** and
    the ledger sees no conflict at all. That is the single commonest belief-update shape there is,
    and it was invisible. The frame is over tags, so no preposition and no noun is listed.
    """
    try:
        tokens = tag_tokens(surface)
        for index, token in enumerate(tokens):
            if token.text not in _COPULA or index == 0:
                continue
            rest = tokens[index + 1:]
            while rest and (rest[0].tag in (Tag.ADV, Tag.NEG) or rest[0].text in _TIME_WORDS):
                rest = rest[1:]
            if not rest or rest[0].tag != Tag.PREP:
                continue
            subject = _bare(" ".join(t.text for t in tokens[:index]))
            place = _bare(" ".join(t.text for t in rest))
            if subject and place:
                return subject, "is_at", place
        return None
    except Exception:  # noqa: BLE001
        return None


def _asserted_clauses(surface: str) -> List[str]:
    """The whole sentence, then each clause a factive subordinator introduces.

    Tried in that order so an ordinary sentence costs nothing. *"When I visited Delhi last year I
    was tired"* is unreadable whole — the compiler has no frame for a fronted subordinate clause —
    and the claim the speaker actually made is sitting inside it.
    """
    out = [str(surface or "")]
    try:
        tokens = tag_tokens(surface)
        starts = [i for i, t in enumerate(tokens)
                  if t.tag == Tag.SUB and t.text in _FACTIVE_SUB]
        for start in starts:
            piece = tokens[start + 1:]
            # Where the subordinate clause stops and the main clause starts. A second
            # subordinator, a pronoun, or **the clause's own subject said again** — the third is
            # not a nicety: once the pronouns have been resolved into names,
            # *"when Master visited Delhi last year Master was tired"* has no pronoun left in it
            # and the cut has to find the repeated name instead.
            head = piece[0].text if piece else ""
            ends = [j for j, t in enumerate(piece)
                    if t.tag == Tag.SUB or (j > 1 and (t.tag == Tag.PRON or t.text == head))]
            cut = ends[0] if ends else len(piece)
            clause = " ".join(t.text for t in piece[:cut])
            if clause:
                out.append(clause)
    except Exception:  # noqa: BLE001
        return out
    return out


def read_claim(surface: str, meaning: Optional[Meaning] = None, *, turn: int = 0
               ) -> Optional[Claim]:
    """A sentence read as a claim, with its universal and its change licence read off the tokens.

    Returns ``None`` for anything that is not a complete assertion — a question asserts nothing,
    and a claim built out of one would put the questioner's presupposition on the ledger as
    though they had stated it.
    """
    try:
        found: Optional[Meaning] = None
        if meaning is not None and meaning.readable and meaning.kind == "assertion" \
                and meaning.complete:
            found = meaning
        else:
            for clause in _asserted_clauses(surface):
                candidate = compile_meaning(clause)
                if candidate.readable and candidate.kind == "assertion" and candidate.complete:
                    found = candidate
                    break
        if found is None:
            return None
        meaning = found
        words = [t.text for t in tag_tokens(surface)]
        universal = any(w in _UNIVERSAL_NEG for w in words)
        if not universal and not meaning.negated:
            universal = any(w in _UNIVERSAL_POS for w in words)
        change = any(w in _CHANGE for w in words)
        subject, relation, obj = (_untimed(_bare(meaning.subject)), meaning.relation,
                                  _untimed(_bare(meaning.object)))
        placed = _locative(surface)
        if placed is not None:
            subject, relation, obj = placed
        return Claim(subject=subject, relation=relation,
                     object=obj, negated=bool(meaning.negated),
                     universal=bool(universal), change=bool(change),
                     temporal=meaning.temporal, turn=int(turn), surface=str(surface or ""))
    except Exception:  # noqa: BLE001
        return None


class Ledger:
    """Every claim a conversation has made, and what each new one does to the ones before it.

    The single decision this class exists to make is **contradicts vs updates**, and the two
    things that make it are in the sentence rather than in the topic: a change licence
    (:data:`_CHANGE`) and a universal (:data:`_UNIVERSAL_NEG`, :data:`_UNIVERSAL_POS`).

    **Nothing is overwritten.** A contradicted claim stays on the ledger, marked ``contested``,
    beside the claim that contradicts it — because a store that overwrites has destroyed the only
    evidence that anything was ever wrong, and the question *"which of these is right?"* cannot be
    asked from a record that keeps one of them.
    """

    #: The most claims kept. A brain that lives for a week is a ledger that grows for a week, and
    #: the oldest **uncontested** claims are the ones a later turn is least likely to be about.
    #: A contested pair is never trimmed: it is the evidence that something is wrong, and evidence
    #: that ages out is evidence that was never kept.
    max_claims = 2000

    def __init__(self) -> None:
        self.claims: List[Claim] = []
        #: Relations a speaker has used a change marker with. Evidence, sitting in the transcript,
        #: that this is a relation where *later* differs from *wrong* — so a second object for one
        #: of these without a licence is a conflict, and a second object for a relation nobody has
        #: shown that behaviour for is simply more information.
        self.stateful: Set[str] = set()
        self.conflicts: List[Tuple[Claim, Claim]] = []

    def note(self, claim: Optional[Claim]) -> Verdict:
        """File one claim and say what it did."""
        out = Verdict(claim=claim)
        try:
            if claim is None:
                out.kind, out.why = "none", "not a complete assertion"
                return out
            priors = [c for c in self.claims if c.key == claim.key and not c.superseded]
            if claim.change:
                self.stateful.add(claim.relation)
            for prior in priors:
                if prior.object == claim.object:
                    if prior.negated == claim.negated:
                        prior.turn = claim.turn
                        out.kind, out.prior = "corroborates", prior
                        out.why = "the same claim, said again"
                        self.claims.append(claim)
                        return out
                    if prior.universal or claim.universal:
                        return self._conflict(
                            out, prior, claim,
                            "a universal cannot be superseded by an instance: one of these is "
                            "wrong rather than out of date")
                    if claim.change:
                        prior.superseded = True
                        self.claims.append(claim)
                        out.kind, out.prior = "updates", prior
                        out.why = ("the speaker marked the change, so this is later "
                                   "rather than a denial")
                        return out
                    return self._conflict(
                        out, prior, claim,
                        "the same claim asserted and denied, with nothing said about time")
                # A different object for the same subject and relation.
                if prior.negated != claim.negated:
                    continue
                if claim.change:
                    prior.superseded = True
                    self.claims.append(claim)
                    out.kind, out.prior = "updates", prior
                    out.why = "a marked change of a relation that holds one value at a time"
                    return out
                if claim.relation in self.stateful:
                    return self._conflict(
                        out, prior, claim,
                        "this relation has been shown to hold one value at a time, and nothing "
                        "in this sentence says the value changed")
            self.claims.append(claim)
            out.kind = "new"
            out.why = "nothing on the ledger speaks to this"
            self._trim()
            return out
        except Exception:  # noqa: BLE001
            return out

    def _trim(self) -> None:
        if len(self.claims) <= self.max_claims:
            return
        keep = [c for c in self.claims if c.contested]
        rest = [c for c in self.claims if not c.contested]
        self.claims = keep + rest[-(self.max_claims - len(keep)):] if keep else \
            rest[-self.max_claims:]

    def _conflict(self, out: Verdict, prior: Claim, claim: Claim, why: str) -> Verdict:
        prior.contested = claim.contested = True
        self.claims.append(claim)
        self.conflicts.append((prior, claim))
        out.kind, out.prior, out.why = "contradicts", prior, why
        out.question = (f"you told me {_render(prior)}, and also {_render(claim)} — "
                        f"which of those holds?")
        return out

    # -- reading it back ----------------------------------------------------- #
    def holds(self, subject: str, relation: str) -> str:
        """What the ledger currently says, or ``""`` where it says two things at once.

        The empty string for a contested pair is the point of the class. Returning the later claim
        would be exactly the silent overwrite this exists to refuse, and returning the earlier one
        would be the same mistake facing the other way.
        """
        try:
            found = [c for c in self.claims
                     if c.subject == _bare(subject) and c.relation == relation
                     and not c.superseded]
            if not found:
                return ""
            if any(c.contested for c in found):
                return ""
            live = [c for c in found if not c.negated]
            return live[-1].object if live else ""
        except Exception:  # noqa: BLE001
            return ""

    def contested_claims(self) -> List[Tuple[Claim, Claim]]:
        return list(self.conflicts)

    def dispute(self, subject: str, relation: str) -> str:
        """The question a contested pair raises, or ``""``.

        The difference between this and :meth:`holds` returning nothing is the difference between
        *"I was never told"* and *"I was told two things"*, and a caller that cannot tell them
        apart will report the second as the first.
        """
        try:
            want = (_bare(subject), relation)
            for prior, now in reversed(self.conflicts):
                if prior.key == want:
                    return (f"you told me {_render(prior)}, and also {_render(now)} — "
                            f"which of those holds?")
            return ""
        except Exception:  # noqa: BLE001
            return ""

    def stats(self) -> Dict[str, Any]:
        return {"claims": len(self.claims), "conflicts": len(self.conflicts),
                "stateful": sorted(self.stateful),
                "superseded": sum(1 for c in self.claims if c.superseded)}


def _render(claim: Claim) -> str:
    """A claim as the shortest English that keeps its polarity. The polarity is the whole reason
    this function exists: *"Master visit delhi"* is what the brain said back to a denial, and it is
    the opposite of what was said to her."""
    relation = str(claim.relation).replace("_", " ")
    core = f"{claim.subject} {relation} {claim.object}".strip()
    if not claim.negated:
        return core
    negator = "never" if claim.universal else "does not"
    return f"{claim.subject} {negator} {relation} {claim.object}".strip()


# --------------------------------------------------------------------------- #
# 4 · other minds — bounded recursion, driven by a sentence
# --------------------------------------------------------------------------- #

def _possessive(surface: str) -> Optional[Tuple[str, str, str]]:
    """``NP have NP`` read as ``(subject, "have", object)``, or ``None``.

    ``has`` is tagged ``AUX``, correctly — it is the perfect auxiliary — and the consequence is
    that *"the box has the red ball"* has no frame at all. The shape here is ``AUX`` followed by a
    determiner and open material, which is where the perfect cannot be: a perfect takes a
    participle, not a noun phrase.
    """
    try:
        tokens = tag_tokens(surface)
        for index, token in enumerate(tokens):
            if token.text not in ("has", "have", "had") or index == 0:
                continue
            rest = tokens[index + 1:]
            if not rest or rest[0].tag not in (Tag.DET, Tag.WORD):
                continue
            subject = _bare(" ".join(t.text for t in tokens[:index]))
            obj = _bare(" ".join(t.text for t in rest))
            if subject and obj:
                return subject, "have", obj
        return None
    except Exception:  # noqa: BLE001
        return None


def proposition(clause: str) -> Tuple[str, str]:
    """A clause as ``(key, value)`` — ``"box|have"`` and ``"red ball"``.

    Keyed on subject and relation with the **object as the value**, which is the shape a
    false-belief question needs: *"Ravi thinks the box has the blue ball"* is not a belief about a
    different proposition from *"the box has the red ball"*, it is a different **answer to the same
    one**, and a key that folded the object in could never see the two meet.
    """
    try:
        text = str(clause or "").strip()
        for reader in (_locative, _possessive):
            found = reader(text)
            if found is not None:
                subject, relation, obj = found
                return f"{subject}|{relation}", _untimed(obj)
        meaning = compile_meaning(text)
        if meaning.readable and meaning.complete:
            key = f"{_untimed(_bare(meaning.subject))}|{meaning.relation}"
            value = _untimed(_bare(meaning.object))
            return key, (f"¬{value}" if meaning.negated else value)
        return "", ""
    except Exception:  # noqa: BLE001
        return "", ""


def _attitude(surface: str) -> Optional[Tuple[str, str, bool, str]]:
    """``(holder, strength, negated, clause)`` for ``X thinks/knows …``, or ``None``."""
    try:
        tokens = tag_tokens(surface)
        for index, token in enumerate(tokens):
            if token.text not in _MENTAL or index == 0:
                continue
            head = tokens[:index]
            negated = any(t.tag == Tag.NEG for t in head)
            holder = [t for t in head if t.tag in (Tag.WORD, Tag.PRON)]
            if not holder:
                continue
            rest = tokens[index + 1:]
            while rest and rest[0].text in ("that", "ki", "कि"):
                rest = rest[1:]
            clause = " ".join(t.text for t in rest).strip()
            if not clause:
                continue
            return holder[-1].text, _MENTAL[token.text], negated, clause
        return None
    except Exception:  # noqa: BLE001
        return None


class Minds:
    """What is true, what she holds, and what somebody else holds — from what was said.

    :mod:`nyxara.social.tom` has been a real recursive belief engine in this repository for many
    versions, and until now **nothing could put a sentence into it**. That is the gap this class
    closes, and it is the whole of what it claims: the store, the nesting and the false-belief
    query were already there and are used rather than rewritten.

    Two properties are this class's own. **Ignorance is not a false belief**: *"Ravi thinks I
    don't know"* stores :data:`UNKNOWN` at depth two rather than the negation of anything, because
    an agent who holds no view and an agent who holds the wrong one behave differently and a test
    that conflated them would pass for the wrong reason. And **depth stops paying**: a level whose
    content repeats the level below it is not stored, which is the bound worth having — cheaper
    than a depth limit and, unlike one, it is about the content rather than about the counting.
    """

    #: Levels of nesting kept. Beyond three the attributions this module can read from a sentence
    #: are not attested in anything anyone says, and an unbounded store of them is a memory leak
    #: dressed as a capability.
    max_depth = 3

    def __init__(self, *, speaker: str = "Master", max_depth: Optional[int] = None) -> None:
        self.speaker = str(speaker)
        if max_depth is not None:
            self.max_depth = max(1, int(max_depth))
        self.collapsed = 0
        self.heard = 0
        self.deepest = 0
        try:
            from nyxara.social.tom import TheoryOfMind
            self.store: Any = TheoryOfMind()
            self.backing = "social.tom"
        except Exception:  # noqa: BLE001
            self.store = _Nest()
            self.backing = "internal"

    # -- hearing ------------------------------------------------------------- #
    def hear(self, surface: str) -> Dict[str, Any]:
        """Read one sentence into the world, or into somebody's head, however deep it nests."""
        out: Dict[str, Any] = {"depth": 0, "path": [], "key": "", "value": None}
        try:
            self.heard += 1
            path: List[str] = []
            clause = str(surface or "").strip()
            ignorant = False
            while len(path) < self.max_depth:
                found = _attitude(clause)
                if found is None:
                    break
                holder, strength, negated, clause = found
                path.append(self._name(holder))
                if negated and strength == "know":
                    ignorant = True
                    break
                if negated:
                    # "Ravi does not think X" attributes no belief in X, which is a different
                    # state from believing not-X and is stored as the one it is.
                    ignorant = True
                    break
            key, value = proposition(clause)
            if not key:
                return out
            out["key"], out["path"] = key, list(path)
            if not path:
                self.store.set_world(key, value)
                out["value"] = value
                return out
            stored: Any = UNKNOWN if ignorant else value
            if len(path) > 1:
                below = self.store.belief(path[:-1], key)
                if below is not None and below == stored:
                    self.collapsed += 1
                    out["value"], out["depth"] = stored, len(path) - 1
                    return out
            self.store.set_belief(path, key, stored)
            out["value"], out["depth"] = stored, len(path)
            self.deepest = max(self.deepest, len(path))
            return out
        except Exception:  # noqa: BLE001
            return out

    def _name(self, holder: str) -> str:
        """The speaker under one name, whether they were named or pronounced.

        :class:`Reference` resolves *"I"* into the speaker's name **before** this sees the
        sentence, so a nested belief arrived under ``master`` while every query asked for
        ``Master`` — the belief was stored correctly and could not be read back, which is the
        worst shape a defect can take because every organ reports success.
        """
        person = _pronoun(holder)
        if person == "speaker":
            return self.speaker
        if str(holder).strip().lower() == self.speaker.strip().lower():
            return self.speaker
        return str(holder)

    # -- asking -------------------------------------------------------------- #
    def world(self, key: str) -> Any:
        try:
            return self.store.get_world(key)
        except Exception:  # noqa: BLE001
            return None

    def believes(self, agent: str, key: str) -> Any:
        try:
            return self.store.belief([str(agent)], key)
        except Exception:  # noqa: BLE001
            return None

    def false_belief(self, agent: str, key: str) -> bool:
        """Does this agent hold something the world contradicts? ``UNKNOWN`` is not a false
        belief and neither is having no belief at all."""
        try:
            held = self.believes(agent, key)
            truth = self.world(key)
            return bool(held is not None and held != UNKNOWN and truth is not None
                        and held != truth)
        except Exception:  # noqa: BLE001
            return False

    def attributes(self, observer: str, target: str, key: str) -> Any:
        try:
            return self.store.belief([str(observer), str(target)], key)
        except Exception:  # noqa: BLE001
            return None

    def attributes_ignorance(self, observer: str, target: str, key: str) -> bool:
        return self.attributes(observer, target, key) == UNKNOWN

    def stats(self) -> Dict[str, Any]:
        return {"backing": self.backing, "heard": self.heard, "deepest": self.deepest,
                "collapsed": self.collapsed, "max_depth": self.max_depth}


class _Nest:
    """The fallback store, with :mod:`nyxara.social.tom`'s four methods and nothing else.

    Present so this module keeps working if that one is ever unavailable, and deliberately not a
    second implementation of it: everything interesting — the recursion, the false-belief query —
    lives there, and duplicating it here would be two engines drifting apart.
    """

    def __init__(self) -> None:
        self._world: Dict[str, Any] = {}
        self._beliefs: Dict[Tuple[str, ...], Dict[str, Any]] = {}

    def set_world(self, prop: str, value: Any) -> None:
        self._world[prop] = value

    def get_world(self, prop: str) -> Any:
        return self._world.get(prop)

    def set_belief(self, path: Any, prop: str, value: Any) -> None:
        key = tuple([path] if isinstance(path, str) else list(path))
        self._beliefs.setdefault(key, {})[prop] = value

    def belief(self, path: Any, prop: str) -> Any:
        key = tuple([path] if isinstance(path, str) else list(path))
        return self._beliefs.get(key, {}).get(prop)


# --------------------------------------------------------------------------- #
# 5 · register — as much as the hearer needs, never more than she holds
# --------------------------------------------------------------------------- #

@dataclass
class Said:
    """One rendering of one meaning, with the claim that was parsed back kept separate."""

    audience: str = ""
    #: The sentence that carries the claim. This is the part that is verified.
    claim: str = ""
    #: The claim plus whatever this register adds around it.
    text: str = ""
    verified: bool = False
    #: Which of the meaning's fields this register said out loud.
    carried: Tuple[str, ...] = ()

    @property
    def words(self) -> int:
        return len(str(self.text or "").split())

    def to_dict(self) -> Dict[str, Any]:
        return {"audience": self.audience, "text": self.text, "claim": self.claim,
                "verified": self.verified, "carried": list(self.carried),
                "words": self.words}


class Register:
    """One meaning, four hearers, and a rendering that is thrown away unless it reads back.

    :meth:`nyxara.njp.language.Grammar.say` established the rule and it is the only one that makes
    generated language safe here: **she says only what she can read back**. A rendering is parsed
    again and must return the same subject, relation, object and polarity; if it does not, it is
    not returned. A fluent surface she cannot verify is babble with her name on it.

    **What varies is how much of what she holds is said**, not how the words are chosen. She has
    no dictionary and no thesaurus, so lexical simplification is a resource this package does not
    have, and claiming it would be the kind of dressed-up gap the rest of this repository exists
    to refuse. What she genuinely can control is scope: the bare claim; the claim with the
    condition it was stated under; with its time and its modality; with what she is entitled to
    assert it on. That is a real communication strategy and it preserves truth by construction,
    because every register is a **subset of one meaning** and never an addition to it.
    """

    def __init__(self) -> None:
        self.rendered = 0
        self.refused = 0

    # -- one rendering ------------------------------------------------------- #
    def say(self, meaning: Meaning, audience: str = "student") -> Said:
        out = Said(audience=str(audience or "student"))
        try:
            if meaning is None or not meaning.readable:
                return out
            core = self._claim(meaning)
            if not core:
                return out
            out.claim, out.text, out.carried = core, core, ("claim",)
            out.verified = self._survives(meaning, core)
            if not out.verified:
                self.refused += 1
                return out
            self.rendered += 1
            rank = REGISTERS.index(out.audience) if out.audience in REGISTERS else 1
            extras: List[str] = []
            carried = ["claim"]
            if rank >= 1 and meaning.condition:
                extras.append(f"this holds if {meaning.condition}")
                carried.append("condition")
            if rank >= 2:
                if meaning.temporal:
                    extras.append(f"as of {meaning.temporal}")
                    carried.append("temporal")
                if meaning.modality:
                    extras.append(f"held as {meaning.modality} rather than certain")
                    carried.append("modality")
            if rank >= 3:
                extras.append(f"on {meaning.evidential or 'stated'} grounds, "
                              f"parse confidence {meaning.confidence:.2f}")
                carried.append("evidential")
            out.text = "; ".join([core] + extras)
            out.carried = tuple(carried)
            return out
        except Exception:  # noqa: BLE001
            return out

    def spread(self, meaning: Meaning) -> Dict[str, Said]:
        """All four, so the thing that is meant to be constant across them can be checked."""
        return {audience: self.say(meaning, audience) for audience in REGISTERS}

    # -- the claim, and reading it back -------------------------------------- #
    @staticmethod
    def _claim(meaning: Meaning) -> str:
        subject, relation, obj = meaning.subject, meaning.relation, meaning.object
        if not subject or not relation:
            return ""
        if meaning.negated:
            return " ".join(x for x in (subject, "does not", relation, obj) if x).strip()
        return " ".join(x for x in (subject, relation, obj) if x).strip()

    @staticmethod
    def _survives(meaning: Meaning, sentence: str) -> bool:
        """Does the sentence read back as the meaning it was rendered from?"""
        try:
            back = compile_meaning(sentence)
            if not back.readable:
                return False
            return (_bare(back.subject) == _bare(meaning.subject)
                    and back.relation == meaning.relation
                    and _bare(back.object) == _bare(meaning.object)
                    and bool(back.negated) == bool(meaning.negated))
        except Exception:  # noqa: BLE001
            return False

    def stats(self) -> Dict[str, Any]:
        return {"rendered": self.rendered, "refused": self.refused}


# --------------------------------------------------------------------------- #
# 6 · figurative — true without being literal
# --------------------------------------------------------------------------- #

@dataclass
class Figurative:
    """Whether a reading should be taken literally, and the evidence either way."""

    figurative: bool = False
    relation: str = ""
    subject: str = ""
    #: The kind every witnessed subject of this relation shares and this one does not.
    violated: str = ""
    witnesses: Tuple[str, ...] = ()
    why: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {"figurative": self.figurative, "relation": self.relation,
                "subject": self.subject, "violated": self.violated,
                "witnesses": list(self.witnesses), "why": self.why}


class Figure:
    """A sentence read as non-literal, and only ever **on the store's own evidence**.

    *"The market swallowed the shock"* is not a claim about a market's digestion, and the honest
    way to know that is to have seen what swallows things. So: where a relation's witnessed
    subjects all share a kind, and the new subject demonstrably does not have it, the reading is
    flagged and nothing is filed. Where there are no witnesses, or where the new subject's kinds
    are unknown, **nothing is claimed** — with an empty store no sentence is figurative, and that
    is the right answer rather than a shortcoming. A module that shipped a list of which nouns may
    swallow would be the ontology this package refuses everywhere else.

    ``kinds`` is injected rather than imported: the caller decides where a kind comes from, and
    :class:`~nyxara.njp.brain.NJPBrain` hands it its own ``is_a`` store.
    """

    #: How many distinct witnessed subjects a relation needs before its shared kind is evidence
    #: rather than coincidence. Two subjects share a kind by accident all the time.
    min_witnesses = 3

    def __init__(self, kinds: Optional[Callable[[str], Sequence[str]]] = None, *,
                 min_witnesses: Optional[int] = None) -> None:
        self.kinds = kinds
        if min_witnesses is not None:
            self.min_witnesses = max(2, int(min_witnesses))
        self.witnesses: Dict[str, List[str]] = {}
        self.flagged = 0

    def witness(self, relation: str, subject: str) -> None:
        """Record one literal use, which is the only thing that ever makes evidence here."""
        try:
            relation, subject = str(relation or ""), _bare(subject)
            if not relation or not subject:
                return
            seen = self.witnesses.setdefault(relation, [])
            if subject not in seen:
                seen.append(subject)
        except Exception:  # noqa: BLE001
            return

    def _kinds(self, name: str) -> Set[str]:
        try:
            if self.kinds is None:
                return set()
            return {str(k).strip().lower() for k in (self.kinds(name) or ()) if str(k).strip()}
        except Exception:  # noqa: BLE001
            return set()

    def judge(self, meaning: Meaning) -> Figurative:
        out = Figurative()
        try:
            if meaning is None or not meaning.readable or not meaning.complete:
                return out
            relation, subject = meaning.relation, _bare(meaning.subject)
            out.relation, out.subject = relation, subject
            seen = [s for s in self.witnesses.get(relation, ()) if s != subject]
            out.witnesses = tuple(seen)
            if len(seen) < self.min_witnesses:
                out.why = (f"{len(seen)} witnessed subjects for {relation!r}, "
                           f"below {self.min_witnesses} — nothing to violate")
                return out
            shared: Optional[Set[str]] = None
            for name in seen:
                found = self._kinds(name)
                if not found:
                    out.why = (f"the kinds of {name!r} are unknown, so the witnesses "
                               f"agree on nothing")
                    return out
                shared = found if shared is None else (shared & found)
            if not shared:
                out.why = f"the witnessed subjects of {relation!r} share no kind"
                return out
            mine = self._kinds(subject)
            if not mine:
                out.why = f"the kinds of {subject!r} are unknown, so nothing is claimed"
                return out
            if shared & mine:
                out.why = f"{subject!r} is a {sorted(shared & mine)[0]}, as every witness is"
                return out
            out.figurative = True
            out.violated = sorted(shared)[0]
            out.why = (f"every one of {len(seen)} witnessed subjects of {relation!r} is a "
                       f"{out.violated}, and {subject!r} is not")
            self.flagged += 1
            return out
        except Exception:  # noqa: BLE001
            return out

    def plausible(self, name: str, relation: str) -> bool:
        """Could this name literally fill the subject of this relation, on the evidence?

        The same test as :meth:`judge`, asked of a candidate antecedent rather than of a sentence,
        and it is what lets *"Ravi walked home. He was tired."* resolve while *"Ravi met Arun. He
        was tired."* stays ambiguous. Both are structurally identical discourses; what separates
        them is knowing that a home is not the sort of thing that gets tired — so this returns
        ``True`` whenever she does not know that, and the pronoun stays unresolved.
        """
        try:
            seen = [s for s in self.witnesses.get(relation, ()) if s != _bare(name)]
            if len(seen) < self.min_witnesses:
                return True
            shared: Optional[Set[str]] = None
            for other in seen:
                found = self._kinds(other)
                if not found:
                    return True
                shared = found if shared is None else (shared & found)
            if not shared:
                return True
            mine = self._kinds(name)
            return True if not mine else bool(shared & mine)
        except Exception:  # noqa: BLE001
            return True

    def stats(self) -> Dict[str, Any]:
        return {"relations": len(self.witnesses), "flagged": self.flagged,
                "witnesses": sum(len(v) for v in self.witnesses.values())}


# --------------------------------------------------------------------------- #
# the meaning adversary — two readings of one sentence
# --------------------------------------------------------------------------- #

#: The prepositions this module will call an attachment ambiguity on. **One**, and the narrowness
#: is deliberate. *"I saw the man with the telescope"* is genuinely two sentences and everybody
#: agrees it is; *"Ravi met Arun in the garden"* is technically two and nobody asks. A module that
#: flagged every trailing prepositional phrase would turn abstention — the property this package
#: spends everything else to keep — into a tic.
_AMBIGUOUS_PREP: frozenset = frozenset(("with",))


@dataclass
class Readings:
    """Two interpretations of one sentence, and the question that would settle it."""

    ambiguous: bool = False
    preposition: str = ""
    inner: str = ""
    outer: str = ""
    interpretations: Tuple[Tuple[str, float], ...] = ()
    question: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {"ambiguous": self.ambiguous, "preposition": self.preposition,
                "interpretations": [[text, round(score, 4)]
                                    for text, score in self.interpretations],
                "question": self.question}


def attachment(surface: str, meaning: Optional[Meaning] = None) -> Readings:
    """*"I saw the man with the telescope."* — read both ways, and neither chosen.

    The evidence that there is an ambiguity at all is **in the compiler's own output**: it has no
    frame for a prepositional phrase after an object, so it fuses the two noun phrases into one
    object string — ``man telescope``, with ``with the`` silently dropped. That fusion is a
    reliable signal, and reading it back out gives both attachment sites without needing a parser
    this package does not have.

    Nothing here picks a winner. The two interpretations come back at 0.50 each and the question
    comes back with them, which is the Master's specification exactly: a genuinely ambiguous
    sentence read confidently one way is worse than one refused.
    """
    out = Readings()
    try:
        meaning = compile_meaning(surface) if meaning is None else meaning
        if not meaning.readable or not meaning.object:
            return out
        tokens = tag_tokens(surface)
        preps = [i for i, t in enumerate(tokens)
                 if t.tag == Tag.PREP and t.text in _AMBIGUOUS_PREP]
        if not preps:
            return out
        at = preps[0]
        before = [t.text for t in tokens[:at] if t.tag == Tag.WORD]
        after = [t.text for t in tokens[at + 1:] if t.tag == Tag.WORD]
        if not before or not after:
            return out
        inner, outer = before[-1], after[-1]
        pieces = str(meaning.object).split()
        if inner not in pieces or outer not in pieces:
            return out                          # the compiler did not fuse them; nothing to read
        out.ambiguous = True
        out.preposition = tokens[at].text
        out.inner, out.outer = inner, outer
        subject, relation = meaning.subject, meaning.relation
        out.interpretations = (
            (f"{subject} {relation} {inner}, using {outer}", 0.5),
            (f"{subject} {relation} {inner}, and {inner} has {outer}", 0.5),
        )
        out.question = (f"did {subject} {relation} the {inner} using the {outer}, "
                        f"or does the {inner} have the {outer}?")
        return out
    except Exception:  # noqa: BLE001
        return out


# --------------------------------------------------------------------------- #
# the whole turn
# --------------------------------------------------------------------------- #

@dataclass
class Uptake:
    """One turn, heard: what it said, what it did, what it referred to, and what it cost."""

    surface: str = ""
    speaker: str = ""
    turn: int = 0
    #: The surface with every settled pronoun replaced. An ambiguous one is left alone, so this
    #: is safe to hand to a grounder: an unresolved pronoun grounds to nothing.
    rewritten: str = ""
    meaning: Optional[Meaning] = None
    interpretation: Optional[Interpretation] = None
    resolutions: List[Resolution] = field(default_factory=list)
    verdict: Optional[Verdict] = None
    figurative: Optional[Figurative] = None
    #: The two readings of a prepositional attachment, when there are two — see
    #: :func:`attachment`.
    readings: Optional[Readings] = None
    #: The one thing worth asking back, or ``""``. Ordered by what blocks understanding hardest:
    #: an act nobody can read, then a referent nothing settles, then two claims that cannot both
    #: hold.
    question: str = ""

    @property
    def literal(self) -> bool:
        """Should this sentence be filed as a fact about the world?"""
        return not (self.figurative is not None and self.figurative.figurative)

    @property
    def conflict(self) -> bool:
        return self.verdict is not None and self.verdict.conflict

    @property
    def ambiguous(self) -> bool:
        return bool(self.question)

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {"surface": self.surface, "turn": self.turn,
                               "speaker": self.speaker, "literal": self.literal}
        if self.rewritten and self.rewritten != self.surface:
            out["rewritten"] = self.rewritten
        if self.interpretation is not None:
            out["act"] = self.interpretation.to_dict()
        if self.resolutions:
            out["reference"] = [r.to_dict() for r in self.resolutions]
        if self.verdict is not None and self.verdict.kind != "none":
            out["ledger"] = self.verdict.to_dict()
        if self.figurative is not None and self.figurative.figurative:
            out["figurative"] = self.figurative.to_dict()
        if self.readings is not None and self.readings.ambiguous:
            out["readings"] = self.readings.to_dict()
        if self.question:
            out["question"] = self.question
        return out


class Communicator:
    """The six organs above, run over a conversation, one turn at a time.

    The order inside :meth:`hear` is the mechanism rather than a convenience, and two steps in it
    are load-bearing:

    * **Reference before grounding.** A pronoun is resolved into the sentence *before* the claim
      is read off it, so *"He was tired"* becomes a claim about somebody or it becomes nothing.
      Resolving it afterwards would mean the ledger and the store had already filed a claim about
      a pronoun.
    * **Figurative before the ledger.** A non-literal sentence never reaches the ledger and is
      never witnessed, so one metaphor cannot become the evidence that licenses the next.
    """

    def __init__(self, *, speaker: str = "Master", addressee: str = "NYXARA",
                 kinds: Optional[Callable[[str], Sequence[str]]] = None,
                 max_history: int = 64) -> None:
        self.acts = ActLearner()
        self.reference = Reference(speaker=speaker, addressee=addressee)
        self.ledger = Ledger()
        self.minds = Minds(speaker=speaker)
        self.register = Register()
        self.figure = Figure(kinds)
        self.reference.plausible = self.figure.plausible
        self.speaker = str(speaker)
        self.turn = 0
        self.max_history = max(1, int(max_history))
        self.history: List[Uptake] = []

    # -- being taught -------------------------------------------------------- #
    def show(self, surface: str, act: str) -> None:
        """Demonstrate that this surface does this. See :meth:`ActLearner.show`."""
        self.acts.show(surface, act)

    def misread(self, surface: str, *, took_as: str, meant: str) -> bool:
        """A communication that failed, handed back as a lesson.
        See :meth:`ActLearner.misread`."""
        return self.acts.misread(surface, took_as=took_as, meant=meant)

    def teach_kind(self, name: str, kind: str) -> None:
        """Witness one literal filler, so the selectional evidence has somewhere to come from."""
        self.figure.witness(str(kind), str(name))

    # -- one turn ------------------------------------------------------------ #
    def hear(self, surface: str, *, speaker: Optional[str] = None) -> Uptake:
        out = Uptake(surface=str(surface or ""), speaker=str(speaker or self.speaker))
        try:
            self.turn += 1
            out.turn = self.reference.turn = self.turn
            first = compile_meaning(out.surface)
            out.rewritten, out.resolutions = self.reference.rewrite(
                out.surface, meaning=first, turn=self.turn)
            out.meaning = (compile_meaning(out.rewritten)
                           if out.rewritten != out.surface else first)
            out.interpretation = self.acts.read(out.surface, meaning=out.meaning)
            out.figurative = self.figure.judge(out.meaning)
            out.readings = attachment(out.rewritten, out.meaning)
            literal_assertion = (out.interpretation.intended in ("assertion", "exclamation")
                                 or out.interpretation.literal == "assertion")
            if literal_assertion and out.literal:
                claim = read_claim(out.rewritten, turn=self.turn)
                out.verdict = self.ledger.note(claim)
                if claim is not None:
                    self.figure.witness(claim.relation, claim.subject)
                self.minds.hear(out.rewritten)
            elif not out.literal:
                out.verdict = Verdict(kind="none", why="read as non-literal; nothing filed")
            if out.meaning is not None:
                self.reference.introduce(out.meaning, turn=self.turn)
            out.question = self._question(out)
            self.history.append(out)
            if len(self.history) > self.max_history:
                self.history = self.history[-self.max_history:]
            return out
        except Exception:  # noqa: BLE001
            return out

    @staticmethod
    def _question(out: Uptake) -> str:
        """The one thing worth asking back. Ordered by what blocks understanding hardest."""
        try:
            if out.interpretation is not None and out.interpretation.ambiguous:
                alternatives = " or ".join(out.interpretation.alternatives)
                return (f"I cannot tell whether that is {alternatives} — "
                        f"both have been shown to me in that shape")
            for resolved in out.resolutions:
                if resolved.ambiguous and resolved.question:
                    return resolved.question
            if out.verdict is not None and out.verdict.question:
                return out.verdict.question
            if out.readings is not None and out.readings.ambiguous:
                return out.readings.question
            return ""
        except Exception:  # noqa: BLE001
            return ""

    def reset(self) -> None:
        """A new conversation, the same learned conventions.

        The ledger, the referents and the other minds are **about this conversation** and start
        empty; the speech-act shapes, the witnessed subjects and the kinds are **about the
        language** and survive. Conflating the two would mean either that a convention had to be
        re-taught every time somebody new spoke, or that one conversation's referents could answer
        a pronoun in the next — and the second of those is how a resolver invents an antecedent.
        """
        self.reference = Reference(speaker=self.reference.speaker,
                                   addressee=self.reference.addressee)
        self.reference.plausible = self.figure.plausible
        self.ledger = Ledger()
        self.minds = Minds(speaker=self.speaker)
        self.turn = 0
        self.history = []

    # -- asking it things ---------------------------------------------------- #
    def say(self, meaning: Meaning, audience: str = "student") -> Said:
        return self.register.say(meaning, audience)

    def figurative(self, surface: str) -> bool:
        """Is this sentence one that must not be filed literally?

        The brain's store guard, and it is read-only: nothing is learned, nothing is witnessed,
        and a sentence about which she has no evidence comes back ``False``.
        """
        try:
            return bool(self.figure.judge(compile_meaning(surface)).figurative)
        except Exception:  # noqa: BLE001
            return False

    def holds(self, subject: str, relation: str) -> str:
        return self.ledger.holds(subject, relation)

    def stats(self) -> Dict[str, Any]:
        return {"turns": self.turn, "acts": self.acts.stats(),
                "reference": self.reference.stats(), "ledger": self.ledger.stats(),
                "minds": self.minds.stats(), "register": self.register.stats(),
                "figure": self.figure.stats()}
