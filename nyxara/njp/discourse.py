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

**Nothing that is a claim about *meaning* is written down here.** The first version of this module
shipped four tables that were: which words license an update, which quantify over all times, which
verbs take a belief, and which preposition attaches two ways — plus three tuned numbers in the
pronoun resolver. They are gone. What replaced them is :class:`MarkerLearner`,
:class:`AttachLearner`, :func:`clause_proposition` and :meth:`Reference.fit`, and the difference is
not cosmetic: a table cannot be wrong in a way any measurement could show, and every one of those
four is now a subject on the report card with a floor, a lesson and a gain. The tables that remain
— the pronouns, the copula, the factive subordinators, the time words — are the closed class
:mod:`nyxara.njp.semantics` already defends, where a list is the honest representation.

Eleven organs, one discipline, and it is the discipline of :mod:`nyxara.njp.language` rather
than a new one: **a lesson leaves a shape, never an answer; a shape is kept only once it has read
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

**7 · Why, and what for** (:class:`Connective`). The Master's anchor names eleven slots and six
were reachable — entity, relation, time, belief, uncertainty, and space through ``is_at``. The
three that were not are here. **Cause** is the interesting one: nothing structural separates
*"A because B"* from *"B so A"*, both being two clauses with a word between, and which of them
puts the cause first is a fact about English that a module could only be told. So it is shown
instead, per connective, and both wordings of one fact land on the same cause. **Purpose** is the
same mechanism over an infinitival tail, which is told from a prepositional phrase structurally —
a determiner between the preposition and the noun. And **whether a relation holds or happens** is
read off how it has behaved: one a speaker marked a change on is a state, one carrying several
live values for a subject is an event, and until she has seen a marked change at all she cannot
tell those apart and says so.

Two things keep the connective honest, and both were found by measurement. A **determiner is not
a connective** — ``the`` splits *"devi lit | the | lamp because the room was dark"* into two
halves that both read, and it was induced as causal and gave a cause of ``lamp because the room
was dark``. And the winner is the **best-supported** candidate rather than the leftmost, which is
the rule :mod:`nyxara.njp.semantics` and :mod:`nyxara.njp.compile` both state and which this
class was breaking.

**8 · The same meaning, said a different way round** (:class:`Alternation`). Four ordinary English
shapes came back unreadable or worse: *"The window was opened by Ravi."*, *"Ravi has been opening
the door."*, *"He was tired."* and *"Ravi opened the door and Arun the window."* — the last of
which had a **wrong** reading rather than none, a single claim whose object was ``door arun
window``, filed at the confidence of a fact about something. Detecting the shapes is structural
and belongs with the closed class (:func:`frames_of`). **Reading them is not**: that a passive
puts the patient in front and the agent after the preposition is a fact about English, not about
the shape ``NP be V PREP NP``, and a module that wrote the mapping down could never be told
otherwise. So a demonstration is two sentences that mean the same thing — one the compiler reads,
one it does not — and what is induced is where each slot's filler turned up in the reading that
already existed. Untaught, none of these shapes is read at all.

Three shapes need no mapping because no role moves in them, and they are read structurally beside
:func:`_locative` and :func:`_possessive`: a copula with a predicate (:func:`_predication`), which
is what *"He was tired."* is. **As a fallback and never an override** — put ahead of the compiler
it read *"When I visited Delhi last year I was tired."* as a claim about ``tired`` and lost the
claim the speaker actually made.

**9 · A turn is a reply, and what it replies to decides what it is** (:class:`Exchange`,
:mod:`nyxara.social.common_ground`). The Master listed seventeen things a conversation does and
eleven of them were unreachable, for one reason: this module read every turn **alone**. What makes
a turn a clarification rather than an answer is the turn before it. So adjacency pairs are induced
from demonstrated exchanges — over the **acts**, never the sentences — and the negative half is
the point: a reply she has no pairing for is a non-sequitur and she says so, while a turn
following an act she has been shown nothing about draws no complaint at all.

Beside it, the other half of *how much to say*: what the hearer already has is left unsaid.
:mod:`nyxara.social.common_ground` has modelled given-versus-new since it was written and nothing
in this package had ever put a sentence into it. Now :meth:`Communicator.say` proposes what it
says and the hearer's next turn acknowledges it, which is Clark's mechanism run by the module that
implements it. Measured: the same meaning to the same hearer, eighteen words the first time and
two the second, with the claim itself unchanged — and a fresh hearer still told everything.

**10 · She expects the turn before it arrives, and is corrected by it** (:class:`Anticipation`).
The Master's fifth deep mechanism — ``PREDICTION → OBSERVATION → ERROR → MODEL UPDATE`` — and
this module did not have it at all. Three things are predicted about a turn before it is looked
at: what the speaker will be *doing*, what they will be *about*, and what it will do to the
ledger. Everything is predicted from **counts she has kept herself**: no transition table ships,
an exchange where a question is followed by an answer teaches that, and an organ that has heard
nothing predicts nothing and says so. A prediction she declined to make scores neither way, and
one about a field the turn does not carry — a question has no ledger verdict — is not a miss.

**And the prediction does work rather than only being scored.** Where the surface carries no
convention for its shape, a confident expectation supplies the act, and the reading says so.
That is deliberately allowed **only** into a gap: a convention that was actually demonstrated is
never overridden by a habit. It is also graded on evidence it did not produce — the act recorded
is the one the *sentence* carried, never the one the expectation supplied, because scoring a
prediction against a reading it wrote is the same self-confirming loop :class:`Figure` was found
to have. Measured with that loop closed: the control exchange's act distribution collapsed to one
successor and reported 1.00 confidence on a sequence built to be unpredictable.

**11 · And where one sentence has two readings, it has two** (:func:`attachment`). *"I saw the man
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
    "GAP", "MORE", "LEVELS", "REGISTERS", "LICENCE", "UNIVERSAL",
    "Readings", "attachment", "AttachLearner", "Marker", "MarkerLearner",
    "Anticipation", "Expectation", "Surprise", "Exchange", "Pair", "Reply",
    "Alternation", "Frame", "Mapping", "frames_of",
    "Connective", "Link", "CAUSE_BEFORE", "CAUSE_AFTER", "GOAL_AFTER",
    "free_words", "clause_proposition",
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

#: The two roles a learned marker can carry — see :class:`MarkerLearner`.
#:
#: **Neither is shipped.** V.26 as first written held ``_CHANGE = {"now", "currently", …}`` and
#: ``_UNIVERSAL_NEG = {"never"}``, and those are not closed-class tables like the pronouns below:
#: they are *semantic claims about particular words*, written by hand, in one language. A module
#: that ships them has not learned that ``now`` licenses an update — it has been told, and it can
#: never be told anything else. Both are now induced from demonstrated verdicts by the same rule
#: everything else here is kept by: two demonstrations minimum, differing in content, unanimous,
#: and dropped the moment one demonstration disagrees.
LICENCE = "licence"
UNIVERSAL = "universal"

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


def _content_of(meaning: Optional[Meaning]) -> Set[str]:
    """Every word the reading claimed, so what is left over is the sentence's *marking*."""
    words: Set[str] = set()
    try:
        if meaning is None:
            return words
        for part in (meaning.subject, meaning.relation, meaning.object, meaning.verb):
            for token in str(part or "").split():
                words.add(token)
        for filler in (meaning.roles or {}).values():
            for token in str(filler).split():
                words.add(token)
    except Exception:  # noqa: BLE001
        return words
    return words


def free_words(surface: str, meaning: Optional[Meaning] = None) -> Set[str]:
    """The words of a sentence that its reading did **not** account for.

    A claim is what the reading extracted; everything else in the sentence is how the speaker
    *marked* that claim — the tense, the negator, the adverb that says time has moved. Those
    leftovers are the candidates a marker can be induced from, and taking them as *whatever the
    reading did not use* means no list decides which words may be markers.
    """
    try:
        meaning = compile_meaning(surface) if meaning is None else meaning
        claimed = _content_of(meaning)
        return {t.text for t in tag_tokens(surface) if t.text not in claimed}
    except Exception:  # noqa: BLE001
        return set()


@dataclass
class Marker:
    """One word induced as carrying a verdict, with the demonstrations behind it."""

    word: str = ""
    role: str = ""
    demonstrations: Tuple[Tuple[str, ...], ...] = ()

    @property
    def support(self) -> int:
        return len(set(self.demonstrations))

    def to_dict(self) -> Dict[str, Any]:
        return {"word": self.word, "role": self.role, "support": self.support}


class MarkerLearner:
    """Which words license an update, and which make a claim immune to one — **induced**.

    This class exists because the first version of this module did not have it. It shipped
    ``{"now", "currently", "anymore", …}`` and ``{"never", "always"}`` as constants, and those
    are not the closed-class tables :mod:`nyxara.njp.semantics` defends: a determiner list is a
    statement about a language's grammar, and *"``now`` means the world moved"* is a statement
    about what a word **means**. Written by hand it is a script with a mechanism's name on it,
    and it cannot be wrong in a way that any measurement could show.

    **What a demonstration is.** Two sentences and the verdict a speaker would give them:
    ``updates``, ``contradicts``, or ``corroborates``. Nothing says which word did it.

    **What is induced.** The words that *differ* between the two sentences, in the direction the
    verdict points:

    * an ``updates`` demonstration credits :data:`LICENCE` to the words the **second** sentence
      has that the first does not — a licence is something the later claim carries;
    * a ``contradicts`` demonstration credits :data:`UNIVERSAL` to the words the **first**
      sentence has that the second does not — a universal is what makes the earlier claim refuse
      to be superseded;
    * every other demonstration credits the same differences to ``none``, which is the negative
      evidence that does most of the work here.

    **Why the differences and not the whole sentence.** Both sentences of *"the key is in the
    drawer"* / *"the key is now on the table"* contain ``is``, so crediting whole sentences would
    make the copula a change marker and every copular claim licensed. The difference is ``now``
    and the preposition — and the preposition is contested away the moment two demonstrations
    vary it, which is exactly why the syllabus varies it.

    **Nothing is kept on one demonstration.** Two minimum, differing in content, unanimous in
    role. A word demonstrated two ways is contested and is never consulted again — the rule
    :class:`ActLearner` applies to conventions and :class:`~nyxara.njp.language.Grammar` applies
    to constructions, applied here to markers.

    **Untaught, it says nothing, and that degrades honestly.** With no markers induced, no claim
    carries a licence and none is universal — so a polarity flip on the same object is still a
    contradiction (nothing had to be marked for that), and a second value for a relation is
    *new information* rather than an update. That is the right floor: it is what a listener who
    has never heard anybody signal a change would conclude.
    """

    min_support = 2

    def __init__(self, *, min_support: Optional[int] = None) -> None:
        if min_support is not None:
            self.min_support = max(1, int(min_support))
        #: ``word → role → [content signature, …]``, everything ever demonstrated.
        self.seen: Dict[str, Dict[str, List[Tuple[str, ...]]]] = {}
        self.kept: Dict[str, Marker] = {}
        self.contested: Set[str] = set()
        self.shown = 0

    def show(self, first: str, second: str, verdict: str) -> None:
        """One demonstration: these two sentences, and what a speaker would call the pair."""
        try:
            verdict = str(verdict or "").strip().lower()
            before, after = free_words(first), free_words(second)
            gained, lost = after - before, before - after
            signature = tuple(sorted(_content_of(compile_meaning(first))
                                     | _content_of(compile_meaning(second))))
            role_of: List[Tuple[Set[str], str]] = []
            if verdict == "updates":
                role_of = [(gained, LICENCE), (lost, "none")]
            elif verdict == "contradicts":
                role_of = [(lost, UNIVERSAL), (gained, "none")]
            else:
                role_of = [(gained, "none"), (lost, "none")]
            for words, role in role_of:
                for word in words:
                    self.seen.setdefault(word, {}).setdefault(role, []).append(signature)
            self.shown += 1
            self._settle()
        except Exception:  # noqa: BLE001
            return

    def _settle(self) -> None:
        kept: Dict[str, Marker] = {}
        contested: Set[str] = set()
        for word, roles in self.seen.items():
            live = [role for role, demos in roles.items() if demos]
            if len(live) > 1:
                contested.add(word)
                continue
            role = live[0]
            if role == "none":
                continue
            demos = tuple(roles[role])
            if len(set(demos)) < self.min_support:
                continue
            kept[word] = Marker(word=word, role=role, demonstrations=demos)
        self.kept, self.contested = kept, contested

    def role(self, word: str) -> str:
        found = self.kept.get(str(word or "").strip().lower())
        return found.role if found is not None else ""

    def carries(self, words: Sequence[str], role: str) -> bool:
        return any(self.role(word) == role for word in words)

    def stats(self) -> Dict[str, Any]:
        return {"shown": self.shown, "kept": {w: m.role for w, m in self.kept.items()},
                "contested": len(self.contested)}


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

    #: How far apart the top two candidates must be before a pronoun is treated as settled, and
    #: how much each cue is worth. **Starting values, not constants** — :meth:`fit` moves all
    #: three from demonstrations, and until it is called these are a prior rather than a claim.
    #:
    #: They were three hand-tuned numbers in the first version of this module, which is the same
    #: objection the marker tables answered: a threshold nobody can measure wrong is a threshold
    #: that is not doing any work. At the fitted values *"Ravi met Arun. He was tired."* comes back
    #: ambiguous with both ranked, which is the honest reading of that discourse and the one a
    #: confident resolver would destroy.
    margin = 0.30
    #: **Neutral until fitted.** Both cues start at zero, so an untaught resolver ranks on recency
    #: alone and abstains wherever recency ties — which is the honest floor and is what makes the
    #: syllabus's ``reference`` subject a lesson rather than a report that the constants shipped
    #: were already right. What :meth:`fit` finds from demonstrations is *not* what was hand-tuned
    #: here first: topicality turns out to be worth several times role parallelism, and the
    #: here first: fitted on demonstrations whose recency is controlled, role parallelism comes
    #: out at **zero** — the evidence does not support it at all — and the whole of the work is
    #: done by topicality and by a much narrower margin. The hand-tuned 0.35 was not measured.
    cues: Dict[str, float] = {"parallel": 0.0, "topical": 0.0}

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
                    score += float(self.cues.get("parallel", 0.0))
                if referent.mentions > 1:
                    score += float(self.cues.get("topical", 0.0))
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

    # -- learning the weights ------------------------------------------------ #
    def fit(self, cases: Sequence[Tuple[Sequence[Referent], str, str, str, str]]) -> Dict[str, Any]:
        """Move :attr:`cues` and :attr:`margin` to whatever answers these demonstrations best.

        A case is ``(referents, pronoun, role, clause_subject, expected)``, where ``expected`` is
        the name the discourse settles on or ``""`` when it settles on nothing. **Both kinds are
        needed**, and the reason is weaker and more exact than it first looks: on cases that all
        resolve, a setting that *never* abstains fits the data perfectly and nothing excludes it.
        The search returns the smallest weights that fit, so it often will not choose one — but it
        is not being stopped from it. One case the discourse does not settle rules the whole family
        out, which is why the ambiguous cases are half the lesson exactly as they are half the
        exam.

        A small deterministic grid rather than anything cleverer. Three parameters and a few dozen
        demonstrations do not support more, and a search that reported a better number than the
        data can carry would be the same dishonesty as a hand-tuned constant with a formula in
        front of it.
        """
        best = (self.score(cases), dict(self.cues), self.margin)
        for parallel in [i / 20 for i in range(0, 13)]:
            for topical in [i / 20 for i in range(0, 9)]:
                for margin in [i / 20 for i in range(1, 13)]:
                    scratch = Reference(speaker=self.speaker, addressee=self.addressee)
                    scratch.plausible = self.plausible
                    scratch.cues = {"parallel": parallel, "topical": topical}
                    scratch.margin = margin
                    got = scratch.score(cases)
                    if got > best[0]:
                        best = (got, dict(scratch.cues), scratch.margin)
        self.cues, self.margin = best[1], best[2]
        return {"fitted": best[0], "cues": dict(self.cues), "margin": self.margin,
                "cases": len(cases)}

    def score(self, cases: Sequence[Tuple[Sequence[Referent], str, str, str, str]]) -> float:
        """How many of these demonstrations the current weights get right, as a rate."""
        right = 0
        for referents, pronoun, role, clause_subject, expected in cases:
            scratch = Reference(speaker=self.speaker, addressee=self.addressee)
            scratch.plausible = self.plausible
            scratch.cues, scratch.margin = dict(self.cues), self.margin
            scratch.referents = list(referents)
            scratch.turn = max([r.turn for r in referents] or [0]) + 1
            got = scratch.resolve(pronoun, role=role, clause_subject=clause_subject)
            if (got.referent if got.settled else "") == expected:
                right += 1
        return right / len(cases) if cases else 0.0

    def stats(self) -> Dict[str, Any]:
        return {"referents": len(self.referents), "turn": self.turn,
                "cues": dict(self.cues), "margin": round(self.margin, 4),
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
    #: The claim quantifies over **all** times. A universal is not an instance and cannot be
    #: superseded by one. Set from a word :class:`MarkerLearner` induced, never from a list.
    universal: bool = False
    #: The speaker signalled that time has moved — see :data:`_CHANGE`. This is the licence that
    #: makes a later claim an update rather than a contradiction, and it has to be **in the
    #: sentence**: inferring it from the topic is how a store silently overwrites its evidence.
    change: bool = False
    #: The words of this sentence its own reading did not account for — see :func:`free_words`.
    #: This is what a marker is looked up in, and it is carried on the claim so that a learner
    #: taught **after** the claim was filed still sees the evidence it needs.
    markers: Tuple[str, ...] = ()
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


def read_claim(surface: str, meaning: Optional[Meaning] = None, *, turn: int = 0,
               markers: Optional["MarkerLearner"] = None,
               alternation: Optional["Alternation"] = None) -> Optional[Claim]:
    """A sentence read as a claim, with its universal and its change licence **looked up**.

    Returns ``None`` for anything that is not a complete assertion — a question asserts nothing,
    and a claim built out of one would put the questioner's presupposition on the ledger as
    though they had stated it.

    ``markers`` is the induced table. Passing ``None`` means no word licenses anything, which is
    the correct reading for a listener who has never been shown a speaker signalling a change —
    not a degraded one.
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
        if found is None and alternation is not None:
            # A shape the compiler has no frame for, read through a mapping she was shown — see
            # :class:`Alternation`. Untaught this is ``None`` and the sentence stays unreadable,
            # which is where this module stood before the mappings existed.
            found = alternation.read(surface)
        if found is None:
            # And the shapes this module reads structurally: a copula with a predicate, a
            # location, a possession. *"He was tired."* is one of the commonest sentences in
            # English and had no reading at all.
            for reader in (_predication, _locative, _possessive):
                shape = reader(surface)
                if shape is not None:
                    shaped_subject, shaped_relation, shaped_object = shape
                    found = Meaning(kind="assertion", subject=shaped_subject,
                                    relation=shaped_relation, object=shaped_object,
                                    confidence=0.7, frame="structural")
                    break
        if found is None:
            return None
        meaning = found
        marking = tuple(sorted(free_words(surface, meaning)))
        universal = bool(markers is not None and markers.carries(marking, UNIVERSAL))
        change = bool(markers is not None and markers.carries(marking, LICENCE))
        subject, relation, obj = (_untimed(_bare(meaning.subject)), meaning.relation,
                                  _untimed(_bare(meaning.object)))
        # Only the locative overrides a reading that already succeeded. Adding the predication
        # here read *"When I visited Delhi last year I was tired."* as a claim about `tired`,
        # because the sentence ends in a copula and one word — and the claim the speaker actually
        # made, inside the fronted clause, was lost. A structural shape is a **fallback** for a
        # sentence nothing else reads, not an override of one that was read.
        placed = _locative(surface)
        if placed is not None:
            subject, relation, obj = placed
        return Claim(subject=subject, relation=relation,
                     object=obj, negated=bool(meaning.negated),
                     universal=bool(universal), change=bool(change), markers=marking,
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

    def kind_of(self, relation: str) -> str:
        """``state`` · ``event`` · ``""`` — induced from how the relation has **behaved**.

        The Master's anchor separates an event from a state, and the difference is not in any
        word: it is that a state holds one value at a time and an event happens more than once.
        Both of those are visible in the ledger without anything being declared. A relation a
        speaker has marked a change on, or one that produced a conflict when re-asserted
        differently, is a state. A relation that carries several live values for one subject and
        nobody has treated as a conflict is an event — things happen repeatedly, and a second
        happening does not deny the first.

        The empty string where the transcript has not shown either, which is most relations most
        of the time and is the honest answer for them.
        """
        try:
            relation = str(relation or "")
            if not relation:
                return ""
            if relation in self.stateful:
                return "state"
            if any(prior.relation == relation for prior, _now in self.conflicts):
                return "state"
            if not self.stateful:
                # Several live values look identical for a state nobody marked a change on and
                # for an event that simply happened twice. Until she has seen at least one marked
                # change *somewhere*, she has no way to tell those apart, and the honest answer is
                # that she cannot. Measured without this: a location that had moved once, with no
                # markers taught, came back an `event`.
                return ""
            live: Dict[str, Set[str]] = {}
            for claim in self.claims:
                if claim.relation != relation or claim.superseded or claim.negated:
                    continue
                live.setdefault(claim.subject, set()).add(claim.object)
            if any(len(values) > 1 for values in live.values()):
                return "event"
            return ""
        except Exception:  # noqa: BLE001
            return ""

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


def clause_proposition(clause: str) -> Tuple[str, str]:
    """:func:`proposition`, but only for a string that is genuinely a **clause**.

    This is what replaced the attitude-verb list. A complement is a clause when it has a
    predicate of its own — a copula (:func:`_locative`), a possessive (:func:`_possessive`), or a
    transitive reading with a real object. What it must **not** be is the compiler's bare
    ``np-verb`` frame, which fires on any two nouns in a row: *"the box in the room"* reads as
    ``box --room-->`` and *"arun the book"* as ``arun --book-->``, so accepting that frame made
    *"ravi opened the box in the room"* an attributed belief.

    A verb list is not needed for this and never was. *"Thinks"* is not an attitude verb because
    it is in a table; it is one because what follows it is a whole proposition, and *"opened"* is
    not one because what follows it is a noun phrase.
    """
    try:
        text = str(clause or "").strip()
        for reader in (_locative, _possessive, _predication):
            found = reader(text)
            if found is not None:
                subject, relation, obj = found
                return f"{subject}|{relation}", _untimed(obj)
        meaning = compile_meaning(text)
        if meaning.readable and meaning.complete and meaning.frame != "np-verb":
            key = f"{_untimed(_bare(meaning.subject))}|{meaning.relation}"
            value = _untimed(_bare(meaning.object))
            return key, (f"¬{value}" if meaning.negated else value)
        return "", ""
    except Exception:  # noqa: BLE001
        return "", ""


def proposition(clause: str) -> Tuple[str, str]:
    """A clause as ``(key, value)`` — ``"box|have"`` and ``"red ball"``.

    Keyed on subject and relation with the **object as the value**, which is the shape a
    false-belief question needs: *"Ravi thinks the box has the blue ball"* is not a belief about a
    different proposition from *"the box has the red ball"*, it is a different **answer to the same
    one**, and a key that folded the object in could never see the two meet.
    """
    try:
        text = str(clause or "").strip()
        for reader in (_locative, _possessive, _predication):
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


def _attitude(surface: str) -> Optional[Tuple[str, bool, str]]:
    """``(holder, negated, clause)`` for a sentence that embeds one, or ``None``.

    **Structural, with no verb list.** A word introduces an attitude when somebody is named in
    front of it and what follows it is a whole clause — see :func:`clause_proposition`. That is
    the definition of a clausal complement, and it is the same move this package makes
    everywhere: *"tag the closed class exhaustively, treat everything else as open material,
    match over the structure rather than over the words."*

    V.26 as first written held a dictionary of thirty-odd attitude verbs and called it closed in
    the sense modals are. It is not: ``think``, ``believe``, ``reckon``, ``suspect``, ``gather``,
    ``figure``, ``hold``, ``maintain``, ``sochta``, ``maanta`` — the class takes new members, it
    differs by register, and a sentence embedding a belief with a verb nobody listed was read as
    a flat assertion. It is now read by what it does.

    The one thing lost with the list is that an **intransitive** complement — *"ravi thinks the
    ball breaks"* — is refused, because to this compiler it is indistinguishable from two nouns
    in a row. Refusing is the safe direction: the alternative reads *"ravi opened the box in the
    room"* as a belief.
    """
    try:
        tokens = tag_tokens(surface)
        for index, token in enumerate(tokens):
            if index == 0 or token.tag != Tag.WORD:
                continue
            head = tokens[:index]
            holder = [t for t in head if t.tag in (Tag.WORD, Tag.PRON)]
            if not holder:
                continue
            rest = tokens[index + 1:]
            while rest and rest[0].text in ("that", "ki", "कि"):
                rest = rest[1:]
            clause = " ".join(t.text for t in rest).strip()
            if not clause or not clause_proposition(clause)[0]:
                continue
            negated = any(t.tag == Tag.NEG for t in head)
            return holder[-1].text, negated, clause
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
                holder, negated, clause = found
                path.append(self._name(holder))
                if negated:
                    # *"Ravi does not think X"* and *"Ravi thinks I do not know X"* both attribute
                    # **no belief** in X, which is a different state from believing not-X and is
                    # stored as the one it is. The distinction between ``think`` and ``know`` was
                    # carried in the verb table and was never read: both branches set this flag,
                    # so the table's second column did nothing and has gone with it.
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
    #: Which it left out because the hearer already had them.
    omitted: Tuple[str, ...] = ()
    #: Every elaboration this register would say to a hearer who had nothing.
    extras: Tuple[str, ...] = ()

    @property
    def words(self) -> int:
        return len(str(self.text or "").split())

    def to_dict(self) -> Dict[str, Any]:
        out = {"audience": self.audience, "text": self.text, "claim": self.claim,
               "verified": self.verified, "carried": list(self.carried),
               "words": self.words}
        if self.omitted:
            out["omitted"] = list(self.omitted)
        return out


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
    def say(self, meaning: Meaning, audience: str = "student",
            given: Any = ()) -> Said:
        """One rendering. ``given`` is what the hearer already has, and is left unsaid.

        This is the half of *how much to say* that scope alone cannot reach. Two hearers who need
        the same amount are still owed different sentences if one of them has already been told
        the condition — repeating it is not thoroughness, it is not listening. What counts as
        already-had is :mod:`nyxara.social.common_ground`'s business, not this class's.
        """
        given = frozenset(str(x) for x in (given or ()))
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
            candidates: List[Tuple[str, str]] = []
            if rank >= 1 and meaning.condition:
                candidates.append(("condition", f"this holds if {meaning.condition}"))
            if rank >= 2:
                if meaning.temporal:
                    candidates.append(("temporal", f"as of {meaning.temporal}"))
                if meaning.modality:
                    candidates.append(
                        ("modality", f"held as {meaning.modality} rather than certain"))
            if rank >= 3:
                candidates.append(("evidential",
                                   f"on {meaning.evidential or 'stated'} grounds, "
                                   f"parse confidence {meaning.confidence:.2f}"))
            extras, carried, omitted = [], ["claim"], []
            for name, text in candidates:
                if text in given:
                    omitted.append(name)
                    continue
                extras.append(text)
                carried.append(name)
            out.text = "; ".join([core] + extras)
            out.carried = tuple(carried)
            out.omitted = tuple(omitted)
            out.extras = tuple(text for _name, text in candidates)
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

    **And the guard must be correctable, which the first version of it was not.** Refusing to file
    a flagged sentence suppresses exactly the evidence that would show the relation is broader than
    its witnesses, so the first exception is indistinguishable from the hundredth and no amount of
    counter-evidence can ever reach it. Measured, that reached three organs away: ten birds
    witnessed for ``need``, then **14 of the next 20 ordinary sentences about plants needing
    light** were suppressed as metaphors, :mod:`nyxara.njp.discover` never saw a
    counterexample,
    and every kind rule it proposed was confirmed and none refuted. So a flagged reading is
    **recorded** (:meth:`note_exception`) even though it is not filed, and a kind seen violating
    a relation :attr:`min_witnesses` times stops being a violation of it. *"Roses need light"* is
    news, not metaphor; *"the market swallowed the shock"* stays metaphor until several more
    institutions swallow things.

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
        #: ``relation → kind → {subject, …}`` — the subjects already flagged as violating a
        #: relation, so the guard can be corrected by evidence instead of confirming itself.
        self.exceptions: Dict[str, Dict[str, Set[str]]] = {}
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
            # A kind that keeps turning up as a violation is not a violation. It is the language
            # being broader than the witnesses so far, and *"roses need light"* is news rather
            # than metaphor. Below the threshold a reading is flagged; at or above it the kind is
            # ordinary for this relation and stays ordinary.
            known = self.exceptions.get(relation, {})
            for kind in sorted(mine):
                already = known.get(kind, set()) - {subject}
                if len(already) >= self.min_witnesses:
                    out.why = (f"{len(already)} other subjects of kind {kind!r} have taken "
                               f"{relation!r} already — the relation is broader than its "
                               f"witnesses rather than being stretched")
                    return out
            out.figurative = True
            out.violated = sorted(shared)[0]
            out.why = (f"every one of {len(seen)} witnessed subjects of {relation!r} is a "
                       f"{out.violated}, and {subject!r} is not")
            self.flagged += 1
            return out
        except Exception:  # noqa: BLE001
            return out

    def note_exception(self, judged: Figurative) -> None:
        """Record a flagged reading, so the guard can be corrected by evidence.

        Deliberately separate from :meth:`judge`, which stays pure so
        :meth:`Communicator.figurative` remains the read-only store guard it is documented as. A
        sentence is counted once, by the caller that actually heard it.
        """
        try:
            if not judged.figurative or not judged.subject:
                return
            for kind in self._kinds(judged.subject):
                bucket = self.exceptions.setdefault(judged.relation, {}).setdefault(kind, set())
                bucket.add(judged.subject)
        except Exception:  # noqa: BLE001
            return

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
                "witnesses": sum(len(v) for v in self.witnesses.values()),
                "exceptions": {r: {k: len(v) for k, v in kinds.items()}
                               for r, kinds in self.exceptions.items()}}


# --------------------------------------------------------------------------- #
# the meaning adversary — two readings of one sentence
# --------------------------------------------------------------------------- #

class AttachLearner:
    """Which prepositions attach two ways — **discovered from demonstrations that disagree**.

    V.26 as first written held ``_AMBIGUOUS_PREP = {"with"}`` and defended the narrowness in a
    docstring. That is a script: one English preposition, chosen by hand, and no measurement could
    ever have shown it wrong.

    Here a demonstration is a surface and which site the phrase attached to — ``event`` (it
    modifies what happened) or ``object`` (it modifies the thing). A preposition demonstrated
    **both ways, with different content**, is one the language genuinely uses both ways, and that
    is what ambiguity *is*. The mechanism is :class:`ActLearner`'s contest, reused a third time:
    a shape two lessons disagree about is not a shape this package reads confidently.

    Untaught, nothing is ambiguous and no sentence is questioned — which is right. A listener who
    has only ever heard *"with"* used one way has no reason to ask.
    """

    min_support = 2

    def __init__(self, *, min_support: Optional[int] = None) -> None:
        if min_support is not None:
            self.min_support = max(1, int(min_support))
        self.seen: Dict[str, Dict[str, List[Tuple[str, ...]]]] = {}
        self.ambiguous: Set[str] = set()
        self.shown = 0

    def show(self, surface: str, attachment_site: str) -> None:
        """One demonstration: in this sentence the phrase modified the event, or the object."""
        try:
            site = str(attachment_site or "").strip().lower()
            tokens = tag_tokens(surface)
            fillers = fillers_of(surface)
            for token in tokens:
                if token.tag != Tag.PREP:
                    continue
                self.seen.setdefault(token.text, {}).setdefault(site, []).append(fillers)
            self.shown += 1
            self._settle()
        except Exception:  # noqa: BLE001
            return

    def _settle(self) -> None:
        found: Set[str] = set()
        for word, sites in self.seen.items():
            live = [site for site, demos in sites.items()
                    if len(set(demos)) >= self.min_support]
            if len(live) > 1:
                found.add(word)
        self.ambiguous = found

    def stats(self) -> Dict[str, Any]:
        return {"shown": self.shown, "ambiguous": sorted(self.ambiguous),
                "prepositions": len(self.seen)}


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


def attachment(surface: str, meaning: Optional[Meaning] = None, *,
               learner: Optional[AttachLearner] = None) -> Readings:
    """*"I saw the man with the telescope."* — read both ways, and neither chosen.

    The evidence that there is an ambiguity at all is **in the compiler's own output**: it has no
    frame for a prepositional phrase after an object, so it fuses the two noun phrases into one
    object string — ``man telescope``, with ``with the`` silently dropped. That fusion is a
    reliable signal, and reading it back out gives both attachment sites without needing a parser
    this package does not have.

    **Which prepositions this fires on is learned** — see :class:`AttachLearner`. Untaught, none
    of them, so a fused object is read exactly as it was before this function existed.

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
        known = learner.ambiguous if learner is not None else frozenset()
        preps = [i for i, t in enumerate(tokens)
                 if t.tag == Tag.PREP and t.text in known]
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
# 8 · connectives — what a joining word does to the two clauses either side
# --------------------------------------------------------------------------- #

#: The three things a joining word has been demonstrated doing. Not a table of words: a table of
#: **roles**, which is a different object — nothing here says which word plays which, and
#: :class:`Connective` will not read a word it has not been shown.
CAUSE_BEFORE = "cause-before"
CAUSE_AFTER = "cause-after"
GOAL_AFTER = "goal-after"


@dataclass
class Link:
    """Two clauses, and what the sentence said about how they go together."""

    kind: str = ""              # ``cause`` · ``goal``
    connective: str = ""
    cause: str = ""
    effect: str = ""
    goal: str = ""
    why: str = ""

    def to_dict(self) -> Dict[str, Any]:
        out = {"kind": self.kind, "connective": self.connective}
        for name in ("cause", "effect", "goal", "why"):
            value = getattr(self, name)
            if value:
                out[name] = value
        return out


def _splits(surface: str) -> List[Tuple[str, str, str]]:
    """``(left, token, right)`` at every position where both sides read as a clause.

    Structural and exhaustive: every position is tried and none is privileged, which is the rule
    :mod:`nyxara.njp.semantics` and :mod:`nyxara.njp.compile` both state. What a particular token
    *does* at such a split is not decided here.
    """
    out: List[Tuple[str, str, str]] = []
    try:
        tokens = tag_tokens(surface)
        for index in range(1, len(tokens) - 1):
            # A determiner is not a connective, and neither is a preposition, a pronoun or an
            # auxiliary — they belong **inside** a clause rather than between two. Without this,
            # ``the`` was induced as a causal connective: it splits *"devi lit | the | lamp
            # because the room was dark"* into two halves that both happen to read, it sat
            # earlier in the list than ``because``, and it won — giving a cause of ``lamp
            # because the room was dark``. A structural constraint over the closed class, which
            # is where a constraint of that kind belongs.
            if tokens[index].tag in (Tag.DET, Tag.PREP, Tag.PRON, Tag.AUX, Tag.NEG):
                continue
            left = " ".join(t.text for t in tokens[:index])
            right = " ".join(t.text for t in tokens[index + 1:])
            if not left or not right:
                continue
            if proposition(left)[0] and proposition(right)[0]:
                out.append((left, tokens[index].text, right))
        return out
    except Exception:  # noqa: BLE001
        return out


def _infinitivals(surface: str) -> List[Tuple[str, str, str]]:
    """``(main, token, reconstructed goal clause)`` for every ``PREP WORD …`` tail.

    *"Ravi went to the shop to buy bread"* has two prepositions and only one of them introduces a
    purpose; the difference is structural — a purpose clause is a preposition followed straight by
    an open word, where a prepositional phrase has a determiner in between. The goal clause has no
    subject of its own, so the main clause's is put back for it, which is what makes it readable
    at all.
    """
    out: List[Tuple[str, str, str]] = []
    try:
        tokens = tag_tokens(surface)
        for index in range(1, len(tokens) - 1):
            if tokens[index].tag != Tag.PREP or tokens[index + 1].tag != Tag.WORD:
                continue
            head = tokens[:index]
            main = " ".join(t.text for t in head)
            opens = [t.text for t in head if t.tag == Tag.WORD]
            if not opens:
                continue
            tail = " ".join(t.text for t in tokens[index + 1:])
            clause = f"{opens[0]} {tail}"
            # Only the reconstructed goal clause has to read. Requiring the main clause to read as
            # well threw away *"Ravi went to the shop to buy bread"* — the compiler has no frame
            # for a verb with a prepositional phrase, so the main half of the commonest purpose
            # sentence there is came back unreadable and the goal went with it.
            if len(opens) >= 2 and proposition(clause)[0]:
                out.append((main, tokens[index].text, clause))
        return out
    except Exception:  # noqa: BLE001
        return out


class Connective:
    """What a joining word does — **induced**, because which word does which is a language fact.

    *"The farmer watered the plant because it was dry"* and *"The plant was dry so the farmer
    watered it"* say the same thing with the cause on opposite sides. Nothing structural
    distinguishes them: both are two clauses with a word between. What differs is ``because``
    against ``so``, and which of those puts the cause first is a fact about English that a module
    could only have been told — so it is shown instead.

    A demonstration is a sentence and which of its clauses is the **cause** (or the **goal**).
    What is induced is the role of the word at the split, at the same terms as everything else
    here: two demonstrations minimum, differing in content, unanimous, dropped the moment one
    disagrees. Untaught, no sentence carries a cause and no sentence carries a goal, which is
    exactly where this module stood.

    Purpose is the same mechanism over :func:`_infinitivals` rather than :func:`_splits`, because
    a goal clause has no subject of its own and has to be reconstructed before it reads.
    """

    min_support = 2

    def __init__(self, *, min_support: Optional[int] = None) -> None:
        if min_support is not None:
            self.min_support = max(1, int(min_support))
        self.seen: Dict[str, Dict[str, List[Tuple[str, ...]]]] = {}
        self.kept: Dict[str, str] = {}
        #: How many distinct demonstrations stand behind each kept token. Read by :meth:`read`,
        #: which picks the **best-supported** candidate rather than the leftmost — the rule
        #: :mod:`nyxara.njp.semantics` and :mod:`nyxara.njp.compile` both state, because ``or``
        #: short-circuits and a short-circuit is how a sentence gets read by the wrong pattern.
        self.support: Dict[str, int] = {}
        self.contested: Set[str] = set()
        self.shown = 0

    @staticmethod
    def _same(one: str, other: str) -> bool:
        return {t.text for t in tag_tokens(one) if t.tag == Tag.WORD} == \
            {t.text for t in tag_tokens(other) if t.tag == Tag.WORD}

    def show(self, surface: str, *, cause: str = "", goal: str = "") -> None:
        """One demonstration: in this sentence, **this** clause is the cause (or the goal)."""
        try:
            signature = tuple(sorted(t.text for t in tag_tokens(surface) if t.tag == Tag.WORD))
            if cause:
                for left, token, right in _splits(surface):
                    if self._same(left, cause):
                        self._note(token, CAUSE_BEFORE, signature)
                    elif self._same(right, cause):
                        self._note(token, CAUSE_AFTER, signature)
            if goal:
                for _main, token, clause in _infinitivals(surface):
                    if self._same(clause, goal) or goal in clause:
                        self._note(token, GOAL_AFTER, signature)
            self.shown += 1
            self._settle()
        except Exception:  # noqa: BLE001
            return

    def _note(self, token: str, role: str, signature: Tuple[str, ...]) -> None:
        self.seen.setdefault(token, {}).setdefault(role, []).append(signature)

    def _settle(self) -> None:
        kept: Dict[str, str] = {}
        support: Dict[str, int] = {}
        contested: Set[str] = set()
        for token, roles in self.seen.items():
            live = [role for role, sigs in roles.items() if sigs]
            if len(live) > 1:
                contested.add(token)
                continue
            sigs = roles[live[0]]
            if len(set(sigs)) >= self.min_support:
                kept[token] = live[0]
                support[token] = len(set(sigs))
        self.kept, self.contested, self.support = kept, contested, support

    def read(self, surface: str) -> Optional[Link]:
        """The cause or the goal this sentence carries, or ``None``."""
        try:
            # Every candidate is tried and the winner is chosen on **evidence, never on order**.
            # Taking the leftmost split let a token with two demonstrations behind it beat one
            # with four, purely because it occurred earlier in the sentence.
            causal = [(self.support.get(token, 0), left, token, right)
                      for left, token, right in _splits(surface)
                      if self.kept.get(token) in (CAUSE_BEFORE, CAUSE_AFTER)]
            if causal:
                _best, left, token, right = max(causal, key=lambda item: item[0])
                if self.kept[token] == CAUSE_BEFORE:
                    return Link(kind="cause", connective=token, cause=left, effect=right,
                                why=f"{token!r} was shown putting the cause first")
                return Link(kind="cause", connective=token, cause=right, effect=left,
                            why=f"{token!r} was shown putting the cause second")
            for main, token, clause in _infinitivals(surface):
                if self.kept.get(token) == GOAL_AFTER:
                    return Link(kind="goal", connective=token, effect=main, goal=clause,
                                why=f"{token!r} was shown introducing a purpose")
            return None
        except Exception:  # noqa: BLE001
            return None

    def stats(self) -> Dict[str, Any]:
        return {"shown": self.shown, "kept": dict(self.kept),
                "contested": sorted(self.contested)}


# --------------------------------------------------------------------------- #
# 9 · alternations — the same meaning, said a different way round
# --------------------------------------------------------------------------- #

def _predication(surface: str) -> Optional[Tuple[str, str, str]]:
    """``NP be WORD`` read as ``(subject, predicate, "")``, or ``None``.

    *"He was tired."* has no frame in the compiler at all, and it is one of the commonest shapes
    in English. Read here for the same reason :func:`_locative` and :func:`_possessive` are — it
    is a shape over the closed class — and with one property neither of those has: **no relation
    name is invented.** The relation is the predicate word the sentence used, so nothing in this
    function decides what anything means.
    """
    try:
        tokens = tag_tokens(surface)
        for index, token in enumerate(tokens):
            if token.text not in _COPULA or index == 0:
                continue
            rest = [t for t in tokens[index + 1:] if t.tag not in (Tag.ADV,)]
            if len(rest) != 1 or rest[0].tag != Tag.WORD:
                continue
            subject = _bare(" ".join(t.text for t in tokens[:index]))
            if subject:
                return subject, rest[0].text, ""
        return None
    except Exception:  # noqa: BLE001
        return None


@dataclass
class Frame:
    """One structural shape found over the closed class, with its slots unnamed as to role.

    The shape is structure and the **roles are not in it**. *"The window was opened by Ravi"* has
    a noun phrase before the auxiliary, a word after it, and a noun phrase after the preposition —
    and which of those is the agent is a fact about English rather than about the shape. That is
    the split this class exists to keep: detection is structural, interpretation is learned.
    """

    name: str = ""
    slots: Dict[str, str] = field(default_factory=dict)
    marker: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "slots": dict(self.slots), "marker": self.marker}


def frames_of(surface: str) -> List[Frame]:
    """Every alternation shape this sentence matches. Structural, over the closed class only."""
    out: List[Frame] = []
    try:
        tokens = tag_tokens(surface)
        texts = [t.text for t in tokens]
        tags = [t.tag for t in tokens]

        def phrase(span: Sequence[Any]) -> str:
            return _bare(" ".join(t.text for t in span))

        # NP be V PREP NP — the shape a passive has. No preposition is named: which one a
        # language uses for the agent is a fact about that language, and naming ``by`` here would
        # be the table this module spent V.26 removing.
        for index, token in enumerate(tokens):
            if token.text not in _COPULA or index == 0:
                continue
            rest = tokens[index + 1:]
            while rest and rest[0].tag == Tag.ADV:
                rest = rest[1:]
            if len(rest) < 3 or rest[0].tag != Tag.WORD:
                continue
            preps = [j for j, t in enumerate(rest) if t.tag == Tag.PREP]
            if not preps or preps[0] == 0:
                continue
            cut = preps[0]
            left, verb = phrase(tokens[:index]), rest[0].text
            right = phrase(rest[cut + 1:])
            if left and verb and right:
                out.append(Frame(name="be-prep",
                                 slots={"left": left, "verb": verb, "right": right},
                                 marker=rest[cut].text))
                break

        # NP AUX AUX V NP — an auxiliary chain, which is where aspect lives.
        for index in range(len(tokens) - 2):
            if tags[index] != Tag.AUX or tags[index + 1] != Tag.AUX:
                continue
            rest = tokens[index + 2:]
            if not rest or rest[0].tag != Tag.WORD:
                continue
            left, verb = phrase(tokens[:index]), rest[0].text
            right = phrase(rest[1:])
            if left and verb:
                out.append(Frame(name="aux-chain",
                                 slots={"left": left, "verb": verb, "right": right},
                                 marker=f"{texts[index]} {texts[index + 1]}"))
                break

        # NP V NP CONJ NP NP — a coordinate whose second half has no verb of its own.
        for index, tag in enumerate(tags):
            if tag != Tag.CONJ or index < 3:
                continue
            head, tail = tokens[:index], tokens[index + 1:]
            opens = [t for t in tail if t.tag == Tag.WORD]
            first_opens = [t for t in head if t.tag == Tag.WORD]
            if len(opens) != 2 or len(first_opens) != 3:
                continue
            out.append(Frame(name="gapped",
                             slots={"left": first_opens[0].text,
                                    "verb": first_opens[1].text,
                                    "right": first_opens[2].text,
                                    "left2": opens[0].text, "right2": opens[1].text},
                             marker=texts[index]))
            break
        return out
    except Exception:  # noqa: BLE001
        return out


@dataclass
class Mapping:
    """One learned reading of a frame: which slot holds which role."""

    frame: str = ""
    roles: Dict[str, str] = field(default_factory=dict)
    demonstrations: Tuple[Tuple[str, ...], ...] = ()

    @property
    def support(self) -> int:
        return len(set(self.demonstrations))

    def to_dict(self) -> Dict[str, Any]:
        return {"frame": self.frame, "roles": dict(self.roles), "support": self.support}


class Alternation:
    """The same meaning said a different way round, and **which way round is learned**.

    Four shapes were coming back unreadable or mis-read, and every one of them is ordinary
    English: *"The window was opened by Ravi"*, *"Ravi has been opening the door"*, *"Ravi said
    that he was tired"*, *"Ravi opened the door and Arun the window"*. Detecting them is
    structural and belongs with the closed class (:func:`frames_of`). **Reading them is not.**

    That a passive puts the *patient* in front and the *agent* after the preposition is a fact
    about English, not about the shape ``NP be V PREP NP``: a language exists where that shape
    means the opposite, and a module that wrote the mapping down could never be told so. So a
    demonstration is two sentences that **mean the same thing** — one the compiler already reads,
    one it does not — and what is induced is where each slot's filler turned up in the reading it
    already has.

    Two demonstrations minimum, differing in fillers, unanimous; a slot two demonstrations
    disagree about is contested and the frame is not read. Untaught, none of these shapes is read
    at all, which is exactly where this module stood before — the capability arrives with the
    evidence and not with the file.
    """

    min_support = 2

    def __init__(self, *, min_support: Optional[int] = None) -> None:
        if min_support is not None:
            self.min_support = max(1, int(min_support))
        #: ``frame → slot → role → [signature, …]``
        self.seen: Dict[str, Dict[str, Dict[str, List[Tuple[str, ...]]]]] = {}
        self.kept: Dict[str, Mapping] = {}
        self.contested: Set[Tuple[str, str]] = set()
        self.shown = 0

    def show(self, plain: str, marked: str) -> None:
        """Two sentences that mean the same thing; the first is one the compiler already reads."""
        try:
            reading = compile_meaning(plain)
            if not reading.readable or not reading.subject:
                return
            where = {_bare(reading.subject): "subject", reading.relation: "relation",
                     _lemma_of(reading.verb or reading.relation): "relation"}
            if reading.object:
                where[_bare(reading.object)] = "object"
            signature = tuple(sorted(where))
            for frame in frames_of(marked):
                for slot, filler in frame.slots.items():
                    role = where.get(filler) or where.get(_lemma_of(filler))
                    if not role:
                        continue
                    (self.seen.setdefault(frame.name, {}).setdefault(slot, {})
                     .setdefault(role, []).append(signature))
            self.shown += 1
            self._settle()
        except Exception:  # noqa: BLE001
            return

    def _settle(self) -> None:
        kept: Dict[str, Mapping] = {}
        contested: Set[Tuple[str, str]] = set()
        for frame, slots in self.seen.items():
            roles: Dict[str, str] = {}
            demos: List[Tuple[str, ...]] = []
            for slot, found in slots.items():
                live = [role for role, sigs in found.items() if sigs]
                if len(live) > 1:
                    contested.add((frame, slot))
                    continue
                sigs = found[live[0]]
                if len(set(sigs)) < self.min_support:
                    continue
                roles[slot] = live[0]
                demos.extend(sigs)
            if roles:
                kept[frame] = Mapping(frame=frame, roles=roles, demonstrations=tuple(demos))
        self.kept, self.contested = kept, contested

    #: Which slots belong to which clause. A structural fact about the shape rather than about
    #: the language — a gapped coordinate has two clauses by definition, and which words are in
    #: which of them is what :func:`frames_of` found. What each slot *means* is still learned.
    CLAUSES: Dict[str, Tuple[Tuple[str, ...], ...]] = {
        "be-prep": (("left", "verb", "right"),),
        "aux-chain": (("left", "verb", "right"),),
        "gapped": (("left", "verb", "right"), ("left2", "verb", "right2")),
    }

    def readings(self, surface: str) -> List[Meaning]:
        """Every clause this sentence carries, read through learned mappings.

        A list rather than one, because a gapped coordinate is **two** claims and returning one of
        them would silently lose the other — which is what *"Ravi opened the door and Arun the
        window"* did before: it came back as a single claim whose object was ``door arun window``.
        """
        out: List[Meaning] = []
        try:
            for frame in frames_of(surface):
                found = self.kept.get(frame.name)
                if found is None:
                    continue
                for group in self.CLAUSES.get(frame.name, (tuple(frame.slots),)):
                    filled: Dict[str, str] = {}
                    for slot in group:
                        role = found.roles.get(slot)
                        if role and frame.slots.get(slot):
                            filled.setdefault(role, frame.slots[slot])
                    if not (filled.get("subject") and filled.get("relation")):
                        continue
                    out.append(Meaning(kind="assertion", subject=filled["subject"],
                                       relation=_lemma_of(filled["relation"]),
                                       object=filled.get("object", ""),
                                       confidence=0.7, frame=f"learned:{frame.name}",
                                       verb=filled["relation"]))
                if out:
                    return out
            return out
        except Exception:  # noqa: BLE001
            return out

    def read(self, surface: str) -> Optional[Meaning]:
        """The first clause this sentence carries, or ``None`` if no mapping applies."""
        found = self.readings(surface)
        return found[0] if found else None

    def stats(self) -> Dict[str, Any]:
        return {"shown": self.shown, "contested": len(self.contested),
                "kept": {name: dict(m.roles) for name, m in self.kept.items()}}


# --------------------------------------------------------------------------- #
# 9 · the exchange — what counts as a reply to what
# --------------------------------------------------------------------------- #

@dataclass
class Pair:
    """One induced adjacency pair: an act, an act that follows it, and what that following is."""

    first: str = ""
    second: str = ""
    relation: str = ""
    demonstrations: Tuple[Tuple[str, ...], ...] = ()

    @property
    def support(self) -> int:
        return len(set(self.demonstrations))

    def to_dict(self) -> Dict[str, Any]:
        return {"first": self.first, "second": self.second, "relation": self.relation,
                "support": self.support}


@dataclass
class Reply:
    """What the current turn is doing to the one before it."""

    relation: str = ""
    confidence: float = 0.0
    fits: bool = True
    #: What she had grounds to expect instead, when the pair does not fit.
    expected: Tuple[str, ...] = ()
    why: str = ""

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {"relation": self.relation,
                               "confidence": round(self.confidence, 4), "fits": self.fits}
        if self.expected:
            out["expected"] = list(self.expected)
        if self.why:
            out["why"] = self.why
        return out


class Exchange:
    """What counts as a reply to what — induced, and the half that matters is the refusal.

    The Master's sixth capability listed seventeen things a conversation does: answering,
    clarifying, correcting, agreeing, disagreeing, refusing, negotiating, explaining, teaching,
    summarising. Six of them were reachable and eleven were not, and the reason is that this
    module read every turn **alone**. A turn is not alone: it is a reply, and what makes it a
    clarification rather than an answer is the turn before it.

    A demonstration is two consecutive turns and what the second is doing to the first. What is
    induced is the pair of **acts** — not the sentences — so a demonstration of one question being
    answered reads any question being answered. Two demonstrations minimum, differing in fillers,
    unanimous; a pair two demonstrations disagree about is contested and never read. That is
    :class:`ActLearner`'s rule for the fourth time in this module, and it is deliberate: one
    discipline applied everywhere is a claim, and four mechanisms would be four chances to get it
    wrong differently.

    **And the negative half is the point.** :meth:`fits` says whether this reply is one she has
    ever been shown following that kind of turn. A request answered with a question is not
    *wrong*, it is a clarification — but a request followed by something she has no pairing for is
    a non-sequitur and she can say so. Where she knows nothing about what follows an act, she
    complains about nothing: no evidence, no objection.
    """

    min_support = 2

    def __init__(self, *, min_support: Optional[int] = None) -> None:
        if min_support is not None:
            self.min_support = max(1, int(min_support))
        self.seen: Dict[Tuple[str, str], Dict[str, List[Tuple[str, ...]]]] = {}
        self.kept: Dict[Tuple[str, str], Pair] = {}
        self.contested: Set[Tuple[str, str]] = set()
        self.shown = 0

    def show(self, first_act: str, second_act: str, relation: str,
             fillers: Sequence[str] = ()) -> None:
        """One demonstration: after this kind of turn, this kind of turn is doing this."""
        try:
            key = (str(first_act or ""), str(second_act or ""))
            relation = str(relation or "").strip().lower()
            if not (key[0] and key[1] and relation):
                return
            self.seen.setdefault(key, {}).setdefault(relation, []).append(tuple(fillers))
            self.shown += 1
            self._settle()
        except Exception:  # noqa: BLE001
            return

    def _settle(self) -> None:
        kept: Dict[Tuple[str, str], Pair] = {}
        contested: Set[Tuple[str, str]] = set()
        for key, relations in self.seen.items():
            live = [name for name, demos in relations.items() if demos]
            if len(live) > 1:
                contested.add(key)
                continue
            demos = tuple(relations[live[0]])
            if len(set(demos)) < self.min_support:
                continue
            kept[key] = Pair(first=key[0], second=key[1], relation=live[0],
                             demonstrations=demos)
        self.kept, self.contested = kept, contested

    # -- reading ------------------------------------------------------------- #
    def read(self, first_act: str, second_act: str) -> Reply:
        out = Reply()
        try:
            key = (str(first_act or ""), str(second_act or ""))
            if key in self.contested:
                out.why = "two relations demonstrated for this pair; it does not choose"
                out.fits = True
                return out
            known = self.expects(key[0])
            found = self.kept.get(key)
            if found is not None:
                out.relation = found.relation
                out.confidence = min(0.95, 0.6 + 0.08 * found.support)
                out.why = f"{found.support} demonstrations of {key[0]} → {key[1]}"
                return out
            if not known:
                out.why = f"nothing has ever been shown following a {key[0] or 'turn'}"
                return out
            out.fits = False
            out.expected = tuple(sorted({pair.second for pair in known}))
            out.why = (f"a {key[1]} has never been shown following a {key[0]}; "
                       f"what has is {', '.join(out.expected)}")
            return out
        except Exception:  # noqa: BLE001
            return out

    def expects(self, first_act: str) -> List[Pair]:
        """Every kind of turn she has been shown following this one."""
        return [pair for key, pair in self.kept.items() if key[0] == str(first_act or "")]

    def fits(self, first_act: str, second_act: str) -> bool:
        return self.read(first_act, second_act).fits

    def relations(self) -> Set[str]:
        return {pair.relation for pair in self.kept.values()}

    def stats(self) -> Dict[str, Any]:
        return {"shown": self.shown, "kept": len(self.kept),
                "contested": len(self.contested), "relations": sorted(self.relations())}


# --------------------------------------------------------------------------- #
# 9 · language as prediction — expect the turn, then be corrected by it
# --------------------------------------------------------------------------- #

@dataclass
class Expectation:
    """What she expects of a turn **before** it arrives, and how strongly."""

    act: str = ""
    act_confidence: float = 0.0
    #: ``subject|relation`` she expects the turn to be about.
    topic: str = ""
    topic_confidence: float = 0.0
    #: ``new`` · ``corroborates`` · ``updates`` · ``contradicts``
    verdict: str = ""
    verdict_confidence: float = 0.0
    why: str = ""

    @property
    def empty(self) -> bool:
        return not (self.act or self.topic or self.verdict)

    def to_dict(self) -> Dict[str, Any]:
        return {"act": self.act, "act_confidence": round(self.act_confidence, 4),
                "topic": self.topic, "topic_confidence": round(self.topic_confidence, 4),
                "verdict": self.verdict,
                "verdict_confidence": round(self.verdict_confidence, 4), "why": self.why}


@dataclass
class Surprise:
    """What the turn actually was, scored against what was expected."""

    act: bool = False
    topic: bool = False
    verdict: bool = False
    #: 0.0 when everything predicted was right, 1.0 when nothing was. An expectation that was
    #: **not made** scores neither a hit nor a miss — silence is not a wrong prediction, and
    #: counting it as one would make an untaught organ look confidently wrong instead of quiet.
    error: float = 0.0
    predicted: int = 0
    news: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {"act": self.act, "topic": self.topic, "verdict": self.verdict,
                "error": round(self.error, 4), "predicted": self.predicted, "news": self.news}


class Anticipation:
    """The turn she expected, the turn that came, and the difference between them.

    The Master's fifth deep mechanism, and the one this module did not have at all::

        PREDICTION → OBSERVATION → ERROR → MODEL UPDATE

    Everything predicted here is predicted from **counts she has kept herself**. No transition
    table ships: an exchange where a question is followed by an answer teaches that, an exchange
    where it is not teaches that instead, and an organ that has heard nothing predicts nothing and
    says so. The three things worth predicting about a turn before it arrives are what the speaker
    will be *doing* (the act), what they will be *about* (the topic), and what it will do to the
    ledger (the verdict) — and each is scored separately, because getting the act right while
    getting the topic wrong is a different state from getting both right.

    **A prediction that is never wrong is not a prediction.** :attr:`Surprise.error` is the
    fraction of what she committed to that reality refused, and it is reported on every turn
    rather than only when it is low. An expectation she declined to make scores neither way.

    **And the prediction does work rather than only being scored.** Where the surface carries no
    convention for its shape — no demonstration has ever been given for that skeleton — a
    confident expectation supplies the act instead, and the reading says so
    (:attr:`Interpretation.why`). That is top-down evidence filling a gap bottom-up evidence
    cannot, which is the whole point of predicting; it is deliberately allowed **only** into a
    gap, so a convention that was actually demonstrated is never overridden by a habit.
    """

    #: How sure an expectation must be before it is allowed to supply an act the surface did not.
    #: Below it the expectation is still reported and still scored — it simply does not act.
    floor = 0.7

    def __init__(self, *, floor: Optional[float] = None) -> None:
        if floor is not None:
            self.floor = float(floor)
        self.after_act: Dict[str, Dict[str, int]] = {}
        self.after_verdict: Dict[str, Dict[str, int]] = {}
        self.topic_stay = 0
        self.topic_switch = 0
        self.turns = 0
        self.committed = 0
        self.hits: Dict[str, int] = {"act": 0, "topic": 0, "verdict": 0}
        self.misses: Dict[str, int] = {"act": 0, "topic": 0, "verdict": 0}
        self.last_act = ""
        self.last_topic = ""
        self.last_verdict = ""

    # -- predicting ---------------------------------------------------------- #
    @staticmethod
    def _best(counts: Dict[str, int]) -> Tuple[str, float]:
        total = sum(counts.values())
        if not total:
            return "", 0.0
        name = max(sorted(counts), key=lambda k: counts[k])
        return name, counts[name] / total

    def expect(self) -> Expectation:
        """What the next turn probably is, from what has followed this kind of turn before."""
        out = Expectation()
        try:
            act, confidence = self._best(self.after_act.get(self.last_act, {}))
            if act:
                out.act, out.act_confidence = act, confidence
            verdict, sure = self._best(self.after_verdict.get(self.last_verdict, {}))
            if verdict:
                out.verdict, out.verdict_confidence = verdict, sure
            seen = self.topic_stay + self.topic_switch
            if seen and self.last_topic and self.topic_stay >= self.topic_switch:
                out.topic = self.last_topic
                out.topic_confidence = self.topic_stay / seen
            out.why = (f"after {self.last_act or 'nothing'}, "
                       f"over {self.turns} turns of this exchange")
            return out
        except Exception:  # noqa: BLE001
            return out

    # -- being corrected ----------------------------------------------------- #
    def observe(self, expected: Optional[Expectation], *, act: str, topic: str,
                verdict: str) -> Surprise:
        """Score the expectation against the turn, then fold the turn into the counts."""
        out = Surprise()
        try:
            if expected is not None and not expected.empty:
                checks = (("act", expected.act, act), ("topic", expected.topic, topic),
                          ("verdict", expected.verdict, verdict))
                wrong: List[str] = []
                for name, guess, truth in checks:
                    if not guess:
                        continue
                    if not truth:
                        # The turn had no such field at all — a question carries no ledger
                        # verdict. A prediction about something that did not occur is neither a
                        # hit nor a miss, and counting it as a miss would mark her wrong for
                        # every question in an exchange that is mostly assertions.
                        continue
                    out.predicted += 1
                    if guess == truth:
                        setattr(out, name, True)
                        self.hits[name] += 1
                    else:
                        self.misses[name] += 1
                        wrong.append(f"{name}: expected {guess!r}, heard {truth!r}")
                if out.predicted:
                    self.committed += 1
                    hit = sum(1 for name in ("act", "topic", "verdict") if getattr(out, name))
                    out.error = 1.0 - (hit / out.predicted)
                    out.news = "; ".join(wrong)
            # The update, and it happens whether or not anything was predicted — a turn nobody
            # expected is still the evidence the next expectation is made from.
            self.turns += 1
            if act:
                self.after_act.setdefault(self.last_act, {})
                self.after_act[self.last_act][act] = \
                    self.after_act[self.last_act].get(act, 0) + 1
                self.last_act = act
            if verdict:
                self.after_verdict.setdefault(self.last_verdict, {})
                self.after_verdict[self.last_verdict][verdict] = \
                    self.after_verdict[self.last_verdict].get(verdict, 0) + 1
                self.last_verdict = verdict
            if topic:
                if self.last_topic:
                    if topic == self.last_topic:
                        self.topic_stay += 1
                    else:
                        self.topic_switch += 1
                self.last_topic = topic
            return out
        except Exception:  # noqa: BLE001
            return out

    def reset(self) -> None:
        """A new conversation. The **counts survive** and the position in the exchange does not:
        what follows a question is a fact about the language, and what was just said is not."""
        self.last_act = self.last_topic = self.last_verdict = ""

    def accuracy(self, name: str = "act") -> float:
        hit, miss = self.hits.get(name, 0), self.misses.get(name, 0)
        return hit / (hit + miss) if (hit + miss) else 0.0

    def stats(self) -> Dict[str, Any]:
        return {"turns": self.turns, "committed": self.committed,
                "hits": dict(self.hits), "misses": dict(self.misses),
                "act_accuracy": round(self.accuracy("act"), 4),
                "topic_accuracy": round(self.accuracy("topic"), 4),
                "verdict_accuracy": round(self.accuracy("verdict"), 4),
                "topic_stay": self.topic_stay, "topic_switch": self.topic_switch}


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
    #: The cause or the purpose this sentence carried, when a connective she has been shown said
    #: so — see :class:`Connective`.
    link: Optional[Link] = None
    #: What she expected of this turn **before** hearing it, and what the turn did to that
    #: expectation — see :class:`Anticipation`. ``expected`` is made from the exchange so far and
    #: never from this sentence; ``surprise`` is the only field on an uptake that is about her
    #: rather than about the turn.
    expected: Optional[Expectation] = None
    surprise: Optional[Surprise] = None
    #: What this turn is doing to the one before it, and whether that is something she has ever
    #: been shown following such a turn — see :class:`Exchange`.
    reply: Optional[Reply] = None
    #: The one thing worth asking back, or ``""``. Ordered by what blocks understanding hardest:
    #: an act nobody can read, then a referent nothing settles, then two claims that cannot both
    #: hold.
    question: str = ""

    @property
    def literal(self) -> bool:
        """Should this sentence be filed as a fact about the world?"""
        return not (self.figurative is not None and self.figurative.figurative)

    def anchor(self, ledger: Optional["Ledger"] = None) -> Dict[str, Any]:
        """The Master's semantic reality anchor, filled from this turn — and **only** filled.

        Every slot here is something the turn actually carried; a slot nothing established is
        absent rather than empty, because a schema that always returns eleven keys cannot be told
        apart from one that fills none of them. Six were reachable when the gap report was
        written: entity, relation, time, belief, uncertainty and — through ``is_at`` — space.
        Cause and goal arrive with :class:`Connective`, and the event-versus-state distinction
        with :meth:`Ledger.kind_of`, which reads it off how the relation has behaved rather than
        off any list.
        """
        out: Dict[str, Any] = {}
        try:
            meaning = self.meaning
            if meaning is not None and meaning.readable:
                entities = [x for x in (_bare(meaning.subject), _bare(meaning.object)) if x]
                if entities:
                    out["entity"] = entities
                if meaning.subject:
                    out["agent"] = _bare(meaning.subject)
                if meaning.relation:
                    out["relation"] = meaning.relation
                    if meaning.relation == "is_at" and meaning.object:
                        out["space"] = _bare(meaning.object)
                if meaning.temporal:
                    out["time"] = meaning.temporal
                if meaning.confidence:
                    out["uncertainty"] = round(1.0 - float(meaning.confidence), 4)
            if self.link is not None:
                if self.link.kind == "cause" and self.link.cause:
                    out["cause"] = self.link.cause
                if self.link.kind == "goal" and self.link.goal:
                    out["goal"] = self.link.goal
            claim = self.verdict.claim if self.verdict is not None else None
            if ledger is not None and claim is not None:
                found = ledger.kind_of(claim.relation)
                if found:
                    out["occurrence"] = found
            return out
        except Exception:  # noqa: BLE001
            return out

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
        if self.link is not None:
            out["link"] = self.link.to_dict()
        if self.expected is not None and not self.expected.empty:
            out["expected"] = self.expected.to_dict()
        if self.surprise is not None and self.surprise.predicted:
            out["surprise"] = self.surprise.to_dict()
        if self.reply is not None and (self.reply.relation or not self.reply.fits):
            out["reply"] = self.reply.to_dict()
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
        #: Which words license an update and which make a claim immune to one. Induced, empty
        #: until demonstrated — see :class:`MarkerLearner`.
        self.markers = MarkerLearner()
        #: Which prepositions attach two ways. Discovered from demonstrations that disagree,
        #: empty until then — see :class:`AttachLearner`.
        self.attach = AttachLearner()
        #: How a shape the compiler has no frame for maps onto roles — see :class:`Alternation`.
        self.alternation = Alternation()
        #: What a joining word does to the clauses either side — see :class:`Connective`.
        self.connective = Connective()
        #: What she expects of a turn before it arrives, learned from what has actually
        #: followed what — see :class:`Anticipation`.
        self.anticipation = Anticipation()
        #: What counts as a reply to what — see :class:`Exchange`.
        self.exchange = Exchange()
        #: What both parties have taken as established. Driven from :meth:`hear`, which is the
        #: thing :mod:`nyxara.social.common_ground` has never had: it has been a real model of
        #: given-versus-new since it was written and nothing in this package could put a sentence
        #: into it.
        self.ground: Any = None
        try:
            from nyxara.social.common_ground import CommonGround
            self.ground = CommonGround((speaker, addressee))
        except Exception:  # noqa: BLE001
            self.ground = None
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

    def show_change(self, first: str, second: str, verdict: str) -> None:
        """Demonstrate what a speaker would call this pair of claims —
        see :meth:`MarkerLearner.show`."""
        self.markers.show(first, second, verdict)

    def show_cause(self, surface: str, *, cause: str = "", goal: str = "") -> None:
        """Demonstrate which clause of this sentence is the cause, or which is the purpose —
        see :meth:`Connective.show`. Which *word* does that is what gets induced."""
        self.connective.show(surface, cause=cause, goal=goal)

    def show_alternation(self, plain: str, marked: str) -> None:
        """Demonstrate that these two sentences mean the same thing, one of them in a shape the
        compiler cannot read — see :meth:`Alternation.show`."""
        self.alternation.show(plain, marked)

    def show_exchange(self, first: str, second: str, relation: str) -> None:
        """Demonstrate what the second turn is doing to the first — see :meth:`Exchange.show`.

        The **acts** of the two surfaces are what is recorded, never the sentences, so one
        demonstration of a question being answered reads any question being answered.
        """
        try:
            self.exchange.show(self.acts.read(first).intended,
                               self.acts.read(second).intended, relation,
                               fillers=fillers_of(first) + fillers_of(second))
        except Exception:  # noqa: BLE001
            return

    def show_attachment(self, surface: str, site: str) -> None:
        """Demonstrate which site a prepositional phrase attached to —
        see :meth:`AttachLearner.show`."""
        self.attach.show(surface, site)

    def fit_reference(self, cases: Sequence[Any]) -> Dict[str, Any]:
        """Move the resolver's cue weights to whatever answers these demonstrations best —
        see :meth:`Reference.fit`."""
        return self.reference.fit(cases)

    # -- one turn ------------------------------------------------------------ #
    def hear(self, surface: str, *, speaker: Optional[str] = None) -> Uptake:
        out = Uptake(surface=str(surface or ""), speaker=str(speaker or self.speaker))
        try:
            self.turn += 1
            out.turn = self.reference.turn = self.turn
            # **Before** the sentence is looked at. An expectation formed after reading it would
            # be a description rather than a prediction, and could never be wrong.
            out.expected = self.anticipation.expect()
            first = compile_meaning(out.surface)
            out.rewritten, out.resolutions = self.reference.rewrite(
                out.surface, meaning=first, turn=self.turn)
            out.meaning = (compile_meaning(out.rewritten)
                           if out.rewritten != out.surface else first)
            # A shape the compiler has no frame for. Read through a mapping she was shown, or
            # through one of the structural shapes this module reads — and left unreadable when
            # neither applies, which is where every one of these sentences stood before.
            # A learned mapping **outranks** the compiler where one applies. It has to: the
            # compiler reads *"Ravi opened the door and Arun the window"* perfectly happily as a
            # single claim whose object is ``door arun window``, so a fallback that only fired on
            # `unreadable` would never have been reached for the shape that needed it most.
            # `frames_of` matches three shapes and a mapping needs demonstrations, so an ordinary
            # active sentence reaches none of this.
            # What the sentence *asserts* when it also says why. *"Devi lit the lamp because the
            # room was dark"* asserts that Devi lit the lamp; the other clause is the reason, and
            # reading the whole sentence as one claim gave an agent of
            # ``devi lit the lamp because the room`` — a phrase, filed as a thing.
            out.link = self.connective.read(out.rewritten)
            asserted = out.rewritten
            if out.link is not None and out.link.effect:
                asserted = out.link.effect
                effect = compile_meaning(asserted)
                if effect.readable:
                    out.meaning = effect

            learned = self.alternation.readings(asserted)
            if learned:
                out.meaning = learned[0]
            elif not out.meaning.readable:
                # The structural shapes, as a fallback and never as an override — see
                # :func:`read_claim`. Without this the *claim* was recovered and the **meaning**
                # was not, so a sentence whose only reading is structural introduced no referent
                # and was judged by nothing.
                for reader in (_predication, _locative, _possessive):
                    shape = reader(asserted)
                    if shape is not None:
                        out.meaning = Meaning(kind="assertion", subject=shape[0],
                                              relation=shape[1], object=shape[2],
                                              confidence=0.7, frame="structural")
                        break
            out.interpretation = self.acts.read(out.surface, meaning=out.meaning)
            out.figurative = self.figure.judge(out.meaning)
            self.figure.note_exception(out.figurative)
            out.readings = attachment(out.rewritten, out.meaning, learner=self.attach)
            literal_assertion = (out.interpretation.intended in ("assertion", "exclamation")
                                 or out.interpretation.literal == "assertion")
            if literal_assertion and out.literal:
                claim = read_claim(asserted, learned[0] if learned else None,
                                   turn=self.turn, markers=self.markers,
                                   alternation=self.alternation)
                # A claim about a pronoun is a claim about nobody — but only when the pronoun
                # was never resolved. First and second person **are** resolved and are
                # deliberately not substituted into the surface (that broke a Hinglish question
                # in V.26), so the ledger has to do the substitution itself. Dropping them
                # instead cost every claim the Master made about himself: *"I never visited
                # Delhi."* stopped reaching the ledger at all, and the contradiction two turns
                # later had nothing to contradict.
                if claim is not None and _pronoun(claim.subject):
                    named = next((r.referent for r in out.resolutions
                                  if r.settled and r.pronoun == claim.subject), "")
                    if named:
                        claim.subject = _bare(named.lower())
                    else:
                        claim = None
                out.verdict = self.ledger.note(claim)
                if claim is not None:
                    self.figure.witness(claim.relation, claim.subject)
                # A gapped coordinate is **two** claims, and filing one of them would lose the
                # other silently.
                for extra in learned[1:]:
                    more = read_claim(out.rewritten, extra, turn=self.turn,
                                      markers=self.markers)
                    if more is not None and not _pronoun(more.subject):
                        self.ledger.note(more)
                        self.figure.witness(more.relation, more.subject)
                self.minds.hear(out.rewritten)
            elif not out.literal:
                out.verdict = Verdict(kind="none", why="read as non-literal; nothing filed")
            if out.meaning is not None:
                self.reference.introduce(out.meaning, turn=self.turn)
            # What the **sentence** said, kept before the expectation is allowed to touch it.
            # Scoring a prediction against a reading the prediction itself supplied is a
            # self-fulfilling prophecy: measured, it collapsed the control exchange's act
            # distribution to a single successor and reported 1.00 confidence on an exchange
            # built to be unpredictable. A prediction must be graded by evidence it did not
            # produce.
            heard_act = out.interpretation.intended if out.interpretation is not None else ""

            # The prediction does work rather than only being scored: where no convention has
            # ever been demonstrated for this shape, a confident expectation supplies the act.
            # Only into that gap — a demonstrated convention is never overridden by a habit.
            if (out.interpretation is not None and not out.interpretation.support
                    and out.expected.act
                    and out.expected.act_confidence >= self.anticipation.floor
                    and out.expected.act != out.interpretation.intended):
                out.interpretation.intended = out.expected.act
                out.interpretation.confidence = float(out.expected.act_confidence)
                out.interpretation.level = "expected"
                out.interpretation.why = (
                    f"no convention demonstrated for this shape; taken from the exchange, "
                    f"where {out.expected.act!r} follows {self.anticipation.last_act or 'this'} "
                    f"{out.expected.act_confidence:.0%} of the time")

            # What this turn does to the one before it. Read from the acts, so it is about the
            # exchange rather than about either sentence.
            before = self.history[-1] if self.history else None
            if before is not None and before.interpretation is not None \
                    and out.interpretation is not None:
                out.reply = self.exchange.read(before.interpretation.intended,
                                               out.interpretation.intended)

            # What the conversation has taken as established, which is what decides how much of a
            # thing needs saying next time — see :meth:`Register.say`.
            self._establish(out)

            claim = out.verdict.claim if out.verdict is not None else None
            out.surprise = self.anticipation.observe(
                out.expected,
                act=heard_act,
                topic=f"{claim.subject}|{claim.relation}" if claim is not None else "",
                verdict=out.verdict.kind if out.verdict is not None else "")

            out.question = self._question(out)
            self.history.append(out)
            if len(self.history) > self.max_history:
                self.history = self.history[-self.max_history:]
            return out
        except Exception:  # noqa: BLE001
            return out

    def _establish(self, out: Uptake) -> None:
        """Put this turn's claim into the common ground, and acknowledge the last one.

        A proposition enters the common ground when one party proposes it and the other gives
        evidence of having taken it — which, in a two-party exchange where she is the hearer, is
        the turn after it. So the previous claim is acknowledged as this one arrives, and the
        current claim is proposed. Nothing is grounded by being said once.
        """
        try:
            if self.ground is None:
                return
            claim = out.verdict.claim if out.verdict is not None else None
            if claim is None:
                return
            content = f"{claim.subject}|{claim.relation}|{claim.object}"
            previous = getattr(self, "_proposed", "")
            if previous and previous != content:
                self.ground.acknowledge(self.reference.addressee, previous)
            # And everything she said last turn: the speaker carrying on without objecting is the
            # positive evidence of understanding that grounding needs.
            for text in tuple(getattr(self, "_offered", ())):
                self.ground.acknowledge(out.speaker or self.speaker, text)
            self._offered = ()                  # noqa: attribute defined outside __init__
            self.ground.propose(out.speaker or self.speaker, content)
            self._proposed = content            # noqa: attribute defined outside __init__
        except Exception:  # noqa: BLE001
            return

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
        empty; the speech-act shapes, the induced markers, the ambiguous prepositions, the fitted
        cue weights, the witnessed subjects and the kinds are **about the language** and survive.
        Conflating the two would mean either that a convention had to be
        re-taught every time somebody new spoke, or that one conversation's referents could answer
        a pronoun in the next — and the second of those is how a resolver invents an antecedent.
        """
        cues, margin = dict(self.reference.cues), self.reference.margin
        self.reference = Reference(speaker=self.reference.speaker,
                                   addressee=self.reference.addressee)
        self.reference.cues, self.reference.margin = cues, margin
        self.reference.plausible = self.figure.plausible
        self.ledger = Ledger()
        self.minds = Minds(speaker=self.speaker)
        self.anticipation.reset()
        self.turn = 0
        self.history = []

    # -- asking it things ---------------------------------------------------- #
    def say(self, meaning: Meaning, audience: str = "student") -> Said:
        """One meaning said for one hearer, with what they already have left unsaid.

        Whatever she says is **proposed** into the common ground, and the hearer's next turn
        acknowledges it — Clark's mechanism, run by the module that has implemented it all along
        and never been driven by a sentence.
        """
        given: Set[str] = set()
        try:
            if self.ground is not None:
                for text in self.register.say(meaning, audience).extras:
                    if self.ground.is_grounded(text):
                        given.add(text)
        except Exception:  # noqa: BLE001
            given = set()
        said = self.register.say(meaning, audience, given=given)
        try:
            if self.ground is not None:
                offered = []
                for text in said.extras:
                    if text not in given:
                        self.ground.propose(self.reference.addressee, text)
                        offered.append(text)
                self._offered = tuple(offered)  # noqa: attribute defined outside __init__
        except Exception:  # noqa: BLE001
            pass
        return said

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
        ground = {}
        try:
            ground = self.ground.status() if self.ground is not None else {}
        except Exception:  # noqa: BLE001
            ground = {}
        return {"turns": self.turn, "acts": self.acts.stats(),
                "alternation": self.alternation.stats(),
                "connective": self.connective.stats(),
                "exchange": self.exchange.stats(), "ground": ground,
                "anticipation": self.anticipation.stats(),
                "markers": self.markers.stats(), "attach": self.attach.stats(),
                "reference": self.reference.stats(), "ledger": self.ledger.stats(),
                "minds": self.minds.stats(), "register": self.register.stats(),
                "figure": self.figure.stats()}
