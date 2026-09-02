"""NYXARA · njp/lessons.py — English and Hindi, taught by hand (📖, NJP V.20).

Everything the language faculty had learned until now was a language **nobody speaks**. Thirty-eight
of them — two school dialects and thirty-six from the hard banks — every one minted per seed to
prove that the mechanism is a mechanism and not a table of English. That was the right thing to
measure and it left one question unasked: *can it learn a real one?*

Asked plainly, the answer before this file was **no, and not because of the mechanism**. She had
no vocabulary at all: not one content word, in any language. What ships is a tokeniser that
handles every script, 242 closed-class words across English, romanised Hinglish and Devanagari, and
88 hand-written extraction patterns. That is grammar *scaffolding*. A sentence in Spanish came back
``('el', 'gato', 'persigue al perro')`` — confidently wrong, from the positional frame — and a
sentence in Japanese came back unreadable.

So this is a curriculum, written out by hand, for two real languages.

**What honesty requires saying about it up front.** This is not "everything the author knows about
English". It is what could be *written down and mechanically verified*: a few hundred words with
their real paradigms, a corpus for the classes to form from, and demonstrations of the
constructions both languages actually use. The gap between that and a speaker's competence is
enormous and is not closed here — no idiom, no register beyond a marked honorific, no pragmatics,
no world knowledge, and a vocabulary you could read in a minute. What it does buy is the first
real answer to *"does the faculty work on a language that was not built for it"*, and the answer is
measured on held-out sentences rather than asserted.

**Both languages, and Hindi twice.** English because it is the language the rest of the package
already half-handles. Hindi in **both** Devanagari and romanised Hinglish, because those are two
different surfaces for one grammar — subject-object-verb, postpositions rather than prepositions,
gender agreement on the verb — and teaching both is the cleanest available test that what she
learned is a grammar rather than a spelling.

**What is taught, in each:**

* a **lexicon** — nouns, verbs, adjectives — each with its real paradigm, so morphology has
  something to induce from rather than a rule to be told;
* a **corpus** — sentences with no meanings attached, which is most of what anyone hears, so the
  word classes form from distribution;
* **constructions** — intransitive, transitive, ditransitive, copular, negation, polar and
  content questions, past and future, modals, possessives, place phrases, coordination, relative
  and complement clauses;
* **inflections** — demonstrated pairs binding a feature to a form, which
  :meth:`~nyxara.njp.language.Morphology.bind` will only accept where the vocabulary corroborates
  it.

**And an exam of sentences no lesson contains**, built from the same words in combinations never
demonstrated — which is the only thing that separates a grammar from a list of sentences.

``python -m nyxara.njp.lessons`` teaches both and prints the report.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from nyxara.njp.semantics import Meaning

__all__ = [
    "Course", "Sentence", "Report", "ENGLISH", "HINDI", "HINGLISH", "COURSES",
    "teach", "examine", "main",
]


def _m(kind: str = "assertion", *, negated: bool = False, temporal: str = "",
       modality: str = "", focus: str = "", **roles: str) -> Meaning:
    """A meaning from named roles. The legacy three stay in their own fields."""
    out = Meaning(kind=kind, negated=negated, temporal=temporal, modality=modality, focus=focus)
    for name, value in roles.items():
        if not value:
            continue
        if name == "subject":
            out.subject = value
        elif name == "verb":
            out.relation = value
        elif name == "object":
            out.object = value
        else:
            out.roles[name] = value
    return out


Sentence = Tuple[str, Meaning]


@dataclass
class Course:
    """One real language: its words, what it sounds like, and what its sentences mean."""

    name: str = ""
    title: str = ""
    #: Word forms, for the morphology to induce paradigms from. Every form of every word.
    forms: List[str] = field(default_factory=list)
    #: Sentences with no meaning attached — exposure, for the classes to form from.
    heard: List[str] = field(default_factory=list)
    #: Demonstrations: a surface and what it says.
    lessons: List[Sentence] = field(default_factory=list)
    #: ``(base, inflected, feature)`` — what an ending *means*, shown rather than declared.
    inflections: List[Tuple[str, str, str]] = field(default_factory=list)
    #: Held-out sentences, in combinations no lesson contains.
    exam: List[Sentence] = field(default_factory=list)
    #: ``(stem, feature, expected)`` on words no lesson inflected — including real nonce words.
    wug: List[Tuple[str, str, str]] = field(default_factory=list)
    note: str = ""


# --------------------------------------------------------------------------- #
# English
# --------------------------------------------------------------------------- #

#: Real English, not a tidied version of it. The first draft regularised the awkward paradigms —
#: ``pushs``, ``carrys``, ``mouses`` — and the exam then asked for the real forms, so she was
#: marked wrong for producing exactly what she had been taught. Teaching a language a simplified
#: version of itself and examining it on the real one measures the curriculum, not the learner.
_EN_NOUNS = [
    # singular, plural
    ("dog", "dogs"), ("cat", "cats"), ("bird", "birds"),
    ("horse", "horses"), ("cow", "cows"), ("goat", "goats"),
    ("book", "books"), ("stone", "stones"), ("river", "rivers"), ("tree", "trees"),
    ("house", "houses"), ("door", "doors"), ("window", "windows"), ("table", "tables"),
    ("apple", "apples"), ("seed", "seeds"), ("root", "roots"),
    ("teacher", "teachers"), ("farmer", "farmers"), ("king", "kings"),
    ("wind", "winds"), ("rain", "rains"),
    ("letter", "letters"), ("song", "songs"), ("road", "roads"), ("field", "fields"),
]

_EN_VERBS = [
    # bare, third-person singular, past, progressive
    ("chase", "chases", "chased", "chasing"),
    ("push", "pushes", "pushed", "pushing"),
    ("watch", "watches", "watched", "watching"),
    ("wash", "washes", "washed", "washing"),
    ("open", "opens", "opened", "opening"),
    ("pull", "pulls", "pulled", "pulling"),
    ("plant", "plants", "planted", "planting"),
    ("paint", "paints", "painted", "painting"),
    ("count", "counts", "counted", "counting"),
    ("follow", "follows", "followed", "following"),
    ("answer", "answers", "answered", "answering"),
    ("cook", "cooks", "cooked", "cooking"),
    ("clean", "cleans", "cleaned", "cleaning"),
    ("walk", "walks", "walked", "walking"),
    ("work", "works", "worked", "working"),
]

_EN_ADJECTIVES = [
    ("small", "smaller", "smallest"), ("great", "greater", "greatest"),
    ("quick", "quicker", "quickest"), ("dark", "darker", "darkest"),
    ("warm", "warmer", "warmest"), ("cold", "colder", "coldest"),
    ("clean", "cleaner", "cleanest"), ("loud", "louder", "loudest"),
    ("young", "younger", "youngest"), ("old", "older", "oldest"),
    ("strong", "stronger", "strongest"), ("light", "lighter", "lightest"),
]

#: The irregulars. Every one is a **memorised pair** — see the ``mixed-irregulars`` problem in
#: :mod:`nyxara.njp.hard`: an irregular is bound where no process corroborates it, and then it
#: never generalises to a stem it was not shown on. Listed rather than derived, because that is
#: what an irregular is.
_EN_IRREGULAR: Tuple[Tuple[str, str, str], ...] = (
    ("go", "went", "past"), ("come", "came", "past"), ("take", "took", "past"),
    ("give", "gave", "past"), ("see", "saw", "past"), ("run", "ran", "past"),
    ("child", "children", "plural"), ("mouse", "mice", "plural"),
    ("foot", "feet", "plural"), ("tooth", "teeth", "plural"),
    ("leaf", "leaves", "plural"), ("fish", "fish", "plural"),
    # The sibilant ``-es`` and the ``y`` → ``ies`` change are memorised here rather than induced,
    # and that is a limitation worth naming rather than hiding. Both are conditioned by the
    # *consonant* the stem ends in, and :class:`~nyxara.njp.language.Rule` conditions on its last
    # **vowel** — the shape a harmony rule has. So she holds these as pairs and does not extend
    # them to a stem she was not shown, which is the correct behaviour for a rule she has not got.
    ("push", "pushes", "third"), ("watch", "watches", "third"),
    ("wash", "washes", "third"),
    ("push", "pushed", "past"), ("watch", "watched", "past"),
    ("wash", "washed", "past"),
    ("push", "pushing", "progressive"), ("watch", "watching", "progressive"),
    # ``carry`` → ``carries`` is left out of the sentence lessons on purpose and kept only as a
    # pair. The lemma is not **in** its own inflected form — `carries` contains `carr`, not
    # `carry` — so a demonstration of it hands the grammar a meaning whose verb is not in its
    # surface, the lemma is taken for a feature of the shape, and the whole transitive family goes
    # with it: 20 constructions down to 18 and four held-out sentences unreadable. A stem change
    # under a suffix is a thing this module holds as a pair and does not build clauses out of.
    ("carry", "carries", "third"), ("carry", "carried", "past"),
)


#: Every inflected English form back to its lemma, for **writing** this curriculum. Bookkeeping
#: for the author, not a rule she is given — every pair in it is one this file also teaches her.
#:
#: It exists because the first draft wrote the lemma as ``form[:-2]`` for the past tense, which
#: makes ``chased`` into ``chas``. She read the sentence correctly as ``chase``, disagreed with
#: the label, and :meth:`~nyxara.njp.language.Grammar._verify` did exactly what it is for: threw
#: the whole past-tense shape away for contradicting its own lesson. The lesson was wrong.
_EN_LEMMA: Dict[str, str] = {}
for _base, _third, _past, _prog in _EN_VERBS:
    _EN_LEMMA.update({_third: _base, _past: _base, _prog: _base})
for _base, _form, _feature in _EN_IRREGULAR:
    if _feature in ("third", "past", "progressive"):
        _EN_LEMMA[_form] = _base


def _lemma(form: str) -> str:
    """The lemma of an English verb form, from the paradigms this file already lists."""
    if form in _EN_LEMMA:
        return _EN_LEMMA[form]
    for ending in ("ed", "s"):
        if form.endswith(ending) and form[: -len(ending)] in _EN_LEMMA.values():
            return form[: -len(ending)]
    return form


def _english() -> Course:
    """English: articles as literals, prepositions as literals, and endings as paradigms.

    The determiner is a **literal token** of a construction rather than material on a slot,
    because that is what it is — English marks definiteness with a separate word. So ``the dog
    runs`` and ``a dog runs`` are two shapes that the paradigm pass folds into one with a marker,
    and the marker's material is a whole token rather than an affix, which is the case
    :class:`~nyxara.njp.language.Marker` was written to hold.
    """
    course = Course(name="en", title="English")

    for singular, plural in _EN_NOUNS:
        course.forms.extend((singular, plural))
    for bare, third, past, progressive in _EN_VERBS:
        course.forms.extend((bare, third, past, progressive))
    for plain, more, most in _EN_ADJECTIVES:
        course.forms.extend((plain, more, most))
    for base, odd, _feature in _EN_IRREGULAR:
        course.forms.extend((base, odd))

    # What an ending means, shown on two words each and no more — the claim being tested is that
    # the ending generalises, so demonstrating it on twenty would leave the exam unable to tell a
    # rule from a well-stocked table.
    course.inflections = [
        ("dog", "dogs", "plural"), ("book", "books", "plural"),
        ("chase", "chased", "past"), ("open", "opened", "past"),
        ("chase", "chasing", "progressive"), ("open", "opening", "progressive"),
        ("small", "smaller", "comparative"), ("quick", "quicker", "comparative"),
        ("small", "smallest", "superlative"), ("quick", "quickest", "superlative"),
    ]
    course.inflections.extend(_EN_IRREGULAR)

    # Exposure. No meanings attached, which is most of what anyone hears.
    for subject, verb, obj in (("dog", "chases", "cat"), ("cat", "follows", "bird"),
                               ("bird", "counts", "seed"), ("farmer", "plants", "tree"),
                               ("teacher", "opens", "book"), ("king", "cleans", "stone"),
                               ("cow", "pulls", "road"), ("goat", "paints", "door")):
        course.heard.append(f"the {subject} {verb} the {obj}")
        course.heard.append(f"a {subject} {verb} a {obj}")

    def transitive(subject: str, verb: str, obj: str, *, det: str = "the") -> Sentence:
        return (f"{det} {subject} {verb} {det} {obj}",
                _m(subject=subject, verb=verb.rstrip("s") if verb.endswith("s") else verb,
                   object=obj))

    # -- the constructions, one family at a time --------------------------- #
    lessons: List[Sentence] = []

    # Transitive, definite and indefinite — and the ``-es`` verbs are in here rather than only in
    # the word list. An ending she has met in a *paradigm* but never in a *sentence* has no
    # construction to sit in: `pushes` was read through the `<verb>s` shape as `pushe`, which is
    # not a word, because nothing had ever shown her `-es` inside a clause.
    for subject, verb, obj in (("dog", "chases", "cat"), ("cat", "follows", "bird"),
                               ("bird", "counts", "seed"), ("farmer", "plants", "tree"),
                               ("teacher", "opens", "door"),
                               ("king", "pushes", "table"), ("goat", "watches", "window"),
                               ("cow", "washes", "root")):
        lessons.append((f"the {subject} {verb} the {obj}",
                        _m(subject=subject, verb=_lemma(verb), object=obj)))
        lessons.append((f"a {subject} {verb} a {obj}",
                        _m(subject=subject, verb=_lemma(verb), object=obj, definite="no")))

    # intransitive
    for subject, verb in (("dog", "walks"), ("cat", "works"), ("river", "cleans"),
                          ("king", "counts")):
        lessons.append((f"the {subject} {verb}", _m(subject=subject, verb=_lemma(verb))))

    # copular: a property, and a kind
    for subject, adjective in (("river", "cold"), ("stone", "old"), ("wind", "warm"),
                               ("song", "loud")):
        lessons.append((f"the {subject} is {adjective}",
                        _m(subject=subject, verb="is", property=adjective)))
    for member, kind in (("dog", "animal"), ("cat", "animal"), ("apple", "fruit"),
                         ("river", "place")):
        lessons.append((f"a {member} is a {kind}", _m(subject=member, verb="is", kind=kind)))

    # negation
    for subject, verb, obj in (("dog", "chase", "cat"), ("cat", "follow", "bird"),
                               ("bird", "count", "seed"), ("teacher", "open", "door")):
        lessons.append((f"the {subject} does not {verb} the {obj}",
                        _m(subject=subject, verb=verb, object=obj, negated=True)))

    # past and future
    for subject, verb, obj in (("dog", "chased", "cat"), ("cat", "followed", "bird"),
                               ("farmer", "planted", "tree"), ("teacher", "opened", "door")):
        lessons.append((f"the {subject} {verb} the {obj}",
                        _m(subject=subject, verb=_lemma(verb), object=obj, temporal="past")))
    for subject, verb, obj in (("dog", "chase", "cat"), ("cat", "follow", "bird"),
                               ("farmer", "plant", "tree"), ("teacher", "open", "door")):
        lessons.append((f"the {subject} will {verb} the {obj}",
                        _m(subject=subject, verb=verb, object=obj, temporal="future")))

    # modal
    for subject, verb, obj in (("dog", "chase", "cat"), ("cat", "follow", "bird"),
                               ("teacher", "open", "door")):
        lessons.append((f"the {subject} can {verb} the {obj}",
                        _m(subject=subject, verb=verb, object=obj, modality="possible")))

    # polar question
    for subject, verb, obj in (("dog", "chase", "cat"), ("cat", "follow", "bird"),
                               ("bird", "count", "seed"), ("teacher", "open", "door")):
        lessons.append((f"does the {subject} {verb} the {obj}?",
                        _m("polar_question", focus="truth",
                           subject=subject, verb=verb, object=obj)))

    # content questions, one for each slot
    for subject, verb in (("dog", "chase"), ("cat", "follow"), ("bird", "count"),
                          ("teacher", "open")):
        lessons.append((f"what does the {subject} {verb}?",
                        _m("question", focus="object", subject=subject, verb=verb)))
    for verb, obj in (("chase", "cat"), ("follow", "bird"), ("count", "seed"), ("open", "door")):
        lessons.append((f"who {verb}s the {obj}?",
                        _m("question", focus="subject", verb=verb, object=obj)))

    # ditransitive, both orders English allows
    for subject, verb, obj, goal in (("teacher", "gives", "book", "king"),
                                     ("farmer", "gives", "seed", "bird"),
                                     ("king", "gives", "horse", "farmer")):
        lessons.append((f"the {subject} {verb} the {goal} the {obj}",
                        _m(subject=subject, verb=_lemma(verb), object=obj, recipient=goal)))
        lessons.append((f"the {subject} {verb} the {obj} to the {goal}",
                        _m(subject=subject, verb=_lemma(verb), object=obj, goal=goal)))

    # a place phrase
    for subject, verb, place in (("cat", "cleans", "table"), ("bird", "counts", "tree"),
                                 ("book", "opens", "door")):
        lessons.append((f"the {subject} {verb} on the {place}",
                        _m(subject=subject, verb=_lemma(verb), location=place)))

    # possessive
    for owner, thing, verb, obj in (("farmer", "dog", "chases", "cat"),
                                    ("teacher", "book", "opens", "door"),
                                    ("king", "horse", "pulls", "stone")):
        lessons.append((f"the {owner} 's {thing} {verb} the {obj}",
                        _m(subject=thing, verb=_lemma(verb), object=obj, owner=owner)))

    # coordination
    for one, two, verb, obj in (("dog", "cat", "chase", "bird"),
                                ("goat", "farmer", "plant", "tree"),
                                ("king", "teacher", "count", "book")):
        lessons.append((f"the {one} and the {two} {verb} the {obj}",
                        _m(subject=one, verb=verb, object=obj, subject2=two)))

    # comparative
    for one, adjective, two in (("dog", "quicker", "cat"), ("river", "colder", "stone"),
                                ("road", "warmer", "wind")):
        lessons.append((f"the {one} is {adjective} than the {two}",
                        _m(subject=one, verb="is", property=adjective, than=two)))

    # a relative clause, and a complement clause
    for subject, rverb, robj, verb, obj in (("dog", "chases", "cat", "counts", "seed"),
                                            ("goat", "opens", "door", "cleans", "stone"),
                                            ("farmer", "plants", "tree", "paints", "cow")):
        lessons.append((f"the {subject} that {rverb} the {robj} {verb} the {obj}",
                        _m(subject=subject, verb=_lemma(verb), object=obj,
                           rel_verb=_lemma(rverb), rel_object=robj)))
    for subject, csubject, cverb, cobj in (("teacher", "dog", "chases", "cat"),
                                           ("goat", "bird", "counts", "seed"),
                                           ("king", "farmer", "plants", "tree")):
        lessons.append((f"the {subject} says that the {csubject} {cverb} the {cobj}",
                        _m(subject=subject, verb="say", comp_subject=csubject,
                           comp_verb=_lemma(cverb), comp_object=cobj)))

    course.lessons = lessons

    # -- the exam: the same grammar, in combinations no lesson contains ----- #
    course.exam = [
        (f"the horse pushes the cow", _m(subject="horse", verb="push", object="cow")),
        (f"a goat cooks a root", _m(subject="goat", verb="cook", object="root",
                                     definite="no")),
        (f"the teacher walks", _m(subject="teacher", verb="walk")),
        (f"the window is dark", _m(subject="window", verb="is", property="dark")),
        (f"a horse is a animal", _m(subject="horse", verb="is", kind="animal")),
        (f"the goat does not push the road",
         _m(subject="goat", verb="push", object="road", negated=True)),
        (f"the king painted the house",
         _m(subject="king", verb="paint", object="house", temporal="past")),
        (f"the cow will clean the field",
         _m(subject="cow", verb="clean", object="field", temporal="future")),
        (f"the bird can pull the seed",
         _m(subject="bird", verb="pull", object="seed", modality="possible")),
        (f"does the goat clean the table?",
         _m("polar_question", focus="truth", subject="goat", verb="clean", object="table")),
        (f"what does the horse count?",
         _m("question", focus="object", subject="horse", verb="count")),
        (f"who answers the letter?",
         _m("question", focus="subject", verb="answer", object="letter")),
        (f"the farmer gives the king the apple",
         _m(subject="farmer", verb="give", object="apple", recipient="king")),
        (f"the cow gives the song to the teacher",
         _m(subject="cow", verb="give", object="song", goal="teacher")),
        (f"the goat works on the road", _m(subject="goat", verb="work", location="road")),
        (f"the king 's cow pushes the tree",
         _m(subject="cow", verb="push", object="tree", owner="king")),
        (f"the horse and the goat clean the window",
         _m(subject="horse", verb="clean", object="window", subject2="goat")),
        (f"the wind is louder than the rain",
         _m(subject="wind", verb="is", property="louder", than="rain")),
        (f"the goat that pushes the cow counts the seed",
         _m(subject="goat", verb="count", object="seed", rel_verb="push", rel_object="cow")),
        (f"the farmer says that the horse pulls the road",
         _m(subject="farmer", verb="say", comp_subject="horse", comp_verb="pull",
            comp_object="road")),
    ]

    # The wug test, on the word the test is named after and on stems no lesson inflected.
    course.wug = [
        ("wug", "plural", "wugs"), ("blicket", "plural", "blickets"),
        ("tove", "plural", "toves"), ("mountain", "plural", "mountains"),
        ("wug", "past", "wuged"), ("blicket", "past", "blicketed"),
        ("gostak", "progressive", "gostaking"), ("wug", "progressive", "wuging"),
        ("bright", "comparative", "brighter"), ("sharp", "superlative", "sharpest"),
        # And the irregulars, which must be recalled and must not have leaked onto anything else.
        ("go", "past", "went"), ("child", "plural", "children"),
    ]
    course.note = ("determiners and prepositions as literal tokens; endings as induced "
                   "paradigms; ten irregulars memorised")
    return course


# --------------------------------------------------------------------------- #
# Hindi — the same grammar, in two scripts
# --------------------------------------------------------------------------- #

#: ``(devanagari, hinglish)`` for every word, so the two courses are the same language twice.
_HI_NOUNS: Tuple[Tuple[str, str, str, str], ...] = (
    # deva singular, deva plural, roman singular, roman plural
    # Enough vowel-final nouns for the class to be a class. With two of them the vowel-change
    # plural reached no corroboration at all — `min_stems` is three — so it was kept as a pair of
    # irregulars and every nonce vowel-final stem got the consonant-final suffix instead. A rule
    # needs a paradigm; two words are an anecdote.
    ("लड़का", "लड़के", "ladka", "ladke"),
    ("कुत्ता", "कुत्ते", "kutta", "kutte"),
    ("बच्चा", "बच्चे", "bachcha", "bachche"),
    ("कमरा", "कमरे", "kamra", "kamre"),
    ("रास्ता", "रास्ते", "raasta", "raaste"),
    ("कपड़ा", "कपड़े", "kapda", "kapde"),
    ("बिल्ली", "बिल्लियाँ", "billi", "billiyan"),
    ("आदमी", "आदमियाँ", "aadmi", "aadmiyan"),
    ("किताब", "किताबें", "kitaab", "kitaaben"),
    ("पेड़", "पेड़ें", "ped", "peden"),
    ("घर", "घरें", "ghar", "gharen"),
    ("पानी", "पानियाँ", "paani", "paaniyan"),
    ("आम", "आमें", "aam", "aamen"),
    ("रोटी", "रोटियाँ", "roti", "rotiyan"),
    ("गाय", "गायें", "gaay", "gaayen"),
    ("शिक्षक", "शिक्षकें", "shikshak", "shikshaken"),
)

_HI_VERBS: Tuple[Tuple[str, str, str, str], ...] = (
    # deva stem, deva habitual-masc, roman stem, roman habitual-masc
    ("खा", "खाता", "kha", "khaata"),
    ("पढ़", "पढ़ता", "padh", "padhta"),
    ("देख", "देखता", "dekh", "dekhta"),
    ("लिख", "लिखता", "likh", "likhta"),
    ("चला", "चलाता", "chala", "chalaata"),
    ("बना", "बनाता", "bana", "banaata"),
    ("तोड़", "तोड़ता", "tod", "todta"),
    ("उठा", "उठाता", "utha", "uthaata"),
)


def _hindi(script: str) -> Course:
    """Hindi, in Devanagari or in romanised Hinglish — one grammar, two surfaces.

    Subject-object-verb, a postposition where English puts a preposition, the copula last, and
    ``नहीं`` before the verb. Nothing about it resembles the English course, which is the point:
    if what she learned from English were a spelling rather than a grammar, none of it would
    transfer and this course would have to be learned from nothing. It is, and it is measured
    separately.
    """
    deva = script == "deva"
    name = "hi" if deva else "hi-latn"
    course = Course(name=name, title="Hindi (Devanagari)" if deva else "Hindi (romanised)")

    nouns = [(row[0], row[1]) if deva else (row[2], row[3]) for row in _HI_NOUNS]
    verbs = [(row[0], row[1]) if deva else (row[2], row[3]) for row in _HI_VERBS]
    ko, ne, se, mein, par = (("को", "ने", "से", "में", "पर") if deva
                             else ("ko", "ne", "se", "mein", "par"))
    hai, tha, nahi, kya, kaun, kahan = (("है", "था", "नहीं", "क्या", "कौन", "कहाँ") if deva
                                        else ("hai", "tha", "nahi", "kya", "kaun", "kahan"))
    ka = "का" if deva else "ka"
    aur = "और" if deva else "aur"
    sakta = "सकता" if deva else "sakta"

    for singular, plural in nouns:
        course.forms.extend((singular, plural))
    for stem, habitual in verbs:
        course.forms.extend((stem, habitual))

    # One pair from each class, on both counts. Hindi's plural is a vowel change on a
    # vowel-final noun (लड़का → लड़के) and a suffix on a consonant-final one (किताब → किताबें),
    # and its habitual is `-ता` after a consonant and `-ाता` after a vowel. Demonstrating two of
    # the same class leaves the other with no evidence at all, which is what happened first: both
    # plural pairs were vowel-final, and she abstained on every consonant-final wug item — the
    # correct answer to a rule she had never been shown.
    # Chosen by **class**, not by position. They were picked by index, and adding four nouns to
    # the list moved the index: both demonstrated plurals became vowel-final, the consonant-final
    # class had no demonstration at all, and every nonce consonant-final stem came back with the
    # vowel-change plural. A curriculum indexed by position is one edit away from teaching only
    # half a paradigm.
    def _first(rows: Sequence[Tuple[str, str]], *, vowel: bool) -> Tuple[str, str]:
        for base, inflected in rows:
            if (base[-1] in "aeiouाीूेोैौ") is vowel:
                return base, inflected
        return rows[0]

    course.inflections = []
    for rows, feature in ((nouns, "plural"), (verbs, "habitual")):
        for vowel in (True, False):
            base, inflected = _first(rows, vowel=vowel)
            course.inflections.append((base, inflected, feature))

    people = [row[0] for row in nouns[:4]]
    things = [row[0] for row in nouns[4:]]
    stems = [row[0] for row in verbs]
    habituals = [row[1] for row in verbs]

    for index in range(8):
        subject, obj = people[index % len(people)], things[index % len(things)]
        course.heard.append(f"{subject} {obj} {habituals[index % len(habituals)]} {hai}")

    lessons: List[Sentence] = []

    # transitive, habitual present: S O V-ta hai
    for index in range(5):
        subject, obj = people[index % len(people)], things[index % len(things)]
        stem, habitual = stems[index % len(stems)], habituals[index % len(habituals)]
        lessons.append((f"{subject} {obj} {habitual} {hai}",
                        _m(subject=subject, verb=stem, object=obj)))

    # the object marked with the accusative postposition
    for index in range(4):
        subject, obj = people[index % len(people)], people[(index + 1) % len(people)]
        stem, habitual = stems[index % len(stems)], habituals[index % len(habituals)]
        lessons.append((f"{subject} {obj} {ko} {habitual} {hai}",
                        _m(subject=subject, verb=stem, object=obj, marked="yes")))

    # past
    for index in range(4):
        subject, obj = people[index % len(people)], things[index % len(things)]
        stem, habitual = stems[index % len(stems)], habituals[index % len(habituals)]
        lessons.append((f"{subject} {obj} {habitual} {tha}",
                        _m(subject=subject, verb=stem, object=obj, temporal="past")))

    # negation
    for index in range(4):
        subject, obj = people[index % len(people)], things[index % len(things)]
        stem, habitual = stems[index % len(stems)], habituals[index % len(habituals)]
        lessons.append((f"{subject} {obj} {nahi} {habitual} {hai}",
                        _m(subject=subject, verb=stem, object=obj, negated=True)))

    # polar question
    for index in range(4):
        subject, obj = people[index % len(people)], things[index % len(things)]
        stem, habitual = stems[index % len(stems)], habituals[index % len(habituals)]
        lessons.append((f"{kya} {subject} {obj} {habitual} {hai}",
                        _m("polar_question", focus="truth",
                           subject=subject, verb=stem, object=obj)))

    # content questions
    for index in range(4):
        subject = people[index % len(people)]
        stem, habitual = stems[index % len(stems)], habituals[index % len(habituals)]
        lessons.append((f"{subject} {kya} {habitual} {hai}",
                        _m("question", focus="object", subject=subject, verb=stem)))
    for index in range(4):
        obj = things[index % len(things)]
        stem, habitual = stems[index % len(stems)], habituals[index % len(habituals)]
        lessons.append((f"{kaun} {obj} {habitual} {hai}",
                        _m("question", focus="subject", verb=stem, object=obj)))

    # instrument, place, possession, coordination, modal
    for index in range(3):
        subject, obj = people[index % len(people)], things[index % len(things)]
        stem, habitual = stems[index % len(stems)], habituals[index % len(habituals)]
        tool = things[(index + 2) % len(things)]
        lessons.append((f"{subject} {tool} {se} {obj} {habitual} {hai}",
                        _m(subject=subject, verb=stem, object=obj, instrument=tool)))
        place = things[(index + 1) % len(things)]
        lessons.append((f"{subject} {place} {mein} {obj} {habitual} {hai}",
                        _m(subject=subject, verb=stem, object=obj, location=place)))
        owner = people[(index + 2) % len(people)]
        lessons.append((f"{owner} {ka} {subject} {obj} {habitual} {hai}",
                        _m(subject=subject, verb=stem, object=obj, owner=owner)))
        other = people[(index + 1) % len(people)]
        lessons.append((f"{subject} {aur} {other} {obj} {habitual} {hai}",
                        _m(subject=subject, verb=stem, object=obj, subject2=other)))
        lessons.append((f"{subject} {obj} {stem} {sakta} {hai}",
                        _m(subject=subject, verb=stem, object=obj, modality="possible")))

    course.lessons = lessons

    # -- held out: same grammar, combinations no lesson contains ------------ #
    a, b, c, d = people[3], people[2], things[5], things[4]
    stem_x, hab_x = stems[6], habituals[6]
    stem_y, hab_y = stems[7], habituals[7]
    course.exam = [
        (f"{a} {c} {hab_x} {hai}", _m(subject=a, verb=stem_x, object=c)),
        (f"{a} {b} {ko} {hab_x} {hai}", _m(subject=a, verb=stem_x, object=b, marked="yes")),
        (f"{b} {d} {hab_y} {tha}", _m(subject=b, verb=stem_y, object=d, temporal="past")),
        (f"{a} {d} {nahi} {hab_y} {hai}",
         _m(subject=a, verb=stem_y, object=d, negated=True)),
        (f"{kya} {b} {c} {hab_x} {hai}",
         _m("polar_question", focus="truth", subject=b, verb=stem_x, object=c)),
        (f"{a} {kya} {hab_y} {hai}", _m("question", focus="object", subject=a, verb=stem_y)),
        (f"{kaun} {c} {hab_x} {hai}", _m("question", focus="subject", verb=stem_x, object=c)),
        (f"{b} {d} {se} {c} {hab_y} {hai}",
         _m(subject=b, verb=stem_y, object=c, instrument=d)),
        (f"{a} {c} {mein} {d} {hab_x} {hai}",
         _m(subject=a, verb=stem_x, object=d, location=c)),
        (f"{b} {ka} {a} {c} {hab_y} {hai}",
         _m(subject=a, verb=stem_y, object=c, owner=b)),
        (f"{a} {aur} {b} {d} {hab_x} {hai}",
         _m(subject=a, verb=stem_x, object=d, subject2=b)),
        (f"{b} {c} {stem_y} {sakta} {hai}",
         _m(subject=b, verb=stem_y, object=c, modality="possible")),
    ]
    # Nonce stems, in both classes, and **not** in the vocabulary. The first version used real
    # words from the noun list, whose plurals are in ``forms`` — so the answer was a word she had
    # already heard and the item was not held out at all. A test found that one rather than she
    # did, which is the second time in this work that the flaw the learner did not surface was
    # the one that flattered the score.
    course.wug = ([("बलक", "plural", "बलकें"), ("पिलक", "plural", "पिलकें"),
                   ("टिमा", "plural", "टिमे"), ("रोका", "plural", "रोके"),
                   ("ग्लिम", "habitual", "ग्लिमता"), ("ड्रन", "habitual", "ड्रनता"),
                   ("ज़िला", "habitual", "ज़िलाता")]
                  if deva else
                  [("balak", "plural", "balaken"), ("pilak", "plural", "pilaken"),
                   ("tima", "plural", "time"), ("roka", "plural", "roke"),
                   ("glim", "habitual", "glimta"), ("dran", "habitual", "dranta"),
                   ("zila", "habitual", "zilaata")])
    course.note = ("subject-object-verb, postpositions, the copula last, "
                   f"{'Devanagari' if deva else 'romanised'}")
    return course


#: What each word is in each language. Three columns, and every row is a fact somebody has to
#: state — there is no way to derive that ``dog`` and ``कुत्ता`` are each other's from any amount
#: of monolingual evidence, which is exactly why a bilingual child needs both languages spoken to
#: them and not just more of one.
#:
#: The Devanagari and romanised columns are the *same* Hindi words in two scripts, so that pairing
#: is free and exact. The English column is a glossary, with all a glossary's faults: one sense
#: each, no context, and nothing about when a word would actually be the right choice.
GLOSSARY: Tuple[Tuple[str, str, str], ...] = (
    ("dog", "कुत्ता", "kutta"), ("cat", "बिल्ली", "billi"), ("teacher", "शिक्षक", "shikshak"),
    ("book", "किताब", "kitaab"), ("tree", "पेड़", "ped"), ("house", "घर", "ghar"),
    ("apple", "आम", "aam"), ("cow", "गाय", "gaay"), ("king", "आदमी", "aadmi"),
    ("farmer", "लड़का", "ladka"), ("bird", "रोटी", "roti"), ("river", "पानी", "paani"),
    ("chase", "देख", "dekh"), ("open", "पढ़", "padh"), ("count", "लिख", "likh"),
    ("clean", "बना", "bana"), ("plant", "उठा", "utha"), ("follow", "तोड़", "tod"),
    ("pull", "चला", "chala"), ("paint", "खा", "kha"),
)


def glosses(faculty: Any) -> int:
    """Teach the word correspondences. Returns how many were recorded."""
    taught = 0
    for english, deva, roman in GLOSSARY:
        taught += int(faculty.pair(english, deva, frm="en", into="hi"))
        taught += int(faculty.pair(english, roman, frm="en", into="hi-latn"))
        taught += int(faculty.pair(deva, roman, frm="hi", into="hi-latn"))
    return taught


ENGLISH = _english()
HINDI = _hindi("deva")
HINGLISH = _hindi("latn")
COURSES: Tuple[Course, ...] = (ENGLISH, HINDI, HINGLISH)


# --------------------------------------------------------------------------- #
# teaching, and being examined
# --------------------------------------------------------------------------- #

@dataclass
class Report:
    """What one language scored, with the three abilities kept apart."""

    language: str = ""
    title: str = ""
    shapes: int = 0
    rejected: int = 0
    read_right: int = 0
    read_total: int = 0
    said_right: int = 0
    said_total: int = 0
    wug_right: int = 0
    wug_total: int = 0
    misses: List[str] = field(default_factory=list)
    note: str = ""

    @property
    def score(self) -> float:
        total = self.read_total + self.said_total + self.wug_total
        right = self.read_right + self.said_right + self.wug_right
        return right / total if total else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {"language": self.language, "title": self.title, "shapes": self.shapes,
                "rejected": self.rejected, "score": round(self.score, 3),
                "read": f"{self.read_right}/{self.read_total}",
                "said": f"{self.said_right}/{self.said_total}",
                "wug": f"{self.wug_right}/{self.wug_total}",
                "note": self.note, "misses": self.misses[:6]}


def _tokens(text: str) -> List[str]:
    """What she actually produced, as tokens.

    Production is graded on tokens rather than on the string because the apostrophe in ``the king
    's cow`` is punctuation the tokeniser drops before she ever sees it — she cannot put back a
    character that was never in her input, and marking her wrong for it would be measuring the
    detokeniser. Every other difference still counts, including word order and every ending.
    """
    from nyxara.njp.language import tokenize_surface
    return tokenize_surface(text)


def _same(got: Optional[Meaning], want: Meaning) -> bool:
    """Right, with no partial credit — every role, the polarity, the tense and the mood."""
    if got is None or not getattr(got, "readable", False):
        return False

    def roles(meaning: Meaning) -> Dict[str, str]:
        out = dict(getattr(meaning, "roles", {}) or {})
        for key, value in (("subject", meaning.subject), ("verb", meaning.relation),
                           ("object", meaning.object)):
            if value:
                out[key] = value
        return {k: " ".join(str(v).lower().split()) for k, v in out.items() if str(v).strip()}

    return (got.kind == (want.kind or "assertion")
            and bool(got.negated) == bool(want.negated)
            and (got.temporal or "") == (want.temporal or "")
            and (got.modality or "") == (want.modality or "")
            and (got.focus or "") == (want.focus or "")
            and roles(got) == roles(want))


def teach(course: Course, faculty: Any) -> Report:
    """Hear the words, hear the corpus, be shown the sentences, and generalise once."""
    report = Report(language=course.name, title=course.title, note=course.note)
    faculty.hear_words(course.forms, tongue=course.name)
    for text in course.heard:
        faculty.hear(text, tongue=course.name)
    for surface, meaning in course.lessons:
        faculty.show(surface, meaning, tongue=course.name)
    for base, inflected, feature in course.inflections:
        faculty.bind(base, inflected, feature, tongue=course.name)
    learned = faculty.learn(tongue=course.name)
    report.shapes, report.rejected = learned.kept, learned.rejected
    return report


def examine(course: Course, faculty: Any, report: Optional[Report] = None) -> Report:
    """Held-out sentences, and inflections of words no lesson touched."""
    report = report or Report(language=course.name, title=course.title, note=course.note)
    for surface, meaning in course.exam:
        report.read_total += 1
        got = faculty.read(surface, tongue=course.name)
        if _same(got, meaning):
            report.read_right += 1
        elif len(report.misses) < 12:
            report.misses.append(f"read {surface!r} → "
                                 f"{got.to_dict() if got is not None else None}")
        report.said_total += 1
        said = faculty.say(meaning, tongue=course.name)
        if _tokens(said) == _tokens(surface):
            report.said_right += 1
        elif len(report.misses) < 12:
            report.misses.append(f"say {surface!r} → {said!r}")
    for stem, feature, want in course.wug:
        report.wug_total += 1
        got = faculty.inflect(stem, feature, tongue=course.name)
        if got == want:
            report.wug_right += 1
        elif len(report.misses) < 12:
            report.misses.append(f"wug {stem!r}+{feature} → {got!r}, wanted {want!r}")
    return report


def enrol(faculty: Any = None, courses: Sequence[Course] = COURSES) -> Tuple[Any, List[Report]]:
    """Teach every course to one faculty, then examine each of them.

    **One** faculty, not one per language, because that is the claim: three grammars held at once,
    each in its own tongue, none of them contaminating the others. Hindi and romanised Hinglish
    are the same grammar in two scripts and are kept as two tongues on purpose — a language is not
    a spelling, and if what she learned were a spelling the second would come free.
    """
    if faculty is None:
        from nyxara.njp.language import LanguageFaculty
        faculty = LanguageFaculty()
    reports = []
    for course in courses:
        report = teach(course, faculty)
        reports.append(examine(course, faculty, report))
    glosses(faculty)
    return faculty, reports


def summary(reports: Sequence[Report]) -> str:
    lines = ["", "NYXARA · NJP — English and Hindi, taught by hand", ""]
    lines.append("  language     shapes   read       said       wug        note")
    lines.append("  " + "-" * 76)
    for report in reports:
        lines.append(f"  {report.language:<12} {report.shapes:>4}   "
                     f"{report.read_right:>3}/{report.read_total:<6} "
                     f"{report.said_right:>3}/{report.said_total:<6} "
                     f"{report.wug_right:>3}/{report.wug_total:<6} {report.note[:34]}")
    right = sum(r.read_right + r.said_right + r.wug_right for r in reports)
    total = sum(r.read_total + r.said_total + r.wug_total for r in reports)
    lines.append("")
    lines.append(f"  items         {right} / {total}")
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """``python -m nyxara.njp.lessons [--json] [--misses]``."""
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Teach NJP English and Hindi.")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--misses", action="store_true", help="print what she got wrong")
    args = parser.parse_args(list(argv) if argv is not None else None)

    _faculty, reports = enrol()
    if args.json:
        print(json.dumps([report.to_dict() for report in reports], indent=2, ensure_ascii=False))
        return 0
    print(summary(reports))
    if args.misses:
        for report in reports:
            for miss in report.misses:
                print(f"  {report.language}: {miss[:150]}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
