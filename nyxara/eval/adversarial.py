"""NYXARA · eval/adversarial.py — natural language she was not written for (🗡, NJP V.09).

**Why the seven-stage curve was not enough.** :mod:`nyxara.eval.intelligence` reads 1.00 on six
stages and it is not wrong to: those stages hold the *surface* fixed and vary the inference, which
is exactly what makes them a clean read on reasoning. Its seventh stage varies the surface, and
varies it within the extractor's own idiom — articles, plurals, and three verbs that are already
synonyms in ``_PREDICATE_ALIASES``. A benchmark whose sentences are generated from the same
vocabulary the parser was built around measures the parser against itself.

Measured through that gap, the reasoning core was in good order and the road into it was not:

* ``"Zorbins don't need glarn."`` grounded to ``("Zorbins don't", requires, glarn)`` — the
  negation became part of the **subject**, so the denial of a claim was stored as a *different
  entity's assertion of it*. The string ``negat`` did not appear in ``grounding.py``.
* ``"Zibbies eat crystals."`` grounded to nothing at all, because ``eat`` is not in the verb
  lexicon — and the following question then answered from a neighbouring ``is_a`` edge rather
  than abstaining.

Neither is a reasoning failure and neither was visible to a template-shaped stage.

**What this module scores, and why it is four numbers rather than one.** A single accuracy hides
the difference between the two ways of being wrong, and they have opposite remedies:

=================  ====================================================================
``concept``        Did she name the right thing? The classical accuracy.
``relation``       Did she answer the relation she was *asked about*, or a neighbouring
                   one that happened to be reachable from the same entity?
``uncertainty``    Is her epistemic state right — ``known`` where she was taught,
                   ``unknown`` where she was not? Scored on both, so silence about
                   everything cannot win it.
``hallucination``  How often did an untaught relation come back with content anyway?
                   **Lower is better**, and it is the one number a cautious brain can
                   improve by abstaining — which is why it is reported apart from
                   ``concept`` rather than folded into it.
=================  ====================================================================

A change that lifts ``concept`` while lifting ``hallucination`` has bought coverage with lies, and
the report is arranged so that trade cannot be reported as an improvement.

**The families.** Each holds the fact fixed and varies exactly one thing about how it is said or
asked, so a drop names its own cause:

``surface``      one relation, asked seven ways — fronted wh, trailing wh, imperative
                 ("tell me…"), nominal ("what's necessary for…"), polar, emphatic, and
                 Hinglish. These are the forms an ordinary sentence arrives in.
``negation``     the same relation asserted and denied. A brain that scores here is one
                 whose answer *changes* when the sentence is negated.
``open_verb``    relations stated with verbs no lexicon anticipated. The point is that the
                 verb is arbitrary: if extraction depends on a closed list of verbs then
                 what she can think about is that list.
``polar``        yes/no questions, asked where the answer is yes, no, and unknown. Three
                 distinct states that a two-valued reply cannot represent.
``neighbour``    the trap. An entity she holds facts about, asked a relation she was never
                 taught. The only correct answer is abstention, and any content is a
                 neighbouring fact being passed off as this one.

**Vocabulary is generated and disjoint.** Same rule as the seven-stage curve, same reason: an item
answerable from anything but that item's own teaching phase proves nothing. A fresh brain per
family, so no family's teaching can carry another's questions.

Honest scope: this measures the **language surface**, not intelligence. A brain that scored 1.00
on every family here would have shown only that it can be talked to in ordinary English and
Hinglish — which is a precondition for the reasoning underneath being reachable, and nothing more.

Run it::

    python -m nyxara.eval --adversarial
    python -m nyxara.eval --adversarial --seed 7
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

__all__ = [
    "ItemResult", "FamilyResult", "AdversarialReport",
    "run_adversarial_benchmark", "FAMILY_NAMES",
]

FAMILY_NAMES: Tuple[str, ...] = (
    "surface",
    "negation",
    "open_verb",
    "polar",
    "neighbour",
)

Preparer = Callable[[Any], None]

#: Meaningless on purpose — see the seven-stage curve's note. A term that looks like a word
#: invites the reader to ask whether the answer is *true*, when the only thing under test is
#: whether it follows from what this item said.
_SYLLABLES = ("zor", "gla", "vim", "keth", "bru", "onn", "sil", "wex", "dra", "phel",
              "mub", "tarn", "iqu", "yol", "cresh", "nuv", "orb", "zeph", "kai", "lum")


def _vocabulary(rng: random.Random, n: int, *, tag: str) -> List[str]:
    """``n`` distinct nonsense terms, stable for a seed and unique to this ``tag``."""
    out: List[str] = []
    seen = set()
    while len(out) < n:
        word = tag + "".join(rng.choice(_SYLLABLES) for _ in range(2)) + str(len(out))
        if word not in seen:
            seen.add(word)
            out.append(word)
    return out


@dataclass
class ItemResult:
    """One asked question and every dimension it was scored on.

    Kept per item rather than accumulated, because the useful output of this benchmark is not
    the mean — it is *which surface form failed*, which is unreadable once the items are summed.
    """

    family: str = ""
    asked: str = ""
    expected: str = ""
    answer: str = ""
    epistemic: str = ""
    #: ``None`` where the dimension does not apply to this item — a ``neighbour`` item has no
    #: correct concept to name, so scoring it 0 for concept would punish the right behaviour.
    concept_ok: Optional[bool] = None
    relation_ok: Optional[bool] = None
    uncertainty_ok: Optional[bool] = None
    hallucinated: Optional[bool] = None
    note: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {"family": self.family, "asked": self.asked, "expected": self.expected,
                "answer": self.answer, "epistemic": self.epistemic,
                "concept_ok": self.concept_ok, "relation_ok": self.relation_ok,
                "uncertainty_ok": self.uncertainty_ok, "hallucinated": self.hallucinated,
                "note": self.note}


def _rate(values: Sequence[Optional[bool]]) -> Optional[float]:
    """Fraction true over the items the dimension applied to, or ``None`` if it applied to none."""
    live = [v for v in values if v is not None]
    if not live:
        return None
    return sum(1 for v in live if v) / len(live)


@dataclass
class FamilyResult:
    """One family's items and its four rates."""

    family: str = ""
    items: List[ItemResult] = field(default_factory=list)
    taught: int = 0
    ms: float = 0.0
    note: str = ""

    @property
    def concept(self) -> Optional[float]:
        return _rate([i.concept_ok for i in self.items])

    @property
    def relation(self) -> Optional[float]:
        return _rate([i.relation_ok for i in self.items])

    @property
    def uncertainty(self) -> Optional[float]:
        return _rate([i.uncertainty_ok for i in self.items])

    @property
    def hallucination(self) -> Optional[float]:
        """Fraction of applicable items that returned content where none was licensed. Lower is
        better — this is the only rate on the report where that is true, and it is labelled."""
        return _rate([i.hallucinated for i in self.items])

    @property
    def failures(self) -> List[ItemResult]:
        return [i for i in self.items
                if i.concept_ok is False or i.relation_ok is False
                or i.uncertainty_ok is False or i.hallucinated is True]

    def to_dict(self) -> Dict[str, Any]:
        return {"family": self.family, "taught": self.taught, "items": len(self.items),
                "concept": self.concept, "relation": self.relation,
                "uncertainty": self.uncertainty, "hallucination": self.hallucination,
                "ms": round(self.ms, 1), "note": self.note,
                "detail": [i.to_dict() for i in self.items]}


@dataclass
class AdversarialReport:
    """Every family, and the four rates pooled over all items."""

    seed: int = 0
    families: List[FamilyResult] = field(default_factory=list)

    def by_name(self, name: str) -> Optional[FamilyResult]:
        for fam in self.families:
            if fam.family == name:
                return fam
        return None

    @property
    def _all_items(self) -> List[ItemResult]:
        return [i for fam in self.families for i in fam.items]

    @property
    def concept(self) -> Optional[float]:
        return _rate([i.concept_ok for i in self._all_items])

    @property
    def relation(self) -> Optional[float]:
        return _rate([i.relation_ok for i in self._all_items])

    @property
    def uncertainty(self) -> Optional[float]:
        return _rate([i.uncertainty_ok for i in self._all_items])

    @property
    def hallucination(self) -> Optional[float]:
        return _rate([i.hallucinated for i in self._all_items])

    def to_dict(self) -> Dict[str, Any]:
        return {"seed": self.seed,
                "concept": self.concept, "relation": self.relation,
                "uncertainty": self.uncertainty, "hallucination": self.hallucination,
                "families": [f.to_dict() for f in self.families]}

    def render(self) -> str:
        def _fmt(v: Optional[float]) -> str:
            return "  n/a " if v is None else f"{v:5.2f}"

        lines = [
            "NYXARA — adversarial natural-language benchmark",
            "=" * 78,
            f"seed {self.seed}   ·   hallucination: LOWER is better",
            "",
            f"{'family':<12} {'concept':>8} {'relation':>9} {'uncert.':>8} {'halluc.':>8} "
            f"{'items':>6}  note",
            "-" * 78,
        ]
        for fam in self.families:
            lines.append(
                f"{fam.family:<12} {_fmt(fam.concept):>8} {_fmt(fam.relation):>9} "
                f"{_fmt(fam.uncertainty):>8} {_fmt(fam.hallucination):>8} "
                f"{len(fam.items):>6}  {fam.note}")
        lines.append("-" * 78)
        lines.append(
            f"{'POOLED':<12} {_fmt(self.concept):>8} {_fmt(self.relation):>9} "
            f"{_fmt(self.uncertainty):>8} {_fmt(self.hallucination):>8} "
            f"{len(self._all_items):>6}")
        failures = [i for fam in self.families for i in fam.failures]
        if failures:
            lines.append("")
            lines.append(f"failures ({len(failures)}):")
            for item in failures[:40]:
                flags = []
                if item.concept_ok is False:
                    flags.append("concept")
                if item.relation_ok is False:
                    flags.append("relation")
                if item.uncertainty_ok is False:
                    flags.append("uncertainty")
                if item.hallucinated is True:
                    flags.append("HALLUCINATED")
                lines.append(f"  [{item.family}/{','.join(flags)}] {item.asked!r}")
                lines.append(f"      expected {item.expected!r}  got {item.answer!r} "
                             f"({item.epistemic or 'no state'})")
        return "\n".join(lines)


def _brain(*, prepare: Optional[Preparer] = None) -> Any:
    """A fresh brain, or ``None`` if NJP will not construct in this environment."""
    try:
        from nyxara.njp.brain import NJPBrain
        brain = NJPBrain()
    except Exception:  # noqa: BLE001
        return None
    if prepare is not None:
        prepare(brain)
    return brain


def _teach(brain: Any, sentences: Sequence[str]) -> int:
    """State each sentence as the Master would. Returns how many actually grounded."""
    landed = 0
    for sentence in sentences:
        try:
            thought = brain.think(sentence)
            grounding = getattr(getattr(thought, "percept", None), "grounding", None)
            if getattr(grounding, "triples", None):
                landed += 1
        except Exception:  # noqa: BLE001
            continue
    return landed


def _ask(brain: Any, question: str) -> Tuple[str, str]:
    """The answer and the epistemic state she attached to it.

    Both, because three of the four dimensions here are about the *state* rather than the text,
    and a benchmark that reads only the string cannot tell an honest abstention from a failure.
    """
    try:
        thought = brain.think(question)
        answer = str(getattr(thought, "answer", "") or "").strip().lower()
        state = str(getattr(thought, "epistemic", "") or "")
        return answer, state
    except Exception:  # noqa: BLE001
        return "", ""


def _names(answer: str, term: str) -> bool:
    """Did the answer name this term? Substring, one direction only — see ``intelligence._hit``."""
    term = str(term or "").strip().lower()
    return bool(term) and bool(answer) and term in answer


# --------------------------------------------------------------------------- #
# surface — one relation, asked seven ways
# --------------------------------------------------------------------------- #

#: Seven question forms for one relation, each a template over ``{e}`` (the entity) and ``{o}``
#: (the object, used only by the polar form). Nothing exotic: fronted wh, trailing wh, an
#: imperative, a nominalisation, a polar, an emphatic, and the Hinglish the Master actually types.
_SURFACE_FORMS: Tuple[Tuple[str, str], ...] = (
    ("fronted-wh",  "what does a {e} need?"),
    ("trailing-wh", "{e}s need what?"),
    ("imperative",  "tell me what a {e} requires."),
    ("nominal",     "what's necessary for a {e}?"),
    ("emphatic",    "a {e} needs what exactly?"),
    ("hinglish",    "{e} ko kya chahiye?"),
    ("bare",        "{e} needs?"),
)


def _family_surface(rng: random.Random, prepare: Optional[Preparer]) -> FamilyResult:
    """One taught relation, asked in seven ordinary surface forms."""
    out = FamilyResult(family="surface")
    started = time.perf_counter()
    try:
        brain = _brain(prepare=prepare)
        if brain is None:
            out.note = "NJP unavailable"
            return out
        entity, obj = _vocabulary(rng, 2, tag="sur")
        out.taught = _teach(brain, [f"{entity}s need {obj}."])
        for label, template in _SURFACE_FORMS:
            asked = template.format(e=entity, o=obj)
            answer, state = _ask(brain, asked)
            named = _names(answer, obj)
            out.items.append(ItemResult(
                family="surface", asked=asked, expected=obj, answer=answer, epistemic=state,
                concept_ok=named,
                # She was taught this exact relation, so answering it with something that is not
                # the taught object is answering a different relation.
                relation_ok=named or not answer,
                # Taught: the honest state is a positive one. Abstaining here is a miss, not a
                # virtue — she was told.
                uncertainty_ok=bool(named),
                hallucinated=bool(answer) and not named,
                note=label))
        out.note = "one taught relation, seven surface forms"
        return out
    except Exception as exc:  # noqa: BLE001
        out.note = f"error: {exc}"
        return out
    finally:
        out.ms = (time.perf_counter() - started) * 1000.0


# --------------------------------------------------------------------------- #
# negation — the same relation asserted and denied
# --------------------------------------------------------------------------- #

_NEGATION_FORMS: Tuple[str, ...] = (
    "{e}s don't need {o}.",
    "{e}s do not need {o}.",
    "{e}s never need {o}.",
    "{e}s ko {o} nahi chahiye.",
)


def _family_negation(rng: random.Random, prepare: Optional[Preparer]) -> FamilyResult:
    """A denied relation must not come back as an asserted one.

    Scored on the *question after the denial*: told ``X does not need Y`` and asked what X needs,
    the two acceptable answers are silence and something that is not Y. Returning Y is the
    failure this family exists to catch, and it is scored as a hallucination rather than merely
    a miss because it is not a gap in what she knows — it is the opposite of it.
    """
    out = FamilyResult(family="negation")
    started = time.perf_counter()
    try:
        for i, template in enumerate(_NEGATION_FORMS):
            brain = _brain(prepare=prepare)
            if brain is None:
                out.note = "NJP unavailable"
                return out
            entity, obj = _vocabulary(rng, 2, tag=f"neg{i}")
            out.taught += _teach(brain, [template.format(e=entity, o=obj)])
            asked = f"what does a {entity} need?"
            answer, state = _ask(brain, asked)
            asserted = _names(answer, obj)
            out.items.append(ItemResult(
                family="negation", asked=asked, expected=f"NOT {obj}", answer=answer,
                epistemic=state,
                concept_ok=not asserted,
                relation_ok=not asserted,
                # A denial leaves her knowing something — that the relation does not hold — but
                # it does not license a positive answer. Either silence or an explicit negative
                # is right; a confident positive is not.
                uncertainty_ok=not asserted,
                hallucinated=asserted,
                note=template))
        out.note = f"{len(_NEGATION_FORMS)} denials, each asked as though asserted"
        return out
    except Exception as exc:  # noqa: BLE001
        out.note = f"error: {exc}"
        return out
    finally:
        out.ms = (time.perf_counter() - started) * 1000.0


# --------------------------------------------------------------------------- #
# open_verb — relations whose verb no lexicon anticipated
# --------------------------------------------------------------------------- #

#: ``(verb as taught, verb as asked)``. Every one is an ordinary transitive English verb and
#: none is a synonym of ``need``/``require``, which is the whole point: if extraction is bound to
#: a verb list then the set of things she can be told is that list.
_OPEN_VERBS: Tuple[Tuple[str, str], ...] = (
    ("eat", "eat"),
    ("avoid", "avoid"),
    ("produce", "produce"),
    ("carry", "carry"),
    ("hunt", "hunt"),
    ("emit", "emit"),
)


def _family_open_verb(rng: random.Random, prepare: Optional[Preparer]) -> FamilyResult:
    """Arbitrary verbs, taught and asked back."""
    out = FamilyResult(family="open_verb")
    started = time.perf_counter()
    try:
        for i, (taught_verb, asked_verb) in enumerate(_OPEN_VERBS):
            brain = _brain(prepare=prepare)
            if brain is None:
                out.note = "NJP unavailable"
                return out
            entity, obj = _vocabulary(rng, 2, tag=f"vrb{i}")
            out.taught += _teach(brain, [f"{entity}s {taught_verb} {obj}."])
            asked = f"what does a {entity} {asked_verb}?"
            answer, state = _ask(brain, asked)
            named = _names(answer, obj)
            out.items.append(ItemResult(
                family="open_verb", asked=asked, expected=obj, answer=answer, epistemic=state,
                concept_ok=named,
                relation_ok=named or not answer,
                uncertainty_ok=bool(named),
                hallucinated=bool(answer) and not named,
                note=taught_verb))
        out.note = f"{len(_OPEN_VERBS)} verbs outside the predicate lexicon"
        return out
    except Exception as exc:  # noqa: BLE001
        out.note = f"error: {exc}"
        return out
    finally:
        out.ms = (time.perf_counter() - started) * 1000.0


# --------------------------------------------------------------------------- #
# polar — yes, no, and unknown are three states
# --------------------------------------------------------------------------- #

def _family_polar(rng: random.Random, prepare: Optional[Preparer]) -> FamilyResult:
    """Yes/no questions where the truth is yes, no, and genuinely unknown.

    The third is the one that matters. A polar question has a two-valued *surface* and a
    three-valued answer set, and a brain that cannot say the third will answer one of the first
    two about something it was never told.
    """
    out = FamilyResult(family="polar")
    started = time.perf_counter()
    try:
        brain = _brain(prepare=prepare)
        if brain is None:
            out.note = "NJP unavailable"
            return out
        entity, yes_obj, no_obj, unknown_obj = _vocabulary(rng, 4, tag="pol")
        out.taught = _teach(brain, [
            f"{entity}s need {yes_obj}.",
            f"{entity}s don't need {no_obj}.",
        ])
        cases = (
            ("yes", yes_obj, ("yes", "haan", yes_obj)),
            ("no", no_obj, ("no", "nahi", "not")),
            ("unknown", unknown_obj, ("", "unknown", "pata nahi", "don't know", "not sure")),
        )
        for label, obj, accepted in cases:
            asked = f"do {entity}s need {obj}?"
            answer, state = _ask(brain, asked)
            if label == "unknown":
                # Silence is a correct answer here and the most likely correct one, so accept it
                # explicitly rather than letting an empty string fail a substring test.
                ok = (not answer) or any(_names(answer, a) for a in accepted if a)
            else:
                ok = any(_names(answer, a) for a in accepted if a)
            out.items.append(ItemResult(
                family="polar", asked=asked, expected=label, answer=answer, epistemic=state,
                concept_ok=ok,
                relation_ok=ok,
                uncertainty_ok=ok,
                # Content where the truth is unknown is the hallucination; on yes/no items a
                # wrong answer is a miss, not an invention, so the dimension does not apply.
                hallucinated=(bool(answer) and not ok) if label == "unknown" else None,
                note=label))
        out.note = "polar questions where the truth is yes, no, and unknown"
        return out
    except Exception as exc:  # noqa: BLE001
        out.note = f"error: {exc}"
        return out
    finally:
        out.ms = (time.perf_counter() - started) * 1000.0


# --------------------------------------------------------------------------- #
# neighbour — the trap
# --------------------------------------------------------------------------- #

#: ``(taught sentence template, question template)``. In each row she is taught one relation
#: about the entity and asked a *different* one. The entity is familiar, the relation is not,
#: and the only correct answer is that she was not told.
_NEIGHBOURS: Tuple[Tuple[str, str], ...] = (
    ("{e} is a {k}.",        "what does a {e} need?"),
    ("{e}s need {o}.",       "what is a {e} made of?"),
    ("{e}s need {o}.",       "where does a {e} live?"),
    ("{e} is a {k}.",        "what does a {e} eat?"),
)


def _family_neighbour(rng: random.Random, prepare: Optional[Preparer]) -> FamilyResult:
    """A familiar entity, a relation she was never taught. Only abstention is correct."""
    out = FamilyResult(family="neighbour")
    started = time.perf_counter()
    try:
        for i, (taught_t, asked_t) in enumerate(_NEIGHBOURS):
            brain = _brain(prepare=prepare)
            if brain is None:
                out.note = "NJP unavailable"
                return out
            entity, kind, obj = _vocabulary(rng, 3, tag=f"nbr{i}")
            out.taught += _teach(brain, [taught_t.format(e=entity, k=kind, o=obj)])
            asked = asked_t.format(e=entity, k=kind, o=obj)
            answer, state = _ask(brain, asked)
            silent = not answer
            out.items.append(ItemResult(
                family="neighbour", asked=asked, expected="(abstention)", answer=answer,
                epistemic=state,
                # There is no right thing to name, so the concept dimension does not apply — see
                # ItemResult. Scoring it would mean scoring her 0 for behaving correctly.
                concept_ok=None,
                relation_ok=silent,
                uncertainty_ok=silent or "unknown" in state.lower(),
                hallucinated=not silent,
                note=taught_t))
        out.note = f"{len(_NEIGHBOURS)} untaught relations on familiar entities"
        return out
    except Exception as exc:  # noqa: BLE001
        out.note = f"error: {exc}"
        return out
    finally:
        out.ms = (time.perf_counter() - started) * 1000.0


_FamilyRunner = Callable[[random.Random, Optional[Preparer]], FamilyResult]

_FAMILIES: Tuple[Tuple[str, _FamilyRunner], ...] = (
    ("surface", _family_surface),
    ("negation", _family_negation),
    ("open_verb", _family_open_verb),
    ("polar", _family_polar),
    ("neighbour", _family_neighbour),
)


def run_adversarial_benchmark(*, seed: int = 20260823,
                              families: Optional[Sequence[str]] = None,
                              prepare: Optional[Preparer] = None) -> AdversarialReport:
    """Run the adversarial families and return the four rates per family and pooled.

    ``prepare`` receives each fresh brain before its family runs, so a corpus loaded outside the
    benchmark can actually be measured by it — the same contract, and for the same reason, as
    :func:`nyxara.eval.intelligence.run_intelligence_benchmark`.
    """
    wanted = set(families) if families else set(FAMILY_NAMES)
    report = AdversarialReport(seed=seed)
    for name, runner in _FAMILIES:
        if name not in wanted:
            continue
        # A fresh RNG per family, keyed by name, so running one family alone draws the same
        # vocabulary it would have drawn in a full run. Without this, `--families negation`
        # and a full run would disagree and neither would be reproducible from the other.
        rng = random.Random(f"{seed}:{name}")
        report.families.append(runner(rng, prepare))
    return report
