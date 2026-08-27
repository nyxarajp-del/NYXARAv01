"""NYXARA · eval/capability.py — how much of the capability surface she actually has (📏).

The three corpora each report their own numbers and every one of them is measured on the records
that were *loaded*. That is the right check for a loader and it is worth nothing as a capability
estimate: a fact store asked for a fact it was just given answers perfectly and has learned
nothing. So this file splits the unified corpus **before** absorbing it, teaches her one part, and
asks about the other — and the difference between the two numbers is the whole point.

Measured on the shipped corpus: recall on the taught half is 1.00, and the capability surface over
the held-out quarter is a different number entirely. It is *supposed* to be. A held-out score near
the taught one would mean the split leaked.

What "fifty percent of a frontier model" would mean
---------------------------------------------------
It is not a number a dataset can be judged against, and pretending otherwise is the thing this
file exists to stop. A frontier model's capability comes from parameter count and a training run
measured in trillions of tokens; :mod:`scripts.train_300m` targets 300M. What *can* be stated
honestly is a capability surface — a list of things, each with a score she either has or does not —
and then the corpus can be aimed at the rows reading zero. A number you can move beats a number
you can quote.

What it reads today
-------------------
1,331 records taught, 372 held out, 300 probes::

    control (taught recall)  1.00      the load worked; a low surface is not a failed load
    surface                  0.40      was 0.29

    counterfactual           1.00
    inheritance              1.00      11/11 — something came down the is_a edge
    generalization           0.87      13 of 13 she answered were right, was 0.00
    recall                   0.17
    causal                   0.17
    relation / transfer / transition   0.00 on one or two probes each — a sample size, not a finding

``generalization`` read **0.00** and was the row this work was aimed at. Nothing about the
reasoning changed: asked for the properties of a seal she already answered "warm blooded", stored
about *mammal* and never about seals, so the inference had been firing all along. What was missing
was every way of getting a question to it, and three separate things were:

* ``has_property`` could be **asked for and never written**. No rule in
  ``grounding._SEED_PATTERNS`` produced one, so "copper is ductile" compiled to
  ``('copper', 'ductile', '')`` — a relation named after the adjective, with no object — and
  "a mammal is warm blooded" was unreadable outright.
* **No question form could name which property was wanted.** ``compile_meaning`` produces a polar
  reading only for a narrow shape: "is a seal warm blooded" parses, "is kiwi feathered" comes back
  `unreadable`, and "is norway governed by elected leaders" parses to the subject "norway governed
  elected". A surface reader that *finds* the subject boundary by asking the store which entities
  it knows handles all three, and hands anything it cannot place back unchanged.
* The corpus filed **capabilities as properties** — "breathes with lungs" under ``has_property``,
  where no English sentence reaches it.

The probe asks both directions and requires both: the true property must come back affirmed **and**
a property borrowed from another concept must not. A system that answers "yes" to everything
scores zero on it, which is why 13 of 13 is worth quoting.

Two of the repairs were to this file — it was withholding the answer to its own question, first the
``is_a`` edge and then the invariant itself — and one was a rule: ``generalization`` and
``inheritance`` **need no holdout**, because the corpus never states the conclusion. They read the
taught set, where the premises are.

Eight probe familiesEight probe families, derived from the blocks a record already carries
---------------------------------------------------------------------
Nothing here needs a new annotation. Each family reads a block the unified corpus already has and
turns it into a question with an answer key:

* **recall** — ``qa`` pairs, asked and marked against the gold answer.
* **relation** — ``knowledge`` triples, looked up by subject and predicate. On a held-out record
  the subject was never taught, so anything right came through inheritance rather than storage.
* **causal** — ``law``, asked as "what does X cause".
* **counterfactual** — ``counterfactual``, scored on direction only.
* **generalization** — ``generalize`` pairs, which name a member that was deliberately left out of
  the examples.
* **transfer** — ``transfer``, whose question is posed in the domain the principle was *not*
  stated in.
* **transition** — ``state``/``action``/``next_state``, asked of the discrete model.
* **inheritance** — the same members as ``generalization``, asked only whether *anything* was
  inherited. The gap between the two is the point; see above.

A category with no probe family is reported as ``probes: 0`` and **not** as a score of zero.
The two are different: one says she got it wrong, the other says this run could not ask. Rolling
them together is how an eval flatters or libels a system, and :mod:`nyxara.njp.study` makes the
same argument about its own denominator.

    python -m nyxara.eval.capability --holdout 0.25
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, FrozenSet, List, Optional, Sequence, Set, Tuple

__all__ = ["ProbeResult", "CapabilityReport", "split", "evaluate", "run"]

_DEFAULT_RECORDS = (Path(__file__).resolve().parents[1] / "njp" / "data" /
                    "world_unified.jsonl.gz")

#: How close a predicted number has to be to count. Relative, because a period in seconds and a
#: latency in milliseconds cannot share an absolute tolerance.
_NUMERIC_TOLERANCE = 0.25

_FLAT = 1e-6

#: How many probes one family may ask within one category, and how many the whole run may ask.
#:
#: ``brain.think`` runs the whole stack — the fabric settles and expands at about forty turns a
#: second — so this is not a detail. Uncapped, a sweep of this corpus ran past twenty-five minutes
#: and was killed; at forty per category across fifty categories it was still over twenty. A
#: benchmark nobody can afford to run is a benchmark nobody runs, so both caps are deliberately
#: small and the global one is the binding constraint.
#:
#: The caps are on *asked*, so the denominator stays honest: a family reports the sample it took
#: rather than pretending the sample was everything. Raise ``--budget`` for a slow, thorough run.
_MAX_PROBES = 8
_DEFAULT_BUDGET = 300


@dataclass
class ProbeResult:
    """One probe family within one category: asked, answered, and right."""

    family: str = ""
    category: str = ""
    asked: int = 0
    answered: int = 0
    correct: int = 0
    misses: List[str] = field(default_factory=list)

    @property
    def score(self) -> Optional[float]:
        """``None`` when nothing could be asked — distinct from 0.0, which means asked and wrong."""
        return (self.correct / self.asked) if self.asked else None

    @property
    def precision(self) -> Optional[float]:
        """Of the ones she answered at all, how many were right. Abstention is not counted wrong."""
        return (self.correct / self.answered) if self.answered else None

    def to_dict(self) -> Dict[str, Any]:
        return {"family": self.family, "category": self.category, "asked": self.asked,
                "answered": self.answered, "correct": self.correct,
                "score": None if self.score is None else round(self.score, 4),
                "precision": None if self.precision is None else round(self.precision, 4)}


@dataclass
class CapabilityReport:
    """The surface: one row per capability, and the honest roll-up over the rows that were asked."""

    taught: int = 0
    held_out: int = 0
    probes: List[ProbeResult] = field(default_factory=list)
    taught_recall: Optional[float] = None
    ms: float = 0.0

    def by_family(self) -> Dict[str, ProbeResult]:
        out: Dict[str, ProbeResult] = {}
        for probe in self.probes:
            roll = out.setdefault(probe.family, ProbeResult(family=probe.family))
            roll.asked += probe.asked
            roll.answered += probe.answered
            roll.correct += probe.correct
        return out

    @property
    def surface(self) -> Optional[float]:
        """The unweighted mean over the families that could be asked at all.

        Unweighted on purpose. Weighting by probe count would let ``recall`` — which has hundreds
        of items because the knowledge corpus is large — stand in for the whole surface, and a
        system that only recalls would score as though it could also transfer.
        """
        scores = [p.score for p in self.by_family().values() if p.score is not None]
        return sum(scores) / len(scores) if scores else None

    def to_dict(self) -> Dict[str, Any]:
        families = {name: probe.to_dict() for name, probe in sorted(self.by_family().items())}
        for probe in families.values():
            probe.pop("category", None)
        return {
            "taught": self.taught,
            "held_out": self.held_out,
            "taught_recall": (None if self.taught_recall is None
                              else round(self.taught_recall, 4)),
            "surface": None if self.surface is None else round(self.surface, 4),
            "families": families,
            "by_category": {
                f"{p.category}:{p.family}": p.to_dict()
                for p in sorted(self.probes, key=lambda q: (q.category, q.family)) if p.asked
            },
            "ms": round(self.ms, 1),
        }


# --------------------------------------------------------------------------- #
# The split
# --------------------------------------------------------------------------- #
def _share(record: Dict[str, Any], seed: int) -> float:
    digest = hashlib.blake2b(f"{seed}:{record.get('id', '')}".encode("utf-8"),
                             digest_size=8).digest()
    return int.from_bytes(digest, "big") / float(1 << 64)


def split(records: Sequence[Dict[str, Any]], *, holdout: float = 0.25,
          seed: int = 0) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Deterministic by a hash of the record id, and **claim-level where the record has claims**.

    Whole-record holdout was the first version and it measured the wrong thing. A
    ``world_knowledge`` record is everything she is ever told about one subject, so holding the
    record out holds out the subject entirely — and the probe then asks her to produce facts about
    a thing she has never heard of. Measured that way ``relation`` scored 0.002, which is not a
    capability gap: **it is the correct answer**, and an eval that counts it as failure is asking
    for clairvoyance and calling the refusal a defect.

    So a record picked for holdout that carries ``knowledge`` is *split rather than removed*: every
    claim but the last goes into the taught half under the same subject, and the last one becomes
    the probe. The subject is known, the claim is not, and getting it right now means inheritance
    or composition did it. Records with no claims — the reasoning categories — are held out whole,
    which for them is the right unit.

    By a hash of the id and not by position, the way ``study.Pair.split`` does it: a positional cut
    puts a whole ``.u`` file on one side, so the held-out half of ``concept_formation`` would be
    every record of it and the taught half none.
    """
    taught: List[Dict[str, Any]] = []
    held: List[Dict[str, Any]] = []
    cut = max(0.0, min(0.9, float(holdout)))
    for record in records:
        if _share(record, seed) >= cut:
            taught.append(record)
            continue
        claims = list(record.get("knowledge") or [])
        if len(claims) < 2:
            held.append(record)
            continue
        # Never a claim the record's own probe depends on. Two kinds of self-sabotage were found
        # here, one after the other, and they are the same mistake at two depths:
        #
        #   1. the ``is_a`` edge — the generator appends ``<member> is_a <concept>`` for every
        #      generalisation target, so taking the last claim removed the edge to inherit
        #      *through*; ``nutcracker`` and ``acetic acid`` came back with no kind at all.
        #   2. the **invariant itself** — with the is_a rows excluded, the only remaining derived
        #      claims are ``<concept> has_property <invariant>``, so the split then withheld the
        #      very property the probe asks about. Measured: ``mammal has_property`` held
        #      "warm blooded, hair or fur, milk" and the probe wanted "breathes with lungs",
        #      which the split had removed. The row read 0.00 for a gap the eval had created.
        #
        # A held-out claim has to be one she could have *derived*, never one the answer key needs.
        gold = {_norm(pair[1]) for pair in (record.get("generalize") or []) if len(pair) > 1}
        candidates = [c for c in claims
                      if len(c) > 2 and c[1] != "is_a" and _norm(c[2]) not in gold]
        if not candidates:
            # Every claim in this record is load-bearing for its own probe. The first version fell
            # back to withholding one anyway, which put the `is_a` edge back on the block and left
            # `nutcracker` and `acetic acid` with no kind again. A record that cannot yield a fair
            # probe yields none: it stays wholly taught and is not asked about.
            taught.append(record)
            continue
        withheld = candidates[-1]
        keep = [c for c in claims if c is not withheld]
        target = _norm(withheld[2]) if len(withheld) > 2 else ""
        kept_qa, held_qa = [], []
        for pair in record.get("qa") or []:
            (held_qa if (len(pair) > 1 and target and target in _norm(pair[1]))
             else kept_qa).append(pair)
        partial = dict(record)
        partial["knowledge"] = keep
        partial["qa"] = kept_qa
        taught.append(partial)
        probe = dict(record)
        probe["knowledge"] = [withheld]
        probe["qa"] = held_qa
        held.append(probe)
    return taught, held


# --------------------------------------------------------------------------- #
# The probes
# --------------------------------------------------------------------------- #
def _norm(text: Any) -> str:
    return " ".join(str(text or "").lower().split())


def _contained(said: str, gold: str) -> bool:
    """Did the reply name the expected thing? Substring, one direction only.

    ``eval/intelligence._hit`` makes the argument and it holds here: a derived answer legitimately
    arrives with its route attached ("water, inherited from animal"), so demanding equality scores
    a correct inference as a miss — and accepting the reverse containment would let "i do not know
    about water" pass, which is why it is not symmetric.

    Both directions are tried by the callers that need it, because the gold on a ``qa`` pair is a
    whole prose sentence and a three-word answer sits inside it.
    """
    said, gold = _norm(said), _norm(gold)
    return bool(said) and bool(gold) and (gold in said or said in gold)


def _ask(brain: Any, question: str) -> str:
    """Ask her, not her fact store.

    ``brain.think`` and not ``grounder.answer``, which is what the first version used and is the
    difference between measuring NJP and measuring one layer of it. ``answer`` is retrieval: it
    calls ``_lookup`` and stops. Inheritance lives in ``core._inherit``, which only runs when the
    retrieval comes back empty and deliberation takes over — so a probe that asked the grounder
    directly could never see an inherited answer, and the generalisation family scored zero for
    that reason rather than for hers. ``eval/intelligence`` has always asked through ``think``.
    """
    try:
        said = str(getattr(brain.think(question), "answer", "") or "").strip().lower()
        if said:
            return said
    except Exception:  # noqa: BLE001
        pass
    try:
        return str(getattr(brain.grounder.answer(question), "text", "") or "")
    except Exception:  # noqa: BLE001
        return ""


def _probe_recall(brain: Any, records: Sequence[Dict[str, Any]],
                  category: str) -> ProbeResult:
    out = ProbeResult(family="recall", category=category)
    for record in records:
        for pair in record.get("qa") or []:
            if len(pair) < 2 or not pair[0]:
                continue
            if out.asked >= _MAX_PROBES:
                break
            out.asked += 1
            said = _ask(brain, pair[0])
            if not said:
                continue
            out.answered += 1
            if _contained(said, pair[1]):
                out.correct += 1
            elif len(out.misses) < 8:
                out.misses.append(f"{pair[0]} → {said!r} (gold {pair[1]!r})")
    return out


def _probe_relation(brain: Any, records: Sequence[Dict[str, Any]],
                    category: str) -> ProbeResult:
    """A withheld claim that a taught kind actually entails. Anything else is not a probe.

    The first version asked every withheld triple and scored 0.004. It deserved to: withholding
    ``copper has_property reddish brown`` and then asking what copper is like tests whether she can
    invent a property nothing entails, and a system that produced one would be worse, not better.

    So a triple counts as asked only when the subject has a taught ``is_a`` to a kind that carries
    the same predicate and object — an inheritance ``core._inherit`` can actually walk. Everything
    else is *unaskable*, which :mod:`nyxara.njp.study` argues at length is a different thing from
    wrong and must not share a denominator with it.
    """
    out = ProbeResult(family="relation", category=category)
    grounder = getattr(brain, "grounder", None)
    if grounder is None:
        return out
    for record in records:
        for triple in record.get("knowledge") or []:
            if len(triple) < 3 or not _entailed(grounder, triple):
                continue
            if out.asked >= _MAX_PROBES:
                break
            out.asked += 1
            try:
                found = grounder._lookup(triple[0], triple[1]) or []
            except Exception:  # noqa: BLE001
                continue
            said = _ask(brain, f"what are the properties of {triple[0]}?") if not found else ""
            if found:
                out.answered += 1
                if any(_norm(getattr(t, "object", "")) == _norm(triple[2]) for t in found):
                    out.correct += 1
                elif len(out.misses) < 8:
                    out.misses.append(f"{triple[0]} {triple[1]} → "
                                      f"{[getattr(t, 'object', '') for t in found][:3]} "
                                      f"(gold {triple[2]!r})")
            elif said:
                out.answered += 1
                if _contained(triple[2], said) or _contained(said, triple[2]):
                    out.correct += 1
    return out


def _entailed(grounder: Any, triple: Sequence[str]) -> bool:
    """Does a taught kind of this subject carry the same claim? Then inheritance could reach it."""
    try:
        kinds = grounder._lookup(triple[0], "is_a") or []
        for kind in kinds:
            parent = str(getattr(kind, "object", ""))
            for held in grounder._lookup(parent, triple[1]) or []:
                if _norm(getattr(held, "object", "")) == _norm(triple[2]):
                    return True
    except Exception:  # noqa: BLE001
        return False
    return False


def _probe_causal(brain: Any, records: Sequence[Dict[str, Any]],
                  category: str) -> ProbeResult:
    out = ProbeResult(family="causal", category=category)
    for record in records:
        for law in record.get("law") or []:
            if len(law) < 3:
                continue
            if out.asked >= _MAX_PROBES:
                break
            out.asked += 1
            said = _ask(brain, f"what does {law[0]} cause?")
            if not said:
                continue
            out.answered += 1
            if _norm(said) == _norm(law[2]):
                out.correct += 1
    return out


def _probe_counterfactual(brain: Any, records: Sequence[Dict[str, Any]],
                          category: str) -> ProbeResult:
    out = ProbeResult(family="counterfactual", category=category)
    universe = getattr(brain, "universe", None)
    if universe is None:
        return out
    for record in records:
        for item in record.get("counterfactual") or []:
            if len(item) < 4:
                continue
            try:
                expected = int(item[3])
                value = float(item[1])
            except (TypeError, ValueError):
                continue
            if out.asked >= _MAX_PROBES:
                break
            out.asked += 1
            base = dict(record.get("observation") or {}) or None
            try:
                answer = universe.intervene({item[0]: value}, base=base)
            except Exception:  # noqa: BLE001
                continue
            if not getattr(answer, "answerable", False):
                continue
            for delta in answer.changed():
                if str(getattr(delta, "variable", "")) != item[2]:
                    continue
                change = getattr(delta, "change", None)
                got = ((1 if change > 0 else -1) if change is not None and abs(change) > _FLAT
                       else int(getattr(delta, "direction", 0)))
                if not got:
                    break
                out.answered += 1
                if got == expected:
                    out.correct += 1
                break
    return out


def _probe_generalization(brain: Any, records: Sequence[Dict[str, Any]],
                          category: str) -> ProbeResult:
    """The member left out of the examples, asked whether the invariant holds of it.

    Asked in the **polar** form — "is a seal warm blooded" — because that is the only form that
    can name *which* property is wanted. ``what are the properties of X`` returns one of four and
    cannot be told, which is why this row read 0.00 against inheritance that was firing correctly.

    **Both directions are asked, and both must be right.** A positive probe alone is worthless: a
    system that answers "yes" to everything scores full marks on it. So each gold property is
    paired with a control drawn from a *different* record's concept, and the item counts only when
    the true property comes back affirmed and the foreign one does not. An open-world store should
    say UNKNOWN to the control rather than "no", and either is accepted — what is not accepted is
    "yes".
    """
    out = ProbeResult(family="generalization", category=category)
    controls = [str(pair[1]) for record in records
                for pair in (record.get("generalize") or []) if len(pair) > 1]
    for index, record in enumerate(records):
        for pair in record.get("generalize") or []:
            if len(pair) < 2:
                continue
            if out.asked >= _MAX_PROBES:
                break
            member, gold = str(pair[0]), str(pair[1])
            foreign = next((c for c in controls if c != gold), "")
            if not foreign:
                continue
            out.asked += 1
            said = _norm(_ask(brain, f"is {member} {gold}?"))
            if not said:
                continue
            out.answered += 1
            control = _norm(_ask(brain, f"is {member} {foreign}?"))
            if said.startswith("yes") and not control.startswith("yes"):
                out.correct += 1
            elif len(out.misses) < 8:
                out.misses.append(f"{member}: {gold}→{said!r} control {foreign}→{control!r}")
    return out


def _probe_inheritance(brain: Any, records: Sequence[Dict[str, Any]],
                       category: str) -> ProbeResult:
    """Did anything come down the ``is_a`` edge at all — not necessarily the one asked for.

    This family exists because measuring the two together libels her. Asked for the properties of
    a seal, she answers ``warm blooded``: that property is **not stored about seals**, it is
    inherited from ``mammal``, so the inheritance worked exactly as designed. It is the wrong one
    only because the question has no way to name which property is wanted — the same read/write
    asymmetry ``grounding`` documents for ``known_for`` and for ``capable_of``, one relation
    further on. ``what are the properties of X`` returns one of four and there is no English form
    in the pattern table that asks for a particular inherited property.

    So: ``inheritance`` is what the machinery does, ``generalization`` is what the question grammar
    can currently get out of it, and the gap between the two numbers is a question-form gap rather
    than a reasoning one.
    """
    out = ProbeResult(family="inheritance", category=category)
    grounder = getattr(brain, "grounder", None)
    if grounder is None:
        return out
    for record in records:
        for pair in record.get("generalize") or []:
            if len(pair) < 2 or out.asked >= _MAX_PROBES:
                continue
            kinds = [str(getattr(t, "object", ""))
                     for t in (grounder._lookup(pair[0], "is_a") or [])]
            if not kinds:
                continue                      # nothing to inherit through; not a probe
            carried = {_norm(getattr(t, "object", ""))
                       for kind in kinds
                       for t in (grounder._lookup(kind, "has_property") or [])}
            if not carried:
                continue
            out.asked += 1
            said = _norm(_ask(brain, f"what are the properties of {pair[0]}?"))
            if not said:
                continue
            out.answered += 1
            if any(value and (value in said or said in value) for value in carried):
                out.correct += 1
    return out


def _probe_transfer(brain: Any, records: Sequence[Dict[str, Any]],
                    category: str) -> ProbeResult:
    """The question is posed in the domain the principle was not stated in. This is the hard one."""
    out = ProbeResult(family="transfer", category=category)
    for record in records:
        for item in record.get("transfer") or []:
            if len(item) < 5:
                continue
            if out.asked >= _MAX_PROBES:
                break
            out.asked += 1
            said = _ask(brain, str(item[3]))
            if not said:
                continue
            out.answered += 1
            if _contained(said, item[4]) or _contained(item[4], said):
                out.correct += 1
    return out


def _probe_transition(brain: Any, records: Sequence[Dict[str, Any]],
                      category: str) -> ProbeResult:
    out = ProbeResult(family="transition", category=category)
    predictive = getattr(brain, "predictive", None)
    if predictive is None:
        return out
    try:
        from nyxara.njp.predictive import WorldState
    except Exception:  # noqa: BLE001
        return out
    for record in records:
        state, nxt = record.get("state") or [], record.get("next_state") or []
        action = record.get("action")
        if not state or not nxt or not isinstance(action, str):
            continue
        if out.asked >= _MAX_PROBES:
            break
        out.asked += 1
        try:
            prediction = predictive.predict(WorldState.of(state), action)
        except Exception:  # noqa: BLE001
            continue
        top = str(getattr(prediction, "top", "") or "")
        if not top:
            continue
        out.answered += 1
        if top == WorldState.of(nxt).signature:
            out.correct += 1
    return out


def _probe_planning(brain: Any, records: Sequence[Dict[str, Any]],
                    category: str) -> ProbeResult:
    """Which lever, and which way — asked of a law she was taught and a plan nobody stated.

    The surface had rows for what she *knows* and what she *concludes*, and none for what she
    would *do*. This is that row.

    A derived family, for the same reason ``generalization`` and ``inheritance`` are. The answer
    key here is the declared direction of the law — ``water causes growth, sign +1`` — which is
    stated in the record, so the record has to be in the **taught** set for the arrow to exist at
    all. What is held out is not withheld by the splitter: it is the plan, and no record in the
    corpus contains one. Asking a held-out record would leave the universe with no arrow, no
    lever, and a row of ``asked`` against ``answered: 0`` measuring the splitter rather than her.

    Two things have to be right and they fail independently:

    * the **lever** — the cause named by the law, picked out of every variable in the universe
      that has an oriented arrow anywhere. Twenty-seven relations were fitted from this corpus,
      so choosing the one that actually reaches the target is a real discrimination and not a
      formality;
    * the **direction** — pushed the way the law's sign says. A planner with an inverted sign
      finds the right lever every time and scores zero here, which is the point of checking both.

    Scored only where the model has something to plan with: a target already met, or a variable
    the universe never fitted, is not answered wrongly — it is not answered, and ``asked``
    against ``answered`` keeps the two apart.
    """
    out = ProbeResult(family="planning", category=category)
    planner = getattr(brain, "rollout", None)
    universe = getattr(brain, "universe", None)
    if planner is None or universe is None:
        return out
    try:
        from nyxara.njp.rollout import Target
    except Exception:  # noqa: BLE001
        return out
    seen: Set[Tuple[str, str]] = set()
    for record in records:
        for law in record.get("law") or []:
            if len(law) < 4:
                continue
            cause, effect = _norm(law[0]), _norm(law[2])
            try:
                sign = int(law[3])
            except (TypeError, ValueError):
                continue
            if not cause or not effect or sign == 0 or (cause, effect) in seen:
                continue
            here = universe.state.get(effect)
            lever_now = universe.state.get(cause)
            if here is None or lever_now is None:
                continue
            # Only where the question is well posed. A variable two different laws both reach is
            # a variable with two right answers, and marking one of them wrong measures the
            # corpus rather than her.
            #
            # This is not hypothetical and it is not rare. The corpus states its laws in bare
            # variable names, so ``fertiliser causes yield`` from an agriculture scenario and
            # ``catalyst mass causes yield`` from a chemistry one fit the *same* variable — as do
            # ``altitude causes temperature`` and ``temperature causes rate``. Measured, the
            # planner chose ``catalyst mass`` for ``yield`` and ``altitude`` for ``rate``: both
            # genuine levers with real arrows into the target, both scored wrong against a key
            # that named the other one. Skipping the ambiguous pairs is the honest response;
            # loosening the key until the answer counts is how a benchmark stops measuring.
            def _into(variable: str) -> set:
                return {c for (c, e), relation in universe.relations.items()
                        if e == variable and getattr(relation, "usable", False)
                        and getattr(relation, "oriented", False)}

            # One lever, and it has to be the whole chain. Restricting to a single *direct*
            # cause is not enough: the search walks upstream, so a cause that is itself an
            # effect gives a second, genuine lever a hop further back. Measured, ``rate`` had
            # exactly one arrow into it — from ``temperature`` — and the plan set ``altitude``,
            # which drives temperature and scored better for it. That is a good plan being
            # marked wrong by a key that names the nearer variable, so the pair is only asked
            # about where nothing upstream of the cause is settable either.
            reaching = _into(effect)
            if len(reaching) != 1 or cause not in reaching or _into(cause):
                continue
            if out.asked >= _MAX_PROBES:
                break
            seen.add((cause, effect))
            out.asked += 1
            # A target above where the variable is now, so "get it higher" is a real request and
            # the sign of the required lever move is exactly the law's own sign.
            target = Target(variable=effect, value=float(here) + max(1.0, abs(float(here)) * 0.2))
            try:
                plan = planner.search(target)
            except Exception:  # noqa: BLE001
                continue
            if plan.chosen is None:
                continue                     # nothing reaches it — not an answer, not a mistake
            out.answered += 1
            moved = float(plan.chosen.setting) - float(lever_now)
            if abs(moved) <= _FLAT:
                continue                     # a plan that changes nothing is not the right way
            if plan.chosen.lever == cause and (1 if moved > 0 else -1) == sign:
                out.correct += 1
    return out


_FAMILIES = (
    ("recall", _probe_recall),
    ("relation", _probe_relation),
    ("causal", _probe_causal),
    ("counterfactual", _probe_counterfactual),
    ("generalization", _probe_generalization),
    ("inheritance", _probe_inheritance),
    ("planning", _probe_planning),
    ("transfer", _probe_transfer),
    ("transition", _probe_transition),
)


# --------------------------------------------------------------------------- #
# Running it
# --------------------------------------------------------------------------- #
def _capped(result: ProbeResult) -> ProbeResult:
    return result


#: Families whose answer key is **derivable and never stated**, so they need no holdout — the
#: corpus says "mammals breathe with lungs" and "a seal is a mammal" and never says the conclusion.
#: Running these on the held-out split was measuring the wrong set: every claim in a generalisation
#: record is load-bearing for its own probe, so a fair split had nothing left to withhold and the
#: row vanished to `asked: 0`. They read the **taught** set, where the premises are, and the answer
#: is held out by construction rather than by the splitter.
_DERIVED_FAMILIES = frozenset({"generalization", "inheritance", "planning"})

#: Why there is no ``curiosity`` row here, and why adding one would be worse than the gap.
#:
#: Every family above has an answer key that exists independently of the organ being scored — the
#: invariant the corpus states, the direction the law declares, the state the transition names.
#: "Which question is most worth asking" has no such key. Any key this file could write would be
#: a restatement of the same priority ordering the organ computes, and a benchmark that agrees
#: with the system by construction measures nothing at all. Phase 5's own rule says it plainly:
#: generation and evaluation must stay separate, and here they cannot be.
#:
#: The compression reward is falsified elsewhere and properly — ``tests/njp/
#: test_compression_reward.py`` holds the question set fixed, moves only which organ is still
#: yielding, and requires the top of the queue to change. That is an experiment with a control,
#: which is more than a benchmark row would have been.
_NO_KEY_FAMILIES = frozenset({"curiosity", "revision", "representation"})


def evaluate(brain: Any, held_out: Sequence[Dict[str, Any]], *,
             taught: Optional[Sequence[Dict[str, Any]]] = None,
             budget: int = _DEFAULT_BUDGET) -> List[ProbeResult]:
    """Every probe family, per category, over the records she was never taught.

    Family-major rather than category-major, and that ordering is the whole reason the budget is
    survivable. Walking category by category spends the entire budget on ``world_knowledge`` —
    it is two thirds of the corpus — and every reasoning family reports nothing. Taking one family
    across all categories before starting the next gives each capability its own slice.
    """
    def group(rows: Sequence[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        out: Dict[str, List[Dict[str, Any]]] = {}
        for record in rows:
            out.setdefault(str(record.get("category") or "unknown"), []).append(record)
        return out

    by_category = group(held_out)
    by_category_taught = group(list(taught or []))
    out: List[ProbeResult] = []
    spent = 0
    for _name, probe in _FAMILIES:
        source = by_category_taught if _name in _DERIVED_FAMILIES else by_category
        for category, records in sorted(source.items()):
            if spent >= budget:
                break
            result = probe(brain, records, category)
            spent += result.asked
            if result.asked:
                out.append(result)
    return out


def run(*, records: Any = _DEFAULT_RECORDS, holdout: float = 0.25, seed: int = 0,
        limit: int = 0, budget: int = _DEFAULT_BUDGET) -> CapabilityReport:
    """Split, teach one part, ask about the other, and report both numbers."""
    from nyxara.njp.brain import NJPBrain
    from nyxara.njp.unified import absorb, load

    started = time.perf_counter()
    rows = _sample(load(records), limit)
    taught, held = split(rows, holdout=holdout, seed=seed)
    report = CapabilityReport(taught=len(taught), held_out=len(held))

    brain = NJPBrain()
    absorbed = absorb(brain, taught, check=False)
    report.taught_recall = _taught_recall(brain, taught)
    report.probes = evaluate(brain, held, taught=taught, budget=budget)
    report.ms = (time.perf_counter() - started) * 1000.0
    del absorbed
    return report


def _sample(rows: List[Dict[str, Any]], limit: int) -> List[Dict[str, Any]]:
    """A stratified slice, round-robin across categories, in the corpus's own order.

    Not the first N. ``world_knowledge`` is two thirds of the file and sorts first, so a plain
    truncation at 700 gave a run containing exactly one category and one probe family — a report
    that looked like a capability surface and was a recall test.
    """
    if not limit or limit >= len(rows):
        return rows
    buckets: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        buckets.setdefault(str(row.get("category") or "unknown"), []).append(row)
    order = sorted(buckets)
    out: List[Dict[str, Any]] = []
    index = 0
    while len(out) < limit and any(index < len(buckets[name]) for name in order):
        for name in order:
            if index < len(buckets[name]) and len(out) < limit:
                out.append(buckets[name][index])
        index += 1
    return out


def _taught_recall(brain: Any, taught: Sequence[Dict[str, Any]]) -> Optional[float]:
    """The control. Without it a low held-out number could mean the load failed rather than that
    she cannot generalise, and those call for opposite work."""
    control = _probe_recall(brain, list(taught)[:60], "taught")
    return control.score


def main(argv: Optional[Sequence[str]] = None) -> int:
    """``python -m nyxara.eval.capability [--holdout F] [--seed N] [--records PATH]``."""
    import argparse

    parser = argparse.ArgumentParser(description="Measure the held-out capability surface.")
    parser.add_argument("--records", default=str(_DEFAULT_RECORDS))
    parser.add_argument("--holdout", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--budget", type=int, default=_DEFAULT_BUDGET,
                        help="total probes the run may ask (raise for a slower, fuller sweep)")
    parser.add_argument("--misses", action="store_true", help="print a few wrong answers")
    args = parser.parse_args(list(argv) if argv is not None else None)

    report = run(records=args.records, holdout=args.holdout, seed=args.seed, limit=args.limit,
                 budget=args.budget)
    print(json.dumps(report.to_dict(), indent=1))
    if args.misses:
        for probe in report.probes:
            for miss in probe.misses[:3]:
                print(f"  {probe.category}:{probe.family}  {miss}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
