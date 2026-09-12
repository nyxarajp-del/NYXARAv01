"""NYXARA · njp/compile.py — language into an executable cognitive operation (⚙🧠, NJP V.08).

**The failure this module exists for, reproduced exactly.** Asked *"agar paani aadha kar doon to
kya hoga"*, she answered nothing. Every organ worked: the speech-act reader called it a
``causal_query`` with confidence 0.85 and listed ``SIMULATE`` among the pathways it permits; the
intent reader parsed the conditional; the internal universe's do-operator answers this class of
question correctly when it is called. What was missing was the sentence reaching the operator.

**Why it was missing, precisely.** One turn was being read three separate times by three parsers
that share nothing:

======================  ==========================================  ==========================
what it produces        what it got right on that sentence          what it is allowed to do
======================  ==========================================  ==========================
:class:`~nyxara.njp.intent.Intent`     the whole conditional clause  decide ``about_self``
:class:`~nyxara.njp.relevance.Act`     ``causal_query``, SIMULATE ok gate the *LLM cortex*
a two-key ``dict``      **nothing**                                 the only route to the engine
======================  ==========================================  ==========================

The third one is four bare regexes joined by ``or``, and ``or`` short-circuits. The English
subject-verb-object pattern matched the Hinglish subject-object-verb sentence, took the *verb*
(``doon``) as the variable, and the pattern written for exactly this word order never ran. The
variable was unresolvable, the key was absent, and — the part that made it expensive — the
problem classifier reads that key, so its absence also flipped the problem from CAUSAL (0.95) to
EMPIRICAL (0.50), selecting three strategies that cannot answer an intervention. Each of them
returned nothing, and ``metareason._miss`` charged the loss to ``simulate``: the bug was teaching
the bandit that the do-operator is bad at causal questions.

**What this module changes.** One typed :class:`CognitiveOp`, compiled from all three readings at
once, carrying what each of them knew. Nothing is deleted: the existing parsers keep running and
keep their own tests. What changes is that their results now meet.

**The selection rule, and why it is not pattern order.** Every pattern is tried, never the first
that matches, and the winner is chosen on evidence:

1. **Does the variable resolve to something she has actually observed?** ``paani`` does and
   ``doon`` does not, and that is not a fact about Hindi — it is a fact about her own state. This
   is the strong signal and it needs no word list.
2. **How close is the operator to the variable?** ``paani aadha`` beats ``aadha kar doon`` by four
   characters. Word order differs between the two languages; adjacency does not.

A stop-list of Hindi verb endings would also have fixed the reported sentence. It would have been
the same kind of thing that broke it.

**The half the numbers cannot supply.** *"aadha kar doon"* states a direction and no magnitude,
and the universe's qualitative pass reads direction by comparing the new value against the factual
one — so with nothing measured it reports ``direction=0``, unknown. But the sentence said *down*.
An operation therefore carries :attr:`CognitiveOp.direction` derived from the operator word alone,
which is knowledge the language has and arithmetic does not, and it is what lets a stated law with
no quantities anywhere still answer *"what if I halve it"* with the direction of the change and an
honest refusal of its size.

Pure standard library. No LLM: a compiler whose output is a language model's guess about what was
meant is a guess about what was meant. Every entry point is fail-soft — an uncompilable turn
returns :data:`NONE_OP` and the turn completes exactly as it did before.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

__all__ = ["Verb", "Operation", "CognitiveOp", "ThoughtCompiler", "NONE_OP"]


class Verb:
    """What the turn asks her to *do*, as opposed to what it is about."""

    INTERVENE = "intervene"    # set a variable and report what follows — the do-operator
    EXPLAIN = "explain"        # why is this so
    PREDICT = "predict"        # what happens next
    RECALL = "recall"          # what do you hold about this
    ANSWER = "answer"          # a question with no special machinery behind it
    NONE = "none"              # nothing to execute — a greeting, a statement, an echo

    ALL: Tuple[str, ...] = (INTERVENE, EXPLAIN, PREDICT, RECALL, ANSWER, NONE)


class Operation:
    """How the target is to be changed. Only forms whose size the sentence states."""

    SET = "set"        # to a stated absolute value
    SCALE = "scale"    # by a stated factor — halve, double
    REMOVE = "remove"  # to zero — the total intervention, a different question from halving

    ALL: Tuple[str, ...] = (SET, SCALE, REMOVE)


#: What an operator word does, and by how much. Only interventions whose magnitude the sentence
#: actually states are here: "reduce the water" is deliberately absent, because a factor she chose
#: herself would come back through the do-operator carrying the same confidence as one the Master
#: gave, and nothing downstream could tell the two apart afterwards.
_FACTOR: Dict[str, float] = {
    "halve": 0.5, "halving": 0.5, "halved": 0.5, "half": 0.5,
    "aadha": 0.5, "adha": 0.5, "aadhi": 0.5,
    "double": 2.0, "doubling": 2.0, "doubled": 2.0, "dugna": 2.0, "duguna": 2.0,
    "remove": 0.0, "removing": 0.0, "removed": 0.0, "stop": 0.0, "stopping": 0.0,
    "stopped": 0.0, "band": 0.0, "bina": 0.0,
}

_OPERATORS = "|".join(sorted(_FACTOR, key=len, reverse=True))

#: "halve the water" — operator first, the English order.
_LEADING = re.compile(
    rf"\b(?P<op>{_OPERATORS})\b(?:\s+(?:the|le|kar|of|its))*\s+(?P<var>[a-z][a-z_]*)\b",
    re.IGNORECASE)

#: "paani aadha kar doon", "the water is halved" — the variable first. Hinglish puts the object
#: before the verb, and the English passive puts the subject before it, so one pattern serves both.
_TRAILING = re.compile(
    rf"\b(?:the\s+)?(?P<var>[a-z][a-z_]*)\s+(?:(?:is|was|were|be|gets?|got|ko|ka|ki)\s+)?"
    rf"(?P<op>{_OPERATORS})\b",
    re.IGNORECASE)

#: "what if the water were 3" — the same question stated as an absolute rather than a factor.
_ABSOLUTE = re.compile(
    r"\b(?:the\s+)?(?P<var>[a-z][a-z_]*)\s+(?:were|was|is|becomes?|ho)\s+"
    r"(?P<val>-?\d+(?:\.\d+)?)\b",
    re.IGNORECASE)

#: "paani 3 ho jaye" — the same absolute with the copula last, which is where Hindi puts it. The
#: pattern above was written in English word order only, so the commonest Hinglish phrasing of a
#: stated-value counterfactual matched nothing at all — the same shape of gap as the one that made
#: this module necessary, one layer down.
_ABSOLUTE_TRAILING = re.compile(
    r"\b(?P<var>[a-z][a-z_]*)\s+(?P<val>-?\d+(?:\.\d+)?)\s+(?:ho|hoke|kar|kardo|kar\s+doon)\b",
    re.IGNORECASE)


@dataclass
class CognitiveOp:
    """One turn, compiled: what to do, to what, by how much, and on whose authority.

    ``sources`` is not decoration. It records which reading contributed each field, so a wrong op
    can be traced to the parser that produced it rather than to the compiler that assembled it —
    the distinction that took a reproduced failure to find the first time.
    """

    verb: str = Verb.NONE
    target: str = ""                 # the universe's own variable name, or "" if unresolved
    raw_target: str = ""             # what the sentence actually said, before resolution
    operation: str = ""
    magnitude: float = 0.0
    #: Which way the change goes: -1 down, +1 up, 0 unknown. Read from the operator word, so it
    #: survives a sentence that states no quantities at all.
    direction: int = 0
    kind: str = ""                   # a metareason.ProblemKind, from the speech act
    pathways: Tuple[str, ...] = ()   # what CognitivePolicy permits this act
    sources: Tuple[str, ...] = ()
    confidence: float = 0.0
    why: str = ""

    @property
    def executable(self) -> bool:
        """Enough to run: a verb with machinery behind it, and a target it can be run on."""
        return self.verb == Verb.INTERVENE and bool(self.target)

    def permits(self, pathway: str) -> bool:
        """Whether this turn's speech act allows a given kind of cognition at all.

        No pathways recorded means no act was read, and an unread act constrains nothing — the
        turn is handled the way it was before this module existed.
        """
        return (not self.pathways) or (pathway in self.pathways)

    def to_context(self) -> Dict[str, Any]:
        """The shape ``_strategy_simulate`` and ``ProblemClassifier`` already read.

        Deliberately the same two keys as before plus additions, so nothing downstream has to
        change to keep working, and anything that wants more can ask the op itself.
        """
        out: Dict[str, Any] = {"op": self}
        if self.executable:
            out["variable"] = self.target
            out["value"] = self.magnitude
            out["operation"] = self.operation
            out["direction"] = self.direction
            # The turn named an operation, and one strategy executes operations. Naming it is not
            # a preference over the bandit — the outcome is scored the same either way — it is the
            # difference between a strategy being *eligible* and the turn actually being about it.
            out["prefer"] = "simulate"
        return out

    def to_dict(self) -> Dict[str, Any]:
        return {"verb": self.verb, "target": self.target, "raw_target": self.raw_target,
                "operation": self.operation, "magnitude": round(self.magnitude, 4),
                "direction": self.direction, "kind": self.kind,
                "pathways": list(self.pathways), "sources": list(self.sources),
                "confidence": round(self.confidence, 4), "why": self.why}


#: What an unparseable turn compiles to. A shared instance because it is immutable in practice and
#: a turn that compiled to nothing should not allocate to say so.
NONE_OP = CognitiveOp(why="nothing in this turn names an operation")


@dataclass
class _Candidate:
    """One pattern's reading of the sentence, before anything has chosen between them."""

    raw: str
    op_word: str
    pattern: str
    gap: int                     # characters between the operator and the variable
    resolved: str = ""

    def score(self) -> Tuple[int, int]:
        """Resolution first, adjacency second. Higher is better; ``gap`` is negated to sort."""
        return (2 if self.resolved else 0, -self.gap)


class ThoughtCompiler:
    """Language, an intent reading and a speech act, compiled into one typed operation.

    Holds no state and owns no parser. Every input is optional: with none of them it still returns
    a usable op, and with all of them it returns a better one.
    """

    def __init__(self) -> None:
        self.compiled = 0
        self.executable = 0
        self.unresolved = 0          # named an intervention on a variable she has never observed
        self.no_operation = 0        # nothing in the turn to execute

    # ---- the act's contribution -------------------------------------------- #
    @staticmethod
    def _kind_from_act(act: Any) -> str:
        """A speech act already decided what kind of question this is. Ask it, do not re-guess.

        ``ProblemClassifier`` scores a bag of words and lands on EMPIRICAL for a sentence the act
        reader called ``causal_query`` with a 0.65 margin. Two classifiers with no shared channel
        is one classifier too many, and the one that read the sentence as a *speech act* is the
        one that saw the "what if".
        """
        try:
            from nyxara.njp.metareason import ProblemKind
            from nyxara.njp.relevance import SpeechAct
        except Exception:  # noqa: BLE001
            return ""
        mapping = {
            SpeechAct.CAUSAL_QUERY: ProblemKind.CAUSAL,
            SpeechAct.KNOWLEDGE_QUERY: ProblemKind.FACTUAL,
            SpeechAct.STATE_QUERY: ProblemKind.INTROSPECTIVE,
            SpeechAct.CAPABILITY_QUERY: ProblemKind.INTROSPECTIVE,
            SpeechAct.CORRECTION: ProblemKind.CONTRADICTION,
        }
        return mapping.get(str(getattr(act, "kind", "")), "")

    @staticmethod
    def _pathways(act: Any) -> Tuple[str, ...]:
        if act is None:
            return ()
        try:
            from nyxara.njp.relevance import CognitivePolicy
            return tuple(CognitivePolicy().allowed(act))
        except Exception:  # noqa: BLE001
            return ()

    @staticmethod
    def _verb_from_act(act: Any) -> str:
        try:
            from nyxara.njp.relevance import SpeechAct
        except Exception:  # noqa: BLE001
            return Verb.NONE
        return {
            SpeechAct.CAUSAL_QUERY: Verb.EXPLAIN,
            SpeechAct.KNOWLEDGE_QUERY: Verb.ANSWER,
            SpeechAct.STATE_QUERY: Verb.RECALL,
            SpeechAct.CAPABILITY_QUERY: Verb.RECALL,
        }.get(str(getattr(act, "kind", "")), Verb.NONE)

    # ---- the intervention --------------------------------------------------- #
    @staticmethod
    def _candidates(text: str) -> List[_Candidate]:
        """Every pattern's reading, never the first one that matched."""
        found: List[_Candidate] = []
        for pattern, name in ((_LEADING, "leading"), (_TRAILING, "trailing")):
            for match in pattern.finditer(text):
                var, op = match.group("var"), match.group("op")
                # The characters between the two groups: how far the sentence had to reach to
                # associate this operator with this variable.
                if name == "leading":
                    gap = match.start("var") - match.end("op")
                else:
                    gap = match.start("op") - match.end("var")
                found.append(_Candidate(raw=var, op_word=op, pattern=name, gap=max(0, gap)))
        return found

    def _intervention(self, text: str, resolve: Callable[[str], str],
                      state: Dict[str, Any]) -> Optional[CognitiveOp]:
        """The intervention this sentence asks for, or ``None`` if it asks for none."""
        # Both word orders, and the one whose variable she has actually observed wins — the same
        # rule the factor patterns are chosen by, for the same reason.
        absolutes = [(m, name) for pattern, name in
                     ((_ABSOLUTE, "absolute"), (_ABSOLUTE_TRAILING, "absolute-trailing"))
                     for m in pattern.finditer(text)]
        scored = [(bool(resolve(m.group("var"))), m, name) for m, name in absolutes]
        scored.sort(key=lambda row: row[0], reverse=True)
        if scored and scored[0][0]:
            _, absolute, pattern_name = scored[0]
            resolved = resolve(absolute.group("var"))
            value = float(absolute.group("val"))
            before = state.get(resolved)
            direction = 0
            if before is not None:
                direction = 1 if value > float(before) else (-1 if value < float(before) else 0)
            return CognitiveOp(
                verb=Verb.INTERVENE, target=resolved, raw_target=absolute.group("var"),
                operation=Operation.SET, magnitude=value, direction=direction,
                sources=(f"pattern:{pattern_name}",), confidence=0.9,
                why="the sentence states the value outright")

        candidates = self._candidates(text)
        if not candidates:
            return None
        for candidate in candidates:
            candidate.resolved = resolve(candidate.raw)
        best = max(candidates, key=lambda c: c.score())

        factor = _FACTOR.get(best.op_word.lower())
        if factor is None:                      # unreachable via the generated pattern; guard anyway
            return None
        operation = Operation.REMOVE if factor == 0.0 else Operation.SCALE
        # The direction the *sentence* states, which arithmetic cannot supply when nothing has
        # been measured. A factor below one is a decrease whatever the baseline turns out to be.
        direction = 0 if factor == 1.0 else (1 if factor > 1.0 else -1)

        if not best.resolved:
            self.unresolved += 1
            return CognitiveOp(
                verb=Verb.INTERVENE, target="", raw_target=best.raw,
                operation=operation, magnitude=factor, direction=direction,
                sources=(f"pattern:{best.pattern}",), confidence=0.2,
                why=f"names an intervention on {best.raw!r}, which she has never observed")

        observed = state.get(best.resolved)
        if observed is None:
            # Known to her as a variable, but never given a number. A magnitude is impossible and
            # a direction is not, so the op is executable qualitatively and says so.
            return CognitiveOp(
                verb=Verb.INTERVENE, target=best.resolved, raw_target=best.raw,
                operation=operation, magnitude=0.0, direction=direction,
                sources=(f"pattern:{best.pattern}",), confidence=0.55,
                why="direction is stated by the sentence; no quantity has ever been observed")

        return CognitiveOp(
            verb=Verb.INTERVENE, target=best.resolved, raw_target=best.raw,
            operation=operation, magnitude=float(observed) * factor, direction=direction,
            sources=(f"pattern:{best.pattern}",), confidence=0.85,
            why=f"{best.op_word} applied to what {best.resolved} has actually been")

    # ---- the whole turn ------------------------------------------------------ #
    def compile(self, text: str, *, intent: Any = None, act: Any = None,
                resolve: Optional[Callable[[str], str]] = None,
                state: Optional[Dict[str, Any]] = None) -> CognitiveOp:
        """Compile one turn. Never raises; an uncompilable turn is :data:`NONE_OP`."""
        try:
            self.compiled += 1
            said = str(text or "")
            resolver = resolve or (lambda name: "")
            observed = dict(state or {})

            op = self._intervention(said, resolver, observed)
            if op is None:
                op = CognitiveOp(verb=self._verb_from_act(act),
                                 confidence=float(getattr(act, "confidence", 0.0) or 0.0),
                                 sources=("act",) if act is not None else (),
                                 why="no intervention named in this turn")
                if op.verb == Verb.NONE:
                    self.no_operation += 1

            op.kind = self._kind_from_act(act)
            op.pathways = self._pathways(act)
            if act is not None and "act" not in op.sources:
                op.sources = op.sources + ("act",)
            if intent is not None:
                op.sources = op.sources + ("intent",)
                # An intent reading that found no reading at all is a turn she did not understand,
                # and an op compiled from it should not claim more confidence than that.
                if getattr(intent, "ambiguous", False):
                    op.confidence = min(op.confidence, 0.6)

            # A turn whose speech act forbids simulation cannot carry an executable intervention,
            # however cleanly the sentence parsed. This is the table that stops a greeting from
            # reaching physics, applied to the operation rather than only to the reply.
            if op.verb == Verb.INTERVENE and not op.permits("simulate"):
                op.verb = Verb.NONE
                op.target = ""
                op.why = "this speech act does not permit simulation"

            if op.executable:
                self.executable += 1
            return op
        except Exception:  # noqa: BLE001 — an uncompilable turn is not a broken turn
            return NONE_OP

    def stats(self) -> Dict[str, Any]:
        """What the compiler managed, including what it did not.

        ``executable_rate`` is over turns that named an operation at all, not over every turn —
        a greeting is not a compiler failure, and averaging it in would flatter the number.
        """
        named = self.compiled - self.no_operation
        return {
            "compiled": self.compiled,
            "named_an_operation": named,
            "executable": self.executable,
            "unresolved_target": self.unresolved,
            "executable_rate": (round(self.executable / named, 4) if named else None),
        }
