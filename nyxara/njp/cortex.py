"""NYXARA · njp/cortex.py — the cortex proposes, NJP disposes (🧠⚡, NJP V.06).

**The finding this module exists for.** Grep ``nyxara/njp/`` for a language model and you find
exactly one consumer: :mod:`nyxara.njp.voice`, whose documented division of labour is *"content
comes from NYX's cycle; the LLM only renders that content into natural language."* Every other
organ in the package says *"Pure standard library. No LLM."* in its own header, and means it.

That is a good rule for a rendering surface and it had an unmeasured consequence: **the strength of
the model on the ladder could not affect her reasoning at all.** Swapping a 2 GB instruct model for
a 9B post-trained reasoner buys better sentences and, by construction, nothing else — because
nothing ever asks it for a hypothesis, a cause, or a prediction. The Master loaded weights, watched
for capability, and found none. The weights were never the missing part.

This module is the missing part, and it is deliberately small:

    cortex  →  competing hypotheses / relations / causal laws / predictions
            →  the gates NJP already had
            →  survived · refuted · undecided

**Competing hypotheses, never one answer.** :meth:`Cortex.hypotheses` asks for three to five rival
explanations — *A caused X*, *B caused X*, *C caused both*, *coincidence*, *measurement error* —
and hands them to :class:`~nyxara.njp.reason.ProblemState`, which has carried live *and rejected*
hypotheses from the start. NJP's job then becomes **model selection**, which is a thing evidence can
do. Asking for one answer and checking it is not: a single proposal has nothing to be selected
against, so the only available verdict is "it sounds right", which is not a verdict.

**Nothing here is stored as a fact.** Every proposal carries
:attr:`~nyxara.njp.grounding.Provenance.PROPOSED` and a confidence capped below the KNOWN floor, so
a cortex claim can be *considered* and can never be *stated*. The cap is not politeness: without it
two proposals from two prompts would corroborate each other — ``Grounder`` counts distinct sources —
and a guess would become a fact by being asked for twice.

**Causal laws are attacked before they are believed.** A proposed law does **not** go to
:meth:`~nyxara.njp.world.WorldView.state_law`, whose docstring is explicit that it records what
*"the Master (or a source) stated outright"*. It goes to
:class:`~nyxara.njp.adversary.SelfAttacker` first. Refuted → destroyed, with the reason written into
:attr:`~nyxara.njp.beliefs.Belief.history`. Survived, on evidence *independent* of the claim →
promoted to ``DERIVED`` and stated. Undecided → it stays a proposal and becomes work for
:mod:`nyxara.njp.epistemic`, because undecided is not a pass.

**The cortex answers as itself or is absent.** Generation goes through
``LLM.complete_with("qwythos", …)``, which raises rather than substituting another rung. The
alternative would be the n-gram floor producing "hypotheses" — babble dressed as reasoning, which is
the exact failure :mod:`nyxara.njp.voice` was written to refuse for the language surface.

**Its output is untrusted text.** The cortex reads documents and web pages, so what comes back is
model text, not Master text. Every response is screened by
:class:`~nyxara.senses.web.InjectionScanner` — the same gate :mod:`nyxara.growth.acquire` uses — and
a flagged response is dropped and counted, never parsed.

**Extraction is structured, and failure is counted rather than guessed at.** The cortex is asked for
strict JSON; a row that does not parse is dropped. :meth:`Cortex.stats` reports
``extraction_rate``, and that number is a statement about the extractor, never a compliment to the
model.

Every public entry point is fail-soft: with no cortex reachable, each returns an empty result and
the turn completes exactly as it does today.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from nyxara.njp.grounding import GroundedTriple, Provenance

log = logging.getLogger("nyxara.njp.cortex")

__all__ = [
    "HypothesisKind",
    "Proposal",
    "CortexTriple",
    "CortexLaw",
    "CortexPrediction",
    "CortexHypothesis",
    "CortexReport",
    "Cortex",
]

#: The hardest a proposal may ever be held on the cortex's word alone. Below every reasonable
#: ``Grounder.known_floor`` on purpose: a proposal must be able to be *considered* and must never be
#: able to be *stated*. It is a ceiling, not a starting value — a cortex that reports 0.99 gets this.
_PROPOSAL_CEILING = 0.45

#: How many rival explanations to ask for. Fewer than three is not a competition; more than five
#: is a list rather than a set of candidates, and the tail is padding the model invented to fill the
#: quota — which is the failure mode this whole module is trying to avoid.
_MIN_HYPOTHESES = 3
_MAX_HYPOTHESES = 5

#: Cortex text is untrusted; at or above this injection score it is dropped unparsed.
_INJECTION_CEILING = 0.5

#: Ceilings on what one response may contain. A model that returns 400 "relations" has stopped
#: extracting and started generating, and letting that into the store is how a fact base rots.
_MAX_TRIPLES = 32
_MAX_LAWS = 16
_MAX_PREDICTIONS = 16


class HypothesisKind:
    """What *sort* of explanation a candidate is. The last two are why the list is not just causes.

    A hypothesis set that contains only causal candidates cannot represent the two answers most
    often correct in practice — that nothing caused it, and that nothing happened. Naming them makes
    them selectable rather than unthinkable.
    """

    CAUSE = "cause"                     # A produced X
    CONFOUND = "confound"               # C produced both A and X
    REVERSE = "reverse"                 # X produced A, not the other way round
    COINCIDENCE = "coincidence"         # they co-occurred; nothing produced anything
    MEASUREMENT = "measurement_error"   # the observation itself is what is wrong
    OTHER = "other"

    ALL = (CAUSE, CONFOUND, REVERSE, COINCIDENCE, MEASUREMENT, OTHER)

    @staticmethod
    def of(raw: Any) -> str:
        kind = str(raw or "").strip().lower().replace(" ", "_").replace("-", "_")
        return kind if kind in HypothesisKind.ALL else HypothesisKind.OTHER


# --------------------------------------------------------------------------- #
# Proposals
# --------------------------------------------------------------------------- #
@dataclass
class Proposal:
    """What every cortex output has in common: a claim, and the receipt for where it came from.

    The provenance fields are not bookkeeping. ``BeliefLedger.why`` reconstructs "why do you believe
    this" from a record, and a record that cannot say *a language model proposed this, from this
    prompt, out of this span of its reply* will confabulate a better-sounding answer than the truth.
    """

    #: The prompt that produced it — the question, not the whole system text.
    prompt: str = ""
    #: Which model said it, as the provider reported itself.
    model: str = ""
    #: The raw fragment of the reply this was read out of, so the claim can be checked against it.
    span: str = ""
    #: What the cortex claimed, already capped at :data:`_PROPOSAL_CEILING`.
    confidence: float = 0.0
    at: float = field(default_factory=time.time)
    provenance: str = Provenance.PROPOSED

    def receipt(self) -> Dict[str, Any]:
        return {"model": self.model, "prompt": self.prompt[:200], "span": self.span[:300],
                "provenance": self.provenance, "at": self.at}


@dataclass
class CortexTriple(Proposal):
    """A proposed relation. Enters through the same ``_assert`` path corpus ingest uses."""

    subject: str = ""
    predicate: str = ""
    object: str = ""
    condition: str = ""
    modality: str = ""

    def as_grounded(self, *, source: str) -> GroundedTriple:
        """The store's own type, with the provenance and the cap already applied."""
        return GroundedTriple(
            subject=self.subject, predicate=self.predicate, object=self.object,
            confidence=min(self.confidence, _PROPOSAL_CEILING), source=source,
            text=self.span or f"{self.subject} {self.predicate} {self.object}",
            condition=self.condition, modality=self.modality,
            provenance=Provenance.PROPOSED)

    def to_dict(self) -> Dict[str, Any]:
        return {"subject": self.subject, "predicate": self.predicate, "object": self.object,
                "condition": self.condition, "modality": self.modality,
                "confidence": round(self.confidence, 4), **self.receipt()}


@dataclass
class CortexLaw(Proposal):
    """A proposed causal law. Attacked before it is believed, never stated on the model's word."""

    cause: str = ""
    effect: str = ""
    kind: str = "causes"                # causes | requires
    condition: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {"cause": self.cause, "effect": self.effect, "kind": self.kind,
                "condition": self.condition, "confidence": round(self.confidence, 4),
                **self.receipt()}


@dataclass
class CortexPrediction(Proposal):
    """A proposed expectation, with what would settle it.

    :attr:`under` is the load-bearing field. A prediction with no stated condition cannot be tested
    against anything, and one that cannot be tested is not a prediction — it is a mood.
    """

    key: str = ""
    expected: str = ""
    under: str = ""                     # the circumstance it is asserted under
    horizon: str = ""                   # when it should be checked, in the source's own words

    @property
    def testable(self) -> bool:
        return bool(self.key and self.expected)

    def to_dict(self) -> Dict[str, Any]:
        return {"key": self.key, "expected": self.expected, "under": self.under,
                "horizon": self.horizon, "testable": self.testable,
                "confidence": round(self.confidence, 4), **self.receipt()}


@dataclass
class CortexHypothesis(Proposal):
    """One candidate explanation among rivals.

    :attr:`predicts` is what makes the set selectable. Two hypotheses that predict the same things
    are not two hypotheses; the observation that separates them is the only thing worth going to
    get, and :mod:`nyxara.njp.epistemic` reads exactly this field to find it.
    """

    claim: str = ""
    kind: str = HypothesisKind.OTHER
    #: Observations this hypothesis says we WOULD see — the discriminating half.
    predicts: List[str] = field(default_factory=list)
    #: Observations it says we would NOT see. A hypothesis that forbids nothing risks nothing.
    forbids: List[str] = field(default_factory=list)
    rank: int = 0

    @property
    def discriminating(self) -> bool:
        return bool(self.predicts or self.forbids)

    def to_dict(self) -> Dict[str, Any]:
        return {"claim": self.claim, "kind": self.kind, "predicts": list(self.predicts),
                "forbids": list(self.forbids), "rank": self.rank,
                "discriminating": self.discriminating,
                "confidence": round(self.confidence, 4), **self.receipt()}


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #
@dataclass
class CortexReport:
    """What one pass actually did. Every field is a count, and the drops are counted too.

    A bridge that reported only its successes would make a model that returns garbage 90% of the
    time look identical to one that does not.
    """

    asked: int = 0
    answered: int = 0
    unavailable: int = 0
    screened: int = 0                   # dropped by the injection scanner, unparsed
    unparsable: int = 0                 # the reply was not the JSON it was asked for
    rows_seen: int = 0
    rows_dropped: int = 0
    hypotheses: List[CortexHypothesis] = field(default_factory=list)
    triples: List[CortexTriple] = field(default_factory=list)
    laws: List[CortexLaw] = field(default_factory=list)
    predictions: List[CortexPrediction] = field(default_factory=list)
    # what the gates then did with them
    asserted: int = 0
    attacked: int = 0
    refuted: int = 0
    promoted: int = 0                   # survived independent evidence -> DERIVED and stated
    undecided: int = 0
    reasons: List[str] = field(default_factory=list)

    @property
    def thin(self) -> bool:
        """Fewer rival explanations than a competition needs. Reported, never padded."""
        return 0 < len(self.hypotheses) < _MIN_HYPOTHESES

    @property
    def extraction_rate(self) -> float:
        """Rows kept over rows seen — a statement about the extractor, not about the model."""
        if not self.rows_seen:
            return 0.0
        return max(0.0, min(1.0, (self.rows_seen - self.rows_dropped) / self.rows_seen))

    def to_dict(self) -> Dict[str, Any]:
        return {"asked": self.asked, "answered": self.answered, "unavailable": self.unavailable,
                "screened": self.screened, "unparsable": self.unparsable,
                "rows_seen": self.rows_seen, "rows_dropped": self.rows_dropped,
                "extraction_rate": round(self.extraction_rate, 4),
                "hypotheses": [h.to_dict() for h in self.hypotheses],
                "triples": [t.to_dict() for t in self.triples],
                "laws": [law.to_dict() for law in self.laws],
                "predictions": [p.to_dict() for p in self.predictions],
                "thin": self.thin, "asserted": self.asserted, "attacked": self.attacked,
                "refuted": self.refuted, "promoted": self.promoted,
                "undecided": self.undecided, "reasons": list(self.reasons)}


# --------------------------------------------------------------------------- #
# Prompts — asked for structure, because prose cannot be dropped honestly
# --------------------------------------------------------------------------- #
_SYSTEM = (
    "You produce structured hypotheses for a reasoning system that will test them against "
    "evidence. You are not answering the user and you are not writing prose. Return ONLY the JSON "
    "object described, with no commentary. State what you do not know as a lower confidence rather "
    "than by omitting it."
)

_HYPOTHESES_PROMPT = """\
Question: {question}
{context}
Give {n} DIFFERENT candidate explanations that compete with each other. Include at least one \
non-causal candidate (coincidence, or an error in how this was observed) where it is possible at all.

For each, list what we WOULD observe if it were true, and what we would NOT observe. Those two \
lists are what will be used to tell the candidates apart, so make them differ between candidates.

JSON: {{"hypotheses": [{{"claim": str, "kind": one of \
["cause","confound","reverse","coincidence","measurement_error","other"], \
"predicts": [str], "forbids": [str], "confidence": 0.0-1.0}}]}}"""

_WORLD_MODEL_PROMPT = """\
Text: {text}
{context}
Extract only what the text supports. Do not add world knowledge it does not state.

- relations: subject/predicate/object; set "condition" when it holds only in some circumstance, \
and "modality" to one of "", "possible", "typical", "necessary".
- laws: cause/effect claims about mechanism, with "kind" of "causes" or "requires".
- predictions: something that should be observable later, with the condition it holds under. \
Omit any prediction you cannot state a condition for.

JSON: {{"relations": [{{"subject": str, "predicate": str, "object": str, "condition": str, \
"modality": str, "confidence": 0.0-1.0}}], \
"laws": [{{"cause": str, "effect": str, "kind": str, "condition": str, "confidence": 0.0-1.0}}], \
"predictions": [{{"key": str, "expected": str, "under": str, "horizon": str, \
"confidence": 0.0-1.0}}]}}"""


def _clamp(value: Any, *, default: float = 0.3) -> float:
    """A confidence the cortex stated, brought inside the cap. Junk becomes the default."""
    try:
        out = float(value)
    except (TypeError, ValueError):
        out = default
    if out != out or out < 0.0:          # NaN or negative
        out = default
    return min(out, _PROPOSAL_CEILING)


def _text(value: Any, *, limit: int = 400) -> str:
    return str(value or "").strip()[:limit]


def _str_list(value: Any, *, limit: int = 8) -> List[str]:
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, (list, tuple)):
        return []
    return [_text(v) for v in value if _text(v)][:limit]


# --------------------------------------------------------------------------- #
# The cortex
# --------------------------------------------------------------------------- #
class Cortex:
    """Asks her cortex for proposals, and hands each to the gate that judges its kind.

    Construct it with nothing and it is honestly unavailable; construct it with a fake ``llm`` and
    it is fully testable without 5.6 GB of weights, which is how every test in
    ``tests/njp/test_cortex.py`` runs.
    """

    def __init__(self, *, llm: Any = None, provider: str = "qwythos",
                 max_tokens: int = 768, scanner: Any = None) -> None:
        self._llm = llm
        self._tried = llm is not None
        self.provider = str(provider)
        self.max_tokens = int(max_tokens)
        self._scanner = scanner
        self._scanner_tried = scanner is not None
        #: Which model actually answered the last call, as the provider reported itself. Stamped
        #: onto every proposal so a receipt names the model rather than the configured intent —
        #: those are different claims, and only one of them is a fact about what happened.
        self._last_model = ""
        # lifetime counters — the report is per pass, these are per process
        self.asked = 0
        self.answered = 0
        self.screened = 0
        self.unparsable = 0
        self.rows_seen = 0
        self.rows_dropped = 0

    # ---- reachability ------------------------------------------------------ #
    def _get_llm(self) -> Any:
        if not self._tried:
            self._tried = True
            try:
                from nyxara.mind.llm import LLM
                self._llm = LLM()
            except Exception:  # noqa: BLE001 — no facade is simply no cortex
                self._llm = None
        return self._llm

    def _get_scanner(self) -> Any:
        if not self._scanner_tried:
            self._scanner_tried = True
            try:
                from nyxara.senses.web import InjectionScanner
                self._scanner = InjectionScanner()
            except Exception:  # noqa: BLE001
                self._scanner = None
        return self._scanner

    def why_unavailable(self) -> str:
        """Empty when she has a cortex; otherwise the reason, in words an operator can act on."""
        llm = self._get_llm()
        if llm is None:
            return "no LLM facade could be built"
        try:
            status = llm.provider_status() or {}
        except Exception as exc:  # noqa: BLE001
            return f"provider status unreadable: {exc}"
        if self.provider not in status:
            return f"provider '{self.provider}' is not registered"
        if not status[self.provider]:
            provider = getattr(llm, "_providers", {}).get(self.provider)
            view = None
            if provider is not None and hasattr(provider, "learning_view"):
                try:
                    view = provider.learning_view().get("reason_unavailable")
                except Exception:  # noqa: BLE001
                    view = None
            return view or f"provider '{self.provider}' is unavailable"
        return ""

    def available(self) -> bool:
        """Is her *cortex* reachable — not merely some language model.

        The distinction is the whole safety property of this module. ``LLM.complete`` would happily
        fall down to the n-gram floor and return something; hypotheses generated that way would be
        noise wearing the shape of reasoning, and every gate downstream would then spend real
        evidence testing it.
        """
        return not self.why_unavailable()

    # ---- one structured call ------------------------------------------------ #
    def _ask_json(self, prompt: str, report: CortexReport) -> Optional[Any]:
        """One cortex call: screened, parsed, counted. ``None`` on any failure, never an exception."""
        report.asked += 1
        self.asked += 1
        reason = self.why_unavailable()
        if reason:
            report.unavailable += 1
            report.reasons.append(reason)
            return None
        llm = self._get_llm()
        try:
            from nyxara.mind.llm import LLMRequest
            req = LLMRequest.from_prompt(prompt, system=_SYSTEM, json_mode=True,
                                         max_tokens=self.max_tokens)
            # complete_with, never complete: the cortex answers as itself or is recorded as absent.
            resp = llm.complete_with(self.provider, req)
        except Exception as exc:  # noqa: BLE001 — an unreachable cortex is a quiet turn, not a crash
            report.unavailable += 1
            report.reasons.append(f"cortex call failed: {exc}")
            return None

        text = (resp.text or "").strip()
        if not text:
            report.unparsable += 1
            self.unparsable += 1
            report.reasons.append("cortex returned nothing")
            return None

        scanner = self._get_scanner()
        if scanner is not None:
            try:
                scan = scanner.scan(text)
                if getattr(scan, "suspicious", False) or float(getattr(scan, "score", 0.0)) >= _INJECTION_CEILING:
                    report.screened += 1
                    self.screened += 1
                    flags = [getattr(f, "name", "?") for f in getattr(scan, "flags", ())]
                    report.reasons.append(f"cortex output screened out ({', '.join(flags) or 'flagged'})")
                    log.warning("cortex output dropped by the injection scanner: %s", flags)
                    return None
            except Exception:  # noqa: BLE001 — a broken scanner must not become a bypass
                report.screened += 1
                report.reasons.append("injection scan failed; output dropped rather than trusted")
                return None

        try:
            data = resp.parse_json()
        except Exception:  # noqa: BLE001
            report.unparsable += 1
            self.unparsable += 1
            report.reasons.append("cortex reply was not the JSON it was asked for")
            return None
        report.answered += 1
        self.answered += 1
        self._last_model = resp.model
        return data

    def _rows(self, data: Any, key: str, report: CortexReport, cap: int) -> List[Dict[str, Any]]:
        """The list under ``key``, with every non-dict row dropped and counted."""
        raw = (data or {}).get(key) if isinstance(data, dict) else None
        if not isinstance(raw, (list, tuple)):
            if raw is not None:
                report.rows_dropped += 1
                self.rows_dropped += 1
            return []
        out: List[Dict[str, Any]] = []
        for row in raw:
            report.rows_seen += 1
            self.rows_seen += 1
            if not isinstance(row, dict):
                report.rows_dropped += 1
                self.rows_dropped += 1
                continue
            out.append(row)
            if len(out) >= cap:
                break
        return out

    # ---- the two generators ------------------------------------------------- #
    def hypotheses(self, question: str, *, context: str = "",
                   n: int = _MAX_HYPOTHESES,
                   report: Optional[CortexReport] = None) -> List[CortexHypothesis]:
        """Three to five *competing* explanations for ``question``.

        Never padded. A cortex that returns one candidate produces a report whose
        :attr:`CortexReport.thin` is True, because inventing rivals to reach a quota would defeat
        the only thing a hypothesis set is for.
        """
        report = report if report is not None else CortexReport()
        question = _text(question, limit=1000)
        if not question:
            return []
        n = max(_MIN_HYPOTHESES, min(int(n), _MAX_HYPOTHESES))
        prompt = _HYPOTHESES_PROMPT.format(
            question=question, n=n,
            context=f"What is already known: {_text(context, limit=2000)}\n" if context else "")
        data = self._ask_json(prompt, report)
        if data is None:
            return []

        model = getattr(self, "_last_model", "")
        out: List[CortexHypothesis] = []
        seen: set = set()
        for i, row in enumerate(self._rows(data, "hypotheses", report, _MAX_HYPOTHESES)):
            claim = _text(row.get("claim"))
            if not claim:
                report.rows_dropped += 1
                self.rows_dropped += 1
                continue
            fingerprint = claim.lower()
            if fingerprint in seen:
                # Two identically-worded candidates are one candidate written twice, and counting
                # them as two would make a set look like a competition when it is not.
                report.rows_dropped += 1
                self.rows_dropped += 1
                continue
            seen.add(fingerprint)
            out.append(CortexHypothesis(
                claim=claim, kind=HypothesisKind.of(row.get("kind")),
                predicts=_str_list(row.get("predicts")), forbids=_str_list(row.get("forbids")),
                confidence=_clamp(row.get("confidence")), rank=i,
                prompt=question, model=model, span=json.dumps(row)[:300]))
        report.hypotheses.extend(out)
        if report.thin:
            report.reasons.append(
                f"only {len(report.hypotheses)} candidate(s) came back — that is not a competition, "
                "and nothing was invented to make it look like one")
        return out

    def world_model(self, text: str, *, context: str = "",
                    report: Optional[CortexReport] = None) -> CortexReport:
        """The World Model Generator: text → relations, causal laws, testable predictions.

        Returns the report rather than the lists, because the drops are as much of the answer as
        the keeps: a pass that read forty rows and kept two is a different event from one that read
        two and kept two, and only the report can tell them apart.
        """
        report = report if report is not None else CortexReport()
        text = _text(text, limit=4000)
        if not text:
            return report
        prompt = _WORLD_MODEL_PROMPT.format(
            text=text,
            context=f"Already known: {_text(context, limit=2000)}\n" if context else "")
        data = self._ask_json(prompt, report)
        if data is None:
            return report
        model = getattr(self, "_last_model", "")

        for row in self._rows(data, "relations", report, _MAX_TRIPLES):
            subject, predicate, obj = (_text(row.get("subject"), limit=120),
                                       _text(row.get("predicate"), limit=80),
                                       _text(row.get("object"), limit=120))
            if not (subject and predicate and obj):
                report.rows_dropped += 1
                self.rows_dropped += 1
                continue
            report.triples.append(CortexTriple(
                subject=subject, predicate=predicate, object=obj,
                condition=_text(row.get("condition"), limit=200),
                modality=_text(row.get("modality"), limit=20),
                confidence=_clamp(row.get("confidence")),
                prompt=text[:200], model=model, span=json.dumps(row)[:300]))

        for row in self._rows(data, "laws", report, _MAX_LAWS):
            cause, effect = _text(row.get("cause"), limit=120), _text(row.get("effect"), limit=120)
            if not (cause and effect) or cause.lower() == effect.lower():
                report.rows_dropped += 1
                self.rows_dropped += 1
                continue
            kind = _text(row.get("kind"), limit=20).lower()
            report.laws.append(CortexLaw(
                cause=cause, effect=effect,
                kind=kind if kind in ("causes", "requires") else "causes",
                condition=_text(row.get("condition"), limit=200),
                confidence=_clamp(row.get("confidence")),
                prompt=text[:200], model=model, span=json.dumps(row)[:300]))

        for row in self._rows(data, "predictions", report, _MAX_PREDICTIONS):
            prediction = CortexPrediction(
                key=_text(row.get("key"), limit=120),
                expected=_text(row.get("expected"), limit=200),
                under=_text(row.get("under"), limit=200),
                horizon=_text(row.get("horizon"), limit=80),
                confidence=_clamp(row.get("confidence")),
                prompt=text[:200], model=model, span=json.dumps(row)[:300])
            if not prediction.testable:
                # An untestable prediction is not a weaker prediction; it is not one at all, and
                # carrying it would inflate every count that is supposed to mean "checkable".
                report.rows_dropped += 1
                self.rows_dropped += 1
                continue
            report.predictions.append(prediction)
        return report

    # ---- handing proposals to the gates ------------------------------------- #
    def offer(self, report: CortexReport, *, grounder: Any = None, world: Any = None,
              attacker: Any = None, beliefs: Any = None,
              source: str = "cortex") -> CortexReport:
        """Put the proposals through the gates NJP already had. Mutates and returns ``report``.

        Relations are asserted as proposals — capped, ``PROPOSED``, and therefore unable to reach
        ``KNOWN`` however often they recur. Laws are **attacked first**: nothing the cortex says
        becomes a stated law on its own word, because
        :meth:`~nyxara.njp.world.WorldView.state_law` records what a *source* stated outright and a
        hypothesis generator is not one.
        """
        if grounder is not None:
            for triple in report.triples:
                try:
                    folded = triple.predicate
                    if hasattr(grounder, "_predicate"):
                        # The same fold corpus ingest applies. An unfolded predicate is not a
                        # slightly worse fact — it is one the reasoner structurally cannot see.
                        folded = grounder._predicate(triple.predicate)
                    stored = triple.as_grounded(source=source)
                    stored.predicate = folded
                    grounder._assert(stored)
                    report.asserted += 1
                except Exception as exc:  # noqa: BLE001 — one bad row never costs the rest
                    # Deliberately NOT counted against ``rows_dropped``: that number is the
                    # extraction rate's denominator and describes what the EXTRACTOR made of the
                    # model's reply. A store that refused a well-extracted row is a different
                    # failure, it is recorded here, and folding it in would let the rate drift
                    # below zero once more rows failed to store than were ever read.
                    report.reasons.append(f"could not assert a proposed relation: {exc}")

        for law in report.laws:
            if attacker is None:
                report.undecided += 1
                report.reasons.append(
                    f"'{law.cause} → {law.effect}' was not attacked (no adversary attached) "
                    "and therefore stays a proposal")
                continue
            try:
                attack = attacker.attack(law.cause, law.effect, confidence=law.confidence,
                                         claim=f"{law.cause} causes {law.effect}")
                report.attacked += 1
            except Exception as exc:  # noqa: BLE001
                report.undecided += 1
                report.reasons.append(f"attack failed for '{law.cause} → {law.effect}': {exc}")
                continue

            if attack.refuted_by:
                report.refuted += 1
                self._record_death(beliefs, law, attack)
                continue
            if self._survived_independently(attack):
                # Promoted, and only now: it has met evidence that does not come from the claim
                # itself. PROPOSED → DERIVED is the one legal promotion; OBSERVED is unreachable
                # from here by construction, because nothing was observed.
                law.provenance = Provenance.promote(law.provenance, Provenance.DERIVED)
                if world is not None:
                    try:
                        world.state_law(law.cause, law.effect, kind=law.kind)
                    except Exception as exc:  # noqa: BLE001
                        report.reasons.append(f"could not state '{law.cause} → {law.effect}': {exc}")
                report.promoted += 1
                continue
            report.undecided += 1
            report.reasons.append(
                f"'{law.cause} → {law.effect}': {attack.verdict} — stays a proposal")
        return report

    @staticmethod
    def _survived_independently(attack: Any) -> bool:
        """Did it survive an attack settled by evidence *independent* of the claim?

        ``survived`` alone is not enough. :class:`~nyxara.njp.adversary.Attack` carries
        ``independent`` precisely because an attack answered from the claim's own support proves
        nothing new — and a proposal promoted on that basis would be a model agreeing with itself
        through one extra layer.
        """
        try:
            from nyxara.njp.adversary import Stance
            return any(a.stance == Stance.SURVIVED and getattr(a, "independent", False)
                       for a in getattr(attack, "attacks", ()))
        except Exception:  # noqa: BLE001
            return False

    @staticmethod
    def _record_death(beliefs: Any, law: CortexLaw, attack: Any) -> None:
        """Write the refutation into the belief ledger, so the destruction is part of the record.

        A model that is simply dropped leaves no trace, and the same proposal arrives again next
        turn to be tested again. What she used to think — and what killed it — is part of what she
        knows about herself.
        """
        if beliefs is None:
            return
        landed = getattr(attack, "refuted_by", ()) or ()
        why = "; ".join(str(getattr(a, "finding", "") or getattr(a, "kind", "")) for a in landed)
        claim = f"{law.cause} causes {law.effect}"
        try:
            if hasattr(beliefs, "revise"):
                beliefs.revise(claim, confidence=0.0,
                               reason=f"refuted by self-attack ({why or 'no supporting evidence'}); "
                                      "proposed by the cortex, never observed")
                return
        except Exception:  # noqa: BLE001
            pass
        try:
            if hasattr(beliefs, "record"):
                beliefs.record(claim, confidence=0.0, reason=f"refuted: {why}")
        except Exception:  # noqa: BLE001 — the ledger is a record, never a dependency
            pass

    # ---- reporting ---------------------------------------------------------- #
    def stats(self) -> Dict[str, Any]:
        """Lifetime counters. ``extraction_rate`` describes the extractor, not the model."""
        rate = 0.0 if not self.rows_seen else (self.rows_seen - self.rows_dropped) / self.rows_seen
        return {"provider": self.provider, "available": self.available(),
                "reason_unavailable": self.why_unavailable() or None,
                "asked": self.asked, "answered": self.answered,
                "screened": self.screened, "unparsable": self.unparsable,
                "rows_seen": self.rows_seen, "rows_dropped": self.rows_dropped,
                "extraction_rate": round(rate, 4),
                "proposal_ceiling": _PROPOSAL_CEILING}
