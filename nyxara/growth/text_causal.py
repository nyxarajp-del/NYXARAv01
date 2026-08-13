"""NYXARA · growth/text_causal.py — L-CAUSAL for prose: sentences become a DAG (📖⇢).

:mod:`~nyxara.growth.dataset_causal` turns a table into a causal graph. Most of what the Master
will hand her is not a table — it is prose, and prose is full of causal assertions that a
next-token model absorbs as word order and nothing more. This module reads them as claims.

Extraction is a deterministic **cue grammar** over sentences: ``causes``, ``leads to``,
``results in``, ``because``, ``due to``, ``if … then``, ``therefore`` for forward claims;
``prevents``, ``inhibits``, ``blocks``, ``reduces`` for negative ones. No LLM is involved, so the
same text always yields the same claims and every claim can be traced to the sentence it came from.

**What this is, stated plainly: extraction of *claimed* causality, not verification of it.** A
sentence saying "X causes Y" is evidence that someone asserted it, nothing more — text is exactly
the medium in which false causal claims travel. Three things follow, and they are the difference
between a useful layer and a confident liar:

* Text-derived links carry a **strictly lower confidence** than table-derived ones, and their
  ``CausalLink.evidence`` records the cue, the quoted sentence and the source file, so an audit can
  always tell an asserted edge from a measured one.
* Every claim is run through :class:`~nyxara.causal.causal_knots.KnotLattice` — the signed
  parity union-find already in the package — so a corpus that says "X increases Y" in one place and
  "X decreases Y" in another has that **contradiction detected and the second claim refused**,
  rather than both being absorbed into the weights.
* Concessive sentences ("despite", "although", "no evidence that") are **defeaters**: they are
  detected and dropped, because the surface pattern in them points the wrong way.
* **Retractions** ("actually…", "I was wrong", "maine galat kaha") are defeaters too, for a
  different reason. A speaker correcting themselves is the one case where "X causes Y" followed
  by "X prevents Y" is not a contradiction — it is one person updating. Read as an assertion it
  would trip the knot gate and get an honest correction flagged as a probable hallucination.
  Dropped, the prior claim simply stands: nothing in a cue grammar can tell which of the two the
  speaker now means, and guessing would be worse than declining to guess.

Depends on ``senses/nlp`` (sentence splitting, stdlib), ``causal/causal_knots`` (the contradiction
gate) and ``mind/causal_world_model`` (the graph). Nothing imports back.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

__all__ = ["CausalClaim", "TextCausalReport", "extract_claims", "claims_to_graph",
           "learn_from_text"]

# Confidence ceiling for anything read out of prose. A measured table edge earns up to 1.0; an
# asserted sentence must never be able to outrank it.
_TEXT_CONFIDENCE = 0.35

# Sentences whose surface pattern points the wrong way — a concession is not an assertion.
_DEFEATERS = re.compile(
    r"\b(despite|although|though|even if|no evidence|not caused|does not cause|"
    r"doesn't cause|is not caused|unrelated to|contrary to|myth that)\b", re.I)

# Sentences that *retract* something said earlier. A speaker correcting themselves is the one
# case where "X causes Y" then "X prevents Y" is not a contradiction at all — it is the same
# person updating. Left unhandled, the knot lattice reads the pair as a Knot Mutation Failure
# and flags an honest correction as a probable hallucination.
#
# The weak cues ("actually", "no wait") are anchored to the start of the sentence: mid-sentence
# they are ordinary filler, and dropping every sentence containing the word "actually" would
# cost far more real claims than it saves. The explicit retractions match anywhere.
# Each alternative must consume the *whole* cue phrase: what is left after it is stripped
# becomes the replacement claim, so a half-eaten cue would leave "kaha" or "wrong" glued to the
# cause and mint a junk node.
_CORRECTIONS = re.compile(
    r"^\s*(?:actually|correction|no[, ]+wait|wait[, ]+no|on second thought(?:s)?|"
    r"scratch that|sorry[, ]+i meant|i meant)"
    r"|\bi (?:was wrong|misspoke|take that back|stand corrected)"
    r"|\b(?:my mistake|let me correct(?: myself)?|to correct myself)"
    r"|\b(?:maine\s+)?galat\s+(?:kaha|bola|tha)"
    r"|(?:मैंने\s*)?(?:ग़लत|गलत)\s*(?:कहा|बोला|था)", re.I)

# Punctuation and filler left dangling once the cue is removed ("Actually, X…" → "X…").
_CUE_RESIDUE = re.compile(r"^[\s,;:.—-]+(?:that\s+|but\s+)?", re.I)


def is_correction(sentence: str) -> bool:
    """True when a sentence retracts an earlier claim rather than asserting a new one."""
    return bool(_CORRECTIONS.search(sentence or ""))


def strip_correction(sentence: str) -> str:
    """Remove the retraction cue, leaving the claim the speaker is replacing it with.

    ``"Actually, caffeine reduces my focus"`` → ``"caffeine reduces my focus"``. Returns the
    sentence unchanged when it carries no cue.
    """
    match = _CORRECTIONS.search(sentence or "")
    if match is None:
        return sentence or ""
    rest = (sentence[:match.start()] + " " + sentence[match.end():]).strip()
    return _CUE_RESIDUE.sub("", rest).strip()

# (name, pattern, cause_group, effect_group, polarity)
_PATTERNS: Tuple[Tuple[str, "re.Pattern[str]", int, int, int], ...] = (
    # effect-first: "Y because of X" / "Y due to X"
    ("because_of", re.compile(
        r"^(.+?)\s+(?:because of|due to|owing to|as a result of|thanks to)\s+(.+)$", re.I),
     2, 1, 1),
    # cause-first clause: "because X, Y"
    ("because_clause", re.compile(r"^because\s+(.+?)\s*,\s*(.+)$", re.I), 1, 2, 1),
    # conditional: "if X then Y" / "if X, Y"
    ("if_then", re.compile(r"^if\s+(.+?)[,\s]+then\s+(.+)$", re.I), 1, 2, 1),
    # explicit causal verbs
    ("causes", re.compile(
        r"^(.+?)\s+(?:causes?|caused|leads? to|led to|results? in|resulted in|triggers?|"
        r"triggered|brings? about|produces?|drives?|induces?)\s+(.+)$", re.I), 1, 2, 1),
    # magnitude verbs — "X increases Y" is a signed causal claim, and pairing it with its
    # opposite below is what lets the knot lattice catch a corpus contradicting itself.
    ("increases", re.compile(
        r"^(.+?)\s+(?:increases?|increased|raises?|raised|boosts?|boosted|improves?|"
        r"improved|enhances?|enhanced|strengthens?|strengthened|worsens?|worsened)\s+(.+)$",
        re.I), 1, 2, 1),
    # preventive / diminishing verbs — a real causal claim with negative polarity
    ("prevents", re.compile(
        r"^(.+?)\s+(?:prevents?|prevented|inhibits?|inhibited|blocks?|blocked|reduces?|"
        r"reduced|suppresses?|suppressed|lowers?|lowered|weakens?|weakened|"
        r"decreases?|decreased)\s+(.+)$", re.I), 1, 2, -1),
    # inference connectives: "X, therefore Y"
    ("therefore", re.compile(
        r"^(.+?)\s*[,;]\s*(?:therefore|hence|consequently|so|thus)\s+(.+)$", re.I), 1, 2, 1),
)

# Trimmed from the edges of an extracted phrase so "the heavy rain" and "heavy rain" are one node.
_EDGE_STOPWORDS = frozenset({
    "a", "an", "the", "this", "that", "these", "those", "it", "its", "his", "her", "their",
    "of", "in", "on", "at", "to", "for", "and", "or", "but", "is", "are", "was", "were", "be",
    "been", "being", "has", "have", "had", "will", "would", "can", "could", "may", "might",
    "often", "usually", "always", "sometimes", "generally", "typically", "very", "much",
})
_MAX_LABEL_WORDS = 4


def _normalize_label(phrase: str) -> str:
    """Reduce a phrase to a stable node label so paraphrases land on the same node.

    Lowercase, strip punctuation, drop leading/trailing function words, and keep at most
    :data:`_MAX_LABEL_WORDS` content words. Crude on purpose: a fancier normaliser would be a
    source of silent, untraceable merges."""
    words = re.findall(r"[a-z0-9'-]+", (phrase or "").lower())
    while words and words[0] in _EDGE_STOPWORDS:
        words.pop(0)
    while words and words[-1] in _EDGE_STOPWORDS:
        words.pop()
    if not words:
        return ""
    return " ".join(words[:_MAX_LABEL_WORDS])


@dataclass
class CausalClaim:
    """One causal assertion read out of a sentence, with the sentence kept for audit."""

    cause: str
    effect: str
    cue: str
    polarity: int = 1               # +1 "causes", -1 "prevents"
    quote: str = ""
    source: str = ""
    confidence: float = _TEXT_CONFIDENCE
    corrective: bool = False        # retracts an earlier claim rather than adding to them

    def as_tuple(self) -> Tuple[str, str, int]:
        """``(cause, effect, sign)`` — the shape :meth:`KnotLattice.check` takes."""
        return self.cause, self.effect, self.polarity

    def to_dict(self) -> Dict[str, Any]:
        return {"cause": self.cause, "effect": self.effect, "cue": self.cue,
                "polarity": self.polarity, "quote": self.quote, "source": self.source,
                "confidence": round(self.confidence, 4), "corrective": self.corrective}


@dataclass
class TextCausalReport:
    """Countable account of a prose pass — including every claim that was refused."""

    sentences: int = 0
    claims: List[CausalClaim] = field(default_factory=list)
    defeated: int = 0                  # concessive sentences skipped
    corrections: int = 0               # retractions read ("actually, X prevents Y")
    retracted: int = 0                 # strands un-tied to make room for a correction
    contradicted: int = 0              # claims refused by the knot lattice
    registered: int = 0                # links written onto the CausalWorldModel
    contradictions: List[str] = field(default_factory=list)
    # one record per retraction: what was un-tied, and the sentence that un-said it
    retractions: List[Dict[str, Any]] = field(default_factory=list)

    def summary(self) -> str:
        lines = [f"· text-causal: {len(self.claims)} claim(s) from {self.sentences} sentence(s); "
                 f"{self.defeated} concessive sentence(s) skipped, "
                 f"{self.corrections} retraction(s) ({self.retracted} strand(s) un-tied), "
                 f"{self.contradicted} contradiction(s) refused"]
        for claim in self.claims[:10]:
            arrow = "→" if claim.polarity > 0 else "⊣"
            lines.append(f"    {claim.cause} {arrow} {claim.effect}  [{claim.cue}]")
        lines.extend(f"    ! {c}" for c in self.contradictions[:5])
        if self.claims:
            lines.append("    (claimed causality, not verified — confidence is capped below "
                         "anything measured from data)")
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {"sentences": self.sentences, "claims": [c.to_dict() for c in self.claims],
                "defeated": self.defeated, "corrections": self.corrections,
                "retracted": self.retracted, "contradicted": self.contradicted,
                "registered": self.registered, "contradictions": list(self.contradictions),
                "retractions": [dict(r) for r in self.retractions]}


def _sentences(text: str) -> List[str]:
    """Split into sentences with the stdlib NLP module, falling back to a regex."""
    try:
        from nyxara.senses.nlp import sentences
        return [s for s in sentences(text) if s.strip()]
    except Exception:  # noqa: BLE001 — the fallback keeps this working on any box
        return [s for s in re.split(r"(?<=[.!?])\s+", text or "") if s.strip()]


def extract_claims(text: str, *, source: str = "",
                   report: Optional["TextCausalReport"] = None
                   ) -> Tuple[List[CausalClaim], int, int]:
    """Read causal claims out of prose. Returns ``(claims, n_sentences, n_defeated)``.

    Deterministic: the same text always yields the same claims, and each carries the sentence it
    came from so nothing is asserted without a traceable quote.

    ``n_defeated`` counts every sentence skipped — concessions and retractions alike. Pass a
    ``report`` to have the two counted apart; the returned tuple keeps its historical shape.
    """
    claims: List[CausalClaim] = []
    sentences = _sentences(text)
    defeated = 0
    for sentence in sentences:
        clean = sentence.strip().rstrip(".!?")
        if not clean:
            continue
        # A retraction is not an assertion *and* not a contradiction: it replaces what was said
        # before. Strip the cue and read what is left as the replacement claim, flagged so the
        # gate retracts the old strand instead of refusing the new one.
        corrective = bool(_CORRECTIONS.search(clean))
        if corrective:
            clean = strip_correction(clean)
            if not clean:
                defeated += 1                 # a bare "I was wrong." — nothing to put in its place
                if report is not None:
                    report.corrections += 1
                continue
        if _DEFEATERS.search(clean):
            # "X does not cause Y" matches the causal pattern too — the concession must win.
            defeated += 1
            if report is not None:
                report.defeated += 1
            continue
        matched = False
        for cue, pattern, cause_group, effect_group, polarity in _PATTERNS:
            match = pattern.match(clean)
            if not match:
                continue
            cause = _normalize_label(match.group(cause_group))
            effect = _normalize_label(match.group(effect_group))
            if not cause or not effect or cause == effect:
                continue
            claims.append(CausalClaim(cause=cause, effect=effect, cue=cue, polarity=polarity,
                                      quote=sentence.strip()[:200], source=source,
                                      corrective=corrective))
            matched = True
            break            # one claim per sentence: the first cue that fires is the reading
        if corrective and not matched:
            # a retraction with no readable causal claim behind it — still not an assertion
            defeated += 1
            if report is not None:
                report.corrections += 1
    return claims, len(sentences), defeated


def claims_to_graph(claims: Sequence[CausalClaim], *, model: Any = None,
                    lattice: Any = None, report: Optional[TextCausalReport] = None
                    ) -> TextCausalReport:
    """Run claims through the contradiction gate and register the survivors.

    Claims are tied one at a time rather than as a batch: a single contradiction should cost that
    one claim, not every claim in the document."""
    report = report if report is not None else TextCausalReport()
    if lattice is None:
        try:
            from nyxara.causal.causal_knots import KnotLattice
            lattice = KnotLattice()
        except Exception:  # noqa: BLE001 — without the gate we still register, but say so
            lattice = None
            report.contradictions.append("knot lattice unavailable: claims not cross-checked")

    accepted: List[CausalClaim] = []
    for claim in claims:
        if lattice is not None:
            if claim.corrective:
                # The speaker said this instead of what they said before, so un-tie the old
                # strand first. Only the pair being corrected is touched; everything else the
                # lattice holds survives the rebuild. The quoted sentence rides along as the
                # reason, so the lattice's own log can say who un-said what.
                report.corrections += 1
                try:
                    before = _lattice_strands(lattice, claim.cause, claim.effect)
                    removed = lattice.retract(claim.cause, claim.effect,
                                              reason=f"retracted by: {claim.quote}")
                    report.retracted += removed
                    if removed:
                        report.retractions.append({
                            "cause": claim.cause, "effect": claim.effect,
                            "by": claim.quote, "source": claim.source,
                            "new_polarity": claim.polarity, "removed": before})
                except TypeError:  # pragma: no cover — a lattice predating the reason kwarg
                    report.retracted += lattice.retract(claim.cause, claim.effect)
                except Exception:  # noqa: BLE001 — a lattice that cannot retract at all: the
                    pass                       # tie below simply refuses, as it did before
            try:
                # pass the lattice's own sign constants rather than a bool or a raw int
                lattice.tie(claim.cause, claim.effect, claim.polarity, reason=claim.cue)
            except Exception as exc:  # noqa: BLE001 — KnotMutationFailure and anything like it
                report.contradicted += 1
                report.contradictions.append(
                    f"{claim.cause} {'→' if claim.polarity > 0 else '⊣'} {claim.effect}: {exc}")
                continue
        accepted.append(claim)
    report.claims.extend(accepted)

    if model is not None:
        report.registered += _register_claims(model, accepted)
    return report


def _register_claims(model: Any, claims: Sequence[CausalClaim]) -> int:
    """Write surviving claims onto a CausalWorldModel with full provenance (never raises).

    A pre-existing link is never overwritten: an edge measured from data outranks one asserted in
    a sentence, and this must not be able to demote it.

    The one exception is a **retraction**, and only over an edge that was itself asserted in text:
    a speaker replacing what they said must be able to replace what their words wrote, or the
    graph would keep quoting the sentence they just took back. A measured edge still wins — being
    corrected about a claim is not evidence against a measurement."""
    try:
        from nyxara.mind.causal_world_model import CAUSAL, CausalLink
    except Exception:  # noqa: BLE001
        return 0
    links = getattr(model, "_links", None)
    if links is None:
        return 0

    written = 0
    for claim in claims:
        key = (claim.cause, claim.effect)
        if key in links and not _may_supersede(claim, links[key]):
            continue          # measured evidence already holds this edge; assertions do not win
        try:
            links[key] = CausalLink(
                cause=claim.cause, effect=claim.effect,
                strength=float(claim.polarity) * claim.confidence,
                confidence=claim.confidence, verdict=CAUSAL, lag=0.0,
                evidence={"source": "text_claim", "cue": claim.cue, "quote": claim.quote,
                          "path": claim.source, "polarity": claim.polarity})
            written += 1
        except Exception:  # noqa: BLE001
            continue
    return written


def _lattice_strands(lattice: Any, cause: str, effect: str) -> List[Dict[str, Any]]:
    """The strands a retraction is about to remove, captured for the audit trail."""
    pair = {(cause, effect), (effect, cause)}
    try:
        return [k.to_dict() for k in lattice._knots      # noqa: SLF001 — reading for the record
                if (k.cause, k.effect) in pair]
    except Exception:  # noqa: BLE001 — a lattice without that internal is still usable
        return []


def _may_supersede(claim: CausalClaim, existing: Any) -> bool:
    """True when ``claim`` is a retraction and ``existing`` is itself only a text assertion."""
    if not claim.corrective:
        return False
    evidence = getattr(existing, "evidence", None) or {}
    try:
        return evidence.get("source") == "text_claim"
    except AttributeError:      # pragma: no cover — an evidence blob that isn't a mapping
        return False


def learn_from_text(text: str, *, model: Any = None, source: str = "",
                    lattice: Any = None) -> TextCausalReport:
    """Extract causal claims from prose and fold the consistent ones into the graph."""
    report = TextCausalReport()
    claims, n_sentences, _defeated = extract_claims(text, source=source, report=report)
    report.sentences = n_sentences        # ``report`` already counted defeats and retractions
    return claims_to_graph(claims, model=model, lattice=lattice, report=report)


def learn_from_dataset_store(store_path: Any, *, model: Any = None,
                             limit: Optional[int] = None) -> TextCausalReport:
    """Read every prose record out of an ingested dataset store and mine it for causal claims."""
    from nyxara.growth.dataset import load_dataset_docs

    report = TextCausalReport()
    lattice = None
    try:
        from nyxara.causal.causal_knots import KnotLattice
        lattice = KnotLattice()        # one lattice for the whole store, so cross-file claims clash
    except Exception:  # noqa: BLE001
        lattice = None

    name = Path(str(store_path)).name
    for doc in load_dataset_docs(store_path, limit=limit):
        claims, n_sentences, _defeated = extract_claims(doc, source=name, report=report)
        report.sentences += n_sentences   # ``report`` already counted defeats and retractions
        claims_to_graph(claims, model=model, lattice=lattice, report=report)
    return report
