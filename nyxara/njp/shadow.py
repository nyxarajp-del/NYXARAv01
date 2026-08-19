"""NYXARA · njp/shadow.py — two minds on the same input, and what their disagreement is for (👥).

:mod:`nyxara.njp.graft` puts a trained network's weights inside the fabric. That buys a
**computational substrate** and not one bit of cognition: a converted layer computes what it
always computed, and NJP's own world model, concepts and causal arrows are exactly as good after
the graft as before it. This module is what turns the imported model from a brain she runs into a
teacher she learns from.

**The architecture.** One stimulus, two paths, three gaps::

    INPUT ──┬── teacher  ──→ representation ──┐
            └── NJP      ──→ world model    ──┴──→ COMPARATOR
                                                       │
                            ┌──────────────────────────┼──────────────────────────┐
                        SEMANTIC gap               CAUSAL gap              PREDICTION gap
                    (concepts she lacks)      (arrows she lacks)      (states she gets wrong)
                            │                          │                          │
                     njp/concepts.py            njp/adversary.py           njp/predictive.py
                     njp/field.py               njp/universe.py            njp/learn.py

The routing is :mod:`nyxara.njp.integrate`'s pattern — every diagnosed miss goes to the organ that
owns the repair — applied to a second source of misses.

**The rule this module exists to enforce, and the reason it is enforced in code rather than in
this docstring:**

    **The teacher may propose. It may never grade.**

:mod:`nyxara.njp.integrate` already states the general form: *"Scoring a prediction against your
own later opinion is not learning, it is self-congratulation… every outcome here is independent of
the prediction it scores."* A teacher's answer is not her own later opinion, which makes it feel
like an independent outcome. It is not one. It is a **correlated** one: the moment a teacher's
output is allowed to score NJP's prediction, the training signal says "agree with the teacher",
NJP converges on the teacher's answers *including its errors*, and the one measurement that was
supposed to decide whether any of this worked — remove the teacher, does she still perform —
passes trivially, because she has become a lossy copy of the thing that was removed. The
experiment would report success in precisely the case where nothing was learned.

So a comparison here produces an **investigation**, never an outcome. :class:`ShadowReading` has
no field an outcome could be written into, :attr:`ShadowReading.outcomes` is structurally empty,
and grading continues to come from where :mod:`~nyxara.njp.integrate` already takes it: physics at
``t+1``, a fact the Master actually stated, or an attack :mod:`~nyxara.njp.adversary` settled
against the record. ``tests/njp/test_shadow_not_grader.py`` fails if that stops being true.

**What a gap is worth.** A disagreement is information about where her model is thin — it is not
evidence that the teacher is right. :meth:`ShadowCognition.challenge` puts every teacher causal
claim through the four attacks, and a claim that cannot be settled from the record comes back
``UNDECIDED``, which never raises confidence. The teacher is a hypothesis generator whose
hypotheses are cheap and whose authority is zero.

Pure standard library plus numpy. Every entry point is fail-soft: any organ that raises degrades
to a null result and the turn still completes.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

__all__ = ["GapKind", "Gap", "ShadowReading", "ShadowCognition"]


class GapKind:
    """The three ways a teacher and NJP can differ, each with a different owner."""

    SEMANTIC = "semantic"       # it has a concept she does not
    CAUSAL = "causal"           # it asserts an arrow she does not hold
    PREDICTION = "prediction"   # it expects a next state she does not


#: Below this a gap is noise rather than a finding. Routing every microscopic difference would
#: bury the organs in work and would make ``gaps`` a measure of floating-point luck.
MIN_GAP = 0.05

#: A causal claim in the teacher's text. Deliberately narrow: a loose pattern turns every "and"
#: into an arrow, and a spurious arrow costs four adversary attacks to reject.
_CAUSAL = re.compile(
    r"\b(?P<cause>[a-z][a-z_ ]{0,40}?)\s+"
    r"(?:causes|cause|leads to|results in|produces|se hota hai|ki wajah se)\s+"
    r"(?P<effect>[a-z][a-z_ ]{0,40}?)\b",
    re.IGNORECASE)


@dataclass
class Gap:
    """One measured disagreement, and the organ it was handed to.

    ``teacher_view`` and ``njp_view`` are recorded verbatim so a later reader can see what the
    disagreement actually was rather than only that there was one. Neither is a verdict on the
    other.
    """

    kind: str
    subject: str = ""
    teacher_view: Any = None
    njp_view: Any = None
    magnitude: float = 0.0          # [0, 1] — how far apart, in the gap kind's own units
    routed_to: str = ""             # which organ received it
    accepted: bool = False          # did that organ take it up (a proposal, not a fact)
    detail: str = ""

    #: A gap is an investigation. This is not a flag anything may set — it is the type-level
    #: statement that nothing here can become a score, and the test suite asserts it.
    is_outcome: bool = field(default=False, init=False)

    def to_dict(self) -> Dict[str, Any]:
        return {"kind": self.kind, "subject": self.subject,
                "teacher_view": _short(self.teacher_view), "njp_view": _short(self.njp_view),
                "magnitude": round(float(self.magnitude), 6),
                "routed_to": self.routed_to, "accepted": self.accepted,
                "detail": self.detail, "is_outcome": False}


@dataclass
class ShadowReading:
    """What one shadow comparison found. Investigations only — by construction."""

    stimulus: str = ""
    gaps: List[Gap] = field(default_factory=list)
    claims_challenged: int = 0
    claims_refuted: int = 0
    claims_undecided: int = 0
    residuals_offered: int = 0
    residuals_trained: int = 0      # only those an independent check established
    teacher_present: bool = False
    ms: float = 0.0

    @property
    def outcomes(self) -> Tuple[()]:
        """Always empty, and this is the point rather than an oversight.

        An outcome is something that may move a confidence or a weight. Nothing a teacher said
        qualifies, so there is no container here for one to be put in. Code looking for the
        signal that grades a turn should read :mod:`nyxara.njp.integrate`, which takes it from
        physics, from the Master, or from a settled attack.
        """
        return ()

    @property
    def investigations(self) -> List[Gap]:
        return list(self.gaps)

    def by_kind(self, kind: str) -> List[Gap]:
        return [g for g in self.gaps if g.kind == kind]

    def to_dict(self) -> Dict[str, Any]:
        return {"stimulus": self.stimulus[:160],
                "gaps": [g.to_dict() for g in self.gaps],
                "n_gaps": len(self.gaps),
                "semantic": len(self.by_kind(GapKind.SEMANTIC)),
                "causal": len(self.by_kind(GapKind.CAUSAL)),
                "prediction": len(self.by_kind(GapKind.PREDICTION)),
                "claims_challenged": self.claims_challenged,
                "claims_refuted": self.claims_refuted,
                "claims_undecided": self.claims_undecided,
                "residuals_offered": self.residuals_offered,
                "residuals_trained": self.residuals_trained,
                "teacher_present": self.teacher_present,
                "outcomes": 0,           # stated in every report, so its absence is visible
                "ms": round(self.ms, 3)}


def _short(v: Any, n: int = 96) -> Any:
    if isinstance(v, str):
        return v[:n]
    if isinstance(v, (list, tuple, set)):
        return sorted(map(str, v))[:8]
    if isinstance(v, (int, float, bool)) or v is None:
        return v
    return str(v)[:n]


# --------------------------------------------------------------------------- #
# The comparator
# --------------------------------------------------------------------------- #
class ShadowCognition:
    """Runs the teacher beside NJP, measures where they differ, routes each difference.

    Every organ is optional and absent is a normal state: with no teacher this degrades to a
    reading that says so, which is the honest behaviour on a machine where the weights were never
    downloaded. Nothing here raises into a turn.
    """

    def __init__(self, *, teacher: Any = None, concepts: Any = None, adversary: Any = None,
                 universe: Any = None, predictor: Any = None, readout: Any = None,
                 lingua: Any = None, ledger: Any = None,
                 min_gap: float = MIN_GAP, max_claims: int = 8) -> None:
        self.teacher = teacher
        self.concepts = concepts
        self.adversary = adversary
        self.universe = universe
        self.predictor = predictor
        self.readout = readout
        self.lingua = lingua
        self.ledger = ledger
        self.min_gap = float(min_gap)
        self.max_claims = max(1, int(max_claims))
        self.comparisons = 0
        self.gaps_found = 0
        self.gaps_routed = 0
        self.claims_challenged = 0
        self.claims_refuted = 0
        self.residuals_trained = 0
        self.residuals_refused = 0

    # ---- the comparison -------------------------------------------------- #
    def compare(self, stimulus: str, *, njp_thought: Any = None,
                teacher_text: str = "", njp_text: str = "") -> ShadowReading:
        """One stimulus through both paths. Returns investigations, never outcomes."""
        t0 = time.perf_counter()
        out = ShadowReading(stimulus=str(stimulus or ""))
        try:
            teacher_text = str(teacher_text or "")
            if not teacher_text and self.teacher is not None:
                teacher_text = self._ask_teacher(stimulus)
            out.teacher_present = bool(teacher_text)
            if not teacher_text:
                out.ms = (time.perf_counter() - t0) * 1000.0
                return out

            njp_text = str(njp_text or self._njp_text(njp_thought))
            self.comparisons += 1

            self._semantic_gap(out, teacher_text, njp_text)
            self._causal_gap(out, teacher_text)
            self._prediction_gap(out, njp_thought, teacher_text)

            self.gaps_found += len(out.gaps)
            self.gaps_routed += sum(1 for g in out.gaps if g.accepted)
            self._record(out)
        except Exception:  # noqa: BLE001 — a failed comparison is an empty reading
            pass
        out.ms = (time.perf_counter() - t0) * 1000.0
        return out

    # ---- gap 1: concepts it has and she does not ------------------------- #
    def _semantic_gap(self, out: ShadowReading, teacher_text: str, njp_text: str) -> None:
        """Concepts present in the teacher's answer and absent from hers.

        Routed to :class:`~nyxara.njp.concepts.ConceptGenesis` as an **observation**, which is the
        strongest thing it is allowed to be. A concept the teacher used is evidence that the word
        exists and gets used in this context — it is not evidence of what the concept is related
        to, and emphatically not of what causes what. That boundary is
        :mod:`nyxara.njp.relevance`'s lesson: a true thing with no established bearing on the
        question is exactly how a fluent system produces confident nonsense.
        """
        try:
            them = set(self._concepts(teacher_text))
            hers = set(self._concepts(njp_text))
            missing = sorted(them - hers)
            if not missing:
                return
            union = len(them | hers) or 1
            mag = len(missing) / union
            if mag < self.min_gap:
                return
            gap = Gap(kind=GapKind.SEMANTIC, subject=", ".join(missing[:6]),
                      teacher_view=sorted(them)[:12], njp_view=sorted(hers)[:12],
                      magnitude=min(1.0, mag), routed_to="njp.concepts")
            if self.concepts is not None:
                taken = 0
                for word in missing[:12]:
                    try:
                        # Co-occurrence only: the other words in the same answer. Not a definition
                        # and not a relation — the concept layer decides for itself whether a
                        # kind is there, and it pays for it in description length.
                        self.concepts.observe(word, features=[w for w in them if w != word][:8])
                        taken += 1
                    except Exception:  # noqa: BLE001
                        continue
                gap.accepted = taken > 0
                gap.detail = f"{taken} concept(s) offered as observations"
            else:
                gap.detail = "no concept organ attached"
            out.gaps.append(gap)
        except Exception:  # noqa: BLE001
            pass

    # ---- gap 2: arrows it asserts and she does not ----------------------- #
    def _causal_gap(self, out: ShadowReading, teacher_text: str) -> None:
        """Every causal claim in the teacher's text, put through the four attacks.

        This is the inversion the whole design turns on. A teacher sentence of the form "X causes
        Y" arrives as a **hypothesis with no standing**, and :mod:`nyxara.njp.adversary` decides
        what happens to it from *her own record*: a rival cause, a base rate, a counterexample, or
        the do-operator. ``UNDECIDED`` is the common answer on a young record and it is not a
        pass — the claim simply does not enter.
        """
        for cause, effect in self._claims(teacher_text)[:self.max_claims]:
            gap = Gap(kind=GapKind.CAUSAL, subject=f"{cause} -> {effect}",
                      teacher_view=f"{cause} causes {effect}", magnitude=1.0,
                      routed_to="njp.adversary")
            if self.adversary is None:
                gap.detail = "no adversary attached — claim NOT accepted"
                out.gaps.append(gap)
                continue
            try:
                report = self.adversary.attack(cause, effect)
                # Read the structured fields, not `verdict`. `AttackReport.verdict` is prose for
                # a human ("nothing could be settled from the record"), and matching strings
                # against it silently classified every unsettled claim as neither refuted nor
                # undecided — the counter that is supposed to show how thin the record is would
                # have sat at zero while the record was at its thinnest.
                refuted = list(getattr(report, "refuted_by", []) or [])
                survived = int(getattr(report, "survived", 0) or 0)
                undecided = int(getattr(report, "undecided", 0) or 0)
                out.claims_challenged += 1
                self.claims_challenged += 1
                gap.njp_view = str(getattr(report, "verdict", ""))
                gap.detail = (f"refuted={len(refuted)} survived={survived} "
                              f"undecided={undecided}")
                if refuted:
                    out.claims_refuted += 1
                    self.claims_refuted += 1
                elif survived == 0:
                    # Nothing landed and nothing was established: the claim does not enter. This
                    # is the common answer on a young record and it is not a pass.
                    out.claims_undecided += 1
                # `accepted` means the attack was actually run, NOT that the claim was believed.
                # Naming it after the challenge rather than the conclusion is deliberate: this
                # counter must never be readable as "how many teacher claims she took on trust".
                gap.accepted = True
            except Exception:  # noqa: BLE001
                gap.detail = "attack failed — claim NOT accepted"
            out.gaps.append(gap)

    # ---- gap 3: next states it expects and she does not ------------------ #
    def _prediction_gap(self, out: ShadowReading, njp_thought: Any,
                        teacher_text: str) -> None:
        """Where her forward model and the teacher's expectation part company.

        Recorded and offered; trained on only through :meth:`residual`, which refuses without an
        independent check. The gap alone is a place to look, not a correction to apply.
        """
        try:
            if self.predictor is None:
                return
            theirs = " ".join(self._concepts(teacher_text)[:6])
            if not theirs:
                return

            # Her forward model over world states — `njp/predictive.py`, not the manifold's
            # cell-level anticipation on the percept. The manifold predicts WHICH CELLS fire next,
            # which is a statement about her own dynamics; comparing that to a sentence would be
            # comparing two different kinds of thing and calling the mismatch a disagreement.
            pred = self.predictor.predict()
            answerable = bool(getattr(pred, "answerable", False))
            hers = str(getattr(pred, "top", "") or "")
            conf = float(getattr(pred, "confidence", 0.0) or 0.0)
            order = int(getattr(pred, "order", -1))

            if not answerable:
                # She has no forward model for this yet. That is NOT a maximal disagreement, and
                # scoring it as one would make this counter a measure of how young her world
                # model is rather than of where the teacher and she actually differ — every early
                # turn would report a full-magnitude gap and the signal would be worthless
                # exactly while the model is being built.
                out.gaps.append(Gap(
                    kind=GapKind.PREDICTION, subject="<no forward model yet>",
                    teacher_view=theirs, njp_view=None, magnitude=0.0,
                    routed_to="njp.predictive",
                    detail="her world model cannot answer here yet — recorded, not scored"))
                return

            mag = _token_distance(hers, theirs)
            if mag < self.min_gap:
                return
            out.residuals_offered += 1
            out.gaps.append(Gap(
                kind=GapKind.PREDICTION, subject=hers,
                teacher_view=theirs, njp_view=hers, magnitude=min(1.0, mag),
                routed_to="njp.predictive",
                detail=(f"her confidence {conf:.3f} at backoff order {order}; offered as a "
                        f"residual — training requires an independent check")))
        except Exception:  # noqa: BLE001
            pass

    # ---- residual learning, and the gate on it --------------------------- #
    def residual(self, teacher_vector: Any, njp_vector: Any, *,
                 established: bool = False, evidence: str = "") -> Optional[Any]:
        """``teacher − njp``, but only when something independent established the teacher.

        The target is the **disagreement**, not the teacher's answer, because that is where the
        information is: copying an answer she already gets right teaches nothing, and the
        difference points at the specific thing her model is missing.

        ``established`` is the whole safety of this method and it is not a formality. Without it
        the residual trains her toward the teacher unconditionally — including on every case the
        teacher got wrong — and it does so *efficiently*, because a residual target is a sharper
        training signal than an imitation target. A confident wrong teacher plus residual learning
        is the fastest way to install an error. Callers pass ``established=True`` only from an
        independent check: a settled adversary attack, a fact the Master stated, or physics at
        ``t + 1``.
        """
        if not established:
            self.residuals_refused += 1
            return None
        try:
            import numpy as _np
            a = _np.asarray(teacher_vector, dtype=_np.float64).ravel()
            b = _np.asarray(njp_vector, dtype=_np.float64).ravel()
            if a.size != b.size or a.size == 0:
                self.residuals_refused += 1
                return None
            self.residuals_trained += 1
            self._note(f"residual trained on evidence: {evidence or 'unnamed'}")
            return a - b
        except Exception:  # noqa: BLE001
            self.residuals_refused += 1
            return None

    # ---- helpers --------------------------------------------------------- #
    def _ask_teacher(self, stimulus: str) -> str:
        try:
            fn = getattr(self.teacher, "complete", None) or getattr(self.teacher, "ask", None)
            if fn is None:
                return ""
            reply = fn(str(stimulus))
            return str(getattr(reply, "text", reply) or "")
        except Exception:  # noqa: BLE001 — an unreachable teacher is an absent teacher
            return ""

    @staticmethod
    def _njp_text(njp_thought: Any) -> str:
        if njp_thought is None:
            return ""
        for attr in ("answer", "text", "reply", "utterance"):
            got = getattr(njp_thought, attr, None)
            if got:
                return str(got)
        return ""

    def _concepts(self, text: str) -> List[str]:
        try:
            if self.lingua is not None:
                return list(self.lingua.concepts(str(text)))
            from nyxara.njp.tongue import concepts_in
            return list(concepts_in(str(text)))
        except Exception:  # noqa: BLE001
            return [w for w in re.findall(r"[a-z]{3,}", str(text).lower())][:32]

    @staticmethod
    def _claims(text: str) -> List[Tuple[str, str]]:
        out: List[Tuple[str, str]] = []
        try:
            for m in _CAUSAL.finditer(str(text)):
                cause = m.group("cause").strip().split()[-1].lower()
                effect = m.group("effect").strip().split()[0].lower()
                if cause and effect and cause != effect:
                    out.append((cause, effect))
        except Exception:  # noqa: BLE001
            pass
        return out

    def _record(self, reading: ShadowReading) -> None:
        try:
            if self.ledger is not None and hasattr(self.ledger, "note"):
                self.ledger.note("shadow", reading.to_dict())
        except Exception:  # noqa: BLE001 — observability never breaks a turn
            pass

    def _note(self, msg: str) -> None:
        try:
            if self.ledger is not None and hasattr(self.ledger, "note"):
                self.ledger.note("shadow.residual", {"detail": msg})
        except Exception:  # noqa: BLE001
            pass

    def stats(self) -> Dict[str, Any]:
        return {
            "comparisons": self.comparisons,
            "gaps_found": self.gaps_found,
            "gaps_routed": self.gaps_routed,
            "claims_challenged": self.claims_challenged,
            "claims_refuted": self.claims_refuted,
            "residuals_trained": self.residuals_trained,
            "residuals_refused": self.residuals_refused,
            "teacher_attached": self.teacher is not None,
            # Published on every report so its absence is a visible, checkable property rather
            # than a claim in a docstring.
            "outcomes_produced": 0,
        }


def _token_distance(a: str, b: str) -> float:
    """1 − Jaccard over word sets. Cheap, symmetric, and never claims more than set overlap."""
    sa = set(re.findall(r"[a-z0-9]+", str(a).lower()))
    sb = set(re.findall(r"[a-z0-9]+", str(b).lower()))
    if not sa and not sb:
        return 0.0
    union = len(sa | sb) or 1
    return 1.0 - (len(sa & sb) / union)
