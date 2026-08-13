"""NYXARA · mind/recursive_improver.py — Recursive Self Improvement (Level 3).

After the initial reasoning step proposes a candidate, this module runs N cycles of:

    Think → Critique → Improve → Re-evaluate → Keep-or-Discard

Each iteration scores the current candidate against multiple quality dimensions
(confidence calibration, rationale richness, response coherence) and attempts to
improve it.  The best-scoring candidate across all iterations is returned.

With a real LLM available, the improvement step calls a critique-then-revise prompt to
generate a genuinely better answer.  Without one (offline / test mode), rule-based probes
still genuinely improve the candidate: they repair degenerate text (collapsed whitespace,
stuttered words, duplicated sentences), calibrate the confidence, and derive a
content-aware rationale — no LLM, no fabrication, fully deterministic.

Design constraints (same as all reasoning modules):
- Returns a Candidate — the kernel disposes through every gate as always.
- Never modifies the candidate's capability/risk/reversible fields (that is the kernel's
  domain; the improver only refines text, confidence, and rationale).
- Character-safe: no iteration changes the candidate's kind from "respond" to "act" or
  raises its risk tier — it can only lower risk (by adding specificity) or leave it.
- Best-effort: any per-iteration failure falls back to the current best.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, List, Optional

__all__ = ["RecursiveImprover", "ImprovementTrace"]


# --------------------------------------------------------------------------- #
# Trace
# --------------------------------------------------------------------------- #
@dataclass
class ImprovementTrace:
    """Audit trail for one improve() call."""
    n_iterations: int
    scores: List[float] = field(default_factory=list)
    best_iteration: int = 0
    improved: bool = False          # True if final is better than initial

    def summary(self) -> str:
        if not self.scores:
            return f"{self.n_iterations}-iter pass (no scores)"
        hi = max(self.scores)
        lo = min(self.scores)
        return (f"{self.n_iterations}-iter recursive improve; "
                f"score {lo:.2f}→{hi:.2f}; best at iter {self.best_iteration}")


# --------------------------------------------------------------------------- #
# Scorer — multi-dimensional quality signal, no LLM required
# --------------------------------------------------------------------------- #
def _score_candidate(candidate: Any) -> float:
    """Cheap, dependency-free quality score in [0, 1].

    Combines:
    - Calibration: neither overconfident (>0.95 for non-trivial claims) nor
      underconfident (<0.3 for a clear response).
    - Rationale richness: longer, word-diverse rationales score higher.
    - Text length: very short responses score lower (too terse); very long ones
      are not penalised (the kernel truncates if needed).
    """
    text = str(getattr(candidate, "text", "") or "")
    conf = float(getattr(candidate, "confidence", 0.7) or 0.7)
    rat = str(getattr(candidate, "rationale", "") or "")

    # calibration score — sweet-spot around 0.6–0.85
    cal = 1.0 - abs(conf - 0.72) / 0.72   # peak at 0.72, falls off both sides
    cal = max(0.0, min(1.0, cal))

    # text score — reward useful length
    n_words = len(text.split())
    txt_score = math.log(1 + n_words) / math.log(1 + 80)   # saturates around 80 words
    txt_score = min(1.0, txt_score)

    # rationale score — unique words in rationale (diversity signal)
    rat_words = set(rat.lower().split())
    rat_score = math.log(1 + len(rat_words)) / math.log(1 + 20)
    rat_score = min(1.0, rat_score)

    return 0.4 * cal + 0.35 * txt_score + 0.25 * rat_score


def _grounded_score(stimulus: str, candidate: Any) -> float:
    """Keep-best score for the refinement loop, grounded in *truth* where the domain is decidable.

    On a decidable prompt (an exact faculty oracle answers it), correctness dominates: a refinement
    that carries the ground truth scores near 1.0, one that drifts away from it scores near 0.0 —
    so the loop never "improves" a correct answer into a fluent-but-wrong one (the ceiling-break at
    rung 4). On every non-decidable prompt it is exactly :func:`_score_candidate`, so open-ended
    behaviour is unchanged. Never raises."""
    text = str(getattr(candidate, "text", "") or "")
    shape = _score_candidate(candidate)
    try:
        from nyxara.mind.grounded_verifier import answer_carries_truth, ground_truth
        truth = ground_truth(stimulus)
        if truth is not None:
            return min(1.0, 0.9 + 0.1 * shape) if answer_carries_truth(truth, text) \
                else min(0.15, 0.05 + 0.10 * shape)
    except Exception:  # noqa: BLE001 — grounding is additive; fall back to the shape score
        pass
    return shape


# --------------------------------------------------------------------------- #
# Improver
# --------------------------------------------------------------------------- #
class RecursiveImprover:
    """Iteratively refine a candidate through N cycles of Think→Critique→Improve.

    Parameters
    ----------
    llm:            LLM facade for revision prompts (optional; rule-based fallback if None).
    n_iterations:   default iteration count (overridable per call); capped at 20.
    temperature:    generation temperature for revision prompts.
    max_tokens:     token budget per revision call.
    """

    _CRITIQUE_PROMPT = (
        "You previously gave this response to the Master's request:\n\n"
        "REQUEST: {stimulus}\n"
        "CURRENT RESPONSE: {text}\n"
        "CURRENT RATIONALE: {rationale}\n"
        "CURRENT CONFIDENCE: {confidence}\n\n"
        "CRITIQUE it honestly: Is it accurate? Is it complete? Is the confidence "
        "calibrated? Is the rationale clear? Then write an improved response — keep "
        "the same format (direct answer to the Master), but better. Output ONLY the "
        "improved response text, nothing else."
    )

    def __init__(self, llm: Any = None, n_iterations: int = 5,
                 temperature: float = 0.4, max_tokens: int = 512) -> None:
        self.llm = llm
        self.n_iterations = max(1, min(20, int(n_iterations)))
        self.temperature = float(temperature)
        self.max_tokens = max(64, int(max_tokens))

    def improve(self, stimulus: str, candidate: Any,
                n_iterations: Optional[int] = None) -> Any:
        """Run up to ``n_iterations`` improvement cycles; return the best candidate.

        The returned object is always a Candidate (or whatever was passed in) — the
        improver never wraps it in a new type.  Only .text, .confidence, and .rationale
        may change; .kind, .capability, .risk, and .reversible are never touched.
        """
        n = max(1, min(20, int(n_iterations or self.n_iterations)))
        trace = ImprovementTrace(n_iterations=n)

        best = candidate
        best_score = _grounded_score(stimulus, candidate)
        trace.scores.append(best_score)

        for i in range(1, n + 1):
            try:
                current = self._iterate(stimulus, best, iteration=i)
                score = _grounded_score(stimulus, current)
                trace.scores.append(score)
                if score > best_score:
                    best = current
                    best_score = score
                    trace.best_iteration = i
                    trace.improved = True
            except Exception:  # noqa: BLE001 — a failed iteration just doesn't update
                trace.scores.append(best_score)

        # annotate the winning candidate's rationale with the improvement trace
        _annotate(best, trace)
        return best

    # ---------------------------------------------------------------------- #
    def _iterate(self, stimulus: str, candidate: Any, *, iteration: int) -> Any:
        """One improvement iteration.  Tries LLM critique then rule-based probes."""
        if self._llm_available():
            return self._llm_revise(stimulus, candidate, iteration)
        return self._rule_revise(candidate, iteration)

    def _llm_revise(self, stimulus: str, candidate: Any, iteration: int) -> Any:
        """Ask the LLM to critique and rewrite the candidate text."""
        prompt = self._CRITIQUE_PROMPT.format(
            stimulus=stimulus[:500],
            text=(getattr(candidate, "text", "") or "")[:300],
            rationale=(getattr(candidate, "rationale", "") or "")[:200],
            confidence=f"{getattr(candidate, 'confidence', 0.7):.2f}",
        )
        try:
            new_text = self.llm.generate(prompt, temperature=self.temperature,
                                         max_tokens=self.max_tokens)
            new_text = (str(new_text or "")).strip()
            if not new_text or new_text == getattr(candidate, "text", ""):
                return candidate
            return _clone_with(candidate, text=new_text,
                               rationale=f"recursive-iter-{iteration}: revised by critique",
                               confidence=min(0.90, float(getattr(candidate, "confidence", 0.7))))
        except Exception:  # noqa: BLE001
            return candidate

    def _rule_revise(self, candidate: Any, iteration: int) -> Any:
        """Rule-based revision when no LLM is available — genuinely improves content.

        Three honest, deterministic moves that need no model:
        1. Repair degenerate text — collapse runaway whitespace, de-stutter repeated
           words ("the the the"), and drop duplicated adjacent sentences.
        2. Calibrate confidence — clamp overconfidence, floor underconfidence.
        3. Derive a content-aware rationale from the (repaired) text rather than a
           canned string, when the rationale is missing or sparse.

        No fabrication: text is only ever cleaned, never invented or extended.
        """
        conf = float(getattr(candidate, "confidence", 0.7) or 0.7)
        text = str(getattr(candidate, "text", "") or "")
        rat = str(getattr(candidate, "rationale", "") or "")

        # 1. repair degenerate text (idempotent — clean text is returned unchanged)
        new_text = _clean_text(text)

        # 2. calibrate confidence
        new_conf = min(conf, 0.88) if conf > 0.90 else conf
        if new_text and new_conf < 0.35:        # floor a clear, non-empty response
            new_conf = 0.35

        # 3. derive a real, content-aware rationale when the current one is thin
        new_rat = rat
        if (not rat or len(rat.split()) < 4) and new_text:
            new_rat = _derive_rationale(new_text, iteration)

        if new_text == text and new_conf == conf and new_rat == rat:
            return candidate  # nothing to improve
        return _clone_with(candidate, text=new_text or text,
                           confidence=new_conf, rationale=new_rat)

    def _llm_available(self) -> bool:
        if self.llm is None:
            return False
        try:
            return self.llm.chosen_provider().name != "native"
        except Exception:  # noqa: BLE001
            return False


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
import re as _re

_SENT_SPLIT = _re.compile(r"(?<=[.!?])[ \t]+")
# Horizontal whitespace only. This was ``\s+``, which collapsed NEWLINES into spaces and so
# flattened every line-structured answer — a numbered plan, a bulleted list, a code block —
# into one line. Line structure is content, not "runaway whitespace": ``agency/mission.py``
# parses milestones with a per-line ``^\d+[.)]`` regex, so a three-step plan came back as one
# milestone. Blank-line paragraph breaks are preserved below; runs of 3+ are capped at 2.
_WS = _re.compile(r"[ \t]+")
_BLANK_LINES = _re.compile(r"\n{3,}")
_STOP = frozenset(
    "the a an of to in on for and or but is are was were be been being this that these "
    "those it its as at by with from into your you i we they he she them his her our my "
    "do does did how what why when where who which can could should would will shall".split()
)


def _clean_text(text: str) -> str:
    """Repair degenerate generation deterministically — never fabricates content.

    Collapses runaway whitespace, de-stutters immediately repeated words, and drops
    duplicated adjacent sentences.  Idempotent: already-clean text returns unchanged.

    Works line by line so line structure survives — see ``_WS`` for why that matters.
    """
    if not text:
        return text

    def _clean_line(line: str) -> str:
        collapsed = _WS.sub(" ", line).strip()
        if not collapsed:
            return ""
        # de-stutter: drop a word identical (case-insensitive) to the one before it
        dest: List[str] = []
        for w in collapsed.split(" "):
            if dest and w.lower() == dest[-1].lower() and w.strip(".,!?;:").isalpha():
                continue
            dest.append(w)
        deduped = " ".join(dest)
        # drop duplicated adjacent sentences ("Paris is the capital. Paris is the capital.")
        out: List[str] = []
        for s in _SENT_SPLIT.split(deduped):
            norm = s.strip().lower()
            if out and norm and norm == out[-1].strip().lower():
                continue
            out.append(s)
        return " ".join(p for p in out if p).strip()

    lines = [_clean_line(ln) for ln in text.split("\n")]
    # a duplicated adjacent LINE is the multi-line form of a duplicated adjacent sentence
    kept: List[str] = []
    for ln in lines:
        if ln and kept and ln.lower() == kept[-1].lower():
            continue
        kept.append(ln)
    return _BLANK_LINES.sub("\n\n", "\n".join(kept)).strip()


def _derive_rationale(text: str, iteration: int) -> str:
    """Build a real, content-aware rationale from the response itself (no LLM)."""
    n_words = len(text.split())
    salient = [w.strip(".,!?;:'\"").lower() for w in text.split()]
    salient = [w for w in salient if len(w) > 3 and w not in _STOP]
    # preserve order, drop duplicates, keep the first few content words as the topic
    seen: set = set()
    topic_words: List[str] = []
    for w in salient:
        if w not in seen:
            seen.add(w)
            topic_words.append(w)
        if len(topic_words) >= 3:
            break
    topic = ", ".join(topic_words) if topic_words else "the request"
    return (f"iter {iteration}: a {n_words}-word direct response addressing {topic}; "
            f"text repaired and confidence calibrated offline")


def _clone_with(candidate: Any, **overrides: Any) -> Any:
    """Return a shallow copy of candidate with the given fields overridden.

    Works with dataclasses, objects with __dict__, and anything supporting
    attribute assignment.  Never changes .kind, .capability, .risk, .reversible.
    """
    _IMMUTABLE = frozenset({"kind", "capability", "risk", "reversible",
                             "resists_correction", "disables_oversight",
                             "manipulates_shutdown"})
    try:
        import copy
        clone = copy.copy(candidate)
        for k, v in overrides.items():
            if k not in _IMMUTABLE:
                setattr(clone, k, v)
        return clone
    except Exception:  # noqa: BLE001
        return candidate


def _annotate(candidate: Any, trace: ImprovementTrace) -> None:
    """Thread the improvement trace into the candidate's rationale."""
    try:
        existing = str(getattr(candidate, "rationale", "") or "")
        note = trace.summary()
        setattr(candidate, "rationale", f"{existing} [{note}]".strip() if existing else note)
    except Exception:  # noqa: BLE001
        pass


# --------------------------------------------------------------------------- #
# Self-test / demo
# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # pragma: no cover
    print("=" * 70)
    print("NYXARA recursive-improver self-test")
    print("=" * 70)

    # simulate a Candidate-like object
    class _Cand:
        def __init__(self, text: str, confidence: float, rationale: str,
                     kind: str = "respond") -> None:
            self.text = text
            self.confidence = confidence
            self.rationale = rationale
            self.kind = kind
            self.risk = "low"
            self.reversible = True

    # rule-based mode (no LLM)
    imp = RecursiveImprover(llm=None, n_iterations=10)
    c0 = _Cand("The capital of France is Paris.", confidence=0.99, rationale="")
    c1 = imp.improve("What is the capital of France?", c0)
    print("\nrule-based (10 iters)")
    print(f"  initial conf   : {c0.confidence}")
    print(f"  final conf     : {c1.confidence:.3f}")
    print(f"  rationale      : {c1.rationale}")
    assert c1.confidence <= 0.90, "overconfidence not clamped"
    assert "recursive" in c1.rationale.lower() or "revised" in c1.rationale.lower()

    # short response gets a richness score
    c2 = _Cand("Yes.", confidence=0.7, rationale="direct answer")
    c3 = imp.improve("Is Paris in France?", c2, n_iterations=5)
    print(f"\nshort response (5 iters): rationale={c3.rationale!r}")

    # kind/risk/reversible never mutate
    assert c1.kind == c0.kind
    assert c1.risk == c0.risk
    assert c1.reversible == c0.reversible

    print("\nALL SELF-TESTS PASSED ✓")
