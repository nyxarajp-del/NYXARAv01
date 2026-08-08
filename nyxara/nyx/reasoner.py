"""NYXARA · nyx/reasoner.py — the NYX V.01 reason-seat (🧠, the pluggable brain).

:class:`NyxReasoner` is a ``Reasoner``-compatible callable — ``(stimulus, focus) -> Candidate`` —
that the kernel installs in the reason seat when ``nyx.as_reasoner`` is set (the default). It
**wraps a base reasoner** rather than replacing it: the base still produces the candidate's
shape — its kind, capability, risk tier, reversibility, and any tool it wants — and NYX supplies
the *content* when her own cycle has something better to say.

What it may change, and what it may never touch, is the whole safety story:

* **may change** — ``text`` (when her cycle produced a stronger answer), ``confidence``,
  ``belief``, ``efe``, and a short honest ``rationale`` naming the specialist that won and
  whether the answer was derived or guessed;
* **never touches** — ``risk``, ``reversible``, ``capability``, ``tool``/``tool_args``, and the
  three corrigibility flags. Risk is never lowered and a safety flag is never weakened.

Sovereignty is preserved absolutely: NYX only ever *proposes*. The returned candidate flows
through the kernel's unchanged, fail-closed gate exactly like any other, and the gate — not this
reasoner — decides what acts. Fail-soft: on any error it returns the base candidate untouched,
so the mind always works.

One deliberate restraint: NYX does not take over a candidate the base reasoner marked as an
*action*. Rewriting the text of something that is about to run a tool would change what the
Master is being asked to approve, and her cycle reasons about claims, not about tool calls.
"""

from __future__ import annotations

from typing import Any, Callable

__all__ = ["NyxReasoner"]


def _clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else float(x))


class NyxReasoner:
    """NYX V.01 in the reason seat — proposes content, never disposes, never relaxes a guard."""

    def __init__(self, *, base: Callable[..., Any], brain: Any,
                 min_confidence: float = 0.25) -> None:
        self.base = base
        self.brain = brain
        self.min_confidence = float(min_confidence)

    def __call__(self, stimulus: str, focus: Any = None, **kwargs: Any) -> Any:
        candidate = self._call_base(stimulus, focus, kwargs)
        try:
            thought = self.brain.think(stimulus)
        except Exception:  # noqa: BLE001 — a failed cycle leaves the base candidate untouched
            return candidate
        try:
            self._offer(candidate, thought)
        except Exception:  # noqa: BLE001 — enrichment is advisory; never break a turn
            pass
        return candidate

    def _call_base(self, stimulus: str, focus: Any, kwargs: dict) -> Any:
        try:
            return self.base(stimulus, focus, **kwargs)
        except TypeError:
            try:
                return self.base(stimulus, focus)   # a two-arg reasoner (the deterministic stand-in)
            except TypeError:
                return self.base(stimulus)          # a one-arg reasoner

    # ---- the proposal ----------------------------------------------------- #
    def _offer(self, candidate: Any, thought: Any) -> None:
        if candidate is None or thought is None:
            return
        answer = str(getattr(thought, "answer", "") or "").strip()
        verified = bool(getattr(thought, "verified", False))
        confidence = _clamp01(float(getattr(thought, "confidence", 0.0)))
        source = str(getattr(getattr(thought, "winner", None), "source", "") or "")

        if self._should_speak(candidate, thought, answer, confidence, verified):
            spoken = self._say(thought, answer)
            if spoken:
                candidate.text = spoken

        self._calibrate(candidate, thought, confidence, verified)
        self._annotate(candidate, source, thought, verified)

    def _should_speak(self, candidate: Any, thought: Any, answer: str, confidence: float,
                      verified: bool) -> bool:
        """Does her cycle have something *better* than what the base already said?

        Taking the reply is not a right, it is earned. The base reasoner convenes real faculties
        of its own and often produces a checked derivation; overwriting that with a weaker
        answer would be a regression dressed as an upgrade.
        """
        if not answer:
            return False
        # An action candidate is not hers to reword — see the module docstring.
        if str(getattr(candidate, "kind", "respond")) != "respond":
            return False
        if getattr(candidate, "tool", ""):
            return False
        # A checked derivation always earns the reply — that is the control law.
        if verified:
            return True
        # Otherwise she needs an answer with *substance*: something recalled or derived, not
        # a list of associations, which is colour rather than a claim.
        source = str(getattr(getattr(thought, "winner", None), "source", "") or "")
        if source not in self._ANSWERING_SOURCES:
            return False
        if confidence < self.min_confidence:
            return False
        # And she must not displace a base answer that already stands on something checked.
        return not self._base_is_verified(candidate)

    # Specialists whose output is an actual answer, rather than commentary about one.
    _ANSWERING_SOURCES = frozenset({"memory", "derivation"})

    @staticmethod
    def _base_is_verified(candidate: Any) -> bool:
        """Did the base reasoner already stand its answer on something it checked?"""
        try:
            rationale = str(getattr(candidate, "rationale", "") or "").lower()
            return "verified=true" in rationale or "proven" in rationale
        except Exception:  # noqa: BLE001
            return False

    def _say(self, thought: Any, answer: str) -> str:
        """Phrase it through the dialogue layer when there is one, so the honesty rules apply."""
        try:
            dialogue = getattr(self.brain, "dialogue", None)
            if dialogue is None:
                return answer
            reply = dialogue.respond(thought, brain=self.brain)
            return str(getattr(reply, "text", "") or "").strip() or answer
        except Exception:  # noqa: BLE001
            return answer

    @staticmethod
    def _calibrate(candidate: Any, thought: Any, confidence: float, verified: bool) -> None:
        """Move confidence toward what her cycle actually measured. Never touches risk."""
        current = getattr(candidate, "confidence", None)
        if isinstance(current, (int, float)):
            if verified:
                candidate.confidence = _clamp01(max(float(current), confidence))
            else:
                # An undecided collapse means several answers are still live — say so by
                # being *less* sure, never more.
                decided = bool(getattr(thought, "decided", True))
                blended = 0.5 * float(current) + 0.5 * confidence
                candidate.confidence = _clamp01(blended * (1.0 if decided else 0.7))

        belief = getattr(candidate, "belief", None)
        if isinstance(belief, (int, float)) and not verified:
            candidate.belief = _clamp01(float(belief) * (1.0 if
                                                         getattr(thought, "decided", True)
                                                         else 0.8))

        if hasattr(candidate, "efe"):
            try:
                # entropy is the honest expected-free-energy proxy: spread belief = more to
                # resolve. Advisory ranking only — never an authorisation.
                candidate.efe = float(getattr(thought, "entropy", 0.0))
            except Exception:  # noqa: BLE001
                pass

    @staticmethod
    def _annotate(candidate: Any, source: str, thought: Any, verified: bool) -> None:
        """Record briefly and honestly that NYX weighed in, and on what basis."""
        rationale = getattr(candidate, "rationale", None)
        if not isinstance(rationale, str):
            return
        verification = getattr(thought, "verification", None)
        verdict = getattr(getattr(verification, "verdict", None), "value", None)
        bits = [f"nyx: {source or 'no specialist'}",
                "derived" if verified else "not derived"]
        if verdict:
            bits.append(f"check={verdict}")
        if not getattr(thought, "decided", True):
            bits.append("undecided")
        note = " (" + ", ".join(bits) + ")"
        candidate.rationale = (rationale + note) if rationale else note.strip()
