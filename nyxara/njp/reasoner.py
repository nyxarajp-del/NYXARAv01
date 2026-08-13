"""NYXARA · njp/reasoner.py — the NJP V.01 reason-seat (🜂, the only one, still gated).

:class:`NJPReasoner` is a ``Reasoner``-compatible callable — ``(stimulus, focus) -> Candidate`` —
that the kernel installs at the outermost position of the reason-seat chain when
``njp.as_reasoner`` is set (the default). Inside it sits ``NyxaraReasoner`` and the base, each
still doing its own work.

It **wraps** rather than replaces. The base produces the candidate's *shape*; NJP supplies the
*content* when its own cycle has something better to say. The safety contract is the one the
reason-seat has always kept, because that contract is the reason the seat is safe to occupy:

* **may change** — ``text`` (only when the claim was **established** by the Truth-Seeking Gauntlet
  or genuinely outscores the base), ``confidence``, ``belief``, ``efe``, and a short honest
  ``rationale`` naming the verdict;
* **may propose a tool** — ``tool``/``tool_args``, and *only* when the base left them empty. It may
  say "this turn wants ``git_status``"; it may never rewrite a tool the base already chose, and
  never change what that tool is allowed to do;
* **never touches** — ``risk``, ``reversible``, ``capability``, and the three corrigibility flags.
  Risk is never lowered and a safety flag is never weakened. A proposed tool changes what is being
  *asked for*, never what the gate will let through: the gate reads the tool's own registered
  capability and risk, not anything written here.

Three restraints beyond that, all deliberate:

* **It does not take over an action candidate.** Rewriting the text of something about to run a
  tool changes what the Master is being asked to approve.
* **An unestablished claim never overwrites.** If the gauntlet returned anything but
  ``established``, the base's text stands and NJP only annotates. This is the anti-hallucination
  rule at the outermost edge: the last thing that touches a reply refuses to state as fact
  something that was not corroborated.
* **A latent want is surfaced, never acted on.** What the Soul-Sync inferred the Master *also*
  wants goes into the rationale so it is visible and can be declined — it does not silently widen
  the work.

Fail-soft: on any error the base candidate is returned untouched, so the mind always works.
Sovereignty is absolute — NJP only ever *proposes*, and the returned candidate flows through the
kernel's unchanged, fail-closed gate exactly like any other.
"""

from __future__ import annotations

from typing import Any, Callable

__all__ = ["NJPReasoner"]


def _clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else float(x))


class NJPReasoner:
    """The outermost reason-seat. Supplies content; never touches the safety envelope."""

    def __init__(self, base: Callable[..., Any], brain: Any, *,
                 may_propose_tools: bool = True, min_confidence: float = 0.25) -> None:
        self.base = base
        self.brain = brain
        self.may_propose_tools = bool(may_propose_tools)
        self.min_confidence = float(min_confidence)
        self.turns = 0
        self.took_over = 0
        self.annotated = 0
        self.proposed_tools = 0

    def __call__(self, stimulus: str, focus: Any = None, *args: Any, **kwargs: Any) -> Any:
        candidate = self.base(stimulus, focus, *args, **kwargs)
        try:
            self.turns += 1
            if self.brain is None:
                return candidate
            # Never take over an action: it would change what the Master is approving.
            if str(getattr(candidate, "kind", "")).lower() in ("action", "tool", "command"):
                return self._annotate_only(candidate)

            thought = self.brain.think(stimulus, remember=True)
            if thought is None or not str(getattr(thought, "answer", "") or "").strip():
                return candidate

            # An unestablished claim annotates but never overwrites.
            if not thought.assertable:
                self.annotated += 1
                return self._set(candidate, rationale=self._rationale(thought, took_over=False))

            base_conf = float(getattr(candidate, "confidence", 0.0) or 0.0)
            if thought.confidence < max(self.min_confidence, base_conf):
                self.annotated += 1
                return self._set(candidate, rationale=self._rationale(thought, took_over=False))

            self.took_over += 1
            out = self._set(candidate, text=thought.answer,
                            confidence=_clamp01(thought.confidence),
                            rationale=self._rationale(thought, took_over=True))
            self._maybe_propose_tool(out, thought)
            return out
        except Exception:  # noqa: BLE001 — the seat never breaks a turn; the base stands
            return candidate

    # ---- the only mutations allowed --------------------------------------- #
    def _set(self, candidate: Any, *, text: str = "", confidence: float = -1.0,
             rationale: str = "") -> Any:
        """Write only the permitted fields. Risk, capability and corrigibility are never touched."""
        try:
            if text:
                setattr(candidate, "text", text)
            if confidence >= 0.0:
                setattr(candidate, "confidence", _clamp01(confidence))
                for field in ("belief", "efe"):
                    if hasattr(candidate, field):
                        setattr(candidate, field, _clamp01(confidence))
            if rationale and hasattr(candidate, "rationale"):
                prior = str(getattr(candidate, "rationale", "") or "")
                setattr(candidate, "rationale", f"{prior} {rationale}".strip()[:600])
        except Exception:  # noqa: BLE001
            pass
        return candidate

    def _annotate_only(self, candidate: Any) -> Any:
        """For action candidates: say what NJP thinks, change nothing else."""
        try:
            fabric = getattr(self.brain, "fabric", None)
            note = "[NJP V.01] did not take over an action candidate"
            if fabric is not None:
                note += f"; fabric {fabric.n_cells} cells / {fabric.n_synapses} synapses"
            self.annotated += 1
            return self._set(candidate, rationale=note)
        except Exception:  # noqa: BLE001
            return candidate

    def _maybe_propose_tool(self, candidate: Any, thought: Any) -> None:
        """Fill an EMPTY tool field only. Never replaces a tool the base chose."""
        if not self.may_propose_tools:
            return
        try:
            if getattr(candidate, "tool", None):
                return                       # the base already chose — do not touch
            registry = getattr(self.brain, "tools", None)
            intent = getattr(thought, "intent", None)
            wants_tool = bool(getattr(intent, "kind", "") == "command")
            if registry is None or not wants_tool:
                return
            router = getattr(registry, "route", None)
            pick = router(thought.stimulus) if callable(router) else None
            name = getattr(pick, "tool", None) if pick is not None else None
            if not name:
                return
            setattr(candidate, "tool", name)
            if hasattr(candidate, "tool_args"):
                setattr(candidate, "tool_args", dict(getattr(pick, "args", {}) or {}))
            self.proposed_tools += 1
        except Exception:  # noqa: BLE001 — a failed proposal leaves the field empty, correctly
            pass

    @staticmethod
    def _rationale(thought: Any, *, took_over: bool) -> str:
        """Honest, short, and names the verdict rather than implying a stronger one."""
        try:
            judgement = getattr(thought, "judgement", None)
            verdict = getattr(judgement, "verdict", "unjudged") if judgement else "unjudged"
            verb = "answered" if took_over else "concurred"
            note = (f"[NJP V.01] {verb} ({verdict}, confidence {thought.confidence:.2f})")
            reading = getattr(thought, "reading", None)
            if reading is not None and getattr(reading, "has_latent", False):
                wants = "; ".join(w.want for w in reading.latent[:2])
                # Surfaced so it can be declined — never acted on silently.
                note += f"; you may also want: {wants}"
            return note
        except Exception:  # noqa: BLE001
            return "[NJP V.01]"

    def stats(self) -> dict:
        return {"turns": self.turns, "took_over": self.took_over,
                "annotated": self.annotated, "proposed_tools": self.proposed_tools,
                "takeover_rate": round(self.took_over / self.turns, 4) if self.turns else None}
