"""NYXARA · nyx001/reasoner.py — the NYX V.001 reason-seat (🜂, outermost, still gated).

:class:`NyxV001Reasoner` is a ``Reasoner``-compatible callable — ``(stimulus, focus) -> Candidate``
— that the kernel installs at the *outermost* position of the reason-seat chain when
``nyx001.as_reasoner`` is set (the default). Inside it sit ``NyxReasoner`` (V.01–.03),
``Nyx5Reasoner``, ``NyxaraReasoner`` and the base, each still doing its own work.

It **wraps** rather than replaces. The base produces the candidate's *shape*; NYX V.001 supplies
the *content* when its fused cycle has something better to say. The safety contract is copied
verbatim from :mod:`nyxara.nyx.reasoner`, because that contract is the reason the reason-seat is
safe to occupy at all:

* **may change** — ``text`` (when the fused thought is stronger), ``confidence``, ``belief``,
  ``efe``, and a short honest ``rationale`` naming which brain won and whether the answer was
  verifiable;
* **may propose a tool** — ``tool``/``tool_args``, and *only* when the base left them empty. It may
  say "this turn wants ``git_status``"; it may never rewrite a tool the base already chose, and
  never change what that tool is allowed to do;
* **never touches** — ``risk``, ``reversible``, ``capability``, and the three corrigibility flags.
  Risk is never lowered and a safety flag is never weakened. A proposed tool changes what is being
  *asked for*, never what the gate will let through: the gate reads the tool's own registered
  capability and risk, not anything written here.

Two restraints beyond the base contract, both deliberate:

* **It does not take over an action candidate.** Rewriting the text of something about to run a
  tool changes what the Master is being asked to approve.
* **A contested fusion does not overwrite.** When the three brains disagree sharply
  (:attr:`~nyxara.nyx001.fusion.Fused.contested`), NYX V.001 leaves the base's text alone and only
  annotates the rationale. Low agreement dressed as a confident answer is precisely the failure
  the fusion layer measures; acting on it here would throw that measurement away.

Fail-soft: on any error the base candidate is returned untouched, so the mind always works.
Sovereignty is absolute — NYX V.001 only ever *proposes*, and the returned candidate flows through
the kernel's unchanged, fail-closed gate exactly like any other.
"""

from __future__ import annotations

from typing import Any, Callable

__all__ = ["NyxV001Reasoner"]


def _clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else float(x))


class NyxV001Reasoner:
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
                return self._annotate_only(candidate, stimulus)

            # Hand the base's answer in as V.01-.03's vote rather than letting the brain ask it
            # again. NyxReasoner sits inside this chain and has already deliberated; recomputing
            # would double the per-turn cost AND fuse a second, possibly different answer than the
            # one the chain below actually produced.
            thought = self.brain.think(stimulus, remember=True,
                                       v03_vote=self._vote_from(candidate))
            fused = getattr(thought, "fused", None)
            if fused is None or not thought.answer.strip():
                return candidate

            # A contested fusion annotates but does not overwrite — see the module docstring.
            if thought.contested:
                self.annotated += 1
                return self._set(candidate, rationale=self._rationale(thought, took_over=False))

            base_conf = float(getattr(candidate, "confidence", 0.0) or 0.0)
            take_over = (thought.verifiable
                         or (thought.confidence >= max(self.min_confidence, base_conf)))
            if not take_over:
                self.annotated += 1
                return self._set(candidate, rationale=self._rationale(thought, took_over=False))

            self.took_over += 1
            out = self._set(candidate,
                            text=thought.answer,
                            confidence=_clamp01(thought.confidence),
                            rationale=self._rationale(thought, took_over=True))
            self._maybe_propose_tool(out, thought)
            return out
        except Exception:  # noqa: BLE001 — the seat never breaks a turn; the base stands
            return candidate

    @staticmethod
    def _vote_from(candidate: Any) -> Any:
        """The base candidate, expressed as V.01-.03's vote in the fusion.

        ``verifiable`` is read from the candidate rather than assumed: only V.02's proof core or
        an induced rule that predicted a held-out demonstration sets it, and this seat must not
        promote a plausible answer to a checked one.
        """
        from nyxara.nyx001.fusion import Vote
        return Vote(source="v03",
                    text=str(getattr(candidate, "text", "") or ""),
                    confidence=float(getattr(candidate, "confidence", 0.0) or 0.0),
                    verifiable=bool(getattr(candidate, "verified", False)
                                    or getattr(candidate, "verifiable", False)),
                    rationale="deliberated by the chain below this seat")

    # ---- the only mutations allowed ---- #
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

    def _annotate_only(self, candidate: Any, stimulus: str) -> Any:
        """For action candidates: say what NYX V.001 thinks, change nothing else."""
        try:
            if self.brain is None or self.brain.stack is None:
                return candidate
            learning = self.brain.is_learning()
            note = ("NYX V.001 did not take over an action candidate"
                    + (f"; substrate is learning ({learning})" if learning is not None else ""))
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
                return                                  # the base already chose — do not touch
            brain = self.brain
            v03 = getattr(brain, "v03", None)
            if v03 is None or not hasattr(v03, "act"):
                return
            reach = getattr(thought, "stimulus", "")
            proposal = getattr(v03, "hands", None)
            if proposal is None:
                return
            pick = proposal.route(reach) if hasattr(proposal, "route") else None
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
        """Honest, short, and names the label rather than implying a stronger one."""
        try:
            fused = getattr(thought, "fused", None)
            winner = getattr(fused, "winner", None)
            src = getattr(winner, "source", "?") if winner is not None else "?"
            label = "verified" if thought.verifiable else "plausible"
            agree = getattr(fused, "agreement", 0.0) if fused is not None else 0.0
            if thought.contested:
                return (f"[NYX V.001] the three brains disagree (agreement {agree:.2f}); "
                        f"leaving the base answer and flagging it")
            verb = "answered" if took_over else "concurred"
            return (f"[NYX V.001] {src} {verb} ({label}, confidence {thought.confidence:.2f}, "
                    f"agreement {agree:.2f})")
        except Exception:  # noqa: BLE001
            return "[NYX V.001]"

    def stats(self) -> dict:
        return {"turns": self.turns, "took_over": self.took_over,
                "annotated": self.annotated, "proposed_tools": self.proposed_tools,
                "takeover_rate": round(self.took_over / self.turns, 4) if self.turns else None}
