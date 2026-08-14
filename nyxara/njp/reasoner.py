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

from typing import Any, Callable, Optional

__all__ = ["NJPReasoner"]


def _clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else float(x))


class NJPReasoner:
    """The outermost reason-seat. Supplies content; never touches the safety envelope."""

    def __init__(self, base: Callable[..., Any], brain: Any, *,
                 may_propose_tools: bool = True, min_confidence: float = 0.25,
                 novelty_damping: float = 0.75) -> None:
        self.base = base
        self.brain = brain
        self.may_propose_tools = bool(may_propose_tools)
        self.min_confidence = float(min_confidence)
        self.novelty_damping = float(novelty_damping)
        self.turns = 0
        self.took_over = 0
        self.annotated = 0
        self.tempered = 0
        self.proposed_tools = 0
        self.vetoed = 0

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
            if thought is None:
                return candidate

            # Having nothing of her own to say is NOT a reason to leave the base's number alone —
            # it is the strongest reason to discount it. An early return here would skip tempering
            # in exactly the case it exists for: a cold or surprised brain, where the base's
            # confident-sounding reply is least likely to be checkable.
            if not str(getattr(thought, "answer", "") or "").strip():
                self.annotated += 1
                return self._temper(candidate, thought)

            # An unestablished claim annotates but never overwrites — and tempers the confidence.
            if not thought.assertable:
                self.annotated += 1
                return self._temper(candidate, thought)

            # Her own record can veto a takeover outright. A faculty this answer rests on that she
            # is *measurably* poor at means the claim is not hers to state as fact, however
            # confident the gauntlet's number looks — the number describes the claim, not the
            # organ that produced it. This is the self-model with teeth: it stops being a report
            # and starts deciding what she is entitled to say.
            weak = self._weak_faculties(thought)
            if weak:
                self.vetoed += 1
                return self._temper(candidate, thought, weak=weak)

            base_conf = float(getattr(candidate, "confidence", 0.0) or 0.0)
            if thought.confidence < max(self.min_confidence, base_conf):
                self.annotated += 1
                return self._temper(candidate, thought)

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

    def _weak_faculties(self, thought: Any) -> list:
        """Faculties this answer rests on that her own counts say are unreliable."""
        try:
            model = getattr(self.brain, "self_model", None)
            if model is None:
                return []
            kind = str(getattr(getattr(thought, "intent", None), "kind", "") or "question")
            return list(model.unreliable_for(kind))
        except Exception:  # noqa: BLE001
            return []

    def _temper(self, candidate: Any, thought: Any, *, weak: Optional[list] = None) -> Any:
        """Discount a confident-looking reply by how unfamiliar the turn actually was to her.

        This is the seat's most important behaviour and it only ever moves confidence **down**.
        The base reasoner reports how sure *it* is; it has no way to know whether NJP recognised
        the situation at all. On a turn the fabric could not anticipate — a cold brain, or simply
        something it has never met — a canned, fluent-sounding reply arrives carrying the base's
        full confidence, and downstream honesty gates (``AgentLoop._gate_answer``) let it through
        because the number looks fine. That is exactly the confident-but-wrong answer those gates
        exist to catch.

        So: ``reported = base × (1 − novelty × damping)``. Novelty is 1.0 when the manifold had no
        trusted prediction for the turn, and falls as she comes to recognise it, so the discount
        disappears on familiar ground rather than permanently muting her. Raising is impossible
        here by construction — the factor is never above 1.
        """
        try:
            percept = getattr(thought, "percept", None)
            if weak:
                note = (self._rationale(thought, took_over=False)
                        + f"; not asserted — measurably unreliable at: {', '.join(weak)}")
                if percept is None:
                    return self._set(candidate, rationale=note)
            base_conf = float(getattr(candidate, "confidence", 0.0) or 0.0)
            factor = max(0.0, 1.0 - float(percept.novelty) * self.novelty_damping)
            tempered = base_conf * factor
            if weak:
                # A weak faculty caps the reported confidence as well as blocking the takeover:
                # letting the base's number stand would leak the same over-confidence downstream
                # through a different field.
                self.tempered += 1
                return self._set(candidate, confidence=min(tempered, 0.4),
                                 rationale=(self._rationale(thought, took_over=False,
                                                            novelty=percept.novelty)
                                            + f"; unreliable at: {', '.join(weak)}"))
            if tempered < base_conf:
                self.tempered += 1
                return self._set(candidate, confidence=tempered,
                                 rationale=self._rationale(thought, took_over=False,
                                                           novelty=percept.novelty))
            return self._set(candidate, rationale=self._rationale(thought, took_over=False))
        except Exception:  # noqa: BLE001 — a failed temper leaves the base's number alone
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
    def _rationale(thought: Any, *, took_over: bool,
                   novelty: Optional[float] = None) -> str:
        """Honest, short, and names the verdict rather than implying a stronger one."""
        try:
            judgement = getattr(thought, "judgement", None)
            verdict = getattr(judgement, "verdict", "unjudged") if judgement else "unjudged"
            verb = "answered" if took_over else "concurred"
            note = (f"[NJP V.01] {verb} ({verdict}, confidence {thought.confidence:.2f})")
            if novelty is not None:
                # Say that the number was moved, and why — a silently discounted confidence is
                # the same opacity this seat exists to remove.
                note += f"; confidence discounted, novelty {novelty:.2f}"
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
                "annotated": self.annotated, "tempered": self.tempered,
                "vetoed_by_self_model": self.vetoed,
                "proposed_tools": self.proposed_tools,
                "takeover_rate": round(self.took_over / self.turns, 4) if self.turns else None}
