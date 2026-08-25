"""NYXARA · njp/economy.py — required computation, not maximum computation (⏱, NJP V.10).

**The rule this module enforces.** Every organ in this package is written to be worth running, and
none of them asks whether *this turn* is worth running it on. That is fine while the organs are
microseconds of arithmetic and stops being fine the moment a 9B model is attached to one of them.

**Where the cost actually sits, measured rather than assumed.** The obvious suspect is reasoning,
and reasoning is already governed: :class:`~nyxara.njp.relevance.CognitivePolicy` forbids the
world-knowledge pathways to a social act, and :class:`~nyxara.njp.router.Router` reports
``cortex_permitted=False`` and ``extraction_permitted=False`` for every one of
``greeting``/``thanks``/``farewell``/``social_checkin``. A greeting cannot reach the cortex, and
could not before this module existed.

The **rendering** surface had no such gate. :meth:`nyxara.njp.voice.Dialogue.respond` calls
:meth:`~nyxara.njp.voice.Dialogue._render` for every turn that produced content, so
``"Hii Master 👋"`` — a complete natural sentence her own organs wrote — was being sent to a
language model with the instruction *"rewrite this as one natural, direct reply"*. On this machine
an NJP turn is 2–12 ms end to end; a 9B rendering call is seconds. The expensive half of a
greeting was never the thinking.

**Why the gate is a rule and not a list of acts.** "Skip the model for greetings" is a special
case that has to be extended every time a new act appears. The general statement is about what
rendering *is for*: the renderer's own contract is that it phrases content it is given and may add
nothing to it, and its output is discarded outright when
:meth:`~nyxara.njp.voice.Dialogue._faithful` finds it drifted. So a call is worth making only when
the content is not already the sentence she would say — and content that is already a complete
natural sentence, written by her own social path, has nothing left for a renderer to do. The model
can only reword it, and a reword that survives the faithfulness check says the same thing while a
reword that fails is discarded. Either way the call bought nothing and cost seconds.

That is the whole heuristic, and it is stated as a property of the *content* rather than of the
act, so an act nobody has invented yet is governed by it too:

    render when the content is a **conclusion** — a fact, a claim, a derivation — and not when it
    is **already an utterance**.

**Budgets are advisory, never a second gate on truth.** :class:`Budget` says how much machinery a
turn is worth. It cannot permit anything the sovereign gate, the router, or the cognitive policy
refuses — every consumer reads it with ``and``, never with ``or`` — and nothing here touches what
is true, what may be stated, or what confidence attaches to it. A cheap turn and an expensive turn
are answered with the same honesty; they are answered with different amounts of hardware.

Pure standard library. No LLM. Fail-soft: on any error the budget returned is the permissive one,
because a broken accountant must not be able to silence her.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

__all__ = ["Tier", "Budget", "CognitiveEconomy", "budget_for"]


class Tier:
    """How much machinery a turn is worth. Ordered, and comparable through :data:`TIER_ORDER`."""

    TRIVIAL = "trivial"      # a greeting; answered from state, rendered by nobody
    LOW = "low"              # a stated fact, a lookup
    MEDIUM = "medium"        # an explanation, a derivation
    HIGH = "high"            # a causal or counterfactual question
    MAXIMUM = "maximum"      # a problem she has no strategy for


TIER_ORDER: Tuple[str, ...] = (Tier.TRIVIAL, Tier.LOW, Tier.MEDIUM, Tier.HIGH, Tier.MAXIMUM)

#: The acts that are answered *from the relationship and her own state*, which is exactly the set
#: whose replies her social path writes as finished sentences. Kept as a set of act kinds because
#: that is what the speech-act reader produces; the *rule* that uses it is about the content.
_SOCIAL_ACTS = frozenset({"greeting", "thanks", "farewell", "social_checkin", "state_query"})

#: Acts that genuinely justify the expensive machinery, whatever their surface length.
_DEEP_ACTS = frozenset({"causal_query", "task"})

#: A finished utterance ends the way a sentence ends, or the way she ends one.
_UTTERANCE_END = re.compile(r"[.!?…।]\s*$|[\U0001F300-\U0001FAFF]\s*$")


@dataclass
class Budget:
    """What this turn has earned. Advisory: every consumer reads it with ``and``, never ``or``."""

    tier: str = Tier.MEDIUM
    #: May the language surface be asked to phrase this? ``False`` does not mean silence — it
    #: means she says the content in her own words, which for an already-finished sentence is
    #: the identical string.
    render: bool = True
    #: How long a rendering may run for, when one is permitted. Scaled to the tier so a lookup
    #: cannot spend an essay's worth of tokens.
    max_tokens: int = 220
    #: May the slow, optional organs run this turn? Advisory only — the router still decides, and
    #: a ``True`` here never overrides a ``False`` there.
    deliberate: bool = True
    why: str = ""

    @property
    def rank(self) -> int:
        try:
            return TIER_ORDER.index(self.tier)
        except ValueError:
            return TIER_ORDER.index(Tier.MEDIUM)

    def at_least(self, tier: str) -> bool:
        try:
            return self.rank >= TIER_ORDER.index(tier)
        except ValueError:
            return True

    def to_dict(self) -> Dict[str, Any]:
        return {"tier": self.tier, "render": self.render, "max_tokens": self.max_tokens,
                "deliberate": self.deliberate, "why": self.why}


class CognitiveEconomy:
    """Decides what a turn is worth, and counts what that decision saved.

    The counters are the point. A budget layer that cannot say how often it declined is a claim
    about efficiency rather than a measurement of one, and this repo does not accept those.
    """

    def __init__(self, *, max_tokens: int = 220) -> None:
        self.max_tokens = max(16, int(max_tokens))
        self.turns = 0
        #: Renderings declined because the content was already an utterance. This is the number
        #: that says whether the module earned its place.
        self.renders_declined = 0
        self.renders_permitted = 0
        self.by_tier: Dict[str, int] = {tier: 0 for tier in TIER_ORDER}

    def assess(self, *, content: str = "", act: Any = None,
               thought: Any = None) -> Budget:
        """What this turn is worth. Never raises; on any error the permissive budget is returned."""
        try:
            return self._assess(content=content, act=act, thought=thought)
        except Exception:  # noqa: BLE001 — a broken accountant may not silence her
            return Budget(why="budget unavailable; nothing withheld")

    # ---- the decision ----------------------------------------------------- #
    def _assess(self, *, content: str, act: Any, thought: Any) -> Budget:
        self.turns += 1
        kind = str(getattr(act, "kind", "") or "")
        text = str(content or "").strip()
        out = Budget(why="")

        out.tier = self._tier(kind, text, thought)
        self.by_tier[out.tier] = self.by_tier.get(out.tier, 0) + 1

        # Tokens scale with the tier. A lookup answered in one word does not get an essay's
        # budget, and the cap is what stops a cheap turn becoming an expensive one by accident.
        out.max_tokens = {
            Tier.TRIVIAL: 32, Tier.LOW: 96, Tier.MEDIUM: self.max_tokens,
            Tier.HIGH: self.max_tokens, Tier.MAXIMUM: self.max_tokens,
        }.get(out.tier, self.max_tokens)

        # The rule, stated on the content rather than the act — see the module docstring.
        already_said = self._is_utterance(text, kind)
        out.render = not already_said
        if already_said:
            self.renders_declined += 1
            out.why = ("the content is already the sentence she would say; a renderer may only "
                       "reword it, and a reword that drifts is discarded anyway")
        else:
            self.renders_permitted += 1
            out.why = "the content is a conclusion, and a conclusion still has to be phrased"

        # Advisory, and read with `and` by whoever consults it. A trivial turn has nothing for
        # the slow organs to work on; everything else may deliberate if its router agrees.
        out.deliberate = out.tier != Tier.TRIVIAL
        return out

    @staticmethod
    def _tier(kind: str, text: str, thought: Any) -> str:
        """Which tier this turn falls in, hardest signal first."""
        if kind in _SOCIAL_ACTS:
            return Tier.TRIVIAL
        if kind in _DEEP_ACTS:
            return Tier.HIGH
        # She has no answer and no strategy — the one case that earns everything she has. Read
        # from the thought rather than the text, because "I don't know" is a state, not a phrasing.
        if thought is not None:
            epistemic = str(getattr(thought, "epistemic", "") or "").lower()
            if epistemic == "unknown" and not text:
                return Tier.MAXIMUM
            if epistemic in ("conflicting", "contested"):
                return Tier.HIGH
        if not text:
            return Tier.LOW
        # A one- or two-word answer is a lookup however it was reached; a paragraph is an
        # explanation. Length of the *answer*, not of the question, because the question's length
        # says what was asked and the answer's says what it took.
        return Tier.LOW if len(text.split()) <= 3 else Tier.MEDIUM

    @staticmethod
    def _is_utterance(text: str, kind: str) -> bool:
        """Is this content already the sentence she would say, rather than a conclusion to phrase?

        Three signals, and all of them are about the string:

        * a **social act's** reply is written by her own social path as a finished sentence —
          this is the case the module was built for and it is checked first;
        * anything ending in sentence punctuation or an emoji is a finished utterance whoever
          wrote it;
        * a bare fact — one or two words, no terminator — is emphatically *not* one, and is
          exactly what rendering exists for: "water" becomes "A sparrow needs water."
        """
        if not text:
            return False
        if kind in _SOCIAL_ACTS:
            return True
        return bool(_UTTERANCE_END.search(text)) and len(text.split()) >= 4

    # ---- what it saved ---------------------------------------------------- #
    def stats(self) -> Dict[str, Any]:
        total = self.renders_declined + self.renders_permitted
        return {
            "turns": self.turns,
            "renders_permitted": self.renders_permitted,
            "renders_declined": self.renders_declined,
            # The fraction of turns on which the language surface was not asked to do work that
            # could not have changed the reply. Zero is a real answer and means this module is
            # not earning its place on this workload.
            "declined_fraction": round(self.renders_declined / total, 4) if total else 0.0,
            "by_tier": dict(self.by_tier),
        }


#: A process-wide economy for callers that have no reason to hold one. Constructing a fresh
#: :class:`CognitiveEconomy` per call would work and would lose the counters, which are the only
#: evidence that any of this helps.
_DEFAULT = CognitiveEconomy()


def budget_for(content: str = "", *, act: Any = None, thought: Any = None,
               economy: Optional[CognitiveEconomy] = None) -> Budget:
    """What this turn is worth, against ``economy`` or the process-wide one."""
    return (economy or _DEFAULT).assess(content=content, act=act, thought=thought)
