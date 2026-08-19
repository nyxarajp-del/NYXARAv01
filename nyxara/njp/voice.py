"""NYXARA · nyx/dialogue.py — how she actually talks (💬, NYX V.01).

Two failure modes bracket this module, and it exists to avoid both.

**A script.** Canned lines, keyword→reply tables, a persona string pasted onto every turn. It
looks like conversation and is not: the same input always yields the same output, nothing it
says depends on what it knows, and it cannot tell you why it said anything. There is not one
user-facing reply literal in this file, and there must never be one.

**Babble dressed as speech.** The opposite failure, and the live one here. NYXARA's language
ladder is ``litertlm → self → native``; only ``litertlm`` (Gemma-4-E2B, on-device) is genuinely
fluent. Without those weights on disk the floor is a word n-gram, which on a real prompt emits
things like ``'::::::::::::::::'``. Piping that to the Master as an answer would be a lie about
what she can do. So:

    **A degraded language surface is reported, never disguised.**

When no fluent surface exists she still answers — from her own cycle, in plain readable form,
with a one-line note about what is missing and how to fix it. What she cannot do is *sound*
fluent when she is not.

The division of labour when a fluent surface *is* present is the repo's own law (``mind/llm.py``:
"the LLM is a provider, not the driver"):

* **content** comes from NYX's cycle — the winning broadcast, the collapsed superposition, the
  symbolic derivation, the memories actually retrieved;
* **the LLM only renders** that content into natural language;
* the rendering is then **checked back** against the content it was given, so a fluent surface
  cannot smuggle in claims NYX never made.

Multi-turn state comes from :mod:`nyxara.nyx.holomem` — content-addressed, so a thing said
fifty turns ago is as reachable as the last one, and there is no window to fall out of.

Fail-soft throughout: any failure degrades to her own words, never to silence and never to a
crash.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

__all__ = ["Surface", "Reply", "Dialogue"]

# Rungs that can genuinely hold a conversation. Everything else on the ladder is a fallback
# that must be reported rather than spoken through.
FLUENT_PROVIDERS = frozenset({"gguf", "litertlm"})


@dataclass
class Surface:
    """What her voice is actually running on right now — named, and honest about it."""

    provider: str = ""
    fluent: bool = False
    available: List[str] = field(default_factory=list)
    note: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {"provider": self.provider, "fluent": self.fluent,
                "available": self.available, "note": self.note}


@dataclass
class Reply:
    """One turn of conversation, with everything needed to audit it."""

    text: str = ""
    source: str = ""                 # which specialist's content this came from
    asked: bool = False              # True when she asked instead of answering
    rendered: bool = False           # True when a fluent model phrased it
    verified: bool = False           # True when a derivation or her evidence backed it
    confidence: float = 0.0
    decided: bool = True             # False => she is genuinely unsure and says so
    surface: Optional[Surface] = None
    evidence: List[str] = field(default_factory=list)
    why: List[str] = field(default_factory=list)
    # What arrived, as read by nyxara.nyx.lingua: the language it was written in, and its
    # tone. Recorded, never generated — she does not learn to be funny by noticing a joke.
    language: str = ""
    register: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {"text": self.text, "source": self.source, "rendered": self.rendered,
                "verified": self.verified, "confidence": round(self.confidence, 4),
                "decided": self.decided, "evidence": self.evidence, "why": self.why,
                "language": self.language, "register": dict(self.register),
                "asked": self.asked,
                "surface": self.surface.to_dict() if self.surface is not None else None}


class Dialogue:
    """Turns a thought into speech — her content, the model's phrasing, and no pretending."""

    def __init__(self, *, llm: Any = None, require_fluent_surface: bool = True,
                 soul: Any = None, max_tokens: int = 220) -> None:
        self._llm = llm
        self._tried = llm is not None
        self.require_fluent_surface = bool(require_fluent_surface)
        self.soul = soul
        self.max_tokens = max(16, int(max_tokens))

    # ---- the voice ------------------------------------------------------- #
    def _get_llm(self) -> Any:
        if not self._tried:
            self._tried = True
            try:
                from nyxara.mind.llm import LLM
                self._llm = LLM()
            except Exception:  # noqa: BLE001 — no model means she speaks in her own words
                self._llm = None
        return self._llm

    def surface(self) -> Surface:
        """Which rung is live, and whether it can actually hold a conversation."""
        out = Surface()
        llm = self._get_llm()
        if llm is None:
            out.note = "no language model is configured; answering in her own words"
            return out
        try:
            status = llm.provider_status() or {}
            out.available = sorted(name for name, ok in status.items() if ok)
            chosen = llm.chosen_provider()
            out.provider = str(getattr(chosen, "name", "") or "")
            out.fluent = out.provider in FLUENT_PROVIDERS
            if not out.fluent:
                out.note = (
                    f"the fluent on-device brain is not installed (running '{out.provider}', "
                    f"a fallback that cannot hold a conversation); fetch one with: "
                    f"python scripts/fetch_litertlm_model.py  (2.4 GB, her primary)  or  "
                    f"python scripts/fetch_qwythos_model.py  (5.6 GB, the Qwythos teacher)")
            return out
        except Exception:  # noqa: BLE001
            out.note = "the language surface could not be inspected; answering in her own words"
            return out

    # ---- speaking -------------------------------------------------------- #
    def respond(self, thought: Any, *, brain: Any = None) -> Reply:
        """Turn a :class:`~nyxara.nyx.brain.NyxThought` into what she actually says."""
        out = Reply()
        try:
            if thought is None:
                return out
            out.surface = self.surface()
            content = str(getattr(thought, "answer", "") or "").strip()
            out.confidence = float(getattr(thought, "confidence", 0.0))
            out.verified = bool(getattr(thought, "verified", False))
            # Read the fields an NJPThought actually carries. These four used to be addressed at
            # `thought.winner` and `thought.assessment` — a shape from the superseded NYX brains
            # that NJPThought never had, so `getattr(..., None)` silently swallowed every one of
            # them: source was always "", evidence and why were always empty, and `decided` was
            # always True however unsure she really was. Silent because they degrade to falsy
            # rather than raising, which is exactly how a wiring gap survives a test suite.
            judgement = getattr(thought, "judgement", None)
            out.source = str(getattr(judgement, "verdict", "") or "")
            out.evidence = [e.source for e in (getattr(judgement, "evidence", None) or [])
                            if getattr(e, "supports", False)]
            out.why = [f"{e.source}: {e.detail}" for e in (getattr(judgement, "evidence", None) or [])
                       if getattr(e, "detail", "")][:4]
            # "Decided" is a claim about *her*, not about the gauntlet: an unestablished verdict is
            # normal (most turns are), whereas two live readings of the instruction mean she
            # genuinely does not know which question she was asked.
            out.decided = not bool(getattr(getattr(thought, "intent", None), "ambiguous", False))
            self._read_register(out, thought, brain)

            # NYX V.02: when two readings of the instruction are still live, or she was asked
            # to do and not do the same thing, the right output is a question. Guessing which
            # was meant and acting on it is the failure this whole gap was about.
            question = self._clarify(thought, brain)
            if question:
                out.text, out.asked, out.decided = question, True, False
                return out

            if not content:
                out.text = self._nothing_to_say(thought)
                return out

            if out.surface.fluent or not self.require_fluent_surface:
                rendered = self._render(content, thought)
                if rendered:
                    out.text, out.rendered = self._hedge(rendered, thought), True
                    return out

            # No fluent surface (or rendering failed): say it in her own words, plainly.
            out.text = self._hedge(self._plainly(content, out), thought)
            return out
        except Exception:  # noqa: BLE001 — she always says something, never crashes mid-sentence
            out.text = str(getattr(thought, "answer", "") or "")
            return out

    def _render(self, content: str, thought: Any) -> str:
        """Let the model phrase *her* content. It may reword; it may not add claims."""
        llm = self._get_llm()
        if llm is None:
            return ""
        try:
            system = self._voice()
            prompt = (
                "Rewrite the following conclusion as one natural, direct reply.\n"
                "Use only what is given — do not add facts, examples, or caveats of your own.\n"
                f"\nQuestion: {getattr(thought, 'stimulus', '')}\n"
                f"Conclusion: {content}\n\nReply:")
            text = str(llm.generate(prompt, system=system, max_tokens=self.max_tokens) or "").strip()
            if not text or not self._faithful(text, content):
                return ""                 # a rendering that drifted is worse than no rendering
            return text
        except Exception:  # noqa: BLE001
            return ""

    def _voice(self) -> Optional[str]:
        """Her own voice fragment, from identity — not a persona string invented here."""
        try:
            if self.soul is None:
                return None
            # `system_fragment` is a method on VoiceProfile, not a property. Without the call this
            # stringified the bound method — "<bound method VoiceProfile.system_fragment of ...>" —
            # and handed *that* to the model as her system prompt. Tolerate both shapes so a
            # future property does not silently reintroduce the same class of bug.
            fragment = self.soul.voice().system_fragment
            return str(fragment() if callable(fragment) else fragment)
        except Exception:  # noqa: BLE001
            return None

    @staticmethod
    def _faithful(rendered: str, content: str) -> bool:
        """Did the rendering stay about the thing it was given, or wander off?

        A coverage check, not entailment: it catches a model that ignored the conclusion and
        answered from its own priors, which is exactly the smuggling this guards against.
        """
        try:
            from nyxara.njp.tongue import concepts_in
            wanted = set(concepts_in(content))
            if not wanted:
                return True
            got = set(concepts_in(rendered))
            return len(wanted & got) / float(len(wanted)) >= 0.4
        except Exception:  # noqa: BLE001
            return True

    @staticmethod
    def _clarify(thought: Any, brain: Any) -> str:
        """The one thing worth asking before acting — empty when nothing needs asking.

        Only fires on a *contradiction* or a genuinely live second reading. An under-specified
        detail is surfaced with the answer rather than in place of it: stopping to ask about
        every vague word would be its own kind of uselessness.

        And an ambiguity she has already resolved is not an ambiguity. If grounding answered the
        question from structure, the competing surface readings are moot — she knows what was
        asked because she found what it was asking *for*. Without this check, "where do I live"
        came back as *"I can read that two ways: it carries a Hindi imperative ending (85%), or it
        opens with a question word (80%). Which one?"* — a false positive from a suffix rule, asked
        of a Master whose answer she was holding the whole time. Asking a question you can answer
        is worse than not asking one.
        """
        try:
            config = getattr(brain, "config", None)
            if config is not None and not getattr(config, "intent_drives_dialogue", True):
                return ""
            intent = getattr(thought, "intent", None)
            if intent is None:
                return ""
            if not (intent.contradictions or intent.ambiguous):
                return ""
            answer = getattr(getattr(getattr(thought, "percept", None), "grounding", None),
                             "answer", None)
            if answer is not None and getattr(answer, "answered", False):
                return ""
            return str(intent.clarifying_question() or "")
        except Exception:  # noqa: BLE001 — she answers rather than crashing on a parse
            return ""

    @staticmethod
    def _read_register(reply: Reply, thought: Any, brain: Any) -> None:
        """Record how the turn was written — language and tone — from the tongue, if she has one."""
        try:
            # `brain.tongue` — NJPBrain names it that (brain.py:166). Reading `brain.lingua` here
            # meant this returned before it read anything, so every reply came back with no
            # language and no register however clearly the tongue had identified them.
            lingua = getattr(brain, "tongue", None)
            stimulus = str(getattr(thought, "stimulus", "") or "")
            if lingua is None or not stimulus:
                return
            read = lingua.read(stimulus)
            reply.language = "hi-Latn" if read.hinglish else read.primary
            reply.register = read.register.to_dict()
        except Exception:  # noqa: BLE001 — tone is a garnish; it never costs a reply
            return

    @staticmethod
    def _hedge(text: str, thought: Any) -> str:
        """Attach the epistemic caveat, if the claim has not earned the right to go bare.

        Applied **after** rendering, never before. The caveat is framing, not content: folding it
        into the content would make a faithful rendering look unfaithful (a model that answers
        "Jay" has not dropped the claim just because it did not repeat the word "confidence"), and
        it would also hand the caveat to the model as something it is free to reword.

        A ``known`` claim is stated plainly — that is what establishment buys. A ``believed`` one
        always carries its number, because "Jay" and "I think Jay" are different assertions and
        the Master is entitled to know which one he is being given.
        """
        try:
            state = str(getattr(thought, "epistemic", "") or "")
            if not text or state != "believed":
                return text
            confidence = float(getattr(thought, "epistemic_confidence", 0.0) or 0.0)
            if confidence <= 0.0:
                return f"{text}\n(I believe this, but I am not certain.)"
            return f"{text}\n(I believe this — confidence {confidence:.2f}, not established.)"
        except Exception:  # noqa: BLE001 — a missing hedge is worse than a plain reply, but a
            return text                # crash mid-sentence is worse than both

    def _plainly(self, content: str, reply: Reply) -> str:
        """Her own words, with the caveats that are actually true of this turn."""
        parts = [content]
        if not reply.decided:
            parts.append("(I am not settled on this — more than one answer is still live.)")
        surface = reply.surface
        if surface is not None and not surface.fluent and surface.note:
            parts.append(f"[{surface.note}]")
        # She understood the language; she cannot *speak* it. Saying so is the same law that
        # keeps the n-gram floor out of her mouth, applied one level up.
        if reply.language and reply.language not in ("en", "und", ""):
            parts.append(f"[read as {reply.language}; I am answering in English — I have no "
                         f"fluent surface for {reply.language}, and will not fake one.]")
        return "\n".join(parts)

    def _nothing_to_say(self, thought: Any) -> str:
        """Nothing reached awareness. Say that, with what she does have — never invent.

        The ungrounded concepts are read from the percept's own grounding result. This used to
        look for ``percept.grounded``, an attribute NJPPercept does not have, so the list was
        always empty and every silent turn gave the same bare sentence — hiding *which* words she
        had failed to ground, which is the one detail that makes the admission useful.
        """
        percept = getattr(thought, "percept", None)
        grounded = getattr(percept, "grounding", None)
        ungrounded = list(getattr(grounded, "ungrounded", None) or [])
        if ungrounded:
            return ("I do not have anything grounded to say about that yet — "
                    f"these are still just words to me: {', '.join(ungrounded[:6])}.")
        return "I do not have anything to say about that yet."

    # ---- introspection ---------------------------------------------------- #
    def stats(self) -> Dict[str, Any]:
        try:
            return {"surface": self.surface().to_dict(),
                    "require_fluent_surface": self.require_fluent_surface}
        except Exception:  # noqa: BLE001
            return {}
