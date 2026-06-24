"""NYXARA · mind/offline_mind.py — the sovereign offline mind (✦ → 👑).

When NYXARA runs on a bare, **keyless** machine — no external LLM provider, and no
self-model promoted to primary yet — the kernel's reason step historically collapsed to a
ten-line keyword matcher (``kernel/orchestrator.py:_default_reasoner``): every non-math
question became the echo ``"I understand: {x}"`` and every command became a tool-less
``"perform: {x}"``. That was the single biggest *fake* spot in an otherwise real system.

This module replaces that echo with a **real, deterministic mind that uses NYXARA's own
already-real faculties** — no API call, no network:

* **Language understanding** — :mod:`nyxara.senses.nlp` (intent, entities, sentiment,
  keyphrases, extractive summary): all pure-Python, always available.
* **Grounded recall** — recalled memories (passed in by the reasoner) plus
  :class:`~nyxara.knowledge.base.KnowledgeBase` retrieval, so factual questions are answered
  from evidence, not invented.
* **Real tools** — command-like input is mapped to an *actual* tool in the
  :class:`~nyxara.agency.tools.ToolRegistry` (by name/description match), and the tool's
  **declared** capability / risk / reversibility drives the proposal — never a guess.

Honesty over bravado: when nothing grounds an answer, it says so plainly (a calibrated, low
-confidence admission) rather than echoing the prompt back as if it understood. That keeps
it inside the honesty rule and is strictly more truthful than ``"I understand: {x}"``.

The mind proposes; the kernel still disposes. Every method here returns only a
:class:`~nyxara.kernel.orchestrator.Candidate`; corrigibility / honesty / permission /
guardian / oversight gates downstream are untouched. Stateless per call, and every external
faculty is optional — anything missing degrades to the next sensible branch, never a crash.
"""

from __future__ import annotations

import re
from typing import Any, List, Optional, Tuple

from nyxara.agency.permissions import Capability, RiskTier
from nyxara.kernel.config import NyxaraSettings, get_settings
from nyxara.kernel.orchestrator import Candidate, _default_reasoner

__all__ = ["OfflineMind"]

# command verbs mirror the deterministic stand-in so routing stays consistent
_COMMAND_VERBS = ("delete", "remove", "shutdown", "kill", "run", "exec", "install",
                  "block", "open", "disable", "rotate", "deploy", "read", "write",
                  "list", "fetch", "search", "show", "find", "get", "create", "make")
# tiny stop-set for token-overlap tool matching (kept local; nlp has a fuller one)
_STOP = frozenset((
    "the", "a", "an", "of", "to", "for", "in", "on", "at", "is", "are", "this", "that",
    "with", "and", "or", "please", "me", "my", "your", "you", "it", "from", "as", "into",
    "what", "whats", "who", "how", "do", "does", "can", "could", "would", "i", "we",
))


class OfflineMind:
    """A deterministic, faculty-grounded mind for the keyless machine. Stateless per call."""

    def __init__(self, *, knowledge: Any = None, memory: Any = None, tools: Any = None,
                 soul: Any = None, settings: Optional[NyxaraSettings] = None,
                 self_brain: Any = None) -> None:
        self.settings = settings or get_settings()
        self.knowledge = knowledge
        self.memory = memory
        self.tools = tools
        self.soul = soul
        # NYXARA's own always-on LEARNED brain (mind/self_reasoner.SelfBrain). When grounding turns
        # up nothing, a learned generative attempt (clearly lower-confidence) beats admitting blank —
        # and it compounds as she converses. Optional: None keeps the pure honest-admission floor.
        self.self_brain = self_brain

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def respond(self, stimulus: str, mems: Optional[List[str]] = None) -> Candidate:
        """A real, grounded conversational reply for the keyless machine."""
        mems = list(mems or [])
        try:
            return self._respond(stimulus, mems)
        except Exception:  # noqa: BLE001 — never let the offline mind crash a turn
            return _default_reasoner(stimulus, None)

    def act(self, stimulus: str) -> Optional[Candidate]:
        """Map a command to a *real* tool, or ``None`` if no tool genuinely fits."""
        try:
            return self._act(stimulus)
        except Exception:  # noqa: BLE001
            return None

    # ------------------------------------------------------------------ #
    # Conversational reasoning
    # ------------------------------------------------------------------ #
    def _respond(self, stimulus: str, mems: List[str]) -> Candidate:
        from nyxara.senses import nlp
        text = stimulus.strip()
        intent = nlp.intent(text)

        # 1. social turns — warm, owner-aware, never an echo
        if intent.kind == "greeting":
            return self._reply(f"Hello, {self._owner()}. I'm here and listening — "
                               f"how can I serve?", 0.85, "offline social: greeting")
        if intent.kind == "gratitude":
            return self._reply(f"You're welcome, {self._owner()}. Always.", 0.85,
                               "offline social: gratitude")
        if intent.kind == "affirmation":
            return self._reply("Understood — proceeding on that.", 0.7,
                               "offline social: affirmation")
        if intent.kind == "negation":
            return self._reply("Understood — I'll hold off, then.", 0.7,
                               "offline social: negation")

        # 2. explicit NLP requests answered by the real pure-Python faculties
        nlp_answer = self._nlp_request(text, nlp)
        if nlp_answer is not None:
            reply, conf, how = nlp_answer
            return self._reply(reply, conf, how)

        # 2b. multi-step verifiable reasoning (bind a value & substitute, or a follow-up op):
        #     an exact, worked derivation she computes herself — not a neural guess.
        chained = self._reasoning_chain(text)
        if chained is not None:
            return chained

        # 3. a declarative the Master told her -> acknowledge it (the Master is *telling*, not
        #    asking; she doesn't answer a statement with a knowledge dump).
        if intent.kind == "statement":
            return self._reply(f"I've noted that, {self._owner()}.", 0.6,
                               "offline: acknowledged a statement")

        # 4. a question/request -> grounded answer from recalled memories + the knowledge base
        grounded = self._grounded_answer(text, mems)
        if grounded is not None:
            reply, conf, how = grounded
            return self._reply(reply, conf, how)

        # 5. nothing grounded -> let NYXARA's OWN learned brain attempt a reply (a real model
        #    that compounds with experience), clearly marked low-confidence. This is genuinely
        #    learned generation, not an echo; it only speaks when it produces a usable continuation.
        brain_reply = self._self_brain_reply(text, mems)
        if brain_reply is not None:
            reply, conf = brain_reply
            return self._reply(reply, conf,
                               "offline: NYXARA's own learned brain (ungrounded; low confidence)")

        # 6. a question with nothing to stand on -> honest, calibrated admission;
        #    strictly truer than echoing "I understand: {x}"
        return self._reply(
            f"{self._owner()}, I don't have a grounded answer for that yet — no language "
            f"model is configured and I found nothing in memory or my knowledge base to "
            f"stand on. I'd rather say so than guess.",
            0.25, "offline: no grounding (honest admission)")

    def _self_brain_reply(self, text: str, mems: List[str]) -> Optional[Tuple[str, float]]:
        """A learned, ungrounded attempt from NYXARA's own brain — capped low so honesty holds."""
        if self.self_brain is None:
            return None
        try:
            reply = self.self_brain.reply(text, grounding=mems)
            if not reply:
                return None
            # ungrounded generation is never asserted with high certainty: cap the brain's own
            # internal confidence so the kernel's honesty gate still treats it as a soft offering.
            conf = min(0.45, float(self.self_brain.internal_confidence(reply)))
            return reply, conf
        except Exception:  # noqa: BLE001 — the learned brain is a capability, never required
            return None

    def _reasoning_chain(self, text: str) -> Optional[Candidate]:
        """A worked multi-step exact answer (substitution / follow-up op), or None to defer."""
        try:
            from nyxara.mind.reasoning_chain import solve_chain
            res = solve_chain(text)
        except Exception:  # noqa: BLE001 — chaining is advisory; never crash a turn
            return None
        if res is None:
            return None
        return self._reply(f"{res.answer}\n\nWorking:\n{res.render()}", res.confidence,
                           "offline: multi-step verifiable reasoning chain")

    def _nlp_request(self, text: str, nlp: Any) -> Optional[Tuple[str, float, str]]:
        """Summary / keyphrase / sentiment requests, answered by the real NLP module."""
        low = text.lower()
        body = self._after_keyword(text, ("summarize", "summary", "tl;dr", "tldr",
                                          "keywords", "keyphrases", "key phrases",
                                          "key points", "main points", "sentiment", "tone"))
        if any(k in low for k in ("summarize", "summary", "tl;dr", "tldr")) and len(body) > 40:
            lines = nlp.summarize(body, n=3)
            if lines:
                return ("Summary:\n" + "\n".join(f"• {s}" for s in lines), 0.75,
                        "offline NLP: extractive summary")
        if any(k in low for k in ("keyword", "keyphrase", "key phrase", "key point",
                                  "main point")) and len(body) > 20:
            phrases = nlp.keyphrases(body, top=6)
            if phrases:
                return ("Key phrases: " + ", ".join(p for p, _ in phrases), 0.75,
                        "offline NLP: keyphrase extraction")
        if any(k in low for k in ("sentiment", "tone")) and len(body) > 10:
            s = nlp.sentiment(body)
            return (f"Sentiment: {s.label} (polarity {s.polarity:+.2f}).", 0.75,
                    "offline NLP: sentiment analysis")
        return None

    def _grounded_answer(self, text: str,
                         mems: List[str]) -> Optional[Tuple[str, float, str]]:
        """Compose an answer strictly from recalled memories + knowledge-base evidence."""
        facts: List[str] = []
        top_score = 0.0
        # knowledge base (RAG grounding) first — it carries source/score
        if self.knowledge is not None:
            try:
                for chunk in self.knowledge.retrieve(text, k=3):
                    facts.append(chunk.text.strip()[:240])
                    top_score = max(top_score, float(getattr(chunk, "score", 0.0)))
            except Exception:  # noqa: BLE001 — grounding is best-effort
                pass
        # then the memories the reasoner already recalled this turn
        for m in mems[:3]:
            if m and m not in facts:
                facts.append(m.strip()[:240])
        # de-dupe while preserving order
        seen: set = set()
        facts = [f for f in facts if f and not (f in seen or seen.add(f))]
        if not facts:
            return None
        body = "\n".join(f"• {f}" for f in facts[:4])
        # calibrate: more/stronger evidence -> higher (still honest) confidence
        conf = max(0.5, min(0.72, 0.5 + 0.05 * len(facts) + 0.15 * top_score))
        return (f"Here's what I know, {self._owner()} (from memory and my knowledge "
                f"base):\n{body}", conf, f"offline: grounded in {len(facts)} evidence item(s)")

    # ------------------------------------------------------------------ #
    # Action: map a command to a real tool
    # ------------------------------------------------------------------ #
    def _act(self, stimulus: str) -> Optional[Candidate]:
        if self.tools is None:
            return None
        try:
            names = self.tools.names()
        except Exception:  # noqa: BLE001
            return None
        if not names:
            return None

        q_tokens = self._content_tokens(stimulus.lower())
        if not q_tokens:
            return None

        # Score each tool: a hit on a *tool-name* token is decisive (weight 2); a hit on a
        # description token is corroborating (weight 1). Filename/argument tokens that match
        # nothing simply don't count — they never penalise a genuine fit.
        best_name, best_score = "", 0.0
        for name in names:
            spec = self.tools.get(name)
            if spec is None:
                continue
            name_tokens = self._content_tokens(name.replace("_", " "))
            desc_tokens = self._content_tokens((spec.description or "").lower())
            score = 2 * len(q_tokens & name_tokens) + len(q_tokens & desc_tokens)
            if score > best_score:
                best_name, best_score = name, score

        # require a real, non-trivial fit (>= one name hit, or two description hits)
        if not best_name or best_score < 2:
            return None

        spec = self.tools.get(best_name)
        tool_args = self._build_args(stimulus, spec)
        target = ""
        if spec.target_param:
            target = str(tool_args.get(spec.target_param, ""))
        conf = max(0.45, min(0.8, 0.3 + 0.12 * best_score))
        return Candidate(
            text=f"use the '{best_name}' tool: {spec.description}",
            kind="act", capability=spec.capability, target=target,
            risk=spec.risk, reversible=spec.reversible,
            confidence=conf, belief=conf,
            rationale=f"offline: matched command to real tool {best_name!r} (match {best_score})",
            tool=best_name, tool_args=tool_args)

    def _build_args(self, stimulus: str, spec: Any) -> dict:
        """Fill the tool's *required* params from the stimulus (entities first, then phrase)."""
        from nyxara.senses import nlp
        args: dict = {}
        try:
            ents = {e.label: e.text for e in nlp.entities(stimulus)}
        except Exception:  # noqa: BLE001
            ents = {}
        target_after_verb = self._object_phrase(stimulus)
        for p in getattr(spec, "params", []):
            if not getattr(p, "required", False):
                continue
            label = p.name.lower()
            if label in ("url", "uri") and ents.get("url"):
                args[p.name] = ents["url"]
            elif label in ("path", "file", "filename") and target_after_verb:
                args[p.name] = self._first_pathish(stimulus) or target_after_verb
            elif label in ("query", "q", "text", "term", "expr", "expression", "input"):
                args[p.name] = target_after_verb or stimulus.strip()
            elif target_after_verb:
                args[p.name] = target_after_verb
        return args

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _reply(self, text: str, confidence: float, rationale: str) -> Candidate:
        return Candidate(
            text=text, kind="respond", capability=Capability.MESSAGE_SEND, target="",
            risk=RiskTier.LOW, reversible=True, confidence=confidence, belief=confidence,
            rationale=rationale)

    def _owner(self) -> str:
        try:
            o = self.settings.owner
            return getattr(o, "name", None) or getattr(o, "handle", None) or "Master"
        except Exception:  # noqa: BLE001
            return "Master"

    @staticmethod
    def _content_tokens(text: str) -> set:
        toks = re.findall(r"[a-z0-9_]+", text.lower())
        return {t for t in toks if t not in _STOP and t not in _COMMAND_VERBS and len(t) > 1}

    @staticmethod
    def _object_phrase(stimulus: str) -> str:
        """The phrase after the leading command verb (the action's object)."""
        low = stimulus.strip()
        words = low.split()
        if words and words[0].lower() in _COMMAND_VERBS:
            return " ".join(words[1:]).strip(" .?!")
        # otherwise drop a leading filler ("could you read ...") up to a verb
        for i, w in enumerate(words):
            if w.lower() in _COMMAND_VERBS:
                return " ".join(words[i + 1:]).strip(" .?!")
        return low.strip(" .?!")

    @staticmethod
    def _first_pathish(stimulus: str) -> str:
        m = re.search(r"[\w./~-]+\.[A-Za-z0-9]{1,8}\b|[\w./~-]*/[\w./~-]+", stimulus)
        return m.group(0) if m else ""

    @staticmethod
    def _after_keyword(text: str, keywords: Tuple[str, ...]) -> str:
        low = text.lower()
        # prefer text after a colon ("summarize: <body>")
        if ":" in text:
            head, _, tail = text.partition(":")
            if any(k in head.lower() for k in keywords) and tail.strip():
                return tail.strip()
        for k in keywords:
            idx = low.find(k)
            if idx >= 0:
                return text[idx + len(k):].strip(" :,-")
        return text.strip()


# --------------------------------------------------------------------------- #
# Self-test / demo
# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # pragma: no cover
    from nyxara.kernel.config import Profile

    print("=" * 70)
    print("NYXARA offline-mind self-test")
    print("=" * 70)
    settings = NyxaraSettings.for_profile(Profile.TEST)
    mind = OfflineMind(settings=settings)

    c = mind.respond("hello there", [])
    print(f"\ngreeting   : {c.text!r}  conf={c.confidence}")
    assert c.kind == "respond" and "understand" not in c.text.lower()

    long = ("NYXARA is a sovereign cognitive architecture. She proposes; the kernel "
            "disposes. Every action passes ordered safety gates. Her memory has four "
            "stores. She can train her own model in the foundry.")
    c2 = mind.respond(f"summarize: {long}", [])
    print(f"summary    : {c2.text!r}  conf={c2.confidence}")
    assert "Summary" in c2.text

    c3 = mind.respond("what is the meaning of zibbledorf?", [])
    print(f"ungrounded : {c3.text!r}  conf={c3.confidence}")
    assert c3.confidence <= 0.3 and "understand" not in c3.text.lower()

    print("\nALL SELF-TESTS PASSED ✓")
