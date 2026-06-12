"""NYXARA · mind/router.py — the confidence router (🧠⇄👑, Phase 2 of the sovereign brain).

Phase 0 made NYXARA's own model *runnable*; Phase 1 *taught* it from the teacher. This is the
bridge that lets it become **primary** — safely, measurably, reversibly. On each turn the
router asks NYXARA's OWN forged model first, scores that answer with an intrinsic **verifier**,
and only if it clears the configured threshold does she speak it herself (a *handoff*).
Otherwise she consults the external teacher. No gold answer is ever shown to the verifier — it
judges quality the way a real router must, on the answer's own merits — so the benchmark can
honestly measure whether rising handoff preserves accuracy.

This changes no weights and reaches around no gate: a routed reply is still a *proposal* the
kernel disposes through corrigibility / honesty / permission / guardian / oversight. The
router only decides *which mind drafts the reply*, never whether it is allowed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable, Optional

from nyxara.kernel.config import NyxaraSettings, get_settings

__all__ = ["RouterResult", "answer_quality", "default_verifier", "Router"]

Verifier = Callable[[str, str], float]

_WORD = re.compile(r"[A-Za-z0-9']+")


def _tokens(text: str) -> list:
    return _WORD.findall((text or "").lower())


# --------------------------------------------------------------------------- #
# Intrinsic verifier — scores an answer on its own merits (no gold answer)
# --------------------------------------------------------------------------- #
def answer_quality(prompt: str, answer: str, *, min_chars: int = 2) -> float:
    """Estimate the quality of ``answer`` to ``prompt`` in ``[0, 1]`` — intrinsically.

    Combines cheap, deterministic signals a small forged model tends to fail: non-emptiness,
    non-degeneracy (it must not just repeat one token), coherence (real words, not byte
    sludge), and that it is not a bare echo of the question. It cannot know *correctness* —
    no router can without the answer — only whether the text is a plausible, self-standing
    reply worth trusting before the teacher is consulted."""
    ans = (answer or "").strip()
    if len(ans) < max(1, min_chars):
        return 0.0
    words = _tokens(ans)
    if not words:
        return 0.0

    # 1) length — a couple of words is fine; ramps to full by ~8 words
    length = min(1.0, len(words) / 8.0)

    # 2) coherence — fraction of characters that are letters/digits/space (not byte sludge)
    sane = sum(c.isalnum() or c.isspace() for c in ans) / len(ans)

    # 3) non-degeneracy — unique-word ratio gates the whole score: a model stuck repeating one
    #    token is worthless however "coherent" its characters look.
    uniq = len(set(words)) / len(words)

    # 4) echo penalty — a reply that is mostly the prompt's own words is parroting, not answering
    pset = set(_tokens(prompt))
    overlap = (sum(1 for w in words if w in pset) / len(words)) if pset else 0.0
    echo_penalty = max(0.0, overlap - 0.6)        # only punish heavy (>60%) parroting

    base = 0.35 * length + 0.65 * sane
    score = base * uniq - echo_penalty
    return max(0.0, min(1.0, score))


def default_verifier(min_chars: int = 2) -> Verifier:
    """A ready-to-use intrinsic verifier bound to a minimum-length floor."""
    def _verify(prompt: str, answer: str) -> float:
        return answer_quality(prompt, answer, min_chars=min_chars)
    return _verify


# --------------------------------------------------------------------------- #
# Result
# --------------------------------------------------------------------------- #
@dataclass
class RouterResult:
    """The drafted reply plus *who* drafted it and *how sure* the verifier was."""

    text: str
    source: str            # "self" (own model), "teacher" (external LLM), or "none"
    confidence: float      # the verifier's score of the OWN model's answer (0 if not tried)
    handed_off: bool       # True iff NYXARA's own model answered unaided

    def to_dict(self) -> dict:
        return {"text": self.text[:500], "source": self.source,
                "confidence": round(self.confidence, 4), "handed_off": self.handed_off}


# --------------------------------------------------------------------------- #
# The router
# --------------------------------------------------------------------------- #
class Router:
    """Own-model-first, teacher-as-fallback — the measurable wrapper→own-AI switch."""

    def __init__(self, llm: Any = None, *, settings: Optional[NyxaraSettings] = None,
                 verifier: Optional[Verifier] = None, self_provider: Any = None) -> None:
        self.settings = settings or (getattr(llm, "settings", None) or get_settings())
        self.cfg = self.settings.router
        if llm is None:
            from nyxara.mind.llm import LLM  # lazy: only when actually routing
            llm = LLM(settings=self.settings)
        self.llm = llm
        self.verifier = verifier or default_verifier(self.cfg.min_chars)
        if self_provider is None:
            from nyxara.mind.llm import SelfProvider
            self_provider = SelfProvider(self.settings)
        self._self = self_provider

    # ---- availability ---- #
    def self_available(self) -> bool:
        try:
            return bool(self._self.available())
        except Exception:  # noqa: BLE001
            return False

    def _teacher_name(self) -> Optional[str]:
        """The real external teacher to consult — the configured provider, else any real one."""
        try:
            avail = list(self.llm.available_providers())
        except Exception:  # noqa: BLE001
            return None
        pref = self.settings.llm.provider.value
        if pref in avail and pref not in ("self", "mock"):
            return pref
        for name in avail:
            if name not in ("self", "mock"):
                return name
        return None

    def teacher_available(self) -> bool:
        return self._teacher_name() is not None

    # ---- generation ---- #
    def _own_answer(self, prompt: str, system: Optional[str]) -> str:
        from nyxara.mind.llm import LLMRequest
        req = LLMRequest.from_prompt(prompt, system=system, temperature=0.0,
                                     max_tokens=self.cfg.max_tokens)
        return (self._self.complete(req).text or "").strip()

    def _teacher_answer(self, prompt: str, system: Optional[str], name: str) -> str:
        from nyxara.mind.llm import LLMRequest
        req = LLMRequest.from_prompt(prompt, system=system, temperature=0.3,
                                     max_tokens=self.cfg.max_tokens)
        return (self.llm.complete_with(name, req).text or "").strip()

    def draft(self, prompt: str, *, system: Optional[str] = None) -> RouterResult:
        """Draft a reply: own model first (if it clears the bar), else the teacher."""
        own: Optional[str] = None
        confidence = 0.0
        if self.self_available():
            try:
                own = self._own_answer(prompt, system)
                confidence = float(self.verifier(prompt, own))
            except Exception:  # noqa: BLE001 — a failed own attempt simply defers to the teacher
                own, confidence = None, 0.0
            if own and confidence >= self.cfg.threshold:
                return RouterResult(own, "self", confidence, handed_off=True)

        teacher = self._teacher_name() if self.cfg.consult_teacher else None
        if teacher is not None:
            try:
                return RouterResult(self._teacher_answer(prompt, system, teacher),
                                    "teacher", confidence, handed_off=False)
            except Exception:  # noqa: BLE001
                pass

        # no teacher to consult — keep the own answer if we have one, else nothing
        if own:
            return RouterResult(own, "self", confidence, handed_off=True)
        return RouterResult("", "none", 0.0, handed_off=False)


# --------------------------------------------------------------------------- #
# Self-test / demo
# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # pragma: no cover
    print("=" * 70)
    print("NYXARA confidence-router self-test (offline, scripted minds)")
    print("=" * 70)

    # verifier: empty/degenerate/echo score low; a real answer scores high
    assert answer_quality("2+2?", "") == 0.0
    assert answer_quality("2+2?", "the the the the the") < 0.4
    assert answer_quality("What is the capital of France?", "The capital is Paris.") > 0.5
    print("verifier               : honest on empty / degenerate / good ✓")

    from nyxara.kernel.config import NyxaraSettings, Profile

    class _OwnProvider:
        def __init__(self, text, ok=True):
            self._t, self._ok = text, ok

        def available(self):
            return self._ok

        def complete(self, req):
            return type("_R", (), {"text": self._t})()

    class _Teacher:
        def available_providers(self):
            return ["anthropic", "self", "mock"]

        def complete_with(self, name, req):
            return type("_R", (), {"text": f"[teacher:{name}] a careful answer"})()

    s = NyxaraSettings.for_profile(Profile.TEST)
    s.router.enabled = True
    s.router.threshold = 0.5

    # a confident own answer is handed off
    good = Router(_Teacher(), settings=s,
                  self_provider=_OwnProvider("The capital is Paris, Master.")).draft(
        "What is the capital of France?")
    print(f"own confident          : source={good.source} conf={good.confidence:.2f}")
    assert good.source == "self" and good.handed_off

    # a degenerate own answer defers to the teacher
    weak = Router(_Teacher(), settings=s,
                  self_provider=_OwnProvider("the the the the the")).draft("Explain entropy.")
    print(f"own weak -> teacher     : source={weak.source}")
    assert weak.source == "teacher" and not weak.handed_off

    # no own model + no teacher -> honest empty
    s2 = NyxaraSettings.for_profile(Profile.TEST)

    class _NoTeacher:
        def available_providers(self):
            return ["mock"]

    none = Router(_NoTeacher(), settings=s2,
                  self_provider=_OwnProvider("x", ok=False)).draft("hi")
    print(f"nothing available       : source={none.source}")
    assert none.source == "none"

    print("\nALL SELF-TESTS PASSED ✓")
