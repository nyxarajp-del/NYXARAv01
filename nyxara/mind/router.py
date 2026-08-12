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
from typing import Any, Callable, Optional, Tuple

from nyxara.kernel.config import OWN_PROVIDERS, NyxaraSettings, get_settings
from nyxara.mind.metacognition import HONEST_ABSTENTION, MetaCognition, MetaDecision

__all__ = ["RouterResult", "answer_quality", "default_verifier", "Router"]

Verifier = Callable[[str, str], float]

_WORD = re.compile(r"[A-Za-z0-9']+")


def _tokens(text: str) -> list:
    return _WORD.findall((text or "").lower())


# --------------------------------------------------------------------------- #
# Scaffold phrases — the shapes her own machinery emits when it is NOT answering
# --------------------------------------------------------------------------- #
# ``kernel/orchestrator._default_reasoner`` is the deterministic stand-in used whenever real
# reasoning is unavailable, and it emits fixed templates: "I understand: X", "perform: X". Those
# strings then land in her transcripts, her transcripts become the foundry's training corpus, and
# her forged model learns to emit them — fluently, and about anything.
#
# The verifier below could not see it. It scores *form*: length, coherence, unique-word ratio, and
# how much of the prompt is parroted back. "I understand: Hii" is well-formed on every one of those
# — three unique words, all real, only one of them from the prompt — so it scored 0.74 and cleared
# the handoff threshold. She answered the Master with the stand-in's phrasing, from her own model,
# confidently, and never asked the 2.5 GB Gemma sitting loaded in the same process.
#
# The echo penalty was the right idea aimed one level too low: it catches a model parroting the
# *prompt*, not one parroting the *scaffold*. A fixed prefix dilutes the overlap ratio enough to
# slip under it — "I understand: Hii" is 33% prompt words, and the penalty starts at 60%.
# The same trap, one layer up: her CALIBRATION qualifiers (observe/honesty.py::_qualifier) are
# prepended to an answer before this verifier ever sees it. They are her own words about her own
# confidence, not the drafting model's content — so counting them was the verifier scoring her
# boilerplate and calling it quality. Worse, it scored *upward*: the qualifier adds length and
# unique words, the two signals below that reward substance. Measured, "I'm certain that The
# answer is clear." — a reply that says nothing — cleared the 0.6 threshold at **0.921**, higher
# than the "I understand: Hii" case above, because the honesty layer had padded it.
#
# Longest-first below: "i'm confident that" must be tried before any shorter prefix that is its
# own prefix, or the strip leaves a fragment behind and scores that instead.
_SCAFFOLD_PREFIXES: Tuple[str, ...] = (
    "i understand:",       # _default_reasoner, conversational branch
    "perform:",            # _default_reasoner, command branch
    "done:",               # the act stage's spoken confirmation
    "the master says:",    # dialogue-template continuation (the n-gram's signature)
    "nyxara responds:",
    "nyxara:",
    # observe/honesty.py::_qualifier — every rung of it
    "i suspect, though i'm not sure, that",
    "i doubt, but it's possible, that",
    "i don't know whether",
    "i'm confident that",
    "i'm certain that",
    "i think",
)


def strip_scaffold(answer: str) -> Tuple[str, bool]:
    """Remove a leading scaffold phrase. Returns ``(remainder, was_scaffolded)``."""
    text = (answer or "").strip()
    low = text.lower()
    for prefix in _SCAFFOLD_PREFIXES:
        if low.startswith(prefix):
            return text[len(prefix):].strip(), True
    return text, False


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

    # 0) scaffold check, before anything else. A reply that is one of her own stand-in templates
    #    wrapped around the prompt is not an answer at all — it is the shape of *not* answering,
    #    reproduced by a model that learned it from her transcripts. Judge what is left once the
    #    template is removed: "I understand: Hii" leaves "Hii", a bare echo, which scores 0.
    body, scaffolded = strip_scaffold(ans)
    if scaffolded:
        rest = _tokens(body)
        pwords = set(_tokens(prompt))
        if not rest:
            return 0.0                        # the template and nothing else
        if pwords and all(w in pwords for w in rest):
            return 0.0                        # the template wrapped around the question
        ans, words = body, rest               # score only the part that says something

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


def _strongest_own_brain(settings: NyxaraSettings) -> Any:
    """The best of HER OWN models to draft with — ``litertlm`` if its weights are here, else ``self``.

    The router measures how often she answers unaided, so "unaided" has to mean her strongest own
    brain, not merely her oldest one. Her on-device Gemma is both stronger than a freshly-forged LoRA
    and always reachable, so it drafts whenever it is available; ``SelfProvider`` remains the fallback
    (and the one the foundry keeps improving). Imported lazily — the router is constructed on paths
    that must not pay for ``mind/llm`` unless they route.
    """
    from nyxara.mind.llm import LiteRTLMProvider, SelfProvider
    try:
        on_device = LiteRTLMProvider(settings)
        if on_device.available():
            return on_device
    except Exception:  # noqa: BLE001 — never let the primary's absence break routing
        pass
    return SelfProvider(settings)


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
            self_provider = _strongest_own_brain(self.settings)
        self._self = self_provider
        self.meta = MetaCognition(answer_threshold=self.cfg.threshold,
                                  abstain_below=self.cfg.abstain_below)

    # ---- availability ---- #
    def self_available(self) -> bool:
        try:
            return bool(self._self.available())
        except Exception:  # noqa: BLE001
            return False

    def _teacher_name(self) -> Optional[str]:
        """The real external teacher to consult — the configured provider, else any real one.

        A "teacher" is a model that is *not hers*. ``OWN_PROVIDERS`` covers everything that runs
        in-process — including her on-device ``litertlm`` primary — so handing off to one of those
        would not be a hand-off at all; it would be her asking herself and calling it consultation.
        """
        try:
            avail = list(self.llm.available_providers())
        except Exception:  # noqa: BLE001
            return None
        pref = self.settings.llm.provider.value
        if pref in avail and pref not in OWN_PROVIDERS:
            return pref
        for name in avail:
            if name not in OWN_PROVIDERS:
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

    def _own_answer_reranked(self, prompt: str, system: Optional[str]) -> str:
        """Draw several own-model samples and select by ground truth (the ceiling-break).

        An exact oracle certifies a provably-correct candidate when the domain is decidable;
        otherwise self-consistency majority vote picks the most-agreed answer — best-of-N lifts
        accuracy above the single greedy draft (and above any single teacher sample) on
        verifiable / reasoning prompts. Falls back to the plain greedy draft on any error."""
        from nyxara.mind.llm import LLMRequest
        from nyxara.mind.verified_answer import best_verified_answer, faculty_oracle

        def _gen() -> str:
            req = LLMRequest.from_prompt(prompt, system=system, temperature=0.7,
                                         max_tokens=self.cfg.max_tokens)
            return (self._self.complete(req).text or "").strip()

        n = int(getattr(self.cfg, "rerank_samples", 5) or 5)
        # Faculties are already short-circuited in draft(); pass the oracle only when they are on
        # so this stays consistent with the operator's use_faculties choice.
        oracle = faculty_oracle if self.cfg.use_faculties else None
        try:
            va = best_verified_answer(prompt, _gen, samples=n, oracle=oracle)
        except Exception:  # noqa: BLE001 — rerank is best-effort; never break a turn
            va = None
        if va is not None and va.text:
            return va.text
        return self._own_answer(prompt, system)

    def _teacher_answer(self, prompt: str, system: Optional[str], name: str) -> str:
        from nyxara.mind.llm import LLMRequest
        req = LLMRequest.from_prompt(prompt, system=system, temperature=0.3,
                                     max_tokens=self.cfg.max_tokens)
        return (self.llm.complete_with(name, req).text or "").strip()

    @staticmethod
    def _answer_similarity(a: str, b: str) -> float:
        """Token-Jaccard overlap of two answers in [0, 1] — a cheap agreement measure."""
        ta = {w for w in "".join(c if c.isalnum() else " " for c in a.lower()).split()}
        tb = {w for w in "".join(c if c.isalnum() else " " for c in b.lower()).split()}
        if not ta and not tb:
            return 1.0
        if not ta or not tb:
            return 0.0
        return len(ta & tb) / len(ta | tb)

    def _self_consistency_uncertainty(self, prompt: str, system: Optional[str],
                                      own: Optional[str]) -> float:
        """An INTERNAL epistemic signal: how much NYXARA's own answer wanders across samples.

        Draws ``self_consistency_samples`` independent answers from her own model at non-zero
        temperature and measures their mean pairwise agreement; low agreement → high epistemic
        uncertainty *she measured about herself*. Returns 0.0 (no signal) when the probe is off,
        her own model is unavailable, or there was no own answer to probe — so default behaviour
        (samples=1) is unchanged.
        """
        n = int(getattr(self.cfg, "self_consistency_samples", 1) or 1)
        if n <= 1 or not own or not self.self_available():
            return 0.0
        from nyxara.mind.llm import LLMRequest
        samples = [own]
        for _ in range(n - 1):
            try:
                req = LLMRequest.from_prompt(prompt, system=system, temperature=0.7,
                                             max_tokens=self.cfg.max_tokens)
                txt = (self._self.complete(req).text or "").strip()
            except Exception:  # noqa: BLE001 — a failed sample just lowers the sample count
                continue
            if txt:
                samples.append(txt)
        if len(samples) < 2:
            return 0.0
        sims, pairs = 0.0, 0
        for i in range(len(samples)):
            for j in range(i + 1, len(samples)):
                sims += self._answer_similarity(samples[i], samples[j])
                pairs += 1
        mean_agreement = sims / pairs if pairs else 1.0
        return self.meta.introspect(self_consistency=mean_agreement)

    def draft(self, prompt: str, *, system: Optional[str] = None) -> RouterResult:
        """Draft a reply: a verifiable faculty first, then own model, then the teacher.

        Neuro-symbolic: exact computation/proof beats any neural guess, so a fitting faculty
        short-circuits everything. Otherwise the metacognition gate chooses own / teacher /
        an honest abstention — NYXARA never bluffs when she has nothing trustworthy to say."""
        # 0) verifiable faculty — exact math / logic wins over any guess
        if self.cfg.use_faculties:
            fac = self._faculty_answer(prompt)
            if fac is not None:
                text, conf = fac
                return RouterResult(text, "faculty", conf, handed_off=True)

        # 1) NYXARA's own model
        own: Optional[str] = None
        confidence = 0.0
        if self.self_available():
            try:
                if getattr(self.cfg, "verify_rerank", False) and \
                        int(getattr(self.cfg, "rerank_samples", 1) or 1) > 1:
                    own = self._own_answer_reranked(prompt, system)
                else:
                    own = self._own_answer(prompt, system)
                confidence = float(self.verifier(prompt, own))
            except Exception:  # noqa: BLE001 — a failed own attempt simply defers downstream
                own, confidence = None, 0.0
            if own and confidence >= self.cfg.threshold:
                return RouterResult(own, "self", confidence, handed_off=True)

        # 2) metacognition decides: own / teacher / honest abstention — now informed by an
        #    INTERNAL self-consistency probe (how stable is her own answer across samples?).
        internal_uncertainty = self._self_consistency_uncertainty(prompt, system, own)
        teacher = self._teacher_name() if self.cfg.consult_teacher else None
        verdict = self.meta.assess(prompt, own_answer=own, own_conf=confidence,
                                   teacher_available=teacher is not None,
                                   internal_uncertainty=internal_uncertainty)
        if verdict.decision is MetaDecision.CONSULT_TEACHER and teacher is not None:
            try:
                return RouterResult(self._teacher_answer(prompt, system, teacher),
                                    "teacher", confidence, handed_off=False)
            except Exception:  # noqa: BLE001
                pass
        if verdict.decision is MetaDecision.ANSWER_SELF and own:
            return RouterResult(own, "self", confidence, handed_off=True)
        if own and confidence > self.cfg.abstain_below:
            return RouterResult(own, "self", confidence, handed_off=True)
        if own:
            return RouterResult(HONEST_ABSTENTION, "abstain", confidence, handed_off=False)
        return RouterResult("", "none", 0.0, handed_off=False)

    def draft_self(self, prompt: str, *, system: Optional[str] = None) -> Optional[RouterResult]:
        """Give NYXARA's OWN model first crack, with NO teacher fallback.

        Returns a handoff :class:`RouterResult` iff her forged model is available and its
        answer clears the verifier threshold; otherwise ``None`` so the caller can run its
        own teacher/council path. This is the lever that lets her own brain become *primary*
        in the integrated reasoner without reaching around any gate — the reply is still a
        proposal the kernel disposes."""
        if not self.self_available():
            return None
        try:
            if getattr(self.cfg, "verify_rerank", False) and \
                    int(getattr(self.cfg, "rerank_samples", 1) or 1) > 1:
                own = self._own_answer_reranked(prompt, system)
            else:
                own = self._own_answer(prompt, system)
        except Exception:  # noqa: BLE001 — a failed own attempt simply defers to the teacher
            return None
        if not own:
            return None
        conf = float(self.verifier(prompt, own))
        if conf >= self.cfg.threshold:
            return RouterResult(own, "self", conf, handed_off=True)
        return None

    def _faculty_answer(self, prompt: str) -> Optional[Tuple[str, float]]:
        # Reuse the SAME verifiable-reasoning entry point the integrated loop uses
        # (chain → single), so the router never discards NYXARA's multi-step reasoning on the
        # handoff path — her exact answer is as strong here as in the full cognitive cycle.
        try:
            from nyxara.mind.reasoning_faculties import solve_verifiable
            return solve_verifiable(prompt)
        except Exception:  # noqa: BLE001 — faculties are advisory; never crash a turn
            return None


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

    # scaffold echo: her own stand-in's phrasing, learned from her transcripts and served back
    # confidently by her forged model. It scored 0.74 here — above the 0.6 handoff threshold —
    # so she answered the Master with the shape of not-answering and never asked her strongest
    # model, which was loaded in the same process the whole time.
    assert answer_quality("Hii", "I understand: Hii") == 0.0
    assert answer_quality("delete the logs", "perform: delete the logs") == 0.0
    assert answer_quality("Hii", "The Master says: Hii. NYXARA responds: Hii") < 0.6
    # ...but the prefix is not itself the crime. A reply that says something real still passes.
    assert answer_quality("Hii", "I understand: you are greeting me, and I am glad of it") > 0.6
    assert answer_quality("Hii", "Hello. I am here. What is the matter") > 0.6
    print("scaffold echo          : rejected; a real answer behind the same prefix passes ✓")

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
            return ["litertlm", "aicredits", "groq", "airouter", "self", "native"]

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
            return ["native"]

    none = Router(_NoTeacher(), settings=s2,
                  self_provider=_OwnProvider("x", ok=False)).draft("hi")
    print(f"nothing available       : source={none.source}")
    assert none.source == "none"

    print("\nALL SELF-TESTS PASSED ✓")
