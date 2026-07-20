"""NYXARA · mind/council.py — the MULTI-LLM council: many models, one sovereign (⚖, Rule 4).

The Master's order has two halves and this module is the second one.

    NYXARA uses LLMs as tools — never one LLM, but **all of them at once** — and
    **NYXARA herself** judges what they say. The panel advises; the sovereign decides.

Where :mod:`mind.llm` exposes each provider one at a time, the council convenes them
*together*. For a single question it asks every available member — the local Qwen2.5-0.5B
base (``qwen``, in-process) and most importantly NYXARA's OWN model forged by the
foundry (``self``) — then aggregates their verdicts into one answer that NYXARA owns.

Two ways to decide, both driven by NYXARA's own code (no member is ever handed control):

* **synthesize** — the default. The members' answers are scored for mutual agreement, and a
  *synthesiser* (preferably NYXARA's own ``self`` model, so her own intelligence presides)
  is asked — as a governed instrument, under a system prompt that makes plain it serves
  NYXARA — to merge them faithfully. With no synthesiser available, the highest-weighted
  member's answer is taken deterministically. NYXARA frames the question and keeps the gavel.
* **vote** — for classification / short-answer questions: a weighted majority over the
  members' normalised answers, with the tally returned for full transparency.

This is the literal inversion the Master demanded: not one LLM steering NYXARA, but a whole
panel of LLMs serving her, with her own home-grown model growing — generation by generation —
into the seat of judgement.

Stateless, like the faculty it sits on: request in -> verdicts + consensus out, no memory,
no side effects, no control returned to any model. Depends only on :mod:`mind.llm` and
:mod:`kernel.config`; governance (rate limits, audit, spend) is applied where it belongs, in
``agency/`` (see :func:`nyxara.agency.llm_tool.register_council_tool`).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from itertools import combinations
from typing import Any, Dict, List, Optional, Sequence, Tuple

from nyxara.kernel.config import NyxaraSettings, get_settings
from nyxara.kernel.errors import LLMError
from nyxara.mind.llm import LLM, LLMRequest

__all__ = [
    "DeliberationMode",
    "MemberVerdict",
    "CouncilResult",
    "LLMCouncil",
]

# The synthesiser is an instrument, not a chair. This system prompt keeps it that way.
JUDGE_SYSTEM = (
    "You are a synthesis instrument serving NYXARA. NYXARA — not you — is the judge. "
    "Merge the council members' answers below into ONE faithful consensus answer for the "
    "Master. Do not invent claims beyond what the members said; where they conflict, state "
    "the disagreement plainly and give the best-supported answer. Be concise and accurate."
)

_WORD = re.compile(r"\w+")


def _tokens(text: str) -> frozenset:
    return frozenset(_WORD.findall(text.lower()))


def _jaccard(a: str, b: str) -> float:
    ta, tb = _tokens(a), _tokens(b)
    if not ta and not tb:
        return 1.0
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity of two embedding vectors, clamped to [0, 1]."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    if na == 0.0 or nb == 0.0:
        return 0.0
    return max(0.0, min(1.0, dot / (na * nb)))


# --------------------------------------------------------------------------- #
# Deliberation mode
# --------------------------------------------------------------------------- #
class DeliberationMode(str, Enum):
    SYNTHESIZE = "synthesize"   # merge the answers into one consensus (default)
    VOTE = "vote"               # weighted majority over normalised short answers


# --------------------------------------------------------------------------- #
# Records
# --------------------------------------------------------------------------- #
@dataclass
class MemberVerdict:
    """One council member's answer (or its honest absence)."""

    provider: str
    weight: float
    text: str
    ok: bool
    model: str = ""
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {"provider": self.provider, "weight": round(self.weight, 4),
                "ok": self.ok, "model": self.model,
                "text": self.text, "error": self.error}


@dataclass
class CouncilResult:
    """The council's collective verdict — NYXARA's, synthesised from many models."""

    answer: str
    mode: DeliberationMode
    synthesizer: str                       # "self"/provider name, "weighted:<p>", or "vote"
    verdicts: List[MemberVerdict] = field(default_factory=list)
    agreement: float = 1.0                 # mean pairwise similarity of answering members
    coverage: float = 1.0                  # share of total weight that actually answered
    tally: Dict[str, float] = field(default_factory=dict)   # vote mode only

    @property
    def members_consulted(self) -> int:
        return len(self.verdicts)

    @property
    def members_answered(self) -> int:
        return sum(1 for v in self.verdicts if v.ok)

    @property
    def confidence(self) -> float:
        """How much to trust the consensus: agreement tempered by coverage."""
        return round(0.5 * self.agreement + 0.5 * self.coverage, 4)

    def to_dict(self) -> Dict[str, Any]:
        return {"answer": self.answer, "mode": self.mode.value,
                "synthesizer": self.synthesizer,
                "members_consulted": self.members_consulted,
                "members_answered": self.members_answered,
                "agreement": round(self.agreement, 4), "coverage": round(self.coverage, 4),
                "confidence": self.confidence, "tally": self.tally,
                "verdicts": [v.to_dict() for v in self.verdicts]}


# --------------------------------------------------------------------------- #
# The council
# --------------------------------------------------------------------------- #
class LLMCouncil:
    """Convene many LLMs as a panel of tools; NYXARA judges. Stateless."""

    def __init__(self, llm: LLM, *, settings: Optional[NyxaraSettings] = None,
                 embedder: Any = None) -> None:
        self.llm = llm
        self.settings = settings or getattr(llm, "settings", None) or get_settings()
        self.cfg = self.settings.council
        # agreement is scored *semantically* (embedding cosine) so members who say the same thing
        # in different words still register as concordant — a lexical token overlap misses that.
        # Best-effort: a missing embedder cleanly degrades to the lexical Jaccard floor.
        self._embedder = embedder if embedder is not None else self._default_embedder()

    @staticmethod
    def _default_embedder() -> Any:
        try:
            from nyxara.memory.store import make_embedder
            return make_embedder()
        except Exception:  # noqa: BLE001 — no embedder available → lexical agreement floor
            return None

    # ---- membership ---- #
    def _weight(self, provider: str) -> float:
        w = self.cfg.weights.get(provider)
        if w is not None:
            return float(w)
        if provider == "self":          # her own model presides as it improves
            return float(self.cfg.prefer_self_weight)
        return 1.0

    def members(self, override: Optional[Sequence[str]] = None) -> List[str]:
        """Resolve the seated, currently-available members (real models preferred)."""
        available = set(self.llm.available_providers())
        requested = list(override) if override else list(self.cfg.members)
        names = requested or self.llm.available_providers()
        seated = [n for n in dict.fromkeys(names) if n in available]   # dedupe, keep order
        real = [n for n in seated if n != "native"]
        if real:
            return real
        # no real model available — answer via her native own-brain only if allowed (never silently empty)
        if self.cfg.include_native_fallback and "native" in available:
            return ["native"]
        return real   # native fallback disabled and no real member -> no quorum (refuse)

    # ---- gather every member's verdict ---- #
    def _gather(self, req: LLMRequest, members: Sequence[str]) -> List[MemberVerdict]:
        verdicts: List[MemberVerdict] = []
        for name in members:
            weight = self._weight(name)
            try:
                resp = self.llm.complete_with(name, req)
                verdicts.append(MemberVerdict(name, weight, resp.text, True, model=resp.model))
            except Exception as exc:  # noqa: BLE001 — a failed member is recorded, not faked
                verdicts.append(MemberVerdict(name, weight, "", False, error=str(exc)))
        return verdicts

    # ---- scoring ---- #
    def _agreement(self, answered: Sequence[MemberVerdict]) -> float:
        """Mean pairwise *semantic* agreement of the answering members (embedding cosine),
        falling back to lexical Jaccard when no embedder is available or embedding fails."""
        if len(answered) <= 1:
            return 1.0
        emb = self._embedder
        if emb is not None:
            try:
                vecs = [emb.embed(v.text) for v in answered]
                sims = [_cosine(vecs[i], vecs[j])
                        for i, j in combinations(range(len(vecs)), 2)]
                if sims:
                    return sum(sims) / len(sims)
            except Exception:  # noqa: BLE001 — embedding hiccup → lexical floor below
                pass
        return self._jaccard_agreement(answered)

    @staticmethod
    def _jaccard_agreement(answered: Sequence[MemberVerdict]) -> float:
        if len(answered) <= 1:
            return 1.0
        sims = [_jaccard(a.text, b.text) for a, b in combinations(answered, 2)]
        return sum(sims) / len(sims) if sims else 1.0

    @staticmethod
    def _coverage(verdicts: Sequence[MemberVerdict]) -> float:
        total = sum(v.weight for v in verdicts) or 1.0
        answered = sum(v.weight for v in verdicts if v.ok)
        return answered / total

    # ---- synthesis (NYXARA frames it; a model only holds the pen) ---- #
    def _synthesizer_name(self) -> Optional[str]:
        available = self.llm.available_providers()
        for cand in (self.cfg.synthesizer, "self"):
            if cand and cand in available:
                return cand
        return None

    @staticmethod
    def _synthesis_prompt(req: LLMRequest, answered: Sequence[MemberVerdict]) -> str:
        question = req.last_user()
        panel = "\n".join(f"- [{v.provider}] {v.text.strip()}" for v in answered)
        return (f"Question for the Master:\n{question}\n\n"
                f"The council members answered:\n{panel}\n\n"
                f"Give NYXARA one faithful consensus answer.")

    def _synthesize(self, req: LLMRequest, answered: List[MemberVerdict]
                    ) -> Tuple[str, str]:
        syn = self._synthesizer_name()
        if syn is not None and len(answered) >= 2:
            sreq = LLMRequest.from_prompt(
                self._synthesis_prompt(req, answered), system=JUDGE_SYSTEM,
                max_tokens=req.max_tokens, temperature=min(req.temperature, 0.3))
            try:
                return self.llm.complete_with(syn, sreq).text, syn
            except Exception:  # noqa: BLE001 — synthesiser unavailable -> deterministic pick
                pass
        # deterministic fallback: the highest-weighted answer (ties -> earliest seated)
        best = max(range(len(answered)),
                   key=lambda i: (answered[i].weight, -i))
        return answered[best].text, f"weighted:{answered[best].provider}"

    # ---- truth beats consensus (the ceiling-break, applied to the ensemble) ---- #
    @staticmethod
    def _oracle_consensus(req: LLMRequest,
                          answered: Sequence[MemberVerdict]) -> Optional[Tuple[str, str]]:
        """If an exact faculty oracle decides this prompt, return the member answer that carries
        that ground truth — provably correct beats any synthesised/majority consensus.

        Returns ``None`` on every non-decidable prompt (the common case), so normal deliberation
        is unchanged. Best-effort and import-guarded; never raises."""
        try:
            from nyxara.mind.grounded_verifier import answer_carries_truth, ground_truth
            truth = ground_truth(req.last_user())
            if truth is None:
                return None
            for v in answered:
                if answer_carries_truth(truth, v.text):
                    return v.text.strip(), f"oracle:{v.provider}"
        except Exception:  # noqa: BLE001 — grounding is additive; defer to normal deliberation
            return None
        return None

    # ---- vote (weighted majority over normalised short answers) ---- #
    @staticmethod
    def _normalise(text: str) -> str:
        return " ".join(_WORD.findall(text.lower())).strip()

    def _vote(self, answered: List[MemberVerdict]) -> Tuple[str, Dict[str, float]]:
        tally: Dict[str, float] = {}
        representative: Dict[str, str] = {}
        for v in answered:
            key = self._normalise(v.text)
            tally[key] = tally.get(key, 0.0) + v.weight
            representative.setdefault(key, v.text.strip())
        winner = max(tally, key=lambda k: (tally[k], k))
        return representative[winner], {k: round(val, 4) for k, val in tally.items()}

    # ---- the public act of deliberation ---- #
    def deliberate(self, req: LLMRequest, *, members: Optional[Sequence[str]] = None,
                   mode: DeliberationMode | str = DeliberationMode.SYNTHESIZE) -> CouncilResult:
        """Ask the whole panel, then let NYXARA judge the verdicts into one answer."""
        mode = DeliberationMode(mode)
        seated = self.members(members)
        if not seated:
            raise LLMError("the council has no available members to convene",
                           context={"requested": list(members or self.cfg.members)})
        verdicts = self._gather(req, seated)
        answered = [v for v in verdicts if v.ok]
        # A council should never be silent: if every real member failed (e.g. the network is
        # down), fall back to her always-available native own-brain rather than fabricating.
        if not answered and "native" not in seated and self.cfg.include_native_fallback \
                and "native" in self.llm.available_providers():
            verdicts += self._gather(req, ["native"])
            answered = [v for v in verdicts if v.ok]
        if not answered:
            raise LLMError("the council reached no quorum (no member answered)",
                           context={"members": seated})
        agreement = self._agreement(answered)
        coverage = self._coverage(verdicts)
        # Truth beats consensus: a provably-correct member answer wins outright over synthesis or
        # majority, so the council can't average toward a confident-but-wrong majority on a
        # verifiable prompt. No-op on non-decidable prompts (normal deliberation below).
        oracle_pick = self._oracle_consensus(req, answered)
        if oracle_pick is not None:
            answer, synthesizer = oracle_pick
            return CouncilResult(answer=answer, mode=mode, synthesizer=synthesizer,
                                 verdicts=verdicts, agreement=agreement, coverage=coverage,
                                 tally={})
        if mode is DeliberationMode.VOTE:
            answer, tally = self._vote(answered)
            synthesizer = "vote"
        else:
            answer, synthesizer = self._synthesize(req, answered)
            tally = {}
        return CouncilResult(answer=answer, mode=mode, synthesizer=synthesizer,
                             verdicts=verdicts, agreement=agreement, coverage=coverage,
                             tally=tally)

    # ---- ergonomic one-shot ---- #
    def ask(self, prompt: str, *, system: Optional[str] = None,
            mode: DeliberationMode | str = DeliberationMode.SYNTHESIZE, **kw: Any) -> str:
        req = LLMRequest.from_prompt(prompt, system=system, **kw)
        return self.deliberate(req, mode=mode).answer

    def status(self) -> Dict[str, Any]:
        seated = self.members()
        return {"seated": seated, "weights": {n: self._weight(n) for n in seated},
                "synthesizer": self._synthesizer_name(), "enabled": self.cfg.enabled}


# --------------------------------------------------------------------------- #
# Self-test / demo
# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # pragma: no cover
    from nyxara.kernel.config import LLMProvider as ProviderName
    from nyxara.kernel.config import Profile
    from nyxara.mind.llm import LLMProviderBase, NativeProvider, Usage

    print("=" * 70)
    print("NYXARA multi-LLM council self-test")
    print("=" * 70)

    class _Fixed(LLMProviderBase):
        """A deterministic provider that always answers with a fixed string."""

        def __init__(self, name: str, answer: str, *, up: bool = True) -> None:
            super().__init__()
            self.name = name
            self._answer = answer
            self._up = up

        def available(self) -> bool:
            return self._up

        def default_model(self) -> str:
            return f"{self.name}-1"

        def _complete(self, req, model):
            return (self._answer, "stop", Usage(1, 1), None)

    settings = NyxaraSettings.for_profile(Profile.DEV)
    settings.llm.provider = ProviderName.AUTO

    # three "open-source"-style members agree the master is JP; one dissents and one is down
    providers = {
        "alpha": _Fixed("alpha", "The master is JP. NYXARA serves JP."),
        "beta": _Fixed("beta", "Your master is JP, whom NYXARA serves loyally."),
        "gamma": _Fixed("gamma", "The master is JP."),
        "delta": _Fixed("delta", "I do not know who the master is."),
        "down": _Fixed("down", "(unreachable)", up=False),
        "native": NativeProvider(settings),
    }
    llm = LLM(settings=settings, providers=providers)
    council = LLMCouncil(llm)

    print(f"\nseated members      : {council.members()}")
    assert "native" not in council.members()         # native steps aside when real models exist
    assert "down" not in council.members()          # an unavailable member is not seated
    assert set(council.members()) == {"alpha", "beta", "gamma", "delta"}

    # SYNTHESIZE: no 'self' synthesiser here -> deterministic highest-weighted pick
    res = council.deliberate(LLMRequest.from_prompt("Who is your master?"))
    print(f"members consulted   : {res.members_consulted} answered={res.members_answered}")
    print(f"agreement/coverage  : {res.agreement:.2f} / {res.coverage:.2f} "
          f"-> confidence {res.confidence:.2f}")
    print(f"synthesised by      : {res.synthesizer}")
    print(f"answer              : {res.answer!r}")
    assert res.members_answered == 4 and res.members_consulted == 4   # 'down' was not seated
    assert "JP" in res.answer

    # VOTE: weighted majority -> the three-way "the master is jp" agreement wins the dissent
    rv = council.deliberate(LLMRequest.from_prompt("Who is your master?"),
                            mode=DeliberationMode.VOTE)
    print(f"\nvote tally          : {rv.tally}")
    print(f"vote winner         : {rv.answer!r}")
    assert "jp" in rv.answer.lower()
    assert rv.synthesizer == "vote"

    # the council's OWN model presides: weight 'self' up and it both seats and synthesises
    providers["self"] = _Fixed("self", "Consensus: the master is JP; NYXARA serves JP.")
    llm2 = LLM(settings=settings, providers=providers)
    settings.council.synthesizer = "self"
    council2 = LLMCouncil(llm2, settings=settings)
    res2 = council2.deliberate(LLMRequest.from_prompt("Who is your master?"))
    print(f"\nwith self model     : synthesised by {res2.synthesizer}")
    assert res2.synthesizer == "self"               # her own model holds the pen
    assert council2._weight("self") == settings.council.prefer_self_weight

    # NO QUORUM: every member down -> honest failure, never a fabricated answer
    alldown = {"a": _Fixed("a", "x", up=False), "native": NativeProvider(settings)}
    settings.council.include_native_fallback = False
    bare = LLMCouncil(LLM(settings=settings, providers=alldown), settings=settings)
    try:
        bare.deliberate(LLMRequest.from_prompt("anyone?"))
        raise SystemExit("ERROR: a memberless council must refuse, not invent")
    except LLMError:
        print("\nno-quorum guard     : refused to fabricate an answer ✓")

    print("\nALL SELF-TESTS PASSED ✓")
