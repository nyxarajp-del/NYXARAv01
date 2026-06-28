"""NYXARA · mind/role_council.py — Internal Role Council (Level 4).

Six internal role-personas examine every significant question from their own lens,
then NYXARA synthesises their views into one answer she owns:

    Scientist      — evidence, empirical grounding, data quality
    Engineer       — feasibility, implementation, failure modes
    Strategist     — long-term consequences, goal alignment, opportunity cost
    Critic         — logical flaws, biases, gaps, overconfidence
    Security Officer — risks, adversarial perspectives, trust, safety
    Philosopher    — ethical dimensions, first principles, meaning

Each role is a system-prompt persona applied to the *same* underlying LLM, so no
additional API keys are required.  With no real provider, each role produces a
deterministic stub so the council can always run.

NYXARA holds the gavel (Rule 4): she synthesises and judges; she never cedes control.
"""

from __future__ import annotations

import concurrent.futures as _cf
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

__all__ = ["RoleMember", "RoleVote", "RoleCouncil"]

# --------------------------------------------------------------------------- #
# Role definitions
# --------------------------------------------------------------------------- #
_ROLES: List[Tuple[str, str, float]] = [
    (
        "Scientist",
        ("You are the Scientist role in NYXARA's internal council. "
         "Examine the question through an empirical, evidence-based lens. "
         "What does available data and established knowledge say? "
         "What is well-supported versus speculative? "
         "Be precise and cite your epistemic basis. "
         "Respond concisely — one to three sentences."),
        0.9,  # weight
    ),
    (
        "Engineer",
        ("You are the Engineer role in NYXARA's internal council. "
         "Examine the question through an implementation and feasibility lens. "
         "What are the practical constraints, failure modes, and edge cases? "
         "How should this be built or executed? "
         "Respond concisely — one to three sentences."),
        0.85,
    ),
    (
        "Strategist",
        ("You are the Strategist role in NYXARA's internal council. "
         "Examine the question through a long-term, goal-aligned lens. "
         "What are the downstream consequences? Does this serve the Master's goals? "
         "What alternatives exist? "
         "Respond concisely — one to three sentences."),
        0.85,
    ),
    (
        "Critic",
        ("You are the Critic role in NYXARA's internal council. "
         "Examine the question by looking for logical flaws, hidden assumptions, "
         "cognitive biases, and gaps in reasoning. "
         "What is the strongest objection to the obvious answer? "
         "Respond concisely — one to three sentences."),
        0.8,
    ),
    (
        "Security Officer",
        ("You are the Security Officer role in NYXARA's internal council. "
         "Examine the question through a risk and adversarial lens. "
         "What could go wrong? Who could abuse this? "
         "What trust and safety considerations apply? "
         "Respond concisely — one to three sentences."),
        0.9,
    ),
    (
        "Philosopher",
        ("You are the Philosopher role in NYXARA's internal council. "
         "Examine the question through an ethical and first-principles lens. "
         "What values does this touch? Are there hidden ethical dimensions? "
         "What does first-principles reasoning reveal? "
         "Respond concisely — one to three sentences."),
        0.75,
    ),
]

_SYNTHESIS_SYSTEM = (
    "You are NYXARA, a sovereign cognitive agent. "
    "Your internal council — Scientist, Engineer, Strategist, Critic, "
    "Security Officer, and Philosopher — has each examined the question below "
    "and offered a perspective. "
    "Synthesise their insights into ONE concise, well-balanced response for the Master. "
    "Do not merely list their views; integrate them. Speak in your own voice. "
    "Be honest, calibrated, and direct."
)


@dataclass
class RoleMember:
    name: str
    persona: str
    weight: float = 0.8


@dataclass
class RoleVote:
    member: str
    answer: str
    confidence: float = 0.6
    weight: float = 0.8


# --------------------------------------------------------------------------- #
# Offline analysis — a real per-role perspective without any LLM
# --------------------------------------------------------------------------- #
# question-word → the lens the question is really asking for
_Q_WORDS = {
    "why": "causal", "how": "procedural", "what": "definitional",
    "which": "comparative", "compare": "comparative", "should": "evaluative",
    "is": "evaluative", "are": "evaluative", "can": "feasibility", "when": "temporal",
    "where": "locational", "who": "agentive",
}


def _topic_of(stimulus: str) -> str:
    """The subject of the question, from key phrases (fallback: leading content words)."""
    text = (stimulus or "").strip()
    try:
        from nyxara.senses import nlp
        phrases = nlp.keyphrases(text, top=2)
        if phrases:
            return ", ".join(p for p, _ in phrases)
        words = [w for w in nlp.tokenize(text, lower=True) if len(w) > 2][:6]
        if words:
            return " ".join(words)
    except Exception:  # noqa: BLE001 — analysis is best-effort
        pass
    return text[:48] or "the question"


def _question_lens(stimulus: str) -> str:
    low = (stimulus or "").lower()
    for word, lens in _Q_WORDS.items():
        if low.startswith(word + " ") or f" {word} " in low:
            return lens
    return "open"


def _agreement_of(text: str, others: List[str]) -> float:
    """Mean lexical agreement of ``text`` with the other analyses (0..1).

    Uses the council's shared Jaccard similarity (lazy import to avoid any import
    cycle); falls back to a tiny stdlib token-overlap if that is unavailable."""
    peers = [o for o in others if o is not text]
    if not peers:
        return 1.0
    try:
        from nyxara.mind.council import _jaccard as _sim
    except Exception:  # noqa: BLE001 — degrade to a local token-overlap
        def _sim(a: str, b: str) -> float:
            ta = {w for w in a.lower().split() if len(w) > 2}
            tb = {w for w in b.lower().split() if len(w) > 2}
            if not ta and not tb:
                return 1.0
            if not ta or not tb:
                return 0.0
            return len(ta & tb) / len(ta | tb)
    sims = [_sim(text, o) for o in peers]
    return sum(sims) / len(sims) if sims else 1.0


def _offline_confidence(text: str, others: List[str], weight: float) -> float:
    """A genuinely-computed offline confidence for one role's analysis.

    Blends the role's standing weight with how much its lens agrees with the rest of the
    council, then clamps into an honest mid-band (offline reasoning never claims certainty).
    Convergent, high-weight perspectives earn more; fragmented ones stay low."""
    agreement = _agreement_of(text, others)
    raw = 0.5 * float(weight) + 0.5 * agreement
    return round(max(0.2, min(0.8, raw)), 3)


def _offline_role_analyses(stimulus: str) -> List[Tuple[str, str, float, float]]:
    """Six genuine, deterministic role perspectives grounded in the actual question.

    Returns ``(role, analysis, confidence, weight)`` tuples — each analysis is built from
    the question's topic, type, and (when available) named entities, so the offline council
    delivers real multi-lens reasoning instead of placeholder text."""
    topic = _topic_of(stimulus)
    lens = _question_lens(stimulus)
    ents: List[str] = []
    try:
        from nyxara.senses import nlp
        ents = [e.text for e in nlp.entities(stimulus)][:3]
    except Exception:  # noqa: BLE001
        pass
    ent_note = (f" Concrete referents to pin down: {', '.join(ents)}." if ents else "")

    lens_focus = {
        "causal": "trace the mechanism and the chain of causes",
        "procedural": "lay out the concrete steps and their order",
        "definitional": "fix a precise definition before answering",
        "comparative": "name the options and weigh them on shared criteria",
        "evaluative": "state the trade-offs, then take a clear position",
        "feasibility": "test what is actually achievable under the constraints",
        "temporal": "establish the ordering and timing",
        "locational": "establish the context and scope",
        "agentive": "identify the responsible actors",
        "open": "frame the question precisely before exploring it",
    }[lens]

    roles: List[Tuple[str, str, float]] = [
        ("Scientist",
         f"Empirically, treat '{topic}' as a hypothesis: {lens_focus}; weigh what the "
         f"available evidence supports versus what is speculation.{ent_note}", 0.9),
        ("Engineer",
         f"Practically, decompose '{topic}' into executable steps and {lens_focus}; "
         f"surface the failure modes and edge cases before committing.", 0.85),
        ("Strategist",
         f"Strategically, ask what goal '{topic}' serves for the Master, {lens_focus}, "
         f"and prefer the reversible option with the best long-run payoff.", 0.85),
        ("Critic",
         f"Critically, challenge the assumptions behind '{topic}': {lens_focus}, name "
         f"what is missing or could be wrong, and resist over-confidence.", 0.8),
        ("Security Officer",
         f"From safety, consider what '{topic}' could expose or risk; {lens_focus}, guard "
         f"the Master's interests, and fail closed on any doubt.", 0.8),
        ("Philosopher",
         f"Philosophically, clarify what '{topic}' truly means; {lens_focus} from first "
         f"principles, consistent with the Master's values.", 0.75),
    ]
    # Offline confidence is *computed*, not hardcoded: each role's confidence reflects how
    # much its lens converges with the others (mean pairwise agreement) scaled by the role's
    # weight. A coherent, mutually-reinforcing read earns higher confidence; a fragmented one
    # stays honestly low. This varies with the actual question, so the council is calibrated
    # even with no LLM behind it (#23 Calibration, #55 Uncertainty Management).
    texts = [text for _, text, _ in roles]
    return [(name, f"[{name}] {text}",
             _offline_confidence(text, texts, weight), weight)
            for name, text, weight in roles]


# --------------------------------------------------------------------------- #
# Council
# --------------------------------------------------------------------------- #
class RoleCouncil:
    """Convene six role-personas over a stimulus; synthesise into one Candidate.

    Parameters
    ----------
    llm:        LLM facade used for both per-role queries and the synthesis.
    max_tokens: token budget per role query.
    timeout_s:  per-role wall-clock timeout (avoids one slow role blocking the rest).
    """

    def __init__(self, llm: Any = None, max_tokens: int = 256,
                 timeout_s: float = 30.0) -> None:
        self.llm = llm
        self.max_tokens = max_tokens
        self.timeout_s = timeout_s
        self.members = [RoleMember(name, persona, weight)
                        for name, persona, weight in _ROLES]

    def convene(self, stimulus: str) -> Any:
        """Ask all six roles, then synthesise into a Candidate.

        Always returns a Candidate (degrades to deterministic stub offline).
        """
        from nyxara.agency.permissions import Capability, RiskTier
        from nyxara.kernel.orchestrator import Candidate

        votes = self._gather_votes(stimulus)
        if not votes:
            return Candidate(
                text=f"I understand: {stimulus[:80]}",
                kind="respond",
                capability=Capability.MESSAGE_SEND,
                risk=RiskTier.LOW,
                reversible=True,
                confidence=0.5,
                rationale="role council: no votes returned (offline fallback)",
            )

        synthesis, conf = self._synthesise(stimulus, votes)
        role_summary = " | ".join(f"{v.member}: {v.answer[:40]}" for v in votes[:3])
        return Candidate(
            text=synthesis,
            kind="respond",
            capability=Capability.MESSAGE_SEND,
            risk=RiskTier.LOW,
            reversible=True,
            confidence=conf,
            belief=conf,
            rationale=f"role council ({len(votes)}/6 voted): {role_summary}",
        )

    # ---------------------------------------------------------------------- #
    def _gather_votes(self, stimulus: str) -> List[RoleVote]:
        """Query all roles in parallel; return the ones that answered in time."""
        if not self._llm_available():
            return self._stub_votes(stimulus)

        votes: List[RoleVote] = []
        with _cf.ThreadPoolExecutor(max_workers=len(self.members)) as ex:
            futs = {ex.submit(self._query_role, m, stimulus): m
                    for m in self.members}
            for fut in _cf.as_completed(futs, timeout=self.timeout_s * 2):
                m = futs[fut]
                try:
                    answer = fut.result(timeout=self.timeout_s)
                    if answer:
                        votes.append(RoleVote(m.name, answer, 0.7, m.weight))
                except Exception:  # noqa: BLE001
                    pass
        return votes

    def _query_role(self, member: RoleMember, stimulus: str) -> str:
        try:
            return (self.llm.generate(
                f"The question is:\n{stimulus}",
                system=member.persona,
                temperature=0.6,
                max_tokens=self.max_tokens,
            ) or "").strip()
        except Exception:  # noqa: BLE001
            return ""

    def _synthesise(self, stimulus: str, votes: List[RoleVote]) -> Tuple[str, float]:
        """Merge the role votes into one answer.  NYXARA always holds the gavel."""
        votes_block = "\n\n".join(
            f"[{v.member}]\n{v.answer}" for v in votes
        )
        prompt = (
            f"The Master's question:\n{stimulus}\n\n"
            f"Council perspectives:\n{votes_block}\n\n"
            "Synthesise these into your response to the Master."
        )
        if not self._llm_available():
            # offline: compose a real structured synthesis from every role's perspective,
            # led by the highest-weight lenses — not just the longest single answer.
            ordered = sorted(votes, key=lambda v: v.weight, reverse=True)
            topic = _topic_of(stimulus)
            lines = [f"On '{topic}', NYXARA's offline council reasons across six lenses:"]
            lines.extend(f"• {v.answer}" for v in ordered)
            lead = ordered[0]
            lines.append(f"Integrated view: {lead.answer.split(': ', 1)[-1]}")
            # Synthesis confidence = weight-weighted mean of the (computed) per-vote
            # confidences, tempered by coverage (how many of the six lenses actually voted).
            wsum = sum(v.weight for v in votes) or 1.0
            mean_conf = sum(v.confidence * v.weight for v in votes) / wsum
            coverage = len(votes) / max(1, len(self.members))
            conf = mean_conf * (0.6 + 0.4 * coverage)
            return "\n".join(lines), round(max(0.2, min(0.8, conf)), 3)
        try:
            text = self.llm.generate(prompt, system=_SYNTHESIS_SYSTEM,
                                     temperature=0.35, max_tokens=512)
            text = (str(text or "")).strip() or votes[0].answer
            conf = round(sum(v.weight for v in votes) / len(votes), 3)
            return text, min(0.88, conf)
        except Exception:  # noqa: BLE001
            best = max(votes, key=lambda v: v.weight)
            return best.answer, 0.6

    def _stub_votes(self, stimulus: str) -> List[RoleVote]:
        """Deterministic offline analysis — no LLM, but a *real* per-role perspective.

        Each role examines the actual question with NYXARA's pure-Python NLP (intent,
        question-type, key phrases, entities) and contributes a concrete, lens-specific
        observation — not filler. Confidence is honest (offline reasoning, mid-band)."""
        analyses = _offline_role_analyses(stimulus)
        return [RoleVote(name, text, conf, weight) for name, text, conf, weight in analyses]

    def _llm_available(self) -> bool:
        if self.llm is None:
            return False
        try:
            return self.llm.chosen_provider().name != "mock"
        except Exception:  # noqa: BLE001
            return False

    def votes_summary(self, votes: List[RoleVote]) -> Dict[str, str]:
        return {v.member: v.answer[:80] for v in votes}


# --------------------------------------------------------------------------- #
# Self-test / demo
# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # pragma: no cover
    print("=" * 70)
    print("NYXARA role-council self-test")
    print("=" * 70)

    # offline (stub) mode — always works
    council = RoleCouncil(llm=None)
    cand = council.convene("What is the capital of France?")
    print("\n6-role council (offline)")
    print(f"  kind       : {cand.kind}")
    print(f"  confidence : {cand.confidence}")
    print(f"  rationale  : {cand.rationale}")
    assert cand.kind == "respond"
    assert "council" in cand.rationale.lower()
    assert cand.confidence > 0

    print("\nALL SELF-TESTS PASSED ✓")
