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
            # offline: weighted-average confidence + longest answer as text
            best = max(votes, key=lambda v: v.weight * len(v.answer))
            conf = sum(v.weight for v in votes) / (len(votes) * max(1, len(self.members)))
            return best.answer, round(min(0.85, conf), 3)
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
        """Deterministic offline stubs — no LLM, just role-tagged observations."""
        stubs = [
            (m.name, f"[{m.name}] Considering '{stimulus[:40]}': a thoughtful response "
             f"grounded in {m.name.lower()} principles is needed.", 0.5, m.weight)
            for m in self.members
        ]
        return [RoleVote(n, a, c, w) for n, a, c, w in stubs]

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
