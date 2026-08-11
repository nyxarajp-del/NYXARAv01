"""NYXARA · nyx/workspace.py — the conscious bottleneck (💡, NYX V.01 pillar 3).

This is where the specialists stop being a list and become a *mind*. Each one bids; the bids
compete on salience; coalitions form between bids that share tags; and exactly one coalition
crosses the narrow bottleneck and is **broadcast** as the momentary content of awareness.

The competition itself is not reimplemented here — :class:`~nyxara.kernel.workspace.GlobalWorkspace`
already implements Baars/Dehaene Global Workspace Theory properly (salience weighting, coalition
synergy, habituation, inhibition-of-return, a real broadcast). What this module adds is the two
things that make it *NYX's* workspace rather than a generic one:

* **specialists as sources** — the cast from :mod:`nyxara.nyx.modules` is run and adapted into
  workspace :class:`~nyxara.kernel.workspace.Content`;
* **learned attention** — each bid's ``source_priority`` is scaled by that specialist's
  *measured* reliability (:mod:`nyxara.nyx.metacog`), so being repeatedly wrong actually costs a
  faculty its access to consciousness. Nothing here is hand-tuned after the first turn.

Honest framing: this simulates the *architecture* Global Workspace Theory describes — competition
into a narrow bottleneck, then broadcast. It is not, and does not claim to be, evidence of
phenomenal experience. What is real and measurable is that only one thing wins per cycle, that
the winner is broadcast to everything listening, and that losing is remembered.

Fail-soft: a specialist that raises simply does not bid; a workspace failure yields an empty
deliberation rather than a broken turn.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from nyxara.nyx.modules import Proposal, Situation, SpecialistModule, default_specialists

__all__ = ["Deliberation", "NyxWorkspace"]


def _clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else float(x))


@dataclass
class Deliberation:
    """One pass of the workspace: everyone who bid, and the one that reached awareness."""

    winner: Optional[Proposal] = None
    proposals: List[Proposal] = field(default_factory=list)
    salience: float = 0.0
    coalition: List[str] = field(default_factory=list)
    cycle: int = 0
    starved: bool = False            # nobody crossed the access threshold — a real outcome

    @property
    def sources(self) -> List[str]:
        return [p.source for p in self.proposals]

    def to_dict(self) -> Dict[str, Any]:
        return {"winner": self.winner.source if self.winner else None,
                "content": self.winner.content if self.winner else "",
                "salience": round(self.salience, 4), "coalition": self.coalition,
                "cycle": self.cycle, "starved": self.starved,
                "proposals": [p.to_dict() for p in self.proposals]}


class NyxWorkspace:
    """Runs the specialists, lets them compete, and broadcasts the winner."""

    def __init__(self, *, specialists: Optional[Sequence[SpecialistModule]] = None,
                 metacog: Any = None, capacity: int = 1, access_threshold: float = 1.0,
                 bus: Any = None, workspace: Any = None) -> None:
        self.specialists: List[SpecialistModule] = list(
            specialists if specialists is not None else default_specialists())
        self.metacog = metacog
        self.cycles = 0
        # ``workspace`` lets a *shared* GlobalWorkspace be injected instead of a private one.
        # A mind with two bottlenecks is two minds; :mod:`nyxara.nyx.monad` passes the single
        # instance every substrate bids into. Left None, this builds its own exactly as before.
        self._gw = workspace if workspace is not None else self._build_workspace(
            capacity=capacity, access_threshold=access_threshold, bus=bus)

    def add_specialist(self, specialist: SpecialistModule) -> bool:
        """Register another bidder. Returns False when the name is already present, so joining
        the same substrate twice cannot give it two votes."""
        if any(s.name == specialist.name for s in self.specialists):
            return False
        self.specialists.append(specialist)
        return True

    @staticmethod
    def _build_workspace(*, capacity: int, access_threshold: float, bus: Any) -> Any:
        try:
            from nyxara.kernel.workspace import GlobalWorkspace
            return GlobalWorkspace(capacity=max(1, int(capacity)),
                                   access_threshold=float(access_threshold), bus=bus)
        except Exception:  # noqa: BLE001 — without it we fall back to a plain argmax
            return None

    # ---- the cycle -------------------------------------------------------- #
    def deliberate(self, situation: Situation, *, goals: Optional[Dict[str, float]] = None
                   ) -> Deliberation:
        """Collect every specialist's bid, run the competition, return what reached awareness."""
        out = Deliberation(cycle=self.cycles)
        try:
            self.cycles += 1
            out.cycle = self.cycles
            out.proposals = [p for p in
                             (s.safe_consider(situation) for s in self.specialists)
                             if p is not None]
            if not out.proposals:
                out.starved = True
                return out

            if self._gw is None:
                return self._fallback(out)

            from nyxara.kernel.workspace import Content
            self._gw.clear()
            if goals:
                self._gw.set_goals(dict(goals))

            by_id: Dict[str, Proposal] = {}
            for proposal in out.proposals:
                # Trust modulates the *bid itself*, not just a tie-break: a faculty she has
                # measured to be wrong registers more weakly, which is what makes the
                # metacog feedback loop actually close.
                strength = _clamp01(proposal.confidence * self._trust(proposal.source))
                content = Content(
                    payload=proposal.content, source=proposal.source,
                    activation=strength,
                    novelty=_clamp01(proposal.novelty),
                    urgency=_clamp01(proposal.urgency),
                    goal_relevance=strength,
                    emotional_intensity=0.0,
                    source_priority=self._priority(proposal),
                    # The workspace's habituation and inhibition-of-return key off
                    # ``source:tags``. With a *fixed cast* bidding every cycle that would
                    # suppress whichever faculty was right last time, rotating the winner for
                    # reasons unrelated to the question. Folding a fingerprint of the content
                    # into the tags restores the mechanism's actual purpose: repeating the same
                    # thing is damped, having something new to say is not penalised.
                    tags=frozenset(proposal.tags) | {self._fingerprint(proposal.content)})
                by_id[content.item_id] = proposal
                self._gw.submit(content)

            broadcasts = self._gw.cycle()
            if not broadcasts:
                out.starved = True
                return self._fallback(out)      # awareness starved — still answer, honestly

            broadcast = broadcasts[0]
            out.winner = by_id.get(broadcast.winner.item_id)
            out.salience = float(broadcast.salience)
            out.coalition = sorted({c.source for c in broadcast.coalition.members})
            if out.winner is None:              # defensive: never return a broadcast we can't map
                return self._fallback(out)
            return self._apply_verifiable_law(out)
        except Exception:  # noqa: BLE001 — deliberation degrades, it does not crash
            return self._fallback(out)

    @staticmethod
    def _fingerprint(content: str) -> str:
        """A short, stable tag for *what was said*, so habituation tracks content not faculty."""
        try:
            import hashlib
            digest = hashlib.sha1(str(content or "").strip().lower().encode()).hexdigest()
            return f"said:{digest[:12]}"
        except Exception:  # noqa: BLE001
            return "said:?"

    def _trust(self, source: str) -> float:
        """How much her *measured* record says this faculty is worth listening to (0.5 … 1.5)."""
        try:
            return float(self.metacog.weight_for(source)) if self.metacog is not None else 1.0
        except Exception:  # noqa: BLE001
            return 1.0

    def _priority(self, proposal: Proposal) -> float:
        """Source priority = the specialist's declared prior, scaled by verifiability."""
        try:
            base = 0.5
            for spec in self.specialists:
                if spec.name == proposal.source:
                    base = float(spec.priority)
                    break
            if proposal.verifiable:
                base *= 1.5
            return max(0.0, min(2.0, base))
        except Exception:  # noqa: BLE001
            return 0.5

    def _apply_verifiable_law(self, out: Deliberation) -> Deliberation:
        """The control law, enforced rather than merely weighted: **verifiable beats probabilistic**.

        Salience is a good model of attention, but it is a *blend* — a sufficiently loud guess can
        out-shout a checked proof, and that is precisely the failure this repo refuses (see
        ``mind/faculties.py``: a verifiable engine that fits wins over a probabilistic one). So
        when anything on the table is actually *derived and checked*, it takes the broadcast.
        Among several verified proposals the most confident wins; the competition still decides
        everything else.
        """
        try:
            if out.winner is None or out.winner.verifiable:
                return out
            verified = [p for p in out.proposals if p.verifiable]
            if not verified:
                return out
            best = max(verified, key=lambda p: p.confidence)
            out.winner = best
            out.coalition = sorted(set(out.coalition) | {best.source})
            return out
        except Exception:  # noqa: BLE001
            return out

    def _fallback(self, out: Deliberation) -> Deliberation:
        """No workspace, or nothing crossed the bottleneck: pick the best bid and say so."""
        try:
            if not out.proposals:
                out.starved = True
                return out
            out.winner = max(out.proposals,
                             key=lambda p: (p.verifiable,
                                            p.confidence * self._trust(p.source)
                                            * self._priority(p)))
            out.salience = out.winner.confidence
            out.coalition = [out.winner.source]
            return out
        except Exception:  # noqa: BLE001
            return out

    # ---- introspection ---------------------------------------------------- #
    def stream(self, n: int = 5) -> List[Dict[str, Any]]:
        """The recent stream of consciousness — what won, cycle by cycle."""
        try:
            if self._gw is None:
                return []
            return [b.to_dict() for b in self._gw.stream(n)]
        except Exception:  # noqa: BLE001
            return []

    def stats(self) -> Dict[str, Any]:
        try:
            out: Dict[str, Any] = {
                "cycles": self.cycles,
                "specialists": [s.name for s in self.specialists],
                "unavailable": [s.name for s in self.specialists if not s.available()],
            }
            if self._gw is not None:
                out["workspace"] = self._gw.metrics.snapshot()
            return out
        except Exception:  # noqa: BLE001
            return {}
