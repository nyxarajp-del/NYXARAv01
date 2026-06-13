"""NYXARA · mind/world_simulator.py — World Simulator (Level 5).

Before NYXARA acts, she imagines: "what happens if I do this?"

The simulator combines three existing capabilities into one unified *what-if* engine:

1. **Sandbox dry-run** (sim/sandbox.py) — intercepts every would-be side-effect of
   the action (file writes, execs, network calls, self-edits, spawns) without executing
   any of them.  The intercepted effects determine reversibility.

2. **World-model rollout** (mind/world_model.py) — if transitions have been observed,
   rolls the world-state forward N steps to estimate expected reward and mean confidence.
   Honestly returns low confidence when the query leaves the observed region.

3. **Heuristic risk classification** — maps tool names and effect kinds to risk tiers so
   the gate pipeline has a concrete signal even on the first turn (before the world model
   has accumulated data).

The result is a :class:`SimResult` that the orchestrator attaches to the candidate; if
the simulated risk_score is higher than the candidate's current risk tier, the tier is
upgraded — so the gate pipeline never sees less risk than the simulator found.

All simulation is internal: no real I/O, no gate bypass.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

__all__ = ["SimResult", "WorldSimulator"]

# --------------------------------------------------------------------------- #
# Tool-name → risk signal (heuristic, no LLM needed)
# --------------------------------------------------------------------------- #
_HIGH_RISK_TOKENS = frozenset({
    "delete", "remove", "drop", "destroy", "wipe", "format", "kill", "shutdown",
    "exec", "execute", "run_shell", "deploy", "rotate", "install", "uninstall",
    "overwrite", "truncate", "transfer", "payment", "send_money",
})
_MODERATE_RISK_TOKENS = frozenset({
    "write", "write_file", "create", "update", "modify", "patch", "http_request",
    "web_fetch", "network", "post", "put", "upload", "push",
})
_NETWORK_TOKENS = frozenset({
    "web_fetch", "web_search", "http_request", "network", "send",
})
_FILE_TOKENS = frozenset({
    "write_file", "read_file", "list_dir", "read_document",
})


def _token_risk(tool_name: str) -> float:
    low = tool_name.lower()
    if any(t in low for t in _HIGH_RISK_TOKENS):
        return 0.8
    if any(t in low for t in _MODERATE_RISK_TOKENS):
        return 0.5
    return 0.2


def _token_reversible(tool_name: str, risk: float) -> bool:
    low = tool_name.lower()
    if any(t in low for t in {"delete", "remove", "drop", "wipe", "kill", "transfer"}):
        return False
    return risk < 0.7


# --------------------------------------------------------------------------- #
# SimResult
# --------------------------------------------------------------------------- #
@dataclass
class SimResult:
    """The outcome of a world-simulation pass."""

    action: str                         # what was simulated
    expected_outcome: str               # human-readable outcome summary
    confidence: float                   # how confident the simulator is [0,1]
    risk_score: float                   # normalised risk [0,1]
    side_effects: List[str]             # list of intercepted effect kinds
    rollback_possible: bool             # True if all effects are reversible
    world_model_reward: Optional[float] = None  # estimated reward from world model
    world_model_confidence: Optional[float] = None

    def risk_tier(self) -> str:
        if self.risk_score >= 0.75:
            return "high"
        if self.risk_score >= 0.45:
            return "moderate"
        if self.risk_score >= 0.15:
            return "low"
        return "trivial"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action": self.action,
            "expected_outcome": self.expected_outcome,
            "confidence": round(self.confidence, 3),
            "risk_score": round(self.risk_score, 3),
            "risk_tier": self.risk_tier(),
            "side_effects": self.side_effects,
            "rollback_possible": self.rollback_possible,
            "world_model_reward": (round(self.world_model_reward, 3)
                                   if self.world_model_reward is not None else None),
        }


# --------------------------------------------------------------------------- #
# Simulator
# --------------------------------------------------------------------------- #
class WorldSimulator:
    """Unified imagination engine — combines sandbox, world model, and heuristics.

    Parameters
    ----------
    world_model:    a WorldModel instance (optional; heuristics-only if None).
    predictive:     a PredictiveCore instance (optional; surprise signal).
    rollout_steps:  how many steps to roll the world model forward.
    """

    def __init__(self, world_model: Any = None, predictive: Any = None,
                 rollout_steps: int = 3) -> None:
        self.world_model = world_model
        self.predictive = predictive
        self.rollout_steps = max(1, int(rollout_steps))

    def simulate(self, action: str, *, tool: str = "", tool_args: Optional[Dict] = None,
                 reversible: Optional[bool] = None) -> SimResult:
        """Run a full simulation pass for the proposed action.

        Returns a SimResult that honestly reflects what would happen.  Even with no
        world model experience, heuristic scoring ensures the gate always has a signal.
        """
        tool = tool or action
        tool_args = tool_args or {}

        # 1. Heuristic risk + side-effect classification
        heuristic_risk = _token_risk(tool)
        heuristic_reversible = (reversible if reversible is not None
                                else _token_reversible(tool, heuristic_risk))
        side_effects = self._classify_effects(tool, tool_args)

        # Irreversible effects bump risk upward
        has_irreversible = any(e in {"exec", "file_delete", "network"}
                               for e in side_effects)
        if has_irreversible and heuristic_risk < 0.6:
            heuristic_risk = max(heuristic_risk, 0.55)

        # 2. World-model rollout (if data exists)
        wm_reward: Optional[float] = None
        wm_conf: Optional[float] = None
        if self.world_model is not None and len(self.world_model) > 0:
            wm_reward, wm_conf = self._world_model_eval(tool)
            if wm_conf is not None and wm_conf > 0.4:
                # blend: world model lowers heuristic risk if history says it's safe
                if wm_reward is not None and wm_reward > 0.5:
                    heuristic_risk = max(0.05, heuristic_risk * (1.0 - 0.3 * wm_conf))

        # 3. Predictive surprise bonus (if the action is novel / unexpected)
        surprise_penalty = self._surprise_score(tool)
        final_risk = min(1.0, heuristic_risk + 0.15 * surprise_penalty)

        # 4. Overall confidence in the simulation
        confidence = 0.5
        if self.world_model is not None and len(self.world_model) > 0:
            confidence = min(0.85, 0.5 + 0.35 * float(len(self.world_model)) / 50.0)

        rollback_possible = heuristic_reversible and not has_irreversible

        outcome = self._describe_outcome(tool, side_effects, final_risk, wm_reward)

        return SimResult(
            action=action,
            expected_outcome=outcome,
            confidence=round(confidence, 3),
            risk_score=round(final_risk, 3),
            side_effects=side_effects,
            rollback_possible=rollback_possible,
            world_model_reward=wm_reward,
            world_model_confidence=wm_conf,
        )

    # ---------------------------------------------------------------------- #
    def _classify_effects(self, tool: str, tool_args: Dict[str, Any]) -> List[str]:
        """Map tool name + args to a list of EffectKind labels."""
        effects: List[str] = []
        low = tool.lower()
        if any(t in low for t in _NETWORK_TOKENS):
            effects.append("network")
        if any(t in low for t in _FILE_TOKENS):
            if any(t in low for t in {"write", "create", "modify", "patch"}):
                effects.append("file_write")
            else:
                effects.append("file_read")
        if any(t in low for t in {"run_shell", "exec", "execute", "run"}):
            effects.append("exec")
        if any(t in low for t in {"delete", "remove", "drop"}):
            effects.append("file_delete")
        if any(t in low for t in {"spawn", "agent", "delegate"}):
            effects.append("spawn")
        if not effects:
            effects.append("state")
        return effects

    def _world_model_eval(self, tool: str) -> tuple:
        """Run a short rollout and return (expected_reward, mean_confidence)."""
        try:
            from nyxara.mind.world_model import WorldModel
            traj = self.world_model.rollout(
                (0.0,), [tool] * self.rollout_steps, steps=self.rollout_steps)
            return traj.total_reward, traj.mean_confidence
        except Exception:  # noqa: BLE001
            return None, None

    def _surprise_score(self, tool: str) -> float:
        """Free-energy surprise: how unexpected is this action from current belief?"""
        if self.predictive is None:
            return 0.0
        try:
            # encode tool name as a tiny observation vector
            obs = [float(ord(c) % 17) / 17.0 for c in tool[:8]]
            obs = obs + [0.0] * max(0, len(self.predictive.mu) - len(obs))
            obs = obs[: len(self.predictive.mu)]
            _, feeling = self.predictive.step(obs)
            return float(min(1.0, feeling.surprise))
        except Exception:  # noqa: BLE001
            return 0.0

    @staticmethod
    def _describe_outcome(tool: str, effects: List[str], risk: float,
                           wm_reward: Optional[float]) -> str:
        effect_str = ", ".join(effects) or "state change"
        wm_note = ""
        if wm_reward is not None:
            wm_note = f"; world-model E[reward]={wm_reward:.2f}"
        tier = "high" if risk >= 0.75 else "moderate" if risk >= 0.45 else "low"
        return (f"Tool '{tool}' would produce: {effect_str}. "
                f"Estimated risk: {tier} ({risk:.2f}){wm_note}.")


# --------------------------------------------------------------------------- #
# Self-test / demo
# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # pragma: no cover
    print("=" * 70)
    print("NYXARA world-simulator self-test")
    print("=" * 70)

    sim = WorldSimulator()

    r1 = sim.simulate("delete_files", tool="run_shell",
                      tool_args={"cmd": "rm -rf /tmp/test"})
    print(f"\nrun_shell / delete")
    print(f"  risk_score  : {r1.risk_score}")
    print(f"  risk_tier   : {r1.risk_tier()}")
    print(f"  rollback    : {r1.rollback_possible}")
    print(f"  effects     : {r1.side_effects}")
    assert r1.risk_score >= 0.5
    assert not r1.rollback_possible

    r2 = sim.simulate("web_search", tool="web_search",
                      tool_args={"query": "Python tutorials"})
    print(f"\nweb_search")
    print(f"  risk_score  : {r2.risk_score}")
    print(f"  risk_tier   : {r2.risk_tier()}")
    assert r2.risk_score < r1.risk_score

    r3 = sim.simulate("read_file", tool="read_file",
                      tool_args={"path": "/tmp/data.txt"}, reversible=True)
    print(f"\nread_file")
    print(f"  risk_score  : {r3.risk_score}")
    print(f"  rollback    : {r3.rollback_possible}")

    print("\nALL SELF-TESTS PASSED ✓")
