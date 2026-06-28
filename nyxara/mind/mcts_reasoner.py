"""NYXARA · mind/mcts_reasoner.py — Monte Carlo Tree Search deep reasoning (⬆, Pillar B4).

The kernel reasoner used to turn a stimulus into a decision in a single (or N-sampled)
LLM call. A single shot commits before it reasons; N-sample voting is breadth without
depth. This module gives NYXARA genuine **search over reasoning**: she branches the
problem into a tree of candidate reasoning steps, simulates each branch forward to a
terminal answer, scores that answer with an *independent* value function, and
backpropagates the value so the most-promising line of thought accrues the most visits —
the four canonical MCTS phases:

1. **Selection** — descend from the root by the UCT rule
   (``Q + c·sqrt(ln N_parent / N_child)``) to the most-promising expandable node.
2. **Expansion** — ask the model for several *distinct* next reasoning moves given the
   trace so far, and graft them on as children (this is where the problem branches).
3. **Simulation** — roll the chosen branch forward a few quick steps to a terminal
   decision, then score it in ``[0,1]`` by an answer-quality verifier, a deterministic
   verifiable check (exact math/logic beats any guess), and a cognitive-bias critic.
4. **Backpropagation** — propagate that value up the path, incrementing visit/value stats.

The decision returned is the highest-scoring terminal the search found; the tree merely
focuses compute on the branches worth deepening. It is the **same decision dict** the
deliberate reasoner returns, so safety is untouched — the LLM only ever *proposes*; the
kernel still disposes through every gate. With no real provider (or any failure) the search
raises :class:`MCTSUnavailable` and the caller falls back to the deliberate / single-shot
path, so a turn never crashes and behaves identically offline.

Provider-agnostic (needs only ``llm.generate`` / ``llm.generate_json``); pure stdlib plus
in-repo imports otherwise. Stateless per call, like the faculties it composes.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

__all__ = ["MCTSReasoner", "MCTSResult", "MCTSUnavailable"]

# (prompt, answer_text) -> quality score in [0,1]; reused from the deliberate reasoner.
Verifier = Callable[[str, str], float]


class MCTSUnavailable(Exception):
    """Raised when the search cannot produce any parseable decision (caller falls back)."""


# --------------------------------------------------------------------------- #
# Tree node
# --------------------------------------------------------------------------- #
@dataclass
class _Node:
    """One reasoning state: the ordered partial reasoning trace + its search statistics."""

    steps: List[str]                              # reasoning moves taken to reach this node
    parent: Optional["_Node"] = None
    children: List["_Node"] = field(default_factory=list)
    untried: Optional[List[str]] = None           # candidate next moves, generated lazily
    visits: int = 0
    value: float = 0.0                            # cumulative backpropagated value

    @property
    def q(self) -> float:
        return self.value / self.visits if self.visits else 0.0

    def expandable(self) -> bool:
        # untried is None until expansion is attempted; an empty list means fully expanded
        return self.untried is None or bool(self.untried)


# --------------------------------------------------------------------------- #
# Result
# --------------------------------------------------------------------------- #
@dataclass
class MCTSResult:
    """The outcome of one MCTS reasoning pass: the chosen decision plus an audit trace."""

    decision: Dict[str, Any]
    score: float = 0.0
    iterations: int = 0
    nodes: int = 0
    best_trace: List[str] = field(default_factory=list)
    rlsp_rounds: int = 0

    def trace(self) -> str:
        bits = [f"MCTS over {self.nodes} nodes / {self.iterations} simulations "
                f"(best value {self.score:.2f})"]
        if self.rlsp_rounds:
            bits.append(f"adversarially hardened in {self.rlsp_rounds} round(s)")
        return "; ".join(bits)


# --------------------------------------------------------------------------- #
# The searcher
# --------------------------------------------------------------------------- #
class MCTSReasoner:
    """Run Monte Carlo Tree Search over reasoning steps, returning a decision dict."""

    _EXPAND_INSTRUCTIONS = (
        "You are NYXARA reasoning toward the best way to serve the Master. Given the problem "
        "and the reasoning so far, propose {k} DISTINCT and genuinely different next reasoning "
        "moves — each a single short sentence naming a concrete approach, sub-question, check, "
        "or consideration to pursue next. Do NOT answer yet. Reply with ONLY a JSON array of "
        "{k} short strings."
    )
    _ROLLOUT_INSTRUCTIONS = (
        "Continue this reasoning by ONE concise step that moves toward the answer. Reply with a "
        "single short sentence only — no preamble, no answer yet."
    )

    def __init__(self, llm: Any, *, decision_instructions: str,
                 iterations: int = 24, max_children: int = 4, rollout_depth: int = 2,
                 c_puct: float = 1.4, max_seconds: float = 20.0,
                 temperature: float = 0.5, max_tokens: int = 4096,
                 think_tokens: int = 512, critic: Any = None,
                 verifier: Optional[Verifier] = None, rlsp: Any = None) -> None:
        self.llm = llm
        self.decision_instructions = decision_instructions
        self.iterations = max(1, int(iterations))
        self.max_children = max(1, int(max_children))
        self.rollout_depth = max(0, int(rollout_depth))
        self.c_puct = max(0.0, float(c_puct))
        self.max_seconds = max(0.5, float(max_seconds))
        self.temperature = max(0.0, float(temperature))
        self.max_tokens = max(1, int(max_tokens))
        self.think_tokens = max(64, int(think_tokens))
        self.critic = critic
        self.verifier = verifier
        self.rlsp = rlsp
        self._nodes = 0
        self._best: Optional[Dict[str, Any]] = None
        self._best_score = -1.0
        self._best_trace: List[str] = []

    # ------------------------------------------------------------------ #
    # public entry point
    # ------------------------------------------------------------------ #
    def search(self, *, stimulus: str, context: str = "",
               system: Optional[str] = None) -> MCTSResult:
        """Grow the tree under the budget and return the best terminal decision found."""
        self._nodes = 1
        self._best, self._best_score, self._best_trace = None, -1.0, []
        root = _Node(steps=[])
        deadline = time.monotonic() + self.max_seconds

        sims = 0
        for _ in range(self.iterations):
            if time.monotonic() >= deadline:
                break
            leaf = self._select(root)
            child = self._expand(leaf, stimulus, context, system) or leaf
            reward = self._simulate(child, stimulus, context, system)
            self._backprop(child, reward)
            sims += 1

        if self._best is None:
            # Not a single branch produced a parseable terminal decision — let the caller
            # fall back to its own deliberate / single-shot path.
            raise MCTSUnavailable("MCTS produced no parseable decision")

        decision = self._best
        result = MCTSResult(decision=decision, score=self._best_score, iterations=sims,
                            nodes=self._nodes, best_trace=list(self._best_trace))

        # Harden the winning answer against its own logical flaws (Generator/Discriminator).
        result = self._maybe_harden(stimulus, result)
        self._annotate(result)
        return result

    # ------------------------------------------------------------------ #
    # 1. selection — descend by UCT to the most-promising expandable node
    # ------------------------------------------------------------------ #
    def _select(self, root: _Node) -> _Node:
        node = root
        while not node.expandable() and node.children:
            node = self._uct_child(node)
        return node

    def _uct_child(self, node: _Node) -> _Node:
        log_n = math.log(node.visits + 1.0)
        best, best_u = node.children[0], -1e18
        for ch in node.children:
            exploit = ch.q
            explore = self.c_puct * math.sqrt(log_n / (ch.visits + 1e-9))
            u = exploit + explore
            if u > best_u:
                best, best_u = ch, u
        return best

    # ------------------------------------------------------------------ #
    # 2. expansion — branch the problem into candidate next reasoning steps
    # ------------------------------------------------------------------ #
    def _expand(self, node: _Node, stimulus: str, context: str,
                system: Optional[str]) -> Optional[_Node]:
        if node.untried is None:
            node.untried = self._candidate_steps(node, stimulus, context, system)
        if not node.untried:
            return None
        step = node.untried.pop(0)
        child = _Node(steps=node.steps + [step], parent=node)
        node.children.append(child)
        self._nodes += 1
        return child

    def _candidate_steps(self, node: _Node, stimulus: str, context: str,
                         system: Optional[str]) -> List[str]:
        prompt = (f"The Master says:\n{stimulus}{context}\n\n"
                  f"{self._trace_block(node)}\n\n"
                  f"{self._EXPAND_INSTRUCTIONS.format(k=self.max_children)}")
        try:
            data = self.llm.generate_json(prompt, system=system,
                                          temperature=min(1.0, self.temperature + 0.2),
                                          max_tokens=self.think_tokens)
        except Exception:  # noqa: BLE001 — a flaky model simply yields no branches here
            return []
        steps = _as_str_list(data)
        # de-dup while preserving order; cap to the branching factor
        seen, out = set(), []
        for s in steps:
            key = s.lower().strip()
            if key and key not in seen:
                seen.add(key)
                out.append(s.strip())
            if len(out) >= self.max_children:
                break
        return out

    # ------------------------------------------------------------------ #
    # 3. simulation — roll forward to a terminal decision and score it
    # ------------------------------------------------------------------ #
    def _simulate(self, node: _Node, stimulus: str, context: str,
                  system: Optional[str]) -> float:
        steps = list(node.steps)
        for _ in range(self.rollout_depth):
            nxt = self._rollout_step(steps, stimulus, context, system)
            if not nxt:
                break
            steps.append(nxt)
        decision = self._terminal_decision(steps, stimulus, context, system)
        if decision is None:
            return 0.0
        score = self._value(stimulus, decision)
        if score > self._best_score:
            self._best, self._best_score, self._best_trace = decision, score, steps
        return score

    def _rollout_step(self, steps: List[str], stimulus: str, context: str,
                      system: Optional[str]) -> str:
        prompt = (f"The Master says:\n{stimulus}{context}\n\n"
                  f"{self._trace_block_from(steps)}\n\n{self._ROLLOUT_INSTRUCTIONS}")
        try:
            text = self.llm.generate(prompt, system=system,
                                     temperature=self.temperature, max_tokens=128)
            return str(text or "").strip().split("\n", 1)[0][:280]
        except Exception:  # noqa: BLE001 — rollout is advisory; stop extending on failure
            return ""

    def _terminal_decision(self, steps: List[str], stimulus: str, context: str,
                           system: Optional[str]) -> Optional[Dict[str, Any]]:
        think = self._trace_block_from(steps)
        prompt = (f"The Master says:\n{stimulus}{context}\n\n{think}\n\n"
                  f"Using the reasoning above (do not repeat it verbatim), "
                  f"{self.decision_instructions}")
        try:
            data = self.llm.generate_json(prompt, system=system,
                                          temperature=min(self.temperature, 0.3),
                                          max_tokens=self.max_tokens)
        except Exception:  # noqa: BLE001
            return None
        return data if isinstance(data, dict) and data.get("text") is not None else None

    # ------------------------------------------------------------------ #
    # value function — verifier + deterministic exact check + bias critic
    # ------------------------------------------------------------------ #
    def _value(self, stimulus: str, decision: Dict[str, Any]) -> float:
        text = str(decision.get("text", ""))
        # base: an independent answer-quality verifier, else the model's stated confidence
        if self.verifier is not None:
            try:
                score = float(self.verifier(stimulus, text))
            except Exception:  # noqa: BLE001 — fall back to self-reported confidence
                score = _as_float(decision.get("confidence"), 0.5)
        else:
            score = _as_float(decision.get("confidence"), 0.5)
        score = max(0.0, min(1.0, score))
        # exact, verifiable answers beat any neural guess — a strong, earned bonus
        score = min(1.0, score + 0.25 * self._verifiable_bonus(stimulus, text))
        # cognitive-bias / calibration penalty from the deterministic Critic
        score = max(0.0, score - 0.1 * self._critic_penalty(decision))
        return score

    @staticmethod
    def _verifiable_bonus(stimulus: str, text: str) -> float:
        """1.0 when an exact faculty answer is present in the text, else 0.0 (best-effort)."""
        exact: Optional[str] = None
        try:
            from nyxara.mind.reasoning_chain import solve_chain
            chain = solve_chain(stimulus)
            if chain is not None:
                exact = str(chain.answer)
        except Exception:  # noqa: BLE001
            exact = None
        if exact is None:
            try:
                from nyxara.mind.reasoning_faculties import solve_with_faculties
                res = solve_with_faculties(stimulus)
                if res is not None:
                    exact = str(res[0])
            except Exception:  # noqa: BLE001
                exact = None
        if not exact:
            return 0.0
        return 1.0 if exact.strip().lower() in (text or "").strip().lower() else 0.0

    def _critic_penalty(self, decision: Dict[str, Any]) -> float:
        if self.critic is None:
            return 0.0
        try:
            from nyxara.mind.critique import CritiqueContext
            ctx = CritiqueContext(
                claim=str(decision.get("text", "")),
                confidence=_as_float(decision.get("confidence"), 0.7),
                rationale=str(decision.get("rationale", "")),
                risk=_risk_to_float(decision.get("risk")),
                reversibility=1.0 if decision.get("reversible", True) else 0.2)
            crit = self.critic.critique(ctx)
            return min(1.0, 0.25 * len(getattr(crit, "findings", []) or []))
        except Exception:  # noqa: BLE001 — the critic is advisory
            return 0.0

    # ------------------------------------------------------------------ #
    # 4. backpropagation
    # ------------------------------------------------------------------ #
    @staticmethod
    def _backprop(node: _Node, reward: float) -> None:
        cur: Optional[_Node] = node
        while cur is not None:
            cur.visits += 1
            cur.value += reward
            cur = cur.parent

    # ------------------------------------------------------------------ #
    # adversarial hardening of the winning answer (Generator/Discriminator)
    # ------------------------------------------------------------------ #
    def _maybe_harden(self, stimulus: str, result: MCTSResult) -> MCTSResult:
        if self.rlsp is None:
            return result
        try:
            hardened = self.rlsp.harden(stimulus, result.decision)
        except Exception:  # noqa: BLE001 — hardening is advisory, never fatal
            return result
        if hardened is None:
            return result
        result.decision = hardened.decision
        result.rlsp_rounds = hardened.rounds
        return result

    # ------------------------------------------------------------------ #
    # helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _trace_block(node: _Node) -> str:
        return MCTSReasoner._trace_block_from(node.steps)

    @staticmethod
    def _trace_block_from(steps: List[str]) -> str:
        if not steps:
            return "Reasoning so far: (none yet — this is the first step)."
        lines = "\n".join(f"  {i + 1}. {s}" for i, s in enumerate(steps))
        return f"Reasoning so far:\n{lines}"

    def _annotate(self, result: MCTSResult) -> None:
        try:
            base = str(result.decision.get("rationale", "")).strip()
            note = result.trace()
            result.decision["rationale"] = f"{base} [{note}]" if base else note
        except Exception:  # noqa: BLE001
            pass


# --------------------------------------------------------------------------- #
# module-level helpers
# --------------------------------------------------------------------------- #
def _as_float(v: Any, default: float) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _risk_to_float(v: Any) -> float:
    return {"trivial": 0.05, "low": 0.2, "moderate": 0.5, "medium": 0.5,
            "high": 0.8, "critical": 0.95}.get(str(v).lower(), 0.2)


def _as_str_list(data: Any) -> List[str]:
    """Coerce a model reply into a list of step strings (array, or {'steps': [...]})."""
    if isinstance(data, list):
        return [str(x) for x in data if str(x).strip()]
    if isinstance(data, dict):
        for key in ("steps", "moves", "options", "next", "candidates"):
            v = data.get(key)
            if isinstance(v, list):
                return [str(x) for x in v if str(x).strip()]
        # a single {"step": "..."} object
        one = data.get("step") or data.get("text")
        if one:
            return [str(one)]
    if isinstance(data, str) and data.strip():
        return [data.strip()]
    return []


# --------------------------------------------------------------------------- #
# Self-test / demo
# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # pragma: no cover
    print("=" * 70)
    print("NYXARA MCTS-reasoner self-test")
    print("=" * 70)

    class _FakeLLM:
        """A scripted LLM: branches into steps, rolls forward, then drifts toward 42."""

        def __init__(self) -> None:
            self.calls = 0

        def generate(self, prompt, *, system=None, **kw):
            return "Consider the arithmetic directly."

        def generate_json(self, prompt, *, system=None, **kw):
            self.calls += 1
            if "JSON array" in prompt:        # expansion: propose branches
                return ["check the units", "compute step by step",
                        "estimate then verify", "look for a shortcut"]
            # terminal decision: deeper branches report higher confidence
            depth = prompt.count("\n  ")
            conf = min(0.95, 0.55 + 0.05 * depth)
            return {"kind": "respond", "text": "The product is 42.",
                    "confidence": conf, "rationale": "worked it out",
                    "risk": "low", "reversible": True}

    # a verifier that rewards a confident, non-empty answer
    def _verifier(_prompt: str, answer: str) -> float:
        return 0.9 if "42" in answer else 0.2

    r = MCTSReasoner(_FakeLLM(), decision_instructions="(JSON decision spec)",
                     iterations=12, max_children=4, rollout_depth=2, verifier=_verifier)
    out = r.search(stimulus="What is 6 times 7?", context="")
    print(f"\ndecision   : kind={out.decision['kind']} text={out.decision['text']!r}")
    print(f"score      : {out.score:.2f}")
    print(f"nodes      : {out.nodes}  iterations: {out.iterations}")
    print(f"best trace : {out.best_trace}")
    print(f"trace note : {out.decision['rationale']}")
    assert out.decision["kind"] == "respond" and "42" in out.decision["text"]
    assert out.nodes > 1 and out.iterations >= 1
    assert "MCTS" in out.decision["rationale"]
    assert out.score >= 0.9              # verifier-selected the good answer

    # graceful fallback: a model that never returns a parseable decision -> MCTSUnavailable
    class _DeadLLM:
        def generate(self, *a, **k):
            return ""
        def generate_json(self, *a, **k):
            return None
    try:
        MCTSReasoner(_DeadLLM(), decision_instructions="x", iterations=3).search(
            stimulus="hello", context="")
        raise SystemExit("expected MCTSUnavailable")
    except MCTSUnavailable:
        print("\ngraceful fallback   : MCTSUnavailable raised when no decision parses ✓")

    print("\nALL SELF-TESTS PASSED ✓")
