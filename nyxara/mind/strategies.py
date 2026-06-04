"""NYXARA · mind/strategies.py — pluggable reasoning strategies.

A drop-in library of distinct ways to think, each implementing the
:class:`~nyxara.mind.reasoner.ReasoningStrategy` contract so the
:class:`~nyxara.mind.reasoner.Reasoner` can select among them or have them vote:

* **Bayesian** — update priors with evidence likelihoods to a posterior (respects
  base rates; the cure for confident nonsense).
* **Causal** — a structural causal model: ``do``-interventions, forward effect
  propagation along paths, counterfactual contrasts, and "what causes X".
* **Abductive** — inference to the best *explanation*: the hypothesis that covers the
  most observations, weighted by prior and penalised for complexity (Occam).
* **Analogical** — structure mapping (Gentner): carry relations from a source domain
  to a target via an entity correspondence ("the atom is like the solar system").
* **Neuro-symbolic** — verifiable-first: a hard *symbolic* constraint filters
  candidates, then a *probabilistic* score ranks the survivors (Rule: verifiable >
  probabilistic).

Each reads its structured inputs from ``query.payload`` and reports ``applicability``
of 0 when its inputs are absent, so it only fires when relevant.

Depends on :mod:`mind.reasoner`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

from nyxara.mind.reasoner import Conclusion, ReasoningQuery, ReasoningStrategy

__all__ = [
    "BayesianStrategy",
    "CausalModel",
    "CausalStrategy",
    "AbductiveStrategy",
    "AnalogicalStrategy",
    "NeuroSymbolicStrategy",
    "default_strategies",
]


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def _p(query: ReasoningQuery) -> Dict[str, Any]:
    return query.payload if isinstance(query.payload, dict) else {}


# --------------------------------------------------------------------------- #
# Bayesian
# --------------------------------------------------------------------------- #
class BayesianStrategy(ReasoningStrategy):
    """Posterior = normalise(prior × ∏ likelihoods). MAP hypothesis is the answer.

    payload = {
        "hypotheses": {name: prior, ...},
        "evidence": [ {hyp: P(e|hyp), ...}, ... ],   # one dict per observation
    }
    """

    name = "bayesian"

    def applicability(self, query: ReasoningQuery) -> float:
        p = _p(query)
        return 0.9 if "hypotheses" in p and "evidence" in p else 0.0

    def reason(self, query: ReasoningQuery) -> Conclusion:
        p = _p(query)
        priors: Dict[str, float] = dict(p["hypotheses"])
        total = sum(priors.values()) or 1.0
        post = {h: pr / total for h, pr in priors.items()}
        for ev in p["evidence"]:
            for h in post:
                post[h] *= float(ev.get(h, 1.0))     # missing likelihood == uninformative
            z = sum(post.values()) or 1.0
            post = {h: v / z for h, v in post.items()}
        if not post:
            return Conclusion(None, 0.0, "no hypotheses", self.name)
        best = max(post, key=post.get)
        return Conclusion(
            answer=best, confidence=_clamp(post[best]),
            rationale=f"posterior P({best})={post[best]:.3f}", strategy=self.name,
            support={"posterior": {h: round(v, 4) for h, v in post.items()}})


# --------------------------------------------------------------------------- #
# Causal
# --------------------------------------------------------------------------- #
class CausalModel:
    """A directed structural causal model with weighted edges (linear propagation)."""

    def __init__(self, edges: Sequence[Tuple[str, str, float]] = ()) -> None:
        self._out: Dict[str, List[Tuple[str, float]]] = {}
        self._in: Dict[str, List[Tuple[str, float]]] = {}
        for c, e, w in edges:
            self.add(c, e, w)

    def add(self, cause: str, effect: str, weight: float = 1.0) -> None:
        self._out.setdefault(cause, []).append((effect, weight))
        self._in.setdefault(effect, []).append((cause, weight))
        self._out.setdefault(effect, self._out.get(effect, []))
        self._in.setdefault(cause, self._in.get(cause, []))

    def causes_of(self, var: str) -> List[Tuple[str, float]]:
        return sorted(self._in.get(var, []), key=lambda cw: abs(cw[1]), reverse=True)

    def effects_of(self, var: str) -> List[Tuple[str, float]]:
        return sorted(self._out.get(var, []), key=lambda ew: abs(ew[1]), reverse=True)

    def _paths(self, source: str, target: str, seen: Optional[Set[str]] = None
               ) -> List[List[Tuple[str, float]]]:
        seen = seen or set()
        if source == target:
            return [[]]
        if source in seen:
            return []
        seen = seen | {source}
        paths: List[List[Tuple[str, float]]] = []
        for nxt, w in self._out.get(source, []):
            for sub in self._paths(nxt, target, seen):
                paths.append([(nxt, w)] + sub)
        return paths

    def effect_of(self, do_var: str, value: float, target: str) -> float:
        """Total linear effect on ``target`` of setting ``do_var = value`` (do-operator)."""
        total = 0.0
        for path in self._paths(do_var, target):
            prod = 1.0
            for _, w in path:
                prod *= w
            total += prod
        return total * value

    def counterfactual(self, do_var: str, target: str, factual: float,
                       counter: float) -> float:
        return self.effect_of(do_var, counter, target) - self.effect_of(do_var, factual, target)


class CausalStrategy(ReasoningStrategy):
    """Causal queries over a :class:`CausalModel`.

    payload = {"graph": [(cause, effect, weight), ...],
               "effect_of": {"do": {var: value}, "on": target}}   # -> numeric effect
            or {"graph": ..., "causes_of": var}                    # -> ranked causes
    """

    name = "causal"

    def applicability(self, query: ReasoningQuery) -> float:
        p = _p(query)
        if "graph" in p and ("effect_of" in p or "causes_of" in p):
            return 0.85
        return 0.0

    def reason(self, query: ReasoningQuery) -> Conclusion:
        p = _p(query)
        model = CausalModel(p["graph"])
        if "causes_of" in p:
            causes = model.causes_of(p["causes_of"])
            return Conclusion(
                answer=[c for c, _ in causes], confidence=1.0 if causes else 0.0,
                rationale=f"direct causes of {p['causes_of']}: {causes}",
                strategy=self.name, support={"causes": causes})
        spec = p["effect_of"]
        do = spec["do"]
        target = spec["on"]
        (var, value), = do.items()
        effect = model.effect_of(var, float(value), target)
        confidence = _clamp(0.5 + 0.5 * min(1.0, abs(effect)))
        return Conclusion(
            answer=effect, confidence=confidence if effect != 0 else 0.2,
            rationale=f"do({var}={value}) -> {target} = {effect:.3f}",
            strategy=self.name, support={"effect": effect})


# --------------------------------------------------------------------------- #
# Abductive
# --------------------------------------------------------------------------- #
class AbductiveStrategy(ReasoningStrategy):
    """Inference to the best explanation (coverage × prior ÷ complexity).

    payload = {
        "observations": [obs, ...],
        "hypotheses": {name: {"explains": [obs...], "prior": p, "complexity": c}, ...},
    }
    """

    name = "abductive"

    def applicability(self, query: ReasoningQuery) -> float:
        p = _p(query)
        return 0.85 if "observations" in p and "hypotheses" in p else 0.0

    def reason(self, query: ReasoningQuery) -> Conclusion:
        p = _p(query)
        obs: Set[Any] = set(p["observations"])
        if not obs:
            return Conclusion(None, 0.0, "no observations", self.name)
        scores: Dict[str, float] = {}
        details: Dict[str, Dict[str, float]] = {}
        for name, h in p["hypotheses"].items():
            explained = obs & set(h.get("explains", []))
            coverage = len(explained) / len(obs)
            prior = float(h.get("prior", 0.5))
            complexity = max(1e-6, float(h.get("complexity", 1.0)))
            score = coverage * prior / complexity
            scores[name] = score
            details[name] = {"coverage": coverage, "score": score}
        if not scores or max(scores.values()) <= 0:
            return Conclusion(None, 0.0, "no hypothesis explains the observations", self.name)
        best = max(scores, key=scores.get)
        total = sum(scores.values()) or 1.0
        confidence = _clamp(scores[best] / total)
        return Conclusion(
            answer=best, confidence=confidence,
            rationale=f"best explanation: {best} "
                      f"(coverage {details[best]['coverage']:.2f})",
            strategy=self.name, support={"scores": {k: round(v, 4) for k, v in scores.items()}})


# --------------------------------------------------------------------------- #
# Analogical (structure mapping)
# --------------------------------------------------------------------------- #
class AnalogicalStrategy(ReasoningStrategy):
    """Carry relations from a source domain to a target via an entity mapping.

    payload = {
        "source_relations": [(a, rel, b), ...],
        "mapping": {source_entity: target_entity, ...},
        "transfer": (a, rel, b) | None,   # optional: a specific relation to project
    }
    """

    name = "analogical"

    def applicability(self, query: ReasoningQuery) -> float:
        p = _p(query)
        return 0.8 if "source_relations" in p and "mapping" in p else 0.0

    def reason(self, query: ReasoningQuery) -> Conclusion:
        p = _p(query)
        mapping: Dict[Any, Any] = p["mapping"]
        rels: List[Tuple[Any, Any, Any]] = list(p["source_relations"])

        def project(rel: Tuple[Any, Any, Any]) -> Optional[Tuple[Any, Any, Any]]:
            a, r, b = rel
            if a in mapping and b in mapping:
                return (mapping[a], r, mapping[b])
            return None

        if p.get("transfer") is not None:
            projected = project(tuple(p["transfer"]))
            inferred = [projected] if projected else []
        else:
            inferred = [pr for pr in (project(r) for r in rels) if pr is not None]

        # structural support = fraction of source entities with a mapping
        entities = {e for (a, _, b) in rels for e in (a, b)}
        mapped = sum(1 for e in entities if e in mapping)
        support = mapped / len(entities) if entities else 0.0
        confidence = _clamp(support * (1.0 if inferred else 0.0))
        return Conclusion(
            answer=inferred, confidence=confidence,
            rationale=f"projected {len(inferred)} relation(s) via {len(mapping)}-entity mapping",
            strategy=self.name, support={"structural_support": support})


# --------------------------------------------------------------------------- #
# Neuro-symbolic (verifiable gate + probabilistic rank)
# --------------------------------------------------------------------------- #
class NeuroSymbolicStrategy(ReasoningStrategy):
    """Hard symbolic constraint filters; soft probabilistic score ranks survivors.

    payload = {
        "candidates": [c, ...],
        "constraint": callable(c) -> bool,    # symbolic, verifiable, MUST pass
        "score": callable(c) -> float,        # probabilistic preference [0,1]
    }
    """

    name = "neuro_symbolic"

    def applicability(self, query: ReasoningQuery) -> float:
        p = _p(query)
        return 0.8 if "candidates" in p and "constraint" in p and "score" in p else 0.0

    def reason(self, query: ReasoningQuery) -> Conclusion:
        p = _p(query)
        constraint: Callable[[Any], bool] = p["constraint"]
        score: Callable[[Any], float] = p["score"]
        survivors = [c for c in p["candidates"] if constraint(c)]
        if not survivors:
            return Conclusion(None, 0.0,
                              "no candidate satisfies the symbolic constraint", self.name)
        best = max(survivors, key=score)
        s = _clamp(score(best))
        return Conclusion(
            answer=best, confidence=s,
            rationale=f"{len(survivors)}/{len(p['candidates'])} passed the symbolic "
                      f"constraint; top probabilistic score {s:.2f}",
            strategy=self.name, support={"survivors": len(survivors)})


# --------------------------------------------------------------------------- #
# Convenience
# --------------------------------------------------------------------------- #
def default_strategies() -> List[ReasoningStrategy]:
    return [BayesianStrategy(), CausalStrategy(), AbductiveStrategy(),
            AnalogicalStrategy(), NeuroSymbolicStrategy()]


# --------------------------------------------------------------------------- #
# Self-test / demo
# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # pragma: no cover
    from nyxara.mind.reasoner import Reasoner, ReasoningMode

    print("=" * 70)
    print("NYXARA reasoning-strategies self-test")
    print("=" * 70)

    # 1) Bayesian — base-rate respecting (a positive test for a rare disease)
    bayes = BayesianStrategy()
    q = ReasoningQuery(payload={
        "hypotheses": {"sick": 0.01, "healthy": 0.99},
        "evidence": [{"sick": 0.99, "healthy": 0.05}],   # positive test
    })
    c = bayes.reason(q)
    print(f"\nbayesian (rare dz)  : {c.answer} (P={c.confidence:.3f})")
    assert c.answer == "healthy"           # base rate dominates a single positive test
    assert 0.8 < c.confidence < 0.85

    # 2) Causal — rain -> wet -> slippery
    causal = CausalStrategy()
    graph = [("rain", "wet", 0.9), ("wet", "slippery", 0.8), ("sprinkler", "wet", 0.7)]
    eff = causal.reason(ReasoningQuery(payload={
        "graph": graph, "effect_of": {"do": {"rain": 1.0}, "on": "slippery"}}))
    print(f"causal do(rain)     : slippery effect = {eff.answer:.3f}")
    assert abs(eff.answer - 0.72) < 1e-6   # 0.9 * 0.8
    causes = causal.reason(ReasoningQuery(payload={"graph": graph, "causes_of": "wet"}))
    print(f"causes of wet       : {causes.answer}")
    assert set(causes.answer) == {"rain", "sprinkler"}

    # 3) Abductive — best explanation of symptoms
    abd = AbductiveStrategy()
    c = abd.reason(ReasoningQuery(payload={
        "observations": ["fever", "cough", "fatigue"],
        "hypotheses": {
            "flu": {"explains": ["fever", "cough", "fatigue"], "prior": 0.3, "complexity": 1},
            "cold": {"explains": ["cough"], "prior": 0.5, "complexity": 1},
            "allergy": {"explains": ["cough", "fatigue"], "prior": 0.2, "complexity": 2},
        }}))
    print(f"\nabductive           : best = {c.answer} ({c.rationale})")
    assert c.answer == "flu"               # explains everything

    # 4) Analogical — solar system -> atom
    ana = AnalogicalStrategy()
    c = ana.reason(ReasoningQuery(payload={
        "source_relations": [("sun", "attracts", "planet"), ("planet", "orbits", "sun")],
        "mapping": {"sun": "nucleus", "planet": "electron"},
    }))
    print(f"analogical          : {c.answer}")
    assert ("nucleus", "attracts", "electron") in c.answer
    assert ("electron", "orbits", "nucleus") in c.answer

    # 5) Neuro-symbolic — symbolic gate then probabilistic rank
    ns = NeuroSymbolicStrategy()
    c = ns.reason(ReasoningQuery(payload={
        "candidates": [3, 4, 6, 7, 8, 9],
        "constraint": lambda x: x % 2 == 0,        # must be even (verifiable)
        "score": lambda x: x / 10.0,               # prefer larger (probabilistic)
    }))
    print(f"neuro-symbolic      : {c.answer} (largest even)")
    assert c.answer == 8

    # all five voting in a Reasoner — each only fires on its own input shape
    reasoner = Reasoner(default_strategies())
    res = reasoner.reason(ReasoningQuery(payload={
        "hypotheses": {"a": 0.5, "b": 0.5}, "evidence": [{"a": 0.9, "b": 0.1}]}),
        mode=ReasoningMode.CONSENSUS)
    print(f"\nreasoner consensus  : {res.conclusion.answer} "
          f"(only bayesian applied)")
    assert res.conclusion.answer == "a"

    print("\nALL SELF-TESTS PASSED ✓")
