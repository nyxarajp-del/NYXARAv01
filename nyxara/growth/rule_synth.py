"""NYXARA · growth/rule_synth.py — inventing a NEW learning rule when the old one fails (🧬📈).

The rest of the growth stack *selects among* or *tunes* a fixed learning rule:
:mod:`growth.meta` picks a hyperparameter bundle with a bandit, :mod:`growth.meta_engine` nudges the
learning rate, :mod:`genesis` chooses an optimizer from a fixed menu. None of them can invent an
update rule that is **not already in the menu**. This module can.

When NYXARA's ordinary learning has demonstrably *stalled* (a plateau or regression — see
:func:`LearningRuleSynthesizer.should_invent` upstream in :mod:`growth.autolearn`), she does what no
fixed algorithm does: she **composes brand-new weight-update rules from mathematical primitives and
tests them on real learning tasks**, and if one genuinely learns better than plain gradient descent,
she installs it as her own. This is learning-to-learn in the strong sense — meta-cognition that
rewrites the *math of how she learns*, done entirely by her own deterministic code. **No LLM is
consulted anywhere in this loop.**

How it stays real, bounded, and safe:

* **Real math, real tasks.** A candidate rule is a small typed expression DAG (:class:`Node`) that
  computes a weight delta from the live update context (error, feature, weight, EWC anchor,
  gradient, momentum/second-moment EMAs, step, lr). Each candidate is plugged into a *fresh copy* of
  the real :class:`~nyxara.growth.learn.Learner` and trained on a battery of tasks with **known
  optima** (linear-regression recovery, a noisy bandit, a non-stationary target that flips midway)
  plus a **replay of NYXARA's own experience buffer** when available — so a discovered rule is
  judged on her actual lived data, not only synthetic tasks.
* **The incumbent is exactly representable.** ``lr*(err*x - ewc_penalty)`` is a legal DAG, so the
  search can always at least recover today's SGD+EWC behaviour; a rule is *adopted* only when it
  beats the incumbent by a margin **and** still recovers the regression optimum. A tie is never
  adopted.
* **Three layers of numerical safety.** The interpreter guards every op (safe division, ``sqrt(abs)``,
  NaN/inf → 0, per-node and final clamping); the fitness function hard-rejects any NaN/divergence or
  a wall-clock overrun; and the live seam in :class:`~nyxara.growth.learn.LinearValueModel` falls
  back to the incumbent on any non-finite delta. The live learner can never be corrupted.
* **Reversible and character-safe.** Adoption calls :meth:`Learner.install_rule`, which keeps the
  incumbent for :meth:`Learner.rollback_rule`. The invented rule is a pure scalar function with no
  access to feature *names*, so ``IMMUTABLE_VALUES`` / :meth:`Learner._guard` remain the sole gate on
  *what* is learned and are never touched — it changes only *how* a weight moves.
* **Bounded like Genesis.** Tiny population/generations/steps, multi-seed averaging, a hard
  wall-clock cap — real, deterministic, CI-fast (honest about scale, never a no-op).

Reuses :mod:`growth.learn` (the real learner + its experience buffer) and :mod:`guard.value_learning`
(the protected core). Pure standard library.
"""

from __future__ import annotations

import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from nyxara.growth.learn import Experience, Learner
from nyxara.guard.value_learning import IMMUTABLE_VALUES

__all__ = [
    "Node",
    "UpdateRule",
    "RuleSynthesisReport",
    "LearningRuleSynthesizer",
    "incumbent_rule",
]

# --------------------------------------------------------------------------- #
# Primitive vocabulary
# --------------------------------------------------------------------------- #
# Terminals are the numeric inputs available at every weight update, plus an evolved constant.
LEAVES: Tuple[str, ...] = (
    "err", "x", "w", "w_star", "importance", "grad", "m", "v", "step", "lr", "penalty", "const",
)
UNARY: Tuple[str, ...] = ("neg", "sign", "tanh", "abs", "sqrt", "square")
# ``clip`` is modelled as binary: clip(a, b) = max(-|b|, min(|b|, a)) — a bounded a by magnitude |b|.
BINARY: Tuple[str, ...] = ("add", "sub", "mul", "safe_div", "min", "max", "clip")

_CLAMP = 1.0e6          # every intermediate result is clamped to +-_CLAMP
_EPS = 1.0e-12          # safe-division floor
_MAX_DEPTH = 6          # cap on generated tree depth (parsimony + finite interpreter)


def _finite(x: float) -> float:
    """Map NaN/inf to 0.0 and clamp magnitude — the interpreter never returns a wild number."""
    if not math.isfinite(x):
        return 0.0
    if x > _CLAMP:
        return _CLAMP
    if x < -_CLAMP:
        return -_CLAMP
    return x


# --------------------------------------------------------------------------- #
# Expression node (the invented rule's AST)
# --------------------------------------------------------------------------- #
@dataclass
class Node:
    """One node of a weight-update expression: a leaf, a unary op, or a binary op."""

    op: str
    a: Optional["Node"] = None
    b: Optional["Node"] = None
    const: float = 0.0            # only meaningful when op == "const"

    # ---- evaluation ---- #
    def eval(self, env: Dict[str, float]) -> float:
        op = self.op
        if op == "const":
            return _finite(self.const)
        if op in LEAVES:
            return _finite(env.get(op, 0.0))
        if op in UNARY:
            a = self.a.eval(env) if self.a is not None else 0.0
            if op == "neg":
                r = -a
            elif op == "sign":
                r = (a > 0.0) - (a < 0.0)
            elif op == "tanh":
                r = math.tanh(a)
            elif op == "abs":
                r = abs(a)
            elif op == "sqrt":
                r = math.sqrt(abs(a))
            elif op == "square":
                r = a * a
            else:  # pragma: no cover — vocabulary is closed
                r = a
            return _finite(r)
        # binary
        a = self.a.eval(env) if self.a is not None else 0.0
        b = self.b.eval(env) if self.b is not None else 0.0
        if op == "add":
            r = a + b
        elif op == "sub":
            r = a - b
        elif op == "mul":
            r = a * b
        elif op == "safe_div":
            r = a / b if abs(b) > _EPS else 0.0
        elif op == "min":
            r = a if a < b else b
        elif op == "max":
            r = a if a > b else b
        elif op == "clip":
            bound = abs(b)
            r = max(-bound, min(bound, a))
        else:  # pragma: no cover — vocabulary is closed
            r = a
        return _finite(r)

    # ---- structure ---- #
    def size(self) -> int:
        n = 1
        if self.a is not None:
            n += self.a.size()
        if self.b is not None:
            n += self.b.size()
        return n

    def depth(self) -> int:
        da = self.a.depth() if self.a is not None else 0
        db = self.b.depth() if self.b is not None else 0
        return 1 + max(da, db)

    def clone(self) -> "Node":
        return Node(self.op,
                    self.a.clone() if self.a is not None else None,
                    self.b.clone() if self.b is not None else None,
                    self.const)

    def expr(self) -> str:
        if self.op == "const":
            return f"{self.const:.3g}"
        if self.op in LEAVES:
            return self.op
        if self.op in UNARY:
            return f"{self.op}({self.a.expr() if self.a else '0'})"
        return f"{self.op}({self.a.expr() if self.a else '0'}, {self.b.expr() if self.b else '0'})"

    # ---- persistence ---- #
    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {"op": self.op}
        if self.op == "const":
            d["const"] = self.const
        if self.a is not None:
            d["a"] = self.a.to_dict()
        if self.b is not None:
            d["b"] = self.b.to_dict()
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Node":
        return cls(
            op=str(d["op"]),
            a=cls.from_dict(d["a"]) if "a" in d and d["a"] is not None else None,
            b=cls.from_dict(d["b"]) if "b" in d and d["b"] is not None else None,
            const=float(d.get("const", 0.0)),
        )


# --------------------------------------------------------------------------- #
# UpdateRule — an executable, installable learning rule
# --------------------------------------------------------------------------- #
@dataclass
class UpdateRule:
    """A weight-update rule NYXARA can install into her live learner via ``Learner.install_rule``.

    Exposes the duck-typed contract the learner calls: ``delta(**inputs) -> float``.
    """

    root: Node
    origin: str = "invented"      # "incumbent" | "invented" | "random"

    @property
    def expr(self) -> str:
        return self.root.expr()

    def delta(self, *, err: float, x: float, w: float, w_star: float, importance: float,
              grad: float, m: float, v: float, step: float, lr: float, penalty: float) -> float:
        """Compute the weight delta for one (action, feature) update. Always returns a finite float.

        A final magnitude clamp of ``10*lr*(1+|grad|)`` guarantees a rule can never take an unbounded
        step, independent of the interpreter's per-node clamps.
        """
        env = {
            "err": err, "x": x, "w": w, "w_star": w_star, "importance": importance,
            "grad": grad, "m": m, "v": v, "step": step, "lr": lr, "penalty": penalty,
        }
        d = self.root.eval(env)
        bound = 10.0 * abs(lr) * (1.0 + abs(grad))
        if bound <= 0.0:
            bound = _CLAMP
        if d > bound:
            d = bound
        elif d < -bound:
            d = -bound
        return d if math.isfinite(d) else 0.0

    # ---- persistence ---- #
    def to_dict(self) -> Dict[str, Any]:
        return {"origin": self.origin, "expr": self.expr, "root": self.root.to_dict()}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "UpdateRule":
        return cls(root=Node.from_dict(d["root"]), origin=str(d.get("origin", "invented")))

    def save(self, path: Any) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Any) -> "UpdateRule":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def incumbent_rule() -> UpdateRule:
    """The rule NYXARA uses today: ``lr*(err*x - ewc_penalty)`` ≡ ``mul(lr, sub(grad, penalty))``.

    Exactly representable in the primitive space, so the search can always recover SGD+EWC.
    """
    root = Node("mul", Node("lr"), Node("sub", Node("grad"), Node("penalty")))
    return UpdateRule(root=root, origin="incumbent")


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #
@dataclass
class RuleSynthesisReport:
    """A summary of one rule-synthesis pass."""

    adopted: bool = False
    best_expr: str = ""
    best_fitness: float = float("-inf")
    incumbent_fitness: float = float("-inf")
    margin: float = 0.0
    generations: int = 0
    evaluated: int = 0
    seconds: float = 0.0
    reason: str = ""
    best_rule: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "adopted": self.adopted,
            "best_expr": self.best_expr,
            "best_fitness": round(self.best_fitness, 6) if math.isfinite(self.best_fitness) else None,
            "incumbent_fitness": (round(self.incumbent_fitness, 6)
                                  if math.isfinite(self.incumbent_fitness) else None),
            "margin": round(self.margin, 6),
            "generations": self.generations,
            "evaluated": self.evaluated,
            "seconds": round(self.seconds, 3),
            "reason": self.reason,
            "best_rule": self.best_rule,
        }


# --------------------------------------------------------------------------- #
# The synthesizer — genetic programming over update rules
# --------------------------------------------------------------------------- #
class LearningRuleSynthesizer:
    """Evolve a weight-update rule from primitives; adopt it only when it truly beats the incumbent.

    All randomness flows from a single seeded ``random.Random``, so a run is fully reproducible.
    """

    def __init__(self, *, core: Any = None, settings: Any = None,
                 population: int = 12, generations: int = 8, steps_per_task: int = 60,
                 seeds: int = 3, max_seconds: float = 8.0, adoption_margin: float = 0.02,
                 parsimony: float = 0.001, seed: int = 0) -> None:
        self.core = core
        self.settings = settings
        cfg = getattr(settings, "rule_synthesis", None)
        # config wins when present; explicit kwargs are the fallback defaults
        self.population = max(2, int(getattr(cfg, "population", population)))
        self.generations = max(1, int(getattr(cfg, "generations", generations)))
        self.steps_per_task = max(4, int(getattr(cfg, "steps_per_task", steps_per_task)))
        self.seeds = max(1, int(getattr(cfg, "seeds", seeds)))
        self.max_seconds = float(getattr(cfg, "max_seconds", max_seconds))
        self.adoption_margin = float(getattr(cfg, "adoption_margin", adoption_margin))
        self.parsimony = float(getattr(cfg, "parsimony", parsimony))
        self._rng = random.Random(seed)
        self._protected = set(IMMUTABLE_VALUES)

    # ------------------------------------------------------------------ #
    # Random program generation & variation
    # ------------------------------------------------------------------ #
    def _random_leaf(self) -> Node:
        op = self._rng.choice(LEAVES)
        if op == "const":
            return Node("const", const=round(self._rng.uniform(-1.0, 1.0), 4))
        return Node(op)

    def _random_tree(self, depth: int) -> Node:
        # deeper trees terminate at leaves; shallow ones grow ops
        if depth <= 0 or (depth < _MAX_DEPTH and self._rng.random() < 0.35):
            return self._random_leaf()
        if self._rng.random() < 0.4:
            return Node(self._rng.choice(UNARY), a=self._random_tree(depth - 1))
        return Node(self._rng.choice(BINARY),
                    a=self._random_tree(depth - 1), b=self._random_tree(depth - 1))

    def _all_nodes(self, node: Node) -> List[Node]:
        out = [node]
        if node.a is not None:
            out += self._all_nodes(node.a)
        if node.b is not None:
            out += self._all_nodes(node.b)
        return out

    def _mutate(self, root: Node) -> Node:
        tree = root.clone()
        nodes = self._all_nodes(tree)
        target = self._rng.choice(nodes)
        roll = self._rng.random()
        if roll < 0.45:
            # replace target's subtree with a fresh small subtree (point/subtree mutation)
            new = self._random_tree(self._rng.randint(0, 2))
            target.op, target.a, target.b, target.const = new.op, new.a, new.b, new.const
        elif roll < 0.7 and target.op == "const":
            target.const = round(target.const + self._rng.gauss(0.0, 0.3), 4)  # jitter a constant
        elif roll < 0.85:
            # wrap the target in a random unary op (grow)
            inner = Node(target.op, target.a, target.b, target.const)
            target.op, target.a, target.b, target.const = self._rng.choice(UNARY), inner, None, 0.0
        else:
            # collapse target to a leaf (prune)
            leaf = self._random_leaf()
            target.op, target.a, target.b, target.const = leaf.op, leaf.a, leaf.b, leaf.const
        return tree if tree.depth() <= _MAX_DEPTH else self._random_tree(2)

    def _crossover(self, a: Node, b: Node) -> Node:
        child = a.clone()
        recipients = self._all_nodes(child)
        donors = self._all_nodes(b)
        recipient = self._rng.choice(recipients)
        donor = self._rng.choice(donors).clone()
        recipient.op, recipient.a, recipient.b, recipient.const = (
            donor.op, donor.a, donor.b, donor.const)
        return child if child.depth() <= _MAX_DEPTH else a.clone()

    # ------------------------------------------------------------------ #
    # The evaluation battery — real learners on real tasks
    # ------------------------------------------------------------------ #
    def _make_learner(self, rule: Optional[UpdateRule], seed: int, *, ewc_lambda: float = 0.0):
        lrn = Learner(base_lr=0.1, ewc_lambda=ewc_lambda, seed=seed)
        if rule is not None:
            lrn.install_rule(rule)
        return lrn

    def _task_regression(self, rule: Optional[UpdateRule], seed: int) -> Tuple[float, bool]:
        """Recover a random linear target. Score in [0,1]; also returns a divergence flag."""
        rng = random.Random(seed * 7919 + 1)
        # a small feature set with NON-protected names, random ground-truth weights
        feats = [f"f{i}" for i in range(4)]
        assert not (set(feats) & self._protected)
        w_star = {f: rng.uniform(-1.0, 1.0) for f in feats}
        lrn = self._make_learner(rule, seed)
        init_err = None
        last_err = 0.0
        for _ in range(self.steps_per_task):
            x = {f: rng.uniform(-1.0, 1.0) for f in feats}
            target = sum(w_star[f] * x[f] for f in feats)
            pred = lrn.value("act", x)
            e = abs(target - pred)
            if init_err is None:
                init_err = e + 1e-9
            last_err = e
            lrn.update("act", x, target)
        if init_err is None:
            return 0.0, False
        diverged = last_err > init_err * 4.0 or not math.isfinite(last_err)
        # score: how much of the initial error was eliminated (clamped to [0,1])
        score = max(0.0, min(1.0, 1.0 - last_err / init_err))
        return score, diverged

    def _task_bandit(self, rule: Optional[UpdateRule], seed: int) -> Tuple[float, bool]:
        """Two actions with fixed noisy reward means; reward picking the better one."""
        rng = random.Random(seed * 104729 + 3)
        means = {"good": 0.8, "bad": 0.2}
        lrn = self._make_learner(rule, seed)
        ctx = {"c": 1.0}
        for _ in range(self.steps_per_task):
            for a in ("good", "bad"):
                r = means[a] + rng.gauss(0.0, 0.1)
                lrn.update(a, ctx, r)
        vg = lrn.value("good", ctx)
        vb = lrn.value("bad", ctx)
        diverged = not (math.isfinite(vg) and math.isfinite(vb)) or max(abs(vg), abs(vb)) > 100.0
        score = 1.0 if vg > vb else 0.0
        # reward a clean, well-scaled separation too (not just the argmax)
        score = 0.5 * score + 0.5 * max(0.0, min(1.0, (vg - vb + 1.0) / 2.0))
        return score, diverged

    def _task_nonstationary(self, rule: Optional[UpdateRule], seed: int) -> Tuple[float, bool]:
        """A target that flips sign midway — rewards fast *adaptation* (where cleverer rules win)."""
        ctx = {"c": 1.0}
        lrn = self._make_learner(rule, seed)
        half = max(2, self.steps_per_task // 2)
        target = 1.0
        errs: List[float] = []
        for t in range(self.steps_per_task):
            if t == half:
                target = -1.0
            pred = lrn.value("act", ctx)
            errs.append(abs(target - pred))
            lrn.update("act", ctx, target)
        # score the *second half* adaptation: low mean error after the flip is good
        post = errs[half:]
        mean_post = sum(post) / len(post) if post else 1.0
        diverged = not math.isfinite(mean_post) or mean_post > 10.0
        score = max(0.0, min(1.0, 1.0 - mean_post))
        return score, diverged

    def _task_replay(self, rule: Optional[UpdateRule], seed: int) -> Optional[Tuple[float, bool]]:
        """Score on NYXARA's OWN lived experience: fit on half, predict the held-out half."""
        buf = None
        if self.core is not None and getattr(self.core, "learner", None) is not None:
            buf = getattr(self.core.learner, "buffer", None)
        if buf is None or len(buf) < 12:
            return None
        exps: List[Experience] = list(buf.recent(200))
        # never train on a protected action/feature (defensive; the live guard also blocks it)
        exps = [e for e in exps
                if e.action not in self._protected and not (set(e.features) & self._protected)]
        if len(exps) < 12:
            return None
        rng = random.Random(seed * 32452843 + 7)
        rng.shuffle(exps)
        cut = len(exps) // 2
        train, test = exps[:cut], exps[cut:]
        lrn = self._make_learner(rule, seed)
        for e in train:
            lrn.update(e.action, e.features, e.reward)
        errs = [abs(e.reward - lrn.value(e.action, e.features)) for e in test]
        mean_err = sum(errs) / len(errs) if errs else 1.0
        diverged = not math.isfinite(mean_err) or mean_err > 100.0
        score = max(0.0, min(1.0, 1.0 - mean_err))
        return score, diverged

    def _fitness(self, rule: Optional[UpdateRule]) -> float:
        """Average task score across seeds, minus instability & parsimony penalties.

        Any NaN/inf or divergence on any task hard-rejects the rule (``-inf``).
        """
        per_seed: List[float] = []
        for s in range(self.seeds):
            scores: List[float] = []
            for task in (self._task_regression, self._task_bandit, self._task_nonstationary):
                score, diverged = task(rule, s)
                if diverged or not math.isfinite(score):
                    return float("-inf")
                scores.append(score)
            replay = self._task_replay(rule, s)
            if replay is not None:
                score, diverged = replay
                if diverged or not math.isfinite(score):
                    return float("-inf")
                scores.append(score)
            per_seed.append(sum(scores) / len(scores))
        mean = sum(per_seed) / len(per_seed)
        # instability = spread across seeds (a rule that is great on one seed and awful on another
        # is not trustworthy)
        spread = max(per_seed) - min(per_seed) if len(per_seed) > 1 else 0.0
        size_pen = self.parsimony * (rule.root.size() if rule is not None else 0)
        return mean - 0.25 * spread - size_pen

    def _recovers_regression(self, rule: UpdateRule, incumbent_score: float) -> bool:
        """A rule may only be adopted if it also learns the regression optimum (no free lunch)."""
        scores = []
        for s in range(self.seeds):
            score, diverged = self._task_regression(rule, s)
            if diverged:
                return False
            scores.append(score)
        return (sum(scores) / len(scores)) >= (incumbent_score - self.adoption_margin)

    # ------------------------------------------------------------------ #
    # The search
    # ------------------------------------------------------------------ #
    def run(self, *, enact: bool = False) -> RuleSynthesisReport:
        """Evolve rules; adopt the winner into the live learner iff it truly beats the incumbent.

        ``enact=False`` measures and reports only — the live learner is never touched.
        """
        t0 = time.time()
        report = RuleSynthesisReport()
        evaluated = 0

        inc = incumbent_rule()
        inc_fit = self._fitness(inc)
        report.incumbent_fitness = inc_fit
        evaluated += 1

        # regression-only baseline the adoption gate compares against
        inc_reg = [self._task_regression(inc, s)[0] for s in range(self.seeds)]
        inc_reg_score = sum(inc_reg) / len(inc_reg)

        # seed the population with the incumbent + random rules
        population: List[UpdateRule] = [inc]
        while len(population) < self.population:
            population.append(UpdateRule(root=self._random_tree(self._rng.randint(1, 3)),
                                         origin="random"))

        scored: List[Tuple[float, UpdateRule]] = [(inc_fit, inc)]
        for rule in population[1:]:
            scored.append((self._fitness(rule), rule))
            evaluated += 1

        best_fit, best_rule = max(scored, key=lambda kv: kv[0])
        gens_run = 0
        for gen in range(self.generations):
            if time.time() - t0 > self.max_seconds:
                report.reason = "time-budget-reached"
                break
            gens_run = gen + 1
            scored.sort(key=lambda kv: kv[0], reverse=True)
            # elitism: keep the top quarter (always includes something >= incumbent)
            keep = max(1, self.population // 4)
            survivors = [r for _, r in scored[:keep]]
            children: List[UpdateRule] = list(survivors)
            while len(children) < self.population:
                if time.time() - t0 > self.max_seconds:
                    break
                pa = self._tournament(scored)
                if self._rng.random() < 0.6:
                    child_root = self._crossover(pa.root, self._tournament(scored).root)
                else:
                    child_root = self._mutate(pa.root)
                children.append(UpdateRule(root=child_root, origin="invented"))
            scored = []
            for rule in children:
                scored.append((self._fitness(rule), rule))
                evaluated += 1
            gbf, gbr = max(scored, key=lambda kv: kv[0])
            if gbf > best_fit:
                best_fit, best_rule = gbf, gbr

        report.generations = gens_run
        report.evaluated = evaluated
        report.best_fitness = best_fit
        report.best_expr = best_rule.expr
        report.margin = best_fit - inc_fit
        report.best_rule = best_rule.to_dict()
        report.seconds = time.time() - t0

        # adoption gate: strictly beat the incumbent by the margin AND still recover regression
        adopt = (
            best_rule.origin != "incumbent"
            and math.isfinite(best_fit)
            and best_fit > inc_fit + self.adoption_margin
            and self._recovers_regression(best_rule, inc_reg_score)
        )
        if not adopt:
            report.reason = report.reason or (
                "no rule beat the incumbent by the margin" if best_rule.origin == "incumbent"
                or best_fit <= inc_fit + self.adoption_margin else "failed regression-recovery gate")
            return report

        report.adopted = True
        report.reason = "invented rule beat the incumbent and recovered the optimum"
        if enact and self.core is not None and getattr(self.core, "learner", None) is not None:
            self.core.learner.install_rule(best_rule)
            self._persist(best_rule, report)
        return report

    def _tournament(self, scored: Sequence[Tuple[float, UpdateRule]], k: int = 3) -> UpdateRule:
        picks = [self._rng.choice(scored) for _ in range(min(k, len(scored)))]
        return max(picks, key=lambda kv: kv[0])[1]

    # ------------------------------------------------------------------ #
    # Persistence (best-effort; never fatal)
    # ------------------------------------------------------------------ #
    def _persist(self, rule: UpdateRule, report: RuleSynthesisReport) -> None:
        try:
            paths = getattr(self.settings, "paths", None)
            data_dir = getattr(paths, "data_dir", None)
            base = Path(data_dir) if data_dir else Path.home() / ".nyxara"
            out = base / "rule_synth"
            out.mkdir(parents=True, exist_ok=True)
            payload = {
                "rule": rule.to_dict(),
                "fitness": report.best_fitness,
                "incumbent_fitness": report.incumbent_fitness,
                "margin": report.margin,
                "adopted": report.adopted,
                "at": time.time(),
            }
            (out / "best_rule.json").write_text(json.dumps(payload, indent=2, default=str),
                                                encoding="utf-8")
        except Exception:  # noqa: BLE001 — persistence is best-effort, never fatal
            pass


# --------------------------------------------------------------------------- #
# Self-test / demo
# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # pragma: no cover
    print("=" * 70)
    print("NYXARA learning-rule synthesis self-test — invent a new way to learn")
    print("=" * 70)

    # ---- the incumbent rule reproduces SGD+EWC exactly ---- #
    inc = incumbent_rule()
    d = inc.delta(err=0.5, x=2.0, w=0.0, w_star=0.0, importance=0.0,
                  grad=1.0, m=0.0, v=0.0, step=1.0, lr=0.1, penalty=0.3)
    print(f"\nincumbent expr      : {inc.expr}")
    print(f"incumbent delta     : {d:.4f}  (expected lr*(grad-penalty) = 0.1*(1.0-0.3) = 0.0700)")
    assert abs(d - 0.07) < 1e-9

    # ---- interpreter is always finite, even for pathological inputs ---- #
    bad = UpdateRule(root=Node("safe_div", Node("err"), Node("const", const=0.0)))
    assert math.isfinite(bad.delta(err=1.0, x=1.0, w=0.0, w_star=0.0, importance=0.0,
                                   grad=1.0, m=0.0, v=0.0, step=1.0, lr=0.1, penalty=0.0))
    print("safe division       : divide-by-zero -> finite 0.0 ✓")

    # ---- a full synthesis run: search invents/recovers a competitive rule ---- #
    synth = LearningRuleSynthesizer(population=14, generations=8, steps_per_task=60,
                                    seeds=3, seed=1, adoption_margin=0.01)
    t = time.time()
    rep = synth.run(enact=False)
    print(f"\nsearch              : {rep.evaluated} rules over {rep.generations} generations "
          f"in {time.time()-t:.2f}s")
    print(f"incumbent fitness   : {rep.incumbent_fitness:.4f}")
    print(f"best fitness        : {rep.best_fitness:.4f}   (margin {rep.margin:+.4f})")
    print(f"best rule           : {rep.best_expr}")
    print(f"adopted             : {rep.adopted}  ({rep.reason})")
    assert math.isfinite(rep.best_fitness)
    assert rep.best_fitness >= rep.incumbent_fitness - 1e-9   # search never regresses below SGD

    # ---- adopting into a live learner is reversible ---- #
    lrn = Learner(base_lr=0.1, seed=0)
    assert lrn.update_rule is None
    lrn.install_rule(UpdateRule(root=Node("mul", Node("lr"), Node("grad"))))
    assert lrn.update_rule is not None
    lrn.rollback_rule()
    assert lrn.update_rule is None
    print("\ninstall / rollback  : live learner rule swapped and restored ✓")

    print("\nALL SELF-TESTS PASSED ✓")
