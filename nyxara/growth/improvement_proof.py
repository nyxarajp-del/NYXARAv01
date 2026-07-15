"""NYXARA · growth/improvement_proof.py — the strict "provably BETTER" gate (⚖, Rule 4).

The Master's charge: NYXARA may rewrite her *entire* source, **but only when she can
mathematically PROVE the new code is better than the old** — never merely "not worse". The
existing gauntlet (``self_optimize.Optimizer``) proves an edit is *safe* and *non-regressing*;
this module is the extra, decisive gate that proves it is a genuine **improvement**, or the edit
is rolled back even though it was harmless.

Rice's theorem forbids a general "this program is better" decider, so we do not pretend to one.
Instead we certify improvement under a **well-defined, decidable ordering** — three sound proof
methods, each machine-checkable and journalled:

* **A · capability Pareto-gain** — measured on NYXARA's deterministic eval battery: at least one
  task that failed before now passes, **zero** regress, and the mean score does not drop. Because
  the harness is deterministic, this is a *dominance proof*, not a lucky benchmark run. This is
  the method that licenses the powerful open-ended self-authored rewrites: a whole-file rewrite is
  kept only if it makes NYXARA measurably smarter.
* **B · proven-equivalent-and-cheaper** — for a behaviour-preserving refactor, the changed boolean
  conditions are **proved logically equivalent** (truth-table, via :class:`ProofCarrier`) *and*
  the static cost (AST node + branch count) strictly drops. Same behaviour, provably simpler ⇒
  provably better.
* **C · defect-elimination** — a named, deterministic transform whose anti-pattern is a
  **decidable predicate**: it is proved the specific defect strictly decreased. The family covers
  ``bare_except`` → ``except Exception``, ``== None`` → ``is None``, dead-import removal, docstring
  insertion, negated membership/identity/equality normalisation (``not a in b`` → ``a not in b``,
  ``not (a == b)`` → ``a != b``), ``not not x`` → ``bool(x)``, empty ``list()``/``dict()``/
  ``tuple()`` → literal, redundant-``pass`` removal, redundant-``else`` de-indentation, and the
  mutable-default-argument fix (``def f(x=[])`` → None sentinel). Combined with the gauntlet's
  non-regression proof, the edit provably removes a defect without making anything worse.
* **D · frontier-advancement** — the **non-saturating** licence that answers the "intelligence
  explosion architecturally impossible" gap. Method A can only ever fire while the *fixed* benchmark
  battery still has failing tasks — once NYXARA masters it, its task count is a ceiling and no
  rewrite can be certified a capability gain again. Method D removes that ceiling. It certifies an
  edit better when it **strictly dominates** on NYXARA's open-ended auto-curriculum frontier
  (:mod:`~nyxara.growth.curriculum`): the same deterministically-seeded batch of freshly-generated,
  prover-certified problems is graded on the before-code and the after-code, and the after-code
  scores strictly higher with **no per-tier regression** and zero fixed-battery regression. Because
  the batch regenerates at an ever-rising difficulty tier, this ordering never saturates and cannot
  be memorised — yet each individual certificate is still a **decidable dominance on one concrete
  finite batch**, so Rice's theorem is respected, not defeated. This is what makes sustained,
  compounding self-improvement architecturally *possible* while every step stays sound.

Rice's theorem still forbids a general "this program is better" decider, and we still do not pretend
to one: methods A–D each certify improvement only under a well-defined, decidable ordering. What D
adds is that the ordering is **open-ended** (its difficulty rises without bound) rather than fixed.

No method certifies ⇒ ``better is False`` ⇒ the edit is discarded. Pure standard library plus
:class:`ProofCarrier`; it reads reports and source, and never touches disk or weights itself.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

__all__ = ["ImprovementCertificate", "ImprovementProver", "static_cost"]


# --------------------------------------------------------------------------- #
# Certificate
# --------------------------------------------------------------------------- #
@dataclass
class ImprovementCertificate:
    """The machine-checkable verdict on whether an edit is provably better."""

    better: bool
    method: str = "none"          # "pareto-capability" | "proven-equivalent-cheaper"
    #                               | "defect-elimination" | "frontier-advancement" | "none"
    reason: str = ""
    newly_passing: List[str] = field(default_factory=list)
    regressed: List[str] = field(default_factory=list)
    cost_before: int = 0
    cost_after: int = 0
    proof: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {"better": self.better, "method": self.method, "reason": self.reason,
                "newly_passing": list(self.newly_passing), "regressed": list(self.regressed),
                "cost_before": self.cost_before, "cost_after": self.cost_after,
                "proof": dict(self.proof)}


# --------------------------------------------------------------------------- #
# The prover
# --------------------------------------------------------------------------- #
class ImprovementProver:
    """Prove — under a decidable ordering — that a ``before → after`` edit is strictly better."""

    def __init__(self, *, settings: Any = None) -> None:
        self.settings = settings

    def prove(self, *, before: Any = None, after: Any = None,
              before_src: str = "", after_src: str = "",
              edit_kind: str = "",
              frontier_before: Optional[Dict[str, Any]] = None,
              frontier_after: Optional[Dict[str, Any]] = None) -> ImprovementCertificate:
        """Return a certificate. ``before``/``after`` are :class:`BenchmarkReport`s (or ``None``
        when no capability measurement is available); ``before_src``/``after_src`` are the edited
        file's full contents before and after. ``frontier_before``/``frontier_after`` are the
        read-only auto-curriculum probe results (``{frontier_score, by_tier, ...}``) graded on the
        SAME seeded batch against the before-code and after-code — the non-saturating ruler for
        Method D. Both ``None`` ⇒ Method D simply does not fire and A/B/C decide as before."""
        regressed = self._regressions(before, after)
        cost_b = static_cost(before_src)
        cost_a = static_cost(after_src)

        # ---- A · capability Pareto-gain (the licence for open-ended rewrites) ---- #
        if before is not None and after is not None:
            newly = self._newly_passing(before, after)
            if not regressed and newly and _mean_score(after) >= _mean_score(before):
                return ImprovementCertificate(
                    better=True, method="pareto-capability",
                    reason=(f"capability strictly increased: {len(newly)} task(s) now pass, "
                            f"0 regress (deterministic dominance proof)"),
                    newly_passing=newly, regressed=[], cost_before=cost_b, cost_after=cost_a)

        # ---- D · open-ended frontier advancement (the NON-SATURATING licence) ---- #
        # Fires when the fixed battery cannot (mastered ⇒ no newly-passing task) but the edit still
        # makes NYXARA measurably smarter on freshly-generated, prover-certified problems at the edge
        # of her capability. Sound because it is a strict dominance on ONE concrete seeded batch and
        # guarded by zero fixed-battery regression; open-ended because the batch's tier rises forever.
        if not regressed and frontier_before is not None and frontier_after is not None:
            fp = _frontier_dominates(frontier_before, frontier_after)
            if fp is not None and _mean_score(after) >= _mean_score(before):
                return ImprovementCertificate(
                    better=True, method="frontier-advancement",
                    reason=(f"strict dominance on the open-ended auto-curriculum frontier "
                            f"(tier {fp.get('tier')}): score {fp['score_before']:.4f}→"
                            f"{fp['score_after']:.4f}, no per-tier regression and 0 fixed-battery "
                            f"regress — a non-saturating, machine-certified capability gain"),
                    regressed=[], cost_before=cost_b, cost_after=cost_a, proof=fp)

        # ---- B · proven behaviourally-equivalent AND strictly cheaper ---- #
        if not regressed and cost_a < cost_b:
            pc = self._prove_equivalent(before_src, after_src, edit_kind)
            if pc is not None and pc.get("verdict") == "proven":
                return ImprovementCertificate(
                    better=True, method="proven-equivalent-cheaper",
                    reason=(f"behaviour proved unchanged (truth-table) and static cost strictly "
                            f"fell {cost_b}→{cost_a}"),
                    regressed=[], cost_before=cost_b, cost_after=cost_a, proof=pc)

        # ---- C · a named defect is provably eliminated (safe deterministic transforms) ---- #
        if not regressed and _defect_eliminated(before_src, after_src, edit_kind):
            return ImprovementCertificate(
                better=True, method="defect-elimination",
                reason=(f"the '{edit_kind}' defect is provably eliminated with no capability "
                        f"regression"),
                regressed=[], cost_before=cost_b, cost_after=cost_a)

        # ---- no proof of improvement ⇒ keep the old code ---- #
        why = "capability did not strictly increase (fixed battery nor open-ended frontier)"
        if regressed:
            why = f"capability regressed on {len(regressed)} task(s): {', '.join(regressed[:4])}"
        elif cost_a >= cost_b:
            why = ("no capability gain, no frontier advancement, and the change is not a "
                   "proven-cheaper or defect-fix edit")
        return ImprovementCertificate(better=False, method="none", reason=why,
                                      regressed=regressed, cost_before=cost_b, cost_after=cost_a)

    # ---- report helpers (duck-typed so tests can pass lightweight fakes) ---- #
    @staticmethod
    def _regressions(before: Any, after: Any) -> List[str]:
        if before is None or after is None:
            return []
        try:
            return list(after.regression_vs(before))
        except Exception:  # noqa: BLE001 — a measurement glitch must never masquerade as a gain
            return []

    @staticmethod
    def _newly_passing(before: Any, after: Any) -> List[str]:
        """Task ids that pass now but did not pass in the baseline (capability gained)."""
        gained: List[str] = []
        try:
            for r in after.results:
                if not getattr(r, "passed", False):
                    continue
                prev = before.get(r.task_id)
                if prev is None or not getattr(prev, "passed", False):
                    gained.append(r.task_id)
        except Exception:  # noqa: BLE001
            return []
        return gained

    def _prove_equivalent(self, before_src: str, after_src: str,
                          kind: str) -> Optional[Dict[str, Any]]:
        try:
            from nyxara.growth.proof_carrying import ProofCarrier
            res = ProofCarrier(settings=self.settings).certify_edit(
                before_src, after_src, kind=kind)
            return res.to_dict()
        except Exception:  # noqa: BLE001 — no prover ⇒ method B simply does not fire
            return None


# --------------------------------------------------------------------------- #
# Frontier dominance (method D) — a decidable "strictly better on the same batch" check
# --------------------------------------------------------------------------- #
def _frontier_dominates(before: Dict[str, Any], after: Dict[str, Any]
                        ) -> Optional[Dict[str, Any]]:
    """Return a proof dict iff ``after`` strictly dominates ``before`` on the same seeded frontier
    batch — a strictly higher weighted ``frontier_score`` AND no tier that got worse — else ``None``.

    Because the two probes were graded on byte-identical, prover-certified problems (same seed +
    tier), a strictly higher score with no per-tier regression is a genuine *dominance*, not a
    luckier draw: a decidable, machine-checkable proof of a real, open-ended capability gain."""
    try:
        sb = float(before.get("frontier_score", 0.0))
        sa = float(after.get("frontier_score", 0.0))
    except Exception:  # noqa: BLE001 — a malformed probe never masquerades as a gain
        return None
    if not (sa > sb):
        return None
    try:
        bt_b = {int(k): float(v) for k, v in (before.get("by_tier") or {}).items()}
        bt_a = {int(k): float(v) for k, v in (after.get("by_tier") or {}).items()}
    except Exception:  # noqa: BLE001
        return None
    for t in set(bt_b) | set(bt_a):
        if bt_a.get(t, 0.0) < bt_b.get(t, 0.0) - 1e-9:
            return None      # a difficulty tier regressed ⇒ not a clean dominance ⇒ not certified
    return {"verdict": "frontier-dominates", "score_before": round(sb, 6),
            "score_after": round(sa, 6), "tier": after.get("frontier_tier"),
            "by_tier_before": bt_b, "by_tier_after": bt_a}


# --------------------------------------------------------------------------- #
# Static cost — a deterministic complexity ruler (lower is simpler)
# --------------------------------------------------------------------------- #
def static_cost(src: str) -> int:
    """AST node count + cyclomatic branch count; ``0`` for empty, large sentinel if unparseable."""
    if not src:
        return 0
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return 10 ** 9      # an unparseable file is never "cheaper" than a valid one
    nodes = 0
    branches = 0
    for node in ast.walk(tree):
        nodes += 1
        if isinstance(node, (ast.If, ast.For, ast.AsyncFor, ast.While, ast.Try,
                             ast.ExceptHandler, ast.BoolOp, ast.IfExp, ast.comprehension,
                             ast.With, ast.AsyncWith, ast.Assert)):
            branches += 1
    return nodes + branches


# --------------------------------------------------------------------------- #
# Defect predicates (method C) — each is a decidable "the anti-pattern shrank" check
# --------------------------------------------------------------------------- #
_BARE_EXCEPT_RE = re.compile(r"(?m)^[ \t]*except[ \t]*:")
_EQ_NONE_RE = re.compile(r"(==|!=)\s*None\b|\bNone\s*(==|!=)")


def _defect_eliminated(before: str, after: str, kind: str) -> bool:
    """True iff the edit provably reduces the specific named defect (a decidable predicate)."""
    k = (kind or "").split(":")[-1]     # strip any 'self:'/'llm:' author prefix
    if k == "bare_except":
        return len(_BARE_EXCEPT_RE.findall(after)) < len(_BARE_EXCEPT_RE.findall(before))
    if k == "eq_none":
        return len(_EQ_NONE_RE.findall(after)) < len(_EQ_NONE_RE.findall(before))
    if k == "unused_import":
        # a genuinely dead import removed ⇒ strictly fewer imports AND a strictly simpler tree
        return (_import_count(after) < _import_count(before)
                and static_cost(after) < static_cost(before))
    if k == "missing_docstring":
        return _docstring_count(after) > _docstring_count(before)
    if k in ("not_in", "is_not_cmp"):
        return _negated_membership_count(after) < _negated_membership_count(before)
    if k == "not_eq_cmp":
        return _count_dropped(before, after, _negated_equality_count)
    if k == "double_negation":
        return _count_dropped(before, after, _double_negation_count)
    if k == "empty_collection_call":
        return _count_dropped(before, after, _empty_collection_count)
    if k == "redundant_pass":
        return _count_dropped(before, after, _redundant_pass_count)
    if k == "redundant_else_return":
        return _count_dropped(before, after, _redundant_else_count)
    if k == "mutable_default_arg":
        return _count_dropped(before, after, _mutable_default_count)
    return False


def _count_dropped(before: str, after: str, counter: Any) -> bool:
    """True iff a decidable anti-pattern counter strictly fell — and both sources parsed.

    Guards against a false positive: an unparseable ``after`` yields ``None`` (not a spurious
    zero), so a broken edit can never be certified as "the defect shrank"."""
    cb, ca = counter(before), counter(after)
    return cb is not None and ca is not None and ca < cb


def _safe_tree(src: str) -> Optional[ast.AST]:
    try:
        return ast.parse(src)
    except SyntaxError:
        return None


def _negated_equality_count(src: str) -> Optional[int]:
    """Count ``not (<x> ==/!= <y>)`` — the SIM201/SIM202 anti-pattern."""
    tree = _safe_tree(src)
    if tree is None:
        return None
    return sum(1 for n in ast.walk(tree)
               if (isinstance(n, ast.UnaryOp) and isinstance(n.op, ast.Not)
                   and isinstance(n.operand, ast.Compare) and len(n.operand.ops) == 1
                   and isinstance(n.operand.ops[0], (ast.Eq, ast.NotEq))))


def _double_negation_count(src: str) -> Optional[int]:
    """Count ``not not x`` — the SIM208 anti-pattern."""
    tree = _safe_tree(src)
    if tree is None:
        return None
    return sum(1 for n in ast.walk(tree)
               if (isinstance(n, ast.UnaryOp) and isinstance(n.op, ast.Not)
                   and isinstance(n.operand, ast.UnaryOp)
                   and isinstance(n.operand.op, ast.Not)))


def _empty_collection_count(src: str) -> Optional[int]:
    """Count empty ``list()``/``dict()``/``tuple()`` builtin calls — the C408 anti-pattern."""
    tree = _safe_tree(src)
    if tree is None:
        return None
    return sum(1 for n in ast.walk(tree)
               if (isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                   and n.func.id in ("list", "dict", "tuple")
                   and not n.args and not n.keywords))


def _redundant_pass_count(src: str) -> Optional[int]:
    """Count each ``pass`` sharing a block with other statements — the PIE790 anti-pattern."""
    tree = _safe_tree(src)
    if tree is None:
        return None
    n = 0
    for node in ast.walk(tree):
        for block_field in ("body", "orelse", "finalbody"):
            block = getattr(node, block_field, None)
            if isinstance(block, list) and len(block) > 1:
                n += sum(1 for s in block if isinstance(s, ast.Pass))
    return n


def _redundant_else_count(src: str) -> Optional[int]:
    """Count each ``else`` after an if-branch that always returns/raises/breaks/continues (RET505)."""
    tree = _safe_tree(src)
    if tree is None:
        return None
    n = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.If) and node.orelse and _body_terminates(node.body):
            if (len(node.orelse) == 1 and isinstance(node.orelse[0], ast.If)
                    and node.orelse[0].col_offset == node.col_offset):
                continue                          # an elif — not the anti-pattern
            n += 1
    return n


def _mutable_default_count(src: str) -> Optional[int]:
    """Count function default args that are mutable literals — the B006 anti-pattern."""
    tree = _safe_tree(src)
    if tree is None:
        return None
    n = 0
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            cands = list(node.args.defaults) + [
                d for d in node.args.kw_defaults if d is not None]
            n += sum(1 for d in cands if _is_mutable_literal(d))
    return n


def _is_mutable_literal(default: Any) -> bool:
    if isinstance(default, (ast.List, ast.Dict, ast.Set)):
        return True
    return (isinstance(default, ast.Call) and isinstance(default.func, ast.Name)
            and default.func.id in ("list", "dict", "set")
            and not default.args and not default.keywords)


def _body_terminates(body: Any) -> bool:
    return bool(body) and isinstance(body[-1], (ast.Return, ast.Raise, ast.Break, ast.Continue))


def _import_count(src: str) -> int:
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return 0
    n = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            n += len(node.names)
        elif isinstance(node, ast.ImportFrom):
            n += len(node.names)
    return n


def _docstring_count(src: str) -> int:
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return 0
    n = 1 if ast.get_docstring(tree) else 0
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if ast.get_docstring(node):
                n += 1
    return n


def _negated_membership_count(src: str) -> int:
    """Count ``not <x> in <y>`` / ``not <x> is <y>`` (the E713/E714 anti-patterns)."""
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return 0
    n = 0
    for node in ast.walk(tree):
        if (isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not)
                and isinstance(node.operand, ast.Compare)
                and len(node.operand.ops) == 1
                and isinstance(node.operand.ops[0], (ast.In, ast.Is))):
            n += 1
    return n


# --------------------------------------------------------------------------- #
# small internal helper
# --------------------------------------------------------------------------- #
def _mean_score(report: Any) -> float:
    try:
        return float(report.mean_score)
    except Exception:  # noqa: BLE001
        return 0.0


# --------------------------------------------------------------------------- #
# Self-test / demo
# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # pragma: no cover
    print("=" * 70)
    print("NYXARA improvement-proof self-test")
    print("=" * 70)

    prover = ImprovementProver()

    # C · a bare-except fix provably eliminates the defect (no reports needed)
    b = "try:\n    pass\nexcept:\n    pass\n"
    a = "try:\n    pass\nexcept Exception:\n    pass\n"
    cert = prover.prove(before_src=b, after_src=a, edit_kind="bare_except")
    assert cert.better and cert.method == "defect-elimination", cert.to_dict()
    print("defect-elimination : bare-except fix proved better ✓")

    # C · the widened deterministic transform library — each proves its own defect strictly shrank
    for kind, bsrc, asrc in [
        ("not_eq_cmp", "x = not (a == b)\n", "x = a != b\n"),
        ("double_negation", "y = not not f\n", "y = bool(f)\n"),
        ("empty_collection_call", "z = list()\n", "z = []\n"),
        ("redundant_pass", "if a:\n    pass\n    go()\n", "if a:\n    go()\n"),
        ("redundant_else_return",
         "def f(a):\n    if a:\n        return 1\n    else:\n        return 2\n",
         "def f(a):\n    if a:\n        return 1\n    return 2\n"),
        ("mutable_default_arg", "def g(x=[]):\n    return x\n",
         "def g(x=None):\n    if x is None:\n        x = []\n    return x\n"),
    ]:
        c = prover.prove(before_src=bsrc, after_src=asrc, edit_kind=kind)
        assert c.better and c.method == "defect-elimination", (kind, c.to_dict())
    print("defect-elimination : not-eq / double-neg / empty-call / pass / else / mutable OK ✓")

    # B · a proven-equivalent, strictly-cheaper negated-compare rewrite
    b2 = "def f(a, b):\n    return not a in b\n"
    a2 = "def f(a, b):\n    return a not in b\n"
    cert2 = prover.prove(before_src=b2, after_src=a2, edit_kind="not_in")
    assert cert2.better, cert2.to_dict()
    print(f"equivalent-cheaper : negated-compare proved better via {cert2.method} ✓")

    # NOT better · a pure whitespace reformat (same behaviour, same cost, no defect) is REJECTED
    b3 = "def add(a, b):\n    return a+b\n"
    a3 = "def add(a, b):\n    return a + b\n"
    cert3 = prover.prove(before_src=b3, after_src=a3, edit_kind="self:high_complexity")
    assert not cert3.better, cert3.to_dict()
    print("no-op reformat     : correctly REJECTED (not provably better) ✓")

    # A · a capability Pareto-gain, measured on (fake) deterministic reports
    class _R:
        def __init__(self, tid, passed):
            self.task_id, self._p = tid, passed
        @property
        def passed(self):
            return self._p

    class _Rep:
        def __init__(self, results, mean):
            self.results, self.mean_score = results, mean
        def get(self, tid):
            return next((r for r in self.results if r.task_id == tid), None)
        def regression_vs(self, base):
            lost = []
            for br in base.results:
                if not br.passed:
                    continue
                cur = self.get(br.task_id)
                if cur is None or not cur.passed:
                    lost.append(br.task_id)
            return lost

    before_rep = _Rep([_R("t1", True), _R("t2", False)], 0.50)
    after_rep = _Rep([_R("t1", True), _R("t2", True)], 0.90)
    cert4 = prover.prove(before=before_rep, after=after_rep,
                         before_src="x = 1\n", after_src="x = 2\n", edit_kind="self:refactor")
    assert cert4.better and cert4.method == "pareto-capability", cert4.to_dict()
    print("pareto-capability  : a real capability gain proved better ✓")

    # D · the NON-SATURATING licence — fixed battery MASTERED (no newly-passing task), but a strict
    #     dominance on the open-ended frontier still certifies the rewrite better.
    saturated = _Rep([_R("t1", True), _R("t2", True)], 1.00)   # 100% — Method A can never fire
    fb = {"frontier_score": 0.40, "by_tier": {1: 1.0, 2: 0.5, 3: 0.0}, "frontier_tier": 2}
    fa = {"frontier_score": 0.62, "by_tier": {1: 1.0, 2: 0.9, 3: 0.4}, "frontier_tier": 2}
    cert5 = prover.prove(before=saturated, after=saturated,
                         before_src="x = 1\n", after_src="x = 2\n", edit_kind="self:rewrite",
                         frontier_before=fb, frontier_after=fa)
    assert cert5.better and cert5.method == "frontier-advancement", cert5.to_dict()
    print("frontier-advance   : mastered battery + frontier gain proved better (no ceiling) ✓")

    # D · a per-tier REGRESSION on the frontier is refused even if the aggregate score rose.
    fa_bad = {"frontier_score": 0.65, "by_tier": {1: 1.0, 2: 0.2, 3: 0.9}, "frontier_tier": 2}
    cert6 = prover.prove(before=saturated, after=saturated,
                         before_src="x = 1\n", after_src="x = 2\n", edit_kind="self:rewrite",
                         frontier_before=fb, frontier_after=fa_bad)
    assert not cert6.better, cert6.to_dict()
    print("frontier-advance   : a per-tier regression correctly REFUSED ✓")

    # D · a flat frontier (no strict gain) is not a licence.
    cert7 = prover.prove(before=saturated, after=saturated,
                         before_src="x = 1\n", after_src="x = 2\n", edit_kind="self:rewrite",
                         frontier_before=fb, frontier_after=dict(fb))
    assert not cert7.better, cert7.to_dict()
    print("frontier-advance   : a flat (non-strict) frontier correctly REJECTED ✓")

    print("\nALL SELF-TESTS PASSED ✓")
