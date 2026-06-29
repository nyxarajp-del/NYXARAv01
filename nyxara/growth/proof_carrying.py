"""NYXARA · growth/proof_carrying.py — proof-carrying self-modification (✔♻, Gödel-machine, Rule 4).

The Master's sixth gap: a self-edit was justified **only empirically** — apply it, run the
benchmark, keep it if the number did not drop. A Gödel-machine-style mind rewrites itself only when
it can *prove* the rewrite is at least as good; the proof, not a lucky benchmark run, is the licence.
NYXARA already has a machine-checkable theorem-checker (:class:`~nyxara.growth.prover.Prover`); this
module **wires it into the self-edit gauntlet** so that, where a property is decidable, an edit
carries a proof — and where it is not, it falls back honestly to the existing verify-or-rollback.

Two real, sound checks (it never blocks a good edit, only a provable regression):

* **Decidable claim attached to an edit** — a caller (or the edit's own rationale) may assert a claim
  in a decidable domain (arithmetic / algebra / logic / number theory). :meth:`certify_claim` hands it
  to the Prover: ``PROVEN`` licenses, ``REFUTED`` blocks, ``UNPROVABLE`` defers to the empirical gate.
* **Behaviour-preserving boolean rewrites** — the deterministic transforms NYXARA applies most
  (``not a in b`` → ``a not in b``; De Morgan; ``a != None`` → ``a is not None``) *claim* to preserve
  meaning. :meth:`certify_edit` extracts the changed boolean conditions and **proves the before/after
  are logically equivalent** by exhaustive truth-table enumeration over canonicalised atoms. The
  equivalence engine is *sound by construction*: it returns ``REFUTED`` only when the two formulas
  share the same atoms yet disagree on some assignment (a genuine semantic change), and ``UNPROVABLE``
  whenever it cannot be sure — so a valid refactor is never wrongly blocked.

Pure standard library (``ast`` + the Prover). Certificates are returned for journalling. Touches no
source or weights itself — it only *decides whether* the gauntlet may keep an edit.
"""

from __future__ import annotations

import ast
import difflib
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from nyxara.growth.prover import ProofClaim, ProofResult, ProofVerdict, Prover

__all__ = ["ProofCarryResult", "ProofCarrier"]


# --------------------------------------------------------------------------- #
# Result
# --------------------------------------------------------------------------- #
@dataclass
class ProofCarryResult:
    """The proof verdict on a self-edit, plus the certificates anyone can re-check."""

    verdict: ProofVerdict
    certificate: str = ""
    method: str = ""
    claims_checked: int = 0
    proven: int = 0
    refuted: int = 0
    details: List[str] = field(default_factory=list)

    @property
    def blocks(self) -> bool:
        """The gauntlet must reject the edit iff a decidable property was *refuted*."""
        return self.verdict is ProofVerdict.REFUTED

    def to_dict(self) -> Dict[str, Any]:
        return {"verdict": self.verdict.value, "certificate": self.certificate,
                "method": self.method, "claims_checked": self.claims_checked,
                "proven": self.proven, "refuted": self.refuted, "details": list(self.details)}


# --------------------------------------------------------------------------- #
# The carrier
# --------------------------------------------------------------------------- #
class ProofCarrier:
    """Attach machine-checkable proofs to self-edits where the property is decidable."""

    def __init__(self, *, prover: Any = None, settings: Any = None) -> None:
        from nyxara.kernel.config import get_settings
        self.settings = settings or get_settings()
        self.prover = prover or Prover()

    # ---- a decidable claim a caller asserts about the edit ---- #
    def certify_claim(self, kind: str, statement: str,
                      candidate: Optional[str] = None) -> ProofResult:
        """Prove a single decidable claim (the reusable proof-carrying primitive)."""
        return self.prover.prove(ProofClaim(kind=kind, statement=statement,
                                            candidate_answer=candidate))

    # ---- the gauntlet entry point: certify a before→after source edit ---- #
    def certify_edit(self, before: str, after: str, *, kind: str = "") -> ProofCarryResult:
        """Prove what is decidable about a ``before → after`` edit.

        For the behaviour-preserving boolean transforms, every changed condition must stay logically
        equivalent — proved by truth-table enumeration. Returns ``PROVEN`` when at least one such
        equivalence is certified and none is broken, ``REFUTED`` when a rewrite provably changed a
        boolean's meaning, and ``UNPROVABLE`` when nothing decidable could be extracted (the honest
        signal for "fall back to the empirical gauntlet")."""
        details: List[str] = []
        proven = refuted = 0
        for old_line, new_line in self._changed_line_pairs(before, after):
            be = _extract_bool_expr(old_line)
            ae = _extract_bool_expr(new_line)
            if be is None or ae is None:
                continue
            verdict, cert = _prove_equivalent(be, ae)
            if verdict is ProofVerdict.PROVEN:
                proven += 1
                details.append(f"≡ {be!r} ⇔ {ae!r}: {cert}")
            elif verdict is ProofVerdict.REFUTED:
                refuted += 1
                details.append(f"✗ {be!r} ⇎ {ae!r}: {cert}")
        if refuted:
            return ProofCarryResult(ProofVerdict.REFUTED,
                                    certificate="; ".join(details[:4]),
                                    method="boolean equivalence (truth-table)",
                                    claims_checked=proven + refuted, proven=proven, refuted=refuted,
                                    details=details)
        if proven:
            return ProofCarryResult(ProofVerdict.PROVEN,
                                    certificate="; ".join(details[:4]),
                                    method="boolean equivalence (truth-table)",
                                    claims_checked=proven, proven=proven, details=details)
        return ProofCarryResult(ProofVerdict.UNPROVABLE,
                                certificate="no decidable invariant in the edit — empirical gate decides",
                                method="abstain", claims_checked=0)

    # ---- find the line pairs an edit changed (one-to-one within replace blocks) ---- #
    @staticmethod
    def _changed_line_pairs(before: str, after: str) -> List[Tuple[str, str]]:
        a = before.splitlines()
        b = after.splitlines()
        pairs: List[Tuple[str, str]] = []
        sm = difflib.SequenceMatcher(a=a, b=b, autojunk=False)
        for tag, i1, i2, j1, j2 in sm.get_opcodes():
            if tag != "replace":
                continue
            old, new = a[i1:i2], b[j1:j2]
            for k in range(min(len(old), len(new))):
                pairs.append((old[k], new[k]))
        return pairs


# --------------------------------------------------------------------------- #
# Boolean expression extraction + sound equivalence proof
# --------------------------------------------------------------------------- #
def _extract_bool_expr(line: str) -> Optional[str]:
    """Pull the boolean expression out of a line (``if``/``elif``/``while``/``return``/``assert``).

    Returns the expression source only when it actually parses as a boolean-shaped expression
    (a BoolOp, a ``not``, or a comparison) — otherwise ``None`` so the caller skips it."""
    s = line.strip()
    for kw in ("if ", "elif ", "while "):
        if s.startswith(kw):
            body = s[len(kw):].rstrip()
            if body.endswith(":"):
                body = body[:-1]
            return _bool_or_none(body)
    for kw in ("return ", "assert "):
        if s.startswith(kw):
            body = s[len(kw):].strip()
            if "," in body or "#" in body:        # assert msg / comment — too ambiguous to extract
                return None
            return _bool_or_none(body)
    return None


def _bool_or_none(expr: str) -> Optional[str]:
    try:
        node = ast.parse(expr.strip(), mode="eval").body
    except Exception:  # noqa: BLE001
        return None
    if isinstance(node, ast.BoolOp):
        return expr.strip()
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
        return expr.strip()
    if isinstance(node, ast.Compare):
        return expr.strip()
    return None


def _prove_equivalent(expr_a: str, expr_b: str) -> Tuple[ProofVerdict, str]:
    """Prove two boolean expressions are logically equivalent (sound; abstains when unsure).

    Each comparison is canonicalised to a positive atom + a sign (``a not in b`` ⇒ atom ``a in b``,
    negated), so a negated-compare rewrite shares atoms with its origin. Equivalence is then decided
    by exhaustive truth-table enumeration over the shared atoms. Returns ``REFUTED`` **only** when the
    two formulas share exactly the same atoms yet differ on some assignment (a real semantic change);
    any structural uncertainty yields ``UNPROVABLE`` so a valid refactor is never wrongly blocked."""
    try:
        na = ast.parse(expr_a, mode="eval").body
        nb = ast.parse(expr_b, mode="eval").body
    except Exception:  # noqa: BLE001
        return ProofVerdict.UNPROVABLE, "unparseable"
    atoms: Dict[str, int] = {}
    fa = _to_formula(na, atoms)
    fb = _to_formula(nb, atoms)
    if fa is None or fb is None:
        return ProofVerdict.UNPROVABLE, "expression outside the decidable boolean fragment"
    used_a = _atoms_in(na, atoms)
    used_b = _atoms_in(nb, atoms)
    if used_a != used_b:
        # different atom sets → cannot soundly compare (e.g. `<` vs `>=`) → abstain, never block
        return ProofVerdict.UNPROVABLE, "atoms differ — not decidable here"
    keys = sorted(used_a)
    n = len(keys)
    if n > 16:
        return ProofVerdict.UNPROVABLE, "too many atoms for exhaustive check"
    for i in range(2 ** n):
        env = {k: bool((i >> b) & 1) for b, k in enumerate(keys)}
        if fa(env) != fb(env):
            return ProofVerdict.REFUTED, f"differ at {env}"
    return ProofVerdict.PROVEN, f"equivalent over all {2 ** n} assignments"


def _canon_compare(node: ast.Compare) -> Optional[Tuple[str, bool]]:
    """Canonicalise a single-op comparison to ``(positive_atom_key, negated)``. None if unsupported."""
    if len(node.ops) != 1:
        return None
    try:
        left = ast.dump(node.left)
        right = ast.dump(node.comparators[0])
    except Exception:  # noqa: BLE001
        return None
    op = node.ops[0]
    base = f"{left}|{type(op).__name__}|{right}"
    if isinstance(op, ast.In):
        return f"IN::{left}|{right}", False
    if isinstance(op, ast.NotIn):
        return f"IN::{left}|{right}", True
    if isinstance(op, ast.Is):
        return f"IS::{left}|{right}", False
    if isinstance(op, ast.IsNot):
        return f"IS::{left}|{right}", True
    if isinstance(op, ast.Eq):
        return f"EQ::{left}|{right}", False
    if isinstance(op, ast.NotEq):
        return f"EQ::{left}|{right}", True
    # ordering comparisons (<, >, <=, >=): treat each as its own positive atom (no cross-relation)
    return f"{base}", False


def _to_formula(node: ast.AST, atoms: Dict[str, int]):
    """Build an ``env→bool`` evaluator over canonical atoms, or None if outside the fragment."""
    if isinstance(node, ast.BoolOp):
        subs = [_to_formula(v, atoms) for v in node.values]
        if any(s is None for s in subs):
            return None
        if isinstance(node.op, ast.And):
            return lambda env: all(s(env) for s in subs)
        return lambda env: any(s(env) for s in subs)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
        sub = _to_formula(node.operand, atoms)
        return (lambda env: not sub(env)) if sub is not None else None
    if isinstance(node, ast.Compare):
        canon = _canon_compare(node)
        if canon is None:
            return None
        key, negated = canon
        atoms.setdefault(key, 1)
        return (lambda env, k=key, neg=negated: (not env[k]) if neg else env[k])
    if isinstance(node, ast.Constant) and isinstance(node.value, bool):
        return lambda env, v=bool(node.value): v
    return None


def _atoms_in(node: ast.AST, atoms: Dict[str, int]) -> set:
    out: set = set()
    if isinstance(node, ast.BoolOp):
        for v in node.values:
            out |= _atoms_in(v, atoms)
    elif isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
        out |= _atoms_in(node.operand, atoms)
    elif isinstance(node, ast.Compare):
        canon = _canon_compare(node)
        if canon is not None:
            out.add(canon[0])
    return out


# --------------------------------------------------------------------------- #
# Self-test / demo
# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # pragma: no cover
    print("=" * 70)
    print("NYXARA proof-carrying self-modification self-test")
    print("=" * 70)

    pc = ProofCarrier()

    # 1) a decidable claim attached to an edit is proven / refuted, never bluffed
    assert pc.certify_claim("arithmetic", "2 + 2 = 4").verdict is ProofVerdict.PROVEN
    assert pc.certify_claim("arithmetic", "2 + 2 = 5").verdict is ProofVerdict.REFUTED
    assert pc.certify_claim("number_theory", "is 17 prime").verdict is ProofVerdict.PROVEN
    print("decidable claims    : proven / refuted correctly ✓")

    # 2) a behaviour-preserving boolean rewrite is PROVEN equivalent (the edit carries a proof)
    before = "def f(a, b):\n    if not a in b:\n        return 1\n    return 0\n"
    after = "def f(a, b):\n    if a not in b:\n        return 1\n    return 0\n"
    r = pc.certify_edit(before, after)
    print(f"not-in rewrite      : {r.to_dict()}")
    assert r.verdict is ProofVerdict.PROVEN and not r.blocks

    # De Morgan is proven equivalent too
    dm_b = "x = 1\nif not (a == b or c == d):\n    pass\n"
    dm_a = "x = 1\nif a != b and c != d:\n    pass\n"
    rdm = pc.certify_edit(dm_b, dm_a)
    print(f"De Morgan rewrite   : {rdm.verdict.value}")
    assert rdm.verdict is ProofVerdict.PROVEN

    # 3) a rewrite that SILENTLY CHANGES boolean meaning is REFUTED → the gauntlet must block it
    bad_b = "if a == b and c == d:\n    pass\n"
    bad_a = "if a == b or c == d:\n    pass\n"
    rbad = pc.certify_edit(bad_b, bad_a)
    print(f"meaning-changing    : {rbad.to_dict()}")
    assert rbad.verdict is ProofVerdict.REFUTED and rbad.blocks

    # 4) an ordinary refactor with no decidable invariant abstains (→ empirical gate decides)
    plain_b = "def g():\n    return compute(x) + helper(y)\n"
    plain_a = "def g():\n    return helper(y) + compute(x)\n"
    rp = pc.certify_edit(plain_b, plain_a)
    print(f"opaque refactor     : {rp.verdict.value} (falls back to empirical)")
    assert rp.verdict is ProofVerdict.UNPROVABLE and not rp.blocks

    print("\nALL SELF-TESTS PASSED ✓")
