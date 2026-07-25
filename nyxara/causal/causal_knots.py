"""NYXARA · causal/causal_knots.py — Temporal-Causal Knot Mechanics (CAUSAL · 2).

*The Master's ask:* build every cause→effect relationship as a **mathematical knot** so that a
logically-false statement **physically fails to link** — a *Knot Mutation Failure* — driving the
probability of a *causal-logic* hallucination toward zero.

What that means here, honestly. Each causal claim is a **signed link** between two nodes:

    * ``+1`` — a *concordant* link (A causes / promotes / agrees-with B),
    * ``-1`` — a *discordant* link (A prevents / inhibits / contradicts B).

A set of such links "closes into a consistent knot" iff the signed graph is **balanced** — 2-
colourable so that every ``+`` link joins same-coloured nodes and every ``-`` link joins
differently-coloured ones (Harary's structural-balance theorem). We test that incrementally with a
**parity union-find**: adding a link whose sign disagrees with the parity already forced by the
existing links closes an **inconsistent cycle** — the knot cannot be tied — and we raise
:class:`KnotMutationFailure`, reconstructing the offending cycle as the proof.

This is genuinely rigorous for the class of claims it can express (signed relations, including the
special anchor "⊤/TRUE" that turns a bare proposition into a link, so asserting both ``P`` and
``¬P`` fails to link). The honest boundary: it does **not** make *all* hallucination impossible —
only contradictions expressible as sign-balance over the claims actually fed to it are caught, and
exactly, at ``O(α(n))`` per link. Claims it never sees, it cannot police.

Bridges: complements :mod:`nyxara.mind.causal_world_model` (which *learns* edges from observation)
and :mod:`nyxara.mind.grounded_verifier`; when a check is genuinely undecided it should stay
superposed via :mod:`nyxara.quantum.superposition_states` rather than bluff.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, List, Optional, Sequence, Tuple

__all__ = [
    "CONCORDANT", "DISCORDANT", "TRUTH_ANCHOR",
    "CausalKnot", "KnotCheck", "KnotMutationFailure", "KnotLattice",
]

CONCORDANT = 1      # A causes / promotes / agrees-with B
DISCORDANT = -1     # A prevents / inhibits / contradicts B
TRUTH_ANCHOR = "⊤"  # the fixed "TRUE" node; a proposition links to it to become a claim


def _norm_sign(sign: Any) -> int:
    """Coerce a sign or relation word to +1 / -1."""
    if isinstance(sign, (int, float)):
        return CONCORDANT if sign >= 0 else DISCORDANT
    s = str(sign).strip().lower()
    if s in {"+", "+1", "1", "causes", "promotes", "implies", "same", "agrees", "supports", "true"}:
        return CONCORDANT
    if s in {"-", "-1", "prevents", "inhibits", "contradicts", "opposite", "negates", "false", "not"}:
        return DISCORDANT
    raise ValueError(f"unrecognised knot sign: {sign!r}")


# --------------------------------------------------------------------------- #
# Records
# --------------------------------------------------------------------------- #
@dataclass
class CausalKnot:
    """One signed causal link — a single strand of the knot."""

    cause: str
    effect: str
    sign: int = CONCORDANT
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {"cause": self.cause, "effect": self.effect, "sign": self.sign,
                "relation": "concordant" if self.sign == CONCORDANT else "discordant",
                "reason": self.reason}


@dataclass
class KnotCheck:
    """The verdict of trying to tie a set of claims into the lattice."""

    consistent: bool
    tied: int = 0
    failure: Optional["KnotMutationFailure"] = None

    def to_dict(self) -> Dict[str, Any]:
        return {"consistent": self.consistent, "tied": self.tied,
                "failure": None if self.failure is None else self.failure.to_dict()}


class KnotMutationFailure(Exception):
    """Raised when a claim cannot be tied — it closes a sign-inconsistent cycle."""

    def __init__(self, knot: CausalKnot, cycle: Sequence[str], expected: int, got: int) -> None:
        self.knot = knot
        self.cycle = list(cycle)
        self.expected = expected
        self.got = got
        path = " → ".join(self.cycle) if self.cycle else f"{knot.cause}~{knot.effect}"
        super().__init__(
            f"Knot Mutation Failure: '{knot.cause}'→'{knot.effect}' "
            f"({'concordant' if knot.sign == CONCORDANT else 'discordant'}) contradicts the "
            f"lattice along [{path}] (forced {expected:+d}, claimed {got:+d})")

    def to_dict(self) -> Dict[str, Any]:
        return {"knot": self.knot.to_dict(), "cycle": self.cycle,
                "expected": self.expected, "got": self.got, "message": str(self)}


# --------------------------------------------------------------------------- #
# The lattice
# --------------------------------------------------------------------------- #
class KnotLattice:
    """A signed causal graph kept perpetually balanced by rejecting contradictions."""

    def __init__(self) -> None:
        self._parent: Dict[str, str] = {}
        self._parity: Dict[str, int] = {}   # parity (0/1) of a node relative to its parent
        self._rank: Dict[str, int] = {}
        # accepted signed adjacency, for reconstructing the offending cycle on failure
        self._adj: Dict[str, List[Tuple[str, int]]] = {}
        self._knots: List[CausalKnot] = []

    # -- union-find with parity -------------------------------------------- #
    def _ensure(self, node: str) -> None:
        if node not in self._parent:
            self._parent[node] = node
            self._parity[node] = 0
            self._rank[node] = 0
            self._adj.setdefault(node, [])

    def _find(self, node: str) -> Tuple[str, int]:
        """Return (root, parity-to-root) with path compression."""
        self._ensure(node)
        parity = 0
        root = node
        while self._parent[root] != root:
            parity ^= self._parity[root]
            root = self._parent[root]
        # compress
        cur = node
        p = 0
        while self._parent[cur] != root:
            nxt = self._parent[cur]
            pp = self._parity[cur]
            self._parent[cur] = root
            self._parity[cur] = parity ^ p
            p ^= pp
            cur = nxt
        return root, parity

    # -- tying knots -------------------------------------------------------- #
    def tie(self, cause: str, effect: str, sign: Any = CONCORDANT, *, reason: str = "",
            dry_run: bool = False) -> CausalKnot:
        """Tie one causal link into the lattice.

        Raises :class:`KnotMutationFailure` if it closes a sign-inconsistent cycle. With
        ``dry_run=True`` the lattice is never mutated (used by :meth:`check`).
        """
        s = _norm_sign(sign)
        knot = CausalKnot(cause=cause, effect=effect, sign=s, reason=reason)
        # parity constraint between cause & effect: 0 if concordant, 1 if discordant
        want = 0 if s == CONCORDANT else 1
        rc, pc = self._find(cause)
        re_, pe = self._find(effect)
        if rc == re_:
            forced = pc ^ pe            # parity the lattice already forces between the two
            if forced != want:
                cycle = self._explain(cause, effect)
                raise KnotMutationFailure(
                    knot, cycle,
                    expected=CONCORDANT if forced == 0 else DISCORDANT,
                    got=s)
            # already consistent — nothing new to union, but record the strand
            if not dry_run:
                self._record(knot)
            return knot
        if dry_run:
            return knot
        # union the two trees, keeping parity coherent
        self._union(rc, pc, re_, pe, want)
        self._record(knot)
        return knot

    def _union(self, rc: str, pc: int, re_: str, pe: int, want: int) -> None:
        # parity of re_'s root relative to rc's root so that parity(cause,effect)==want
        rel = pc ^ pe ^ want
        if self._rank[rc] < self._rank[re_]:
            rc, re_ = re_, rc
            # rel is symmetric (xor), so it still holds
        self._parent[re_] = rc
        self._parity[re_] = rel
        if self._rank[rc] == self._rank[re_]:
            self._rank[rc] += 1

    def _record(self, knot: CausalKnot) -> None:
        self._knots.append(knot)
        self._adj.setdefault(knot.cause, []).append((knot.effect, knot.sign))
        self._adj.setdefault(knot.effect, []).append((knot.cause, knot.sign))

    def _explain(self, cause: str, effect: str) -> List[str]:
        """BFS over accepted strands to reconstruct a path cause→…→effect (the cycle)."""
        prev: Dict[str, str] = {cause: cause}
        q: Deque[str] = deque([cause])
        while q:
            u = q.popleft()
            if u == effect:
                break
            for v, _sign in self._adj.get(u, []):
                if v not in prev:
                    prev[v] = u
                    q.append(v)
        if effect not in prev:
            return [cause, effect]
        path: List[str] = [effect]
        while path[-1] != cause:
            path.append(prev[path[-1]])
        path.reverse()
        path.append(cause)  # close the loop back to the start via the new claimed strand
        return path

    # -- assertions on bare propositions ----------------------------------- #
    def assert_fact(self, proposition: str, value: bool = True, *, reason: str = "",
                    dry_run: bool = False) -> CausalKnot:
        """Assert ``proposition`` is True/False by linking it to the truth anchor.

        Asserting both a proposition and its negation then fails to link.
        """
        return self.tie(proposition, TRUTH_ANCHOR,
                        CONCORDANT if value else DISCORDANT,
                        reason=reason, dry_run=dry_run)

    # -- batch verification ------------------------------------------------- #
    def check(self, claims: Sequence[Tuple[str, str, Any]]) -> KnotCheck:
        """Dry-run a batch of ``(cause, effect, sign)`` claims *against* the live lattice.

        Never mutates the lattice. Returns a :class:`KnotCheck`; the first contradiction (if
        any) is captured in ``failure`` — this is the hallucination signal.
        """
        # Simulate on a shallow clone so the live lattice is untouched even across the batch.
        clone = self._clone()
        tied = 0
        for cause, effect, sign in claims:
            try:
                clone.tie(cause, effect, sign)
                tied += 1
            except KnotMutationFailure as f:
                return KnotCheck(consistent=False, tied=tied, failure=f)
        return KnotCheck(consistent=True, tied=tied)

    def commit(self, claims: Sequence[Tuple[str, str, Any]]) -> KnotCheck:
        """Check then, only if fully consistent, tie every claim into the live lattice."""
        verdict = self.check(claims)
        if verdict.consistent:
            for cause, effect, sign in claims:
                self.tie(cause, effect, sign)
        return verdict

    def _clone(self) -> "KnotLattice":
        c = KnotLattice()
        c._parent = dict(self._parent)
        c._parity = dict(self._parity)
        c._rank = dict(self._rank)
        c._adj = {k: list(v) for k, v in self._adj.items()}
        c._knots = list(self._knots)
        return c

    # -- introspection ------------------------------------------------------ #
    def __len__(self) -> int:
        return len(self._knots)

    def nodes(self) -> int:
        return len(self._parent)

    def status(self) -> Dict[str, Any]:
        return {"knots": len(self._knots), "nodes": len(self._parent),
                "components": len({self._find(n)[0] for n in self._parent})}
