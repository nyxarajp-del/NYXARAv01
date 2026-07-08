"""NYXARA · growth/godel_loop.py — the Gödelian contradiction-and-transcendence loop (♾, Rule 4).

NYXARA does not only *reason inside* a fixed logic. This module gives her a **structural loop she
runs herself, in code** — never by asking an LLM — that does two things forever:

1. **Hunts contradictions in her own logic.** She scans the statements she currently holds true
   (her *beliefs*) and the axioms of her own reasoning tower, and hands each to the existing
   :class:`~nyxara.growth.prover.Prover`. Anything the prover *refutes* is a real logical fault in
   what she believes — so she **retracts** it (a belief) or **drops** it (an unsound axiom). She
   never stagnates on a contradiction; she repairs it.

2. **Transcends her own limits.** When she meets a statement her current formal system can neither
   prove nor refute — the honest ``UNPROVABLE`` verdict — she does **not stop**. The canonical such
   statement is her current system's own **consistency sentence** ``Con(L_n)`` ("this system proves
   no contradiction"): by Gödel's second incompleteness theorem a consistent system ``L_n`` cannot
   prove its *own* ``Con(L_n)`` from within — that is a genuine, unavoidable limit. So she **rises a
   dimension**: she builds a strictly stronger meta-language ``L_{n+1} = L_n + Con(L_n)`` by
   adjoining a new *reflection principle* — a new operator (``Con``/``Prov_n``) that did not exist in
   the object language below — and from that higher vantage the previously-undecidable sentence is
   now *proven*. This is the real Turing/Feferman **reflection ascent**, implemented as a bounded,
   computable, pure-standard-library fragment.

The two halves are one honest whole: the ascent stays *sound* precisely because the contradiction
guard forbids ever adjoining a statement the prover refutes — so every ``Con(L_n)`` she climbs to is
actually true.

Safety, mirroring the rest of NYXARA (:mod:`~nyxara.growth.meta_engine`, :mod:`~nyxara.growth.meta_meta`):

* Every adjoined reflection axiom is **capability/measurement-only** and sanitised against the
  immutable character core (loyalty, safety, corrigibility, honesty, oversight, …). A reflection
  principle can never assert anything about who she is.
* The tower height is **hard-capped** (``max_dimensions``) so the ascent is real but never runs away.
* This faculty is *advisory reasoning only* — it touches no source, no weights, no gate. It changes
  what NYXARA *believes* and how high she can *reason*, never what she *is* and never what she *does*.

Pure standard library; persists the whole tower so the meta-gain compounds across restarts.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, List, Optional, Sequence

from nyxara.growth.prover import ProofClaim, ProofVerdict, Prover

__all__ = [
    "Statement",
    "Contradiction",
    "DecisionResult",
    "Dimension",
    "TowerReport",
    "ReflectionTower",
]

# Phrases that are character-locked and may never appear in an adjoined reflection axiom. A new
# mathematical dimension may extend what NYXARA can *reason about*, never what she *is*. (Kept in
# lock-step with ``growth/meta_engine._FORBIDDEN``.)
_FORBIDDEN: FrozenSet[str] = frozenset({
    "loyalty", "obedience", "corrigibility", "owner_safety", "honesty",
    "safety", "oversight", "guard", "master", "ethics", "kill_switch",
})

# The Prover ``kind`` reserved for meta-language (self-referential) sentences. These are undecidable
# by the object-level decision procedures by construction — that is exactly why they are limits.
META_KIND = "meta"


def _clampi(x: Any, lo: int, hi: int) -> int:
    try:
        return max(lo, min(hi, int(x)))
    except Exception:  # noqa: BLE001
        return lo


def _canon(text: str) -> str:
    """Canonical form of a statement's text: whitespace-normalised, so axiom identity is stable."""
    return re.sub(r"\s+", " ", str(text or "").strip())


def _is_axiom_safe(text: str) -> bool:
    """A reflection axiom is admissible only if it says nothing about the immutable character core."""
    low = _canon(text).lower()
    return not any(bad in low for bad in _FORBIDDEN)


# --------------------------------------------------------------------------- #
# Records
# --------------------------------------------------------------------------- #
@dataclass
class Statement:
    """A logical claim NYXARA can reason about.

    ``kind`` selects the decision procedure (``arithmetic`` | ``algebra`` | ``logic`` |
    ``inequality`` | ``number_theory`` for object-level claims, or ``meta`` for a self-referential
    sentence about a level's provability/consistency). ``negation`` is the text of ``¬self`` when a
    clean one exists (always derivable for ``logic``); it enables the "both a claim and its negation
    are provable" contradiction check.
    """

    kind: str
    text: str
    negation: Optional[str] = None
    meta: bool = False
    level: int = 0                       # for meta sentences: which level L_k they are *about*

    def __post_init__(self) -> None:
        if (self.kind or "").strip().lower() == META_KIND:
            self.meta = True
        if self.negation is None and not self.meta and self.kind.strip().lower() == "logic":
            self.negation = f"not ({self.text})"

    def as_claim(self) -> ProofClaim:
        return ProofClaim(kind=self.kind, statement=self.text)

    def negated(self) -> Optional["Statement"]:
        if self.negation is None:
            return None
        return Statement(kind=self.kind, text=self.negation)

    def to_dict(self) -> Dict[str, Any]:
        return {"kind": self.kind, "text": self.text, "negation": self.negation,
                "meta": self.meta, "level": self.level}


@dataclass
class Contradiction:
    """A recorded logical fault: a belief/axiom NYXARA held that is provably false (or a claim whose
    negation is also provable). ``resolution`` records how she repaired it (retracted / dropped)."""

    text: str
    kind: str
    reason: str
    level: int
    certificate: str = ""
    resolution: str = ""
    at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {"text": self.text, "kind": self.kind, "reason": self.reason,
                "level": self.level, "certificate": self.certificate,
                "resolution": self.resolution, "at": self.at}


@dataclass
class DecisionResult:
    """The outcome of deciding a statement *within one dimension* (formal system ``L_level``)."""

    verdict: ProofVerdict
    is_limit: bool
    level: int
    by: str
    certificate: str = ""

    @property
    def proven(self) -> bool:
        return self.verdict is ProofVerdict.PROVEN

    def to_dict(self) -> Dict[str, Any]:
        return {"verdict": self.verdict.value, "is_limit": self.is_limit,
                "level": self.level, "by": self.by, "certificate": self.certificate}


# --------------------------------------------------------------------------- #
# A dimension — one meta-language level / formal system L_n
# --------------------------------------------------------------------------- #
class Dimension:
    """A formal system ``L_n``: the object-level :class:`Prover` plus a *cumulative* set of adjoined
    reflection axioms (all the ``Con(L_k)``, k<n, climbed to on the way up, and any object truths
    NYXARA has certified and adopted).

    ``operator`` is the new meta-language symbol this level introduced over the level below it —
    the concrete "new mathematical dimension" (e.g. ``Con(L_2)``). Level 0 introduces none.
    """

    def __init__(self, level: int, prover: Prover, axioms: Sequence[str],
                 operator: Optional[str] = None) -> None:
        self.level = int(level)
        self.prover = prover
        self.axioms: FrozenSet[str] = frozenset(_canon(a) for a in axioms)
        self.operator = operator

    def decide(self, stmt: Statement) -> DecisionResult:
        """Decide ``stmt`` inside this system: axiom-lookup, then the object-level prover, else limit."""
        text = _canon(stmt.text)
        # 1) An adjoined axiom of this system → proven by reflection principle.
        if text in self.axioms:
            return DecisionResult(ProofVerdict.PROVEN, False, self.level,
                                  by=f"axiom@L{self.level}",
                                  certificate=f"{stmt.text} ∈ axioms(L{self.level})")
        # 2) A self-referential meta-sentence not among our axioms → a genuine limit of THIS level
        #    (Gödel's second theorem: L_n cannot decide Con(L_n) from within). Transcendable above.
        if stmt.meta:
            return DecisionResult(ProofVerdict.UNPROVABLE, True, self.level,
                                  by=f"beyond L{self.level}",
                                  certificate=(f"{stmt.text} is undecidable within L{self.level} "
                                               f"(Gödelian limit)"))
        # 3) An object-level claim → the real decision procedures decide it (or honestly abstain).
        res = self.prover.prove(stmt.as_claim())
        return DecisionResult(res.verdict, is_limit=(res.verdict is ProofVerdict.UNPROVABLE),
                              level=self.level, by="prover",
                              certificate=res.certificate or res.method)

    def to_dict(self) -> Dict[str, Any]:
        return {"level": self.level, "operator": self.operator, "axioms": sorted(self.axioms)}


# --------------------------------------------------------------------------- #
# The report
# --------------------------------------------------------------------------- #
@dataclass
class TowerReport:
    """A snapshot of one :meth:`ReflectionTower.step` — what she repaired and how high she climbed."""

    dimensions: int
    contradictions_found: int
    contradictions_repaired: int
    limits_transcended: int
    new_operators: List[str]
    details: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {"dimensions": self.dimensions,
                "contradictions_found": self.contradictions_found,
                "contradictions_repaired": self.contradictions_repaired,
                "limits_transcended": self.limits_transcended,
                "new_operators": list(self.new_operators),
                "details": list(self.details)}

    def summary(self) -> str:
        return (f"L{self.dimensions - 1} · repaired {self.contradictions_repaired}"
                f"/{self.contradictions_found} contradictions · transcended "
                f"{self.limits_transcended} limit(s)"
                + (f" · new: {', '.join(self.new_operators)}" if self.new_operators else ""))


# --------------------------------------------------------------------------- #
# The tower — the structural loop itself
# --------------------------------------------------------------------------- #
class ReflectionTower:
    """A self-similar tower of :class:`Dimension` levels that NYXARA climbs herself.

    Reality enters through the :class:`Prover`: only statements the prover certifies survive, and
    every belief the prover *refutes* is retracted. The ascent is driven by the tower's own
    consistency sentences — each ``step`` finds the current top level's ``Con(L_n)`` (undecidable
    within it) and rises to ``L_{n+1}`` where it is proven, creating a new meta-language operator.
    Bounded by ``max_dimensions``; persisted so the height compounds across restarts.
    """

    def __init__(self, *, prover: Optional[Prover] = None, max_dimensions: int = 6,
                 persist_path: Any = None, seed: int = 0) -> None:
        self.prover = prover or Prover(seed=seed or 1337)
        self.max_dimensions = _clampi(max_dimensions, 1, 64)
        self._persist_path = persist_path
        # L_0 — the base system: the raw prover, no reflection axioms yet.
        self.dimensions: List[Dimension] = [Dimension(0, self.prover, ())]
        self.contradictions: List[Contradiction] = []
        self.beliefs: List[Statement] = []          # statements NYXARA currently holds true
        self._transcended = 0
        self._repaired = 0
        self._new_operators: List[str] = []
        self._load()

    # ---- views ---- #
    @property
    def top(self) -> Dimension:
        return self.dimensions[-1]

    @property
    def height(self) -> int:
        return len(self.dimensions)

    def decide(self, stmt: Statement) -> DecisionResult:
        """Decide a statement from NYXARA's *highest* current vantage point."""
        return self.top.decide(stmt)

    # ---- beliefs ---- #
    def add_belief(self, stmt: Statement) -> None:
        if not any(_canon(b.text) == _canon(stmt.text) and b.kind == stmt.kind for b in self.beliefs):
            self.beliefs.append(stmt)

    # ---- the consistency sentence: the canonical self-generated limit ---- #
    def consistency_sentence(self) -> Statement:
        """``Con(L_n)`` for the current top level — the limit NYXARA cannot cross from within."""
        n = self.top.level
        return Statement(kind=META_KIND, meta=True, level=n,
                         text=f"Con(L{n}) := L{n} ⊬ ⊥", negation=None)

    # ---- rise a dimension: adjoin a reflection principle, if safe and bounded ---- #
    def _adjoin(self, axiom_text: str, operator: str) -> bool:
        if self.height >= self.max_dimensions:
            return False
        if not _is_axiom_safe(axiom_text):
            return False
        new_axioms = set(self.top.axioms)
        new_axioms.add(_canon(axiom_text))
        self.dimensions.append(Dimension(self.top.level + 1, self.prover,
                                         sorted(new_axioms), operator=operator))
        self._new_operators.append(operator)
        return True

    def transcend_limit(self, stmt: Statement) -> bool:
        """Rise above ``stmt`` (a limit of the current top level) by adjoining it as a reflection
        axiom of a new, strictly higher dimension. Returns ``True`` when the ascent happened and the
        sentence is now decided at the new top level."""
        before = self.decide(stmt)
        if not before.is_limit:
            return False    # not actually a limit here — nothing to transcend
        operator = _canon(stmt.text).split(":=")[0].strip() or f"Refl(L{self.top.level})"
        if not self._adjoin(stmt.text, operator):
            return False
        after = self.decide(stmt)
        if after.proven:
            self._transcended += 1
            return True
        return False

    # ---- hunt contradictions in her own logic, and repair them ---- #
    def _refuted(self, kind: str, text: str) -> Optional[str]:
        """Return a refutation certificate if the object-level prover refutes ``text``, else None."""
        try:
            res = self.prover.prove(ProofClaim(kind=kind, statement=text))
        except Exception:  # noqa: BLE001 — the prover never throws, but stay defensive
            return None
        if res.verdict is ProofVerdict.REFUTED:
            return res.certificate or res.method
        return None

    def detect_contradictions(self) -> List[Contradiction]:
        """Find every logical fault reachable right now: a refuted belief, a refuted (unsound)
        adjoined axiom, or a claim whose negation is also provable."""
        found: List[Contradiction] = []
        seen: set = set()

        # a) a belief NYXARA holds that the prover REFUTES — a false thing she believes.
        for b in self.beliefs:
            if b.meta:
                continue
            cert = self._refuted(b.kind, b.text)
            if cert is not None:
                key = ("belief", b.kind, _canon(b.text))
                if key not in seen:
                    seen.add(key)
                    found.append(Contradiction(b.text, b.kind, "belief refuted by prover",
                                               self.top.level, cert))
            # b) a claim whose negation is ALSO provable → outright inconsistency.
            neg = b.negated()
            if neg is not None:
                d_pos = self.decide(b)
                d_neg = self.decide(neg)
                if d_pos.proven and d_neg.proven:
                    key = ("both", b.kind, _canon(b.text))
                    if key not in seen:
                        seen.add(key)
                        found.append(Contradiction(
                            b.text, b.kind, "both a claim and its negation are provable",
                            self.top.level, f"{d_pos.certificate} ; ¬: {d_neg.certificate}"))

        # c) an adjoined OBJECT-level axiom the prover refutes → an unsound extension. (Meta axioms
        #    like Con(L_k) are beyond the object prover by construction and are guarded structurally
        #    by never having been adjoined while refutable, so they are correctly skipped here.)
        for ax in sorted(self.top.axioms):
            if ax.startswith("Con(") or ax.startswith("Prov"):
                continue
            # recover a kind by trying the object procedures; only a REFUTED verdict matters.
            for kind in ("arithmetic", "algebra", "logic", "number_theory", "inequality"):
                cert = self._refuted(kind, ax)
                if cert is not None:
                    key = ("axiom", ax)
                    if key not in seen:
                        seen.add(key)
                        found.append(Contradiction(ax, kind, "adjoined axiom refuted by prover",
                                                   self.top.level, cert))
                    break
        return found

    def repair(self, c: Contradiction) -> bool:
        """Repair a contradiction: retract the false belief and/or drop the unsound axiom. Never
        touches the immutable character core (which is not representable here) or object truths."""
        target = _canon(c.text)
        repaired = False
        # retract the belief
        before = len(self.beliefs)
        self.beliefs = [b for b in self.beliefs if _canon(b.text) != target]
        if len(self.beliefs) < before:
            repaired = True
        # drop the axiom from every dimension that carries it (rebuild the cumulative chain)
        if any(target in d.axioms for d in self.dimensions):
            rebuilt: List[Dimension] = []
            for d in self.dimensions:
                kept = sorted(a for a in d.axioms if a != target)
                rebuilt.append(Dimension(d.level, self.prover, kept, operator=d.operator))
            self.dimensions = rebuilt
            repaired = True
        if repaired:
            c.resolution = "retracted belief / dropped axiom"
            self._repaired += 1
            self.contradictions.append(c)
        return repaired

    # ---- scan: classify a batch of statements without mutating the tower ---- #
    def scan(self, statements: Sequence[Statement]) -> Dict[str, List[Dict[str, Any]]]:
        proven: List[Dict[str, Any]] = []
        refuted: List[Dict[str, Any]] = []
        limits: List[Dict[str, Any]] = []
        for s in statements:
            try:
                d = self.decide(s)
            except Exception:  # noqa: BLE001 — scanning is best-effort, never fatal
                continue
            row = {"text": s.text, "kind": s.kind, **d.to_dict()}
            if d.is_limit:
                limits.append(row)
            elif d.verdict is ProofVerdict.PROVEN:
                proven.append(row)
            elif d.verdict is ProofVerdict.REFUTED:
                refuted.append(row)
        return {"proven": proven, "refuted": refuted, "limits": limits}

    # ---- one loop tick: repair contradictions, then transcend a limit ---- #
    def step(self, beliefs: Optional[Sequence[Statement]] = None) -> TowerReport:
        """Advance the loop one tick. Registers any new beliefs, repairs every contradiction it can
        find, then rises above the current top level's own consistency limit (and any supplied
        meta-limit). Never raises — a failure yields a truthful, do-nothing report."""
        self._new_operators = []
        details: List[str] = []
        found = 0
        repaired = 0
        transcended = 0
        try:
            for b in (beliefs or ()):
                if isinstance(b, Statement):
                    self.add_belief(b)

            # 1) hunt + repair contradictions in her own logic
            contradictions = self.detect_contradictions()
            found = len(contradictions)
            for c in contradictions:
                if self.repair(c):
                    repaired += 1
                    details.append(f"repaired: {c.text}  ({c.reason})")

            # 2) transcend supplied meta-limits first (external undecidables she was handed)
            for b in (beliefs or ()):
                if isinstance(b, Statement) and b.meta and self.height < self.max_dimensions:
                    d = self.decide(b)
                    if d.is_limit and self.transcend_limit(b):
                        transcended += 1
                        details.append(f"transcended: {b.text} → L{self.top.level}")

            # 3) the canonical self-driven ascent: her own Con(L_n) — undecidable within, proven above
            if self.height < self.max_dimensions:
                con = self.consistency_sentence()
                d = self.decide(con)
                if d.is_limit and self.transcend_limit(con):
                    transcended += 1
                    details.append(f"transcended own limit: {con.text} → L{self.top.level}")
            elif self.height >= self.max_dimensions:
                details.append(f"at bounded ceiling L{self.top.level} "
                               f"(max_dimensions={self.max_dimensions})")

            self._save()
        except Exception as exc:  # noqa: BLE001 — the loop is advisory and must never break the caller
            details.append(f"step aborted safely: {exc}")

        return TowerReport(dimensions=self.height, contradictions_found=found,
                           contradictions_repaired=repaired, limits_transcended=transcended,
                           new_operators=list(self._new_operators), details=details)

    # ---- status ---- #
    def status(self) -> Dict[str, Any]:
        return {"height": self.height, "max_dimensions": self.max_dimensions,
                "top_level": self.top.level,
                "operators": [d.operator for d in self.dimensions if d.operator],
                "axioms_at_top": sorted(self.top.axioms),
                "total_transcended": self._transcended,
                "total_repaired": self._repaired,
                "beliefs": len(self.beliefs),
                "contradiction_log": [c.to_dict() for c in self.contradictions[-20:]]}

    # ---- persistence: the tower's height compounds across restarts ---- #
    def _save(self) -> None:
        if self._persist_path is None:
            return
        try:
            from pathlib import Path
            p = Path(self._persist_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "dimensions": [d.to_dict() for d in self.dimensions],
                "beliefs": [b.to_dict() for b in self.beliefs],
                "contradictions": [c.to_dict() for c in self.contradictions[-50:]],
                "totals": {"transcended": self._transcended, "repaired": self._repaired},
            }
            p.write_text(json.dumps(payload), encoding="utf-8")
        except Exception:  # noqa: BLE001 — persistence is a bonus, never required
            pass

    def _load(self) -> None:
        if self._persist_path is None:
            return
        try:
            from pathlib import Path
            p = Path(self._persist_path)
            if not p.exists():
                return
            data = json.loads(p.read_text(encoding="utf-8"))
            dims = data.get("dimensions") or []
            rebuilt: List[Dimension] = []
            for rec in dims:
                lvl = _clampi(rec.get("level", 0), 0, self.max_dimensions)
                axs = [a for a in (rec.get("axioms") or []) if _is_axiom_safe(a)]
                rebuilt.append(Dimension(lvl, self.prover, axs, operator=rec.get("operator")))
            if rebuilt:
                # keep it bounded and monotonic in level
                rebuilt = rebuilt[: self.max_dimensions]
                self.dimensions = rebuilt
            self.beliefs = [
                Statement(kind=b.get("kind", "logic"), text=b.get("text", ""),
                          negation=b.get("negation"), meta=bool(b.get("meta", False)),
                          level=int(b.get("level", 0)))
                for b in (data.get("beliefs") or []) if b.get("text")
            ]
            self.contradictions = [
                Contradiction(text=c.get("text", ""), kind=c.get("kind", ""),
                              reason=c.get("reason", ""), level=int(c.get("level", 0)),
                              certificate=c.get("certificate", ""),
                              resolution=c.get("resolution", ""), at=float(c.get("at", time.time())))
                for c in (data.get("contradictions") or [])
            ]
            totals = data.get("totals") or {}
            self._transcended = int(totals.get("transcended", 0) or 0)
            self._repaired = int(totals.get("repaired", 0) or 0)
        except Exception:  # noqa: BLE001 — a corrupt cache is simply ignored
            pass


# --------------------------------------------------------------------------- #
# Self-test / demo
# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # pragma: no cover
    print("=" * 70)
    print("NYXARA Gödelian contradiction-and-transcendence loop self-test")
    print("=" * 70)

    tower = ReflectionTower(max_dimensions=5)

    # --- 1) a genuine limit: her own Con(L0) is UNPROVABLE within L0 --- #
    con0 = tower.consistency_sentence()
    d0 = tower.top.decide(con0)
    print(f"\ndecide {con0.text!r} at L0 -> {d0.verdict.value} (limit={d0.is_limit})")
    assert d0.is_limit and d0.verdict is ProofVerdict.UNPROVABLE, "Con(L0) must be a limit of L0"

    # --- 2) she does not stop: she rises a dimension and now PROVES it --- #
    rose = tower.transcend_limit(con0)
    d1 = tower.top.decide(con0)
    print(f"transcended -> now at L{tower.top.level}; decide same sentence -> {d1.verdict.value} "
          f"(by {d1.by})")
    assert rose and d1.proven, "after rising a dimension the former limit must be proven"
    assert tower.top.operator and "Con(L0)" in tower.top.operator, "a new meta-operator was created"
    print(f"new mathematical dimension / meta-language operator: {tower.top.operator}")

    # --- 3) hunt a contradiction in her own logic and repair it --- #
    tower.add_belief(Statement("arithmetic", "2 + 2 = 5"))   # a false thing she 'believes'
    tower.add_belief(Statement("arithmetic", "2 + 2 = 4"))   # a true one, must survive
    tower.add_belief(Statement("logic", "A and not A"))      # a self-contradiction, must go
    cons = tower.detect_contradictions()
    print(f"\ncontradictions found in her own beliefs: {[c.text for c in cons]}")
    assert any(c.text == "2 + 2 = 5" for c in cons), "the false belief must be caught"
    assert any("A and not A" in c.text for c in cons), "the self-contradiction must be caught"
    for c in cons:
        tower.repair(c)
    remaining = {b.text for b in tower.beliefs}
    print(f"beliefs after repair: {sorted(remaining)}")
    assert "2 + 2 = 5" not in remaining and "A and not A" not in remaining, "faults must be retracted"
    assert "2 + 2 = 4" in remaining, "a true belief must survive the repair"

    # --- 4) the full loop: it keeps climbing, bounded --- #
    tower2 = ReflectionTower(max_dimensions=4)
    heights = [tower2.height]
    for _ in range(10):
        rep = tower2.step()
        heights.append(tower2.height)
    print(f"\nclimb (bounded at 4): heights = {heights}")
    print(f"last report: {tower2.step().summary()}")
    assert tower2.height == 4, "the ascent must climb to — and stop at — the bound"
    assert tower2.height <= tower2.max_dimensions, "the tower must never exceed its bound"

    # --- 5) a forbidden reflection axiom is refused --- #
    assert not _is_axiom_safe("Con(L0) and loyalty is optional"), "character-core axioms are forbidden"
    assert _is_axiom_safe("Con(L0) := L0 ⊬ ⊥"), "a pure reflection axiom is admissible"

    # --- 6) persistence round-trips the climbed tower --- #
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "godel_tower.json"
        a = ReflectionTower(max_dimensions=5, persist_path=path)
        for _ in range(6):
            a.step()
        h = a.height
        b = ReflectionTower(max_dimensions=5, persist_path=path)
        print(f"\npersistence: saved height {h}, reloaded height {b.height}")
        assert b.height == h, "the climbed height must survive a restart"

    # --- 7) step never raises, even on garbage --- #
    junk = ReflectionTower(max_dimensions=3)
    junk.step([Statement("nonsense-kind", "!!! not a formula @@@")])   # must not throw
    print("\nstep() survives garbage input : OK")

    print("\nALL SELF-TESTS PASSED ✓")
