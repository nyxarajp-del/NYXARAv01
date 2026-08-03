"""NYXARA · growth/vsa_reasoner.py — Vectorized reasoning, certified by the exact prover (🌌, Stage C).

Two real subsystems already existed but were **disjoint**: the exact symbolic prover
(:mod:`nyxara.growth.prover` — z3 / sympy / stdlib decision procedures) and the 10,000-D
hyperdimensional space (:mod:`nyxara.cognition.hyper_dimensional_vectors`). This module bridges them
into a single discipline the Master asked for — *propose in vector space, then **prove** it*:

* **Relational inference in hyperspace.** Facts ``(subject, predicate, object)`` are encoded as
  role-bound hypervectors and superposed into one **holographic field**; a query ``(subject,
  predicate)`` is answered by a single *unbind + cleanup* — associative, near-constant work over the
  whole field, not a per-fact scan. This is the fast, geometric **proposal**.
* **Certification (never trust the vector alone).** Every HDC proposal is checked against ground
  truth — the asserted fact base — before it is returned. When the field saturates (HDC's honest
  capacity limit) and cleanup starts snapping to the *wrong* symbol, the certifier catches it and she
  **ABSTAINS** instead of confidently returning a hallucinated answer. So on decidable questions the
  probability of asserting a *false* conclusion is driven toward zero **by certification** — not by
  the impossible claim that a probabilistic guess is exactly 0.
* **Decidable math / logic** claims are routed straight to the exact :class:`~nyxara.growth.prover.Prover`
  (arithmetic / algebra / logic / inequality / number-theory) and returned through the same
  PROVEN / REFUTED / ABSTAIN verdict — the prover, never token-guessing, is always the disposer.

Pure standard library (numpy only accelerates the HDC space); **no LLM anywhere**. Degrades honestly:
with no facts it abstains, and it never invents a certificate it cannot back.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional, Set, Tuple

from nyxara.cognition.hyper_dimensional_vectors import HyperSpace, ItemMemory
from nyxara.growth.prover import ProofClaim, ProofResult, ProofVerdict, Prover

__all__ = ["VSAVerdict", "VSAResult", "VSAReasoner"]


class VSAVerdict(str, Enum):
    PROVEN = "proven"       # HDC proposed it AND the exact check certified it
    REFUTED = "refuted"     # certified false
    ABSTAIN = "abstain"     # HDC unsure, field saturated, or outside the decidable domain — never bluff


@dataclass
class VSAResult:
    """A vector-space proposal plus its exact certificate — the witness anyone can re-check."""

    query: str
    verdict: VSAVerdict
    proposal: Optional[str] = None      # what HDC proposed (the cleaned-up symbol / answer)
    certificate: str = ""
    method: str = ""
    hdc_confidence: float = 0.0         # cleanup cosine of the proposal in [-1, 1]
    certified: bool = False             # did the exact check confirm the proposal?

    @property
    def proven(self) -> bool:
        return self.verdict is VSAVerdict.PROVEN

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "verdict": self.verdict.value,
            "proposal": self.proposal,
            "certificate": self.certificate,
            "method": self.method,
            "hdc_confidence": round(float(self.hdc_confidence), 4),
            "certified": self.certified,
        }


class VSAReasoner:
    """Propose with hyperdimensional algebra; dispose with the exact prover / ground truth."""

    def __init__(self, *, dim: int = 10000, seed: int = 1337,
                 prover: Optional[Prover] = None, cleanup_floor: float = 0.15) -> None:
        self.space = HyperSpace(dim, seed=seed)
        self.sym = ItemMemory(self.space)      # all atomic symbols (roles / subjects / predicates / objects)
        self.obj = ItemMemory(self.space)      # object vectors only — the cleanup targets
        self.prover = prover or Prover()
        # below this cleanup cosine the recall is treated as noise → abstain (honest capacity limit)
        self.cleanup_floor = float(cleanup_floor)
        self._field = self.space.zero()        # the holographic superposition of every fact
        self._facts: Set[Tuple[str, str, str]] = set()
        self.queries = 0
        self.abstentions = 0

    # ------------------------------------------------------------------ #
    # Fact base
    # ------------------------------------------------------------------ #
    def assert_fact(self, subject: str, predicate: str, obj: str) -> None:
        """Add ``(subject, predicate, object)`` to the fact base and fold it into the field."""
        s, p, o = subject.strip(), predicate.strip(), obj.strip()
        if (s, p, o) in self._facts:
            return
        # the object vector is shared between the binding vocabulary and the cleanup targets
        o_vec = self.obj.symbol(o)
        self.sym.add(o, o_vec)
        fact_vec = self.space.bind(self._key(s, p), o_vec)
        self._field = self.space.bundle(self._field, fact_vec)
        self._facts.add((s, p, o))

    def _key(self, subject: str, predicate: str) -> Any:
        """The (subject, predicate) retrieval key: bind their role-tagged atomic vectors."""
        return self.space.bind(self.sym.symbol(f"S:{subject.strip()}"),
                               self.sym.symbol(f"P:{predicate.strip()}"))

    # ------------------------------------------------------------------ #
    # HDC proposal + certification
    # ------------------------------------------------------------------ #
    def infer(self, subject: str, predicate: str) -> VSAResult:
        """Answer ``(subject, predicate, ?)`` by holographic recall, certified against ground truth."""
        self.queries += 1
        q = f"{subject} --{predicate}--> ?"
        if not self._facts or len(self.obj) == 0:
            self.abstentions += 1
            return VSAResult(q, VSAVerdict.ABSTAIN, method="hdc",
                             certificate="no facts asserted")
        # PROPOSE — unbind the field with the key, clean up to the nearest object symbol
        noisy = self.space.unbind(self._field, self._key(subject, predicate))
        cr = self.obj.cleanup(noisy)
        proposal, conf = cr.name, float(cr.score)
        # CERTIFY — the vector is never trusted alone: confirm against the asserted facts
        if proposal is None or conf < self.cleanup_floor:
            self.abstentions += 1
            return VSAResult(q, VSAVerdict.ABSTAIN, proposal=proposal, hdc_confidence=conf,
                             method="hdc", certificate="recall below noise floor (field saturated?)")
        certified = (subject.strip(), predicate.strip(), proposal) in self._facts
        if certified:
            return VSAResult(q, VSAVerdict.PROVEN, proposal=proposal, hdc_confidence=conf,
                             method="hdc+certify", certified=True,
                             certificate=f"({subject}, {predicate}, {proposal}) ∈ fact base")
        # HDC snapped to a symbol that is NOT a real fact — a caught hallucination → abstain, never assert
        self.abstentions += 1
        return VSAResult(q, VSAVerdict.ABSTAIN, proposal=proposal, hdc_confidence=conf,
                         method="hdc+certify", certified=False,
                         certificate=f"proposal ({subject}, {predicate}, {proposal}) not certified")

    def check_fact(self, subject: str, predicate: str, obj: str) -> VSAResult:
        """Verify a specific triple: HDC similarity proposes, the fact base certifies PROVEN/REFUTED."""
        self.queries += 1
        q = f"({subject}, {predicate}, {obj})?"
        if not self._facts:
            self.abstentions += 1
            return VSAResult(q, VSAVerdict.ABSTAIN, method="hdc", certificate="no facts asserted")
        # similarity of the encoded triple to the field is the vector-space evidence
        fact_vec = self.space.bind(self._key(subject, predicate), self.obj.symbol(obj.strip()))
        conf = HyperSpace.similarity(self._field, fact_vec)
        if (subject.strip(), predicate.strip(), obj.strip()) in self._facts:
            return VSAResult(q, VSAVerdict.PROVEN, proposal=obj, hdc_confidence=conf,
                             method="hdc+certify", certified=True,
                             certificate="triple ∈ fact base")
        return VSAResult(q, VSAVerdict.REFUTED, proposal=obj, hdc_confidence=conf,
                         method="hdc+certify", certified=False,
                         certificate="triple ∉ fact base")

    # ------------------------------------------------------------------ #
    # Decidable math / logic — the exact prover is always the disposer
    # ------------------------------------------------------------------ #
    def prove(self, claim: ProofClaim) -> VSAResult:
        """Route a decidable symbolic claim to the exact prover, wrapped in the unified verdict.

        There is no token-guessing: the answer is a machine-checkable certificate or an honest
        abstention. (The HDC layer proposes for *relational* recall, above; math/logic is decided
        by strict procedures, which is exactly the point — geometric/strict proof, not fluency.)"""
        self.queries += 1
        res: ProofResult = self.prover.prove(claim)
        v = {ProofVerdict.PROVEN: VSAVerdict.PROVEN,
             ProofVerdict.REFUTED: VSAVerdict.REFUTED,
             ProofVerdict.UNPROVABLE: VSAVerdict.ABSTAIN}[res.verdict]
        if v is VSAVerdict.ABSTAIN:
            self.abstentions += 1
        return VSAResult(claim.statement, v, proposal=claim.candidate_answer,
                         certificate=res.certificate, method=f"prover:{res.method}",
                         hdc_confidence=float(res.confidence),
                         certified=res.verdict is ProofVerdict.PROVEN)

    # ------------------------------------------------------------------ #
    def report(self) -> Dict[str, Any]:
        return {
            "facts": len(self._facts),
            "objects": len(self.obj),
            "queries": self.queries,
            "abstentions": self.abstentions,
            "dim": self.space.dim,
            "cleanup_floor": self.cleanup_floor,
        }
