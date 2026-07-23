"""Stage C — vectorized (HDC) reasoning, certified by the exact prover.

The 10,000-D hyperdimensional space and the exact z3/sympy prover used to be disjoint. These tests
prove the bridge: HDC *proposes* by holographic recall, and every proposal is *certified* (or the
reasoner abstains). The headline honest property — as the field saturates, a wrong recall is caught
and she abstains instead of asserting a hallucination — is exercised directly.
"""

from __future__ import annotations

from nyxara.growth.prover import ProofClaim
from nyxara.growth.vsa_reasoner import VSAReasoner, VSAVerdict


def test_holographic_recall_is_certified_proven():
    r = VSAReasoner(dim=10000, seed=7)
    r.assert_fact("chess", "part_of", "strategy")
    r.assert_fact("strategy", "part_of", "defense")
    r.assert_fact("nyxara", "owned_by", "master")
    res = r.infer("chess", "part_of")
    assert res.verdict is VSAVerdict.PROVEN
    assert res.proposal == "strategy"
    assert res.certified is True
    assert res.hdc_confidence > 0.15


def test_abstains_when_nothing_asserted():
    r = VSAReasoner()
    res = r.infer("a", "b")
    assert res.verdict is VSAVerdict.ABSTAIN
    assert r.report()["facts"] == 0


def test_uncertified_proposal_is_abstained_not_asserted():
    """A query whose (subject, predicate) has no true object must never return a confident lie."""
    r = VSAReasoner(dim=10000, seed=3)
    r.assert_fact("chess", "part_of", "strategy")
    # ask about a subject/predicate pair that was never asserted
    res = r.infer("poker", "part_of")
    assert res.verdict is VSAVerdict.ABSTAIN
    assert res.certified is False


def test_field_saturation_forces_honest_abstention_not_hallucination():
    """As the holographic field saturates, HDC recall degrades. The certifier must catch every
    wrong snap: over many queries the reasoner returns PROVEN only for genuinely-true facts and
    ABSTAINS on the rest — it must never certify a triple that is not in the fact base."""
    r = VSAReasoner(dim=2048, seed=11)  # small dim → saturates fast, stressing the guarantee
    truth = {}
    for i in range(60):
        s, p, o = f"s{i}", "rel", f"o{i}"
        r.assert_fact(s, p, o)
        truth[(s, p)] = o
    false_certifications = 0
    correct = 0
    for (s, p), o in truth.items():
        res = r.infer(s, p)
        if res.verdict is VSAVerdict.PROVEN:
            # the safety property: anything PROVEN must be an actual fact
            assert (s, p, res.proposal) in r._facts
            if res.proposal == o:
                correct += 1
            else:
                false_certifications += 1
    # certification is sound: zero false facts ever asserted, even under heavy saturation
    assert false_certifications == 0
    # and it still recalls a real, non-trivial fraction correctly (the vector layer earns its keep)
    assert correct > 0


def test_check_fact_proves_true_and_refutes_false():
    r = VSAReasoner(seed=5)
    r.assert_fact("nyxara", "is_a", "sovereign_cognitive_agent")
    assert r.check_fact("nyxara", "is_a", "sovereign_cognitive_agent").verdict is VSAVerdict.PROVEN
    assert r.check_fact("nyxara", "is_a", "toaster").verdict is VSAVerdict.REFUTED


def test_math_claims_route_to_exact_prover():
    r = VSAReasoner()
    ok = r.prove(ProofClaim(kind="arithmetic", statement="2 + 2 = 4"))
    assert ok.verdict is VSAVerdict.PROVEN
    assert ok.certified is True and "prover" in ok.method
    bad = r.prove(ProofClaim(kind="arithmetic", statement="2 + 2 = 5"))
    assert bad.verdict is VSAVerdict.REFUTED
