"""Tests for growth/certify.py — the proof reaching the data she trains on.

The bug these guard is not a crash. It is that every machine-checkable witness the system built
was discarded at the corpus boundary: `ProvenItem.to_doc()` rendered prompt and answer with
`self.certificate` in scope and unused, `synthesis._feed_flywheel` computed the proof string and
threw it away, and `DistillationExample` had no field to put one in. The model was trained on
the output of a prover it had no evidence existed.

The second property is the one that keeps it honest: sampled agreement must NOT be rendered as
a proof, or the data teaches that sampling and proving are the same act.
"""

from __future__ import annotations

from nyxara.growth.certify import (
    PROOF_CLOSE,
    PROOF_OPEN,
    Certificate,
    available_backends,
    certify_claim,
    extract_certificate,
    render_certified_doc,
)


# --------------------------------------------------------------------------- #
# The witness has to survive into the document
# --------------------------------------------------------------------------- #
def test_an_exact_certificate_is_embedded_in_the_document() -> None:
    cert = Certificate("2+2 = 4", "exact rational evaluation gives 4", "arithmetic")
    doc = render_certified_doc("What is 2+2?", "4", cert)
    assert PROOF_OPEN in doc and PROOF_CLOSE in doc
    assert "exact rational evaluation" in doc
    assert "4" in doc


def test_the_certificate_round_trips_out_of_the_document() -> None:
    cert = Certificate("s", "z3: negation unsatisfiable", "z3 (universal)")
    got = extract_certificate(render_certified_doc("q", "a", cert))
    assert got is not None and "negation unsatisfiable" in got


def test_a_document_without_a_proof_extracts_none() -> None:
    assert extract_certificate(render_certified_doc("q", "a", None)) is None
    assert extract_certificate("") is None


def test_the_answer_still_precedes_the_proof() -> None:
    """The proof is attached to the answer, not substituted for it."""
    doc = render_certified_doc("q", "THE-ANSWER", Certificate("s", "THE-WITNESS", "m"))
    assert doc.index("THE-ANSWER") < doc.index("THE-WITNESS")


def test_a_very_long_witness_is_truncated_not_dropped() -> None:
    cert = Certificate("s", "x" * 5000, "m")
    body = extract_certificate(render_certified_doc("q", "a", cert))
    assert body is not None and 0 < len(body) < 1000


# --------------------------------------------------------------------------- #
# Sampling must not wear a proof's clothes
# --------------------------------------------------------------------------- #
def test_a_sampled_result_is_never_embedded_as_a_proof() -> None:
    """Schwartz–Zippel agreement is overwhelming evidence and is not a decision procedure."""
    sampled = Certificate("s", "agrees at 24 random integer points",
                          "polynomial identity testing", exact=False)
    assert sampled.embeddable is False
    assert extract_certificate(render_certified_doc("q", "a", sampled)) is None


def test_an_empty_witness_is_not_embedded() -> None:
    assert Certificate("s", "   ", "m").embeddable is False


def test_a_proof_result_predating_the_exact_field_is_treated_as_inexact() -> None:
    """Absence of evidence about exactness must not be read as evidence of exactness."""

    class _Old:
        proven = True
        certificate = "something"
        method = "legacy"
        claim = None

    cert = Certificate.from_proof_result(_Old(), statement="s")
    assert cert is not None and cert.exact is False and cert.embeddable is False


def test_a_refuted_or_abstained_result_yields_no_certificate() -> None:
    class _NotProven:
        proven = False
        certificate = "counter-example found"
        method = "m"
        claim = None

    assert Certificate.from_proof_result(_NotProven()) is None
    assert Certificate.from_proof_result(None) is None


# --------------------------------------------------------------------------- #
# Against the real prover
# --------------------------------------------------------------------------- #
def test_certify_claim_proves_a_real_identity() -> None:
    cert = certify_claim("(x+1)^2 = x^2+2*x+1", kind="algebra")
    assert cert is not None and cert.embeddable
    assert cert.witness


def test_certify_claim_returns_nothing_for_a_falsehood() -> None:
    assert certify_claim("2+2 = 5", kind="arithmetic") is None


def test_certify_claim_returns_nothing_for_a_contingent_formula() -> None:
    """Undecided is not proved — and must not become a certificate."""
    assert certify_claim("A or B", kind="logic") is None


def test_certify_claim_proves_a_tautology() -> None:
    assert certify_claim("A or not A", kind="logic") is not None


# --------------------------------------------------------------------------- #
# The three paths where the certificate used to die
# --------------------------------------------------------------------------- #
def test_proven_item_carries_its_certificate_into_the_training_doc() -> None:
    from nyxara.growth.oracle import ProvenItem

    item = ProvenItem(domain="math", prompt="Is 91 prime?", answer="No, 91 = 7 x 13.",
                      certificate="trial division found factor 7")
    got = extract_certificate(item.to_doc())
    assert got is not None and "factor 7" in got


def test_distillation_example_stores_and_renders_a_certificate() -> None:
    from nyxara.growth.distill import DistillationExample

    ex = DistillationExample(prompt="p", answer="a", certificate="z3: unsat")
    assert extract_certificate(ex.to_training_doc()) is not None
    assert DistillationExample.from_dict(ex.to_dict()).certificate == "z3: unsat"


def test_a_record_written_before_certificates_existed_still_loads() -> None:
    from nyxara.growth.distill import DistillationExample

    old = DistillationExample.from_dict({"prompt": "p", "answer": "a", "source": "teacher"})
    assert old.certificate == "" and extract_certificate(old.to_training_doc()) is None


def test_the_flywheel_accepts_and_stores_a_certificate(tmp_path) -> None:
    from nyxara.growth.flywheel import DataFlywheel

    fw = DataFlywheel(store_path=tmp_path / "flywheel.jsonl")
    prompt = "Prove that the sum of two even integers is even."
    answer = "Let a = 2m and b = 2n; then a + b = 2(m + n), which is even."
    decision = fw.consider(prompt, answer, verified=True, certificate="algebraic construction")
    assert decision.collected, decision.reason
    stored = fw.examples()
    assert stored and stored[-1].certificate == "algebraic construction"


# --------------------------------------------------------------------------- #
# Report what is installed, not what is aspired to
# --------------------------------------------------------------------------- #
def test_backends_report_reality() -> None:
    backends = available_backends()
    assert backends["z3"] is True, "z3-solver is a pinned dependency and should be present"
    assert isinstance(backends["lean4"], bool)
