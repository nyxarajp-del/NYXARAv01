"""Integration: the orchestrator distils a successful TEACHER turn (Part B wiring).

Confirms _feed_flywheel's epistemic hook fires only for teacher-answered owner turns and lazily builds
the distiller from the live core (reusing its KnowledgeGraph / LatentSpaceMap). Hermetic (TEST profile).
"""
from __future__ import annotations

from nyxara.kernel.orchestrator import Authority, Candidate, NyxaraCore


def _core() -> NyxaraCore:
    # default construction under the TEST profile (conftest); growth/memory on so graph+HD exist.
    return NyxaraCore()


def test_teacher_turn_distilled_into_local_knowledge():
    core = _core()
    cand = Candidate(text="42", kind="respond", confidence=0.9, rationale="cloud teacher answered")
    assert core._classify_answer_source(cand) == "teacher"

    core._feed_flywheel("what is 6*7?", "42", cand, Authority.OWNER)

    d = getattr(core, "_epistemic_distiller", None)
    assert d is not None and d.count() == 1
    assert len(core.knowledge_graph) >= 1
    # the HD-vector rule lands in the SAME map the kernel's reasoning-time recall consults,
    # so future turns can answer locally (this is the cloud-dependency-shrinking mechanism)
    assert len(core.hyperdimensional) == 1
    assert core.hyperdimensional.nearest({"condition": "what is 6*7?"}, k=1)


def test_native_turn_not_distilled():
    core = _core()
    # rationale marks her OWN chain of thought -> classified "native", so teacher_only gate skips it
    cand = Candidate(text="42", kind="respond", confidence=0.9, rationale="native reasoning chain")
    assert core._classify_answer_source(cand) == "native"
    core._feed_flywheel("what is 6*7?", "42", cand, Authority.OWNER)
    d = getattr(core, "_epistemic_distiller", None)
    assert d is None or d.count() == 0
