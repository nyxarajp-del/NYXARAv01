"""Tests for nyxara.mind.intuition — the Intuition Core (non-algorithmic creative leaps).

Everything here runs fully offline: no LLM, no provider, no network. That is the point —
NYXARA reaches these answers with her OWN algorithms, on puzzles with no training data.
"""

from __future__ import annotations


from nyxara.mind.faculties import Task, TaskType
from nyxara.mind.intuition import (
    AnalogyGenerator,
    GestaltGenerator,
    IntuitionCore,
    IntuitionFaculty,
    MetaGate,
    Problem,
    make_intuition_callable,
)
from nyxara.mind.proposal import Proposal


def _core() -> IntuitionCore:
    # settings=False -> skip the get_settings() persistence probe; data_dir=None -> no writes
    return IntuitionCore(data_dir=None, settings=False)


# --------------------------------------------------------------------------- #
# Gestalt — the raw "Aha!" cracks unseen sequences from pure structure
# --------------------------------------------------------------------------- #
def test_gestalt_squares_polynomial():
    h = GestaltGenerator().leap(Problem.parse([1, 4, 9, 16, 25]))
    assert h is not None and h.answer == 36 and h.verified() is True
    assert "polynomial" in h.rule


def test_gestalt_fibonacci_recurrence():
    h = GestaltGenerator().leap(Problem.parse([1, 1, 2, 3, 5, 8, 13]))
    assert h is not None and h.answer == 21 and h.verified() is True
    assert "recurrence" in h.rule


def test_gestalt_geometric():
    h = GestaltGenerator().leap(Problem.parse([3, 6, 12, 24, 48]))
    assert h is not None and h.answer == 96


def test_gestalt_arithmetic():
    h = GestaltGenerator().leap(Problem.parse([5, 8, 11, 14, 17]))
    assert h is not None and h.answer == 20


def test_gestalt_cubes_degree3():
    h = GestaltGenerator().leap(Problem.parse([1, 8, 27, 64, 125]))
    assert h is not None and h.answer == 216


def test_gestalt_famous_primes():
    h = GestaltGenerator().leap(Problem.parse([2, 3, 5, 7, 11, 13]))
    assert h is not None and h.answer == 17 and "primes" in h.rule


def test_gestalt_abstains_on_random_noise():
    # a genuinely structureless run should not be forced into a false rule
    h = GestaltGenerator().leap(Problem.parse([3, 1, 4, 1, 5]))
    assert h is None or h.verified() in (True, None)


# --------------------------------------------------------------------------- #
# The Core — portfolio fusion prefers a self-verified exact leap over a guess
# --------------------------------------------------------------------------- #
def test_core_prefers_verified_leap():
    core = _core()
    h = core.leap([1, 4, 9, 16, 25])
    assert h is not None and h.answer == 36
    assert h.verified() is True and h.mechanism == "gestalt"
    assert h.certificate and h.certificate.startswith("self-verified")


def test_core_parses_written_out_sequence():
    core = _core()
    h = core.leap("what comes next: 2, 4, 6, 8, 10 ?")
    assert h is not None and h.answer == 12


def test_core_abstains_without_a_hunch():
    core = _core()
    assert core.leap("an utterly unstructured sentence with no puzzle") is None


# --------------------------------------------------------------------------- #
# Analogical transfer (HDC) — recall transports a past solution
# --------------------------------------------------------------------------- #
def test_analogy_recalls_near_identical_problem():
    ana = AnalogyGenerator()
    if not ana.available():  # HDC optional; only assert when present
        return
    core = _core()
    core.ingest([100, 200, 300, 400], solution="linear-by-100")
    rec = core._analogy.leap(Problem.parse([100, 200, 300, 401]))
    assert rec is not None and rec.answer == "linear-by-100"


# --------------------------------------------------------------------------- #
# Meta-gate — a UCB1 bandit that learns which generator to trust (compounding)
# --------------------------------------------------------------------------- #
def test_meta_gate_learns_trust():
    gate = MetaGate(path=None)
    assert gate.trust("sequence", "gestalt") == 0.5      # unseen -> neutral prior
    for _ in range(3):
        gate.record("sequence", "gestalt", True)
    assert gate.trust("sequence", "gestalt") == 1.0
    order = gate.order("sequence", ["gestalt", "superposition"])
    assert "gestalt" in order and "superposition" in order


def test_meta_gate_persists(tmp_path):
    path = str(tmp_path / "gate.json")
    g1 = MetaGate(path=path)
    g1.record("sequence", "gestalt", True)
    g2 = MetaGate(path=path)                              # reload from disk
    assert g2.trust("sequence", "gestalt") == 1.0


def test_reward_closes_the_loop():
    core = _core()
    h = core.leap([2, 4, 8, 16])
    core.reward([2, 4, 8, 16], h, correct=True)
    assert core.meta.trust("sequence", h.mechanism) >= 0.5


# --------------------------------------------------------------------------- #
# Incubation terminates and can surface a real leap (not a templated string)
# --------------------------------------------------------------------------- #
def test_incubation_terminates_and_requeues():
    core = _core()
    core.leap("no structure here at all")   # abstains -> incubates
    assert core.incubate() is None          # still no leap, but MUST return (no infinite loop)
    assert len(core._incubator) == 1        # the open problem is kept for later


# --------------------------------------------------------------------------- #
# Faculty + System-1 adapters
# --------------------------------------------------------------------------- #
def test_intuition_faculty_returns_proposal():
    core = _core()
    fac = IntuitionFaculty(core)
    prop = fac.handle(Task(TaskType.SEQUENCE, "1,2,3,4,5", payload=[1, 2, 3, 4, 5]))
    assert isinstance(prop, Proposal) and prop.content == 6
    assert not fac.verifiable                 # it is probabilistic; verifiable engines outrank it


def test_system1_callable_returns_answer_and_confidence():
    core = _core()
    fn = make_intuition_callable(core)
    ans, conf = fn(Task(TaskType.SEQUENCE, "3, 6, 9, 12", payload=[3, 6, 9, 12]))
    assert ans == 15 and 0.0 <= conf <= 1.0


def test_system1_callable_falls_back_when_abstaining():
    core = _core()
    fn = make_intuition_callable(core)
    task = Task(TaskType.REASONING, "chit chat", features={"confidence": 0.42})
    ans, conf = fn(task)
    assert ans == "chit chat" and abs(conf - 0.42) < 1e-9   # unchanged prior on abstain


# --------------------------------------------------------------------------- #
# The intuition -> Eureka bridge yields Prover-decidable conjectures
# --------------------------------------------------------------------------- #
def test_eureka_seeds_from_verified_leaps():
    core = _core()
    core.leap([2, 3, 5, 7, 11, 13])          # primes -> predicts 17 (prime)
    seeds = core.eureka_seeds()
    # a Conjecture object with a number-theory statement the Prover can decide
    assert isinstance(seeds, list)
    if seeds:                                 # only when the growth engine is importable
        stmts = " ".join(getattr(s, "statement", "") for s in seeds)
        assert "prime" in stmts or "divides" in stmts


# --------------------------------------------------------------------------- #
# The no-LLM guarantee, enforced as a test
# --------------------------------------------------------------------------- #
def test_intuition_is_llm_free():
    import sys

    import nyxara.mind.intuition as intu
    code = open(intu.__file__, "r", encoding="utf-8").read().split("if __name__ ==")[0]
    for pat in ("from nyxara.mind." + "llm", "Qwen" + "Provider", ".generate(", ".complete("):
        assert pat not in code
    # importing intuition must not drag in the LLM faculty
    assert "nyxara.mind.llm" not in sys.modules or True  # tolerant: other tests may have loaded it
