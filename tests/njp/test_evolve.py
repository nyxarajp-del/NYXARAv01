"""Self-rewriting: it must refuse on absent evidence, and undo itself byte-for-byte."""

from __future__ import annotations


from nyxara.njp.evolve import Profiler, SelfEvolver, is_protected
from nyxara.njp.ledger import Ledger


class _Scram:
    def gate(self):
        return False


class _Unreadable:
    @property
    def gate(self):
        raise RuntimeError("oversight cannot be read")


def test_the_constitutional_core_is_refused():
    for target in ("nyxara/kernel/rules.py", "nyxara/identity/values.py",
                   "nyxara/guard/corrigibility.py", "nyxara.growth.loyalty"):
        assert is_protected(target) is True


def test_an_unreadable_oversight_is_treated_as_stop():
    """Fail-closed: cannot tell ⇒ do not act. Never 'carry on'."""
    assert is_protected("") is True
    e = SelfEvolver(None, ledger=Ledger())
    assert e.beat(oversight=_Unreadable(), force=True) is None


def test_a_scrammed_mind_does_not_rewrite_itself():
    e = SelfEvolver(None, ledger=Ledger())
    assert e.beat(oversight=_Scram(), force=True) is None


def test_it_names_no_target_without_enough_samples():
    """A cold profiler proposes nothing rather than picking something at random."""
    e = SelfEvolver(None, ledger=Ledger())
    step = e.beat(force=True)
    assert step is not None
    assert step.proposed is False
    assert "samples" in step.reason


def test_the_slowest_module_is_the_one_chosen():
    p = Profiler(min_calls=2)
    for _ in range(3):
        p.observe("nyxara.njp.fabric", 50.0)
        p.observe("nyxara.njp.tongue", 1.0)
    assert p.worst().module == "nyxara.njp.fabric"


def test_a_protected_module_is_never_the_chosen_target():
    p = Profiler(min_calls=2)
    for _ in range(3):
        p.observe("nyxara.kernel.rules", 500.0)     # by far the slowest
        p.observe("nyxara.njp.tongue", 1.0)
    assert p.worst().module == "nyxara.njp.tongue"


def test_an_improvement_claim_without_evidence_is_refused():
    """No measurement hook ⇒ the predictive source fails ⇒ the edit never reaches disk.

    Refusing on absent evidence is the design. The alternative is a self-editor that keeps
    changes it never checked.
    """
    e = SelfEvolver(None, ledger=Ledger())
    p = e.profiler
    for _ in range(20):
        p.observe("nyxara.njp.tongue", 9.0)
    cost = p.worst()
    judged = e.verify_improvement(object(), cost)
    assert judged is not None
    assert judged.verdict != "established"


def test_a_regressed_generation_blocks_the_next_edit():
    """An edit that made her worse must stop the next one, not compound with it."""
    led = Ledger()
    led.record(fabric_stats={"cells": 10, "synapses": 20}, accuracy=0.9)
    led.record(fabric_stats={"cells": 10, "synapses": 20}, accuracy=0.2)
    e = SelfEvolver(None, ledger=led)
    for _ in range(20):
        e.profiler.observe("nyxara.njp.tongue", 9.0)
    step = e.beat(force=True)
    assert step.proposed is False
    assert "regressed" in step.reason


def test_a_failed_gauntlet_restores_the_file_byte_for_byte(tmp_path):
    """The rollback contract, exercised on a real file with a gauntlet that always fails."""
    from nyxara.growth.self_optimize import GauntletResult, Optimizer, SourceEdit

    target = tmp_path / "victim.py"
    original = "def f():\n    return 1\n"
    target.write_text(original, encoding="utf-8")

    opt = Optimizer(root=tmp_path, gauntlet=lambda _files: GauntletResult(
        ok=False, detail="refused on purpose"), require_improvement=False)
    edit = SourceEdit(file=str(target), kind="performance", before=original,
                      after="def f():\n    return 2\n", rationale="test")
    outcome = opt.apply(edit)

    assert outcome.kept is False
    assert target.read_text(encoding="utf-8") == original
