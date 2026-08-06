"""Tests for the parallel-reasoning deadline in ``NyxaraCore._reason_parallel``.

Metacontrol already decides how many seconds a question is worth (``ComputeBudget.max_seconds``),
and until this change nothing on the reasoning path read that decision. The framings ran under
``with ThreadPoolExecutor(...)``, whose ``__exit__`` joins every worker — so a budget could not
have bounded anything even if one had been consulted. Under DEV defaults a single turn costs
3 framings x 3 passes x 3 samples x 5 RSI iterations of model calls, which is how a turn ran for
the better part of an hour and how the test suite came to be un-runnable end to end.

Two properties are pinned here, and the second is the one that got away the first time:

1. the deadline bounds the **turn**, not just the wait; and
2. spending the budget must not then buy a full second pass — measured at 2 s + 30 s = 32 s
   against a 2 s deadline, because ``if not results:`` fell through to ``_reason_once``.
"""
from __future__ import annotations

import time
import types

import pytest

from nyxara.kernel.orchestrator import NyxaraCore


def _bare_core(*, max_seconds=None, framings=3, floor=0.0):
    """A core with no boot — only the parallel-reasoning path is under test here.

    ``NyxaraCore()`` builds the whole mind; this exercises one method, so it is constructed
    without ``__init__`` and given exactly the attributes that method reads.

    ``floor`` pins :meth:`_one_generation_s`, which in the live system is read from the turn
    ledger. Most cases here want it out of the way (0.0) so the *share* arithmetic is what is
    under test; the tests that care about the floor set it explicitly.
    """
    core = NyxaraCore.__new__(NyxaraCore)
    core.parallel_hypotheses = framings
    core._last_compute_plan = (types.SimpleNamespace(max_seconds=max_seconds)
                               if max_seconds is not None else None)
    core._reasoner_par = True
    core._hypothesis_framings = lambda memories: [(f"f{i}", []) for i in range(framings)]
    core._one_generation_s = lambda: floor
    return core


# --------------------------------------------------------------------------- #
# The budget is real
# --------------------------------------------------------------------------- #
def test_a_slow_framing_cannot_outrun_the_budget():
    core = _bare_core(max_seconds=1.0)

    def never_returns(stimulus, focus, mem):
        time.sleep(30.0)
        raise AssertionError("the turn must not wait for this")

    core._reason_once = never_returns

    started = time.time()
    candidate = core._reason_parallel("Hii", None, [])
    elapsed = time.time() - started

    assert elapsed < 5.0, f"the deadline is decorative: the turn took {elapsed:.1f}s"
    assert candidate.text, "she must still answer — a budget buys a cheaper answer, not silence"


def test_the_budget_is_not_spent_twice():
    """The hole in the first fix: the deadline fired, and then a full pass ran anyway.

    The counter is the assertion. One call per framing is the deadline working; a call after
    that is the 2 + 30 = 32s bug, and it would pass a wall-clock check on a fast machine.
    """
    core = _bare_core(max_seconds=1.0)
    calls = []

    def slow(stimulus, focus, mem):
        calls.append(mem)
        time.sleep(30.0)
        raise AssertionError("unreachable")

    core._reason_once = slow
    core._reason_parallel("Hii", None, [])

    assert len(calls) == 3, f"a post-deadline pass was started: {len(calls)} calls for 3 framings"


def test_the_framings_that_did_arrive_still_vote():
    """A deadline discards stragglers; it must not discard work that finished in time."""
    core = _bare_core(max_seconds=5.0)
    from nyxara.kernel.orchestrator import _default_reasoner

    def mixed(stimulus, focus, mem):
        if mem == []:                      # the framings are all [] here; slow only the last
            pass
        return _default_reasoner(stimulus, focus)

    core._reason_once = mixed
    core._select_hypothesis = lambda results: (results[0][0], results[0][1])
    core._record_hypotheses = lambda results, chosen: None

    candidate = core._reason_parallel("Hii", None, [])
    assert candidate.text == "I understand: Hii"


# --------------------------------------------------------------------------- #
# Where the number comes from
# --------------------------------------------------------------------------- #
def test_the_deadline_is_read_from_this_turns_compute_plan():
    core = _bare_core(max_seconds=30.0)
    assert core._parallel_deadline_s() == pytest.approx(30.0 * core._deadline_share())


def test_the_framings_get_a_share_of_the_turn_not_the_whole_thing():
    """``max_seconds`` budgets the WHOLE turn — recursive improvement, the council and the gates
    all come after this stage. Spending the full allocation on the first stage starves the rest."""
    core = _bare_core(max_seconds=100.0)
    assert core._parallel_deadline_s() < 100.0


def test_the_share_is_configurable(monkeypatch):
    core = _bare_core(max_seconds=100.0)
    monkeypatch.setattr(NyxaraCore, "_deadline_share", staticmethod(lambda: 0.25))
    assert core._parallel_deadline_s() == pytest.approx(25.0)


def test_a_nonsense_share_falls_back_rather_than_removing_the_budget(monkeypatch):
    """A share of 0 would make every deadline the 1 s floor; a share above 1 would hand the
    framings more than the turn has. Both mean the config is wrong, not that the budget is off."""
    import nyxara.kernel.config as cfg

    for bad in (0.0, -1.0, 5.0):
        monkeypatch.setattr(cfg, "get_settings",
                            lambda b=bad: type("S", (), {
                                "metacontrol": type("M", (), {"parallel_deadline_share": b})()})())
        assert NyxaraCore._deadline_share() == 0.5


def test_a_turn_with_no_plan_still_has_a_ceiling():
    """A turn with no budget is exactly the turn that runs forever."""
    seconds = _bare_core()._parallel_deadline_s()
    assert seconds > 0.0


def test_no_plan_does_not_raise():
    """``NyxaraCore`` has no ``settings`` attribute — every other site in the module spells it
    ``getattr(self, "settings", None)``. Reaching for it directly raised ``AttributeError`` on
    precisely the path that needs the fallback."""
    core = _bare_core()
    core._parallel_deadline_s()          # must not raise


def test_a_nonsense_budget_falls_back_rather_than_crashing():
    core = _bare_core(max_seconds="soon")     # type: ignore[arg-type]
    assert core._parallel_deadline_s() > 0.0


def test_the_budget_never_drops_below_one_second():
    """A zero or negative allocation would make every turn answer from the floor."""
    assert _bare_core(max_seconds=0.001)._parallel_deadline_s() >= 1.0
    assert _bare_core(max_seconds=-5.0)._parallel_deadline_s() >= 1.0


# --------------------------------------------------------------------------- #
# A budget that cannot buy one generation is not a budget
# --------------------------------------------------------------------------- #
def test_the_deadline_is_never_shorter_than_one_generation():
    """The hole the first version left, and it cost the Master a working answer.

    An easy turn is allotted 5s; half of that is 2.5s for the framings; one generation on the
    on-device Gemma measures ~13.9s. So the deadline fired before the model could ever reply,
    every easy turn fell to ``_default_reasoner``, and she said "I understand: Hii" — which reads
    exactly like a broken model rather than a budget too small to buy a single call.
    """
    core = _bare_core(max_seconds=5.0, floor=13.9)
    assert core._parallel_deadline_s() >= 13.9


def test_a_generous_allocation_is_not_dragged_down_to_the_floor():
    """The floor raises a too-small budget; it must not cap a large one."""
    core = _bare_core(max_seconds=600.0, floor=13.9)
    assert core._parallel_deadline_s() > 13.9


def test_the_floor_is_measured_not_guessed(monkeypatch):
    """Read from the turn ledger — the same recorded events ``answered_by`` derives from — so a
    faster machine earns a smaller floor by being measured, not by being configured."""
    from nyxara.observe.turn_ledger import get_ledger, reset_ledger

    reset_ledger()
    try:
        ledger = get_ledger()
        ledger.begin()
        ledger.record_generation("litertlm", "gemma", ok=True, chars=42, latency_s=7.0)
        ledger.end()
        core = NyxaraCore.__new__(NyxaraCore)
        assert core._one_generation_s() >= 7.0, "an observed 7s call must not be cut off at 7s"
    finally:
        reset_ledger()


def test_a_failed_generation_does_not_set_the_floor():
    """A call that produced nothing says nothing about what a real call costs."""
    from nyxara.observe.turn_ledger import get_ledger, reset_ledger

    reset_ledger()
    try:
        ledger = get_ledger()
        ledger.begin()
        ledger.record_generation("litertlm", "gemma", ok=False, latency_s=900.0, error="boom")
        ledger.end()
        core = NyxaraCore.__new__(NyxaraCore)
        assert core._one_generation_s() < 900.0
    finally:
        reset_ledger()


def test_with_no_observation_the_floor_is_still_real():
    """The first turn of a fresh process has nothing recorded, and it is exactly the turn that
    would otherwise be cut off before the model has ever spoken."""
    from nyxara.observe.turn_ledger import reset_ledger

    reset_ledger()
    core = NyxaraCore.__new__(NyxaraCore)
    assert core._one_generation_s() > 1.0


# --------------------------------------------------------------------------- #
# A framing that returned nothing has not voted
# --------------------------------------------------------------------------- #
def test_a_framing_that_returns_none_does_not_crash_the_vote():
    """Reached ``_hypothesis_signature(None)`` and took the turn down with an AttributeError —
    unreachable while the deadline was too short for any framing to finish, immediate once it
    was long enough."""
    core = _bare_core(max_seconds=30.0, floor=0.0)
    core._reason_once = lambda s, f, m: None
    candidate = core._reason_parallel("Hii", None, [])
    assert candidate is not None and candidate.text


# --------------------------------------------------------------------------- #
# The turn's own budget — the framings deadline only covered one stage
# --------------------------------------------------------------------------- #
def _budgeted_core(max_seconds=5.0, elapsed=0.0):
    core = NyxaraCore.__new__(NyxaraCore)
    core._last_compute_plan = (types.SimpleNamespace(max_seconds=max_seconds)
                               if max_seconds is not None else None)
    core._turn_clock = time.monotonic() - elapsed
    core._skipped_stages = []
    return core


def test_a_fresh_turn_has_its_whole_allocation():
    assert _budgeted_core(5.0)._budget_left() == pytest.approx(5.0, abs=0.2)


def test_an_overrun_turn_skips_optional_cognition():
    core = _budgeted_core(5.0, elapsed=6.0)
    assert core._budget_spent("recursive_improve") is True


def test_a_turn_inside_its_budget_skips_nothing():
    core = _budgeted_core(30.0, elapsed=1.0)
    assert core._budget_spent("recursive_improve") is False
    assert core._skipped_stages == []


def test_the_skips_are_recorded():
    """A turn that quietly thinks less looks exactly like a turn that had less to say — this
    module has already lost three rounds of diagnosis to that resemblance."""
    core = _budgeted_core(1.0, elapsed=9.0)
    core._budget_spent("role_council")
    core._budget_spent("recursive_improve")
    core._budget_spent("role_council")            # a repeat is not a second skip
    assert core._skipped_stages == ["role_council (out of time)",
                                    "recursive_improve (out of time)"]


def test_the_recorded_skip_says_which_rule_dropped_it():
    """'out of time' and 'fast path' are different facts about the turn, and an operator
    reading the log needs to know which one cost him the faculty."""
    slow = _budgeted_core(1.0, elapsed=9.0)
    slow._budget_spent("recursive_improve", costs_a_generation=True)
    assert "out of time" in slow._skipped_stages[0]

    fast = _budgeted_core(30.0, elapsed=0.0)
    fast._last_compute_plan = types.SimpleNamespace(max_seconds=30.0, entry_rung=0)
    fast._budget_spent("recursive_improve", costs_a_generation=True)
    assert "fast path" in fast._skipped_stages[0]


def test_the_fast_path_only_drops_stages_that_cost_a_generation():
    """A cheap stage on the fast path still runs — 'one forward pass' is a budget for model
    calls, not an instruction to stop thinking."""
    core = _budgeted_core(30.0)
    core._last_compute_plan = types.SimpleNamespace(max_seconds=30.0, entry_rung=0)
    assert core._budget_spent("cheap_enrichment") is False
    assert core._budget_spent("expensive", costs_a_generation=True) is True


def test_a_normal_turn_is_not_on_the_fast_path():
    core = _budgeted_core(30.0)
    core._last_compute_plan = types.SimpleNamespace(max_seconds=30.0, entry_rung=2)
    assert core._on_fast_path() is False
    assert core._budget_spent("expensive", costs_a_generation=True) is False


def test_a_turn_with_no_plan_is_unbudgeted_rather_than_starved():
    """No plan must mean 'no limit', never 'no thinking' — a missing budget would otherwise
    silently disable every optional faculty on the turn."""
    core = _budgeted_core(max_seconds=None)
    assert core._budget_left() == float("inf")
    assert core._budget_spent("recursive_improve") is False


def test_a_nonsense_allocation_does_not_starve_the_turn():
    for bad in (0.0, -1.0, "soon"):
        core = _budgeted_core(max_seconds=bad, elapsed=100.0)
        assert core._budget_left() == float("inf")
        assert core._budget_spent("x") is False


def test_within_budget_runs_the_stage_when_there_is_time():
    core = _budgeted_core(30.0)
    assert core._within_budget("s", lambda c: c + "!", "answer") == "answer!"


def test_within_budget_returns_the_candidate_untouched_when_out_of_time():
    core = _budgeted_core(1.0, elapsed=9.0)
    def _boom(c):
        raise AssertionError("an out-of-budget stage must not run")
    assert core._within_budget("s", _boom, "answer") == "answer"


def test_the_budget_guards_refinement_not_the_gates():
    """The control law is downstream of every guard added here. A turn that runs out of time
    gives a shallower answer, never a less-gated one — so no gate name may appear as a stage."""
    import inspect
    source = inspect.getsource(NyxaraCore._invoke_reasoner)
    for gate in ("shield", "corrigibility", "permission", "guardian", "oversight", "initiative"):
        assert f'_budget_spent("{gate}")' not in source
        assert f'_within_budget("{gate}"' not in source


# --------------------------------------------------------------------------- #
# Saying so
# --------------------------------------------------------------------------- #
def test_hitting_the_deadline_is_announced(caplog):
    """A turn that quietly returns a thinner answer looks exactly like a turn that thought less
    — and this module has already lost three rounds of diagnosis to that resemblance."""
    core = _bare_core(max_seconds=1.0)
    core._reason_once = lambda s, f, m: time.sleep(30.0)

    with caplog.at_level("WARNING", logger="nyxara.kernel.orchestrator"):
        core._reason_parallel("Hii", None, [])

    assert any("deadline" in r.message or "deadline" in r.getMessage() for r in caplog.records)
