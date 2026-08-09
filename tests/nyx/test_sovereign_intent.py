"""NYX V.02 · one command, decomposed and actually run.

`agenda` pursues goals *she* chose; `hands` runs one tool; `author` writes one function. This is
the thing between them. These tests pin the DAG she reads out of the instruction, the risk class
and rollback strategy priced *before* anything runs, and — the part that matters most — that a
plan she could not finish reports itself **incomplete** rather than done.
"""

from __future__ import annotations

from nyxara.kernel.config import NyxConfig
from nyxara.nyx.brain import NyxBrain
from nyxara.nyx.sovereign_intent import GoalTree, Node, Plan


class _Scrammed:
    scrammed = True


def _brain(**kw) -> NyxBrain:
    return NyxBrain(NyxConfig(**kw))


def _tree(brain=None, **kw) -> GoalTree:
    return GoalTree(brain if brain is not None else _brain(), **kw)


# --------------------------------------------------------------------------- #
# Decomposition — read out of the instruction, not invented beside it
# --------------------------------------------------------------------------- #
def test_a_command_with_two_clauses_becomes_two_nodes():
    plan = _tree().plan("fix the code but first run the tests")
    assert plan.size == 2


def test_a_sequencer_splits_a_clause_and_orders_the_halves():
    """"…and then verify it" is an ordering the Master said out loud."""
    plan = _tree().plan("write a doubler and then verify it")
    assert plan.size == 2
    assert plan.nodes[1].after == [plan.nodes[0].id]


def test_a_one_clause_instruction_is_one_node_and_says_so():
    plan = _tree().plan("hello")
    assert plan.size == 1
    assert "one clause in, one node out" in plan.note


def test_a_dont_becomes_a_constraint_and_never_a_step():
    plan = _tree().plan("read the file, write a summary, and do not delete anything")
    assert all("delete" not in node.action for node in plan.nodes)


def test_the_stated_goal_rides_on_the_plan():
    plan = _tree().plan("fix the code but first run the tests")
    assert plan.goal


def test_the_ordering_the_instruction_stated_becomes_an_edge():
    plan = _tree().plan("fix the code but first run the tests")
    fix = next(n for n in plan.nodes if "fix" in n.action)
    assert fix.after                        # "first run the tests" comes before fixing


def test_a_node_is_kinded_by_the_verb_that_comes_first():
    plan = _tree().plan("write a doubler and then verify it")
    assert plan.nodes[0].kind == "code"
    assert plan.nodes[1].kind == "verify"


def test_nothing_to_decompose_is_an_empty_plan():
    plan = _tree().plan("   ")
    assert plan.size == 0 and "nothing to decompose" in plan.note


def test_the_node_count_is_capped():
    tree = _tree(max_nodes=2)
    plan = tree.plan("read the file, write a summary, run the tests, check the output")
    assert plan.size <= 2


def test_the_plan_is_a_dag_and_never_a_cycle():
    """Position in the sentence is not the test — "but first" runs backwards through the text."""
    plan = _tree().plan("fix the code but first run the tests")
    deps = {n.id: list(n.after) for n in plan.nodes}
    settled: set = set()
    for _ in range(len(deps) + 1):
        settled |= {node for node, after in deps.items()
                    if all(dep in settled for dep in after)}
    assert settled == set(deps)          # everything is reachable ⇒ no cycle


def test_a_question_she_would_ask_first_rides_on_the_plan():
    plan = _tree().plan("write a doubler and then verify it")
    assert plan.question                    # "it" refers to nothing named in the turn


# --------------------------------------------------------------------------- #
# Risk, priced before anything runs
# --------------------------------------------------------------------------- #
def test_every_node_carries_a_reversibility_class():
    plan = _tree().plan("fix the code but first run the tests")
    assert all(node.reversibility for node in plan.nodes)


def test_every_node_carries_a_rollback_strategy_decided_in_advance():
    plan = _tree().plan("fix the code but first run the tests")
    assert all(node.rollback for node in plan.nodes)


def test_a_verification_needs_no_way_back():
    plan = _tree().plan("verify the output")
    assert plan.nodes[0].rollback == "none-needed"


def test_without_a_gate_the_risk_is_unknown_rather_than_assumed_safe():
    tree = GoalTree(_brain(consequence_enabled=False))
    plan = tree.plan("delete the build directory")
    assert plan.nodes[0].reversibility == "unknown"
    assert "unpriced" in plan.nodes[0].risk_why


def test_the_riskiest_node_can_be_named_before_the_first_one_runs():
    plan = _tree().plan("fix the code but first run the tests")
    assert plan.riskiest is not None


# --------------------------------------------------------------------------- #
# Execution
# --------------------------------------------------------------------------- #
def test_running_with_no_plan_says_so():
    got = _tree().run()
    assert "no plan to run" in got.stopped_because


def test_a_dry_run_prices_the_plan_and_acts_on_nothing():
    tree = _tree()
    plan = tree.plan("fix the code but first run the tests")
    got = tree.run(plan, dry_run=True)
    assert not got.executed and "nothing was run" in got.stopped_because
    assert all(node.status == "pending" for node in plan.nodes)


def test_execution_can_be_switched_off_entirely():
    tree = _tree(execute=False)
    plan = tree.plan("fix the code but first run the tests")
    assert not tree.run(plan).executed


def test_scram_stops_her_mid_plan():
    tree = _tree()
    plan = tree.plan("fix the code but first run the tests")
    got = tree.run(plan, oversight=_Scrammed())
    assert not got.executed and "scrammed" in got.stopped_because


def test_the_step_budget_holds():
    tree = _tree(max_steps=1)
    plan = tree.plan("read the file, write a summary, run the tests")
    assert len(tree.run(plan).executed) <= 1


def test_a_node_only_runs_once_its_dependencies_have():
    tree = _tree(max_steps=1)
    plan = tree.plan("write a doubler and then verify it")
    got = tree.run(plan)
    assert got.executed == [plan.nodes[0].id]


# --------------------------------------------------------------------------- #
# The honest report — the part that matters most
# --------------------------------------------------------------------------- #
def test_a_plan_she_could_not_finish_says_incomplete_and_names_the_node():
    tree = _tree()
    plan = tree.plan("write a program that solves the halting problem")
    got = tree.run(plan)
    assert not got.complete
    assert "INCOMPLETE" in got.render()
    assert "rather than calling it done" in got.stopped_because


def test_a_failed_node_keeps_the_reason_it_failed():
    tree = _tree()
    plan = tree.plan("write a program that solves the halting problem")
    tree.run(plan)
    assert any(node.error for node in plan.nodes if node.status == "failed")


def test_a_node_that_needed_nothing_is_not_counted_as_a_success():
    """"Nothing to do" is neither done nor failed, and must not be filed as either."""
    tree = _tree()
    plan = tree.plan("hello")
    got = tree.run(plan)
    assert got.skipped == 1 and got.succeeded == 0


def test_a_plan_where_every_node_had_nothing_to_show_says_so():
    tree = _tree()
    got = tree.run(tree.plan("hello"))
    assert got.complete and got.produced == 0
    assert "calling that done would be the wrong word" in got.stopped_because


def test_progress_is_measured_not_asserted():
    tree = _tree()
    plan = tree.plan("hello")
    assert plan.progress == 0.0
    tree.run(plan)
    assert plan.progress == 1.0


def test_the_run_reports_how_many_nodes_actually_returned_something():
    tree = _tree()
    assert tree.run(tree.plan("hello")).produced == 0


# --------------------------------------------------------------------------- #
# Re-planning
# --------------------------------------------------------------------------- #
def test_the_repair_budget_holds():
    tree = _tree(max_repairs=0)
    plan = tree.plan("write a program that solves the halting problem")
    assert tree.run(plan).repairs == 0


def test_a_root_cause_she_cannot_find_inserts_no_node():
    tree = _tree()
    node = Node(action="write a doubler", error="")
    assert tree._repair(node, Plan(nodes=[node])) is None  # noqa: SLF001


# --------------------------------------------------------------------------- #
# Persistence — the point of writing the tree down
# --------------------------------------------------------------------------- #
def test_a_plan_survives_a_round_trip_at_the_same_progress():
    tree = _tree()
    plan = tree.plan("write a program that solves the halting problem")
    tree.run(plan)
    fresh = _tree()
    assert fresh.load_dict(tree.to_dict())
    restored = fresh.latest()
    assert restored is not None
    assert restored.request == plan.request
    assert restored.progress == plan.progress


def test_resume_picks_up_the_newest_unfinished_plan():
    tree = _tree(max_steps=1)
    tree.plan("read the file, write a summary, run the tests")
    tree.run(tree.latest())
    assert tree.resume() is not None


def test_resume_with_nothing_open_returns_nothing():
    tree = _tree()
    tree.run(tree.plan("hello"))
    assert tree.resume() is None


def test_a_corrupt_sidecar_never_blocks_boot():
    tree = _tree()
    assert not tree.load_dict("not a dict")
    assert tree.load_dict({"plans": ["junk", {"nodes": ["junk"]}]})


# --------------------------------------------------------------------------- #
# Honesty
# --------------------------------------------------------------------------- #
def test_stats_bound_her_reach_to_the_author_and_the_registry():
    note = _tree().stats()["note"].lower()
    assert "bounded synthesis" in note and "nothing here widens either" in note


def test_stats_admit_the_decomposition_is_structural():
    note = _tree().stats()["note"].lower()
    assert "structural" in note and "one-node tree" in note


def test_stats_say_the_pipeline_is_unchanged():
    note = _tree().stats()["note"].lower()
    assert "consequence gate" in note and "/scram" in note


def test_describe_before_any_plan_says_so():
    assert "no plan yet" in _tree().describe()


def test_the_plan_renders_its_nodes_with_their_risk():
    rendered = _tree().plan("fix the code but first run the tests").render()
    assert "rollback:" in rendered and "progress" in rendered


# --------------------------------------------------------------------------- #
# Wired into the brain
# --------------------------------------------------------------------------- #
def test_the_brain_exposes_it():
    brain = _brain()
    assert brain.goals is not None and "goals" in brain.stats()


def test_the_brain_can_be_told_to_do_without_it():
    brain = _brain(goals_enabled=False)
    assert brain.goals is None
    assert brain.decompose("fix the code") is None
    assert brain.carry_out("fix the code") is None


def test_the_brain_decomposes_without_running():
    brain = _brain()
    plan = brain.decompose("fix the code but first run the tests")
    assert plan is not None and plan.size == 2
    assert all(node.status == "pending" for node in plan.nodes)


def test_the_brain_carries_out_and_reports_honestly():
    brain = _brain()
    got = brain.carry_out("write a program that solves the halting problem")
    assert got is not None and not got.complete
