"""Tests for nyxara.eval.continual — the continual-learning (forgetting) benchmark."""

from __future__ import annotations

import json

import pytest

from nyxara.eval.continual import (ContinualMetrics, ContinualTask, evaluate_continual,
                                   make_task_suite, matrix_metrics, run_sequential)
from nyxara.growth.learn import Learner


# -------------------- the task suite -------------------- #
def test_suite_is_deterministic_for_a_seed():
    a = make_task_suite(3, 20, seed=11)
    b = make_task_suite(3, 20, seed=11)
    assert [t.name for t in a] == [t.name for t in b]
    assert a[0].train == b[0].train
    assert a[2].test == b[2].test


def test_suite_tasks_share_features_by_construction():
    tasks = make_task_suite(2, 10, seed=1)
    feats0 = set(tasks[0].train[0][0])
    feats1 = set(tasks[1].train[0][0])
    assert {"shared_a", "shared_b"} <= (feats0 & feats1)   # the interference channel
    assert feats0 != feats1                                 # ...plus per-task context


# -------------------- the metrics, on a hand-built matrix -------------------- #
def test_matrix_metrics_formulas():
    # 3 tasks: task 0 is learned perfectly then fully forgotten; task 2 is learned last
    matrix = [
        [1.0, 0.5, 0.5],
        [0.5, 1.0, 0.5],
        [0.0, 1.0, 1.0],
    ]
    m = matrix_metrics(matrix, untrained=[0.5, 0.5, 0.5])
    assert m.average_accuracy == pytest.approx((0.0 + 1.0 + 1.0) / 3)
    assert m.forgetting_per_task[0] == pytest.approx(1.0)   # max 1.0 -> final 0.0
    assert m.forgetting_per_task[1] == pytest.approx(0.0)
    assert m.forgetting == pytest.approx((1.0 + 0.0 + 0.0) / 3)
    assert m.bwt == pytest.approx(((0.0 - 1.0) + (1.0 - 1.0)) / 2)
    assert m.fwt == pytest.approx(((0.5 - 0.5) + (0.5 - 0.5)) / 2)


def test_metrics_to_dict_is_json_serializable():
    m = matrix_metrics([[1.0]])
    json.dumps(m.to_dict())


# -------------------- sequential training -------------------- #
def test_run_sequential_returns_full_matrix():
    tasks = make_task_suite(3, 15, seed=2)
    matrix, learner = run_sequential(lambda: Learner(base_lr=0.1, seed=2), tasks)
    assert len(matrix) == 3 and all(len(row) == 3 for row in matrix)
    assert isinstance(learner, Learner)
    assert matrix[0][0] > 0.8          # the first task is actually learned


def test_run_sequential_on_task_end_hook_fires():
    tasks = make_task_suite(2, 10, seed=3)
    seen: list = []
    run_sequential(lambda: Learner(seed=3), tasks,
                   on_task_end=lambda lrn, t: seen.append(t.name))
    assert seen == ["task-0", "task-1"]


# -------------------- the head-to-head proof -------------------- #
def test_protected_beats_baseline():
    report = evaluate_continual(seed=7)
    assert report.protected.forgetting < report.baseline.forgetting
    assert report.protected.average_accuracy > report.baseline.average_accuracy
    assert report.forgetting_reduction > 0.0
    assert report.accuracy_gain > 0.0


def test_baseline_actually_forgets():
    # the suite must be discriminating: with no protection, forgetting is real
    report = evaluate_continual(seed=7)
    assert report.baseline.forgetting > 0.1


def test_report_to_dict_is_json_serializable():
    report = evaluate_continual(seed=7)
    blob = json.dumps(report.to_dict())
    parsed = json.loads(blob)
    assert parsed["n_tasks"] == 4
    assert "baseline" in parsed and "protected" in parsed


def test_continual_task_dataclass_defaults():
    t = ContinualTask(name="x")
    assert t.train == [] and t.test == []
    assert isinstance(matrix_metrics([[0.0]]), ContinualMetrics)
