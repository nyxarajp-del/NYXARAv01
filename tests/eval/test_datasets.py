"""Tests for the real-world held-out validation loader (eval/datasets.py).

The loader is deterministic and offline. We assert that the *bundled* corpus is well-formed and
gradeable (an oracle aces it), that it is genuinely **held out** (its ids never overlap the training
battery, so it cannot leak into weakness-detection), that an external JSONL override loads, and that
malformed/missing datasets fail *honestly* rather than yielding a silent empty battery."""

from __future__ import annotations

import json

import pytest

from nyxara.eval.benchmark import Benchmark, Grader, build_default_benchmark
from nyxara.eval.datasets import (DatasetError, REALWORLD_HOLDOUT, build_realworld_benchmark,
                                  load_jsonl_benchmark, realworld_holdout_path)
from nyxara.eval.hard_benchmark import build_hard_benchmark


def _oracle(bench: Benchmark):
    def solve(prompt: str) -> str:
        for t in bench.tasks():
            if t.prompt == prompt:
                if t.grader == Grader.MULTIPLE_CHOICE:
                    return f"The answer is {t.answer}."
                return str(t.answer)
        return ""
    return solve


# --------------------------------------------------------------------------- #
# The bundled real-world corpus
# --------------------------------------------------------------------------- #
def test_bundled_realworld_loads_and_is_nonempty():
    bench = build_realworld_benchmark()
    assert len(bench) >= 40
    assert set(bench.categories()) >= {"math", "factual", "reading", "logic", "units"}


def test_bundled_realworld_ids_are_unique_and_namespaced():
    bench = build_realworld_benchmark()
    ids = [t.id for t in bench.tasks()]
    assert len(ids) == len(set(ids)), "duplicate task ids in the bundled corpus"
    assert all(i.startswith("rw-") for i in ids), "real-world ids must be rw- namespaced"


def test_oracle_aces_the_bundled_realworld_set():
    # Every gold answer must be gradeable by its declared grader — otherwise the corpus is unusable.
    bench = build_realworld_benchmark()
    report = bench.run(_oracle(bench))
    assert report.accuracy == 1.0, report.summary()


def test_realworld_is_held_out_from_the_training_battery():
    # The single most important property: the validation set shares NO task id with anything the
    # self-improvement loop trains / detects weaknesses against, so it can never leak.
    rw_ids = {t.id for t in build_realworld_benchmark().tasks()}
    train_ids = {t.id for t in build_default_benchmark().tasks()}
    adversarial_ids = {t.id for t in build_hard_benchmark().tasks()}
    assert rw_ids.isdisjoint(train_ids)
    assert rw_ids.isdisjoint(adversarial_ids)


# --------------------------------------------------------------------------- #
# load_jsonl_benchmark — paths, overrides, and honest failure
# --------------------------------------------------------------------------- #
def test_load_by_bundled_name_matches_builder():
    by_name = load_jsonl_benchmark(REALWORLD_HOLDOUT)
    assert len(by_name) == len(build_realworld_benchmark())


def test_load_external_jsonl_path(tmp_path):
    rows = [
        {"id": "ext-0", "category": "math", "grader": "numeric_final",
         "answer": "4", "prompt": "What is 2 + 2? Give just the number."},
        {"id": "ext-1", "category": "factual", "grader": "contains",
         "answer": "paris", "prompt": "Capital of France?"},
    ]
    p = tmp_path / "mini.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    bench = load_jsonl_benchmark(str(p))
    assert len(bench) == 2
    assert bench.run(_oracle(bench)).accuracy == 1.0


def test_external_override_via_settings(tmp_path):
    p = tmp_path / "ov.jsonl"
    p.write_text(json.dumps({"id": "ov-0", "category": "math", "grader": "numeric_final",
                             "answer": "1", "prompt": "1?"}), encoding="utf-8")

    class _Cfg:
        validation_realworld_path = str(p)

    class _Settings:
        self_improvement = _Cfg()

    assert realworld_holdout_path(_Settings()) == str(p)
    bench = build_realworld_benchmark(_Settings())
    assert [t.id for t in bench.tasks()] == ["ov-0"]


def test_env_override(tmp_path, monkeypatch):
    p = tmp_path / "env.jsonl"
    p.write_text(json.dumps({"id": "env-0", "category": "x", "grader": "exact",
                             "answer": "y", "prompt": "q"}), encoding="utf-8")
    monkeypatch.setenv("NYXARA_EVAL_HOLDOUT_PATH", str(p))
    bench = build_realworld_benchmark()
    assert [t.id for t in bench.tasks()] == ["env-0"]


# --------------------------------------------------------------------------- #
# Honest failure
# --------------------------------------------------------------------------- #
def test_missing_dataset_raises_clearly():
    with pytest.raises(DatasetError):
        load_jsonl_benchmark("does_not_exist_anywhere.jsonl")


def test_malformed_json_raises(tmp_path):
    p = tmp_path / "bad.jsonl"
    p.write_text("{not valid json}", encoding="utf-8")
    with pytest.raises(DatasetError):
        load_jsonl_benchmark(str(p))


def test_missing_required_field_raises(tmp_path):
    p = tmp_path / "incomplete.jsonl"
    p.write_text(json.dumps({"id": "x", "prompt": "q"}), encoding="utf-8")  # no answer
    with pytest.raises(DatasetError):
        load_jsonl_benchmark(str(p))


def test_unknown_grader_raises(tmp_path):
    p = tmp_path / "badgrader.jsonl"
    p.write_text(json.dumps({"id": "x", "prompt": "q", "answer": "a", "grader": "nope"}),
                 encoding="utf-8")
    with pytest.raises(Exception):
        load_jsonl_benchmark(str(p))


def test_empty_dataset_raises(tmp_path):
    p = tmp_path / "empty.jsonl"
    p.write_text("\n# only a comment\n", encoding="utf-8")
    with pytest.raises(DatasetError):
        load_jsonl_benchmark(str(p))
