"""Tests for nyxara.growth.dataset_causal — a table becomes a causal graph, or honestly abstains."""

from __future__ import annotations

import random

import pytest

from nyxara.growth.dataset_causal import (learn_causal_graph, learn_from_dataset_store,
                                          simulate_intervention)
from nyxara.mind.causal_world_model import CausalWorldModel


def _chain(n: int = 200, seed: int = 0):
    """Ground truth: x → m → y. y has NO direct dependence on x once m is known."""
    rng = random.Random(seed)
    rows = []
    for _ in range(n):
        x = rng.gauss(0, 1)
        m = 2.0 * x + rng.gauss(0, 0.2)
        y = 3.0 * m + rng.gauss(0, 0.2)
        rows.append({"x": x, "m": m, "y": y})
    return ["x", "m", "y"], rows


def _confounded(n: int = 200, seed: int = 1):
    """Ground truth: z → a and z → b. a does NOT cause b — they only correlate."""
    rng = random.Random(seed)
    rows = []
    for _ in range(n):
        z = rng.gauss(0, 1)
        rows.append({"z": z, "a": 2.0 * z + rng.gauss(0, 0.3), "b": 3.0 * z + rng.gauss(0, 0.3)})
    return ["z", "a", "b"], rows


# -------------------- structure recovery -------------------- #
def test_declared_order_recovers_the_chain_with_correct_coefficients():
    columns, rows = _chain()
    report = learn_causal_graph(columns, rows, order=["x", "m", "y"])

    assert report.orientation == "declared"
    edges = {(c, e): w for c, e, w in report.edges}
    assert ("x", "m") in edges and ("m", "y") in edges
    assert edges[("x", "m")] == pytest.approx(2.0, abs=0.1)
    assert edges[("m", "y")] == pytest.approx(3.0, abs=0.1)


def test_the_fully_mediated_direct_edge_is_not_claimed():
    """The whole point of fitting every parent at once: with m in the model, x has no residual
    effect on y. A single-predictor fit would report ~6.0 here and be confidently wrong."""
    columns, rows = _chain()
    report = learn_causal_graph(columns, rows, order=["x", "m", "y"])
    assert ("x", "y") not in {(c, e) for c, e, _w in report.edges}


def test_a_confounded_pair_is_not_called_causal():
    """z drives both a and b. Once z is controlled for there is no a→b effect, and none is
    claimed — this is the correlation-is-not-causation case, decided by the arithmetic."""
    columns, rows = _confounded()
    report = learn_causal_graph(columns, rows, order=["z", "a", "b"])
    pairs = {(c, e) for c, e, _w in report.edges}
    assert ("z", "a") in pairs and ("z", "b") in pairs
    assert ("a", "b") not in pairs


def test_notears_recovers_the_chain_without_any_declared_order():
    """A static table has no arrow of time, so direction must come from the data itself."""
    pytest.importorskip("numpy")
    columns, rows = _chain()
    report = learn_causal_graph(columns, rows)
    assert report.orientation == "notears"
    pairs = {(c, e) for c, e, _w in report.edges}
    assert ("x", "m") in pairs and ("m", "y") in pairs


# -------------------- abstention -------------------- #
def test_without_order_or_numpy_she_abstains_rather_than_guessing(monkeypatch):
    """Guessing a direction and calling it causal is worse than saying nothing."""
    import nyxara.growth.dataset_causal as mod

    monkeypatch.setattr(mod, "_notears_edges", lambda *a, **k: [])
    columns, rows = _chain()
    report = learn_causal_graph(columns, rows)

    assert report.orientation == "unknown"
    assert report.edges == []
    assert "direction unknown" in report.summary()


def test_too_few_rows_claims_nothing():
    columns, rows = _chain(n=4)
    report = learn_causal_graph(columns, rows, order=["x", "m", "y"])
    assert report.edges == [] and report.rows == 4
    assert any("usable row" in n for n in report.notes)


def test_a_single_column_relates_to_nothing():
    report = learn_causal_graph(["only"], [{"only": float(i)} for i in range(50)])
    assert report.edges == []
    assert any("two numeric columns" in n for n in report.notes)


def test_rows_missing_a_column_are_dropped_not_guessed():
    columns, rows = _chain(n=40)
    rows[0].pop("m")
    report = learn_causal_graph(columns, rows, order=["x", "m", "y"])
    assert report.rows == 39


# -------------------- the graph answers questions -------------------- #
def test_registered_edges_make_do_work_over_the_dataset():
    columns, rows = _chain()
    model = CausalWorldModel()
    report = learn_causal_graph(columns, rows, model=model, order=["x", "m", "y"])
    assert report.registered == len(report.edges)

    effects = simulate_intervention(model, {"x": 1.0})
    assert effects["m"] == pytest.approx(2.0, abs=0.2)
    assert effects["y"] == pytest.approx(6.0, abs=0.6)


def test_registered_links_carry_their_provenance():
    """An audit must be able to tell a table-learned edge from a stream-measured one."""
    columns, rows = _chain()
    model = CausalWorldModel()
    learn_causal_graph(columns, rows, model=model, order=["x", "m", "y"])

    link = model._links[("x", "m")]
    assert link.evidence["source"] == "dataset_table"
    assert link.evidence["orientation"] == "declared"
    assert link.evidence["n_rows"] == 200
    assert 0.0 < link.confidence <= 1.0


def test_register_false_leaves_the_model_untouched():
    columns, rows = _chain()
    model = CausalWorldModel()
    report = learn_causal_graph(columns, rows, model=model, order=["x", "m", "y"], register=False)
    assert report.edges and report.registered == 0
    assert not model._links


def test_undeclared_columns_are_placed_last_not_dropped():
    columns, rows = _chain()
    report = learn_causal_graph(columns, rows, order=["x", "m"])
    assert "y" in report.columns
    assert any("not in --order" in n for n in report.notes)


def test_learn_from_dataset_store_reads_ingested_rows(tmp_path):
    import csv

    from nyxara.growth.dataset import ingest_dataset

    columns, rows = _chain(n=60)
    path = tmp_path / "d" / "chain.csv"
    path.parent.mkdir(parents=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        for row in rows:
            writer.writerow([f"{row[c]:.4f}" for c in columns])

    store = tmp_path / "dataset.jsonl"
    ingest_dataset(tmp_path / "d", store_path=store)
    report = learn_from_dataset_store(store, order=["x", "m", "y"])
    assert {(c, e) for c, e, _w in report.edges} == {("x", "m"), ("m", "y")}
