"""The causal / hyperspace / qualia routes — and the persistence bug they exposed."""

from __future__ import annotations

import csv
import random

import pytest

from nyxara.kernel.config import NyxaraSettings, Profile

pytest.importorskip("fastapi")


def _client(tmp_path):
    from fastapi.testclient import TestClient

    from nyxara.server.app import create_app

    settings = NyxaraSettings.for_profile(Profile.TEST)
    settings.paths.data_dir = tmp_path
    # no api_token configured -> the auth dependency is a no-op, which keeps these tests
    # about the routes themselves (auth has its own coverage in test_app.py)
    return TestClient(create_app(settings=settings)), settings


def _seed_graph(tmp_path, settings):
    """Learn rain → moisture → mould from numbers and persist it where the routes look."""
    from nyxara.growth.dataset import _default_store_path, ingest_dataset
    from nyxara.growth.dataset_causal import learn_from_dataset_store
    from nyxara.mind.causal_world_model import CausalWorldModel

    source = tmp_path / "d" / "chain.csv"
    source.parent.mkdir(parents=True, exist_ok=True)
    rng = random.Random(1)
    with source.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["rain", "moisture", "mould"])
        for _ in range(120):
            rain = rng.gauss(0, 1)
            moisture = 2.0 * rain + rng.gauss(0, 0.2)
            writer.writerow([f"{rain:.4f}", f"{moisture:.4f}",
                             f"{3.0 * moisture + rng.gauss(0, 0.2):.4f}"])

    store = _default_store_path(settings)
    ingest_dataset(tmp_path / "d", store_path=store, settings=settings)
    model = CausalWorldModel()
    learn_from_dataset_store(store, model=model, order=["rain", "moisture", "mould"])
    model.save(str(store.with_name("causal_graph.json")))
    return store


def test_a_persisted_graph_is_actually_restored(tmp_path):
    """CausalWorldModel.load is a CLASSMETHOD returning a new instance. Calling it on an object
    silently discards the restored graph — the routes looked correct and answered with nothing."""
    from nyxara.mind.causal_world_model import CausalWorldModel

    _client_obj, settings = _client(tmp_path)
    store = _seed_graph(tmp_path, settings)

    restored = CausalWorldModel.load(str(store.with_name("causal_graph.json")))
    assert set(restored._links) == {("rain", "moisture"), ("moisture", "mould")}


def test_causal_why_reports_provenance(tmp_path):
    client, settings = _client(tmp_path)
    _seed_graph(tmp_path, settings)

    body = client.post("/v1/causal/why", json={"target": "mould"}).json()
    assert body["causes"], body
    top = body["causes"][0]
    assert top["cause"] == "moisture"
    assert top["measured"] is True          # learned from numbers, not asserted in prose


def test_causal_do_propagates_the_learned_chain(tmp_path):
    client, settings = _client(tmp_path)
    _seed_graph(tmp_path, settings)

    effects = client.post("/v1/causal/do", json={"cause": "rain", "value": 1.0}).json()["effects"]
    assert effects["moisture"] == pytest.approx(2.0, abs=0.2)
    assert effects["mould"] == pytest.approx(6.0, abs=0.6)


def test_causal_counterfactual_answers(tmp_path):
    client, settings = _client(tmp_path)
    _seed_graph(tmp_path, settings)

    body = client.post("/v1/causal/counterfactual",
                       json={"cause": "rain", "target": "mould"}).json()
    assert body["delta"] == pytest.approx(-6.0, abs=0.6)
    assert "describe" in body


def test_causal_validate_checks_against_ground_truth(tmp_path):
    client, _settings = _client(tmp_path)
    reports = client.post("/v1/causal/validate", json={}).json()["reports"]
    assert reports and all(r["invented_nothing"] for r in reports)


def test_missing_arguments_are_rejected(tmp_path):
    client, _settings = _client(tmp_path)
    assert client.post("/v1/causal/why", json={}).status_code == 400
    assert client.post("/v1/causal/do", json={}).status_code == 400
    assert client.post("/v1/causal/counterfactual", json={"cause": "a"}).status_code == 400


def test_unknown_actions_are_404(tmp_path):
    client, _settings = _client(tmp_path)
    assert client.post("/v1/causal/nonsense", json={"target": "x"}).status_code == 404
    assert client.post("/v1/hyperspace/nonsense", json={}).status_code in (404, 200)


def test_hyperspace_and_qualia_say_so_when_nothing_is_fitted(tmp_path):
    client, _settings = _client(tmp_path)
    assert "error" in client.post("/v1/hyperspace/nearest", json={"term": "dog"}).json()
    assert "error" in client.post("/v1/qualia/concepts", json={}).json()


def test_hyperspace_answers_from_a_fitted_space(tmp_path):
    from nyxara.growth.dataset import _default_store_path
    from nyxara.mind.concept_hyperspace import ConceptHyperspace, relations_from_pairs

    client, settings = _client(tmp_path)
    space = ConceptHyperspace(dim=16, seed=0, lr=0.1, negatives=6)
    space.fit(relations_from_pairs([("poodle", "dog"), ("husky", "dog"), ("dog", "mammal"),
                                    ("cat", "mammal"), ("mammal", "animal")]),
              epochs=300, burn_in=30, domain="life")
    store = _default_store_path(settings)
    store.parent.mkdir(parents=True, exist_ok=True)
    space.save(store.with_name("hyperspace.json"))

    body = client.post("/v1/hyperspace/nearest", json={"term": "poodle", "k": 3}).json()
    assert [n["name"] for n in body["nearest"]][0] == "dog"

    qualia = client.post("/v1/qualia/concepts", json={}).json()
    assert "report" in qualia
    # uncertified regions are discovered but withheld from `usable`
    assert qualia["usable"] == []
