"""Tests for nyxara.cognition.concept_formation — the abstraction ladder.

The motivating example, end to end: ``Car, Bike, Truck → Vehicle → Transport``.
"""

from __future__ import annotations

from nyxara.cognition.concept_formation import AbstractionLadder, Concept, Instance


def _vehicles() -> AbstractionLadder:
    L = AbstractionLadder()
    L.observe("Car",   ["has_wheels", "has_engine", "carries_people", "moves", "road"])
    L.observe("Bike",  ["has_wheels", "carries_people", "moves", "road"])
    L.observe("Truck", ["has_wheels", "has_engine", "carries_people", "carries_cargo",
                        "moves", "road"])
    return L


# -------------------- abstraction = shared structure -------------------- #
def test_abstract_keeps_shared_features_only():
    L = _vehicles()
    vehicle = L.abstract(["Car", "Bike", "Truck"], name="Vehicle")
    assert isinstance(vehicle, Concept) and vehicle.name == "Vehicle"
    # intersection of the three: engine/cargo drop out, the rest is kept
    assert vehicle.features == {"has_wheels", "carries_people", "moves", "road"}
    assert vehicle.level == 1 and vehicle.support == 3
    assert set(vehicle.members) == {"Car", "Bike", "Truck"}


def test_abstract_reparents_members():
    L = _vehicles()
    L.abstract(["Car", "Bike", "Truck"], name="Vehicle")
    assert L.concept_of("Car") == "Vehicle"
    assert L.concept_of("Bike") == "Vehicle"
    assert L.concept_of("Vehicle") is None  # top of the ladder so far


def test_abstract_needs_two_members():
    L = _vehicles()
    assert L.abstract(["Car"]) is None
    assert L.abstract([]) is None


# -------------------- ascend = climb a level (autonomously) -------------------- #
def test_ascend_builds_the_next_level():
    L = _vehicles()
    L.abstract(["Car", "Bike", "Truck"], name="Vehicle")
    L.observe("Plane", ["has_wings", "has_engine", "carries_people", "moves", "sky"])
    L.observe("Ship",  ["has_hull", "carries_people", "carries_cargo", "moves", "water"])

    formed = L.ascend()
    assert len(formed) == 1
    transport = formed[0]
    assert transport.features == {"carries_people", "moves"}
    assert transport.level == 2 and transport.support == 5
    assert set(transport.members) == {"Vehicle", "Plane", "Ship"}


def test_ascend_with_nothing_to_cluster_returns_empty():
    L = AbstractionLadder()
    L.observe("Lonely", ["unique_feature"])
    assert L.ascend() == []


# -------------------- the Car → Vehicle → Transport chain -------------------- #
def test_path_reads_off_the_full_chain():
    L = _vehicles()
    L.abstract(["Car", "Bike", "Truck"], name="Vehicle")
    L.observe("Plane", ["has_wings", "has_engine", "carries_people", "moves", "sky"])
    L.observe("Ship",  ["has_hull", "carries_people", "carries_cargo", "moves", "water"])
    transport = L.ascend()[0]

    assert L.path("Car") == ["Car", "Vehicle", transport.name]
    assert L.ancestors("Truck") == ["Vehicle", transport.name]
    assert L.is_a("Car", "Vehicle") and L.is_a("Car", transport.name)
    assert not L.is_a("Car", "Ship")


# -------------------- classify an unseen thing by subsumption -------------------- #
def test_classify_unseen_picks_most_specific_concept():
    L = _vehicles()
    L.abstract(["Car", "Bike", "Truck"], name="Vehicle")
    L.observe("Plane", ["has_wings", "has_engine", "carries_people", "moves", "sky"])
    L.observe("Ship",  ["has_hull", "carries_people", "carries_cargo", "moves", "water"])
    transport = L.ascend()[0]

    # a never-seen Scooter still lands under Vehicle (the most specific match), not Transport
    hit = L.classify(["has_wheels", "has_engine", "carries_people", "moves", "road"])
    assert hit is not None and hit[0] == "Vehicle"

    # something that only shares the broad features lands on the meta-concept
    far = L.classify(["carries_people", "moves", "rails"])
    assert far is not None and far[0] == transport.name


def test_classify_returns_none_without_concepts():
    L = AbstractionLadder()
    L.observe("Car", ["has_wheels"])
    assert L.classify(["has_wheels"]) is None


# -------------------- auto-naming when no name is given -------------------- #
def test_auto_name_from_shared_features():
    L = AbstractionLadder()
    L.observe("Sparrow", ["has_wings", "lays_eggs", "feathers", "flies"])
    L.observe("Eagle",   ["has_wings", "lays_eggs", "feathers", "flies", "predator"])
    concept = L.abstract(["Sparrow", "Eagle"])
    assert concept is not None and concept.name.startswith("≈")
    assert {"has_wings", "feathers", "flies", "lays_eggs"} <= concept.features


# -------------------- relational features fold through anti-unification -------------------- #
def test_relational_features_generalize():
    L = AbstractionLadder()
    L.observe("a", ["owns(jp, car)", "mortal"])
    L.observe("b", ["owns(jp, bike)", "mortal"])
    concept = L.abstract(["a", "b"])
    assert "mortal" in concept.features
    # owns(jp, car) ⊓ owns(jp, bike) → owns(jp, ?X): the shared subject is preserved
    assert any(f.startswith("owns(jp,") and "?" in f for f in concept.features)


# -------------------- introspection -------------------- #
def test_to_dict_and_report_shape():
    L = _vehicles()
    L.abstract(["Car", "Bike", "Truck"], name="Vehicle")
    d = L.to_dict()
    assert d["instances"] == 3
    assert any(c["name"] == "Vehicle" for c in d["concepts"])
    assert "Vehicle" in d["roots"]
    report = L.report()
    assert "Vehicle" in report and "Car" in report


def test_observe_merges_features():
    L = AbstractionLadder()
    L.observe("Car", ["has_wheels"])
    inst = L.observe("Car", ["has_engine"])
    assert isinstance(inst, Instance)
    assert inst.features == {"has_wheels", "has_engine"}
