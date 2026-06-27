"""Tests for the real cad_model tool in nyxara.agency.default_tools."""

from __future__ import annotations

import json
import math

import pytest

from nyxara.agency.default_tools import build_default_tools, cad_model
from nyxara.agency.permissions import Authority
from nyxara.agency.tools import ToolRegistry


def test_box_geometry_is_exact():
    m = cad_model(json.dumps({"shape": "box", "size": [50, 50, 50]}))
    assert m["volume"] == 125000.0
    assert m["surface_area"] == 15000.0
    assert m["bounding_box"] == [50, 50, 50]
    assert m["centroid"] == [25, 25, 25]


def test_cylinder_geometry_is_exact():
    m = cad_model(json.dumps({"shape": "cylinder", "radius": 10, "height": 50}))
    assert m["volume"] == pytest.approx(math.pi * 100 * 50)
    assert m["surface_area"] == pytest.approx(2 * math.pi * 100 + 2 * math.pi * 10 * 50)


def test_sphere_geometry_is_exact():
    m = cad_model(json.dumps({"shape": "sphere", "radius": 5}))
    assert m["volume"] == pytest.approx(4.0 / 3.0 * math.pi * 125)


def test_box_with_through_hole_subtracts_volume():
    spec = {"shape": "box", "size": [50, 50, 50],
            "hole": {"shape": "cylinder", "radius": 10, "height": 50}}
    m = cad_model(json.dumps(spec))
    assert m["volume"] == pytest.approx(125000 - math.pi * 100 * 50)
    assert "difference()" in m["scad"]
    assert "cylinder" in m["scad"]


def test_mass_uses_density():
    m = cad_model(json.dumps({"shape": "box", "size": [10, 10, 10], "density": 0.00785}))
    assert m["mass"] == pytest.approx(1000 * 0.00785)


def test_scad_source_is_emitted():
    m = cad_model(json.dumps({"shape": "box", "size": [1, 2, 3], "name": "blk"}))
    assert m["scad"].startswith("// NYXARA cad_model — blk")
    assert "cube([1.0, 2.0, 3.0]" in m["scad"]


def test_unsupported_shape_raises():
    with pytest.raises(ValueError):
        cad_model(json.dumps({"shape": "torus", "radius": 1}))


def test_registered_and_invocable_through_registry():
    reg = build_default_tools(ToolRegistry())
    assert "cad_model" in reg.names()
    result = reg.invoke("cad_model",
                        {"spec": json.dumps({"shape": "box", "size": [2, 2, 2]})},
                        authority=Authority.OWNER)
    assert result.ok
    assert result.value["volume"] == 8.0
