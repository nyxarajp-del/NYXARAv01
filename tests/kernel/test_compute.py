"""Tests for nyxara.kernel.compute — honest compute/device introspection."""

from __future__ import annotations

from nyxara.kernel.compute import ComputeReport, compute_report, recommend_device


def test_report_has_expected_shape():
    d = compute_report().to_dict()
    assert d["cpu_count"] >= 1
    assert isinstance(d["gpu"], dict) and "available" in d["gpu"]
    assert d["recommended_device"] in ("cuda", "mps", "cpu")


def test_recommend_device_never_raises():
    assert recommend_device() in ("cuda", "mps", "cpu")


def test_torchless_machine_recommends_cpu():
    rep = compute_report()
    if not rep.torch_available:
        assert rep.recommend_device() == "cpu"
        assert rep.gpu.available is False


def test_report_summary_is_string():
    assert isinstance(compute_report().summary(), str)


def test_construct_default_report():
    rep = ComputeReport()
    assert rep.recommend_device() == "cpu"
