"""Tests for nyxara.void.dark_data_mining."""

from __future__ import annotations

import math
import random

import pytest

from nyxara.void.dark_data_mining import (
    AnomalyReport,
    DarkDataMiner,
    GapReport,
    PeriodicityReport,
    SilenceReport,
)


# -------------------- anomaly in noise -------------------- #
def test_anomalies_found_in_noise():
    rng = random.Random(0)
    series = [rng.gauss(0.0, 1.0) for _ in range(200)]
    series[50] = 12.0
    series[120] = -11.0
    rep = DarkDataMiner().mine_anomalies(series)
    assert isinstance(rep, AnomalyReport)
    found = {a.index for a in rep.anomalies}
    assert 50 in found and 120 in found
    assert rep.count <= 5
    hi = next(a for a in rep.anomalies if a.index == 50)
    lo = next(a for a in rep.anomalies if a.index == 120)
    assert hi.direction == "high" and lo.direction == "low"


def test_pure_noise_few_anomalies():
    rng = random.Random(1)
    rep = DarkDataMiner().mine_anomalies([rng.gauss(0.0, 1.0) for _ in range(200)])
    assert rep.count <= 2


def test_anomalies_too_short_series():
    rep = DarkDataMiner().mine_anomalies([1.0, 2.0])
    assert rep.count == 0


def test_anomalies_constant_series_no_false_positive():
    rep = DarkDataMiner().mine_anomalies([5.0] * 50)
    assert rep.count == 0


# -------------------- gaps / silence in time -------------------- #
def test_gap_detected():
    times = [float(i) for i in range(10)] + [t + 9.0 for t in range(10, 20)]
    rep = DarkDataMiner().mine_gaps(times, expected_interval=1.0)
    assert isinstance(rep, GapReport)
    assert rep.count == 1
    assert rep.gaps[0].duration > 5.0
    assert rep.gaps[0].start == 9.0


def test_no_gap_in_regular_series():
    rep = DarkDataMiner().mine_gaps([float(i) for i in range(30)])
    assert rep.count == 0


def test_gaps_too_few_events():
    rep = DarkDataMiner().mine_gaps([0.0, 1.0])
    assert rep.count == 0


# -------------------- silence / absence -------------------- #
def test_missing_categories_detected():
    observed = ["yes", "yes", "no", "yes", "no"]
    expected = ["yes", "no", "maybe", "abstain"]
    rep = DarkDataMiner().mine_silence(observed, expected)
    assert isinstance(rep, SilenceReport)
    missing = {a.item for a in rep.absences}
    assert "maybe" in missing and "abstain" in missing
    assert rep.total_missing == 2


def test_silence_with_count_mappings():
    observed = {"a": 10, "b": 1}
    expected = {"a": 10, "b": 10, "c": 10}
    rep = DarkDataMiner().mine_silence(observed, expected)
    items = {a.item: a for a in rep.absences}
    assert "a" not in items                 # met expectation
    assert items["c"].observed == 0.0
    assert items["b"].deficit_ratio == pytest.approx(0.9)
    # 'c' (fully absent) is louder than 'b' (under-represented)
    assert rep.absences[0].item == "c"


def test_no_absence_when_all_met():
    rep = DarkDataMiner().mine_silence(["a", "b", "c"], ["a", "b"])
    assert rep.count == 0
    assert rep.total_missing == 0


# -------------------- hidden periodicity -------------------- #
def test_hidden_cycle_found():
    rng = random.Random(2)
    signal = [math.sin(2 * math.pi * i / 5.0) + rng.gauss(0.0, 0.3) for i in range(80)]
    rep = DarkDataMiner().mine_periodicity(signal)
    assert isinstance(rep, PeriodicityReport)
    assert rep.has_cycle
    assert rep.dominant_lag == 5


def test_noise_has_no_cycle():
    rng = random.Random(3)
    rep = DarkDataMiner().mine_periodicity([rng.gauss(0.0, 1.0) for _ in range(80)])
    assert not rep.has_cycle
    assert rep.dominant_lag is None


def test_periodicity_too_short():
    rep = DarkDataMiner().mine_periodicity([1.0, 2.0, 3.0])
    assert not rep.has_cycle


# -------------------- serialization -------------------- #
def test_reports_to_dict():
    miner = DarkDataMiner()
    a = miner.mine_anomalies([0.0, 0.0, 0.0, 9.0, 0.0, 0.0]).to_dict()
    assert {"count", "anomalies", "median", "scale"} <= set(a)
    s = miner.mine_silence(["x"], ["x", "y"]).to_dict()
    assert {"count", "total_missing", "absences"} <= set(s)
