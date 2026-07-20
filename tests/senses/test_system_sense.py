"""Tests for nyxara.senses.system + nyxara.senses.watch — host interoception.

Hermetic where determinism matters (injected snapshots, tmp dirs); the readers that
touch /proc are additionally exercised for real on Linux CI, asserting only invariants
(fractions in [0, 1], honest omission) — never fabricated values.
"""

from __future__ import annotations

from pathlib import Path

from nyxara.senses.system import DeviceSensor, NetworkSensor, SystemSensor
from nyxara.senses.watch import FileWatcher


# --------------------------------------------------------------------------- #
# SystemSensor — real vitals, honest omission
# --------------------------------------------------------------------------- #
def test_sample_returns_bounded_real_fractions():
    sensor = SystemSensor()
    sensor.sample()                      # primes CPU/IO deltas
    vitals = sensor.sample()
    for key in ("cpu", "mem", "disk", "battery"):
        if key in vitals:
            assert 0.0 <= vitals[key] <= 1.0
    if "cpu_cores" in vitals:
        assert all(0.0 <= c <= 1.0 for c in vitals["cpu_cores"])
    assert "note" in vitals              # missing readings are named, not invented


def test_first_cpu_sample_omits_rate_honestly():
    sensor = SystemSensor()
    first = sensor.sample()
    # a rate needs a delta; the very first sample must not fabricate one
    if "cpu" in first:                   # only possible if /proc/stat was unreadable→retry
        assert sensor._last_cpu is not None
    assert "unavailable" in first["note"] or "cpu" in first or first["note"] == ""


def test_new_processes_primes_then_diffs():
    sensor = SystemSensor()
    assert sensor.new_processes() == []          # first scan primes
    again = sensor.new_processes()               # steady state: mostly nothing new
    assert isinstance(again, list)
    # injected snapshot diff: a brand-new name is reported exactly once
    sensor._seen_procs = {"bash"}
    sensor._procs_primed = True
    sensor._process_names = lambda: {"1": "bash", "2": "hacker-tool"}  # type: ignore
    assert sensor.new_processes() == ["hacker-tool"]
    assert sensor.new_processes() == []          # seen once, never re-announced


def test_kernel_churn_is_filtered():
    sensor = SystemSensor()
    sensor._seen_procs = set()
    sensor._procs_primed = True
    sensor._process_names = lambda: {"1": "kworker/0:1", "2": "myapp"}  # type: ignore
    assert sensor.new_processes() == ["myapp"]


# --------------------------------------------------------------------------- #
# NetworkSensor — transitions, port diff
# --------------------------------------------------------------------------- #
def test_network_first_transition_call_primes():
    net = NetworkSensor(probe=False)
    assert net.transitions() == []


def test_offline_transition_and_port_diff_detected():
    net = NetworkSensor(probe=False)
    states = [{"online": True, "ports": [22, 80]},
              {"online": False, "ports": [22, 80, 8080]}]
    net.sample = lambda: states.pop(0)  # type: ignore
    assert net.transitions() == []                       # primes
    out = net.transitions()
    kinds = [k for k, _, _ in out]
    assert "network_change" in kinds and "port_change" in kinds
    messages = " ".join(m for _, m, _ in out)
    assert "down" in messages and "8080" in messages


def test_steady_network_state_is_quiet():
    net = NetworkSensor(probe=False)
    net.sample = lambda: {"online": True, "ports": [22]}  # type: ignore
    net.transitions()
    assert net.transitions() == []


# --------------------------------------------------------------------------- #
# DeviceSensor — hot-plug diffing
# --------------------------------------------------------------------------- #
def test_device_diff_primes_then_reports_plug_and_unplug():
    dev = DeviceSensor()
    snaps = [{"/dev/video0": "camera (/dev/video0)"},
             {"/dev/video0": "camera (/dev/video0)", "usb1": "USB device 'Mouse'"},
             {"/dev/video0": "camera (/dev/video0)"}]
    dev.snapshot = lambda: dict(snaps.pop(0))  # type: ignore
    assert dev.diff() == ({}, {})                        # primes
    added, removed = dev.diff()
    assert added == {"usb1": "USB device 'Mouse'"} and removed == {}
    added, removed = dev.diff()
    assert added == {} and removed == {"usb1": "USB device 'Mouse'"}


# --------------------------------------------------------------------------- #
# FileWatcher — real files, real diffs, churn filtered
# --------------------------------------------------------------------------- #
def test_watcher_detects_create_modify_delete(tmp_path):
    (tmp_path / "a.txt").write_text("one")
    w = FileWatcher([tmp_path])
    assert w.diff() == ([], [], [])                      # primes
    (tmp_path / "b.txt").write_text("two")
    (tmp_path / "a.txt").write_text("one plus more")
    created, modified, deleted = w.diff()
    assert [Path(p).name for p in created] == ["b.txt"]
    assert [Path(p).name for p in modified] == ["a.txt"]
    (tmp_path / "b.txt").unlink()
    assert [Path(p).name for p in w.diff()[2]] == ["b.txt"]


def test_watcher_ignores_churn_and_hidden(tmp_path):
    w = FileWatcher([tmp_path])
    w.diff()
    (tmp_path / ".secret").write_text("x")
    (tmp_path / "junk.pyc").write_text("x")
    (tmp_path / "__pycache__").mkdir()
    (tmp_path / "__pycache__" / "m.py").write_text("x")
    assert w.diff() == ([], [], [])


def test_watcher_respects_depth_and_ignore_paths(tmp_path):
    deep = tmp_path / "l1" / "l2" / "l3"
    deep.mkdir(parents=True)
    ignored = tmp_path / "data"
    ignored.mkdir()
    w = FileWatcher([tmp_path], depth=2, ignore_paths=[ignored])
    w.diff()
    (deep / "toodeep.txt").write_text("x")               # beyond depth 2
    (ignored / "journal.jsonl").write_text("x")          # ignored subtree
    (tmp_path / "l1" / "seen.txt").write_text("x")       # within depth
    created, _, _ = w.diff()
    assert [Path(p).name for p in created] == ["seen.txt"]


def test_watcher_survives_missing_root(tmp_path):
    w = FileWatcher([tmp_path / "nope"])
    assert w.diff() == ([], [], [])
    assert w.diff() == ([], [], [])
