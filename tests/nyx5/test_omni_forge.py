"""Omni-forge: default-deny, gate-approved run, dangerous screened, cap, all logged."""

from __future__ import annotations

from nyxara.nyx5.omni_forge import OmniForge


def test_default_deny():
    f = OmniForge(max_forges_per_turn=5)
    f.begin_turn()
    r = f.forge("add", "def tool(a, b): return a + b", args=(2, 3))
    assert r.status == "refused-gate"       # no gate supplied -> default deny


def test_gate_approved_runs_sandboxed():
    f = OmniForge(max_forges_per_turn=5, permission_gate=lambda n, s: True)
    f.begin_turn()
    r = f.forge("add", "def tool(a, b): return a + b", args=(2, 3))
    assert r.status == "ran" and r.result == 5


def test_dangerous_source_refused_by_screen():
    f = OmniForge(max_forges_per_turn=5, permission_gate=lambda n, s: True)
    f.begin_turn()
    r = f.forge("evil", "import os\ndef tool(): os.system('echo x')")
    assert r.status == "refused-screen"     # screened out BEFORE the gate / any execution


def test_sandbox_has_no_imports():
    f = OmniForge(max_forges_per_turn=5, permission_gate=lambda n, s: True)
    f.begin_turn()
    # even if it slipped past screening, the sandbox has no __import__
    r = f.forge("x", "def tool():\n    return __import__")
    assert r.status in ("error", "refused-screen")


def test_forge_cap_and_ledger():
    f = OmniForge(max_forges_per_turn=1, permission_gate=lambda n, s: True)
    f.begin_turn()
    f.forge("a", "def tool(): return 1")
    r2 = f.forge("b", "def tool(): return 2")
    assert r2.status == "refused-gate"      # cap reached
    assert len(f.ledger()) == 2             # every attempt logged
