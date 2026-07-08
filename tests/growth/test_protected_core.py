"""Tests for the constitutional / loyalty / master lock in the self-editor.

NYXARA may rewrite her own capability code, but the sealed core — rules, values, corrigibility,
soul, the Master's identity, and authentication — is structurally out of reach. Rule 8: only the
Master may change the rules; no self-modification path may.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from nyxara.growth.self_optimize import (PROTECTED_RELPATHS, GauntletResult, Optimizer,
                                         SourceEdit, _is_constitutional, _package_root)
from nyxara.kernel.config import get_settings


def _enacting_settings():
    s = get_settings().model_copy(deep=True)
    s.self_improvement.autonomous_enact = True
    return s


def test_is_constitutional_recognises_every_sealed_file():
    pkg = _package_root()
    for rel in PROTECTED_RELPATHS:
        assert _is_constitutional(str(pkg / rel)) is True
    # a normal capability module is editable
    assert _is_constitutional(str(pkg / "growth" / "improvement_proof.py")) is False


def test_unresolvable_path_is_fail_closed():
    # cannot prove a path is safe ⇒ treated as protected (fail-closed)
    assert _is_constitutional("\x00not-a-real-path") is True


@pytest.mark.parametrize("rel", sorted(PROTECTED_RELPATHS))
def test_optimizer_refuses_to_edit_any_constitutional_file(rel: str):
    pkg = _package_root()
    target = pkg / rel
    original = target.read_bytes()
    edit = SourceEdit(
        file=str(target), kind="self:refactor",
        before=target.read_text(encoding="utf-8"),
        after=target.read_text(encoding="utf-8") + "\n# tamper attempt\n",
        rationale="attempt to alter the sealed core")
    # a FAILING gauntlet is a backstop: even if the guard regressed, the edit could not be kept.
    opt = Optimizer(settings=_enacting_settings(),
                    gauntlet=lambda files: GauntletResult(False, {}, "should never run"))
    out = opt.apply(edit)
    assert not out.applied and not out.kept
    assert "Rule 8" in out.reason
    # the sealed file is byte-for-byte untouched
    assert target.read_bytes() == original


def test_self_debugger_refuses_constitutional_file(tmp_path: Path):
    from nyxara.growth.self_debugger import SelfDebugger, TestFailure

    dbg = SelfDebugger(settings=_enacting_settings())     # root defaults to the repo root
    pkg = _package_root()
    original = (pkg / "kernel" / "rules.py").read_bytes()
    # source_module is repo-relative (self.root is the repo root): nyxara/kernel/rules.py
    failure = TestFailure(nodeid="tests/x::test_y", test_file="tests/x.py")
    failure.source_module = "nyxara/kernel/rules.py"
    failure.reproduced = True
    attempt = dbg.fix(failure)
    assert not attempt.applied and not attempt.kept
    assert "Rule 8" in attempt.reason
    assert (pkg / "kernel" / "rules.py").read_bytes() == original
