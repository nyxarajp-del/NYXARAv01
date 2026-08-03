"""Tests for nyxara.growth.module_loader — the missing import wire, and its fences."""

from __future__ import annotations

import sys

import pytest

from nyxara.growth.module_loader import forged_root, load_module_from_path, screen_source


@pytest.fixture()
def forged(request):
    """A throwaway module inside the forged root, cleaned up afterwards."""
    root = forged_root()
    root.mkdir(parents=True, exist_ok=True)
    init = root / "__init__.py"
    created_init = not init.exists()
    if created_init:
        init.write_text("", encoding="utf-8")
    path = root / f"_t_{abs(hash(request.node.name)) % 10_000_000}.py"
    yield path
    if path.exists():
        path.unlink()
    sys.modules.pop(f"nyxara_forged_{path.stem}", None)
    if created_init and init.exists():
        init.unlink()


def test_a_file_she_just_wrote_can_be_imported(forged):
    """929 files and not one spec_from_file_location: she could write Python, sandbox it, hot-swap
    a reference and ctypes-load a .so — but never import a module she had just authored."""
    forged.write_text("def apply(x):\n    return x * 2\n", encoding="utf-8")
    result = load_module_from_path(forged)
    assert result.ok, result.reason
    assert result.module.apply(21) == 42
    assert sys.modules.get(result.name) is result.module


def test_forbidden_imports_never_reach_execution(forged):
    forged.write_text("import os\nos.system('echo pwned')\n", encoding="utf-8")
    result = load_module_from_path(forged)
    assert not result.ok
    assert "not allowed" in result.reason


def test_forbidden_builtins_are_refused(forged):
    forged.write_text("x = eval('1 + 1')\n", encoding="utf-8")
    result = load_module_from_path(forged)
    assert not result.ok and "eval" in result.reason


def test_paths_outside_the_forged_root_are_refused(tmp_path):
    outside = tmp_path / "anywhere.py"
    outside.write_text("X = 1\n", encoding="utf-8")
    result = load_module_from_path(outside)
    assert not result.ok
    assert "outside the allowed root" in result.reason


def test_an_explicit_root_is_honoured(tmp_path):
    module = tmp_path / "ok.py"
    module.write_text("VALUE = 7\n", encoding="utf-8")
    result = load_module_from_path(module, allowed_roots=[tmp_path], register=False)
    assert result.ok and result.module.VALUE == 7


def test_a_module_that_fails_to_import_is_data_not_a_crash(forged):
    forged.write_text("raise ValueError('boom')\n", encoding="utf-8")
    result = load_module_from_path(forged)
    assert not result.ok and "import failed" in result.reason
    assert sys.modules.get(f"nyxara_forged_{forged.stem}") is None


def test_a_missing_file_is_reported(tmp_path):
    result = load_module_from_path(tmp_path / "nope.py", allowed_roots=[tmp_path])
    assert not result.ok and result.reason == "no such file"


def test_non_python_files_are_refused(tmp_path):
    blob = tmp_path / "data.txt"
    blob.write_text("hello", encoding="utf-8")
    result = load_module_from_path(blob, allowed_roots=[tmp_path])
    assert not result.ok and "not a Python source file" in result.reason


def test_the_screen_shares_one_policy_with_the_capability_foundry():
    """require_handle is the forged-TOOL contract, not a safety rule — a module legitimately has
    no handle(). Everything security-relevant still comes from the one shared policy."""
    assert screen_source("def apply(x):\n    return x\n") is None
    assert screen_source("import socket\n")
    assert screen_source("")


def test_screening_can_be_skipped_for_already_gauntleted_code(tmp_path):
    module = tmp_path / "trusted.py"
    module.write_text("import os\nNAME = os.sep\n", encoding="utf-8")
    result = load_module_from_path(module, allowed_roots=[tmp_path], screen=False,
                                   register=False)
    assert result.ok
