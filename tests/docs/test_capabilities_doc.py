"""P3 — the capability map cannot drift from reality.

``docs/CAPABILITIES.md`` claims a module owns each of the 70 requested capabilities. This test
keeps that map honest: every ``nyxara.*`` module path it cites must actually import, and all 70
capability rows must be present. If a module is renamed/removed or a row is dropped, this fails —
so the documentation can never quietly overstate what exists (honesty applied to the docs).
"""

from __future__ import annotations

import importlib
import re
from pathlib import Path

import pytest

_DOC = Path(__file__).resolve().parents[2] / "docs" / "CAPABILITIES.md"
_MODULE_RE = re.compile(r"`(nyxara\.[a-zA-Z0-9_.]+)`")
_ROW_RE = re.compile(r"^\|\s*(\d+)\s*\|")


def _doc_text() -> str:
    assert _DOC.exists(), f"missing capability map: {_DOC}"
    return _DOC.read_text(encoding="utf-8")


def test_all_seventy_capabilities_present():
    rows = {int(m.group(1)) for line in _doc_text().splitlines()
            if (m := _ROW_RE.match(line))}
    missing = set(range(1, 71)) - rows
    assert not missing, f"capability rows missing from the map: {sorted(missing)}"


@pytest.mark.parametrize("module_path", sorted(set(_MODULE_RE.findall(_doc_text()))))
def test_cited_module_imports(module_path: str):
    # the map may not claim a module that does not exist / does not import
    importlib.import_module(module_path)
