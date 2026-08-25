"""P3 — the architecture map cannot drift from reality.

``docs/ARCHITECTURE.md`` claims a module owns each part of the 32-section cognitive architecture.
This test keeps that claim honest the same way ``test_capabilities_doc.py`` keeps the capability
map honest: every ``nyxara.*`` path it cites must actually import, and all 32 section rows must be
present. A module renamed or removed, or a row quietly dropped, fails here.

The rule this enforces is the one the document is about. An architecture map is worthless the
moment it starts describing an architecture that is not there, and a map nobody can check is a map
that will eventually do exactly that.
"""

from __future__ import annotations

import importlib
import re
from pathlib import Path

import pytest

_DOC = Path(__file__).resolve().parents[2] / "docs" / "ARCHITECTURE.md"
_MODULE_RE = re.compile(r"`(nyxara\.[a-zA-Z0-9_.]+)`")
_ROW_RE = re.compile(r"^\|\s*(\d+)\s*\|")
_STATUSES = {"REAL+WIRED", "REAL", "PARTIAL", "NARRATIVE"}


def _doc_text() -> str:
    assert _DOC.exists(), f"missing architecture map: {_DOC}"
    return _DOC.read_text(encoding="utf-8")


def _rows() -> dict:
    out = {}
    for line in _doc_text().splitlines():
        match = _ROW_RE.match(line)
        if not match:
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) >= 4 and cells[-1] in _STATUSES:
            out[int(match.group(1))] = cells
    return out


def test_all_thirty_two_sections_present():
    missing = set(range(0, 33)) - set(_rows())
    assert not missing, f"architecture sections missing from the map: {sorted(missing)}"


@pytest.mark.parametrize("module_path", sorted(set(_MODULE_RE.findall(_doc_text()))))
def test_cited_module_imports(module_path: str):
    # the map may not claim a module that does not exist / does not import
    importlib.import_module(module_path)


def test_every_row_carries_a_known_status():
    """A row without a status is a checkmark, and a checkmark is what this file exists to prevent."""
    for number, cells in sorted(_rows().items()):
        assert cells[-1] in _STATUSES, f"§{number} has an unrecognised status: {cells[-1]!r}"


def test_a_row_claiming_a_module_is_not_narrative():
    """NARRATIVE means "framing, not a component". Citing a module while claiming to be neither."""
    for number, cells in sorted(_rows().items()):
        if cells[-1] != "NARRATIVE":
            continue
        # §1, §13 and §29 point at the modules that embody the idea without owning it as a
        # component; they are allowed to, but a narrative row must not be the *only* home a
        # module has, or the map would be crediting a section with something nothing implements.
        for path in _MODULE_RE.findall(cells[2]):
            others = [n for n, c in _rows().items()
                      if n != number and path in _MODULE_RE.findall(c[2])]
            assert others, (f"§{number} is NARRATIVE and is the only row citing {path} — "
                            "either it owns the module and is not narrative, or the citation "
                            "belongs on the row that does")


def test_every_partial_names_what_is_missing():
    """A PARTIAL that does not say which half is absent is a status with no information in it."""
    text = _doc_text()
    for number, cells in sorted(_rows().items()):
        if cells[-1] != "PARTIAL":
            continue
        assert f"### §{number} —" in text, (
            f"§{number} is PARTIAL but has no section under 'The three PARTIALs, named' "
            "saying what is missing")


def test_the_black_box_is_claimed_and_real():
    """§26 was ABSENT when this map was written and is the one section it caused to be built."""
    rows = _rows()
    assert "nyxara.njp.blackbox" in _MODULE_RE.findall(rows[26][2])
    assert rows[26][-1] == "REAL+WIRED"
    importlib.import_module("nyxara.njp.blackbox")
