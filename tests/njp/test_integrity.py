"""An experiment may not quietly edit the instruments.

Twice this package destroyed a working module by writing a new one over it — `tasks.py` in V.58,
`economy.py` in V.92 — and neither was caught by a test, because the tests that would have caught
them belonged to the module that had just been deleted. Both historical paths are pinned here.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from nyxara.njp.integrity import (
    INSTRUMENTS, Disturbed, Fingerprint, Occupied, claim, compare, fingerprint, watch,
)


# --------------------------------------------------------------------------------------------- #
#  the one-second check that was not run
# --------------------------------------------------------------------------------------------- #
def test_it_refuses_the_two_paths_this_package_actually_destroyed():
    for path in ("nyxara/njp/economy.py", "nyxara/njp/tasks.py"):
        with pytest.raises(Occupied, match="already exists"):
            claim(path)


def test_a_free_path_is_handed_over():
    got = claim("nyxara/njp/nothing_answers_to_this.py")
    assert got.name == "nothing_answers_to_this.py"
    assert not got.exists()


def test_replacing_a_file_has_to_be_something_somebody_typed():
    """`rewriting=True` makes the dangerous case deliberate rather than the default."""
    assert claim("nyxara/njp/economy.py", rewriting=True).exists()


def test_the_refusal_says_what_is_in_the_way():
    with pytest.raises(Occupied) as caught:
        claim("nyxara/njp/worth.py")
    assert "bytes" in str(caught.value) and "pick a name" in str(caught.value)


# --------------------------------------------------------------------------------------------- #
#  fingerprints
# --------------------------------------------------------------------------------------------- #
def test_a_fingerprint_covers_the_instruments(tmp_path):
    assert len(fingerprint(INSTRUMENTS).files) > 50
    assert all(f.endswith(".py") for f in fingerprint(INSTRUMENTS).files)


def test_an_edit_shows_and_an_addition_shows_differently(tmp_path):
    (tmp_path / "pkg").mkdir()
    one = tmp_path / "pkg" / "a.py"
    one.write_text("x = 1\n")
    before = fingerprint(("pkg",), at=tmp_path)
    one.write_text("x = 2\n")
    (tmp_path / "pkg" / "b.py").write_text("y = 1\n")
    got = compare(before, fingerprint(("pkg",), at=tmp_path))
    assert got.edited == ["pkg/a.py"] and got.added == ["pkg/b.py"] and got.removed == []
    assert got.any


def test_a_removal_is_a_change_and_an_addition_is_not_undeclared(tmp_path):
    """A new file harms nothing. An edited or removed one is an instrument that moved."""
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "a.py").write_text("x = 1\n")
    before = fingerprint(("pkg",), at=tmp_path)
    (tmp_path / "pkg" / "a.py").unlink()
    (tmp_path / "pkg" / "new.py").write_text("z = 1\n")
    got = compare(before, fingerprint(("pkg",), at=tmp_path))
    assert got.removed == ["pkg/a.py"]
    assert got.undeclared([]) == ["pkg/a.py"], "the addition is not a complaint"
    assert got.undeclared(["pkg/a.py"]) == []


def test_nothing_moving_is_nothing_to_report(tmp_path):
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "a.py").write_text("x = 1\n")
    one = fingerprint(("pkg",), at=tmp_path)
    assert not compare(one, fingerprint(("pkg",), at=tmp_path)).any


# --------------------------------------------------------------------------------------------- #
#  watching an experiment
# --------------------------------------------------------------------------------------------- #
def test_an_undeclared_edit_fails_the_experiment(tmp_path):
    """However calm its own numbers looked."""
    (tmp_path / "pkg").mkdir()
    victim = tmp_path / "pkg" / "instrument.py"
    victim.write_text("MEASURE = 1\n")
    with pytest.raises(Disturbed, match="never declared"):
        with watch(("pkg",), at=tmp_path):
            victim.write_text("MEASURE = 999\n")


def test_a_declared_edit_is_allowed(tmp_path):
    """Change is not forbidden. **Undeclared** change is."""
    (tmp_path / "pkg").mkdir()
    mine = tmp_path / "pkg" / "mine.py"
    mine.write_text("v = 1\n")
    with watch(("pkg",), touching=["pkg/mine.py"], at=tmp_path):
        mine.write_text("v = 2\n")


def test_writing_a_brand_new_file_is_never_a_complaint(tmp_path):
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "old.py").write_text("v = 1\n")
    with watch(("pkg",), at=tmp_path):
        (tmp_path / "pkg" / "fresh.py").write_text("v = 1\n")


def test_the_failure_names_the_file_that_moved(tmp_path):
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "a.py").write_text("v = 1\n")
    with pytest.raises(Disturbed) as caught:
        with watch(("pkg",), at=tmp_path):
            (tmp_path / "pkg" / "a.py").write_text("v = 2\n")
    assert "pkg/a.py" in str(caught.value)
    assert "instruments that moved while it ran" in str(caught.value)


def test_an_empty_fingerprint_claims_nothing():
    assert Fingerprint().files == []
    assert not compare(Fingerprint(), Fingerprint()).any
    assert fingerprint(("no/such/place",)).files == []
