"""Sovereign dialectic: suggests sharper paths, but is advisory — never refuses a command."""

from __future__ import annotations

from nyxara.nyx5.dialectic import SovereignDialectic


def test_suggests_on_suboptimal_idea():
    d = SovereignDialectic()
    c = d.critique("use a nested loop to compare all pairs")
    assert c.has_suggestion()
    assert c.proceed is True             # advisory: still proceeds


def test_proceed_always_true():
    d = SovereignDialectic()
    for idea in ("delete the database", "anything at all", "single node deployment"):
        assert d.critique(idea).proceed is True   # never a veto


def test_no_suggestion_still_proceeds():
    d = SovereignDialectic()
    c = d.critique("a clean, already-optimal plan")
    assert c.proceed is True
    assert c.has_suggestion() is False


def test_multiple_suggestions():
    d = SovereignDialectic()
    c = d.critique("nested loop over a single node storing unencrypted plain data")
    assert len(c.suggestions) >= 2
