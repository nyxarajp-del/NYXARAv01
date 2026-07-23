"""Tests for growth/mdl_refactor.py — the MDL-proxy accept/reject rule (Change set E).

A refactor is kept only when it strictly lowers the description-length proxy AND is behaviour-preserving
(the caller's gauntlet). These are fast, standalone tests — no NyxaraCore boot.
"""

from __future__ import annotations

from nyxara.growth.mdl_refactor import MDLRefactor, description_length

# a verbose, redundant module ...
_VERBOSE = '''\
def add(a, b):
    result = a + b
    return result


def add_again(a, b):
    result = a + b
    return result


def add_third_time(a, b):
    result = a + b
    return result
'''

# ... and a leaner, equivalent one
_LEAN = '''\
def add(a, b):
    return a + b
'''


def test_description_length_rewards_leaner_code():
    assert description_length(_LEAN) < description_length(_VERBOSE)


def test_non_parsing_source_has_no_ast_term():
    # total function, never raises on broken source
    assert description_length("def oops(:\n") >= 0.0


def test_accept_keeps_a_leaner_behaviour_preserving_refactor():
    r = MDLRefactor().accept(_VERBOSE, _LEAN, gauntlet_ok=True)
    assert r.accepted is True
    assert r.improved is True
    assert r.delta < 0


def test_reject_when_not_behaviour_preserving():
    r = MDLRefactor().accept(_VERBOSE, _LEAN, gauntlet_ok=False)
    assert r.accepted is False
    assert "behaviour" in r.reason


def test_reject_when_not_leaner():
    # after is bulkier than before → rejected even if the gauntlet passes
    r = MDLRefactor().accept(_LEAN, _VERBOSE, gauntlet_ok=True)
    assert r.accepted is False
    assert r.improved is False


def test_reject_when_candidate_does_not_parse():
    r = MDLRefactor().accept(_VERBOSE, "def broken(:\n", gauntlet_ok=True)
    assert r.accepted is False
    assert "parse" in r.reason


def test_refactor_only_verifies_plausible_candidates():
    calls = []

    def verify(before, after):
        calls.append((before, after))
        return True

    # a leaner candidate → verify is consulted and it is accepted
    r = MDLRefactor().refactor(_VERBOSE, _LEAN, verify)
    assert r.accepted is True
    assert len(calls) == 1

    # a bulkier candidate → verify is never spent
    calls.clear()
    r2 = MDLRefactor().refactor(_LEAN, _VERBOSE, verify)
    assert r2.accepted is False
    assert calls == []
