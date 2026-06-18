"""Tests for the intrinsic-drive wiring — motivation now SELECTS autonomous work.

NYXARA's :class:`~nyxara.identity.motivation.MotivationSystem` used to compute rich
intrinsic scores (novelty, info-gain, competence, owner-relevance) that nothing ever
read. It is now built into the kernel and drives which queued idle task she pursues
first: serving the Master outranks pure curiosity (Rule 1), and novelty habituates so
she does not fixate on one theme. These tests assert the behaviour, not the dashboard.
"""

from __future__ import annotations

from nyxara.kernel.orchestrator import NyxaraCore


def _core(**kw):
    return NyxaraCore(**kw)


def test_motivation_is_built_into_the_kernel():
    nyx = _core()
    # identity is on by default -> the drive system exists and is affect-modulated
    assert nyx.motivation is not None
    assert nyx.motivation.affect is nyx.affect


def test_owner_relevant_work_is_chosen_first():
    nyx = _core()
    queue = ["re-read cached notes", "study a novel subsystem",
             "how to better protect the Master"]
    picked = nyx._drain_motivated(queue)
    assert "master" in picked.lower()           # Rule 1: service outranks curiosity
    assert picked not in queue                   # and it was removed from the queue


def test_novelty_habituates_so_she_does_not_fixate():
    nyx = _core()
    # visit one theme repeatedly; a fresh, unseen theme should then win the next pick
    for _ in range(3):
        nyx._drain_motivated(["study quantum computing"])  # single item -> always picked, visited
    picked = nyx._drain_motivated(["study quantum computing", "explore a brand new topic"])
    assert picked == "explore a brand new topic"


def test_drain_is_lossless_and_falls_back_without_motivation():
    nyx = _core()
    q = ["only one item"]
    assert nyx._drain_motivated(q) == "only one item"
    assert q == []
    assert nyx._drain_motivated([]) is None
    # with the drive system absent, selection degrades to FIFO without losing work
    nyx.motivation = None
    q2 = ["first", "second"]
    assert nyx._drain_motivated(q2) == "first"
