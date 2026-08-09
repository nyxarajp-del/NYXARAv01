"""NYX V.02 · what happened while you were away — compression, not accumulation.

`aura` takes events in one at a time and `car` thinks between prompts; neither ever turns a
thousand arrivals into something a person could read on returning. These tests pin the four
scales, the fact that a digest keeps what was *new* / *recurring* / *changed* rather than the
stream itself, and the honest limit — "never sleeping" is true only while the process runs.
"""

from __future__ import annotations

import time

from nyxara.kernel.config import NyxConfig
from nyxara.nyx.brain import NyxBrain
from nyxara.nyx.stream import SCALES, Digest, PerpetualStream


class _Scrammed:
    scrammed = True


def _stream(**kw) -> PerpetualStream:
    kw.setdefault("every_s", 0.0)
    return PerpetualStream(**kw)


# --------------------------------------------------------------------------- #
# Absorbing
# --------------------------------------------------------------------------- #
def test_a_moment_is_taken_in():
    stream = _stream()
    assert stream.absorb("event", concepts=["rain"], detail="it rained")
    assert stream.absorbed == 1


def test_a_moment_with_nothing_in_it_is_refused():
    assert not _stream().absorb("event")


def test_a_turn_is_absorbed_as_its_concepts():
    stream = _stream()
    assert stream.absorb_turn("the engine turns the flywheel")
    assert stream.stats()["concepts_seen"] >= 2


def test_an_empty_turn_is_refused():
    assert not _stream().absorb_turn("   ")


def test_the_buffer_is_bounded():
    stream = _stream(capacity=32)
    for index in range(200):
        stream.absorb("event", concepts=[f"c{index}"])
    assert stream.stats()["held"] <= 32
    assert stream.absorbed == 200            # what arrived is still counted honestly


# --------------------------------------------------------------------------- #
# Compression at four scales
# --------------------------------------------------------------------------- #
def test_every_scale_gets_its_own_digest():
    stream = _stream()
    stream.absorb("event", concepts=["rain"])
    digests = stream.tick(force=True)
    assert set(digests) == {name for name, _window in SCALES}


def test_a_digest_keeps_what_was_new():
    stream = _stream()
    stream.absorb("turn", concepts=["flywheel"])
    assert "flywheel" in stream.tick(force=True)["seconds"].novel


def test_a_digest_keeps_what_repeated():
    stream = _stream()
    for _ in range(3):
        stream.absorb("turn", concepts=["flywheel"])
    recurring = dict(stream.tick(force=True)["seconds"].recurring)
    assert recurring.get("flywheel") == 3


def test_something_seen_once_is_not_called_recurring():
    stream = _stream()
    stream.absorb("turn", concepts=["flywheel"])
    assert not stream.tick(force=True)["seconds"].recurring


def test_a_digest_keeps_what_changed():
    stream = _stream()
    stream.absorb("change", concepts=["hebbian_rate"], detail="tuned graph.hebbian_rate")
    assert stream.tick(force=True)["seconds"].changes


def test_a_scale_that_saw_nothing_says_so():
    """An empty summary would read like a quiet week. Saying nothing arrived is the honest form."""
    digest = _stream().tick(force=True)["seconds"]
    assert digest.empty and "nothing arrived" in digest.render()


def test_a_moment_older_than_the_window_falls_out_of_the_short_scale():
    stream = _stream()
    stream.absorb("turn", concepts=["old"])
    stream._moments[-1].at = time.time() - 300.0   # noqa: SLF001 — reaching in to age it
    digests = stream.tick(force=True)
    assert digests["seconds"].empty and not digests["hours"].empty


def test_a_digest_does_not_store_the_stream_it_compressed():
    stream = _stream()
    for index in range(50):
        stream.absorb("turn", concepts=[f"c{index}"], detail="x" * 200)
    assert "moments" not in stream.to_dict()


# --------------------------------------------------------------------------- #
# The cadence, and /scram
# --------------------------------------------------------------------------- #
def test_a_tick_before_its_time_does_nothing():
    stream = _stream(every_s=3600.0)
    stream.absorb("turn", concepts=["rain"])
    stream.tick(force=True)
    assert stream.tick() == {}
    assert stream.ticks == 1


def test_due_is_true_once_the_interval_has_passed():
    stream = _stream(every_s=0.0)
    assert stream.due()


def test_scram_stops_the_beat_dead():
    stream = _stream()
    stream.absorb("turn", concepts=["rain"])
    assert stream.tick(oversight=_Scrammed(), force=True) == {}
    assert stream.ticks == 0


def test_an_unreadable_oversight_is_treated_as_stop():
    class _Broken:
        @property
        def scrammed(self):
            raise RuntimeError("no")

    assert _stream().tick(oversight=_Broken(), force=True) == {}


# --------------------------------------------------------------------------- #
# The briefing
# --------------------------------------------------------------------------- #
def test_the_briefing_reports_what_arrived():
    stream = _stream()
    for _ in range(2):
        stream.absorb_turn("rain makes the road wet")
    stream.tick(force=True)
    briefing = stream.briefing()
    assert "While you were away" in briefing and "rain" in briefing


def test_the_briefing_with_nothing_says_it_does_not_go_looking():
    briefing = _stream().briefing()
    assert "Nothing arrived" in briefing
    assert "do not go" in briefing


def test_the_briefing_compresses_on_demand_when_no_beat_has_run():
    stream = _stream()
    stream.absorb_turn("the engine turns the flywheel")
    assert "While you were away" in stream.briefing()


def test_describe_before_any_beat_says_so():
    assert "no digest yet" in _stream().describe()


# --------------------------------------------------------------------------- #
# Honesty
# --------------------------------------------------------------------------- #
def test_stats_bound_the_never_sleeping_claim_to_the_process():
    note = _stream().stats()["note"].lower()
    assert "only while the process is running" in note
    assert "cli session that exits does not" in note


def test_stats_say_it_rides_the_existing_heartbeat():
    note = _stream().stats()["note"].lower()
    assert "no second thread" in note and "never reaches out" in note


def test_the_tick_count_keeps_the_claim_checkable():
    stream = _stream()
    stream.tick(force=True)
    stream.tick(force=True)
    assert stream.stats()["ticks"] == 2


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #
def test_digests_and_counters_survive_a_round_trip():
    stream = _stream()
    stream.absorb_turn("the engine turns the flywheel")
    stream.tick(force=True)
    fresh = _stream()
    assert fresh.load_dict(stream.to_dict())
    assert fresh.absorbed == stream.absorbed
    assert set(fresh.digests) == set(stream.digests)


def test_a_corrupt_sidecar_never_blocks_boot():
    stream = _stream()
    assert not stream.load_dict([1, 2])
    assert stream.load_dict({"digests": ["junk", {}]})
    assert not stream.digests


def test_a_digest_serialises():
    assert Digest(scale="seconds", moments=1).to_dict()["scale"] == "seconds"


# --------------------------------------------------------------------------- #
# Wired into the brain
# --------------------------------------------------------------------------- #
def test_the_brain_exposes_it():
    brain = NyxBrain(NyxConfig())
    assert brain.stream is not None and "stream" in brain.stats()


def test_the_brain_can_be_told_to_do_without_it():
    brain = NyxBrain(NyxConfig(stream_enabled=False))
    assert brain.stream is None and brain.catch_up() == ""


def test_a_turn_is_absorbed_by_the_brain():
    brain = NyxBrain(NyxConfig())
    before = brain.stream.absorbed
    brain.think("the engine turns the flywheel")
    assert brain.stream.absorbed == before + 1


def test_the_beat_compresses_without_a_second_thread():
    brain = NyxBrain(NyxConfig(stream_every_s=0.0))
    brain.think("the engine turns the flywheel")
    brain.tick()
    assert brain.stream.ticks >= 1


def test_her_own_omega_change_shows_up_as_a_change():
    brain = NyxBrain(NyxConfig(stream_every_s=0.0))
    brain.digest(force=True)
    assert isinstance(brain.catch_up(), str)
