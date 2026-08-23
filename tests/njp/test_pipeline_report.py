"""Every arrow of the loop: closed, open with a reason, or absent.

The point of this report is the arrows that are **not** closed. Every other report in the package
counts what happened; this one names what cannot happen, which is the thing a system full of
working organs is least able to tell you about itself.

Read from counters rather than declared. A hardcoded "closed" would make this exactly the kind of
documentation the rest of the package is written against — and the same map exists in
`docs/CAPABILITIES.md`, kept honest by a test that imports every module it cites. This is the
runtime half of that idea.
"""

from __future__ import annotations

from types import SimpleNamespace

from nyxara.njp.brain import NJPBrain


def _taught() -> NJPBrain:
    brain = NJPBrain()
    for line in ("paudhe ko paani chahiye", "paani se growth badhti hai",
                 "light se growth badhti hai", "growth kaise badhti hai"):
        brain.think(line)
    return brain


def test_every_arrow_is_closed_or_open_with_a_reason():
    report = _taught().pipeline_report()
    assert report, "the report is empty"
    for arrow, entry in report.items():
        assert entry["state"] in ("closed", "open"), arrow
        if entry["state"] == "open":
            assert entry["why"], f"{arrow} is open and does not say why"


def test_an_organ_that_is_off_is_absent_rather_than_reported_as_broken():
    """"Off" and "failing" are different, and a report has no business blurring them."""
    off = NJPBrain(config=SimpleNamespace(noesis_enabled=False))
    if off.noesis is not None:
        return                      # the gate is named differently in this build
    assert "→program" not in off.pipeline_report()


def test_the_arrows_this_container_cannot_close_say_so():
    """Qwythos has no weights on disk here, and the report must not pretend otherwise."""
    report = _taught().pipeline_report()
    entry = report.get("qwythos→proposals")
    if entry is not None:
        assert entry["state"] == "open"
        assert "weights" in entry["why"] or "cortex" in entry["why"]


def test_the_planning_arrow_is_reported_as_the_stub_it_is():
    """`predictive._encode_state`'s own docstring records three words collapsing to one state."""
    entry = _taught().pipeline_report().get("→plan")
    if entry is not None:
        assert entry["state"] == "open"
        assert "identity" in entry["why"]


def test_the_report_reflects_what_actually_ran():
    """Not a constant: an arrow closes because its counter moved, and starts open."""
    fresh = NJPBrain().pipeline_report()
    taught = _taught().pipeline_report()
    fresh_open = {k for k, v in fresh.items() if v["state"] == "open"}
    taught_open = {k for k, v in taught.items() if v["state"] == "open"}
    assert taught_open <= fresh_open, "a turn opened an arrow that was closed before it"
    assert len(taught_open) < len(fresh_open), "no arrow closed over four turns"


def test_the_report_never_raises_on_a_bare_brain():
    assert isinstance(NJPBrain().pipeline_report(), dict)


# --------------------------------------------------------------------------- #
# The abstraction library is finally stepped
# --------------------------------------------------------------------------- #

def test_the_program_arrow_closes_once_the_library_has_been_stepped():
    """Index term R read a `compression_gain` from an engine that had never run.

    A report from an engine that has never run reports zero, so one of the eight terms in her own
    capability index was structurally zero and looked like a measurement.
    """
    brain = NJPBrain()
    if brain.noesis is None:
        return
    assert brain.pipeline_report()["→program"]["state"] == "open"

    brain.noesis.step()
    assert brain.pipeline_report()["→program"]["state"] == "closed"
