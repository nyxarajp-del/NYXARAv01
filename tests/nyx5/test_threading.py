"""Anticipatory threading: distinct next-directions grounded in the query subject; count capped."""

from __future__ import annotations

from nyxara.nyx5.threading import AnticipatoryThreads


def test_three_distinct_axes():
    t = AnticipatoryThreads(directions=3)
    dirs = t.anticipate("optimize the spiking network throughput")
    axes = [d.axis for d in dirs]
    assert axes == ["technical", "conceptual", "strategic"]
    assert len({d.line for d in dirs}) == 3      # distinct lines


def test_subject_grounded_in_query():
    t = AnticipatoryThreads(directions=1)
    dirs = t.anticipate("build the reasoner seat")
    assert "reasoner seat" in dirs[0].line


def test_direction_count_capped():
    t = AnticipatoryThreads(directions=99)        # clamped to available axes
    assert len(t.anticipate("x")) == 3


def test_empty_query_safe():
    t = AnticipatoryThreads(directions=2)
    dirs = t.anticipate("")
    assert len(dirs) == 2
