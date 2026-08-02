"""Epistemic mirroring: casual -> shallow/sharp, deep -> high depth; voice always preserved."""

from __future__ import annotations

from nyxara.nyx5.mirroring import EpistemicMirror


def test_casual_prompt_is_shallow_and_sharp():
    m = EpistemicMirror()
    p = m.read("hey bro quick fix lol")
    assert p.depth <= 3
    assert p.style == "sharp"
    assert p.voice_preserved is True


def test_deep_prompt_expands_depth():
    m = EpistemicMirror()
    p = m.read("Design a distributed recursive architecture, prove its topology is optimal, "
               "and discuss the philosophical implications for machine consciousness.")
    assert p.depth >= 6
    assert p.style == "deep"
    assert p.voice_preserved is True


def test_depth_within_bounds():
    m = EpistemicMirror(max_depth=10)
    for prompt in ("", "a", "x " * 500):
        p = m.read(prompt)
        assert 1 <= p.depth <= 10
