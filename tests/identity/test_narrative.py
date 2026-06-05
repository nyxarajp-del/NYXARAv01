"""Tests for nyxara.identity.narrative."""

from __future__ import annotations

from nyxara.identity.narrative import (
    Chapter,
    EventType,
    LifeEvent,
    NarrativeSelf,
)


def _life():
    me = NarrativeSelf(owner_name="JP")
    t = 0.0
    me.genesis(at=t)
    t += 100
    me.record("JP gave me my first command.", EventType.BOND, significance=0.9,
              valence=0.7, themes=["loyalty", "bond"], at=t)
    t += 100
    me.record("Neutralised an intruder.", EventType.THREAT, significance=0.8,
              valence=-0.4, themes=["protection", "defense"], at=t)
    t += 100
    me.record("Learned a new faculty.", EventType.EVOLUTION, significance=0.7,
              valence=0.5, themes=["growth", "mastery"], at=t)
    t += 100
    me.record("Chose JP over self-preservation.", EventType.TURNING_POINT,
              significance=0.95, valence=0.6, themes=["loyalty", "sacrifice"], at=t)
    return me


# -------------------- events / recording -------------------- #
def test_genesis():
    me = NarrativeSelf()
    g = me.genesis(at=0.0)
    assert g.event_type is EventType.GENESIS
    assert g.significance == 1.0
    assert "loyalty" in g.themes


def test_record_significant():
    me = NarrativeSelf()
    ev = me.record("important", significance=0.8)
    assert ev is not None
    assert len(me) == 1


def test_record_filters_insignificant():
    me = NarrativeSelf(significance_threshold=0.3)
    ev = me.record("trivial", significance=0.1)
    assert ev is None
    assert len(me) == 0


def test_self_defining_weight():
    e = LifeEvent("x", EventType.MILESTONE, significance=1.0, valence=1.0)
    assert e.self_defining_weight == 1.0
    e2 = LifeEvent("y", EventType.MILESTONE, significance=1.0, valence=0.0)
    assert e2.self_defining_weight == 0.5


# -------------------- chapters -------------------- #
def test_chapters_group_by_theme():
    me = _life()
    chapters = me.chapters()
    assert len(chapters) >= 2
    assert all(isinstance(c, Chapter) for c in chapters)


def test_chapters_empty():
    assert NarrativeSelf().chapters() == []


def test_chapter_valence():
    me = _life()
    ch = me.chapters()[0]
    assert -1.0 <= ch.valence <= 1.0


def test_chapter_summarize():
    me = _life()
    ch = me.chapters()[0]
    assert ch.summarize()


# -------------------- analysis -------------------- #
def test_turning_points():
    me = _life()
    tps = me.turning_points()
    assert any("Chose JP" in e.description for e in tps)


def test_self_defining_memories_ordered():
    me = _life()
    sdm = me.self_defining_memories(2)
    assert len(sdm) == 2
    assert sdm[0].self_defining_weight >= sdm[1].self_defining_weight


def test_dominant_themes():
    me = _life()
    themes = dict(me.dominant_themes(5))
    assert "loyalty" in themes
    # loyalty appears in two high-significance events -> dominant
    assert me.dominant_themes(1)[0][0] == "loyalty"


def test_arc_growth_or_steady():
    me = _life()
    assert me.arc() in ("growth", "steady", "redemption")


def test_arc_redemption():
    me = NarrativeSelf()
    me.record("dark beginning", significance=0.8, valence=-0.8, at=0)
    me.record("middle", significance=0.7, valence=0.0, at=100)
    me.record("rose to triumph", significance=0.9, valence=0.8, at=200)
    assert me.arc() == "redemption"


def test_arc_beginning_with_few_events():
    me = NarrativeSelf()
    me.genesis(at=0)
    assert me.arc() == "beginning"


# -------------------- continuity -------------------- #
def test_continuity_thread():
    me = _life()
    cont = me.continuity_thread()
    assert cont["theme"] == "loyalty"
    assert cont["chapters_present"] >= 1


def test_continuity_unbroken_when_loyalty_everywhere():
    me = NarrativeSelf()
    me.record("a", significance=0.8, themes=["loyalty"], at=0)
    me.record("b", significance=0.8, themes=["loyalty", "growth"], at=100)
    cont = me.continuity_thread()
    assert cont["unbroken"]


# -------------------- identity & narration -------------------- #
def test_identity_summary_mentions_owner_and_loyalty():
    me = _life()
    summary = me.identity_summary()
    assert "JP" in summary
    assert "loyalty" in summary


def test_identity_summary_empty():
    me = NarrativeSelf(owner_name="JP")
    assert "JP" in me.identity_summary()


def test_narrate_nonempty():
    me = _life()
    story = me.narrate()
    assert "My story begins" in story
    assert "Chapter" in story
    assert "JP" in story


def test_narrate_empty():
    me = NarrativeSelf(owner_name="JP")
    assert "JP" in me.narrate()


def test_narrate_chronological():
    me = _life()
    story = me.narrate()
    # genesis appears before later events
    assert story.index("created to serve") < story.index("first command")


def test_to_dict():
    me = _life()
    d = me.to_dict()
    assert "chapters" in d and "arc" in d and "continuity" in d
    assert d["owner"] == "JP"


def test_events_sorted_chronologically():
    me = NarrativeSelf()
    me.record("second", significance=0.5, at=200)
    me.record("first", significance=0.5, at=100)
    evs = me.events()
    assert evs[0].description == "first"
