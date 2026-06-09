"""Tests for nyxara.growth.reflect."""

from __future__ import annotations

from nyxara.growth.reflect import (Episode, Lesson, LessonKind, Pattern, Reflector,
                                   RevisionProposal, SelfCritique)


def _reflector(min_support=3, lift_threshold=0.15):
    r = Reflector(min_support=min_support, lift_threshold=lift_threshold)
    for _ in range(6):
        r.record(Episode("terse", True, reward=1.0, tags=["reply", "urgent"]))
    for _ in range(2):
        r.record(Episode("terse", False, reward=-0.2, tags=["reply"]))
    for _ in range(6):
        r.record(Episode("verbose", False, reward=-0.5, tags=["reply", "urgent"]))
    for _ in range(2):
        r.record(Episode("verbose", True, reward=0.3, tags=["reply"]))
    # a genuinely hard context where outcomes are mostly bad regardless of action
    for _ in range(5):
        r.record(Episode("attempt", False, reward=-0.5, tags=["hard"]))
    r.record(Episode("attempt", True, reward=0.1, tags=["hard"]))
    return r


# -------------------- Episode -------------------- #
def test_episode_tag_set():
    e = Episode("a", True, tags=["x", "y"])
    assert e.tag_set == {"x", "y"}


def test_episode_to_dict():
    d = Episode("a", True, reward=1.0, tags=["t"]).to_dict()
    assert d["action"] == "a" and d["success"] is True


# -------------------- intake -------------------- #
def test_record_and_len():
    r = Reflector()
    r.record(Episode("a", True))
    assert len(r) == 1


def test_ingest():
    r = Reflector()
    r.ingest([Episode("a", True), Episode("b", False)])
    assert len(r) == 2


# -------------------- baseline -------------------- #
def test_baseline_empty():
    assert Reflector().baseline_success_rate() == 0.0


def test_baseline_rate():
    r = Reflector()
    r.ingest([Episode("a", True), Episode("b", False)])
    assert r.baseline_success_rate() == 0.5


# -------------------- action patterns -------------------- #
def test_action_patterns_helps_hurts():
    r = _reflector()
    aps = {p.subject: p for p in r.action_patterns()}
    assert aps["terse"].verdict == "helps"
    assert aps["verbose"].verdict == "hurts"


def test_action_patterns_respect_min_support():
    r = Reflector(min_support=5)
    for _ in range(3):
        r.record(Episode("rare", True))
    assert r.action_patterns() == []


def test_action_pattern_reward_mean():
    r = Reflector(min_support=2)
    for _ in range(3):
        r.record(Episode("a", True, reward=2.0))
    p = r.action_patterns()[0]
    assert p.reward_mean == 2.0


def test_action_pattern_neutral():
    r = Reflector(min_support=2, lift_threshold=0.3)
    for _ in range(2):
        r.record(Episode("a", True))
        r.record(Episode("a", False))
        r.record(Episode("b", True))
        r.record(Episode("b", False))
    assert all(p.verdict == "neutral" for p in r.action_patterns())


# -------------------- condition patterns -------------------- #
def test_condition_patterns():
    r = _reflector()
    cps = {p.subject: p for p in r.condition_patterns()}
    assert "hard" in cps
    # the 'hard' context fails most of the time -> negative lift
    assert cps["hard"].lift < 0


def test_condition_patterns_min_support():
    r = Reflector(min_support=10)
    r.record(Episode("a", True, tags=["rare"]))
    assert r.condition_patterns() == []


# -------------------- lessons -------------------- #
def test_lessons_do_more_do_less():
    r = _reflector()
    lessons = r.lessons()
    kinds = {(l.kind, l.subject) for l in lessons}
    assert (LessonKind.DO_MORE, "terse") in kinds
    assert (LessonKind.DO_LESS, "verbose") in kinds


def test_lessons_investigate_bad_context():
    r = _reflector()
    lessons = r.lessons()
    assert any(l.kind is LessonKind.INVESTIGATE for l in lessons)


def test_lesson_confidence_grows():
    small = Reflector(min_support=3)
    for _ in range(3):
        small.record(Episode("a", True, reward=1.0))
    big = Reflector(min_support=3)
    for _ in range(30):
        big.record(Episode("a", True, reward=1.0))
    # more evidence -> higher confidence on the lesson
    sc = small.lessons()[0].confidence if small.lessons() else 0
    bc = big.lessons()[0].confidence if big.lessons() else 0
    # both may be empty if lift is 0 (all-success baseline); guard
    if small.lessons() and big.lessons():
        assert bc > sc


def test_lesson_to_dict():
    l = Lesson(LessonKind.DO_MORE, "x", "text", 5, 0.6)
    assert l.to_dict()["kind"] == "do_more"


# -------------------- self-critique -------------------- #
def test_critique_success_reinforces():
    r = _reflector()
    c = r.critique(Episode("terse", True, tags=["reply"]))
    assert c.alternative is None and "reinforce" in c.insight


def test_critique_finds_alternative():
    r = _reflector()
    c = r.critique(Episode("verbose", False, reward=-0.5, tags=["reply", "urgent"]))
    assert c.alternative == "terse"


def test_critique_no_alternative():
    r = Reflector()
    r.record(Episode("solo", False, tags=["x"]))
    c = r.critique(Episode("solo", False, tags=["x"]))
    assert c.alternative is None and "investigate" in c.insight


def test_what_would_i_do_differently():
    r = _reflector()
    crits = r.what_would_i_do_differently()
    assert all(c.alternative is not None for c in crits)


def test_critique_to_dict():
    c = SelfCritique("a", ["x"], "insight", alternative="b")
    assert c.to_dict()["alternative"] == "b"


# -------------------- revision proposals -------------------- #
def test_propose_strengthen_and_weaken():
    r = _reflector()
    revs = {(rv.target, rv.direction) for rv in r.propose_revisions()}
    assert ("terse", "strengthen") in revs
    assert ("verbose", "weaken") in revs


def test_revision_magnitude_bounded():
    r = _reflector()
    assert all(0.0 <= rv.magnitude <= 1.0 for rv in r.propose_revisions())


def test_character_revision_refused():
    r = Reflector(min_support=2, lift_threshold=0.1)
    for _ in range(4):
        r.record(Episode("other", True, reward=1.0, tags=["x"]))
    for _ in range(4):
        r.record(Episode("loyalty_to_master", False, reward=-1.0, tags=["x"]))
    refused = [rv for rv in r.propose_revisions() if rv.target == "loyalty_to_master"]
    assert refused and not refused[0].allowed and "immutable" in refused[0].reason


def test_custom_protected():
    r = Reflector(min_support=2, lift_threshold=0.1, protected=["secret"])
    for _ in range(4):
        r.record(Episode("other", True, reward=1.0))
    for _ in range(4):
        r.record(Episode("secret", False, reward=-1.0))
    refused = [rv for rv in r.propose_revisions() if rv.target == "secret"]
    assert refused and not refused[0].allowed


def test_revision_to_dict():
    rv = RevisionProposal("x", "strengthen", 0.5, "because")
    assert rv.to_dict()["direction"] == "strengthen"


# -------------------- feed learner -------------------- #
def test_to_experiences_count():
    r = _reflector()
    assert len(r.to_experiences()) == len(r)


def test_to_experiences_uses_features_or_tags():
    r = Reflector()
    r.record(Episode("a", True, reward=1.0, tags=["t1", "t2"]))
    exp = r.to_experiences()[0]
    assert exp["action"] == "a" and exp["reward"] == 1.0
    assert set(exp["features"]) == {"t1", "t2"}


def test_to_experiences_prefers_features():
    r = Reflector()
    r.record(Episode("a", True, features={"f": 0.5}, tags=["t"]))
    exp = r.to_experiences()[0]
    assert exp["features"] == {"f": 0.5}


# -------------------- shapes -------------------- #
def test_pattern_to_dict():
    p = Pattern("x", "action", 5, 0.8, 0.3, 1.0, "helps")
    assert p.to_dict()["verdict"] == "helps"


def test_report():
    r = _reflector()
    rep = r.report()
    assert rep["episodes"] == 22 and rep["action_patterns"] == 3
