"""Tests for the Aesthetic Feedback Loop (P24 · F19, organ 5)."""

from nyxara.mind.aesthetic import AestheticJudge

_VERSE = ("Silver river, silver stone —\n"
          "the storm remembers light,\n"
          "the shadow keeps the fire's key,\n"
          "and silence ends the night.")
_FLAT = "the the the the\nthe the the the"
_SYM_ART = ('<svg><polyline points="100,100 200,150 300,100 200,50 100,100" '
            'stroke="hsl(0,70%,50%)"/><polyline points="100,300 200,350 300,300" '
            'stroke="hsl(120,70%,50%)"/><path stroke="hsl(240,70%,50%)" '
            'd="M 150 200 L 250 200"/></svg>')
_LOP_ART = '<svg><polyline points="10,10 11,12 10,14 12,10 11,11" stroke="hsl(7,70%,50%)"/></svg>'


def test_features_stay_in_range():
    judge = AestheticJudge()
    for content, modality in ((_VERSE, "verse"), (_SYM_ART, "art"),
                              ("a mirror engine drives the wheel", "idea")):
        for name, value in judge.features(content, modality).items():
            assert 0.0 <= value <= 1.0, (name, value)


def test_crafted_verse_beats_flat_repetition():
    judge = AestheticJudge()
    assert judge.resonance(_VERSE, "verse") > judge.resonance(_FLAT, "verse")


def test_balanced_art_beats_lopsided():
    judge = AestheticJudge()
    assert judge.resonance(_SYM_ART, "art") > judge.resonance(_LOP_ART, "art")


def test_resonance_is_deterministic():
    judge = AestheticJudge()
    assert judge.resonance(_VERSE, "verse") == judge.resonance(_VERSE, "verse")


def test_learning_moves_weights_toward_signal_and_back():
    judge = AestheticJudge()
    w0 = judge.weights["imagery"]
    for _ in range(5):
        judge.learn(_VERSE, "verse", 1.0)
    w_up = judge.weights["imagery"]
    assert w_up > w0
    for _ in range(5):
        judge.learn(_VERSE, "verse", 0.0)
    assert judge.weights["imagery"] < w_up


def test_weights_stay_bounded():
    judge = AestheticJudge()
    for _ in range(300):
        judge.learn(_VERSE, "verse", 1.0)
    assert all(0.2 <= w <= 3.0 for w in judge.weights.values())


def test_persistence_round_trip():
    judge = AestheticJudge()
    judge.learn(_VERSE, "verse", 0.9)
    clone = AestheticJudge.from_dict(judge.to_dict())
    for name, weight in judge.weights.items():
        assert abs(clone.weights[name] - weight) < 1e-5
    assert clone.ratings_seen == judge.ratings_seen
