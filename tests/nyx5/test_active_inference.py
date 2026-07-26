"""Active inference: surprise high for novel, low for repeated; suggest() gated; delegates if given."""

from __future__ import annotations

from nyxara.nyx5.active_inference import SurpriseMeter


def test_repeated_input_becomes_unsurprising():
    m = SurpriseMeter(n_neurons=32, alpha=0.5, gate=0.6)
    first = m.appraise([1, 2, 3])
    for _ in range(10):
        last = m.appraise([1, 2, 3])
    assert first.surprise > last.surprise      # the model learned to predict it
    assert last.surprise < 0.6


def test_novel_input_is_surprising():
    m = SurpriseMeter(n_neurons=32, alpha=0.3, gate=0.5)
    for _ in range(10):
        m.appraise([1, 2, 3])                  # settle on a routine
    novel = m.appraise([20, 21, 22, 23])       # never-seen neurons
    assert novel.surprise >= 0.5
    assert novel.preemptive is True


def test_suggest_gated_by_surprise():
    m = SurpriseMeter(n_neurons=32, alpha=0.3, gate=0.5)
    for _ in range(10):
        m.appraise([1, 2, 3])
    assert m.suggest() is None                 # unsurprising -> no suggestion
    m.appraise([25, 26, 27, 28])
    s = m.suggest()
    assert s is not None and s.gain > 0.0


def test_delegates_to_free_energy_engine():
    class FakeEngine:
        def learning_progress(self):
            return 0.99

    m = SurpriseMeter(n_neurons=16, alpha=0.3, gate=0.4, free_energy=FakeEngine())
    m.appraise([10, 11, 12])                    # surprising on a fresh model
    s = m.suggest()
    assert s is not None and s.gain >= 0.99     # engine refined the gain upward


def test_entropy_in_unit_range():
    m = SurpriseMeter(n_neurons=16)
    r = m.appraise([1, 2, 3, 4])
    assert 0.0 <= r.entropy <= 1.0
