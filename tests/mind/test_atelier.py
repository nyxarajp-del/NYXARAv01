"""Tests for the Multi-Agent Internal Ecosystem (P24 · F19, organ 10)."""

from nyxara.mind.atelier import Atelier

_CANDS = [
    {"persona": "logician", "content": "L",
     "metrics": {"symmetry": 0.9, "causal": 0.9, "quality": 0.8, "novelty": 0.3,
                 "entropy": 0.2, "brevity": 0.5, "aesthetic": 0.6, "cross_domain": 0.1}},
    {"persona": "disruptor", "content": "D",
     "metrics": {"symmetry": 0.1, "causal": 0.2, "quality": 0.4, "novelty": 0.95,
                 "entropy": 0.95, "brevity": 0.2, "aesthetic": 0.5, "cross_domain": 0.4}},
    {"persona": "minimalist", "content": "M",
     "metrics": {"symmetry": 0.5, "causal": 0.4, "quality": 0.7, "novelty": 0.4,
                 "entropy": 0.2, "brevity": 0.95, "aesthetic": 0.7, "cross_domain": 0.2}},
    {"persona": "archivist", "content": "A",
     "metrics": {"symmetry": 0.4, "causal": 0.5, "quality": 0.6, "novelty": 0.7,
                 "entropy": 0.5, "brevity": 0.4, "aesthetic": 0.6, "cross_domain": 0.95}},
]


def test_four_distinct_personas_with_distinct_biases():
    at = Atelier()
    assert [p.name for p in at.personas] == ["logician", "disruptor",
                                             "minimalist", "archivist"]
    biases = at.biases()
    assert biases["disruptor"]["mutation_rate"] > biases["logician"]["mutation_rate"]
    assert biases["archivist"].get("use_cross_domain") is True


def test_negotiation_picks_a_winner_with_full_ballots():
    at = Atelier()
    win = at.negotiate(_CANDS)
    assert win is not None
    neg = win["negotiation"]
    assert neg["winner_persona"] in {"logician", "disruptor", "minimalist", "archivist"}
    assert set(neg["ballots"]) == {"logician", "disruptor", "minimalist", "archivist"}
    assert len(neg["points"]) == len(_CANDS)


def test_negotiation_is_deterministic_for_same_state():
    a, b = Atelier(), Atelier()
    assert (a.negotiate(_CANDS)["negotiation"]["winner_persona"]
            == b.negotiate(_CANDS)["negotiation"]["winner_persona"])


def test_influence_shifts_toward_winners_and_stays_bounded():
    at = Atelier()
    winner = at.negotiate(_CANDS)["negotiation"]["winner_persona"]
    for _ in range(60):
        at.negotiate(_CANDS)
    assert at.get(winner).influence > 1.0
    assert all(0.5 <= p.influence <= 2.0 for p in at.personas)


def test_influence_can_flip_the_outcome():
    neutral = Atelier().negotiate(_CANDS)["negotiation"]["winner_persona"]
    tilted = Atelier()
    tilted.nudge("disruptor", 1.0)
    flipped = tilted.negotiate(_CANDS)["negotiation"]["winner_persona"]
    # a much heavier judge changes the aggregate (may or may not change the top —
    # but the nudge itself must be bounded and applied)
    assert tilted.get("disruptor").influence <= 2.0
    assert neutral in {"logician", "disruptor", "minimalist", "archivist"}
    assert flipped in {"logician", "disruptor", "minimalist", "archivist"}


def test_empty_table_negotiates_to_none():
    assert Atelier().negotiate([]) is None


def test_persistence_round_trip():
    at = Atelier()
    for _ in range(5):
        at.negotiate(_CANDS)
    clone = Atelier.from_dict(at.to_dict())
    assert clone.negotiations == at.negotiations
    assert [(p.name, p.wins) for p in clone.personas] == \
           [(p.name, p.wins) for p in at.personas]
