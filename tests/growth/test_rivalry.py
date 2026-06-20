"""Tests for nyxara.growth.rivalry — peer-AGI out-compete (Pillar F · Edge 4)."""

from __future__ import annotations

from nyxara.eval.benchmark import Benchmark, BenchmarkTask
from nyxara.growth.rivalry import Arena, HeadToHead, Rival


def _bench() -> Benchmark:
    return Benchmark("demo", [
        BenchmarkTask(id="m1", category="math", prompt="2+2?", answer="4", grader="exact"),
        BenchmarkTask(id="m2", category="math", prompt="3+5?", answer="8", grader="exact"),
        BenchmarkTask(id="l1", category="lang", prompt="opposite of hot?", answer="cold",
                      grader="contains"),
        BenchmarkTask(id="l2", category="lang", prompt="opposite of up?", answer="down",
                      grader="contains"),
    ])


def _me(p: str) -> str:            # strong at math, weak at language
    return {"2+2?": "4", "3+5?": "8"}.get(p, "?")


def _rival(p: str) -> str:         # strong at language, weak at math
    return {"opposite of hot?": "cold", "opposite of up?": "down"}.get(p, "?")


# --------------------------------------------------------------------------- #
# head-to-head
# --------------------------------------------------------------------------- #
def test_head_to_head_scores_both_per_category():
    h2h = Arena(benchmark=_bench()).head_to_head(_me, _rival, rival_name="GPT-X")
    assert h2h.per_category["math"]["self"] == 1.0
    assert h2h.per_category["math"]["rival"] == 0.0
    assert h2h.per_category["lang"]["self"] == 0.0
    assert h2h.per_category["lang"]["rival"] == 1.0
    assert abs(h2h.delta) < 1e-9            # 50% each -> tie overall

def test_gaps_lists_only_where_rival_leads():
    h2h = Arena(benchmark=_bench()).head_to_head(_me, _rival, rival_name="GPT-X")
    cats = [g["category"] for g in h2h.gaps()]
    assert cats == ["lang"]                 # rival only leads language
    assert h2h.gaps()[0]["gap"] == 1.0

def test_winning_flag():
    # rival that answers nothing -> NYXARA strictly ahead
    h2h = Arena(benchmark=_bench()).head_to_head(_me, lambda p: "", rival_name="weak")
    assert h2h.winning and h2h.delta > 0


# --------------------------------------------------------------------------- #
# rival modeling (Theory of Mind)
# --------------------------------------------------------------------------- #
def test_rival_is_modeled_with_strategy():
    arena = Arena(benchmark=_bench())
    arena.head_to_head(_me, _rival, rival_name="GPT-X")
    rival = arena.rival("GPT-X")
    assert isinstance(rival, Rival)
    assert "lang" in rival.strengths()
    assert "math" in rival.weaknesses()
    assert "lang" in rival.strategy

def test_tom_records_rival_competence():
    arena = Arena(benchmark=_bench())
    arena.head_to_head(_me, _rival, rival_name="GPT-X")
    tom = arena.tom()
    assert "GPT-X" in tom.agents()
    assert tom.belief("GPT-X", "competence:lang") == 1.0


# --------------------------------------------------------------------------- #
# routing the gaps into growth
# --------------------------------------------------------------------------- #
def test_gaps_become_ranked_weaknesses():
    arena = Arena(benchmark=_bench())
    h2h = arena.head_to_head(_me, _rival, rival_name="GPT-X")
    rep = arena.gaps_to_weaknesses(h2h)
    assert rep.to_dict()["n_weaknesses"] == 1
    top = rep.ranked()[0]
    assert "lang" in top.title and top.severity == 1.0

def test_gaps_become_curiosity_topics():
    arena = Arena(benchmark=_bench())
    h2h = arena.head_to_head(_me, _rival, rival_name="GPT-X")
    assert arena.gaps_to_topics(h2h) == ["hard problems in lang"]

def test_out_compete_returns_full_pass():
    out = Arena(benchmark=_bench()).out_compete(_me, _rival, rival_name="GPT-X")
    assert set(out) == {"head_to_head", "rival", "weaknesses", "topics"}
    assert out["rival"]["name"] == "GPT-X"
    assert out["topics"] == ["hard problems in lang"]


# --------------------------------------------------------------------------- #
# robustness
# --------------------------------------------------------------------------- #
def test_solver_errors_do_not_crash_the_arena():
    def boom(p: str) -> str:
        raise RuntimeError("rival crashed")
    h2h = Arena(benchmark=_bench()).head_to_head(_me, boom, rival_name="flaky")
    assert h2h.rival_accuracy == 0.0 and h2h.winning   # a crash scores 0, never propagates
