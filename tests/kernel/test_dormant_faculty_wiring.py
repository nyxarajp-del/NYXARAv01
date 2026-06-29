"""Tests that the four formerly-dormant faculties are now genuinely WIRED into the live
cognitive cycle — superposition into hypothesis selection, the dark-data miner into idle
maintenance, and the timeline simulator + butterfly effect into the embodied loop.

These guard against the faculties silently regressing back to "built but never called".
Everything here runs offline (no LLM / no API key) — proving NYXARA does it herself.
"""

from __future__ import annotations

from nyxara.kernel.orchestrator import NyxaraCore, Candidate, ThoughtKind


# --------------------------------------------------------------------------- #
# Quantum · 1 — Superposition drives hypothesis selection
# --------------------------------------------------------------------------- #
def test_superposition_selects_consensus_grounded_hypothesis():
    core = NyxaraCore()
    assert core.superposition_factory is not None
    a = Candidate(text="answer is 4", kind="respond", confidence=0.60)
    b = Candidate(text="answer is 5", kind="respond", confidence=0.55)
    a2 = Candidate(text="answer is 4", kind="respond", confidence=0.62)
    name, cand = core._select_hypothesis([("base", a), ("role_council", b),
                                          ("grounded", a2)])
    # consensus (two agree) + grounding bonus, folded as Bayesian evidence, wins
    assert cand.text == "answer is 4", (name, cand.text)


def test_superposition_records_collapse_to_mindscope():
    core = NyxaraCore()
    a = Candidate(text="left", kind="act", tool="t1", confidence=0.7)
    b = Candidate(text="right", kind="act", tool="t2", confidence=0.4)
    picked = core._collapse_hypotheses([("base", a), ("role_council", b)])
    assert picked is not None
    _name, cand = picked
    assert cand.text == "left"
    inferences = [t for t in core.mind.by_kind(ThoughtKind.INFERENCE)
                  if "superposition collapse" in (t.content or "")]
    assert inferences, "collapse should be auditable in the MindScope"


def test_superposition_single_hypothesis_falls_back_to_floor():
    core = NyxaraCore()
    a = Candidate(text="only", kind="respond", confidence=0.5)
    # one unique signature -> nothing to superpose -> deterministic floor still returns it
    name, cand = core._select_hypothesis([("base", a)])
    assert cand.text == "only" and name == "base"


# --------------------------------------------------------------------------- #
# Void · 1 — Dark-data miner runs on idle over NYXARA's own vitals
# --------------------------------------------------------------------------- #
def test_dark_data_mines_anomaly_and_seeds_curiosity():
    core = NyxaraCore()
    assert core.dark_data_miner is not None
    t = 1000.0
    for i in range(40):
        conf = 0.6
        if i == 20:            # one unusually-unsure turn buried in steady noise
            conf = 0.02
        dt = 1.0 if i != 30 else 90.0   # one long silence between turns
        t += dt
        core._signal_log.append((t, conf, 0.01))
    out = core._mine_dark_data()
    assert out.get("anomalies", 0) >= 1
    assert out.get("gaps", 0) >= 1
    assert "finding" in out
    # a real gap becomes a real question she sets herself
    seeds = [s for s in core._curiosity_seeds if "in my own behaviour" in s]
    assert seeds, "dark-data finding should seed curiosity"


def test_signal_log_grows_with_real_turns():
    core = NyxaraCore()
    before = len(core._signal_log)
    core.process("hello")
    core.process("what is 2 + 2")
    assert len(core._signal_log) >= before + 2
    # each entry is (timestamp, confidence, latency) — all numeric
    ts, conf, lat = core._signal_log[-1]
    assert isinstance(ts, float) and 0.0 <= conf <= 1.0 and lat >= 0.0


# --------------------------------------------------------------------------- #
# Abyss · 1 & 2 — Timeline planner + butterfly drive the embodied loop
# --------------------------------------------------------------------------- #
def test_embodied_agent_honors_planner():
    from nyxara.sim.embodied import EmbodiedAgent
    calls = {"n": 0}

    def planner(state, options):
        calls["n"] += 1
        return options[-1]            # always force the last safe action

    with EmbodiedAgent(seed=3, epsilon=0.0, planner=planner) as agent:
        state = agent.perceive().state
        choice = agent.decide(state)
        assert calls["n"] == 1
        assert choice == (agent.safe_actions() or ["noop"])[-1]


def test_orchestrator_planner_is_wired_and_lazy():
    core = NyxaraCore()
    assert core.embodied_agent is not None
    # the planner reference is injected even though the simulator is built afterwards
    assert core.embodied_agent.planner == core._embodied_planner
    # with no lived experience yet, the planner honestly declines (greedy floor stands)
    out = core._embodied_planner((0.0,) * 8, ["noop", "observe"])
    assert out is None


def test_butterfly_attend_is_safe_without_experience():
    core = NyxaraCore()
    report: dict = {}
    # too little world-model experience -> records nothing, never raises
    core._butterfly_attend((0.0,) * 8, report)
    assert "butterfly" not in report
