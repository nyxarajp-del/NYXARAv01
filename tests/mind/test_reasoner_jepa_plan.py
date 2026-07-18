"""The reasoner's imagination step prefers the JEPA latent planner when present."""

from __future__ import annotations

from typing import Any, List

from nyxara.mind.nyxara_reasoner import NyxaraReasoner


class _Plan:
    actions: List[str] = ["clarify", "answer", "verify"]
    expected_reward = 1.5
    confidence = 0.8


class _PlannerModel:
    """A grounded-JEPA double exposing plan()."""

    def __len__(self) -> int:
        return 20

    def encode_state(self, state: Any):
        return (0.0,)

    def plan(self, state: Any, *, horizon: int = 8, **kw: Any) -> _Plan:
        self.planned_from = state
        return _Plan()

    def rollout(self, *a: Any, **kw: Any):  # must NOT be reached when plan works
        raise AssertionError("rollout should not be used when plan() is available")


class _RolloutOnlyModel:
    def __len__(self) -> int:
        return 20

    def rollout(self, start: Any, policy: Any, steps: int = 3):
        return type("_T", (), {"total_reward": 0.5, "mean_confidence": 0.4})()


def test_rollout_eval_prefers_latent_plan():
    model = _PlannerModel()
    r = NyxaraReasoner(llm=None, world_model=model)
    out = r._rollout_eval("please summarize the report")
    assert "latent plan" in out
    assert "clarify,answer,verify" in out
    assert model.planned_from == "please summarize the report"


def test_rollout_eval_falls_back_without_plan():
    r = NyxaraReasoner(llm=None, world_model=_RolloutOnlyModel())
    out = r._rollout_eval("please summarize the report")
    assert "rollout" in out


def test_rollout_eval_honest_without_model():
    r = NyxaraReasoner(llm=None, world_model=None)
    assert "no world model" in r._rollout_eval("anything")
