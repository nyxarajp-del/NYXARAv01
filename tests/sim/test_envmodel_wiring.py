"""The embodied agent now learns a structured environment model from its lived transitions.

sim/envmodel.py was a complete, tested module that nothing in the running loop used. The
embodied agent now refines an EnvironmentModel of its workspace on every real before→after
step, so its symbolic action templates (and their confidence) actually grow with experience.
"""

from __future__ import annotations

from nyxara.sim.embodied import EmbodiedAgent
from nyxara.sim.envmodel import EnvironmentModel


def test_agent_builds_a_workspace_env_model():
    with EmbodiedAgent(seed=2) as agent:
        assert isinstance(agent.env_model, EnvironmentModel)
        assert agent.env_model.get_entity("workspace") is not None


def test_env_model_learns_templates_from_lived_steps():
    with EmbodiedAgent(seed=2) as agent:
        assert agent.env_coverage() == 0.0           # nothing observed yet
        agent.run(steps=12)
        assert len(agent.env_model.templates) > 0     # actions were observed as templates
        assert agent.env_coverage() > 0.0             # confidence grew with observations


def test_status_reports_env_coverage():
    with EmbodiedAgent(seed=5) as agent:
        agent.run(steps=8)
        st = agent.status()
        assert "env_model_coverage" in st
        assert "env_actions_learned" in st
        assert st["env_actions_learned"] == len(agent.env_model.templates)


def test_injected_env_model_is_used():
    shared = EnvironmentModel("shared", "workspace")
    with EmbodiedAgent(seed=1, env_model=shared) as agent:
        agent.run(steps=6)
    assert agent.env_model is shared
    assert len(shared.templates) > 0
