"""Tests for nyxara.sim.envmodel."""

from __future__ import annotations

from nyxara.sim.envmodel import (
    ActionTemplate,
    Entity,
    EnvPrediction,
    EnvironmentModel,
    EnvironmentRegistry,
    build_accounts_model,
    build_default_registry,
    build_network_model,
    build_os_model,
)
from nyxara.sim.sandbox import Sandbox


# -------------------- entities & templates -------------------- #
def test_entity_copy_is_independent():
    e = Entity("f", "file", {"exists": True})
    c = e.copy()
    c.attributes["exists"] = False
    assert e.attributes["exists"] is True


def test_entity_to_dict():
    e = Entity("f", "file", {"exists": True})
    d = e.to_dict()
    assert d["eid"] == "f" and d["etype"] == "file" and d["attributes"]["exists"] is True


def test_template_confidence_grows_with_observations():
    t0 = ActionTemplate("a", observations=0)
    t1 = ActionTemplate("a", observations=3)
    t2 = ActionTemplate("a", observations=30)
    assert t0.confidence == 0.0
    assert 0.0 < t1.confidence < t2.confidence < 1.0


def test_add_and_get_entity():
    m = EnvironmentModel("os", "os")
    m.add_entity("/etc/x", "file", exists=True)
    assert m.get_entity("/etc/x").attributes["exists"] is True
    assert m.get_entity("missing") is None


def test_entities_of_type():
    m = EnvironmentModel("os", "os")
    m.add_entity("a", "file")
    m.add_entity("b", "file")
    m.add_entity("p", "process")
    assert len(m.entities_of_type("file")) == 2
    assert len(m.entities_of_type("process")) == 1


# -------------------- prediction -------------------- #
def test_predict_applies_attr_changes():
    m = build_os_model()
    m.add_entity("/etc/config", "file", exists=True, modified=False)
    p = m.predict("write_file", "/etc/config")
    assert isinstance(p, EnvPrediction)
    assert p.after["modified"] is True
    assert p.known and p.reversible


def test_predict_delete_is_irreversible():
    m = build_os_model()
    m.add_entity("/etc/config", "file", exists=True)
    p = m.predict("delete_file", "/etc/config")
    assert p.after["exists"] is False
    assert p.reversible is False
    assert p.known and p.confidence > 0.5


def test_predict_unknown_action_is_honest():
    m = build_os_model()
    m.add_entity("/etc/config", "file", exists=True)
    p = m.predict("teleport_file", "/etc/config")
    assert not p.known
    assert p.confidence == 0.0
    assert p.reversible is False
    assert p.after == p.before


def test_predict_params_override():
    m = build_os_model()
    m.add_entity("f", "file", exists=True)
    p = m.predict("chmod", "f", perms="0644")
    assert p.after["perms"] == "0644"


def test_predict_does_not_mutate_entity():
    m = build_os_model()
    m.add_entity("/etc/config", "file", exists=True)
    m.predict("delete_file", "/etc/config")
    # the real entity is untouched by a prediction
    assert m.get_entity("/etc/config").attributes["exists"] is True


def test_prediction_to_dict():
    m = build_os_model()
    m.add_entity("f", "file", exists=True)
    d = m.predict("delete_file", "f").to_dict()
    assert d["known"] is True and d["reversible"] is False and "confidence" in d


# -------------------- learning -------------------- #
def test_observe_learns_new_template():
    m = EnvironmentModel("os", "os")
    m.add_entity("/var/log", "file", exists=True, size=100)
    t = m.observe("rotate_log", "/var/log",
                  {"exists": True, "size": 100}, {"exists": True, "size": 0})
    assert "rotate_log" in m.templates
    assert t.attr_changes["size"] == 0
    assert t.observations == 1


def test_observe_raises_confidence_over_repeated_observations():
    m = EnvironmentModel("os", "os")
    m.add_entity("/var/log", "file", exists=True, size=100)
    before = {"exists": True, "size": 100}
    after = {"exists": True, "size": 0}
    for _ in range(6):
        m.observe("rotate_log", "/var/log", before, after)
    p = m.predict("rotate_log", "/var/log")
    assert p.known and p.confidence > 0.5
    assert p.after["size"] == 0


def test_observe_commits_after_state_to_entity():
    m = EnvironmentModel("os", "os")
    m.add_entity("/var/log", "file", exists=True, size=100)
    m.observe("rotate_log", "/var/log",
              {"exists": True, "size": 100}, {"exists": True, "size": 0})
    assert m.get_entity("/var/log").attributes["size"] == 0


def test_observe_reversible_flag_honored():
    m = EnvironmentModel("os", "os")
    m.add_entity("f", "file", exists=True)
    t = m.observe("shred", "f", {"exists": True}, {"exists": False}, reversible=False)
    assert t.reversible is False


def test_coverage_empty_is_zero():
    assert EnvironmentModel("x", "x").coverage() == 0.0


def test_coverage_mean_confidence():
    m = build_os_model()
    assert 0.0 < m.coverage() <= 1.0


# -------------------- registry -------------------- #
def test_registry_register_get_domains():
    reg = EnvironmentRegistry()
    reg.register(build_os_model())
    reg.register(build_network_model())
    assert set(reg.domains()) == {"os", "network"}
    assert reg.get("os").domain == "os"
    assert reg.get("missing") is None


def test_registry_predict():
    reg = build_default_registry()
    reg.get("os").add_entity("/etc/x", "file", exists=True)
    p = reg.predict("os", "delete_file", "/etc/x")
    assert p is not None and p.after["exists"] is False
    assert reg.predict("nope", "x", "y") is None


def test_default_registry_has_three_domains():
    reg = build_default_registry()
    assert set(reg.domains()) == {"os", "network", "accounts"}


# -------------------- prebuilt builders -------------------- #
def test_os_model_actions():
    m = build_os_model()
    assert {"create_file", "write_file", "delete_file", "chmod",
            "kill_process", "start_process"} <= set(m.templates)
    assert m.templates["delete_file"].reversible is False
    assert m.templates["kill_process"].reversible is False


def test_network_model_actions():
    m = build_network_model()
    assert {"open_port", "close_port", "block_ip", "connect"} <= set(m.templates)


def test_accounts_model_actions():
    m = build_accounts_model()
    assert {"disable_account", "enable_account", "grant_role", "revoke_role"} <= set(m.templates)


# -------------------- sandbox seeding -------------------- #
def test_seed_sandbox_populates_state():
    reg = build_default_registry()
    reg.get("os").add_entity("/etc/config", "file", exists=True)
    reg.get("network").add_entity("22", "port", open=True)
    sb = Sandbox()
    env = reg.seed_sandbox(sb)
    assert "/etc/config" in sb.state["environment"]["os"]["entities"]
    assert sb.state["environment"]["network"]["entities"]["22"]["open"] is True
    assert env is sb.state["environment"]


def test_seed_sandbox_is_deep_copy():
    reg = build_default_registry()
    reg.get("os").add_entity("/etc/config", "file", exists=True)
    sb = Sandbox()
    reg.seed_sandbox(sb)
    # mutating the seeded copy must not touch the real model
    sb.state["environment"]["os"]["entities"]["/etc/config"]["exists"] = False
    assert reg.get("os").get_entity("/etc/config").attributes["exists"] is True


def test_prediction_against_unknown_host_does_not_mutate_model():
    reg = build_default_registry()
    net = reg.get("network")
    p = net.predict("block_ip", "10.0.0.5")
    # unknown target -> empty before, but no entity is created as a side-effect
    assert net.get_entity("10.0.0.5") is None
    assert p.known  # the action template exists even if the host doesn't


def test_to_dict_round_shape():
    reg = build_default_registry()
    d = reg.to_dict()
    assert set(d) == {"os", "network", "accounts"}
    assert "coverage" in d["os"]
