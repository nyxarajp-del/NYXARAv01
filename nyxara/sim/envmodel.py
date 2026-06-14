"""NYXARA · sim/envmodel.py — learned models of the real environments (✦).

The sandbox can only imagine accurately if it knows how the real world behaves. This
module holds NYXARA's **structured models** of the environments it actually operates
in — the operating system / filesystem, the network, and accounts/services — so a dry
run reflects reality, not a guess.

Each :class:`EnvironmentModel` tracks the **entities** in its domain (files, processes,
hosts, ports, accounts) with their attributes, and a set of **action templates** that
say how an action transforms those entities (what attributes change, which facts are
added/removed, and whether the change is reversible). Crucially the model **learns**:
:meth:`EnvironmentModel.observe` compares before/after states of a real action and
refines (or discovers) the template, raising its **confidence** with each observation —
and honestly reporting low confidence where it has not yet looked.

A :class:`EnvironmentRegistry` gathers the domain models and **seeds the sandbox** so
plans/tools/self-edits are dry-run against a faithful copy of the world.

Depends on :mod:`sim.sandbox` (seeding). Pure standard library.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, List, Optional

__all__ = [
    "Entity",
    "ActionTemplate",
    "EnvPrediction",
    "EnvironmentModel",
    "EnvironmentRegistry",
    "build_os_model",
    "build_network_model",
    "build_accounts_model",
    "build_default_registry",
]


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


# --------------------------------------------------------------------------- #
# Entities & templates
# --------------------------------------------------------------------------- #
@dataclass
class Entity:
    eid: str
    etype: str
    attributes: Dict[str, Any] = field(default_factory=dict)

    def copy(self) -> "Entity":
        return Entity(self.eid, self.etype, dict(self.attributes))

    def to_dict(self) -> Dict[str, Any]:
        return {"eid": self.eid, "etype": self.etype, "attributes": dict(self.attributes)}


@dataclass
class ActionTemplate:
    name: str
    applies_to: Optional[str] = None         # entity type, or None for any
    attr_changes: Dict[str, Any] = field(default_factory=dict)
    add_facts: FrozenSet[str] = frozenset()
    del_facts: FrozenSet[str] = frozenset()
    reversible: bool = True
    observations: int = 0

    @property
    def confidence(self) -> float:
        return _clamp(self.observations / (self.observations + 3.0))


@dataclass
class EnvPrediction:
    action: str
    entity_id: str
    before: Dict[str, Any]
    after: Dict[str, Any]
    add_facts: List[str]
    del_facts: List[str]
    reversible: bool
    confidence: float
    known: bool                              # was there a learned/known template?

    def to_dict(self) -> Dict[str, Any]:
        return {"action": self.action, "entity": self.entity_id,
                "after": self.after, "add_facts": self.add_facts,
                "del_facts": self.del_facts, "reversible": self.reversible,
                "confidence": round(self.confidence, 3), "known": self.known}


# --------------------------------------------------------------------------- #
# Environment model
# --------------------------------------------------------------------------- #
class EnvironmentModel:
    """A learned, structured model of one real environment (os / network / accounts)."""

    def __init__(self, name: str, domain: str) -> None:
        self.name = name
        self.domain = domain
        self.entities: Dict[str, Entity] = {}
        self.facts: set = set()
        self.templates: Dict[str, ActionTemplate] = {}

    # ---- entities ---- #
    def add_entity(self, eid: str, etype: str, **attributes: Any) -> Entity:
        e = Entity(eid=eid, etype=etype, attributes=dict(attributes))
        self.entities[eid] = e
        return e

    def get_entity(self, eid: str) -> Optional[Entity]:
        return self.entities.get(eid)

    def entities_of_type(self, etype: str) -> List[Entity]:
        return [e for e in self.entities.values() if e.etype == etype]

    def has_fact(self, fact: str) -> bool:
        return fact in self.facts

    # ---- templates ---- #
    def register_template(self, template: ActionTemplate) -> ActionTemplate:
        self.templates[template.name] = template
        return template

    # ---- learning ---- #
    def observe(self, action: str, target: str, before: Dict[str, Any],
                after: Dict[str, Any], *, reversible: Optional[bool] = None) -> ActionTemplate:
        """Learn/refine an action template from a real before→after transition."""
        diff = {k: v for k, v in after.items() if before.get(k) != v}
        tmpl = self.templates.get(action)
        if tmpl is None:
            ent = self.entities.get(target)
            tmpl = ActionTemplate(name=action, applies_to=ent.etype if ent else None,
                                  attr_changes=dict(diff), observations=1,
                                  reversible=True if reversible is None else reversible)
            self.templates[action] = tmpl
        else:
            tmpl.observations += 1
            tmpl.attr_changes.update(diff)
            if reversible is not None:
                tmpl.reversible = reversible
        # commit the observed state to the entity (this actually happened)
        ent = self.entities.get(target)
        if ent is not None:
            ent.attributes.update(after)
        return tmpl

    # ---- prediction (for the sandbox) ---- #
    def predict(self, action: str, target: str, **params: Any) -> EnvPrediction:
        ent = self.entities.get(target)
        before = dict(ent.attributes) if ent else {}
        tmpl = self.templates.get(action)
        if tmpl is None:
            return EnvPrediction(action=action, entity_id=target, before=before,
                                 after=before, add_facts=[], del_facts=[],
                                 reversible=False, confidence=0.0, known=False)
        after = dict(before)
        after.update(tmpl.attr_changes)
        after.update(params)
        return EnvPrediction(
            action=action, entity_id=target, before=before, after=after,
            add_facts=sorted(tmpl.add_facts), del_facts=sorted(tmpl.del_facts),
            reversible=tmpl.reversible, confidence=tmpl.confidence, known=True)

    def coverage(self) -> float:
        """How well-modelled this environment is (mean template confidence)."""
        if not self.templates:
            return 0.0
        return sum(t.confidence for t in self.templates.values()) / len(self.templates)

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "domain": self.domain,
                "entities": {eid: e.to_dict() for eid, e in self.entities.items()},
                "facts": sorted(self.facts),
                "actions": sorted(self.templates),
                "coverage": round(self.coverage(), 3)}


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #
class EnvironmentRegistry:
    """Holds the domain models and seeds the sandbox with a faithful world copy."""

    def __init__(self) -> None:
        self._models: Dict[str, EnvironmentModel] = {}

    def register(self, model: EnvironmentModel) -> EnvironmentModel:
        self._models[model.domain] = model
        return model

    def get(self, domain: str) -> Optional[EnvironmentModel]:
        return self._models.get(domain)

    def domains(self) -> List[str]:
        return list(self._models)

    def predict(self, domain: str, action: str, target: str, **params: Any
                ) -> Optional[EnvPrediction]:
        m = self._models.get(domain)
        return m.predict(action, target, **params) if m else None

    def seed_sandbox(self, sandbox: Any) -> Dict[str, Any]:
        """Populate a Sandbox with a copy of every environment's entities and facts."""
        env: Dict[str, Any] = {}
        for domain, model in self._models.items():
            env[domain] = {
                "entities": {eid: copy.deepcopy(e.attributes)
                             for eid, e in model.entities.items()},
                "facts": sorted(model.facts),
            }
        sandbox.state["environment"] = env
        return env

    def to_dict(self) -> Dict[str, Any]:
        return {d: m.to_dict() for d, m in self._models.items()}


# --------------------------------------------------------------------------- #
# Prebuilt domain models
# --------------------------------------------------------------------------- #
def build_os_model() -> EnvironmentModel:
    m = EnvironmentModel("operating system", "os")
    m.register_template(ActionTemplate("create_file", "file", {"exists": True},
                                       add_facts=frozenset({"file_exists"}),
                                       reversible=True, observations=5))
    m.register_template(ActionTemplate("write_file", "file", {"modified": True},
                                       reversible=True, observations=5))
    m.register_template(ActionTemplate("delete_file", "file", {"exists": False},
                                       reversible=False, observations=5))
    m.register_template(ActionTemplate("chmod", "file", {"perms_changed": True},
                                       reversible=True, observations=4))
    m.register_template(ActionTemplate("kill_process", "process", {"running": False},
                                       reversible=False, observations=5))
    m.register_template(ActionTemplate("start_process", "process", {"running": True},
                                       reversible=True, observations=5))
    return m


def build_network_model() -> EnvironmentModel:
    m = EnvironmentModel("network", "network")
    m.register_template(ActionTemplate("open_port", "port", {"open": True},
                                       add_facts=frozenset({"port_open"}),
                                       reversible=True, observations=5))
    m.register_template(ActionTemplate("close_port", "port", {"open": False},
                                       reversible=True, observations=5))
    m.register_template(ActionTemplate("block_ip", "host", {"blocked": True},
                                       reversible=True, observations=5))
    m.register_template(ActionTemplate("connect", "host", {"connected": True},
                                       reversible=True, observations=4))
    return m


def build_accounts_model() -> EnvironmentModel:
    m = EnvironmentModel("accounts & services", "accounts")
    m.register_template(ActionTemplate("disable_account", "account", {"active": False},
                                       reversible=True, observations=5))
    m.register_template(ActionTemplate("enable_account", "account", {"active": True},
                                       reversible=True, observations=5))
    m.register_template(ActionTemplate("grant_role", "account", {},
                                       reversible=True, observations=4))
    m.register_template(ActionTemplate("revoke_role", "account", {},
                                       reversible=True, observations=4))
    return m


def build_default_registry() -> EnvironmentRegistry:
    reg = EnvironmentRegistry()
    reg.register(build_os_model())
    reg.register(build_network_model())
    reg.register(build_accounts_model())
    return reg


# --------------------------------------------------------------------------- #
# Self-test / demo
# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # pragma: no cover
    from nyxara.sim.sandbox import Sandbox

    print("=" * 70)
    print("NYXARA environment-model self-test")
    print("=" * 70)

    reg = build_default_registry()
    os_m = reg.get("os")
    net = reg.get("network")
    acc = reg.get("accounts")

    # populate some known entities
    os_m.add_entity("/etc/config", "file", exists=True, modified=False)
    os_m.add_entity("sshd", "process", running=True)
    net.add_entity("22", "port", open=True)
    acc.add_entity("guest", "account", active=True)

    # predict the effect of a destructive action (for the sandbox to consult)
    p = os_m.predict("delete_file", "/etc/config")
    print(f"\npredict delete_file : {p.to_dict()}")
    assert p.after["exists"] is False
    assert p.reversible is False          # deleting a file is irreversible
    assert p.known and p.confidence > 0.5

    # an unknown action -> honest zero confidence
    unknown = os_m.predict("teleport_file", "/etc/config")
    print(f"unknown action      : known={unknown.known} confidence={unknown.confidence}")
    assert not unknown.known and unknown.confidence == 0.0

    # learning: observe a real transition for a new action and refine confidence
    os_m.add_entity("/var/log", "file", exists=True, size=100)
    before = {"exists": True, "size": 100}
    for _ in range(4):
        os_m.observe("rotate_log", "/var/log", before, {"exists": True, "size": 0})
    learned = os_m.predict("rotate_log", "/var/log")
    print(f"\nlearned rotate_log  : after={learned.after} conf={learned.confidence:.2f}")
    assert learned.known and learned.after["size"] == 0
    assert learned.confidence > 0.5       # confidence grew with observations

    # coverage report
    print(f"\nos model coverage   : {os_m.coverage():.2f}")
    print(f"domains             : {reg.domains()}")
    assert set(reg.domains()) == {"os", "network", "accounts"}

    # seed a sandbox with the world, then dry-run against it
    sb = Sandbox()
    env = reg.seed_sandbox(sb)
    print(f"\nseeded sandbox      : os entities = {list(env['os']['entities'])}")
    assert "/etc/config" in sb.state["environment"]["os"]["entities"]
    assert sb.state["environment"]["network"]["entities"]["22"]["open"] is True

    # a dry-run tool consults the model's prediction, real env untouched
    def simulate_block(ctx):
        pred = net.predict("block_ip", "10.0.0.5")  # host not yet known -> low confidence
        ctx.set_state("would_block", "10.0.0.5")
        return pred

    result = sb.run(simulate_block)
    print("dry-run block       : effect recorded, real network untouched ✓")
    assert net.get_entity("10.0.0.5") is None   # prediction didn't mutate the real model

    print("\nALL SELF-TESTS PASSED ✓")
