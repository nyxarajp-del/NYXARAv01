"""Tests for nyxara.growth.skill_factory — autonomous skill creation.

Fully offline: a fake skill memory, toolsmith registry, and episode stand in for the live
components, so the detect → draft → test → store pipeline runs deterministically in CI.
"""

from __future__ import annotations

from nyxara.growth.skill_factory import SkillFactory


# --------------------------------------------------------------------------- #
# Test doubles
# --------------------------------------------------------------------------- #
class _Spec:
    def __init__(self, description: str) -> None:
        self.description = description


class _Registry:
    def __init__(self, mapping: dict) -> None:
        self._m = mapping

    def get(self, name: str):
        return self._m.get(name)


class _Toolsmith:
    def __init__(self, registry: _Registry) -> None:
        self.registry = registry


class _Episode:
    def __init__(self, tools=None, rationale: str = "") -> None:
        self.tools = tools or []
        self.rationale = rationale


class _Skill:
    def __init__(self, goal: str) -> None:
        self.goal = goal


class _Memory:
    def __init__(self) -> None:
        self.learned = []

    def recall(self, goal_type: str, k: int = 1):
        return []

    def learn(self, goal_type: str, procedure: str, tools=None):
        self.learned.append((goal_type, procedure, list(tools or [])))
        return _Skill(goal_type)


# --------------------------------------------------------------------------- #
# _draft_skill — Toolsmith composition vs. fallbacks
# --------------------------------------------------------------------------- #
def test_draft_uses_toolsmith_when_tools_present():
    reg = _Registry({"fetch": _Spec("fetch a url"), "parse": _Spec("parse json")})
    sf = SkillFactory(toolsmith=_Toolsmith(reg))
    proc, tools = sf._draft_skill("collect data", _Episode(["fetch", "parse"]))
    assert tools == ["fetch", "parse"]
    assert "fetch" in proc and "parse" in proc           # grounded, names both tools
    assert "fetch a url" in proc                          # annotated with the registered spec
    assert "1." in proc and "2." in proc                 # numbered, real pipeline


def test_draft_falls_back_to_rationale_without_toolsmith():
    sf = SkillFactory()                                   # no toolsmith
    proc, tools = sf._draft_skill("do thing", _Episode(rationale="step one then step two"))
    assert proc == "step one then step two"
    assert tools == []


def test_draft_minimal_when_nothing_available():
    sf = SkillFactory()
    proc, tools = sf._draft_skill("achieve world peace", None)
    assert proc.startswith("achieve:")
    assert tools == []


def test_compose_with_toolsmith_empty_without_tools():
    sf = SkillFactory(toolsmith=_Toolsmith(_Registry({})))
    assert sf._compose_with_toolsmith("goal", []) == ""   # nothing to compose


# --------------------------------------------------------------------------- #
# maybe_create_skill — full pipeline
# --------------------------------------------------------------------------- #
def test_threshold_must_be_reached_before_creating():
    mem = _Memory()
    sf = SkillFactory(skill_memory=mem, threshold=2)
    first = sf.maybe_create_skill("scrape the site", _Episode(["fetch"]))
    assert not first.triggered and not first.skill_created


def test_triggers_and_composes_after_threshold():
    reg = _Registry({"fetch": _Spec("fetch a url")})
    mem = _Memory()
    sf = SkillFactory(skill_memory=mem, toolsmith=_Toolsmith(reg), threshold=2)
    ep = _Episode(["fetch"], rationale="r")
    sf.maybe_create_skill("scrape the site", ep)          # count 1 — not yet
    result = sf.maybe_create_skill("scrape the site", ep)  # count 2 — fires
    assert result.triggered and result.skill_created
    assert mem.learned and "fetch" in mem.learned[0][1]   # composed procedure stored


def test_does_not_recreate_existing_skill():
    class _MemHas(_Memory):
        def recall(self, goal_type, k=1):
            return [_Skill(goal_type)]                     # pretend it already exists

    sf = SkillFactory(skill_memory=_MemHas(), threshold=2)
    ep = _Episode(["fetch"])
    sf.maybe_create_skill("scrape the site", ep)
    result = sf.maybe_create_skill("scrape the site", ep)
    assert result.triggered and not result.skill_created
    assert result.reason == "skill already stored"
