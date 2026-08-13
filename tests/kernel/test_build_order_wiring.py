"""Faculties that must actually be CONSTRUCTED at boot, not just declared.

``NyxaraCore`` builds every faculty behind a blanket ``except Exception: return None`` so a
missing optional dependency degrades honestly instead of killing the boot. The cost of that
pattern is that an ordinary programming error inside a builder is indistinguishable from
"this machine lacks torch" — the faculty simply comes up ``None`` and nothing says so.

That is exactly what had happened to Levels 3 and 4. ``_build_recursive_improver`` and
``_build_role_council`` both read ``self.reasoner.llm``, but were called ~300 lines BEFORE
``self.reasoner`` was assigned. Accessing ``self.reasoner`` raised ``AttributeError``, the
blanket handler swallowed it, and both returned ``None`` on every single boot — silently, with
``strategic_intelligence`` (documented as composing the role council) also receiving ``None``.

These tests pin the construction order so the regression cannot come back quietly.
"""

from __future__ import annotations

import types

from nyxara.kernel.orchestrator import NyxaraCore


def test_level3_recursive_improver_is_built_at_boot():
    core = NyxaraCore()
    assert core.recursive_improver is not None, (
        "Level 3 (recursive self-improvement) must be constructed at boot")


def test_level4_role_council_is_built_at_boot():
    # the suite seals Level 4 for speed (see conftest), so opt back in to test the wiring
    core = _core_with(NYXARA_ROLE_COUNCIL__ENABLED="true")
    assert core.role_council is not None, (
        "Level 4 (six-role internal council) must be constructed at boot")


def test_role_council_reaches_strategic_intelligence():
    """strategic_intelligence composes the council — it must get the real one, not None."""
    core = _core_with(NYXARA_ROLE_COUNCIL__ENABLED="true")
    strategic = core.strategic_intelligence
    assert strategic is not None
    council = getattr(strategic, "council", None)
    # not-None first: with both sides dead, a bare identity check passes on None is None.
    assert council is not None, "strategic_intelligence received no role council"
    assert council is core.role_council


def test_builders_run_after_the_reasoner_exists():
    """The ordering itself, not just the end state: both builders must see a live reasoner."""
    seen: dict[str, bool] = {}

    def spy(name, fn):
        def inner(self):
            seen[name] = getattr(self, "reasoner", None) is not None
            return fn(self)
        return inner

    original = {n: getattr(NyxaraCore, n)
                for n in ("_build_recursive_improver", "_build_role_council")}
    try:
        for name, fn in original.items():
            setattr(NyxaraCore, name, spy(name, fn))
        _core_with(NYXARA_ROLE_COUNCIL__ENABLED="true")
    finally:
        for name, fn in original.items():
            setattr(NyxaraCore, name, fn)

    assert seen.get("_build_recursive_improver") is True, (
        "_build_recursive_improver ran before self.reasoner was assigned")
    assert seen.get("_build_role_council") is True, (
        "_build_role_council ran before self.reasoner was assigned")


def test_no_builder_silently_returns_none_except_documented_optional_deps():
    """A boot-wide sweep: any ``_build_*`` returning None is an unavailable capability.

    ``continuous_physics`` is the one legitimate None on a torch-less machine (its builder
    checks ``torch_available()`` and declines honestly). Anything else coming up None means a
    builder threw and the blanket handler ate it — the failure mode this module exists for.
    """
    returned_none: list[str] = []
    original: dict[str, types.FunctionType] = {}

    def spy(name, fn):
        def inner(self, *a, **kw):
            out = fn(self, *a, **kw)
            if out is None:
                returned_none.append(name)
            return out
        return inner

    for name in dir(NyxaraCore):
        if not name.startswith("_build_"):
            continue
        fn = getattr(NyxaraCore, name)
        if isinstance(fn, types.FunctionType):
            original[name] = fn
    try:
        for name, fn in original.items():
            setattr(NyxaraCore, name, spy(name, fn))
        _core_with(NYXARA_ROLE_COUNCIL__ENABLED="true")
    finally:
        for name, fn in original.items():
            setattr(NyxaraCore, name, fn)

    try:
        from nyxara.mind.continuous_world import torch_available
        torch_present = torch_available()
    except Exception:  # noqa: BLE001
        torch_present = False

    allowed = set() if torch_present else {"_build_continuous_physics"}
    unexpected = sorted(set(returned_none) - allowed)
    assert not unexpected, f"builders silently returned None at boot: {unexpected}"


# --------------------------------------------------------------------------- #
# The Level-4 config block must actually be connected to something
# --------------------------------------------------------------------------- #
# ``RoleCouncilConfig`` was defined, attached to ``NyxaraSettings`` and documented, but nothing
# read it: ``_build_role_council`` hardcoded 256/30.0 and never consulted ``enabled``. The
# council is a SECOND full deliberation on every respond turn, so a decorative off-switch meant
# its cost could not be declined. (config.py makes the same complaint about FeatureFlags that
# "were decorative — flipping them did nothing".)
def _core_with(**env):
    import importlib
    import os
    from nyxara.kernel.config import reload_settings
    old = {k: os.environ.get(k) for k in env}
    os.environ.update({k: str(v) for k, v in env.items()})
    try:
        reload_settings()
        orch = importlib.import_module("nyxara.kernel.orchestrator")
        return orch.NyxaraCore()
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        reload_settings()


def test_role_council_enabled_flag_is_honoured():
    core = _core_with(NYXARA_ROLE_COUNCIL__ENABLED="false")
    assert core.role_council is None, (
        "NYXARA_ROLE_COUNCIL__ENABLED=false must actually disable Level 4")


def test_role_council_tuning_knobs_reach_the_council():
    core = _core_with(NYXARA_ROLE_COUNCIL__ENABLED="true",
                      NYXARA_ROLE_COUNCIL__MAX_TOKENS_PER_ROLE="64",
                      NYXARA_ROLE_COUNCIL__TIMEOUT_S="5")
    assert core.role_council is not None
    assert core.role_council.max_tokens == 64
    assert core.role_council.timeout_s == 5.0


def test_role_council_defaults_are_unchanged():
    core = _core_with(NYXARA_ROLE_COUNCIL__ENABLED="true")
    assert core.role_council is not None
    assert core.role_council.max_tokens == 256
    assert core.role_council.timeout_s == 30.0
