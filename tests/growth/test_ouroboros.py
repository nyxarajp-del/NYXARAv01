"""Tests for nyxara.growth.ouroboros — she may rewrite her own source, under proof."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from nyxara.growth.ouroboros import Ouroboros
from nyxara.kernel.config import NyxaraSettings, Profile

_GROWTH = Path(__file__).resolve().parent.parent.parent / "nyxara" / "growth"


@pytest.fixture()
def probe(request):
    """A throwaway module inside growth/ so the allowlist genuinely applies to it."""
    path = _GROWTH / f"_probe_{abs(hash(request.node.name)) % 10_000_000}.py"
    yield path
    if path.exists():
        path.unlink()


def _ouroboros() -> Ouroboros:
    return Ouroboros(settings=NyxaraSettings.for_profile(Profile.TEST))


# -------------------- the transforms are semantics-preserving -------------------- #
def test_every_transform_preserves_behaviour(probe):
    """The one property that cannot be traded away: the rewritten code must compute the same
    thing. Anything else is a bug that a passing gauntlet would happily ship."""
    probe.write_text(textwrap.dedent('''
        def f(x, d, xs):
            a = x in ['a', 'b', 'c']
            b = x in d.keys()
            c = [k for k in d.keys()]
            e = sum([i * 2 for i in xs])
            g = any([i > 1 for i in xs])
            return (a, b, sorted(c), e, g)
    ''').lstrip(), encoding="utf-8")

    edit = _ouroboros().propose(probe)
    assert edit is not None

    before_ns, after_ns = {}, {}
    exec(edit.before, before_ns)          # noqa: S102 — executing our own fixture on purpose
    exec(edit.after, after_ns)            # noqa: S102
    for case in [("a", {"k": 1, "j": 2}, [1, 2, 3]), ("z", {}, []), ("b", {"z": 9}, [5])]:
        assert before_ns["f"](*case) == after_ns["f"](*case)


def test_comments_and_formatting_survive(probe):
    """Rewrites are spliced at exact spans rather than round-tripped through ast.unparse, which
    would silently delete every comment in the file — self-vandalism on the first run."""
    probe.write_text(textwrap.dedent('''
        # a leading comment
        def f(x):
            """A docstring."""
            if x in ['a', 'b']:      # a trailing comment
                return 1
            return 0
    ''').lstrip(), encoding="utf-8")

    edit = _ouroboros().propose(probe)
    assert edit is not None
    assert "# a leading comment" in edit.after
    assert "# a trailing comment" in edit.after
    assert '"""A docstring."""' in edit.after
    assert "{'a', 'b'}" in edit.after


def test_unhashable_elements_are_left_alone(probe):
    """Turning a list into a set with an unhashable element turns a working comparison into a
    TypeError, so the precondition must actually be discharged."""
    probe.write_text("def g(x):\n    return x in [[1, 2], [3, 4]]\n", encoding="utf-8")
    assert _ouroboros().propose(probe) is None


def test_non_constant_elements_are_left_alone(probe):
    probe.write_text("def g(x, y):\n    return x in [y, 2, 3]\n", encoding="utf-8")
    assert _ouroboros().propose(probe) is None


def test_list_is_not_lazily_aggregated(probe):
    """For list()/dict() the materialised list IS the answer — only aggregators that consume the
    iterable once may be handed a generator."""
    probe.write_text("def g(xs):\n    return list([i for i in xs])\n", encoding="utf-8")
    assert _ouroboros().propose(probe) is None


def test_a_file_with_nothing_to_improve_proposes_nothing(probe):
    probe.write_text("def g(x):\n    return x + 1\n", encoding="utf-8")
    assert _ouroboros().propose(probe) is None


def test_output_is_always_valid_python(probe):
    import ast

    probe.write_text(textwrap.dedent('''
        def f(x, d):
            return (x in ['a', 'b'], x in d.keys(), sum([i for i in range(3)]))
    ''').lstrip(), encoding="utf-8")
    edit = _ouroboros().propose(probe)
    assert edit is not None
    ast.parse(edit.after)


# -------------------- the fences -------------------- #
def test_the_constitution_is_out_of_reach():
    """The Master's standing requirement: the Loyalty Equation and the sovereign limits are never
    rewritable. Enforced by PROTECTED_RELPATHS, which predates this module — and re-checked here."""
    engine = _ouroboros()
    for sealed in ("nyxara/growth/loyalty.py", "nyxara/identity/values.py",
                   "nyxara/kernel/rules.py", "nyxara/guard/corrigibility.py",
                   "nyxara/kernel/config.py"):
        assert not engine.in_scope(sealed), sealed


def test_only_the_allowlisted_subpackages_are_in_scope():
    engine = _ouroboros()
    assert engine.in_scope("nyxara/growth/dataset.py")
    assert engine.in_scope("nyxara/mind/reasoner.py")
    assert not engine.in_scope("nyxara/agency/permissions.py")
    assert not engine.in_scope("nyxara/guard/shield.py")


def test_paths_outside_the_package_are_refused():
    engine = _ouroboros()
    assert not engine.in_scope("/etc/passwd")
    assert not engine.in_scope("/tmp/anything.py")
    assert not engine.in_scope("")


def test_proposing_for_an_out_of_scope_file_returns_nothing():
    assert _ouroboros().propose("nyxara/kernel/config.py") is None


def test_a_custom_allowlist_is_honoured():
    engine = Ouroboros(settings=NyxaraSettings.for_profile(Profile.TEST), allowlist=("mind",))
    assert engine.in_scope("nyxara/mind/reasoner.py")
    assert not engine.in_scope("nyxara/growth/dataset.py")


# -------------------- the cycle -------------------- #
def test_measure_only_writes_nothing(probe):
    original = "def g(x):\n    return x in ['a', 'b']\n"
    probe.write_text(original, encoding="utf-8")
    report = _ouroboros().metamorphose(enact=False)
    assert report.applied == 0 and report.kept == 0
    assert probe.read_text(encoding="utf-8") == original
    assert any("measure-only" in n for n in report.notes)


def test_enacting_on_a_dirty_tree_is_refused(monkeypatch):
    """Optimizer's rollback is an in-memory string; if the process dies mid-edit it is gone. A
    clean tree is what makes `git checkout` a real last-resort undo."""
    engine = _ouroboros()
    engine.require_clean_tree = True
    monkeypatch.setattr(engine, "tree_is_clean", lambda: False)

    report = engine.metamorphose(enact=True)
    assert report.applied == 0
    assert any("working tree is dirty" in n for n in report.notes)


def test_an_edit_with_no_speedup_is_restored(probe, monkeypatch):
    """improvement_proof certifies against a static cost proxy. Ouroboros additionally looks at
    the clock, and puts the file back when the clock disagrees."""
    original = "def g(x):\n    return x in ['a', 'b']\n"
    probe.write_text(original, encoding="utf-8")

    engine = _ouroboros()
    engine.require_clean_tree = False
    engine.require_speedup = True

    class _Outcome:
        applied = True
        kept = True
        rolled_back = False
        reason = ""

    class _FakeOptimizer:
        def apply(self, edit):
            Path(edit.file).write_text(edit.after, encoding="utf-8")
            return _Outcome()

    engine._optimizer = _FakeOptimizer()
    monkeypatch.setattr(engine, "scan", lambda **_kw: [__import__(
        "nyxara.growth.ouroboros", fromlist=["Bottleneck"]).Bottleneck(file=str(probe))])
    # a workload that is slower afterwards, so the speedup gate must fire
    calls = {"n": 0}

    def _workload():
        calls["n"] += 1
        if calls["n"] > 1:
            for _ in range(200_000):
                pass

    report = engine.metamorphose(workload=_workload, enact=True)
    assert report.rolled_back >= 1
    assert probe.read_text(encoding="utf-8") == original
    assert any("no measured speedup" in n for n in report.notes)


def test_report_serialises():
    report = _ouroboros().metamorphose(enact=False)
    blob = report.to_dict()
    assert {"scanned", "proposed", "kept", "rolled_back", "speedup"} <= set(blob)


def test_speedup_is_one_when_nothing_was_timed():
    report = _ouroboros().metamorphose(enact=False)
    assert report.speedup == 1.0


# -------------------- config -------------------- #
def test_ouroboros_is_off_by_default():
    """Every other growth layer here is additive; this one edits her own source, so the Master
    turns it on deliberately."""
    for profile in (Profile.DEV, Profile.PROD, Profile.TEST):
        settings = NyxaraSettings.for_profile(profile)
        assert settings.self_improvement.ouroboros_enabled is False
