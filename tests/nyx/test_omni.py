"""NYX V.01 · L-OMNI — she really does compile herself, and it really does come back.

These tests compile C for real (the container has ``gcc``) and load it into this process, but
never against the live package: every forge target is a module written into ``tmp_path``. That is
the same posture ``tests/conftest.py`` takes for the rest of NYXARA's self-modification — exercise
the mechanism, never the live source tree.
"""

from __future__ import annotations

import shutil
import sys
import textwrap

import pytest

from nyxara.kernel.config import get_settings
from nyxara.nyx.brain import NyxBrain
from nyxara.nyx.omni import DEFAULT_PACKAGES, Kernel, MetamorphicCompiler, _CLowerer, _Unlowerable

_HAS_CC = any(shutil.which(c) for c in ("gcc", "clang", "cc"))
needs_cc = pytest.mark.skipif(not _HAS_CC, reason="no C compiler on PATH")

# A module of ordinary Python. Nothing here is written for the compiler's benefit — the point is
# that these are the shapes real numeric helpers already have.
_SOURCE = '''
def digit_sum_run(n: int) -> int:
    """Sum every digit of every number below n."""
    total = 0
    for i in range(n):
        v = i
        while v > 0:
            total = total + v % 10
            v = v // 10
    return total


def wobble(x: float, n: int) -> float:
    acc = 0.0
    for i in range(n):
        acc = acc + (x * i) / (1.0 + abs(x - i))
    return acc


def negative_floor(a: int, b: int) -> int:
    """Floor division and modulo on mixed signs — where C and Python genuinely disagree."""
    total = 0
    for i in range(b):
        total = total + (-a - i) // (i + 2) + (-a - i) % (i + 2)
    return total


def reads_a_global(n: int) -> int:
    return n * SCALE


def takes_a_list(rows: list) -> int:
    return len(rows)


def clamp(x: float) -> float:
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)


SCALE = 7
'''


@pytest.fixture
def probe(tmp_path, monkeypatch):
    """A real importable module on disk, so the compiler reads and rebinds actual source."""
    (tmp_path / "nyx_omni_probe.py").write_text(textwrap.dedent(_SOURCE), encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    for stale in [m for m in sys.modules if m.startswith("nyx_omni_probe")]:
        del sys.modules[stale]
    yield "nyx_omni_probe"
    for stale in [m for m in sys.modules if m.startswith("nyx_omni_probe")]:
        del sys.modules[stale]


@pytest.fixture
def omni(probe):
    return MetamorphicCompiler(hot_swap=True, packages=(probe,), cases=16)


# -------------------- 1. she reads herself -------------------- #
def test_she_finds_the_loops_and_not_the_rest(omni, probe):
    """The candidate set is the narrow numeric functions — everything else is left alone."""
    found = {k.name for k in omni.scan()}
    assert {"digit_sum_run", "wobble", "negative_floor"} <= found


def test_a_function_reading_a_global_is_not_a_candidate(omni):
    """A free name is not a parameter, so its value cannot be passed to a C kernel."""
    assert "reads_a_global" not in {k.name for k in omni.scan()}


def test_an_unannotated_or_non_numeric_signature_is_not_a_candidate(omni):
    assert "takes_a_list" not in {k.name for k in omni.scan()}


def test_straight_line_arithmetic_is_not_worth_compiling(omni):
    """``clamp`` lowers perfectly well; a ctypes call would simply cost more than it saves."""
    assert "clamp" not in {k.name for k in omni.scan()}


def test_candidates_are_ranked_by_why_native_would_win(omni):
    ranked = omni.scan()
    assert ranked == sorted(ranked, key=lambda k: k.score, reverse=True)
    assert ranked[0].loops >= 1


def test_scanning_names_a_real_place_in_her_source(omni):
    kernel = next(k for k in omni.scan() if k.name == "wobble")
    assert kernel.file.endswith("nyx_omni_probe.py") and kernel.lineno > 0
    assert kernel.params == [("x", "float"), ("n", "int")]
    assert kernel.returns == "float" and kernel.integral is False


# -------------------- 2. the lowering keeps Python's semantics -------------------- #
def test_floor_division_rounds_the_way_python_rounds(omni):
    """C truncates toward zero and Python floors toward −∞. Emitting ``/`` would be a bug."""
    kernel = next(k for k in omni.scan() if k.name == "negative_floor")
    c = _CLowerer(omni._ast_of(kernel), kernel.params, integral=True).lower(symbol="k")
    assert "nyx_fdiv" in c and "nyx_fmod" in c
    assert " / " not in c.split("long long k(")[1]


def test_the_lowerer_refuses_what_it_cannot_promise(omni, probe):
    """Refusal is the normal outcome. It raises a private signal, never emits doubtful C."""
    import ast
    for body in ("return sum(xs)", "return a.b", "return [i for i in range(a)]",
                 "return a < b < a", "return a ** b"):
        node = ast.parse(f"def f(a: int, b: int) -> int:\n    {body}\n").body[0]
        with pytest.raises(_Unlowerable):
            _CLowerer(node, [("a", "int"), ("b", "int")], integral=True).lower(symbol="k")


def test_the_emitted_kernel_never_needs_libm(omni):
    """The forge links without -lm, so an undefined ``floor`` would fail to load."""
    kernel = next(k for k in omni.scan() if k.name == "wobble")
    c = _CLowerer(omni._ast_of(kernel), kernel.params, integral=False).lower(symbol="k")
    assert "math.h" not in c and "nyx_floor" in c


# -------------------- 3-5. compile, verify, adopt -------------------- #
@needs_cc
def test_she_compiles_an_integer_kernel_and_it_gives_identical_answers(omni, probe):
    import importlib
    module = importlib.import_module(probe)
    kernel = next(k for k in omni.scan() if k.name == "digit_sum_run")
    before = [module.digit_sum_run(n) for n in (0, 1, 9, 10, 97, 500)]

    got = omni.forge(kernel)
    assert got.status == "adopted", got.reason
    assert got.speedup >= omni.min_speedup and got.language == "c"
    assert getattr(module.digit_sum_run, "__nyx_native__", False) is True
    assert [module.digit_sum_run(n) for n in (0, 1, 9, 10, 97, 500)] == before


@needs_cc
def test_she_compiles_a_floating_point_kernel_bit_for_bit(omni, probe):
    """Not "close enough": the same IEEE doubles in the same order give the same answer."""
    import importlib
    module = importlib.import_module(probe)
    kernel = next(k for k in omni.scan() if k.name == "wobble")
    before = module.wobble(1.5, 60)

    assert omni.forge(kernel).status == "adopted"
    assert module.wobble(1.5, 60) == before


@needs_cc
def test_the_certificate_says_what_was_actually_checked(omni):
    kernel = next(k for k in omni.scan() if k.name == "digit_sum_run")
    got = omni.forge(kernel)
    assert "identical on" in got.certificate and "faster" in got.certificate
    assert "fresh cases over" in got.certificate      # the domain, not just the claim
    assert got.cases >= 4


@needs_cc
def test_mixed_sign_floor_semantics_survive_the_equivalence_gate(omni, probe):
    """The one place the lowering could be subtly wrong, checked against Python itself."""
    import importlib
    module = importlib.import_module(probe)
    kernel = next(k for k in omni.scan() if k.name == "negative_floor")
    reference = module.negative_floor
    before = [reference(a, b) for a in (3, 17) for b in (1, 5, 12)]

    got = omni.forge(kernel)
    assert got.status in ("adopted", "rejected")       # speed is not guaranteed; truth is
    assert [module.negative_floor(a, b) for a in (3, 17) for b in (1, 5, 12)] == before


# -------------------- the gauntlet cannot be waved through -------------------- #
@needs_cc
def test_a_kernel_that_is_not_faster_is_not_adopted(probe):
    strict = MetamorphicCompiler(hot_swap=True, packages=(probe,), cases=8,
                                 min_speedup=10_000.0)
    kernel = next(k for k in strict.scan() if k.name == "digit_sum_run")
    got = strict.forge(kernel)
    assert got.status == "rejected" and "under 10000.0x" in got.reason
    import importlib
    assert getattr(importlib.import_module(probe).digit_sum_run,
                   "__nyx_native__", False) is False


@needs_cc
def test_a_kernel_that_disagrees_after_the_swap_is_rolled_back(omni, probe, monkeypatch):
    """The rewriter re-checks on *fresh* inputs after applying. This is that path."""
    import importlib
    module = importlib.import_module(probe)
    original = module.digit_sum_run
    kernel = next(k for k in omni.scan() if k.name == "digit_sum_run")

    # Hand back a "native" kernel that is wrong. Everything else is the real pipeline.
    monkeypatch.setattr("nyxara.nyx.omni._wrap",
                        lambda fn, names, pyname: lambda *a, **k: -12345)
    got = omni.forge(kernel)
    assert got.status == "rejected" and "baseline" in (got.reason or "")
    assert module.digit_sum_run is original


# -------------------- the safety core is not a candidate -------------------- #
def test_rewriting_the_safety_core_is_refused_before_anything_compiles():
    omni = MetamorphicCompiler(hot_swap=True)
    for name in ("loyalty", "corrigibility", "oversight", "honesty", "guard.shield"):
        kernel = Kernel(module=f"nyxara.growth.{name}", name="score", file="x.py", lineno=1,
                        params=[("n", "int")], returns="int", integral=True, loops=1, ops=8)
        got = omni.forge(kernel)
        assert got.status == "refused" and "safety core" in got.reason


def test_a_constitutional_file_is_never_scanned():
    omni = MetamorphicCompiler(hot_swap=True, packages=("nyxara.kernel",))
    assert omni._scan_one("nyxara.kernel.config") == []
    assert omni._protected_file("/opt/nyxara/identity/values.py") is True
    assert omni._protected_file("/opt/nyxara/nyx/graph.py") is False


def test_an_unresolvable_path_is_treated_as_protected():
    """Fail-closed: cannot prove it is safe ⇒ it is not a candidate."""
    assert MetamorphicCompiler._protected_file(None) is True     # type: ignore[arg-type]


# -------------------- what drives the policy is her own latency -------------------- #
def test_the_target_comes_from_where_she_is_measurably_slow(probe):
    omni = MetamorphicCompiler(hot_swap=True, packages=("nyxara.nyx",))
    omni.observe(probe, 40.0)
    omni.observe("nyxara.nyx.graph", 1.0)
    assert omni.hottest(1) == [(probe, 40.0)]
    assert omni.next_target().module == probe          # the hot module, not the configured one


def test_a_real_thought_records_where_its_time_went():
    brain = NyxBrain(get_settings().nyx.model_copy(update={
        "holo_dim": 512, "aura_enabled": False, "car_enabled": False,
        "episteme_enabled": False, "dialogue_enabled": False, "chronos_enabled": False}))
    brain.think("what happens when an apple falls")
    assert brain.omni is not None
    assert "nyxara.nyx.workspace" in brain.omni.latency
    assert sum(brain.omni.latency.values()) > 0.0


def test_no_function_is_nominated_by_hand():
    """The candidate list is parsed out of her source; there is no hard-coded target anywhere."""
    omni = MetamorphicCompiler(hot_swap=True)
    assert DEFAULT_PACKAGES and all(p.startswith("nyxara.") for p in DEFAULT_PACKAGES)
    assert not any(hasattr(omni, attr) for attr in ("targets", "TARGETS", "_targets"))


def test_reading_herself_is_spread_across_beats():
    omni = MetamorphicCompiler(hot_swap=True, scan_per_beat=2)
    omni.next_target()
    assert len(omni._scanned) <= 2
    omni.next_target()
    assert len(omni._scanned) <= 4


# -------------------- reversal -------------------- #
@needs_cc
def test_rollback_restores_the_python_function(omni, probe):
    import importlib
    module = importlib.import_module(probe)
    original = module.digit_sum_run
    omni.forge(next(k for k in omni.scan() if k.name == "digit_sum_run"))
    assert module.digit_sum_run is not original

    assert omni.rollback_all() == 1
    assert module.digit_sum_run is original
    assert omni.adopted == {}


def test_rolling_back_something_never_adopted_is_not_an_error(omni):
    assert omni.rollback("nyxara.nowhere.nothing") is False
    assert omni.rollback_all() == 0


# -------------------- budget and oversight -------------------- #
@needs_cc
def test_the_hourly_forge_budget_is_respected(probe):
    once = MetamorphicCompiler(hot_swap=True, packages=(probe,), cases=8,
                               max_forges_per_hour=1)
    assert once.beat() is not None
    assert once.beat() is None                        # budget spent for the hour
    once.rollback_all()


def test_a_scrammed_mind_does_not_recompile_itself(omni):
    class _Gate:
        def gate(self):
            return False

    assert omni.beat(oversight=_Gate()) is None
    assert omni.attempts == []


@needs_cc
def test_an_open_gate_permits_one_attempt(omni):
    class _Gate:
        def gate(self):
            return True

    assert omni.beat(oversight=_Gate()) is not None
    omni.rollback_all()


def test_the_brain_recompiles_itself_on_a_slower_cadence_than_it_thinks():
    brain = NyxBrain(get_settings().nyx.model_copy(update={
        "holo_dim": 512, "aura_enabled": False, "car_enabled": False,
        "episteme_enabled": False, "dialogue_enabled": False,
        "omni_every_s": 3600.0}))
    for _ in range(5):
        brain.tick()
    assert len(brain.omni.attempts) <= 1


# -------------------- no toolchain, no layer -------------------- #
def test_without_the_live_swap_it_is_a_clean_no_op(probe):
    scan_only = MetamorphicCompiler(hot_swap=False, packages=(probe,))
    assert scan_only.available() is False
    assert scan_only.beat() is None
    assert scan_only.scan()                            # she can still read herself
    got = scan_only.forge(scan_only.scan()[0])
    assert got.status == "unavailable" and "hot-swap is off" in got.reason


def test_the_suite_runs_with_the_live_swap_sealed():
    """conftest pins NYXARA_NYX__OMNI_HOT_SWAP=false so CI never invokes a compiler."""
    assert get_settings().nyx.omni_hot_swap is False
    assert get_settings().nyx.omni_enabled is True
    assert NyxBrain(get_settings().nyx).omni.available() is False


def test_a_missing_toolchain_is_reported_not_faked(omni, monkeypatch):
    monkeypatch.setattr("nyxara.growth.native_forge.NativeForge.available",
                        staticmethod(lambda language: False))
    assert omni.available() is False
    assert omni.stats()["toolchain"] == []
    assert omni.beat() is None


# -------------------- robustness -------------------- #
def test_a_function_that_has_since_moved_fails_quietly(omni):
    got = omni.forge(Kernel(module="nyx_omni_probe", name="vanished", file="/nope.py",
                            lineno=1, params=[("n", "int")], returns="int", integral=True,
                            loops=1, ops=8))
    assert got.status in ("rejected", "unavailable") and got.adopted is False


def test_unparseable_source_is_simply_not_a_candidate(tmp_path, monkeypatch):
    (tmp_path / "nyx_omni_broken.py").write_text("def f(:\n", encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    assert MetamorphicCompiler()._scan_one("nyx_omni_broken") == []


def test_stats_are_honest_about_reversibility(omni):
    stats = omni.stats()
    assert stats["reversible"] is True
    assert stats["budget_per_hour"] == omni.max_forges_per_hour
    assert stats["min_speedup"] == omni.min_speedup


def test_state_round_trips(omni, probe):
    omni.observe(probe, 12.5)
    omni.forge(omni.scan()[0])
    data = omni.to_dict()

    other = MetamorphicCompiler(hot_swap=False)
    assert other.load_dict(data) is True
    assert other.latency[probe] == 12.5
    assert other._tried == omni._tried
    omni.rollback_all()


def test_a_compiled_kernel_is_deliberately_not_persisted(omni, probe):
    """A restart must come back as pure Python. Persisting the swap would break that promise."""
    omni.observe(probe, 1.0)
    assert "adopted" not in omni.to_dict()
    assert "so_path" not in str(omni.to_dict())


def test_corrupt_state_does_not_block(omni):
    assert omni.load_dict(None) is False
    assert omni.load_dict("nonsense") is False
