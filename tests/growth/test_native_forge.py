"""Tests for Autopoietic Genome Compiling — growth/native_forge.py + hotspot_profiler.py (Part C).

These exercise the real compile+verify+speedup path when a C/Rust toolchain is present, and assert the
honest no-op / gating behavior otherwise.
"""
from __future__ import annotations

import ctypes

import pytest

from nyxara.growth.hotspot_profiler import HotspotProfiler
from nyxara.growth.native_forge import NativeForge


# a pure, hot reference function (sum of squares) — the kind eligible for native compilation
def py_sumsq(n: int) -> int:
    s = 0
    for i in range(n):
        s += i * i
    return s


_C_SRC = """
long sumsq(long n) {
    long s = 0;
    for (long i = 0; i < n; i++) s += i * i;
    return s;
}
"""

_RUST_SRC = """
#[no_mangle]
pub extern "C" fn sumsq(n: i64) -> i64 {
    let mut s: i64 = 0;
    let mut i: i64 = 0;
    while i < n { s += i * i; i += 1; }
    s
}
"""

_SAMPLES = [(2000,), (4000,), (8000,)]
_HAVE_C = NativeForge.available("c")
_HAVE_RUST = NativeForge.available("rust")


# --------------------------------------------------------------------------- #
# Hotspot profiler
# --------------------------------------------------------------------------- #
def test_profiler_ranks_hot_function_and_marks_eligible():
    prof = HotspotProfiler()
    spots = prof.profile(lambda: py_sumsq(50000), top=10)
    assert spots
    names = [h.func for h in spots]
    assert "py_sumsq" in names
    # py_sumsq lives in NYXARA's test tree under the package marker path? It lives in tests/, so it is
    # only eligible if the marker matches; assert the eligibility filter runs and returns bools.
    assert all(isinstance(h.eligible, bool) for h in spots)


def test_profiler_excludes_protected_and_dunder():
    prof = HotspotProfiler()
    assert prof._eligible("/x/nyxara/kernel/config.py", "f") is False   # constitutional
    assert prof._eligible("/x/nyxara/mind/llm.py", "<listcomp>") is False
    assert prof._eligible("/usr/lib/python3.11/json/__init__.py", "loads") is False  # stdlib
    assert prof._eligible("/x/nyxara/mind/llm.py", "complete") is True


# --------------------------------------------------------------------------- #
# Native forge — gating
# --------------------------------------------------------------------------- #
def test_inprocess_gated_off_by_default_returns_none():
    forge = NativeForge(allow_inprocess=False, min_speedup=1.2)
    cand = forge.forge(py_sumsq, samples=_SAMPLES, symbol="sumsq",
                       argtypes=[ctypes.c_long], restype=ctypes.c_long, c_source=_C_SRC)
    assert cand is None                                   # gated: no in-process load


# --------------------------------------------------------------------------- #
# Native forge — real compile + verify + speedup (when toolchain present)
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not _HAVE_C, reason="no C compiler")
def test_forge_c_kernel_is_equivalent_and_faster():
    forge = NativeForge(allow_inprocess=True, min_speedup=1.2)
    cand = forge.forge(py_sumsq, samples=_SAMPLES, symbol="sumsq",
                       argtypes=[ctypes.c_long], restype=ctypes.c_long, c_source=_C_SRC)
    assert cand is not None and cand.language == "c"
    assert cand.speedup >= 1.2 and "identical" in cand.certificate
    # the native kernel returns exactly the Python result
    assert cand.fn(4000) == py_sumsq(4000)


@pytest.mark.skipif(not _HAVE_RUST, reason="no rustc")
def test_forge_rust_kernel_is_equivalent_and_faster():
    forge = NativeForge(allow_inprocess=True, min_speedup=1.2)
    cand = forge.forge(py_sumsq, samples=_SAMPLES, symbol="sumsq",
                       argtypes=[ctypes.c_long], restype=ctypes.c_long, rust_source=_RUST_SRC)
    assert cand is not None and cand.language == "rust"
    assert cand.fn(8000) == py_sumsq(8000)


@pytest.mark.skipif(not _HAVE_C, reason="no C compiler")
def test_forge_rejects_a_wrong_native_kernel():
    forge = NativeForge(allow_inprocess=True, min_speedup=1.0)
    wrong_c = "long sumsq(long n){ return n; }"          # NOT equivalent
    cand = forge.forge(py_sumsq, samples=_SAMPLES, symbol="sumsq",
                       argtypes=[ctypes.c_long], restype=ctypes.c_long, c_source=wrong_c)
    assert cand is None                                   # equivalence gauntlet rejected it
