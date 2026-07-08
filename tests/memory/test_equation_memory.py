"""Tests for nyxara.memory.equation_memory — data stored as mathematical equations.

Covers each coder's round-trip fidelity, the honesty of the measured ratios (structured
data shrinks, random data does not), the literal Mandelbrot generator, determinism (no LLM,
no randomness), and end-to-end integration with :class:`MemoryStore` persistence.
"""

from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest

from nyxara.memory.equation_memory import (
    EquationCode,
    EquationMemory,
    MandelbrotIFSCoder,
    PolynomialCoder,
    RunLengthCoder,
    SpectralCoder,
    SymbolicCoder,
)
from nyxara.memory.store import MemoryStore, MemoryType


# -------------------- engine round-trips -------------------- #
def _rel_err(a, b):
    a = np.asarray(a, dtype="float64").ravel()
    b = np.asarray(b, dtype="float64").ravel()
    return float(np.linalg.norm(b - a) / (np.linalg.norm(a) + 1e-12))


def test_constant_is_lossless_and_tiny():
    em = EquationMemory()
    arr = np.full(256, 3.14159)
    code = em.compress(arr)
    recon = em.decompress(code)
    assert _rel_err(arr, recon) < 1e-4
    assert code.ratio > 5.0  # a whole array from a couple of numbers


def test_arithmetic_progression_lossless():
    em = EquationMemory()
    arr = np.arange(200) * 0.5 + 2.0
    code = em.compress(arr)
    assert _rel_err(arr, em.decompress(code)) < 1e-3
    assert code.ratio > 5.0


def test_sinusoid_uses_few_terms():
    em = EquationMemory()
    x = np.linspace(0, 8 * np.pi, 256)
    arr = 3.0 * np.sin(x) + 1.0
    code = em.compress(arr)
    recon = em.decompress(code)
    assert _rel_err(arr, recon) <= 0.09       # within the aggressive budget
    assert code.ratio > 3.0
    assert code.kind in {"spectral", "symbolic"}


def test_smooth_polynomial_signal():
    em = EquationMemory()
    t = np.linspace(-1, 1, 256)
    arr = t ** 3 - 0.5 * t
    code = em.compress(arr)
    assert _rel_err(arr, em.decompress(code)) < 1e-3
    assert code.ratio > 4.0


def test_random_data_does_not_fake_shrink():
    """Honesty: incompressible noise must not report a large ratio, and must round-trip."""
    em = EquationMemory()
    arr = np.random.RandomState(0).randn(256)
    code = em.compress(arr)
    recon = em.decompress(code)
    assert code.kind == "raw"          # nothing structured to exploit
    assert code.ratio < 2.0            # no magical compression of noise
    assert _rel_err(arr, recon) < 1e-3  # float32 quantise stays faithful


# -------------------- individual coders -------------------- #
def test_symbolic_coder_recovers_closed_form():
    coder = SymbolicCoder()
    arr = np.arange(64) * 3.0 - 7.0  # exact arithmetic progression
    code = coder.try_encode(arr, tol=1e-6)
    assert code is not None
    assert code.measured_error < 1e-6
    assert "f(i)" in code.equation           # a real, human-readable equation
    assert _rel_err(arr, coder.decode(code)) < 1e-6


def test_runlength_coder_lossless_on_blocky():
    coder = RunLengthCoder()
    arr = np.array([1.0] * 40 + [5.0] * 40 + [0.0] * 40)
    code = coder.try_encode(arr, tol=1e-9)
    assert code is not None
    assert _rel_err(arr, coder.decode(code)) < 1e-9
    assert code.ratio > 3.0


def test_polynomial_and_spectral_coders_apply():
    t = np.linspace(-1, 1, 128)
    poly = PolynomialCoder().try_encode(t ** 2, tol=1e-4)
    assert poly is not None and poly.measured_error < 1e-3
    spec = SpectralCoder().try_encode(np.sin(np.linspace(0, 6 * np.pi, 128)), tol=0.02)
    assert spec is not None and spec.measured_error <= 0.02


# -------------------- the literal Mandelbrot generator -------------------- #
def test_mandelbrot_field_is_lossless_and_deterministic():
    coder = MandelbrotIFSCoder()
    field = MandelbrotIFSCoder._render_field("mandelbrot", 0j, 48, 48, iters=64)
    code = coder.try_encode(field, tol=1e-6)
    assert code is not None
    assert code.measured_error < 1e-6         # the map reproduces its own field exactly
    assert code.ratio > 20.0                  # a 48x48 field from a handful of numbers
    # byte-identical reconstruction across independent runs (no randomness, no LLM)
    r1 = coder.decode(code)
    r2 = coder.decode(EquationCode.from_dict(code.to_dict()))
    assert np.array_equal(np.asarray(r1, dtype="float32"), np.asarray(r2, dtype="float32"))


def test_engine_is_deterministic():
    a = EquationMemory().compress(np.linspace(-2, 2, 256) ** 3).to_dict()
    b = EquationMemory().compress(np.linspace(-2, 2, 256) ** 3).to_dict()
    assert a == b                              # same input -> identical code, every time


def test_code_dict_roundtrip():
    em = EquationMemory()
    code = em.compress(np.arange(128) * 2.0)
    restored = EquationCode.from_dict(code.to_dict())
    assert np.allclose(em.decompress(code), em.decompress(restored))


# -------------------- MemoryStore integration -------------------- #
def test_store_persists_embeddings_as_equations_and_recall_is_faithful():
    ms = MemoryStore()
    facts = [
        ("The owner is Jaypal Khoja, called JP.", MemoryType.SEMANTIC, ["owner"]),
        ("Suspicious SSH login on port 22 at 3am.", MemoryType.EPISODIC, ["security"]),
        ("To harden SSH: disable root, use keys, fail2ban.", MemoryType.PROCEDURAL, ["howto"]),
    ]
    for i in range(30):
        for text, mt, tags in facts:
            ms.remember(f"{text} #{i}", mem_type=mt, tags=tags)

    d = tempfile.mkdtemp()
    p_on = os.path.join(d, "compressed.json")
    p_off = os.path.join(d, "raw.json")
    ms.save(p_on)
    # engine stats are reported while compression is on (real bytes moved)
    assert "equation_memory" in ms.stats()
    ms.compress_embeddings = False
    ms.save(p_off)

    # real, measured on-disk shrink
    assert os.path.getsize(p_on) < os.path.getsize(p_off)

    # faithfulness: loading the compressed file yields the SAME recall as the raw file
    a = MemoryStore(); a.load(p_on)
    b = MemoryStore(); b.load(p_off)
    query = "who is my master?"
    ra = [h[0].mem_id for h in a.recall(query, k=5)]
    rb = [h[0].mem_id for h in b.recall(query, k=5)]
    assert ra == rb            # equation-coded embeddings recall identically to raw ones


def test_store_disabled_compression_still_round_trips():
    ms = MemoryStore(compress_embeddings=False)
    ms.remember("hello world", mem_type=MemoryType.SEMANTIC)
    d = tempfile.mkdtemp()
    p = os.path.join(d, "m.json")
    ms.save(p)
    ms2 = MemoryStore()
    assert ms2.load(p) == 1
    assert ms2.recall("hello", k=1)
