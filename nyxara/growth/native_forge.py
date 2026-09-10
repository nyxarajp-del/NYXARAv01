"""NYXARA · growth/native_forge.py — Autopoietic Genome Compiling (Part C2).

NYXARA turns a hot, pure Python function into a native **C** or **Rust** kernel, compiles it to a C-ABI
shared library, and adopts it **only** when it is proven behaviorally identical AND measurably faster —
reversibly. Over time her own kernel gets faster without her architecture ever being static.

Scope + safety (honest, and enforced):
* bounded to **narrow, pure** functions (numeric/string, no side-effects) — never kernel internals or the
  constitutional core;
* both backends compile to a plain C-ABI shared object (`extern "C"` / Rust `#[no_mangle] cdylib`), so **no
  PyO3/maturin** — only ``gcc``/``clang`` and ``rustc`` on PATH; absent → :meth:`available` is False and the
  forge is a clean no-op;
* adoption passes an **equivalence + speedup gauntlet** (identical output on every sample AND
  ``speedup ≥ min_speedup``), producing a re-checkable certificate — the same "provably better" discipline
  the existing ``Optimizer`` requires;
* the **in-process** ctypes load (the one thing that opens the containment wall) is OFF by default and
  gated by ``genome.allow_inprocess_native`` + the ``Capability.NATIVE_COMPILE`` grant. The compile+verify
  runs regardless; only the live load is gated. Reversible: the Python source is never touched, so any
  restart or a ``PolymorphRuntime`` rollback restores pure-Python behavior.
"""
from __future__ import annotations

import ctypes
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, List, Optional, Sequence, Tuple

__all__ = ["NativeCandidate", "NativeForge"]


@dataclass
class NativeCandidate:
    """A compiled, verified native kernel ready to (optionally) hot-swap in."""

    language: str
    symbol: str
    so_path: str
    fn: Callable[..., Any]
    speedup: float
    cases: int
    certificate: str
    _lib: Any = None            # keep the CDLL alive so `fn` stays valid

    def to_dict(self) -> dict:
        return {"language": self.language, "symbol": self.symbol, "speedup": round(self.speedup, 2),
                "cases": self.cases, "certificate": self.certificate}


class NativeForge:
    """Compile a pure Python function to native C/Rust, gauntlet it, and (optionally) hot-swap it."""

    def __init__(self, *, min_speedup: Optional[float] = None,
                 allow_inprocess: Optional[bool] = None) -> None:
        cfg = self._cfg()
        self.min_speedup = float(min_speedup if min_speedup is not None
                                 else getattr(cfg, "min_speedup", 1.2))
        self.allow_inprocess = bool(allow_inprocess if allow_inprocess is not None
                                    else getattr(cfg, "allow_inprocess_native", False))
        #: Why the last :meth:`forge` returned nothing, one line per language tried. Cleared at
        #: the start of every forge, so it always describes the most recent attempt.
        self.refusals: List[str] = []
        #: What the compiler said the last time it failed — its stderr, or the exception raised
        #: trying to start it at all.
        self._last_build = ""
        #: The two wall times behind the last speedup, in seconds: (pure Python, native).
        self._last_times = (0.0, 0.0)

    @staticmethod
    def _cfg() -> Any:
        try:
            from nyxara.kernel.config import get_settings
            return get_settings().genome
        except Exception:  # noqa: BLE001
            return None

    # ---- toolchain ---- #
    @staticmethod
    def available(language: str) -> bool:
        if language == "c":
            return any(shutil.which(c) for c in ("gcc", "clang", "cc"))
        if language == "rust":
            return shutil.which("rustc") is not None
        return False

    @staticmethod
    def _c_compiler() -> Optional[str]:
        for c in ("gcc", "clang", "cc"):
            if shutil.which(c):
                return c
        return None

    # ---- compilation (fail-closed) ---- #
    def _compile_c(self, source: str, workdir: Path) -> Optional[Path]:
        compiler = self._c_compiler()
        if compiler is None:
            return None
        src, so = workdir / "kernel.c", workdir / "kernel.so"
        src.write_text(source, encoding="utf-8")
        cmd = [compiler, "-O2", "-shared", "-fPIC", str(src), "-o", str(so)]
        return so if self._run(cmd, workdir) and so.exists() else None

    def _compile_rust(self, source: str, workdir: Path) -> Optional[Path]:
        if shutil.which("rustc") is None:
            return None
        src, so = workdir / "kernel.rs", workdir / "kernel.so"
        src.write_text(source, encoding="utf-8")
        cmd = ["rustc", "-O", "--crate-type", "cdylib", str(src), "-o", str(so)]
        return so if self._run(cmd, workdir) and so.exists() else None

    def _run(self, cmd: List[str], cwd: Path) -> bool:
        """Run the compiler, and keep what it said when it did not work.

        `subprocess.run` here can fail for reasons that have nothing to do with the source — the
        fork itself can fail when the parent process is large, which is what happens to this forge
        after a long test session and is invisible if the error is discarded. Both the compiler's
        own stderr and any exception raised getting to it are recorded.
        """
        try:
            r = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True, timeout=60)
            if r.returncode == 0:
                return True
            self._last_build = (r.stderr or r.stdout or "").strip()[:400] or \
                f"exit {r.returncode}"
            return False
        except Exception as error:  # noqa: BLE001 — a build failure is an honest no-op
            self._last_build = f"{type(error).__name__}: {error}"
            return False

    @staticmethod
    def _load(so: Path, symbol: str, argtypes: Sequence[Any], restype: Any) -> Tuple[Any, Callable]:
        lib = ctypes.CDLL(str(so))
        fn = getattr(lib, symbol)
        fn.argtypes = list(argtypes)
        fn.restype = restype
        return lib, fn

    # ---- gauntlet ---- #
    @staticmethod
    def _equivalent(reference: Callable, native: Callable, samples: Sequence[Tuple]) -> bool:
        for args in samples:
            if native(*args) != reference(*args):
                return False
        return True

    def _speedup(self, reference: Callable, native: Callable, samples: Sequence[Tuple],
                 *, repeats: int = 50, rounds: int = 5) -> float:
        """How much faster the native kernel is, taken as the **best** of several rounds each.

        One timing of each side is what this used to do, and it decides a correctness gate on a
        single sample of a quantity that only ever moves one way. Everything that can interfere —
        another process on the machine, the scheduler, a page fault, frequency scaling — makes a
        measurement *slower*; nothing makes it spuriously faster. So the minimum of several rounds
        is the honest estimate of what the code costs, and the mean is an estimate of what the
        machine was doing at the time.

        The symptom this fixes: a C kernel that measures 39x to 77x faster when the machine is
        quiet came back at **0.23x** — four times slower than Python — during a long test session,
        and the forge duly refused it. Not reproducibly; that is the point. A gate that flips on
        load is not measuring a property of the kernel.
        """
        def _time(fn: Callable) -> float:
            best = float("inf")
            for _ in range(max(1, rounds)):
                t0 = time.perf_counter()
                for _ in range(repeats):
                    for args in samples:
                        fn(*args)
                best = min(best, time.perf_counter() - t0)
            return best
        t_native = _time(native) or 1e-9
        t_ref = _time(reference)
        # Kept so a refusal can say which side was anomalous. A ratio alone cannot: "0.23x" is
        # equally consistent with a slow kernel and with a reference that was not the pure-Python
        # function it was supposed to be, and those need opposite repairs.
        self._last_times = (t_ref, t_native)
        return t_ref / t_native

    # ---- the forge ---- #
    def forge(self, reference: Callable, *, samples: Sequence[Tuple], symbol: str,
              argtypes: Sequence[Any], restype: Any, c_source: Optional[str] = None,
              rust_source: Optional[str] = None) -> Optional[NativeCandidate]:
        """Compile provided C/Rust source, verify equivalence + speedup vs ``reference``, and return the
        best certified :class:`NativeCandidate` — or ``None`` if nothing qualifies. The in-process load
        is gated by ``allow_inprocess``; without it the forge refuses (returns None), honoring the
        containment choice."""
        self.refusals: List[str] = []
        if not self.allow_inprocess:
            self._refuse("*", "in-process loading is gated off")
            return None                       # in-process ctypes load is the gated tier (default OFF)
        best: Optional[NativeCandidate] = None
        for language, source in (("c", c_source), ("rust", rust_source)):
            if not source:
                continue
            if not self.available(language):
                self._refuse(language, "no toolchain on PATH")
                continue
            cand = self._forge_one(language, source, reference, samples, symbol, argtypes, restype)
            if cand is not None and (best is None or cand.speedup > best.speedup):
                best = cand
        return best

    def _refuse(self, language: str, why: str) -> None:
        """Record why a kernel was not forged.

        Refusing is right — every branch below stays pure Python rather than risk a wrong answer
        fast — but refusing **silently** is not. Every failure used to collapse into one `None`:
        a missing compiler, a compile error, a kernel that computed the wrong thing and a kernel
        that was merely slow were indistinguishable to the caller, and a test asserting
        `cand is not None` could only report that nothing came back. The fail-closed behaviour is
        unchanged; what is added is the reason.
        """
        self.refusals.append(f"{language}: {why}")

    def _forge_one(self, language: str, source: str, reference: Callable,
                   samples: Sequence[Tuple], symbol: str, argtypes: Sequence[Any],
                   restype: Any) -> Optional[NativeCandidate]:
        try:
            workdir = Path(tempfile.mkdtemp(prefix=f"nyxara-forge-{language}-"))
            so = self._compile_c(source, workdir) if language == "c" \
                else self._compile_rust(source, workdir)
            if so is None:
                self._refuse(language, f"did not compile: {self._last_build or 'no detail'}")
                return None
            lib, fn = self._load(so, symbol, argtypes, restype)
            if not self._equivalent(reference, fn, samples):
                self._refuse(language, f"not identical to the reference on {len(samples)} cases")
                return None                   # NOT behaviorally identical → reject (fail-closed)
            speedup = self._speedup(reference, fn, samples)
            if speedup < self.min_speedup:
                t_ref, t_native = self._last_times
                self._refuse(language,
                             f"{speedup:.2f}x, under the {self.min_speedup}x bar "
                             f"(python {t_ref * 1e3:.1f}ms, native {t_native * 1e3:.1f}ms, "
                             f"reference {getattr(reference, '__qualname__', reference)!r})")
                return None                   # not measurably faster → keep pure Python
            cert = (f"{language}: identical on {len(samples)} cases; "
                    f"{speedup:.1f}x faster (>= {self.min_speedup}x)")
            return NativeCandidate(language=language, symbol=symbol, so_path=str(so), fn=fn,
                                   speedup=speedup, cases=len(samples), certificate=cert, _lib=lib)
        except Exception as error:  # noqa: BLE001 — any failure means: stay pure Python
            self._refuse(language, f"{type(error).__name__}: {error}")
            return None

    # ---- reversible hot-swap ---- #
    @staticmethod
    def hot_swap(candidate: NativeCandidate, *, cell: Any = None) -> Callable:
        """Adopt the native kernel. If a PolymorphRuntime cell is given, swap it (reversible); else just
        return the native callable. The Python source on disk is never touched, so a restart/rollback
        always restores pure-Python behavior."""
        if cell is not None:
            try:
                from nyxara.causal.polymorph_runtime import PolymorphRuntime  # noqa: F401
                if hasattr(cell, "hot_swap"):
                    cell.hot_swap(candidate.fn)
            except Exception:  # noqa: BLE001 — swap is best-effort; the callable is still returned
                pass
        return candidate.fn
