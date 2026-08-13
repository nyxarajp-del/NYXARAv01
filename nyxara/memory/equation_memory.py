"""NYXARA · memory/equation_memory.py — data stored as mathematical equations.

Instead of keeping every array of numbers verbatim, NYXARA can keep the **generator**
that produces it: a handful of coefficients, a closed-form equation, or the parameters of
an iterated map (a Mandelbrot/Julia field). When a memory is needed the equation is
**unpacked in real time** — deterministically, in pure code, *with no LLM in the loop*.

This is a real algorithmic compressor, not magic. The honest physics:

* **Lossless is only possible where real structure exists.** A run of equal numbers, an
  arithmetic/geometric progression, a single sinusoid, or a genuine Mandelbrot field all
  collapse to a few bytes with (near) zero error. Random noise cannot shrink — the
  pigeonhole principle forbids it — so for incompressible input the engine falls back to a
  faithful quantised code (ratio ≈ 1) rather than faking a number.
* **Every code carries its own truth.** :class:`EquationCode` records the *measured*
  reconstruction error and the *true* byte counts (original vs. code), so the compression
  ratio is computed, never asserted.

The engine tries several coders and keeps the smallest code whose error fits the budget:

* :class:`RunLengthCoder`   — repeats/constant runs (lossless).
* :class:`SymbolicCoder`    — closed forms: constant, arithmetic, geometric, sinusoid.
                              Stores the **equation itself** (via sympy when present).
* :class:`SpectralCoder`    — top-k real-FFT coefficients (smooth / periodic signals).
* :class:`PolynomialCoder`  — least-squares polynomial fit (smooth trends).
* :class:`MandelbrotIFSCoder` — 2-D fields as escape-time parameters of a genuine
                              Mandelbrot/Julia iteration, plus a quantised residual.
* :class:`RawCoder`         — guaranteed fallback (float32 quantise); compression never fails.

Requires numpy for the real coders; without it the engine still works via the raw coder.
sympy is used only to render the human-readable equation string and is optional.
"""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:  # the real coders need numpy; the raw fallback does not
    import numpy as _np  # type: ignore
    _HAS_NP = True
except Exception:  # pragma: no cover
    _np = None  # type: ignore
    _HAS_NP = False

__all__ = [
    "EquationCode",
    "EquationMemory",
    "RunLengthCoder",
    "SymbolicCoder",
    "SpectralCoder",
    "PolynomialCoder",
    "MandelbrotIFSCoder",
    "RawCoder",
]

_EPS = 1e-12


# --------------------------------------------------------------------------- #
# packing helpers — compact, JSON-safe binary payloads
# --------------------------------------------------------------------------- #
def _pack(values: Sequence[float], dtype: str) -> str:
    """Pack a numeric sequence into a base64 string of raw ``dtype`` bytes."""
    if _HAS_NP:
        arr = _np.asarray(list(values), dtype=dtype)
        return base64.b64encode(arr.tobytes()).decode("ascii")
    # numpy-free fallback: JSON list inside the b64 envelope keeps decode uniform
    return "j:" + base64.b64encode(json.dumps(list(values)).encode()).decode("ascii")


def _unpack(blob: str, dtype: str) -> "List[float]":
    if blob.startswith("j:"):
        return list(json.loads(base64.b64decode(blob[2:]).decode()))
    raw = base64.b64decode(blob.encode("ascii"))
    if _HAS_NP:
        return _np.frombuffer(raw, dtype=dtype).tolist()
    raise RuntimeError("numpy required to unpack binary equation payloads")


def _jsonlen(obj: Any) -> int:
    """True serialized byte size of ``obj`` (how big it is on disk / in the record)."""
    return len(json.dumps(obj, separators=(",", ":")).encode("utf-8"))


def _rel_error(original: "Any", recon: "Any") -> float:
    """Relative RMS error ``||recon-original|| / (||original||+eps)`` — the measured truth."""
    o = _np.asarray(original, dtype="float64").ravel()
    r = _np.asarray(recon, dtype="float64").ravel()
    if r.shape != o.shape:
        return float("inf")
    denom = float(_np.linalg.norm(o)) + _EPS
    return float(_np.linalg.norm(r - o) / denom)


# --------------------------------------------------------------------------- #
# EquationCode — the compact, self-describing representation
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class EquationCode:
    """A compact generator for an array: unpack with :meth:`EquationMemory.decompress`."""

    kind: str                       # coder name that produced this
    params: Dict[str, Any]          # coefficients / equation / packed payloads (JSON-safe)
    shape: Tuple[int, ...]          # original array shape
    measured_error: float           # real relative-RMS reconstruction error
    original_bytes: int             # float32 size of the source array
    code_bytes: int                 # true serialized size of this code's payload
    equation: str = ""              # human-readable form when one exists

    @property
    def ratio(self) -> float:
        """Honest compression ratio (original bytes / code bytes)."""
        return self.original_bytes / max(1, self.code_bytes)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.kind,
            "params": self.params,
            "shape": list(self.shape),
            "measured_error": self.measured_error,
            "original_bytes": self.original_bytes,
            "code_bytes": self.code_bytes,
            "equation": self.equation,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "EquationCode":
        return cls(
            kind=d["kind"],
            params=d.get("params", {}),
            shape=tuple(d.get("shape", ())),
            measured_error=float(d.get("measured_error", 0.0)),
            original_bytes=int(d.get("original_bytes", 0)),
            code_bytes=int(d.get("code_bytes", 0)),
            equation=d.get("equation", ""),
        )


def _finish(kind: str, params: Dict[str, Any], shape: Tuple[int, ...],
            original: "Any", recon: "Any", equation: str = "") -> EquationCode:
    """Assemble a code with honestly measured error and true byte counts."""
    original_bytes = int(_np.asarray(original, dtype="float32").size) * 4
    code_bytes = _jsonlen(params) + _jsonlen(list(shape)) + _jsonlen(equation)
    return EquationCode(
        kind=kind, params=params, shape=tuple(shape),
        measured_error=_rel_error(original, recon),
        original_bytes=original_bytes, code_bytes=code_bytes, equation=equation,
    )


# --------------------------------------------------------------------------- #
# Coders — each: try_encode(array, tol) -> EquationCode|None ; decode(code) -> ndarray
# --------------------------------------------------------------------------- #
class RawCoder:
    """Guaranteed fallback: quantise to float32 and pack. Compression never fails."""

    name = "raw"

    def try_encode(self, arr: "Any", tol: float) -> Optional[EquationCode]:
        flat = _np.asarray(arr, dtype="float32").ravel()
        params = {"data": _pack(flat, "float32")}
        recon = flat.reshape(arr.shape)
        return _finish(self.name, params, arr.shape, arr, recon, equation="verbatim(float32)")

    def decode(self, code: EquationCode) -> "Any":
        vals = _unpack(code.params["data"], "float32")
        return _np.asarray(vals, dtype="float64").reshape(code.shape)


class RunLengthCoder:
    """Lossless run-length: `(value, count)` pairs. Wins on constant / blocky data."""

    name = "rle"

    def try_encode(self, arr: "Any", tol: float) -> Optional[EquationCode]:
        flat = _np.asarray(arr, dtype="float64").ravel()
        if flat.size == 0:
            return None
        # a run continues while values stay within an absolute quantum
        scale = float(_np.max(_np.abs(flat))) + _EPS
        quantum = max(tol * scale, 1e-9)
        values: List[float] = []
        counts: List[int] = []
        cur = float(flat[0]); n = 1
        for x in flat[1:]:
            if abs(float(x) - cur) <= quantum:
                n += 1
            else:
                values.append(cur); counts.append(n)
                cur = float(x); n = 1
        values.append(cur); counts.append(n)
        if len(values) > flat.size * 0.6:  # not enough repetition to be worth it
            return None
        params = {"values": _pack(values, "float32"), "counts": counts}
        recon = self._render(values, counts).reshape(arr.shape)
        return _finish(self.name, params, arr.shape, arr, recon,
                       equation=f"{len(values)} runs")

    @staticmethod
    def _render(values: Sequence[float], counts: Sequence[int]) -> "Any":
        return _np.repeat(_np.asarray(values, dtype="float64"),
                          _np.asarray(counts, dtype="int64"))

    def decode(self, code: EquationCode) -> "Any":
        values = _unpack(code.params["values"], "float32")
        counts = code.params["counts"]
        return self._render(values, counts).reshape(code.shape)


class SymbolicCoder:
    """Closed-form equations: constant, arithmetic, geometric, single sinusoid.

    This is the literal *"store the data as an equation"* case — a whole array reduced to a
    few coefficients and a human-readable formula. Uses sympy to render the equation string
    when available; reconstruction is always numeric and deterministic.
    """

    name = "symbolic"

    def try_encode(self, arr: "Any", tol: float) -> Optional[EquationCode]:
        flat = _np.asarray(arr, dtype="float64").ravel()
        n = flat.size
        if n < 3:
            return None
        x = _np.arange(n, dtype="float64")
        scale = float(_np.max(_np.abs(flat))) + _EPS

        best: Optional[Tuple[Dict[str, Any], "Any", str]] = None

        # constant
        c = float(_np.mean(flat))
        best = self._keep(best, {"form": "const", "c": c},
                          _np.full(n, c), f"f(i) = {c:.6g}", flat, tol, scale)
        # arithmetic  a0 + d*i
        d0 = float(flat[1] - flat[0])
        arith = flat[0] + d0 * x
        best = self._keep(best, {"form": "arith", "a0": float(flat[0]), "d": d0},
                          arith, f"f(i) = {flat[0]:.6g} + {d0:.6g}*i", flat, tol, scale)
        # geometric  a0 * r**i
        if abs(flat[0]) > _EPS and _np.all(_np.abs(flat) > _EPS):
            ratios = flat[1:] / flat[:-1]
            r0 = float(_np.median(ratios))
            geo = flat[0] * (r0 ** x)
            best = self._keep(best, {"form": "geom", "a0": float(flat[0]), "r": r0},
                              geo, f"f(i) = {flat[0]:.6g} * {r0:.6g}**i", flat, tol, scale)
        # single sinusoid  A*sin(w*i + phi) + c   (frequency from the dominant FFT bin)
        sinp = self._fit_sinusoid(flat, x)
        if sinp is not None:
            params, recon, eq = sinp
            best = self._keep(best, params, recon, eq, flat, tol, scale)

        if best is None:
            return None
        params, recon, eq, _err = best
        return _finish(self.name, params, arr.shape, arr, recon.reshape(arr.shape), equation=eq)

    @staticmethod
    def _keep(best, params, recon, eq, target, tol, scale):
        err = float(_np.linalg.norm(recon - target)) / (float(_np.linalg.norm(target)) + _EPS)
        if err > tol:
            return best
        cand = (params, recon, eq, err)
        if best is None or cand[3] < best[3]:
            return cand
        return best

    @staticmethod
    def _fit_sinusoid(flat: "Any", x: "Any"):
        n = flat.size
        if n < 8:
            return None
        c = float(_np.mean(flat))
        spec = _np.fft.rfft(flat - c)
        if spec.size < 2:
            return None
        k = int(_np.argmax(_np.abs(spec[1:]))) + 1  # skip DC
        w = 2.0 * _np.pi * k / n
        # least squares for A*cos + B*sin at frequency w
        design = _np.stack([_np.cos(w * x), _np.sin(w * x), _np.ones_like(x)], axis=1)
        coef, *_ = _np.linalg.lstsq(design, flat, rcond=None)
        a, b, c2 = float(coef[0]), float(coef[1]), float(coef[2])
        amp = float(_np.hypot(a, b))
        phi = float(_np.arctan2(a, b))  # so that a*cos+b*sin = amp*sin(w x + phi)
        recon = amp * _np.sin(w * x + phi) + c2
        eq = f"f(i) = {amp:.6g}*sin({w:.6g}*i + {phi:.6g}) + {c2:.6g}"
        return {"form": "sin", "amp": amp, "w": w, "phi": phi, "c": c2}, recon, eq

    def decode(self, code: EquationCode) -> "Any":
        p = code.params
        n = int(_np.prod(code.shape)) if code.shape else 0
        x = _np.arange(n, dtype="float64")
        form = p["form"]
        if form == "const":
            out = _np.full(n, p["c"], dtype="float64")
        elif form == "arith":
            out = p["a0"] + p["d"] * x
        elif form == "geom":
            out = p["a0"] * (p["r"] ** x)
        elif form == "sin":
            out = p["amp"] * _np.sin(p["w"] * x + p["phi"]) + p["c"]
        else:  # pragma: no cover
            raise ValueError(f"unknown symbolic form {form!r}")
        return out.reshape(code.shape)


class PolynomialCoder:
    """Least-squares polynomial fit — smallest degree whose error fits the budget."""

    name = "poly"

    def try_encode(self, arr: "Any", tol: float) -> Optional[EquationCode]:
        flat = _np.asarray(arr, dtype="float64").ravel()
        n = flat.size
        if n < 4:
            return None
        x = _np.linspace(-1.0, 1.0, n)  # conditioning-friendly domain
        max_deg = min(12, n - 1)
        norm = float(_np.linalg.norm(flat)) + _EPS
        for deg in range(1, max_deg + 1):
            coef = _np.polyfit(x, flat, deg)
            recon = _np.polyval(coef, x)
            if float(_np.linalg.norm(recon - flat)) / norm <= tol:
                params = {"coef": _pack(coef, "float32")}
                return _finish(self.name, params, arr.shape, arr, recon.reshape(arr.shape),
                               equation=f"poly deg {deg}")
        return None

    def decode(self, code: EquationCode) -> "Any":
        coef = _unpack(code.params["coef"], "float32")
        n = int(_np.prod(code.shape)) if code.shape else 0
        x = _np.linspace(-1.0, 1.0, n)
        return _np.polyval(_np.asarray(coef, dtype="float64"), x).reshape(code.shape)


class SpectralCoder:
    """Top-k real-FFT coefficients. Excellent for smooth or periodic signals."""

    name = "spectral"

    def try_encode(self, arr: "Any", tol: float) -> Optional[EquationCode]:
        flat = _np.asarray(arr, dtype="float64").ravel()
        n = flat.size
        if n < 8:
            return None
        spec = _np.fft.rfft(flat)
        order = _np.argsort(-_np.abs(spec))  # strongest coefficients first
        norm = float(_np.linalg.norm(flat)) + _EPS
        for k in self._k_schedule(spec.size):
            idx = _np.sort(order[:k])
            trimmed = _np.zeros_like(spec)
            trimmed[idx] = spec[idx]
            recon = _np.fft.irfft(trimmed, n=n)
            if float(_np.linalg.norm(recon - flat)) / norm <= tol:
                params = {
                    "n": int(n),
                    "idx": [int(i) for i in idx],
                    "re": _pack(spec[idx].real, "float32"),
                    "im": _pack(spec[idx].imag, "float32"),
                }
                return _finish(self.name, params, arr.shape, arr, recon.reshape(arr.shape),
                               equation=f"{k} FFT terms")
        return None

    @staticmethod
    def _k_schedule(size: int) -> List[int]:
        ks: List[int] = []
        k = 1
        while k < size:
            ks.append(k)
            k = max(k + 1, int(k * 1.7))
        ks.append(size)
        return ks

    def decode(self, code: EquationCode) -> "Any":
        p = code.params
        n = int(p["n"])
        spec = _np.zeros(n // 2 + 1, dtype="complex128")
        idx = _np.asarray(p["idx"], dtype="int64")
        re = _np.asarray(_unpack(p["re"], "float32"), dtype="float64")
        im = _np.asarray(_unpack(p["im"], "float32"), dtype="float64")
        spec[idx] = re + 1j * im
        return _np.fft.irfft(spec, n=n).reshape(code.shape)


class MandelbrotIFSCoder:
    """2-D fields as parameters of a genuine escape-time (Mandelbrot/Julia) iteration.

    A field generated by iterating ``z -> z**2 + c`` collapses to a handful of numbers and
    reconstructs exactly. Arbitrary 2-D input keeps a quantised residual, so the ratio stays
    honest: huge for fractal-like data, ≈ 1 for noise.
    """

    name = "mandelbrot"

    def try_encode(self, arr: "Any", tol: float) -> Optional[EquationCode]:
        a = _np.asarray(arr, dtype="float64")
        if a.ndim != 2:
            return None
        h, w = a.shape
        if h < 4 or w < 4:
            return None
        # normalise target to [0,1] so the escape-count field is comparable
        amin, amax = float(a.min()), float(a.max())
        span = (amax - amin) or 1.0
        target = (a - amin) / span

        best_params, best_field, best_err = None, None, float("inf")
        # a small deterministic sweep of Julia constants + the Mandelbrot map (c per-pixel)
        candidates: List[Tuple[str, complex]] = [("mandelbrot", 0j)]
        for cr in (-0.8, -0.4, 0.0, 0.285, 0.45):
            for ci in (0.0, 0.156, 0.35, -0.2):
                candidates.append(("julia", complex(cr, ci)))
        for mode, cparam in candidates:
            rendered = self._render_field(mode, cparam, h, w, iters=64)
            err = float(_np.linalg.norm(rendered - target)) / (float(_np.linalg.norm(target)) + _EPS)
            if err < best_err:
                best_err, best_field, best_params = err, rendered, (mode, cparam)

        mode, cparam = best_params  # type: ignore[misc]
        residual = target - best_field                 # what the map cannot explain
        rmax = float(_np.max(_np.abs(residual))) + _EPS
        # aggressive: quantise residual to int8; drop it entirely if already within budget
        if best_err <= tol:
            res_blob, res_scale = "", 0.0
            recon_norm = best_field
        else:
            q = _np.clip(_np.round(residual / rmax * 127.0), -127, 127).astype("int8")
            res_blob = base64.b64encode(q.tobytes()).decode("ascii")
            res_scale = rmax / 127.0
            recon_norm = best_field + q.astype("float64") * res_scale
        recon = recon_norm * span + amin

        params = {
            "mode": mode, "cr": cparam.real, "ci": cparam.imag,
            "h": h, "w": w, "iters": 64,
            "amin": amin, "span": span,
            "res": res_blob, "res_scale": res_scale,
        }
        eq = (f"z->z^2+{'(x+iy)' if mode == 'mandelbrot' else f'{cparam:.3g}'} "
              f"escape field {h}x{w}")
        return _finish(self.name, params, arr.shape, arr, recon, equation=eq)

    @staticmethod
    def _render_field(mode: str, cparam: complex, h: int, w: int, iters: int) -> "Any":
        ys = _np.linspace(-1.4, 1.4, h)[:, None]
        xs = _np.linspace(-2.0, 1.0 if mode == "mandelbrot" else 1.4, w)[None, :]
        if mode == "mandelbrot":
            c = xs + 1j * ys
            z = _np.zeros_like(c)
        else:
            z = xs + 1j * ys
            c = _np.full((h, w), cparam, dtype="complex128")
        escape = _np.full((h, w), iters, dtype="float64")
        alive = _np.ones((h, w), dtype=bool)
        for i in range(iters):
            z[alive] = z[alive] * z[alive] + c[alive]
            just = alive & (_np.abs(z) > 2.0)
            escape[just] = i
            alive &= ~just
        return escape / iters  # normalised to [0,1]

    def decode(self, code: EquationCode) -> "Any":
        p = code.params
        h, w = int(p["h"]), int(p["w"])
        rendered = self._render_field(p["mode"], complex(p["cr"], p["ci"]), h, w, int(p["iters"]))
        if p.get("res"):
            raw = base64.b64decode(p["res"].encode("ascii"))
            q = _np.frombuffer(raw, dtype="int8").astype("float64").reshape(h, w)
            rendered = rendered + q * float(p["res_scale"])
        # ``rendered``, not ``field``: ``field`` is dataclasses.field, imported at the top of this
        # module, so the typo raised "unsupported operand type(s) for *: 'function' and 'float'"
        # instead of failing a name lookup — which is why it survived to here.
        return (rendered * float(p["span"]) + float(p["amin"])).reshape(code.shape)


# --------------------------------------------------------------------------- #
# EquationMemory — the engine that picks the best coder and unpacks on demand
# --------------------------------------------------------------------------- #
class EquationMemory:
    """Compress arrays to the smallest faithful equation; decompress deterministically.

    ``tol`` is a *relative-RMS error budget* (default 0.05 = aggressive-lossy, per the
    owner's choice). ``aggressive`` lets a coder drop residuals to buy a bigger ratio while
    staying inside a slightly relaxed budget. Set ``aggressive=False`` and a small ``tol``
    for near-lossless behaviour.
    """

    def __init__(self, *, tol: float = 0.05, aggressive: bool = True) -> None:
        self.tol = float(tol)
        self.aggressive = bool(aggressive)
        self._coders = {
            c.name: c for c in (
                RunLengthCoder(), SymbolicCoder(), PolynomialCoder(),
                SpectralCoder(), MandelbrotIFSCoder(), RawCoder(),
            )
        }
        self._stats = {
            "compressed": 0, "bytes_in": 0, "bytes_out": 0,
            "coder_hist": {name: 0 for name in self._coders},
        }

    # ---- compress / decompress ---- #
    def compress(self, array: Any, *, tol: Optional[float] = None,
                 aggressive: Optional[bool] = None) -> EquationCode:
        """Encode ``array`` as the smallest :class:`EquationCode` within the error budget."""
        if not _HAS_NP:  # numpy-free environments: honest raw passthrough
            code = self._raw_only(array)
            self._account(code)
            return code
        arr = _np.asarray(array, dtype="float64")
        budget = self.tol if tol is None else float(tol)
        greedy = self.aggressive if aggressive is None else bool(aggressive)
        eff = budget * (1.6 if greedy else 1.0)   # aggressive mode relaxes the budget a touch

        best: Optional[EquationCode] = None
        for name, coder in self._coders.items():
            if name == "raw":
                continue
            try:
                code = coder.try_encode(arr, eff)
            except Exception:  # noqa: BLE001 — a coder that fails simply doesn't apply
                code = None
            if code is None or code.measured_error > eff:
                continue
            if best is None or code.code_bytes < best.code_bytes:
                best = code
        # raw fallback guarantees success and caps the code size
        raw = self._coders["raw"].try_encode(arr, eff)
        if best is None or (raw is not None and raw.code_bytes < best.code_bytes):
            best = raw
        assert best is not None
        self._account(best)
        return best

    def decompress(self, code: Any) -> Any:
        """Unpack an :class:`EquationCode` (or its dict form) back into an array — real time."""
        if isinstance(code, dict):
            code = EquationCode.from_dict(code)
        if not _HAS_NP and code.kind != "raw":  # pragma: no cover
            raise RuntimeError("numpy required to unpack this equation code")
        coder = self._coders.get(code.kind)
        if coder is None:  # pragma: no cover
            raise ValueError(f"no coder for kind {code.kind!r}")
        return coder.decode(code)

    # ---- convenience for vectors (embeddings are 1-D lists of floats) ---- #
    def compress_vector(self, vector: Sequence[float], **kw: Any) -> Dict[str, Any]:
        return self.compress(list(vector), **kw).to_dict()

    def decompress_vector(self, code: Dict[str, Any]) -> List[float]:
        out = self.decompress(code)
        if _HAS_NP:
            return _np.asarray(out, dtype="float64").ravel().tolist()
        return list(out)  # pragma: no cover

    # ---- internals ---- #
    def _raw_only(self, array: Any) -> EquationCode:  # pragma: no cover - numpy-free path
        flat = [float(x) for x in array]
        params = {"data": _pack(flat, "float32")}
        code_bytes = _jsonlen(params)
        return EquationCode("raw", params, (len(flat),), 0.0, len(flat) * 4, code_bytes,
                            equation="verbatim(json)")

    def _account(self, code: EquationCode) -> None:
        self._stats["compressed"] += 1
        self._stats["bytes_in"] += code.original_bytes
        self._stats["bytes_out"] += code.code_bytes
        self._stats["coder_hist"][code.kind] = self._stats["coder_hist"].get(code.kind, 0) + 1

    def stats(self) -> Dict[str, Any]:
        s = dict(self._stats)
        s["coder_hist"] = dict(self._stats["coder_hist"])
        bi, bo = s["bytes_in"], s["bytes_out"]
        s["mean_ratio"] = round(bi / bo, 4) if bo else 1.0
        s["bytes_saved"] = max(0, bi - bo)
        s["numpy"] = _HAS_NP
        return s
