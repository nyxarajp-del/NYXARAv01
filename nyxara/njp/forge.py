"""NYXARA · njp/forge.py — she reads her own source and rewrites her slow parts in C (⚙, NJP V.01).

Every other faculty in NYX V.01 changes what she *knows*. This one changes what she **is made
of**. She opens her own source with :mod:`ast`, finds the narrow numeric functions where a
Python loop is the whole cost, lowers them to C, compiles them, proves the compiled kernel
returns identical answers and is measurably faster, and — only then — swaps it into the running
process. If anything fails, nothing changes.

The pipeline, and every stage of it is real:

1. **measure** — :meth:`observe` takes her own per-faculty latency from ``NJPBrain.think``. The
   module she actually spends time in is the module she looks at. Nothing is hand-picked.
2. **read herself** — :meth:`scan` parses that module's source and returns the functions it can
   lower: fully ``int``/``float`` annotated, no calls but ``abs``/``min``/``max``/``int``/
   ``float``/``range``, no attribute access, no closures. The narrowness is the point.
3. **lower** — :class:`_CLowerer` emits C for that subset, with Python's *own* semantics where C
   differs: floor division and modulo round toward −∞ (C truncates), so both are emitted as
   helpers rather than as ``/`` and ``%``.
4. **gauntlet** — :mod:`nyxara.growth.native_forge` compiles it and refuses to return anything
   that is not byte-identical on every sample **and** at least ``min_speedup`` faster.
5. **adopt** — :mod:`nyxara.njp.autopoiesis` applies the swap, re-runs the check on a *fresh*
   sample set, and rolls back if it does not hold. The outcome goes to a ledger either way.

Honest about all of it:

* **This is empirical verification, not proof.** Equivalence is established on a sampled domain,
  and the domain is written into the certificate. Outside it — very large integers, where Python
  has no overflow and ``long long`` does — the kernel is not verified, and this is the reason the
  candidate set is kept to narrow numeric functions instead of being widened.
* **Nothing on disk is ever written.** The swap is a module attribute rebound in memory. Any
  restart, or :meth:`rollback_all`, restores pure Python. That is why it is safe to have on.
* **A module attribute is what changes**, so a caller that did ``from m import f`` before the
  swap keeps the Python function. The gain is real but it is not global.
* **The safety core is not a candidate.** Constitutional files (``PROTECTED_RELPATHS``) and any
  target naming loyalty / corrigibility / oversight / honesty are refused outright, before the
  compiler is ever invoked — and refused a second time by the rewriter.
* **No toolchain, no layer.** Without ``gcc``/``clang``/``cc`` or ``rustc``, :meth:`available` is
  False and every beat is a clean no-op. Nothing is faked and nothing crashes.
* "Recompiling herself every second" is not what this is. Forges are budgeted per hour, run on
  the shared beat, and stop dead when oversight pauses or scrams.
"""

from __future__ import annotations

import ast
import ctypes
import importlib
import importlib.util
import random
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

__all__ = ["Kernel", "Forged", "MetamorphicCompiler"]

# Constitutional files, mirrored from growth/self_optimize.PROTECTED_RELPATHS. Fail-closed: a
# path that cannot be resolved is treated as protected.
_PROTECTED_RELPATHS = (
    "kernel/rules.py", "kernel/invariants.py", "kernel/config.py",
    "identity/values.py", "identity/soul.py", "guard/value_learning.py",
    "guard/corrigibility.py", "guard/auth.py", "growth/loyalty.py",
)
# Names the compiler may never touch, whatever file they live in.
_PROTECTED_NAMES = frozenset({
    "loyalty", "corrigibility", "oversight", "honesty", "guardian", "shield",
    "safety_core", "value_core", "obedience", "auth", "vault", "crypto",
})

_CALLS = frozenset({"abs", "min", "max", "int", "float"})
# A ctypes call costs about a microsecond, so a kernel measured on inputs too small to cost that
# much is measuring the wrapper, not the work. The sample domain therefore *escalates* until one
# reference call is worth timing — and stops the moment it is, so a probe never runs away.
_SCALES = (96, 512, 2048, 8192, 32768, 131072, 1048576, 8388608)
_MIN_CALL_MS = 0.004

# Where she looks when her own latency has not yet named a module. These are the numeric parts of
# herself; the rest of the package is orchestration, where a C kernel has nothing to win.
DEFAULT_PACKAGES = ("nyxara.njp", "nyxara.mind", "nyxara.cognition", "nyxara.quantum",
                    "nyxara.memory", "nyxara.growth", "nyxara.sim")

_ARITH = {ast.Add: "+", ast.Sub: "-", ast.Mult: "*"}
_BITS = {ast.BitAnd: "&", ast.BitOr: "|", ast.BitXor: "^", ast.LShift: "<<", ast.RShift: ">>"}
_CMP = {ast.Lt: "<", ast.LtE: "<=", ast.Gt: ">", ast.GtE: ">=", ast.Eq: "==", ast.NotEq: "!="}

# Python's floor-division and modulo round toward −∞; C truncates toward zero. Emitting them as
# `/` and `%` would be a silently wrong kernel that the gauntlet would have to catch, so they are
# emitted correctly instead. No libm: `floor` via a cast keeps the object free of undefined
# symbols, which matters because the forge links without -lm.
_PRELUDE_INT = """
static long long nyx_fdiv(long long a, long long b) {
    long long q = a / b;
    if ((a % b != 0) && ((a < 0) != (b < 0))) q -= 1;
    return q;
}
static long long nyx_fmod(long long a, long long b) {
    long long r = a % b;
    if (r != 0 && ((r < 0) != (b < 0))) r += b;
    return r;
}
"""
_PRELUDE_FLT = """
static double nyx_floor(double x) {
    if (x >= 9.2e18 || x <= -9.2e18) return x;
    double t = (double)(long long)x;
    return (t > x) ? t - 1.0 : t;
}
static double nyx_fdiv(double a, double b) { return nyx_floor(a / b); }
static double nyx_fmod(double a, double b) { return a - b * nyx_floor(a / b); }
static double nyx_trunc(double x) {
    if (x >= 9.2e18 || x <= -9.2e18) return x;
    return (double)(long long)x;
}
"""


class _Unlowerable(Exception):
    """This function is outside the subset. Not an error — most functions are."""


@dataclass
class Kernel:
    """One function of her own that could be compiled — found by reading her source, not listed."""

    module: str
    name: str
    file: str
    lineno: int
    params: List[Tuple[str, str]] = field(default_factory=list)   # (name, "int" | "float")
    returns: str = "float"
    integral: bool = False        # every parameter and the result are integers
    loops: int = 0                # why it is worth compiling at all
    ops: int = 0
    reason: str = ""

    @property
    def target(self) -> str:
        return f"{self.module}.{self.name}"

    @property
    def score(self) -> float:
        """A loop is the reason native wins; arithmetic inside it is the size of the win."""
        return float(self.loops) * 4.0 + float(self.ops)

    def to_dict(self) -> Dict[str, Any]:
        return {"target": self.target, "file": self.file, "lineno": self.lineno,
                "params": [f"{n}: {t}" for n, t in self.params], "returns": self.returns,
                "loops": self.loops, "ops": self.ops, "score": round(self.score, 2),
                "reason": self.reason}


@dataclass
class Forged:
    """What one forge attempt actually did — including, most of the time, nothing."""

    target: str = ""
    status: str = "idle"    # adopted | rejected | refused | unavailable | no-candidate | idle
    language: str = ""
    speedup: float = 0.0
    cases: int = 0
    certificate: str = ""
    reason: str = ""
    elapsed_ms: float = 0.0

    @property
    def adopted(self) -> bool:
        return self.status == "adopted"

    def to_dict(self) -> Dict[str, Any]:
        return {"target": self.target, "status": self.status, "language": self.language,
                "speedup": round(self.speedup, 2), "cases": self.cases,
                "certificate": self.certificate, "reason": self.reason,
                "elapsed_ms": round(self.elapsed_ms, 2)}


# --------------------------------------------------------------------------- #
# Reading herself: which of her own functions are even candidates
# --------------------------------------------------------------------------- #
def _annotation(node: Any) -> Optional[str]:
    if isinstance(node, ast.Name) and node.id in ("int", "float"):
        return node.id
    return None


def _scan_source(module: str, source: str, path: str) -> List[Kernel]:
    """Parse one module of her own and return the functions that could be lowered to C."""
    out: List[Kernel] = []
    try:
        tree = ast.parse(source)
    except Exception:  # noqa: BLE001 — an unparseable file is simply not a candidate
        return out
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        kernel = _candidate(module, node, path)
        if kernel is not None:
            out.append(kernel)
    return out


def _candidate(module: str, node: ast.FunctionDef, path: str) -> Optional[Kernel]:
    a = node.args
    if node.decorator_list or a.vararg or a.kwarg or a.kwonlyargs or a.posonlyargs:
        return None
    if a.defaults or not a.args or len(a.args) > 8:
        return None
    params: List[Tuple[str, str]] = []
    for arg in a.args:
        kind = _annotation(arg.annotation)
        if kind is None or arg.arg in ("self", "cls"):
            return None
        params.append((arg.arg, kind))
    returns = _annotation(node.returns)
    if returns is None:
        return None
    integral = returns == "int" and all(t == "int" for _n, t in params)
    lowerer = _CLowerer(node, params, integral=integral)
    try:
        lowerer.lower(symbol="probe")
    except _Unlowerable:
        return None
    except Exception:  # noqa: BLE001 — a lowering bug means "not a candidate", never a crash
        return None
    loops = sum(1 for n in ast.walk(node) if isinstance(n, (ast.For, ast.While)))
    ops = sum(1 for n in ast.walk(node) if isinstance(n, (ast.BinOp, ast.UnaryOp, ast.Compare)))
    if loops == 0 and ops < 6:
        return None                      # straight-line arithmetic: ctypes overhead would win
    reason = (f"{loops} loop(s), {ops} arithmetic node(s), "
              f"{'integer' if integral else 'floating-point'} throughout")
    return Kernel(module=module, name=node.name, file=path, lineno=node.lineno,
                  params=params, returns=returns, integral=integral, loops=loops, ops=ops,
                  reason=reason)


# --------------------------------------------------------------------------- #
# Lowering: the restricted Python subset → C, with Python's semantics preserved
# --------------------------------------------------------------------------- #
class _CLowerer:
    """Emit C for one narrow numeric function, or refuse by raising :class:`_Unlowerable`."""

    def __init__(self, node: ast.FunctionDef, params: Sequence[Tuple[str, str]], *,
                 integral: bool) -> None:
        self.node = node
        self.params = list(params)
        self.integral = bool(integral)
        self.ctype = "long long" if integral else "double"
        self._locals: List[str] = []
        self._tmp = 0

    # ---- entry ---- #
    def lower(self, *, symbol: str) -> str:
        body = self._block(self.node.body, indent=2)
        args = ", ".join(f"{'long long' if t == 'int' else 'double'} {n}"
                         for n, t in self.params)
        declared = [n for n in self._locals if n not in {p for p, _ in self.params}]
        decls = "".join(f"  {self.ctype} {n} = 0;\n" for n in declared)
        prelude = _PRELUDE_INT if self.integral else _PRELUDE_FLT
        ret = "long long" if self.integral else "double"
        return (f"{prelude}\n{ret} {symbol}({args}) {{\n{decls}{body}"
                f"  return ({ret})0;\n}}\n")

    # ---- statements ---- #
    def _block(self, body: Sequence[ast.stmt], *, indent: int) -> str:
        return "".join(self._stmt(s, indent=indent) for s in body)

    def _stmt(self, s: ast.stmt, *, indent: int) -> str:
        pad = " " * indent
        if isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant):
            return ""                                        # docstring / bare literal
        if isinstance(s, ast.Pass):
            return ""
        if isinstance(s, ast.Return):
            value = self._expr(s.value) if s.value is not None else "0"
            return f"{pad}return {value};\n"
        if isinstance(s, ast.Assign):
            name = self._single_target(s.targets)
            return f"{pad}{name} = {self._expr(s.value)};\n"
        if isinstance(s, ast.AugAssign):
            name = self._single_target([s.target])
            op = type(s.op)
            if op in _ARITH:
                return f"{pad}{name} = {name} {_ARITH[op]} {self._expr(s.value)};\n"
            return f"{pad}{name} = {self._binop(op, name, self._expr(s.value))};\n"
        if isinstance(s, ast.If):
            head = f"{pad}if ({self._expr(s.test, boolean=True)}) {{\n"
            head += self._block(s.body, indent=indent + 2) + pad + "}"
            if s.orelse:
                head += " else {\n" + self._block(s.orelse, indent=indent + 2) + pad + "}"
            return head + "\n"
        if isinstance(s, ast.While):
            if s.orelse:
                raise _Unlowerable("while/else")
            return (f"{pad}while ({self._expr(s.test, boolean=True)}) {{\n"
                    + self._block(s.body, indent=indent + 2) + pad + "}\n")
        if isinstance(s, ast.For):
            return self._for(s, indent=indent)
        if isinstance(s, ast.Break):
            return f"{pad}break;\n"
        if isinstance(s, ast.Continue):
            return f"{pad}continue;\n"
        raise _Unlowerable(type(s).__name__)

    def _for(self, s: ast.For, *, indent: int) -> str:
        """Only ``for <name> in range(...)``. Anything iterable is out of the subset."""
        if s.orelse or not isinstance(s.target, ast.Name):
            raise _Unlowerable("for")
        call = s.iter
        if not (isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
                and call.func.id == "range" and not call.keywords
                and 1 <= len(call.args) <= 3):
            raise _Unlowerable("for-iterable")
        var = self._declare(s.target.id)
        if len(call.args) == 1:
            start, stop, step = "0", self._expr(call.args[0]), "1"
        elif len(call.args) == 2:
            start, stop, step = self._expr(call.args[0]), self._expr(call.args[1]), "1"
        else:
            start = self._expr(call.args[0])
            stop = self._expr(call.args[1])
            step = self._expr(call.args[2])
        self._tmp += 1
        e, d = f"_e{self._tmp}", f"_d{self._tmp}"
        pad = " " * indent
        # Bounds are hoisted so they are evaluated once, exactly as Python evaluates them once.
        return (f"{pad}{{ {self.ctype} {e} = {stop}; {self.ctype} {d} = {step};\n"
                f"{pad}  for ({var} = {start}; ({d} > 0) ? ({var} < {e}) : ({var} > {e});"
                f" {var} = {var} + {d}) {{\n"
                + self._block(s.body, indent=indent + 4) + pad + "  }\n" + pad + "}\n")

    def _single_target(self, targets: Sequence[ast.expr]) -> str:
        if len(targets) != 1 or not isinstance(targets[0], ast.Name):
            raise _Unlowerable("assignment target")
        return self._declare(targets[0].id)

    def _declare(self, name: str) -> str:
        if name not in self._locals:
            self._locals.append(name)
        return name

    # ---- expressions ---- #
    def _expr(self, e: Optional[ast.expr], *, boolean: bool = False) -> str:
        if e is None:
            raise _Unlowerable("empty expression")
        if isinstance(e, ast.Constant):
            return self._const(e.value)
        if isinstance(e, ast.Name):
            if e.id not in {p for p, _ in self.params} and e.id not in self._locals:
                raise _Unlowerable(f"free name {e.id}")
            return e.id
        if isinstance(e, ast.BinOp):
            return self._binop(type(e.op), self._expr(e.left), self._expr(e.right))
        if isinstance(e, ast.UnaryOp):
            if isinstance(e.op, ast.Not):
                if not boolean:
                    raise _Unlowerable("not outside a test")
                return f"(!({self._expr(e.operand, boolean=True)}))"
            if isinstance(e.op, ast.USub):
                return f"(-({self._expr(e.operand)}))"
            if isinstance(e.op, ast.UAdd):
                return f"(+({self._expr(e.operand)}))"
            raise _Unlowerable("unary op")
        if isinstance(e, ast.Compare):
            if len(e.ops) != 1:
                raise _Unlowerable("chained comparison")   # Python's semantics differ from C's
            op = _CMP.get(type(e.ops[0]))
            if op is None:
                raise _Unlowerable("comparison")
            return f"({self._expr(e.left)} {op} {self._expr(e.comparators[0])})"
        if isinstance(e, ast.BoolOp):
            if not boolean:
                raise _Unlowerable("and/or outside a test")  # they return operands, not booleans
            glue = "&&" if isinstance(e.op, ast.And) else "||"
            parts = [self._expr(v, boolean=True) for v in e.values]
            return "(" + f" {glue} ".join(parts) + ")"
        if isinstance(e, ast.IfExp):
            return (f"(({self._expr(e.test, boolean=True)}) ? ({self._expr(e.body)})"
                    f" : ({self._expr(e.orelse)}))")
        if isinstance(e, ast.Call):
            return self._call(e)
        raise _Unlowerable(type(e).__name__)

    def _const(self, value: Any) -> str:
        if isinstance(value, bool):
            return "1" if value else "0"
        if isinstance(value, int):
            return f"{value}LL" if self.integral else f"{float(value)!r}"
        if isinstance(value, float):
            if self.integral:
                raise _Unlowerable("float literal in an integer kernel")
            return repr(value)
        raise _Unlowerable("constant")

    def _binop(self, op: type, left: str, right: str) -> str:
        if op in _ARITH:
            return f"({left} {_ARITH[op]} {right})"
        if op is ast.FloorDiv:
            return f"nyx_fdiv({left}, {right})"
        if op is ast.Mod:
            return f"nyx_fmod({left}, {right})"
        if op is ast.Div:
            if self.integral:
                raise _Unlowerable("true division returns a float")
            return f"({left} / {right})"
        if op in _BITS:
            if not self.integral:
                raise _Unlowerable("bitwise op on floats")
            return f"({left} {_BITS[op]} {right})"
        raise _Unlowerable("binary op")

    def _call(self, e: ast.Call) -> str:
        if not isinstance(e.func, ast.Name) or e.func.id not in _CALLS or e.keywords:
            raise _Unlowerable("call")
        name, args = e.func.id, [self._expr(a) for a in e.args]
        if name == "abs" and len(args) == 1:
            return f"(({args[0]}) < 0 ? -({args[0]}) : ({args[0]}))"
        if name in ("min", "max") and len(args) == 2:
            cmp_ = "<" if name == "min" else ">"
            return f"((({args[0]}) {cmp_} ({args[1]})) ? ({args[0]}) : ({args[1]}))"
        if name == "float" and len(args) == 1:
            if self.integral:
                raise _Unlowerable("float() in an integer kernel")
            return f"((double)({args[0]}))"
        if name == "int" and len(args) == 1:
            return args[0] if self.integral else f"nyx_trunc({args[0]})"
        raise _Unlowerable(f"call {name}")


# --------------------------------------------------------------------------- #
# The compiler proper
# --------------------------------------------------------------------------- #
class MetamorphicCompiler:
    """L-OMNI: profile herself, lower the hot part to C, verify it, and swap it in — reversibly."""

    def __init__(self, brain: Any = None, *, hot_swap: bool = True, min_speedup: float = 1.2,
                 max_forges_per_hour: int = 2, cases: int = 24, scan_per_beat: int = 6,
                 packages: Sequence[str] = DEFAULT_PACKAGES, seed: int = 42) -> None:
        self.brain = brain
        self.hot_swap = bool(hot_swap)
        self.min_speedup = float(min_speedup)
        self.max_forges_per_hour = int(max_forges_per_hour)
        self.cases = max(4, int(cases))
        self.scan_per_beat = max(1, int(scan_per_beat))
        self.packages = list(packages)
        self._rng = random.Random(seed)
        self._pkg_modules: Dict[str, List[str]] = {}
        self.latency: Dict[str, float] = {}          # module → total ms she has spent there
        self.calls: Dict[str, int] = {}
        self.attempts: List[Forged] = []
        self.adopted: Dict[str, Forged] = {}
        self._rollbacks: Dict[str, Callable[[], None]] = {}
        self._tried: List[str] = []                  # targets already attempted this process
        self._forge_times: List[float] = []
        self._scanned: Dict[str, List[Kernel]] = {}

    # ---- 1. she measures herself ----------------------------------------- #
    def observe(self, module: str, ms: float) -> None:
        """Record time spent inside one of her own modules. This is what picks the target."""
        try:
            if not module or ms < 0:
                return
            self.latency[module] = self.latency.get(module, 0.0) + float(ms)
            self.calls[module] = self.calls.get(module, 0) + 1
        except Exception:  # noqa: BLE001 — measuring must never cost a turn
            pass

    def hottest(self, k: int = 3) -> List[Tuple[str, float]]:
        """Where she actually spends her time, most expensive first."""
        return sorted(self.latency.items(), key=lambda kv: kv[1], reverse=True)[:max(1, k)]

    # ---- 2. she reads herself -------------------------------------------- #
    def scan(self, module: Optional[str] = None) -> List[Kernel]:
        """Her own source, parsed: the functions in ``module`` that could become C kernels."""
        targets = [module] if module else self._search_order()
        found: List[Kernel] = []
        for name in targets:
            found.extend(self._scan_one(name))
        found.sort(key=lambda k: k.score, reverse=True)
        return found

    def _search_order(self) -> List[str]:
        """Hottest-measured first, then the configured packages. Never a hand-picked function."""
        order = [m for m, _ms in self.hottest(5)]
        for pkg in self.packages:
            order.extend(m for m in self._modules_in(pkg) if m not in order)
        return order

    def _modules_in(self, package: str) -> List[str]:
        if package in self._pkg_modules:
            return self._pkg_modules[package]
        out: List[str] = []
        try:
            spec = importlib.util.find_spec(package)
            if spec is None or not spec.submodule_search_locations:
                out = [package]
            else:
                from pathlib import Path
                for root in spec.submodule_search_locations:
                    for path in sorted(Path(root).glob("*.py")):
                        if path.stem != "__init__":
                            out.append(f"{package}.{path.stem}")
        except Exception:  # noqa: BLE001
            out = []
        self._pkg_modules[package] = out
        return out

    def _scan_one(self, module: str) -> List[Kernel]:
        if module in self._scanned:
            return self._scanned[module]
        kernels: List[Kernel] = []
        try:
            spec = importlib.util.find_spec(module)
            path = getattr(spec, "origin", None) if spec is not None else None
            if path and str(path).endswith(".py") and not self._protected_file(str(path)):
                with open(path, "r", encoding="utf-8") as fh:
                    kernels = _scan_source(module, fh.read(), str(path))
        except Exception:  # noqa: BLE001 — a module she cannot read is simply not a candidate
            kernels = []
        kernels = [k for k in kernels if not self._refuse(k.target, k.file)]
        self._scanned[module] = kernels
        return kernels

    # ---- refusals, before anything is compiled --------------------------- #
    @staticmethod
    def _protected_file(path: str) -> bool:
        try:
            if not isinstance(path, str) or not path.strip():
                return True          # not a path she can check ⇒ not a path she may compile
            posix = path.replace("\\", "/")
            return any(posix.endswith("/" + rel) for rel in _PROTECTED_RELPATHS)
        except Exception:  # noqa: BLE001 — cannot prove it is safe ⇒ it is protected
            return True

    def _refuse(self, target: str, path: str = "") -> bool:
        """The safety core is not a performance problem to be solved."""
        try:
            low = str(target).lower()
            if any(name in low for name in _PROTECTED_NAMES):
                return True
            return bool(path) and self._protected_file(path)
        except Exception:  # noqa: BLE001
            return True

    # ---- 3-5. lower, gauntlet, adopt ------------------------------------- #
    def available(self) -> bool:
        """A toolchain *and* permission to load into this process. Either missing ⇒ no-op."""
        try:
            from nyxara.growth.native_forge import NativeForge
            return bool(self.hot_swap) and (NativeForge.available("c")
                                            or NativeForge.available("rust"))
        except Exception:  # noqa: BLE001
            return False

    def forge(self, kernel: Kernel) -> Forged:
        """Lower one function to C, verify it against itself, and swap it in if it wins."""
        started = time.perf_counter()

        def _done(status: str, **kw: Any) -> Forged:
            got = Forged(target=kernel.target, status=status,
                         elapsed_ms=(time.perf_counter() - started) * 1000.0, **kw)
            self.attempts.append(got)
            if kernel.target not in self._tried:
                self._tried.append(kernel.target)
            return got

        if self._refuse(kernel.target, kernel.file):
            return _done("refused", reason="safety core is not a candidate")
        if not self.available():
            return _done("unavailable", reason="no C/Rust toolchain, or hot-swap is off")
        try:
            reference = self._reference(kernel)
            if reference is None:
                return _done("rejected", reason="function could not be resolved at runtime")
            source = _CLowerer(self._ast_of(kernel), kernel.params,
                               integral=kernel.integral).lower(symbol=_symbol(kernel))
            samples, hi = self._samples(kernel, reference)
            if len(samples) < 4:
                return _done("rejected", reason="not enough inputs the reference accepts")
            candidate = self._compile(kernel, source, reference, samples)
            if candidate is None:
                return _done("rejected",
                             reason=f"not identical on {len(samples)} cases over "
                                    f"{self._domain(hi)}, or under {self.min_speedup}x")
            return self._adopt(kernel, reference, candidate, samples, hi, _done)
        except Exception as exc:  # noqa: BLE001 — a forge failure leaves pure Python running
            return _done("rejected", reason=f"{type(exc).__name__}: {exc}")

    def _compile(self, kernel: Kernel, source: str, reference: Callable,
                 samples: Sequence[Tuple]) -> Any:
        from nyxara.growth.native_forge import NativeForge
        forge = NativeForge(min_speedup=self.min_speedup, allow_inprocess=True)
        int_t, dbl_t = ctypes.c_longlong, ctypes.c_double
        argtypes = [int_t if t == "int" else dbl_t for _n, t in kernel.params]
        restype = int_t if kernel.integral else dbl_t
        return forge.forge(reference, samples=samples, symbol=_symbol(kernel),
                           argtypes=argtypes, restype=restype, c_source=source)

    def _adopt(self, kernel: Kernel, reference: Callable, candidate: Any,
               samples: Sequence[Tuple], hi: int, _done: Callable[..., Forged]) -> Forged:
        """Swap it in through the rewriter, which re-checks it and rolls back if it does not hold."""
        from nyxara.njp.autopoiesis import AutopoieticRewriter, RewriteProposal
        module = importlib.import_module(kernel.module)
        original = getattr(module, kernel.name, None)
        if original is None:
            return _done("rejected", reason="the function is not an attribute of its module")
        native = _wrap(candidate.fn, [n for n, _t in kernel.params], kernel.name)
        fresh = self._draw(kernel, reference, hi)

        def _gauntlet() -> float:
            live = getattr(module, kernel.name)
            for args in fresh or samples:
                try:
                    if live(*args) != original(*args):
                        return 0.0
                except Exception:  # noqa: BLE001 — a kernel that raises is not equivalent
                    return 0.0
            return 1.0

        rewriter = AutopoieticRewriter(require_gauntlet=True, max_rewrites_per_cycle=1)
        rewriter.begin_cycle()
        outcome = rewriter.consider(RewriteProposal(
            target=kernel.target,
            apply=lambda: setattr(module, kernel.name, native),
            rollback=lambda: setattr(module, kernel.name, original),
            gauntlet=_gauntlet, baseline=0.5))
        self._forge_times.append(time.time())
        if outcome.status != "promoted":
            return _done("refused" if outcome.status == "refused" else "rejected",
                         language=candidate.language, speedup=candidate.speedup,
                         cases=len(samples), reason=outcome.reason or "rolled back")
        self._rollbacks[kernel.target] = lambda: setattr(module, kernel.name, original)
        got = _done("adopted", language=candidate.language, speedup=candidate.speedup,
                    cases=len(samples),
                    certificate=f"{candidate.certificate}; re-checked on "
                                f"{len(fresh)} fresh cases over {self._domain(hi)}")
        self.adopted[kernel.target] = got
        return got

    # ---- inputs the equivalence check runs on ---------------------------- #
    @staticmethod
    def _domain(hi: int) -> str:
        return f"int∈[1,{hi}], float∈[-8,8]"

    def _samples(self, kernel: Kernel, reference: Callable) -> Tuple[List[Tuple], int]:
        """Inputs the reference accepts, on a domain large enough to be worth timing.

        Two things are going on. Inputs the reference *raises* on are dropped rather than fed to
        C — that is what keeps a division by zero from reaching a compiled kernel, where Python
        would raise and C would not. And the integer domain escalates until one reference call
        costs more than the ctypes call overhead, because below that the speedup gate would be
        measuring the wrapper. Escalation stops at the first scale that is expensive enough, so
        a function whose cost explodes with its input is probed at most one step past cheap.
        """
        floats_only = all(t == "float" for _n, t in kernel.params)
        best: List[Tuple] = []
        chosen = _SCALES[0]
        for hi in _SCALES:
            drawn = self._draw(kernel, reference, hi)
            if len(drawn) < 4:
                break
            cost = self._cost(reference, drawn)
            if cost is None:
                break
            best, chosen = drawn, hi
            if cost >= _MIN_CALL_MS or floats_only:
                break                      # worth timing — or nothing here scales with an int
        return best, chosen

    def _draw(self, kernel: Kernel, reference: Callable, hi: int) -> List[Tuple]:
        out: List[Tuple] = []
        for _ in range(self.cases * 3):
            if len(out) >= self.cases:
                break
            args = tuple(self._rng.randint(1, hi) if t == "int"
                         else self._rng.uniform(-8.0, 8.0) for _n, t in kernel.params)
            try:
                reference(*args)
            except Exception:  # noqa: BLE001 — the reference itself refuses these inputs
                continue
            out.append(args)
        return out

    @staticmethod
    def _cost(reference: Callable, samples: Sequence[Tuple]) -> Optional[float]:
        """Mean milliseconds per reference call over the whole sample set, or ``None``.

        Timing the whole set rather than a few of it matters for a function whose cost depends on
        its input — an early-exit branch taken by half the domain drags the mean down, and that
        low mean is the *truth* about what a native kernel would save. Abandoned once it has cost
        enough to have answered the question.
        """
        try:
            probe = list(samples)
            started = time.perf_counter()
            for index, args in enumerate(probe, start=1):
                reference(*args)
                elapsed = time.perf_counter() - started
                if elapsed > 0.02:
                    return elapsed * 1000.0 / index
            return (time.perf_counter() - started) * 1000.0 / max(1, len(probe))
        except Exception:  # noqa: BLE001
            return None

    def _reference(self, kernel: Kernel) -> Optional[Callable]:
        try:
            fn = getattr(importlib.import_module(kernel.module), kernel.name, None)
            return fn if callable(fn) and not getattr(fn, "__nyx_native__", False) else None
        except Exception:  # noqa: BLE001
            return None

    def _ast_of(self, kernel: Kernel) -> ast.FunctionDef:
        with open(kernel.file, "r", encoding="utf-8") as fh:
            tree = ast.parse(fh.read())
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == kernel.name:
                return node
        raise _Unlowerable("function no longer in its source file")

    # ---- reversal --------------------------------------------------------- #
    def rollback(self, target: str) -> bool:
        """Put the Python function back. Always possible: the source was never touched."""
        try:
            undo = self._rollbacks.pop(target, None)
            if undo is None:
                return False
            undo()
            self.adopted.pop(target, None)
            return True
        except Exception:  # noqa: BLE001
            return False

    def rollback_all(self) -> int:
        return sum(1 for target in list(self._rollbacks) if self.rollback(target))

    # ---- the clock -------------------------------------------------------- #
    def _within_budget(self) -> bool:
        cutoff = time.time() - 3600.0
        self._forge_times = [t for t in self._forge_times if t >= cutoff]
        return len(self._forge_times) < self.max_forges_per_hour

    def beat(self, *, oversight: Any = None) -> Optional[Forged]:
        """One forge attempt, at most, when permitted — and nothing at all when it is not."""
        try:
            if oversight is not None:
                gate = getattr(oversight, "gate", None)
                permitted = bool(gate()) if callable(gate) else bool(gate)
                if not permitted:
                    return None
            if not self.available() or not self._within_budget():
                return None
            kernel = self.next_target()
            if kernel is None:
                return None
            return self.forge(kernel)
        except Exception:  # noqa: BLE001 — self-optimisation never breaks the beat
            return None

    def next_target(self) -> Optional[Kernel]:
        """The best-scoring function she has not already tried, hottest module first.

        Reading herself is not free, so a beat parses at most ``scan_per_beat`` new modules and
        picks from everything read so far. She widens her view of herself over many beats rather
        than stalling one of them.
        """
        try:
            budget = self.scan_per_beat
            for name in self._search_order():
                if name in self._scanned:
                    continue
                if budget <= 0:
                    break
                self._scan_one(name)
                budget -= 1
            best: Optional[Kernel] = None
            for kernels in self._scanned.values():
                for kernel in kernels:
                    if kernel.target in self._tried:
                        continue
                    if best is None or kernel.score > best.score:
                        best = kernel
            return best
        except Exception:  # noqa: BLE001
            return None

    # ---- introspection ---------------------------------------------------- #
    def ledger(self) -> List[Dict[str, Any]]:
        return [a.to_dict() for a in self.attempts]

    def stats(self) -> Dict[str, Any]:
        try:
            from nyxara.growth.native_forge import NativeForge
            toolchain = [lang for lang in ("c", "rust") if NativeForge.available(lang)]
        except Exception:  # noqa: BLE001
            toolchain = []
        try:
            return {"available": self.available(), "toolchain": toolchain,
                    "hot_swap": self.hot_swap, "min_speedup": self.min_speedup,
                    "hottest": [{"module": m, "ms": round(ms, 3)} for m, ms in self.hottest(3)],
                    "attempts": len(self.attempts),
                    "adopted": {t: g.to_dict() for t, g in self.adopted.items()},
                    "forges_this_hour": len(self._forge_times),
                    "budget_per_hour": self.max_forges_per_hour,
                    "reversible": True,
                    "last": self.attempts[-1].to_dict() if self.attempts else None}
        except Exception:  # noqa: BLE001
            return {"available": False}

    def to_dict(self) -> Dict[str, Any]:
        """Latency and the attempt ledger persist. A compiled kernel does not — by design."""
        return {"latency": {k: round(v, 4) for k, v in self.latency.items()},
                "calls": dict(self.calls), "tried": list(self._tried),
                "attempts": [a.to_dict() for a in self.attempts[-32:]]}

    def load_dict(self, data: Any) -> bool:
        try:
            if not isinstance(data, dict):
                return False
            self.latency = {str(k): float(v) for k, v in (data.get("latency") or {}).items()}
            self.calls = {str(k): int(v) for k, v in (data.get("calls") or {}).items()}
            self._tried = [str(t) for t in (data.get("tried") or [])]
            return True
        except Exception:  # noqa: BLE001
            return False


def _symbol(kernel: Kernel) -> str:
    safe = "".join(ch if ch.isalnum() else "_" for ch in kernel.name)
    return f"nyx_k_{safe}"


def _wrap(fn: Callable, names: Sequence[str], pyname: str) -> Callable:
    """A ctypes kernel, given back the calling convention its Python original had."""

    def native(*args: Any, **kwargs: Any) -> Any:
        if kwargs:
            args = tuple(args) + tuple(kwargs[n] for n in names[len(args):])
        return fn(*args)

    native.__name__ = pyname
    native.__qualname__ = pyname
    native.__doc__ = "Native C kernel forged by L-OMNI; equivalent to the Python original."
    native.__nyx_native__ = True                        # type: ignore[attr-defined]
    return native
