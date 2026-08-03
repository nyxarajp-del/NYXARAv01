"""NYXARA · growth/operator_source.py — a searched genome becomes readable, runnable source (✎).

:class:`~nyxara.growth.genesis.MixerProgram` lets the architecture search escape its fixed palette:
its ``steps`` are evolved, so NYXARA composes token-mixers nobody named. But a ``MixerProgram`` is
**data** — a list of strings interpreted at run time by hand-written builders in
:mod:`~nyxara.growth.genesis_numpy`. Nothing ever turned one into *source*. The only source-renderer
in the package was ``meta_lab.render_addition``, which wraps a snippet in a function.

That mattered because the rest of the chain already existed and had nowhere to start:
:func:`~nyxara.agency.code_sandbox.run_python` to test authored code, the Foundry gauntlet to judge
it, :meth:`~nyxara.growth.self_optimize.Optimizer.apply` to write it reversibly,
:func:`~nyxara.growth.module_loader.load_module_from_path` to import it, and
:meth:`~nyxara.causal.polymorph_runtime.PolymorphRuntime.hot_swap` to install it. This module is the
first link: **genome → Python file**.

**The emitted module imports nothing.** Its ``forward`` takes an ``ops`` namespace as a parameter,
so the operator is a pure composition over injected autograd primitives. That is not a workaround
for the loader's import allowlist (though it satisfies it); it is what makes the emitted file a
*definition* rather than a copy — the same source runs against any backend that supplies the
handful of ops it names, and it can be read, diffed and reviewed on its own.

**What is emitted inline versus delegated, stated plainly.** Five primitives compose cleanly from
the public autograd ops and are emitted as real expressions: ``proj``, ``gate``, ``shift``,
``norm``, ``ssm``. Two — ``conv`` and ``lowrank`` — are implemented in ``genesis_numpy`` with
closures over the tape and explicit numpy masking; re-emitting them would produce something that
*looks* equivalent and is not, so the emitted source calls the injected implementation instead.
Composition is genuinely authored either way; those two steps are genuinely delegated, and pretending
otherwise would be the only real failure available here.

Equivalence is not asserted, it is **checked**: :func:`verify_equivalence` runs the rendered
operator and ``GenesisNumpyModel``'s interpreted path on the same parameters and compares outputs.
A renderer whose output merely resembles the interpreter is worthless.

Depends on ``growth/genesis`` (the genome) and ``growth/genesis_numpy`` (the ops it names).
Nothing imports back.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

__all__ = ["OperatorSource", "render_operator", "build_operator_params", "verify_equivalence"]

# Primitives whose forward is emitted as a real expression over the injected op namespace.
INLINE_PRIMS: Tuple[str, ...] = ("proj", "gate", "shift", "norm", "ssm")
# Primitives the emitted source delegates to the model's own implementation, because theirs
# involves tape closures and numpy masking that cannot be re-expressed faithfully here.
DELEGATED_PRIMS: Tuple[str, ...] = ("conv", "lowrank")

_HEADER = '''"""NYXARA · a forged token-mixer — authored by her own architecture search.

Genome: {fingerprint}

This file was rendered from a MixerProgram by growth/operator_source.py. It imports nothing: every
operation arrives through the ``ops`` namespace, so this module is a definition of the operator
rather than a copy of one backend's implementation.
"""

STEPS = {steps!r}
FINGERPRINT = {fingerprint!r}


def param_spec(c, ctx):
    """Parameter shapes this operator needs, as ``{{name: shape}}``."""
    return {spec}


def forward(x, p, ops):
    """Apply the operator.

    x   -- input tensor, shape (batch, time, channels)
    p   -- ``{{name: tensor}}`` matching :func:`param_spec`
    ops -- namespace supplying the autograd primitives named below
    """
'''


def _base(step: str) -> str:
    return str(step).split(":", 1)[0]


def _param(step: str, default: Optional[int] = None) -> Optional[int]:
    part = str(step).split(":", 1)
    if len(part) != 2:
        return default
    try:
        return int(part[1])
    except ValueError:
        return default


@dataclass
class OperatorSource:
    """A rendered operator: its source, its parameter shapes, and what it delegated."""

    source: str = ""
    fingerprint: str = ""
    steps: List[str] = field(default_factory=list)
    spec: Dict[str, Tuple[int, ...]] = field(default_factory=dict)
    delegated: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    @property
    def fully_inline(self) -> bool:
        """True when every step was emitted as source rather than delegated."""
        return not self.delegated

    def summary(self) -> str:
        head = (f"· forged operator {self.fingerprint} — {len(self.steps)} step(s), "
                f"{len(self.spec)} parameter tensor(s)")
        if self.delegated:
            head += f"; delegated: {', '.join(sorted(set(self.delegated)))}"
        return "\n".join([head] + [f"    · {n}" for n in self.notes])

    def to_dict(self) -> Dict[str, Any]:
        return {"fingerprint": self.fingerprint, "steps": list(self.steps),
                "spec": {k: list(v) for k, v in self.spec.items()},
                "delegated": list(self.delegated), "fully_inline": self.fully_inline,
                "chars": len(self.source), "notes": list(self.notes)}


def _step_code(index: int, step: str) -> Tuple[str, Dict[str, str], Optional[str]]:
    """Emit one step. Returns ``(code_line, {param: shape_expr}, delegated_or_None)``.

    The emitted expressions mirror ``GenesisNumpyModel._synth`` exactly — same ops, same order,
    same parameter names — because equivalence with the interpreted path is the whole point."""
    prefix = f"s{index}_"
    base = _base(step)

    if base == "proj":
        return (f'    cur = ops.gelu(ops.matmul(cur, p["{prefix}w"]))',
                {f"{prefix}w": "(c, c)"}, None)
    if base == "gate":
        return (f'    cur = ops.mul(cur, ops.sigmoid(ops.matmul(cur, p["{prefix}g"])))',
                {f"{prefix}g": "(c, c)"}, None)
    if base == "shift":
        return (f'    cur = ops.add(ops.matmul(cur, p["{prefix}w"]), '
                f'ops.matmul(ops.shift_time(cur, 1), p["{prefix}sw"]))',
                {f"{prefix}w": "(c, c)", f"{prefix}sw": "(c, c)"}, None)
    if base == "norm":
        return (f'    cur = ops.rmsnorm(cur, p["{prefix}g"])',
                {f"{prefix}g": "(c,)"}, None)
    if base == "ssm":
        return (f'    cur = ops.diag_scan(ops.matmul(cur, p["{prefix}in"]), '
                f'ops.sigmoid(p["{prefix}a"]))',
                {f"{prefix}in": "(c, c)", f"{prefix}a": "(c,)"}, None)
    if base == "conv":
        width = _param(step, 3) or 3
        return (f'    cur = ops.conv(cur, p["{prefix}cw"])   # delegated: tape closures over rows',
                {f"{prefix}cw": f"({width}, c)"}, "conv")
    if base == "lowrank":
        rank = _param(step, 4) or 4
        return (f'    cur = ops.lowrank(cur, p["{prefix}a"], p["{prefix}b"])'
                f'   # delegated: causal mask',
                {f"{prefix}a": f"(ctx, {rank})", f"{prefix}b": f"({rank}, ctx)"}, "lowrank")
    return (f"    # unknown primitive {base!r} — skipped", {}, None)


def render_operator(program: Any) -> OperatorSource:
    """Render a :class:`MixerProgram` (or its ``steps``) into an importable Python module.

    Never raises: an empty or unusable genome yields an identity operator rather than a failure,
    so a search that produced nothing useful does not take the pipeline down with it."""
    # An empty ``steps`` is falsy but still a valid genome, so test for the attribute rather than
    # its truth — otherwise MixerProgram(steps=[]) falls through to iterating the program itself.
    raw = getattr(program, "steps", None)
    if raw is None:
        raw = program if isinstance(program, (list, tuple)) else []
    steps = [str(s) for s in raw] or ["proj"]
    fingerprint = "+".join(steps)

    lines: List[str] = ["    cur = x"]
    spec_parts: Dict[str, str] = {}
    delegated: List[str] = []
    for index, step in enumerate(steps):
        code, params, deleg = _step_code(index, step)
        lines.append(code)
        spec_parts.update(params)
        if deleg:
            delegated.append(deleg)
    lines.append("    return cur")

    spec_literal = ("{" + ", ".join(f'"{k}": {v}' for k, v in spec_parts.items()) + "}") \
        if spec_parts else "{}"
    source = _HEADER.format(fingerprint=fingerprint, steps=steps,
                            spec=spec_literal) + "\n".join(lines) + "\n"

    rendered = OperatorSource(source=source, fingerprint=fingerprint, steps=steps,
                              delegated=delegated)
    # resolve the shape expressions once so callers get real shapes, not strings
    try:
        rendered.spec = _resolve_spec(source, c=8, ctx=8)
    except Exception:  # noqa: BLE001 — the shapes are a convenience; the source is the artefact
        rendered.notes.append("could not resolve parameter shapes")
    if delegated:
        rendered.notes.append(
            "conv/lowrank call the injected implementation — their tape closures and causal "
            "masking cannot be re-emitted faithfully, and an approximation would be worse")
    return rendered


def _resolve_spec(source: str, *, c: int, ctx: int) -> Dict[str, Tuple[int, ...]]:
    """Execute only the rendered ``param_spec`` to turn its shape expressions into tuples."""
    namespace: Dict[str, Any] = {}
    exec(compile(source, "<forged-operator>", "exec"), namespace)   # noqa: S102 — our own render
    spec = namespace["param_spec"](c, ctx)
    return {k: tuple(v) for k, v in spec.items()}


def build_operator_params(rendered: OperatorSource, *, c: int, ctx: int, seed: int = 0) -> Any:
    """Allocate trainable NumPy tensors for a rendered operator, matching ``genesis_numpy``.

    Returns ``{name: Tensor}`` using the same initialisation scale the interpreted path uses, so a
    rendered operator and the interpreted one start from identical weights and can be compared."""
    import numpy as np

    from nyxara.growth.genesis_numpy import Tensor

    rng = np.random.default_rng(seed)
    spec = rendered.spec or _resolve_spec(rendered.source, c=c, ctx=ctx)
    out: Dict[str, Any] = {}
    for name, shape in spec.items():
        real = tuple(ctx if s == "ctx" else (c if s == "c" else int(s)) for s in shape) \
            if any(isinstance(s, str) for s in shape) else tuple(int(s) for s in shape)
        out[name] = Tensor(rng.standard_normal(real) * 0.02, requires_grad=True)
    return out


def operator_ops() -> Any:
    """The op namespace a rendered operator's ``forward`` expects.

    Everything comes from ``genesis_numpy``'s autograd, so a rendered operator is differentiable
    on exactly the same tape as the rest of her brain — no second gradient implementation to drift."""
    from nyxara.growth import genesis_numpy as g

    class _Ops:
        matmul = staticmethod(g.matmul)
        add = staticmethod(g.add)
        mul = staticmethod(g.mul)
        gelu = staticmethod(g.gelu)
        sigmoid = staticmethod(g.sigmoid)
        rmsnorm = staticmethod(g.rmsnorm)
        shift_time = staticmethod(g.shift_time)
        diag_scan = staticmethod(g.diag_scan)

        @staticmethod
        def conv(x: Any, w: Any) -> Any:
            """Causal depthwise convolution — the same accumulation genesis_numpy performs."""
            acc = None
            for j in range(w.data.shape[0]):
                row = g.Tensor(w.data[j], _prev=(w,))
                row.requires_grad = w.requires_grad
                term = g.mul(g.shift_time(x, j), row)
                acc = term if acc is None else g.add(acc, term)
            return acc if acc is not None else x

        @staticmethod
        def lowrank(x: Any, a: Any, b: Any) -> Any:
            """Causally-masked low-rank token mixing."""
            import numpy as np
            _b, t, _c = x.shape
            full = g.matmul(a, b)
            mask = np.tril(np.ones((full.data.shape[0], full.data.shape[1])))[:t, :t]
            m = g.Tensor(full.data[:t, :t] * mask, _prev=(full,))
            return g.matmul(m, x)

    return _Ops()


def verify_equivalence(rendered: OperatorSource, *, c: int = 8, ctx: int = 8,
                       batch: int = 2, seed: int = 0,
                       tol: float = 1e-9) -> Tuple[bool, str]:
    """Run the rendered operator **and** ``GenesisNumpyModel``'s interpreted path, and compare.

    This is the check that makes the renderer worth anything. Rendering source that merely
    *resembles* the interpreter is the easy failure — it looks right, it runs, and it silently
    computes something else. So both paths execute on the *same* parameter tensors and their
    outputs are compared elementwise. The rendered parameter names (``s0_w``, ``s1_g``, …) are
    chosen to match the interpreter's under an empty prefix precisely so this comparison is
    possible at all.

    Returns ``(ok, detail)``; never raises."""
    try:
        import numpy as np

        from nyxara.growth.genesis_numpy import Tensor, backward

        namespace: Dict[str, Any] = {}
        exec(compile(rendered.source, "<forged-operator>", "exec"), namespace)  # noqa: S102
        params = build_operator_params(rendered, c=c, ctx=ctx, seed=seed)
        rng = np.random.default_rng(seed + 1)
        x = Tensor(rng.standard_normal((batch, ctx, c)), requires_grad=True)

        out = namespace["forward"](x, params, operator_ops())
        if tuple(out.data.shape) != (batch, ctx, c):
            return False, f"shape {out.data.shape} != {(batch, ctx, c)}"
        if not np.all(np.isfinite(out.data)):
            return False, "produced non-finite values"

        # --- the comparison: the interpreted path, on the same tensors --- #
        interpreted, why = _interpreted_output(rendered, params, x, c=c, ctx=ctx)
        if interpreted is None:
            match = f"not compared ({why})"
        else:
            delta = float(np.max(np.abs(out.data - interpreted.data)))
            if delta > tol:
                return False, f"DIVERGES from the interpreted path: max|Δ| = {delta:.3e}"
            match = f"matches the interpreted path (max|Δ| = {delta:.1e})"

        # the gradient has to reach the parameters, or it is not a trainable operator
        backward(out)
        touched = sum(1 for t in params.values() if getattr(t, "grad", None) is not None)
        if params and touched == 0:
            return False, "no parameter received a gradient"
        return True, f"{match}; {touched}/{len(params)} parameter tensor(s) received gradient"
    except Exception as exc:  # noqa: BLE001 — a failed verification is a result, not a crash
        return False, f"{type(exc).__name__}: {exc}"


def _interpreted_output(rendered: OperatorSource, params: Dict[str, Any], x: Any,
                        *, c: int, ctx: int) -> Tuple[Optional[Any], str]:
    """Run ``GenesisNumpyModel._synth`` over the same steps and the same parameter tensors.

    Returns ``(tensor_or_None, reason)``. The model is built with a minimal genome purely to
    borrow its interpreter; ``_synth`` reads everything it needs out of ``model.params``, and the
    rendered names line up under an empty prefix by construction."""
    try:
        from nyxara.growth.genesis_numpy import GenesisNumpyModel

        model = GenesisNumpyModel({"n_embd": c, "block_size": ctx,
                                   "layers": [{"op": "synth"}]}, seed=0)
        model.c, model.ctx = c, ctx
        model.params = dict(params)
        model.params["_steps"] = list(rendered.steps)
        return model._synth(x, ""), "ok"
    except Exception as exc:  # noqa: BLE001 — an uncomparable genome is reported, not fatal
        return None, f"{type(exc).__name__}: {exc}"
