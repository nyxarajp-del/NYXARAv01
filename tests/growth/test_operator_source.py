"""A searched genome becomes source — and the source computes the same thing the genome did."""

from __future__ import annotations

import ast
import shutil

import pytest

from nyxara.growth.genesis import MixerProgram
from nyxara.growth.operator_source import (build_operator_params, operator_ops, render_operator,
                                           verify_equivalence)

pytest.importorskip("numpy")

_GENOMES = (
    ["proj"],
    ["proj", "gate", "norm"],
    ["shift", "ssm", "proj"],
    ["gate", "shift"],
    ["conv:5", "gate"],
    ["lowrank:4", "norm"],
    ["norm", "proj", "gate", "ssm", "shift"],
)


# -------------------- the emitted source is real Python -------------------- #
@pytest.mark.parametrize("steps", _GENOMES)
def test_rendered_source_parses(steps):
    rendered = render_operator(MixerProgram(steps=list(steps)))
    ast.parse(rendered.source)
    assert rendered.fingerprint == "+".join(steps)


def test_the_emitted_module_imports_nothing():
    """Ops arrive as a parameter, which is what makes the file a definition rather than a copy of
    one backend — and it is why the loader's import allowlist has nothing to object to."""
    rendered = render_operator(MixerProgram(steps=["proj", "gate"]))
    tree = ast.parse(rendered.source)
    assert not [n for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom))]


def test_an_empty_genome_renders_an_operator_rather_than_failing():
    rendered = render_operator(MixerProgram(steps=[]))
    ast.parse(rendered.source)
    ok, detail = verify_equivalence(rendered)
    assert ok, detail


# -------------------- and it computes the same thing -------------------- #
@pytest.mark.parametrize("steps", _GENOMES)
def test_rendered_source_matches_the_interpreted_path_exactly(steps):
    """The failure that matters: source that LOOKS equivalent, runs, and silently computes
    something else. Both paths run on the same parameter tensors and are compared elementwise."""
    rendered = render_operator(MixerProgram(steps=list(steps)))
    ok, detail = verify_equivalence(rendered)
    assert ok, detail
    assert "matches the interpreted path" in detail


@pytest.mark.parametrize("steps", _GENOMES)
def test_gradients_reach_every_parameter(steps):
    """An operator no gradient reaches is not a trainable operator, whatever it computes."""
    rendered = render_operator(MixerProgram(steps=list(steps)))
    ok, detail = verify_equivalence(rendered)
    assert ok and "received gradient" in detail


def test_a_divergent_render_is_caught_not_waved_through():
    """Guard the guard: corrupt the emitted forward and the comparison must fail."""
    rendered = render_operator(MixerProgram(steps=["proj", "gate"]))
    rendered.source = rendered.source.replace("ops.gelu(", "ops.sigmoid(")
    ok, detail = verify_equivalence(rendered)
    assert not ok
    assert "DIVERGES" in detail


# -------------------- shapes and delegation -------------------- #
def test_parameter_shapes_are_resolved():
    rendered = render_operator(MixerProgram(steps=["proj", "norm"]))
    assert rendered.spec["s0_w"] == (8, 8)
    assert rendered.spec["s1_g"] == (8,)


def test_delegation_is_declared_not_hidden():
    """conv and lowrank use tape closures and causal masking that cannot be re-emitted
    faithfully. They are delegated, and the report says so rather than implying otherwise."""
    inline = render_operator(MixerProgram(steps=["proj", "gate"]))
    assert inline.fully_inline and inline.delegated == []

    delegated = render_operator(MixerProgram(steps=["conv:5", "lowrank:4"]))
    assert not delegated.fully_inline
    assert set(delegated.delegated) == {"conv", "lowrank"}
    assert "delegated" in delegated.summary()
    assert any("cannot be re-emitted faithfully" in n for n in delegated.notes)


def test_a_parameterised_step_carries_its_parameter_into_the_shape():
    rendered = render_operator(MixerProgram(steps=["conv:5"]))
    assert rendered.spec["s0_cw"][0] == 5
    assert render_operator(MixerProgram(steps=["lowrank:8"])).spec["s0_a"][1] == 8


def test_report_serialises():
    blob = render_operator(MixerProgram(steps=["proj", "conv:3"])).to_dict()
    assert blob["fingerprint"] == "proj+conv:3"
    assert blob["fully_inline"] is False
    assert blob["chars"] > 0


# -------------------- the chain: genome -> source -> screened -> imported -------------------- #
def test_a_rendered_operator_passes_the_source_screen_and_loads_and_runs():
    """The whole reason this module exists: it is the first link of a chain whose every other
    link already existed and had nothing to start from."""
    import numpy as np

    from nyxara.growth.genesis_numpy import Tensor
    from nyxara.growth.module_loader import forged_root, load_module_from_path, screen_source

    rendered = render_operator(MixerProgram(steps=["proj", "gate", "ssm", "norm"]))
    assert screen_source(rendered.source) is None          # clean under the shared policy

    root = forged_root()
    root.mkdir(parents=True, exist_ok=True)
    (root / "__init__.py").write_text("", encoding="utf-8")
    path = root / "_t_chain_op.py"
    path.write_text(rendered.source, encoding="utf-8")
    try:
        result = load_module_from_path(path)
        assert result.ok, result.reason

        params = build_operator_params(rendered, c=8, ctx=8, seed=0)
        x = Tensor(np.random.default_rng(1).standard_normal((2, 8, 8)), requires_grad=True)
        out = result.module.forward(x, params, operator_ops())
        assert out.data.shape == (2, 8, 8)
        assert np.all(np.isfinite(out.data))
    finally:
        import sys
        sys.modules.pop(f"nyxara_forged_{path.stem}", None)
        shutil.rmtree(root, ignore_errors=True)
