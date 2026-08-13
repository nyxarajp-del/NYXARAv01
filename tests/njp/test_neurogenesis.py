"""New cells appear only when she cannot predict herself — measured, never scheduled."""

from __future__ import annotations

from nyxara.njp.fabric import Fabric


def test_no_neurogenesis_while_she_predicts_herself():
    """A fabric modelling itself well grows no cells, however busy it is."""
    f = Fabric(seed=3, manifold_dim=512, neurogenesis_window=4)
    for _ in range(12):
        f.stimulate([1, 2, 3])
        f.expand(result=f.settle())
    # errors only accrue on TRUSTED predictions that then missed; a stable repeated input
    # should not be manufacturing capacity.
    assert f.stats()["cells"] <= 8


def test_sustained_self_prediction_error_mints_cells():
    """The trigger is the error window, so drive it directly rather than hoping to provoke it."""
    f = Fabric(seed=3, manifold_dim=512, neurogenesis_window=4,
               neurogenesis_error=0.5, neurogenesis_batch=3)
    f.stimulate([1, 2, 3, 4])
    res = f.settle()
    f._errors = [0.9] * 8              # she has been wrong about herself, repeatedly
    before = f.n_cells
    rep = f.expand(result=res)
    assert rep.born == 3
    assert f.n_cells == before + 3


def test_a_newborn_cell_is_wired_in_both_directions():
    """A cell nothing reaches can never fire; a cell that reaches nothing can never matter."""
    f = Fabric(seed=3, manifold_dim=512, neurogenesis_window=2,
               neurogenesis_error=0.5, neurogenesis_batch=1)
    f.stimulate([1, 2, 3, 4])
    res = f.settle()
    known = set(f.cells)
    f._errors = [0.9] * 4
    f.expand(result=res)
    newborn = [c for c in f.cells if c not in known]
    assert newborn
    for cid in newborn:
        assert f.out.get(cid), "a newborn must reach something"
        assert f.inn.get(cid), "a newborn must be reachable"


def test_the_error_window_resets_after_minting():
    """Capacity was added; re-measure before adding more rather than minting every pass."""
    f = Fabric(seed=3, manifold_dim=512, neurogenesis_window=2,
               neurogenesis_error=0.5, neurogenesis_batch=1)
    f.stimulate([1, 2])
    res = f.settle()
    f._errors = [0.9] * 4
    f.expand(result=res)
    assert f._errors == []
