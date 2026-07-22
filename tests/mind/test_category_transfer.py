"""Tests for F7 — category-theoretic functorial transfer (mind/category_transfer.py).

A functor must be well-defined (signature-preserving), its image of a proven abstraction must be a
zero-shot concept in a sibling domain, and it is adopted ONLY when verified to solve a held-out task —
otherwise she abstains. No LLM anywhere.
"""

from __future__ import annotations

import random

from nyxara.growth.noesis import (
    INT,
    INTLIST,
    Abstraction,
    NoesisEngine,
    app,
    base_library,
    hole,
    synthetic_tasks,
    var,
)
from nyxara.mind.category_transfer import (
    Functor,
    FunctorTransfer,
    apply_functor,
    discover_functors,
)


def _sum_scaled_abs():
    return Abstraction("absK", (INT,), INT,
                       app("sum_", INT, [app("scale_each", INTLIST, [var(INTLIST), hole(0, INT)])]))


def test_functor_well_defined_requires_matching_signatures():
    lib = base_library()
    # scale_each and inc_each share (intlist,int)->intlist → a real functor
    assert Functor("s->i", {"scale_each": "inc_each"}).well_defined(lib)
    # scale_each -> sum_ does NOT preserve the signature → not a functor
    assert not Functor("bad", {"scale_each": "sum_"}).well_defined(lib)


def test_apply_functor_is_the_image_on_morphisms():
    src = _sum_scaled_abs()
    image = apply_functor(src.body, {"scale_each": "inc_each"})
    assert image.pretty() == "sum_(inc_each(x, $0))"


def test_verified_transfer_is_adopted_zero_shot():
    lib = base_library()
    src = _sum_scaled_abs()
    lib.add_abstraction(src)
    probes = [t for t in synthetic_tasks(random.Random(0), 30) if t.name.startswith("sum_inc")][:5]
    adopted = FunctorTransfer().transfer_all(lib, [src], probes)
    assert len(adopted) == 1
    assert adopted[0].body.pretty() == "sum_(inc_each(x, $0))"


def test_unverified_transfer_abstains():
    lib = base_library()
    src = _sum_scaled_abs()
    lib.add_abstraction(src)
    # give it NO sibling probes it could solve → the transfer must abstain (adopt nothing)
    probes = [t for t in synthetic_tasks(random.Random(0), 20) if t.name.startswith("count_over")][:5]
    adopted = FunctorTransfer().transfer_all(lib, [src], probes)
    assert adopted == []


def test_discover_functors_finds_signature_siblings():
    lib = base_library()
    fs = {f.name for f in discover_functors(lib)}
    # scale_each and inc_each are (intlist,int)->intlist siblings
    assert "scale_each->inc_each" in fs or "inc_each->scale_each" in fs


def test_engine_with_transfer_produces_verified_transfers():
    eng = NoesisEngine(seed=1, tasks_per_cycle=14, transfer=FunctorTransfer())
    reports = eng.run(6)
    assert any(r.transferred > 0 for r in reports)
    assert any(name.startswith("xfer") for name in eng.library.abstractions)
