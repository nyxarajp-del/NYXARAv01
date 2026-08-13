"""The ``nyxara.nyx`` facade must not bind two different classes to one exported name.

The package deliberately aliases every colliding re-export — ``Verdict as ConsequenceVerdict``,
``Verdict as ProofVerdict``, ``Retrieved as VsaRetrieved``. Two escaped that discipline:
``hybrid.Verdict`` (the symbolic/subsymbolic fusion verdict) and ``ascent.Verdict``. Imported
later, ascent's silently shadowed the fusion one, so ``nyxara.nyx.Verdict`` no longer meant what
the V.01 block advertised — and ``"Verdict"`` appeared twice in ``__all__``, so ``import *``
exported only the survivor while the other was unreachable through the facade entirely.
"""

from __future__ import annotations

import collections

import nyxara.nyx as nyx


def test_all_has_no_duplicate_entries():
    dupes = [n for n, k in collections.Counter(nyx.__all__).items() if k > 1]
    assert not dupes, f"__all__ binds these names more than once: {dupes}"


def test_every_exported_name_resolves():
    missing = [n for n in nyx.__all__ if not hasattr(nyx, n)]
    assert not missing, f"__all__ advertises names the facade does not define: {missing}"


def test_fusion_verdict_keeps_the_bare_name():
    from nyxara.nyx.hybrid import Verdict as FusionVerdict
    assert nyx.Verdict is FusionVerdict


def test_ascent_verdict_is_reachable_under_its_alias():
    from nyxara.nyx.ascent import Verdict as RealAscentVerdict
    assert nyx.AscentVerdict is RealAscentVerdict


def test_the_two_verdicts_are_genuinely_different_types():
    """If these ever became the same class the aliasing would be pointless — pin that they differ."""
    assert nyx.Verdict is not nyx.AscentVerdict
