"""NYXARA · njp/__main__.py — ``python -m nyxara.njp``.

The package had no ``__main__``, so ``python -m nyxara.njp`` failed outright and the only way to
train the brain was to know that the entry point lived one module deeper, in
:mod:`nyxara.njp.study`. Training is what one does *to* NJP, so it is what the package does when
it is run.

This forwards to :func:`nyxara.njp.study.main` and adds nothing of its own — one entry point, not
two that can drift apart.
"""

from __future__ import annotations

from nyxara.njp.study import main

if __name__ == "__main__":
    raise SystemExit(main())
