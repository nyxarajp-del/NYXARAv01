"""NYXARA · growth/meta_lab.py — helpers for integrating sandbox-grown optimizations.

The :class:`~nyxara.growth.meta_research.MetaResearcher` validates invented optimizations in a
sandbox. When (and only when) the Master authorises integration, the validated technique is
appended to a dedicated *grown-code* file under a reversible, gauntlet-gated edit. This module
just names that file and renders a well-formed, documented addition for it — it performs no I/O
itself (the Optimizer writes the file under its verify-or-rollback discipline).
"""

from __future__ import annotations

import re
from pathlib import Path

__all__ = ["lab_path", "render_addition", "GROWN_HEADER"]

GROWN_HEADER = (
    '"""NYXARA · growth/_meta_grown.py — sandbox-grown, gauntlet-verified helpers (auto-generated).\n\n'
    "Each function below was *invented* by the MetaResearcher, *validated* in the sandbox, and\n"
    "*integrated* only after clearing the verify-or-rollback gauntlet with the Master's standing\n"
    "authorisation. Do not edit by hand — additions are appended programmatically.\n"
    '"""\n\nfrom __future__ import annotations\n')


def lab_path() -> Path:
    """Absolute path to the grown-code file (a sibling of this module)."""
    return Path(__file__).resolve().parent / "_meta_grown.py"


def _slug(title: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "_", title.strip().lower()).strip("_")
    return s or "technique"


def render_addition(title: str, description: str, code: str) -> str:
    """Render a validated technique as an indented, documented function appended to the lab file."""
    name = _slug(title)[:48]
    body_lines = [f"    {ln}" if ln.strip() else "" for ln in code.splitlines()]
    body = "\n".join(body_lines)
    safe_desc = description.replace('"""', "'''")
    return (f"\n\ndef grown_{name}():\n"
            f'    """{title}\n\n    {safe_desc}\n\n'
            f"    Validated in the sandbox (result == expected) before integration.\n"
            f'    """\n'
            f"{body}\n"
            f"    return result == expected\n")


# --------------------------------------------------------------------------- #
# Self-test / demo
# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # pragma: no cover
    print("=" * 70)
    print("NYXARA meta-lab self-test")
    print("=" * 70)

    add = render_addition("The sum of first n odds equals n^2",
                          "Closed-form identity.",
                          "n = 5\nresult = sum(2*k-1 for k in range(1, n+1))\nexpected = n*n\n")
    full = GROWN_HEADER + add
    # the rendered module must be valid, importable Python
    compile(full, "<meta-grown>", "exec")
    print(add)
    assert "def grown_" in add and "return result == expected" in add
    assert str(lab_path()).endswith("_meta_grown.py")

    print("\nALL SELF-TESTS PASSED ✓")
