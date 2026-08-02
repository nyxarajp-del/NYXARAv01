"""The honesty invariant — pinned exactly like tests/senses/test_hardware.py.

NYX-5 is a *software simulation* on commodity silicon. These strings must remain in the package and
config docstrings so the honest framing is load-bearing, not decorative: no false neuromorphic-hardware
claims, no backpropagation.
"""

from __future__ import annotations

import nyxara.nyx5 as nyx5
from nyxara.kernel.config import Nyx5Config


def test_package_docstring_is_honest():
    doc = (nyx5.__doc__ or "").lower()
    assert "commodity silicon" in doc
    assert "not neuromorphic" in doc
    assert "no backpropagation" in doc


def test_config_docstring_is_honest():
    doc = (Nyx5Config.__doc__ or "").lower()
    assert "commodity silicon" in doc
    assert "not neuromorphic" in doc
    assert "no backpropagation" in doc


def test_no_ghz_or_power_claims_as_fact():
    # the honest ceiling is stated as analogy; make sure the module never asserts the marketing numbers
    doc = (nyx5.__doc__ or "").lower()
    assert "ghz" not in doc or "no literal ghz" in doc
