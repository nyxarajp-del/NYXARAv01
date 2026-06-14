"""Tests for nyxara.agency.toolpacks — domain tool packs (Pillar E1)."""

from __future__ import annotations

from nyxara.agency.default_tools import build_default_tools
from nyxara.agency.permissions import Capability, RiskTier
from nyxara.agency.tools import ToolRegistry
from nyxara.agency.toolpacks import (
    build_domain_packs,
    python_check,
    register_packs,
    summarize_text,
)


# --------------------------------------------------------------------------- #
# Pack-specific tools
# --------------------------------------------------------------------------- #
def test_summarize_picks_salient_sentences_and_keeps_order():
    text = ("Cats are small animals. The mitochondria is the powerhouse of the cell. "
            "Cats purr when content. Cats are small animals that purr. "
            "Weather today is mild.")
    out = summarize_text(text, max_sentences=2)
    # the cat-dense sentences dominate the frequency score; the off-topic ones drop
    assert "cats" in out.lower()
    assert "mitochondria" not in out.lower()
    # never longer than requested
    assert out.count(".") <= 2


def test_summarize_short_text_is_returned_whole():
    assert summarize_text("One sentence only.", max_sentences=3) == "One sentence only."
    assert summarize_text("") == ""


def test_python_check_accepts_valid_and_flags_invalid():
    assert python_check("x = 1\nprint(x)")["ok"] is True
    bad = python_check("def f(:\n    pass")
    assert bad["ok"] is False and bad["line"] == 1 and bad["error"]


def test_python_check_never_executes_code():
    # if it ran the code this would raise; it must only compile, so it stays ok
    assert python_check("raise RuntimeError('boom')")["ok"] is True


# --------------------------------------------------------------------------- #
# Packs
# --------------------------------------------------------------------------- #
def test_register_packs_adds_only_the_new_tools():
    reg = ToolRegistry()
    build_default_tools(reg)
    before = set(reg.names())
    added = register_packs(reg)
    assert added["researcher"] == ["summarize_text"]
    assert added["coder"] == ["python_check"]
    assert added["maker"] == []                       # maker reuses existing tools only
    assert set(reg.names()) - before == {"summarize_text", "python_check"}


def test_register_packs_is_idempotent():
    reg = ToolRegistry()
    build_default_tools(reg)
    register_packs(reg)
    again = register_packs(reg)                        # nothing new the second time
    assert again["researcher"] == [] and again["coder"] == []


def test_pack_catalog_lists_the_roles_available_tools():
    reg = ToolRegistry()
    build_default_tools(reg)
    register_packs(reg)
    coder = build_domain_packs()["coder"]
    cat = coder.catalog(reg)
    assert "python_check" in cat and "read_file" in cat
    assert len(cat) == len(set(cat))                  # no duplicates


def test_pack_tools_are_gated_trivial_and_reversible():
    reg = ToolRegistry()
    register_packs(reg)
    for name in ("summarize_text", "python_check"):
        spec = reg.get(name)
        assert spec is not None
        assert spec.capability is Capability.TOOL_CALL
        assert spec.risk is RiskTier.TRIVIAL and spec.reversible


def test_named_subset_registers_only_that_pack():
    reg = ToolRegistry()
    build_default_tools(reg)
    added = register_packs(reg, names=("researcher",))
    assert "researcher" in added and "coder" not in added
    assert reg.get("summarize_text") is not None
    assert reg.get("python_check") is None            # coder pack not requested
