"""Tests for the real-RSI upgrade of nyxara.growth.self_optimize.

Covers the two new deterministic transforms (E713/E714) and the LLM-authored whole-file edit
path — all with a *stub* LLM, so the suite is fully offline and deterministic. The proof these
tests carry: an LLM edit is only ever attempted when triple-gated, a good proposal is kept by
the gauntlet, and a broken/oversized/unfenced proposal is rejected or rolled back byte-for-byte.
"""

from __future__ import annotations

from pathlib import Path

from nyxara.growth.self_optimize import (EditGenerator, GauntletResult, LLMEditGenerator,
                                         Optimizer, _fix_negated_compare, _parses,
                                         _strip_code_fence)
from nyxara.kernel.config import get_settings


# --------------------------------------------------------------------------- #
# Stubs — a fake LLM facade with a selectable "chosen provider"
# --------------------------------------------------------------------------- #
class _StubProvider:
    def __init__(self, name: str = "self", ok: bool = True) -> None:
        self.name = name
        self._ok = ok

    def available(self) -> bool:
        return self._ok


class _StubResp:
    def __init__(self, text: str) -> None:
        self.text = text


class _StubLLM:
    """A fake LLM facade with one selectable provider, exposing the surface the self-editor uses
    (``available_providers`` / ``complete_with``) plus a legacy ``generate``. Default provider is
    ``self`` — NYXARA's own model — which is the author the feature is built around."""

    def __init__(self, text: str = "", *, provider: str = "self", ok: bool = True) -> None:
        self._text = text
        self._prov = _StubProvider(provider, ok)
        self.calls: list = []

    def chosen_provider(self):
        return self._prov

    def available_providers(self):
        return [self._prov.name] if self._prov._ok else []

    def complete_with(self, name: str, req):
        self.calls.append({"provider": name})
        return _StubResp(self._text)

    def generate(self, prompt: str, *, system: str | None = None, **kw):
        self.calls.append({"prompt": prompt, "system": system, **kw})
        return self._text


class _W:
    """A minimal weakness stand-in."""

    def __init__(self, *, id: str, locus: str, remediation: str = "fix it",
                 title: str = "weakness", evidence=None, edit_strategy: str = "llm") -> None:
        self.id = id
        self.locus = locus
        self.remediation = remediation
        self.title = title
        self.evidence = evidence or []
        self.edit_strategy = edit_strategy
        self.is_source_edit = True


def _llm_settings(**over):
    s = get_settings().model_copy(deep=True)
    s.self_improvement.autonomous_enact = True
    s.self_improvement.allow_llm_edits = True
    for k, v in over.items():
        setattr(s.self_improvement, k, v)
    return s


# --------------------------------------------------------------------------- #
# New deterministic transforms — E713 / E714
# --------------------------------------------------------------------------- #
def test_fix_not_in_transform():
    out = _fix_negated_compare("if not a in b:\n    pass\n", 1)
    assert out is not None and "a not in b" in out and "not a in b" not in out
    assert _parses(out)


def test_fix_is_not_transform():
    out = _fix_negated_compare("x = not a is None\n", 1)
    assert out is not None and "a is not None" in out and "not a is None" not in out
    assert _parses(out)


def test_fix_negated_compare_leaves_chained_alone():
    # chained comparison (two operators) is never rewritten — too subtle to do safely
    assert _fix_negated_compare("if not a in b in c:\n    pass\n", 1) is None


def test_edit_generator_fixes_not_in(tmp_path: Path):
    f = tmp_path / "m.py"
    f.write_text("def f(a, b):\n    return not a in b\n", encoding="utf-8")
    edit = EditGenerator().generate(
        _W(id="code-not_in-m.py-2", locus=f"{f}:2", edit_strategy="deterministic"))
    assert edit is not None and "a not in b" in edit.after
    assert _parses(edit.after)


# --------------------------------------------------------------------------- #
# LLM edit generator — gating
# --------------------------------------------------------------------------- #
def test_llm_generator_disabled_without_flag(tmp_path: Path):
    s = get_settings().model_copy(deep=True)
    s.self_improvement.allow_llm_edits = False          # flag off
    gen = LLMEditGenerator(llm=_StubLLM("x = 1\n"), settings=s)
    assert gen.available() is False


def test_self_model_is_an_author_but_mock_is_not(tmp_path: Path):
    s = _llm_settings()
    # NYXARA's OWN model authors her redesigns — this is the whole point of the feature
    assert LLMEditGenerator(llm=_StubLLM(provider="self"), settings=s).available() is True
    # the mock impersonator is never an author (it would only fake a fix)
    assert LLMEditGenerator(llm=_StubLLM(provider="native"), settings=s).available() is False
    # an unavailable own-model can't author
    assert LLMEditGenerator(llm=_StubLLM(provider="self", ok=False), settings=s).available() is False


def test_external_provider_only_when_self_authored_only_off():
    # default (self_authored_only=True): the base LLM is NOT permitted to author — khud NYXARA
    assert LLMEditGenerator(
        llm=_StubLLM(provider="airouter"), settings=_llm_settings()).available() is False
    # with the flag off, the real base provider may author when the own-model isn't ready
    s = _llm_settings(self_authored_only=False)
    assert LLMEditGenerator(llm=_StubLLM(provider="airouter"), settings=s).available() is True


def test_llm_generator_none_llm_is_unavailable():
    assert LLMEditGenerator(llm=None, settings=_llm_settings()).available() is False


# --------------------------------------------------------------------------- #
# LLM edit generator — proposal validation
# --------------------------------------------------------------------------- #
def test_llm_good_edit_is_proposed(tmp_path: Path):
    f = tmp_path / "g.py"
    f.write_text("def add(a, b):\n    return a+b\n", encoding="utf-8")
    fixed = "def add(a, b):\n    return a + b\n"
    gen = LLMEditGenerator(llm=_StubLLM(fixed), settings=_llm_settings())
    edit = gen.generate(_W(id="code-high_complexity-g.py-1", locus=f"{f}:1"))
    assert edit is not None
    assert edit.after == fixed and edit.kind == "self:high_complexity"
    assert _parses(edit.after)


def test_llm_strips_markdown_fence(tmp_path: Path):
    f = tmp_path / "g.py"
    f.write_text("x = 1\n", encoding="utf-8")
    fenced = "```python\nx = 2\n```"
    gen = LLMEditGenerator(llm=_StubLLM(fenced), settings=_llm_settings())
    edit = gen.generate(_W(id="code-todo-g.py-1", locus=f"{f}:1"))
    assert edit is not None and edit.after == "x = 2\n" and "```" not in edit.after


def test_llm_rejects_unparseable_edit(tmp_path: Path):
    f = tmp_path / "g.py"
    f.write_text("x = 1\n", encoding="utf-8")
    gen = LLMEditGenerator(llm=_StubLLM("def (:\n"), settings=_llm_settings())
    assert gen.generate(_W(id="code-todo-g.py-1", locus=f"{f}:1")) is None


def test_llm_rejects_runaway_rewrite(tmp_path: Path):
    f = tmp_path / "g.py"
    f.write_text("x = 1\n", encoding="utf-8")
    runaway = "x = 1\n" + "y = 2  # padding\n" * 200      # far beyond the size-delta bound
    gen = LLMEditGenerator(llm=_StubLLM(runaway), settings=_llm_settings())
    assert gen.generate(_W(id="code-todo-g.py-1", locus=f"{f}:1")) is None


def test_llm_skips_oversize_file(tmp_path: Path):
    f = tmp_path / "big.py"
    f.write_text("# pad\n" * 5000, encoding="utf-8")      # exceeds llm_edit_max_file_bytes
    gen = LLMEditGenerator(llm=_StubLLM("x = 1\n"),
                           settings=_llm_settings(llm_edit_max_file_bytes=1000))
    assert gen.generate(_W(id="code-todo-big.py-1", locus=f"{f}:1")) is None


def test_llm_provider_exception_is_swallowed(tmp_path: Path):
    f = tmp_path / "g.py"
    f.write_text("x = 1\n", encoding="utf-8")

    class _BoomLLM(_StubLLM):
        def complete_with(self, name, req):
            raise RuntimeError("author down")

    gen = LLMEditGenerator(llm=_BoomLLM(), settings=_llm_settings())
    assert gen.generate(_W(id="code-todo-g.py-1", locus=f"{f}:1")) is None


def test_self_model_authors_through_complete_with(tmp_path: Path):
    # the edit is authored by NYXARA's OWN 'self' provider, explicitly (never an external LLM)
    f = tmp_path / "g.py"
    f.write_text("def add(a, b):\n    return a+b\n", encoding="utf-8")
    fixed = "def add(a, b):\n    return a + b\n"
    llm = _StubLLM(fixed, provider="self")
    edit = LLMEditGenerator(llm=llm, settings=_llm_settings()).generate(
        _W(id="code-high_complexity-g.py-1", locus=f"{f}:1"))
    assert edit is not None and edit.kind == "self:high_complexity"
    assert llm.calls and llm.calls[0]["provider"] == "self"   # her own brain authored it


def test_big_file_is_now_eligible(tmp_path: Path):
    # a file far past the old 24 KB linter-only cap is now eligible for a real redesign
    f = tmp_path / "big.py"
    body = "x = 1\n" + "# padding line to grow the file well past 24 KB\n" * 800   # ~38 KB
    f.write_text(body, encoding="utf-8")
    assert len(body.encode("utf-8")) > 24000
    fixed = body.replace("x = 1\n", "x = 2\n", 1)
    edit = LLMEditGenerator(llm=_StubLLM(fixed, provider="self"),
                            settings=_llm_settings()).generate(
        _W(id="code-todo-big.py-1", locus=f"{f}:1"))
    assert edit is not None and edit.after == fixed          # not pre-rejected by the size cap


def test_architecture_locus_module_name_resolves(tmp_path: Path):
    # architecture weaknesses carry a dotted module locus (e.g. 'nyxara.growth.weakness'); the
    # self-editor must resolve it to a real file so NYXARA can attempt the redesign
    from nyxara.growth.self_optimize import _resolve_locus_path

    assert _resolve_locus_path("nyxara.growth.weakness") is not None
    assert _resolve_locus_path("nyxara.kernel") is not None          # package __init__.py
    assert _resolve_locus_path("nyxara.does.not.exist") is None


# --------------------------------------------------------------------------- #
# End-to-end through the real Optimizer gauntlet (stubbed verdict)
# --------------------------------------------------------------------------- #
def test_llm_edit_kept_when_gauntlet_passes(tmp_path: Path):
    f = tmp_path / "g.py"
    f.write_text("def add(a, b):\n    return a+b\n", encoding="utf-8")
    fixed = "def add(a, b):\n    return a + b\n"
    s = _llm_settings()
    edit = LLMEditGenerator(llm=_StubLLM(fixed), settings=s).generate(
        _W(id="code-high_complexity-g.py-1", locus=f"{f}:1"))
    assert edit is not None
    # isolate the LLM-edit → gauntlet → keep MECHANISM (require_improvement=False); the separate
    # provable-improvement gate is exercised in tests/growth/test_improvement_proof.py.
    opt = Optimizer(settings=s, gauntlet=lambda files: GauntletResult(True, {}, "ok"),
                    require_improvement=False)
    out = opt.apply(edit)
    assert out.kept and not out.rolled_back
    assert f.read_text(encoding="utf-8") == fixed


def test_llm_edit_rolled_back_byte_for_byte_when_gauntlet_fails(tmp_path: Path):
    f = tmp_path / "g.py"
    original = "def add(a, b):\n    return a+b\n"
    f.write_text(original, encoding="utf-8")
    s = _llm_settings()
    edit = LLMEditGenerator(llm=_StubLLM("def add(a, b):\n    return a + b\n"),
                            settings=s).generate(
        _W(id="code-high_complexity-g.py-1", locus=f"{f}:1"))
    assert edit is not None
    opt = Optimizer(settings=s, gauntlet=lambda files: GauntletResult(False, {}, "stub fail"))
    out = opt.apply(edit)
    assert out.rolled_back and not out.kept
    assert f.read_text(encoding="utf-8") == original


# --------------------------------------------------------------------------- #
# Generator selection in the RSI loop (deterministic first, LLM fallback)
# --------------------------------------------------------------------------- #
def test_rsi_generate_edit_prefers_deterministic_then_llm(tmp_path: Path):
    from nyxara.growth.recursive_improvement import RecursiveSelfImprovement

    rsi = RecursiveSelfImprovement(settings=_llm_settings())
    det_gen = EditGenerator()
    llm_gen = LLMEditGenerator(llm=_StubLLM("def f(a, b):\n    return a + b\n"),
                               settings=_llm_settings())

    # deterministic weakness → deterministic transform is used (LLM never consulted)
    f1 = tmp_path / "d.py"
    f1.write_text("def f(a, b):\n    return not a in b\n", encoding="utf-8")
    det_w = _W(id="code-not_in-d.py-2", locus=f"{f1}:2", edit_strategy="deterministic")
    edit = rsi._generate_edit(det_w, det_gen, llm_gen)
    assert edit is not None and "a not in b" in edit.after
    assert llm_gen.llm.calls == []                       # LLM was not called

    # llm-strategy weakness the transforms cannot express → LLM fallback authors it
    f2 = tmp_path / "c.py"
    f2.write_text("def f(a, b):\n    return a+b\n", encoding="utf-8")
    llm_w = _W(id="code-high_complexity-c.py-1", locus=f"{f2}:1", edit_strategy="llm")
    edit2 = rsi._generate_edit(llm_w, det_gen, llm_gen)
    assert edit2 is not None and edit2.kind == "self:high_complexity"
    assert len(llm_gen.llm.calls) == 1                   # NYXARA's own model authored it once
    assert llm_gen.llm.calls[0]["provider"] == "self"    # …and it was her own brain, not an LLM


def test_strip_code_fence_plain_passthrough():
    assert _strip_code_fence("x = 1") == "x = 1\n"
    assert _strip_code_fence("x = 1\n") == "x = 1\n"


# --------------------------------------------------------------------------- #
# Genuine recursion: a kept edit re-reviews the file and chains the next fix
# --------------------------------------------------------------------------- #
def test_recursion_chains_deterministic_edits(tmp_path: Path):
    from nyxara.growth.recursive_improvement import (RecursiveSelfImprovement,
                                                     SelfImprovementReport)

    f = tmp_path / "r.py"
    # two independent E713 issues on different lines — only one can be fixed per review pass
    # (line numbers shift), so reaching both *proves* the re-review recursion fired
    f.write_text(
        "def f(a, b, c):\n"
        '    """A documented helper (so docstring hygiene is not a confound)."""\n'
        "    x = not a in b\n"
        "    y = not b in c\n"
        "    return x and y\n", encoding="utf-8")
    s = _llm_settings(llm_edit_recursion_depth=3)
    rsi = RecursiveSelfImprovement(settings=s)          # default root → tmp file gets abs locus
    opt = Optimizer(settings=s, gauntlet=lambda files: GauntletResult(True, {}, "ok"))
    report = SelfImprovementReport()
    init = _W(id="code-not_in-r.py-3", locus=f"{f}:3", edit_strategy="deterministic")
    rsi._enact_edits([init], EditGenerator(), opt, budget=10, report=report, llm_gen=None)

    text = f.read_text(encoding="utf-8")
    assert "not a in b" not in text and "not b in c" not in text
    assert "a not in b" in text and "b not in c" in text
    assert report.kept == 2                             # outer edit + 1 recursion edit

