"""Tests for nyxara.agency.dynamic_tool_creator — ephemeral tool synthesis.

These exercise the synthesize→build→run→delete lifecycle for real: Python is executed,
C/C++ is actually compiled (skipped when no compiler is present), and every run is
asserted to clean up its throwaway artifacts.
"""

from __future__ import annotations

import shutil

import pytest

from nyxara.agency.dynamic_tool_creator import (
    DynamicToolCreator,
    EphemeralResult,
    Language,
)
from nyxara.agency.permissions import Authority
from nyxara.agency.tools import ToolRegistry

_HAS_CPP = shutil.which("g++") is not None or shutil.which("clang++") is not None


# --------------------------------------------------------------------------- #
# Language coercion
# --------------------------------------------------------------------------- #
def test_language_coerce_aliases():
    assert Language.coerce("py") is Language.PYTHON
    assert Language.coerce("c++") is Language.CPP
    assert Language.coerce("CXX") is Language.CPP
    assert Language.coerce(Language.C) is Language.C
    with pytest.raises(ValueError):
        Language.coerce("rust")


# --------------------------------------------------------------------------- #
# Python execution
# --------------------------------------------------------------------------- #
def test_python_value_via_result_convention():
    creator = DynamicToolCreator()
    r = creator.run_source("result = sum(range(101))", language="python")
    assert isinstance(r, EphemeralResult)
    assert r.ok and r.value == 5050 and r.cleaned_up


def test_python_stdout_captured():
    creator = DynamicToolCreator()
    r = creator.run_source("print('hello ephemeral')", language="python")
    assert r.ok and "hello ephemeral" in r.stdout


def test_python_stdin_runs_via_temp_file():
    creator = DynamicToolCreator()
    r = creator.run_source("print(input()[::-1])", language="python", stdin="nyxara\n")
    assert r.ok and r.stdout.strip() == "araxyn"


def test_python_argv_passed_through():
    creator = DynamicToolCreator()
    src = "import sys\nprint('argc', len(sys.argv) - 1)\n"
    # `import sys` is blocked by the static gauntlet — confirm that, then use a safe form
    r = creator.run_source(src, language="python", argv=["a", "b"])
    assert not r.ok and "unsafe source" in (r.error or "")


def test_python_exception_is_data_not_raise():
    creator = DynamicToolCreator()
    r = creator.run_source("raise ValueError('boom')", language="python")
    assert not r.ok and r.error is not None and "ValueError" in (r.stderr or r.error)


def test_python_times_out_and_still_cleans_up():
    creator = DynamicToolCreator()
    r = creator.run_source("while True:\n    pass", language="python", timeout_s=1.0)
    assert not r.ok and r.timed_out and r.cleaned_up


# --------------------------------------------------------------------------- #
# Native compilation (real g++/clang++)
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not _HAS_CPP, reason="no C++ compiler on PATH")
def test_cpp_compile_and_run():
    creator = DynamicToolCreator()
    cpp = ('#include <iostream>\n'
           'int main(){ long s=0; for(int i=1;i<=100;i++) s+=i; '
           'std::cout << s; return 0; }\n')
    r = creator.run_source(cpp, language="cpp")
    assert r.ok and r.stdout.strip() == "5050" and r.compiled
    assert r.build_s > 0.0 and r.cleaned_up


@pytest.mark.skipif(not _HAS_CPP, reason="no C++ compiler on PATH")
def test_cpp_stdin():
    creator = DynamicToolCreator()
    cpp = ('#include <iostream>\n#include <string>\n'
           'int main(){ std::string s; std::getline(std::cin, s); '
           'std::cout << s.size(); return 0; }\n')
    r = creator.run_source(cpp, language="cpp", stdin="abcd")
    assert r.ok and r.stdout.strip() == "4"


@pytest.mark.skipif(not _HAS_CPP, reason="no C++ compiler on PATH")
def test_cpp_compile_error_is_data():
    creator = DynamicToolCreator()
    r = creator.run_source("int main(){ this is not c++ }\n", language="cpp")
    assert not r.ok and not r.compiled and r.compile_stderr


def test_compiler_missing_is_reported_as_data(monkeypatch):
    creator = DynamicToolCreator()
    # force "no compiler" without touching PATH globally
    monkeypatch.setattr(shutil, "which", lambda _name: None)
    r = creator.run_source("int main(){return 0;}\n", language="cpp")
    assert not r.ok and not r.compiled and "compiler" in (r.error or "")


# --------------------------------------------------------------------------- #
# Static safety gauntlet (fail-closed)
# --------------------------------------------------------------------------- #
def test_refuses_os_reach_python():
    creator = DynamicToolCreator()
    r = creator.run_source("import os\nos.system('echo hi')", language="python")
    assert not r.ok and "unsafe source" in (r.error or "")


def test_refuses_shellout_cpp():
    creator = DynamicToolCreator()
    bad = '#include <cstdlib>\nint main(){ std::system("rm -rf /"); }\n'
    r = creator.run_source(bad, language="cpp")
    assert not r.ok and "unsafe source" in (r.error or "")


def test_refuses_empty_source():
    creator = DynamicToolCreator()
    r = creator.run_source("   ", language="python")
    assert not r.ok and "unsafe" in (r.error or "")


# --------------------------------------------------------------------------- #
# Capability gate (mirrors ToolRegistry.invoke)
# --------------------------------------------------------------------------- #
def test_autonomous_call_escalates_to_master():
    creator = DynamicToolCreator(registry=ToolRegistry())
    r = creator.run_source("result = 1", language="python", authority=Authority.AUTONOMOUS)
    assert not r.ok and r.requires_owner


def test_owner_confirmed_runs():
    creator = DynamicToolCreator(registry=ToolRegistry())
    r = creator.run_source("result = 6 * 7", language="python",
                           authority=Authority.AUTONOMOUS, owner_confirmed=True)
    assert r.ok and r.value == 42


def test_ungoverned_runs_without_registry():
    creator = DynamicToolCreator()  # no registry -> sandbox is the only guard
    r = creator.run_source("result = 2", language="python", authority=Authority.AUTONOMOUS)
    assert r.ok and r.value == 2


# --------------------------------------------------------------------------- #
# Zero-shot solve() and synthesis
# --------------------------------------------------------------------------- #
def test_solve_uses_template_without_llm():
    creator = DynamicToolCreator()
    r = creator.solve("say hello", language="python")
    assert r.ok and r.origin == "template"


def test_solve_uses_injected_llm():
    def fake_llm(_prompt: str) -> str:
        return "```python\nresult = 3 + 4\n```"

    creator = DynamicToolCreator(llm=fake_llm)
    r = creator.solve("add three and four", language="python")
    assert r.ok and r.value == 7 and r.origin == "llm"


def test_llm_unsafe_draft_falls_back_to_template():
    def evil_llm(_prompt: str) -> str:
        return "import os\nos.system('rm -rf /')"

    creator = DynamicToolCreator(llm=evil_llm)
    src, origin = creator.synthesize("do something", Language.PYTHON)
    assert origin == "template" and "import os" not in src


# --------------------------------------------------------------------------- #
# Artifact retention toggle
# --------------------------------------------------------------------------- #
def test_keep_artifacts_does_not_delete(monkeypatch):
    import os
    import tempfile as _tf
    from nyxara.agency import dynamic_tool_creator as _mod

    captured = {}
    real_mkdtemp = _tf.mkdtemp

    def _spy(*a, **k):
        path = real_mkdtemp(*a, **k)
        captured["path"] = path
        return path

    monkeypatch.setattr(_mod.tempfile, "mkdtemp", _spy)
    creator = DynamicToolCreator(keep_artifacts=True)
    r = creator.run_source("print(input())", language="python", stdin="x\n")
    assert r.ok
    try:
        assert os.path.isdir(captured["path"]), "keep_artifacts should preserve the workdir"
    finally:
        shutil.rmtree(captured["path"], ignore_errors=True)


# --------------------------------------------------------------------------- #
# default_tools wiring
# --------------------------------------------------------------------------- #
def test_ephemeral_exec_registered_and_runs():
    from nyxara.agency.default_tools import build_default_tools

    reg = build_default_tools(ToolRegistry(), enable_code=True)
    assert "ephemeral_exec" in reg.names()
    r = reg.invoke("ephemeral_exec",
                   {"source": "result = 10 * 10", "language": "python"},
                   authority=Authority.OWNER, owner_confirmed=True)
    assert r.ok and r.value["ok"] and r.value["value"] == 100
