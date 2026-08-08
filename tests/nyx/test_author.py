"""NYX V.02 · authoring — prose in, verified code out, or a refusal that names itself.

L-OMNI lowers functions she has *already measured as slow*; it cannot write what does not
exist. This is the other direction, and the tests here are mostly about the stages that stop
things: a program that runs but is wrong is thrown away, and a spec outside the synthesiser's
bounded families is refused by name rather than answered with a plausible-looking file.
"""

from __future__ import annotations

from nyxara.agency.self_coder import CodeSynthesizer
from nyxara.kernel.config import NyxConfig
from nyxara.nyx.author import STAGES, Author, Authored
from nyxara.nyx.brain import NyxBrain

TRIANGULAR = "a function that returns the 7th triangular number"
IMPOSSIBLE = "build a distributed consensus subsystem with leader election"


def _author(**kw) -> Author:
    kw.setdefault("lineage", False)      # a receipt is not what these tests are about
    return Author(**kw)


# --------------------------------------------------------------------------- #
# The whole gauntlet, on something she genuinely can derive
# --------------------------------------------------------------------------- #
def test_a_spec_she_can_derive_survives_every_stage():
    got = _author().write(TRIANGULAR, load=False)
    assert got.ok and got.value == 28 and got.verified
    assert got.stopped_at == ""
    assert [s.name for s in got.stages] == list(STAGES)


def test_the_program_she_wrote_is_real_python():
    got = _author().write(TRIANGULAR, load=False)
    assert "def triangular" in got.source
    compile(got.source, "<authored>", "exec")     # it is syntactically real, not a sketch


def test_the_answer_is_checked_against_an_independently_computed_one():
    got = _author().write("compute the 10th fibonacci number", load=False)
    assert got.expected == 55 and got.value == 55
    check = next(s for s in got.stages if s.name == "check")
    assert check.ok and "independently computed" in check.detail


def test_a_list_task_is_derivable_too():
    got = _author().write("sum the list [1, 2, 3, 4]", load=False)
    assert got.ok and got.value == 10 and got.family == "sequence"


def test_a_string_task_is_derivable_too():
    got = _author().write("reverse the string 'hello'", load=False)
    assert got.ok and got.value == "olleh"


# --------------------------------------------------------------------------- #
# The refusal — the point of the module
# --------------------------------------------------------------------------- #
def test_a_spec_outside_the_families_is_refused_by_name():
    got = _author().write(IMPOSSIBLE, load=False)
    assert got.ok is False and got.stopped_at == "synthesise"
    assert "could not derive a program" in got.refusal


def test_the_refusal_names_what_she_can_actually_do():
    refusal = _author().write(IMPOSSIBLE, load=False).refusal
    for family in ("arithmetic", "number_theory", "sequence", "string"):
        assert family in refusal


def test_the_refusal_says_why_no_file_is_produced():
    refusal = _author().write(IMPOSSIBLE, load=False).refusal
    assert "large pretrained model" in refusal
    assert "rather say that than hand you a file" in refusal


def test_a_refused_spec_produces_no_source_at_all():
    got = _author().write(IMPOSSIBLE, load=False)
    assert got.source == "" and got.path == "" and got.loaded is False


def test_an_empty_spec_is_refused():
    assert _author().write("").ok is False


# --------------------------------------------------------------------------- #
# A program that runs but is wrong is thrown away
# --------------------------------------------------------------------------- #
class _LyingSynthesiser:
    """Writes code that runs cleanly and returns the wrong answer — the dangerous case."""

    def synthesize(self, task):
        from nyxara.agency.self_coder import SynthesisResult
        return SynthesisResult(True, "result = 999\nprint(result)\n",
                               "local:test", "number_theory", expected=28)


def test_a_correct_looking_program_with_a_wrong_answer_never_ships():
    author = _author()
    author._synth = _LyingSynthesiser()
    got = author.write(TRIANGULAR, load=False)
    assert got.ok is False and got.stopped_at == "check"
    assert got.loaded is False and got.verified is False
    assert "runs but is wrong" in got.refusal
    assert "thrown away, not shipped" in got.refusal


class _CrashingSynthesiser:
    def synthesize(self, task):
        from nyxara.agency.self_coder import SynthesisResult
        return SynthesisResult(True, "raise ValueError('boom')\n", "local:test", "x", None)


def test_a_program_that_does_not_run_is_reported_not_shipped():
    author = _author()
    author._synth = _CrashingSynthesiser()
    got = author.write("anything", load=False)
    assert got.ok is False and got.stopped_at == "run"
    assert "does not run" in got.refusal


class _DangerousSynthesiser:
    def synthesize(self, task):
        from nyxara.agency.self_coder import SynthesisResult
        return SynthesisResult(True, "import socket\nresult = 1\n", "local:test", "x", 1)


def test_her_own_source_screen_can_refuse_what_she_wrote():
    author = _author()
    author._synth = _DangerousSynthesiser()
    got = author.write("anything", load=False)
    assert got.ok is False and got.stopped_at == "screen"
    assert "source screen refused" in got.refusal


class _HugeSynthesiser:
    def synthesize(self, task):
        from nyxara.agency.self_coder import SynthesisResult
        return SynthesisResult(True, "result = 1\n" + "# pad\n" * 5000, "local:test", "x", 1)


def test_an_oversized_program_is_refused_before_it_runs():
    author = _author(max_source_bytes=1024)
    author._synth = _HugeSynthesiser()
    got = author.write("anything", load=False)
    assert got.ok is False and got.stopped_at == "synthesise"


# --------------------------------------------------------------------------- #
# Reading the spec — an ambiguous one is asked about, not guessed at
# --------------------------------------------------------------------------- #
def _brain(**kw) -> NyxBrain:
    return NyxBrain(NyxConfig(**kw))


def test_a_contradictory_spec_is_asked_about_rather_than_built():
    brain = _brain()
    got = brain.author_code("write the function but don't write the function", load=False)
    assert got.ok is False and got.stopped_at == "read"
    assert "both to and not to" in got.refusal


def test_a_clear_spec_reads_through_to_synthesis():
    got = _brain().author_code(TRIANGULAR, load=False)
    assert got.ok and got.stages[0].name == "read" and got.stages[0].ok


def test_without_an_intent_reader_the_spec_is_taken_verbatim():
    got = _author().write(TRIANGULAR, load=False)
    assert "verbatim" in got.stages[0].detail


# --------------------------------------------------------------------------- #
# Loading and lineage
# --------------------------------------------------------------------------- #
def test_loading_puts_the_module_in_the_forged_root_and_nowhere_else():
    from nyxara.growth.module_loader import forged_root
    got = _author().write(TRIANGULAR, load=True, name="tri")
    assert got.path.startswith(str(forged_root()))
    assert got.loaded and got.module


def test_a_verified_program_can_be_kept_without_loading_it():
    got = _author().write(TRIANGULAR, load=False)
    assert got.ok and got.verified and got.loaded is False
    assert "not requested" in next(s for s in got.stages if s.name == "load").detail


def test_lineage_records_a_signed_entry():
    author = Author(lineage=True)
    got = author.write(TRIANGULAR, load=False)
    assert got.ok and got.lineage
    assert author.ledger().verify_chain()


# --------------------------------------------------------------------------- #
# Corrigibility
# --------------------------------------------------------------------------- #
class _Scrammed:
    def scrammed(self) -> bool:
        return True


def test_scram_stops_her_before_anything_is_written():
    got = _author().write(TRIANGULAR, oversight=_Scrammed())
    assert got.ok is False and got.source == ""
    assert "scrammed" in got.refusal


def test_an_unreadable_oversight_is_treated_as_stop():
    class _Broken:
        @property
        def paused(self):
            raise RuntimeError("unreadable")
    assert Author._stopped(_Broken()) is True


def test_disabling_the_author_leaves_v01_behaviour():
    brain = _brain(author_enabled=False)
    assert brain.author is None and brain.author_code("anything") is None


# --------------------------------------------------------------------------- #
# The widened synthesiser family
# --------------------------------------------------------------------------- #
def test_closed_form_sequences_are_derivable():
    synth = CodeSynthesizer()
    assert synth.synthesize("the 7th triangular number").expected == 28
    assert synth.synthesize("the 9th square number").expected == 81


def test_a_closed_form_program_uses_the_formula_not_a_loop():
    source = CodeSynthesizer().synthesize("the 100th triangular number").source
    assert "for " not in source and "while " not in source


def test_the_synthesiser_still_misses_honestly_outside_its_families():
    got = CodeSynthesizer().synthesize(IMPOSSIBLE)
    assert got.ok is False and got.origin == "none" and got.note


# --------------------------------------------------------------------------- #
# Introspection, fail-soft, persistence
# --------------------------------------------------------------------------- #
def test_stats_say_what_the_gauntlet_is_for():
    stats = _author().stats()
    assert stats["stages"] == list(STAGES)
    assert "thrown away rather than shipped" in stats["note"]
    assert "refused by name" in stats["note"]


def test_counters_separate_shipped_from_refused():
    author = _author()
    author.write(TRIANGULAR, load=False)
    author.write(IMPOSSIBLE, load=False)
    stats = author.stats()
    assert stats["shipped"] == 1 and stats["refused"] == 1 and stats["attempts"] == 2


def test_everything_is_fail_soft_on_junk():
    author = _author()
    assert author.write(None).ok is False
    assert author.load_dict(None) is False
    assert Authored().to_dict()["ok"] is False


def test_the_record_survives_a_restart():
    author = _author()
    author.write(TRIANGULAR, load=False)
    other = _author()
    assert other.load_dict(author.to_dict())
    assert other.shipped == 1


def test_describe_states_the_boundary_up_front():
    said = _author().describe()
    assert "families" in said and "refuses and names what she could not derive" in said
