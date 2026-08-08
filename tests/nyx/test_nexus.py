"""NYX V.01 · L-NEXUS-OMNI — her own notation, which must run, and must always translate."""

from __future__ import annotations

import pytest

from nyxara.kernel.config import get_settings
from nyxara.nyx.brain import NyxBrain
from nyxara.nyx.nexus import Notation, OntologyGenesis


@pytest.fixture
def nexus():
    return OntologyGenesis()


# -------------------- 1. she mints her own symbols -------------------- #
def test_a_notation_is_private_symbols_not_english_renamed(nexus):
    got = nexus.invent("thermo", ["force", "temperature", "n"])
    assert got is not None and len(got.symbols) == 3
    for concept, glyph in got.symbols.items():
        assert glyph != concept
        assert not glyph.isascii()               # a glyph, not a word in disguise


def test_every_concept_gets_a_distinct_symbol(nexus):
    got = nexus.invent("many", ["a", "b", "c", "d", "e"])
    assert len(set(got.symbols.values())) == len(got.symbols)


def test_repeated_concepts_are_minted_once(nexus):
    got = nexus.invent("dupes", ["force", "force", "n"])
    assert len(got.symbols) == 2


def test_inventing_nothing_is_nothing(nexus):
    assert nexus.invent("", ["a"]) is None
    assert nexus.invent("x", []) is None


def test_notations_are_capped(nexus):
    small = OntologyGenesis(max_notations=2)
    for i in range(6):
        small.invent(f"n{i}", ["a", "b"])
    assert len(small.notations) <= 2


# -------------------- 2 & 3. it must actually run -------------------- #
def test_a_statement_is_written_in_her_symbols_and_executes(nexus):
    notation = nexus.invent("thermo", ["temperature", "n"])
    got = nexus.express(notation, "3 * (temperature * n) + 2",
                        bindings={"temperature": 10.0, "n": 4.0})
    assert got.realised is True
    assert got.value == pytest.approx(122.0)
    assert got.program                            # real bytecode, not a rendering


def test_her_written_form_uses_the_glyphs_she_minted(nexus):
    notation = nexus.invent("thermo", ["temperature", "n"])
    got = nexus.express(notation, "temperature * n", bindings={"temperature": 2.0, "n": 3.0})
    assert notation.symbols["temperature"] in got.written
    assert "temperature" not in got.written


def test_a_statement_that_cannot_run_is_refused(nexus):
    """A notation that cannot be executed is not a language — it is decoration."""
    notation = nexus.invent("thermo", ["temperature", "n"])
    got = nexus.express(notation, "temperature * n")          # no bindings, nothing to run
    assert got.realised is False and "not a language" in got.reason


def test_unparseable_input_is_refused(nexus):
    notation = nexus.invent("thermo", ["x"])
    assert nexus.express(notation, "3 + + )").realised is False
    assert nexus.express(notation, "").realised is False


def test_an_unknown_notation_says_so(nexus):
    assert nexus.express("no-such-notation", "1 + 1").reason == "no such notation"


def test_arithmetic_is_correct_including_precedence(nexus):
    notation = nexus.invent("m", ["a"])
    assert nexus.express(notation, "2 + 3 * 4").value == pytest.approx(14.0)
    assert nexus.express(notation, "(2 + 3) * 4").value == pytest.approx(20.0)
    assert nexus.express(notation, "-(2 + 3)").value == pytest.approx(-5.0)
    assert nexus.express(notation, "10 / 4").value == pytest.approx(2.5)


# -------------------- 4. translation is mandatory -------------------- #
def test_whatever_reaches_a_person_is_translated(nexus):
    notation = nexus.invent("thermo", ["temperature"])
    got = nexus.express(notation, "temperature * 2", bindings={"temperature": 5.0})
    assert got.translated and "temperature" in got.translated


def test_the_translation_is_labelled_a_lossy_projection(nexus):
    notation = nexus.invent("thermo", ["temperature"])
    got = nexus.express(notation, "temperature * 2", bindings={"temperature": 5.0})
    rendered = nexus.translate(got)
    assert "lossy projection" in rendered and got.written in rendered


def test_the_guard_is_real_and_can_be_proven_off():
    """The flag exists so a test can show the guard is enforced, not assumed."""
    quiet = OntologyGenesis(translate_always=False)
    notation = quiet.invent("thermo", ["temperature"])
    got = quiet.express(notation, "temperature * 2", bindings={"temperature": 5.0})
    assert got.translated == ""


# -------------------- alternative logics -------------------- #
def test_an_alternative_axiom_system_can_be_forged(nexus):
    system = nexus.alternative_axioms("paraconsistent")
    assert system is not None and getattr(system, "name", "") == "paraconsistent"


def test_asking_a_system_with_nothing_forged_is_not_an_error(nexus):
    assert nexus.holds(None, "anything") is None


# -------------------- state -------------------- #
def test_statements_are_bounded(nexus):
    notation = nexus.invent("m", ["a"])
    for i in range(50):
        nexus.express(notation, f"{i} + 1")
    assert len(nexus.statements) <= 32


def test_notations_round_trip(nexus):
    nexus.invent("thermo", ["force", "temperature"])
    data = nexus.to_dict()

    other = OntologyGenesis()
    assert other.load_dict(data) is True
    assert other.notations["thermo"].symbols == nexus.notations["thermo"].symbols


def test_corrupt_state_does_not_block(nexus):
    assert nexus.load_dict(None) is False
    assert nexus.load_dict({"notations": "nonsense"}) is True
    assert nexus.notations == {}


def test_the_inverse_mapping_is_available():
    notation = Notation(name="x", symbols={"force": "◇"})
    assert notation.inverse == {"◇": "force"}


# -------------------- wired into the brain -------------------- #
def test_a_law_she_established_is_given_symbols_of_her_own():
    b = NyxBrain(get_settings().nyx.model_copy(
        update={"holo_dim": 1024, "episteme_trials": 20, "aura_enabled": False,
                "car_enabled": False}))
    if b.nexus is None or b.episteme is None or not b.episteme.available():
        pytest.skip("no sandbox or notation engine available")
    finding = b.discover()
    if finding is None or not finding.promoted:
        pytest.skip("this run did not establish a law")
    assert b.nexus.notations
    statement = b.nexus.statements[-1]
    assert statement.realised is True             # it compiled and ran
    assert statement.translated                    # and it came back to words


def test_a_brain_without_nexus_still_works():
    b = NyxBrain(get_settings().nyx.model_copy(
        update={"holo_dim": 1024, "nexus_enabled": False}))
    assert b.nexus is None
    assert b.think("what pulls an apple down") is not None
