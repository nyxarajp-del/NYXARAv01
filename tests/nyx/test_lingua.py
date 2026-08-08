"""NYX V.02 · the tongue — scripts, code-mixing, function words, register, transliteration.

The measurement this file exists to pin: before ``lingua``,
``concepts_in("गुरुत्वाकर्षण सेब को खींचता है")`` returned ``[]`` and Hinglish grammar became
the hubs of the concept graph. Both are asserted against here.
"""

from __future__ import annotations

from nyxara.nyx.graph import DynamicNeuralGraph, concepts_in
from nyxara.nyx.lingua import (
    Lingua,
    content_tokens,
    detect_language,
    is_function_word,
    register_of,
    roman_key,
    script_of,
    tokenize,
    transliterate,
)

DEVA = "गुरुत्वाकर्षण सेब को खींचता है"          # "gravity pulls the apple"
HINGLISH = "mera code fix kar do lekin pehle test chala"


# --------------------------------------------------------------------------- #
# Scripts — no alphabet is privileged
# --------------------------------------------------------------------------- #
def test_script_of_reads_the_character_database():
    assert script_of("a") == "latin"
    assert script_of("क") == "devanagari"
    assert script_of("д") == "cyrillic"
    assert script_of("7") == "digit"
    assert script_of("😂") == "emoji"
    assert script_of("!") == "common"


def test_script_of_is_fail_soft():
    assert script_of("") == "unknown"


def test_tokenizer_tags_every_token_with_its_script():
    toks = tokenize("NYX पढ़ती 3.14 😂 ok!", keep_punct=True)
    by_kind = {t.text: (t.kind, t.script) for t in toks}
    assert by_kind["NYX"] == ("word", "latin")
    assert by_kind["3.14"] == ("number", "digit")
    assert by_kind["😂"] == ("emoji", "emoji")
    assert by_kind["!"] == ("punct", "common")
    assert any(t.script == "devanagari" and t.kind == "word" for t in toks)


def test_punctuation_is_dropped_unless_asked_for():
    assert not any(t.kind == "punct" for t in tokenize("hello, world!"))
    assert any(t.kind == "punct" for t in tokenize("hello, world!", keep_punct=True))


# --------------------------------------------------------------------------- #
# Gap 2, measured: Devanagari used to be invisible
# --------------------------------------------------------------------------- #
def test_devanagari_now_mints_concepts():
    """V.01 returned [] for this exact string. That is the whole reason this module exists."""
    got = content_tokens(DEVA)
    assert got, "Devanagari must produce concepts, not silence"
    assert "गुरुत्वाकर्षण" in got and "सेब" in got


def test_devanagari_function_words_are_not_concepts():
    got = content_tokens(DEVA)
    assert "को" not in got and "है" not in got


def test_graph_grows_devanagari_nodes():
    g = DynamicNeuralGraph(max_nodes=64, max_edges=256, decay_rate=0.0)
    g.activate(DEVA)
    assert "गुरुत्वाकर्षण" in g
    assert g.weight("गुरुत्वाकर्षण", "सेब") > 0.0


def test_concepts_in_routes_through_the_tongue():
    assert concepts_in(DEVA) == content_tokens(DEVA)


# --------------------------------------------------------------------------- #
# Hinglish function words must not become graph hubs
# --------------------------------------------------------------------------- #
def test_hinglish_grammar_is_not_a_concept():
    got = content_tokens("ye kya hai aur ye kaise kaam karta hai")
    for grammar in ("ye", "kya", "hai", "aur", "kaise"):
        assert grammar not in got
    assert "kaam" in got


def test_hinglish_words_never_become_graph_hubs():
    """The measured failure: 'hai'/'kya'/'aur' co-occur with everything, so they hubbed."""
    g = DynamicNeuralGraph(max_nodes=128, max_edges=512, decay_rate=0.0)
    for line in ("ye kya hai aur kaise chalta hai",
                 "wo kya hai aur kyun rukta hai",
                 "server kya hai aur cache kaise banta hai"):
        g.activate(line)
    for grammar in ("hai", "kya", "aur"):
        assert grammar not in g


def test_ambiguous_hinglish_words_stay_english_in_an_english_turn():
    """'mat' is *don't* in Hindi and a doormat in English. Context decides, not the list."""
    assert "mat" in content_tokens("The cat sat on the warm mat")
    assert "mat" not in content_tokens("aisa mat karo bhai")


def test_english_extraction_is_unchanged_from_v01():
    assert content_tokens("The cat sat on the warm mat") == ["cat", "sat", "warm", "mat"]


def test_numbers_are_measurements_not_concepts():
    assert content_tokens("host mem rose to 0.0906") == ["host", "mem", "rose"]


def test_is_function_word_spans_all_three_registers():
    assert is_function_word("the") and is_function_word("kyunki") and is_function_word("और")
    assert not is_function_word("gravity")


# --------------------------------------------------------------------------- #
# Language identification — evidence, or an abstention
# --------------------------------------------------------------------------- #
def test_devanagari_is_read_as_hindi_not_unknown():
    """``senses.nlp.detect_language`` answers 'unknown' here. This one has evidence."""
    assert Lingua().read(DEVA).primary == "hi"
    assert detect_language(DEVA) == "hi"


def test_romanised_hinglish_is_first_class():
    read = Lingua().read(HINGLISH)
    assert read.primary == "hi-Latn"
    assert read.hinglish is True


def test_pure_devanagari_is_hindi_not_hinglish():
    assert Lingua().read(DEVA).hinglish is False


def test_two_scripts_in_one_turn_is_code_mixing():
    read = Lingua().read("गुरुत्वाकर्षण pulls the apple down")
    assert read.code_mixed is True
    assert set(read.scripts) == {"devanagari", "latin"}


def test_a_latin_content_word_is_not_evidence_of_a_language():
    """No lexicon ships, so 'code' proves nothing — and must not be labelled as if it did."""
    read = Lingua().read(HINGLISH)
    code = next(t for t in read.tokens if t.lower == "code")
    assert code.evidence is False
    assert code.lang == read.primary          # inherited, and flagged as inherited


def test_language_is_abstained_on_when_nothing_is_evidence():
    assert Lingua().read("zxqv pflmr").primary == "und"


# --------------------------------------------------------------------------- #
# Transliteration bridge
# --------------------------------------------------------------------------- #
def test_transliteration_deletes_the_word_final_schwa():
    assert transliterate("सेब") == "seb"          # not "seba"
    assert transliterate("गुरुत्वाकर्षण") == "gurutvaakarshan"


def test_roman_key_bridges_the_two_scripts():
    assert roman_key("गुरुत्वाकर्षण") == roman_key("gurutvakarshan") == "gurutvakarshan"
    assert roman_key("सेब") == roman_key("seb") == "seb"


def test_roman_key_folds_spelling_variants():
    assert roman_key("waqt") == roman_key("vaqt")
    assert roman_key("phal") == roman_key("fal")


def test_same_word_is_orthographic_only():
    lin = Lingua()
    assert lin.same_word("गुरुत्वाकर्षण", "gurutvakarshan")
    # Meaning is semantics' job, not the tongue's — and the tongue does not pretend otherwise.
    assert not lin.same_word("गुरुत्वाकर्षण", "gravity")


def test_transliterate_leaves_latin_alone():
    assert transliterate("gravity") == "gravity"


# --------------------------------------------------------------------------- #
# Register and tone — recorded, never generated
# --------------------------------------------------------------------------- #
def test_politeness_and_formality_are_read_off_the_surface():
    polite = register_of("Kindly explain how gravity works, sir.")
    blunt = register_of("arre yaar just fix it lol")
    assert polite.politeness > blunt.politeness
    assert polite.formality > blunt.formality
    assert "polite" in polite.markers and "casual" in blunt.markers


def test_emphasis_counts_capitals_and_repetition():
    assert register_of("this is TOTALLY broken!!!").emphasis > register_of("this is broken").emphasis


def test_humor_notices_laughter_and_emoji():
    assert register_of("haha 😂 nice").humor > 0.0
    assert register_of("the server restarted").humor == 0.0


# --------------------------------------------------------------------------- #
# Fail-soft and persistence — the house rules
# --------------------------------------------------------------------------- #
def test_everything_is_fail_soft_on_junk():
    assert content_tokens("") == [] and content_tokens(None) == []
    assert tokenize(None) == []
    assert transliterate(None) == ""
    assert roman_key(None) == ""
    assert Lingua().read(None).tokens == []
    assert register_of(None).markers == []


def test_stats_report_what_she_has_actually_met():
    lin = Lingua()
    lin.read(DEVA)
    lin.read("the cat sat on the mat")
    stats = lin.stats()
    assert stats["reads"] == 2
    assert stats["scripts"]["devanagari"] > 0 and stats["scripts"]["latin"] > 0
    assert "hi" in stats["languages"]


def test_round_trips_through_a_sidecar():
    lin = Lingua()
    lin.read(DEVA)
    lin.read(HINGLISH)
    other = Lingua()
    assert other.load_dict(lin.to_dict())
    assert other.stats()["scripts"] == lin.stats()["scripts"]
    assert other.reads == lin.reads


def test_a_corrupt_sidecar_never_raises():
    lin = Lingua()
    assert lin.load_dict(None) is False
    assert lin.load_dict("junk") is False
    assert lin.reads == 0
