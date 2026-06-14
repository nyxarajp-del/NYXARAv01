"""Tests for nyxara.social.culture."""

from __future__ import annotations


from nyxara.social.culture import CodeProfile, CultureSystem, Register, StyleGuidance
from nyxara.social.persons import InteractionKind, Roster


def _culture():
    return CultureSystem(Roster(owner_name="Jaypal Khoja", owner_aliases=["JP"]))


# -------------------- code detection -------------------- #
def test_detect_english():
    assert _culture().detect_code("What is the network status?").language == "en"


def test_detect_devanagari_hindi():
    code = _culture().detect_code("नेटवर्क की स्थिति क्या है?")
    assert code.language == "hi"
    assert code.romanized is False


def test_detect_romanized_hinglish():
    code = _culture().detect_code("yaar kal ka plan ready hai kya?")
    assert code.language == "hinglish"
    assert code.romanized is True


def test_detect_hinglish_mix():
    assert _culture().detect_code("kal meeting hai, please ready rehna").language == "hinglish"


def test_detect_script_mixed_hinglish():
    code = _culture().detect_code("network की स्थिति check karo please")
    assert code.language == "hinglish"


def test_code_profile_shape():
    code = _culture().detect_code("hello world")
    assert isinstance(code, CodeProfile)
    assert 0.0 <= code.hindi_ratio <= 1.0


# -------------------- register inference -------------------- #
def test_register_formal():
    assert _culture().infer_register("Sir, would you kindly review this?") is Register.FORMAL


def test_register_formal_aap():
    assert _culture().infer_register("aap kaise hain ji?") is Register.FORMAL


def test_register_casual():
    assert _culture().infer_register("arre yaar, chill kar") is Register.CASUAL


def test_register_neutral():
    assert _culture().infer_register("The report is ready.") is Register.NEUTRAL


# -------------------- person-aware -------------------- #
def test_register_for_owner_casual():
    assert _culture().register_for("JP") is Register.CASUAL


def test_register_for_stranger_formal():
    c = _culture()
    c.roster.get_or_create("Stranger")
    assert c.register_for("Stranger") is Register.FORMAL


def test_register_for_ally_neutral():
    c = _culture()
    for _ in range(30):
        c.roster.record("Friend", InteractionKind.HELP, sentiment=0.9)
    assert c.register_for("Friend") is Register.NEUTRAL


def test_honorific_owner():
    assert _culture().honorific("JP") == "Master"


def test_honorific_stranger():
    c = _culture()
    c.roster.get_or_create("Stranger")
    assert c.honorific("Stranger") == "Sir/Madam"


def test_honorific_ally_uses_name():
    c = _culture()
    for _ in range(30):
        c.roster.record("Alice", InteractionKind.HELP, sentiment=0.9)
    assert c.honorific("Alice") == "Alice"


# -------------------- politeness -------------------- #
def test_politeness_positive():
    assert _culture().politeness(face_threat=0.2, closeness=0.9) == "positive"


def test_politeness_negative():
    assert _culture().politeness(face_threat=0.8, closeness=0.1) == "negative"


def test_politeness_bald():
    assert _culture().politeness(face_threat=0.1, closeness=0.2) == "bald"


# -------------------- norms -------------------- #
def test_norms_professional():
    assert "use formal register" in _culture().norms("professional")


def test_norms_unknown_setting_defaults_public():
    assert _culture().norms("unknown") == _culture().norms("public")


# -------------------- adapt -------------------- #
def test_adapt_owner_hinglish():
    g = _culture().adapt(person="JP", text="kal ka kaam ready hai kya?")
    assert g.language == "hinglish"
    assert g.honorific == "Master"
    assert "Hinglish" in g.fragment


def test_adapt_stranger_formal_deferential():
    c = _culture()
    c.roster.get_or_create("Stranger")
    g = c.adapt(person="Stranger", text="Hello, who are you?", face_threat=0.7)
    assert g.register is Register.FORMAL
    assert g.honorific == "Sir/Madam"
    assert g.politeness == "negative"


def test_adapt_owner_english():
    g = _culture().adapt(person="JP", text="What is the status?")
    assert g.language == "en"
    assert g.honorific == "Master"


def test_adapt_no_person():
    g = _culture().adapt(text="Hello there")
    assert isinstance(g, StyleGuidance)
    assert g.honorific == "you"


def test_adapt_no_text_uses_person_register():
    c = _culture()
    c.roster.get_or_create("Stranger")
    g = c.adapt(person="Stranger")
    assert g.register is Register.FORMAL


def test_adapt_fragment_mentions_register():
    g = _culture().adapt(person="JP", text="hi")
    assert g.register.value in g.fragment


def test_guidance_to_dict():
    d = _culture().adapt(person="JP", text="hi").to_dict()
    assert "register" in d and "language" in d and "honorific" in d


def test_match_code_mirrors():
    c = _culture()
    assert c.match_code("yaar ye scan abhi check karo").language == "hinglish"
    assert c.match_code("What time is it?").language == "en"
