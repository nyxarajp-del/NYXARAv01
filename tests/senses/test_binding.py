"""Tests for nyxara.senses.binding."""

from __future__ import annotations

from nyxara.senses.binding import (Association, Binder, Modality, Percept,
                                   PerceptualFrame, Provenance, text_simhash)
from nyxara.senses.vision import hamming


# -------------------- text_simhash -------------------- #
def test_simhash_deterministic():
    assert text_simhash("the quick brown fox") == text_simhash("the quick brown fox")


def test_simhash_empty():
    assert text_simhash("") == 0


def test_simhash_near_duplicate_close():
    a = text_simhash("the quick brown fox jumps over the lazy dog")
    b = text_simhash("the quick brown fox jumps over the lazy cat")
    c = text_simhash("completely unrelated sentence about quantum physics maths")
    assert hamming(a, b) < hamming(a, c)


# -------------------- Provenance -------------------- #
def test_provenance_trusted():
    assert Provenance("owner", 0.9).trusted
    assert not Provenance("web", 0.2).trusted


def test_provenance_to_dict():
    d = Provenance("src", 0.7).to_dict()
    assert d["trusted"] is True and d["source"] == "src"


# -------------------- Percept factories -------------------- #
def test_from_text():
    p = Percept.from_text("hello world", source="s", entities=["X"], tags=["t"])
    assert p.modality is Modality.TEXT and p.fingerprint != 0
    assert p.entities == ["X"] and "t" in p.tags


def test_from_analysis():
    class FakeAnalysis:
        text = "Acme Corp is a company. It builds firewalls."
        summary = ["Acme Corp is a company."]
        entities = [type("E", (), {"text": "Acme Corp"})()]
        intent = type("I", (), {"kind": "statement"})()
        sentiment = type("S", (), {"label": "neutral"})()
        language = "en"

    p = Percept.from_analysis(FakeAnalysis(), source="owner")
    assert p.modality is Modality.TEXT
    assert "Acme Corp" in p.entities
    assert "statement" in p.tags


def test_from_webpage():
    class FakePage:
        url = "https://x.com"
        title = "Title"
        text = "some body text"
        headings = ["H1"]
        injection = type("R", (), {"suspicious": True})()

    p = Percept.from_webpage(FakePage())
    assert p.modality is Modality.WEB
    assert "injection" in p.tags and p.data["injection"] is True


def test_from_document():
    class FakeDoc:
        source = "f.md"
        text = "doc text here"
        doc_type = type("DT", (), {"value": "markdown"})()
        metadata = {"headings": ["Intro"]}

    p = Percept.from_document(FakeDoc())
    assert p.modality is Modality.DOCUMENT and "markdown" in p.tags


def test_from_image():
    class FakeInfo:
        width, height, format = 800, 600, "PNG"

    class FakeImg:
        info = FakeInfo()
        perceptual_hash = 0x1234
        average_hash = 0xABCD
        ocr_text = None

    p = Percept.from_image(FakeImg(), source="img.png")
    assert p.modality is Modality.IMAGE and p.fingerprint == 0x1234
    assert "800x600" in p.content


def test_from_audio():
    class FakeInfo:
        duration = 3.5

    class FakeAudio:
        info = FakeInfo()
        fingerprint = 0x9999
        silence_ratio = 0.9
        transcript = None

    p = Percept.from_audio(FakeAudio(), source="a.wav")
    assert p.modality is Modality.AUDIO and "silent" in p.tags


def test_percept_to_dict():
    p = Percept.from_text("hi", source="s")
    assert p.to_dict()["modality"] == "text"


# -------------------- PerceptualFrame: add / dedup -------------------- #
def test_frame_add_new():
    f = PerceptualFrame()
    p = Percept.from_text("unique content here", source="s")
    bound, is_new = f.add(p)
    assert is_new and len(f) == 1


def test_frame_dedups_near_duplicate():
    f = PerceptualFrame()
    f.add(Percept.from_text("the quick brown fox jumps over the lazy dog", source="s"))
    bound, is_new = f.add(Percept.from_text(
        "the quick brown fox jumps over the lazy dog", source="s2"))
    assert not is_new and bound.count == 2
    assert len(f) == 1


def test_frame_keeps_distinct():
    f = PerceptualFrame()
    f.add(Percept.from_text("alpha beta gamma delta epsilon", source="s"))
    f.add(Percept.from_text("quantum mechanics relativity physics maths", source="s"))
    assert len(f) == 2


def test_frame_dedup_only_same_modality():
    f = PerceptualFrame()
    # same fingerprint but different modality -> not a duplicate
    p1 = Percept(Modality.TEXT, "a", Provenance("s"), fingerprint=0xFF)
    p2 = Percept(Modality.IMAGE, "b", Provenance("s"), fingerprint=0xFF)
    f.add(p1)
    _, is_new = f.add(p2)
    assert is_new and len(f) == 2


def test_frame_zero_fingerprint_never_dedups():
    f = PerceptualFrame()
    f.add(Percept(Modality.TEXT, "a", Provenance("s"), fingerprint=0))
    _, is_new = f.add(Percept(Modality.TEXT, "b", Provenance("s"), fingerprint=0))
    assert is_new and len(f) == 2


# -------------------- frame queries -------------------- #
def test_by_modality():
    f = PerceptualFrame()
    f.add(Percept(Modality.TEXT, "a", Provenance("s"), fingerprint=1))
    f.add(Percept(Modality.IMAGE, "b", Provenance("s"), fingerprint=2))
    assert len(f.by_modality(Modality.TEXT)) == 1


def test_trusted_untrusted_split():
    f = PerceptualFrame()
    f.add(Percept(Modality.TEXT, "a", Provenance("s", 0.9), fingerprint=1))
    f.add(Percept(Modality.WEB, "b", Provenance("s", 0.1), fingerprint=2))
    assert len(f.trusted()) == 1 and len(f.untrusted()) == 1


def test_ranked_by_salience():
    f = PerceptualFrame()
    p1 = Percept(Modality.TEXT, "a", Provenance("s"), fingerprint=0xFFFF0000, salience=0.3)
    p2 = Percept(Modality.TEXT, "b", Provenance("s"), fingerprint=0x0000FFFF, salience=0.9)
    f.add(p1)
    f.add(p2)
    assert f.ranked()[0].id == p2.id
    assert f.most_salient().id == p2.id


def test_most_salient_empty():
    assert PerceptualFrame().most_salient() is None


# -------------------- associations -------------------- #
def test_association_shared_entity():
    f = PerceptualFrame(window=60.0)
    a = Percept(Modality.TEXT, "x", Provenance("s"), fingerprint=1,
                entities=["Acme"], at=100.0)
    b = Percept(Modality.IMAGE, "y", Provenance("s"), fingerprint=2,
                entities=["Acme"], at=105.0)
    f.add(a)
    f.add(b)
    assocs = f.associations()
    assert len(assocs) == 1
    assert "acme" in assocs[0].shared and assocs[0].cross_modal


def test_association_shared_tag():
    f = PerceptualFrame(window=60.0)
    a = Percept(Modality.TEXT, "x", Provenance("s"), fingerprint=0xFFFF0000,
                tags=["threat"], at=100.0)
    b = Percept(Modality.TEXT, "y", Provenance("s"), fingerprint=0x0000FFFF,
                tags=["threat"], at=100.0)
    f.add(a)
    f.add(b)
    assert any("threat" in x.shared for x in f.associations())


def test_association_generic_tags_excluded():
    f = PerceptualFrame()
    a = Percept(Modality.WEB, "x", Provenance("s"), fingerprint=0xFFFF0000, tags=["web"], at=100.0)
    b = Percept(Modality.WEB, "y", Provenance("s"), fingerprint=0x0000FFFF, tags=["web"], at=100.0)
    f.add(a)
    f.add(b)
    # 'web' alone is a generic modality tag, not a meaningful shared concept
    assert f.associations() == []


def test_association_none_without_overlap():
    f = PerceptualFrame()
    a = Percept(Modality.TEXT, "x", Provenance("s"), fingerprint=0xFFFF0000, entities=["A"], at=100.0)
    b = Percept(Modality.TEXT, "y", Provenance("s"), fingerprint=0x0000FFFF, entities=["B"], at=100.0)
    f.add(a)
    f.add(b)
    assert f.associations() == []


def test_association_to_dict():
    a = Association("1", "2", 0.7, ["acme"], True)
    assert a.to_dict()["cross_modal"] is True


# -------------------- Binder: trust -------------------- #
def test_binder_owner_trust():
    b = Binder()
    p = Percept.from_text("hi", source="owner")
    b.perceive(p)
    assert p.provenance.trust == 0.95


def test_binder_web_untrusted():
    b = Binder()
    p = Percept(Modality.WEB, "x", Provenance("https://site.com", 0.0), fingerprint=1)
    b.perceive(p)
    assert p.provenance.trust == 0.25 and not p.provenance.trusted


def test_binder_injection_barely_trusted():
    b = Binder()
    p = Percept(Modality.WEB, "x", Provenance("https://evil.com", 0.0), fingerprint=1,
                tags=["web", "injection"])
    b.perceive(p)
    assert p.provenance.trust <= 0.1


def test_binder_document_trust():
    b = Binder()
    p = Percept(Modality.DOCUMENT, "x", Provenance("f.txt", 0.0), fingerprint=1)
    b.perceive(p)
    assert p.provenance.trust == 0.7


def test_binder_trusted_source_override():
    b = Binder(trusted_sources=["internal-db"])
    p = Percept(Modality.WEB, "x", Provenance("internal-db", 0.0), fingerprint=1)
    b.perceive(p)
    assert p.provenance.trust == 0.85


def test_binder_explicit_trust_preserved():
    b = Binder()
    p = Percept(Modality.WEB, "x", Provenance("site", 0.42), fingerprint=1)
    b.perceive(p)
    assert p.provenance.trust == 0.42  # not overwritten when already set


# -------------------- Binder: salience -------------------- #
def test_binder_threat_more_salient():
    b = Binder()
    calm = Percept(Modality.TEXT, "x", Provenance("owner", 0.9), fingerprint=0xFFFF0000)
    threat = Percept(Modality.TEXT, "y", Provenance("owner", 0.9), fingerprint=0x0000FFFF,
                     tags=["threat"])
    b.perceive(calm)
    b.perceive(threat)
    assert threat.salience > calm.salience


def test_binder_novelty_bonus():
    b = Binder()
    p = Percept.from_text("a wholly novel sentence about turtles", source="owner")
    bound, is_new = b.perceive(p)
    assert is_new and bound.salience > 0.3


def test_binder_dedup_keeps_max_salience():
    b = Binder()
    p1 = Percept.from_text("same content here friends", source="owner", tags=["threat"])
    b.perceive(p1)
    p2 = Percept.from_text("same content here friends", source="owner")
    bound, is_new = b.perceive(p2)
    assert not is_new and bound.count == 2


def test_perceive_all():
    b = Binder()
    ps = [Percept.from_text(f"distinct sentence number {i} about topic {i}", source="s")
          for i in range(3)]
    results = b.perceive_all(ps)
    assert len(results) == 3 and all(isinstance(p, Percept) for p, _ in results)


def test_new_frame_resets():
    b = Binder()
    b.perceive(Percept.from_text("hello there world", source="s"))
    assert len(b.frame) == 1
    b.new_frame()
    assert len(b.frame) == 0


# -------------------- frame summary -------------------- #
def test_frame_summary():
    b = Binder()
    b.perceive(Percept(Modality.TEXT, "x", Provenance("owner", 0.9),
                       fingerprint=1, entities=["A"], at=100.0))
    b.perceive(Percept(Modality.IMAGE, "y", Provenance("cam", 0.6),
                       fingerprint=2, entities=["A"], at=101.0))
    s = b.frame.summary()
    assert s["percepts"] == 2 and s["by_modality"]["text"] == 1
    assert s["associations"] == 1
