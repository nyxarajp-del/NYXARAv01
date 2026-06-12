"""Tests for Step-9 multimodal grounding — process() binds image/audio/document
percepts into the *same* perceptual frame as the text, so attention and cross-modal
association span modalities rather than text alone. Heavy ML stays optional: ready
analyses / percepts need no torch or Pillow.
"""

from __future__ import annotations

from nyxara.agency.permissions import Authority
from nyxara.kernel.orchestrator import NyxaraCore
from nyxara.observe.mindscope import ThoughtKind
from nyxara.senses.audio import AudioAnalysis
from nyxara.senses.binding import Modality, Percept, Provenance
from nyxara.senses.vision import ImageAnalysis


def _core(**kw):
    return NyxaraCore(**kw)


# -------------------- backward compatibility -------------------- #
def test_text_only_process_unchanged():
    nyx = _core()
    r = nyx.process("how are you?", authority=Authority.OWNER)
    assert r.acted and r.candidate.kind == "respond"


# -------------------- binding extra modalities -------------------- #
def test_image_analysis_is_bound_into_the_frame():
    nyx = _core()
    img = ImageAnalysis(info=None, perceptual_hash=0xABCD, ocr_text="STOP")
    nyx.process("what is in this picture?", authority=Authority.OWNER,
                media=[{"image": img}])
    assert nyx.binder.frame.by_modality(Modality.IMAGE)


def test_audio_analysis_is_bound_into_the_frame():
    nyx = _core()
    aud = AudioAnalysis(info=None, fingerprint=0x1234, transcript="hello master")
    nyx.process("did you hear that?", authority=Authority.OWNER, media=[{"audio": aud}])
    assert nyx.binder.frame.by_modality(Modality.AUDIO)


def test_prebuilt_percept_and_text_spec_bind():
    nyx = _core()
    p = Percept.from_text("an attached note", source="clip", tags=["note"])
    nyx.process("look at these", authority=Authority.OWNER,
                media=[p, {"text": "another note", "source": "clip2"}])
    contents = [pc.content for pc in nyx.binder.frame.percepts()]
    assert "an attached note" in contents
    assert "another note" in contents


def test_binding_records_a_multimodal_thought():
    nyx = _core()
    img = ImageAnalysis(info=None, perceptual_hash=0x55)
    aud = AudioAnalysis(info=None, fingerprint=0x66)
    r = nyx.process("ground these", authority=Authority.OWNER,
                    media=[{"image": img}, {"audio": aud}])
    perceptions = nyx.mind.by_kind(ThoughtKind.PERCEPTION)
    assert any("bound" in t.content and "image" in t.content and "audio" in t.content
               for t in perceptions)
    assert r is not None


# -------------------- cross-modal association -------------------- #
def test_cross_modal_tie_is_surfaced():
    nyx = _core()
    img = Percept(Modality.IMAGE, "photo of the acme logo",
                  Provenance("camera", 0.6), fingerprint=0x111,
                  entities=["acme"], tags=["image"])
    txt = Percept(Modality.TEXT, "acme quarterly report",
                  Provenance("report.txt", 0.7), fingerprint=0x222,
                  entities=["acme"], tags=["text"])
    nyx.process("connect these", authority=Authority.OWNER, media=[img, txt])
    inferences = nyx.mind.by_kind(ThoughtKind.INFERENCE)
    assert any("cross-modal tie" in t.content and "acme" in t.content for t in inferences)


# -------------------- robustness -------------------- #
def test_missing_vision_dep_degrades_gracefully():
    nyx = _core()
    # a bare file path with no vision backend available -> a placeholder, never a crash
    r = nyx.process("what's in this file?", authority=Authority.OWNER,
                    media=[{"image": "/no/such/image.png"}])
    assert r is not None
    assert any("image" in (p.tags or []) for p in nyx.binder.frame.percepts())


def test_empty_and_bad_media_are_safe():
    nyx = _core()
    r1 = nyx.process("nothing attached", authority=Authority.OWNER, media=[])
    r2 = nyx.process("junk attached", authority=Authority.OWNER, media=[None, 42, "raw"])
    assert r1.acted and r2 is not None
