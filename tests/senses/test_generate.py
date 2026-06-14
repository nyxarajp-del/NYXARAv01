"""Tests for nyxara.senses.generate — generative image & speech output."""

from __future__ import annotations

import pytest

from nyxara.senses.generate import (ImageGenerator, SpeechSynthesizer, identicon_rows,
                                     write_png)

# These cases pin the dependency-free identicon fallback (deterministic, offline). With
# diffusers+torch installed the real diffusion backend takes over — non-deterministic and
# engine != "identicon" — so the fallback contract isn't exercised; skip it there.
_skip_if_diffusion = pytest.mark.skipif(
    ImageGenerator.diffusion_available(),
    reason="diffusers installed — the identicon fallback path is not exercised")


def test_write_png_is_valid(tmp_path):
    p = tmp_path / "x.png"
    n = write_png(str(p), 16, 16, identicon_rows("hello", size=16, grid=8))
    assert n > 0
    assert p.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


@_skip_if_diffusion
def test_image_generate_fallback(tmp_path):
    p = tmp_path / "art.png"
    res = ImageGenerator().generate("a teal sovereign mind", str(p), size=64)
    assert res.ok and res.kind == "image" and res.bytes_written > 0
    # without diffusers the deterministic identicon path is used
    assert res.engine == "identicon"
    assert p.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


@_skip_if_diffusion
def test_image_is_deterministic(tmp_path):
    a = ImageGenerator().generate("same prompt", str(tmp_path / "a.png"), size=64)
    b = ImageGenerator().generate("same prompt", str(tmp_path / "b.png"), size=64)
    assert (tmp_path / "a.png").read_bytes() == (tmp_path / "b.png").read_bytes()
    assert a.bytes_written == b.bytes_written


def test_image_differs_by_prompt(tmp_path):
    ImageGenerator().generate("alpha", str(tmp_path / "a.png"), size=64)
    ImageGenerator().generate("beta", str(tmp_path / "b.png"), size=64)
    assert (tmp_path / "a.png").read_bytes() != (tmp_path / "b.png").read_bytes()


def test_speech_writes_real_wav(tmp_path):
    p = tmp_path / "say.wav"
    res = SpeechSynthesizer().synthesize("NYXARA serves JP.", str(p))
    assert res.ok and res.kind == "speech" and res.bytes_written > 44
    assert p.read_bytes()[:4] == b"RIFF"


def test_availability_is_honest():
    # these report a bool without raising regardless of what's installed
    assert isinstance(ImageGenerator.diffusion_available(), bool)
    assert isinstance(SpeechSynthesizer.tts_available(), bool)
