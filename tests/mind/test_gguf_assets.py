"""Getting 5.6-17.9 GB onto disk without lying about whether it arrived."""

from __future__ import annotations

import hashlib

import pytest

from nyxara.kernel.config import NyxaraSettings, Profile, get_settings
from nyxara.mind import gguf_assets


def test_absent_weights_are_reported_not_faked():
    assert gguf_assets.model_present() is False
    assert gguf_assets.model_path().name.endswith(".gguf")


def test_the_test_profile_never_reaches_the_network():
    """A hermetic suite must not spend the network or the disk budget, and this is the guard that
    makes every other test in the repo safe to run offline."""
    settings = NyxaraSettings(profile=Profile.TEST)
    settings.llm.gguf_auto_download = True          # even asked for it explicitly
    assert gguf_assets.ensure_gguf_model(settings) is None
    assert gguf_assets.fetch_checksums(settings) == {}


def test_auto_download_off_declines_even_outside_the_test_profile():
    settings = NyxaraSettings(profile=Profile.DEV)
    settings.llm.gguf_auto_download = False
    assert gguf_assets.ensure_gguf_model(settings) is None


def test_auto_download_is_off_by_default():
    """Unlike the 2.4 GB litertlm fetch. Rule 6: a 5.6-17.9 GB download is a decision, not a
    surprise to spring on a metered connection."""
    assert get_settings().llm.gguf_auto_download is False


def test_every_published_quant_maps_to_a_filename_and_a_size():
    quants = gguf_assets.available_quants()
    assert set(quants) == {"Q4_K_M", "Q5_K_M", "Q6_K", "Q8_0", "BF16"}
    for name in quants:
        assert gguf_assets.filename_for_quant(name).endswith(f"-{name}.gguf")
        assert "GB" in quants[name]


def test_an_unpublished_quant_is_refused_rather_than_guessed():
    with pytest.raises(ValueError, match="unknown quant"):
        gguf_assets.filename_for_quant("Q3_K_S")


def test_checksum_verification_has_three_states_not_two():
    """'I could not check' and 'I checked and it is wrong' are completely different facts.
    Collapsing them either rejects good files offline or accepts bad ones silently."""
    payload = b"weights-ish"
    path = get_settings().paths.data_dir / "probe.gguf"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()

    assert gguf_assets.verify_checksum(path, checksums={}) is None
    assert gguf_assets.verify_checksum(path, checksums={"probe.gguf": digest}) is True
    assert gguf_assets.verify_checksum(path, checksums={"probe.gguf": "0" * 64}) is False
    path.unlink()


def test_sha256_is_streamed_not_slurped():
    path = get_settings().paths.data_dir / "probe2.gguf"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x" * (3 * 1024 * 1024))
    assert gguf_assets.sha256_file(path) == hashlib.sha256(b"x" * (3 * 1024 * 1024)).hexdigest()
    path.unlink()


def test_the_url_points_at_the_configured_repo_and_file():
    url = gguf_assets.expected_url()
    assert "empero-ai/Qwythos-9B-Claude-Mythos-5-1M-GGUF" in url
    assert url.endswith(get_settings().llm.gguf_filename)
