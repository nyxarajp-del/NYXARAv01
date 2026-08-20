"""Network-free tests for the cortex weights fetcher — mind/gguf_assets.py.

This module is the only thing that puts her cortex on disk, and it runs on every boot. The
properties worth pinning are the same three ``mind/litertlm_assets.py`` is built around — restraint,
atomicity, and never taking the boot down — plus the one this module adds:

    a checksum that MISMATCHES is fatal; a checksum that cannot be OBTAINED is reported, not assumed.

Both fetch backends are monkeypatched, so no test here opens a socket or writes 5.6 GB.
"""
from __future__ import annotations

import hashlib

import pytest

import nyxara.mind.gguf_assets as assets
from nyxara.kernel.config import NyxaraSettings, Profile

_WEIGHTS = b"the cortex weights"
_PROJECTOR = b"the vision projector"
_MODEL = "Qwythos-9B-Claude-Mythos-5-1M-Q4_K_M.gguf"
_MMPROJ = "mmproj-Qwythos-9B-Claude-Mythos-5-1M-F16.gguf"


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _settings(tmp_path, profile=Profile.DEV, **over):
    # The suite-wide config pins qwythos off under TEST for hermeticity, so a test that wants to
    # exercise the fetcher has to opt back in explicitly — on its own object.
    s = NyxaraSettings(profile=profile)
    s.llm.qwythos_model_path = tmp_path / _MODEL
    s.llm.qwythos_mmproj_path = tmp_path / _MMPROJ
    s.llm.qwythos_enabled = True
    s.llm.qwythos_auto_download = True
    s.llm.qwythos_vision_enabled = False
    for k, v in over.items():
        setattr(s.llm, k, v)
    return s


@pytest.fixture(autouse=True)
def _no_real_fetching(monkeypatch):
    """Fail loudly if a test reaches a real backend or the real checksum manifest."""
    def _boom(*a, **k):
        raise AssertionError("a test reached a real fetch backend")

    monkeypatch.setattr(assets, "_fetch_via_hub", _boom)
    monkeypatch.setattr(assets, "_fetch_via_http", _boom)
    monkeypatch.setattr(assets, "published_checksums", lambda *a, **k: {})
    assets._VERIFIED.clear()


def _writes(content=_WEIGHTS, calls=None, projector=_PROJECTOR):
    """A fake backend that writes the right bytes for whichever file was asked for."""
    def _fake(settings, filename, dest, progress):
        if calls is not None:
            calls.append(filename)
        dest.write_bytes(projector if filename.startswith("mmproj") else content)
        if progress is not None:
            progress(len(content), len(content))
        return True
    return _fake


# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
def test_paths_prefer_the_configured_locations(tmp_path):
    s = _settings(tmp_path)
    assert assets.model_path(s) == tmp_path / _MODEL
    assert assets.mmproj_path(s) == tmp_path / _MMPROJ


def test_paths_default_under_the_data_dir():
    s = NyxaraSettings(profile=Profile.DEV)
    s.llm.qwythos_model_path = None
    s.llm.qwythos_mmproj_path = None
    assert assets.model_path(s) == s.paths.data_dir / "models" / s.llm.qwythos_filename
    assert assets.mmproj_path(s) == s.paths.data_dir / "models" / s.llm.qwythos_mmproj_filename


def test_presence_tracks_the_two_files_separately(tmp_path):
    s = _settings(tmp_path)
    assert assets.model_present(s) is False and assets.mmproj_present(s) is False
    (tmp_path / _MODEL).write_bytes(b"x")
    assert assets.model_present(s) is True
    assert assets.mmproj_present(s) is False, "text works without sight"


def test_expected_url_points_at_the_configured_repo(tmp_path):
    url = assets.expected_url(_settings(tmp_path))
    assert url.startswith("https://huggingface.co/empero-ai/")
    assert url.endswith(f"/resolve/main/{_MODEL}")


# --------------------------------------------------------------------------- #
# Restraint — the reasons not to fetch
# --------------------------------------------------------------------------- #
def test_present_weights_are_never_refetched(tmp_path):
    s = _settings(tmp_path)
    dest = tmp_path / _MODEL
    dest.write_bytes(b"already here")
    assert assets.ensure_gguf_model(s) == dest      # the autouse guard would fire on a fetch
    assert dest.read_bytes() == b"already here"


def test_auto_download_off_declines_honestly(tmp_path):
    s = _settings(tmp_path, qwythos_auto_download=False)
    assert assets.ensure_gguf_model(s) is None
    assert not (tmp_path / _MODEL).exists()


def test_test_profile_never_fetches(tmp_path, monkeypatch):
    """A hermetic suite must not pull 5.6 GB — even if something flips the flag back on."""
    s = _settings(tmp_path, profile=Profile.TEST)
    s.llm.qwythos_auto_download = True
    s.llm.qwythos_enabled = True
    monkeypatch.setattr(assets, "_fetch_via_hub", _writes())
    assert assets.ensure_gguf_model(s) is None
    assert not (tmp_path / _MODEL).exists()


def test_a_disabled_cortex_is_not_fetched_for(tmp_path):
    assert assets.ensure_gguf_model(_settings(tmp_path, qwythos_enabled=False)) is None


# --------------------------------------------------------------------------- #
# The download itself
# --------------------------------------------------------------------------- #
def test_successful_fetch_lands_atomically(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(assets, "_fetch_via_hub", _writes(calls=calls))
    s = _settings(tmp_path)
    got = assets.ensure_gguf_model(s)
    assert got == tmp_path / _MODEL
    assert got.read_bytes() == _WEIGHTS
    assert calls == [_MODEL]
    assert not (tmp_path / f"{_MODEL}.part").exists(), "the .part must not survive a success"


def test_the_http_backend_is_the_fallback(tmp_path, monkeypatch):
    monkeypatch.setattr(assets, "_fetch_via_hub", lambda *a, **k: False)
    monkeypatch.setattr(assets, "_fetch_via_http", _writes())
    assert assets.ensure_gguf_model(_settings(tmp_path)) == tmp_path / _MODEL


def test_a_killed_download_leaves_no_truncated_file(tmp_path, monkeypatch):
    """The whole reason for the .part: a truncated 5.6 GB file would make available() lie."""
    def _dies(settings, filename, dest, progress):
        dest.write_bytes(b"half of the wei")
        raise OSError("the process was killed")

    monkeypatch.setattr(assets, "_fetch_via_hub", _dies)
    s = _settings(tmp_path)
    assert assets.ensure_gguf_model(s) is None
    assert not (tmp_path / _MODEL).exists()
    assert not (tmp_path / f"{_MODEL}.part").exists()
    assert assets.model_present(s) is False


def test_no_usable_backend_returns_none_rather_than_raising(tmp_path, monkeypatch):
    monkeypatch.setattr(assets, "_fetch_via_hub", lambda *a, **k: False)
    monkeypatch.setattr(assets, "_fetch_via_http", lambda *a, **k: False)
    assert assets.ensure_gguf_model(_settings(tmp_path)) is None


# --------------------------------------------------------------------------- #
# Integrity — the property this module adds
# --------------------------------------------------------------------------- #
def test_a_matching_checksum_installs_and_is_reported_verified(tmp_path, monkeypatch):
    monkeypatch.setattr(assets, "_fetch_via_hub", _writes())
    monkeypatch.setattr(assets, "published_checksums", lambda *a, **k: {_MODEL: _sha(_WEIGHTS)})
    s = _settings(tmp_path)
    got = assets.ensure_gguf_model(s)
    assert got is not None
    assert assets.verification_state(got) == "verified"


def test_a_mismatching_checksum_installs_nothing(tmp_path, monkeypatch):
    """A corrupt model that loads is worse than one that does not: it answers, and every answer is
    garbage attributed to her cortex."""
    monkeypatch.setattr(assets, "_fetch_via_hub", _writes(content=b"tampered"))
    monkeypatch.setattr(assets, "published_checksums", lambda *a, **k: {_MODEL: _sha(_WEIGHTS)})
    s = _settings(tmp_path)
    assert assets.ensure_gguf_model(s) is None
    assert not (tmp_path / _MODEL).exists()
    assert not (tmp_path / f"{_MODEL}.part").exists()


def test_an_unobtainable_checksum_is_reported_not_assumed(tmp_path, monkeypatch):
    monkeypatch.setattr(assets, "_fetch_via_hub", _writes())
    monkeypatch.setattr(assets, "published_checksums", lambda *a, **k: {})
    s = _settings(tmp_path)
    got = assets.ensure_gguf_model(s)
    assert got is not None, "an unreadable manifest must not block an install"
    assert assets.verification_state(got) == "unverified"


def test_verification_off_skips_the_manifest_entirely(tmp_path, monkeypatch):
    def _boom(*a, **k):
        raise AssertionError("the manifest was read with verification off")

    monkeypatch.setattr(assets, "_fetch_via_hub", _writes())
    monkeypatch.setattr(assets, "published_checksums", _boom)
    s = _settings(tmp_path, qwythos_verify_sha256=False)
    assert assets.ensure_gguf_model(s) is not None


def test_a_file_we_did_not_place_is_unverified_not_verified(tmp_path):
    (tmp_path / _MODEL).write_bytes(_WEIGHTS)
    assert assets.verification_state(tmp_path / _MODEL) == "unverified"
    assert assets.verification_state(tmp_path / "nothing-here.gguf") == "absent"


def test_the_manifest_parser_reads_both_sha256sum_formats():
    text = (f"{_sha(_WEIGHTS)}  {_MODEL}\n"
            f"{_sha(_PROJECTOR)} *{_MMPROJ}\n"
            "# a header line\n"
            "\n"
            "not-a-hash-but-64-characters-long-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa  x.gguf\n")
    assert assets.parse_sums(text) == {_MODEL: _sha(_WEIGHTS), _MMPROJ: _sha(_PROJECTOR)}


def test_the_manifest_parser_tolerates_nothing_at_all():
    assert assets.parse_sums("") == {} and assets.parse_sums(None) == {}


# --------------------------------------------------------------------------- #
# Two files, and sight is optional
# --------------------------------------------------------------------------- #
def test_the_projector_is_fetched_when_vision_is_on(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(assets, "_fetch_via_hub", _writes(calls=calls))
    s = _settings(tmp_path, qwythos_vision_enabled=True)
    assert assets.ensure_gguf_model(s) == tmp_path / _MODEL
    assert calls == [_MODEL, _MMPROJ]
    assert (tmp_path / _MMPROJ).read_bytes() == _PROJECTOR


def test_the_projector_is_not_fetched_when_vision_is_off(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(assets, "_fetch_via_hub", _writes(calls=calls))
    assert assets.ensure_gguf_model(_settings(tmp_path)) is not None
    assert calls == [_MODEL]


def test_a_failed_projector_still_leaves_a_working_cortex(tmp_path, monkeypatch):
    """A cortex that can read but not see is a working cortex."""
    def _text_only(settings, filename, dest, progress):
        if filename.startswith("mmproj"):
            raise OSError("the projector download died")
        dest.write_bytes(_WEIGHTS)
        return True

    monkeypatch.setattr(assets, "_fetch_via_hub", _text_only)
    s = _settings(tmp_path, qwythos_vision_enabled=True)
    assert assets.ensure_gguf_model(s) == tmp_path / _MODEL
    assert not (tmp_path / _MMPROJ).exists()


# --------------------------------------------------------------------------- #
# The suite-wide seal — Profile.TEST is not the guard it looks like
# --------------------------------------------------------------------------- #
def test_a_dev_settings_object_cannot_reach_the_network_from_inside_the_suite():
    """Every test above builds its own settings and opts back in. This pins the default.

    ``Profile.TEST`` seals ``qwythos_auto_download`` and this module refuses to fetch under it, so
    the guard reads as complete — but the suite does not run under TEST, and plenty of tests
    legitimately build a **DEV** object and then call a boot path.
    ``tests/growth/test_bootstrap.py::_self_settings`` is five of them, and each reaches
    ``growth/bootstrap._ensure_qwythos`` with auto-download on. Measured before the seal in
    ``tests/conftest.py``: ten real fetches, 6.5 GB into the suite home plus 6.1 GB of HuggingFace
    cache, and a second run that died on "No space left on device".
    """
    import os

    assert os.environ.get("NYXARA_LLM__QWYTHOS_AUTO_DOWNLOAD") == "false"
    for profile in (Profile.DEV, Profile.PROD, Profile.TEST):
        settings = NyxaraSettings.for_profile(profile)
        assert settings.llm.qwythos_auto_download is False, profile
        assert settings.llm.litertlm_auto_download is False, profile


def test_the_boot_path_declines_when_the_seal_is_in_force(tmp_path, monkeypatch):
    """The seal has to stop the *caller*, not merely the fetcher — so drive the real boot path."""
    def _boom(*a, **k):
        raise AssertionError("growth/bootstrap reached the network from inside the suite")

    monkeypatch.setattr(assets, "_fetch_via_hub", _boom)
    monkeypatch.setattr(assets, "_fetch_via_http", _boom)

    from nyxara.growth.bootstrap import ensure_primary_model

    s = NyxaraSettings.for_profile(Profile.DEV)
    s.llm.self_model_dir = tmp_path / "foundry"
    ensure_primary_model(s, log=lambda _m: None)
    assert not assets.model_present(s)
