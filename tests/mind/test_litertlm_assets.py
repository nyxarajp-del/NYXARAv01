"""Network-free tests for the weights fetcher — mind/litertlm_assets.py.

This module is the only thing that puts her PRIMARY brain on disk, and it runs on every boot. The
properties worth pinning are all about *restraint and honesty* rather than throughput:

* it never fetches when it should not (already present, auto-download off, TEST profile);
* a killed download can never leave a truncated file that ``available()`` would trust;
* nothing it can fail at is allowed to take the boot down with it.

Both fetch backends are monkeypatched, so no test here opens a socket or writes 2.4 GB.
"""
from __future__ import annotations

import pytest

import nyxara.mind.litertlm_assets as assets
from nyxara.kernel.config import NyxaraSettings, Profile


def _settings(tmp_path, profile=Profile.DEV, **over):
    # The suite-wide conftest pins ``litertlm_enabled``/``auto_download`` off for hermeticity, so a
    # test that wants to exercise the fetcher has to opt back in explicitly — on its own object.
    s = NyxaraSettings(profile=profile)
    s.llm.litertlm_model_path = tmp_path / "model.litertlm"
    s.llm.litertlm_enabled = True
    s.llm.litertlm_auto_download = True
    for k, v in over.items():
        setattr(s.llm, k, v)
    return s


@pytest.fixture(autouse=True)
def _no_real_fetching(monkeypatch):
    """Fail loudly if a test reaches a real backend without saying so."""
    def _boom(*a, **k):
        raise AssertionError("a test reached a real fetch backend")

    monkeypatch.setattr(assets, "_fetch_via_hub", _boom)
    monkeypatch.setattr(assets, "_fetch_via_http", _boom)


def _writes(content=b"weights", calls=None):
    """A fake backend that writes ``content`` to the .part file and reports success."""
    def _fake(settings, dest, progress):
        if calls is not None:
            calls.append(dest)
        dest.write_bytes(content)
        if progress is not None:
            progress(len(content), len(content))
        return True
    return _fake


# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
def test_model_path_prefers_the_configured_path(tmp_path):
    s = _settings(tmp_path)
    assert assets.model_path(s) == tmp_path / "model.litertlm"


def test_model_path_defaults_under_the_data_dir():
    s = NyxaraSettings(profile=Profile.DEV)
    s.llm.litertlm_model_path = None
    expected = s.paths.data_dir / "models" / s.llm.litertlm_filename
    assert assets.model_path(s) == expected


def test_model_present_tracks_the_file(tmp_path):
    s = _settings(tmp_path)
    assert assets.model_present(s) is False
    (tmp_path / "model.litertlm").write_bytes(b"x")
    assert assets.model_present(s) is True


def test_expected_url_points_at_the_configured_repo(tmp_path):
    s = _settings(tmp_path)
    url = assets.expected_url(s)
    assert url.startswith("https://huggingface.co/")
    assert s.llm.litertlm_repo_id in url
    assert url.endswith("/resolve/main/model.litertlm")


# --------------------------------------------------------------------------- #
# Restraint — the three reasons not to fetch
# --------------------------------------------------------------------------- #
def test_present_weights_are_never_refetched(tmp_path, monkeypatch):
    s = _settings(tmp_path)
    dest = tmp_path / "model.litertlm"
    dest.write_bytes(b"already here")
    assert assets.ensure_litertlm_model(s) == dest        # the autouse guard would fire on a fetch
    assert dest.read_bytes() == b"already here"


def test_auto_download_off_declines_honestly(tmp_path):
    s = _settings(tmp_path, litertlm_auto_download=False)
    assert assets.ensure_litertlm_model(s) is None
    assert not (tmp_path / "model.litertlm").exists()


def test_test_profile_never_fetches(tmp_path, monkeypatch):
    """A hermetic suite must not pull 2.4 GB — even if something flips the flag back on."""
    s = _settings(tmp_path, profile=Profile.TEST)
    s.llm.litertlm_auto_download = True
    monkeypatch.setattr(assets, "_fetch_via_hub", _writes())
    assert assets.ensure_litertlm_model(s) is None
    assert not (tmp_path / "model.litertlm").exists()


def test_disabled_provider_is_not_fetched_for(tmp_path):
    s = _settings(tmp_path, litertlm_enabled=False)
    assert assets.ensure_litertlm_model(s) is None


# --------------------------------------------------------------------------- #
# The download itself
# --------------------------------------------------------------------------- #
def test_successful_fetch_lands_atomically(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(assets, "_fetch_via_hub", _writes(b"the weights", calls))
    s = _settings(tmp_path)
    out = assets.ensure_litertlm_model(s)
    assert out == tmp_path / "model.litertlm"
    assert out.read_bytes() == b"the weights"
    assert calls and calls[0].name.endswith(".part"), "must download to .part, then rename"
    assert not calls[0].exists(), "the .part file must not survive a successful fetch"


def test_http_is_the_fallback_when_the_hub_is_unusable(tmp_path, monkeypatch):
    monkeypatch.setattr(assets, "_fetch_via_hub", lambda *a, **k: False)
    monkeypatch.setattr(assets, "_fetch_via_http", _writes(b"via http"))
    s = _settings(tmp_path)
    assert assets.ensure_litertlm_model(s).read_bytes() == b"via http"


def test_progress_is_reported(tmp_path, monkeypatch):
    seen = []
    monkeypatch.setattr(assets, "_fetch_via_hub", _writes(b"12345"))
    s = _settings(tmp_path)
    assets.ensure_litertlm_model(s, progress=lambda done, total: seen.append((done, total)))
    assert seen == [(5, 5)]


def test_no_backend_available_returns_none(tmp_path, monkeypatch):
    monkeypatch.setattr(assets, "_fetch_via_hub", lambda *a, **k: False)
    monkeypatch.setattr(assets, "_fetch_via_http", lambda *a, **k: False)
    s = _settings(tmp_path)
    assert assets.ensure_litertlm_model(s) is None


# --------------------------------------------------------------------------- #
# Failure — never leave a file the provider would trust
# --------------------------------------------------------------------------- #
def test_a_failed_fetch_leaves_no_partial_file(tmp_path, monkeypatch):
    """The whole point of the .part dance: a truncated 2.4 GB file would look ready and load broken."""
    def _half_then_die(settings, dest, progress):
        dest.write_bytes(b"half a model")
        raise OSError("connection reset")

    monkeypatch.setattr(assets, "_fetch_via_hub", _half_then_die)
    s = _settings(tmp_path)
    assert assets.ensure_litertlm_model(s) is None
    assert not (tmp_path / "model.litertlm").exists()
    assert not (tmp_path / "model.litertlm.part").exists()
    assert assets.model_present(s) is False


def test_a_failed_fetch_never_raises(tmp_path, monkeypatch):
    """Boot integrity: the ladder just starts one rung lower."""
    def _die(settings, dest, progress):
        raise RuntimeError("catastrophe")

    monkeypatch.setattr(assets, "_fetch_via_hub", _die)
    assert assets.ensure_litertlm_model(_settings(tmp_path)) is None


def test_force_redownloads_over_an_existing_file(tmp_path, monkeypatch):
    monkeypatch.setattr(assets, "_fetch_via_hub", _writes(b"fresh"))
    s = _settings(tmp_path)
    dest = tmp_path / "model.litertlm"
    dest.write_bytes(b"stale")
    assert assets.ensure_litertlm_model(s, force=True) == dest
    assert dest.read_bytes() == b"fresh"


# --------------------------------------------------------------------------- #
# Boot wiring
# --------------------------------------------------------------------------- #
def test_bootstrap_asks_for_the_weights(tmp_path, monkeypatch):
    """``ensure_primary_model`` is the first-boot trigger; it must call through and survive failure."""
    from nyxara.growth import bootstrap

    called = []
    monkeypatch.setattr(assets, "ensure_litertlm_model",
                        lambda s, **k: called.append(s) or None)
    s = _settings(tmp_path)
    s.foundry.enabled = False
    bootstrap._ensure_litertlm(s, lambda _m: None)
    assert len(called) == 1


def test_bootstrap_survives_a_fetch_explosion(tmp_path, monkeypatch):
    from nyxara.growth import bootstrap

    def _die(*a, **k):
        raise RuntimeError("no disk")

    monkeypatch.setattr(assets, "ensure_litertlm_model", _die)
    said = []
    bootstrap._ensure_litertlm(_settings(tmp_path), said.append)
    assert any("skipped" in m for m in said)
