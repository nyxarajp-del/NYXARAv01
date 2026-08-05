"""Tests for mind/vulkan_shim.py — running her on-device brain with no Vulkan loader installed.

``liblitert-lm.so`` declares ``DT_NEEDED: libvulkan.so.1`` even on the CPU backend while importing
no symbol from it, so a host without the loader cannot load the runtime at all. The shim generates a
344-byte ELF carrying only that SONAME, which satisfies the linker and lets the CPU path run.

The properties worth pinning are mostly about **restraint**: the stub is a workaround for one narrow,
verified fact, and every case where that fact stops holding must be a case where it declines.
"""
from __future__ import annotations

import struct

import pytest

import nyxara.mind.vulkan_shim as shim
from nyxara.kernel.config import NyxaraSettings, Profile


@pytest.fixture(autouse=True)
def _fresh_process_state(monkeypatch):
    """The shim latches per process; each test gets a clean slate."""
    monkeypatch.setattr(shim, "_LOADED", False)


def _settings(tmp_path, **over):
    s = NyxaraSettings(profile=Profile.DEV)
    s.llm.litertlm_model_path = tmp_path / "model.litertlm"
    for k, v in over.items():
        setattr(s.llm, k, v)
    return s


# --------------------------------------------------------------------------- #
# The generated ELF
# --------------------------------------------------------------------------- #
def test_stub_is_a_well_formed_elf64_shared_object():
    blob = shim.build_stub(machine="x86_64")
    assert blob[:4] == b"\x7fELF"
    assert blob[4] == 2 and blob[5] == 1          # ELFCLASS64, ELFDATA2LSB
    e_type, e_machine = struct.unpack_from("<HH", blob, 16)
    assert e_type == 3, "must be ET_DYN — a shared object, not an executable"
    assert e_machine == 62                        # EM_X86_64


def test_stub_carries_the_soname_the_linker_is_looking_for():
    """The whole point: DT_SONAME is the only content that matters."""
    blob = shim.build_stub(machine="x86_64")
    assert b"libvulkan.so.1\0" in blob
    # walk the dynamic array and find DT_SONAME (tag 14)
    off_ph, = struct.unpack_from("<Q", blob, 32)
    tags = {}
    for i in range(2):
        p_type, _, p_offset = struct.unpack_from("<IIQ", blob, off_ph + 56 * i)
        if p_type == 2:                            # PT_DYNAMIC
            off = p_offset
            while True:
                tag, val = struct.unpack_from("<qQ", blob, off)
                if tag == 0:
                    break
                tags[tag] = val
                off += 16
    assert 14 in tags, "no DT_SONAME — the linker would not accept it"
    strtab = tags[5]
    name = blob[strtab + tags[14]:blob.index(b"\0", strtab + tags[14])]
    assert name == b"libvulkan.so.1"


def test_stub_is_tiny():
    """It exists to satisfy a name, not to do anything."""
    assert len(shim.build_stub(machine="x86_64")) < 1024


def test_arm64_is_supported():
    assert struct.unpack_from("<H", shim.build_stub(machine="aarch64"), 18)[0] == 183


def test_an_unknown_architecture_is_an_honest_failure():
    """Better to decline than to emit an ELF the loader rejects for a confusing reason."""
    with pytest.raises(ValueError):
        shim.build_stub(machine="s390x")


# --------------------------------------------------------------------------- #
# Restraint — every case where the stub would be wrong
# --------------------------------------------------------------------------- #
def test_a_real_loader_is_never_shadowed(tmp_path, monkeypatch):
    monkeypatch.setattr(shim, "loader_present", lambda: True)
    assert shim.ensure_vulkan_loader(_settings(tmp_path)) is None
    assert not (tmp_path / "vulkan-shim").exists(), "nothing should be written when it is not needed"


def test_it_refuses_on_a_gpu_backend(tmp_path, monkeypatch):
    """A do-nothing Vulkan under a GPU backend is worse than an honest failure to start."""
    monkeypatch.setattr(shim, "loader_present", lambda: False)
    for backend in ("gpu", "npu"):
        s = _settings(tmp_path, litertlm_backend=backend)
        assert shim.ensure_vulkan_loader(s) is None
        assert not (tmp_path / "vulkan-shim").exists()


def test_it_can_be_switched_off(tmp_path, monkeypatch):
    monkeypatch.setattr(shim, "loader_present", lambda: False)
    s = _settings(tmp_path, litertlm_vulkan_shim=False)
    assert shim.ensure_vulkan_loader(s) is None


def test_an_unknown_architecture_declines_rather_than_raising(tmp_path, monkeypatch):
    monkeypatch.setattr(shim, "loader_present", lambda: False)
    monkeypatch.setattr(shim, "build_stub", lambda *a, **k: (_ for _ in ()).throw(ValueError("nope")))
    assert shim.ensure_vulkan_loader(_settings(tmp_path)) is None


def test_a_write_failure_is_not_fatal(tmp_path, monkeypatch):
    """The shim is an aid. If it cannot help, the rung stays unavailable — nothing crashes."""
    monkeypatch.setattr(shim, "loader_present", lambda: False)

    def _boom(*a, **k):
        raise OSError("read-only file system")

    monkeypatch.setattr(shim.Path, "mkdir", _boom)
    assert shim.ensure_vulkan_loader(_settings(tmp_path)) is None


# --------------------------------------------------------------------------- #
# The happy path
# --------------------------------------------------------------------------- #
def test_it_writes_and_loads_the_stub_when_the_loader_is_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(shim, "loader_present", lambda: False)
    loaded = []
    monkeypatch.setattr(shim.ctypes, "CDLL", lambda path, mode=0: loaded.append(path))

    s = _settings(tmp_path)
    note = shim.ensure_vulkan_loader(s)
    assert note and "stub" in note

    path = shim.stub_path(s)
    assert path.is_file()
    assert path.read_bytes()[:4] == b"\x7fELF"
    assert loaded == [str(path)], "the stub must actually be dlopened, not merely written"
    assert not path.with_suffix(path.suffix + ".part").exists(), "atomic write leaves no .part"


def test_it_only_loads_once_per_process(tmp_path, monkeypatch):
    monkeypatch.setattr(shim, "loader_present", lambda: False)
    calls = []
    monkeypatch.setattr(shim.ctypes, "CDLL", lambda path, mode=0: calls.append(path))
    s = _settings(tmp_path)
    shim.ensure_vulkan_loader(s)
    shim.ensure_vulkan_loader(s)
    shim.ensure_vulkan_loader(s)
    assert len(calls) == 1


def test_a_stale_stub_is_rewritten(tmp_path, monkeypatch):
    """A shim from an older version must not be trusted just because the path exists."""
    monkeypatch.setattr(shim, "loader_present", lambda: False)
    monkeypatch.setattr(shim.ctypes, "CDLL", lambda path, mode=0: None)
    s = _settings(tmp_path)
    path = shim.stub_path(s)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"junk from a previous life")
    shim.ensure_vulkan_loader(s)
    assert path.read_bytes() == shim.build_stub()
