"""Tests for NYXARA's whole-disk filesystem faculty.

Two layers are locked in here:

* the engine (:class:`nyxara.agency.filesystem.Filesystem`) — real read/write/list/walk/glob/
  search/copy/move/delete, its caps, its scope + deny-glob guards, and errors-as-data;
* the governed wiring — the FS tools register with the right capability/risk and are gated by
  the permission pipeline, so an irreversible delete ESCALATES under a bare policy yet EXECUTES
  under full control (proving the gate is real, not bypassed).
"""

from __future__ import annotations

import base64

import pytest

from nyxara.agency.default_tools import build_default_tools
from nyxara.agency.filesystem import Filesystem, FilesystemError
from nyxara.agency.permissions import (
    Authority,
    Capability,
    PermissionRequest,
    build_default_policy,
    build_filesystem_policy,
    grant_filesystem_access,
)
from nyxara.agency.tools import ToolRegistry


# ------------------------------------------------------------------ #
# Engine — round-trips
# ------------------------------------------------------------------ #
def test_write_read_round_trip(tmp_path):
    fs = Filesystem()
    target = tmp_path / "hello.txt"
    fs.write_text(str(target), "hello world")
    assert fs.read_text(str(target)) == "hello world"
    assert fs.exists(str(target))


def test_append_text(tmp_path):
    fs = Filesystem()
    target = str(tmp_path / "log.txt")
    fs.write_text(target, "a")
    fs.append_text(target, "b")
    assert fs.read_text(target) == "ab"


def test_read_bytes_binary_safe(tmp_path):
    fs = Filesystem()
    target = str(tmp_path / "blob.bin")
    raw = bytes(range(256))
    fs.write_bytes(target, base64.b64encode(raw).decode("ascii"))
    decoded = base64.b64decode(fs.read_bytes(target))
    assert decoded == raw


def test_read_offset(tmp_path):
    fs = Filesystem()
    target = str(tmp_path / "f.txt")
    fs.write_text(target, "0123456789")
    assert fs.read_text(target, offset=5) == "56789"


def test_stat_and_list_dir(tmp_path):
    fs = Filesystem()
    (tmp_path / "a.txt").write_text("x")
    (tmp_path / "sub").mkdir()
    st = fs.stat(str(tmp_path / "a.txt"))
    assert st["type"] == "file" and st["size"] == 1 and "mode" in st
    listing = fs.list_dir(str(tmp_path))
    kinds = {e["name"]: e["type"] for e in listing}
    assert kinds == {"a.txt": "file", "sub": "dir"}


def test_make_dir_move_copy_delete(tmp_path):
    fs = Filesystem()
    d = str(tmp_path / "x" / "y")
    fs.make_dir(d)
    assert fs.exists(d)
    f = str(tmp_path / "x" / "y" / "f.txt")
    fs.write_text(f, "data")
    # copy file
    f2 = str(tmp_path / "copy.txt")
    fs.copy(f, f2)
    assert fs.read_text(f2) == "data"
    # move file
    f3 = str(tmp_path / "moved.txt")
    fs.move(f2, f3)
    assert fs.exists(f3) and not fs.exists(f2)
    # delete file
    fs.delete(f3)
    assert not fs.exists(f3)
    # recursive delete of a tree
    fs.delete(str(tmp_path / "x"), recursive=True)
    assert not fs.exists(str(tmp_path / "x"))


def test_copy_tree(tmp_path):
    fs = Filesystem()
    src = tmp_path / "src"
    (src / "nested").mkdir(parents=True)
    (src / "nested" / "a.txt").write_text("hi")
    dst = str(tmp_path / "dst")
    fs.copy(str(src), dst)
    assert fs.read_text(str(tmp_path / "dst" / "nested" / "a.txt")) == "hi"


def test_disk_usage(tmp_path):
    fs = Filesystem()
    usage = fs.disk_usage(str(tmp_path))
    assert usage["total"] > 0 and usage["free"] >= 0 and usage["used"] >= 0


# ------------------------------------------------------------------ #
# Engine — walk / glob / search with caps
# ------------------------------------------------------------------ #
def test_walk_depth_cap(tmp_path):
    deep = tmp_path
    for i in range(5):
        deep = deep / f"level{i}"
    deep.mkdir(parents=True)
    (deep / "leaf.txt").write_text("x")
    fs = Filesystem()
    shallow = fs.walk(str(tmp_path), max_depth=1)
    names = {e["name"] for e in shallow}
    assert "level0" in names and "leaf.txt" not in names
    full = fs.walk(str(tmp_path), max_depth=10)
    assert any(e["name"] == "leaf.txt" for e in full)


def test_walk_result_cap(tmp_path):
    for i in range(10):
        (tmp_path / f"f{i}.txt").write_text("x")
    fs = Filesystem()
    assert len(fs.walk(str(tmp_path), max_results=3)) == 3


def test_glob(tmp_path):
    (tmp_path / "a.py").write_text("x")
    (tmp_path / "b.txt").write_text("x")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "c.py").write_text("x")
    fs = Filesystem()
    py = fs.glob("**/*.py", root=str(tmp_path))
    assert len(py) == 2 and all(p.endswith(".py") for p in py)


def test_search_literal_and_regex(tmp_path):
    (tmp_path / "a.txt").write_text("alpha\nNEEDLE here\nbeta")
    (tmp_path / "b.txt").write_text("nothing")
    fs = Filesystem()
    hits = fs.search(str(tmp_path), "NEEDLE")
    assert len(hits) == 1 and hits[0]["line"] == 2 and "NEEDLE" in hits[0]["text"]
    rx = fs.search(str(tmp_path), r"N\w+E", regex=True)
    assert len(rx) == 1


def test_search_match_cap(tmp_path):
    (tmp_path / "a.txt").write_text("x\n" * 20)
    fs = Filesystem()
    assert len(fs.search(str(tmp_path), "x", max_matches=5)) == 5


def test_read_bytes_cap(tmp_path):
    target = str(tmp_path / "big.bin")
    (tmp_path / "big.bin").write_bytes(b"A" * 1000)
    fs = Filesystem(max_read_bytes=10)
    decoded = base64.b64decode(fs.read_bytes(target))
    assert len(decoded) == 10


# ------------------------------------------------------------------ #
# Engine — errors-as-data / raising guards
# ------------------------------------------------------------------ #
def test_missing_file_raises(tmp_path):
    fs = Filesystem()
    with pytest.raises(OSError):
        fs.read_text(str(tmp_path / "nope.txt"))


def test_no_overwrite_guard(tmp_path):
    fs = Filesystem()
    target = str(tmp_path / "f.txt")
    fs.write_text(target, "first")
    with pytest.raises(FilesystemError):
        fs.write_text(target, "second", overwrite=False)


# ------------------------------------------------------------------ #
# Engine — scope + deny-glob guards
# ------------------------------------------------------------------ #
def test_root_scoping_denies_outside(tmp_path):
    root = tmp_path / "jail"
    root.mkdir()
    fs = Filesystem(whole_disk=False, root=root)
    # inside root: fine
    fs.write_text(str(root / "ok.txt"), "x")
    # outside root: refused
    with pytest.raises(FilesystemError):
        fs.read_text(str(tmp_path / "outside.txt"))


def test_whole_disk_reaches_outside_a_root(tmp_path):
    # whole_disk=True: no scoping, so a path anywhere is allowed by the engine's guards.
    fs = Filesystem(whole_disk=True)
    target = str(tmp_path / "anywhere.txt")
    fs.write_text(target, "x")
    assert fs.read_text(target) == "x"


def test_deny_glob_protects_sensitive_path(tmp_path):
    secret = tmp_path / "keys" / "master.key"
    secret.parent.mkdir()
    secret.write_text("s3cr3t")
    fs = Filesystem(deny_globs=["*/keys/*"])
    with pytest.raises(FilesystemError):
        fs.read_text(str(secret))


def test_allow_glob_is_allow_only(tmp_path):
    (tmp_path / "a.txt").write_text("x")
    (tmp_path / "a.log").write_text("x")
    fs = Filesystem(allow_globs=["*.txt"])
    assert fs.read_text(str(tmp_path / "a.txt")) == "x"
    with pytest.raises(FilesystemError):
        fs.read_text(str(tmp_path / "a.log"))


# ------------------------------------------------------------------ #
# Grant helper
# ------------------------------------------------------------------ #
def _auto(cap, **kw):
    return PermissionRequest(cap, authority=Authority.AUTONOMOUS, **kw)


def test_grant_filesystem_access_covers_all_three_caps():
    pol = build_filesystem_policy()
    for cap in (Capability.FS_READ, Capability.FS_WRITE, Capability.FS_DELETE):
        d = pol.check(_auto(cap, target="/anywhere", reversible=False))
        assert d.allowed, f"{cap} should be autonomously allowed under filesystem access"
        assert d.rule_basis == "grant"


def test_scoped_filesystem_grant_confines_targets():
    pol = grant_filesystem_access(build_default_policy(), whole_disk=False,
                                  scopes=["/tmp/*"])
    inside = pol.check(_auto(Capability.FS_DELETE, target="/tmp/x", reversible=False))
    outside = pol.check(_auto(Capability.FS_DELETE, target="/etc/passwd", reversible=False))
    assert inside.allowed
    assert not outside.allowed  # scope mismatch -> escalates


# ------------------------------------------------------------------ #
# Governed wiring — the gate is real, not bypassed
# ------------------------------------------------------------------ #
def _registry(policy):
    reg = ToolRegistry(policy=policy)
    build_default_tools(reg, fs=Filesystem(), enable_code=False)
    return reg


def test_delete_escalates_under_bare_policy(tmp_path):
    target = str(tmp_path / "f.txt")
    (tmp_path / "f.txt").write_text("x")
    reg = _registry(build_default_policy())
    res = reg.invoke("delete_path", {"path": target})
    # HIGH + irreversible under the conservative envelope -> requires the Master, file untouched.
    assert not res.ok and res.requires_owner
    assert (tmp_path / "f.txt").exists()


def test_delete_executes_under_filesystem_grant(tmp_path):
    target = str(tmp_path / "f.txt")
    (tmp_path / "f.txt").write_text("x")
    reg = _registry(build_filesystem_policy())
    res = reg.invoke("delete_path", {"path": target})
    assert res.ok
    assert not (tmp_path / "f.txt").exists()


def test_full_fs_tool_family_registered():
    reg = ToolRegistry(policy=build_filesystem_policy())
    build_default_tools(reg, fs=Filesystem(), enable_code=False)
    names = set(reg.names())
    expected = {
        "read_file", "read_bytes", "stat_path", "path_exists", "list_dir", "walk_dir",
        "glob_paths", "search_files", "disk_usage", "write_file", "write_bytes",
        "append_file", "make_dir", "copy_path", "move_path", "delete_path",
    }
    assert expected <= names


def test_write_then_read_through_the_registry(tmp_path):
    target = str(tmp_path / "wired.txt")
    reg = _registry(build_filesystem_policy())
    wrote = reg.invoke("write_file", {"path": target, "content": "via NYXARA"})
    assert wrote.ok
    read = reg.invoke("read_file", {"path": target})
    assert read.ok and read.value == "via NYXARA"


# ------------------------------------------------------------------ #
# Orchestrator wiring — standalone FS grant when full_control is off
# ------------------------------------------------------------------ #
def test_orchestrator_grants_filesystem_when_whole_disk_on_and_full_control_off(monkeypatch):
    monkeypatch.setenv("NYXARA_AGENCY__FULL_CONTROL", "false")
    monkeypatch.setenv("NYXARA_AGENCY__AUTONOMOUS_INTERNET", "false")
    monkeypatch.setenv("NYXARA_AGENCY__AUTONOMOUS_REMOTE", "false")
    monkeypatch.setenv("NYXARA_AGENCY__FILESYSTEM__WHOLE_DISK", "true")
    from nyxara.kernel import config as cfg
    cfg.reload_settings()
    try:
        from nyxara.kernel.orchestrator import NyxaraCore
        core = NyxaraCore()
        # full_control off, but whole-disk FS on -> the standalone FS envelope is installed.
        d = core.permissions.check(_auto(Capability.FS_DELETE, target="/tmp/x", reversible=False))
        assert d.allowed and d.rule_basis == "grant"
        # ...and it does NOT confer the shell surface (that still escalates).
        assert core.permissions.check(_auto(Capability.PROC_EXEC, target="ls")).escalated
    finally:
        for var in ("NYXARA_AGENCY__FULL_CONTROL", "NYXARA_AGENCY__AUTONOMOUS_INTERNET",
                    "NYXARA_AGENCY__AUTONOMOUS_REMOTE", "NYXARA_AGENCY__FILESYSTEM__WHOLE_DISK"):
            monkeypatch.delenv(var, raising=False)
        cfg.reload_settings()
