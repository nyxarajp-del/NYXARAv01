"""Tests for NYXARA's filesystem-wide toolset (agency/default_tools.py).

These exercise the *real* file operations — write/read/append/binary/list/find/search/
stat/copy/move/mkdir/delete — through the governed ToolRegistry, proving they take real
effects on disk with no path confinement, and that the FS_DELETE capability (HIGH,
irreversible) is properly gated: autonomous delete escalates under the conservative
policy but runs under the owner's standing full_control grant.
"""

from __future__ import annotations

import base64
import os

from nyxara.agency.default_tools import build_default_tools
from nyxara.agency.permissions import Authority, build_sovereign_policy
from nyxara.agency.tools import ToolRegistry


def _reg(**kw):
    return build_default_tools(ToolRegistry(), **kw)


def _sovereign():
    return build_default_tools(ToolRegistry(policy=build_sovereign_policy()))


# --------------------------------------------------------------------------- #
# Registration
# --------------------------------------------------------------------------- #
def test_full_filesystem_toolset_registered():
    names = set(_reg().names())
    assert {
        "read_file", "read_file_bytes", "list_dir", "path_stat", "path_exists",
        "find_files", "search_in_files", "disk_usage",
        "write_file", "write_file_bytes", "make_dir", "copy_path", "move_path",
        "delete_file", "delete_dir",
    } <= names


# --------------------------------------------------------------------------- #
# Write / read round-trips (real effects, absolute paths outside cwd)
# --------------------------------------------------------------------------- #
def test_write_then_read_text(tmp_path):
    reg = _reg()
    target = tmp_path / "hello.txt"
    w = reg.invoke("write_file", {"path": str(target), "content": "sovereign"},
                   authority=Authority.OWNER)
    assert w.ok
    # the effect is real on disk
    assert target.read_text() == "sovereign"
    r = reg.invoke("read_file", {"path": str(target)}, authority=Authority.OWNER)
    assert r.ok and r.value == "sovereign"


def test_write_append_and_make_parents(tmp_path):
    reg = _reg()
    nested = tmp_path / "a" / "b" / "c" / "log.txt"
    # make_parents (default True) creates the missing tree
    reg.invoke("write_file", {"path": str(nested), "content": "one\n"},
               authority=Authority.OWNER)
    reg.invoke("write_file", {"path": str(nested), "content": "two\n", "append": True},
               authority=Authority.OWNER)
    assert nested.read_text() == "one\ntwo\n"


def test_read_file_offset_and_max_bytes(tmp_path):
    reg = _reg()
    target = tmp_path / "data.txt"
    target.write_text("0123456789")
    r = reg.invoke("read_file", {"path": str(target), "offset": 3, "max_bytes": 4},
                   authority=Authority.OWNER)
    assert r.ok and r.value == "3456"


def test_binary_round_trip(tmp_path):
    reg = _reg()
    target = tmp_path / "blob.bin"
    raw = bytes(range(256))
    b64 = base64.b64encode(raw).decode("ascii")
    w = reg.invoke("write_file_bytes", {"path": str(target), "base64_data": b64},
                   authority=Authority.OWNER)
    assert w.ok and target.read_bytes() == raw
    r = reg.invoke("read_file_bytes", {"path": str(target)}, authority=Authority.OWNER)
    assert r.ok and base64.b64decode(r.value["base64"]) == raw
    assert r.value["bytes"] == 256


# --------------------------------------------------------------------------- #
# Listing / find / search / stat / usage
# --------------------------------------------------------------------------- #
def test_list_dir_recursive_with_metadata(tmp_path):
    reg = _reg()
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "x.txt").write_text("hi")
    (tmp_path / "top.txt").write_text("yo")
    r = reg.invoke("list_dir", {"path": str(tmp_path), "recursive": True},
                   authority=Authority.OWNER)
    assert r.ok
    kinds = {e["name"]: e["type"] for e in r.value}
    assert kinds.get("sub") == "dir" and kinds.get("x.txt") == "file"
    assert kinds.get("top.txt") == "file"
    # metadata is present
    top = next(e for e in r.value if e["name"] == "top.txt")
    assert top["size"] == 2 and "mtime" in top


def test_find_files_glob(tmp_path):
    reg = _reg()
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "a.py").write_text("x")
    (tmp_path / "pkg" / "b.txt").write_text("y")
    r = reg.invoke("find_files", {"root": str(tmp_path), "pattern": "**/*.py"},
                   authority=Authority.OWNER)
    assert r.ok
    assert any(p.endswith("a.py") for p in r.value)
    assert not any(p.endswith("b.txt") for p in r.value)


def test_search_in_files(tmp_path):
    reg = _reg()
    (tmp_path / "one.txt").write_text("alpha\nNEEDLE here\nbeta\n")
    (tmp_path / "two.txt").write_text("nothing\n")
    r = reg.invoke("search_in_files", {"root": str(tmp_path), "query": "NEEDLE"},
                   authority=Authority.OWNER)
    assert r.ok and len(r.value) == 1
    hit = r.value[0]
    assert hit["line_no"] == 2 and "NEEDLE" in hit["line"]


def test_path_stat_and_exists(tmp_path):
    reg = _reg()
    target = tmp_path / "f.txt"
    target.write_text("abc")
    s = reg.invoke("path_stat", {"path": str(target)}, authority=Authority.OWNER)
    assert s.ok and s.value["exists"] and s.value["is_file"] and s.value["size"] == 3
    e = reg.invoke("path_exists", {"path": str(target)}, authority=Authority.OWNER)
    assert e.ok and e.value is True
    missing = reg.invoke("path_exists", {"path": str(tmp_path / "nope")},
                         authority=Authority.OWNER)
    assert missing.ok and missing.value is False


def test_disk_usage(tmp_path):
    reg = _reg()
    r = reg.invoke("disk_usage", {"path": str(tmp_path)}, authority=Authority.OWNER)
    assert r.ok and r.value["total"] > 0 and r.value["free"] >= 0


# --------------------------------------------------------------------------- #
# copy / move / mkdir
# --------------------------------------------------------------------------- #
def test_copy_and_move_and_mkdir(tmp_path):
    reg = _reg()
    src = tmp_path / "src.txt"
    src.write_text("payload")
    dst = tmp_path / "copy.txt"
    assert reg.invoke("copy_path", {"src": str(src), "dst": str(dst)},
                      authority=Authority.OWNER).ok
    assert dst.read_text() == "payload" and src.exists()

    moved = tmp_path / "moved.txt"
    assert reg.invoke("move_path", {"src": str(dst), "dst": str(moved)},
                      authority=Authority.OWNER).ok
    assert moved.read_text() == "payload" and not dst.exists()

    newdir = tmp_path / "deep" / "nested"
    assert reg.invoke("make_dir", {"path": str(newdir)}, authority=Authority.OWNER).ok
    assert newdir.is_dir()


def test_copy_directory_tree(tmp_path):
    reg = _reg()
    (tmp_path / "tree").mkdir()
    (tmp_path / "tree" / "leaf.txt").write_text("leaf")
    dst = tmp_path / "tree_copy"
    assert reg.invoke("copy_path", {"src": str(tmp_path / "tree"), "dst": str(dst)},
                      authority=Authority.OWNER).ok
    assert (dst / "leaf.txt").read_text() == "leaf"


# --------------------------------------------------------------------------- #
# delete — the FS_DELETE capability (HIGH, irreversible), properly gated
# --------------------------------------------------------------------------- #
def test_delete_file_owner_runs(tmp_path):
    reg = _reg()
    target = tmp_path / "gone.txt"
    target.write_text("bye")
    r = reg.invoke("delete_file", {"path": str(target)}, authority=Authority.OWNER)
    assert r.ok and not target.exists()


def test_autonomous_delete_escalates_under_default_policy(tmp_path):
    # FS_DELETE is HIGH + irreversible: on the conservative default policy an autonomous
    # delete escalates to the Master rather than running — governance intact.
    reg = _reg()
    target = tmp_path / "keep.txt"
    target.write_text("still here")
    r = reg.invoke("delete_file", {"path": str(target)}, authority=Authority.AUTONOMOUS)
    assert not r.ok and r.requires_owner
    assert target.exists()  # not deleted


def test_autonomous_delete_runs_under_full_control(tmp_path):
    # "NYXARA khude kare": with the owner's standing full_control grant she deletes on her
    # own initiative — real effect, still gated by the policy.
    reg = _sovereign()
    target = tmp_path / "sovereign_gone.txt"
    target.write_text("bye")
    r = reg.invoke("delete_file", {"path": str(target)}, authority=Authority.AUTONOMOUS)
    assert r.ok and not target.exists()


def test_delete_dir_requires_recursive_for_nonempty(tmp_path):
    reg = _reg()
    d = tmp_path / "full"
    d.mkdir()
    (d / "child.txt").write_text("x")
    # non-empty dir without recursive fails safe (OSError -> ToolResult not ok)
    r = reg.invoke("delete_dir", {"path": str(d)}, authority=Authority.OWNER)
    assert not r.ok and d.exists()
    # recursive=True removes the whole tree
    r2 = reg.invoke("delete_dir", {"path": str(d), "recursive": True},
                    authority=Authority.OWNER)
    assert r2.ok and not d.exists()


def test_delete_empty_dir(tmp_path):
    reg = _reg()
    d = tmp_path / "empty"
    d.mkdir()
    r = reg.invoke("delete_dir", {"path": str(d)}, authority=Authority.OWNER)
    assert r.ok and not d.exists()


# --------------------------------------------------------------------------- #
# Full-disk reach — no workspace confinement
# --------------------------------------------------------------------------- #
def test_no_path_confinement_absolute_outside_cwd(tmp_path):
    # tmp_path is an absolute path outside the process cwd; writing/reading it proves the
    # tools reach anywhere on disk (confinement is the capability policy, not a chroot).
    reg = _reg()
    assert os.path.isabs(str(tmp_path))
    target = tmp_path / "anywhere.txt"
    reg.invoke("write_file", {"path": str(target), "content": "reach"},
               authority=Authority.OWNER)
    r = reg.invoke("read_file", {"path": str(target)}, authority=Authority.OWNER)
    assert r.ok and r.value == "reach"
