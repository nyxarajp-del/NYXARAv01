"""NYXARA · mind/vulkan_shim.py — let her on-device brain run on a host with no Vulkan loader.

``liblitert-lm.so`` (the runtime behind ``mind/llm.py::LiteRTLMProvider``) carries a ``DT_NEEDED``
entry for ``libvulkan.so.1`` **unconditionally** — including on the CPU backend, which is the one she
ships with. On a host without the Vulkan loader the dynamic linker refuses to load the library at
all, and every turn dies with::

    libvulkan.so.1: cannot open shared object file: No such file or directory

The usual answer is ``sudo apt install libvulkan1``. That is one command, and it is still the right
answer where someone has root — but "install a system package" is not always available: a locked-down
box, a rootless container, a hosted notebook. So this module removes the requirement instead.

**Why a stub is sound here, and not a lie.** The runtime declares the dependency but imports **no
symbols** from it — verified on the shipped wheel::

    $ nm -D --undefined-only liblitert-lm.so | grep -ci vk
    0

So the linker needs the *library to exist*, not anything inside it. On the CPU backend nothing ever
calls through it. A 344-byte ELF whose only real content is ``DT_SONAME = libvulkan.so.1`` satisfies
the loader completely, and the CPU path then runs exactly as it does with the real loader present
(verified end-to-end with the system library removed: engine load 2.3 s, correct generation).

**Where it stops.** A stub cannot do Vulkan. If the configured backend is ``gpu`` or ``npu`` this
module refuses to shim and lets the rung report itself unavailable, because a GPU backend that
silently loads a do-nothing Vulkan is far worse than one that says it cannot start. The real loader,
when present, always wins — this never shadows it.

Pure stdlib: the ELF is written by :func:`build_stub`, so no compiler, no wheel, and no binary blob
committed to the tree.
"""

from __future__ import annotations

import ctypes
import logging
import platform
import struct
from pathlib import Path
from typing import Optional

log = logging.getLogger("nyxara.mind.vulkan_shim")

__all__ = ["SONAME", "loader_present", "build_stub", "stub_path", "ensure_vulkan_loader"]

SONAME = "libvulkan.so.1"

# ELF e_machine values for the architectures the litert-lm wheels target.
_E_MACHINE = {"x86_64": 62, "amd64": 62, "aarch64": 183, "arm64": 183}

_LOADED = False          # process-wide: the stub only ever needs loading once


def loader_present() -> bool:
    """True when a real ``libvulkan.so.1`` can be dlopened from the normal search path."""
    try:
        ctypes.CDLL(SONAME)
        return True
    except OSError:
        return False


def build_stub(soname: str = SONAME, machine: Optional[str] = None) -> bytes:
    """Return a minimal ELF64 shared object whose only purpose is to carry ``soname``.

    Hand-assembled rather than compiled: a shared object needs only an ELF header, a ``PT_LOAD``
    covering itself, a ``PT_DYNAMIC``, and a dynamic array naming the SONAME plus the string/symbol
    tables the loader dereferences. That is 344 bytes and no toolchain.

    Raises :class:`ValueError` on an architecture we have no ``e_machine`` for — an honest failure is
    better than emitting an ELF the loader will reject for a confusing reason.
    """
    machine = (machine or platform.machine()).lower()
    if machine not in _E_MACHINE:
        raise ValueError(f"no ELF machine id for architecture {machine!r}")

    dynstr = b"\0" + soname.encode() + b"\0"          # index 1 is the soname
    ehdr_size, phdr_size = 64, 56
    off_ph = ehdr_size
    off_dynsym = off_ph + 2 * phdr_size
    off_dynstr = off_dynsym + 24                      # exactly one (null) Elf64_Sym
    off_hash = (off_dynstr + len(dynstr) + 7) & ~7    # 8-aligned
    off_dyn = off_hash + 16                           # nbucket, nchain, bucket[0], chain[0]
    entries = [
        (14, 1),                 # DT_SONAME  -> dynstr index 1
        (4, off_hash),           # DT_HASH    — glibc dereferences this even with no symbols
        (5, off_dynstr),         # DT_STRTAB
        (6, off_dynsym),         # DT_SYMTAB
        (10, len(dynstr)),       # DT_STRSZ
        (11, 24),                # DT_SYMENT
        (0, 0),                  # DT_NULL
    ]
    size = off_dyn + 16 * len(entries)

    buf = bytearray(size)
    struct.pack_into(
        "<16sHHIQQQIHHHHHH", buf, 0,
        b"\x7fELF\x02\x01\x01" + b"\0" * 9,           # ELF64, little-endian, current version
        3,                                            # e_type = ET_DYN
        _E_MACHINE[machine], 1, 0, off_ph, 0, 0,
        ehdr_size, phdr_size, 2, 64, 0, 0)
    # PT_LOAD over the whole file; PT_DYNAMIC over the dynamic array
    struct.pack_into("<IIQQQQQQ", buf, off_ph, 1, 7, 0, 0, 0, size, size, 0x1000)
    struct.pack_into("<IIQQQQQQ", buf, off_ph + phdr_size, 2, 6,
                     off_dyn, off_dyn, off_dyn,
                     16 * len(entries), 16 * len(entries), 8)
    buf[off_dynstr:off_dynstr + len(dynstr)] = dynstr
    struct.pack_into("<IIII", buf, off_hash, 1, 1, 0, 0)   # 1 bucket, 1 chain, both empty
    for i, (tag, val) in enumerate(entries):
        struct.pack_into("<qQ", buf, off_dyn + 16 * i, tag, val)
    return bytes(buf)


def stub_path(settings) -> Path:
    """Where the generated stub lives — beside the weights, under her own data dir."""
    from nyxara.mind.litertlm_assets import model_path
    return model_path(settings).parent / "vulkan-shim" / SONAME


def ensure_vulkan_loader(settings) -> Optional[str]:
    """Make ``libvulkan.so.1`` resolvable for this process. Returns a note, or ``None``.

    ``None`` means "nothing was done", which covers both the good case (a real loader is present)
    and the cases where shimming would be wrong (a GPU/NPU backend, an unknown architecture). Never
    raises: a failure here just leaves the rung unavailable, which the provider reports honestly.
    """
    global _LOADED
    if _LOADED or loader_present():
        return None

    backend = str(getattr(settings.llm, "litertlm_backend", "cpu")).lower()
    if backend != "cpu":
        # A do-nothing Vulkan under a GPU/NPU backend would be a silent lie about what is running.
        log.warning("no Vulkan loader and backend=%s — refusing to shim; install libvulkan1", backend)
        return None
    if not bool(getattr(settings.llm, "litertlm_vulkan_shim", True)):
        return None

    try:
        blob = build_stub()
    except ValueError as exc:                  # unknown architecture — say so, do nothing
        log.warning("cannot build a Vulkan shim here: %s", exc)
        return None

    try:
        path = stub_path(settings)
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists() or path.read_bytes() != blob:
            tmp = path.with_suffix(path.suffix + ".part")
            tmp.write_bytes(blob)
            tmp.replace(path)                  # atomic: never a half-written .so
        ctypes.CDLL(str(path), mode=ctypes.RTLD_GLOBAL)
    except Exception as exc:  # noqa: BLE001 — best-effort; the rung stays honestly unavailable
        log.warning("could not install the Vulkan shim: %s", exc)
        return None

    _LOADED = True
    note = (f"no system Vulkan loader; loaded a {len(blob)}-byte stub so the CPU backend can run "
            f"({path}). Install libvulkan1 to use the real one.")
    log.info("vulkan shim: %s", note)
    return note
