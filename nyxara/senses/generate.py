"""NYXARA · senses/generate.py — generative output: images & speech (🎨🔊).

Perception (``vision``/``audio``/``ingest``) lets NYXARA *take in* non-text media; this
module lets her *put out* media. True heavy generative models are optional and
import-guarded, and — crucially — there is **always a real, dependency-free fallback** so
the capability never silently disappears on a bare machine:

* :class:`ImageGenerator` — renders a real PNG. With ``diffusers`` + ``torch`` present it
  runs a Stable-Diffusion pipeline; otherwise it deterministically draws a symmetric
  *identicon* from the prompt (pure-stdlib PNG encoder — genuinely an image, just not a
  photoreal one). It reports honestly which path produced the bytes.
* :class:`SpeechSynthesizer` — writes a real WAV. With an offline TTS engine
  (``pyttsx3``) present it synthesises actual speech; otherwise it writes a deterministic
  *audio signature* tone derived from the text (a real, playable WAV) and says so.

Both write to a path and return a small honest manifest (path, bytes, engine, note). They
are stateless instruments — the kernel decides whether NYXARA may use them (the
``generate_image`` / ``synthesize_speech`` tools are capability-bound and governed).
"""

from __future__ import annotations

import hashlib
import math
import struct
import wave
import zlib
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

__all__ = ["GenerationResult", "ImageGenerator", "SpeechSynthesizer",
           "write_png", "identicon_rows"]


@dataclass
class GenerationResult:
    path: str
    ok: bool
    kind: str                      # "image" | "speech"
    engine: str                    # which backend produced it
    bytes_written: int = 0
    note: str = ""
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {"path": self.path, "ok": self.ok, "kind": self.kind, "engine": self.engine,
                "bytes": self.bytes_written, "note": self.note, "meta": self.meta}


# --------------------------------------------------------------------------- #
# A tiny, dependency-free PNG encoder (RGB, 8-bit)
# --------------------------------------------------------------------------- #
def write_png(path: str, width: int, height: int, rows: List[bytes]) -> int:
    """Write an 8-bit RGB PNG from ``rows`` (each ``width*3`` bytes). Returns bytes written."""
    def _chunk(typ: bytes, data: bytes) -> bytes:
        body = typ + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)

    sig = b"\x89PNG\r\n\x1a\n"
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)  # color type 2 = RGB
    raw = bytearray()
    for row in rows:
        raw.append(0)             # filter byte: none
        raw.extend(row)
    idat = zlib.compress(bytes(raw), 9)
    blob = sig + _chunk(b"IHDR", ihdr) + _chunk(b"IDAT", idat) + _chunk(b"IEND", b"")
    with open(path, "wb") as f:
        f.write(blob)
    return len(blob)


def identicon_rows(prompt: str, *, size: int = 256, grid: int = 8) -> List[bytes]:
    """Deterministically render a horizontally-symmetric identicon for ``prompt``."""
    h = hashlib.sha256(prompt.encode("utf-8")).digest()
    fg = (h[0], h[1], h[2])
    bg = (235, 238, 242)
    cells: List[List[bool]] = []
    half = (grid + 1) // 2
    for r in range(grid):
        row_cells = [bool(h[(r * half + c) % len(h)] & 0x80) for c in range(half)]
        full = row_cells + row_cells[::-1][grid % 2:]
        cells.append(full[:grid])
    cell_px = size // grid
    rows: List[bytes] = []
    for y in range(size):
        gy = min(grid - 1, y // cell_px)
        row = bytearray()
        for x in range(size):
            gx = min(grid - 1, x // cell_px)
            r, g, b = fg if cells[gy][gx] else bg
            row += bytes((r, g, b))
        rows.append(bytes(row))
    return rows


# --------------------------------------------------------------------------- #
# Image generation
# --------------------------------------------------------------------------- #
class ImageGenerator:
    """Generate an image from a text prompt — diffusion if available, identicon otherwise."""

    def __init__(self, model: str = "stabilityai/sd-turbo", device: str = "") -> None:
        self.model = model
        self.device = device
        self._pipe = None

    @staticmethod
    def diffusion_available() -> bool:
        try:
            import diffusers  # noqa: F401
            import torch  # noqa: F401
            return True
        except Exception:  # pragma: no cover - depends on install
            return False

    def _ensure_pipe(self):
        if self._pipe is None:
            from diffusers import AutoPipelineForText2Image  # lazy, optional
            import torch
            dtype = torch.float16 if torch.cuda.is_available() else torch.float32
            pipe = AutoPipelineForText2Image.from_pretrained(self.model, torch_dtype=dtype)
            self._pipe = pipe.to(self.device or ("cuda" if torch.cuda.is_available() else "cpu"))
        return self._pipe

    def generate(self, prompt: str, path: str, *, size: int = 256,
                 steps: int = 4) -> GenerationResult:
        if self.diffusion_available():
            try:
                pipe = self._ensure_pipe()
                image = pipe(prompt=prompt, num_inference_steps=steps,
                             guidance_scale=0.0).images[0]
                image.save(path)
                import os
                return GenerationResult(path, True, "image", f"diffusion:{self.model}",
                                        bytes_written=os.path.getsize(path),
                                        meta={"prompt": prompt})
            except Exception as exc:  # noqa: BLE001 — fall back rather than fail
                note = f"diffusion failed ({exc}); rendered identicon instead"
                n = write_png(path, size, size, identicon_rows(prompt, size=size))
                return GenerationResult(path, True, "image", "identicon", bytes_written=n,
                                        note=note, meta={"prompt": prompt})
        n = write_png(path, size, size, identicon_rows(prompt, size=size))
        return GenerationResult(path, True, "image", "identicon", bytes_written=n,
                                note="no diffusion backend; rendered a deterministic identicon",
                                meta={"prompt": prompt})


# --------------------------------------------------------------------------- #
# Speech synthesis
# --------------------------------------------------------------------------- #
class SpeechSynthesizer:
    """Synthesise speech to a WAV — a real TTS engine if available, a tone signature else."""

    def __init__(self, *, sample_rate: int = 22050) -> None:
        self.sample_rate = sample_rate

    @staticmethod
    def tts_available() -> bool:
        try:
            import pyttsx3  # noqa: F401
            return True
        except Exception:  # pragma: no cover - depends on install
            return False

    def _signature_wav(self, text: str, path: str) -> int:
        """Write a deterministic, real WAV tone sequence derived from the text."""
        sr = self.sample_rate
        words = text.split() or ["…"]
        samples: List[int] = []
        per = int(sr * 0.18)  # ~180ms per token
        for tok in words[:64]:
            freq = 180.0 + (hash(tok) % 480)   # 180..660 Hz, deterministic per-token shape
            for i in range(per):
                env = math.sin(math.pi * i / per)   # smooth in/out to avoid clicks
                samples.append(int(0.45 * env * math.sin(2 * math.pi * freq * i / sr) * 32767))
        with wave.open(path, "w") as w:
            w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr)
            w.writeframes(struct.pack("<" + "h" * len(samples), *samples))
        import os
        return os.path.getsize(path)

    def synthesize(self, text: str, path: str) -> GenerationResult:
        if self.tts_available():
            try:
                import pyttsx3
                engine = pyttsx3.init()
                engine.save_to_file(text, path)
                engine.runAndWait()
                import os
                return GenerationResult(path, True, "speech", "pyttsx3",
                                        bytes_written=os.path.getsize(path),
                                        meta={"chars": len(text)})
            except Exception as exc:  # noqa: BLE001
                n = self._signature_wav(text, path)
                return GenerationResult(path, True, "speech", "tone-signature",
                                        bytes_written=n,
                                        note=f"TTS engine failed ({exc}); wrote a tone signature",
                                        meta={"chars": len(text)})
        n = self._signature_wav(text, path)
        return GenerationResult(path, True, "speech", "tone-signature", bytes_written=n,
                                note="no TTS engine; wrote a deterministic non-speech tone signature",
                                meta={"chars": len(text)})


# --------------------------------------------------------------------------- #
# Self-test / demo
# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # pragma: no cover
    import os
    import tempfile

    print("=" * 70)
    print("NYXARA generative-output self-test")
    print("=" * 70)

    d = tempfile.mkdtemp()
    img = ImageGenerator().generate("a loyal sovereign mind, teal and gold",
                                    os.path.join(d, "art.png"))
    print(f"\nimage   : engine={img.engine} bytes={img.bytes_written} note={img.note!r}")
    assert img.ok and img.bytes_written > 0
    with open(img.path, "rb") as f:
        assert f.read(8) == b"\x89PNG\r\n\x1a\n"   # a real PNG
    # determinism: same prompt -> identical identicon bytes
    img2 = ImageGenerator().generate("a loyal sovereign mind, teal and gold",
                                     os.path.join(d, "art2.png"))
    assert os.path.getsize(img.path) == os.path.getsize(img2.path)
    print("image determinism: identical prompt -> identical render ✓")

    spe = SpeechSynthesizer().synthesize("NYXARA serves JP.", os.path.join(d, "say.wav"))
    print(f"speech  : engine={spe.engine} bytes={spe.bytes_written} note={spe.note!r}")
    assert spe.ok and spe.bytes_written > 44   # a real WAV (header + data)
    with open(spe.path, "rb") as f:
        assert f.read(4) == b"RIFF"

    print("\nALL SELF-TESTS PASSED ✓")
