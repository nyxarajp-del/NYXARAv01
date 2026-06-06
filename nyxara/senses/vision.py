"""NYXARA · senses/vision.py — visual perception, header-deep & dependency-light (👁).

Sight, like every NYXARA sense, must work on a bare machine and grow sharper when better
tools are present. The *core* of this module is pure standard library and needs no image
stack at all:

* **Format & dimensions** — :func:`parse_header` decodes PNG / JPEG / GIF / BMP / WEBP
  *from raw bytes*, reading width, height and format straight out of the file header. No
  decoder, no dependency.
* **Perceptual hashing** — :func:`average_hash`, :func:`difference_hash` and
  :func:`perceptual_hash` (a pure-python 2-D DCT) operate on a grayscale matrix, with
  :func:`hamming` for similarity. These power **deduplication** and near-duplicate
  detection: two visually similar images have a small Hamming distance.
* **Dominant colours** — :func:`dominant_colors` quantises a pixel list into the palette
  that defines the image.

The :class:`Vision` facade bridges *real* image files to these algorithms using ``Pillow``
when it is installed — extracting the grayscale matrix and pixel list — and degrades
honestly (header-only, with a note) when it is not. OCR (``pytesseract``) and CLIP-style
embeddings are optional hooks that return ``None`` plus a note when their libraries are
absent. Nothing here raises because a library is missing.

Pure standard library at the core; ``Pillow`` / ``pytesseract`` / CLIP optional.
"""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass, field
from statistics import median
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:  # optional, never required
    from PIL import Image as _PILImage  # type: ignore
    _HAS_PIL = True
except Exception:  # noqa: BLE001
    _HAS_PIL = False

__all__ = [
    "ImageInfo",
    "ImageAnalysis",
    "parse_header",
    "dominant_colors",
    "resize_gray",
    "average_hash",
    "difference_hash",
    "perceptual_hash",
    "hamming",
    "hex_hash",
    "Vision",
]

Matrix = List[List[int]]
Pixel = Tuple[int, int, int]


# --------------------------------------------------------------------------- #
# Image info
# --------------------------------------------------------------------------- #
@dataclass
class ImageInfo:
    format: str
    width: int
    height: int
    n_bytes: int = 0

    @property
    def aspect_ratio(self) -> float:
        return self.width / self.height if self.height else 0.0

    @property
    def megapixels(self) -> float:
        return round(self.width * self.height / 1_000_000, 4)

    def to_dict(self) -> Dict[str, Any]:
        return {"format": self.format, "width": self.width, "height": self.height,
                "aspect_ratio": round(self.aspect_ratio, 4), "megapixels": self.megapixels,
                "n_bytes": self.n_bytes}


# --------------------------------------------------------------------------- #
# Header parsing (pure python)
# --------------------------------------------------------------------------- #
def _parse_jpeg(data: bytes) -> Optional[Tuple[int, int]]:
    i = 2
    n = len(data)
    sof = {0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF}
    while i + 9 < n:
        if data[i] != 0xFF:
            i += 1
            continue
        marker = data[i + 1]
        if marker in sof:
            height = int.from_bytes(data[i + 5:i + 7], "big")
            width = int.from_bytes(data[i + 7:i + 9], "big")
            return width, height
        if marker in (0xD8, 0xD9) or 0xD0 <= marker <= 0xD7:
            i += 2
            continue
        seg_len = int.from_bytes(data[i + 2:i + 4], "big")
        if seg_len <= 0:
            break
        i += 2 + seg_len
    return None


def _parse_webp(data: bytes) -> Optional[Tuple[int, int]]:
    if data[8:12] != b"WEBP":
        return None
    chunk = data[12:16]
    try:
        if chunk == b"VP8 ":
            w = int.from_bytes(data[26:28], "little") & 0x3FFF
            h = int.from_bytes(data[28:30], "little") & 0x3FFF
            return w, h
        if chunk == b"VP8L":
            b = int.from_bytes(data[21:25], "little")
            w = (b & 0x3FFF) + 1
            h = ((b >> 14) & 0x3FFF) + 1
            return w, h
        if chunk == b"VP8X":
            w = int.from_bytes(data[24:27], "little") + 1
            h = int.from_bytes(data[27:30], "little") + 1
            return w, h
    except Exception:  # noqa: BLE001
        return None
    return None


def parse_header(data: bytes) -> Optional[ImageInfo]:
    """Decode format/width/height from raw image bytes (PNG/JPEG/GIF/BMP/WEBP)."""
    n = len(data)
    if n < 24:
        return None
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        w = int.from_bytes(data[16:20], "big")
        h = int.from_bytes(data[20:24], "big")
        return ImageInfo("PNG", w, h, n)
    if data[:6] in (b"GIF87a", b"GIF89a"):
        w = int.from_bytes(data[6:8], "little")
        h = int.from_bytes(data[8:10], "little")
        return ImageInfo("GIF", w, h, n)
    if data[:2] == b"BM":
        w = struct.unpack_from("<i", data, 18)[0]
        h = abs(struct.unpack_from("<i", data, 22)[0])
        return ImageInfo("BMP", w, h, n)
    if data[:2] == b"\xff\xd8":
        wh = _parse_jpeg(data)
        if wh:
            return ImageInfo("JPEG", wh[0], wh[1], n)
    if data[:4] == b"RIFF":
        wh = _parse_webp(data)
        if wh:
            return ImageInfo("WEBP", wh[0], wh[1], n)
    return None


# --------------------------------------------------------------------------- #
# Dominant colours
# --------------------------------------------------------------------------- #
def dominant_colors(pixels: Sequence[Pixel], *, k: int = 5, step: int = 32
                    ) -> List[Tuple[Pixel, float]]:
    """Quantise pixels into a coarse palette and return the top-k with proportions."""
    if not pixels:
        return []
    buckets: Dict[Pixel, int] = {}
    for r, g, b in pixels:
        key = (r // step, g // step, b // step)
        buckets[key] = buckets.get(key, 0) + 1
    total = len(pixels)
    ranked = sorted(buckets.items(), key=lambda kv: -kv[1])[:k]
    half = step // 2
    return [((br * step + half, bg * step + half, bb * step + half), cnt / total)
            for (br, bg, bb), cnt in ranked]


# --------------------------------------------------------------------------- #
# Grayscale resize + perceptual hashes
# --------------------------------------------------------------------------- #
def resize_gray(m: Matrix, tw: int, th: int) -> Matrix:
    """Area-average resize of a grayscale matrix to ``tw`` × ``th``."""
    sh = len(m)
    sw = len(m[0]) if sh else 0
    if sh == 0 or sw == 0:
        return [[0] * tw for _ in range(th)]
    out = [[0] * tw for _ in range(th)]
    for ty in range(th):
        y0 = ty * sh // th
        y1 = max(y0 + 1, (ty + 1) * sh // th)
        for tx in range(tw):
            x0 = tx * sw // tw
            x1 = max(x0 + 1, (tx + 1) * sw // tw)
            tot = cnt = 0
            for y in range(y0, min(y1, sh)):
                row = m[y]
                for x in range(x0, min(x1, sw)):
                    tot += row[x]
                    cnt += 1
            out[ty][tx] = tot // cnt if cnt else 0
    return out


def _pack(bits: Sequence[int]) -> int:
    h = 0
    for b in bits:
        h = (h << 1) | (1 if b else 0)
    return h


def average_hash(m: Matrix, *, size: int = 8) -> int:
    g = resize_gray(m, size, size)
    flat = [px for row in g for px in row]
    mean = sum(flat) / len(flat)
    return _pack([1 if px > mean else 0 for px in flat])


def difference_hash(m: Matrix, *, size: int = 8) -> int:
    g = resize_gray(m, size + 1, size)
    bits = [1 if g[y][x] > g[y][x + 1] else 0 for y in range(size) for x in range(size)]
    return _pack(bits)


def _dct1(vec: Sequence[float]) -> List[float]:
    N = len(vec)
    out = [0.0] * N
    for u in range(N):
        s = 0.0
        for x in range(N):
            s += vec[x] * math.cos(math.pi * (2 * x + 1) * u / (2 * N))
        cu = (1.0 / math.sqrt(2)) if u == 0 else 1.0
        out[u] = 0.5 * cu * s
    return out


def perceptual_hash(m: Matrix, *, size: int = 8, dct_size: int = 32) -> int:
    """A DCT-based perceptual hash (robust to scaling/minor edits)."""
    g = [[float(px) for px in row] for row in resize_gray(m, dct_size, dct_size)]
    rows = [_dct1(r) for r in g]
    cols = [_dct1(list(c)) for c in zip(*rows)]
    d = [list(r) for r in zip(*cols)]
    low = [d[y][x] for y in range(size) for x in range(size)]
    med = median(low[1:])  # exclude the DC term from the median
    return _pack([1 if v > med else 0 for v in low])


def hamming(a: int, b: int) -> int:
    """Bit-distance between two hashes (number of differing bits)."""
    return bin(a ^ b).count("1")


def hex_hash(h: int, *, bits: int = 64) -> str:
    return format(h, f"0{bits // 4}x")


# --------------------------------------------------------------------------- #
# Analysis result
# --------------------------------------------------------------------------- #
@dataclass
class ImageAnalysis:
    info: Optional[ImageInfo]
    dominant_colors: List[Tuple[Pixel, float]] = field(default_factory=list)
    average_hash: Optional[int] = None
    difference_hash: Optional[int] = None
    perceptual_hash: Optional[int] = None
    ocr_text: Optional[str] = None
    note: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {"info": self.info.to_dict() if self.info else None,
                "dominant_colors": [[list(c), round(p, 4)] for c, p in self.dominant_colors],
                "average_hash": hex_hash(self.average_hash) if self.average_hash is not None else None,
                "difference_hash": hex_hash(self.difference_hash) if self.difference_hash is not None else None,
                "perceptual_hash": hex_hash(self.perceptual_hash) if self.perceptual_hash is not None else None,
                "ocr_text": self.ocr_text, "note": self.note}


# --------------------------------------------------------------------------- #
# Facade
# --------------------------------------------------------------------------- #
class Vision:
    """Bridges real image files to the pure-python perception algorithms via Pillow."""

    def __init__(self, *, dup_threshold: int = 6) -> None:
        self.dup_threshold = dup_threshold

    # ---- header-only (always works) ---- #
    @staticmethod
    def inspect_bytes(data: bytes) -> Optional[ImageInfo]:
        return parse_header(data)

    def inspect(self, path: str) -> Optional[ImageInfo]:
        with open(path, "rb") as f:
            return parse_header(f.read(64)) or self._pil_info(path)

    def _pil_info(self, path: str) -> Optional[ImageInfo]:
        if not _HAS_PIL:
            return None
        try:  # pragma: no cover - exercised only with Pillow
            with _PILImage.open(path) as im:
                import os
                return ImageInfo(im.format or "?", im.width, im.height,
                                 os.path.getsize(path))
        except Exception:  # noqa: BLE001
            return None

    # ---- pixel access via Pillow ---- #
    def gray_matrix(self, path: str, *, max_dim: int = 256) -> Optional[Matrix]:
        if not _HAS_PIL:
            return None
        try:  # pragma: no cover - exercised only with Pillow
            with _PILImage.open(path) as im:
                im = im.convert("L")
                im.thumbnail((max_dim, max_dim))
                w, h = im.size
                px = list(im.getdata())
                return [[px[y * w + x] for x in range(w)] for y in range(h)]
        except Exception:  # noqa: BLE001
            return None

    def pixels(self, path: str, *, max_dim: int = 128) -> Optional[List[Pixel]]:
        if not _HAS_PIL:
            return None
        try:  # pragma: no cover - exercised only with Pillow
            with _PILImage.open(path) as im:
                im = im.convert("RGB")
                im.thumbnail((max_dim, max_dim))
                return list(im.getdata())
        except Exception:  # noqa: BLE001
            return None

    # ---- optional OCR / embedding hooks ---- #
    def ocr(self, path: str) -> Tuple[Optional[str], str]:
        try:
            import pytesseract  # type: ignore
        except ImportError:
            return None, "pytesseract not installed; OCR unavailable"
        if not _HAS_PIL:
            return None, "Pillow not installed; OCR unavailable"
        try:  # pragma: no cover - exercised only with the lib
            with _PILImage.open(path) as im:
                return pytesseract.image_to_string(im).strip(), ""
        except Exception as exc:  # noqa: BLE001
            return None, f"OCR failed: {exc}"

    # ---- full analysis ---- #
    def analyze(self, path: str, *, ocr: bool = False) -> ImageAnalysis:
        info = self.inspect(path)
        gray = self.gray_matrix(path)
        if gray is None:
            return ImageAnalysis(info=info, note="Pillow not installed; header-only analysis")
        px = self.pixels(path) or []
        ocr_text = None
        note = ""
        if ocr:
            ocr_text, note = self.ocr(path)
        return ImageAnalysis(
            info=info, dominant_colors=dominant_colors(px),
            average_hash=average_hash(gray), difference_hash=difference_hash(gray),
            perceptual_hash=perceptual_hash(gray), ocr_text=ocr_text, note=note)

    # ---- deduplication ---- #
    def is_duplicate(self, hash_a: int, hash_b: int, *, threshold: Optional[int] = None
                     ) -> bool:
        thr = self.dup_threshold if threshold is None else threshold
        return hamming(hash_a, hash_b) <= thr


# --------------------------------------------------------------------------- #
# Self-test / demo
# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # pragma: no cover
    print("=" * 70)
    print("NYXARA vision self-test")
    print("=" * 70)

    # craft a valid PNG header (no image library needed)
    png = (b"\x89PNG\r\n\x1a\n" + b"\x00\x00\x00\rIHDR"
           + (640).to_bytes(4, "big") + (480).to_bytes(4, "big")
           + b"\x08\x06\x00\x00\x00")
    info = parse_header(png)
    print(f"\nPNG header          : {info.to_dict()}")
    assert info.format == "PNG" and info.width == 640 and info.height == 480
    assert abs(info.aspect_ratio - 4 / 3) < 1e-6

    # GIF + BMP headers
    gif = b"GIF89a" + (320).to_bytes(2, "little") + (200).to_bytes(2, "little") + b"\x00" * 20
    assert parse_header(gif).format == "GIF" and parse_header(gif).width == 320
    bmp = b"BM" + b"\x00" * 16 + struct.pack("<i", 100) + struct.pack("<i", 50) + b"\x00" * 4
    assert parse_header(bmp).format == "BMP" and parse_header(bmp).width == 100
    print("GIF/BMP headers     : decoded ✓")

    # dominant colours
    pixels = [(250, 5, 5)] * 70 + [(5, 5, 250)] * 30
    dom = dominant_colors(pixels, k=2)
    print(f"\ndominant colours    : {dom}")
    assert dom[0][1] == 0.7 and dom[0][0][0] > 200   # mostly red

    # perceptual hashing & dedup
    grad = [[min(255, (x + y) * 8) for x in range(16)] for y in range(16)]
    near = [[min(255, (x + y) * 8 + (3 if (x + y) % 4 == 0 else 0)) for x in range(16)]
            for y in range(16)]
    invert = [[255 - v for v in row] for row in grad]

    ah, ah2, ah3 = average_hash(grad), average_hash(near), average_hash(invert)
    ph, ph2, ph3 = perceptual_hash(grad), perceptual_hash(near), perceptual_hash(invert)
    print(f"\naverage_hash grad   : {hex_hash(ah)}")
    print(f"hamming near        : {hamming(ah, ah2)} (similar)")
    print(f"hamming invert      : {hamming(ah, ah3)} (different)")
    assert hamming(ah, ah2) < hamming(ah, ah3)        # near-dup closer than inverted
    assert hamming(ph, ph2) < hamming(ph, ph3)

    vis = Vision(dup_threshold=6)
    assert vis.is_duplicate(ah, ah2)                  # a near-duplicate
    assert not vis.is_duplicate(ah, ah3)              # not the inverted image
    print(f"\ndedup verdict       : near=dup ✓, inverted=not-dup ✓")

    # difference hash sanity
    dh, dh2 = difference_hash(grad), difference_hash(near)
    assert hamming(dh, dh2) <= 8
    print(f"difference_hash     : near distance = {hamming(dh, dh2)}")

    # graceful note when Pillow is absent
    print(f"\nPillow available    : {_HAS_PIL}")
    print("\nALL SELF-TESTS PASSED ✓")
