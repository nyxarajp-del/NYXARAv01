"""Tests for nyxara.senses.vision."""

from __future__ import annotations

import struct

import pytest

from nyxara.senses.vision import (ImageAnalysis, ImageInfo, Vision, average_hash,
                                  difference_hash, dominant_colors, hamming, hex_hash,
                                  parse_header, perceptual_hash, resize_gray)


# -------------------- header parsing -------------------- #
def _png(w, h):
    return (b"\x89PNG\r\n\x1a\n" + b"\x00\x00\x00\rIHDR"
            + w.to_bytes(4, "big") + h.to_bytes(4, "big") + b"\x08\x06\x00\x00\x00")


def test_parse_png():
    info = parse_header(_png(640, 480))
    assert info.format == "PNG" and info.width == 640 and info.height == 480


def test_parse_gif():
    gif = b"GIF89a" + (320).to_bytes(2, "little") + (200).to_bytes(2, "little") + b"\x00" * 20
    info = parse_header(gif)
    assert info.format == "GIF" and info.width == 320 and info.height == 200


def test_parse_gif87():
    gif = b"GIF87a" + (10).to_bytes(2, "little") + (20).to_bytes(2, "little") + b"\x00" * 20
    assert parse_header(gif).format == "GIF"


def test_parse_bmp():
    bmp = b"BM" + b"\x00" * 16 + struct.pack("<i", 100) + struct.pack("<i", 50) + b"\x00" * 4
    info = parse_header(bmp)
    assert info.format == "BMP" and info.width == 100 and info.height == 50


def test_parse_bmp_negative_height():
    bmp = b"BM" + b"\x00" * 16 + struct.pack("<i", 100) + struct.pack("<i", -50) + b"\x00" * 4
    assert parse_header(bmp).height == 50  # abs


def test_parse_jpeg():
    # minimal JPEG: SOI + SOF0 marker with dimensions
    jpeg = (b"\xff\xd8" + b"\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00"
            + b"\xff\xc0\x00\x11\x08" + (200).to_bytes(2, "big") + (300).to_bytes(2, "big")
            + b"\x03\x01\x22\x00\x02\x11\x01\x03\x11\x01")
    info = parse_header(jpeg)
    assert info.format == "JPEG" and info.height == 200 and info.width == 300


def test_parse_webp_vp8():
    # VP8 lossy: "VP8 " + size(4) + frame-tag(3) + start-code(3) + width(2) + height(2)
    body = b"VP8 " + b"\x00" * 7 + b"\x9d\x01\x2a" + (640).to_bytes(2, "little") \
        + (480).to_bytes(2, "little") + b"\x00" * 8
    webp = b"RIFF" + b"\x00\x00\x00\x00" + b"WEBP" + body
    info = parse_header(webp)
    assert info.format == "WEBP" and info.width == 640 and info.height == 480


def test_parse_unknown_returns_none():
    assert parse_header(b"not an image at all really here padding") is None


def test_parse_too_short():
    assert parse_header(b"short") is None


# -------------------- ImageInfo -------------------- #
def test_image_info_aspect_and_megapixels():
    i = ImageInfo("PNG", 1000, 500, 12345)
    assert i.aspect_ratio == 2.0 and i.megapixels == 0.5


def test_image_info_zero_height():
    assert ImageInfo("PNG", 100, 0).aspect_ratio == 0.0


def test_image_info_to_dict():
    d = ImageInfo("PNG", 4, 2).to_dict()
    assert d["format"] == "PNG" and d["aspect_ratio"] == 2.0


# -------------------- dominant colours -------------------- #
def test_dominant_colors_proportions():
    pixels = [(250, 0, 0)] * 70 + [(0, 0, 250)] * 30
    dom = dominant_colors(pixels, k=2)
    assert dom[0][1] == 0.7 and dom[1][1] == 0.3
    assert dom[0][0][0] > 200  # red dominant


def test_dominant_colors_respects_k():
    pixels = [(i, i, i) for i in range(0, 256, 8)] * 2
    assert len(dominant_colors(pixels, k=3)) <= 3


def test_dominant_colors_empty():
    assert dominant_colors([]) == []


def test_dominant_colors_quantizes():
    # near-identical colours fall in the same bucket
    pixels = [(10, 10, 10), (12, 12, 12), (11, 9, 13)]
    dom = dominant_colors(pixels, step=32)
    assert len(dom) == 1 and dom[0][1] == 1.0


# -------------------- resize_gray -------------------- #
def test_resize_gray_dimensions():
    m = [[i for i in range(16)] for _ in range(16)]
    out = resize_gray(m, 8, 8)
    assert len(out) == 8 and len(out[0]) == 8


def test_resize_gray_uniform():
    m = [[100] * 10 for _ in range(10)]
    out = resize_gray(m, 4, 4)
    assert all(px == 100 for row in out for px in row)


def test_resize_gray_empty():
    assert resize_gray([], 8, 8) == [[0] * 8 for _ in range(8)]


# -------------------- hashing -------------------- #
def _grad(n=16):
    return [[min(255, (x + y) * 8) for x in range(n)] for y in range(n)]


def test_average_hash_deterministic():
    m = _grad()
    assert average_hash(m) == average_hash(m)


def test_average_hash_identical_zero_distance():
    m = _grad()
    assert hamming(average_hash(m), average_hash(m)) == 0


def test_average_hash_near_duplicate_close():
    grad = _grad()
    near = [[min(255, v + (2 if (i % 5 == 0) else 0)) for i, v in enumerate(row)]
            for row in grad]
    invert = [[255 - v for v in row] for row in grad]
    assert hamming(average_hash(grad), average_hash(near)) < \
        hamming(average_hash(grad), average_hash(invert))


def test_difference_hash_64_bits():
    h = difference_hash(_grad())
    assert 0 <= h < (1 << 64)


def test_difference_hash_near_close():
    grad = _grad()
    near = [[min(255, v + 1) for v in row] for row in grad]
    assert hamming(difference_hash(grad), difference_hash(near)) <= 8


def test_perceptual_hash_robust_to_scale():
    grad = _grad(16)
    # a true 2x nearest-neighbour upscale of the SAME pattern
    grad_big = []
    for row in grad:
        big_row = [v for v in row for _ in range(2)]
        grad_big.append(big_row)
        grad_big.append(big_row)
    # the DCT hash should see them as similar despite different resolution
    assert hamming(perceptual_hash(grad), perceptual_hash(grad_big)) <= 12


def test_perceptual_hash_distinguishes_inverted():
    grad = _grad()
    invert = [[255 - v for v in row] for row in grad]
    assert hamming(perceptual_hash(grad), perceptual_hash(invert)) > 12


def test_hamming():
    assert hamming(0b1010, 0b1000) == 1
    assert hamming(0xFF, 0x00) == 8


def test_hex_hash():
    assert hex_hash(255, bits=64) == "00000000000000ff"


# -------------------- Vision facade -------------------- #
def test_inspect_bytes():
    info = Vision.inspect_bytes(_png(8, 8))
    assert info.format == "PNG"


def test_is_duplicate():
    vis = Vision(dup_threshold=6)
    grad = _grad()
    near = [[min(255, v + 1) for v in row] for row in grad]
    invert = [[255 - v for v in row] for row in grad]
    assert vis.is_duplicate(average_hash(grad), average_hash(near))
    assert not vis.is_duplicate(average_hash(grad), average_hash(invert))


def test_is_duplicate_custom_threshold():
    vis = Vision()
    assert vis.is_duplicate(0b1100, 0b1101, threshold=1)
    assert not vis.is_duplicate(0b1100, 0b0011, threshold=1)


def test_inspect_file(tmp_path):
    p = tmp_path / "img.png"
    p.write_bytes(_png(64, 32))
    info = Vision().inspect(str(p))
    assert info.format == "PNG" and info.width == 64


def test_analyze_header_only_without_pil(tmp_path):
    # without Pillow, analyze falls back to header-only with an honest note
    from nyxara.senses import vision as v
    p = tmp_path / "img.png"
    p.write_bytes(_png(100, 100))
    res = Vision().analyze(str(p))
    assert isinstance(res, ImageAnalysis)
    if not v._HAS_PIL:
        assert res.info.format == "PNG"
        assert res.average_hash is None and "Pillow" in res.note


def test_analysis_to_dict():
    a = ImageAnalysis(info=ImageInfo("PNG", 8, 8), average_hash=255)
    d = a.to_dict()
    assert d["info"]["format"] == "PNG" and d["average_hash"] == hex_hash(255)


# -------------------- pure-stdlib pixel decoders -------------------- #
import zlib  # noqa: E402

from nyxara.senses.vision import decode_bmp, decode_image, decode_png  # noqa: E402


def _paeth(a, b, c):
    p = a + b - c
    pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
    return a if (pa <= pb and pa <= pc) else (b if pb <= pc else c)


def _encode_png(w, h, pixels, filter_type=0):
    """Encode an 8-bit RGB PNG applying one filter type uniformly (exercises decode)."""
    def chunk(typ, data):
        return (struct.pack(">I", len(data)) + typ + data
                + struct.pack(">I", zlib.crc32(typ + data) & 0xFFFFFFFF))

    bpp, stride = 3, w * 3
    recon_rows = []
    for y in range(h):
        row = bytearray()
        for x in range(w):
            row += bytes(pixels[y * w + x])
        recon_rows.append(row)
    raw = bytearray()
    prev = bytearray(stride)
    for recon in recon_rows:
        raw.append(filter_type)
        filt = bytearray(stride)
        for i in range(stride):
            a = recon[i - bpp] if i >= bpp else 0
            b = prev[i]
            c = prev[i - bpp] if i >= bpp else 0
            pred = (0, a, b, (a + b) // 2, _paeth(a, b, c))[filter_type]
            filt[i] = (recon[i] - pred) & 255
        raw += filt
        prev = recon
    ihdr = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr)
            + chunk(b"IDAT", zlib.compress(bytes(raw))) + chunk(b"IEND", b""))


_PIXELS = [(250, 5, 5), (5, 250, 5), (5, 5, 250), (200, 200, 50),
           (30, 60, 90), (120, 130, 140)]  # 3x2


@pytest.mark.parametrize("ftype", [0, 1, 2, 3, 4])
def test_decode_png_all_filters_roundtrip(ftype):
    png = _encode_png(3, 2, _PIXELS, filter_type=ftype)
    assert decode_png(png) == (3, 2, _PIXELS)
    assert decode_image(png) == (3, 2, _PIXELS)


def test_decode_bmp_roundtrip():
    w, h = 2, 2
    rows_top_down = [[(10, 20, 30), (40, 50, 60)], [(70, 80, 90), (100, 110, 120)]]
    row_bytes = ((24 * w + 31) // 32) * 4
    body = bytearray()
    for row in reversed(rows_top_down):  # BMP positive height is bottom-up
        rb = bytearray()
        for (r, g, b) in row:
            rb += bytes((b, g, r))
        rb += bytes(row_bytes - len(rb))
        body += rb
    offset = 54
    header = b"BM" + struct.pack("<IHHI", offset + len(body), 0, 0, offset)
    dib = struct.pack("<IiiHHIIiiII", 40, w, h, 1, 24, 0, len(body), 2835, 2835, 0, 0)
    bmp = bytes(header + dib + body)
    assert decode_bmp(bmp) == (2, 2, [(10, 20, 30), (40, 50, 60),
                                      (70, 80, 90), (100, 110, 120)])


def test_decode_image_defers_unknown_format():
    # JPEG/WEBP need a heavy codec → stdlib decode returns None (Pillow handles them)
    assert decode_image(b"\xff\xd8\xff\xe0" + b"\x00" * 40) is None


def test_perception_works_without_pillow(tmp_path, monkeypatch):
    # the whole point of Step E: real hashing + dominant colours on a bare machine (no PIL)
    from nyxara.senses import vision as v
    monkeypatch.setattr(v, "_HAS_PIL", False)
    p = tmp_path / "img.png"
    p.write_bytes(_encode_png(3, 2, _PIXELS, filter_type=4))
    res = Vision().analyze(str(p))
    assert res.info.format == "PNG" and res.info.width == 3
    assert res.average_hash is not None and res.perceptual_hash is not None
    assert res.difference_hash is not None
    assert res.dominant_colors  # real palette, decoded with zero dependencies


def test_decode_png_rejects_oversized(monkeypatch):
    from nyxara.senses import vision as v
    monkeypatch.setattr(v, "_MAX_DECODE_PIXELS", 4)  # tiny cap
    png = _encode_png(3, 2, _PIXELS)  # 6 pixels > cap → defer
    assert v.decode_png(png) is None


# -------------------- real OCR (skips cleanly without Tesseract) -------------------- #
def _tesseract_ready():
    try:
        import pytesseract  # type: ignore
        pytesseract.get_tesseract_version()
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _tesseract_ready(),
                    reason="requires pytesseract + the tesseract binary")
def test_real_ocr_reads_rendered_text(tmp_path):
    """Render real text into an image and confirm Tesseract reads it back."""
    Image = pytest.importorskip("PIL.Image")
    from PIL import ImageDraw, ImageFont
    p = tmp_path / "hello.png"
    im = Image.new("RGB", (340, 90), "white")
    dr = ImageDraw.Draw(im)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 40)
    except Exception:
        font = ImageFont.load_default()
    dr.text((12, 22), "NYXARA SEES", fill="black", font=font)
    im.save(str(p))

    res = Vision().analyze(str(p), ocr=True)
    assert res.note == "", f"OCR errored: {res.note}"
    assert res.ocr_text and "NYXARA" in res.ocr_text.upper()
    # real Pillow-backed perception too: hashes and colours are populated
    assert res.perceptual_hash is not None and res.dominant_colors
