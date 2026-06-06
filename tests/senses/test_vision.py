"""Tests for nyxara.senses.vision."""

from __future__ import annotations

import struct

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
