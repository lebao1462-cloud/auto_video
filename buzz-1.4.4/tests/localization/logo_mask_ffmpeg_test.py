"""End-to-end FFmpeg tests for multi-region source-logo blur.

The tests create purely synthetic checkerboard frames; they require neither
videos from users nor Gemini/TTS.
"""
from __future__ import annotations

import shutil
import subprocess

import pytest
from PIL import Image, ImageChops, ImageDraw

from buzz.localization.logo_mask import LogoMaskConfig, LogoMaskRegion
from buzz.localization.final_render import build_custom_logo_mask_filter_graph

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="FFmpeg unavailable")


@pytest.mark.parametrize("width,height", [(320, 180), (180, 320)])
def test_custom_blur_affects_multiple_corners_without_covering_center(tmp_path, width, height):
    image = Image.new("RGB", (width, height), "#ffffff")
    brush = ImageDraw.Draw(image)
    for y in range(0, height, 6):
        for x in range(0, width, 6):
            if (x // 6 + y // 6) % 2:
                brush.rectangle((x, y, x + 5, y + 5), fill="#101010")
    source = tmp_path / "source.png"
    image.save(source)

    regions = (
        LogoMaskRegion("top-left", "Top left", True, .02, .02, .20, .13, 18),
        LogoMaskRegion("bottom-right", "Bottom right", True, .77, .82, .20, .15, 21),
        LogoMaskRegion("bottom-left", "Bottom left", True, .02, .82, .20, .15, 12),
    )
    config = LogoMaskConfig(mode="custom", regions=regions)
    graph = build_custom_logo_mask_filter_graph(config)
    result = tmp_path / "filtered.png"
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
           "-i", str(source), "-filter_complex", graph, "-map", "[vout]",
           "-frames:v", "1", str(result)]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=45)
    assert proc.returncode == 0, proc.stderr
    assert result.is_file()

    out = Image.open(result).convert("RGB")
    assert out.size == image.size
    diff = ImageChops.difference(image, out)
    # Center is not under any mask; permit a tiny unavoidable color-conversion delta.
    cx, cy = width // 2, height // 2
    center = diff.crop((cx-12, cy-12, cx+12, cy+12))
    center_max = max(max(pixel) for pixel in center.getdata())
    assert center_max <= 4, f"Unexpected masking of frame center (max delta {center_max})"

    for region in regions:
        x0, y0 = int(region.x*width), int(region.y*height)
        x1, y1 = int((region.x + region.w)*width), int((region.y + region.h)*height)
        affected = diff.crop((x0+2, y0+2, max(x0+3,x1-2), max(y0+3,y1-2)))
        changed = sum(1 for pixel in affected.getdata() if max(pixel) >= 20)
        assert changed >= 10, f"Mask {region.id} did not visibly blur corner"


@pytest.mark.parametrize("width,height", [(320, 180), (180, 320)])
def test_tiny_corner_mask_does_not_crash_ffmpeg(tmp_path, width, height):
    source = tmp_path / "tiny-input.png"
    Image.new("RGB", (width, height), "grey").save(source)
    config = LogoMaskConfig(
        mode="custom",
        regions=(LogoMaskRegion("tiny", "Near edge", True, .99, .99, .01, .01, 40),),
    )
    graph = build_custom_logo_mask_filter_graph(config)
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
           "-i", str(source), "-filter_complex", graph, "-map", "[vout]",
           "-frames:v", "1", str(tmp_path / "tiny-output.png")]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=45)
    assert result.returncode == 0, result.stderr
