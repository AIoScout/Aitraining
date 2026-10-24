"""C-vs-Python parity for the capture -> preprocessing box resize.

The camera captures at kCaptureFrameSide (e.g. 160) while the preprocessing
front-end and the model work at kPreprocessFrameSide / img_size (e.g. 96).
Both the ESP32-P4 (TFLite/frame_resize.h) and the desktop app
(record_controller._resize_square_box_average_u8) must resample the SAME
way, or the device sees a different image than the host preview and the
training captures.

This suite compiles the C header and compares every pixel of:

  gray   FrameResizeBoxAverageGray   vs _resize_square_box_average_u8
  rgb    FrameResizeBoxAverageRgb    vs the same helper per channel after
                                      applying the WB gains to the source
  host   _raw96_to_png(160x160)      vs the helper (the live serial preview)

Run:
    .venv/bin/python tests/test_frame_resize_parity.py
"""
from __future__ import annotations

import io
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import record_controller as rc  # noqa: E402

HEADER_DIR = REPO / "TFLite"
HARNESS = r"""
#include <stdio.h>
#include <stdlib.h>

#include "frame_resize.h"

int main(int argc, char **argv) {
  if (argc != 8) {
    fprintf(stderr, "usage: %s in.raw src_side dst_side channels wb_r wb_b out.raw\n", argv[0]);
    return 2;
  }
  const int src_side = atoi(argv[2]);
  const int dst_side = atoi(argv[3]);
  const int channels = atoi(argv[4]);
  const int wb_r = atoi(argv[5]);
  const int wb_b = atoi(argv[6]);
  if (src_side <= 0 || dst_side <= 0 || (channels != 1 && channels != 3)) return 2;
  const size_t in_n = (size_t)src_side * (size_t)src_side * (size_t)channels;
  const size_t out_n = (size_t)dst_side * (size_t)dst_side * (size_t)channels;
  unsigned char *in = (unsigned char *)malloc(in_n);
  unsigned char *out = (unsigned char *)calloc(out_n, 1);
  if (!in || !out) return 3;
  FILE *f = fopen(argv[1], "rb");
  if (!f || fread(in, 1, in_n, f) != in_n) return 4;
  if (fgetc(f) != EOF) return 4;
  fclose(f);
  if (channels == 1) {
    FrameResizeBoxAverageGray(in, src_side, out, dst_side);
  } else {
    FrameResizeBoxAverageRgb(in, src_side, out, dst_side,
                             (uint16_t)wb_r, (uint16_t)wb_b);
  }
  f = fopen(argv[7], "wb");
  if (!f || fwrite(out, 1, out_n, f) != out_n) return 5;
  fclose(f);
  free(in);
  free(out);
  return 0;
}
"""


def _build_harness(tmp: Path) -> Path:
    cc = shutil.which("cc") or shutil.which("gcc") or shutil.which("clang")
    if cc is None:
        raise RuntimeError("no C compiler found (cc/gcc/clang)")
    src = tmp / "frame_resize_host.c"
    exe = tmp / "frame_resize_host"
    src.write_text(HARNESS, encoding="utf-8")
    subprocess.run(
        [cc, "-O2", "-std=c99", "-Wall", "-Wextra",
         "-I", str(HEADER_DIR), "-o", str(exe), str(src)],
        check=True,
    )
    return exe


def _run(exe: Path, tmp: Path, arr: np.ndarray, out_side: int,
         wb_red: int = 100, wb_blue: int = 100) -> np.ndarray:
    arr = np.ascontiguousarray(arr, dtype=np.uint8)
    src_side = int(arr.shape[0])
    channels = 1 if arr.ndim == 2 else int(arr.shape[2])
    in_path = tmp / "in.raw"
    out_path = tmp / "out.raw"
    in_path.write_bytes(arr.tobytes())
    subprocess.run(
        [str(exe), str(in_path), str(src_side), str(out_side),
         str(channels), str(wb_red), str(wb_blue), str(out_path)],
        check=True,
    )
    out = np.frombuffer(out_path.read_bytes(), dtype=np.uint8)
    shape = (out_side, out_side) if channels == 1 else (out_side, out_side, 3)
    return out.reshape(shape)


def _wb_adjust_rgb(arr: np.ndarray, wb_red: int, wb_blue: int) -> np.ndarray:
    """Apply the C helper's per-source-pixel WB exactly (int truncation)."""
    src = arr.astype(np.int32)
    r = np.clip(src[:, :, 0] * wb_red // 100, 0, 255)
    b = np.clip(src[:, :, 2] * wb_blue // 100, 0, 255)
    return np.stack([r, src[:, :, 1], b], axis=-1).astype(np.uint8)


def test_gray_resize_matches_python_bit_for_bit():
    rng = np.random.default_rng(20261024)
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        exe = _build_harness(tmp)
        for src_side, out_side in [(160, 96), (96, 96), (384, 96),
                                   (17, 7), (5, 9), (128, 48)]:
            arr = rng.integers(0, 256, (src_side, src_side), dtype=np.uint8)
            expected = rc._resize_square_box_average_u8(arr, out_side)
            got = _run(exe, tmp, arr, out_side)
            assert np.array_equal(got, expected), (src_side, out_side)


def test_rgb_resize_and_wb_matches_python_bit_for_bit():
    rng = np.random.default_rng(20261024)
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        exe = _build_harness(tmp)
        for src_side, out_side in [(160, 96), (96, 96), (384, 96), (17, 7)]:
            arr = rng.integers(0, 256, (src_side, src_side, 3), dtype=np.uint8)
            for wb_red, wb_blue in [(100, 100), (200, 200), (80, 150)]:
                expected = rc._resize_square_box_average_u8(
                    _wb_adjust_rgb(arr, wb_red, wb_blue), out_side
                )
                got = _run(exe, tmp, arr, out_side, wb_red, wb_blue)
                assert np.array_equal(got, expected), (
                    src_side, out_side, wb_red, wb_blue
                )


def test_live_serial_preview_uses_the_same_resize():
    """A 160x160 device frame shown/previewed at 96 must be the box average."""
    rng = np.random.default_rng(7)
    raw = rng.integers(0, 256, (160, 160), dtype=np.uint8)
    png = rc._raw96_to_png(raw.tobytes(), None, frame_side=160, out_side=96)
    shown = np.asarray(Image.open(io.BytesIO(png)).convert("L"))
    expected = rc._resize_square_box_average_u8(raw, 96)
    assert np.array_equal(shown, expected)


def test_glyph_project_preprocess_side_is_96():
    assert rc._preprocess_frame_side("sign_ocr", 28) == 96
    assert rc._preprocess_frame_side("ocr", 28) == 96
    assert rc._preprocess_frame_side("image", 96) == 96
    assert rc._preprocess_frame_side("image", 64) == 64


if __name__ == "__main__":
    raise SystemExit(__import__("pytest").main([__file__, "-q"]))
