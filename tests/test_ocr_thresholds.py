"""Automated checks for the OCR editor's Dark Thr / Lum Thr sliders.

The two sliders are the one place a user can change what counts as ink in an
OCR project without retraining, and the bug this suite exists for is that they
used to change nothing: the frontend sent them, the backend read neither, and
the preview re-binarised with exact Otsu wherever the sliders sat.  The fix has
a deliberately conservative shape, and both halves of it are asserted here:

  defaults       0 / 100 means "no slider has been touched" and must stay
                 BIT-IDENTICAL to the plain Otsu pipeline — every model already
                 in the field was trained through that path, and a default
                 that re-cut the training cache would be a silent regression
  adjusted       either slider moving off its default pins the luminance
                 window the image branch defines:
                     ink = (gray > bg_dark_thresh) & (gray < bg_lum_thresh)
                 with both bounds strict, nudged one LSB into binarize_ink's
                 inclusive comparison

Run:
    .venv/bin/python tests/test_ocr_thresholds.py
"""
from __future__ import annotations

import io
import sys
import traceback
from pathlib import Path
from typing import Any, Dict

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import ocr_preprocess
import record_controller


def _png(gray: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(gray, mode="L").save(buf, format="PNG")
    return buf.getvalue()


def _decode(png: bytes) -> np.ndarray:
    return np.asarray(Image.open(io.BytesIO(png)).convert("L"))


def _run(gray: np.ndarray, config: Dict[str, Any], **kw):
    """Call the real backend method, unbound: it touches no instance state."""
    return record_controller.RecordController._preprocess_image_png_ocr(
        None, _png(gray), config, **kw
    )


def _ink_pixels(res: Dict[str, Any]) -> int:
    """Black pixels in the processed preview.

    With `return_crop=True`, `processed_png` is the decision mask cropped
    to the detected glyph / manual ROI and rendered BLACK INK ON PURE
    WHITE — the same contract the Preview pane consumes.  Ink is therefore
    0, not 255; counting `> 0` (the old raw-mask convention) measured the
    white background instead.
    """
    return int((_decode(res["processed_png"]) == 0).sum())


def _frame(ink: int = 40, paper: int = 230, mid: int | None = None) -> np.ndarray:
    """Paper with one dark rectangle, and optionally a mid-tone one.

    Two separated blocks rather than a hollow glyph: these tests are about the
    luminance window, and the projection segmentation is exercised by
    test_sign_ocr.py.  The Otsu split of the resulting histogram is measured
    below, not assumed.
    """
    img = np.full((96, 96), paper, np.uint8)
    img[30:70, 10:25] = ink
    if mid is not None:
        img[30:70, 60:75] = mid
    return img


DARK_AT = (50, 15)   # (row, col) inside the dark block
MID_AT = (50, 67)    # inside the mid-tone block


def test_defaults_are_bit_identical_to_the_plain_otsu_pipeline():
    gray = _frame()
    got = _decode(_run(gray, {}))
    want = np.asarray(
        ocr_preprocess.preprocess_glyph(
            gray,
            out_size=ocr_preprocess.GLYPH_SIZE,
            polarity=ocr_preprocess.INK_DARK,
            apply_segmentation=True,
            convention=ocr_preprocess.NORM_CANON,
            thresh=None,
            dark_thr=0,
        )["glyph"],
        dtype=np.uint8,
    )
    assert np.array_equal(got, want), "defaults must not change a single pixel"


def test_lum_slider_pins_the_window_and_drops_the_brighter_ink():
    # `mid` is an ink tone that DEFAULT Otsu includes: for a 40/120/230
    # histogram the Otsu threshold lands at 121.  Pinning Lum Thr to 120
    # (window upper bound 119) must therefore exclude it while leaving the 40
    # block inked — proving the slider reaches the binariser, which is exactly
    # what used to be missing.
    gray = _frame(ink=40, mid=120)
    assert ocr_preprocess.otsu_threshold_u8(gray) == 121, "fixture drifted"

    default = _run(gray, {}, return_crop=True)
    pinned = _run(gray, {"bg_lum_thresh": 120}, return_crop=True)

    # Both 24x40 blocks are inked by default (1,200 px); the pinned window
    # keeps only the 40 block (600 px).  The processed view is cropped to
    # the detected ink, so compare ink AREA, not original-frame indices.
    assert _ink_pixels(default) == 1200, _ink_pixels(default)
    assert _ink_pixels(pinned) == 600, "the 120 block is above the 119 ceiling"
    assert not np.array_equal(_decode(pinned["png"]), _decode(default["png"]))


def test_dark_slider_drops_ink_at_or_below_it():
    gray = _frame(ink=40)
    default = _run(gray, {}, return_crop=True)
    pinned = _run(gray, {"bg_dark_thresh": 60}, return_crop=True)

    default_glyph = _decode(default["png"])
    pinned_glyph = _decode(pinned["png"])
    assert default_glyph.any(), "a 40-on-230 frame has an obvious glyph"
    # Nothing is left inside the window, so the transform reports an empty
    # glyph rather than inventing a box.
    assert not pinned_glyph.any(), "the 40 block is below the 61 floor"


def test_dark_slider_keeps_ink_above_the_floor():
    # A floor of 30 leaves the 40 block inked.  The same slider move also pins
    # the ceiling (default lum 100 -> 99), which the 40 still clears — so this
    # separates "floor is applied" from "everything vanished".
    gray = _frame(ink=40)
    got = _run(gray, {"bg_dark_thresh": 30}, return_crop=True)
    assert _ink_pixels(got) == 600, _ink_pixels(got)
    assert _decode(got["png"]).any()


def test_both_sliders_together_form_one_strict_window():
    gray = _frame(ink=40, mid=120)
    # 31 <= gray <= 119: the 40 block in, the 120 block out.
    got = _run(gray, {"bg_dark_thresh": 30, "bg_lum_thresh": 120}, return_crop=True)
    assert _ink_pixels(got) == 600, _ink_pixels(got)


def test_polarity_selects_which_side_of_the_split_is_ink():
    gray = _frame(ink=40)
    dark = _run(gray, {"ink_polarity": ocr_preprocess.INK_DARK}, return_crop=True)
    bright = _run(gray, {"ink_polarity": ocr_preprocess.INK_BRIGHT}, return_crop=True)
    # INK_DARK inks the 600 block pixels (its preview is cropped to that
    # block); INK_BRIGHT inks the 8,616 paper pixels (its box is the whole
    # frame).  The split is the same Otsu threshold either way.
    assert _ink_pixels(dark) == 600, _ink_pixels(dark)
    assert _ink_pixels(bright) == 96 * 96 - 600, _ink_pixels(bright)
    assert not np.array_equal(_decode(dark["png"]), _decode(bright["png"]))


def test_sample_config_wins_over_the_class_config():
    # The per-sample override is the documented precedence and the preview
    # passes the sample record; if this flipped, a sample-level slider move
    # would appear to do nothing.
    gray = _frame(ink=40, mid=120)
    got = _run(gray, {"bg_lum_thresh": 255}, sample_config={"bg_lum_thresh": 120},
               return_crop=True)
    # 119 ceiling: only the 40 block is inked.  The class-level 255 would
    # have kept both blocks (1,200 px).
    assert _ink_pixels(got) == 600, _ink_pixels(got)


def main() -> int:
    tests = [(name, fn) for name, fn in sorted(globals().items())
             if name.startswith("test_") and callable(fn)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
        except Exception:  # noqa: BLE001 - a test runner reports everything
            failed += 1
            print(f"FAIL {name}")
            traceback.print_exc()
        else:
            print(f"ok   {name}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
