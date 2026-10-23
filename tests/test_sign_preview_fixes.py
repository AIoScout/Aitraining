"""Regression tests for the three reported sign-preview bugs.

  1. Preview ROI semantics: ROI OFF = full frame + box; ROI ON = the crop,
     no box.  The JavaScript half lives in tests/test_preview_roi_spa.py
     (Playwright); the backend payloads it consumes are exercised here and
     in test_sign_roi_preview.py.
  2. Auto ROI on a real "A 1 2" board: an interior ink cluster with a
     hairline '1' and border clutter must still produce a tight, symmetric
     box — never the full plate, never pulled left by the border noise.
  3. A camera-captured '1' (1-2 px stroke) must be normalised to the
     3-4 px stroke width of its EMNIST counterparts so it is not read as
     `_background`.

Requires the AItraining venv (TensorFlow for the end-to-end case):
    .venv/bin/python -m pytest tests/test_sign_preview_fixes.py -q
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, Path(__file__).resolve().parent):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import ocr_preprocess as op
import record_controller as rc


def _thin_one_frame() -> np.ndarray:
    """160x160 "A 1 2" board: thin middle stroke + border clutter.

    The template plate supplies A and 2; the '1' is replaced with a 3-px
    vertical stroke (the camera-segmented shape from the report).  Dark
    border noise sits in the outer 6 % margin on the left and top, which
    is exactly what used to drag the ROI left.
    """
    sign = np.asarray(
        Image.open(ROOT / "templates" / "sign_A12_match.png").convert("L")
    ).copy()
    sign[30:64, 40:64] = 245       # erase the template's wide '1'
    sign[33:62, 53:56] = 20        # camera-style hairline stroke
    frame = np.full((160, 160), 205, np.uint8)
    plate = np.full((112, 112), 245, np.uint8)
    plate[8:104, 8:104] = sign
    frame[24:136, 24:136] = plate
    # dark plastic frame around the plate
    frame[20:28, 20:140] = 30
    frame[132:140, 20:140] = 30
    frame[20:140, 20:28] = 30
    frame[20:140, 132:140] = 30
    # border clutter inside the 6 % margin (must be discarded)
    frame[145:155, 2:30] = 15
    frame[150:156, 30:45] = 40
    frame[0:6, 60:90] = 25
    return frame


def _png(arr: np.ndarray) -> bytes:
    import io

    buf = io.BytesIO()
    Image.fromarray(np.asarray(arr, dtype=np.uint8), mode="L").save(
        buf, format="PNG"
    )
    return buf.getvalue()


def test_ink_cluster_roi_is_tight_symmetric_and_ignores_border_noise():
    frame = _thin_one_frame()
    roi = op.find_sign_roi(frame, dark_thr=0, lum_thr=59)
    assert roi["method"] == "ink_cluster", roi
    x1, y1, x2, y2 = (int(v) for v in roi["box"])

    # The true character components: A (35..66), '1' (85..88), 2 (100..124),
    # all on rows 65..94.
    left, top, right, bottom = 35, 65, 124, 94
    assert x1 < left and y1 < top and x2 > right and y2 > bottom, roi
    pads = (left - x1, top - y1, x2 - right, y2 - bottom)
    assert max(pads) - min(pads) <= 1, pads
    assert min(pads) >= 3, pads
    assert y2 < 145, ("border clutter leaked into the ROI", roi)
    assert x1 > 10, ("ROI pulled left by the top-left noise", roi)


def _ink_widths(glyph: np.ndarray) -> list[int]:
    """Stroke width per row on the CANON canvas.

    `glyph` is dark ink on a black (zero) pad, so raw `glyph < 80` includes
    the pad.  The plate inside the fitted block stays bright (230), which
    lets us count only the stroke: 0 < v < 100.
    """
    ink = (glyph > 0) & (glyph < 100)
    return [int(row.sum()) for row in ink if row.any()]


def test_thin_one_slot_is_widened_to_emnist_stroke_width():
    # The sign-slot path: the split boxes the stroke, then preprocess_glyph
    # runs with segmentation OFF on a padded crop.  A 1-px stroke used to
    # vanish under nearest-neighbour downsampling; it must now be 3-4 px.
    crop = np.full((35, 20), 230, np.uint8)
    crop[3:32, 9:10] = 20
    res = op.preprocess_glyph(
        crop, out_size=28, polarity=op.INK_DARK,
        apply_segmentation=False, convention=op.NORM_CANON, thresh=100,
    )
    widths = _ink_widths(np.asarray(res["glyph"], dtype=np.uint8))
    assert widths, "the stroke vanished"
    assert min(widths) >= 3, widths
    assert max(widths) <= 4, widths

    # The training path: a tight capture with a 2-px stroke, segmentation
    # ON.  It must receive the same widening.
    tight = np.full((29, 2), 230, np.uint8)
    tight[1:28, 0:2] = 20
    res = op.preprocess_glyph(
        tight, out_size=28, polarity=op.INK_DARK,
        apply_segmentation=True, convention=op.NORM_CANON, thresh=100,
    )
    widths = _ink_widths(np.asarray(res["glyph"], dtype=np.uint8))
    assert widths, "the segmented stroke vanished"
    assert min(widths) >= 3, widths
    assert max(widths) <= 4, widths

    # A wide glyph must not be touched: its ink bbox aspect is >= 0.35, so
    # the narrow-stroke radius is zero and the output keeps its own shape.
    wide_ink = np.zeros((40, 30), dtype=bool)
    wide_ink[5:35, 5:25] = True
    assert op._narrow_stroke_radius(wide_ink, 30, 40, 28, op.NORM_CANON) == 0
    narrow_ink = np.zeros((40, 30), dtype=bool)
    narrow_ink[5:35, 15:17] = True
    assert op._narrow_stroke_radius(narrow_ink, 30, 40, 28, op.NORM_CANON) >= 1


def test_webcam_frames_are_center_cropped_to_a_square_for_glyph_projects():
    frame = np.zeros((90, 160, 3), np.uint8)
    frame[:, :, 0] = np.arange(160, dtype=np.uint8)[None, :]
    glyph = rc._center_crop_webcam_square(frame, "sign_ocr")
    assert glyph.shape == (90, 90, 3), glyph.shape
    assert int(glyph[0, 0, 0]) == 35, "crop must take the horizontal centre"
    assert int(glyph[0, -1, 0]) == 124
    # Image projects keep their native aspect (no device square to match).
    assert rc._center_crop_webcam_square(frame, "image").shape == (90, 160, 3)
    # Already-square frames are untouched.
    square = np.zeros((90, 90, 3), np.uint8)
    assert rc._center_crop_webcam_square(square, "ocr").shape == (90, 90, 3)


def test_thin_one_sign_reads_a12_not_background():
    """End-to-end: train the template project WITH the reject class, then
    read the realistic board.  Slot 2 must say '1', not `_background`."""
    from test_emnist_seeder import _open_template, _workspace

    with tempfile.TemporaryDirectory() as td:
        ctrl, _dataset = _workspace(Path(td))
        state = _open_template(ctrl)
        cfg = dict(state["train_cfg"])
        cfg["ocr_emnist_enabled"] = True
        cfg["ocr_emnist_per_class"] = 200
        cfg["ocr_background_enabled"] = True
        cfg["sign_background_replace_ood"] = True
        cfg["ocr_background_samples"] = 120
        ctrl._train_worker("s", cfg)
        assert not (ctrl._train.get("s") or {}).get("error"), ctrl._train.get("s")

        res = ctrl._preview_predict_sign_ocr(
            "s",
            _png(_thin_one_frame()),
            sign_sequence="A12",
            reading_order="ltr",
            bg_dark_thresh=0,
            bg_lum_thresh=60,
        )

    assert res.get("roi_method") == "ink_cluster", res.get("roi_method")
    slots = res.get("slots") or []
    assert [s.get("char") for s in slots] == ["A", "1", "2"], slots
    assert [s.get("label") for s in slots] == ["A", "1", "2"], slots
    assert not any(s.get("background") for s in slots), slots
    assert all(float(s.get("prob") or 0.0) >= 0.60 for s in slots), slots
    assert res.get("match") is True, res
    assert str(res.get("match_text") or "").startswith(rc.SIGN_MATCH_TEXT), res.get("match_text")

    # ROI OFF loads full_image_b64 (frame-sized); ROI ON loads
    # processed_image_b64 / crop_image_b64 (ROI-sized).  The JS test
    # (tests/test_preview_roi_spa.py) asserts the toggle/overlay half.
    import base64
    import io

    def _shape(key: str) -> tuple[int, ...]:
        return np.asarray(
            Image.open(io.BytesIO(base64.b64decode(res[key])))
        ).shape

    roi_x1, roi_y1, roi_x2, roi_y2 = (int(v) for v in res["roi_box"])
    assert _shape("full_image_b64") == (160, 160), _shape("full_image_b64")
    assert _shape("processed_image_b64") == (roi_y2 - roi_y1, roi_x2 - roi_x1)
    assert _shape("crop_image_b64") == (roi_y2 - roi_y1, roi_x2 - roi_x1)
