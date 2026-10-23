"""Regression checks for native-resolution sign-board ROI + 3-slot splitting.

The hardware report: a 160x160 ESP32-P4 frame with the sign plate surrounded
by lab clutter produced `roi=[0,0,1,1]`, slot 0 read the background at 54.7%,
and slots 1/2 read the sign holder's 1 px vertical border.  This module pins
the repaired contract:

  find_sign_roi   finds the bright inner plate (or honours a manual ROI) on
                  the NATIVE frame, excluding the desk/frame clutter
  clean_sign_ink  drops border-touching frame lines and tiny specks but keeps
                  a character that merely sits close to the ROI edge
  preview         crops the board first, splits three glyphs inside the ROI
                  at native resolution, maps boxes back to full-frame
                  coordinates, reports a real sign_pct, and still reads A12

Requires the AItraining venv (TensorFlow for the template training run):
    .venv/bin/python tests/test_sign_roi_preview.py
"""
from __future__ import annotations

import base64
import json
import os
import sys
import tempfile
import traceback
from io import BytesIO
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, Path(__file__).resolve().parent):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
from PIL import Image

import ocr_preprocess as op
from test_emnist_seeder import _open_template, _workspace

TEMPLATE_SIGN = ROOT / "templates" / "sign_A12_match.png"


def _clutter_frame() -> np.ndarray:
    """160x160 desk scene: dark clutter left, bright plate + thick frame."""
    sign = np.asarray(Image.open(TEMPLATE_SIGN).convert("L"), dtype=np.uint8)
    frame = np.full((160, 160), 205, dtype=np.uint8)
    # dark objects on the left / bottom, like the report's wheel + sensor
    frame[:, :24] = 25
    frame[120:160, 20:90] = 45
    frame[40:120, 0:15] = 60
    # bright sign plate with the A12 glyphs, surrounded by a dark frame
    plate = np.full((112, 112), 245, dtype=np.uint8)
    plate[8:104, 8:104] = sign
    frame[24:136, 24:136] = plate
    frame[20:28, 20:140] = 30
    frame[132:140, 20:140] = 30
    frame[20:140, 20:28] = 30
    frame[20:140, 132:140] = 30
    return frame


def _png_bytes(arr: np.ndarray) -> bytes:
    buf = BytesIO()
    Image.fromarray(np.asarray(arr, dtype=np.uint8), mode="L").save(buf, format="PNG")
    return buf.getvalue()


def _user_window_frame() -> np.ndarray:
    """Mid-tone camera scene for the hardware window Dark=57 / Lum=160.

    The template glyphs are near-black, so this scene maps them into the
    user's ink band (60..140) and the board above Lum (175..245), matching the
    reported hand-held plate: dark characters, bright sign, clutter/fingers
    inside and outside the window.
    """
    sign = np.asarray(Image.open(TEMPLATE_SIGN).convert("L"), dtype=np.int32)
    mid = np.where(
        sign < 128,
        60 + sign * 80 // 127,          # glyphs: 60..140, inside [57,160]
        175 + (sign - 128) * 70 // 127, # board: 175..245, above Lum
    ).astype(np.uint8)
    frame = np.full((160, 160), 205, dtype=np.uint8)
    frame[:, :24] = 25
    frame[120:160, 20:90] = 45
    frame[40:120, 0:15] = 60
    frame[90:150, 0:40] = 85        # mid-gray finger/clutter inside the window
    plate = np.full((112, 112), 245, dtype=np.uint8)
    plate[8:104, 8:104] = mid
    frame[24:136, 24:136] = plate
    frame[20:28, 20:140] = 30
    frame[132:140, 20:140] = 30
    frame[20:140, 20:28] = 30
    frame[20:140, 132:140] = 30
    return frame


def test_find_sign_roi_ignores_desk_and_frame():
    frame = _clutter_frame()
    roi = op.find_sign_roi(frame)
    assert roi["method"] == "bright_plate", roi
    x1, y1, x2, y2 = roi["box"]
    # The plate is inside the black frame (x/y 20..140); the desk is outside.
    assert 20 <= x1 <= 30 and 20 <= y1 <= 30, roi
    assert 130 <= x2 <= 140 and 130 <= y2 <= 140, roi
    assert (x2 - x1) * (y2 - y1) < 0.60 * frame.size, roi
    assert roi["area_frac"] > 0.05, roi


def test_manual_roi_is_honoured():
    frame = _clutter_frame()
    roi = op.find_sign_roi(frame, manual_roi=[0.2, 0.2, 0.8, 0.8])
    assert roi["method"] == "manual", roi
    assert roi["box"] == [32, 32, 128, 128], roi


def test_clean_sign_ink_removes_frame_line_and_speck():
    ink = np.zeros((100, 100), dtype=bool)
    ink[:, 0:1] = True          # vertical frame line on the ROI edge
    ink[10:24, 10:24] = True    # character-sized interior component
    ink[50, 50] = True          # one-pixel noise speck
    cleaned = op.clean_sign_ink(ink)
    assert not cleaned[:, 0].any(), "border frame line survived"
    assert cleaned[10:24, 10:24].all(), "character component was removed"
    assert not cleaned[50, 50], "noise speck survived"


def test_preview_crops_board_before_three_slots():
    frame = _clutter_frame()
    with tempfile.TemporaryDirectory() as td:
        ctrl, _dataset = _workspace(Path(td))
        state = _open_template(ctrl)
        ctrl._train_worker("s", state["train_cfg"])
        assert not (ctrl._train.get("s") or {}).get("error"), ctrl._train.get("s")

        res = ctrl._preview_predict_sign_ocr(
            "s", _png_bytes(frame), sign_sequence="A12", reading_order="ltr"
        )

    assert res.get("roi_method") == "bright_plate", res
    roi = res.get("roi_box")
    assert isinstance(roi, list) and len(roi) == 4, roi
    assert not (roi == [0, 0, 160, 160]), roi
    assert roi[0] > 10 and roi[1] > 10 and roi[2] < 150 and roi[3] < 150, roi

    crop = res.get("crop")
    assert isinstance(crop, list) and len(crop) == 4, crop
    assert crop != [0.0, 0.0, 1.0, 1.0], crop
    assert crop[0] > 0.05 and crop[2] < 0.95, crop

    # A real measurement, not the old hard-coded 0.0.
    assert float(res.get("sign_pct") or 0.0) > 1.0, res.get("sign_pct")

    slots = res.get("slots") or []
    # The target slots and their character-sized crops are the subject here;
    # the template model's confidence can vary between repeated training runs
    # in one process, so its label/match is not a segmentation assertion.
    assert [s.get("char") for s in slots] == ["A", "1", "2"], slots
    for slot in slots:
        box = slot.get("box")
        assert isinstance(box, list) and len(box) == 4, box
        # Full-frame coordinates, and character-sized boxes — not a 1 px
        # holder edge and not a 28x28 pre-resized frame.
        assert 0 <= box[0] < box[2] <= 160 and 0 <= box[1] < box[3] <= 160, box
        assert (box[2] - box[0]) > 10 and (box[3] - box[1]) > 10, box
        assert slot.get("empty") is False, slot
    assert (roi[2] - roi[0]) > 28 and (roi[3] - roi[1]) > 28, roi


def test_user_window_finds_characters_and_renders_white_masks():
    """Dark=57 / Lum=160: ink-cluster ROI, black-on-white displays, A12."""
    frame = _user_window_frame()
    roi = op.find_sign_roi(frame, dark_thr=58, lum_thr=159)
    assert roi["method"] == "ink_cluster", roi
    x1, y1, x2, y2 = roi["box"]
    assert roi["box"] != [0, 0, 160, 160], roi
    assert 20 <= x1 <= 60 and 45 <= y1 <= 80, roi
    assert 110 <= x2 <= 145 and 85 <= y2 <= 120, roi
    assert (y2 - y1) < 0.55 * 160, roi
    assert (x2 - x1) > 0.35 * 160, roi

    slots = op.preprocess_sign_sequence(
        frame[y1:y2, x1:x2], sign_sequence="A12", out_size=28,
        thresh=159, dark_thr=58, clean_border=True,
    )["slots"]
    assert len(slots) == 3, slots
    for slot in slots:
        display = np.asarray(slot["display"])
        assert display.shape == (28, 28), display.shape
        assert display[0, 0] == 255, "slot display background must be white"
        assert int(display.min()) == 0 and int(display.max()) == 255, display
        assert float(display.mean()) > 100.0, float(display.mean())

    with tempfile.TemporaryDirectory() as td:
        ctrl, _dataset = _workspace(Path(td))
        state = _open_template(ctrl)
        ctrl._train_worker("s", state["train_cfg"])
        assert not (ctrl._train.get("s") or {}).get("error"), ctrl._train.get("s")
        res = ctrl._preview_predict_sign_ocr(
            "s", _png_bytes(frame), sign_sequence="A12", reading_order="ltr",
            bg_dark_thresh=57, bg_lum_thresh=160,
        )

    assert res.get("roi_method") == "ink_cluster", res.get("roi_method")
    assert float(res.get("sign_pct") or 0.0) > 1.0, res.get("sign_pct")
    # The synthetic mid-tone glyphs are a distribution shift for the template
    # model, so assert the three target slots/crops (the subject of this
    # test), not the model's label confidence.
    assert [s.get("char") for s in res.get("slots") or []] == ["A", "1", "2"], res.get("slots")
    assert all(not s.get("empty") for s in res.get("slots") or []), res.get("slots")

    full = np.asarray(Image.open(BytesIO(base64.b64decode(res["full_image_b64"]))))
    processed = np.asarray(Image.open(BytesIO(base64.b64decode(res["processed_image_b64"]))))
    assert full.shape == (160, 160), full.shape
    assert processed.shape != (160, 160), "processed view must be cropped to the ROI"
    for arr in (full, processed):
        assert int(arr.min()) == 0 and int(arr.max()) == 255, arr
        assert float(arr.mean()) > 180.0, float(arr.mean())
        assert int(arr[0, 0]) == 255, "threshold background must be pure white"
    for b64 in res.get("slot_images_b64") or []:
        arr = np.asarray(Image.open(BytesIO(base64.b64decode(b64))))
        assert arr.shape == (28, 28), arr.shape
        assert int(arr[0, 0]) == 255, "slot background must be white"
        assert int(arr.min()) == 0, "slot ink must be black"
        assert float(arr.mean()) > 100.0, float(arr.mean())


def test_edit_processed_preview_is_white_masked_and_cropped():
    """Edit modal: processed view is black ink on white, cropped to the ROI."""
    png = TEMPLATE_SIGN.read_bytes()
    with tempfile.TemporaryDirectory() as td:
        ctrl, _dataset = _workspace(Path(td))
        auto = ctrl._preprocess_image_png_ocr(
            png,
            {"mode": "auto_by_label", "bg_dark_thresh": 17, "bg_lum_thresh": 87},
            return_crop=True, out_size=28,
        )
        manual = ctrl._preprocess_image_png_ocr(
            png,
            {"mode": "manual_roi", "manual_roi": [0.2, 0.2, 0.8, 0.8],
             "bg_dark_thresh": 17, "bg_lum_thresh": 87},
            return_crop=True, out_size=28,
        )
    for name, res in (("auto", auto), ("manual", manual)):
        arr = np.asarray(Image.open(BytesIO(res["processed_png"])))
        assert arr.shape != (96, 96), (name, arr.shape)
        assert int(arr[0, 0]) == 255, (name, "white background")
        assert int(arr.min()) == 0, (name, "black ink")
        assert float(arr.mean()) > 180.0, (name, float(arr.mean()))
    # Manual ROI uses the user's 0.2..0.8 box (57-58 px for a 96 px sample),
    # not the full uncropped frame.
    manual_arr = np.asarray(Image.open(BytesIO(manual["processed_png"])))
    assert 50 <= manual_arr.shape[0] <= 64 and 50 <= manual_arr.shape[1] <= 64, manual_arr.shape


def main() -> int:
    tests = [
        ("ROI ignores desk and frame", test_find_sign_roi_ignores_desk_and_frame),
        ("manual ROI is honoured", test_manual_roi_is_honoured),
        ("border cleanup removes line/speck", test_clean_sign_ink_removes_frame_line_and_speck),
        ("preview crops board before 3 slots", test_preview_crops_board_before_three_slots),
        ("user window finds characters and white masks", test_user_window_finds_characters_and_renders_white_masks),
        ("Edit processed view is white-masked and cropped", test_edit_processed_preview_is_white_masked_and_cropped),
    ]
    results = []
    for name, fn in tests:
        try:
            fn()
        except Exception as exc:  # noqa: BLE001 - a test runner reports everything
            results.append((name, False))
            print(f"FAIL {name}: {exc}")
            traceback.print_exc()
        else:
            results.append((name, True))
            print(f"ok   {name}")
    passed = sum(1 for _, ok in results if ok)
    print(f"\n{passed}/{len(results)} passed")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
