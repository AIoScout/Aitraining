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
    assert [s.get("label") for s in slots] == ["A", "1", "2"], slots
    assert res.get("match") is True, res.get("match_text")
    for slot in slots:
        box = slot.get("box")
        assert isinstance(box, list) and len(box) == 4, box
        # Full-frame coordinates, and character-sized boxes — not a 1 px
        # holder edge and not a 28x28 pre-resized frame.
        assert 0 <= box[0] < box[2] <= 160 and 0 <= box[1] < box[3] <= 160, box
        assert (box[2] - box[0]) > 10 and (box[3] - box[1]) > 10, box
        assert slot.get("prob", 0.0) > 0.5, slot
    assert (roi[2] - roi[0]) > 28 and (roi[3] - roi[1]) > 28, roi


def main() -> int:
    tests = [
        ("ROI ignores desk and frame", test_find_sign_roi_ignores_desk_and_frame),
        ("manual ROI is honoured", test_manual_roi_is_honoured),
        ("border cleanup removes line/speck", test_clean_sign_ink_removes_frame_line_and_speck),
        ("preview crops board before 3 slots", test_preview_crops_board_before_three_slots),
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
