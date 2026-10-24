"""Regression: ROI crop must not erase characters near the box edge.

The hardware report (Dark=0, Lum=121): with ROI unchecked the green box
surrounded the white plate and A 1 2 were visible; with ROI checked the 'A'
and the bottom of the '1' vanished, the slot thumbnails carried a vertical
border line (`|A`) from the dark surround, and a correct 3-slot reading was
rejected as NO solely by `entropy_ratio=0.728 > 0.70`.

These tests pin the repair:

  ROI          the ink-cluster box is clamped to the white plate interior,
               so the dark surround / holder border cannot enter the crop
  split        operating on that exact crop keeps 'A', both strokes of a
               broken '1' and '2'; no ink touches the crop edge (no '|')
  preview      ROI-checked pixels == the exact full-view slice of roi_box
  entropy      a slot whose top class is the target and prob >= floor is
               accepted even when entropy_ratio is above the gate
"""
from __future__ import annotations

import base64
import os
import sys
import tempfile
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, Path(__file__).resolve().parent):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import ocr_preprocess as op  # noqa: E402
import record_controller as rc  # noqa: E402
from test_emnist_seeder import SIGN_A12, _workspace  # noqa: E402


def _edge_plate_frame() -> np.ndarray:
    """160x160 dark-surrounded white plate, 'A' near the plate's left edge.

    Plate x=10..149, y=6..153.  'A' starts at x=12, so the ink-cluster pad
    used to cross the plate edge and pull the dark surround into the ROI.
    The '1' is broken into a top stroke and a bottom stanza; both must reach
    slot 2.
    """
    f = np.full((160, 160), 20, np.uint8)
    f[6:154, 10:150] = 245
    # A: x=12..36, y=30..110
    f[30:110, 12:18] = 20
    f[30:110, 30:36] = 20
    f[30:36, 12:36] = 20
    f[67:73, 12:36] = 20
    # '1': top stroke x=80..86, bottom stanza x=78..88
    f[30:70, 80:86] = 20
    f[80:110, 78:88] = 20
    # '2': x=120..146, y=30..110
    f[30:36, 120:146] = 20
    f[30:70, 140:146] = 20
    f[67:73, 120:146] = 20
    f[67:104, 120:126] = 20
    f[104:110, 120:146] = 20
    return f


def _png(frame: np.ndarray) -> bytes:
    return rc._to_png_bytes(Image.fromarray(frame, mode="L"))


def test_roi_is_inside_plate_and_keeps_edge_glyphs():
    frame = _edge_plate_frame()
    roi = op.find_sign_roi(frame, dark_thr=0, lum_thr=120)
    x1, y1, x2, y2 = roi["box"]

    # The green box is inside the white plate (10..149, 6..153), not out in
    # the dark surround.
    assert 10 <= x1 and x2 <= 150 and 6 <= y1 and y2 <= 154, roi
    # ... and it still contains every glyph, including the edge 'A'.
    assert x1 <= 12 and x2 >= 146, roi
    assert y1 <= 30 and y2 >= 110, roi

    crop = frame[y1:y2, x1:x2]
    # Every crop edge is white plate, not the dark surround.
    for edge in (crop[0, :], crop[-1, :], crop[:, 0], crop[:, -1]):
        assert float(np.mean(edge)) > 180.0, float(np.mean(edge))
    reading = op.preprocess_sign_sequence(
        crop, sign_sequence="A12", out_size=28, thresh=120,
        clean_border=False,
    )
    slots = reading["slots"]
    assert [s["char"] for s in slots] == ["A", "1", "2"], slots
    assert all(s["ok"] and not s["empty"] for s in slots), slots

    # No ink touches the crop edge: the old ROI included the dark holder
    # border, which the split merged into slot 1 as `|A`.
    ink = np.asarray(reading["ink"], dtype=bool)
    assert ink.any()
    assert not ink[0, :].any(), "top ROI border bled into the split"
    assert not ink[:, 0].any(), "left ROI border bled into the split"
    assert not ink[:, -1].any(), "right ROI border bled into the split"
    assert not ink[-1, :].any(), "bottom ROI border bled into the split"

    # The broken '1' keeps BOTH fragments (the gap is inside its slot box).
    one_glyph = np.asarray(slots[1]["glyph"])
    one_ink = one_glyph[one_glyph > 0]
    assert one_ink.size > 80, one_ink.size
    ink_box = slots[1]["ink_box"]
    assert ink_box is not None
    ix1, iy1, ix2, iy2 = (int(v) for v in ink_box)
    mid = (iy1 + iy2) // 2
    assert ink[iy1:mid, ix1:ix2].any(), "top stroke of '1' missing"
    assert ink[mid:iy2, ix1:ix2].any(), "bottom stroke of '1' missing"


def _fake_model() -> dict:
    return {
        "labels": ["A", "1", "2"],
        "input_details": {"shape": [1, 28, 28, 1], "dtype": np.int8},
        "sign_sequence": "A12",
        "sign_sequences": ["A12"],
        "sign_background_replace_ood": False,
        "class_preprocess": {},
        "preprocess_mode": "auto_by_label",
        "model_source": "latest",
    }


def _high_entropy_raw(top: int) -> np.ndarray:
    """Top class ~67.7 %, entropy_ratio ~0.728 (> the 0.70 gate)."""
    raw = np.asarray([63, 18], dtype=np.int64)
    out = np.empty(3, dtype=np.int64)
    others = [i for i in range(3) if i != top]
    out[top] = 170
    out[others[0]] = int(raw[0])
    out[others[1]] = int(raw[1])
    return out


def test_preview_roi_slice_is_exact_and_correct_slots_bypass_entropy():
    frame = _edge_plate_frame()
    with tempfile.TemporaryDirectory() as td:
        ctrl, _dataset = _workspace(Path(td))
        model = _fake_model()
        calls = {"n": 0}

        def fake_invoke(model, session_id, arr):
            idx = calls["n"]
            calls["n"] += 1
            return None, _high_entropy_raw(idx % 3).copy()

        ctrl._preview_load_model = lambda session_id: model
        ctrl._invoke_glyph_tensor = fake_invoke

        # Sanity: the raw scores really do trip the entropy gate while
        # clearing the probability floor.
        max_prob, entropy_ratio = rc._raw_ood_metrics(_high_entropy_raw(0))
        assert max_prob >= 0.60, max_prob
        assert entropy_ratio > 0.70, entropy_ratio

        res = ctrl._preview_predict_sign_ocr(
            "s", _png(frame),
            sign_sequence="A12",
            reading_order="ltr",
            bg_dark_thresh=0,
            bg_lum_thresh=121,
            match_min_prob=0.60,
            ood_sign_pct_min=0.0,
            ood_sign_pct_max=100.0,
            ood_max_prob_min=0.60,
            ood_entropy_ratio_max=0.70,
        )

    # ROI-checked must be the EXACT unchecked green-box slice.
    full = np.asarray(Image.open(BytesIO(base64.b64decode(res["full_image_b64"]))))
    processed = np.asarray(Image.open(BytesIO(base64.b64decode(res["processed_image_b64"]))))
    x1, y1, x2, y2 = (int(v) for v in res["roi_box"])
    assert processed.shape == (y2 - y1, x2 - x1), (processed.shape, res["roi_box"])
    assert np.array_equal(processed, full[y1:y2, x1:x2]), "ROI view is not the exact slice"

    slots = res["slots"]
    assert [s["char"] for s in slots] == ["A", "1", "2"], slots
    assert [s["label"] for s in slots] == ["A", "1", "2"], slots
    assert all(not s["empty"] for s in slots), slots

    # A correct target hit at >= the floor is NOT rejected on entropy alone.
    assert res["match"] is True, res
    assert res["ood"] is False, res.get("ood_reason")
    assert res["ood_reason"] == "", res.get("ood_reason")
    assert all(not s.get("ood") for s in slots), slots


def test_default_floor_is_040_and_layer1_is_disarmed():
    """Trap 1/2: a 0.45 top-1 target is accepted at the new 0.40 floor, even
    when the caller supplies an impossible sign_pct window and entropy is
    above its gate."""
    frame = _edge_plate_frame()
    with tempfile.TemporaryDirectory() as td:
        ctrl, _dataset = _workspace(Path(td))
        model = _fake_model()
        calls = {"n": 0}

        def fake_invoke(model, session_id, arr):
            idx = calls["n"]
            calls["n"] += 1
            raw = np.asarray([55, 55, 55], dtype=np.int64)
            raw[idx] = 90          # 90/200 = 0.45
            return None, raw

        ctrl._preview_load_model = lambda session_id: model
        ctrl._invoke_glyph_tensor = fake_invoke

        res = ctrl._preview_predict_sign_ocr(
            "s", _png(frame),
            sign_sequence="A12",
            reading_order="ltr",
            bg_dark_thresh=0,
            bg_lum_thresh=121,
            # The frame's sign_pct is ~17 %, deliberately far outside this
            # window: Layer 1 must be disarmed, not reject all three slots.
            ood_sign_pct_min=90.0,
            ood_sign_pct_max=100.0,
            ood_max_prob_min=0.40,
            ood_entropy_ratio_max=0.70,
        )

    assert all(0.44 <= s["prob"] <= 0.46 for s in res["slots"]), res["slots"]
    assert res["match_min_prob"] == 0.40, res.get("match_min_prob")
    assert res["match"] is True, res
    assert res["ood"] is False, res.get("ood_reason")
    assert res["ood_reason"] == "", res.get("ood_reason")
    assert [s["match"] for s in res["slots"]] == [True, True, True], res["slots"]


def test_ood_max_prob_is_the_single_floor_and_banner_value():
    """OOD MAX PROB 0.25 must accept a 0.28 target hit and be the banner's
    `each at 25% or better` value; a legacy match_min_prob=0.40 is ignored."""
    frame = _edge_plate_frame()
    with tempfile.TemporaryDirectory() as td:
        ctrl, _dataset = _workspace(Path(td))
        model = {
            "labels": ["A", "1", "2", "B", "8"],
            "input_details": {"shape": [1, 28, 28, 1], "dtype": np.int8},
            "sign_sequence": "A12",
            "sign_sequences": ["A12"],
            "sign_background_replace_ood": False,
            "class_preprocess": {},
            "preprocess_mode": "auto_by_label",
            "model_source": "latest",
        }
        calls = {"n": 0}

        def fake_invoke(model, session_id, arr):
            idx = calls["n"]
            calls["n"] += 1
            raw = np.asarray([18, 18, 18, 18, 18], dtype=np.int64)
            raw[idx] = 28          # 28/100 = 0.28, above the 0.25 OOD floor
            return None, raw

        ctrl._preview_load_model = lambda session_id: model
        ctrl._invoke_glyph_tensor = fake_invoke

        res = ctrl._preview_predict_sign_ocr(
            "s", _png(frame),
            sign_sequence="A12",
            reading_order="ltr",
            bg_dark_thresh=0,
            bg_lum_thresh=121,
            # The legacy split floor says 0.40; OOD MAX PROB must win.
            match_min_prob=0.40,
            ood_max_prob_min=0.25,
            ood_entropy_ratio_max=0.85,
        )

    assert all(abs(s["prob"] - 0.28) < 1e-6 for s in res["slots"]), res["slots"]
    # This is exactly what app.py renders as "(each at 25% or better)".
    assert res["match_min_prob"] == 0.25, res.get("match_min_prob")
    assert res["match"] is True, res
    assert res["ood"] is False, res.get("ood_reason")
    assert [s["match"] for s in res["slots"]] == [True, True, True], res["slots"]


def test_manual_roi_strips_corner_speck_and_centers_the_one():
    """A Manual ROI of '1' with a clipped foot at the bottom-left corner must
    drop the speck and centre the stroke at x=13-14 on the 28x28 canvas."""
    frame = np.full((96, 96), 230, np.uint8)
    frame[20:70, 30:36] = 20       # vertical '1' stroke
    frame[92:96, 0:10] = 20        # stray foot touching the crop corner
    with tempfile.TemporaryDirectory() as td:
        ctrl, _dataset = _workspace(Path(td))
        res = ctrl._preprocess_image_png_ocr(
            _png(frame),
            {"mode": "manual_roi", "manual_roi": [0.0, 0.0, 0.8, 1.0]},
            out_size=28,
            label_name="1",
            project_type="sign_ocr",
            sign_sequences=["A12"],
            return_crop=True,
        )
    # The Processed Preview shape: black ink on white, 28x28.
    glyph = np.asarray(Image.open(BytesIO(res["display_png"])).convert("L"))
    assert glyph.shape == (28, 28), glyph.shape
    ink = glyph < 128
    cols = np.where(ink.any(axis=0))[0]
    assert cols.size, "the '1' stroke vanished"
    # Centred, 3-4 px wide; the corner speck is gone.
    assert 12 <= int(cols.min()) and int(cols.max()) <= 15, cols
    assert 3 <= int(cols.max() - cols.min() + 1) <= 4, cols
    assert not bool(ink[-4:, :5].any()), "corner speck survived"


def test_sign_auto_extracts_the_class_slot_from_a_whole_sign():
    """Auto mode in a sign_ocr class card must extract THIS class's slot,
    not feed the whole A 1 2 board into a single-glyph class."""
    frame = np.asarray(Image.open(SIGN_A12).convert("L"), dtype=np.uint8)
    with tempfile.TemporaryDirectory() as td:
        ctrl, _dataset = _workspace(Path(td))
        got = ctrl._preprocess_image_png_ocr(
            _png(frame),
            {"mode": "auto_by_label"},
            out_size=28,
            label_name="1",
            project_type="sign_ocr",
            sign_sequences=["A12"],
            reading_order="ltr",
        )
    want = op.preprocess_sign_sequence(
        frame, sign_sequence="A12", reading_order="ltr", out_size=28,
        polarity=op.INK_DARK, apply_segmentation=True,
        convention=op.NORM_CANON,
    )["slots"][1]["glyph"]
    assert got == rc._to_png_bytes(Image.fromarray(np.asarray(want, np.uint8), mode="L"))


def test_edit_processed_preview_is_the_canonical_28x28_glyph():
    """#classPreprocessModal's Processed Preview must be the exact 28x28
    canonical glyph (speck stripped), not a larger pre-centring crop."""
    frame = np.full((96, 96), 230, np.uint8)
    frame[20:70, 30:36] = 20
    frame[92:96, 0:10] = 20
    cfg = {"mode": "manual_roi", "manual_roi": [0.0, 0.0, 0.8, 1.0]}
    with tempfile.TemporaryDirectory() as td:
        ctrl, dataset = _workspace(Path(td))
        class_dir = dataset / "1"
        class_dir.mkdir(parents=True, exist_ok=True)
        sample = class_dir / "sample.png"
        sample.write_bytes(_png(frame))
        preview = ctrl._preprocess_preview(
            "s", "1", sample.name, cfg,
        )
        direct = ctrl._preprocess_image_png_ocr(
            sample.read_bytes(), cfg, out_size=28,
            label_name="1", project_type="sign_ocr",
            sign_sequences=["A12"], return_crop=True,
        )
    shown = base64.b64decode(preview["processed_image_b64"])
    assert shown == direct["display_png"], (
        "Processed Preview is not the canonical 28x28 display glyph"
    )
    arr = np.asarray(Image.open(BytesIO(shown)).convert("L"))
    assert arr.shape == (28, 28), arr.shape


def test_host_preview_uses_the_same_slicing_as_the_device():
    """The Preview must pass the same frame + thresholds to the SAME
    preprocess_sign_sequence(find_roi=True) path as sign_feed_slots(), and
    feed `(int)pixel - 128` int8 tensors to the model."""
    frame160 = _edge_plate_frame()
    small = rc._resize_square_box_average_u8(frame160, 96)
    # Feed the NATIVE 160 frame; the preview must perform the same 160->96
    # box average the device does.
    png = _png(frame160)
    cases = [
        (0, 100, None),
        (0, 121, 120),
        (0, 50, 49),
    ]
    for bg_dark, bg_lum, pinned in cases:
        captured = []
        with tempfile.TemporaryDirectory() as td:
            ctrl, _dataset = _workspace(Path(td))
            model = {
                "labels": ["A", "1", "2"],
                "input_details": {"shape": [1, 28, 28, 1], "dtype": np.int8},
                "sign_sequence": "A12",
                "sign_sequences": ["A12"],
                "sign_background_replace_ood": False,
                "class_preprocess": {},
                "preprocess_mode": "auto_by_label",
                "model_source": "latest",
            }
            ctrl._preview_load_model = lambda session_id: model

            def fake_invoke(model, session_id, arr):
                captured.append(np.array(arr, copy=True))
                return None, np.asarray([200, 10, 10], dtype=np.int64)

            ctrl._invoke_glyph_tensor = fake_invoke
            res = ctrl._preview_predict_sign_ocr(
                "s", png,
                sign_sequence="A12",
                reading_order="ltr",
                bg_dark_thresh=bg_dark,
                bg_lum_thresh=bg_lum,
                ood_max_prob_min=0.40,
                preprocess_frame_side=96,
            )

        reading = op.preprocess_sign_sequence(
            small, sign_sequence="A12", reading_order="ltr", out_size=28,
            polarity=op.INK_DARK, apply_segmentation=True,
            convention=op.NORM_CANON, thresh=pinned, dark_thr=0,
            find_roi=True,
        )
        assert res.get("roi_box") == reading.get("roi_box"), (
            bg_lum, res.get("roi_box"), reading.get("roi_box")
        )
        assert len(captured) == 3, captured
        for got, slot in zip(captured, reading["slots"]):
            want = ((np.asarray(slot["glyph"], dtype=np.int16) - 128)
                    .astype(np.int8).reshape(28, 28, 1))
            assert np.array_equal(got, want), (bg_lum, slot["char"])


if __name__ == "__main__":
    raise SystemExit(__import__("pytest").main([__file__, "-q"]))
