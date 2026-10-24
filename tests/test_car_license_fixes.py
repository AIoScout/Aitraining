"""Regression tests for the Car License (sign_ocr) class/polarity fixes.

The reported failure: a workspace still holding the auto-named
"Class 1"/"Class 2" folders trained with those as real classes (the EMNIST
mapper reads their trailing digit and fills them), produced
`no readable images` warnings, shifted labels to "Class 2", and let
`_background` swallow real characters.

These tests pin the four contracts:
  classes     the target sequence is the ONLY class authority; stale folders
              in dataset/, .tm_processed_cache/ and .tm_ocr_mixed/ are gone
              before the mixed set is built, and the labels never contain
              "Class N"
  toggle      `sign_background_replace_ood` defaults OFF for a Car License
              project (no `_background`, OOD + 60 % floor) and ON trains the
              extra class; both read an A 1 2 plate as 🟢 YES
  export      model_settings.h carries kSignBackgroundIsTheOnlyRejector 0/1
              matching the run
  polarity    training cache, mixed EMNIST/background samples and the three
              preview slot tensors all use the same dark-ink encoding
              (CANON zero pad; no inverted half anywhere)

Requires the AItraining venv (TensorFlow):
    .venv/bin/python -m pytest tests/test_car_license_fixes.py -q
"""
from __future__ import annotations

import base64
import json
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

import ocr_preprocess as op
import record_controller as rc
from test_emnist_seeder import SIGN_A12, _open_template, _workspace
from test_sign_preview_fixes import _png, _thin_one_frame


def _inject_stale_classes(ctrl, dataset: Path) -> None:
    """Reproduce the reported state: two default class folders + labels."""
    classes = ctrl._classes_load(dataset)
    ctrl._classes_save(dataset, list(classes) + ["Class 1", "Class 2"])
    for name in ("Class 1", "Class 2"):
        (dataset / name).mkdir(exist_ok=True)


def _train(ctrl, state: dict, *, background: bool) -> dict:
    cfg = dict(state["train_cfg"])
    cfg.update(
        {
            "ocr_emnist_enabled": True,
            "ocr_emnist_per_class": 200,
            "ocr_background_enabled": bool(background),
            "sign_background_replace_ood": bool(background),
            "ocr_background_samples": 120,
        }
    )
    ctrl._train_worker("s", cfg)
    status = ctrl._train.get("s") or {}
    assert not status.get("error"), status
    return json.loads(ctrl._train_result_path(ctrl._configs["s"].dataset_root).read_text())


def _predict(ctrl, png: bytes, sequence: str = "A12") -> dict:
    return ctrl._preview_predict_sign_ocr(
        "s", png, sign_sequence=sequence, reading_order="ltr",
        bg_dark_thresh=0, bg_lum_thresh=60,
    )


def _assert_a12_yes(res: dict) -> None:
    slots = res.get("slots") or []
    assert [s.get("char") for s in slots] == ["A", "1", "2"], slots
    assert [s.get("label") for s in slots] == ["A", "1", "2"], slots
    assert not any(s.get("background") for s in slots), slots
    assert all(not s.get("ood") for s in slots), slots
    assert res.get("match") is True, res
    assert str(res.get("match_text") or "").startswith(rc.SIGN_MATCH_TEXT), res.get("match_text")


def _assert_png_encoding(path: Path, *, require_bright_page: bool) -> np.ndarray:
    """One training image must use the canonical dark-ink encoding.

    Every glyph is pasted into a zero-filled CANON canvas, so the border
    pixel is 0 (black), never 255.  A bright page, when the sample has one,
    is the paper tone; the ink is dark.  An inverted pipeline flips exactly
    these three facts.
    """
    arr = np.asarray(Image.open(path).convert("L"))
    assert arr.shape == (28, 28), (path, arr.shape)
    assert int(arr[0, 0]) == 0, (path, "CANON pad must be black, got", int(arr[0, 0]))
    if require_bright_page:
        assert int(arr.max()) > 120, (path, int(arr.max()))
        assert bool((arr < 100).any()), (path, "no dark ink")
    return arr


def _assert_preview_polarity(sign_png: bytes) -> None:
    """The three slot tensors use the same encoding and survive a PNG round-trip."""
    frame = np.asarray(Image.open(BytesIO(sign_png)).convert("L"))
    roi = op.find_sign_roi(frame)
    x1, y1, x2, y2 = roi["box"]
    reading = op.preprocess_sign_sequence(
        frame[y1:y2, x1:x2], sign_sequence="A12", out_size=28,
        convention=op.NORM_CANON, clean_border=True,
    )
    assert len(reading["slots"]) == 3, reading
    for slot in reading["slots"]:
        glyph = np.asarray(slot["glyph"], dtype=np.uint8)
        assert glyph.shape == (28, 28), glyph.shape
        assert int(glyph[0, 0]) == 0, (slot["char"], int(glyph[0, 0]))
        assert bool((glyph < 100).any()), (slot["char"], "no dark ink")
        # Preview feeds glyph.astype(float32) / 255; the training loader
        # reads the same PNG bytes and divides by 255.  Round-trip the array
        # through PNG and assert the exact same float tensor either way.
        buf = BytesIO()
        Image.fromarray(glyph, mode="L").save(buf, format="PNG")
        buf.seek(0)
        reloaded = np.asarray(Image.open(buf).convert("L"), dtype=np.uint8)
        assert np.array_equal(glyph, reloaded), slot["char"]
        preview = glyph.astype(np.float32) / 255.0
        training = reloaded.astype(np.float32) / 255.0
        assert np.array_equal(preview, training), slot["char"]
        assert 0.0 <= float(preview.min()) and float(preview.max()) <= 1.0


def _export_header(ctrl, dataset: Path) -> str:
    export_dir = dataset.parent / "export_check"
    ctrl._export_run("s", export_dir, "tm_sign", "g_model", overwrite=True)
    return (export_dir / "model_settings.h").read_text(encoding="utf-8")


def test_stale_class_dirs_are_pruned_and_background_defaults_off():
    with tempfile.TemporaryDirectory() as td:
        ctrl, dataset = _workspace(Path(td))
        state = _open_template(ctrl)
        _inject_stale_classes(ctrl, dataset)
        meta = _train(ctrl, state, background=False)

        classes = ctrl._classes_load(dataset)
        assert classes == ["A", "1", "2"], classes
        assert {p.name for p in dataset.iterdir() if p.is_dir()} == {"A", "1", "2"}
        processed = dataset.parent / ".tm_processed_cache"
        assert {p.name for p in processed.iterdir() if p.is_dir()} == {"A", "1", "2"}
        mixed = dataset.parent / ".tm_ocr_mixed"
        assert {p.name for p in mixed.iterdir() if p.is_dir()} == {"A", "1", "2"}

        assert set(meta["labels"]) == {"A", "1", "2"}, meta["labels"]
        assert not any(str(x).startswith("Class ") for x in meta["labels"]), meta["labels"]
        message = str((ctrl._train.get("s") or {}).get("message") or "")
        assert "Class 1" not in message and "Class 2" not in message, message
        assert "no readable images" not in message, message

        _assert_a12_yes(_predict(ctrl, SIGN_A12.read_bytes()))
        _assert_a12_yes(_predict(ctrl, _png(_thin_one_frame())))

        # With no _background class, a blank frame must not become a spurious
        # MATCH.  Layer 1 (sign_pct) is disarmed on the 3-slot path, so the
        # rejection is the per-slot ink/empty check (plus Layers 2/3 when the
        # model is invoked on an empty glyph).
        blank = _predict(ctrl, _png(np.full((160, 160), 245, np.uint8)))
        assert blank["match"] is False, blank
        assert any(s.get("empty") or not s.get("label") for s in blank["slots"]), blank["slots"]

        # Training-cache transform is reproducible byte-for-byte (the same
        # PNG the mixed set copies), so cache and training see identical
        # pixel polarity.
        raw = sorted((dataset / "A").glob("emnist_seed_*.png"))[0]
        cached = processed / "A" / f"{raw.stem}.png"
        cfg = ctrl._class_preprocess_load(dataset).get("A", {})
        assert ctrl._preprocess_image_png_ocr(raw.read_bytes(), cfg, out_size=28) == cached.read_bytes()

        # (a) processed cache, (b) mixed EMNIST, (d) preview slots all use
        # the same dark-ink encoding.
        for cls in ("A", "1", "2"):
            _assert_png_encoding(next((processed / cls).glob("*.png")), require_bright_page=True)
            _assert_png_encoding(next((mixed / cls).glob("*.png")), require_bright_page=True)
        _assert_preview_polarity(SIGN_A12.read_bytes())

        header = _export_header(ctrl, dataset)
        assert "#define TM_SIGN_USE_BACKGROUND_OOD 0" in header, header
        assert "constexpr bool kSignBackgroundReplacesOod = false;" in header, header
        assert "constexpr int kBackgroundClassIndex = -1;" in header, header
        assert "#define TM_SIGN_BACKGROUND_IS_THE_ONLY_REJECTOR 0" in header, header
        assert "constexpr bool kSignBackgroundIsTheOnlyRejector = false;" in header, header


def test_edit_auto_roi_extracts_the_class_slot_and_28x28_preview():
    """The Edit worktable's Auto mode in a sign class card must extract
    THIS class's slot from a full board, and the Processed pane is the
    canonical 28x28 glyph (not the whole sign or a pre-centring crop).

    This is the modal path (`/preprocess/preview` -> _preprocess_preview).
    """
    with tempfile.TemporaryDirectory() as td:
        ctrl, dataset = _workspace(Path(td))
        _open_template(ctrl)
        # A full A 1 2 board with border noise, filed under class 'A'.
        Image.fromarray(_thin_one_frame(), mode="L").save(
            dataset / "A" / "board.png"
        )

        def preview(cfg):
            return ctrl._preprocess_preview(
                "s", "A", "board.png", cfg,
                sign_sequences=["A12"], reading_order="ltr",
            )

        default = preview({})
        tuned = preview(
            {"mode": "auto_by_label", "bg_dark_thresh": 0, "bg_lum_thresh": 59}
        )
        for res in (default, tuned):
            crop = res.get("crop")
            assert isinstance(crop, list) and len(crop) == 4, res
            assert crop != [0.0, 0.0, 1.0, 1.0], res
            roi = res.get("auto_roi")
            assert isinstance(roi, list) and len(roi) == 4, res
            x1, y1, x2, y2 = roi
            # Class 'A' -> slot 0, NOT the whole A..2 board.
            assert x2 < 100, ("the whole sign leaked into class A", roi)
            assert x1 < 70 and y1 < 80 and y2 > 80, roi
            proc = np.asarray(
                Image.open(BytesIO(base64.b64decode(res["processed_image_b64"])))
            )
            assert proc.shape == (28, 28), (proc.shape, roi)
            assert set(np.unique(proc)).issubset({0, 255}), np.unique(proc)
            assert int(proc.min()) == 0 and int(proc.max()) == 255
            assert int(proc[0, 0]) == 255, "threshold background must be white"
            # `crop` is the slot's PADDED crop (8 % plate context) and the
            # green `auto_roi` is its ink bbox inside it.
            assert crop[0] <= x1 / 160.0 and crop[2] >= x2 / 160.0, (crop, roi)
            assert x1 / 160.0 - crop[0] <= 4.0 / 160.0, (crop, roi)
            assert crop[2] - x2 / 160.0 <= 4.0 / 160.0, (crop, roi)


def test_project_open_prunes_stale_classes_from_dataset_and_cache():
    """Opening a .tmproj whose target is A 1 2 drops Class 1/Class 2 too."""
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        ctrl, dataset = _workspace(root / "source")
        state = _open_template(ctrl)
        _inject_stale_classes(ctrl, dataset)
        project_state = ctrl._project_state_payload(dataset)
        project_state["project_type"] = "sign_ocr"
        project_state["train_cfg"] = dict(state["train_cfg"])
        saved = root / "stale.tmproj"
        ctrl._project_save("s", saved, project_state=project_state)

        ctrl2, dataset2 = _workspace(root / "opened")
        opened = ctrl2._project_open("s", saved)

        assert opened.get("project_type") == "sign_ocr", opened.get("project_type")
        assert opened.get("classes") == ["A", "1", "2"], opened.get("classes")
        assert {p.name for p in dataset2.iterdir() if p.is_dir()} == {"A", "1", "2"}
        processed = dataset2.parent / ".tm_processed_cache"
        if processed.exists():
            assert {p.name for p in processed.iterdir() if p.is_dir()} == {"A", "1", "2"}


def test_background_toggle_on_trains_the_rejector_and_still_reads_a12():
    with tempfile.TemporaryDirectory() as td:
        ctrl, dataset = _workspace(Path(td))
        state = _open_template(ctrl)
        _inject_stale_classes(ctrl, dataset)
        meta = _train(ctrl, state, background=True)

        classes = ctrl._classes_load(dataset)
        assert classes == ["A", "1", "2"], classes
        assert set(meta["labels"]) == {"A", "1", "2", "_background"}, meta["labels"]
        assert not any(str(x).startswith("Class ") for x in meta["labels"]), meta["labels"]
        message = str((ctrl._train.get("s") or {}).get("message") or "")
        assert "Class 1" not in message and "Class 2" not in message, message
        assert "no readable images" not in message, message

        _assert_a12_yes(_predict(ctrl, SIGN_A12.read_bytes()))
        _assert_a12_yes(_predict(ctrl, _png(_thin_one_frame())))

        processed = dataset.parent / ".tm_processed_cache"
        mixed = dataset.parent / ".tm_ocr_mixed"
        for cls in ("A", "1", "2"):
            _assert_png_encoding(next((processed / cls).glob("*.png")), require_bright_page=True)
            _assert_png_encoding(next((mixed / cls).glob("*.png")), require_bright_page=True)
        # (c) Background crops share the same CANON pad convention.  They may
        # legitimately be all-zero/noise, so only the pad polarity is pinned.
        for path in sorted((mixed / "_background").glob("*.png"))[:12]:
            _assert_png_encoding(path, require_bright_page=False)
        _assert_preview_polarity(SIGN_A12.read_bytes())

        header = _export_header(ctrl, dataset)
        assert "#define TM_SIGN_USE_BACKGROUND_OOD 1" in header, header
        assert "constexpr bool kSignBackgroundReplacesOod = true;" in header, header
        assert "constexpr int kBackgroundClassIndex = 3;" in header, header
        assert "#define TM_SIGN_BACKGROUND_IS_THE_ONLY_REJECTOR 1" in header, header
        assert "constexpr bool kSignBackgroundIsTheOnlyRejector = true;" in header, header


def test_ui_display_name_is_car_license():
    import app

    assert app._tm_project_type_label("sign_ocr") == "🚗 Car License Project"
    source = (ROOT / "app.py").read_text(encoding="utf-8")
    assert "🛑 Road Sign (3-Char)" not in source
    assert "Road Sign Project (3-Char)" not in source
    assert "🚗 Car License Project" in source
    # Internal key is unchanged.
    assert "sign_ocr" in source
