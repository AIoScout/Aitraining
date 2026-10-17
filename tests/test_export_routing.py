"""Which device sketch does an export belong to?

`model_settings.h` is the routing decision: each Arduino sketch is a separate
folder and refuses (at compile time) an export written for a different one.
That makes this file the cheapest place to catch a routing regression —
the sketch's `#error` only fires when a human opens the right folder in the
IDE, while a wrong `TM_*` macro reaching a device is a silently-wrong
preprocessing pipeline.

Checked here:
  image   -> TM_OCR_GLYPH_MODEL 0, TM_SIGN_OCR_MODEL 0
  ocr     -> TM_OCR_GLYPH_MODEL 1, TM_SIGN_OCR_MODEL 0
  sign    -> TM_OCR_GLYPH_MODEL 0, TM_SIGN_OCR_MODEL 1
  sign    -> kTargetSignSequence / kTargetSignClassIndices in LABEL order,
             kSignReadingOrder applied to the boxes (never mirrored),
             kSignMatchMinProb exported
  sign    -> the sequence comes from runtime_config, and from the run's
             train_config.json when an older SPA does not send it
  both    -> TM_THRESHOLD_WINDOW_EXPORTED + kDarkThr/kLumThr

Run with:
    .venv/bin/python -m pytest tests/test_export_routing.py -q
or, with no test runner installed:
    .venv/bin/python tests/test_export_routing.py
"""
from __future__ import annotations

import json
import re
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import dataset_io
import record_controller
import trainer


LABELS = ["_background", "A", "B", "1"]


def _workspace(tmp: Path, project_type: str,
               labels: list[str] | None = None) -> Path:
    """A dataset dir with a train meta pointing at a dummy .tflite."""
    dataset = tmp / "dataset"
    dataset.mkdir(parents=True, exist_ok=True)
    tflite = tmp / "model.tflite"
    tflite.write_bytes(b"not a real flatbuffer; the parsers are stubbed")
    meta = {
        "tflite_path": str(tflite),
        "labels": list(labels if labels is not None else LABELS),
        "img_size": 28,
        "color_mode": "grayscale",
        "project_type": project_type,
    }
    (tmp / "tm_train_latest.json").write_text(json.dumps(meta), encoding="utf-8")
    return dataset


def _controller(dataset: Path, project_type: str) -> record_controller.RecordController:
    ctrl = record_controller.RecordController()
    ctrl._configs["s"] = record_controller.SessionConfig(
        dataset_root=dataset,
        serial_port="",
        serial_baud=115200,
        serial_sync="0xAA55",
        serial_frame_side=96,
        serial_channels=3,
        webcam_index=0,
        fps=30.0,
        crop_box=None,
        project_type=project_type,
    )
    return ctrl


def _export(tmp: Path, project_type: str,
            runtime_config: dict | None = None,
            labels: list[str] | None = None) -> str:
    # An image project has no "_background" contract; the flag would
    # otherwise (correctly) route it to the OCR sketch on the class name.
    if labels is None and project_type == dataset_io.PROJECT_TYPE_IMAGE:
        labels = [x for x in LABELS if x != "_background"]
    dataset = _workspace(tmp, project_type, labels=labels)
    ctrl = _controller(dataset, project_type)
    out = tmp / "out"
    # The export only needs a byte blob behind the model; the two parsers
    # that would reject a fake flatbuffer are stubbed, so the test exercises
    # the header composition rather than TFLite parsing.
    real_sources = trainer.export_tflite_c_sources
    real_resolver = record_controller._generate_model_resolver_h
    trainer.export_tflite_c_sources = lambda data, array_name=None: ("// src", "// hdr")
    record_controller._generate_model_resolver_h = lambda data: "// resolver"
    try:
        out_dir, _legacy = ctrl._export_run(
            "s", out, "model", "g_model", overwrite=True,
            runtime_config=runtime_config,
        )
    finally:
        trainer.export_tflite_c_sources = real_sources
        record_controller._generate_model_resolver_h = real_resolver
    header = (out_dir / "model_settings.h").read_text(encoding="utf-8")
    assert out == out_dir
    return header


def _macro(header: str, name: str) -> str | None:
    m = re.search(rf"^#define\s+{re.escape(name)}\s+(\S+)\s*$", header, re.M)
    return m.group(1) if m else None


def test_image_export_targets_the_legacy_sketch():
    with tempfile.TemporaryDirectory() as td:
        header = _export(Path(td), dataset_io.PROJECT_TYPE_IMAGE)
    assert _macro(header, "TM_OCR_GLYPH_MODEL") == "0"
    assert _macro(header, "TM_SIGN_OCR_MODEL") == "0"
    assert "kTargetSignSequence" not in header
    assert "kTargetSignClassIndices" not in header


def test_ocr_export_targets_tflite_ocr_only():
    with tempfile.TemporaryDirectory() as td:
        header = _export(Path(td), dataset_io.PROJECT_TYPE_OCR)
    assert _macro(header, "TM_OCR_GLYPH_MODEL") == "1"
    assert _macro(header, "TM_SIGN_OCR_MODEL") == "0"
    assert "kTargetSignSequence" not in header


def test_sign_export_targets_tflite_sign_only():
    with tempfile.TemporaryDirectory() as td:
        header = _export(
            Path(td), dataset_io.PROJECT_TYPE_SIGN_OCR,
            runtime_config={"sign_sequence": "a1b", "reading_order": "rtl",
                            "match_min_prob": 0.7},
        )
    # A sign_ocr project is a glyph project, but NOT the single-glyph one.
    assert _macro(header, "TM_OCR_GLYPH_MODEL") == "0"
    assert _macro(header, "TM_SIGN_OCR_MODEL") == "1"
    assert 'constexpr char kTargetSignSequence[] = "A1B";' in header
    # LABEL order (the sequence is read A, 1, B), indices into
    # ["_background", "A", "B", "1"].
    assert "constexpr int kTargetSignClassIndices[3] = {1, 3, 2};" in header
    assert "constexpr int kSignReadingOrder = 1;" in header
    assert "constexpr float kSignMatchMinProb = 0.700000f;" in header


def test_sign_export_falls_back_to_the_runs_train_config():
    """An older SPA sends no sign fields; the run itself must remember."""
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        dataset = _workspace(tmp, dataset_io.PROJECT_TYPE_SIGN_OCR)
        run_dir = tmp / "runs" / "latest"
        run_dir.mkdir(parents=True)
        (run_dir / "train_config.json").write_text(
            json.dumps({"sign_sequence": ["B", "A", "1"],
                        "reading_order": "ltr"}), encoding="utf-8")
        ctrl = _controller(dataset, dataset_io.PROJECT_TYPE_SIGN_OCR)
        out = tmp / "out"
        real_sources = trainer.export_tflite_c_sources
        real_resolver = record_controller._generate_model_resolver_h
        trainer.export_tflite_c_sources = lambda data, array_name=None: ("// src", "// hdr")
        record_controller._generate_model_resolver_h = lambda data: "// resolver"
        try:
            out_dir, _legacy = ctrl._export_run(
                "s", out, "model", "g_model", overwrite=True,
                runtime_config={"sign_sequence": None, "reading_order": None},
            )
        finally:
            trainer.export_tflite_c_sources = real_sources
            record_controller._generate_model_resolver_h = real_resolver
        header = (out_dir / "model_settings.h").read_text(encoding="utf-8")
    assert 'constexpr char kTargetSignSequence[] = "BA1";' in header
    assert "constexpr int kTargetSignClassIndices[3] = {2, 1, 3};" in header
    assert "constexpr int kSignReadingOrder = 0;" in header


def test_sign_export_maps_an_unknown_character_to_minus_one():
    with tempfile.TemporaryDirectory() as td:
        header = _export(
            Path(td), dataset_io.PROJECT_TYPE_SIGN_OCR,
            runtime_config={"sign_sequence": "AZB", "reading_order": "ltr"},
        )
    assert "constexpr int kTargetSignClassIndices[3] = {1, -1, 2};" in header


def test_threshold_window_is_exported_for_both_sketches():
    for project_type in (dataset_io.PROJECT_TYPE_OCR,
                         dataset_io.PROJECT_TYPE_SIGN_OCR):
        with tempfile.TemporaryDirectory() as td:
            header = _export(
                Path(td), project_type,
                runtime_config={"bg_dark_thresh": 20, "bg_lum_thresh": 180},
            )
        assert _macro(header, "TM_THRESHOLD_WINDOW_EXPORTED") == "1"
        assert _macro(header, "kDarkThr") == "20"
        assert _macro(header, "kLumThr") == "180"


def test_threshold_window_defaults_are_the_bit_exact_no_op():
    with tempfile.TemporaryDirectory() as td:
        header = _export(
            Path(td), dataset_io.PROJECT_TYPE_OCR,
            runtime_config={"bg_dark_thresh": 0, "bg_lum_thresh": 100},
        )
    assert _macro(header, "kDarkThr") == "0"
    assert _macro(header, "kLumThr") == "100"


def main() -> int:
    tests = [(name, obj) for name, obj in sorted(globals().items())
             if name.startswith("test_") and callable(obj)]
    failures = 0
    for name, fn in tests:
        try:
            fn()
        except Exception as exc:  # noqa: BLE001 - a test runner reports everything
            failures += 1
            import traceback
            print(f"FAIL {name}: {exc}")
            traceback.print_exc()
        else:
            print(f"ok   {name}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
