"""Backend checks for the auto-fill workflow and the Preview upload endpoint.

These exercise the real HTTP surface (`RecordController.start()`), not the
methods underneath it: the contract the SPA depends on is "call this endpoint,
get the whole runtime state back", and a method-level test cannot catch a
missing key in the JSON.

Covered:
  live/config   turning auto-fill ON backfills classes that ALREADY exist with
                zero samples and returns classes / counts / sample_previews /
                train_enabled in the SAME response
  seed endpoint /classes/seed_emnist with only_empty fills only the empty
                classes, leaves real captures alone, and returns full state
  train/config  saving a TrainConfig with the flag ON backfills too
  upload        /preview/upload accepts raw and JSON image bytes, routes a
                sign_ocr session through _preview_predict_sign_ocr, returns
                orig/roi base64 plus the verdict, and `reuse=1` re-predicts
                the stored picture without a second upload

Requires the AItraining venv (PIL; no TensorFlow training is run here):
    .venv/bin/python tests/test_autofill_preview.py
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import traceback
import urllib.request
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, Path(__file__).resolve().parent):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
from PIL import Image

import record_controller
from dataset_io import PROJECT_TYPE_SIGN_OCR

SEQUENCE = ["A", "1", "2"]


def _session_config(dataset: Path, *, project_type: str = PROJECT_TYPE_SIGN_OCR, img_size: int = 28):
    return record_controller.SessionConfig(
        dataset_root=dataset,
        serial_port="",
        serial_baud=115200,
        serial_sync="AA 55 AA",
        serial_frame_side=96,
        serial_channels=1,
        webcam_index=0,
        fps=8.0,
        crop_box=None,
        img_size=img_size,
        project_type=project_type,
    )


def _workspace(root: Path, classes=SEQUENCE):
    dataset = root / "ws" / "tm_dataset"
    dataset.mkdir(parents=True)
    ctrl = record_controller.RecordController()
    ctrl.set_config("s", _session_config(dataset))
    ctrl._classes_save(dataset, list(classes))
    for name in classes:
        (dataset / name).mkdir(exist_ok=True)
    ctrl.start()
    return ctrl, dataset


def _png_bytes(width: int = 96, height: int = 96, value: int = 128) -> bytes:
    arr = np.full((height, width), value, np.uint8)
    from io import BytesIO

    buf = BytesIO()
    Image.fromarray(arr, mode="L").save(buf, format="PNG")
    return buf.getvalue()


# ── HTTP helpers ──────────────────────────────────────────────────────────
def _get_json(ctrl, path: str) -> Dict[str, Any]:
    url = f"http://127.0.0.1:{ctrl.port}{path}"
    with urllib.request.urlopen(url, timeout=60) as res:
        return json.loads(res.read().decode("utf-8"))


def _post_json(ctrl, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    url = f"http://127.0.0.1:{ctrl.port}{path}"
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=body, method="POST",
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=60) as res:
        return json.loads(res.read().decode("utf-8"))


def _post_raw(
    ctrl,
    path: str,
    body: bytes,
    content_type: str = "image/png",
) -> Dict[str, Any]:
    url = f"http://127.0.0.1:{ctrl.port}{path}"
    req = urllib.request.Request(
        url, data=body, method="POST",
        headers={"Content-Type": content_type},
    )
    with urllib.request.urlopen(req, timeout=60) as res:
        return json.loads(res.read().decode("utf-8"))


def _class_files(dataset: Path, name: str):
    return sorted((dataset / name).glob("*.png"))


# ── auto-fill ─────────────────────────────────────────────────────────────
def test_live_config_toggle_backfills_existing_empty_classes():
    """The exact scenario from the task: ['A','1','2'] exist with 0 samples."""
    with tempfile.TemporaryDirectory() as td:
        ctrl, dataset = _workspace(Path(td))
        try:
            assert all(len(_class_files(dataset, n)) == 0 for n in SEQUENCE)

            res = _get_json(
                ctrl,
                "/live/config?session=s&auto_seed_emnist=1&include_state=1",
            )
            assert res.get("ok") == "1", res
            # The SAME response carries the full state the cards render from.
            assert res.get("classes") == SEQUENCE, res.get("classes")
            assert res.get("counts") == {"A": 24, "1": 24, "2": 24}, res.get("counts")
            assert res.get("train_enabled") is True, res.get("train_enabled")
            assert isinstance(res.get("train_rec"), dict), res.get("train_rec")
            for name in SEQUENCE:
                assert len(res["sample_previews"].get(name) or []) > 0, name

            # Files landed on disk, and the processed cache was updated, so a
            # training run started right away sees them.
            for name in SEQUENCE:
                assert len(_class_files(dataset, name)) == 24, name
            cache = dataset.parent / ".tm_processed_cache"
            for name in SEQUENCE:
                assert len(sorted((cache / name).glob("*.png"))) == 24, name

            # The flag is persisted so a later class edit (sync_ocr, add)
            # backfills without the SPA resending it.
            saved = json.loads(
                (dataset.parent / "runs" / "latest" / "train_config.json").read_text("utf-8")
            )
            assert saved.get("auto_seed_emnist_samples") is True, saved

            # A read-only include_state after the fill does not double-seed.
            again = _get_json(ctrl, "/live/config?session=s&include_state=1")
            assert again.get("counts") == {"A": 24, "1": 24, "2": 24}, again.get("counts")

            # And turning the toggle OFF persists False without deleting.
            off = _get_json(
                ctrl, "/live/config?session=s&auto_seed_emnist=0&include_state=1"
            )
            assert off.get("counts") == {"A": 24, "1": 24, "2": 24}, off.get("counts")
            saved = json.loads(
                (dataset.parent / "runs" / "latest" / "train_config.json").read_text("utf-8")
            )
            assert saved.get("auto_seed_emnist_samples") is False, saved
        finally:
            ctrl.stop()


def test_seed_endpoint_fills_only_empty_classes():
    with tempfile.TemporaryDirectory() as td:
        ctrl, dataset = _workspace(Path(td))
        try:
            # A has one real camera capture: "empty" is a per-class fact.
            (dataset / "A" / "camera_keep.png").write_bytes(_png_bytes())

            res = _post_json(
                ctrl, "/classes/seed_emnist",
                {"session": "s", "only_empty": True, "count": 24},
            )
            assert res.get("ok") == "1", res
            assert res.get("seeded") == ["1", "2"], res.get("seeded")
            assert res.get("skipped") == [], res.get("skipped")
            assert res.get("classes") == SEQUENCE, res.get("classes")
            assert res.get("counts") == {"A": 1, "1": 24, "2": 24}, res.get("counts")
            # Training turns ON: three classes, every one non-empty.
            assert res.get("train_enabled") is True, res.get("train_enabled")
            assert (dataset / "A" / "camera_keep.png").exists()

            # Calling again is the steady state, not an error: it reports
            # nothing seeded but still answers with the full state.
            again = _post_json(
                ctrl, "/classes/seed_emnist",
                {"session": "s", "only_empty": True, "count": 24},
            )
            assert again.get("ok") == "1", again
            assert again.get("seeded") == [], again.get("seeded")
            assert again.get("counts") == {"A": 1, "1": 24, "2": 24}, again.get("counts")

            # A plain (non-only_empty) seed still fills every character class,
            # which is the explicit "load templates" path.
            plain = _post_json(
                ctrl, "/classes/seed_emnist",
                {"session": "s", "classes": ["A"], "count": 24},
            )
            assert plain.get("seeded") == ["A"], plain.get("seeded")
            assert plain.get("counts", {}).get("A") == 25, plain.get("counts")
        finally:
            ctrl.stop()


def test_train_config_save_backfills():
    with tempfile.TemporaryDirectory() as td:
        ctrl, dataset = _workspace(Path(td))
        try:
            res = _post_json(
                ctrl, "/train/config",
                {
                    "session": "s",
                    "cfg": {
                        "img_size": 28,
                        "sign_sequence": "A12",
                        "auto_seed_emnist_samples": True,
                    },
                },
            )
            assert res.get("ok") == "1", res
            assert res.get("counts") == {"A": 24, "1": 24, "2": 24}, res.get("counts")
            assert res.get("train_enabled") is True, res.get("train_enabled")
            for name in SEQUENCE:
                assert len(_class_files(dataset, name)) == 24, name
        finally:
            ctrl.stop()


# ── Preview upload ────────────────────────────────────────────────────────
def _fake_sign_result():
    """The shape _preview_predict_sign_ocr returns, with a YES verdict."""
    return {
        "sign_ocr": True,
        "sign_sequence": SEQUENCE,
        "reading_order": "ltr",
        "slots": [
            {"char": "A", "label": "A", "prob": 0.99, "match": True},
            {"char": "1", "label": "1", "prob": 0.98, "match": True},
            {"char": "2", "label": "2", "prob": 0.97, "match": True},
        ],
        "slot_images_b64": ["", "", ""],
        "match": True,
        "match_text": record_controller.SIGN_MATCH_TEXT,
        "match_min_prob": 0.6,
        "processed_image_b64": "cm9p",
        "processed_variant": "sign_search",
        "crop_image_b64": "Y3JvcA==",
        "full_image_b64": "ZnVsbA==",
    }


def _install_preview_spy(ctrl):
    calls = []

    def spy(session_id, png, **kwargs):
        calls.append({"session_id": session_id, "png": png, "kwargs": kwargs})
        return _fake_sign_result()

    ctrl._preview_predict_sign_ocr = spy  # type: ignore[assignment]
    return calls


def test_preview_upload_raw_bytes_and_reuse():
    with tempfile.TemporaryDirectory() as td:
        ctrl, _dataset = _workspace(Path(td))
        try:
            calls = _install_preview_spy(ctrl)
            png = _png_bytes()

            # 1. Raw binary upload (the drag-and-drop / fetch(Blob) path).
            res = _post_raw(
                ctrl,
                "/preview/upload?session=s&filename=sign_A12.png"
                "&sign_sequence=A12&reading_order=ltr&dark=3&lum=90&match_min=0.6",
                png,
                "image/png",
            )
            assert res.get("ok") == "1", res
            assert len(calls) == 1, calls
            assert calls[0]["session_id"] == "s"
            assert calls[0]["kwargs"].get("sign_sequence") == "A12", calls[0]["kwargs"]
            assert calls[0]["kwargs"].get("bg_dark_thresh") == 3, calls[0]["kwargs"]
            assert calls[0]["kwargs"].get("bg_lum_thresh") == 90, calls[0]["kwargs"]
            assert calls[0]["kwargs"].get("match_min_prob") == 0.6, calls[0]["kwargs"]
            assert res.get("match_text") == record_controller.SIGN_MATCH_TEXT, res
            assert res.get("match") is True, res
            assert res.get("orig_b64"), res.keys()
            assert res.get("roi_b64") == "cm9p", res.get("roi_b64")
            assert res.get("filename") == "sign_A12.png", res.get("filename")
            # The stored copy is a clean PNG, whatever came in.
            stored = ctrl._preview_uploads["s"]["bytes"]
            assert stored.startswith(b"\x89PNG\r\n\x1a\n"), stored[:8]

            # 2. reuse=1 re-runs inference on the stored image (the Dark/Lum
            #    slider path) with no second upload.
            res2 = _post_json(
                ctrl, "/preview/upload",
                {"session": "s", "reuse": True, "match_min": "0.7"},
            )
            assert res2.get("ok") == "1", res2
            assert len(calls) == 2, calls
            assert calls[1]["png"] == stored, "reuse did not forward the stored PNG"
            assert calls[1]["kwargs"].get("match_min_prob") == 0.7, calls[1]["kwargs"]
        finally:
            ctrl.stop()


def test_preview_upload_json_and_dispatch():
    with tempfile.TemporaryDirectory() as td:
        ctrl, _dataset = _workspace(Path(td))
        try:
            calls = _install_preview_spy(ctrl)
            import base64

            res = _post_json(
                ctrl, "/preview/upload",
                {
                    "session": "s",
                    "image_b64": base64.b64encode(_png_bytes()).decode("ascii"),
                    "filename": "a12.png",
                    "sign_sequence": "A12",
                },
            )
            assert res.get("ok") == "1", res
            assert len(calls) == 1, calls
            assert res.get("match_text") == record_controller.SIGN_MATCH_TEXT, res

            # An upload with no image and no stored copy is a client error,
            # not a crash and not a silent empty prediction.
            try:
                _post_json(ctrl, "/preview/upload", {"session": "s", "filename": "x.png"})
            except Exception as exc:  # urllib raises on 4xx
                body = exc.read().decode("utf-8") if hasattr(exc, "read") else ""
                assert "missing image" in body, body
            else:
                raise AssertionError("empty upload was accepted")
        finally:
            ctrl.stop()


# ── runner ────────────────────────────────────────────────────────────────
def main() -> int:
    tests = [
        ("live/config backfills existing empty classes", test_live_config_toggle_backfills_existing_empty_classes),
        ("seed endpoint fills only empty classes", test_seed_endpoint_fills_only_empty_classes),
        ("train/config save backfills", test_train_config_save_backfills),
        ("preview upload raw bytes + reuse", test_preview_upload_raw_bytes_and_reuse),
        ("preview upload JSON + dispatch", test_preview_upload_json_and_dispatch),
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
