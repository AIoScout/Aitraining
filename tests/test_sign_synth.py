"""Backend checks for the synthetic 3-character sign Preview.

The sign Preview used to need a photo of a road sign before it could show
anything.  These tests exercise the replacement: `POST
/preview/synthesize_sign` draws a whole plate from the project's own
samples (EMNIST fallback), splits it through the REAL sign front-end and
answers with the same payload `/preview/upload` returns.

Covered:
  verdicts      match -> YES, mismatch -> NO, reversed -> NO under ltr and
                YES under rtl (the direction probe)
  shape         a 240x96 light plate that `split_sign_glyph_boxes` cuts into
                exactly three glyph runs
  auto-fill     enabling the shortcut in a sign project stores a MATCH plate
                as the session's active Preview frame
  safety        a real upload is never replaced by the auto-synthesized one

Requires the AItraining venv (TensorFlow; the bundled car-sign template is
trained first, ~6 s):
    .venv/bin/python tests/test_sign_synth.py
"""
from __future__ import annotations

import base64
import io
import json
import os
import sys
import tempfile
import traceback
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, Path(__file__).resolve().parent):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
from PIL import Image

import record_controller
from dataset_io import PROJECT_TYPE_IMAGE, PROJECT_TYPE_SIGN_OCR
from ocr_preprocess import INK_DARK, binarize_ink, otsu_threshold_u8, split_sign_glyph_boxes

TEMPLATE_PROJECT = ROOT / "templates" / "car_sign_template.tmproj"
SEQUENCE = ["A", "1", "2"]


def _session_config(dataset: Path, *, project_type: str = PROJECT_TYPE_SIGN_OCR) -> Any:
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
        img_size=28,
        project_type=project_type,
    )


def _workspace(root: Path, classes=SEQUENCE):
    dataset = root / "ws" / "tm_dataset"
    dataset.mkdir(parents=True)
    ctrl = record_controller.RecordController()
    ctrl._configs["s"] = _session_config(dataset)
    ctrl._classes_save(dataset, list(classes))
    for name in classes:
        (dataset / name).mkdir(exist_ok=True)
    return ctrl, dataset


def _post_json(ctrl, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    url = f"http://127.0.0.1:{ctrl.port}{path}"
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=body, method="POST",
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=120) as res:
        return json.loads(res.read().decode("utf-8"))


def _get_json(ctrl, path: str) -> Dict[str, Any]:
    url = f"http://127.0.0.1:{ctrl.port}{path}"
    with urllib.request.urlopen(url, timeout=120) as res:
        return json.loads(res.read().decode("utf-8"))


def _plate(res: Dict[str, Any]) -> np.ndarray:
    raw = base64.b64decode(str(res.get("orig_b64") or res.get("image_b64") or ""))
    return np.asarray(Image.open(io.BytesIO(raw)).convert("L"), dtype=np.uint8)


def _boxes(plate: np.ndarray):
    ink = binarize_ink(plate, otsu_threshold_u8(plate), polarity=INK_DARK)
    return split_sign_glyph_boxes(ink, count=3)


def _train_template(ctrl) -> Dict[str, Any]:
    state = ctrl._project_open("s", TEMPLATE_PROJECT)
    assert state["classes"] == SEQUENCE, state["classes"]
    ctrl._train_worker("s", state["train_cfg"])
    status = ctrl._train.get("s") or {}
    assert not status.get("error"), status
    return state


# ── the three variants ────────────────────────────────────────────────────
def test_synth_variants_answer_yes_no_and_direction():
    with tempfile.TemporaryDirectory() as td:
        ctrl, _dataset = _workspace(Path(td))
        _train_template(ctrl)

        # Training itself prepares a ready-to-test MATCH plate, so the
        # upload source has an image before the user clicks anything.
        stored = ctrl._preview_uploads.get("s") or {}
        assert stored.get("synthetic") is True, stored
        assert stored.get("filename") == "synth_A12_match.png", stored.get("filename")

        ctrl.start()
        try:
            # 1. MATCH: the target, laid out left-to-right.
            yes = _post_json(
                ctrl, "/preview/synthesize_sign",
                {"session": "s", "variant": "match", "sign_sequence": "A12",
                 "reading_order": "ltr"},
            )
            assert yes.get("ok") == "1", yes
            assert yes.get("match") is True, yes
            assert str(yes.get("match_text") or "").startswith(record_controller.SIGN_MATCH_TEXT), yes
            assert [s["label"] for s in yes["slots"]] == SEQUENCE, yes["slots"]
            assert yes.get("synth_variant") == "match", yes
            assert yes.get("synth_chars") == SEQUENCE, yes.get("synth_chars")
            assert yes.get("filename") == "synth_A12_match.png", yes.get("filename")
            plate = _plate(yes)
            assert plate.shape == (96, 240), plate.shape
            assert float(plate.mean()) > 190.0, float(plate.mean())
            # The real front-end splits the synthesized plate into three
            # glyphs — the whole point of drawing gaps into it.
            assert len(_boxes(plate)) == 3, _boxes(plate)
            # The stored frame is the SAME plate the response drew, so a
            # later slider re-predict (`reuse=1`) reads those pixels.
            assert ctrl._preview_uploads["s"]["bytes"] == base64.b64decode(
                yes.get("orig_b64") or ""
            )

            # 2. MISMATCH: last character replaced.  The model may misread an
            # unseen replacement as the target; the endpoint retries until
            # the verdict is actually NO, which is the contract.
            no = _post_json(
                ctrl, "/preview/synthesize_sign",
                {"session": "s", "variant": "mismatch", "sign_sequence": "A12",
                 "reading_order": "ltr"},
            )
            assert no.get("ok") == "1", no
            assert no.get("match") is False, no
            assert no.get("match_text") == record_controller.SIGN_NOMATCH_TEXT, no
            assert no["synth_chars"][-1] != "2", no["synth_chars"]
            assert any(not slot.get("match") for slot in no["slots"]), no["slots"]

            # 3. REVERSED: A12 painted as 21A.  Left-to-right reading says
            # NO; a right-to-left project reads the SAME pixels as A12 and
            # says YES — the LTR/RTL probe the button promises.
            rev_ltr = _post_json(
                ctrl, "/preview/synthesize_sign",
                {"session": "s", "variant": "reversed", "sign_sequence": "A12",
                 "reading_order": "ltr"},
            )
            assert rev_ltr.get("match") is False, rev_ltr
            assert rev_ltr.get("synth_chars") == ["2", "1", "A"], rev_ltr.get("synth_chars")

            rev_rtl = _post_json(
                ctrl, "/preview/synthesize_sign",
                {"session": "s", "variant": "reversed", "sign_sequence": "A12",
                 "reading_order": "rtl"},
            )
            assert rev_rtl.get("match") is True, rev_rtl
            assert [s["label"] for s in rev_rtl["slots"]] == SEQUENCE, rev_rtl["slots"]

            # A MATCH under RTL paints the target right-to-left and still
            # reads YES in label order.
            match_rtl = _post_json(
                ctrl, "/preview/synthesize_sign",
                {"session": "s", "variant": "match", "sign_sequence": "A12",
                 "reading_order": "rtl"},
            )
            assert match_rtl.get("match") is True, match_rtl
            assert match_rtl.get("synth_chars") == ["2", "1", "A"], match_rtl.get("synth_chars")
            assert [s["label"] for s in match_rtl["slots"]] == SEQUENCE, match_rtl["slots"]
        finally:
            ctrl.stop()


# ── auto-fill hook ────────────────────────────────────────────────────────
def test_auto_fill_stores_a_match_plate_without_a_model():
    with tempfile.TemporaryDirectory() as td:
        ctrl, dataset = _workspace(Path(td))
        ctrl.start()
        try:
            assert all(not list((dataset / n).glob("*.png")) for n in SEQUENCE)

            res = _get_json(
                ctrl,
                "/live/config?session=s&auto_seed_emnist=1&include_state=1"
                "&sign_sequence=A12&reading_order=ltr",
            )
            assert res.get("counts") == {"A": 24, "1": 24, "2": 24}, res.get("counts")

            # The seed pass prepared the preview frame in the SAME request:
            # switching Preview to Upload / turning Input on needs no file.
            stored = ctrl._preview_uploads.get("s") or {}
            assert stored.get("synthetic") is True, stored
            assert stored.get("filename") == "synth_A12_match.png", stored.get("filename")
            plate = np.asarray(Image.open(io.BytesIO(stored["bytes"])).convert("L"))
            assert plate.shape == (96, 240), plate.shape
            assert len(_boxes(plate)) == 3, _boxes(plate)

            # A REAL upload is never replaced by a later auto-fill pass.
            ctrl._preview_uploads["s"] = {
                "bytes": b"<real upload>",
                "filename": "my_sign.png",
                "synthetic": False,
            }
            assert ctrl._maybe_autostore_synth_preview(
                "s", replace_synthetic=True
            ) is False
            assert ctrl._preview_uploads["s"]["filename"] == "my_sign.png"
        finally:
            ctrl.stop()


def test_synth_rejects_a_project_without_a_target():
    with tempfile.TemporaryDirectory() as td:
        ctrl, _dataset = _workspace(Path(td))
        ctrl.start()
        try:
            # No saved train config and no request sequence: there is no 3
            # character target to combine.  A clear 400, not a plate of '?'.
            try:
                _post_json(
                    ctrl, "/preview/synthesize_sign",
                    {"session": "s", "variant": "match"},
                )
            except Exception as exc:  # urllib raises on 4xx
                body = exc.read().decode("utf-8") if hasattr(exc, "read") else ""
                assert "target sign" in body, body
            else:
                raise AssertionError("synthesis without a target was accepted")

            # An image project has no glyph pipeline at all.
            ctrl.update_config("s", project_type=PROJECT_TYPE_IMAGE)
            try:
                _post_json(
                    ctrl, "/preview/synthesize_sign",
                    {"session": "s", "variant": "match", "sign_sequence": "A12"},
                )
            except Exception as exc:
                body = exc.read().decode("utf-8") if hasattr(exc, "read") else ""
                assert "sign_ocr" in body, body
            else:
                raise AssertionError("synthesis in an image project was accepted")
        finally:
            ctrl.stop()


# ── runner ────────────────────────────────────────────────────────────────
def main() -> int:
    tests = [
        ("synth variants answer YES / NO / direction", test_synth_variants_answer_yes_no_and_direction),
        ("auto-fill stores a match plate", test_auto_fill_stores_a_match_plate_without_a_model),
        ("synth rejects a project without a target", test_synth_rejects_a_project_without_a_target),
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
