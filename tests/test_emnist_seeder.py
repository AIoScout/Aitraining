"""Backend checks for the EMNIST template-sample seeder + sign template.

The seeder is the developer/demo shortcut behind Settings -> Advanced: it
fills character classes with EMNIST glyphs rendered as 96x96 DARK-INK-ON-
LIGHT captures, because that is the polarity the sign pipeline expects
(raw EMNIST is bright-on-dark and would binarize inverted).

Covered:
  seeding       24 samples/class, 96x96, dark ink on a light page; the
                processed cache gets 24 28x28 glyphs per class
  idempotence   re-seeding replaces only `emnist_seed_*` files — a camera
                capture in the same class folder survives
  template      templates/car_sign_template.tmproj opens as a sign_ocr
                project with the A,1,2 target and 24 samples per class
  end-to-end    that project trains fast and the preview says YES for A12,
                NO for A17, and YES for 21A when reading_order == "rtl"

Requires the AItraining venv (TensorFlow):
    .venv/bin/python tests/test_emnist_seeder.py
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, Path(__file__).resolve().parent):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
from PIL import Image

import record_controller
from dataset_io import PROJECT_TYPE_OCR, PROJECT_TYPE_SIGN_OCR
from ocr_preprocess import preprocess_glyph

TEMPLATES = ROOT / "templates"
TEMPLATE_PROJECT = TEMPLATES / "car_sign_template.tmproj"
SIGN_A12 = TEMPLATES / "sign_A12_match.png"
SIGN_21A = TEMPLATES / "sign_21A_rtl.png"
SIGN_A17 = TEMPLATES / "sign_A17_nomatch.png"
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


def _workspace(root: Path) -> tuple[record_controller.RecordController, Path]:
    dataset = root / "ws" / "tm_dataset"
    dataset.mkdir(parents=True)
    ctrl = record_controller.RecordController()
    ctrl._configs["s"] = _session_config(dataset)
    return ctrl, dataset


def _seed_classes(ctrl, dataset: Path, classes):
    ctrl._classes_save(dataset, list(classes))
    for name in classes:
        (dataset / name).mkdir(exist_ok=True)


def _png_bytes(arr: np.ndarray) -> bytes:
    return record_controller._to_png_bytes(Image.fromarray(arr, mode="L"))


# ── seeding ───────────────────────────────────────────────────────────────
def test_seeder_writes_camera_style_templates():
    with tempfile.TemporaryDirectory() as td:
        ctrl, dataset = _workspace(Path(td))
        _seed_classes(ctrl, dataset, SEQUENCE)

        out = ctrl._samples_seed_emnist("s", classes=SEQUENCE)
        # `classes` is the project's COMPLETE class list (it rides along with
        # the runtime state); `seeded` names what this call drew.
        assert out["seeded"] == SEQUENCE, out["seeded"]
        assert out["classes"] == SEQUENCE, out["classes"]
        assert out["skipped"] == [], out["skipped"]
        assert int(out["count"]) == 24

        cache = dataset.parent / ".tm_processed_cache"
        for name in SEQUENCE:
            raw = sorted((dataset / name).glob("emnist_seed_*.png"))
            assert len(raw) == 24, (name, len(raw))
            assert int(out["states"][name]["count"]) == 24
            for p in raw:
                arr = np.asarray(Image.open(p))
                assert arr.shape == (96, 96), (p.name, arr.shape)
                # Light page, dark ink — the polarity a physical sign has.
                assert float(arr.mean()) > 200.0, (p.name, float(arr.mean()))
                assert int(arr.min()) < 90, (p.name, int(arr.min()))
                assert int(arr.max()) > 230, (p.name, int(arr.max()))
                # And the real capture front-end accepts it as one glyph.
                res = preprocess_glyph(arr, apply_segmentation=True)
                assert res.get("ok"), (p.name, res)

            processed = sorted((cache / name).glob("emnist_seed_*.png"))
            assert len(processed) == 24, (name, len(processed))
            for p in processed:
                arr = np.asarray(Image.open(p))
                assert arr.shape == (28, 28), (p.name, arr.shape)
            assert len(out["processed_previews"][name]) > 0

        # Re-seed: replaces only its own files and never a camera capture.
        camera = dataset / "A" / "camera_keep.png"
        camera.write_bytes(_png_bytes(np.full((96, 96), 128, np.uint8)))
        again = ctrl._samples_seed_emnist("s")
        assert int(again["states"]["A"]["count"]) == 25
        assert camera.exists()
        assert len(list((dataset / "A").glob("emnist_seed_*.png"))) == 24
        assert len(list((cache / "A").glob("emnist_seed_*.png"))) == 24

        # A class name that is not one sign character is skipped, not guessed.
        ctrl._classes_save(dataset, SEQUENCE + ["Sign"])
        out = ctrl._samples_seed_emnist("s", classes=["sign", "a"])
        assert out["seeded"] == ["A"], out
        assert out["skipped"] == ["Sign"], out

        # Basic OCR uses the same glyph pipeline, so the seeder is allowed there.
        ctrl._configs["s"] = _session_config(dataset, project_type=PROJECT_TYPE_OCR)
        assert ctrl._samples_seed_emnist("s", classes=["1"])["seeded"] == ["1"]

        # An image project has no character classes to seed.
        ctrl._configs["s"] = _session_config(dataset, project_type="image")
        try:
            ctrl._samples_seed_emnist("s", classes=["A"])
        except ValueError as exc:
            assert "OCR" in str(exc)
        else:
            raise AssertionError("image project accepted template samples")


# ── bundled template project ──────────────────────────────────────────────
def _open_template(ctrl) -> dict:
    state = ctrl._project_open("s", TEMPLATE_PROJECT)
    assert state["classes"] == SEQUENCE, state["classes"]
    assert state["project_type"] == PROJECT_TYPE_SIGN_OCR, state.get("project_type")
    assert state["train_cfg"]["sign_sequence"] == SEQUENCE, state["train_cfg"].get("sign_sequence")
    assert state["train_cfg"]["reading_order"] == "ltr"
    assert int(state["train_cfg"]["img_size"]) == 28
    for name in SEQUENCE:
        assert int(state["counts"][name]) == 24, (name, state["counts"][name])
        assert len(state["processed_previews"][name]) > 0
    return state


def test_template_project_opens_ready_to_train():
    with tempfile.TemporaryDirectory() as td:
        ctrl, _dataset = _workspace(Path(td))
        _open_template(ctrl)


def test_template_trains_and_reads_the_signs():
    with tempfile.TemporaryDirectory() as td:
        ctrl, dataset = _workspace(Path(td))
        state = _open_template(ctrl)

        ctrl._train_worker("s", state["train_cfg"])
        status = ctrl._train.get("s") or {}
        assert not status.get("error"), status
        latest = ctrl._train_result_path(dataset)
        assert latest.exists()
        meta = json.loads(latest.read_text(encoding="utf-8"))
        assert meta["labels"] == SEQUENCE, meta.get("labels")
        assert Path(meta["tflite_path"]).exists()
        assert float(meta.get("metrics", {}).get("val_accuracy") or 0.0) > 0.8, meta.get("metrics")

        def predict(path: Path, sequence, order):
            res = ctrl._preview_predict_sign_ocr(
                "s",
                path.read_bytes(),
                sign_sequence=sequence,
                reading_order=order,
            )
            return res

        ok = predict(SIGN_A12, SEQUENCE, "ltr")
        assert ok["match"] is True, ok
        assert [s["label"] for s in ok["slots"]] == SEQUENCE, ok["slots"]
        assert str(ok["match_text"] or "").startswith(record_controller.SIGN_MATCH_TEXT)

        wrong = predict(SIGN_A17, SEQUENCE, "ltr")
        assert wrong["match"] is False, wrong
        assert wrong["match_text"] == record_controller.SIGN_NOMATCH_TEXT

        # The same physical sign (A,1,2 painted left to right) read RTL is
        # 2,1,A — pixels are reordered, never mirrored.
        rtl = predict(SIGN_21A, ["2", "1", "A"], "rtl")
        assert rtl["match"] is True, rtl
        assert [s["label"] for s in rtl["slots"]] == ["2", "1", "A"], rtl["slots"]


# ── runner ────────────────────────────────────────────────────────────────
def main() -> int:
    tests = [
        ("seeder writes camera-style templates", test_seeder_writes_camera_style_templates),
        ("template project opens ready to train", test_template_project_opens_ready_to_train),
        ("template trains and reads the signs", test_template_trains_and_reads_the_signs),
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


def test_one_seeding_keeps_only_upright_vertical_bars():
    """RC3: seeded EMNIST '1's must be upright vertical bars.

    EMNIST's handwritten '1's are mostly diagonal slashes (35-45 deg),
    which taught the model to read an upright printed '1' (|) as 'A' or
    '2'.  The pre-widening ink bbox separates them: an upright bar is
    narrow and vertical (w / h <= 0.40), a diagonal slash is about as
    wide as it is tall.  The template bank and the training-time mix
    both drop the slashes.
    """
    import ocr_dataset as od

    def _bar(w: int, h: int, *, slant: int = 0) -> np.ndarray:
        """A dark-on-light glyph: vertical bar or diagonal slash."""
        img = np.zeros((28, 28), dtype=np.uint8)
        for i in range(h):
            x = 13 + (slant * i // h)
            for j in range(w):
                col = x + j
                if 0 <= col < 28:
                    img[10 + i, col] = 255
        return img

    upright = _bar(2, 12)
    slanted = _bar(2, 12, slant=10)   # w/h -> ~0.83
    assert od._is_upright_one(upright), "upright bar rejected"
    assert not od._is_upright_one(slanted), "diagonal slash accepted"
    # Edge: exactly at the 0.40 ratio -> kept (the contract is <=).
    assert od._is_upright_one(_bar(4, 10)), "w/h == 0.40 must be kept"

    # The bank funnel applies the filter for '1' (and only for '1').
    bank: dict = {}
    od._bank_add(bank, "1", upright)
    od._bank_add(bank, "1", slanted)
    assert [g.shape for g in bank["1"]] == [(28, 28)], "slanted '1' leaked into the bank"
    od._bank_add(bank, "A", slanted)
    assert len(bank["A"]) == 1, "the filter must not touch other classes"
