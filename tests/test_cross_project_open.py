"""Backend checks for reopening a project of a different type (Task 2).

Opening a `.tmproj` used to do two things wrong, and both showed up as the
same symptom — a session that looked converted but was not:

  stale model   a model trained in the CURRENT session survived the open, so
                an Image project's 96x96 graph kept `export_enabled` true
                inside a freshly opened OCR project (and the preview kept
                running the previous project's interpreter)
  cross-type    the session kept its own `project_type` / `img_size` instead
                of adopting the archive's, so the SPA stayed in the previous
                project's UI mode while the dataset on disk was the new one

The open path fixes both by (a) tearing the session down before anything from
the archive lands, and (b) only adopting a bundled graph when the archive
itself carried one AND it matches the project being opened.  These tests pin
that behaviour with the real controller, real archives.

Requires the AItraining venv (TensorFlow):
    .venv/bin/python tests/test_cross_project_open.py
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import traceback
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, Path(__file__).resolve().parent):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
from PIL import Image

import record_controller
from dataset_io import PROJECT_MAGIC, project_header


# ── workspace fixtures ────────────────────────────────────────────────────
def _config(dataset: Path, project_type: str, img_size: int) -> record_controller.SessionConfig:
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


def _workspace(root: Path):
    dataset = root / "ws" / "tm_dataset"
    dataset.mkdir(parents=True)
    return record_controller.RecordController(), dataset


def _use(ctrl, dataset: Path, project_type: str, img_size: int) -> None:
    """Make the single session 's' be a project of this type and geometry."""
    ctrl._configs["s"] = _config(dataset, project_type, img_size)


def _png(value: int) -> bytes:
    return record_controller._to_png_bytes(Image.fromarray(np.full((96, 96), value, np.uint8), mode="L"))


def _seed(ctrl, dataset: Path, classes) -> None:
    """Replace the dataset with `classes`, two captures each."""
    if dataset.exists():
        for child in list(dataset.iterdir()):
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
    ctrl._classes_save(dataset, list(classes))
    for name in classes:
        d = dataset / name
        d.mkdir(exist_ok=True)
        (d / "capture_1.png").write_bytes(_png(120))
        (d / "capture_2.png").write_bytes(_png(150))


def _live_model(ctrl, dataset: Path, project_type: str, img_size: int) -> Path:
    """Plant the artefacts a finished training run leaves behind."""
    ws = dataset.parent
    run = ws / "runs" / "latest"
    run.mkdir(parents=True, exist_ok=True)
    (run / "train_config.json").write_text(
        json.dumps({"project_type": project_type, "img_size": img_size}), encoding="utf-8"
    )
    (run / "model.tflite").write_bytes(b"TFL3-this-project")
    (ws / "tm_train_latest.json").write_text(
        json.dumps(
            {
                "run_dir": str(run),
                "tflite_path": str(run / "model.tflite"),
                "project_type": project_type,
                "img_size": img_size,
            }
        ),
        encoding="utf-8",
    )
    return run


def _export_enabled(ctrl, dataset: Path) -> str:
    return str(ctrl._project_state_payload(dataset).get("export_enabled"))


def _craft(path: Path, *, manifest_type: str, manifest_size: int, run_type: str, run_size: int) -> Path:
    """Hand-write a `.tmproj` whose manifest and bundled run disagree.

    `_project_save` never produces this — it stamps both halves from one
    session — so the mismatch has to be built by hand to test the guard.
    """
    assert project_header(manifest_type)["_magic"] == PROJECT_MAGIC  # a file the app really reads
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        manifest = {
            **project_header(manifest_type),
            "format": "tmproj",
            "img_size": manifest_size,
            "dataset_dir": "dataset",
            "train_latest": "tm_train_latest.json",
            "train_run_dir": "runs/latest",
            "project_state": "tm_project_state.json",
            "processed_cache_dir": "",
        }
        state = {
            **project_header(manifest_type),
            "project_type": manifest_type,
            "train_cfg": {"project_type": manifest_type, "img_size": manifest_size},
        }
        zf.writestr("manifest.json", json.dumps(manifest))
        zf.writestr("tm_project_state.json", json.dumps(state))
        zf.writestr("dataset/labels.txt", "A\n1\n")
        zf.writestr("dataset/labels.json", json.dumps({"labels": ["A", "1"]}))
        for name in ("A", "1"):
            zf.writestr(f"dataset/{name}/capture_1.png", _png(120))
        # The record the saving machine wrote: absolute paths that still
        # resolve here, which is exactly how a stale run got adopted.
        zf.writestr(
            "tm_train_latest.json",
            json.dumps(
                {
                    "run_dir": "/original/machine/runs/latest",
                    "tflite_path": "/original/machine/runs/latest/model.tflite",
                }
            ),
        )
        zf.writestr(
            "runs/latest/train_config.json",
            json.dumps({"project_type": run_type, "img_size": run_size}),
        )
        zf.writestr("runs/latest/model.tflite", b"TFL3-bundled")
    return path


# ── tests ─────────────────────────────────────────────────────────────────
def test_open_clears_the_sessions_own_model():
    """An archive with no model must leave the session with no model.

    The session is an OCR project with a trained 28x28 model; the file is an
    Image project saved before it was ever trained.  Before the fix the
    session's graph and `tm_train_latest.json` survived, so the reopened
    project reported itself exportable off a model it never loaded.
    """
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        ctrl, dataset = _workspace(root)

        # The Image project, saved with no training run behind it.
        _use(ctrl, dataset, "image", 96)
        _seed(ctrl, dataset, ["LEFT", "STOP"])
        archive = root / "image_project.tmproj"
        ctrl._project_save("s", archive)

        # The same session becomes an OCR project that HAS trained.
        _use(ctrl, dataset, "ocr", 28)
        _seed(ctrl, dataset, ["A", "1"])
        run = _live_model(ctrl, dataset, "ocr", 28)
        assert _export_enabled(ctrl, dataset) == "1"
        ctrl._train["s"] = "STALE-INTERPRETER"
        ctrl._preview["s"] = "STALE-PREVIEW"

        state = ctrl._project_open("s", archive)

        assert state["project_type"] == "image", state.get("project_type")
        assert int(state["train_cfg"]["img_size"]) == 96, state["train_cfg"]
        assert state["export_enabled"] == "0", state["export_enabled"]
        assert not run.exists(), "the previous project's run survived the open"
        assert not (dataset.parent / "tm_train_latest.json").exists()
        assert "s" not in ctrl._train, "the cached interpreter survived the open"
        assert "s" not in ctrl._preview, "the cached preview survived the open"
        # ...and the live session config is the opened file's, not the old one.
        live = ctrl.get_config("s")
        assert live.project_type == "image", live.project_type
        assert int(live.img_size) == 96, live.img_size
        # The archive's own classes replaced the previous project's.
        assert state["classes"] == ["LEFT", "STOP"], state["classes"]


def test_open_refuses_a_bundled_run_from_another_project():
    """A bundled graph that disagrees with the file's own header is dropped.

    A `.tmproj` carries the run directory when one existed at save time.  Two
    projects saved in the same default workspace share `runs/latest`, so the
    paths in the record routinely resolve on the machine doing the opening —
    which is how an Image 96x96 graph became an OCR project's "trained" model.
    """
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        ctrl, dataset = _workspace(root)
        _use(ctrl, dataset, "sign_ocr", 28)
        _seed(ctrl, dataset, ["A", "1"])

        archive = _craft(
            root / "ocr_with_image_run.tmproj",
            manifest_type="ocr",
            manifest_size=28,
            run_type="image",
            run_size=96,
        )
        state = ctrl._project_open("s", archive)

        assert state["project_type"] == "ocr", state.get("project_type")
        assert int(state["train_cfg"]["img_size"]) == 28, state["train_cfg"]
        assert state["export_enabled"] == "0", state["export_enabled"]
        assert not (dataset.parent / "runs" / "latest").exists(), "the mismatched run was adopted"
        assert not (dataset.parent / "tm_train_latest.json").exists()


def test_open_keeps_a_bundled_run_that_matches():
    """The guard is a filter, not a blanket refusal: a matching graph comes back."""
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        ctrl, dataset = _workspace(root)
        _use(ctrl, dataset, "image", 96)
        _seed(ctrl, dataset, ["LEFT", "STOP"])

        archive = _craft(
            root / "sign_project.tmproj",
            manifest_type="sign_ocr",
            manifest_size=28,
            run_type="sign_ocr",
            run_size=28,
        )
        state = ctrl._project_open("s", archive)

        assert state["project_type"] == "sign_ocr", state.get("project_type")
        assert int(state["train_cfg"]["img_size"]) == 28, state["train_cfg"]
        assert state["export_enabled"] == "1", state["export_enabled"]
        run = dataset.parent / "runs" / "latest"
        assert (run / "model.tflite").read_bytes() == b"TFL3-bundled"
        meta = json.loads((dataset.parent / "tm_train_latest.json").read_text(encoding="utf-8"))
        # The record's absolute paths were rewritten to this machine's copy.
        assert Path(meta["tflite_path"]).resolve() == (run / "model.tflite").resolve()
        live = ctrl.get_config("s")
        assert live.project_type == "sign_ocr", live.project_type
        assert int(live.img_size) == 28, live.img_size


def test_open_takes_the_archives_type_both_ways():
    """Round trip: a trained Image file opened in an OCR session, and back."""
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        ctrl, dataset = _workspace(root)

        # An Image project that has trained, saved with its own 96x96 graph.
        _use(ctrl, dataset, "image", 96)
        _seed(ctrl, dataset, ["LEFT", "STOP"])
        _live_model(ctrl, dataset, "image", 96)
        image_archive = root / "trained_image.tmproj"
        ctrl._project_save("s", image_archive)

        # The session is now an OCR project with a 28x28 model of its own.
        _use(ctrl, dataset, "ocr", 28)
        _seed(ctrl, dataset, ["A", "1"])
        _live_model(ctrl, dataset, "ocr", 28)
        assert _export_enabled(ctrl, dataset) == "1"

        state = ctrl._project_open("s", image_archive)
        assert state["project_type"] == "image", state.get("project_type")
        assert int(state["train_cfg"]["img_size"]) == 96
        # The archive DID bundle a matching graph, so it comes back — and the
        # one that comes back is the archive's, not the OCR session's.
        assert state["export_enabled"] == "1", state["export_enabled"]
        restored = json.loads((dataset.parent / "tm_train_latest.json").read_text(encoding="utf-8"))
        assert Path(restored["tflite_path"]).read_bytes() == b"TFL3-this-project"
        assert "sign_sequence" not in (state.get("train_cfg") or {}), state["train_cfg"]

        # ...and the reverse: back into the OCR file, from the Image session.
        ocr_archive = root / "trained_ocr.tmproj"
        _use(ctrl, dataset, "ocr", 28)
        _seed(ctrl, dataset, ["A", "1"])
        _live_model(ctrl, dataset, "ocr", 28)
        ctrl._project_save("s", ocr_archive)

        _use(ctrl, dataset, "image", 96)
        _seed(ctrl, dataset, ["LEFT", "STOP"])
        _live_model(ctrl, dataset, "image", 96)

        state = ctrl._project_open("s", ocr_archive)
        assert state["project_type"] == "ocr", state.get("project_type")
        assert int(state["train_cfg"]["img_size"]) == 28
        assert state["export_enabled"] == "1", state["export_enabled"]
        assert state["classes"] == ["A", "1"], state["classes"]
        live = ctrl.get_config("s")
        assert live.project_type == "ocr" and int(live.img_size) == 28, live


def main() -> int:
    tests = [
        test_open_clears_the_sessions_own_model,
        test_open_refuses_a_bundled_run_from_another_project,
        test_open_keeps_a_bundled_run_that_matches,
        test_open_takes_the_archives_type_both_ways,
    ]
    passed = 0
    for fn in tests:
        try:
            fn()
        except Exception:  # noqa: BLE001 - a runner reports everything
            print(f"FAIL {fn.__name__}")
            traceback.print_exc()
        else:
            passed += 1
            print(f"ok   {fn.__name__}")
    print(f"\n{passed}/{len(tests)} passed")
    return 0 if passed == len(tests) else 1


if __name__ == "__main__":
    raise SystemExit(main())
