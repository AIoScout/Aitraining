"""Regression: changing a sign slot after training must auto-fill the new class.

The reported symptom: with a model already trained (so Preview/Export are
live), editing a slot in `#signSeqBar` — A-1-2 -> A-1-B — sometimes leaves B
at "0 Image Samples" instead of the 24 EMNIST templates.  The route the SPA
actually calls is `/classes/sync_ocr`; this test drives that route over HTTP
after a real training run and asserts the new character is filled on EVERY
attempt, from disk to the JSON the cards render from.

Covered:
  after training   training preserves the project's auto_seed flag
  sequence edit    A12 -> A1B fills B with 24 on-disk PNGs, returns
                   counts / sample_previews / train_enabled in one response,
                   and removes the dropped '2' (class, cache, counts)
  determinism      A1B <-> XYZ five times: every new character gets 24
  flag sources     persisted-on seeds with no request field; request-on
                   heals a persisted-off project; both-off stays empty

Requires the AItraining venv (TensorFlow for the training step):
    .venv/bin/python tests/test_sign_sequence_autofill.py
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import traceback
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, Path(__file__).resolve().parent):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import record_controller
from dataset_io import PROJECT_TYPE_SIGN_OCR

SEQUENCE = ["A", "1", "2"]


def _session_config(dataset: Path):
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
        project_type=PROJECT_TYPE_SIGN_OCR,
    )


def _post_json(ctrl, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    url = f"http://127.0.0.1:{ctrl.port}{path}"
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=180) as res:
        return json.loads(res.read().decode("utf-8"))


def _sync(ctrl, names: List[str], sequence: str, auto_seed: Optional[bool]):
    """The exact body `syncSignClassesToSequence()` posts from #signSeqBar."""
    payload: Dict[str, Any] = {
        "session": "s",
        "names": names,
        "sign_sequence": sequence,
        "reading_order": "ltr",
    }
    if auto_seed is not None:
        payload["auto_seed_emnist_samples"] = auto_seed
    return _post_json(ctrl, "/classes/sync_ocr", payload)


def _class_pngs(dataset: Path, name: str) -> List[Path]:
    folder = dataset / name
    return sorted(folder.glob("*.png")) if folder.exists() else []


def _saved_cfg(workspace: Path) -> Dict[str, Any]:
    path = workspace / "runs" / "latest" / "train_config.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _workspace(root: Path):
    dataset = root / "ws" / "tm_dataset"
    dataset.mkdir(parents=True)
    ctrl = record_controller.RecordController()
    ctrl.set_config("s", _session_config(dataset))
    ctrl._classes_save(dataset, list(SEQUENCE))
    for name in SEQUENCE:
        (dataset / name).mkdir()
    # The Settings checkbox persisted on, and the initial classes filled —
    # the state the report starts from.
    ctrl._save_train_cfg(
        "s",
        {
            "img_size": 28,
            "sign_sequence": list(SEQUENCE),
            "reading_order": "ltr",
            "auto_seed_emnist_samples": True,
        },
    )
    ctrl._seed_empty_char_classes("s")
    ctrl.start()
    return ctrl, dataset


def _train(ctrl, dataset: Path) -> Dict[str, Any]:
    """A real training run, so export_enabled/preview are live afterwards."""
    ctrl._train_worker(
        "s",
        {
            "img_size": 28,
            "epochs": 1,
            "batch_size": 8,
            "auto_adjust_batch_and_epochs": False,
            "ocr_emnist_enabled": False,
            "ocr_transfer_learning": False,
            "sign_sequence": "A12",
            "reading_order": "ltr",
            "auto_seed_emnist_samples": True,
            "crop_mode": "auto_search",
        },
    )
    status = ctrl._train.get("s") or {}
    assert not status.get("error"), status
    meta = json.loads(ctrl._train_result_path(dataset).read_text(encoding="utf-8"))
    assert Path(meta["tflite_path"]).exists(), meta.get("tflite_path")
    return meta


def test_post_training_sequence_edit_fills_new_class():
    with tempfile.TemporaryDirectory() as td:
        ctrl, dataset = _workspace(Path(td))
        try:
            meta = _train(ctrl, dataset)
            workspace = dataset.parent
            state = ctrl._project_state_payload(dataset)
            assert state["export_enabled"] == "1", state["export_enabled"]
            assert _saved_cfg(workspace).get("auto_seed_emnist_samples") is True
            # The trained run itself records the project's flag (so an
            # archive/export cannot carry a silently-disabled fill).
            run_cfg = json.loads(
                (Path(meta["run_dir"]) / "train_config.json").read_text(encoding="utf-8")
            )
            assert run_cfg.get("auto_seed_emnist_samples") is True, run_cfg

            res = _sync(ctrl, ["A", "1", "B"], "A1B", True)
            assert res.get("ok") == "1", res
            assert res.get("seed_error") == "", res.get("seed_error")
            assert res.get("classes") == ["A", "1", "B"], res.get("classes")
            assert res.get("counts") == {"A": 24, "1": 24, "B": 24}, res.get("counts")
            assert len(res.get("sample_previews", {}).get("B") or []) > 0, res.get(
                "sample_previews", {}
            ).get("B")
            assert res.get("train_enabled") is True, res.get("train_enabled")
            assert res.get("seeded") == ["B"], res.get("seeded")

            # On disk, not just in the response.
            assert len(_class_pngs(dataset, "B")) == 24, _class_pngs(dataset, "B")
            # The dropped character travels with its directory and cache.
            assert not (dataset / "2").exists()
            assert not (
                dataset.parent / ".tm_processed_cache" / "2"
            ).exists()
            assert set(res["counts"]) == {"A", "1", "B"}
            # The target is persisted now, not only when the next training
            # starts: reopen/export/synthesis read it from this file.
            saved = _saved_cfg(workspace)
            assert saved.get("sign_sequence") == ["A", "1", "B"], saved.get("sign_sequence")
            assert saved.get("auto_seed_emnist_samples") is True

            # The model on disk still carries the OLD labels (plus the
            # trainer's reject class) — the SPA's dataset fingerprint now
            # differs, which is what paints "Stale" and keeps Train
            # clickable.  Assert the premise of that marker.
            trained_labels = list(meta.get("labels") or [])
            assert "B" not in trained_labels, trained_labels
            assert res["classes"] != trained_labels
        finally:
            ctrl.stop()


def test_every_character_change_is_filled_100_percent():
    with tempfile.TemporaryDirectory() as td:
        ctrl, dataset = _workspace(Path(td))
        try:
            _train(ctrl, dataset)
            for i in range(5):
                res = _sync(ctrl, ["A", "1", "B"], "A1B", True)
                assert res.get("counts", {}).get("B") == 24, (i, res.get("counts"))
                assert len(_class_pngs(dataset, "B")) == 24, (i, len(_class_pngs(dataset, "B")))
                assert res.get("train_enabled") is True, (i, res.get("train_enabled"))

                res = _sync(ctrl, ["X", "Y", "Z"], "XYZ", True)
                assert res.get("classes") == ["X", "Y", "Z"], (i, res.get("classes"))
                assert res.get("counts") == {"X": 24, "Y": 24, "Z": 24}, (i, res.get("counts"))
                assert res.get("train_enabled") is True, (i, res.get("train_enabled"))
                for name in ("X", "Y", "Z"):
                    assert len(_class_pngs(dataset, name)) == 24, (i, name)
                    assert len(res["sample_previews"].get(name) or []) > 0, (i, name)
                assert set(res["counts"]) == {"X", "Y", "Z"}
        finally:
            ctrl.stop()


def test_flag_comes_from_either_the_request_or_the_project():
    with tempfile.TemporaryDirectory() as td:
        ctrl, dataset = _workspace(Path(td))
        try:
            _train(ctrl, dataset)
            workspace = dataset.parent

            # 1. Persisted ON, request omits the field: the server's copy
            #    still fills.  (An older SPA or a background caller.)
            res = _sync(ctrl, ["A", "1", "C"], "A1C", None)
            assert res.get("counts", {}).get("C") == 24, res.get("counts")

            # 2. Persisted OFF, request OFF: nothing is fabricated.
            ctrl._save_train_cfg("s", {"auto_seed_emnist_samples": False})
            res = _sync(ctrl, ["A", "1", "D"], "A1D", False)
            assert res.get("counts", {}).get("D") == 0, res.get("counts")
            assert _saved_cfg(workspace).get("auto_seed_emnist_samples") is False

            # 3. Persisted OFF, request ON: fill AND heal the persisted flag,
            #    so the next class write cannot silently regress.
            res = _sync(ctrl, ["A", "1", "E"], "A1E", True)
            assert res.get("counts", {}).get("E") == 24, res.get("counts")
            assert _saved_cfg(workspace).get("auto_seed_emnist_samples") is True
        finally:
            ctrl.stop()


def main() -> int:
    tests = [
        ("post-training A12 -> A1B fills B", test_post_training_sequence_edit_fills_new_class),
        ("every A1B/XYZ change is filled", test_every_character_change_is_filled_100_percent),
        ("flag source: request OR persisted", test_flag_comes_from_either_the_request_or_the_project),
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
