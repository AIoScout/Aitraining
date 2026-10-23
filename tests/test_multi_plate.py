"""Gatekeeper whitelist: multi-plate classes, preview and export.

A Car License project may hold one or more allowed 3-character plates.  The
class list is the union of their characters (first appearance), Preview says
YES when the reading matches ANY plate, and the export carries the full table
for the MCU.  These tests pin all three contracts, plus the EMNIST backfill
of newly introduced characters.

Requires the AItraining venv (TensorFlow):
    .venv/bin/python -m pytest tests/test_multi_plate.py -q
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, Path(__file__).resolve().parent):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import record_controller as rc
from test_emnist_seeder import _open_template, _workspace

PLATES = ["A12", "B18"]


def _predict(ctrl, png: bytes) -> dict:
    return ctrl._preview_predict_sign_ocr(
        "s", png, sign_sequence=PLATES[0], sign_sequences=PLATES,
        reading_order="ltr", bg_dark_thresh=0, bg_lum_thresh=60,
    )


def _synth_until_match(ctrl, plate_index: int):
    """Draw the requested plate until the model reads it (the UI retries too)."""
    last = None
    for attempt in range(14):
        syn = ctrl._synthesize_sign_preview_png(
            "s", variant="match", sign_sequences=PLATES,
            plate_index=plate_index, seed=attempt,
        )
        pred = _predict(ctrl, syn["png"])
        last = (syn, pred)
        if pred.get("match") is True and pred.get("matched_sequence") == PLATES[plate_index]:
            return syn, pred
    return last


def test_multi_plate_union_autofill_preview_and_export():
    with tempfile.TemporaryDirectory() as td:
        ctrl, dataset = _workspace(Path(td))
        state = _open_template(ctrl)
        ctrl._save_train_cfg("s", {"auto_seed_emnist_samples": True})

        # The class list is the union of the whitelist, first-appearance order.
        ctrl._classes_sync_ocr("s", ["A", "1", "2", "B", "8"])
        assert ctrl._classes_load(dataset) == ["A", "1", "2", "B", "8"]

        # Newly introduced classes start empty; the auto-fill pass (the same
        # one /classes/sync_ocr calls) draws 24 EMNIST captures for each.
        seeded = ctrl._seed_empty_char_classes("s", sign_sequence="A12", strict=True)
        assert set(seeded) == {"B", "8"}, seeded
        for ch in ("A", "1", "2", "B", "8"):
            count = len(list((dataset / ch).glob("*.png")))
            assert count >= 24, (ch, count)

        cfg = dict(state["train_cfg"])
        cfg.update(
            {
                "sign_sequence": PLATES[0],
                "sign_sequences": PLATES,
                "ocr_background_enabled": False,
                "sign_background_replace_ood": False,
                "ocr_emnist_enabled": True,
                "ocr_emnist_per_class": 200,
            }
        )
        ctrl._train_worker("s", cfg)
        status = ctrl._train.get("s") or {}
        assert not status.get("error"), status
        meta = json.loads(ctrl._train_result_path(dataset).read_text())
        assert set(meta["labels"]) == {"A", "1", "2", "B", "8"}, meta["labels"]
        assert meta.get("sign_sequences") == PLATES, meta.get("sign_sequences")
        assert not any(str(x).startswith("Class ") for x in meta["labels"])

        # Both whitelisted plates read YES; the matched plate is identified.
        for idx, plate in enumerate(PLATES):
            syn, pred = _synth_until_match(ctrl, idx)
            assert pred.get("match") is True, pred
            assert pred.get("matched_sequence") == plate, pred
            assert pred.get("matched_index") == idx, pred
            assert f"Plate #{idx + 1} ({plate})" in str(pred.get("match_text")), pred
            assert [s.get("label") for s in pred["slots"]] == list(plate), pred["slots"]

        # A18 is not whitelisted even though A,1 and 8 are all classes.
        bad = ctrl._synthesize_sign_preview_png(
            "s", variant="mismatch", sign_sequences=PLATES,
            plate_index=0, alt_char="8", seed=0,
        )
        pred = _predict(ctrl, bad["png"])
        assert pred.get("match") is False, pred
        assert pred.get("closest_sequence") == "A12", pred
        assert [s.get("match") for s in pred["slots"]] == [True, True, False], pred["slots"]

        # C99: characters the model knows but no allowed plate contains.
        other = ctrl._synthesize_sign_preview_png(
            "s", variant="match", chars=["C", "9", "9"], reading_order="ltr", seed=0,
        )
        pred = _predict(ctrl, other["png"])
        assert pred.get("match") is False, pred
        assert pred.get("matched_sequence") == "", pred

        export_dir = dataset.parent / "export_multi"
        ctrl._export_run("s", export_dir, "tm_sign", "g_model", overwrite=True)
        header = (export_dir / "model_settings.h").read_text(encoding="utf-8")
        assert "#define TM_SIGN_MULTI_TARGET_EXPORTED 1" in header, header
        assert "constexpr int kTargetSignCount = 2;" in header, header
        assert (
            'constexpr char kTargetSignSequences[kTargetSignCount][4] = { "A12", "B18" };'
            in header
        ), header
        assert "constexpr int kTargetSignClassMatrix[kTargetSignCount][3] = {" in header, header
        assert "    { 0, 1, 2 }," in header, header
        assert "    { 3, 1, 4 }," in header, header
        # Backward-compatible primary target.
        assert 'constexpr char kTargetSignSequence[] = "A12";' in header, header
        assert "constexpr int kTargetSignClassIndices[3] = {0, 1, 2};" in header, header


def test_normalize_sign_sequences_matches_the_plates_contract():
    from ocr_preprocess import normalize_sign_sequences

    assert normalize_sign_sequences(["A12", "B18", "A12"]) == ["A12", "B18"]
    assert normalize_sign_sequences("A12") == ["A12"]
    assert normalize_sign_sequences(["A", "1", "2"]) == ["A12"]
    assert normalize_sign_sequences(["A1", "???"]) == []
    assert normalize_sign_sequences([["a", "1", "2"], ["b", "1", "8"]]) == ["A12", "B18"]


if __name__ == "__main__":
    raise SystemExit(__import__("pytest").main([__file__, "-q"]))
