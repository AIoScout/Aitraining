"""Automated checks for the `sign_ocr` (3-character road sign) project type.

Covers the Python half of the feature end to end:

  registry        dataset_io knows the type and treats it as a glyph project
  sequence        normalisation is total, idempotent, and shared by both sides
  splitting       three glyphs in, three boxes out — including the cases where
                  the projection disagrees with the label length
  reading order   "rtl" reverses the CROPS, never the glyphs or the labels
  predictor       _preview_predict_sign_ocr's YES/NO rule, slot by slot
  train config    the sequence round-trips through pick_train_cfg
  routing         .tmproj classification and the Open-Project whitelist

Run with:
    .venv/bin/python -m pytest tests/test_sign_ocr.py -q
or, with no test runner installed:
    .venv/bin/python tests/test_sign_ocr.py
"""
from __future__ import annotations

import sys
import threading
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import dataset_io
import ocr_preprocess
import record_controller
import trainer


# ── fixture: a synthetic 96x96 sign with three glyphs ──────────────────────
def glyph_block(img, x1, y1, x2, y2, ink, stroke=4):
    """A hollow rectangular glyph: outline only, paper in the middle.

    Deliberately NOT a solid bar.  The split returns boxes that HUG the ink,
    so a solid bar would hand `preprocess_glyph` a crop of one uniform value —
    Otsu on a flat crop has no split point, the slot reads empty, and the test
    would be exercising a degenerate case no real glyph produces.  An outline
    keeps the interior mixed (stroke vs paper), which is what a captured
    character actually looks like.
    """
    img[y1:y1 + stroke, x1:x2] = ink
    img[y2 - stroke:y2, x1:x2] = ink
    img[y1:y2, x1:x1 + stroke] = ink
    img[y1:y2, x2 - stroke:x2] = ink


def sign_frame(widths=(11, 11, 11), gap=14, height=96, ink=40, paper=230):
    """White frame, three hollow glyph blocks — one slot each.

    The blocks are outlined at known coordinates so a test can assert the
    split recovered exactly what was drawn rather than whatever the fallback
    happened to produce.
    """
    img = np.full((height, height), paper, dtype=np.uint8)
    x = 10
    boxes = []
    for w in widths:
        glyph_block(img, x, 30, x + w, 70, ink)
        boxes.append((x, 30, x + w, 70))
        x += w + gap
    return img, boxes


# ── registry ───────────────────────────────────────────────────────────────
def test_registry_treats_sign_ocr_as_a_glyph_project():
    assert dataset_io.PROJECT_TYPE_SIGN_OCR == "sign_ocr"
    assert dataset_io.PROJECT_TYPE_SIGN_OCR in dataset_io.KNOWN_PROJECT_TYPES
    # The whole point of the family predicate: the sample dispatch and the
    # export's device-sketch choice both ask this, so a third glyph type
    # cannot miss half the sites.
    assert dataset_io.is_glyph_project("sign_ocr") is True
    assert dataset_io.is_glyph_project("SIGN_OCR") is True
    assert dataset_io.is_glyph_project("ocr") is True
    assert dataset_io.is_glyph_project("image") is False
    assert dataset_io.is_glyph_project(None) is False


def test_sign_ocr_uses_the_28px_glyph_geometry():
    assert dataset_io.default_img_size("sign_ocr") == 28
    assert trainer.get_train_strategy("sign_ocr") is trainer.get_train_strategy("ocr")
    assert trainer.get_train_strategy("sign_ocr") is not trainer.get_train_strategy("image")


def test_train_config_carries_the_sequence():
    fields = {f.name for f in __import__("dataclasses").fields(trainer.TrainConfig)}
    assert "sign_sequence" in fields
    # A sign project must not inherit an image project's input size.
    assert trainer.TrainConfig(img_size=28, sign_sequence=["A", "1", "A"]).sign_sequence == ["A", "1", "A"]


# ── sequence normalisation ─────────────────────────────────────────────────
def test_normalize_sequence_is_total_and_idempotent():
    n = ocr_preprocess.normalize_sign_sequence
    assert n("a1a") == ["A", "1", "A"]
    assert n("A 1 A") == ["A", "1", "A"]
    assert n(["a", "1", "a"]) == ["A", "1", "A"]
    assert n(["ab", "cd", "ef"]) == ["A", "C", "E"]     # one glyph per slot
    assert n("AB") == ["A", "B", "?"]                    # padded
    assert n("ABCD") == ["A", "B", "C"]                  # truncated
    assert n(None) == ["?", "?", "?"]
    assert n([]) == ["?", "?", "?"]
    assert n("") == ["?", "?", "?"]
    assert n(["", None, " "]) == ["?", "?", "?"]
    for value in ("a1a", "AB", "ABCD", None, ["", "x", ""]):
        once = n(value)
        assert n(once) == once, f"not idempotent for {value!r}"


# ── glyph splitting ───────────────────────────────────────────────────────
def test_split_recovers_three_real_glyphs_verbatim():
    img, drawn = sign_frame()
    ink = ocr_preprocess.binarize_ink(img, 128)
    boxes = ocr_preprocess.split_sign_glyph_boxes(ink, count=3)
    assert boxes == drawn, f"expected the drawn bars {drawn}, got {boxes}"


def test_split_merges_a_speck_rather_than_fusing_two_glyphs():
    # Three glyphs plus a 2px speck at the far right.  The narrowest-run rule
    # must consume the SPECK; the earlier smallest-gap rule fused glyphs 1+2
    # and kept the speck, which is the failure this test exists for.
    img, _ = sign_frame()
    img[20:24, 90:92] = 40
    ink = ocr_preprocess.binarize_ink(img, 128)
    boxes = ocr_preprocess.split_sign_glyph_boxes(ink, count=3)
    assert len(boxes) == 3
    # The speck's column is inside the last box, and the first two bars are
    # each still their own box.
    assert boxes[0][0] == 10 and boxes[0][2] == 21
    assert boxes[1][0] == 35 and boxes[1][2] == 46
    assert boxes[2][0] == 60 and boxes[2][2] >= 92


def test_split_splits_the_widest_box_when_a_glyph_touches_its_neighbour():
    img, _ = sign_frame()
    img[30:70, 21:35] = 40          # bridge bars 1 and 2 into one run
    ink = ocr_preprocess.binarize_ink(img, 128)
    boxes = ocr_preprocess.split_sign_glyph_boxes(ink, count=3)
    assert len(boxes) == 3
    # The merged run is the widest, so it is the one that gets cut.
    assert boxes[0][0] == 10
    assert boxes[0][2] == boxes[1][0]          # tiled, no gap
    assert boxes[2][0] == 60


def test_split_falls_back_to_equal_thirds_when_there_is_no_ink():
    ink = np.zeros((96, 96), dtype=bool)
    boxes = ocr_preprocess.split_sign_glyph_boxes(ink, count=3)
    assert len(boxes) == 3
    assert boxes[0][0] == 0 and boxes[-1][2] == 96
    # Tiled exactly: each box starts where the previous one ended.
    for a, b in zip(boxes, boxes[1:]):
        assert a[2] == b[0]


# ── reading order ─────────────────────────────────────────────────────────
def test_rtl_reverses_the_crops_not_the_labels():
    img, drawn = sign_frame()
    ltr = ocr_preprocess.preprocess_sign_sequence(img, "A1B", reading_order="ltr")
    rtl = ocr_preprocess.preprocess_sign_sequence(img, "A1B", reading_order="rtl")
    assert [s["char"] for s in ltr["slots"]] == ["A", "1", "B"]
    assert [s["char"] for s in rtl["slots"]] == ["A", "1", "B"]
    # Same chars, opposite crops: slot 0 of RTL is slot 2 of LTR.
    assert ltr["slots"][0]["box"] == rtl["slots"][2]["box"]
    assert ltr["slots"][2]["box"] == rtl["slots"][0]["box"]
    # box_norm agrees with the pixel boxes, in the same order.
    assert rtl["slots"][0]["box_norm"][0] == ltr["slots"][2]["box_norm"][0]


def test_rtl_does_not_mirror_the_glyph():
    # A glyph that is not left-right symmetric: a left bar only.  Under RTL it
    # must still come out with its ink on the LEFT — reversing the crop
    # sequence must never reverse the image, which would hand the model a
    # mirrored character it was never trained on.
    img = np.full((96, 96), 230, dtype=np.uint8)
    # Three glyphs of different heights, so every slot's glyph is distinct and
    # "the same crop" cannot be satisfied by two blank glyphs agreeing.
    glyph_block(img, 8, 20, 30, 76, 40)
    glyph_block(img, 40, 30, 62, 66, 40)
    glyph_block(img, 72, 24, 94, 70, 40)
    ltr = ocr_preprocess.preprocess_sign_sequence(img, "A1B", reading_order="ltr")
    rtl = ocr_preprocess.preprocess_sign_sequence(img, "A1B", reading_order="rtl")
    # Slot 0 LTR is the left crop; slot 2 RTL is that same crop.  Identical
    # pixels means the crop was moved, not mirrored.
    assert np.array_equal(ltr["slots"][0]["glyph"], rtl["slots"][2]["glyph"])
    assert np.array_equal(ltr["slots"][2]["glyph"], rtl["slots"][0]["glyph"])
    # And the crops are genuinely different from each other, so the equality
    # above is not two blank glyphs agreeing.
    assert not np.array_equal(ltr["slots"][0]["glyph"], ltr["slots"][2]["glyph"])


def test_sequence_always_yields_one_slot_per_character():
    img, _ = sign_frame()
    out = ocr_preprocess.preprocess_sign_sequence(img, "?1?")
    assert len(out["slots"]) == 3
    assert len(out["glyphs"]) == 3
    for g in out["glyphs"]:
        assert g.shape == (28, 28)
    # A slot holding the pad is still a slot; it is the predictor that refuses
    # to call it a match.
    assert [s["char"] for s in out["slots"]] == ["?", "1", "?"]


# ── train-config coercion ─────────────────────────────────────────────────
def test_pick_train_cfg_coerces_the_sequence():
    ok, value = record_controller._coerce_train_cfg_value("sign_sequence", "a1a")
    assert ok is True and value == ["A", "1", "A"]
    ok, value = record_controller._coerce_train_cfg_value("sign_sequence", ["a", "1", "a"])
    assert ok is True and value == ["A", "1", "A"]
    # An all-pad sequence is "not set": dropping it lets the archive keep the
    # previous target instead of overwriting it with three question marks.
    assert record_controller._coerce_train_cfg_value("sign_sequence", "???") == (False, None)
    assert record_controller._coerce_train_cfg_value("sign_sequence", None) == (False, None)


def test_pick_train_cfg_round_trips_a_project_payload():
    cfg = record_controller.pick_train_cfg(
        {
            "img_size": 28,
            "project_type": "sign_ocr",
            "sign_sequence": "A1A",
            "reading_order": "rtl",
        }
    )
    assert isinstance(cfg, dict)
    assert cfg["sign_sequence"] == ["A", "1", "A"]
    assert cfg["reading_order"] == "rtl"
    assert cfg["img_size"] == 28
    # Keys the blob did not carry are absent, so the caller's fallback chain
    # still sees "this source had nothing to say" for them.
    assert "preprocess_mode" not in cfg


def test_payload_extras_are_absent_for_other_projects():
    assert record_controller.sign_ocr_payload_extras({"labels": [], "probs": []}) == {}
    assert record_controller.sign_ocr_payload_extras(None) == {}
    pred = {
        "sign_ocr": True,
        "sign_sequence": ["A", "1", "A"],
        "reading_order": "ltr",
        "slots": [{"char": "A"}],
        "slot_images_b64": ["x"],
        "match": True,
        "match_text": record_controller.SIGN_MATCH_TEXT,
        "match_min_prob": 0.6,
        "sign_box": [1, 2, 3, 4],
        "labels": ["A → A"],
        "probs": [0.9],
    }
    extras = record_controller.sign_ocr_payload_extras(pred)
    assert set(extras) == set(record_controller._SIGN_OCR_PAYLOAD_KEYS)
    assert extras["match"] is True
    # The generic fields are the caller's job, not this helper's.
    assert "labels" not in extras and "probs" not in extras


# ── the predictor (no model, no server: both are stubbed) ─────────────────
LABELS = ["A", "1", "B", "_background"]


class FakeController:
    """The parts of RecordController the sign predictor actually touches.

    Built with object.__new__ so no HTTP server, capture thread or model
    runtime is ever started, and with the two model-facing methods replaced:
    `_preview_load_model` returns a dict describing a model, and
    `_invoke_glyph_tensor` answers with whatever raw int8-mirror tensor the
    test asks for.  Everything between them — the split, the per-slot match
    rule, the payload — is the real code.
    """

    def __init__(self, raw_by_slot, labels=LABELS, sequence=("A", "1", "A")):
        import types

        self.raw_by_slot = list(raw_by_slot)
        self.calls = 0
        self._lock = threading.RLock()
        self._configs = {}
        self._preview_locks = {}
        self._preview_load_model = types.MethodType(
            lambda _self, session_id=None, **kw: {
                "labels": list(labels),
                "input_details": {"shape": [1, 28, 28, 1], "quantization": (1 / 255, -128)},
                "output_details": {"quantization": (1 / 256, 0)},
                "class_preprocess": {},
                "manual_roi": None,
                "preprocess_mode": "auto_by_label",
                "model_source": "latest",
            },
            self,
        )
        self._invoke_glyph_tensor = types.MethodType(
            lambda _self, model, session_id, arr: (None, self._next_raw()),
            self,
        )
        # These two touch nothing but their arguments, so the REAL ones are
        # used: the preprocessing-record choice and the composite image are
        # part of what the test is checking, not scaffolding around it.
        for name in ("_sign_ocr_preprocess_config", "_sign_ocr_composite"):
            setattr(
                self,
                name,
                types.MethodType(getattr(record_controller.RecordController, name), self),
            )

    def _next_raw(self):
        raw = self.raw_by_slot[min(self.calls, len(self.raw_by_slot) - 1)]
        self.calls += 1
        return np.asarray(raw, dtype=np.int32)


MULTI_LABELS = ["A", "1", "2", "B", "8", "_background"]


def _raw_for(index, best=200, rest=10, n_classes=None):
    """A raw int8-mirror tensor whose argmax is `index`."""
    out = [rest] * int(n_classes or len(LABELS))
    out[index] = best
    return out


def _raw_for_multi(index, best=200, rest=10):
    return _raw_for(index, best=best, rest=rest, n_classes=len(MULTI_LABELS))


def _predict(controller, sequence="A1A", order="ltr", **kw):
    img, _ = sign_frame()
    png = _png(img)
    return record_controller.RecordController._preview_predict_sign_ocr(
        controller, session_id="s", png=png, sign_sequence=sequence, reading_order=order, **kw
    )


def _png(gray: np.ndarray) -> bytes:
    from io import BytesIO

    from PIL import Image

    buf = BytesIO()
    Image.fromarray(np.asarray(gray, dtype=np.uint8), mode="L").save(buf, format="PNG")
    return buf.getvalue()


def test_predictor_says_yes_only_when_all_three_match():
    ctrl = FakeController([_raw_for(0), _raw_for(1), _raw_for(0)])
    pred = _predict(ctrl)
    assert pred["sign_ocr"] is True
    assert pred["match"] is True
    assert pred["top_label"].startswith(record_controller.SIGN_MATCH_TEXT)
    assert [s["char"] for s in pred["slots"]] == ["A", "1", "A"]
    assert [s["label"] for s in pred["slots"]] == ["A", "1", "A"]
    assert all(s["match"] for s in pred["slots"])
    # The sequence survives to the payload in LABEL order, not crop order.
    assert pred["sign_sequence"] == ["A", "1", "A"]
    assert len(pred["slot_images_b64"]) == 3


def test_predictor_says_no_when_one_slot_reads_the_wrong_character():
    ctrl = FakeController([_raw_for(0), _raw_for(1), _raw_for(2)])   # last is 'B'
    pred = _predict(ctrl)
    assert pred["match"] is False
    assert pred["top_label"] == record_controller.SIGN_NOMATCH_TEXT
    assert [s["match"] for s in pred["slots"]] == [True, True, False]


def test_predictor_treats_the_reject_class_as_a_miss():
    ctrl = FakeController([_raw_for(0), _raw_for(1), _raw_for(3)])   # _background
    pred = _predict(ctrl)
    assert pred["match"] is False
    assert pred["slots"][2]["background"] is True
    assert pred["slots"][2]["match"] is False


def test_predictor_requires_each_slot_to_clear_the_confidence_floor():
    # argmax is right, but only 60 of 180 total -> 0.33, below the 0.60 floor.
    weak = _raw_for(0, best=60, rest=40)
    ctrl = FakeController([weak, _raw_for(1), _raw_for(0)])
    pred = _predict(ctrl)
    assert pred["match"] is False
    assert pred["slots"][0]["label"] == "A"
    assert abs(pred["slots"][0]["prob"] - 60 / 180) < 1e-6
    # Same model, a floor the reading clears -> YES.  The threshold is what
    # decides, and it is applied per slot.
    ctrl2 = FakeController([weak, _raw_for(1), _raw_for(0)])
    assert _predict(ctrl2, match_min_prob=0.30)["match"] is True


def test_predictor_accepts_any_allowed_plate():
    # Slot readings A,1,2 -> the first plate in the whitelist.
    ctrl = FakeController([_raw_for_multi(0), _raw_for_multi(1), _raw_for_multi(2)], labels=MULTI_LABELS)
    pred = _predict(ctrl, sequence="A12", sign_sequences=["A12", "B18"])
    assert pred["match"] is True, pred
    assert pred["matched_sequence"] == "A12"
    assert pred["matched_index"] == 0
    assert pred["plate_count"] == 2
    assert "Plate #1 (A12)" in pred["match_text"], pred["match_text"]
    assert [s["char"] for s in pred["slots"]] == ["A", "1", "2"]
    assert all(s["match"] for s in pred["slots"])

    # Slot readings B,1,8 -> the second plate.
    ctrl = FakeController([_raw_for_multi(3), _raw_for_multi(1), _raw_for_multi(4)], labels=MULTI_LABELS)
    pred = _predict(ctrl, sequence="A12", sign_sequences=["A12", "B18"])
    assert pred["match"] is True, pred
    assert pred["matched_sequence"] == "B18"
    assert pred["matched_index"] == 1
    assert "Plate #2 (B18)" in pred["match_text"], pred["match_text"]
    assert [s["char"] for s in pred["slots"]] == ["B", "1", "8"]


def test_predictor_reports_the_closest_plate_when_none_match():
    # A,1,8 is two slots of A12 and two of B18 (tie -> first).  The returned
    # slot headers must follow the closest plate so the failing slot is
    # explainable; the verdict is still NO.
    ctrl = FakeController([_raw_for_multi(0), _raw_for_multi(1), _raw_for_multi(4)], labels=MULTI_LABELS)
    pred = _predict(ctrl, sequence="A12", sign_sequences=["A12", "B18"])
    assert pred["match"] is False, pred
    assert pred["closest_sequence"] == "A12", pred
    assert pred["matched_sequence"] == ""
    assert [s["char"] for s in pred["slots"]] == ["A", "1", "2"]
    assert [s["match"] for s in pred["slots"]] == [True, True, False]


def test_predictor_reports_no_match_without_a_trained_model():
    ctrl = FakeController([_raw_for(0)] * 3, labels=[])
    pred = _predict(ctrl)
    assert pred["match"] is False
    # Still answers with the split — the crops are the useful part before
    # there is a model to read them.
    assert len(pred["slots"]) == 3
    assert all(s["label"] == "" for s in pred["slots"])


def test_predictor_keeps_labels_in_reading_order_under_rtl():
    # Distinct raw answers per call, and a symmetric sequence, so the only
    # thing that changes is which CROP each slot came from.
    ctrl = FakeController([_raw_for(0), _raw_for(1), _raw_for(0)])
    ltr = _predict(ctrl, sequence="A1A", order="ltr")
    ctrl2 = FakeController([_raw_for(0), _raw_for(1), _raw_for(0)])
    rtl = _predict(ctrl2, sequence="A1A", order="rtl")
    assert [s["char"] for s in rtl["slots"]] == ["A", "1", "A"]
    # The crop boxes are mirrored, the label order is not.
    assert ltr["slots"][0]["box"] == rtl["slots"][2]["box"]
    assert ltr["slots"][2]["box"] == rtl["slots"][0]["box"]


def test_predictor_boxes_are_full_frame_even_under_manual_roi():
    # With a Manual ROI the reading is taken on a sub-image; the boxes the SPA
    # draws must be in FULL-frame pixels or the overlay lands in the wrong
    # place.  Pin the manual ROI to the auto ROI's own pixels: the same crop
    # must produce the same full-frame slot boxes.
    ctrl = FakeController([_raw_for(0), _raw_for(1), _raw_for(0)])
    plain = _predict(ctrl)
    ctrl2 = FakeController([_raw_for(0), _raw_for(1), _raw_for(0)])
    ctrl2._preview_load_model = _with_roi(ctrl2, tuple(plain["roi_box_norm"]))
    roi = _predict(ctrl2)
    assert roi["roi_box"] == plain["roi_box"], (roi["roi_box"], plain["roi_box"])
    assert [s["box"] for s in roi["slots"]] == [s["box"] for s in plain["slots"]]
    assert roi["crop"] == plain["crop"]


def _with_roi(ctrl, roi):
    import types

    return types.MethodType(
        lambda _self, session_id=None, **kw: {
            "labels": list(LABELS),
            "input_details": {"shape": [1, 28, 28, 1], "quantization": (1 / 255, -128)},
            "output_details": {"quantization": (1 / 256, 0)},
            "class_preprocess": {},
            "manual_roi": roi,
            "preprocess_mode": "manual_roi",
            "model_source": "latest",
        },
        ctrl,
    )


# ── class list follows the sequence ───────────────────────────────────────
def test_classes_sync_ocr_rewrites_the_class_list():
    """The classes ARE the three slots: keep, add, reorder and prune in one call."""
    import json
    import tempfile
    from types import SimpleNamespace

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        dataset = root / "dataset"
        dataset.mkdir()
        for name in ("N", "A", "Y"):
            (dataset / name).mkdir()
            (dataset / name / "000000.png").write_bytes(b"sample")

        ctrl = object.__new__(record_controller.RecordController)
        ctrl._lock = threading.RLock()
        ctrl._configs = {"s": SimpleNamespace(dataset_root=dataset)}
        record_controller.RecordController._classes_save(
            ctrl, dataset, ["N", "A", "Y"]
        )

        # Repeated characters are ONE class; unlisted classes leave with their
        # directory; the kept character keeps its sample.
        assert ctrl._classes_sync_ocr("s", ["A", "1", "A"]) == ["A", "1"]
        meta = json.loads((root / "tm_classes.json").read_text(encoding="utf-8"))
        assert meta["classes"] == ["A", "1"]
        assert (dataset / "A" / "000000.png").exists()
        assert not (dataset / "N").exists()
        assert not (dataset / "Y").exists()

        # Idempotent, and it matches on the sanitized name (case-insensitive):
        # that is what collides on disk, so it must not become a second class.
        assert ctrl._classes_sync_ocr("s", ["a", "1"]) == ["A", "1"]

        # A route guard, not a truncation: a multi-character "slot" would be a
        # class no single-glyph model output could ever match.
        try:
            ctrl._classes_sync_ocr("s", ["AB"])
        except ValueError as exc:
            assert "one character" in str(exc)
        else:
            raise AssertionError("multi-character slot was accepted")
        try:
            ctrl._classes_sync_ocr("s", ["?"])
        except ValueError as exc:
            assert "Not a sign character" in str(exc)
        else:
            raise AssertionError("pad character was accepted")


# ── project routing ───────────────────────────────────────────────────────
def test_tmproj_detection_and_open_whitelist():
    import app

    p = Path("sign_ocr_20260101.tmproj")
    # Manifest is authoritative.
    assert app._tmproj_detect_project_type(p, {"project_type": "sign_ocr"}) == "sign_ocr"
    # Filename fallback: "sign" must be tested before "ocr", or a legacy
    # "sign_ocr_*.tmproj" (which contains both words) comes back as plain OCR.
    assert app._tmproj_detect_project_type(p, {}) == "sign_ocr"
    assert app._tmproj_detect_project_type(Path("ocr_x.tmproj"), {}) == "ocr"
    # The Open-Project gate accepts every type main() can route to.
    assert "sign_ocr" in app._TM_OPENABLE_PROJECT_TYPES
    assert "ocr" in app._TM_OPENABLE_PROJECT_TYPES
    assert "image" in app._TM_OPENABLE_PROJECT_TYPES


# ── runner ────────────────────────────────────────────────────────────────
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
