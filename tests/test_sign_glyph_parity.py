"""C-vs-Python parity for the sign-sequence glyph front-end.

`TFLite_Sign/sign_glyph.c` is the device's C mirror of
`AItraining/ocr_preprocess.py`.  A model trained through the Python transform
sees a distribution the device can only reproduce if the two agree EXACTLY:
one pixel in a box or one LSB in a resampled value is a different input.
This suite compiles the C reference and compares, for every case:

  sequence   preprocess_sign_sequence_c  vs  preprocess_sign_sequence
             (frame threshold, raw run count, sign box, per-slot box/ok/empty,
              and every glyph pixel)
  split      split_sign_glyph_boxes_c    vs  split_sign_glyph_boxes
  single     sign_preprocess_glyph       vs  preprocess_glyph
  strided    sign_otsu_threshold_strided vs  otsu_threshold_u8 on the crop
  luminance  sign_rgb_to_lum_u8          vs  rgb_to_lum_u8

The corpus is seeded, so a failure is reproducible.  It deliberately includes
the three reconciliation paths of the split (merge, split, equal thirds), RTL,
INK_BRIGHT, pinned thresholds, the Dark Thr floor, segmentation off, and
degenerate/tiny frames — the branches that are easy to get subtly wrong.

Run:
    .venv/bin/python tests/test_sign_glyph_parity.py
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import ocr_preprocess as ocr  # noqa: E402

SOURCE = REPO / "TFLite_Sign" / "sign_glyph.c"
HARNESS = REPO / "TFLite_Sign" / "tools" / "sign_glyph_host.c"
HEADER = ["name", "mode", "w", "h", "out_size", "count", "order", "polarity",
          "seg", "min_run", "thresh", "dark", "conv", "x0", "y0"]
SWEEP = 40


# ── corpus ──────────────────────────────────────────────────────────────────
def _blocks(w, h, rects, paper=230):
    img = np.full((h, w), paper, np.uint8)
    for (x1, y1, x2, y2, v) in rects:
        img[y1:y2, x1:x2] = v
    return img


def _cases():
    rng = np.random.default_rng(20261017)
    cases = []

    def add(mode, name, data, **kw):
        c = dict(mode=mode, name=name, w=data.shape[1], h=data.shape[0],
                 out_size=28, count=3, order=0, polarity=0, seg=1, min_run=2,
                 thresh=-1, dark=0, conv=1, x0=0, y0=0)
        c.update(kw)
        cases.append((c, np.ascontiguousarray(data)))

    def seq(name, gray, **kw):
        add(0, name, gray, **kw)

    def split(name, ink, **kw):
        add(1, name, (ink != 0).astype(np.uint8), **kw)

    base = _blocks(96, 96, [(10, 30, 20, 70, 40), (30, 25, 48, 72, 40),
                            (60, 32, 76, 68, 40)])
    seq("three_clean", base)
    seq("three_rtl", base, order=1)
    seq("three_bright", base, polarity=1)
    seq("three_stretch", base, conv=0)
    seq("three_pinned_200", base, thresh=200)
    seq("three_pinned_0", base, thresh=0)
    seq("three_pinned_255", base, thresh=255)
    seq("three_dark_30", base, dark=30)
    seq("three_dark_60", base, dark=60)
    seq("three_dark_300", base, dark=300)
    seq("three_threshold_window", base, thresh=200, dark=60)
    seq("three_noseg", base, seg=0)

    # > count runs: specks and a detached stroke -> merge path.
    speck = base.copy()
    speck[5:7, 90:92] = 0
    speck[80:82, 5:7] = 0
    seq("many_specks", speck)
    seq("broken_stroke", _blocks(
        96, 96, [(10, 30, 14, 70, 40), (16, 30, 20, 70, 40),
                 (30, 25, 48, 72, 40), (60, 32, 76, 68, 40)]))
    # < count runs: touching glyphs -> split path.
    seq("one_wide_block", _blocks(96, 96, [(10, 30, 80, 70, 40)]))
    seq("one_wide_block_rtl", _blocks(96, 96, [(10, 30, 80, 70, 40)]), order=1)
    two = _blocks(96, 96, [(10, 30, 70, 70, 40)])
    two[30:70, 35:37] = 230      # a bright gap through the block -> 2 runs
    seq("two_runs_gap", two)

    # no ink at all -> equal thirds.
    seq("flat_128", np.full((96, 96), 128, np.uint8))
    seq("flat_0", np.zeros((96, 96), np.uint8))
    seq("flat_255", np.full((96, 96), 255, np.uint8))
    seq("all_black_pinned", np.zeros((96, 96), np.uint8), thresh=128)
    seq("all_ink_pinned255", base, thresh=255, seg=0)

    for (w, h) in [(1, 1), (2, 5), (5, 2), (7, 7), (16, 3), (3, 16)]:
        g = np.full((h, w), 200, np.uint8)
        g[:, 0] = 20
        seq(f"tiny_{w}x{h}", g)

    for (w, h) in [(64, 64), (100, 70), (33, 97), (128, 48)]:
        seq(f"geom_{w}x{h}", _blocks(
            w, h, [(2, 5, max(3, w // 5), h - 5, 30),
                   (w // 2, 3, w // 2 + 7, h - 3, 30),
                   (w - 12, 7, w - 3, h - 7, 30)], paper=220))

    # Narrow single-stroke glyphs: the hairline '1' path.  Python widens
    # these under NORM_CANON (ocr_preprocess._narrow_stroke_radius) and the
    # C port must produce the same widened canvas.  Cover segmentation on
    # (training capture), segmentation off (padded sign slot) and both
    # polarities.
    for (w, h, x, bw) in [(96, 96, 48, 1), (96, 96, 47, 2),
                          (160, 160, 80, 2), (96, 96, 10, 3)]:
        img = np.full((h, w), 230, np.uint8)
        img[20:h - 20, x:x + bw] = 20
        seq(f"narrow_stroke_{w}x{h}_{x}_{bw}", img, thresh=100)
        add(3, f"single_narrow_{w}x{h}_{x}_{bw}", img, thresh=100,
            polarity=0, seg=1, conv=1)
        add(3, f"single_narrow_noseg_{w}x{h}_{x}_{bw}", img, thresh=100,
            polarity=0, seg=0, conv=1)
        # INK_BRIGHT: the same stroke, inverted.  Exercises the max-filter
        # branch and the dark (0) plate fill.
        bright = np.full((h, w), 20, np.uint8)
        bright[20:h - 20, x:x + bw] = 230
        seq(f"narrow_bright_{w}x{h}_{x}_{bw}", bright, thresh=100,
            polarity=1)
        add(3, f"single_narrow_bright_{w}x{h}_{x}_{bw}", bright,
            thresh=100, polarity=1, seg=1, conv=1)
    for (w, h) in [(96, 96), (160, 160)]:
        img = np.full((h, w), 230, np.uint8)
        for x in (w // 4, w // 2, 3 * w // 4):
            img[20:h - 20, x:x + 2] = 20
        seq(f"three_narrow_strokes_{w}", img, thresh=100)

    for i in range(SWEEP):
        w = int(rng.integers(40, 130))
        h = int(rng.integers(40, 130))
        rects = []
        for _ in range(int(rng.integers(1, 7))):
            x1 = int(rng.integers(0, max(1, w - 2)))
            x2 = int(rng.integers(x1 + 1, w + 1))
            y1 = int(rng.integers(0, max(1, h - 2)))
            y2 = int(rng.integers(y1 + 1, h + 1))
            rects.append((x1, y1, x2, y2, int(rng.integers(0, 90))))
        seq(f"rand_rect_{i}", _blocks(w, h, rects,
                                      paper=int(rng.integers(150, 256))),
            order=int(rng.integers(0, 2)),
            polarity=int(rng.integers(0, 2)),
            thresh=int(rng.choice([-1, -1, 100, 200])),
            dark=int(rng.choice([0, 0, 30, 80])))

    for i in range(SWEEP // 2):
        w = int(rng.integers(32, 100))
        h = int(rng.integers(32, 100))
        seq(f"noise_{i}", rng.integers(0, 256, (h, w), dtype=np.uint8))

    for i in range(SWEEP):
        w = int(rng.integers(1, 120))
        h = int(rng.integers(1, 60))
        density = float(rng.choice([0.0, 0.02, 0.1, 0.4, 0.9, 1.0]))
        split(f"mask_{i}", rng.random((h, w)) < density,
              count=int(rng.choice([1, 2, 3, 4])),
              min_run=int(rng.choice([1, 2, 3])))
    split("mask_empty", np.zeros((10, 10), bool))
    split("mask_full", np.ones((10, 10), bool))
    split("mask_one_col", np.ones((10, 1), bool))

    for i in range(SWEEP):
        w = int(rng.integers(8, 100))
        h = int(rng.integers(8, 100))
        add(3, f"single_noise_{i}", rng.integers(0, 256, (h, w),
                                                 dtype=np.uint8),
            polarity=int(rng.integers(0, 2)), seg=int(rng.integers(0, 2)),
            thresh=int(rng.choice([-1, 0, 128, 255])),
            dark=int(rng.choice([0, 0, 40])), conv=int(rng.integers(0, 2)))
    add(3, "single_flat", np.full((20, 20), 200, np.uint8))
    add(3, "single_blocks", base)

    for i in range(SWEEP):
        w = int(rng.integers(5, 100))
        h = int(rng.integers(5, 100))
        g = rng.integers(0, 256, (h, w), dtype=np.uint8)
        cw = int(rng.integers(1, w + 1))
        ch = int(rng.integers(1, h + 1))
        add(2, f"strided_{i}", g, count=cw, order=ch,
            x0=int(rng.integers(0, w - cw + 1)),
            y0=int(rng.integers(0, h - ch + 1)))

    for i in range(SWEEP):
        w = int(rng.integers(1, 80))
        h = int(rng.integers(1, 80))
        add(4, f"rgb_{i}", rng.integers(0, 256, (h, w, 3), dtype=np.uint8))
    return cases


# ── harness ─────────────────────────────────────────────────────────────────
def _build(tmp: Path) -> Path:
    cc = os.environ.get("CC") or shutil.which("cc")
    exe = tmp / "sign_glyph_host"
    subprocess.run([cc, "-std=c99", "-O2", "-Wall", "-Wextra", "-Werror",
                    str(HARNESS), str(SOURCE), "-o", str(exe)], check=True)
    return exe


def _manifest_rows(cases):
    rows = [" ".join(HEADER)]
    for c, data in cases:
        rows.append(" ".join(str(c[k]) for k in HEADER))
    return "\n".join(rows) + "\n"


def _parse_meta(path: Path):
    lines = path.read_text().splitlines()
    head = dict(kv.split("=", 1) for kv in lines[0].split())
    slots = []
    for line in lines[1:]:
        parts = line.split()
        slots.append({
            "thresh": int(parts[1].split("=")[1]),
            "ok": int(parts[2].split("=")[1]),
            "empty": int(parts[3].split("=")[1]),
            "box": tuple(int(v) for v in parts[4].split("=")[1].split(",")),
        })
    return head, slots


def _check_info(name, out_dir, want):
    """Compare the frame fields every entry point reports the same way."""
    head, slots = _parse_meta(out_dir / f"{name}.cmeta")
    errs = []
    if int(head["thresh"]) != int(want["thresh"]):
        errs.append(f"thresh {head['thresh']} != {want['thresh']}")
    if int(head["ok"]) != int(bool(want["ok"])):
        errs.append(f"ok {head['ok']} != {want['ok']}")
    if int(head["empty"]) != int(bool(want["empty"])):
        errs.append(f"empty {head['empty']} != {want['empty']}")
    want_box = want["box"]
    got_box = None if head["box"] == "-" else tuple(
        int(v) for v in head["box"].split(","))
    if want_box is None:
        if got_box is not None:
            errs.append(f"box {got_box} != None")
    elif got_box != tuple(int(v) for v in want_box):
        errs.append(f"box {got_box} != {want_box}")
    return head, slots, errs


def _run_corpus():
    cc = os.environ.get("CC") or shutil.which("cc")
    if not cc:
        return None
    cases = _cases()
    with tempfile.TemporaryDirectory(prefix="sign_glyph_parity_") as td:
        tmp = Path(td)
        exe = _build(tmp)
        (tmp / "manifest.tsv").write_text(_manifest_rows(cases))
        for c, data in cases:
            (tmp / f"{c['name']}.raw").write_bytes(data.tobytes())
        subprocess.run([str(exe), str(tmp)], check=True,
                       stdout=subprocess.DEVNULL)

        results = {"sequence": [], "split": [], "single": [], "strided": [],
                   "luminance": [], "paths": {"merge": 0, "split": 0,
                                               "equal": 0}}
        for c, data in cases:
            name = c["name"]
            errs = []
            if c["mode"] == 1:
                want = ocr.split_sign_glyph_boxes(
                    data != 0, count=c["count"], min_run=c["min_run"])
                text = (tmp / f"{name}.cboxes").read_text().splitlines()
                got_n = int(text[0].split("=")[1])
                got_boxes = [tuple(int(v) for v in ln.split(","))
                             for ln in text[1:]]
                if got_n != len(got_boxes) or got_boxes != want:
                    errs.append(f"got n={got_n} {got_boxes} want {want}")
                results["split"].append(errs)

            elif c["mode"] == 2:
                got = int((tmp / f"{name}.cot").read_text().split("=")[1])
                want = ocr.otsu_threshold_u8(
                    data[c["y0"]:c["y0"] + c["order"],
                         c["x0"]:c["x0"] + c["count"]])
                if got != want:
                    errs.append(f"strided {got} != {want}")
                results["strided"].append(errs)

            elif c["mode"] == 4:
                got = np.fromfile(tmp / f"{name}.cgray", np.uint8)
                want = ocr.rgb_to_lum_u8(data)
                if not np.array_equal(got, want.reshape(-1)):
                    errs.append("luminance differs")
                results["luminance"].append(errs)

            elif c["mode"] == 3:
                want = ocr.preprocess_glyph(
                    data, out_size=c["out_size"], polarity=c["polarity"],
                    apply_segmentation=bool(c["seg"]), min_run=c["min_run"],
                    thresh=(None if c["thresh"] < 0 else c["thresh"]),
                    convention=(ocr.NORM_CANON if c["conv"]
                                else ocr.NORM_STRETCH),
                    dark_thr=c["dark"])
                head, _, errs = _check_info(name, tmp, want)
                if c["seg"] and int(head["nb"]) != len(want["boxes"]):
                    errs.append(f"nb {head['nb']} != {len(want['boxes'])}")
                got = np.fromfile(tmp / f"{name}.cglyphs", np.uint8)
                if not np.array_equal(
                        got.reshape(c["out_size"], c["out_size"]),
                        want["glyph"]):
                    errs.append("glyph differs")
                results["single"].append(errs)

            else:
                want = ocr.preprocess_sign_sequence(
                    data, sign_sequence="ABC",
                    reading_order=("rtl" if c["order"] else "ltr"),
                    out_size=c["out_size"], polarity=c["polarity"],
                    apply_segmentation=bool(c["seg"]),
                    convention=(ocr.NORM_CANON if c["conv"]
                                else ocr.NORM_STRETCH),
                    thresh=(None if c["thresh"] < 0 else c["thresh"]),
                    dark_thr=c["dark"])
                head, slots, errs = _check_info(name, tmp, {
                    "thresh": want["thresh"], "box": want["sign_box"],
                    "ok": want["ok"], "empty": want["empty"],
                })
                if c["seg"]:
                    t = int(want["thresh"])
                    if t == 0 and c["thresh"] < 0:
                        ink = np.zeros(data.shape, bool)
                    else:
                        ink = ocr.binarize_ink(data, t,
                                               polarity=c["polarity"])
                        if c["dark"] > 0:
                            ink &= data >= np.uint8(min(255, int(c["dark"])))
                    nb = len(ocr.segment_glyph_boxes(ink,
                                                     min_run=c["min_run"]))
                    if nb != int(head["nb"]):
                        errs.append(f"nb {head['nb']} != {nb}")
                    key = "merge" if nb > 3 else ("split" if nb < 3
                                                  else "equal")
                    results["paths"][key] += 1
                got = np.fromfile(tmp / f"{name}.cglyphs", np.uint8)
                osz = c["out_size"]
                got = got.reshape(c["count"], osz, osz)
                for i, slot in enumerate(want["slots"]):
                    gs = slots[i]
                    if gs["box"] != tuple(slot["box"]):
                        errs.append(f"slot{i} box {gs['box']} != "
                                    f"{slot['box']}")
                    if gs["ok"] != int(bool(slot["ok"])):
                        errs.append(f"slot{i} ok {gs['ok']} != {slot['ok']}")
                    if gs["empty"] != int(bool(slot["empty"])):
                        errs.append(f"slot{i} empty {gs['empty']} != "
                                    f"{slot['empty']}")
                    if not np.array_equal(got[i], slot["glyph"]):
                        diff = np.argwhere(got[i] != slot["glyph"])[:5]
                        errs.append(f"slot{i} glyph differs at "
                                    f"{diff.tolist()}")
                results["sequence"].append(errs)
        results["count"] = len(cases)
        return results


_CACHE = None


def _results():
    global _CACHE
    if _CACHE is None:
        _CACHE = _run_corpus()
    return _CACHE


def _require_corpus():
    r = _results()
    assert r is not None, (
        "no C compiler found (set CC); the C-vs-Python parity suite cannot run")
    return r


# ── tests ──────────────────────────────────────────────────────────────────
def test_sign_sequence_matches_python_bit_for_bit():
    r = _require_corpus()
    bad = [(i, e) for i, e in enumerate(r["sequence"]) if e]
    assert not bad, f"{len(bad)}/{len(r['sequence'])} sequence cases differ: {bad[:5]}"


def test_box_split_matches_python_bit_for_bit():
    r = _require_corpus()
    bad = [(i, e) for i, e in enumerate(r["split"]) if e]
    assert not bad, f"{len(bad)}/{len(r['split'])} split cases differ: {bad[:5]}"


def test_single_glyph_matches_python_bit_for_bit():
    r = _require_corpus()
    bad = [(i, e) for i, e in enumerate(r["single"]) if e]
    assert not bad, f"{len(bad)}/{len(r['single'])} single-glyph cases differ: {bad[:5]}"


def test_otsu_and_luminance_stages_match():
    r = _require_corpus()
    bad = [(i, e) for i, e in enumerate(r["strided"]) if e]
    assert not bad, f"strided Otsu differs: {bad[:5]}"
    bad = [(i, e) for i, e in enumerate(r["luminance"]) if e]
    assert not bad, f"RGB luminance differs: {bad[:5]}"


def test_corpus_exercises_all_three_reconciliation_paths():
    # A parity suite that only ever sees three clean runs cannot catch a
    # merge or split bug.  Fail loudly if the corpus drifts away from the
    # branches it exists to cover.
    r = _require_corpus()
    paths = r["paths"]
    assert paths["merge"] > 0, f"no merge-path case in {r['count']} cases"
    assert paths["split"] > 0, f"no split-path case in {r['count']} cases"
    assert paths["equal"] > 0, f"no equal-thirds case in {r['count']} cases"


def main() -> int:
    tests = [(name, fn) for name, fn in sorted(globals().items())
             if name.startswith("test_") and callable(fn)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
        except Exception:  # noqa: BLE001 - a test runner reports everything
            failed += 1
            print(f"FAIL {name}")
            traceback.print_exc()
        else:
            print(f"ok   {name}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
