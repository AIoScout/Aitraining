"""Headless-Chrome checks for the Preview ROI rectangle.

The ROI checkbox owns the green/blue crop rectangle: toggling it ON draws
the box on the scene, toggling it OFF hides it.  The box is normalized to
the ORIGINAL frame, so while ROI is the only active layer the pane must
keep a full-frame image (the raw scene) — showing the cropped tensor there
would put the rectangle in the wrong place.  Orig still selects the full
masked frame, and ROI + Orig still shows the processed crop, where a
full-frame box cannot be drawn.

Requires Playwright with a system Chrome:
    .venv/bin/python tests/test_preview_roi_spa.py
"""
from __future__ import annotations

import base64
import io
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, Path(__file__).resolve().parent):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from spa_harness import STUB_FETCH, boot_payload, build_spa_html, serve

from PIL import Image
from playwright.sync_api import sync_playwright


def _png_b64(size, color) -> str:
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


# Four visually distinct frames: the test can tell which payload field the
# pane chose without trusting any state besides the <img> src.
RAW = _png_b64((40, 30), (200, 40, 40))
FULL = _png_b64((40, 30), (220, 210, 60))
CROP = _png_b64((20, 20), (40, 60, 220))
PROCESSED = _png_b64((20, 20), (40, 180, 60))

CROP_RECT = [0.2, 0.25, 0.8, 0.75]


def _url(b64: str) -> str:
    return "data:image/png;base64," + b64


_PAYLOAD = {
    "ok": "1",
    "labels": ["A", "B"],
    "probs": [0.9, 0.1],
    "top_label": "A",
    "top_prob": 0.9,
    "ood": False,
    "image_b64": RAW,
    "processed_image_b64": PROCESSED,
    "crop_image_b64": CROP,
    "full_image_b64": FULL,
    "processed_variant": "masked",
    "crop": CROP_RECT,
}


def _apply(page, payload=_PAYLOAD):
    page.evaluate(
        """(payload) => {
          window.__payload = payload;
          previewLastPredictData = payload;
          applyPreviewSrcToPane(payload);
        }""",
        payload,
    )


def _state(page):
    return page.evaluate(
        """() => {
          const ov = document.getElementById('previewRoiOverlay');
          const img = document.getElementById('previewImage');
          return {
            display: ov ? ov.style.display : 'missing',
            src: img ? img.src : '',
            box: ov ? {
              left: parseFloat(ov.style.left || 'NaN'),
              top: parseFloat(ov.style.top || 'NaN'),
              width: parseFloat(ov.style.width || 'NaN'),
              height: parseFloat(ov.style.height || 'NaN'),
            } : null,
          };
        }"""
    )


def _set_toggle(page, toggle_id: str, checked: bool):
    page.evaluate(
        """({id, checked}) => {
          const el = document.getElementById(id);
          if (el.checked !== checked) el.click();
        }""",
        {"id": toggle_id, "checked": checked},
    )
    # onViewToggleChanged defers persist + renderPreviewCard to a 0 ms
    # timeout; let that run so the assertions see the settled DOM.
    page.wait_for_timeout(80)


def assert_roi_off_shows_full_threshold_and_box(page):
    """ROI OFF: full threshold view by default, with the box on top."""
    _set_toggle(page, "previewRoiToggle", False)
    _set_toggle(page, "previewRawToggle", False)
    _apply(page)
    page.wait_for_function(
        "(() => { const i = document.getElementById('previewImage');"
        " return i && i.complete && i.naturalWidth > 0; })()",
        timeout=5000,
    )
    _apply(page)
    state = _state(page)
    assert state["display"] == "block", state
    assert state["src"] == _url(FULL), state

    expected = page.evaluate(
        """() => {
          const img = document.getElementById('previewImage');
          const stage = document.querySelector('#previewPane .preview-stage');
          const r = stage.getBoundingClientRect();
          const scale = Math.min(r.width / img.naturalWidth,
                                 r.height / img.naturalHeight);
          const w = img.naturalWidth * scale;
          const h = img.naturalHeight * scale;
          const ox = (r.width - w) * 0.5;
          const oy = (r.height - h) * 0.5;
          const c = window.__payload.crop;
          return {
            left: ox + c[0] * w,
            top: oy + c[1] * h,
            width: (c[2] - c[0]) * w,
            height: (c[3] - c[1]) * h,
          };
        }"""
    )
    for key in ("left", "top", "width", "height"):
        assert abs(state["box"][key] - expected[key]) < 0.75, (key, state["box"], expected)


def assert_roi_on_shows_cropped_threshold_without_box(page):
    """ROI ON: the thresholded crop itself; the box is hidden."""
    _set_toggle(page, "previewRoiToggle", True)
    _set_toggle(page, "previewRawToggle", False)
    _apply(page)
    state = _state(page)
    assert state["display"] == "none", state
    assert state["src"] == _url(PROCESSED), state


def assert_roi_off_again_shows_the_box(page):
    _set_toggle(page, "previewRoiToggle", False)
    _apply(page)
    state = _state(page)
    assert state["display"] == "block", state
    assert state["src"] == _url(FULL), state


def assert_orig_on_shows_raw_full_with_box(page):
    """Orig ON: original grayscale full frame; ROI OFF keeps the box."""
    _set_toggle(page, "previewRoiToggle", False)
    _set_toggle(page, "previewRawToggle", True)
    _apply(page)
    state = _state(page)
    assert state["display"] == "block", state
    assert state["src"] == _url(RAW), state


def assert_both_toggles_show_raw_crop_without_box(page):
    """ROI ON + Orig ON: raw cropped ROI, and no overlay."""
    _set_toggle(page, "previewRoiToggle", True)
    _apply(page)
    state = _state(page)
    assert state["display"] == "none", state
    assert state["src"] == _url(CROP), state


def assert_square_stage_keeps_the_frame_sharp(page):
    """A 160x160 device frame fills a 1:1 stage; the ROI box stays on it."""
    raw160 = _png_b64((160, 160), (180, 180, 180))
    full160 = _png_b64((160, 160), (250, 250, 250))
    crop160 = _png_b64((80, 80), (200, 200, 200))
    payload = {
        "ok": "1",
        "labels": ["A", "1", "2"],
        "probs": [0.9, 0.05, 0.05],
        "top_label": "A",
        "top_prob": 0.9,
        "image_b64": raw160,
        "processed_image_b64": PROCESSED,
        "crop_image_b64": crop160,
        "full_image_b64": full160,
        "processed_variant": "sign-threshold",
        "crop": [0.15625, 0.15625, 0.84375, 0.84375],
        "sign_ocr": True,
        "sign_sequence": ["A", "1", "2"],
        "reading_order": "ltr",
        "match": True,
        "match_text": "🟢 YES — MATCH",
        "match_min_prob": 0.6,
        "slots": [
            {"char": "A", "label": "A", "prob": 0.98, "match": True},
            {"char": "1", "label": "1", "prob": 0.97, "match": True},
            {"char": "2", "label": "2", "prob": 0.99, "match": True},
        ],
        "slot_images_b64": ["", "", ""],
    }
    _set_toggle(page, "previewRoiToggle", False)
    _set_toggle(page, "previewRawToggle", False)
    _apply(page, payload)
    page.wait_for_function(
        "(() => { const i = document.getElementById('previewImage');"
        " return i && i.complete && i.naturalWidth === 160; })()",
        timeout=5000,
    )
    _apply(page, payload)
    stage = page.evaluate(
        """() => {
          const s = document.querySelector('#previewPane .preview-stage').getBoundingClientRect();
          const ov = document.getElementById('previewRoiOverlay');
          return {
            w: s.width, h: s.height,
            display: ov.style.display,
            left: parseFloat(ov.style.left || 'NaN'),
            top: parseFloat(ov.style.top || 'NaN'),
            width: parseFloat(ov.style.width || 'NaN'),
            height: parseFloat(ov.style.height || 'NaN'),
          };
        }"""
    )
    # 160x160 device frames get a clean square stage, not a wide letterbox;
    # ROI OFF shows the threshold view with the box on top.
    assert abs(stage["w"] - stage["h"]) < 1.5, stage
    assert stage["display"] == "block", stage
    assert abs(stage["left"] - stage["w"] * 0.15625) < 1.0, stage
    assert abs(stage["top"] - stage["h"] * 0.15625) < 1.0, stage
    assert abs(stage["width"] - stage["w"] * 0.6875) < 1.0, stage
    assert abs(stage["height"] - stage["h"] * 0.6875) < 1.0, stage
    # The box must hug the image content, never the full wide pane.
    assert stage["width"] < stage["w"] * 0.80, stage
    assert _state(page)["src"] == _url(full160), _state(page)

    # ROI ON: the cropped threshold content, no box.
    _set_toggle(page, "previewRoiToggle", True)
    _apply(page, payload)
    state = _state(page)
    assert state["src"] == _url(PROCESSED), state
    assert state["display"] == "none", state

    # ROI ON + Orig ON: raw cropped ROI, still no box.
    _set_toggle(page, "previewRawToggle", True)
    _apply(page, payload)
    state = _state(page)
    assert state["src"] == _url(crop160), state
    assert state["display"] == "none", state

    # ROI OFF + Orig ON: full raw frame + box.
    _set_toggle(page, "previewRoiToggle", False)
    _apply(page, payload)
    state = _state(page)
    assert state["src"] == _url(raw160), state
    assert state["display"] == "block", state


def assert_worktable_is_centered_with_square_stages(page):
    """Edit modal centered; Original / Processed stages are 1:1."""
    page.evaluate(
        """(src) => {
          STATE.sample_previews = Object.assign({}, STATE.sample_previews || {}, {
            A: [{src, filename: 'sample.png'}],
          });
          STATE.counts = Object.assign({}, STATE.counts || {}, {A: 1});
          openClassPreprocessEditor('A');
        }""",
        _url(RAW),
    )
    page.wait_for_selector("#classPreprocessStage", timeout=5000)
    page.wait_for_timeout(250)
    box = page.evaluate(
        """() => {
          const shell = document.querySelector('#classPreprocessModal .class-preprocess-shell').getBoundingClientRect();
          const stage = document.getElementById('classPreprocessStage').getBoundingClientRect();
          const result = document.querySelector('#classPreprocessModal .class-preprocess-result').getBoundingClientRect();
          return {
            vw: innerWidth, vh: innerHeight,
            shell: {l: shell.left, t: shell.top, r: shell.right, b: shell.bottom},
            stage: {w: stage.width, h: stage.height},
            result: {w: result.width, h: result.height},
          };
        }"""
    )
    assert abs((box["shell"]["l"] + box["shell"]["r"]) / 2 - box["vw"] / 2) < 2.0, box
    assert abs((box["shell"]["t"] + box["shell"]["b"]) / 2 - box["vh"] / 2) < 2.0, box
    assert box["stage"]["w"] > 0 and abs(box["stage"]["w"] - box["stage"]["h"]) < 1.5, box
    assert box["result"]["w"] > 0 and abs(box["result"]["w"] - box["result"]["h"]) < 1.5, box
    page.evaluate("closeClassPreprocessEditor(true)")


def assert_input_off_keeps_the_box_visible(page):
    # With Input OFF no prediction arrives to redraw the overlay; the
    # deferred render must re-apply the cached one instead of hiding it.
    # ROI OFF is the state that owns the box now.
    _set_toggle(page, "previewRawToggle", False)
    _set_toggle(page, "previewRoiToggle", False)
    _apply(page)
    page.evaluate("renderPreviewCard()")
    page.wait_for_timeout(80)
    state = _state(page)
    assert state["display"] == "block", state
    assert state["src"] == _url(FULL), state


def main() -> int:
    html = build_spa_html(boot_payload())
    results = []

    def run(name, fn, page):
        try:
            fn(page)
        except Exception as exc:  # noqa: BLE001 - a test runner reports everything
            results.append((name, False))
            print(f"FAIL {name}: {exc}")
            traceback.print_exc()
        else:
            results.append((name, True))
            print(f"ok   {name}")

    with serve(html) as url:
        with sync_playwright() as p:
            browser = p.chromium.launch(channel="chrome", headless=True)
            page = browser.new_page()
            errors = []
            page.on("pageerror", lambda e: errors.append(str(e)))
            page.add_init_script(STUB_FETCH)
            page.goto(url)
            page.wait_for_function("typeof renderPreviewCard === 'function'")
            page.wait_for_timeout(300)

            # Start from a clean, deterministic preview state: no persisted
            # ROI/Orig choices and no live loop.
            page.evaluate(
                """() => {
                  try { localStorage.clear(); } catch (e) {}
                  previewInputOn = false;
                  previewSource = 'upload';
                  STATE.preview_source = 'upload';
                  previewUploadImageSrc = '';
                  previewLastStaticImageSrc = '';
                  previewShowRoi = false;
                  previewShowRaw = false;
                  renderPreviewCard();
                  applyPreviewSrcToPane(window.__payload || null);
                }"""
            )
            page.evaluate("window.__payload = null")

            run("ROI OFF shows full threshold + box", assert_roi_off_shows_full_threshold_and_box, page)
            run("ROI ON shows cropped threshold, no box", assert_roi_on_shows_cropped_threshold_without_box, page)
            run("ROI OFF shows the box again", assert_roi_off_again_shows_the_box, page)
            run("Orig ON shows raw full + box", assert_orig_on_shows_raw_full_with_box, page)
            run("ROI + Orig shows raw crop, no box", assert_both_toggles_show_raw_crop_without_box, page)
            run("160x160 frame gets a square stage", assert_square_stage_keeps_the_frame_sharp, page)
            run("worktable is centered with square stages", assert_worktable_is_centered_with_square_stages, page)
            run("Input OFF keeps the box visible", assert_input_off_keeps_the_box_visible, page)

            browser.close()

    if errors:
        print("\npage errors:")
        for e in errors:
            print("  " + e)
    passed = sum(1 for _, ok in results if ok)
    print(f"\n{passed}/{len(results)} passed")
    return 0 if passed == len(results) and not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
