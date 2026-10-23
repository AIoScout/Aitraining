"""Headless-Chrome checks for the settings-driven Preview controls.

The Preview card no longer carries a source switcher of its own: the Settings
-> Preview selector is the master, and the card renders ONLY the control for
the active source.  These tests drive the shipped SPA through Playwright, so
they assert the DOM the user actually sees.

Covered:
  context bar   webcam -> #previewWebcamSelect; device -> port select +
                Refresh; upload -> Upload button + hidden file input.  The
                other sources' controls are absent, not merely hidden.
  settings      the Settings selector drives STATE.preview_source and the
                bar follows on save
  upload        choosing a file routes it through /preview/upload and the
                sign verdict renders as the YES banner

Requires Playwright with a system Chrome:
    .venv/bin/python tests/test_autofill_preview_spa.py
"""
from __future__ import annotations

import base64
import json
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, Path(__file__).resolve().parent):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from spa_harness import STUB_FETCH, boot_payload, build_spa_html, serve

from playwright.sync_api import sync_playwright

# A real 1x1 PNG, so File/FileReader exercise the same path as a user drop.
_TINY_PNG_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)

_SIGN_YES = {
    "ok": "1",
    "sign_ocr": True,
    "sign_sequence": ["A", "1", "2"],
    "reading_order": "ltr",
    "slots": [
        {"char": "A", "label": "A", "prob": 0.99, "match": True},
        {"char": "1", "label": "1", "prob": 0.98, "match": True},
        {"char": "2", "label": "2", "prob": 0.97, "match": True},
    ],
    "slot_images_b64": ["", "", ""],
    "match": True,
    "match_text": "🟢 YES — MATCH",
    "match_min_prob": 0.6,
    "processed_image_b64": "cm9p",
    "processed_variant": "sign_search",
    "crop_image_b64": "Y3JvcA==",
    "full_image_b64": _TINY_PNG_B64,
    "orig_b64": "b3JpZw==",
    "roi_b64": "cm9p",
    "filename": "sign_A12.png",
}


def _set_source(page, source: str):
    page.evaluate(
        """(src) => {
          STATE.preview_source = src;
          previewSource = src;
          renderPreviewCard();
        }""",
        source,
    )


def assert_context_bar_renders_only_the_active_source(page):
    # Webcam: camera select only.
    _set_source(page, "webcam")
    assert page.evaluate("!!document.getElementById('previewWebcamSelect')") is True
    assert page.evaluate("!!document.getElementById('previewSerialPortSelect')") is False
    assert page.evaluate("!!document.getElementById('previewFileInput')") is False
    assert page.evaluate("!!document.getElementById('previewUploadBtn')") is False

    # Device: port select + Refresh, and nothing from webcam/upload.
    _set_source(page, "device")
    assert page.evaluate("!!document.getElementById('previewSerialPortSelect')") is True
    assert page.evaluate("!!document.getElementById('previewSerialRefreshBtn')") is True
    assert page.evaluate("!!document.getElementById('previewWebcamSelect')") is False
    assert page.evaluate("!!document.getElementById('previewFileInput')") is False

    # Upload: one Upload button, a HIDDEN file input, filename display.
    _set_source(page, "upload")
    assert page.evaluate(
        "document.getElementById('previewUploadBtn').textContent.trim()"
    ) == "📁 Upload Image..."
    assert page.evaluate("!!document.getElementById('previewFileInput')") is True
    assert page.evaluate(
        "getComputedStyle(document.getElementById('previewFileInput')).display"
    ) == "none"
    assert page.evaluate("!!document.getElementById('previewUploadName')") is True
    assert page.evaluate("!!document.getElementById('previewWebcamSelect')") is False
    assert page.evaluate("!!document.getElementById('previewSerialPortSelect')") is False


def assert_settings_selector_drives_the_bar(page):
    page.evaluate("closeGlobalSettings(); localStorage.clear();")
    page.evaluate("STATE.preview_source = 'webcam'; previewSource = 'webcam'; renderPreviewCard();")
    page.evaluate("openGlobalSettings()")
    assert page.evaluate("!!document.getElementById('settingsPreviewSource')") is True
    page.evaluate(
        """() => {
          const sel = document.getElementById('settingsPreviewSource');
          sel.value = 'device';
          sel.dispatchEvent(new Event('change', {bubbles: true}));
        }"""
    )
    # Save applies the selector to the card without a reload.
    page.evaluate("document.getElementById('settingsSave').click()")
    page.wait_for_function(
        "!!document.getElementById('previewSerialPortSelect')",
        timeout=5000,
    )
    assert page.evaluate("STATE.preview_source") == "device"
    assert page.evaluate("!!document.getElementById('previewWebcamSelect')") is False
    page.evaluate("closeGlobalSettings()")
    # Back to webcam for the upload test's clean slate.
    _set_source(page, "webcam")


def assert_toggle_backfills_existing_cards_immediately(page):
    """Classes already on screen with 0 samples update to 24, no reload."""
    page.evaluate(
        """() => {
          // The target sequence must cover the classes, or the sign-project
          // render prunes the ones the slots do not ask for.
          STATE.train_cfg = Object.assign({}, STATE.train_cfg || {}, {
            sign_sequence: ['A', '1', '2'],
          });
          STATE.classes = ['A', '1', '2'];
          STATE.counts = {'A': 0, '1': 0, '2': 0};
          STATE.sample_previews = {};
          STATE.processed_previews = {};
          STATE.train_enabled = false;
          autoSeedEmnistSamples = false;
          autoSeedEmnistAttempted = new Set();
          render();
        }"""
    )
    assert page.evaluate(
        "document.getElementById('trainBtn').disabled"
    ) is True
    assert page.evaluate(
        "document.getElementById('summaryTitle-A').textContent"
    ).strip() == "0 Image Samples"

    preview = {"src": "data:image/png;base64," + _TINY_PNG_B64, "filename": "emnist_seed_00.png"}
    state = {
        "ok": "1",
        "classes": ["A", "1", "2"],
        "counts": {"A": 24, "1": 24, "2": 24},
        "sample_previews": {"A": [preview], "1": [preview], "2": [preview]},
        "processed_previews": {"A": [preview], "1": [preview], "2": [preview]},
        "train_enabled": True,
        "train_rec": {"total_samples": 72, "batch_size": 16, "epochs": 20,
                      "learning_rate": 0.001, "validation_split": 0.2, "reasons": {}},
    }
    page.evaluate(
        """(payload) => {
          window.__liveStateBodies = [];
          const real = window.fetch;
          window.fetch = async (url, opts) => {
            if (String(url).includes('/live/config')) {
              window.__liveStateBodies.push(String(url));
              return new Response(JSON.stringify(payload), {
                status: 200, headers: {'Content-Type': 'application/json'}
              });
            }
            return real(url, opts);
          };
        }""",
        state,
    )

    page.evaluate("openGlobalSettings()")
    page.evaluate("document.getElementById('settingsAutoSeedEmnist').click()")
    page.wait_for_function("STATE.counts['A'] === 24", timeout=5000)

    # The DOM — not just STATE — shows the filled classes and enabled button.
    for name in ("A", "1", "2"):
        assert page.evaluate(
            f"document.getElementById('summaryTitle-{name}').textContent"
        ).strip() == "24 Image Samples", name
    assert page.evaluate("STATE.train_enabled") is True
    assert page.evaluate("document.getElementById('trainBtn').disabled") is False
    # The response's thumbnails were applied too, so the card is not blank.
    assert page.evaluate(
        "document.querySelectorAll('.samples-strip img').length"
    ) > 0
    page.evaluate(
        """() => {
          closeGlobalSettings();
          autoSeedEmnistSamples = false;
          autoSeedEmnistAttempted = new Set();
          try { localStorage.removeItem('auto_seed_emnist_samples'); } catch (e) {}
        }"""
    )


def _stub_upload(page, payload):
    page.evaluate(
        """(payload) => {
          window.__uploadBodies = [];
          const real = window.fetch;
          window.fetch = async (url, opts) => {
            if (String(url).includes('/preview/upload')) {
              try { window.__uploadBodies.push(JSON.parse(opts.body)); } catch (e) {}
              return new Response(JSON.stringify(payload), {
                status: 200, headers: {'Content-Type': 'application/json'}
              });
            }
            return real(url, opts);
          };
        }""",
        payload,
    )


def assert_upload_shows_the_yes_banner(page):
    _set_source(page, "upload")
    _stub_upload(page, _SIGN_YES)
    page.evaluate(
        """async ({b64, name}) => {
          const bytes = Uint8Array.from(atob(b64), (c) => c.charCodeAt(0));
          const file = new File([bytes], name, {type: 'image/png'});
          return await handlePreviewUploadFile(file);
        }""",
        {"b64": _TINY_PNG_B64, "name": "sign_A12.png"},
    )
    page.wait_for_function(
        "document.querySelector('#previewOutput .signmatch') !== null",
        timeout=5000,
    )
    body = page.evaluate("window.__uploadBodies[0]")
    assert body["filename"] == "sign_A12.png", body
    assert body["image_b64"] == _TINY_PNG_B64, body
    assert body["session"] == "test-session", body
    assert "sign_sequence" in body, body

    assert page.evaluate(
        "document.querySelector('#previewOutput .signmatch').classList.contains('yes')"
    ) is True
    assert page.evaluate(
        "document.querySelector('#previewOutput .signmatch-main').textContent"
    ).strip() == "🟢 YES — MATCH"
    assert page.evaluate(
        "document.getElementById('previewUploadName').textContent"
    ).strip() == "sign_A12.png"
    # The note names the uploaded file, not a camera frame.
    note = page.evaluate("document.getElementById('previewNote').textContent")
    assert "Upload: sign_A12.png" in note, note


def assert_drop_is_ignored_off_the_upload_source(page):
    # Dropping while the card reads the camera must not switch what the model
    # is looking at — the handler is armed only in upload mode.
    _set_source(page, "webcam")
    page.evaluate("window.__uploadBodies = []")
    page.evaluate(
        """({b64, name}) => {
          const bytes = Uint8Array.from(atob(b64), (c) => c.charCodeAt(0));
          const file = new File([bytes], name, {type: 'image/png'});
          const dt = new DataTransfer();
          dt.items.add(file);
          const pane = document.getElementById('previewPane');
          pane.dispatchEvent(new DragEvent('drop', {bubbles: true, dataTransfer: dt}));
        }""",
        {"b64": _TINY_PNG_B64, "name": "dropped.png"},
    )
    page.wait_for_timeout(150)
    assert page.evaluate("window.__uploadBodies.length") == 0
    assert page.evaluate("previewContextSource()") == "webcam"


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

            run("context bar renders only the active source", assert_context_bar_renders_only_the_active_source, page)
            run("settings selector drives the bar", assert_settings_selector_drives_the_bar, page)
            run("toggle backfills existing cards immediately", assert_toggle_backfills_existing_cards_immediately, page)
            run("upload renders the YES banner", assert_upload_shows_the_yes_banner, page)
            run("drop is ignored off the upload source", assert_drop_is_ignored_off_the_upload_source, page)

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
