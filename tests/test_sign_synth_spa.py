"""Headless-Chrome checks for the synthetic-sign pills in Preview.

A sign project with no road-sign photo still needs to test the three-slot
split and the YES/NO verdict.  `#previewContextBar` now carries three pills
beside `📁 Upload Image...`: Synth Match, Synth Reversed and Synth
Mismatch.  These tests click them in the shipped SPA (extracted and served
by `spa_harness`) and assert the DOM the user sees.

Covered:
  pills     all three appear in the upload source of a sign project, with
            the tooltips that explain the direction probe and the NO test
  request   a click POSTs /preview/synthesize_sign with the on-screen
            target and reading order
  match     the YES banner, three slot chips and the plate image render,
            Input turns ON, and the filename/note name the synth frame
  mismatch  the banner switches to NO on the second click

Requires Playwright with a system Chrome:
    .venv/bin/python tests/test_sign_synth_spa.py
"""
from __future__ import annotations

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

# A real 1x1 PNG: the plate image the stub returns must survive the same
# data-URL path a real synthesized frame travels.
_TINY_PNG_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)

_SLOTS_YES = [
    {"char": "A", "label": "A", "prob": 0.99, "match": True},
    {"char": "1", "label": "1", "prob": 0.98, "match": True},
    {"char": "2", "label": "2", "prob": 0.97, "match": True},
]
_SLOTS_NO = [
    {"char": "A", "label": "A", "prob": 0.99, "match": True},
    {"char": "1", "label": "1", "prob": 0.98, "match": True},
    {"char": "5", "label": "2", "prob": 0.91, "match": False},
]


def _sign_payload(*, match: bool, variant: str, chars, filename: str):
    return {
        "ok": "1",
        "sign_ocr": True,
        "sign_sequence": ["A", "1", "2"],
        "reading_order": "ltr",
        "slots": _SLOTS_YES if match else _SLOTS_NO,
        "slot_images_b64": ["", "", ""],
        "match": match,
        "match_text": "🟢 YES — MATCH" if match else "🔴 NO — NOT MATCH",
        "top_label": "🟢 YES — MATCH" if match else "🔴 NO — NOT MATCH",
        "top_prob": 0.97 if match else 0.91,
        "match_min_prob": 0.6,
        "processed_image_b64": "cm9p",
        "processed_variant": "sign_search",
        "crop_image_b64": "Y3JvcA==",
        "full_image_b64": _TINY_PNG_B64,
        "image_b64": _TINY_PNG_B64,
        "orig_b64": _TINY_PNG_B64,
        "roi_b64": "cm9p",
        "filename": filename,
        "synth_variant": variant,
        "synth_chars": list(chars),
    }


_YES = _sign_payload(match=True, variant="match", chars=["A", "1", "2"],
                     filename="synth_A12_match.png")
_NO = _sign_payload(match=False, variant="mismatch", chars=["A", "1", "5"],
                    filename="synth_A12_mismatch.png")


def _boot_payload():
    return boot_payload(
        classes=["A", "1", "2"],
        counts={"A": 3, "1": 2, "2": 1},
        train_cfg={
            "img_size": 28,
            "preprocess_mode": "auto_by_label",
            "crop_mode": "auto_search",
            "reading_order": "ltr",
            "sign_sequence": ["A", "1", "2"],
            "ocr_emnist_enabled": True,
            "ocr_transfer_learning": True,
        },
    )


def _set_source(page, source: str):
    page.evaluate(
        """(src) => {
          STATE.preview_source = src;
          previewSource = src;
          renderPreviewCard();
        }""",
        source,
    )


def _stub_synth(page):
    """Serve the two verdict plates and record every request body."""
    page.evaluate(
        """(payloads) => {
          window.__synthBodies = [];
          const real = window.fetch;
          window.fetch = async (url, opts) => {
            if (String(url).includes('/preview/synthesize_sign')) {
              let body = {};
              try { body = JSON.parse(opts.body); } catch (e) {}
              window.__synthBodies.push(body);
              const key = String(body.variant || 'match');
              return new Response(JSON.stringify(payloads[key] || payloads.match), {
                status: 200, headers: {'Content-Type': 'application/json'}
              });
            }
            return real(url, opts);
          };
        }""",
        {"match": _YES, "reversed": _YES, "mismatch": _NO},
    )


def assert_pills_render_in_upload_source(page):
    _set_source(page, "upload")
    assert page.evaluate("!!document.getElementById('previewSynthMatchBtn')") is True
    assert page.evaluate("!!document.getElementById('previewSynthRevBtn')") is True
    assert page.evaluate("!!document.getElementById('previewSynthMismatchBtn')") is True
    assert page.evaluate(
        "document.getElementById('previewSynthMatchBtn').textContent.trim()"
    ) == "🧪 Synth Match"
    assert page.evaluate(
        "document.getElementById('previewSynthRevBtn').textContent.trim()"
    ) == "🔀 Synth Reversed"
    assert page.evaluate(
        "document.getElementById('previewSynthMismatchBtn').textContent.trim()"
    ) == "❌ Synth Mismatch"
    # The tooltips are the only explanation of what reversed/mismatch test.
    assert page.evaluate(
        "document.getElementById('previewSynthMatchBtn').title"
    ) == "Combine current 3 target characters into a synthetic sign image"
    assert page.evaluate(
        "document.getElementById('previewSynthRevBtn').title"
    ) == "Combine in reversed order to test LTR/RTL direction"
    assert page.evaluate(
        "document.getElementById('previewSynthMismatchBtn').title"
    ) == "Combine with a wrong character to test NO verdict"
    # With a target and an exported model, none of them is disabled.
    assert page.evaluate("document.getElementById('previewSynthMatchBtn').disabled") is False


def assert_match_click_draws_a_yes_sign(page):
    _set_source(page, "upload")
    _stub_synth(page)
    assert page.evaluate("previewInputOn") is False

    page.evaluate("document.getElementById('previewSynthMatchBtn').click()")
    page.wait_for_function(
        "document.querySelector('#previewOutput .signmatch') !== null",
        timeout=10000,
    )
    page.wait_for_function(
        "document.querySelector('#previewOutput .signmatch').classList.contains('yes')",
        timeout=10000,
    )

    # The request carried the editor's target, not a stale saved one.
    body = page.evaluate("window.__synthBodies[0]")
    assert body["session"] == "test-session", body
    assert body["variant"] == "match", body
    assert body["sign_sequence"] == "A12", body
    assert body["reading_order"] == "ltr", body

    # The verdict banner and all three slot chips.
    assert page.evaluate(
        "document.querySelector('#previewOutput .signmatch-main').textContent"
    ).strip() == "🟢 YES — MATCH"
    assert page.evaluate(
        "document.querySelectorAll('#previewOutput .signmatch-slot').length"
    ) == 3
    assert page.evaluate(
        "document.querySelector('#previewOutput .signmatch-slot .got').textContent"
    ).strip() == "A"

    # The synthesized plate itself is on the preview canvas, not a blank pane.
    assert page.evaluate(
        "document.getElementById('previewImage').src"
    ) == "data:image/png;base64," + _TINY_PNG_B64

    # Clicking a pill leaves exactly the state a real upload leaves: upload
    # source, Input ON, the synth filename shown, and a note naming it.
    assert page.evaluate("STATE.preview_source") == "upload"
    assert page.evaluate("previewSource") == "upload"
    assert page.evaluate("previewInputOn") is True
    assert page.evaluate("document.getElementById('previewInputToggle').checked") is True
    assert page.evaluate(
        "document.getElementById('previewUploadName').textContent"
    ).strip() == "synth_A12_match.png"
    note = page.evaluate("document.getElementById('previewNote').textContent")
    assert "Synth (match): synth_A12_match.png" in note, note
    assert "🟢 YES — MATCH" in note, note
    assert "A→A 1→1 2→2" in note, note


def assert_mismatch_click_switches_to_no(page):
    page.wait_for_function(
        "document.querySelector('#previewOutput .signmatch') !== null",
        timeout=5000,
    )
    page.evaluate("document.getElementById('previewSynthMismatchBtn').click()")
    page.wait_for_function(
        "document.querySelector('#previewOutput .signmatch').classList.contains('no')",
        timeout=10000,
    )

    body = page.evaluate("window.__synthBodies[1]")
    assert body["variant"] == "mismatch", body
    assert body["sign_sequence"] == "A12", body

    assert page.evaluate(
        "document.querySelector('#previewOutput .signmatch-main').textContent"
    ).strip() == "🔴 NO — NOT MATCH"
    # The wrong slot is red; the two correct ones stay green.
    assert page.evaluate(
        "document.querySelectorAll('#previewOutput .signmatch-slot.bad').length"
    ) == 1
    assert page.evaluate(
        "document.querySelectorAll('#previewOutput .signmatch-slot.ok').length"
    ) == 2
    assert page.evaluate(
        "document.getElementById('previewUploadName').textContent"
    ).strip() == "synth_A12_mismatch.png"
    note = page.evaluate("document.getElementById('previewNote').textContent")
    assert "Synth (mismatch): synth_A12_mismatch.png" in note, note
    assert "🔴 NO — NOT MATCH" in note, note


def main() -> int:
    html = build_spa_html(_boot_payload())
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

            run("synth pills render in the upload source", assert_pills_render_in_upload_source, page)
            run("Synth Match draws a YES sign", assert_match_click_draws_a_yes_sign, page)
            run("Synth Mismatch switches to NO", assert_mismatch_click_switches_to_no, page)

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
