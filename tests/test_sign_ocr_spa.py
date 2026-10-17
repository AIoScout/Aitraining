"""Automated checks for the road-sign sequence UI and MATCH banner in the SPA.

The workspace is JavaScript emitted from an f-string, so there is nothing to
import.  `spa_harness` evaluates that f-string and serves the resulting page on
a real localhost origin, and these tests drive it in Chrome through Playwright
— the same browser the app runs in.  Every assertion therefore exercises the
shipped code, not a paraphrase of it.

Covered:
  normalisation   the JS mirror agrees with ocr_preprocess, input for input
  sequence bar    three slots, dedup note, missing-class warning, read order
  editor modal    open/close, per-slot edits, ◀/▶ reordering, disabled arrows
  banner          YES / NO / idle, slot chips, RTL order, failing-slot reasons
  wiring          the predict request carries sign_seq + reading; the menu item
                  and the bar appear only in a sign project; a settings reset
                  keeps the target

Requires Playwright with a system Chrome:
    .venv/bin/python tests/test_sign_ocr_spa.py
"""
from __future__ import annotations

import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, Path(__file__).resolve().parent):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import ocr_preprocess
from spa_harness import STUB_FETCH, boot_payload, build_spa_html, serve

from playwright.sync_api import sync_playwright


# ── normalisation parity ──────────────────────────────────────────────────
NORMALISE_CASES = [
    "a1a", "A 1 A", "AB", "ABCD", "", "?", "???", "1-2-3", " x ",
    ["a", "1", "a"], ["ab", "cd", "ef"], ["", None, " "], [], None,
    ["a"], ["1", "2"], "a1a ",
]


def assert_normalisation_matches_python(page):
    """The JS mirror must agree with ocr_preprocess on every case."""
    js = page.evaluate(
        "(cases) => cases.map((c) => normalizeSignSequence(c))", NORMALISE_CASES
    )
    for case, got in zip(NORMALISE_CASES, js):
        want = ocr_preprocess.normalize_sign_sequence(case)
        assert got == want, f"normalizeSignSequence({case!r}): JS {got!r} != Python {want!r}"


def assert_js_shape_rules(page):
    assert page.evaluate("JSON.stringify(signSequence())") == '["A","1","A"]'
    # Dedup: a repeated character is ONE class, the sequence still holds both.
    assert page.evaluate("JSON.stringify(signSequenceClasses())") == '["A","1"]'
    assert page.evaluate("signSequence().length") == 3
    assert page.evaluate("signSequenceIsSet()") is True


# ── sequence bar ──────────────────────────────────────────────────────────
def assert_bar_renders_the_target(page):
    n = page.evaluate("document.querySelectorAll('#signSeqBar .signseq-slot').length")
    assert n == 3, f"expected 3 slots, got {n}"
    texts = page.evaluate(
        "Array.from(document.querySelectorAll('#signSeqBar .signseq-slot')).map((e) => e.textContent)"
    )
    assert texts == ["A", "1", "A"]
    # First child of the column: the target is above the class cards.
    assert page.evaluate(
        "document.getElementById('classes').firstElementChild.id"
    ) == "signSeqBar"
    note = page.evaluate("document.getElementById('signSeqBar').textContent")
    assert "Classes trained: A, 1." in note
    # Every character has a class in this fixture, so no warning.
    assert "No class yet" not in note


def assert_bar_warns_about_untrained_characters(page):
    page.evaluate(
        "() => { const c = Object.assign({}, STATE.train_cfg, {sign_sequence: ['A','1','B']});"
        " STATE.train_cfg = c; renderSignSequenceBar(); }"
    )
    note = page.evaluate("document.getElementById('signSeqBar').textContent")
    assert "No class yet for B" in note, note
    texts = page.evaluate(
        "Array.from(document.querySelectorAll('#signSeqBar .signseq-slot')).map((e) => e.textContent)"
    )
    assert texts == ["A", "1", "B"]
    page.evaluate(
        "() => { const c = Object.assign({}, STATE.train_cfg, {sign_sequence: ['A','1','A']});"
        " STATE.train_cfg = c; renderSignSequenceBar(); }"
    )


def assert_bar_says_when_no_target_is_set(page):
    page.evaluate(
        "() => { const c = Object.assign({}, STATE.train_cfg, {sign_sequence: null});"
        " STATE.train_cfg = c; renderSignSequenceBar(); }"
    )
    note = page.evaluate("document.getElementById('signSeqBar').textContent")
    assert "No target set yet" in note, note
    # An unset slot is drawn as EMPTY rather than as a literal '?'.
    empty = page.evaluate("document.querySelectorAll('#signSeqBar .signseq-slot.empty').length")
    assert empty == 3
    page.evaluate(
        "() => { const c = Object.assign({}, STATE.train_cfg, {sign_sequence: ['A','1','A']});"
        " STATE.train_cfg = c; renderSignSequenceBar(); }"
    )


# ── editor modal ──────────────────────────────────────────────────────────
def assert_modal_opens_with_the_sequence(page):
    page.evaluate("openSignCharModal()")
    assert page.evaluate(
        "getComputedStyle(document.getElementById('signCharModal')).display"
    ) == "flex"
    values = page.evaluate(
        "Array.from(document.querySelectorAll('#signCharModal .signchar-input')).map((e) => e.value)"
    )
    assert values == ["A", "1", "A"], values
    # Arrows are disabled at the ends, not unbound: the first slot has no
    # left neighbour and saying so is clearer than a button that does nothing.
    disabled = page.evaluate(
        "Array.from(document.querySelectorAll('#signCharModal .signchar-move')).map("
        "  (m) => [m.children[0].disabled, m.children[1].disabled])"
    )
    assert disabled == [[True, False], [False, False], [False, True]], disabled


def assert_modal_reorders_by_swap(page):
    # Second slot's ◀ swaps it with the first: A,1,A -> 1,A,A.
    page.evaluate("document.querySelectorAll('#signCharModal .signchar-move')[1].children[0].click()")
    assert page.evaluate("JSON.stringify(signSequence())") == '["1","A","A"]'
    # The bar follows immediately — the target is one fact, not two views.
    texts = page.evaluate(
        "Array.from(document.querySelectorAll('#signSeqBar .signseq-slot')).map((e) => e.textContent)"
    )
    assert texts == ["1", "A", "A"]
    # ▶ on the first slot puts it back.
    page.evaluate("document.querySelectorAll('#signCharModal .signchar-move')[0].children[1].click()")
    assert page.evaluate("JSON.stringify(signSequence())") == '["A","1","A"]'


def assert_modal_edits_and_keeps_other_slots(page):
    # Type into slot 1 and then slot 3 without reopening: the second edit must
    # not discard the first, which is what the draft exists for.
    page.evaluate(
        """() => {
          const inputs = document.querySelectorAll('#signCharModal .signchar-input');
          inputs[0].value = 'x';
          inputs[0].dispatchEvent(new Event('input'));
        }"""
    )
    page.evaluate(
        """() => {
          const inputs = document.querySelectorAll('#signCharModal .signchar-input');
          inputs[2].value = '9';
          inputs[2].dispatchEvent(new Event('input'));
        }"""
    )
    assert page.evaluate("JSON.stringify(signSequence())") == '["X","1","9"]'


def assert_modal_clears_a_slot_to_unset(page):
    page.evaluate(
        """() => {
          const inputs = document.querySelectorAll('#signCharModal .signchar-input');
          inputs[1].value = '';
          inputs[1].dispatchEvent(new Event('input'));
        }"""
    )
    assert page.evaluate("JSON.stringify(signSequence())") == '["X","?","9"]'
    assert page.evaluate("signSequenceIsSet()") is True


def assert_modal_close_keeps_the_edit(page):
    page.evaluate("closeSignCharModal()")
    assert page.evaluate(
        "getComputedStyle(document.getElementById('signCharModal')).display"
    ) == "none"
    assert page.evaluate("JSON.stringify(signSequence())") == '["X","?","9"]'


def assert_dedup_adds_only_missing_classes(page):
    # The modal's add button is driven by signSequenceClasses(): a sequence
    # naming one character twice must ask the server for one name, not two.
    asked = page.evaluate(
        """() => {
          STATE.train_cfg = Object.assign({}, STATE.train_cfg, {sign_sequence: ['Q','7','Q']});
          let sent = null;
          const realFetch = window.fetch;
          window.fetch = async (url, opts) => {
            if (String(url).includes('/classes/add_named')) sent = JSON.parse(opts.body).names;
            return realFetch(url, opts);
          };
          openSignCharModal();
          const btns = document.querySelectorAll('#signCharModal .signseq-btn.primary');
          btns[0].click();
          window.fetch = realFetch;
          return sent;
        }"""
    )
    assert asked == ["Q", "7"], asked
    page.evaluate(
        "() => { STATE.train_cfg = Object.assign({}, STATE.train_cfg, {sign_sequence: ['A','1','A']});"
        " closeSignCharModal(); renderSignSequenceBar(); }"
    )


# ── MATCH banner ──────────────────────────────────────────────────────────
def _match_payload(**overrides):
    payload = {
        "sign_ocr": True,
        "sign_sequence": ["A", "1", "A"],
        "reading_order": "ltr",
        "slots": [
            {"index": 0, "char": "A", "label": "A", "prob": 0.94, "match": True,
             "background": False, "empty": False, "box": [10, 30, 21, 70]},
            {"index": 1, "char": "1", "label": "1", "prob": 0.88, "match": True,
             "background": False, "empty": False, "box": [35, 30, 46, 70]},
            {"index": 2, "char": "A", "label": "A", "prob": 0.91, "match": True,
             "background": False, "empty": False, "box": [60, 30, 71, 70]},
        ],
        "slot_images_b64": ["", "", ""],
        "match": True,
        "match_text": "🟢 YES — MATCH",
        "match_min_prob": 0.6,
        "sign_box": [10, 30, 71, 70],
        "labels": ["A → A", "1 → 1", "A → A"],
        "probs": [0.94, 0.88, 0.91],
        "top_label": "🟢 YES — MATCH",
        "top_prob": 0.88,
    }
    payload.update(overrides)
    return payload


def _render_banner(page, payload):
    page.evaluate("(p) => renderSignMatch(p)", payload)


def assert_banner_says_yes(page):
    _render_banner(page, _match_payload())
    assert page.evaluate("!!document.querySelector('#previewOutput .signmatch.yes')")
    assert page.evaluate("document.querySelector('#previewOutput .signmatch-main').textContent") \
        == "🟢 YES — MATCH"
    got = page.evaluate(
        "Array.from(document.querySelectorAll('#previewOutput .signmatch-slot .got')).map((e) => e.textContent)"
    )
    assert got == ["A", "1", "A"]
    assert page.evaluate(
        "Array.from(document.querySelectorAll('#previewOutput .signmatch-slot')).every((e) => e.classList.contains('ok'))"
    )
    # No explanation needed when the answer is yes.
    assert page.evaluate("document.querySelectorAll('#previewOutput .signmatch-hint').length") == 0


def assert_banner_says_no_and_explains(page):
    slots = _match_payload()["slots"]
    slots[2] = dict(slots[2], label="B", prob=0.83, match=False, char="A")
    _render_banner(page, _match_payload(
        slots=slots, match=False, match_text="🔴 NO — NOT MATCH",
        labels=["A → A", "1 → 1", "A → B"],
    ))
    assert page.evaluate("!!document.querySelector('#previewOutput .signmatch.no')")
    assert page.evaluate("document.querySelector('#previewOutput .signmatch-main').textContent") \
        == "🔴 NO — NOT MATCH"
    hint = page.evaluate("document.querySelector('#previewOutput .signmatch-hint').textContent")
    # The reason names the slot, what it read, and the bar it failed to clear.
    assert 'slot 3 read "B" at 83%' in hint, hint
    assert "60%" in hint, hint


def assert_banner_explains_a_background_slot(page):
    slots = _match_payload()["slots"]
    slots[1] = dict(slots[1], label="_background", prob=0.7, match=False, background=True)
    _render_banner(page, _match_payload(slots=slots, match=False))
    hint = page.evaluate("document.querySelector('#previewOutput .signmatch-hint').textContent")
    assert "slot 2 came back as background" in hint, hint


def assert_banner_idle_without_a_target(page):
    _render_banner(page, _match_payload(
        sign_sequence=["?", "?", "?"], match=False, labels=["? → ", "? → ", "? → "],
        slots=[{"index": i, "char": "?", "label": "", "prob": 0.0, "match": False}
               for i in range(3)],
    ))
    assert page.evaluate("!!document.querySelector('#previewOutput .signmatch.idle')")
    main = page.evaluate("document.querySelector('#previewOutput .signmatch-main').textContent")
    # Not a red NO: there was no question to answer.
    assert main == "◻ SET THE TARGET SIGN", main


def assert_banner_idle_without_a_model(page):
    _render_banner(page, _match_payload(
        match=False,
        slots=[{"index": i, "char": c, "label": "", "prob": 0.0, "match": False}
               for i, c in enumerate(["A", "1", "A"])],
        labels=["A → —", "1 → —", "A → —"],
    ))
    assert page.evaluate("!!document.querySelector('#previewOutput .signmatch.idle')")
    main = page.evaluate("document.querySelector('#previewOutput .signmatch-main').textContent")
    assert main == "◻ NO MODEL YET", main
    # And it points at the missing model, not at the (already set) target.
    hint = page.evaluate("document.querySelector('#previewOutput .signmatch-hint').textContent")
    assert "Train, then come back" in hint, hint


def assert_banner_keeps_slots_in_reading_order(page):
    # The server already ordered the slots by reading direction.  Sorting them
    # by their pixel box here would silently undo an RTL project: this payload
    # reads B,1,A while the frame runs A,1,B left to right.
    boxes_ltr = [(10, 30, 21, 70), (35, 30, 46, 70), (60, 30, 71, 70)]
    chars = ["B", "1", "A"]
    rtl = [
        {"index": i, "char": chars[i], "label": chars[i], "prob": 0.9, "match": True,
         "background": False, "empty": False, "box": boxes_ltr[2 - i]}
        for i in range(3)
    ]
    _render_banner(page, _match_payload(
        slots=rtl, sign_sequence=chars, reading_order="rtl", labels=chars,
    ))
    got = page.evaluate(
        "Array.from(document.querySelectorAll('#previewOutput .signmatch-slot')).map("
        "  (e) => e.querySelector('.want').textContent)"
    )
    # Frame order is A,1,B — the payload order is what must reach the screen.
    assert got == ["WANT B", "WANT 1", "WANT A"], got


def assert_preview_output_dispatches(page):
    _render_banner(page, _match_payload())
    assert page.evaluate("!!document.querySelector('#previewOutput .signmatch')")
    # A non-sign payload draws the ordinary score bars instead.
    page.evaluate(
        "(d) => renderPreviewOutput(d)",
        {"labels": ["A", "B"], "probs": [0.9, 0.1], "top_label": "A", "top_prob": 0.9},
    )
    assert page.evaluate("!document.querySelector('#previewOutput .signmatch')")
    assert page.evaluate("document.querySelectorAll('#previewOutput .out-row').length") == 2


# ── wiring ────────────────────────────────────────────────────────────────
def assert_predict_request_carries_the_target(page):
    url = page.evaluate(
        """async () => {
          window.__fetches = [];
          previewInputOn = true;
          previewPredictToken = 4242;
          await refreshPreviewPrediction(4242);
          return window.__fetches.find((u) => u.includes('/preview/predict?')) || '';
        }"""
    )
    assert "sign_seq=A1A" in url, url
    assert "reading=ltr" in url, url
    # And an RTL project asks for RTL.
    url_rtl = page.evaluate(
        """async () => {
          STATE.train_cfg = Object.assign({}, STATE.train_cfg, {reading_order: 'rtl'});
          window.__fetches = [];
          previewPredictToken = 4243;
          await refreshPreviewPrediction(4243);
          const u = window.__fetches.find((x) => x.includes('/preview/predict?')) || '';
          STATE.train_cfg = Object.assign({}, STATE.train_cfg, {reading_order: 'ltr'});
          return u;
        }"""
    )
    assert "reading=rtl" in url_rtl, url_rtl


def assert_menu_item_and_bar_follow_the_project_type(page):
    page.evaluate("syncSignNavVisibility()")
    assert page.evaluate("document.getElementById('navSignChars').style.display") == ""
    # In a plain OCR project the bar and the menu item are both gone.
    page.evaluate(
        """() => {
          STATE.project_type = 'ocr';
          syncSignNavVisibility();
          renderSignSequenceBar();
        }"""
    )
    assert page.evaluate("document.getElementById('navSignChars').style.display") == "none"
    assert page.evaluate("!document.getElementById('signSeqBar')")
    # Back to a sign project: the bar is remounted as the column's first child.
    page.evaluate(
        """() => {
          STATE.project_type = 'sign_ocr';
          syncSignNavVisibility();
          renderSignSequenceBar(document.getElementById('classes'));
        }"""
    )
    assert page.evaluate("!!document.getElementById('signSeqBar')")


def assert_settings_reset_keeps_the_target(page):
    kept = page.evaluate(
        """() => {
          STATE.train_cfg = Object.assign({}, STATE.train_cfg, {sign_sequence: ['A','1','A']});
          resetTrainCfg();
          return signSequence();
        }"""
    )
    assert kept == ["A", "1", "A"], kept


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
            page.wait_for_function("typeof renderSignSequenceBar === 'function'")
            page.wait_for_timeout(300)

            run("normalisation matches Python", assert_normalisation_matches_python, page)
            run("sequence helpers", assert_js_shape_rules, page)
            run("bar renders the target", assert_bar_renders_the_target, page)
            run("bar warns about untrained chars", assert_bar_warns_about_untrained_characters, page)
            run("bar says when no target is set", assert_bar_says_when_no_target_is_set, page)
            run("modal opens with the sequence", assert_modal_opens_with_the_sequence, page)
            run("modal reorders by swap", assert_modal_reorders_by_swap, page)
            run("modal edits keep other slots", assert_modal_edits_and_keeps_other_slots, page)
            run("modal clears a slot", assert_modal_clears_a_slot_to_unset, page)
            run("modal close keeps the edit", assert_modal_close_keeps_the_edit, page)
            run("dedup adds only missing classes", assert_dedup_adds_only_missing_classes, page)
            run("banner says YES", assert_banner_says_yes, page)
            run("banner says NO and explains", assert_banner_says_no_and_explains, page)
            run("banner explains a background slot", assert_banner_explains_a_background_slot, page)
            run("banner idle without a target", assert_banner_idle_without_a_target, page)
            run("banner idle without a model", assert_banner_idle_without_a_model, page)
            run("banner keeps reading order", assert_banner_keeps_slots_in_reading_order, page)
            run("preview output dispatches", assert_preview_output_dispatches, page)
            run("predict request carries the target", assert_predict_request_carries_the_target, page)
            run("menu item and bar follow the type", assert_menu_item_and_bar_follow_the_project_type, page)
            run("settings reset keeps the target", assert_settings_reset_keeps_the_target, page)

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
