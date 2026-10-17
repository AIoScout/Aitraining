"""Automated checks for the road-sign sequence UI and MATCH banner in the SPA.

The workspace is JavaScript emitted from an f-string, so there is nothing to
import.  `spa_harness` evaluates that f-string and serves the resulting page on
a real localhost origin, and these tests drive it in Chrome through Playwright
— the same browser the app runs in.  Every assertion therefore exercises the
shipped code, not a paraphrase of it.

Covered:
  normalisation   the JS mirror agrees with ocr_preprocess, input for input
  inline editor   three <input> slots, no modal: typing upper-cases and
                  auto-advances, ◀/▶ swap neighbours, class cards follow the
                  sequence in order (including create/delete/reorder) with no
                  "Add as classes" button
  fresh project   a new sign project starts with three inline slots and three
                  matching cards; unrelated/placeholder classes disappear
  banner          YES / NO / idle, slot chips, RTL order, failing-slot reasons
  wiring          the predict request carries sign_seq + reading; the Edit
                  menu's character editor is OCR-only; a settings reset keeps
                  the target

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


# ── inline sequence bar ───────────────────────────────────────────────────
def _wait_sync(page):
    """Let the debounced /classes/sync_ocr round-trip land."""
    page.wait_for_timeout(250)


def _reset_sign_state(page):
    page.evaluate(
        """() => {
          STATE.project_type = 'sign_ocr';
          STATE.classes = ['A', '1'];
          STATE.counts = {A: 3, 1: 2};
          STATE.sample_previews = {};
          STATE.processed_previews = {};
          STATE.class_preprocess = {};
          STATE.sample_preprocess = {};
          STATE.train_cfg = Object.assign({}, STATE.train_cfg, {
            sign_sequence: ['A', '1', 'A'], reading_order: 'ltr'
          });
          signSyncSavedTarget = '';
          render();
        }"""
    )
    _wait_sync(page)


def _slot_values(page):
    return page.evaluate(
        "Array.from(document.querySelectorAll('#signSeqBar .signseq-slot')).map((e) => e.value)"
    )


def _class_titles(page):
    return page.evaluate(
        "Array.from(document.querySelectorAll('#classes .class-card .class-title')).map("
        "  (e) => e.firstChild ? String(e.firstChild.textContent).trim() : e.textContent.trim())"
    )


def assert_bar_renders_the_target(page):
    n = page.evaluate("document.querySelectorAll('#signSeqBar .signseq-slot').length")
    assert n == 3, f"expected 3 slots, got {n}"
    tags = page.evaluate(
        "Array.from(document.querySelectorAll('#signSeqBar .signseq-slot')).map((e) => e.tagName)"
    )
    assert tags == ["INPUT", "INPUT", "INPUT"], tags
    assert _slot_values(page) == ["A", "1", "A"]
    # First child of the column: the target is above the class cards.
    assert page.evaluate(
        "document.getElementById('classes').firstElementChild.id"
    ) == "signSeqBar"
    note = page.evaluate("document.getElementById('signSeqBar').textContent")
    assert "Step 1: Type the 3 sign characters in order below" in note, note
    assert "Step 2: Capture samples for each character card" in note, note
    assert "Step 3: Train & Preview (YES/NO)." in note, note
    # The old advise-and-button surface is gone, not merely hidden.
    assert "No class yet" not in note, note
    assert page.evaluate("!document.querySelector('#signSeqBar .signseq-btn')")
    assert page.evaluate("!document.getElementById('signCharModal')")
    assert page.evaluate("typeof openSignCharModal") == "undefined"


def assert_typing_auto_advances_and_syncs(page):
    _reset_sign_state(page)
    page.evaluate("window.__syncBodies = []")

    # Type over slot 1: lower-case input is upper-cased in place, the sequence
    # changes, and the caret moves on to slot 2 with its text selected.
    page.locator("#signSeqSlot-0").click()
    page.keyboard.type("x")
    _wait_sync(page)
    assert _slot_values(page) == ["X", "1", "A"], _slot_values(page)
    assert page.evaluate("JSON.stringify(signSequence())") == '["X","1","A"]'
    assert page.evaluate("document.activeElement.id") == "signSeqSlot-1"
    assert page.evaluate("JSON.stringify(STATE.classes)") == '["X","1","A"]'
    assert _class_titles(page) == ["X", "1", "A"]
    # The new character starts empty; it must not inherit A's samples merely
    # because it occupies the index A used to occupy.
    counts = page.evaluate("STATE.counts")
    assert counts == {"X": 0, "1": 2, "A": 3}, counts

    # The keyboard is already on slot 2, and its character is selected: typing
    # replaces it instead of being refused by maxlength.
    page.keyboard.type("9")
    _wait_sync(page)
    assert page.evaluate("JSON.stringify(signSequence())") == '["X","9","A"]'
    assert page.evaluate("document.activeElement.id") == "signSeqSlot-2"
    assert page.evaluate("JSON.stringify(STATE.classes)") == '["X","9","A"]'
    assert _class_titles(page) == ["X", "9", "A"]

    bodies = page.evaluate("window.__syncBodies")
    assert ["X", "9", "A"] in bodies, bodies
    # No character selector and no "Add as classes" was needed to get here.
    assert page.evaluate("!document.querySelector('.charsel')")
    assert page.evaluate("!document.querySelector('.addclass')")


def assert_arrows_swap_and_sync_classes(page):
    _reset_sign_state(page)
    page.evaluate("window.__syncBodies = []")

    # Slot 1's ▶ swaps it with slot 2: A,1,A -> 1,A,A.
    page.locator("#signSeqRight-0").click()
    _wait_sync(page)
    assert page.evaluate("JSON.stringify(signSequence())") == '["1","A","A"]'
    assert _slot_values(page) == ["1", "A", "A"]
    assert page.evaluate("JSON.stringify(STATE.classes)") == '["1","A"]'
    assert _class_titles(page) == ["1", "A"]
    # Repeated character: one card, badged with both positions.
    assert page.evaluate(
        "Array.from(document.querySelectorAll('#classes .signslot-badge')).map((e) => e.textContent)"
    ) == ["Slots 2 & 3"]
    assert ["1", "A"] in page.evaluate("window.__syncBodies"), page.evaluate("window.__syncBodies")

    # The edge buttons say so rather than doing nothing.
    assert page.evaluate("document.getElementById('signSeqLeft-0').disabled") is True
    assert page.evaluate("document.getElementById('signSeqRight-2').disabled") is True


def assert_unrelated_classes_are_cleaned_up(page):
    # This is the payload an old project can arrive with: two untouched
    # placeholders plus characters from a previous sign, none of which the
    # new target names.
    page.evaluate(
        """() => {
          STATE.project_type = 'sign_ocr';
          STATE.classes = ['Class 1', 'N', 'Y'];
          STATE.counts = {'Class 1': 0, N: 0, Y: 0};
          STATE.sample_previews = {};
          STATE.processed_previews = {};
          STATE.class_preprocess = {};
          STATE.sample_preprocess = {};
          STATE.train_cfg = Object.assign({}, STATE.train_cfg, {
            sign_sequence: ['1', '2', 'A'], reading_order: 'ltr'
          });
          signSyncSavedTarget = '';
          window.__syncBodies = [];
          render();
        }"""
    )
    _wait_sync(page)
    assert page.evaluate("JSON.stringify(STATE.classes)") == '["1","2","A"]'
    assert _class_titles(page) == ["1", "2", "A"]
    assert ["1", "2", "A"] in page.evaluate("window.__syncBodies")
    # A sign card has no manual add/delete/rename/drag controls at all.
    assert page.evaluate("!document.querySelector('.addclass')")
    assert page.evaluate("document.querySelectorAll('#classes .tm-delete-class').length") == 0
    assert page.evaluate("document.querySelectorAll('#classes .tm-rename-btn').length") == 0
    assert page.evaluate("document.querySelectorAll('#classes .class-drag-handle').length") == 0


def assert_fresh_project_starts_with_three_cards(page):
    page.evaluate(
        """() => {
          STATE.project_type = 'sign_ocr';
          STATE.classes = ['Class 1', 'Class 2'];
          STATE.counts = {'Class 1': 0, 'Class 2': 0};
          STATE.sample_previews = {};
          STATE.processed_previews = {};
          STATE.class_preprocess = {};
          STATE.sample_preprocess = {};
          STATE.train_cfg = Object.assign({}, STATE.train_cfg, {
            sign_sequence: null, reading_order: 'ltr'
          });
          signSyncSavedTarget = '';
          window.__syncBodies = [];
          render();
        }"""
    )
    _wait_sync(page)
    assert page.evaluate("JSON.stringify(signSequence())") == '["A","B","C"]'
    assert _slot_values(page) == ["A", "B", "C"]
    assert page.evaluate("JSON.stringify(STATE.classes)") == '["A","B","C"]'
    assert _class_titles(page) == ["A", "B", "C"]
    assert ["A", "B", "C"] in page.evaluate("window.__syncBodies")


# ── initial project load ──────────────────────────────────────────────────
def assert_boot_cleans_unrelated_classes(page):
    """A project arriving with stale classes is fixed by the first render."""
    page.wait_for_timeout(500)
    assert page.evaluate("JSON.stringify(STATE.classes)") == '["1","2","A"]'
    assert _class_titles(page) == ["1", "2", "A"]
    assert _slot_values(page) == ["1", "2", "A"]


def assert_fresh_boot_has_three_cards(page):
    """A brand-new sign project needs zero clicks to become usable."""
    page.wait_for_timeout(500)
    assert page.evaluate("JSON.stringify(signSequence())") == '["A","B","C"]'
    assert _slot_values(page) == ["A", "B", "C"]
    assert _class_titles(page) == ["A", "B", "C"]


def assert_bar_says_when_no_target_is_set(page):
    # A cleared target is visible as three empty dashed slots.  It is not a
    # warning, and it must not delete the project's existing classes.
    page.evaluate(
        """() => {
          STATE.project_type = 'sign_ocr';
          STATE.classes = ['A', '1'];
          STATE.counts = {A: 3, 1: 2};
          STATE.train_cfg = Object.assign({}, STATE.train_cfg, {sign_sequence: null});
          render();
        }"""
    )
    _wait_sync(page)
    note = page.evaluate("document.getElementById('signSeqBar').textContent")
    assert "No target set yet" not in note, note
    assert page.evaluate("document.querySelectorAll('#signSeqBar .signseq-slot.empty').length") == 3
    assert _slot_values(page) == ["", "", ""]
    assert page.evaluate("JSON.stringify(STATE.classes)") == '["A","1"]'
    _reset_sign_state(page)


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
    _reset_sign_state(page)
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
    page.evaluate("syncOcrPanelVisibility()")
    # A sign project edits its characters in the bar; the Edit-menu character
    # editor belongs to basic OCR and there is no second sign-characters item.
    assert page.evaluate("document.getElementById('navEditCharacters').style.display") == "none"
    assert page.evaluate("!document.getElementById('navSignChars')")
    # In a plain OCR project the bar is gone and the character editor returns.
    page.evaluate(
        """() => {
          STATE.project_type = 'ocr';
          syncOcrPanelVisibility();
          renderSignSequenceBar();
        }"""
    )
    assert page.evaluate("document.getElementById('navEditCharacters').style.display") == ""
    assert page.evaluate("!document.getElementById('signSeqBar')")
    # Back to a sign project: the bar is remounted as the column's first child.
    page.evaluate(
        """() => {
          STATE.project_type = 'sign_ocr';
          syncOcrPanelVisibility();
          renderSignSequenceBar(document.getElementById('classes'));
        }"""
    )
    assert page.evaluate("!!document.getElementById('signSeqBar')")
    assert page.evaluate("document.getElementById('navEditCharacters').style.display") == "none"


def assert_settings_reset_keeps_the_target(page):
    _reset_sign_state(page)
    kept = page.evaluate(
        """() => {
          resetTrainCfg();
          return signSequence();
        }"""
    )
    assert kept == ["A", "1", "A"], kept


def main() -> int:
    base_cfg = boot_payload()["train_cfg"]
    html = build_spa_html(boot_payload())
    # Two more real boot payloads: one that arrives with stale classes, and a
    # fresh project.  These exercise the path the app actually takes on load,
    # not a test-only render() call.
    clean_html = build_spa_html(boot_payload(
        classes=["Class 1", "N", "Y"],
        counts={"Class 1": 0, "N": 0, "Y": 0},
        train_cfg=dict(base_cfg, sign_sequence=["1", "2", "A"]),
    ))
    fresh_html = build_spa_html(boot_payload(
        classes=["Class 1", "Class 2"],
        counts={"Class 1": 0, "Class 2": 0},
        train_cfg=dict(base_cfg, sign_sequence=None),
    ))
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

    with serve(html) as url, serve(clean_html) as clean_url, serve(fresh_html) as fresh_url:
        with sync_playwright() as p:
            browser = p.chromium.launch(channel="chrome", headless=True)
            page = browser.new_page()
            errors = []
            page.on("pageerror", lambda e: errors.append(str(e)))
            page.add_init_script(STUB_FETCH)
            page.goto(url)
            page.wait_for_function("typeof renderSignSequenceBar === 'function'")
            page.wait_for_timeout(300)

            page2 = browser.new_page()
            page2.on("pageerror", lambda e: errors.append("stale-boot: " + str(e)))
            page2.add_init_script(STUB_FETCH)
            page2.goto(clean_url)
            page2.wait_for_function("typeof renderSignSequenceBar === 'function'")

            page3 = browser.new_page()
            page3.on("pageerror", lambda e: errors.append("fresh-boot: " + str(e)))
            page3.add_init_script(STUB_FETCH)
            page3.goto(fresh_url)
            page3.wait_for_function("typeof renderSignSequenceBar === 'function'")
            # Record the names each auto-sync sends, without changing what the
            # stub returns (so a test sees the request the SPA really built).
            page.evaluate(
                """() => {
                  window.__syncBodies = [];
                  const real = window.fetch;
                  window.fetch = async (url, opts) => {
                    if (String(url).includes('/classes/sync_ocr') && opts && opts.body) {
                      try { window.__syncBodies.push(JSON.parse(opts.body).names); } catch (e) {}
                    }
                    return real(url, opts);
                  };
                }"""
            )

            run("normalisation matches Python", assert_normalisation_matches_python, page)
            run("sequence helpers", assert_js_shape_rules, page)
            run("bar renders the target inline", assert_bar_renders_the_target, page)
            run("boot cleans unrelated classes", assert_boot_cleans_unrelated_classes, page2)
            run("fresh boot has three cards", assert_fresh_boot_has_three_cards, page3)
            run("typing auto-advances and syncs", assert_typing_auto_advances_and_syncs, page)
            run("arrows swap and sync classes", assert_arrows_swap_and_sync_classes, page)
            run("unrelated classes are cleaned up", assert_unrelated_classes_are_cleaned_up, page)
            run("fresh project starts with three cards", assert_fresh_project_starts_with_three_cards, page)
            run("bar says when no target is set", assert_bar_says_when_no_target_is_set, page)
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
