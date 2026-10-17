"""Automated checks for the workspace's interactive tour (`window.tmTour`).

The tour is JavaScript emitted from the workspace f-string, so there is
nothing to import.  `spa_harness` evaluates that f-string and serves the
resulting page on a real localhost origin, and this test drives it in the
same Chrome the app ships in — launched head-less from the command line, so
the check needs no test-only dependency (no Playwright, no Node).

What it covers:

  curricula     the guide selected for a project type is that project's own
                curriculum: 12 steps for `image`, 10 for `ocr`, 10 for
                `sign_ocr`, with three distinct step lists
  directional marks
                every targeted step paints `#tmTourArrow` aimed between the
                card and the ring, and every step that is WAITING for a
                press also paints the `#tmTourBeacon` on that very control;
                steps with no target carry neither
  trigger phase
                a step that documents a screen which is shut — the Advanced
                panel, the ⚙ Settings modal — spotlights the control that
                OPENS it first, and pressing that control re-seats the same
                step inside instead of teleporting the user there
  click-to-advance
                every step's cue is derived from the DOM target exactly the
                way `armTargetAdvance` derives it (off / click / input /
                disabled / phase), and using the spotlighted control really
                does advance `tmTour.current()` — a button (`#navSettings`),
                a chip (`.class-preprocess-chip`), a select inside a spotlit
                row, and the sequence-bar input a sign project starts on
  disabled guard
                a button that cannot be used yet (`#trainBtn` with no
                samples, `#exportBtn` with no model) reports "disabled"
                instead of trapping the user, does not advance when clicked,
                and says in the card what to do instead
  state-aware hooks
                no step leaves the global Settings modal open behind the
                tour, even on the steps whose own click opens it
  start page    the three project cards all render before any of them acts,
                and cancelling the Open Project picker is a silent no-op

Run it with the interpreter that can import the app's own modules:

    python3 tests/test_tour_spa.py
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, Path(__file__).resolve().parent):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from spa_harness import boot_payload, build_spa_html, serve

CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"

# The curriculum each project type must load.  Hard-coded on purpose: the
# number of steps is the thing a user notices, so it is the thing the test
# states, rather than something read back out of the source it checks.
EXPECTED_STEPS = {"image": 12, "ocr": 10, "sign_ocr": 10}

# Steps whose real interaction the test performs.  Deliberately short: these
# are the spec's own examples plus the two field kinds the tour treats
# differently.  Everything else is asserted on its cue alone, so a failure
# points at one control instead of at the whole walk.
REAL_CLICK = ("#navSettings", ".class-preprocess-chip", ".class-preprocess-mode-row")
REAL_CHANGE = ("#classPreprocessDark", "#signSeqBar")

# `#trainBtn` and `.tm-action-toolbar` are not clicked: the first would begin
# a real training run through the stubbed endpoint and the second opens a
# camera / device stream, neither of which a head-less browser has.  Both
# still assert their cue, which is what click-to-advance arms.
DRIVER = r"""
<script>
(function () {
  var SPEC = __SPEC__;
  var TYPES = __TYPES__;
  var REAL_CLICK = __REAL_CLICK__;
  var REAL_CHANGE = __REAL_CHANGE__;
  var OUT = [];
  var DONE = false;
  var ADV_SEL = 'button, [role="button"], a[href], input, select, textarea, ' +
    '.class-preprocess-chip, .chip, .tm-toolbar-btn, .tm-action-btn, .ocr-empty-cta';

  function finish() {
    if (DONE) return;
    DONE = true;
    try { window.stop(); } catch (e) {}
    for (var i = 1; i < 8000; i++) { window.clearTimeout(i); window.clearInterval(i); }
    var pre = document.createElement('pre');
    pre.id = 'TM_TEST_OUT';
    pre.textContent = 'TM_RESULT_JSON::' + JSON.stringify({
      steps: OUT,
      errors: window.__pageErrors || []
    });
    document.body.appendChild(pre);
  }

  function vis(el) {
    if (!el || !el.getClientRects().length) return false;
    var st = window.getComputedStyle(el);
    return st.display !== 'none' && st.visibility !== 'hidden' && Number(st.opacity) !== 0;
  }

  function firstVisible(selectors) {
    for (var i = 0; i < selectors.length; i++) {
      if (!selectors[i]) continue;
      var el = null;
      try { el = document.querySelector(selectors[i]); } catch (e) { el = null; }
      if (vis(el)) return { el: el, sel: selectors[i] };
    }
    return null;
  }

  function wait(ms) { return new Promise(function (r) { window.setTimeout(r, ms); }); }

  function describe(el) {
    if (!el) return null;
    return String(el.tagName || '') + '#' + String(el.id || '') +
      '.' + String(el.className || '').slice(0, 22);
  }

  function settingsOpen() {
    var modal = document.getElementById('globalSettingsModal');
    return !!(modal && modal.classList.contains('open'));
  }

  /* `trainAdvancedOpen` is a top-level binding of the workspace script,
     shared with this one: it is the same variable the curricula's own
     `phaseIf` reads, so the walk and the app agree on when the panel is
     collapsed. */
  function advancedOpen() {
    return (typeof trainAdvancedOpen !== 'undefined') && !!trainAdvancedOpen;
  }

  /* The screen a step documents is SHUT when its trigger is still needed:
     the ⚙ Settings modal closed, or the Advanced panel collapsed.  This is
     the `phaseIf` each curricula step declares, restated. */
  function screenShut(step) {
    if (!step.phaseTarget) return false;
    if (step.phaseTarget === '#advBtn') return !advancedOpen();
    if (step.phaseTarget === '#navSettings') return !settingsOpen();
    return false;
  }

  /* Mirrors `resolveStepTargets` exactly: the real target while its screen
     is up, the control that opens it while it is shut. */
  function resolveTargets(step) {
    var shut = screenShut(step);
    var found = shut ? null : firstVisible([step.target].concat(step.fallback || []));
    if (!found && step.phaseTarget) {
      var door = firstVisible([step.phaseTarget]);
      if (door) return { found: door, phase: true };
    }
    if (!found) found = firstVisible([step.target].concat(step.fallback || []));
    return { found: found, phase: false };
  }

  function marks() {
    try { return window.tmTour.marks(); } catch (e) { return null; }
  }

  /* The cue the tour is supposed to be showing, derived the same way
     `armTargetAdvance` derives it: from the resolved element, the named
     `advanceTarget`, and that control's own disabled / tag state.  The
     trigger phase (`phase`) is decided by `resolveTargets`, above. */
  function expectedCue(step, found) {
    if (!found) return 'off';
    var el = found.el;
    var control = el;
    if (step.advanceTarget) {
      var named = null;
      try { named = document.querySelector(step.advanceTarget); } catch (e) { named = null; }
      if (named && vis(named)) control = named;
    }
    if (control.disabled) return 'disabled';
    var tag = String(control.tagName || '').toUpperCase();
    return (tag === 'INPUT' || tag === 'SELECT' || tag === 'TEXTAREA') ? 'input' : 'click';
  }

  async function walkType(t) {
    try { STATE.project_type = t; } catch (e) {}
    window.tmTour.select();
    var steps = SPEC[t] || [];
    OUT.push({
      kind: 'total', type: t, projectType: window.tmTour.projectType(),
      total: window.tmTour.total(), parsed: steps.length,
      titles: window.tmTour.titles()
    });
    window.tmTour.start();
    await wait(140);

    var n = window.tmTour.total();
    for (var i = 0; i < n; i++) {
      window.tmTour.goTo(i);
      await wait(300);

      var step = steps[i] || {};
      var res = resolveTargets(step);
      var found = res.found;
      var el = found ? found.el : null;
      var control = el;
      if (el && !res.phase && step.advanceTarget) {
        var named = null;
        try { named = document.querySelector(step.advanceTarget); } catch (e) { named = null; }
        if (named && vis(named)) control = named;
      }

      var cueBox = document.getElementById('tmTourCue');
      var modal = document.getElementById('globalSettingsModal');
      /* Read while the step is on screen, before the action below moves the
         tour on: a step whose control lives inside the Settings modal needs
         it open, and every other step needs it closed — the state the user
         actually sees, not the state its successor's onEnter just set up. */
      var rec = {
        kind: 'step', type: t, i: i,
        title: String(step.title || '').slice(0, 46),
        hasTarget: !!step.target,
        target: step.target || null,
        sel: found ? found.sel : null,
        control: describe(control),
        phase: !!res.phase,
        expected: res.phase ? 'phase' : expectedCue(step, found),
        cue: window.tmTour.cue(),
        cueText: cueBox ? String(cueBox.textContent || '').slice(0, 70) : '',
        marks: marks(),
        action: 'none',
        advanced: null,
        settingsOpen: settingsOpen(),
        settingsExpected: !!(modal && el && modal.contains(el))
      };
      var want = rec.expected;
      var real = (want === 'click' && REAL_CLICK.indexOf(rec.sel) >= 0) ||
                 (want === 'input' && REAL_CHANGE.indexOf(rec.sel) >= 0);

      if (want === 'disabled') {
        /* A disabled button dispatches nothing, so this click is harmless by
           construction — and that is the point: it must not advance. */
        if (control && control.click) control.click();
        rec.action = 'click-disabled';
        await wait(450);
      } else if (want === 'phase') {
        /* The screen this step documents is shut, so the spotlight sits on
           the control that opens it.  Pressing it is the lesson: the tour
           must follow the user in WITHOUT moving off this step. */
        if (control && control.click) control.click();
        rec.action = 'click-door';
        await wait(700);
        rec.cueAfter = window.tmTour.cue();
        rec.indexAfter = window.tmTour.current();
        rec.settingsAfter = settingsOpen();
        rec.advancedAfter = advancedOpen();
        /* Where the re-seated spotlight landed: the step's OWN target, if
           the door really opened the screen it documents. */
        var after = resolveTargets(step);
        rec.selAfter = after.found ? after.found.sel : null;
        rec.marksAfter = marks();
      } else if (real && want === 'input') {
        if (control) {
          /* The value is left alone: a real user's change event is what the
             tour listens for, and the app's own handler then sees the value
             it already had, so the walk leaves no state behind. */
          control.dispatchEvent(new Event('input', { bubbles: true }));
          control.dispatchEvent(new Event('change', { bubbles: true }));
          rec.action = 'change';
          rec.target = describe(control);
        }
        await wait(700);
      } else if (real) {
        var hit = (el.matches && el.matches(ADV_SEL)) ? el : el.querySelector(ADV_SEL);
        if (hit) {
          hit.click();
          rec.action = 'click';
          rec.target = describe(hit);
        }
        await wait(500);
      } else {
        window.tmTour.next();
        rec.action = 'next';
        await wait(90);
      }
      rec.advanced = window.tmTour.current();
      OUT.push(rec);
    }
    window.tmTour.close();
    await wait(80);
  }

  async function run() {
    try {
      for (var i = 0; i < TYPES.length; i++) await walkType(TYPES[i]);
    } catch (e) {
      OUT.push({ kind: 'EXC', v: String((e && e.stack) || e) });
    }
    finish();
  }

  /* The workspace's own boot calls startTour() on a delay when the guide has
     not been seen, and the harness localStorage is empty on every run, so
     that auto-start would fire in the middle of this walk.  Quiet it here —
     the explicit tmTour.start() below forces the tour open regardless. */
  try {
    ['image', 'ocr', 'sign_ocr'].forEach(function (t) {
      window.localStorage.setItem('tm_tour_autoskip_' + t + '_v1', '1');
      window.localStorage.setItem('tm_tour_seen_' + t + '_v1', '1');
    });
  } catch (e) {}

  if (document.readyState === 'complete') window.setTimeout(run, 400);
  else window.addEventListener('load', function () { window.setTimeout(run, 400); });
})();
</script>
"""


# ── reading the curricula back out of the page ───────────────────────────
_CURRICULUM_BLOCK = re.compile(
    r"var (?P<name>IMAGE_STEPS|OCR_STEPS|SIGN_OCR_STEPS) = \[(?P<body>.*?)\n  \];",
    re.S,
)
_BY_NAME = {"IMAGE_STEPS": "image", "OCR_STEPS": "ocr", "SIGN_OCR_STEPS": "sign_ocr"}


def _quoted(block: str, key: str) -> str:
    m = re.search(r"(?<![A-Za-z])" + key + r":\s*'((?:[^'\\]|\\.)*)'", block)
    return m.group(1) if m else ""


def _list(block: str, key: str):
    m = re.search(r"(?<![A-Za-z])" + key + r":\s*\[(.*?)\]", block, re.S)
    if not m:
        return []
    return re.findall(r"'((?:[^'\\]|\\.)*)'", m.group(1))


def parse_curricula(html: str) -> dict:
    """The step metadata the served page itself carries, keyed by project type.

    Read from the page rather than from `app.py` so the walk asserts on what
    the browser was actually given.  The f-string's brace doubling is already
    resolved by then, which is why a step block starts `\\n    {\\n`.
    """
    out = {}
    for m in _CURRICULUM_BLOCK.finditer(html):
        blocks = re.split(r"\n    \{\n", m.group("body"))[1:]
        out[_BY_NAME[m.group("name")]] = [
            {
                "title": _quoted(b, "title"),
                "target": _quoted(b, "target") or None,
                "fallback": _list(b, "fallback"),
                "advanceTarget": _quoted(b, "advanceTarget") or None,
                "phaseTarget": _quoted(b, "phaseTarget") or None,
            }
            for b in blocks
        ]
    return out


def run_page(html: str, spec: dict, types) -> dict:
    """Serve `html` with the driver injected and return the driver's report."""
    driver = (
        DRIVER.replace("__SPEC__", json.dumps(spec))
        .replace("__TYPES__", json.dumps(list(types)))
        .replace("__REAL_CLICK__", json.dumps(list(REAL_CLICK)))
        .replace("__REAL_CHANGE__", json.dumps(list(REAL_CHANGE)))
    )
    html = html.replace("</body>", driver + "\n</body>", 1)
    probe = (
        "<script>window.__pageErrors=[];"
        "window.addEventListener('error',function(e){window.__pageErrors.push(String(e.message));});"
        "</script>"
    )
    html = html.replace("<!doctype html>", "<!doctype html>" + probe, 1)

    with serve(html) as url:
        proc = subprocess.run(
            [
                CHROME,
                "--headless=new",
                "--disable-gpu",
                "--no-sandbox",
                "--window-size=1600,1000",
                "--virtual-time-budget=150000",
                "--dump-dom",
                url,
            ],
            capture_output=True,
            text=True,
            timeout=180,
        )
    m = re.search(
        r'<pre[^>]*id="TM_TEST_OUT"[^>]*>TM_RESULT_JSON::(.*?)</pre>', proc.stdout, re.S
    )
    if m is None:
        raise AssertionError(
            "the page never reported a result (no TM_TEST_OUT); tail of the DOM:\n"
            + proc.stdout[-2000:]
        )
    return json.loads(m.group(1))


def check_curriculum(payload: dict, want_type: str) -> dict:
    """One type's walk: step count, per-step cue, marks, and forward progress."""
    assert payload["errors"] == [], payload["errors"]
    totals = [r for r in payload["steps"] if r["kind"] == "total"]
    assert len(totals) == 1, totals
    total = totals[0]
    assert total["projectType"] == want_type, total
    want = EXPECTED_STEPS[want_type]
    assert total["parsed"] == want, f"curriculum has {total['parsed']} steps, want {want}"
    assert total["total"] == want, f"the tour reports {total['total']} steps, want {want}"

    records = {}
    for r in payload["steps"]:
        if r["kind"] != "step":
            continue
        records[r["i"]] = r
        # A step that names a control must have found it: an unresolved
        # target reads as cue 'off', which would otherwise pass silently.
        if r["hasTarget"]:
            assert r["sel"] is not None, (
                f"{want_type} step {r['i']} ({r['title']}): the spotlight found nothing "
                "for its target"
            )
        assert r["cue"] == r["expected"], (
            f"{want_type} step {r['i']} ({r['title']}): cue is {r['cue']!r}, expected "
            f"{r['expected']!r} for target {r['sel']!r} ({r['control']})"
        )
        # The Settings modal must be in the state THIS step needs: open only
        # for the steps whose control lives inside it, closed for the rest —
        # a step whose own click opens it must put it back before the next
        # step spotlights something hidden behind it.
        assert r["settingsOpen"] == r["settingsExpected"], (
            f"{want_type} step {r['i']} ({r['title']}): Settings "
            f"{'left open' if r['settingsOpen'] else 'closed'} but the step "
            f"{'needs it open' if r['settingsExpected'] else 'needs it closed'}"
        )
        # The arrow links the card to the ring, so it is on exactly when
        # something is spotlit; the beacon sits on the control the card is
        # asking for, so it is on exactly when the step waits for a press.
        assert r["marks"]["arrow"] == (r["sel"] is not None), (
            f"{want_type} step {r['i']} ({r['title']}): arrow "
            f"{'shown' if r['marks']['arrow'] else 'hidden'} for target {r['sel']!r}"
        )
        assert r["marks"]["beacon"] == (r["expected"] in ("click", "input", "phase")), (
            f"{want_type} step {r['i']} ({r['title']}): beacon "
            f"{'shown' if r['marks']['beacon'] else 'hidden'} while the cue is "
            f"{r['expected']!r}"
        )
        if r["phase"]:
            _check_trigger_step(want_type, r)
        elif r["i"] < want - 1:
            assert r["advanced"] == r["i"] + 1, (
                f"{want_type} step {r['i']} ({r['title']}) did not advance"
            )
        else:
            assert r["advanced"] == r["i"], r
    assert len(records) == want, sorted(records)
    return {"total": total, "records": records}


def _check_trigger_step(want_type: str, r: dict) -> None:
    """A step whose screen is shut must ask for the door, then follow the user in.

    The same step index covers both halves: the spotlight starts on the
    control that opens the screen, and pressing it re-seats the step inside
    rather than skipping over the screen it documents.
    """
    assert r["sel"] in ("#advBtn", "#navSettings"), (
        f"{want_type} step {r['i']} ({r['title']}): the trigger phase spotlights "
        f"{r['sel']!r}, want the Advanced toggle or the ⚙ Settings button"
    )
    assert r["cue"] == "phase", r
    # While the door is spotlit the screen behind it is still shut — that is
    # the whole point of the phase.
    assert r["settingsExpected"] is False, r
    assert r["settingsOpen"] is False, r
    assert r["action"] == "click-door", r
    assert r["indexAfter"] == r["i"], (
        f"{want_type} step {r['i']} ({r['title']}): pressing {r['sel']} moved the tour "
        f"to step {r['indexAfter']} instead of re-seating this one"
    )
    assert r["advanced"] == r["i"], r
    assert r["cueAfter"] != "phase", (
        f"{want_type} step {r['i']} ({r['title']}): pressing {r['sel']} did not open "
        "the screen the step documents"
    )
    # Following the user in must re-seat the spotlight on what the step
    # actually documents — the Advanced rows, the OOD gate group, the EMNIST
    # group — rather than leaving the arrow on the button just pressed.
    assert r["selAfter"] == r["target"], (
        f"{want_type} step {r['i']} ({r['title']}): after {r['sel']} the spotlight "
        f"landed on {r['selAfter']!r}, want the step's own {r['target']!r}"
    )
    assert r["marksAfter"] and r["marksAfter"]["arrow"] is True, r
    if r["sel"] == "#navSettings":
        # The ⚙ door opens Settings and nothing else.  The Advanced training
        # panel keeps whatever state an earlier step left it in.
        assert r["settingsAfter"] is True, r
    else:
        # The Advanced door must not open Settings.
        assert r["advancedAfter"] is True, r
        assert r["settingsAfter"] is False, r


def interacted(report: dict, sel: str, action: str) -> None:
    """Using the spotlighted control itself must have moved the tour on."""
    hits = [
        r
        for r in report["records"].values()
        if r["sel"] == sel and not r["phase"]
    ]
    assert len(hits) == 1, f"{sel}: {len(hits)} steps target it, want 1"
    r = hits[0]
    assert r["action"] == action, (
        f"{r['type']} step {r['i']} ({r['title']}): action {r['action']!r}, want {action!r}"
    )
    assert r["advanced"] == r["i"] + 1, (
        f"{r['type']} step {r['i']} ({r['title']}): {action} on {sel} left the tour "
        f"on step {r['advanced']}"
    )


def check_disabled_guard(payload: dict) -> None:
    """Nothing trained, no samples: the two locked buttons must say so."""
    assert payload["errors"] == [], payload["errors"]
    records = [r for r in payload["steps"] if r["kind"] == "step"]
    assert len(records) == EXPECTED_STEPS["image"], len(records)
    for step in records:
        assert step["cue"] == step["expected"], step
        if step["expected"] != "disabled":
            continue
        assert step["action"] == "click-disabled", step
        assert step["advanced"] == step["i"], (
            f"{step['sel']}: clicking the disabled control advanced the tour to "
            f"{step['advanced']}"
        )
        assert "Collect samples first" in step["cueText"], step["cueText"]
    locked = [r["sel"] for r in records if r["expected"] == "disabled"]
    assert "#exportBtn" in locked, locked
    # The training card is reached through its trigger: pressing Advanced
    # lands inside the panel, where Train itself is locked — the same
    # "nothing trained yet" answer, one screen further in.
    train = [r for r in records if r["phase"] and r["sel"] == "#advBtn"]
    assert len(train) == 1, train
    assert train[0]["cueAfter"] == "disabled", train[0]
    assert train[0]["indexAfter"] == train[0]["i"], train[0]


# ── the start page (Streamlit, so checked at the source) ─────────────────
def _block(src: str, header: str) -> str:
    """The indented body of `header`, up to the next line at the same indent."""
    lines = src.splitlines()
    for i, line in enumerate(lines):
        if line.strip() != header:
            continue
        indent = len(line) - len(line.lstrip())
        body = []
        for nxt in lines[i + 1:]:
            if not nxt.strip():
                body.append(nxt)
                continue
            if len(nxt) - len(nxt.lstrip()) <= indent:
                break
            body.append(nxt)
        return "\n".join(body)
    raise AssertionError(f"{header!r} not found in app.py")


def check_start_page() -> None:
    """All three start-page cards render before any of them acts.

    The Open Project picker is a native modal dialog that blocks the script,
    so opening it mid-grid used to leave the OCR and sign cards unrendered
    behind it — and cancelling `return`ed straight out of the column,
    dropping them for good, with a yellow "No file selected." box on top.
    """
    src = (ROOT / "app.py").read_text(encoding="utf-8")
    assert "No file selected." not in src, (
        "the cancelled Open Project dialog still warns the user"
    )
    for col in ("with card_col1:", "with card_col2:", "with card_col3:"):
        body = _block(src, col)
        assert body.strip(), f"{col} is empty"
        assert not re.search(r"^\s*return\b", body, re.M), (
            f"{col} still returns early, which drops the cards after it"
        )
        assert "open_error_slot.error(" not in body, (
            f"{col} reports open errors inside the column instead of above the grid"
        )
        # Each column only RECORDS which card was clicked; the picker runs
        # once, after the grid, so a cancelled dialog cannot cost a card.
        assert body.count("open_request = ") == 1, (
            f"{col} should record exactly one open request, found "
            f"{body.count('open_request = ')}"
        )
    assert src.count("picked, why = _tmproj_pick_and_classify()") == 1, (
        "the project picker must run once, after the grid"
    )
    assert src.index("if open_request:") > src.index("with card_col3:"), (
        "the picker still runs before the last card is rendered"
    )
    assert "open_error_slot = st.empty()" in src


def main() -> int:
    base = boot_payload()
    cfg, defaults = base["train_cfg"], base["train_cfg_defaults"]

    def page(**overrides):
        return build_spa_html(boot_payload(**overrides))

    pages = {
        "image": page(
            project_type="image",
            classes=["LEFT", "STOP"],
            counts={"LEFT": 4, "STOP": 3},
            export_enabled=True,
            train_cfg=dict(cfg, img_size=96, project_type="image", sign_sequence=None),
            train_cfg_defaults=dict(defaults, img_size=96, project_type="image", sign_sequence=None),
        ),
        "ocr": page(
            project_type="ocr",
            classes=["A", "1"],
            counts={"A": 5, "1": 4},
            export_enabled=True,
            train_cfg=dict(cfg, img_size=28, project_type="ocr", sign_sequence=None),
            train_cfg_defaults=dict(defaults, img_size=28, project_type="ocr", sign_sequence=None),
        ),
        "sign_ocr": page(
            project_type="sign_ocr",
            classes=["A", "1"],
            counts={"A": 3, "1": 2},
            export_enabled=True,
            train_cfg=dict(cfg, img_size=28, project_type="sign_ocr"),
            train_cfg_defaults=dict(defaults, img_size=28, project_type="sign_ocr"),
        ),
    }
    locked_html = page(
        project_type="image",
        classes=["LEFT", "STOP"],
        counts={"LEFT": 0, "STOP": 0},
        train_enabled=False,
        export_enabled=False,
        train_cfg=dict(cfg, img_size=96, project_type="image", sign_sequence=None),
        train_cfg_defaults=dict(defaults, img_size=96, project_type="image", sign_sequence=None),
    )

    spec = parse_curricula(pages["image"])
    assert sorted(spec) == sorted(EXPECTED_STEPS), sorted(spec)

    results = []

    def run(name, fn):
        try:
            fn()
        except Exception as exc:  # noqa: BLE001 - a runner reports everything
            results.append((name, False))
            print(f"FAIL {name}: {exc}")
            traceback.print_exc()
        else:
            results.append((name, True))
            print(f"ok   {name}")

    reports = {}
    try:
        for t in ("image", "ocr", "sign_ocr"):
            reports[t] = run_page(pages[t], spec, [t])
        locked = run_page(locked_html, spec, ["image"])
    except Exception as exc:  # noqa: BLE001
        results.append(("headless tour walk", False))
        print(f"FAIL headless tour walk: {exc}")
        traceback.print_exc()
        print(f"\n0/{len(results)} passed")
        return 1

    checked = {}

    def curriculum(t):
        checked[t] = check_curriculum(reports[t], t)

    for t in ("image", "ocr", "sign_ocr"):
        run(f"{t} curriculum and cues", lambda t=t: curriculum(t))

    def distinct():
        assert len(checked) == 3, "a curriculum check failed; not compared"
        titles = [tuple(checked[t]["total"]["titles"]) for t in ("image", "ocr", "sign_ocr")]
        assert len(set(titles)) == 3, "the three project types do not have distinct curricula"

    run("three distinct curricula", distinct)

    def click_to_advance():
        assert len(checked) == 3, "a curriculum check failed; not compared"
        interacted(checked["image"], "#navSettings", "click")
        interacted(checked["image"], ".class-preprocess-chip", "click")
        interacted(checked["image"], ".class-preprocess-mode-row", "click")
        interacted(checked["image"], "#classPreprocessDark", "change")
        interacted(checked["ocr"], ".class-preprocess-chip", "click")
        interacted(checked["sign_ocr"], "#signSeqBar", "change")

    run("using the highlighted control advances", click_to_advance)
    run("a locked control does not trap the user", lambda: check_disabled_guard(locked))
    run("the start page renders every card first", check_start_page)

    passed = sum(1 for _, ok in results if ok)
    print(f"\n{passed}/{len(results)} passed")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
