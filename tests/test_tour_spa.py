"""Automated checks for the workspace's interactive tour (`window.tmTour`).

The tour is JavaScript emitted from the workspace f-string, so there is
nothing to import.  `spa_harness` evaluates that f-string and serves the
resulting page on a real localhost origin, and this test drives it in the
same Chrome the app ships in — launched head-less from the command line, so
the check needs no test-only dependency (no Playwright, no Node).

The guide is built as the rhythm a person actually works in, so that is what
is checked:

  curricula     the guide selected for a project type is that project's own
                curriculum: 23 steps for `image`, 26 for `ocr`, 28 for
                `sign_ocr`, with three distinct step lists
  overview      every type opens on step 0 as a Workflow Overview: a card
                that lists the whole trip as numbered stages, with no
                spotlight, no arrow and no hands-on button — and exactly one
                way forward, Start Step-by-Step Guide →
  explain & point
                every step after the overview spotlights its control and
                draws `#tmTourArrow` at it, and the card names the exact
                action in a `#tmTourCue` banner; both are on exactly when
                they should be, and off for a step with nothing to point at
  hands-on      the wide "Start Doing This" button on the card collapses the
                tour into `#tmTourOrb`: the dim layer, the ring and the card
                all leave the screen — so nothing covers the buttons and
                sliders being used — and the pill is really visible
  live status   the pill carries the step's own live state, and its badge
                turns green only once that state has really changed: sample
                counts, a slider, the target sign, a finished training run.
                Steps with nothing done yet read "not ready"
  no teleport   using the spotlighted control itself — or the door that opens
                the screen a step documents — hands the workspace over
                instead of advancing; and a second press moves the tour no
                further, which is what lets a user capture twenty photos or
                tune a slider without the guide walking off
  orb advances  the pill is what moves the tour on, one step at a time, and
                the closing slide ends the guide
  re-read       `?` on the pill brings the instruction card back for the same
                step, and closing the tour takes the pill away with it
  disabled guard
                a button that cannot be used yet (`#trainBtn` with no
                samples, `#exportBtn` with no model) reports "disabled"
                instead of trapping the user, does not advance when clicked,
                and says in the card what to do instead
  re-armed      the class column is thrown away and rebuilt after every
                capture (`#classes.innerHTML = ''`), which replaces the very
                node the click binding sat on: the guide must re-seat onto
                the fresh node, so pressing the control still hands over
  containers    a step whose spotlight is a container rather than the field
                inside it answers to click AND to input / change — the three
                events the user's own work produces — and the pill's badge
                turns green for each of them
  escape order  Escape belongs to whichever layer is on top: with a modal or
                the drawer open it closes that layer and leaves the guide
                running, rather than killing the tour standing behind it
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
EXPECTED_STEPS = {"image": 23, "ocr": 26, "sign_ocr": 28}

# Steps whose real interaction the test performs, keyed by the TARGET
# selector — that is the string the driver records for a step, not the
# `advanceTarget` inside it.
#
# The walk itself always takes the card's own "Start Doing This" button, so
# these are used by the second pass: it uses the real control instead, and
# states that doing so hands the workspace over WITHOUT moving the tour on.
# Each screen's door, its inner controls and its Close / Cancel button are
# pressed for real: ⚙ Settings, the Webcam tile, the drawer's Capture and ▴
# Close, the Edit chip, the worktable's × (and the select / slider inside it).
REAL_CLICK = (
    "#navSettings",
    "#settingsCancel",
    ".tm-action-toolbar",
    '[id^="sourceCapture-"]',
    '[id^="sourceClose-"]',
    ".class-preprocess-chip",
    "#classPreprocessCloseTop",
)
REAL_CHANGE = (
    "#classPreprocessDark",
    ".class-preprocess-mode-row",
    "#signSeqBar",
    "#advReadingOrderRow",
    "#advInkPolarityRow",
)

# `#trainBtn` and `#exportBtn` are never clicked: the stub would answer and
# the tour would walk the user through a training run or an export that never
# happened.  Both still assert their cue, which is what click-to-advance arms.
DRIVER = r"""
<script>
(function () {
  var SPEC = __SPEC__;
  var TYPES = __TYPES__;
  var REAL_CLICK = __REAL_CLICK__;
  var REAL_CHANGE = __REAL_CHANGE__;
  var OUT = [];
  var DONE = false;
  /* How long the pill's "hands-on mode" greeting stays up before the live
     state takes the text over (the tour promises 1.8s). */
  var ORB_FLASH_MS = 2000;
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

  function byId(id) { return document.getElementById(id); }

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

  /* Where the tour says it put the ring.  The tour re-seats the spotlight
     on the next animation frame, and head-less Chrome hands frames out on
     its own schedule, so a step is "on screen" when the ring on screen says
     it belongs to that step — checking the class list alone would read the
     previous step's ring, still sitting where the last step left it. */
  function placedAt(i) {
    try { return window.tmTour.place().index === i; } catch (e) { return false; }
  }

  function settle(i, ms) {
    var deadline = Date.now() + (ms || 9000);
    return new Promise(function (res) {
      (function tick() {
        if (placedAt(i)) return res(true);
        if (Date.now() >= deadline) return res(false);
        window.setTimeout(tick, 50);
      })();
    });
  }

  function describe(el) {
    if (!el) return null;
    return String(el.tagName || '') + '#' + String(el.id || '') +
      '.' + String(el.className || '').slice(0, 22);
  }

  function settingsOpen() {
    var modal = byId('globalSettingsModal');
    return !!(modal && modal.classList.contains('open'));
  }

  /* `trainAdvancedOpen` is a top-level binding of the workspace script,
     shared with this one: it is the same variable the curricula's own
     `phaseIf` reads, so the walk and the app agree on when the panel is
     collapsed. */
  function advancedOpen() {
    return (typeof trainAdvancedOpen !== 'undefined') && !!trainAdvancedOpen;
  }

  function editMenuOpen() {
    var d = byId('menuEditDropdown');
    return !!(d && d.classList.contains('open'));
  }

  /* `openSourceClass` and `classPreprocessOpen` are top-level bindings of the
     workspace script, shared with this one: the very values the tour's own
     `drawerOpen()` / `worktableOpen()` read. */
  function drawerOpen() {
    return (typeof openSourceClass !== 'undefined') && !!openSourceClass;
  }

  function worktableOpen() {
    return (typeof classPreprocessOpen !== 'undefined') && !!classPreprocessOpen;
  }

  function screens() {
    return {
      advanced: advancedOpen(), settings: settingsOpen(), editMenu: editMenuOpen(),
      worktable: worktableOpen(), drawer: drawerOpen()
    };
  }

  /* The screen a step documents is SHUT when its trigger is still needed.
     This is the `phaseIf` each curricula step declares, restated — the door
     the step names is the key — so the walk and the app agree on when the
     screen behind that door has to be opened before the step can land. */
  function screenShut(step) {
    if (!step.phaseTarget) return false;
    if (step.phaseTarget === '#advBtn') return !advancedOpen();
    if (step.phaseTarget === '#navSettings') return !settingsOpen();
    if (step.phaseTarget === '#menuEditBtn') return !editMenuOpen();
    if (step.phaseTarget === '.class-preprocess-chip') return !worktableOpen();
    if (step.phaseTarget === '.tm-action-toolbar button.sample') return !drawerOpen();
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

  /* The control inside the spotlight that really carries the affordance —
     `advanceTarget`, when the step names one and it is on screen. */
  function controlOf(step, found) {
    var el = found ? found.el : null;
    if (!el) return null;
    if (!found.phase && step.advanceTarget) {
      var named = null;
      try { named = document.querySelector(step.advanceTarget); } catch (e) { named = null; }
      if (named && vis(named)) return named;
    }
    return el;
  }

  function marks() {
    try { return window.tmTour.marks(); } catch (e) { return null; }
  }

  function orb() {
    try { return window.tmTour.orb(); } catch (e) { return { on: false, visible: false, text: '', live: {} }; }
  }

  /* The cue the tour is supposed to be showing, derived the same way
     `armTargetAdvance` derives it: from the resolved element, the named
     `advanceTarget`, and that control's own disabled / tag state.  The
     trigger phase (`phase`) is decided by `resolveTargets`, above. */
  function expectedCue(step, found) {
    if (!found) return 'off';
    var control = controlOf(step, found);
    if (control.disabled) return 'disabled';
    var tag = String(control.tagName || '').toUpperCase();
    return (tag === 'INPUT' || tag === 'SELECT' || tag === 'TEXTAREA') ? 'input' : 'click';
  }

  /* A real press on the control the card is pointing at, the way a user
     would: the element itself, not `tmTour.next()`. */
  function doReal(step, found, control, kind) {
    if (kind === 'change') {
      var el = control || (found && found.el);
      if (!el) return;
      el.dispatchEvent(new Event('input', { bubbles: true }));
      el.dispatchEvent(new Event('change', { bubbles: true }));
      return;
    }
    var host = found.el;
    if (!host) return;
    var hit = (host.matches && host.matches(ADV_SEL)) ? host : host.querySelector(ADV_SEL);
    if (hit) hit.click();
  }

  /* Do the step's own job, for real, so the pill has something live to
     report.  `samples` and `train` arrive the way the backend delivers them
     (no event at all — the pill polls for those); `thresholds` and `signseq`
     arrive the way the user produces them (an input / change event). */
  function bumpLive(kind) {
    if (kind === 'samples') {
      var counts = (typeof STATE !== 'undefined' && STATE.counts) || {};
      var keys = Object.keys(counts).filter(function (k) { return k !== '_background'; });
      if (!keys.length) { counts['A'] = 0; keys = ['A']; }
      counts[keys[0]] = (Number(counts[keys[0]]) || 0) + 12;
      return 'counts.' + keys[0] + '=' + counts[keys[0]];
    }
    if (kind === 'train') {
      try { STATE.trainClassAccuracies = { 'A': 0.97, '1': 0.95 }; } catch (e) {}
      return 'trainClassAccuracies';
    }
    if (kind === 'thresholds') {
      var found = firstVisible(['#classPreprocessDark', '#classPreprocessLum',
        '#previewDarkSlider', '#previewLumSlider']);
      if (!found) return 'no slider on screen';
      var v = Number(found.el.value);
      if (!isFinite(v)) v = 0;
      found.el.value = String(v + 1);
      found.el.dispatchEvent(new Event('input', { bubbles: true }));
      found.el.dispatchEvent(new Event('change', { bubbles: true }));
      return found.sel + '=' + found.el.value;
    }
    if (kind === 'signseq') {
      var slot = byId('signSeqSlot-0');
      if (!slot) return 'no slot on screen';
      slot.value = 'X';
      slot.dispatchEvent(new Event('input', { bubbles: true }));
      slot.dispatchEvent(new Event('change', { bubbles: true }));
      return 'signSeqSlot-0=X';
    }
    return '';
  }

  /* ── step 0: the Workflow Overview ────────────────────────────────── */
  async function walkOverview(t) {
    var card = byId('tmTourCard');
    var spot = byId('tmTourSpotlight');
    var next = byId('tmTourNext');
    var rec = {
      kind: 'overview', type: t, index: window.tmTour.current(),
      title: (card && card.querySelector('.tm-tour-title')) ?
        String(card.querySelector('.tm-tour-title').textContent || '') : '',
      declared: window.tmTour.overview(),
      stages: window.tmTour.stages(),
      stageRows: card ? card.querySelectorAll('.tm-tour-stages li').length : -1,
      cue: window.tmTour.cue(),
      cardOn: !!(card && card.classList.contains('on')),
      spotOn: !!(spot && spot.classList.contains('on')),
      spotDim: !!(spot && spot.classList.contains('dim')),
      marks: marks(),
      hasStart: !!byId('tmTourStart'),
      nextText: next ? String(next.textContent || '').trim() : '',
      orbOn: orb().on,
      orbVisible: orb().visible
    };
    /* The overview's one door: start the walkthrough. */
    if (next) next.click(); else window.tmTour.next();
    await wait(800);
    rec.after = window.tmTour.current();
    rec.orbAfter = orb().on;
    rec.cardAfter = !!(card && card.classList.contains('on'));
    OUT.push(rec);
  }

  /* ── steps 1..n-1: Explain & Point, then the Floating Orb ─────────── */
  async function walkSteps(t, steps, n) {
    var bumped = {};
    for (var i = 1; i < n; i++) {
      window.tmTour.goTo(i);
      await settle(i);
      /* The tour re-seats the ring once the layout has settled (280ms), so
         wait out that second look before reading what is on screen. */
      await wait(340);
      var step = steps[i] || {};
      var res = resolveTargets(step);
      var found = res.found;
      var control = controlOf(step, found);
      var card = byId('tmTourCard');
      var spot = byId('tmTourSpotlight');
      var modal = byId('globalSettingsModal');
      var cueBox = byId('tmTourCue');
      /* Read while the step is on screen, before any action below moves the
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
        live: step.live || '',
        expected: res.phase ? 'phase' : expectedCue(step, found),
        cue: window.tmTour.cue(),
        cueText: cueBox ? String(cueBox.textContent || '').slice(0, 70) : '',
        marks: marks(),
        cardOn: !!(card && card.classList.contains('on')),
        spotOn: !!(spot && spot.classList.contains('on')),
        orbOn: orb().on,
        startPresent: !!byId('tmTourStart'),
        settingsOpen: settingsOpen(),
        settingsExpected: !!(modal && found && found.el && modal.contains(found.el))
      };

      if (i === n - 1) {
        /* The closing slide has nothing to point at, so it has no hands-on
           mode: Finish 🎉 is the only door, and it ends the guide. */
        var fin = byId('tmTourNext');
        rec.finishText = fin ? String(fin.textContent || '').trim() : '';
        if (fin) fin.click(); else window.tmTour.next();
        await wait(800);
        rec.advanced = window.tmTour.current();
        rec.closed = !window.tmTour.isOpen();
        OUT.push(rec);
        break;
      }

      /* Phase A → Phase B: the wide button on the card. */
      var startBtn = byId('tmTourStart');
      if (startBtn) { startBtn.click(); rec.phaseB = 'start-button'; }
      else { window.tmTour.handsOn(); rec.phaseB = 'api'; }
      await wait(700);
      var ob = orb();
      rec.handsOn = {
        on: ob.on, cardOn: ob.cardOn, spotOn: ob.spotOn, visible: ob.visible,
        text: String(ob.text || '').slice(0, 96), ready: !!(ob.live && ob.live.ready)
      };
      /* The pill greets the hand-over before it starts reporting: read its
         live text once that acknowledgement has had its say. */
      await wait(ORB_FLASH_MS);
      var obLive = orb();
      rec.handsOn.text = String(obLive.text || '').slice(0, 96);
      rec.handsOn.ready = !!(obLive.live && obLive.live.ready);
      /* Which screen the hand-over left open, for a step that lives in one. */
      rec.doorSel = res.phase && found ? found.sel : null;
      rec.door = screens();

      /* The live badge: only the first step of each kind is worked on, and
         only the step's own kind can move it. */
      if (step.live && !bumped[step.live]) {
        bumped[step.live] = true;
        rec.bump = bumpLive(step.live);
        var eventKind = (step.live === 'thresholds' || step.live === 'signseq');
        await wait(eventKind ? 560 : 1500);
        var ob2 = orb();
        rec.afterBump = {
          ready: !!(ob2.live && ob2.live.ready),
          text: String(ob2.text || '').slice(0, 96),
          goText: String(ob2.goText || '').slice(0, 40)
        };
      }

      /* The pill is the only thing that moves the tour on. */
      window.tmTour.clickOrb();
      await wait(860);
      var ob3 = orb();
      rec.advanced = window.tmTour.current();
      rec.orbAfterAdvance = { on: ob3.on, cardOn: ob3.cardOn, visible: ob3.visible };
      OUT.push(rec);
    }
  }

  /* ── the real control does the same thing, and moves nothing ──────── */
  async function walkRealControls(t, steps, n) {
    var picked = 0;
    for (var i = 1; i < n && picked < 4; i++) {
      var step = steps[i] || {};
      if (step.overview || !step.target) continue;
      var res = resolveTargets(step);
      /* Doors are the walk's business: with the screen shut the spotlight is
         on the trigger, not on the control the step documents. */
      if (res.phase || !res.found) continue;
      var sel = res.found.sel;
      var kind = REAL_CLICK.indexOf(sel) >= 0 ? 'click'
        : (REAL_CHANGE.indexOf(sel) >= 0 ? 'change' : null);
      if (!kind) continue;

      window.tmTour.start();
      window.tmTour.goTo(i);
      await settle(i);
      await wait(340);
      var control = controlOf(step, res.found);
      var rec = {
        kind: 'auto', type: t, i: i, sel: sel, action: kind,
        before: window.tmTour.current(), orbBefore: orb().on,
        cueBefore: window.tmTour.cue()
      };
      doReal(step, res.found, control, kind);
      await wait(800);
      rec.orb = orb().on;
      rec.cardOn = !!(byId('tmTourCard') && byId('tmTourCard').classList.contains('on'));
      rec.spotOn = !!(byId('tmTourSpotlight') && byId('tmTourSpotlight').classList.contains('on'));
      rec.index = window.tmTour.current();
      /* A user who keeps working — the second photo, the third slider nudge
         — must find the tour exactly where they left it. */
      doReal(step, res.found, control, kind);
      await wait(800);
      rec.indexAgain = window.tmTour.current();
      rec.orbAgain = orb().on;
      OUT.push(rec);
      window.tmTour.close();
      await wait(140);
      picked++;
    }
  }

  /* ── `?` on the pill brings the card back, same step ──────────────── */
  async function walkAskOrb(t) {
    window.tmTour.start();
    window.tmTour.goTo(1);
    await settle(1);
    await wait(340);
    var startBtn = byId('tmTourStart');
    if (startBtn) startBtn.click(); else window.tmTour.handsOn();
    await wait(340);
    var rec = {
      kind: 'ask', type: t,
      before: { on: orb().on, cardOn: !!(byId('tmTourCard') && byId('tmTourCard').classList.contains('on')),
                i: window.tmTour.current() }
    };
    window.tmTour.askOrb();
    await wait(460);
    rec.after = {
      on: orb().on,
      cardOn: !!(byId('tmTourCard') && byId('tmTourCard').classList.contains('on')),
      spotOn: !!(byId('tmTourSpotlight') && byId('tmTourSpotlight').classList.contains('on')),
      i: window.tmTour.current()
    };
    OUT.push(rec);
  }

  /* ── support for the three checks below ───────────────────────────── */
  function stepIndexFor(steps, sel) {
    for (var i = 1; i < steps.length; i++) {
      if (steps[i] && steps[i].target === sel) return i;
    }
    return -1;
  }

  /* The step that documents a shut screen, found by the door it points at. */
  function stepIndexForPhase(steps, door) {
    for (var i = 1; i < steps.length; i++) {
      if (steps[i] && steps[i].phaseTarget === door) return i;
    }
    return -1;
  }

  /* Put the workspace back the way the next sub-check expects to find it:
     whatever screen the previous one left standing, closed. */
  function resetLayers() {
    try {
      if (worktableOpen() && typeof closeClassPreprocessEditor === 'function')
        closeClassPreprocessEditor();
    } catch (e) {}
    try {
      if (settingsOpen() && typeof closeGlobalSettings === 'function') closeGlobalSettings();
    } catch (e) {}
    try {
      if (drawerOpen() && typeof closeSourcePanel === 'function') closeSourcePanel();
    } catch (e) {}
  }

  /* ── a rebuilt class column must not disarm the step ──────────────── */
  async function walkRebind(t, steps) {
    var i = stepIndexFor(steps, '.tm-action-toolbar');
    if (i < 0) return;
    resetLayers();
    await wait(200);
    window.tmTour.start();
    window.tmTour.goTo(i);
    await settle(i);
    await wait(340);
    var rec = {
      kind: 'rebind', type: t, i: i, err: null,
      sel: null, replaced: false, cue: null,
      orbBefore: orb().on, orb: false, cardOn: true, spotOn: true,
      ready: false, text: '', index: null
    };
    var step = steps[i];
    var before = resolveTargets(step);
    var oldNode = before.found ? before.found.el : null;
    /* Mark the node on screen, so the record can state that the press below
       really went to a DIFFERENT node than the one the binding was armed on. */
    if (oldNode) oldNode.__tmPreRender = true;
    /* Exactly what the workspace does after every capture: the class column
       is thrown away and built again, replacing the node the binding sat on. */
    try { render(); } catch (e) { rec.err = String((e && e.message) || e); }
    await wait(600);
    await settle(i, 6000);
    await wait(340);
    var res = resolveTargets(step);
    rec.sel = res.found ? res.found.sel : null;
    rec.replaced = !!(res.found && res.found.el && res.found.el.__tmPreRender !== true);
    rec.cue = window.tmTour.cue();
    /* The fresh node must be armed again: pressing it hands over. */
    doReal(step, res.found, controlOf(step, res.found), 'click');
    await wait(3000);
    var ob = orb();
    rec.orb = ob.on;
    rec.cardOn = ob.cardOn;
    rec.spotOn = ob.spotOn;
    rec.ready = !!(ob.live && ob.live.ready);
    rec.text = String(ob.text || '').slice(0, 96);
    rec.index = window.tmTour.current();
    OUT.push(rec);
    window.tmTour.close();
    await wait(160);
    resetLayers();
  }

  /* ── a container step answers to click AND to input / change ──────── */
  async function walkContainerEvents(t, steps) {
    var i = stepIndexFor(steps, '.tm-action-toolbar');
    if (i < 0) return;
    var actions = ['click', 'input', 'change'];
    for (var k = 0; k < actions.length; k++) {
      var action = actions[k];
      resetLayers();
      await wait(200);
      window.tmTour.start();
      window.tmTour.goTo(i);
      await settle(i);
      await wait(340);
      var step = steps[i];
      var res = resolveTargets(step);
      var el = res.found ? res.found.el : null;
      var rec = {
        kind: 'container', type: t, i: i, action: action,
        sel: res.found ? res.found.sel : null,
        isContainer: !!(el && ['INPUT', 'SELECT', 'TEXTAREA', 'BUTTON']
          .indexOf(String(el.tagName || '').toUpperCase()) < 0),
        indexBefore: window.tmTour.current(), orbBefore: orb().on,
        orb: false, cardOn: true, spotOn: true, ready: false, text: '', index: null
      };
      if (action === 'click') {
        doReal(step, res.found, controlOf(step, res.found), 'click');
      } else if (el) {
        el.dispatchEvent(new Event(action, { bubbles: true }));
      }
      await wait(3000);
      var ob = orb();
      rec.orb = ob.on;
      rec.cardOn = ob.cardOn;
      rec.spotOn = ob.spotOn;
      rec.ready = !!(ob.live && ob.live.ready);
      rec.text = String(ob.text || '').slice(0, 96);
      rec.index = window.tmTour.current();
      OUT.push(rec);
      window.tmTour.close();
      await wait(160);
    }
    resetLayers();
  }

  /* ── Escape closes the open layer, and leaves the guide running ───── */
  async function walkEscapeLayer(t, steps) {
    var cases = [
      { layer: 'settings', door: '#navSettings' },
      { layer: 'drawer', door: '.tm-action-toolbar button.sample' }
    ];
    for (var c = 0; c < cases.length; c++) {
      var i = stepIndexForPhase(steps, cases[c].door);
      if (i < 0) continue;
      resetLayers();
      await wait(200);
      window.tmTour.start();
      window.tmTour.goTo(i);
      await settle(i);
      await wait(340);
      /* Start Doing This presses the step's own door, so the screen the step
         documents is open and the workspace has been handed over. */
      var startBtn = byId('tmTourStart');
      if (startBtn) startBtn.click(); else window.tmTour.handsOn();
      await wait(1000);
      var scr = screens();
      var ob = orb();
      var rec = {
        kind: 'escape', type: t, layer: cases[c].layer, i: i,
        opened: cases[c].layer === 'settings' ? scr.settings : scr.drawer,
        orbBefore: ob.on, cardBefore: ob.cardOn,
        closed: false, open: false, index: null
      };
      /* Escape, the way a keyboard sends it.  Nothing is focused, so the
         key goes straight to the layer stack. */
      try { if (document.activeElement && document.activeElement.blur)
        document.activeElement.blur(); } catch (e) {}
      await wait(120);
      try {
        document.body.dispatchEvent(new KeyboardEvent('keydown', {
          key: 'Escape', code: 'Escape', bubbles: true, cancelable: true
        }));
      } catch (e) {}
      await wait(1000);
      var scr2 = screens();
      rec.closed = cases[c].layer === 'settings' ? !scr2.settings : !scr2.drawer;
      rec.open = window.tmTour.isOpen();
      rec.index = window.tmTour.current();
      OUT.push(rec);
      window.tmTour.close();
      await wait(160);
      resetLayers();
    }
  }

  async function walkType(t) {
    try { STATE.project_type = t; } catch (e) {}
    window.tmTour.select();
    var steps = SPEC[t] || [];
    var n = window.tmTour.total();
    OUT.push({
      kind: 'total', type: t, projectType: window.tmTour.projectType(),
      total: n, parsed: steps.length, titles: window.tmTour.titles(),
      overview: window.tmTour.overview(), stages: window.tmTour.stages(),
      orbVisibleBefore: orb().visible
    });

    window.tmTour.start();
    await settle(0);
    await wait(340);
    await walkOverview(t);
    await walkSteps(t, steps, n);
    window.tmTour.close();
    await wait(140);
    await walkRealControls(t, steps, n);
    await walkAskOrb(t);
    await walkRebind(t, steps);
    await walkContainerEvents(t, steps);
    await walkEscapeLayer(t, steps);

    window.tmTour.close();
    await wait(200);
    OUT.push({
      kind: 'closed', type: t,
      orbVisible: orb().visible,
      cardOn: !!(byId('tmTourCard') && byId('tmTourCard').classList.contains('on')),
      spotOn: !!(byId('tmTourSpotlight') && byId('tmTourSpotlight').classList.contains('on')),
      arrow: marks() ? marks().arrow : false,
      beacon: marks() ? marks().beacon : false
    });
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


def _flag(block: str, key: str) -> bool:
    return bool(re.search(r"(?<![A-Za-z])" + key + r":\s*true", block))


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
                "live": _quoted(b, "live") or None,
                "overview": _flag(b, "overview"),
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
                # Virtual time, not wall clock: the walk is almost entirely
                # `setTimeout` beats, and the three curricula together now
                # spend far more than the old budget waiting on them.
                "--virtual-time-budget=900000",
                "--dump-dom",
                url,
            ],
            capture_output=True,
            text=True,
            timeout=300,
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


def _records(payload: dict, kind: str):
    return [r for r in payload["steps"] if r["kind"] == kind]


def check_overview(payload: dict, want_type: str, want_stages: int) -> dict:
    """The guide opens on the big picture, not on a control.

    Step 0 is a card the user can only read: numbered stages covering the
    whole workflow, no spotlight, no arrow, no beacon, no hands-on mode —
    and exactly one button, which starts the step-by-step guide.
    """
    assert payload["errors"] == [], payload["errors"]
    totals = _records(payload, "total")
    assert len(totals) == 1, totals
    total = totals[0]
    assert total["overview"] is True, "the curriculum has no overview step"
    assert total["stages"] == want_stages, (
        f"the overview lists {total['stages']} stages, want {want_stages}"
    )
    assert total["orbVisibleBefore"] is False, "the pill is on screen before the tour starts"

    rows = _records(payload, "overview")
    assert len(rows) == 1, rows
    r = rows[0]
    assert r["declared"] is True, r
    assert r["index"] == 0, r
    assert r["stages"] == want_stages, r
    # The card really drew them (as stages, not as prose).
    assert r["stageRows"] == want_stages, (
        f"{want_type}: the overview card drew {r['stageRows']} stage rows, "
        f"want {want_stages}"
    )
    assert r["cardOn"] is True, r
    assert r["spotOn"] is True and r["spotDim"] is True, (
        f"{want_type}: the overview must dim the whole workspace, not cut a "
        f"spotlight out of it (ring on={r['spotOn']}, dim={r['spotDim']})"
    )
    # Nothing to point at, and nothing to click but the one button.
    assert r["cue"] == "off", r
    assert r["marks"]["arrow"] is False and r["marks"]["beacon"] is False, r
    assert r["hasStart"] is False, "the overview offers a hands-on button"
    assert r["orbOn"] is False and r["orbVisible"] is False, r
    assert "Start Step-by-Step Guide" in r["nextText"], r["nextText"]
    assert r["after"] == 1, (
        f"{want_type}: Start Step-by-Step Guide left the guide on step {r['after']}"
    )
    assert r["orbAfter"] is False, r
    assert r["cardAfter"] is True, r
    return total


def check_curriculum(payload: dict, want_type: str) -> dict:
    """One type's walk: step count, cues, marks, hand-over and forward progress."""
    assert payload["errors"] == [], payload["errors"]
    totals = _records(payload, "total")
    assert len(totals) == 1, totals
    total = totals[0]
    assert total["projectType"] == want_type, total
    want = EXPECTED_STEPS[want_type]
    assert total["parsed"] == want, f"curriculum has {total['parsed']} steps, want {want}"
    assert total["total"] == want, f"the tour reports {total['total']} steps, want {want}"

    records = {}
    for r in _records(payload, "step"):
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
        # Phase A still explains and points: the arrow links the card to the
        # ring, and the beacon sits on the control the cue is asking for.
        assert r["cardOn"] is True and r["spotOn"] is True, (
            f"{want_type} step {r['i']} ({r['title']}): the step is not on screen to "
            "be explained"
        )
        assert r["orbOn"] is False, (
            f"{want_type} step {r['i']} ({r['title']}): the pill is already up while the "
            "card is still explaining"
        )
        assert r["marks"]["arrow"] == (r["sel"] is not None), (
            f"{want_type} step {r['i']} ({r['title']}): arrow "
            f"{'shown' if r['marks']['arrow'] else 'hidden'} for target {r['sel']!r}"
        )
        assert r["marks"]["beacon"] == (r["expected"] in ("click", "input", "phase")), (
            f"{want_type} step {r['i']} ({r['title']}): beacon "
            f"{'shown' if r['marks']['beacon'] else 'hidden'} while the cue is "
            f"{r['expected']!r}"
        )

        last = r["i"] == want - 1
        if last:
            # The closing slide has no control to work with, so no hands-on
            # mode either: Finish 🎉 ends the guide.
            assert r["startPresent"] is False, r
            assert r["finishText"].startswith("Finish"), r["finishText"]
            assert r["closed"] is True, (
                f"{want_type}: the closing slide left the guide open"
            )
            assert r["advanced"] == r["i"], r
            continue

        if r["phase"]:
            _check_trigger_step(want_type, r)
        _check_hands_on(want_type, r)
        _check_live(want_type, r)
        assert r["advanced"] == r["i"] + 1, (
            f"{want_type} step {r['i']} ({r['title']}): the pill left the guide on "
            f"step {r['advanced']}"
        )
        assert r["orbAfterAdvance"]["on"] is False, (
            f"{want_type} step {r['i']} ({r['title']}): next step did not start in "
            "Explain & Point mode — the pill is still the only thing on screen"
        )
        assert r["orbAfterAdvance"]["cardOn"] is True, (
            f"{want_type} step {r['i']} ({r['title']}): the next step's card never "
            "came back after the pill moved the user on"
        )
    # Step 0 is the overview, which `check_overview` reads on its own: the
    # walk records one entry per step from 1 to the closing slide.
    assert len(records) == want - 1, sorted(records)
    return {"total": total, "records": records}


def _check_hands_on(want_type: str, r: dict) -> None:
    """The card's own button must move the user out of the way of their work."""
    assert r["startPresent"] is True, (
        f"{want_type} step {r['i']} ({r['title']}): a step with a control offers no "
        "Start Doing This button"
    )
    assert r["phaseB"] == "start-button", r
    h = r["handsOn"]
    assert h["on"] is True, (
        f"{want_type} step {r['i']} ({r['title']}): Start Doing This did not hand over "
        "to the floating orb"
    )
    # Unobstructed: the dim layer, the ring and the card all leave the screen.
    assert h["cardOn"] is False, (
        f"{want_type} step {r['i']} ({r['title']}): the card is still covering the "
        "workspace in hands-on mode"
    )
    assert h["spotOn"] is False, (
        f"{want_type} step {r['i']} ({r['title']}): the dim layer is still up in "
        "hands-on mode"
    )
    # …and the pill that replaces them is really on screen, not merely flagged.
    assert h["visible"] is True, (
        f"{want_type} step {r['i']} ({r['title']}): the floating orb is not visible"
    )
    assert "Step " + str(r["i"] + 1) + "/" in h["text"], h["text"]
    # Nothing has been done yet, so the badge must not claim otherwise.
    if r["live"]:
        assert h["ready"] is False, (
            f"{want_type} step {r['i']} ({r['title']}): the badge says the step is done "
            "before the user has done anything"
        )


def _check_live(want_type: str, r: dict) -> None:
    """Doing the step's own job must be what turns the badge green."""
    if not r["live"] or "afterBump" not in r:
        return
    assert r["bump"], f"{want_type} step {r['i']}: nothing to change for {r['live']!r}"
    after = r["afterBump"]
    assert after["ready"] is True, (
        f"{want_type} step {r['i']} ({r['title']}): {r['live']} changed ({r['bump']}) but "
        f"the badge stayed as it was; the pill reads {after['text']!r}"
    )
    assert "Step Ready" in after["text"], after["text"]
    assert "Done" in after["goText"], after["goText"]
    if r["live"] == "samples":
        assert "Collecting Samples" in after["text"], after["text"]
    elif r["live"] == "train":
        assert "Model Trained" in after["text"], after["text"]
    elif r["live"] == "thresholds":
        assert "Adjusting ROI" in after["text"], after["text"]
    elif r["live"] == "signseq":
        assert "X" in after["text"], after["text"]


# Which screen each door opens, and the driver-record key that reports it.
# A step in its trigger phase must leave exactly that screen open behind it.
_DOOR_SCREENS = {
    "#advBtn": "advanced",
    "#navSettings": "settings",
    "#menuEditBtn": "editMenu",
    ".class-preprocess-chip": "worktable",
    ".tm-action-toolbar button.sample": "drawer",
}


def _check_trigger_step(want_type: str, r: dict) -> None:
    """A step whose screen is shut asks for the door, then follows the user in.

    The same step index covers both halves: the spotlight starts on the
    control that opens the screen, and pressing it opens that screen and
    hands the workspace over — without skipping over the screen it
    documents, and without leaving the ring on a button the user has
    already dealt with.
    """
    assert r["sel"] in _DOOR_SCREENS, (
        f"{want_type} step {r['i']} ({r['title']}): the trigger phase spotlights "
        f"{r['sel']!r}, which is not one of the screens this walk knows how to open"
    )
    assert r["cue"] == "phase", r
    # While the door is spotlit the Settings modal is still shut — that is
    # the whole point of the phase.
    assert r["settingsExpected"] is False, r
    assert r["settingsOpen"] is False, r
    # Start Doing This presses the door for the user, so the screen the step
    # documents is open when the workspace is handed over.
    screen = _DOOR_SCREENS[r["sel"]]
    assert r["doorSel"] == r["sel"], r
    assert r["door"][screen] is True, (
        f"{want_type} step {r['i']} ({r['title']}): {r['sel']} did not open {screen}"
    )
    if screen != "settings":
        # Every other door opens its own screen and nothing else.
        assert r["door"]["settings"] is False, r


def check_real_controls(payload: dict, want_type: str) -> dict:
    """Using the spotlighted control hands over — it never teleports.

    This is the complaint the redesign answers: a click used to move the tour
    on before the user had finished working.  Now the same press only swaps
    the card for the pill, and the pill's own click is what advances.
    """
    rows = _records(payload, "auto")
    assert rows, f"{want_type}: no real control was ever exercised"
    checked = {r["sel"]: r for r in rows}
    for r in rows:
        assert r["orbBefore"] is False, r
        assert r["index"] == r["i"] == r["before"], (
            f"{want_type} step {r['i']}: {r['sel']} ({r['action']}) moved the guide to "
            f"step {r['index']} — a real click must never advance the tour"
        )
        assert r["orb"] is True, (
            f"{want_type} step {r['i']}: {r['sel']} did not collapse the card into the "
            "floating orb"
        )
        assert r["cardOn"] is False and r["spotOn"] is False, (
            f"{want_type} step {r['i']}: {r['sel']} left the tour covering the workspace"
        )
        # Keep working: still nothing moves on its own.
        assert r["indexAgain"] == r["i"], (
            f"{want_type} step {r['i']}: a second press on {r['sel']} moved the guide to "
            f"step {r['indexAgain']}"
        )
        assert r["orbAgain"] is True, r
    return checked


def check_rebind(payload: dict, want_type: str) -> None:
    """Rebuilding the class column must not disarm the step.

    `render()` begins `#classes.innerHTML = ''`, so the node the spotlight was
    measured against and the node the click binding sat on are both thrown
    away.  A binding that still looks armed is dead: the guide has to notice
    and re-arm onto the fresh node, or the control the card is pointing at
    stops responding — which is exactly what the user sees after a capture.
    """
    rows = _records(payload, "rebind")
    assert len(rows) == 1, rows
    r = rows[0]
    assert r["err"] is None, r["err"]
    assert r["sel"] == ".tm-action-toolbar", (
        f"{want_type}: the step stopped resolving after the class column was "
        f"rebuilt (found {r['sel']!r})"
    )
    assert r["replaced"] is True, (
        f"{want_type}: render() did not replace the step's node, so this check "
        "never exercised a re-armed binding"
    )
    assert r["orbBefore"] is False, r
    assert r["orb"] is True, (
        f"{want_type}: after the class column was rebuilt, pressing the step's "
        "own control no longer handed over — the advance binding was still on "
        "the node render() had thrown away"
    )
    assert r["cardOn"] is False and r["spotOn"] is False, r
    assert r["ready"] is True, (
        f"{want_type}: the pill's badge stayed grey after a real press on the "
        f"re-armed control; it reads {r['text']!r}"
    )
    assert r["index"] == r["i"], (
        f"{want_type}: the re-armed control moved the guide to step {r['index']}"
    )


def check_container_events(payload: dict, want_type: str) -> None:
    """A container step answers to click AND to input / change.

    Most spotlights are containers (`.tm-action-toolbar`) whose real control is
    a descendant, so gating the binding on the spotlight's own tag left every
    one of them silent.  Each of the three events the user's own work produces
    must hand the workspace over and turn the badge green — and none of them
    may move the tour.
    """
    rows = _records(payload, "container")
    assert len(rows) == 3, rows
    assert [r["action"] for r in rows] == ["click", "input", "change"], rows
    for r in rows:
        assert r["sel"] == ".tm-action-toolbar", r
        assert r["isContainer"] is True, r
        assert r["orbBefore"] is False, r
        assert r["indexBefore"] == r["i"], r
        assert r["orb"] is True, (
            f"{want_type}: a real {r['action']} on the container step did not "
            "hand the workspace over to the pill"
        )
        assert r["cardOn"] is False and r["spotOn"] is False, r
        assert r["ready"] is True, (
            f"{want_type}: the badge stayed grey after a real {r['action']} on "
            f"the container; the pill reads {r['text']!r}"
        )
        assert r["index"] == r["i"], (
            f"{want_type}: a real {r['action']} moved the guide to step {r['index']}"
        )


def check_escape_layer(payload: dict, want_type: str) -> None:
    """Escape closes the layer on top, and leaves the guide running.

    A modal or the source drawer is the topmost layer, so the key belongs to
    it: Escape must close that and leave the tour exactly where it was,
    instead of killing the guide standing behind the panel.
    """
    rows = _records(payload, "escape")
    assert len(rows) == 2, rows
    by_layer = {r["layer"]: r for r in rows}
    assert set(by_layer) == {"settings", "drawer"}, sorted(by_layer)
    for layer, r in by_layer.items():
        assert r["opened"] is True, (
            f"{want_type}: the {layer} never opened, so Escape had nothing to close"
        )
        assert r["orbBefore"] is True and r["cardBefore"] is False, (
            f"{want_type}: the workspace was not handed over before Escape ({r['layer']})"
        )
        assert r["closed"] is True, (
            f"{want_type}: Escape did not close the {layer} covering the workspace"
        )
        assert r["open"] is True, (
            f"{want_type}: Escape closed the guide instead of the open {layer}"
        )
        assert r["index"] == r["i"], (
            f"{want_type}: Escape moved the guide from step {r['i']} to "
            f"step {r['index']}"
        )


def check_ask_orb(payload: dict, want_type: str) -> None:
    """`?` on the pill brings the instruction card back for the same step."""
    rows = _records(payload, "ask")
    assert len(rows) == 1, rows
    r = rows[0]
    assert r["before"]["on"] is True and r["before"]["cardOn"] is False, r
    assert r["after"]["on"] is False, (
        f"{want_type}: `?` left the pill up while re-opening the card"
    )
    assert r["after"]["cardOn"] is True, (
        f"{want_type}: `?` did not bring the instruction card back"
    )
    assert r["after"]["spotOn"] is True, r
    assert r["after"]["i"] == r["before"]["i"], (
        f"{want_type}: `?` moved the guide from step {r['before']['i']} to {r['after']['i']}"
    )


def check_closed(payload: dict, want_type: str) -> None:
    """A closed guide leaves nothing behind — no card, no ring, no pill."""
    rows = _records(payload, "closed")
    assert len(rows) == 1, rows
    r = rows[0]
    assert r["orbVisible"] is False, (
        f"{want_type}: the floating orb is still on screen after the tour closed"
    )
    assert r["cardOn"] is False and r["spotOn"] is False, r
    assert r["arrow"] is False and r["beacon"] is False, r


def check_disabled_guard(payload: dict) -> None:
    """Nothing trained, no samples: the two locked buttons must say so."""
    assert payload["errors"] == [], payload["errors"]
    records = _records(payload, "step")
    assert len(records) == EXPECTED_STEPS["image"] - 1, len(records)
    for step in records:
        assert step["cue"] == step["expected"], step
        if step["expected"] != "disabled":
            continue
        assert "Collect samples first" in step["cueText"], step["cueText"]
        # A locked button has no hands-on mode to offer: pressing it starts
        # nothing, and the card is what tells the user what to do instead.
        assert step["orbOn"] is False or step["phaseB"] == "start-button", step
    locked = [r["sel"] for r in records if r["expected"] == "disabled"]
    assert "#trainBtn" in locked, locked
    assert "#exportBtn" in locked, locked
    # The training card is reached through its trigger: pressing Advanced
    # opens the panel, whose rows are ordinary settings, so the lock sits on
    # Train — the step after it, one screen further in.
    door = [r for r in records if r["phase"] and r["sel"] == "#advBtn"]
    assert len(door) == 1, door
    assert door[0]["cue"] == "phase", door[0]
    assert door[0]["door"]["advanced"] is True, door[0]


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
    # Every curriculum opens on an overview that lists the whole trip.
    for t, steps in spec.items():
        assert steps[0]["overview"] is True, f"{t} does not open on the overview"
        assert len(steps) == EXPECTED_STEPS[t], t

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
    for t in ("image", "ocr", "sign_ocr"):
        run(f"{t} overview opens the guide", lambda t=t: check_overview(reports[t], t, 5))

    def curriculum(t):
        checked[t] = check_curriculum(reports[t], t)

    for t in ("image", "ocr", "sign_ocr"):
        run(f"{t} curriculum, cues and hand-over", lambda t=t: curriculum(t))

    def distinct():
        assert len(checked) == 3, "a curriculum check failed; not compared"
        titles = [tuple(checked[t]["total"]["titles"]) for t in ("image", "ocr", "sign_ocr")]
        assert len(set(titles)) == 3, "the three project types do not have distinct curricula"

    run("three distinct curricula", distinct)

    for t in ("image", "ocr", "sign_ocr"):
        run(
            f"{t} real controls hand over without advancing",
            lambda t=t: check_real_controls(reports[t], t),
        )
        run(f"{t} a rebuilt class column re-arms the step",
            lambda t=t: check_rebind(reports[t], t))
        run(f"{t} containers answer to click and to input/change",
            lambda t=t: check_container_events(reports[t], t))
        run(f"{t} Escape closes the open layer, not the guide",
            lambda t=t: check_escape_layer(reports[t], t))
        run(f"{t} `?` re-opens the card", lambda t=t: check_ask_orb(reports[t], t))
        run(f"{t} a closed guide leaves nothing behind",
            lambda t=t: check_closed(reports[t], t))

    run("a locked control does not trap the user", lambda: check_disabled_guard(locked))
    run("the start page renders every card first", check_start_page)

    passed = sum(1 for _, ok in results if ok)
    print(f"\n{passed}/{len(results)} passed")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
