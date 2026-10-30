"""Automated checks for the workspace's interactive tour (`window.tmTour`).

The tour is JavaScript emitted from the workspace f-string, so there is
nothing to import.  `spa_harness` evaluates that f-string and serves the
resulting page on a real localhost origin, and this test drives it in the
same Chrome the app ships in — launched head-less from the command line, so
the check needs no test-only dependency (no Playwright, no Node).

The guide is built as the scene-based STEM rhythm a student actually works
in, so that is what is checked:

  curricula     the guide selected for a project type is that project's own
                curriculum: 11 steps for `image`, 12 for `ocr`, 14 for
                `sign_ocr`, with three distinct step lists
  overview      every type opens on step 0 as a Project Overview: a concept
                card that teaches What / Why / How and lists the whole trip
                as numbered stages, with no spotlight, no arrow and no
                hands-on button — and exactly one way forward, Start Walkthrough →
  nav steps     performing the highlighted action advances the guide
                DIRECTLY to the next step (180 ms settle, no floating orb
                in between); the footer carries Skip / Back / Next and no
                Start button; steps whose precondition is already met
                (Settings open, the plate typed) auto-advance on entry
  concept steps interacting inside the highlighted panel never collapses
                the card; the Next button (with the step's own label)
                advances, and the training step advances on its own when
                the run finishes
  hands-on      only the two hands-on scenes (collect samples, tune
                thresholds) offer the wide 🛠️ Start button: it collapses
                the tour into `#tmTourOrb`, which carries LIVE progress
                (`X/N Classes ≥20 Samples` / `X/N Classes Tuned`) and turns
                green only when the criteria really hold
  orb           the pill is fixed bottom-right (no dragging), `📖 Review
                Guide` brings the card back for the same step, `Next Step →`
                moves the tour on, and an early advance flashes a warning
                but never soft-locks the student
  re-armed      the class column is thrown away and rebuilt after every
                capture (`#classes.innerHTML = ''`), which replaces the very
                node the click binding sat on: the guide must re-seat onto
                the fresh node, so pressing the control still hands over
  containers    a hands-on step whose spotlight is a container rather than
                the field inside it answers to click AND to input / change —
                the three events the student's own work produces
  escape order  Escape belongs to whichever layer is on top: with a modal
                open it closes that layer and leaves the guide running,
                rather than killing the tour standing behind it
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

from spa_harness import STUB_FETCH, boot_payload, build_spa_html, serve
# The hard-gate / smart-orb, instant-completion and alignment scenarios are
# shared verbatim with the refactored suite so the two never drift: one set
# of drivers, one set of measurements.
from test_tm_tour_refactored import (
    SMART_DRIVER as _SMART_DRIVER,
    _box_frames_toolbar,
    _home_fixture,
    _unescape_html,
    check_hero_description_width,
    check_hero_matches_workspace,
    check_instant_training_completion,
    check_ocr_collection_gate,
    check_ocr_end_to_end,
    check_preview_payload_compaction,
    check_preview_performance,
    check_workspace_alignment,
)

CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"

# The curriculum each project type must load.  Hard-coded on purpose: the
# number of steps is the thing a user notices, so it is the thing the test
# states, rather than something read back out of the source it checks.
EXPECTED_STEPS = {"image": 15, "ocr": 17, "sign_ocr": 18}

# Nav steps whose real interaction the walk performs, keyed by the TARGET
# selector — that is the string the driver records for a step, not the
# `advanceTarget` inside it.  Pressing one of these must advance the guide
# directly, with no floating orb in between.
REAL_CLICK = ("#navSettings", "#settingsSave", "#exportBtn",
              ".tm-action-toolbar", ".class-preprocess-chip")

DRIVER = r"""
<script>
(function () {
  var SPEC = __SPEC__;
  var TYPES = __TYPES__;
  var REAL_CLICK = __REAL_CLICK__;
  var OUT = [];
  var DONE = false;
  /* How long the pill's "hands-on mode" greeting stays up before the live
     state takes the text over (the tour promises 1.8s). */
  var ORB_FLASH_MS = 2200;
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

  /* A comma group is an ORDERED preference list (mirrors the engine's own
     `firstVisible`): `.tm-settings-card, #globalSettingsModal` must try the
     inner card first, or the parent modal wins by document order. */
  function firstVisible(selectors) {
    for (var i = 0; i < selectors.length; i++) {
      if (!selectors[i]) continue;
      var parts = String(selectors[i]).split(',');
      for (var k = 0; k < parts.length; k++) {
        var sel = parts[k].trim();
        if (!sel) continue;
        var el = null;
        try { el = document.querySelector(sel); } catch (e) { el = null; }
        if (vis(el)) return { el: el, sel: selectors[i] };
      }
    }
    return null;
  }

  function wait(ms) { return new Promise(function (r) { window.setTimeout(r, ms); }); }

  async function until(fn, ms) {
    var deadline = Date.now() + (ms || 9000);
    while (Date.now() < deadline) {
      var v = false;
      try { v = !!fn(); } catch (e) {}
      if (v) return true;
      await wait(50);
    }
    return false;
  }

  /* Where the tour says it put the ring.  The tour re-seats the spotlight
     on the next animation frame, and head-less Chrome hands frames out on
     its own schedule, so a step is "on screen" when the ring on screen says
     it belongs to that step. */
  function placedAt(i) {
    try { return window.tmTour.place().index === i; } catch (e) { return false; }
  }

  function settle(i, ms) {
    return until(function () { return placedAt(i); }, ms || 9000);
  }

  function describe(el) {
    if (!el) return null;
    return String(el.tagName || '') + '#' + String(el.id || '') +
      '.' + String(el.className || '').slice(0, 22);
  }


  /* A click on a handler-free control INSIDE the container: the tour's
     capture-phase binding sees a real control click (the ghost matches
     ADV_SEL via [role="button"]) while no app handler runs — the Webcam
     tile would start the app's live-source loop, and a synthetic event on
     the container itself would not match ADV_SEL at all. */
  function ghostClick(container) {
    if (!container) return;
    var ghost = document.createElement('span');
    ghost.setAttribute('role', 'button');
    ghost.style.display = 'none';
    container.appendChild(ghost);
    try { ghost.click(); } catch (e) {}
    try { container.removeChild(ghost); } catch (e) {}
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

  function screens() {
    return { advanced: advancedOpen(), settings: settingsOpen() };
  }

  /* The screen a step documents is SHUT when its trigger is still needed.
     This is the `phaseIf` each curricula step declares, restated — the door
     the step names is the key. */
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
     `armTargetAdvance` derives it: from the step's mode, the resolved
     element, the named `advanceTarget`, and that control's own disabled /
     tag state.  Concept steps are never armed: cue off. */
  function expectedCue(step, found) {
    if (!step.mode || step.mode === 'concept') return 'off';
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

  /* Do the scene's own job, for real, so the pill has something live to
     report.  `samples` and `train` arrive the way the backend delivers them
     (no event at all — the pill polls for those); `thresholds` arrive the
     way the backend persists them (STATE.sample_preprocess). */
  function bumpLive(kind) {
    if (kind === 'samples') {
      var names = (typeof STATE !== 'undefined' && Array.isArray(STATE.classes))
        ? STATE.classes.slice() : [];
      var next = {};
      names.forEach(function (n) { next[n] = 20; });
      STATE.counts = next;
      return 'counts=' + JSON.stringify(next);
    }
    if (kind === 'thresholds') {
      var names2 = (typeof STATE !== 'undefined' && Array.isArray(STATE.classes))
        ? STATE.classes.slice() : [];
      var map = {};
      names2.forEach(function (n) { map[n] = { 'probe.png': { mode: 'auto' } }; });
      STATE.sample_preprocess = map;
      return 'sample_preprocess=' + JSON.stringify(map);
    }
    if (kind === 'train') {
      try { STATE.trainClassAccuracies = { 'A': 0.97, '1': 0.95 }; } catch (e) {}
      return 'trainClassAccuracies';
    }
    return '';
  }

  /* ── step 0: the Project Overview ─────────────────────────────────── */
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

  /* ── steps 1..n-1: per-mode Explain & Point ───────────────────────── */
  async function walkSteps(t, steps, n) {
    var bumped = {};
    for (var i = 1; i < n; i++) {
      window.tmTour.goTo(i);
      /* A nav step whose precondition is already met advances on entry
         (the pre-populated plate, a door already open): record it and
         move on. */
      var autoSkipped = await until(function () { return window.tmTour.current() > i; }, 2500);
      if (autoSkipped) {
        OUT.push({ kind: 'autoskip', type: t, i: i, to: window.tmTour.current(),
          mode: window.tmTour.mode(), orbOn: orb().on });
        continue;
      }
      await settle(i);
      /* The tour re-seats the ring once the layout has settled (280ms), so
         wait out that second look before reading what is on screen. */
      await wait(340);
      var step = steps[i] || {};
      var mode = step.mode || 'handson';
      var res = resolveTargets(step);
      var found = res.found;
      var control = controlOf(step, found);
      var card = byId('tmTourCard');
      var spot = byId('tmTourSpotlight');
      var modal = byId('globalSettingsModal');
      var cueBox = byId('tmTourCue');
      /* Read while the step is on screen, before any action below moves the
         tour on. */
      var layout = null;
      try { layout = window.tmTour.layout(); } catch (e) {}
      var bodyEl = card ? card.querySelector('.tm-tour-body') : null;
      var badgeEl = cueBox ? cueBox.querySelector('.tm-tour-cue-badge') : null;
      var rec = {
        kind: 'step', type: t, i: i,
        title: String(step.title || '').slice(0, 46),
        mode: mode,
        layout: layout,
        dataMode: card ? String(card.getAttribute('data-mode') || '') : '',
        badge: !!badgeEl,
        bodyText: bodyEl ? String(bodyEl.textContent || '') : '',
        bodyHtml: bodyEl ? String(bodyEl.innerHTML || '') : '',
        cardHeight: card ? Math.round(card.getBoundingClientRect().height) : null,
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
        backPresent: !!byId('tmTourBack'),
        startPresent: !!byId('tmTourStart'),
        settingsOpen: settingsOpen(),
        settingsExpected: !!(modal && found && found.el && modal.contains(found.el))
      };

      if (i === n - 1) {
        /* The closing slide ends the guide from its own Next button. */
        var fin = byId('tmTourNext');
        rec.finishText = fin ? String(fin.textContent || '').trim() : '';
        if (fin) fin.click(); else window.tmTour.next();
        await wait(900);
        rec.advanced = window.tmTour.current();
        rec.closed = !window.tmTour.isOpen();
        OUT.push(rec);
        break;
      }

      if (mode === 'handson' && step.settingsPhase) {
        /* A Settings sub-panel auto-advances when its control changes;
           «✓ Confirm & Continue» does the same for the default value. */
        var sBtn = byId('tmTourStart');
        rec.settingsPanel = true;
        rec.startLabel = sBtn ? String(sBtn.textContent || '').trim() : '';
        if (sBtn) sBtn.click();
        await until(function () { return window.tmTour.current() === i + 1; }, 9000);
        rec.advanced = window.tmTour.current();
        rec.orbAfterAdvance = { on: orb().on };
        OUT.push(rec);
        continue;
      }

      if (mode === 'handson') {
        /* Phase A → Phase B: the wide 🛠️ Start button on the card. */
        var startBtn = byId('tmTourStart');
        if (startBtn) { startBtn.click(); rec.phaseB = 'start-button'; }
        else { window.tmTour.handsOn(); rec.phaseB = 'api'; }
        /* Read the hand-over EARLY: live scenes can auto-complete (and
           auto-advance) once their task finishes. */
        await wait(400);
        var ob = orb();
        rec.handsOn = {
          on: ob.on, cardOn: ob.cardOn, spotOn: ob.spotOn, visible: ob.visible,
          text: String(ob.text || '').slice(0, 96), ready: !!(ob.live && ob.live.ready)
        };
        await wait(ORB_FLASH_MS - 400);
        var obLive = orb();
        /* The greeting can outlast the wait under virtual time: keep
           polling until the live "Step N/M" line replaces it. */
        if (String(obLive.text || '').indexOf('Step ') < 0) {
          await until(function () {
            return String(orb().text || '').indexOf('Step ') >= 0;
          }, 4000);
          obLive = orb();
        }
        rec.handsOn.text = String(obLive.text || '').slice(0, 96);
        rec.handsOn.ready = !!(obLive.live && obLive.live.ready);
        /* The live badge: only the first step of each kind is worked on. */
        if (step.live && !bumped[step.live]) {
          bumped[step.live] = true;
          rec.bump = bumpLive(step.live);
          /* Wait for the badge — or for the scene to auto-advance, which
             is the new completion behaviour. */
          await until(function () {
            return orb().ready || window.tmTour.current() === i + 1;
          }, 9000);
          var ob2 = orb();
          rec.afterBump = {
            ready: !!(ob2.live && ob2.live.ready) || window.tmTour.current() === i + 1,
            text: String(ob2.text || '').slice(0, 96),
            goText: String(ob2.goText || '').slice(0, 40)
          };
          if (window.tmTour.current() === i + 1) rec.autoAdvanced = true;
        }
        /* `Next Step →` on the pill (a no-op if auto-advance got there). */
        window.tmTour.orbGo();
        await until(function () { return window.tmTour.current() === i + 1; }, 9000);
        var ob3 = orb();
        rec.advanced = window.tmTour.current();
        rec.orbAfterAdvance = { on: ob3.on, cardOn: ob3.cardOn, visible: ob3.visible };
        OUT.push(rec);
        continue;
      }

      if (mode === 'nav') {
        /* The action IS the answer: press the real control and the guide
           advances directly — never into the floating orb. */
        var sel = found ? found.sel : null;
        if (REAL_CLICK.indexOf(sel) >= 0) {
          var orbSeen = false;
          var watcher = window.setInterval(function () {
            if (orb().on) orbSeen = true;
          }, 40);
          doReal(step, found, control, 'click');
          await until(function () { return window.tmTour.current() === i + 1; }, 9000);
          window.clearInterval(watcher);
          rec.action = sel;
          rec.advanced = window.tmTour.current();
          rec.orbSeenDuringAction = orbSeen;
          rec.orbAfterAction = orb().on;
        } else {
          /* Locked nav steps hide Next: press the control the card names. */
          doReal(step, found, control, 'click');
          await until(function () { return window.tmTour.current() === i + 1; }, 9000);
          rec.advanced = window.tmTour.current();
        }
        OUT.push(rec);
        continue;
      }

      /* concept: free exploration, the card's own Next button advances. */
      byId('tmTourNext').click();
      await until(function () { return window.tmTour.current() === i + 1; }, 9000);
      rec.advanced = window.tmTour.current();
      OUT.push(rec);
    }
  }

  /* ── the real control, per mode: nav advances, hands-on hands over ── */
  async function walkRealControls(t, steps, n) {
    var picked = 0;
    for (var i = 1; i < n && picked < 5; i++) {
      var step = steps[i] || {};
      /* The closing slide's Next button is the main walk's business. */
      if (i >= n - 1) continue;
      if (step.overview || !step.target) continue;
      /* Step 3 documents the modal's Save button: with the modal shut the
         step resolves to its door instead of the button (and its
         `advanceWhen` auto-advances on entry), so open the screen first —
         before anything resolves. */
      if (step.target === '#settingsSave' && !settingsOpen()) {
        try { openGlobalSettings(); } catch (e) {}
        await wait(300);
      }
      var res = resolveTargets(step);
      /* Doors are the walk's business: with the screen shut the spotlight is
         on the trigger, not on the control the step documents. */
      if (res.phase || !res.found) continue;
      var sel = res.found.sel;
      if (REAL_CLICK.indexOf(sel) < 0) continue;

      window.tmTour.start();
      window.tmTour.goTo(i);
      var autoSkipped = await until(function () { return window.tmTour.current() > i; }, 2500);
      if (autoSkipped) { window.tmTour.close(); await wait(140); continue; }
      await settle(i);
      await wait(340);
      var control = controlOf(step, res.found);
      var rec = {
        kind: 'auto', type: t, i: i, sel: sel, action: 'click', mode: step.mode,
        before: window.tmTour.current(), orbBefore: orb().on,
        cueBefore: window.tmTour.cue()
      };
      doReal(step, res.found, control, 'click');
      await wait(800);
      rec.orb = orb().on;
      rec.cardOn = !!(byId('tmTourCard') && byId('tmTourCard').classList.contains('on'));
      rec.spotOn = !!(byId('tmTourSpotlight') && byId('tmTourSpotlight').classList.contains('on'));
      rec.index = window.tmTour.current();
      /* A hands-on student who keeps working must find the tour exactly
         where they left it. */
      if (step.mode === 'handson') {
        doReal(step, res.found, control, 'click');
        await wait(800);
        rec.indexAgain = window.tmTour.current();
        rec.orbAgain = orb().on;
      }
      OUT.push(rec);
      window.tmTour.close();
      await wait(140);
      picked++;
    }
  }

  /* ── `📖 Review Guide` on the pill brings the card back, same step ── */
  async function walkAskOrb(t, steps) {
    var i = stepIndexForMode(steps, 'handson', 'samples');
    if (i < 0) return;
    resetLayers();
    await wait(200);
    window.tmTour.start();
    window.tmTour.goTo(i);
    await settle(i);
    await wait(340);
    var startBtn = byId('tmTourStart');
    if (startBtn) startBtn.click(); else window.tmTour.handsOn();
    await wait(ORB_FLASH_MS);
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
    window.tmTour.close();
    await wait(140);
    resetLayers();
  }

  /* ── support for the three checks below ───────────────────────────── */
  function stepIndexFor(steps, sel) {
    for (var i = 1; i < steps.length; i++) {
      if (steps[i] && steps[i].target === sel) return i;
    }
    return -1;
  }

  function stepIndexForTargetContaining(steps, sel) {
    for (var i = 1; i < steps.length; i++) {
      if (steps[i] && steps[i].target && steps[i].target.indexOf(sel) >= 0) return i;
    }
    return -1;
  }

  function stepIndexForMode(steps, mode, live) {
    for (var i = 1; i < steps.length; i++) {
      if (steps[i] && steps[i].mode === mode && steps[i].live === live) return i;
    }
    return -1;
  }

  function resetLayers() {
    try {
      if (settingsOpen() && typeof closeGlobalSettings === 'function') closeGlobalSettings();
    } catch (e) {}
    try {
      if (advancedOpen()) {
        trainAdvancedOpen = false;
        if (typeof renderAdvancedPanel === 'function') renderAdvancedPanel();
      }
    } catch (e) {}
  }

  /* ── a rebuilt class column must not disarm the scene ───────────────
     `render()` begins `#classes.innerHTML = ''`: the class COLUMN survives
     (it is the container), but every card inside it — and with it the
     armed Edit chip (`.class-preprocess-chip`) — is replaced. */
  async function walkRebind(t, steps) {
    /* The capture scene (not a keep-open one): render() rebuilds the
       class column after every capture, so this is where a stale binding
       would really hurt. */
    var i = stepIndexForMode(steps, 'handson', 'samples');
    if (i < 0) return;
    resetLayers();
    await wait(200);
    window.tmTour.start();
    window.tmTour.goTo(i);
    await settle(i);
    await wait(340);
    var rec = {
      kind: 'rebind', type: t, i: i, err: null,
      keepOpen: !!(steps[i] && steps[i].keepCardOpenOnInteract),
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
    /* The fresh node must be armed again: pressing a control inside it
       hands over — unless the step keeps the card open beside the panel
       it documents, in which case the card must STAY and the tour must
       not move.  The press is a ghost control (no app handler). */
    ghostClick(res.found ? res.found.el : null);
    await wait(3000);
    var ob = orb();
    rec.orb = ob.on;
    rec.cardOn = ob.cardOn;
    rec.spotOn = ob.spotOn;
    rec.text = String(ob.text || '').slice(0, 96);
    rec.index = window.tmTour.current();
    OUT.push(rec);
    window.tmTour.close();
    await wait(160);
    resetLayers();
  }

  /* ── a container scene answers to click AND to input / change ─────── */
  async function walkContainerEvents(t, steps) {
    var i = stepIndexForMode(steps, 'handson', 'samples');
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
        /* Same inert-control press as walkRebind: the Webcam tile starts a
           live-source loop under the stub. */
        ghostClick(el);
      } else if (el) {
        el.dispatchEvent(new Event(action, { bubbles: true }));
      }
      await wait(ORB_FLASH_MS);
      /* The scene's own work: every class reaches the ≥20 bar. */
      bumpLive('samples');
      await until(function () { return orb().ready; }, 9000);
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
    var i = stepIndexForTargetContaining(steps, '#settingsOodGroup');
    if (i < 0) return;
    resetLayers();
    await wait(200);
    window.tmTour.start();
    window.tmTour.goTo(i);
    await settle(i);
    await wait(340);
    /* The concept step's onEnter opened the modal: it is the layer on top. */
    var scr = screens();
    var rec = {
      kind: 'escape', type: t, layer: 'settings', i: i,
      opened: scr.settings, orbBefore: orb().on, cardBefore: !!byId('tmTourCard')
        ? byId('tmTourCard').classList.contains('on') : false,
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
    rec.closed = !scr2.settings;
    rec.open = window.tmTour.isOpen();
    rec.index = window.tmTour.current();
    OUT.push(rec);
    window.tmTour.close();
    await wait(160);
    resetLayers();
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
    await walkAskOrb(t, steps);
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
      arrow: marks() ? marks().arrow : false
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

  /* Watchdog: whatever stalls, the report still lands (with the records
     gathered so far) instead of hanging the harness. */
  window.setTimeout(finish, 600000);

  /* Heartbeat for stall diagnosis: capped at 120 beats. */
  var BEATS = 0;
  window.setInterval(function () {
    if (BEATS < 120) {
      BEATS++;
      OUT.push({ kind: 'beat', t: BEATS, at: (window.tmTour && window.tmTour.current) ? window.tmTour.current() : null });
    }
  }, 2000);

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
                "mode": _quoted(b, "mode") or "handson",
                "target": _quoted(b, "target") or None,
                "fallback": _list(b, "fallback"),
                "advanceTarget": _quoted(b, "advanceTarget") or None,
                "phaseTarget": _quoted(b, "phaseTarget") or None,
                "live": _quoted(b, "live") or None,
                "overview": _flag(b, "overview"),
                "keepOpen": _flag(b, "keepCardOpenOnInteract"),
                "settingsPhase": _flag(b, "settingsPhase"),
                "aimGroup": _quoted(b, "aimGroup") or None,
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
    )
    html = html.replace("</body>", driver + "\n</body>", 1)
    probe = (
        "<script>window.__pageErrors=[];"
        "window.addEventListener('error',function(e){window.__pageErrors.push(String(e.message));});"
        "</script>"
    )
    # The workspace's own fetches must not leave the page (a fetch to the
    # throwaway server returns HTML, so res.json() would throw and abort the
    # handlers that follow it).  STUB_FETCH is bare JS: wrap it in a tag.
    html = html.replace(
        "<!doctype html>",
        "<!doctype html>" + probe + "<script>" + STUB_FETCH + "</script>",
        1,
    )

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

    Step 0 is a concept card the student can only read: What / Why / How
    plus numbered stages covering the whole workflow, no spotlight, no
    arrow, no hands-on mode — and exactly one button, which starts the
    step-by-step guide.
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
    assert r["marks"]["arrow"] is False, r
    assert r["hasStart"] is False, "the overview offers a hands-on button"
    assert r["orbOn"] is False and r["orbVisible"] is False, r
    assert "Start Walkthrough" in r["nextText"], r["nextText"]
    assert r["after"] == 1, (
        f"{want_type}: Start Walkthrough left the guide on step {r['after']}"
    )
    assert r["orbAfter"] is False, r
    assert r["cardAfter"] is True, r
    return total


def check_curriculum(payload: dict, want_type: str) -> dict:
    """One type's walk: step count, modes, cues, marks, forward progress."""
    assert payload["errors"] == [], payload["errors"]
    totals = _records(payload, "total")
    assert len(totals) == 1, totals
    total = totals[0]
    assert total["projectType"] == want_type, total
    want = EXPECTED_STEPS[want_type]
    assert total["parsed"] == want, f"curriculum has {total['parsed']} steps, want {want}"
    assert total["total"] == want, f"the tour reports {total['total']} steps, want {want}"

    autoskips = {r["i"] for r in _records(payload, "autoskip")}
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
        # PART 2: the Action Prompt banner is always on, naming the action.
        assert r["cueText"], (
            f"{want_type} step {r['i']} ({r['title']}): no action prompt rendered"
        )
        # The Settings modal must be in the state THIS step needs: open only
        # for the steps whose control lives inside it, closed for the rest.
        assert r["settingsOpen"] == r["settingsExpected"], (
            f"{want_type} step {r['i']} ({r['title']}): Settings "
            f"{'left open' if r['settingsOpen'] else 'closed'} but the step "
            f"{'needs it open' if r['settingsExpected'] else 'needs it closed'}"
        )
        # Phase A still explains and points: the arrow links the card to the
        # ring.
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
        # ── Compact card: Part B is a goal line + parameter cheat-sheet,
        # and the whole card stays short enough to never wall off the UI.
        assert 'tm-tour-goal' in (r.get("bodyHtml") or ""), (
            f"{want_type} step {r['i']} ({r['title']}): no goal line in the body"
        )
        if r["i"] < want - 1:
            assert 'tm-tour-param-list' in (r.get("bodyHtml") or ""), (
                f"{want_type} step {r['i']} ({r['title']}): no parameter cheat-sheet"
            )
        assert (r.get("cardHeight") or 0) <= 560, (
            f"{want_type} step {r['i']} ({r['title']}): card is "
            f"{r.get('cardHeight')}px tall (want ≤560)"
        )
        # ── Layout: the card never covers the highlighted container, the
        # vector arrow is never a stub, and the prompt wears its badge.
        lay = r.get("layout") or {}
        if lay.get("card") and lay.get("spot") and not lay["spot"].get("dim"):
            c = lay["card"]; s = lay["spot"]
            overlaps = (c["left"] < s["left"] + s["width"] and
                        c["left"] + c["width"] > s["left"] and
                        c["top"] < s["top"] + s["height"] and
                        c["top"] + c["height"] > s["top"])
            assert not overlaps, (
                f"{want_type} step {r['i']} ({r['title']}): #tmTourCard covers "
                f"#tmTourSpotlight (card={c}, spot={s})"
            )
        if r["hasTarget"] and lay.get("arrow"):
            a = lay["arrow"]
            length = ((a["tx"] - a["sx"]) ** 2 + (a["ty"] - a["sy"]) ** 2) ** 0.5
            assert length == length and length >= 32, (
                f"{want_type} step {r['i']} ({r['title']}): vector arrow is only "
                f"{length}px long (want ≥32)"
            )
        assert r["dataMode"] == r["mode"], (
            f"{want_type} step {r['i']} ({r['title']}): card data-mode="
            f"{r['dataMode']!r} but step.mode={r['mode']!r}"
        )
        assert r["badge"] is True, (
            f"{want_type} step {r['i']} ({r['title']}): .tm-tour-cue-badge missing"
        )
        leaked = re.findall(r"#[A-Za-z][\w-]*|\.(?:addclass|class-preprocess-chip|tm-[\w-]+)",
                            r.get("bodyText") or "")
        assert leaked == [], (
            f"{want_type} step {r['i']} ({r['title']}): raw selectors in the body: {leaked}"
        )
        # ← Back must be on EVERY step after the overview.
        if r["i"] > 0:
            assert r["backPresent"] is True, (
                f"{want_type} step {r['i']} ({r['title']}): no ← Back button"
            )
        # Only the hands-on scenes offer the wide Start button.
        assert r["startPresent"] == (r["mode"] == "handson"), (
            f"{want_type} step {r['i']} ({r['title']}): startPresent={r['startPresent']} "
            f"but mode is {r['mode']!r}"
        )

        last = r["i"] == want - 1
        if last:
            # The closing slide ends the guide from its own Next button.
            assert r["finishText"].startswith("Finish"), r["finishText"]
            assert r["closed"] is True, (
                f"{want_type}: the closing slide left the guide open"
            )
            # Finishing consumes the progress: the guide may report 0 or
            # the final index once closed.
            assert r["advanced"] in (r["i"], 0), r
            continue

        if r["mode"] == "handson" and r.get("settingsPanel"):
            # A Settings sub-panel is confirmed and auto-advances; it never
            # flies to the orb (the next panel is the destination).
            assert "Confirm" in r.get("startLabel", ""), r
            assert r["advanced"] == r["i"] + 1, (
                f"{want_type} step {r['i']} ({r['title']}): the panel did not "
                f"auto-advance (landed on {r['advanced']})"
            )
            assert r["orbAfterAdvance"]["on"] is False, r
        elif r["mode"] == "handson":
            _check_hands_on(want_type, r)
            _check_live(want_type, r)
        elif r["mode"] == "nav":
            # Performing the action advances DIRECTLY: the orb never
            # flickers on during the settle beat.
            assert r["advanced"] == r["i"] + 1, (
                f"{want_type} step {r['i']} ({r['title']}): the action left the guide on "
                f"step {r['advanced']}"
            )
            if "action" in r:
                assert r["orbSeenDuringAction"] is False, (
                    f"{want_type} step {r['i']} ({r['title']}): the orb appeared during "
                    "a nav action"
                )
                assert r["orbAfterAction"] is False, r
            assert r.get("orbAfterAdvance", {"on": False})["on"] is False, r
        else:
            assert r["advanced"] == r["i"] + 1, (
                f"{want_type} step {r['i']} ({r['title']}): Next left the guide on "
                f"step {r['advanced']}"
            )
    # Step 0 is the overview, which `check_overview` reads on its own: the
    # walk records one entry per step from 1 to the closing slide, minus any
    # nav step that auto-advanced on entry.
    assert len(records) + len(autoskips) == want - 1, (sorted(records), autoskips)
    return {"total": total, "records": records}


def _check_hands_on(want_type: str, r: dict) -> None:
    """The card's own button must move the student out of the way of their work."""
    assert r["phaseB"] == "start-button", r
    h = r["handsOn"]
    assert h["on"] is True, (
        f"{want_type} step {r['i']} ({r['title']}): Start did not hand over to the "
        "floating orb"
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
    # Nothing has been done yet, so the badge must not claim otherwise —
    # and the in-progress (blue) text must name the scene's live progress.
    # The training scene is the exception: `ready` mirrors the MODEL's
    # state, so a workspace that already has a trained model honestly
    # reads green from the first frame.
    if r["live"] in ("samples", "thresholds"):
        assert h["ready"] is False, (
            f"{want_type} step {r['i']} ({r['title']}): the badge says the scene is done "
            "before the student has done anything"
        )
    if r["live"] == "samples":
        # The smart orb names the NEXT class still short of 20 samples
        # ("Capture Class 'LEFT' (0/20)") instead of a generic count.
        assert ("Capture Class" in h["text"] and "/20" in h["text"]) or (
            "≥20 samples/class" in h["text"]), h["text"]
    elif r["live"] == "thresholds":
        assert "Tuned" in h["text"] or "tuned" in h["text"], h["text"]


def _check_live(want_type: str, r: dict) -> None:
    """Doing the scene's own job must be what turns the badge green."""
    if not r["live"] or "afterBump" not in r:
        return
    assert r["bump"], f"{want_type} step {r['i']}: nothing to change for {r['live']!r}"
    after = r["afterBump"]
    assert after["ready"] is True, (
        f"{want_type} step {r['i']} ({r['title']}): {r['live']} changed ({r['bump']}) but "
        f"the badge stayed as it was; the pill reads {after['text']!r}"
    )
    if r.get("autoAdvanced"):
        # Completion auto-advances: no go-button text is left to read.
        assert r["advanced"] == r["i"] + 1, r
        return
    assert "✅ Next Step" in after["goText"], after["goText"]
    if r["live"] == "samples":
        assert "All 2/2 Classes ≥20 Samples" in after["text"], after["text"]
    elif r["live"] == "thresholds":
        assert "2/2 Classes Tuned" in after["text"], after["text"]


def check_real_controls(payload: dict, want_type: str) -> dict:
    """Using the real control behaves exactly as the step's mode promises.

    A nav press advances the guide directly — no orb, no teleporting.  A
    hands-on press hands the workspace over and a second press moves the
    tour no further, which is what lets a student capture twenty photos
    without the guide walking off.
    """
    rows = _records(payload, "auto")
    assert rows, f"{want_type}: no real control was ever exercised"
    checked = {r["sel"]: r for r in rows}
    for r in rows:
        assert r["orbBefore"] is False, r
        assert r["index"] == r["i"] or r["index"] >= r["i"] + 1, (
            f"{want_type} step {r['i']}: {r['sel']} moved the guide backwards to "
            f"step {r['index']}"
        )
        if r["mode"] == "nav":
            # The press advances directly; a follower step whose
            # precondition is already met (the pre-populated plate) may
            # cascade one step further.
            assert r["index"] >= r["i"] + 1, (
                f"{want_type} step {r['i']}: a nav press on {r['sel']} did not advance "
                f"the guide (still on {r['index']})"
            )
            assert r["orb"] is False, (
                f"{want_type} step {r['i']}: a nav press on {r['sel']} opened the "
                "floating orb"
            )
        else:
            assert r["index"] == r["i"], (
                f"{want_type} step {r['i']}: a real press on {r['sel']} moved the "
                f"guide to step {r['index']} — a hands-on press must never advance"
            )
            assert r["orb"] is True, (
                f"{want_type} step {r['i']}: {r['sel']} did not collapse the card into "
                "the floating orb"
            )
            assert r["cardOn"] is False and r["spotOn"] is False, (
                f"{want_type} step {r['i']}: {r['sel']} left the tour covering the "
                "workspace"
            )
            # Keep working: still nothing moves on its own.
            assert r["indexAgain"] == r["i"], (
                f"{want_type} step {r['i']}: a second press on {r['sel']} moved the "
                f"guide to step {r['indexAgain']}"
            )
            assert r["orbAgain"] is True, r
    return checked


def check_ask_orb(payload: dict, want_type: str) -> None:
    """`📖 Review Guide` brings the instruction card back for the SAME step."""
    rows = _records(payload, "ask")
    assert len(rows) == 1, rows
    r = rows[0]
    assert r["before"]["on"] is True, r
    assert r["after"]["on"] is False, r
    assert r["after"]["cardOn"] is True, r
    assert r["after"]["spotOn"] is True, r
    assert r["after"]["i"] == r["before"]["i"], (
        f"{want_type}: Review Guide changed the step from {r['before']['i']} to "
        f"{r['after']['i']}"
    )


def check_rebind(payload: dict, want_type: str) -> None:
    """Rebuilding the class column must not disarm the scene.

    `render()` begins `#classes.innerHTML = ''`, so the node the spotlight was
    measured against and the node the click binding sat on are both thrown
    away.  A binding that still looks armed is dead: the guide has to notice
    and re-arm onto the fresh node, or the control the card is pointing at
    stops responding — which is exactly what the student sees after a capture.
    """
    rows = _records(payload, "rebind")
    assert len(rows) == 1, rows
    r = rows[0]
    assert r["err"] is None, r["err"]
    assert r["sel"] in (".source-panel, .class-card", ".source-panel", ".class-card"), (
        f"{want_type}: the scene stopped resolving after the class column was "
        f"rebuilt (found {r['sel']!r})"
    )
    assert r["replaced"] is True, (
        f"{want_type}: render() did not replace the step's node, so this check "
        "never exercised a re-armed binding"
    )
    assert r["orbBefore"] is False, r
    if r.get("keepOpen"):
        assert r["orb"] is False and r["cardOn"] is True, (
            f"{want_type}: a keep-open step collapsed its card on an in-panel press"
        )
    else:
        assert r["orb"] is True, (
            f"{want_type}: after the class column was rebuilt, pressing the scene's "
            "own control no longer handed over — the advance binding was still on "
            "the node render() had thrown away"
        )
        assert r["cardOn"] is False and r["spotOn"] is False, r
    assert r["index"] == r["i"], (
        f"{want_type}: the re-armed control moved the guide to step {r['index']}"
    )


def check_container_events(payload: dict, want_type: str) -> None:
    """A container scene answers to click AND to input / change.

    Most scene spotlights are containers (`.col-classes`) whose real control
    is a descendant, so gating the binding on the spotlight's own tag left
    every one of them silent.  Each of the three events the student's own
    work produces must hand the workspace over, and the badge turns green
    once the scene's criteria hold — and none of them may move the tour.
    """
    rows = _records(payload, "container")
    assert len(rows) == 3, rows
    assert [r["action"] for r in rows] == ["click", "input", "change"], rows
    for r in rows:
        assert r["sel"] in (".source-panel, .class-card", ".source-panel", ".class-card", ".col-classes"), r
        assert r["isContainer"] is True, r
        assert r["orbBefore"] is False, r
        assert r["indexBefore"] == r["i"], r
        assert r["orb"] is True, (
            f"{want_type}: a real {r['action']} on the container scene did not "
            f"hand the workspace over to the pill: {r}"
        )
        assert r["cardOn"] is False and r["spotOn"] is False, r
        assert r["ready"] is True, (
            f"{want_type}: the badge stayed grey after the scene's own work; "
            f"the pill reads {r['text']!r}"
        )
        assert "Classes ≥20 Samples" in r["text"], r["text"]
        assert r["index"] == r["i"], (
            f"{want_type}: a real {r['action']} moved the guide to step {r['index']}"
        )


def check_escape_layer(payload: dict, want_type: str) -> None:
    """Escape closes the layer on top, and leaves the guide running.

    The Settings modal is the topmost layer, so the key belongs to it:
    Escape must close that and leave the tour exactly where it was,
    instead of killing the guide standing behind the panel.
    """
    rows = _records(payload, "escape")
    assert len(rows) == 1, rows
    r = rows[0]
    assert r["opened"] is True, (
        f"{want_type}: the concept step's onEnter did not open the Settings modal"
    )
    assert r["closed"] is True, (
        f"{want_type}: Escape did not close the Settings modal on top"
    )
    assert r["open"] is True, (
        f"{want_type}: Escape killed the guide instead of the modal"
    )
    assert r["index"] == r["i"], (
        f"{want_type}: Escape moved the guide from step {r['i']} to {r['index']}"
    )


def check_closed(payload: dict, want_type: str) -> None:
    """Closing the tour takes every layer of it away with it."""
    rows = _records(payload, "closed")
    assert len(rows) == 1, rows
    r = rows[0]
    assert r["orbVisible"] is False, r
    assert r["cardOn"] is False, r
    assert r["spotOn"] is False, r
    assert r["arrow"] is False, r


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


def check_home_cards() -> None:
    """Every home card offers the same three actions; the badge never wraps.

    The Start page is Streamlit, so this asserts the source contract the
    browser test (`test_tm_tour_refactored.check_home_cards`) then measures
    against the app's own CSS.
    """
    src = (ROOT / "app.py").read_text(encoding="utf-8")
    assert src.count('st.button("Import classified class"') == 3, (
        "all three project cards must offer Import classified class"
    )
    assert '"tm_start_classified_ocr"' in src, "OCR card lacks its import button"
    assert 'tm_classified_target = PROJECT_TYPE_OCR' in src, (
        "OCR import does not route into the OCR workspace"
    )
    assert 'project_type = target_mode' in src and '_tm_apply_project_defaults(target_mode)' in src, (
        "classified import does not restore the requesting project mode"
    )
    for token in (".tm-start-card-headrow", "white-space: nowrap; flex-shrink: 0;",
                  "margin-top: auto;", "flex: 1 1 auto; margin: 0 0 16px;"):
        assert token in src, f"home card alignment CSS missing: {token}"
    # `Import classified class` opens the folder picker: macOS osascript
    # first, and the tkinter fallback is fully guarded so a build without
    # `_tkinter` reports "no folder" instead of a traceback.
    pick = _block(src, "def _pick_directory_dialog(initial_dir: Optional[str] = None) -> Optional[str]:")
    assert '["osascript", "-e", script]' in pick, (
        "the folder picker must try osascript before tkinter")
    assert "import tkinter" in pick and "except Exception:" in pick, (
        "the tkinter fallback must be guarded (no `_tkinter` error)")
    assert "_pick_directory_dialog(initial_dir=st.session_state.local_import_path)" in src, (
        "Import classified class does not use the guarded folder picker")
    print("PASS home cards: 3 buttons per card, OCR import wired, aligned CSS "
          "present, folder picker never surfaces a tkinter error")


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


# ── Phase 5➔6 multi-class guard, card chrome, warning-bar lifecycle ───────
def _run_extra(driver: str, out_id: str, marker: str, overrides: dict) -> dict:
    html = build_spa_html(boot_payload(**overrides))
    html = html.replace("</body>", driver + "\n</body>", 1)
    probe = (
        "<script>window.__pageErrors=[];"
        "window.addEventListener('error',function(e){window.__pageErrors.push(String(e.message));});"
        "</script>"
    )
    html = html.replace(
        "<!doctype html>",
        "<!doctype html>" + probe + "<script>" + STUB_FETCH + "</script>", 1)
    with serve(html) as url:
        proc = subprocess.run(
            [CHROME, "--headless=new", "--disable-gpu", "--no-sandbox",
             "--window-size=1600,1000", "--virtual-time-budget=90000",
             "--dump-dom", url],
            capture_output=True, text=True, timeout=300)
    m = re.search(r'<pre[^>]*id="' + out_id + r'"[^>]*>' + marker + r'::(.*?)</pre>',
                  proc.stdout, re.S)
    if m is None:
        raise AssertionError("the page never reported a result; tail:\n"
                             + proc.stdout[-2000:])
    return json.loads(m.group(1))


REGRESSION_DRIVER = r"""
<script>
(function () {
  var OUT = [];
  function byId(id) { return document.getElementById(id); }
  function wait(ms) { return new Promise(function (r) { window.setTimeout(r, ms); }); }
  async function until(fn, ms) {
    var deadline = Date.now() + (ms || 12000);
    while (Date.now() < deadline) {
      var v = false;
      try { v = !!fn(); } catch (e) {}
      if (v) return true;
      await wait(50);
    }
    return false;
  }
  function finish() {
    var pre = document.createElement('pre');
    pre.id = 'TM_REG_OUT';
    pre.textContent = 'TM_REG_JSON::' + JSON.stringify({
      rows: OUT, errors: window.__pageErrors || [] });
    document.body.appendChild(pre);
  }
  function modalText() { var m = byId('trainWarningModal'); return m ? m.textContent : null; }
  function bar() {
    var b = byId('trainStatus');
    return b ? { text: b.textContent, cursor: b.style.cursor } : null;
  }
  function snap(k, extra) {
    var o = { kind: k, bar: bar(), modal: modalText(),
      acc: STATE.trainClassAccuracies, warn: STATE.train_warning };
    if (extra) Object.keys(extra).forEach(function (x) { o[x] = extra[x]; });
    OUT.push(o);
  }
  async function tune(v) {
    var el = byId('classPreprocessLum');
    if (!el) return false;
    el.value = String(v);
    el.dispatchEvent(new Event('input', { bubbles: true }));
    el.dispatchEvent(new Event('change', { bubbles: true }));
    await wait(180);
    return true;
  }
  function chromeMetrics() {
    var close = byId('tmTourClose');
    var reset = byId('tmTourRestart');
    var tip = document.querySelector('.tm-tour-meta-tip');
    var tipLine = tip ? tip.querySelector('span:last-child') : null;
    var phase = document.querySelector('.tm-tour-phase');
    var top = document.querySelector('.tm-tour-head-top');
    var kids = top ? Array.prototype.slice.call(top.children).map(function (n) {
      var r = n.getBoundingClientRect();
      return { cy: r.top + (r.height / 2), h: r.height };
    }) : [];
    return {
      closeH: close ? close.getBoundingClientRect().height : -1,
      closeText: close ? String(close.textContent || '').trim() : '',
      resetH: reset ? reset.getBoundingClientRect().height : -1,
      tipSW: tip ? tip.scrollWidth : -1,
      tipCW: tip ? tip.clientWidth : -1,
      tipOverflow: tipLine ? getComputedStyle(tipLine).overflow : '',
      tipText: tip ? String(tip.textContent || '').trim() : '',
      phase: phase ? String(phase.textContent || '').trim() : '',
      kids: kids
    };
  }
  async function openWorktable() {
    window.tmTour.goTo(12);
    await until(function () {
      return window.tmTour.current() === 12 &&
        (typeof classPreprocessOpen !== 'undefined') && classPreprocessOpen;
    }, 12000);
    await wait(400);
  }
  async function multiClass() {
    window.tmTour.__resetProgress();
    window.tmTour.start();
    await wait(300);
    await openWorktable();
    OUT.push({ kind: 'opened', cls: String(classPreprocessClass),
      chrome: chromeMetrics(),
      pos: document.querySelector('.class-preprocess-classpos')
        ? String(document.querySelector('.class-preprocess-classpos').textContent || '').trim() : '',
      nextText: byId('classPreprocessNextClass')
        ? String(byId('classPreprocessNextClass').textContent || '').trim() : null });
    await tune(130);
    OUT.push({ kind: 'afterA', step: window.tmTour.current(),
      tuned: window.tmTour.tunedClasses(), next: window.tmTour.nextUntunedClass(),
      all: window.tmTour.allSampledClassesTuned(),
      done: window.tmTour.completedSteps().indexOf(12) >= 0,
      cue: byId('tmTourCue') ? String(byId('tmTourCue').textContent || '') : '' });
    var nb = byId('classPreprocessNextClass'); if (nb) nb.click();
    await wait(900);
    OUT.push({ kind: 'switch1', cls: String(classPreprocessClass) });
    await tune(140);
    OUT.push({ kind: 'afterB', step: window.tmTour.current(),
      next: window.tmTour.nextUntunedClass(),
      done: window.tmTour.completedSteps().indexOf(12) >= 0 });
    var nb2 = byId('classPreprocessNextClass'); if (nb2) nb2.click();
    await wait(900);
    await tune(150);
    OUT.push({ kind: 'afterC', step: window.tmTour.current(),
      all: window.tmTour.allSampledClassesTuned(),
      done: window.tmTour.completedSteps().indexOf(12) >= 0,
      tuned: window.tmTour.tunedClasses() });
    await wait(1800);
    OUT.push({ kind: 'advanced', step: window.tmTour.current(),
      title: window.tmTour.stepTitle() });
    window.tmTour.__resetProgress();
    window.tmTour.start();
    await wait(300);
    await openWorktable();
    await tune(135);
    var cancel = byId('classPreprocessCancel'); if (cancel) cancel.click();
    await wait(800);
    var aim = null; try { aim = window.tmTour.aim(); } catch (e) {}
    OUT.push({ kind: 'closedEarly', step: window.tmTour.current(),
      open: (typeof classPreprocessOpen !== 'undefined') && classPreprocessOpen,
      next: window.tmTour.nextUntunedClass(),
      cue: byId('tmTourCue') ? String(byId('tmTourCue').textContent || '') : '',
      aim: aim });
    window.tmTour.__resetProgress();
    window.tmTour.start();
    await wait(300);
    await openWorktable();
    var startBtn = byId('tmTourStart');
    if (startBtn) startBtn.click();
    await wait(800);
    OUT.push({ kind: 'orbStart', step: window.tmTour.current(),
      orbOn: window.tmTour.orb().on });
    await tune(160);
    await wait(1900);
    OUT.push({ kind: 'orbAfterA', step: window.tmTour.current(),
      orbOn: window.tmTour.orb().on, orbText: window.tmTour.orb().text,
      tuned: window.tmTour.tunedClasses(),
      next: window.tmTour.nextUntunedClass() });
    var nb3 = byId('classPreprocessNextClass'); if (nb3) nb3.click();
    await wait(800);
    await tune(170);
    var nb4 = byId('classPreprocessNextClass'); if (nb4) nb4.click();
    await wait(800);
    await tune(180);
    OUT.push({ kind: 'orbAllTuned', all: window.tmTour.allSampledClassesTuned() });
    await wait(2000);
    OUT.push({ kind: 'orbAdvanced', step: window.tmTour.current(),
      orbOn: window.tmTour.orb().on, cardOn: window.tmTour.orb().cardOn,
      worktable: (typeof classPreprocessOpen !== 'undefined') && classPreprocessOpen });
    window.tmTour.close();
    await wait(200);
  }
  async function warnings() {
    window.__trainRun = 0;
    window.__statusCount = 0;
    window.__staleReturned = false;
    var RUN1 = { A: 0.2, '1': 0.3 };
    var RUN2 = { A: 0.95, '1': 0.9 };
    window.fetch = async function (url) {
      var u = String(url);
      var body = { ok: '1' };
      if (u.indexOf('/train/start') >= 0) {
        window.__trainRun += 1; window.__statusCount = 0;
      } else if (u.indexOf('/train/status') >= 0) {
        window.__statusCount += 1;
        if (window.__trainRun <= 1) {
          body = window.__statusCount === 1
            ? { ok: '1', progress: 0.4, message: 'Training…', class_accuracies: RUN1 }
            : { ok: '1', done: '1', progress: 1,
                message: 'Done. WARNINGS: class 1 has only 2 samples',
                class_accuracies: RUN1 };
        } else if (window.__statusCount === 1) {
          window.__staleReturned = true;
          body = { ok: '1', progress: 0.3, message: 'Training…', class_accuracies: RUN1 };
        } else {
          body = { ok: '1', done: '1', progress: 1, message: 'Done.', class_accuracies: RUN2 };
        }
      }
      return { ok: true, status: 200,
        json: async function () { return body; },
        text: async function () { return ''; },
        blob: async function () { return new Blob([]); } };
    };
    byId('trainBtn').click();
    await until(function () { return window.__trainRun === 1; }, 5000);
    await wait(80);
    snap('run1-start');
    await until(function () { return !trainInFlight; }, 15000);
    await wait(300);
    snap('run1-done');
    if (byId('trainWarningModal')) byId('trainWarningModal').remove();
    render(); renderTrainStatus();
    await wait(200);
    snap('after-polls', { modalGone: !byId('trainWarningModal') });
    byId('trainStatus').click();
    await wait(100);
    snap('bar-click');
    if (byId('trainWarningModal')) byId('trainWarningModal').remove();
    byId('trainBtn').click();
    await until(function () { return window.__trainRun === 2; }, 5000);
    await wait(40);
    snap('run2-start');
    await until(function () { return window.__statusCount >= 1; }, 5000);
    await wait(80);
    snap('run2-stale-poll', { stale: window.__staleReturned });
    await until(function () { return !trainInFlight; }, 15000);
    await wait(300);
    snap('run2-done');
    if (byId('trainWarningModal')) byId('trainWarningModal').remove();
    byId('trainStatus').click();
    await wait(100);
    snap('run2-click');
    if (byId('trainWarningModal')) byId('trainWarningModal').remove();
    window.navigateParent = function () {};
    window.cleanupWorkspaceFrameBeforeNavigate = function () {};
    var nr = byId('navReturn');
    if (nr) nr.click();
    await wait(500);
    snap('after-leave');
    window.confirm = function () { return true; };
    await newProject();
    await wait(500);
    snap('after-new-project');
  }
  async function run() {
    try {
      try {
        ['image', 'ocr', 'sign_ocr'].forEach(function (t) {
          window.localStorage.setItem('tm_tour_autoskip_' + t + '_v1', '1');
          window.localStorage.setItem('tm_tour_seen_' + t + '_v1', '1');
        });
      } catch (e) {}
      window.fetch = async function (url) {
        var u = String(url);
        var body = { ok: '1' };
        if (u.indexOf('/class_filenames') >= 0) {
          body = { ok: '1', filenames: ['a.png', 'b.png', 'c.png'] };
        }
        return { ok: true, status: 200,
          json: async function () { return body; },
          text: async function () { return ''; },
          blob: async function () { return new Blob([]); } };
      };
      await multiClass();
      await warnings();
    } catch (e) {
      OUT.push({ kind: 'EXC', v: String((e && e.stack) || e) });
    }
    finish();
  }
  if (document.readyState === 'complete') window.setTimeout(run, 400);
  else window.addEventListener('load', function () { window.setTimeout(run, 400); });
})();
</script>
"""


def check_multiclass_chrome_and_warnings() -> None:
    """The SPA-side copy of the refactored suite's two new scenarios."""
    cfg = dict(boot_payload()["train_cfg"])
    cfg["sign_sequence"] = ["A", "1", "2"]
    payload = _run_extra(
        REGRESSION_DRIVER, "TM_REG_OUT", "TM_REG_JSON",
        {"classes": ["A", "1", "2"], "counts": {"A": 16, "1": 16, "2": 14},
         "train_cfg": cfg})
    assert payload["errors"] == [], payload["errors"]
    rows = {r.get("kind"): r for r in payload["rows"]}
    assert "EXC" not in rows, rows["EXC"]

    opened = rows["opened"]
    assert opened["cls"] == "A", opened
    assert 'Class 1 of 3: "A"' in opened["pos"], opened["pos"]
    assert opened["nextText"] and "1" in opened["nextText"], opened["nextText"]
    ch = opened["chrome"]
    assert ch["closeH"] <= 26, f"✕ Close wrapped to {ch['closeH']}px"
    assert ch["closeText"].replace(" ", "") == "✕Close", ch["closeText"]
    assert ch["resetH"] <= 26, ch["resetH"]
    assert ch["tipSW"] <= ch["tipCW"] + 2, (ch["tipSW"], ch["tipCW"])
    assert ch["tipOverflow"] != "hidden", ch["tipOverflow"]
    assert "Click «✕ Close»" in ch["tipText"] and "Tutorial" in ch["tipText"], ch["tipText"]
    assert "5/8" in ch["phase"] or "6/8" in ch["phase"], ch["phase"]
    cys = [k["cy"] for k in ch["kids"]]
    assert max(cys) - min(cys) <= 1.5, f"head-top wrapped: {ch['kids']}"
    assert all(k["h"] <= 26 for k in ch["kids"]), ch["kids"]

    a = rows["afterA"]
    assert a["step"] == 12, a
    assert a["tuned"] == [0] and a["next"] == 1, a
    assert a["all"] is False and a["done"] is False, a
    assert "1 / 3" in a["cue"] and "Next Class ('1')" in a["cue"], a["cue"]
    assert rows["switch1"]["cls"] == "1", rows["switch1"]
    b = rows["afterB"]
    assert b["step"] == 12 and b["next"] == 2 and b["done"] is False, b
    c = rows["afterC"]
    assert c["all"] is True and c["done"] is True, c
    assert c["tuned"] == [0, 1, 2], c["tuned"]
    assert rows["advanced"]["step"] == 13, rows["advanced"]
    early = rows["closedEarly"]
    assert early["step"] == 12 and early["open"] is False, early
    assert early["next"] == 1, early
    assert "1 / 3" in early["cue"] and "«Edit» on Class '1'" in early["cue"], early["cue"]
    assert early["aim"] and early["aim"]["id"] == "editClass-1", early["aim"]
    orb_a = rows["orbAfterA"]
    assert rows["orbStart"]["orbOn"] is True, rows["orbStart"]
    assert orb_a["step"] == 12 and orb_a["orbOn"] is True, orb_a
    assert orb_a["tuned"] == [0] and orb_a["next"] == 1, orb_a
    assert "1 / 3" in orb_a["orbText"], orb_a["orbText"]
    assert rows["orbAllTuned"]["all"] is True, rows["orbAllTuned"]
    orb_done = rows["orbAdvanced"]
    assert orb_done["step"] == 13, orb_done
    assert orb_done["orbOn"] is False and orb_done["cardOn"] is True, orb_done
    assert orb_done["worktable"] is False, orb_done
    for row in rows.values():
        if isinstance(row.get("bar"), dict):
            row["bar"]["text"] = _unescape_html(row["bar"].get("text"))

    run1 = rows["run1-done"]
    assert run1["acc"] == {"A": 0.2, "1": 0.3}, run1["acc"]
    assert "Trained & Ready" in run1["bar"]["text"], run1["bar"]
    assert "WARNINGS" in str(run1["warn"]), run1["warn"]
    assert run1["modal"] and "20%" in run1["modal"], run1["modal"]
    polls = rows["after-polls"]
    assert polls["modalGone"] is True, polls
    assert polls["acc"] == {"A": 0.2, "1": 0.3}, polls["acc"]
    assert "Trained & Ready" in polls["bar"]["text"], polls["bar"]
    click = rows["bar-click"]
    assert click["modal"] and "20%" in click["modal"] and "WARNINGS" in click["modal"], click["modal"]
    start2 = rows["run2-start"]
    assert start2["acc"] is None and start2["warn"] is None and start2["modal"] is None, start2
    assert "Training" in start2["bar"]["text"], start2["bar"]
    stale = rows["run2-stale-poll"]
    assert stale["stale"] is True and stale["acc"] is None, stale
    assert stale["modal"] is None, f"stale warning popped on Train: {stale}"
    run2 = rows["run2-done"]
    assert run2["acc"] == {"A": 0.95, "1": 0.9}, run2["acc"]
    click2 = rows["run2-click"]
    assert "95%" in click2["modal"] and "20%" not in click2["modal"], click2["modal"]
    left = rows["after-leave"]
    assert left["acc"] is None and left["warn"] is None and left["modal"] is None, left
    assert "· A:" not in left["bar"]["text"], left["bar"]
    fresh = rows["after-new-project"]
    assert fresh["acc"] is None and fresh["warn"] is None, fresh
    assert "Not trained" in fresh["bar"]["text"], fresh["bar"]
    print("PASS SPA regressions: 3-class Edit guard holds until A/1/2 are "
          "tuned, ✕ Close stays single-line, meta tip untruncated, warning "
          "bar persists across polls and clears only on Train / leave")


def check_hard_gates_and_smart_orb() -> None:
    """Collect/Preprocess lock the footer; the pill points at the next class.

    Shared driver with `test_tm_tour_refactored` — same measurements, so a
    regression cannot pass one suite and fail the other.
    """
    cfg = dict(boot_payload()["train_cfg"])
    cfg["sign_sequence"] = ["A", "1", "2"]
    payload = _run_extra(
        _SMART_DRIVER, "TM_SMART_OUT", "TM_SMART_JSON",
        {"classes": ["A", "1", "2"], "counts": {"A": 20, "1": 0, "2": 0},
         "train_cfg": cfg})
    assert payload["errors"] == [], payload["errors"]
    rows = {r.get("kind"): r for r in payload["rows"]}
    assert "EXC" not in rows, rows.get("EXC")

    # No Skip / Next on Collect ② until every class is at 20.
    cg = rows["collectGate"]["gate"]
    assert cg["pending"] is True, cg
    assert cg["next"] is False and cg["skip"] is False, cg
    assert cg["hint"] and "20" in cg["hint"], cg["hint"]

    # Orb points at Class 1, not Class 0.
    co = rows["collectOrb"]
    by_idx = {c["idx"]: c for c in co["classes"]}
    assert co["orbOn"] is True, co
    assert _box_frames_toolbar(co["box"], by_idx.get(1)), (co["box"], by_idx.get(1))
    assert not _box_frames_toolbar(co["box"], by_idx.get(0)), (co["box"], by_idx.get(0))
    assert "Class '1' (0/20)" in co["orbText"], co["orbText"]

    # Class 1 full → arrow flies to Class 2.
    ca = rows["collectAfter1"]
    by_idx = {c["idx"]: c for c in ca["classes"]}
    assert ca["next"] and ca["next"]["name"] == "2", ca["next"]
    assert _box_frames_toolbar(ca["box"], by_idx.get(2)), (ca["box"], by_idx.get(2))
    assert not _box_frames_toolbar(ca["box"], by_idx.get(1)), (ca["box"], by_idx.get(1))
    assert "Class 1 full!" in ca["orbText"], ca["orbText"]
    assert "Class '2' (0/20)" in ca["orbText"], ca["orbText"]

    # All collected: card returns, guide advances.
    cd = rows["collectDone"]
    assert cd["all"] is True and cd["step"] == 11, cd
    assert cd["orbOn"] is False and cd["cardOn"] is True, cd

    # Preprocess ② locked until every sampled class is tuned.
    pg = rows["preGate"]["gate"]
    assert pg["pending"] is True, pg
    assert pg["next"] is False and pg["skip"] is False, pg
    assert pg["hint"] and "tuned" in pg["hint"], pg["hint"]
    pa = rows["preAfterA"]
    assert pa["aim"] and pa["aim"]["id"] == "classPreprocessNextClass", pa["aim"]
    assert "1 / 3" in pa["cue"], pa["cue"]
    pc = rows["preClosed"]
    assert pc["aim"] and pc["aim"]["id"] == "editClass-1", pc["aim"]
    assert pc["gate"]["next"] is False and pc["gate"]["skip"] is False, pc["gate"]
    assert "Progress: 1 / 3 tuned" in pc["cue"], pc["cue"]

    # Orb frames the Edit chip; the last tune closes the modal, docks the
    # card and lands on Step 14 (Phase 6).
    po = rows["preOrb"]
    assert po["orbOn"] is True and po["box"] and po["edit"], po
    assert (po["box"]["left"] <= po["edit"]["l"] + 12 and
            po["box"]["right"] >= po["edit"]["r"] - 12), (po["box"], po["edit"])
    adv = rows["preAdvanced"]
    assert adv["step"] == 13, adv
    assert "Configure Training Parameters" in adv["title"], adv["title"]
    assert adv["orbOn"] is False and adv["cardOn"] is True, adv
    assert adv["worktable"] is False and adv["modal"] is False, adv
    print("PASS hard gates + smart orb: no premature Skip/Next; pill frames "
          "Class 1 then Class 2; last tune closes the modal and lands on Step 14")


def check_home_metrics() -> None:
    """The three home cards' button rows measure equal to ≤1px."""
    html = _home_fixture()
    with serve(html) as url:
        proc = subprocess.run(
            [CHROME, "--headless=new", "--disable-gpu", "--no-sandbox",
             "--window-size=1600,1000", "--virtual-time-budget=20000",
             "--dump-dom", url],
            capture_output=True, text=True, timeout=120,
        )
    m = re.search(r'<pre[^>]*id="TM_HOME_OUT"[^>]*>TM_HOME_JSON::(.*?)</pre>',
                  proc.stdout, re.S)
    assert m, "home fixture never measured the layout"
    data = json.loads(m.group(1))
    for key in ("started", "opened", "imported"):
        tops = data[key]
        assert len(tops) == 3, (key, tops)
        assert abs(tops[0] - tops[1]) <= 1, (key, tops)
        assert abs(tops[1] - tops[2]) <= 1, (key, tops)
    for t in data.get("titles", []):
        assert t["height"] <= 26, f"card title wrapped: {data['titles']}"
    descs = [d["height"] for d in data.get("descs", [])]
    assert len(descs) == 3 and max(descs) - min(descs) <= 1, descs
    print("PASS home metrics: all 3 button rows equal within 1px, titles "
          "single-line, blurbs equal height")


# The overviews list the whole trip as stages: three types, three counts.
WANT_STAGES = {"image": 6, "ocr": 6, "sign_ocr": 6}


def main() -> int:
    base = boot_payload()
    cfg, defaults = base["train_cfg"], base["train_cfg_defaults"]

    def page(**overrides):
        return build_spa_html(boot_payload(**overrides))

    # One page per project type, each with its own honest STATE: the image
    # project has LEFT/STOP classes at 96×96, the glyph projects carry the
    # A/1 character pair at 28×28 with the sign sequence filled in.
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

    spec = parse_curricula(pages["image"])
    assert sorted(spec) == sorted(EXPECTED_STEPS), sorted(spec)
    for t, steps in spec.items():
        assert len(steps) == EXPECTED_STEPS[t], t
        assert steps[0]["overview"] is True, f"{t} does not open on the overview"

    for t in EXPECTED_STEPS:
        payload = run_page(pages[t], spec, [t])
        check_overview(payload, t, WANT_STAGES[t])
        check_curriculum(payload, t)
        check_real_controls(payload, t)
        check_ask_orb(payload, t)
        check_rebind(payload, t)
        check_container_events(payload, t)
        check_escape_layer(payload, t)
        check_closed(payload, t)
    check_home_cards()
    check_start_page()
    check_multiclass_chrome_and_warnings()
    check_hard_gates_and_smart_orb()
    check_home_metrics()
    check_instant_training_completion()
    check_workspace_alignment()
    check_hero_matches_workspace()
    check_hero_description_width()
    check_ocr_end_to_end()
    check_ocr_collection_gate()
    check_preview_payload_compaction()
    check_preview_performance()
    print("ALL PASS: tests/test_tour_spa.py")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except AssertionError:
        traceback.print_exc()
        raise SystemExit(1)
