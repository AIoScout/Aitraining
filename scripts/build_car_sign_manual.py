#!/usr/bin/env python3
"""Build the annotated Student Lab Manual for the Road Sign (3-Char) project.

What this script does
---------------------
1. Launches the real Streamlit app headlessly (streamlit run app.py).
2. Drives a real sign_ocr session with Playwright: Start Page -> A-1-2
   workspace -> duplicate-character demo -> Settings / EMNIST Auto-Fill ->
   capture drawer -> Sample Worktable -> real training -> Synth Match /
   Synth Mismatch -> Export / Save view.
3. Injects numbered red callout boxes (``#e11d48``) through
   ``getBoundingClientRect()`` and captures 1440x900 @2x screenshots.
4. Assembles the student manual with python-docx, following the exact
   section structure of the reference HKUST "TF Lite Training Manual":
   Title -> A) Objective / Background -> Task 0 (hardware) -> Task 1
   (Arduino setup) -> Task 2 (the Road Sign 3-Char app walkthrough).
   The front matter plus Task 0 and Task 1 are deep-copied straight out of
   the reference .docx (photos, callout textboxes, bullets, numbering and
   hyperlinks all travel with them); only the title suffix, the lab-goal
   sentence and a photo-placeholder box in Task 0 Part A-(3) are patched.
   Task 2 is written fresh in middle/high-school language around the ten
   captured screenshots.
5. Writes the .docx to the repository root and to the OneDrive Student_Helper
   folder.

The manual deliberately stops after Export Model / Save Project (end of
Task 2).  It contains no Task 3: no firmware-flashing or Serial-Monitor
instructions beyond what Task 1 already covers.

Usage
-----
    cd AItraining
    .venv/bin/python scripts/build_car_sign_manual.py

    # Reuse already-captured screenshots and only rebuild the Word file:
    .venv/bin/python scripts/build_car_sign_manual.py --only-docx

Requires: streamlit, playwright (+ system Chrome), pillow, python-docx, and
the reference manual at REFERENCE_DOCX (hardcoded OneDrive path).
"""
from __future__ import annotations

import argparse
import copy
import itertools
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from PIL import Image, ImageDraw, ImageFont
from playwright.sync_api import sync_playwright

try:
    from docx import Document
    from docx.enum.style import WD_STYLE_TYPE
    from docx.enum.table import WD_TABLE_ALIGNMENT
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.opc.constants import RELATIONSHIP_TYPE as RT
    from docx.opc.packuri import PackURI
    from docx.opc.part import Part
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn
    from docx.shared import Inches, Pt, RGBColor
except ImportError as exc:  # pragma: no cover - actionable message for students
    raise SystemExit(
        "python-docx is required: .venv/bin/python -m pip install python-docx"
    ) from exc


APP_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = APP_DIR.parent
ASSET_DIR = APP_DIR / "docs" / "manual_assets" / "sign_ocr"
BUILD_LOG = ASSET_DIR / "build.log"
LOCAL_DOCX = REPO_ROOT / "Lab_Manual_Car_Sign.docx"
ONEDRIVE_DOCX = Path(
    "/Users/koil/Library/CloudStorage/OneDrive-HKUSTConnect/"
    "Student_Helper/TF Lite Training Manual - Car Sign.docx"
)
REFERENCE_DOCX = Path(
    "/Users/koil/Library/CloudStorage/OneDrive-HKUSTConnect/"
    "Student_Helper/TF Lite Training Manual.docx"
)

VIEWPORT = {"width": 1440, "height": 900}
DEVICE_SCALE = 2
CALL_OUT = "#e11d48"
BODY_FONT = "Arial"
HEADING_FONT = "Arial"


# ── logging / process helpers ───────────────────────────────────────────────
def log(message: str) -> None:
    stamp = time.strftime("%H:%M:%S")
    line = f"[{stamp}] {message}"
    print(line, flush=True)
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    with BUILD_LOG.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def free_port() -> int:
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


def wait_http(url: str, timeout_s: float = 90.0) -> None:
    deadline = time.time() + timeout_s
    last_error: Optional[Exception] = None
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as res:
                if res.status < 500:
                    return
        except Exception as exc:  # noqa: BLE001 - startup polling
            last_error = exc
        time.sleep(0.4)
    raise RuntimeError(f"server did not become ready at {url}: {last_error}")


class StreamlitApp:
    def __init__(self, port: Optional[int] = None) -> None:
        self.port = int(port or free_port())
        self.url = f"http://127.0.0.1:{self.port}/"
        self.proc: Optional[subprocess.Popen] = None
        self.log_file = None

    def start(self) -> None:
        ASSET_DIR.mkdir(parents=True, exist_ok=True)
        self.log_file = open(ASSET_DIR / "streamlit.log", "w", encoding="utf-8")
        env = dict(os.environ)
        env.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
        env.setdefault("STREAMLIT_BROWSER_GATHER_USAGE_STATS", "false")
        self.proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "streamlit",
                "run",
                "app.py",
                "--server.headless",
                "true",
                "--server.address",
                "127.0.0.1",
                "--server.port",
                str(self.port),
                "--browser.gatherUsageStats",
                "false",
                "--server.fileWatcherType",
                "none",
            ],
            cwd=str(APP_DIR),
            env=env,
            stdout=self.log_file,
            stderr=subprocess.STDOUT,
        )
        wait_http(f"{self.url}_stcore/health")
        log(f"Streamlit ready at {self.url}")

    def stop(self) -> None:
        if self.proc is not None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=12)
            except Exception:  # noqa: BLE001 - best-effort teardown
                self.proc.kill()
                try:
                    self.proc.wait(timeout=5)
                except Exception:
                    pass
        if self.log_file is not None:
            self.log_file.close()
        self.proc = None
        self.log_file = None


# ── browser helpers ─────────────────────────────────────────────────────────
WS_FRAME_JS = """
() => Array.from(document.querySelectorAll('iframe')).find((f) => {
  try { return !!(f.contentDocument && f.contentDocument.querySelector('#classes')); }
  catch (e) { return false; }
})
"""

MAIN_SCROLL_JS = """() => {
  const main = document.querySelector('section[data-testid="stMain"]');
  const rect = main.getBoundingClientRect();
  return { top: rect.top, scrollTop: main.scrollTop, height: main.clientHeight };
}"""

IFRAME_CONTENT_TOP_JS = """
() => {
  const main = document.querySelector('section[data-testid="stMain"]');
  const frame = Array.from(document.querySelectorAll('iframe')).find((f) => {
    try { return !!(f.contentDocument && f.contentDocument.querySelector('#classes')); }
    catch (e) { return false; }
  });
  if (!main || !frame) return null;
  return frame.getBoundingClientRect().top - main.getBoundingClientRect().top + main.scrollTop;
}"""


def wait_workspace(page) -> Any:
    page.wait_for_function(
        "() => Array.from(document.querySelectorAll('iframe')).some((f) => {"
        " try { return !!(f.contentDocument && f.contentDocument.querySelector('#classes')); }"
        " catch (e) { return false; } })",
        timeout=90_000,
    )
    frame = next(
        f
        for f in page.frames
        if f != page.main_frame and f.locator("#classes").count() > 0
    )
    frame.wait_for_selector("#signSeqBar", timeout=30_000)
    return frame


def dismiss_tour(ws) -> None:
    for _ in range(3):
        try:
            if ws.locator("#tmTourSkip").count():
                ws.evaluate("document.getElementById('tmTourSkip').click()")
                log("Tour dismissed")
                return
        except Exception:  # noqa: BLE001 - tour may already be gone
            pass
        time.sleep(0.4)


def close_training_warning(ws) -> None:
    ws.evaluate(
        """() => {
          const nodes = Array.from(document.querySelectorAll('div'))
            .filter((d) => d.textContent.includes('Training Warnings') && d.querySelector('button'))
            .sort((a, b) => a.textContent.length - b.textContent.length);
          if (nodes.length) nodes[0].querySelector('button').click();
        }"""
    )
    time.sleep(0.4)


def scroll_main_content(page, content_top: float, target: float = 60.0) -> None:
    page.evaluate(
        """([contentTop, target]) => {
          const main = document.querySelector('section[data-testid="stMain"]');
          main.scrollTop = Math.max(0, contentTop - target);
        }""",
        [content_top, target],
    )
    time.sleep(0.25)


def scroll_to_iframe_y(page, iframe_y: float, target: float = 0.0) -> None:
    content_top = page.evaluate(IFRAME_CONTENT_TOP_JS)
    if content_top is None:
        raise RuntimeError("workspace iframe not found")
    scroll_main_content(page, float(content_top) + float(iframe_y), target)


def ws_top(ws, selector: str) -> float:
    return float(
        ws.evaluate(
            "(sel) => { const e = document.querySelector(sel);"
            " if (!e) throw new Error('missing ' + sel);"
            " return e.getBoundingClientRect().top; }",
            selector,
        )
    )


def ws_union(ws, selectors: Sequence[str]) -> Dict[str, float]:
    return ws.evaluate(
        """(sels) => {
          let left = 1e9, top = 1e9, right = -1e9, bottom = -1e9;
          for (const sel of sels) {
            const e = document.querySelector(sel);
            if (!e) throw new Error('missing callout target: ' + sel);
            const r = e.getBoundingClientRect();
            left = Math.min(left, r.left); top = Math.min(top, r.top);
            right = Math.max(right, r.right); bottom = Math.max(bottom, r.bottom);
          }
          return { left, top, right, bottom };
        }""",
        list(selectors),
    )


def add_callouts(frame, specs: Sequence[Dict[str, Any]]) -> None:
    """Draw numbered red boxes around one or more DOM selectors per number."""
    frame.evaluate(
        """(specs) => {
          document.querySelectorAll('.__callout_box, .__callout_badge').forEach((e) => e.remove());
          for (const spec of specs) {
            const elements = spec.selectors.map((s) => document.querySelector(s)).filter(Boolean);
            if (!elements.length) throw new Error('callout target missing: ' + spec.selectors.join(', '));
            let left = 1e9, top = 1e9, right = -1e9, bottom = -1e9;
            for (const el of elements) {
              const r = el.getBoundingClientRect();
              left = Math.min(left, r.left); top = Math.min(top, r.top);
              right = Math.max(right, r.right); bottom = Math.max(bottom, r.bottom);
            }
            const pad = spec.pad == null ? 6 : Number(spec.pad);
            const box = document.createElement('div');
            box.className = '__callout_box';
            Object.assign(box.style, {
              position: 'fixed',
              left: (left - pad) + 'px',
              top: (top - pad) + 'px',
              width: (right - left + 2 * pad) + 'px',
              height: (bottom - top + 2 * pad) + 'px',
              border: '3px solid #e11d48',
              borderRadius: '9px',
              boxSizing: 'border-box',
              zIndex: 2147483600,
              pointerEvents: 'none',
              boxShadow: '0 0 0 1px rgba(255,255,255,0.65) inset'
            });
            const badge = document.createElement('div');
            badge.className = '__callout_badge';
            badge.textContent = String(spec.number);
            const badgeInside = !!spec.badgeInside;
            Object.assign(badge.style, {
              position: 'fixed',
              left: (badgeInside ? left - pad + 4 : Math.max(2, left - pad - 13)) + 'px',
              top: (badgeInside ? top - pad + 4 : Math.max(2, top - pad - 13)) + 'px',
              width: '26px',
              height: '26px',
              borderRadius: '50%',
              background: '#e11d48',
              color: '#ffffff',
              font: '700 14px/26px Arial, sans-serif',
              textAlign: 'center',
              zIndex: 2147483601,
              pointerEvents: 'none',
              boxShadow: '0 0 0 2px #ffffff'
            });
            document.body.appendChild(box);
            document.body.appendChild(badge);
          }
        }""",
        list(specs),
    )
    time.sleep(0.15)


def clear_callouts(frame) -> None:
    frame.evaluate(
        "() => document.querySelectorAll('.__callout_box, .__callout_badge').forEach((e) => e.remove())"
    )


def snap(page, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    page.screenshot(path=str(path), full_page=False)
    log(f"screenshot -> {path.name}")


def click_js(ws, expression: str) -> None:
    ws.evaluate(expression)
    time.sleep(0.25)


# ── real-app walkthrough ────────────────────────────────────────────────────
def capture_walkthrough(app: StreamlitApp) -> Dict[str, Path]:
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    shots: Dict[str, Path] = {}

    with sync_playwright() as pw:
        browser = pw.chromium.launch(channel="chrome", headless=True)
        page = browser.new_page(viewport=VIEWPORT, device_scale_factor=DEVICE_SCALE)
        page.goto(app.url, wait_until="domcontentloaded", timeout=90_000)
        page.wait_for_selector(".tm-start-card", timeout=120_000)
        page.wait_for_timeout(1200)

        # ── 01 Start Page ──────────────────────────────────────────────────
        page.wait_for_timeout(500)
        card = page.locator(".tm-start-card").nth(2)
        get_started = page.get_by_role("button", name="Get Started").nth(2)
        open_project = page.get_by_role("button", name="Open Project (.tmproj)").nth(2)
        card.scroll_into_view_if_needed()
        page.wait_for_timeout(400)
        # Make sure the card and both buttons are in the 1440x900 viewport.
        open_box = open_project.bounding_box()
        if open_box and open_box["y"] + open_box["height"] > 850:
            page.evaluate(
                """(dy) => {
                  const main = document.querySelector('section[data-testid="stMain"]');
                  main.scrollTop += dy;
                }""",
                open_box["y"] + open_box["height"] - 830,
            )
            page.wait_for_timeout(350)
        boxes = {
            "card": card.bounding_box(),
            "start": get_started.bounding_box(),
            "open": open_project.bounding_box(),
        }
        # The Streamlit buttons have no stable ids; draw from measured boxes.
        page.evaluate(
            """(boxes) => {
              document.querySelectorAll('.__callout_box, .__callout_badge').forEach((e) => e.remove());
              const specs = [
                { n: 1, r: boxes.card },
                { n: 2, r: boxes.start },
                { n: 3, r: boxes.open },
              ];
              for (const { n, r } of specs) {
                if (!r) continue;
                const box = document.createElement('div');
                box.className = '__callout_box';
                Object.assign(box.style, {
                  position: 'fixed', left: (r.x - 5) + 'px', top: (r.y - 5) + 'px',
                  width: (r.width + 10) + 'px', height: (r.height + 10) + 'px',
                  border: '3px solid #e11d48', borderRadius: '9px',
                  boxSizing: 'border-box', zIndex: 2147483600, pointerEvents: 'none'
                });
                const badge = document.createElement('div');
                badge.className = '__callout_badge';
                badge.textContent = String(n);
                Object.assign(badge.style, {
                  position: 'fixed', left: Math.max(2, r.x - 32) + 'px',
                  top: Math.max(2, r.y - 16) + 'px', width: '26px', height: '26px',
                  borderRadius: '50%', background: '#e11d48', color: '#fff',
                  font: '700 14px/26px Arial, sans-serif', textAlign: 'center',
                  zIndex: 2147483601, pointerEvents: 'none',
                  boxShadow: '0 0 0 2px #fff'
                });
                document.body.appendChild(box);
                document.body.appendChild(badge);
              }
            }""",
            boxes,
        )
        page.wait_for_timeout(200)
        shots["01_start_page"] = ASSET_DIR / "01_start_page.png"
        snap(page, shots["01_start_page"])
        clear_callouts(page)

        # ── Enter the sign workspace ──────────────────────────────────────
        get_started.click()
        ws = wait_workspace(page)
        page.wait_for_timeout(1500)
        dismiss_tour(ws)

        def set_sequence(seq: List[str]) -> None:
            ws.evaluate("(s) => setSignSequence(s)", seq)
            page.wait_for_timeout(1800)

        set_sequence(["A", "1", "2"])

        # ── 02 Workspace overview ─────────────────────────────────────────
        scroll_to_iframe_y(page, 0, target=0)
        page.wait_for_timeout(300)
        add_callouts(
            ws,
            [
                {"number": 1, "selectors": ["#menuFileBtn", "#navSettings", "#navTutorial", "#navReturn"], "pad": 6},
                {"number": 2, "selectors": ["#signSeqBar"], "pad": 5},
                {"number": 3, "selectors": [".class-card"], "pad": 5},
                {"number": 4, "selectors": ["#trainCard"], "pad": 5},
                {"number": 5, "selectors": ["#previewCard"], "pad": 5},
            ],
        )
        shots["02_workspace_overview"] = ASSET_DIR / "02_workspace_overview.png"
        snap(page, shots["02_workspace_overview"])
        clear_callouts(ws)

        # ── 03 Target sequence + duplicate dedup ──────────────────────────
        set_sequence(["A", "1", "A"])
        scroll_to_iframe_y(page, 0, target=0)
        page.wait_for_timeout(300)
        add_callouts(
            ws,
            [
                {"number": 1, "selectors": ["#signSeqSlot-0", "#signSeqSlot-1", "#signSeqSlot-2"], "pad": 7},
                {"number": 2, "selectors": ["#signSeqLeft-1", "#signSeqRight-1"], "pad": 6},
                {"number": 3, "selectors": ["#signSeqBar .signseq-head .signseq-note"], "pad": 6},
                {"number": 4, "selectors": [".signslot-badge"], "pad": 6},
            ],
        )
        shots["03_target_sequence_and_dedup"] = ASSET_DIR / "03_target_sequence_and_dedup.png"
        snap(page, shots["03_target_sequence_and_dedup"])
        clear_callouts(ws)
        set_sequence(["A", "1", "2"])

        # ── 04a Settings: source + EMNIST auto-fill ───────────────────────
        click_js(ws, "openGlobalSettings()")
        ws.wait_for_timeout(500)
        click_js(ws, "document.getElementById('settingsAdvancedToggle').click()")
        ws.evaluate(
            """() => {
              const s = document.getElementById('settingsPreviewSource');
              s.value = 'upload';
              s.dispatchEvent(new Event('change', {bubbles: true}));
            }"""
        )
        click_js(
            ws,
            """() => {
              const c = document.getElementById('settingsAutoSeedEmnist');
              if (!c.checked) { c.checked = true; c.dispatchEvent(new Event('change', {bubbles: true})); }
            }""",
        )
        ws.wait_for_function(
            "STATE.counts.A >= 24 && STATE.counts['1'] >= 24 && STATE.counts['2'] >= 24",
            timeout=180_000,
        )
        page.wait_for_timeout(3200)  # let the auto-fill toast fade
        log(f"Auto-fill complete: {ws.evaluate('JSON.stringify(STATE.counts)')}")

        # Top view: Preview source selector.
        ws.evaluate("document.querySelector('#globalSettingsModal .tm-settings-card').scrollTop = 0")
        scroll_to_iframe_y(page, 0, target=0)
        page.wait_for_timeout(300)
        add_callouts(
            ws,
            [
                {"number": 1, "selectors": ["#settingsPreviewSource"], "pad": 6},
            ],
        )
        settings_top = ASSET_DIR / "_settings_top.png"
        snap(page, settings_top)
        clear_callouts(ws)

        # Bottom view: Advanced -> EMNIST samples + Save.  The modal card is
        # taller than the visible iframe slice, so scroll BOTH the card to its
        # maximum and the Streamlit page to the bottom; the Advanced group and
        # Save then sit in the visible slice.
        ws.evaluate(
            """() => {
              const card = document.querySelector('#globalSettingsModal .tm-settings-card');
              card.scrollTop = card.scrollHeight;
            }"""
        )
        page.evaluate(
            """() => {
              const main = document.querySelector('section[data-testid="stMain"]');
              main.scrollTop = main.scrollHeight;
            }"""
        )
        page.wait_for_timeout(400)
        add_callouts(
            ws,
            [
                {"number": 2, "selectors": [".tm-settings-check"], "pad": 4},
                {"number": 3, "selectors": ["#settingsSeedEmnistNowBtn"], "pad": 4},
                {"number": 4, "selectors": ["#settingsSave"], "pad": 2},
            ],
        )
        settings_bottom = ASSET_DIR / "_settings_bottom.png"
        snap(page, settings_bottom)
        clear_callouts(ws)

        click_js(ws, "document.getElementById('settingsSave').click()")
        page.wait_for_timeout(1200)
        if ws.evaluate("document.getElementById('globalSettingsModal').classList.contains('open')"):
            click_js(ws, "document.getElementById('settingsCancel').click()")
            page.wait_for_timeout(600)

        # ── 04b Capture drawer ────────────────────────────────────────────
        click_js(
            ws,
            """() => {
              const card = document.querySelector('.class-card');
              const tiles = card.querySelectorAll('.tm-action-toolbar .sample');
              tiles[2].click();  // Device
            }""",
        )
        page.wait_for_timeout(2200)
        card_top = ws_top(ws, ".class-card")
        scroll_to_iframe_y(page, card_top, target=35)
        page.wait_for_timeout(350)
        add_callouts(
            ws,
            [
                {"number": 1, "selectors": [".class-card .tm-action-toolbar"], "pad": 5},
                {"number": 2, "selectors": ["#sourceCapture-A"], "pad": 5},
                {"number": 3, "selectors": ["#sourceHold-A"], "pad": 5},
                {"number": 4, "selectors": ["#sourceClose-A"], "pad": 5},
                {"number": 5, "selectors": ["#samplesHost-A"], "pad": 3, "badgeInside": True},
            ],
        )
        shots["04b_capture_drawer"] = ASSET_DIR / "04b_capture_drawer.png"
        snap(page, shots["04b_capture_drawer"])
        clear_callouts(ws)
        click_js(ws, "document.getElementById('sourceClose-A').click()")
        page.wait_for_timeout(500)

        # ── 05 Sample Worktable ───────────────────────────────────────────
        click_js(ws, "document.querySelector('.class-card .class-preprocess-chip').click()")
        ws.wait_for_selector("#classPreprocessImage", timeout=20_000)
        ws.wait_for_selector(".class-preprocess-result img", timeout=30_000)
        page.wait_for_timeout(1200)
        scroll_to_iframe_y(page, 0, target=0)
        page.wait_for_timeout(300)
        add_callouts(
            ws,
            [
                {"number": 1, "selectors": ["#classPreprocessStage"], "pad": 5},
                {"number": 2, "selectors": [".class-preprocess-result"], "pad": 5},
                {"number": 3, "selectors": ["#classPreprocessMode"], "pad": 5},
                {"number": 4, "selectors": ["#classPreprocessDark", "#classPreprocessLum"], "pad": 6},
                {"number": 5, "selectors": ["#classPreprocessSave", "#classPreprocessCancel"], "pad": 5},
            ],
        )
        shots["05_sample_worktable_edit"] = ASSET_DIR / "05_sample_worktable_edit.png"
        snap(page, shots["05_sample_worktable_edit"])
        clear_callouts(ws)
        click_js(ws, "document.getElementById('classPreprocessCancel').click()")
        page.wait_for_timeout(600)

        # ── 06 Train for real ─────────────────────────────────────────────
        click_js(ws, "document.getElementById('advBtn').click()")
        page.wait_for_timeout(500)
        click_js(ws, "document.getElementById('trainBtn').click()")
        log("Training started…")
        ws.wait_for_function(
            "() => { const e = document.getElementById('trainStatus');"
            " return e && e.textContent.includes('Model Trained'); }",
            timeout=1_200_000,
        )
        log(f"Training complete: {ws.locator('#trainStatus').inner_text()}")
        page.wait_for_timeout(1200)
        close_training_warning(ws)
        if ws.evaluate("document.getElementById('trainAdvPanel').style.display === 'none'"):
            click_js(ws, "document.getElementById('advBtn').click()")
            page.wait_for_timeout(400)

        train_top = ws_top(ws, "#trainCard")
        scroll_to_iframe_y(page, train_top, target=20)
        page.wait_for_timeout(350)
        add_callouts(
            ws,
            [
                {"number": 1, "selectors": ["#advInkPolarity"], "pad": 5},
                {"number": 2, "selectors": ["#advReadingOrder"], "pad": 5},
                {"number": 3, "selectors": ["#advUseRec"], "pad": 5},
                {"number": 5, "selectors": ["#trainBtn"], "pad": 5},
            ],
        )
        training_card = ASSET_DIR / "_training_card.png"
        snap(page, training_card)
        clear_callouts(ws)

        # OCR/sign hyperparameter panel lives in Settings → Training
        # Hyperparameters.  Capture it separately for the composite.
        click_js(ws, "openGlobalSettings()")
        page.wait_for_timeout(500)
        ws.evaluate(
            """() => {
              const card = document.querySelector('#globalSettingsModal .tm-settings-card');
              const target = document.getElementById('modalOcrSection');
              const cr = card.getBoundingClientRect();
              const tr = target.getBoundingClientRect();
              card.scrollTop += (tr.top - cr.top - 60);
            }"""
        )
        scroll_to_iframe_y(page, 0, target=0)
        page.wait_for_timeout(300)
        add_callouts(
            ws,
            [
                {"number": 4, "selectors": ["#advOcrPanel"], "pad": 6},
            ],
        )
        ocr_panel = ASSET_DIR / "_ocr_panel.png"
        snap(page, ocr_panel)
        clear_callouts(ws)
        click_js(ws, "document.getElementById('settingsCancel').click()")
        page.wait_for_timeout(600)

        # ── 07a / 07b Synth Match + Mismatch ──────────────────────────────
        if ws.evaluate("previewSource") != "upload":
            log("Preview source was not upload; switching through JS")
            ws.evaluate(
                """() => {
                  STATE.preview_source = 'upload';
                  previewSource = 'upload';
                  renderPreviewCard();
                }"""
            )
            page.wait_for_timeout(500)
        click_js(ws, "document.getElementById('previewSynthMatchBtn').click()")
        ws.wait_for_function(
            "() => document.querySelector('#previewOutput .signmatch')"
            " && document.querySelector('#previewOutput .signmatch').classList.contains('yes')",
            timeout=240_000,
        )
        page.wait_for_timeout(1200)
        preview_top = ws_top(ws, "#previewCard")
        scroll_to_iframe_y(page, preview_top, target=20)
        page.wait_for_timeout(350)
        preview_specs = [
            {"number": 1, "selectors": [".preview-synth-group"], "pad": 5},
            {"number": 2, "selectors": [".preview-controls"], "pad": 6},
            {"number": 3, "selectors": ["#previewImage"], "pad": 5},
            {"number": 4, "selectors": [".signmatch-main"], "pad": 5},
            {"number": 5, "selectors": [".signmatch-slots"], "pad": 5},
        ]
        add_callouts(ws, preview_specs)
        shots["07a_preview_synth_match"] = ASSET_DIR / "07a_preview_synth_match.png"
        snap(page, shots["07a_preview_synth_match"])
        clear_callouts(ws)

        click_js(ws, "document.getElementById('previewSynthMismatchBtn').click()")
        ws.wait_for_function(
            "() => document.querySelector('#previewOutput .signmatch')"
            " && document.querySelector('#previewOutput .signmatch').classList.contains('no')"
            " && document.querySelector('#previewOutput .signmatch-hint')",
            timeout=240_000,
        )
        page.wait_for_timeout(1200)
        add_callouts(ws, preview_specs)
        shots["07b_preview_synth_mismatch"] = ASSET_DIR / "07b_preview_synth_mismatch.png"
        snap(page, shots["07b_preview_synth_mismatch"])
        clear_callouts(ws)

        # ── 08 Export + Save ──────────────────────────────────────────────
        click_js(ws, "document.getElementById('menuFileBtn').click()")
        scroll_to_iframe_y(page, 0, target=0)
        page.wait_for_timeout(300)
        add_callouts(
            ws,
            [
                {"number": 1, "selectors": ["#exportBtn"], "pad": 5},
                {"number": 2, "selectors": ["#menuFileBtn", "#navSaveProject", "#navSaveAsProject"], "pad": 5},
            ],
        )
        shots["08_export_and_save"] = ASSET_DIR / "08_export_and_save.png"
        snap(page, shots["08_export_and_save"])
        clear_callouts(ws)

        browser.close()

    # Composite 04a (Preview source + Advanced EMNIST controls).
    shots["04a_settings_emnist_autofill"] = ASSET_DIR / "04a_settings_emnist_autofill.png"
    _pair_panels(
        [
            (settings_top, (300, 470, 760, 900),
             "Settings → Preview: choose Local Image File"),
            (settings_bottom, (300, 520, 1140, 900),
             "Settings → Advanced: Auto-fill 24 EMNIST samples"),
        ],
        shots["04a_settings_emnist_autofill"],
    )

    # Composite 06 (Training card + Settings → Training Hyperparameters).
    shots["06_training_and_advanced"] = ASSET_DIR / "06_training_and_advanced.png"
    _pair_panels(
        [
            (training_card, (640, 0, 980, 900), "Training card → Advanced"),
            (ocr_panel, (680, 0, 1140, 900), "Settings → Training Hyperparameters"),
        ],
        shots["06_training_and_advanced"],
    )

    # Remove intermediate captures; keep only the ten deliverable PNGs.
    for tmp in (settings_top, settings_bottom, training_card, ocr_panel):
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass
    return shots


# ── Pillow compositing ──────────────────────────────────────────────────────
def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            continue
    return ImageFont.load_default()


def _crop_css(image: Image.Image, css_box: Sequence[float]) -> Image.Image:
    scale = DEVICE_SCALE
    x1, y1, x2, y2 = [int(round(float(v) * scale)) for v in css_box]
    x1 = max(0, min(image.width - 1, x1))
    y1 = max(0, min(image.height - 1, y1))
    x2 = max(x1 + 1, min(image.width, x2))
    y2 = max(y1 + 1, min(image.height, y2))
    return image.crop((x1, y1, x2, y2))


def _pair_panels(
    panels: Sequence[tuple[Path, Sequence[float], str]],
    out_path: Path,
) -> None:
    """Two cropped panels side by side, each with a title strip.

    Used where one screenshot cannot show two distant parts of the UI
    (e.g. Settings → Preview vs Settings → Advanced Settings, or the
    Training card vs the Training Hyperparameters modal section).
    """
    canvas_w = VIEWPORT["width"] * DEVICE_SCALE
    canvas_h = VIEWPORT["height"] * DEVICE_SCALE
    canvas = Image.new("RGB", (canvas_w, canvas_h), "white")
    draw = ImageDraw.Draw(canvas)
    margin = 40
    gap = 50
    label_h = 62
    slot_w = (canvas_w - 2 * margin - gap) // 2
    slot_h = canvas_h - label_h - margin
    x_positions = [margin, margin + slot_w + gap]
    for idx, (path, crop_box, label) in enumerate(panels[:2]):
        crop = _crop_css(Image.open(path).convert("RGB"), crop_box)
        scale = min(slot_w / crop.width, slot_h / crop.height)
        crop = crop.resize((max(1, int(crop.width * scale)), max(1, int(crop.height * scale))),
                           Image.LANCZOS)
        x = x_positions[idx]
        draw.text((x, 14), label, font=_font(26, bold=True), fill=(15, 23, 42))
        canvas.paste(crop, (x, label_h))
    draw.line(
        [(margin + slot_w + gap // 2, 8), (margin + slot_w + gap // 2, canvas_h - 20)],
        fill=(203, 213, 225),
        width=2,
    )
    canvas.save(out_path)
    log(f"composite -> {out_path.name} ({canvas.width}x{canvas.height})")


# ── Word manual ─────────────────────────────────────────────────────────────
def _set_cell_shading(cell, fill: str) -> None:
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), fill)
    tc_pr.append(shd)


def _set_cell_borders(cell, color: str = "E11D48", size: int = 18) -> None:
    tc_pr = cell._tc.get_or_add_tcPr()
    borders = OxmlElement("w:tcBorders")
    for edge in ("top", "left", "bottom", "right"):
        el = OxmlElement(f"w:{edge}")
        el.set(qn("w:val"), "single")
        el.set(qn("w:sz"), str(size))
        el.set(qn("w:space"), "0")
        el.set(qn("w:color"), color)
        borders.append(el)
    tc_pr.append(borders)


def _set_repeat_header(row) -> None:
    tr_pr = row._tr.get_or_add_trPr()
    tbl_header = OxmlElement("w:tblHeader")
    tbl_header.set(qn("w:val"), "true")
    tr_pr.append(tbl_header)


def _set_cell_text(cell, text: str, *, bold: bool = False, size: float = 9.0,
                   color: Optional[str] = None, align: Optional[Any] = None) -> None:
    cell.text = ""
    p = cell.paragraphs[0]
    if align is not None:
        p.alignment = align
    run = p.add_run(text)
    run.bold = bold
    run.font.size = Pt(size)
    run.font.name = BODY_FONT
    if color:
        run.font.color.rgb = RGBColor.from_string(color)


def _add_table(doc, headers: Sequence[str], rows: Sequence[Sequence[str]],
               widths: Optional[Sequence[float]] = None, font_size: float = 9.0):
    table = doc.add_table(rows=1, cols=len(headers))
    table.style = "Table Grid"
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    header = table.rows[0]
    _set_repeat_header(header)
    for i, text in enumerate(headers):
        cell = header.cells[i]
        _set_cell_shading(cell, "E2E8F0")
        _set_cell_text(cell, text, bold=True, size=font_size, color="0F172A")
    for row_values in rows:
        row = table.add_row()
        # Never split a row across pages (the long "Use recommended" row
        # otherwise left an empty half-row at the top of the next page).
        tr_pr = row._tr.get_or_add_trPr()
        cant_split = OxmlElement("w:cantSplit")
        cant_split.set(qn("w:val"), "true")
        tr_pr.append(cant_split)
        for i, text in enumerate(row_values):
            _set_cell_text(row.cells[i], text, size=font_size)
    if widths:
        for row in table.rows:
            for i, width in enumerate(widths):
                row.cells[i].width = Inches(width)
    return table


def _callout_box(doc, title: str, body: Sequence[str], icon: str = "⚠") -> None:
    table = doc.add_table(rows=1, cols=1)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    cell = table.cell(0, 0)
    _set_cell_shading(cell, "FEF2F2")
    _set_cell_borders(cell)
    cell.text = ""
    p = cell.paragraphs[0]
    run = p.add_run(f"{icon} {title}")
    run.bold = True
    run.font.name = BODY_FONT
    run.font.size = Pt(11)
    run.font.color.rgb = RGBColor(0xB9, 0x1C, 0x1C)
    for text in body:
        p = cell.add_paragraph()
        p.paragraph_format.space_after = Pt(2)
        r = p.add_run(text)
        r.font.name = BODY_FONT
        r.font.size = Pt(10)
        r.font.color.rgb = RGBColor(0x7F, 0x1D, 0x1D)
    doc.add_paragraph().paragraph_format.space_after = Pt(2)


def _figure(doc, path: Path, caption: str, width_in: float = 6.5) -> None:
    # Keep the lead-in paragraph, the image and its caption on one page where
    # possible; an orphaned step heading above a page break reads poorly.
    if doc.paragraphs:
        doc.paragraphs[-1].paragraph_format.keep_with_next = True
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(6)
    p.paragraph_format.space_after = Pt(2)
    p.paragraph_format.keep_with_next = True
    p.add_run().add_picture(str(path), width=Inches(width_in))
    cap = doc.add_paragraph()
    cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    cap.paragraph_format.space_after = Pt(10)
    cap.paragraph_format.keep_together = True
    run = cap.add_run(caption)
    run.italic = True
    run.font.size = Pt(8.5)
    run.font.color.rgb = RGBColor(0x64, 0x74, 0x8B)
    run.font.name = BODY_FONT


def _bullet(doc, text: str, bold_prefix: Optional[str] = None) -> None:
    p = doc.add_paragraph(style="List Bullet")
    p.paragraph_format.space_after = Pt(2)
    if bold_prefix and text.startswith(bold_prefix):
        r = p.add_run(bold_prefix)
        r.bold = True
        r.font.name = BODY_FONT
        r.font.size = Pt(10.5)
        rest = text[len(bold_prefix):]
        r2 = p.add_run(rest)
        r2.font.name = BODY_FONT
        r2.font.size = Pt(10.5)
    else:
        r = p.add_run(text)
        r.font.name = BODY_FONT
        r.font.size = Pt(10.5)


def _numbered(doc, text: str) -> None:
    """Manually numbered list item.

    Word's built-in List Number style shares one numbering definition, so
    every list in the document continued from the previous list (Step 4A
    began at "4", Step 7 at "8").  Writing the number as text and resetting
    whenever the previous paragraph is not a numbered item keeps each list
    independent and predictable.
    """
    previous = doc.paragraphs[-1] if doc.paragraphs else None
    number = 1
    if previous is not None and previous.style is not None and previous.style.name == "Manual Number":
        match = re.match(r"^(\d+)\.\s", previous.text)
        if match:
            number = int(match.group(1)) + 1
    p = doc.add_paragraph(style="Manual Number")
    p.paragraph_format.space_after = Pt(2)
    # Keep a short numbered list together; the Step 7 five-item rule list
    # otherwise ended with a lone item on the next page.
    p.paragraph_format.keep_with_next = True
    p.add_run(f"{number}. ")
    r = p.add_run(text)
    r.font.name = BODY_FONT
    r.font.size = Pt(10.5)


def _para(doc, text: str, *, bold: bool = False, size: float = 10.5,
          color: Optional[str] = None, space_after: float = 6.0) -> None:
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(space_after)
    r = p.add_run(text)
    r.bold = bold
    r.font.name = BODY_FONT
    r.font.size = Pt(size)
    if color:
        r.font.color.rgb = RGBColor.from_string(color)


def _heading(doc, text: str, level: int) -> None:
    p = doc.add_heading(level=level)
    p.paragraph_format.keep_with_next = True
    run = p.add_run(text)
    run.font.name = HEADING_FONT
    if level == 2:
        run.font.size = Pt(16)
        run.font.color.rgb = RGBColor(0x0F, 0x17, 0x2A)
        p.paragraph_format.space_before = Pt(14)
        p.paragraph_format.space_after = Pt(6)
    elif level == 3:
        run.font.size = Pt(12)
        run.font.color.rgb = RGBColor(0x0F, 0x17, 0x2A)
        p.paragraph_format.space_before = Pt(10)
        p.paragraph_format.space_after = Pt(3)
    else:  # level 4 — numbered sub-steps inside our Task 2
        run.font.size = Pt(11.5)
        run.font.color.rgb = RGBColor(0x1D, 0x4E, 0xD8)
        p.paragraph_format.space_before = Pt(10)
        p.paragraph_format.space_after = Pt(3)


def _task_heading(doc, text: str):
    """A Task-level heading matching the reference manual's Task 0 / Task 1
    headings: the imported Heading 3 style, no explicit run formatting."""
    p = doc.add_heading(level=3)
    p.paragraph_format.keep_with_next = True
    p.add_run(text)
    return p


def _page_break(doc) -> None:
    doc.add_page_break()


def _add_page_number(paragraph) -> None:
    run = paragraph.add_run()
    fld_char1 = OxmlElement("w:fldChar")
    fld_char1.set(qn("w:fldCharType"), "begin")
    instr = OxmlElement("w:instrText")
    instr.set(qn("xml:space"), "preserve")
    instr.text = "PAGE"
    fld_char2 = OxmlElement("w:fldChar")
    fld_char2.set(qn("w:fldCharType"), "end")
    run._r.append(fld_char1)
    run._r.append(instr)
    run._r.append(fld_char2)
    run.font.name = BODY_FONT
    run.font.size = Pt(8.5)
    run.font.color.rgb = RGBColor(0x64, 0x74, 0x8B)


def _photo_placeholder_table(doc) -> Any:
    """Amber photo-placeholder box for Task 0 Part A-(3).

    The instructor replaces this box with a photo of the physical
    3-character road sign / car plate before printing the manual.
    """
    table = doc.add_table(rows=1, cols=1)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    cell = table.cell(0, 0)
    _set_cell_shading(cell, "FFFBEB")
    _set_cell_borders(cell, "D97706", 18)
    p = cell.paragraphs[0]
    p.clear()
    r = p.add_run("📷 [Photo Placeholder: Physical 3-Character Road Sign Setup]")
    r.bold = True
    r.font.name = BODY_FONT
    r.font.size = Pt(11)
    r.font.color.rgb = RGBColor(0xB4, 0x53, 0x09)
    p2 = cell.add_paragraph()
    r2 = p2.add_run(
        "Instructor: replace this box with a photo of the physical 3-character "
        "road sign / car plate (for example A 1 2) mounted on the signpost."
    )
    r2.font.name = BODY_FONT
    r2.font.size = Pt(10)
    r2.font.color.rgb = RGBColor(0x78, 0x35, 0x0F)
    return table


class _ReferenceImporter:
    """Deep-copy the reference manual's front matter + Task 0 + Task 1.

    The copied paragraphs keep their original look (photos, callout
    textboxes, bullets, hyperlinks): the styles and numbering definitions
    they rely on are imported from the reference, image parts are copied
    into the new package under fresh names, and every relationship /
    numbering / drawing id is remapped.  After copying, three patches are
    applied: the title suffix, the lab-goal sentence, and a
    photo-placeholder box in Task 0 Part A-(3).
    """

    VML_NS = "urn:schemas-microsoft-com:vml"

    def __init__(self, doc: Document, reference_path: Path) -> None:
        if not reference_path.exists():
            raise RuntimeError(
                f"reference manual not found: {reference_path}\n"
                "Task 0 / Task 1 are copied from this file; restore it or "
                "point REFERENCE_DOCX at the right location."
            )
        self.doc = doc
        self.ref = Document(str(reference_path))
        self._img_counter = 0
        self._docpr_ids = itertools.count(4000)
        self._image_rIds: Dict[str, str] = {}
        self._hyperlink_rIds: Dict[str, str] = {}
        self._numid_map: Dict[str, str] = {}
        self._copied: List[Any] = []

    # ── inventory ────────────────────────────────────────────────────────
    @staticmethod
    def _para_text(el) -> str:
        return "".join(t.text or "" for t in el.iter(qn("w:t")))

    def _children(self):
        return list(self.ref.element.body.iterchildren())

    def _block(self, start_prefix: str, end_prefix: str):
        """Body elements from the paragraph starting with start_prefix up to
        (not including) the paragraph starting with end_prefix."""
        children = self._children()
        start = end = None
        for i, ch in enumerate(children):
            if ch.tag != qn("w:p"):
                continue
            text = self._para_text(ch)
            if start is None and text.startswith(start_prefix):
                start = i
            if end is None and text.startswith(end_prefix):
                end = i
        if start is None or end is None:
            raise RuntimeError(
                f"reference section {start_prefix!r} .. {end_prefix!r} not found"
            )
        return children[start:end]

    def _style_ids_needed(self, block) -> set:
        """Styles referenced by the copied block, plus the transitive
        closure over basedOn / link / next, plus Normal."""
        needed = {"Normal"}
        for el in block:
            for p in el.iter(qn("w:p")):
                pPr = p.find(qn("w:pPr"))
                if pPr is not None:
                    pStyle = pPr.find(qn("w:pStyle"))
                    if pStyle is not None:
                        needed.add(pStyle.get(qn("w:val")))
            for r in el.iter(qn("w:r")):
                rPr = r.find(qn("w:rPr"))
                if rPr is not None:
                    rStyle = rPr.find(qn("w:rStyle"))
                    if rStyle is not None:
                        needed.add(rStyle.get(qn("w:val")))
        ref_styles = self._ref_styles()
        queue = list(needed)
        while queue:
            sid = queue.pop()
            st = ref_styles.get(sid)
            if st is None:
                continue
            for child_name in ("w:basedOn", "w:link", "w:next"):
                ref_el = st.find(qn(child_name))
                if ref_el is not None:
                    val = ref_el.get(qn("w:val"))
                    if val and val not in needed and val in ref_styles:
                        needed.add(val)
                        queue.append(val)
        return needed

    def _ref_styles(self) -> Dict[str, Any]:
        return {
            st.get(qn("w:styleId")): st
            for st in self.ref.styles.element
            if st.tag == qn("w:style")
        }

    # ── numbering ────────────────────────────────────────────────────────
    @staticmethod
    def _find_part(doc: Document, reltype: str):
        for rel in doc.part.rels.values():
            if rel.reltype == reltype:
                return rel.target_part
        return None

    def _import_numbering(self, block, style_ids) -> set:
        """Copy the numbering definitions the copied block relies on.

        Returns any extra style ids referenced by the imported numbering
        levels (w:lvl → w:pStyle), which then join the style import.
        """
        ref_numbering = self._find_part(self.ref, RT.NUMBERING)
        if ref_numbering is None:
            return set()
        numids = set()
        for el in block:
            for numPr in el.iter(qn("w:numPr")):
                numId = numPr.find(qn("w:numId"))
                if numId is not None:
                    numids.add(numId.get(qn("w:val")))
        ref_styles = self._ref_styles()
        for sid in style_ids:
            st = ref_styles.get(sid)
            if st is None:
                continue
            for numPr in st.iter(qn("w:numPr")):
                numId = numPr.find(qn("w:numId"))
                if numId is not None:
                    numids.add(numId.get(qn("w:val")))
        if not numids:
            return set()
        ref_root = ref_numbering.element
        ref_nums = {
            n.get(qn("w:numId")): n for n in ref_root.findall(qn("w:num"))
        }
        ref_abstracts = {
            a.get(qn("w:abstractNumId")): a
            for a in ref_root.findall(qn("w:abstractNum"))
        }
        dst_numbering = self._find_part(self.doc, RT.NUMBERING)
        if dst_numbering is None:
            raise RuntimeError("new document has no numbering part")
        dst_root = dst_numbering.element
        extra_styles = set()
        new_abstract_ids: Dict[str, str] = {}
        for nid in sorted(numids, key=int):
            num_el = ref_nums.get(nid)
            if num_el is None:
                raise RuntimeError(f"reference numbering numId {nid} missing")
            if num_el.find(qn("w:numStyleLink")) is not None:
                raise RuntimeError("numStyleLink numbering not supported")
            abst = num_el.find(qn("w:abstractNumId"))
            if abst is None:
                raise RuntimeError(f"reference numId {nid} has no abstractNumId")
            orig_abst = abst.get(qn("w:val"))
            if orig_abst not in ref_abstracts:
                raise RuntimeError(f"reference abstractNum {orig_abst} missing")
            if orig_abst not in new_abstract_ids:
                new_abst_id = str(1000 + len(new_abstract_ids))
                new_abst = copy.deepcopy(ref_abstracts[orig_abst])
                new_abst.set(qn("w:abstractNumId"), new_abst_id)
                for lvl in new_abst.findall(qn("w:lvl")):
                    pStyle = lvl.find(qn("w:pStyle"))
                    if pStyle is not None:
                        extra_styles.add(pStyle.get(qn("w:val")))
                dst_root.append(new_abst)
                new_abstract_ids[orig_abst] = new_abst_id
            if nid not in self._numid_map:
                new_nid = str(1000 + len(self._numid_map))
                new_num = copy.deepcopy(num_el)
                new_num.set(qn("w:numId"), new_nid)
                new_num.find(qn("w:abstractNumId")).set(
                    qn("w:val"), new_abstract_ids[orig_abst]
                )
                dst_root.append(new_num)
                self._numid_map[nid] = new_nid
        return extra_styles

    # ── styles ───────────────────────────────────────────────────────────
    def _import_styles(self, style_ids) -> None:
        ref_styles = self._ref_styles()
        dst = self.doc.styles.element
        for sid in sorted(style_ids):
            src = ref_styles.get(sid)
            if src is None:
                continue
            existing = [
                st
                for st in dst
                if st.tag == qn("w:style") and st.get(qn("w:styleId")) == sid
            ]
            if existing:
                dst.replace(existing[0], copy.deepcopy(src))
            else:
                dst.append(copy.deepcopy(src))
        log(f"imported {len(style_ids)} styles from the reference manual")

    # ── relationship remapping ───────────────────────────────────────────
    def _image_rId(self, orig: str) -> str:
        if orig in self._image_rIds:
            return self._image_rIds[orig]
        part = self.ref.part.related_parts.get(orig)
        if part is None:
            raise RuntimeError(f"reference image relationship {orig} not found")
        # PackURI.ext drops the dot in python-docx 1.2.0, so re-add it here;
        # a dotless partname would leave the image without a content-type
        # Default mapping and trip Word's repair prompt.
        name = PackURI(f"/word/media/ref{self._img_counter:03d}.{part.partname.ext}")
        self._img_counter += 1
        new_part = Part(name, part.content_type, part.blob, self.doc.part.package)
        new_rId = self.doc.part.relate_to(new_part, RT.IMAGE)
        self._image_rIds[orig] = new_rId
        return new_rId

    def _hyperlink_rId(self, orig: str) -> str:
        if orig in self._hyperlink_rIds:
            return self._hyperlink_rIds[orig]
        rel = self.ref.part.rels[orig]
        new_rId = self.doc.part.rels.get_or_add_ext_rel(
            RT.HYPERLINK, rel.target_ref
        )
        self._hyperlink_rIds[orig] = new_rId
        return new_rId

    # ── copying ──────────────────────────────────────────────────────────
    def _remap(self, el) -> None:
        for blip in el.iter(qn("a:blip")):
            rId = blip.get(qn("r:embed"))
            if rId:
                blip.set(qn("r:embed"), self._image_rId(rId))
        for imagedata in el.iter("{%s}imagedata" % self.VML_NS):
            rId = imagedata.get(qn("r:id"))
            if rId:
                imagedata.set(qn("r:id"), self._image_rId(rId))
        for hyperlink in el.iter(qn("w:hyperlink")):
            rId = hyperlink.get(qn("r:id"))
            if rId:
                hyperlink.set(qn("r:id"), self._hyperlink_rId(rId))
        for numPr in el.iter(qn("w:numPr")):
            numId = numPr.find(qn("w:numId"))
            if numId is not None and numId.get(qn("w:val")) in self._numid_map:
                numId.set(qn("w:val"), self._numid_map[numId.get(qn("w:val"))])
        # Drawing ids must stay unique document-wide; the new document's own
        # figures use ids 1..N, so shift the copied ones to 4000+.
        for docPr in el.iter(qn("wp:docPr")):
            docPr.set("id", str(next(self._docpr_ids)))
        for cNvPr in el.iter(qn("pic:cNvPr")):
            cNvPr.set("id", str(next(self._docpr_ids)))

    def _copy_block(self, block) -> None:
        body = self.doc.element.body
        for el in block:
            new_el = copy.deepcopy(el)
            self._remap(new_el)
            body.insert(-1, new_el)  # keep the trailing sectPr last
            if new_el.tag == qn("w:p"):
                self._copied.append(new_el)
        log(f"copied {len(self._copied)} paragraphs from the reference manual")

    # ── patches ──────────────────────────────────────────────────────────
    @staticmethod
    def _set_para_text(el, text: str) -> None:
        """Replace a paragraph's runs with a single run holding `text`,
        keeping the first run's formatting."""
        runs = el.findall(qn("w:r"))
        if not runs:
            r = OxmlElement("w:r")
            pPr = el.find(qn("w:pPr"))
            if pPr is not None:
                pPr.addnext(r)
            else:
                el.insert(0, r)
            runs = [r]
        first = runs[0]
        for t in first.findall(qn("w:t")):
            first.remove(t)
        t = OxmlElement("w:t")
        t.set(qn("xml:space"), "preserve")
        t.text = text
        rPr = first.find(qn("w:rPr"))
        if rPr is not None:
            rPr.addnext(t)
        else:
            first.insert(0, t)
        for extra in runs[1:]:
            el.remove(extra)

    def _patch_title(self) -> None:
        for el in self._copied:
            if self._para_text(el).startswith("Lab0:"):
                self._set_para_text(
                    el,
                    "Lab0: Introduction to TensorFlow Lite Micro and "
                    "TFLiteTraining App (Road Sign / Car Sign Project)",
                )
                return
        raise RuntimeError("reference title paragraph not found")

    def _patch_goal(self) -> None:
        for el in self._copied:
            if self._para_text(el).startswith("In this lab, we will be familiar"):
                self._set_para_text(
                    el,
                    "In this lab, you will become familiar with TFLiteTraining "
                    "and train a model to recognize a 3-character road sign / "
                    "car plate (such as A 1 2).",
                )
                return
        raise RuntimeError("reference lab-goal paragraph not found")

    def _insert_photo_placeholder(self) -> None:
        for el in self._copied:
            if self._para_text(el).startswith("(3) Signpost with three Road Signs"):
                el.addnext(_photo_placeholder_table(self.doc)._tbl)
                return
        raise RuntimeError("Task 0 Part A-(3) heading not found")

    # ── entry point ──────────────────────────────────────────────────────
    def run(self) -> None:
        block = self._block("Lab0:", "Task 2-")
        style_ids = self._style_ids_needed(block)
        style_ids |= self._import_numbering(block, style_ids)
        self._import_styles(style_ids)
        self._copy_block(block)
        self._patch_title()
        self._patch_goal()
        self._insert_photo_placeholder()


def build_docx(assets: Dict[str, Path]) -> Path:
    doc = Document()
    section = doc.sections[0]
    section.page_width = Inches(8.5)
    section.page_height = Inches(11)
    for attr in ("left_margin", "right_margin", "top_margin", "bottom_margin"):
        setattr(section, attr, Inches(1))

    # Base styles for the parts we write ourselves.  The reference importer
    # below then swaps Normal / Heading2 / Heading3 / ... for the reference
    # manual's own definitions, so the copied Task 0 / Task 1 keeps the
    # original look.  Every paragraph we add sets its fonts explicitly, so
    # the swap does not change our Task 2.
    styles = doc.styles
    normal = styles["Normal"]
    normal.font.name = BODY_FONT
    normal.font.size = Pt(10.5)
    normal.paragraph_format.line_spacing = 1.12
    normal.paragraph_format.space_after = Pt(4)
    for style_name in ("List Bullet", "List Number"):
        st = styles[style_name]
        st.font.name = BODY_FONT
        st.font.size = Pt(10.5)
    manual = styles.add_style("Manual Number", WD_STYLE_TYPE.PARAGRAPH)
    manual.base_style = styles["Normal"]
    manual.font.name = BODY_FONT
    manual.font.size = Pt(10.5)
    manual.paragraph_format.left_indent = Inches(0.3)
    manual.paragraph_format.first_line_indent = Inches(-0.22)
    manual.paragraph_format.space_after = Pt(2)

    # Header / footer.
    header = section.header.paragraphs[0]
    header.text = "Student Lab Manual · TFLite Training"
    header.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    for r in header.runs:
        r.font.name = BODY_FONT
        r.font.size = Pt(8.5)
        r.font.color.rgb = RGBColor(0x64, 0x74, 0x8B)
    footer = section.footer.paragraphs[0]
    footer.alignment = WD_ALIGN_PARAGRAPH.CENTER
    fr = footer.add_run("Road Sign (3-Char) · ")
    fr.font.name = BODY_FONT
    fr.font.size = Pt(8.5)
    fr.font.color.rgb = RGBColor(0x64, 0x74, 0x8B)
    _add_page_number(footer)

    # ── Front matter + Task 0 + Task 1 (copied from the reference manual) ──
    # Title "Lab0: Introduction to TensorFlow Lite Micro and TFLiteTraining
    # App (Road Sign / Car Sign Project)", A) Objective / Background,
    # Task 0 (hardware) and Task 1 (Arduino setup) all come verbatim from
    # the reference manual; see _ReferenceImporter for the three patches.
    importer = _ReferenceImporter(doc, REFERENCE_DOCX)
    importer.run()
    inline_before_task2 = len(doc.inline_shapes)

    # ── Task 2 (Road Sign 3-Char) ───────────────────────────────────────
    # page_break_before (not an explicit break paragraph) keeps Task 1's
    # "Continue to next page" arrow on its own page without leaving a blank
    # page in between.
    task2 = _task_heading(doc, "Task 2 – Familiarize Yourself with the TFLiteTraining App (Road Sign 3-Char Project)")
    task2.paragraph_format.page_break_before = True
    _para(
        doc,
        "Now for the fun part — teaching an AI to read a 3-character sign! In this task you "
        "will collect example photos of each character, teach the app to recognize them, test "
        "a whole sign, and export your very own model.",
    )
    _callout_box(
        doc,
        "Why one character per card?",
        [
            "We collect single letters like A, 1 and 2 separately instead of taking one photo "
            "of the whole A12 sign, because teaching the AI individual letters lets it "
            "recognize ANY 3-character combination later! During Preview, the app "
            "automatically slices the whole sign into 3 boxes and checks each letter one by "
            "one.",
        ],
        icon="💡",
    )

    # ── 1. Start page ───────────────────────────────────────────────────
    _heading(doc, "1. Opening the App & Choosing 🛑 Road Sign (3-Char)", 4)
    _para(
        doc,
        "Launch TFLite Training. The Start Page shows three project cards — choose the 🛑 "
        "Road Sign (3-Char) one.",
    )
    _figure(doc, assets["01_start_page"], "Figure 2.1 — Start Page: choose the 🛑 Road Sign (3-Char) project.")
    _numbered(doc, "Click Get Started on the 🛑 Road Sign (3-Char) card. A fresh project opens with a 3-character sign already set up.")
    _numbered(doc, "Tip: Open Project (.tmproj) reopens a project you saved earlier — you will use it again at the very end of this lab.")

    # ── 2. Workspace overview ───────────────────────────────────────────
    _heading(doc, "2. Understanding the Four Main Areas", 4)
    _para(doc, "The workspace is like a small factory with four working areas:")
    _figure(doc, assets["02_workspace_overview"], "Figure 2.2 — The workspace: Target Sign Bar, Sample Area, Train Area and Preview Area.")
    _bullet(doc, "Target Sign Bar (top left) — where you type the 3 letters/numbers on your sign (for example A, 1, 2).")
    _bullet(doc, "Sample Area (left) — where you collect and tidy photos for each character.")
    _bullet(doc, "Train Area (middle) — where you train your AI model.")
    _bullet(doc, "Preview Area (right) — where you test your model and check whether a sign matches (🟢 YES or 🔴 NO).")
    _para(
        doc,
        "Along the top there is also a small toolbar with File ▾, ⚙ Settings, 📖 Tutorial and "
        "← Return Home. You will use ⚙ Settings in a moment.",
        size=10,
        color="475569",
    )

    # ── 3. Target sign ──────────────────────────────────────────────────
    _heading(doc, "3. Setting Your 3-Character Target Sign", 4)
    _para(
        doc,
        "Type the three characters of your sign into the Target Sign Bar. The character cards "
        "below update automatically as you type.",
    )
    _figure(doc, assets["03_target_sequence_and_dedup"], "Figure 2.3 — The Target Sign Bar: the three slots, swap buttons, reading direction and the duplicate badge.")
    _bullet(doc, "Click a slot and type 0–9 or A–Z. A complete character jumps you to the next slot automatically.")
    _bullet(doc, "Use the ◀ / ▶ buttons to swap two neighbouring characters.")
    _bullet(doc, "Reading direction is left → right by default. You can change it in the Train Area → Advanced.")
    _bullet(doc, "Same character twice? A 1 A only needs TWO cards — A and 1. The A card gets a small badge “Slots 1 & 3”; the AI learns A once and the app reuses it for both boxes.")
    _para(doc, "For this lab, set the target to A 1 2.", color="B91C1C", size=10)

    # ── 4. Collect samples ──────────────────────────────────────────────
    _page_break(doc)
    _heading(doc, "4. Collecting Samples in the Sample Area", 4)
    _para(
        doc,
        "The AI needs example photos of every character — around 20 per character is a great "
        "start. You can collect real photos, use the quick Auto-fill shortcut, or both.",
    )
    _para(doc, "The quick way — Auto-fill 24 EMNIST samples:", bold=True)
    _para(
        doc,
        "Open ⚙ Settings, expand ▸ Advanced Settings, and tick Auto-fill empty classes with "
        "EMNIST template samples (24 / class). Each character card instantly receives 24 "
        "built-in handwriting examples of that letter or number. Real photos are never touched "
        "— auto-fill only fills classes that are still empty.",
    )
    _figure(doc, assets["04a_settings_emnist_autofill"], "Figure 2.4a — ⚙ Settings: Preview source (left) and the EMNIST Auto-fill controls in Advanced Settings (right).")
    _numbered(doc, "Click ⚙ Settings.")
    _numbered(doc, "Under Preview, set Preview Image Input Source to Local Image File (you will use it for testing later).")
    _numbered(doc, "Expand ▸ Advanced Settings and tick Auto-fill empty classes with EMNIST template samples (24 / class).")
    _numbered(doc, "If the tick box was already on, click Fill Empty Classes with EMNIST Samples Now instead.")
    _numbered(doc, "Click Save changes.")
    _para(doc, "The real way — Webcam / Upload / Device:", bold=True)
    _para(
        doc,
        "On any character card, click Webcam, Upload or Device. A drawer opens below the card "
        "with a live view and the capture buttons.",
    )
    _figure(doc, assets["04b_capture_drawer"], "Figure 2.4b — The capture drawer on a character card: source tiles, Capture / Hold to Capture and Close.")
    _bullet(doc, "Webcam — uses your laptop camera. Great for photos of printed letters and numbers.")
    _bullet(doc, "Upload — pick photos you already have on your computer.")
    _bullet(doc, "Device — reads pictures from the ESP32-P4 camera on the smart car! Choose the right port in ⚙ Settings first.")
    _bullet(doc, "Capture — takes ONE photo. Hold to Capture (or hold the Space key) — keeps snapping while you hold.")
    _bullet(doc, "Close (Esc) — folds the drawer away.")
    _para(
        doc,
        "Top tip: capture the letter close-up and straight-on so it fills most of the picture. "
        "Vary the distance, angle and lighting a little — variety makes the AI smarter.",
        size=10,
        color="475569",
    )

    # ── 5. Sample Worktable ─────────────────────────────────────────────
    _page_break(doc)
    _heading(doc, "5. Preprocessing the Samples (Edit Page: Image Viewer & Sample Worktable)", 4)
    _para(
        doc,
        "Click Edit on a character card to open the Edit page. On the left, the Image Viewer "
        "shows your Original Sample with a crop box. On the right, the Sample Worktable shows "
        "the Processed Preview — the clean 28×28 picture the AI will actually learn from.",
    )
    _figure(doc, assets["05_sample_worktable_edit"], "Figure 2.5 — The Edit page: Image Viewer (left) and Sample Worktable (right).")
    _para(
        doc,
        "Dark Thr and Lum Thr work together: they turn very dark shadows or bright background "
        "areas into clean white space so the AI only focuses on the letter itself.",
    )
    _add_table(
        doc,
        ["Parameter", "Effects"],
        [
            ["Preprocess Mode (Auto / Manual ROI)", "Auto = the app finds the letter by itself (green box). Manual ROI = you draw the box yourself (blue box) — handy when Auto picks the wrong corner."],
            ["Dark Thr (Darkness Threshold)", "How dark a pixel may be and still count as the letter. Turning it up erases dark shadows around the letter."],
            ["Lum Thr (Luminosity Threshold)", "How bright the background may be. Turning it down erases bright background areas, leaving a clean letter."],
            ["ROI X1, Y1, X2, Y2", "The four numbers that describe your Manual box: X1/Y1 is the top-left corner and X2/Y2 the bottom-right corner (0 = top/left edge, 1 = bottom/right edge)."],
        ],
        widths=[1.9, 4.6],
    )
    _para(doc, "Handy keyboard shortcuts:", bold=True)
    _bullet(doc, "F1 — Auto mode, F2 — Manual ROI mode.")
    _bullet(doc, "Arrow keys — move to the next/previous sample.")
    _bullet(doc, "Shift + Arrow keys — nudge the ROI box in Manual mode.")
    _bullet(doc, "S — Save the sample. D — Delete the sample. Esc — close the Edit page.")
    _para(doc, "Remember to click Save Sample after any change!", color="B91C1C", size=10)

    # ── 6. Training ─────────────────────────────────────────────────────
    _page_break(doc)
    _heading(doc, "6. Training the Model in the Train Area", 4)
    _para(
        doc,
        "Time to teach the AI! Open the Train Area, click ▸ Advanced to see all the settings, "
        "then click Train Embedded Model. The status line shows Model Trained when it finishes "
        "— click it to see how accurate the model is for each character.",
    )
    _figure(doc, assets["06_training_and_advanced"], "Figure 2.6 — The Train Area → Advanced (left) and ⚙ Settings → Training Hyperparameters (right).")
    _para(doc, "The hyperparameters, in plain words:", bold=True)
    _add_table(
        doc,
        ["#", "Hyperparameter", "Default", "Range", "What it does"],
        [
            ["1", "Ink Polarity", "0 — Dark ink on bright bg", "0 / 1", "Which way round your letters are: normal dark-on-light writing (0), or bright letters on a dark sign (1)."],
            ["2", "Sign Reading Direction", "Left to Right", "L ➔ R / R ➔ L", "The order the app reads the three boxes of your sign."],
            ["3", "Image Size", "28", "8–512", "How many pixels wide the small letter picture is. 28×28 fits one letter perfectly — leave it."],
            ["4", "Batch Size", "32", "1 or more", "How many photos the AI practices with in one go."],
            ["5", "Epochs", "20", "1 or more", "How many times the AI reads through all your photos. For this project the app follows its own smart two-phase plan."],
            ["6", "Validation Split", "0.25", "0–0.5", "The part of your photos kept secret for testing."],
            ["7", "Learning Rate", "0.0016", "0 or more", "How big each learning step is. Small steps = careful but slow."],
            ["8", "💡 Use recommended", "—", "button", "One click: the app picks sensible numbers for you based on how many photos you have. Click this first!"],
            ["9", "Mix EMNIST into every class", "On", "on / off", "Mixes built-in handwriting examples with your photos so the AI learns from more variety."],
            ["10", "Synthetic per class", "200", "0–5000", "How many built-in handwriting examples are added to each character."],
            ["11", "Background / noise class", "On", "on / off", "Adds a “no letter here” class, so the AI can tell you when a box is empty."],
            ["12", "Fine-tune pre-trained backbone", "On", "on / off", "Starts from a smart pre-learned letter reader so your model trains fast and accurately even with just 20 photos!"],
            ["13", "Head epochs", "12", "1–200", "Practice rounds for the new part of the AI that learns YOUR letters."],
            ["14", "Fine-tune epochs", "25", "0–200", "Practice rounds where the AI polishes everything together."],
            ["15", "Fine-tune learning rate", "0.0002", "0 or more", "The small, careful step size used while polishing."],
        ],
        widths=[0.3, 1.55, 1.3, 1.0, 2.35],
        font_size=8.5,
    )
    _para(doc, "What to do, step by step:", bold=True)
    _numbered(doc, "Click 💡 Use recommended so the app picks good numbers for your photos.")
    _numbered(doc, "Make sure Mix EMNIST into every class and Fine-tune pre-trained backbone are on — they help a lot when you only have a few photos.")
    _numbered(doc, "Click Train Embedded Model and wait for the status Model Trained.")
    _numbered(doc, "Click the status to check the accuracy for A, 1, 2 (and the “no letter” class).")
    _numbered(doc, "Changed your target or added photos after training? The status shows ⚠ Stale — just train again.")

    # ── 7. Preview / testing ────────────────────────────────────────────
    _page_break(doc)
    _heading(doc, "7. Testing Your Model in the Preview Area", 4)
    _para(
        doc,
        "The Preview Area is your test bench: it shows the whole sign, splits it into 3 boxes, "
        "reads each box with your model, and gives a verdict.",
    )
    _para(doc, "Choose where the test picture comes from:", bold=True)
    _bullet(doc, "Device — live pictures from the ESP32-P4 camera on the smart car (pick the port in ⚙ Settings). This is the real thing!")
    _bullet(doc, "Webcam — your laptop camera.")
    _bullet(doc, "Local Image File — pictures you upload from your computer.")
    _para(doc, "No camera handy? Use the three magic buttons to make pretend test signs:", bold=True)
    _bullet(doc, "🧪 Synth Match — builds a test sign out of YOUR three characters (A 1 2). A well-trained model says YES.")
    _bullet(doc, "🔀 Synth Reversed — the same sign backwards, to double-check the reading direction.")
    _bullet(doc, "❌ Synth Mismatch — swaps in a wrong character. A good model says NO.")
    _para(doc, "Click Input to start reading, and watch the banner:")
    _figure(doc, assets["07a_preview_synth_match"], "Figure 2.7a — 🧪 Synth Match: all three boxes agree → 🟢 YES — MATCH.")
    _figure(doc, assets["07b_preview_synth_mismatch"], "Figure 2.7b — ❌ Synth Mismatch: one box disagrees → 🔴 NO — NOT MATCH.")
    _para(doc, "Reading the 3 character boxes:", bold=True)
    _bullet(doc, "WANT — the character the box should read.")
    _bullet(doc, "The small picture — the cleaned 28×28 image the AI actually looked at.")
    _bullet(doc, "The character the AI read, plus its confidence percentage. Green = correct, red = wrong.")
    _para(
        doc,
        "A box passes only if the AI (1) found a letter shape in it, (2) the letter is not "
        "“no letter”, (3) it matches the wanted character, and (4) it is at least 60% sure. "
        "The banner says 🟢 YES — MATCH when all three boxes pass; otherwise it says 🔴 NO — "
        "NOT MATCH, with a hint telling you which box went wrong.",
    )
    _para(
        doc,
        "Fix a failing box: open Edit for that character (Section 5), tidy its samples, and "
        "train again.",
        size=10,
        color="475569",
    )

    # ── 8. Export + save ────────────────────────────────────────────────
    _page_break(doc)
    _heading(doc, "8. Exporting the Model & Saving Your Project (.tmproj)", 4)
    _para(
        doc,
        "Happy with the YES / NO answers? Export the trained model and save your project so "
        "you can come back to it any time.",
    )
    _figure(doc, assets["08_export_and_save"], "Figure 2.8 — Export Model on the Preview card, and File ▾ → Save / Save as.")
    _para(doc, "Export the model:", bold=True)
    _numbered(doc, "Click Export Model on the Preview card.")
    _numbered(doc, "Choose a folder, then confirm.")
    _para(
        doc,
        "The folder receives a small set of files: the trained model (tm.tflite), the model "
        "packed as C/C++ code for the microcontroller (tm_model_data.h / .cpp), a settings "
        "file that remembers your three characters and the 28×28 picture size "
        "(model_settings.h / .cpp), and labels.txt listing the characters.",
    )
    _para(doc, "Save your project:", bold=True)
    _numbered(doc, "Open File ▾ → Save (⌘S / Ctrl+S), or Save as (⇧⌘S / Shift+Ctrl+S) to pick a new file.")
    _numbered(doc, "The .tmproj file is a little zip archive that remembers EVERYTHING: your characters, photos, preprocessing choices, training settings and the trained model.")
    _numbered(doc, "Reopen it later with Open Project (.tmproj) on the Start Page.")
    _callout_box(
        doc,
        "Save early and often",
        [
            "Save the .tmproj before training and again after a good result — it is the only "
            "way to get back to exactly what you had.",
        ],
    )
    _para(
        doc,
        "That's it — you collected letters, trained an AI to read them, tested a whole sign, "
        "and exported your model. Well done!",
        bold=True,
        color="1D4ED8",
    )

    # Scope guard: this lab stops at the end of Task 2 (Export / Save).
    # A future edit that slips a Task 3 (firmware flashing / serial
    # commands) back in should fail the build rather than silently
    # producing the wrong manual.
    all_text = "\n".join(p.text for p in doc.paragraphs)
    for table in doc.tables:
        for row in table.rows:
            for cell in row.cells:
                all_text += "\n" + cell.text
    for banned in ("Task 3", "TFLite_Sign", "Copy the five files"):
        if banned.lower() in all_text.lower():
            raise RuntimeError(f"manual scope violation: contains {banned!r}")
    our_figures = len(doc.inline_shapes) - inline_before_task2
    if our_figures != 10:
        raise RuntimeError(f"expected 10 Task 2 figures, found {our_figures}")

    LOCAL_DOCX.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(LOCAL_DOCX))
    log(f"docx -> {LOCAL_DOCX}")
    ONEDRIVE_DOCX.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(LOCAL_DOCX, ONEDRIVE_DOCX)
    log(f"docx -> {ONEDRIVE_DOCX}")
    return LOCAL_DOCX


# ── main ────────────────────────────────────────────────────────────────────
def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only-docx", action="store_true",
                        help="skip Playwright and reuse existing screenshots")
    parser.add_argument("--port", type=int, default=None,
                        help="use a fixed Streamlit port instead of a free one")
    args = parser.parse_args(list(argv) if argv is not None else None)

    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    if BUILD_LOG.exists() and not args.only_docx:
        BUILD_LOG.unlink()

    expected = [
        "01_start_page", "02_workspace_overview", "03_target_sequence_and_dedup",
        "04a_settings_emnist_autofill", "04b_capture_drawer", "05_sample_worktable_edit",
        "06_training_and_advanced", "07a_preview_synth_match",
        "07b_preview_synth_mismatch", "08_export_and_save",
    ]
    assets = {name: ASSET_DIR / f"{name}.png" for name in expected}

    if not args.only_docx:
        app = StreamlitApp(port=args.port)
        try:
            app.start()
            captured = capture_walkthrough(app)
            assets.update(captured)
        finally:
            app.stop()
    else:
        missing = [str(p) for p in assets.values() if not p.exists()]
        if missing:
            raise SystemExit("missing screenshots:\n  " + "\n  ".join(missing))

    missing = [str(p) for p in assets.values() if not p.exists()]
    if missing:
        raise SystemExit("manual build incomplete; missing:\n  " + "\n  ".join(missing))

    build_docx(assets)
    manifest = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "project_type": "sign_ocr",
        "screenshots": [str(p.relative_to(APP_DIR)) for p in assets.values()],
        "docx": [str(LOCAL_DOCX), str(ONEDRIVE_DOCX)],
    }
    (ASSET_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    log("Manual build complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
