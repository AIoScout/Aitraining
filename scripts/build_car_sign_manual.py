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
4. Assembles the student manual with python-docx (Steps 1-8), reusing the
   visual tone of the existing HKUST TFLite Training manual.
5. Writes the .docx to the repository root and to the OneDrive Student_Helper
   folder.

The manual deliberately stops after Export Model / Save Project.  It contains
no Arduino IDE, firmware-flashing, UART or Serial-Monitor instructions.

Usage
-----
    cd AItraining
    .venv/bin/python scripts/build_car_sign_manual.py

    # Reuse already-captured screenshots and only rebuild the Word file:
    .venv/bin/python scripts/build_car_sign_manual.py --only-docx

Requires: streamlit, playwright (+ system Chrome), pillow, python-docx.
"""
from __future__ import annotations

import argparse
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


def _callout_box(doc, title: str, body: Sequence[str]) -> None:
    table = doc.add_table(rows=1, cols=1)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    cell = table.cell(0, 0)
    _set_cell_shading(cell, "FEF2F2")
    _set_cell_borders(cell)
    cell.text = ""
    p = cell.paragraphs[0]
    run = p.add_run(f"⚠ {title}")
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
    run.font.color.rgb = RGBColor(0x0F, 0x17, 0x2A)
    if level == 2:
        run.font.size = Pt(16)
        p.paragraph_format.space_before = Pt(14)
        p.paragraph_format.space_after = Pt(6)
    elif level == 3:
        run.font.size = Pt(12)
        p.paragraph_format.space_before = Pt(10)
        p.paragraph_format.space_after = Pt(3)
    else:
        run.font.size = Pt(10.5)


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


def build_docx(assets: Dict[str, Path]) -> Path:
    doc = Document()
    section = doc.sections[0]
    section.page_width = Inches(8.5)
    section.page_height = Inches(11)
    for attr in ("left_margin", "right_margin", "top_margin", "bottom_margin"):
        setattr(section, attr, Inches(1))

    # Consistent styles; the reference manual is an Arial/Times mix, but the
    # app screenshots are clean and modern, so Arial throughout reads best.
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

    # ── Title block ─────────────────────────────────────────────────────
    _heading(doc, "Student Lab Manual", 2)
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(2)
    r = p.add_run("Road Sign / Car Sign Project (3-Char)")
    r.bold = True
    r.font.name = HEADING_FONT
    r.font.size = Pt(22)
    r.font.color.rgb = RGBColor(0x1D, 0x4E, 0xD8)
    _para(doc, "TFLite Training desktop app · sign_ocr workflow", size=12, color="475569")
    _para(
        doc,
        "In this lab you will build a three-character road-sign verifier: declare the target "
        "sequence, collect or auto-generate character glyphs, train an int8 model, verify the "
        "three-slot reading in Preview, then export the model and save the project archive.",
        size=10.5,
    )
    _add_table(
        doc,
        ["Lab at a glance", "Details"],
        [
            ["Project type", "🛑 Road Sign (3-Char)  (sign_ocr)"],
            ["Target in this lab", "A ▸ 1 ▸ 2  (reading direction: Left to Right)"],
            ["Model input", "3 slots × 28 × 28 int8 glyphs"],
            ["Verdict", "🟢 YES — MATCH / 🔴 NO — NOT MATCH"],
            ["Estimated time", "45–60 minutes"],
            ["Hardware", "Not required — EMNIST Auto-Fill + Synth plates"],
            ["Ends with", "Export Model + Save Project (.tmproj)"],
        ],
        widths=[1.7, 4.8],
    )

    _heading(doc, "Before you start", 3)
    _para(
        doc,
        "TensorFlow Lite (LiteRT) for Microcontrollers is designed to run small machine-learning "
        "models on microcontrollers. TFLite Training is the desktop trainer used in this course: "
        "it captures or imports samples, applies the same preprocessing the device will use, "
        "trains an int8 TensorFlow Lite model, and exports C/C++ sources for a downstream project.",
    )
    _para(doc, "By the end of this lab you will be able to:")
    _bullet(doc, "Explain why a sign is trained as one class per unique character, not as one class per sign.")
    _bullet(doc, "Set a three-slot target and read the duplicate-character badge (e.g. A ▸ 1 ▸ A → “Slots 1 & 3”).")
    _bullet(doc, "Collect glyph samples with the camera/device drawer or fill every empty class with EMNIST Auto-Fill.")
    _bullet(doc, "Tune ROI and luminance thresholds in the Sample Worktable.")
    _bullet(doc, "Choose sign-specific hyperparameters and explain the EMNIST mix and two-stage transfer learning.")
    _bullet(doc, "Verify a sign in Preview and read the three per-slot results behind the YES / NO banner.")
    _bullet(doc, "Export the trained artifacts and save/reopen the whole project as a .tmproj archive.")

    _heading(doc, "Key idea: training is per character; splitting is per sign", 3)
    _para(
        doc,
        "The sign project reuses the same 28×28 single-glyph classifier as the OCR project. "
        "Each character card collects glyphs of ONE character. The full three-character plate "
        "is split into three slots only at Preview / inference time. Keeping those two ideas "
        "separate is the most important rule in this lab.",
    )

    # ── Step 1 ──────────────────────────────────────────────────────────
    _heading(doc, "Step 1 — Choose the Road Sign project", 2)
    _para(
        doc,
        "Launch TFLite Training. The Start Page shows three project cards. Choose the third one, "
        "🛑 Road Sign (3-Char).",
    )
    _figure(doc, assets["01_start_page"], "Figure 1 — Start Page: choose the Road Sign (3-Char) project.")
    _numbered(doc, "Click Get Started on the 🛑 Road Sign (3-Char) card. A fresh project opens with the target A ▸ B ▸ C and matching character cards.")
    _numbered(doc, "Alternatively, click Open Project (.tmproj) to reopen a saved sign project. The archive's project type is authoritative; a cross-type file is announced rather than silently switched.")
    _numbered(doc, "The third button, Import classified class, starts the ordinary image importer. You do not need it for this lab.")
    _para(doc, "Tip: the app follows the system dark-mode setting (prefers-color-scheme: dark).", size=10, color="475569")

    # ── Step 2 ──────────────────────────────────────────────────────────
    _heading(doc, "Step 2 — Tour the workspace and set the target", 2)
    _para(
        doc,
        "The workspace has three columns: Character Sample Cards (left), Training (middle), "
        "and Preview (right). The top toolbar holds File/Edit, Settings, Preferences, the "
        "📖 Tutorial, and Return Home.",
    )
    _figure(doc, assets["02_workspace_overview"], "Figure 2 — Workspace overview with the five working areas.")
    _bullet(doc, "Top bar — File ▾ (save/open/export dataset), Settings, 📖 Tutorial, ← Return Home, and the Preview card's Export Model button.")
    _bullet(doc, "Target Sign Sequence Bar (#signSeqBar) — the single writer of the class list and target.")
    _bullet(doc, "Character Sample Cards — one card per unique character; each card is a capture surface.")
    _bullet(doc, "Training card — Train Embedded Model, status/accuracy, Advanced hyperparameters, 💡 Use recommended.")
    _bullet(doc, "Preview card — source controls, Input/ROI/Orig, the 3-slot inference result, and 🧪 Synth pills for sign projects.")
    _para(
        doc,
        "Set the target now: type A, 1, 2 into the three slots (left to right). "
        "The class cards update automatically. If the first-run tour appears, close it with "
        "Skip Tour or Esc — you can reopen it any time from 📖 Tutorial.",
    )

    # ── Step 3 ──────────────────────────────────────────────────────────
    _heading(doc, "Step 3 — Understand the sequence bar and duplicate characters", 2)
    _para(
        doc,
        "The three slots are the single source of truth. Each valid keystroke updates the target, "
        "auto-advances to the next slot, and syncs the class cards to the server.",
    )
    _figure(doc, assets["03_target_sequence_and_dedup"], "Figure 3 — Target slots, swap buttons, reading direction and duplicate-character badge.")
    _bullet(doc, "Slots — click a slot and type 0–9 or A–Z. ✔ A complete character advances to the next slot. Backspace on an empty slot clears the previous one.")
    _bullet(doc, "◀ / ▶ — swap the character with its neighbour. The edge buttons are disabled.")
    _bullet(doc, "Reading direction — the bar echoes left → right (or right → left). Change it in Training → Advanced → Sign Reading Direction. RTL reverses the three boxes after the split; pixels are never mirrored.")
    _bullet(doc, "Duplicate character — A ▸ 1 ▸ A creates only two class cards, A and 1. The A card carries the badge “Slots 1 & 3”; the model trains A once.")
    _bullet(doc, "Auto-sync — every edit posts the new sequence/class list and returns fresh counts and previews atomically. New characters are auto-filled in the same request when Auto-Fill is on.")
    _bullet(doc, "After training — editing the sequence changes the dataset fingerprint; the Training status adds ⚠ Stale until you retrain.")
    _para(doc, "Return to A ▸ 1 ▸ 2 before continuing.", bold=False, color="B91C1C", size=10)

    # ── Step 4 ──────────────────────────────────────────────────────────
    _page_break(doc)
    _heading(doc, "Step 4 — Collect training samples", 2)
    _callout_box(
        doc,
        "Critical training rule",
        [
            "Each character card must collect single-character glyph samples — the input the 28×28 model actually learns.",
            "Use EMNIST Auto-Fill (⚙ Settings → Advanced Settings) for a zero-camera workflow, or capture/upload a frame and isolate ONE character with Auto / Manual ROI.",
            "A whole 3-character sign filed under “A” trains A on the whole sign. The automatic three-slot split belongs to Preview / inference, not training.",
        ],
    )
    _heading(doc, "4A — Zero hardware: EMNIST Auto-Fill (recommended first pass)", 3)
    _para(
        doc,
        "Open ⚙ Settings → Preview and set Preview Image Input Source to Local Image File. "
        "Then expand ▸ Advanced Settings, enable the auto-fill checkbox, and (optionally) click "
        "Fill Empty Classes with EMNIST Samples Now. Each empty character class receives 24 "
        "dark-ink-on-light 96×96 captures; real camera samples are never touched.",
    )
    _figure(doc, assets["04a_settings_emnist_autofill"], "Figure 4a — Preview source and EMNIST Auto-Fill controls in Settings.")
    _numbered(doc, "Set Preview Image Input Source = Local Image File.")
    _numbered(doc, "Enable Auto-fill empty classes with EMNIST template samples (24 / class).")
    _numbered(doc, "Wait for each card to show 24 Image Samples; the fill is immediate. Use the one-shot button if the checkbox is already on.")
    _numbered(doc, "Click Save changes to apply the settings.")
    _para(
        doc,
        "The auto-fill flag is project state. Changing a slot later backfills the new character "
        "in the same class-sync request, so a new class does not stay at 0 samples.",
        size=10,
        color="475569",
    )

    _heading(doc, "4B — Camera, device drawer and upload", 3)
    _para(
        doc,
        "On a character card, click Webcam, Upload, or Device. The drawer opens below the card. "
        "Capture saves one frame; Hold to Capture (or holding Space) records a burst. "
        "Close folds the drawer away.",
    )
    _figure(doc, assets["04b_capture_drawer"], "Figure 4b — Device capture drawer: source tiles, Capture / Hold, Close and sample thumbnails.")
    _para(
        doc,
        "The example above shows the drawer with no board attached: it reports that no "
        "compatible serial device was detected and leaves the live preview empty. Choose the "
        "correct port in Settings to see frames. Webcam and Upload work without a board.",
        size=9.5,
        color="475569",
    )
    _bullet(doc, "Webcam — pick the camera in Settings; live preview appears in the drawer.")
    _bullet(doc, "Device — connect the ESP32-P4 serial source, choose the port in Settings, and capture from the live serial frame. Use Capture or Hold to Capture.")
    _bullet(doc, "Upload — choose existing images. Great for photos of printed characters.")
    _bullet(doc, "Space — while a Webcam/Device drawer is open: tap for one frame, hold ≥220 ms for a burst; release stops.")
    _bullet(doc, "Close — the pill in the drawer header (title: Collapse panel (Esc)).")
    _para(
        doc,
        "If you use camera samples, capture a close view where the current character is the "
        "dominant glyph, or use Manual ROI in Step 5. Vary distance, tilt and lighting; aim for "
        "at least 10–20 varied samples per class.",
        size=10,
        color="475569",
    )

    # ── Step 5 ──────────────────────────────────────────────────────────
    _page_break(doc)
    _heading(doc, "Step 5 — Inspect and tune samples in the Sample Worktable", 2)
    _para(
        doc,
        "Click Edit on a character card to open the worktable. The left shows the Original "
        "Sample with its ROI box; the right shows the Processed Preview — the 28×28 glyph "
        "the model receives.",
    )
    _figure(doc, assets["05_sample_worktable_edit"], "Figure 5 — Sample Worktable: original/ROI, 28×28 processed glyph, mode, thresholds and save controls.")
    _bullet(doc, "Original Sample — the raw capture. The green box is auto-detected; blue is Manual ROI.")
    _bullet(doc, "Processed Preview — integer Otsu + projection + 28×28 normalization. It should show one centered glyph with a clean margin.")
    _bullet(doc, "Preprocess Mode — Auto (find sign) or Manual ROI. Manual ROI pins a fixed box; drag on the original image or type ROI X1/Y1/X2/Y2 (0–1).")
    _bullet(doc, "Dark Thr (0–100) / Lum Thr (50–255) — the luminance window. Defaults 0/100 mean exact Otsu; other values pin the threshold and add a dark floor.")
    _bullet(doc, "Save Sample — stores the current sample's edits. Saved is shown when there is nothing to save.")
    _bullet(doc, "Shortcuts — F1 Auto, F2 Manual ROI, S Save, D Delete Sample, Esc Close, arrows navigate, Shift+Arrows nudge the ROI.")
    _para(
        doc,
        "Remember: the worktable edits a class sample, not the Preview's three-slot split. "
        "Use it to make each character's glyph clean and consistent.",
        size=10,
        color="475569",
    )

    # ── Step 6 ──────────────────────────────────────────────────────────
    _page_break(doc)
    _heading(doc, "Step 6 — Train the model", 2)
    _para(
        doc,
        "Open Training → ▸ Advanced. The project-specific controls are Sign Reading Direction "
        "and Ink Polarity; the OCR/sign block holds EMNIST mixing and two-stage transfer "
        "learning. Start with the defaults, then click Train Embedded Model.",
    )
    _figure(doc, assets["06_training_and_advanced"], "Figure 6 — Training card → Advanced (left) and Settings → Training Hyperparameters (right).")
    _para(doc, "Complete hyperparameter reference for sign_ocr:", bold=True)
    _add_table(
        doc,
        ["Field", "Default", "Range / options", "What it does"],
        [
            ["Ink Polarity", "0 — Dark ink on bright bg", "0 / 1", "Global ink polarity for every class; 1 = bright ink on dark bg."],
            ["Sign Reading Direction", "Left to Right (L ➔ R)", "LTR / RTL", "Order of the three slots. RTL reverses boxes after the split, never pixels."],
            ["Image Size", "28", "8–512", "Model input side. 28×28 is the glyph geometry; keep it for sign_ocr."],
            ["Batch Size", "32", "≥1", "Samples per gradient step; auto-adjust may lower it for small sets."],
            ["Epochs", "20", "≥1", "Base epoch count; the OCR strategy replaces it with Head + Fine-tune epochs."],
            ["Validation Split", "0.25", "0–0.5", "Fraction held out for validation."],
            ["Learning Rate", "0.0016", "≥0", "Base learning rate; the two-stage schedule has its own head/finetune rates."],
            ["💡 Use recommended", "—", "button", "Sets Batch Size, Epochs, Learning Rate and Validation Split from the dataset size (~800 updates, ≥4 steps/epoch). Does not change Image Size or architecture."],
            ["Mix EMNIST into every class", "On", "on/off", "Adds synthetic EMNIST glyphs to every real class."],
            ["EMNIST split", "balanced", "balanced / letters / digits", "Which EMNIST subset supplies the synthetic glyphs."],
            ["Synthetic per class", "200", "0–5000", "Synthetic EMNIST glyphs added per class."],
            ["Background / noise class", "On", "on/off", "Appends the _background rejector used for blank slots."],
            ["Background crops", "0", "0–5000", "Random non-glyph crops; 0 mirrors Synthetic per class."],
            ["Fine-tune pre-trained backbone", "On", "on/off", "Two-stage transfer learning from a cached EMNIST backbone."],
            ["Pre-train samples / class", "1200", "100–6000", "Backbone warm-up corpus size."],
            ["Pre-train epochs", "4", "1–100", "Backbone warm-up epochs."],
            ["Head epochs", "12", "1–200", "Phase 1: train the new head with the backbone frozen."],
            ["Head learning rate", "0.0015", "≥0", "Phase 1 learning rate."],
            ["Fine-tune epochs", "25", "0–200", "Phase 2: unfreeze the last conv block(s)."],
            ["Fine-tune learning rate", "0.0002", "≥0", "Phase 2 learning rate."],
            ["Conv blocks to unfreeze", "1", "0–16", "How many final backbone blocks are unfrozen."],
        ],
        widths=[1.65, 1.35, 1.35, 2.15],
        font_size=8.5,
    )
    _para(
        doc,
        "Under the hood: EMNIST mixing aligns synthetic glyphs to the capture convention, "
        "isolates and centers them, and appends random background crops for the _background "
        "class. Two-stage transfer learning first warms a cached EMNIST backbone, then trains "
        "the new classification head with the backbone frozen, and finally unfreezes the last "
        "conv block(s) at the lower fine-tune rate.",
        size=9.5,
        color="475569",
    )
    _heading(doc, "After training", 3)
    _bullet(doc, "The status changes to Model Trained · tap for accuracy. Click it to see per-class accuracy, including 1, 2, A and _background.")
    _bullet(doc, "The Train button stays available for re-training with new samples or a changed target.")
    _bullet(doc, "If you change a character after training, the status adds ⚠ Stale. Retrain before exporting; the old model's labels may not include the new character.")
    _bullet(doc, "Training uses the same int8 graph Preview runs, so Preview accuracy predicts the exported model's behavior.")

    # ── Step 7 ──────────────────────────────────────────────────────────
    _page_break(doc)
    _heading(doc, "Step 7 — Verify the sign in Preview", 2)
    _para(
        doc,
        "Set Preview Image Input Source to Local Image File. The context bar shows the Upload "
        "control and three sign-only pills: 🧪 Synth Match, 🔀 Synth Reversed and "
        "❌ Synth Mismatch. Toggle Input ON to run predictions.",
    )
    _figure(doc, assets["07a_preview_synth_match"], "Figure 7a — Synth Match: three-slot reading and the 🟢 YES — MATCH verdict.")
    _bullet(doc, "🧪 Synth Match — composes the current target characters into a sign plate and asserts YES.")
    _bullet(doc, "🔀 Synth Reversed — composes the characters in reversed order to probe LTR/RTL direction.")
    _bullet(doc, "❌ Synth Mismatch — substitutes a wrong character and asserts NO.")
    _bullet(doc, "Input — starts/stops the prediction loop; ROI draws the crop box; Orig shows the full masked frame.")
    _bullet(doc, "The banner replaces the class bars: 🟢 YES — MATCH means all three slots agreed; 🔴 NO — NOT MATCH names the failing slot(s).")
    _bullet(doc, "Each .signmatch-slot shows WANT <char>, the read glyph, its 28×28 image, and the confidence percentage. Green = match, red = fail.")
    _para(doc, "A YES requires all three slots to satisfy the same per-slot rule:", bold=True)
    doc.paragraphs[-1].paragraph_format.keep_with_next = True
    _numbered(doc, "The split found ink in the slot.")
    _numbered(doc, "The winning class is not _background.")
    _numbered(doc, "The winning class equals the target character for that slot.")
    _numbered(doc, "The raw confidence (best_raw / sum(raw)) is at least 0.60.")
    _numbered(doc, "The exported model uses the same rule; the sign path opens the generic OOD layers and rejects with _background + the confidence floor.")
    _figure(doc, assets["07b_preview_synth_mismatch"], "Figure 7b — Synth Mismatch: the red NO verdict and the per-slot hint.")
    _para(
        doc,
        "Read a red banner from the hint: “slot 2 read ‘6’ at 41.2%” means the glyph was "
        "classified as the wrong character or missed the 60% floor. A _background result means "
        "the slot crop held no recognized glyph. Use the worktable to clean that character's "
        "samples, then retrain.",
        size=10,
        color="475569",
    )

    # ── Step 8 ──────────────────────────────────────────────────────────
    _page_break(doc)
    _heading(doc, "Step 8 — Export the model and save the project", 2)
    _para(
        doc,
        "When the Preview verdict is reliably correct, export the trained artifacts for your "
        "downstream project and save the whole workspace so you can reopen it later.",
    )
    _figure(doc, assets["08_export_and_save"], "Figure 8 — Export Model in the Preview card and Save / Save as in the File menu.")
    _heading(doc, "Export Model", 3)
    _numbered(doc, "Click Export Model in the Preview card head.")
    _numbered(doc, "Choose the target folder for the downstream project.")
    _numbered(doc, "Confirm overwrite if the folder already contains an export.")
    _para(doc, "With the default export name tm, the folder receives:")
    _add_table(
        doc,
        ["File", "Purpose"],
        [
            ["tm.tflite", "The trained int8 TensorFlow Lite model."],
            ["tm_model_data.h / .cpp", "The model bytes as a C/C++ array (default symbol g_tm_model_data)."],
            ["model_settings.h / .cpp", "Labels, 28×28 geometry, target sequence, class indices, reading order, confidence floor, thresholds and OOD constants."],
            ["model_resolver.h", "The MicroInterpreter operator resolver for this model's ops."],
            ["labels.txt", "Human-readable label order, including _background when enabled."],
        ],
        widths=[2.1, 4.4],
    )
    _para(
        doc,
        "For the downstream project, the usual integration set is tm_model_data.cpp/.h, "
        "model_resolver.h and model_settings.cpp/.h. The .tflite and labels.txt are useful "
        "diagnostics. Exporting also pins Preview to this exact model until the next export.",
        size=10,
        color="475569",
    )
    _heading(doc, "Save Project (.tmproj)", 3)
    _numbered(doc, "Open File ▾ → Save (⌘S / Ctrl+S) to save to the current path, or Save as (⇧⌘S / Shift+Ctrl+S) to choose a new file.")
    _numbered(doc, "The .tmproj is a ZIP archive containing the project header/type, classes and target, samples, preprocessing, training config and the latest run artifacts.")
    _numbered(doc, "Reopen it later with Open Project (.tmproj) on the Start Page. The archive restores the full sign workspace, including sign_sequence, reading_order and the Auto-Fill flag.")
    _callout_box(
        doc,
        "Save early and often",
        [
            "Save the .tmproj before training and again after a good result. A project archive is the only way to return to the exact class list, target, preprocessing and trained model.",
        ],
    )
    _para(
        doc,
        "You have completed the full Road Sign (3-Char) workflow: target sequence → glyph "
        "samples → worktable preprocessing → training → three-slot Preview verification → "
        "export and project save.",
        bold=True,
        size=10.5,
        color="1D4ED8",
    )

    # Scope guard: this lab stops at Export / Save.  A future edit that
    # slips an Arduino/Serial section back in should fail the build rather
    # than silently producing the wrong manual.
    all_text = "\n".join(p.text for p in doc.paragraphs)
    for table in doc.tables:
        for row in table.rows:
            for cell in row.cells:
                all_text += "\n" + cell.text
    for banned in ("Arduino", "Serial Monitor", "TFLite_Sign.ino", "TFLite_Sign"):
        if banned.lower() in all_text.lower():
            raise RuntimeError(f"manual scope violation: contains {banned!r}")
    if len(doc.inline_shapes) < 10:
        raise RuntimeError(f"expected 10 figures, found {len(doc.inline_shapes)}")

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
