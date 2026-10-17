"""Render the workspace SPA outside Streamlit, so its JavaScript can be tested.

The workspace is one hand-written HTML/JS page emitted from a single f-string
in `app._render_tm_old_frontend_html`.  There is no module boundary to import
and no route that serves it standalone, so a JS test has two options: boot
Streamlit and drive a real browser through the iframe, or evaluate the f-string
directly.  This module does the second.

It is not a re-implementation of the f-string.  The text between
`components.html(f'''` and the closing `''')` is compiled and evaluated as the
f-string it already is, with the handful of Python values it interpolates
supplied explicitly.  That means the extracted JS is byte-identical to what the
app ships — brace-doubling and all — rather than a hand-un-doubled copy that
could drift from the real thing without anyone noticing.

`boot_payload()` builds the JSON injected at `const STATE = {data}`; override
any key to put the page in the state a test needs.
"""
from __future__ import annotations

import functools
import http.server
import json
import re
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator

APP_PY = Path(__file__).resolve().parents[1] / "app.py"

# The four values the f-string interpolates.  Three are constants of the page;
# `data` is the boot payload and is supplied per test.
_LOGO = ""
_EMNIST_SPLIT_OPTIONS = "[]"
_BACKGROUND_CLASS_NAME = "_background"


def _frontend_fstring_body() -> str:
    src = APP_PY.read_text(encoding="utf-8")
    start = src.index("def _render_tm_old_frontend_html")
    begin = src.index("f'''", start)
    rest = src[begin + 4:]
    end = re.search(r"\n\s*''',", rest)
    if end is None:
        raise RuntimeError("could not find the end of the workspace f-string")
    return rest[: end.start()]


def build_spa_html(payload: Dict[str, Any]) -> str:
    """The full page HTML, with `payload` as the SPA's boot STATE."""
    body = _frontend_fstring_body()
    # Evaluated exactly as Python would evaluate it inside the app: doubled
    # braces collapse, and the four real fields are substituted.
    code = compile("f'''" + body + "'''", str(APP_PY), "eval")
    names = {
        "_AIO_SCOUTS_LOGO_B64": _LOGO,
        "emnist_split_options": _EMNIST_SPLIT_OPTIONS,
        "BACKGROUND_CLASS_NAME": _BACKGROUND_CLASS_NAME,
        "data": json.dumps(payload),
    }
    return eval(code, names)  # noqa: S307 - evaluating our own source, by design


def boot_payload(**overrides: Any) -> Dict[str, Any]:
    """A minimal but complete STATE for the workspace, in a sign project.

    Every key here is one the app's own payload writer emits
    (`_render_tm_old_frontend_html`); nothing is invented, so the page boots
    into a state the real app could produce.
    """
    payload: Dict[str, Any] = {
        "port": 1,
        "session": "test-session",
        "return_target": "home",
        "project_type": "sign_ocr",
        "classes": ["A", "1"],
        "counts": {"A": 3, "1": 2},
        "train_enabled": True,
        "export_enabled": True,
        "notice": "",
        "serial_ports": [],
        "current_serial_port": "",
        "current_serial_baud": 115200,
        "current_serial_sync": "AA 55 AA",
        "current_serial_frame_side": 96,
        "current_serial_channels": 3,
        "preview_dark_thresh": 0,
        "preview_lum_thresh": 100,
        "webcam_options": [],
        "current_webcam_index": 0,
        "current_webcam_id": "",
        "sample_previews": {},
        "initial_open_source_class": "",
        "initial_open_source_kind": "",
        "class_preprocess": {},
        "sample_preprocess": {},
        "train_cfg": {
            "img_size": 28,
            "preprocess_mode": "auto_by_label",
            "crop_mode": "auto_search",
            "reading_order": "ltr",
            "sign_sequence": ["A", "1", "A"],
            "ocr_emnist_enabled": True,
            "ocr_transfer_learning": True,
        },
        "train_cfg_defaults": {
            "img_size": 28,
            "preprocess_mode": "auto_by_label",
            "crop_mode": "auto_search",
            "reading_order": "ltr",
            "sign_sequence": None,
            "ocr_emnist_enabled": True,
            "ocr_transfer_learning": True,
        },
    }
    payload.update(overrides)
    return payload


@contextmanager
def serve(html: str) -> Iterator[str]:
    """Serve `html` on a throwaway localhost origin; yields its URL.

    Served over HTTP rather than injected with page.set_content() because the
    page uses localStorage, and a `data:`/about:blank document is an opaque
    origin where touching it THROWS.  That throw aborts the whole workspace
    script partway through, which looks like a broken feature rather than a
    broken harness.  The real app is served from a real origin too, so this is
    also the more faithful setup.
    """
    body = html.encode("utf-8")

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802 - BaseHTTPRequestHandler's API
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):  # keep the test output readable
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}/"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


# Every network call the page makes, recorded so a test can assert on the
# request the SPA actually issued rather than on the code that built it.
STUB_FETCH = """
window.__fetches = [];
window.fetch = async (url, opts) => {
  try { window.__fetches.push(String(url)); } catch (e) {}
  return {
    ok: true, status: 200,
    json: async () => ({ok: '1'}),
    text: async () => '',
    blob: async () => new Blob([]),
  };
};
"""
