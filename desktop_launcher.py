from __future__ import annotations

import argparse
import contextlib
import json
import multiprocessing
import os
import signal
import socket
import sys
import threading
import time
import urllib.request
from pathlib import Path

# Cold start of the packaged app measured ~80 s on arm64 (TF + streamlit
# imports from the onedir bundle); leave generous headroom for slower Macs.
_SERVER_READY_TIMEOUT_S = 180.0


def _resource_path(rel_path: str) -> Path:
    if hasattr(sys, "_MEIPASS"):
        return (Path(getattr(sys, "_MEIPASS")) / rel_path).resolve()
    return (Path(__file__).resolve().parent / rel_path).resolve()


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _wait_http_ready(url: str, timeout_s: float = _SERVER_READY_TIMEOUT_S) -> None:
    deadline = time.time() + timeout_s
    last_err: Exception | None = None
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as resp:
                if 200 <= resp.status < 500:
                    return
        except Exception as e:
            last_err = e
        time.sleep(0.25)
    raise RuntimeError(f"Streamlit not ready: {url} ({last_err})")


def _app_data_dir() -> Path:
    env_override = os.getenv("TFLITE_TRAINING_DATA_DIR")
    if env_override:
        return Path(env_override).expanduser().resolve()
    if sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support"
    elif os.name == "nt":
        base = Path(os.environ.get("APPDATA", str(Path.home() / "AppData" / "Roaming")))
    else:
        base = Path(os.environ.get("XDG_DATA_HOME", str(Path.home() / ".local" / "share")))
    return (base / "TFLiteTraining").resolve()


def _debug_post(hypothesis_id: str, location: str, msg: str, data: dict | None = None) -> None:
    env_path = Path(".dbg/open-project-layout.env")
    url = "http://127.0.0.1:7777/event"
    session_id = "open-project-layout"
    try:
        if env_path.exists():
            for line in env_path.read_text(encoding="utf-8").splitlines():
                if line.startswith("DEBUG_SERVER_URL="):
                    url = line.split("=", 1)[1].strip() or url
                elif line.startswith("DEBUG_SESSION_ID="):
                    session_id = line.split("=", 1)[1].strip() or session_id
    except Exception:
        pass
    payload = {
        "sessionId": session_id,
        "runId": "pre-fix",
        "hypothesisId": hypothesis_id,
        "location": location,
        "msg": msg,
        "data": data or {},
        "ts": int(time.time() * 1000),
    }
    try:
        req = urllib.request.Request(
            url,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        urllib.request.urlopen(req, timeout=1.5).read()
    except Exception:
        pass


def _exit_when_parent_dies() -> None:
    # daemon=True only reaps the child on a clean parent exit; Force Quit
    # (SIGKILL), crashes and pkill leave the streamlit server running as an
    # orphan holding ~1 GB of TF memory. When the parent dies the child gets
    # reparented (ppid changes), so poll for that and self-exit.
    parent_pid = os.getppid()

    def _watch() -> None:
        while True:
            if os.getppid() != parent_pid:
                os._exit(1)
            time.sleep(1.0)

    threading.Thread(target=_watch, daemon=True).start()


def _run_streamlit_server(port: int, log_path: str) -> None:
    import traceback

    from streamlit.web import bootstrap

    _exit_when_parent_dies()

    app_py = _resource_path("app.py")
    os.environ.setdefault("STREAMLIT_BROWSER_GATHER_USAGE_STATS", "false")
    os.environ.setdefault("STREAMLIT_GLOBAL_DEVELOPMENT_MODE", "false")

    flag_options = {
        "global_developmentMode": False,
        "server_headless": True,
        "server_port": port,
        "server_address": "127.0.0.1",
        "browser_gatherUsageStats": False,
        "browser_serverPort": port,
        "browser_serverAddress": "127.0.0.1",
    }

    log_file = Path(log_path)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with log_file.open("a", encoding="utf-8") as f:
        with contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
            try:
                bootstrap.load_config_options(flag_options=flag_options)
                bootstrap.run(str(app_py), False, [], flag_options)
            except Exception:
                f.write(traceback.format_exc())
                f.write("\n")
                f.flush()
                raise


_LAYOUT_REFRESH_JS = r"""
(() => {
  const refreshTarget = (win) => {
    if (!win) return;
    try { win.dispatchEvent(new Event('resize')); } catch (e) {}
    try { win.dispatchEvent(new Event('orientationchange')); } catch (e) {}
    try { if (typeof win.scheduleLayoutResync === 'function') win.scheduleLayoutResync(); } catch (e) {}
    try { if (typeof win.queueFrameHeightSync === 'function') win.queueFrameHeightSync(); } catch (e) {}
    try { if (typeof win.syncFrameHeight === 'function') win.syncFrameHeight(); } catch (e) {}
  };
  refreshTarget(window);
  try {
    document.querySelectorAll('iframe').forEach((frame) => {
      try { refreshTarget(frame.contentWindow); } catch (e) {}
    });
  } catch (e) {}
  return true;
})();
"""


def _schedule_window_layout_refresh(window: "webview.Window", reason: str = "") -> None:
    # #region debug-point C:schedule-window-layout-refresh
    _debug_post("C", "desktop_launcher.py:_schedule_window_layout_refresh", "[DEBUG] shell layout refresh scheduled", {"reason": str(reason or "")})
    # #endregion
    def _run_once(delay_s: float) -> None:
        def _inner() -> None:
            try:
                # #region debug-point C:evaluate-layout-refresh-js
                _debug_post("C", "desktop_launcher.py:_schedule_window_layout_refresh", "[DEBUG] shell evaluate_js layout refresh", {"reason": str(reason or ""), "delay_s": float(delay_s)})
                # #endregion
                window.evaluate_js(_LAYOUT_REFRESH_JS)
            except Exception:
                pass

        timer = threading.Timer(delay_s, _inner)
        timer.daemon = True
        timer.start()

    for delay_s in (0.0, 0.12, 0.35, 0.8):
        _run_once(delay_s)


_LAST_NATIVE_NUDGE_AT = 0.0


def _maybe_native_resize_nudge(window: "webview.Window", reason: str = "") -> bool:
    global _LAST_NATIVE_NUDGE_AT
    reason_s = str(reason or "")
    if reason_s.startswith("resized:"):
        return False
    if not (reason_s.startswith("image-project-mount") or reason_s in {"shown", "loaded", "startup", "open-project"}):
        return False
    now = time.time()
    if (now - float(_LAST_NATIVE_NUDGE_AT or 0.0)) < 1.2:
        return False
    resize_fn = getattr(window, "resize", None)
    width = int(getattr(window, "width", 0) or 0)
    height = int(getattr(window, "height", 0) or 0)
    if not callable(resize_fn) or width < 300 or height < 300:
        # #region debug-point C:native-resize-nudge-skip
        _debug_post("C", "desktop_launcher.py:_maybe_native_resize_nudge", "[DEBUG] native resize nudge skipped", {"reason": reason_s, "width": width, "height": height, "has_resize": bool(callable(resize_fn))})
        # #endregion
        return False
    try:
        _LAST_NATIVE_NUDGE_AT = now
        # #region debug-point C:native-resize-nudge
        _debug_post("C", "desktop_launcher.py:_maybe_native_resize_nudge", "[DEBUG] native resize nudge start", {"reason": reason_s, "width": width, "height": height})
        # #endregion
        resize_fn(width + 1, height + 1)
        time.sleep(0.03)
        resize_fn(width, height)
        # #region debug-point C:native-resize-nudge-done
        _debug_post("C", "desktop_launcher.py:_maybe_native_resize_nudge", "[DEBUG] native resize nudge done", {"reason": reason_s, "width": width, "height": height})
        # #endregion
        return True
    except Exception as e:
        # #region debug-point C:native-resize-nudge-error
        _debug_post("C", "desktop_launcher.py:_maybe_native_resize_nudge", "[DEBUG] native resize nudge failed", {"reason": reason_s, "error": str(e)})
        # #endregion
        return False


class _ShellApi:
    def __init__(self) -> None:
        self.window = None

    def bind(self, window: "webview.Window") -> None:
        self.window = window

    def request_reflow(self, reason: str = "") -> bool:
        if self.window is None:
            return False
        # #region debug-point C:request-reflow
        _debug_post("C", "desktop_launcher.py:_ShellApi.request_reflow", "[DEBUG] shell request_reflow invoked", {"reason": str(reason or "")})
        # #endregion
        _schedule_window_layout_refresh(self.window, reason=reason)
        _maybe_native_resize_nudge(self.window, reason=reason)
        return True


def _startup_window_logic(window: "webview.Window") -> None:
    _schedule_window_layout_refresh(window, reason="startup")
    _maybe_native_resize_nudge(window, reason="startup")


def _run_headless(port: int, proc: multiprocessing.Process) -> None:
    # Announce readiness to the embedding supervisor (the AIoScout desktop app
    # reads stdout line-by-line). Everything the Streamlit server prints goes to
    # the log file, so this is the only line on our stdout.
    print(json.dumps({"type": "aioscout:ready", "port": int(port), "pid": os.getpid()}), flush=True)
    # If the embedding app is SIGKILLed we never see a signal — watch for
    # reparenting and self-exit (same watchdog the streamlit child uses).
    _exit_when_parent_dies()
    try:
        proc.join()  # blocks until the Streamlit child exits
    except KeyboardInterrupt:
        _shutdown_and_exit(proc)
    # If the child died on its own, exit with its code so a supervisor can
    # detect the crash and restart us.
    os._exit(proc.exitcode if proc.exitcode is not None else 1)


def main() -> None:
    parser = argparse.ArgumentParser(description="TF Lite Training app launcher")
    parser.add_argument(
        "--headless",
        action="store_true",
        help="run the Streamlit server without a native window; prints a JSON line "
        '{"type": "aioscout:ready", "port": ..., "pid": ...} to stdout once ready '
        "(for embedding in the AIoScout desktop app)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=0,
        help="fixed port for the Streamlit server (default: pick a free ephemeral port)",
    )
    args = parser.parse_args()

    multiprocessing.freeze_support()
    port = args.port if args.port > 0 else _find_free_port()
    url = f"http://127.0.0.1:{port}"
    log_file = (_app_data_dir() / "logs" / "streamlit.log").resolve()
    log_file.parent.mkdir(parents=True, exist_ok=True)
    log_file.touch(exist_ok=True)
    log_path = str(log_file)
    proc = multiprocessing.Process(target=_run_streamlit_server, args=(port, log_path), daemon=True)
    proc.start()
    # Catch SIGTERM (pkill, shutdown) so the streamlit child is reaped even
    # outside the window-close path. SIGKILL (Force Quit) cannot be caught;
    # the child-side watchdog in _exit_when_parent_dies covers that case.
    signal.signal(signal.SIGTERM, lambda *_: _shutdown_and_exit(proc))
    # Ctrl-C during development: reap the child the same way.
    signal.signal(signal.SIGINT, lambda *_: _shutdown_and_exit(proc))
    try:
        deadline = time.time() + _SERVER_READY_TIMEOUT_S
        last_err: Exception | None = None
        while time.time() < deadline:
            if not proc.is_alive():
                break
            try:
                with urllib.request.urlopen(url, timeout=2) as resp:
                    if 200 <= resp.status < 500:
                        last_err = None
                        break
            except Exception as e:
                last_err = e
            time.sleep(0.25)

        if last_err is not None:
            raise RuntimeError(f"Streamlit not ready: {url} ({last_err}). Log: {log_path}")
        if not proc.is_alive():
            raise RuntimeError(f"Streamlit process exited. Log: {log_path}")

        if args.headless:
            _run_headless(port, proc)
            return

        import webview

        shell_api = _ShellApi()
        window = webview.create_window("TF Lite Training", url, width=1200, height=800, js_api=shell_api)
        shell_api.bind(window)
        window.events.loaded += lambda: (_debug_post("C", "desktop_launcher.py:window.events.loaded", "[DEBUG] shell loaded event", {}), _schedule_window_layout_refresh(window, reason="loaded"))
        window.events.shown += lambda: (_debug_post("C", "desktop_launcher.py:window.events.shown", "[DEBUG] shell shown event", {}), _schedule_window_layout_refresh(window, reason="shown"))
        window.events.restored += lambda: (_debug_post("C", "desktop_launcher.py:window.events.restored", "[DEBUG] shell restored event", {}), _schedule_window_layout_refresh(window, reason="restored"))
        window.events.maximized += lambda: (_debug_post("C", "desktop_launcher.py:window.events.maximized", "[DEBUG] shell maximized event", {}), _schedule_window_layout_refresh(window, reason="maximized"))
        window.events.resized += lambda width, height: (_debug_post("C", "desktop_launcher.py:window.events.resized", "[DEBUG] shell resized event", {"width": int(width), "height": int(height)}), _schedule_window_layout_refresh(window, reason=f"resized:{width}x{height}"))
        window.events.closed += lambda: _shutdown_and_exit(proc)
        webview.start(_startup_window_logic, window)
    finally:
        if proc.is_alive():
            proc.terminate()
            proc.join(timeout=5)


def _shutdown_and_exit(proc: multiprocessing.Process) -> None:
    if proc.is_alive():
        proc.terminate()
        proc.join(timeout=5)
    os._exit(0)



if __name__ == "__main__":
    main()
