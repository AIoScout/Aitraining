from __future__ import annotations

import argparse
import contextlib
import json
import multiprocessing
import os
import signal
import socket
import sys
import tempfile
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


def _prepare_log_file() -> str:
    """Return a writable log path; fall back to /tmp if the app-data dir fails."""
    candidates = []
    try:
        candidates.append(str((_app_data_dir() / "logs" / "streamlit.log").resolve()))
    except Exception:
        pass
    candidates.append(str((Path(tempfile.gettempdir()) / "TFLiteTraining" / "logs" / "streamlit.log").resolve()))
    for log_path in candidates:
        try:
            p = Path(log_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.touch(exist_ok=True)
            return str(p)
        except Exception:
            continue
    # Absolute last resort — stderr
    return ""


def _configure_multiprocessing_executable() -> None:
    if sys.platform != "darwin":
        return
    if not getattr(sys, "frozen", False):
        return
    helper = Path(sys.executable).with_name("TFLiteTrainingConsole")
    if helper.exists():
        multiprocessing.set_executable(str(helper))


def _raise_fd_limit() -> None:
    target = 4096
    try:
        import resource

        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        desired_soft = max(soft, target)
        desired_hard = max(hard, desired_soft)
        try:
            resource.setrlimit(resource.RLIMIT_NOFILE, (desired_soft, desired_hard))
        except Exception:
            new_soft = min(desired_soft, hard)
            if new_soft != soft:
                resource.setrlimit(resource.RLIMIT_NOFILE, (new_soft, hard))
        return
    except Exception:
        pass

    try:
        import ctypes
        import ctypes.util

        libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)
        RLIMIT_NOFILE = 8

        class Rlimit(ctypes.Structure):
            _fields_ = [("rlim_cur", ctypes.c_uint64), ("rlim_max", ctypes.c_uint64)]

        rlim = Rlimit(0, 0)
        if libc.getrlimit(RLIMIT_NOFILE, ctypes.byref(rlim)) != 0:
            return
        rlim.rlim_cur = min(max(int(rlim.rlim_cur), target), int(rlim.rlim_max))
        libc.setrlimit(RLIMIT_NOFILE, ctypes.byref(rlim))
    except Exception:
        return


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

    _raise_fd_limit()

    from streamlit.web import bootstrap

    _exit_when_parent_dies()

    try:
        base_dir = _app_data_dir()
        base_dir.mkdir(parents=True, exist_ok=True)
        os.chdir(str(base_dir))
    except Exception:
        pass

    app_py = _resource_path("app.py")
    os.environ.setdefault("STREAMLIT_BROWSER_GATHER_USAGE_STATS", "false")
    os.environ.setdefault("STREAMLIT_GLOBAL_DEVELOPMENT_MODE", "false")
    os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "poll"

    flag_options = {
        "global_developmentMode": False,
        "server_headless": True,
        "server_fileWatcherType": "poll",
        "server_runOnSave": False,
        "server_port": port,
        "server_address": "127.0.0.1",
        "browser_gatherUsageStats": False,
        "browser_serverPort": port,
        "browser_serverAddress": "127.0.0.1",
    }

    if not log_path:
        log_path = _prepare_log_file()
    if log_path:
        log_file = Path(log_path)
    else:
        log_file = (Path(tempfile.gettempdir()) / "TFLiteTraining" / "logs" / "streamlit.log").resolve()
        log_file.parent.mkdir(parents=True, exist_ok=True)
        log_file.touch(exist_ok=True)
    import signal as _signal

    def _on_terminate_signal(signum, frame):
        # Window closed → the launcher calls proc.terminate() → SIGTERM.
        # The default handler would kill this process WITHOUT running
        # atexit, so the camera bridge would never release its
        # AVCaptureSessions (camera stays claimed, pthread trap on
        # finalization).  Release them here while the interpreter is fully
        # alive, then exit normally so every other atexit handler
        # (controller / server / serial cleanup) runs too.
        try:
            import mac_camera

            mac_camera.shutdown_all_caps(wait_s=0.2)
        except Exception:
            pass
        sys.exit(0)

    _signal.signal(_signal.SIGTERM, _on_terminate_signal)
    _signal.signal(_signal.SIGINT, _on_terminate_signal)

    with log_file.open("a", encoding="utf-8") as f:
        try:
            import resource

            f.write(f"[TFLiteTraining] RLIMIT_NOFILE={resource.getrlimit(resource.RLIMIT_NOFILE)}\n")
            f.flush()
        except Exception:
            pass
        with contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
            try:
                bootstrap.load_config_options(flag_options=flag_options)
                try:
                    from streamlit import config as st_config

                    f.write(f"[TFLiteTraining] server.fileWatcherType={st_config.get_option('server.fileWatcherType')}\n")
                    f.flush()
                except Exception:
                    pass
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
    def _run_once(delay_s: float) -> None:
        def _inner() -> None:
            try:
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
        return False
    try:
        _LAST_NATIVE_NUDGE_AT = now
        resize_fn(width + 1, height + 1)
        time.sleep(0.03)
        resize_fn(width, height)
        return True
    except Exception as e:
        return False


class _ShellApi:
    def __init__(self) -> None:
        self.window = None

    def bind(self, window: "webview.Window") -> None:
        self.window = window

    def request_reflow(self, reason: str = "") -> bool:
        if self.window is None:
            return False
        _schedule_window_layout_refresh(self.window, reason=reason)
        _maybe_native_resize_nudge(self.window, reason=reason)
        return True

    def pick_upload_files(self, base_url: str, session: str, class_name: str) -> list:
        """Native file picker for the SPA upload button.

        The SPA's <input type="file"> does not reliably open a dialog inside
        pywebview on macOS, so the SPA calls this JS API instead: open the
        native open-file dialog, read the chosen images, and POST them to
        the local controller's /upload endpoint.

        Returns a list of per-file results:
          {"filename": str, "saved_filename": str, "ok": bool, "error": str, "thumb_b64": str}
        so the SPA can show a detailed summary instead of a single success toast.
        """
        import base64 as _b64
        import pathlib as _pl
        import urllib.error as _ue

        results = []
        try:
            import webview as _webview

            paths = _webview.create_file_dialog(
                _webview.OPEN_DIALOG,
                allow_multiple=True,
                file_types=("Image files (*.png;*.jpg;*.jpeg;*.bmp;*.gif;*.webp)",),
            )
            if not paths:
                return results
            base_url = str(base_url or "").strip()
            session = str(session or "").strip()
            class_name = str(class_name or "").strip()
            if not base_url or not session or not class_name:
                results.append({"filename": "", "saved_filename": "", "ok": False, "error": "missing upload context (reload page)", "thumb_b64": ""})
                return results
            endpoint = f"{base_url}/upload"
            for p in paths:
                fpath = _pl.Path(str(p))
                fname = fpath.name
                saved = ""
                try:
                    data = fpath.read_bytes()
                except Exception as e:
                    results.append({"filename": fname, "saved_filename": "", "ok": False, "error": f"read failed: {e}", "thumb_b64": ""})
                    continue
                payload = {
                    "session": session,
                    "class": class_name,
                    "image_b64": _b64.b64encode(data).decode("ascii"),
                    "filename": fname,
                }
                ok = False
                err = ""
                thumb = ""
                try:
                    req = urllib.request.Request(
                        endpoint,
                        data=json.dumps(payload).encode("utf-8"),
                        headers={"Content-Type": "application/json", "Accept": "application/json"},
                    )
                    with urllib.request.urlopen(req, timeout=45) as resp:
                        raw = resp.read()
                        body = json.loads(raw.decode("utf-8") or "{}")
                    ok = str(body.get("ok") or "0") == "1"
                    err = "" if ok else str(body.get("error") or "upload rejected")
                    thumb = str(body.get("thumb_b64") or "")
                    saved = str(body.get("filename") or "")
                except _ue.HTTPError as he:
                    err_body = ""
                    try:
                        raw = he.read()
                        err_body = json.loads(raw.decode("utf-8") or "{}").get("error", "")
                    except Exception:
                        err_body = ""
                    ok = False
                    err = f"server HTTP {he.code}: {err_body or he.reason}"
                    thumb = ""
                except _ue.URLError as ue:
                    ok = False
                    err = f"network unreachable: {ue.reason}"
                    thumb = ""
                except Exception as e:
                    ok = False
                    err = f"upload failed: {e}"
                    thumb = ""
                results.append({
                    "filename": fname,
                    "saved_filename": saved,
                    "ok": bool(ok),
                    "error": str(err or ""),
                    "thumb_b64": thumb if ok else "",
                })
        except Exception as e:
            results.append({"filename": "", "saved_filename": "", "ok": False, "error": f"picker failed: {e}", "thumb_b64": ""})
        return results

    def pick_single_image_file(self) -> dict:
        """Native single-image picker for the preview Upload-File source."""
        import base64 as _b64
        import pathlib as _pl
        import mimetypes as _mt

        try:
            import webview as _webview

            paths = _webview.create_file_dialog(
                _webview.OPEN_DIALOG,
                allow_multiple=False,
                file_types=("Image files (*.png;*.jpg;*.jpeg;*.bmp;*.gif;*.webp)",),
            )
            if not paths:
                return {}
            fpath = _pl.Path(str(paths[0]))
            data = fpath.read_bytes()
            mime = _mt.guess_type(fpath.name)[0] or "image/png"
            return {"name": fpath.name, "mime": mime, "b64": _b64.b64encode(data).decode("ascii")}
        except Exception:
            return {}


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
    _configure_multiprocessing_executable()
    try:
        multiprocessing.set_start_method("spawn", force=True)
    except Exception:
        pass
    _raise_fd_limit()

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
    log_path = _prepare_log_file()
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
        window.events.loaded += lambda: _schedule_window_layout_refresh(window, reason="loaded")
        window.events.shown += lambda: _schedule_window_layout_refresh(window, reason="shown")
        window.events.restored += lambda: _schedule_window_layout_refresh(window, reason="restored")
        window.events.maximized += lambda: _schedule_window_layout_refresh(window, reason="maximized")
        window.events.resized += lambda width, height: _schedule_window_layout_refresh(window, reason=f"resized:{width}x{height}")
        window.events.closed += lambda: _shutdown_and_exit(proc)
        webview.start(_startup_window_logic, window)
    finally:
        if proc.is_alive():
            proc.terminate()
            proc.join(timeout=5)


def _shutdown_and_exit(proc: multiprocessing.Process) -> None:
    if proc.is_alive():
        # SIGTERM → the child's handler releases the camera sessions and
        # exits gracefully (its atexit handlers run).
        proc.terminate()
        proc.join(timeout=5)
    if proc.is_alive():
        # The child ignored/hung on SIGTERM (e.g. blocked in a camera read).
        # SIGKILL it so it can never outlive the window and keep the camera
        # claimed — the OS reclaims the capture hardware on process death.
        try:
            proc.kill()
        except Exception:
            pass
        proc.join(timeout=3)
    os._exit(0)



if __name__ == "__main__":
    main()
