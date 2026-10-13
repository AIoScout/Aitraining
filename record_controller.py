from __future__ import annotations

import base64
import json
import math
import shutil
import socket
import subprocess
import sys
import threading
import time
import uuid
import zipfile
from dataclasses import dataclass, replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import parse_qs, urlparse


def _ensure_bundle_sys_path() -> None:
    """Make sibling .py files importable when RecordController runs inside a
    PyInstaller one-folder / one-file bundle on macOS.

    Streamlit launches us via subprocess from a different cwd. Tf the bundle
    is built with datas=(('mac_camera.py', '.'), ...), the helper file ends up
    next to this script in the frozen app root but NOT on sys.path. We add
    the obvious candidate directories once during import so
    ``from mac_camera import ...`` succeeds without students editing anything.
    """
    roots: List[Path] = []
    try:
        this_file = Path(__file__).resolve()
        if this_file.exists():
            roots.append(this_file.parent)
    except Exception:
        pass
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        try:
            roots.append(Path(str(meipass)))
        except Exception:
            pass
    try:
        roots.append(Path.cwd())
    except Exception:
        pass
    for root in roots:
        if not isinstance(root, Path):
            continue
        try:
            root_str = str(root)
        except Exception:
            continue
        if root_str and root_str not in sys.path:
            sys.path.insert(0, root_str)


_ensure_bundle_sys_path()
del _ensure_bundle_sys_path

import numpy as np
from PIL import Image, ImageOps

from camera_permission import ensure_camera_access
from serial_device import SerialFrameReader, list_serial_ports, parse_sync_header
from dataset_io import IMAGE_EXTS, sanitize_class_name
from image_preprocess import (
    CROP_MODE_AUTO_SEARCH,
    CROP_MODE_CENTER,
    CLIP_MODE_JUNCTION,
    PREPROCESS_MODE_AUTO_BY_LABEL,
    PREPROCESS_MODE_MANUAL_ROI,
    _BG_LUM_THRESH,
    _BG_DIFF_MIN,
    class_clip_mode,
    normalize_class_preprocess,
    normalize_class_preprocess_map,
    normalize_sample_preprocess_map,
    normalize_manual_roi,
    manual_roi_to_pixels,
    preprocess_blue_diff_array,
    prepare_inference_inputs,
)

_APP_NAME = "TFLiteTraining"


def _app_data_dir() -> Path:
    """Same resolution logic as app.py/desktop_launcher.py (kept in sync)."""
    import os

    env_override = os.getenv("TFLITE_TRAINING_DATA_DIR")
    if env_override:
        return Path(env_override).expanduser().resolve()
    if sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support"
    elif os.name == "nt":
        base = Path(os.environ.get("APPDATA", str(Path.home() / "AppData" / "Roaming")))
    else:
        base = Path(os.environ.get("XDG_DATA_HOME", str(Path.home() / ".local" / "share")))
    return (base / _APP_NAME).resolve()


@dataclass
class SessionConfig:
    dataset_root: Path
    serial_port: str
    serial_baud: int
    serial_sync: str
    serial_frame_side: int
    serial_channels: int
    webcam_index: int
    fps: float
    crop_box: Optional[Tuple[int, int, int, int]]
    # When True, a 1-channel device stream is fed STRAIGHT to the model
    # Legacy: older firmware (kCaptureGray/kInferGray) streamed the
    # preprocessed TENSOR.  Current firmware streams the RAW library gray
    # (BT.601, uncropped) in ALL capture modes, so this must stay False —
    # the host runs the full crop pipeline exactly like training did.
    # Direct-feeding a raw frame into a crop-trained model was the
    # "model always wrong" bug.
    direct_tensor_stream: bool = False
    # macOS AVCaptureDevice uniqueID of the selected camera.  On macOS this
    # is AUTHORITATIVE: the camera is opened by uniqueID (mac_camera bridge),
    # never by positional index — cv2's positional order is NOT the
    # AVFoundation array order (cv2 sorts by uniqueID), so index-based opens
    # silently select the wrong device whenever virtual cameras (Iriun etc.)
    # are installed.  webcam_index is only a UI hint / non-macOS fallback.
    webcam_id: Optional[str] = None
    # Training image size (the "Image Size" hyperparameter).  Captured /
    # uploaded samples and their processed-cache previews are stored at this
    # size so 160×160 (or 48/128/192) projects are never downsampled to 96.
    img_size: int = 96


def _persist_webcam_choice(cfg: "SessionConfig") -> None:
    """Remember the last-used camera (uniqueID + index) in the workspace so a
    fresh session auto-selects the user's camera instead of the first
    non-virtual one."""
    try:
        p = Path(cfg.dataset_root).parent / "webcam.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(
            json.dumps(
                {
                    "webcam_id": str(getattr(cfg, "webcam_id", None) or ""),
                    "webcam_index": int(getattr(cfg, "webcam_index", 0) or 0),
                },
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
    except Exception:
        pass


class ExportConflictError(RuntimeError):
    def __init__(self, conflicts: List[str]) -> None:
        self.conflicts = list(conflicts)
        preview = ", ".join(self.conflicts[:5])
        if len(self.conflicts) > 5:
            preview += ", ..."
        super().__init__(f"Export will overwrite existing files: {preview}")


def _is_truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value or "").strip().lower()
    return text in {"1", "true", "yes", "on"}


_TFLITE_OP_RESOLVER_MAP = {
    "ADD": "Add",
    "AVERAGE_POOL_2D": "AveragePool2D",
    "CONV_2D": "Conv2D",
    "DEPTHWISE_CONV_2D": "DepthwiseConv2D",
    "FULLY_CONNECTED": "FullyConnected",
    "MAX_POOL_2D": "MaxPool2D",
    "MUL": "Mul",
    "PACK": "Pack",
    "PAD": "Pad",
    "QUANTIZE": "Quantize",
    "DEQUANTIZE": "Dequantize",
    "RESHAPE": "Reshape",
    "SHAPE": "Shape",
    "SOFTMAX": "Softmax",
    "STRIDED_SLICE": "StridedSlice",
}


def _generate_model_resolver_h(tflite_bytes: bytes) -> str:
    """Generate model_resolver.h matching the ops in the exported .tflite."""
    ops = []
    try:
        import tensorflow as tf
        interp = tf.lite.Interpreter(model_content=tflite_bytes)
        interp.allocate_tensors()
        for detail in interp._get_ops_details():
            op = str(detail.get("op_name") or "")
            if op and op not in ops:
                ops.append(op)
    except Exception:
        pass
    if not ops:
        ops = ["CONV_2D", "MAX_POOL_2D", "SHAPE", "STRIDED_SLICE", "PACK", "RESHAPE", "FULLY_CONNECTED", "SOFTMAX"]
    resolved = [op for op in ops if op in _TFLITE_OP_RESOLVER_MAP]
    if not resolved:
        resolved = ["CONV_2D", "FULLY_CONNECTED", "SOFTMAX"]
    n = len(resolved)
    lines = [
        "/*",
        "Auto-generated by TFLiteTraining export.",
        "Derived from the exported .tflite builtin operator table.",
        "*/",
        "",
        "#ifndef TFLITE_MODEL_RESOLVER_H_",
        "#define TFLITE_MODEL_RESOLVER_H_",
        "",
        '#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"',
        "",
        f"using ModelOpResolver = tflite::MicroMutableOpResolver<{n}>;",
        "",
        "inline TfLiteStatus RegisterModelOps(ModelOpResolver& resolver) {",
    ]
    for op in resolved:
        add_fn = _TFLITE_OP_RESOLVER_MAP[op]
        lines.append(f"  if (resolver.Add{add_fn}() != kTfLiteOk) return kTfLiteError;")
    lines.append("  return kTfLiteOk;")
    lines.append("}")
    lines.append("")
    lines.append("#endif  // TFLITE_MODEL_RESOLVER_H_")
    lines.append("")
    return "\n".join(lines)


# Live-worker liveness contract (see _live_running / _get_live_preview):
# - LIVE_IDLE_S: a worker that NO consumer has polled for this long is
#   reaped (releases the camera; never while a hold capture is active).
#   last_touch is written by consumers only — NOT by the producer.
# - LIVE_STALE_S: frames older than this are treated as stale; /live/frame
#   then revives a DEAD producer instead of serving the last frame forever.
LIVE_IDLE_S = 15.0
LIVE_STALE_S = 1.5


class RecordController:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._configs: Dict[str, SessionConfig] = {}
        self._active: Dict[str, Dict[str, Any]] = {}
        self._live: Dict[str, Dict[str, Any]] = {}
        self._live_threads: Dict[str, threading.Thread] = {}
        self._live_stops: Dict[str, threading.Event] = {}
        self._record_threads: Dict[str, threading.Thread] = {}
        self._record_conds: Dict[str, threading.Condition] = {}
        self._train: Dict[str, Dict[str, Any]] = {}
        self._train_conds: Dict[str, threading.Condition] = {}
        self._preview: Dict[str, Dict[str, Any]] = {}
        self._preview_locks: Dict[str, threading.Lock] = {}
        self._server: Optional[ThreadingHTTPServer] = None
        self._port: Optional[int] = None
        self._thread: Optional[threading.Thread] = None

    @property
    def port(self) -> int:
        if self._port is None:
            raise RuntimeError("Controller server not started")
        return self._port

    def stop(self) -> None:
        """Shut the HTTP server down and join its thread (hot-reload support)."""
        with self._lock:
            server = self._server
            thread = self._thread
            self._server = None
            self._thread = None
            self._port = None
        if server is not None:
            try:
                server.shutdown()
                server.server_close()
            except Exception:
                pass
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=2.0)
        # Live workers hold AVCaptureSessions whose sample-buffer callbacks
        # run on libdispatch threads; if those sessions are still streaming
        # when the interpreter starts finalizing, the callback's GIL acquire
        # calls pthread_exit() on a dispatch-owned thread and macOS kills the
        # process ("BUG IN CLIENT OF LIBPTHREAD").  Release them here, while
        # the interpreter is fully alive.
        try:
            self._stop_live_all()
        except Exception:
            pass

    def _stop_live_all(self) -> None:
        """Stop every live worker of every session, then release every open
        AVCaptureSession held by the macOS camera bridge (hot-reload /
        shutdown safety)."""
        with self._lock:
            keys = list(self._live.keys())
        for key in keys:
            try:
                session_id, source = str(key).split(":", 1)
            except Exception:
                continue
            try:
                self._stop_live(session_id=session_id, source=source, wait=True, timeout_s=1.0)
            except Exception:
                continue
        try:
            from mac_camera import shutdown_all_caps

            shutdown_all_caps()
        except Exception:
            pass

    def start(self) -> None:
        with self._lock:
            if self._server is not None:
                return
            host = "127.0.0.1"
            port = _find_free_port(host)
            server = ThreadingHTTPServer((host, port), self._make_handler())
            self._server = server
            self._port = int(port)
            t = threading.Thread(target=server.serve_forever, daemon=True)
            self._thread = t
            t.start()
            # Publish the port so the embedding desktop app can reach this
            # API directly (the Streamlit port it knows is a different server).
            try:
                port_file = _app_data_dir() / "record_controller_port.json"
                port_file.parent.mkdir(parents=True, exist_ok=True)
                port_file.write_text(json.dumps({"port": self._port}), encoding="utf-8")
            except Exception:
                pass

    def set_config(self, session_id: str, cfg: SessionConfig) -> None:
        with self._lock:
            self._configs[session_id] = cfg

    def update_config(
        self,
        session_id: str,
        *,
        webcam_index: Optional[int] = None,
        webcam_id: Optional[str] = None,
        img_size: Optional[int] = None,
        serial_port: Optional[str] = None,
        serial_baud: Optional[int] = None,
        serial_sync: Optional[str] = None,
        serial_frame_side: Optional[int] = None,
        serial_channels: Optional[int] = None,
        direct_tensor_stream: Optional[bool] = None,
    ) -> SessionConfig:
        with self._lock:
            cfg = self._configs.get(session_id)
            if cfg is None:
                raise RuntimeError("missing config")
            updated = replace(
                cfg,
                webcam_index=int(cfg.webcam_index if webcam_index is None else webcam_index),
                webcam_id=(str(cfg.webcam_id) if cfg.webcam_id else "") if webcam_id is None else str(webcam_id),
                img_size=int(cfg.img_size if img_size is None else img_size),
                serial_port=str(cfg.serial_port if serial_port is None else serial_port),
                serial_baud=int(cfg.serial_baud if serial_baud is None else serial_baud),
                serial_sync=str(cfg.serial_sync if serial_sync is None else serial_sync),
                serial_frame_side=int(cfg.serial_frame_side if serial_frame_side is None else serial_frame_side),
                serial_channels=int(cfg.serial_channels if serial_channels is None else serial_channels),
            )
            self._configs[session_id] = updated
            _persist_webcam_choice(updated)
            return updated

    def get_config(self, session_id: str) -> Optional["SessionConfig"]:
        with self._lock:
            return self._configs.get(session_id)

    def status(self, session_id: str) -> Dict[str, Any]:
        with self._lock:
            return dict(self._active.get(session_id, {}))

    def preview_webcam_png(self, webcam_index: int, webcam_id: Optional[str] = None, session_id: Optional[str] = None) -> Optional[bytes]:
        uid = str(webcam_id or "").strip()
        # Fast path: when the session's live worker already owns the selected
        # camera, serve its newest preview frame instead of opening a SECOND
        # capture session (macOS allows a single consumer per camera).
        if session_id and uid:
            with self._lock:
                state = self._live.get(self._live_key(session_id, "webcam"))
                if state is not None and state.get("running"):
                    actual = state.get("actual") or {}
                    if str(actual.get("unique_id") or "") == uid:
                        data = state.get("preview_png")
                        if isinstance(data, (bytes, bytearray)):
                            return bytes(data)
        permission = ensure_camera_access(webcam_index=int(webcam_index or 0), probe_open=False, unique_id=uid)
        if not permission.allowed:
            return None
        try:
            cap, _actual = _open_working_camera(int(webcam_index or 0), unique_id=uid)
        except Exception:
            return None
        if cap is None or not cap.isOpened():
            return None
        ok, frame = cap.read()
        cap.release()
        if not ok or frame is None:
            return None
        import cv2
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame)
        return _to_png_bytes(img)

    def preview_serial_png(self, port: str, baud: int, sync_header: str = "", frame_side: int = 96, channels: int = 1) -> Optional[bytes]:
        if not port:
            return None
        try:
            reader = SerialFrameReader(port=port, baud=int(baud), sync_header=sync_header, frame_side=int(frame_side), channels=int(channels))
            reader.open()
            try:
                raw = reader.read_frame()  # auto-scaled timeout (frame_size × baud)
            finally:
                reader.close()
            side = int(frame_side)
            ch = int(channels)
            if ch == 3:
                arr = np.frombuffer(raw, dtype=np.uint8).reshape((side, side, 3))
                img = Image.fromarray(arr, mode="RGB")
            else:
                arr = np.frombuffer(raw, dtype=np.uint8).reshape((side, side))
                img = Image.fromarray(arr, mode="L")
            return _to_png_bytes(img)
        except Exception:
            return None

    def _make_handler(self):
        controller = self

        class Handler(BaseHTTPRequestHandler):
            @staticmethod
            def _error_body_write(handler: BaseHTTPRequestHandler) -> None:
                try:
                    handler.send_response(500)
                    handler.send_header("Content-Type", "text/plain")
                    handler.end_headers()
                    handler.wfile.write(b"error")
                except (BrokenPipeError, ConnectionResetError, OSError):
                    pass

            def do_GET(self) -> None:
                try:
                    controller._handle(self)
                except Exception:
                    self._error_body_write(self)

            def do_POST(self) -> None:
                try:
                    controller._handle_post(self)
                except Exception:
                    self._error_body_write(self)

            def do_OPTIONS(self) -> None:
                try:
                    controller._handle_options(self)
                except (BrokenPipeError, ConnectionResetError, OSError):
                    pass

            def log_message(self, format: str, *args) -> None:
                return

        return Handler

    def _handle(self, req: BaseHTTPRequestHandler) -> None:
        parsed = urlparse(req.path)
        qs = parse_qs(parsed.query)
        path = parsed.path
        if path == "/status":
            session_id = (qs.get("session") or [""])[0]
            data = self.status(session_id)
            _send_json(req, data, cors=True)
            return
        if path == "/start":
            session_id = (qs.get("session") or [""])[0]
            source = (qs.get("source") or [""])[0]
            class_name = (qs.get("class") or [""])[0]
            if not session_id or not source or not class_name:
                _send_json(req, {"ok": "0", "error": "missing params"}, status=400, cors=True)
                return
            self._start_record(session_id=session_id, source=source, class_name=class_name)
            _send_json(req, {"ok": "1"}, cors=True)
            return
        if path == "/stop":
            session_id = (qs.get("session") or [""])[0]
            self._stop_record(session_id=session_id)
            _send_json(req, {"ok": "1"}, cors=True)
            return
        if path == "/record/next":
            session_id = (qs.get("session") or [""])[0]
            since_raw = (qs.get("since") or [""])[0]
            try:
                since = int(since_raw) if since_raw else 0
            except Exception:
                since = 0
            payload = self._wait_next_record(session_id=session_id, since=since, timeout_s=10.0)
            _send_json(req, payload, cors=True)
            return
        if path == "/train/status":
            session_id = (qs.get("session") or [""])[0]
            payload = self._train_status_payload(session_id=session_id)
            _send_json(req, payload, cors=True)
            return
        if path == "/class_state":
            session_id = (qs.get("session") or [""])[0]
            class_name = (qs.get("class") or [""])[0]
            limit_raw = (qs.get("limit") or [""])[0]
            if not session_id or not class_name:
                _send_json(req, {"ok": "0", "error": "missing params"}, status=400, cors=True)
                return
            cfg = self._configs.get(session_id)
            if cfg is None:
                _send_json(req, {"ok": "0", "error": "missing config"}, status=400, cors=True)
                return
            try:
                limit = int(limit_raw) if str(limit_raw).strip() else 12
            except Exception:
                limit = 12
            _send_json(req, self._class_state_payload(cfg.dataset_root, class_name, limit=None if limit <= 0 else limit), cors=True)
            return
        if path == "/class_filenames":
            session_id = (qs.get("session") or [""])[0]
            class_name = (qs.get("class") or [""])[0]
            if not session_id or not class_name:
                _send_json(req, {"ok": "0", "error": "missing params"}, status=400, cors=True)
                return
            cfg = self._configs.get(session_id)
            if cfg is None:
                _send_json(req, {"ok": "0", "error": "missing config"}, status=400, cors=True)
                return
            _send_json(req, self._class_filenames_payload(cfg.dataset_root, class_name), cors=True)
            return
        if path == "/sample_image":
            session_id = (qs.get("session") or [""])[0]
            class_name = (qs.get("class") or [""])[0]
            filename = (qs.get("filename") or [""])[0]
            if not session_id or not class_name or not filename:
                _send_json(req, {"ok": "0", "error": "missing params"}, status=400, cors=True)
                return
            cfg = self._configs.get(session_id)
            if cfg is None:
                _send_json(req, {"ok": "0", "error": "missing config"}, status=400, cors=True)
                return
            try:
                png = self._sample_image_png(cfg.dataset_root, class_name, filename)
            except FileNotFoundError:
                _send_json(req, {"ok": "0", "error": "sample not found"}, status=404, cors=True)
                return
            # Sample filenames are immutable in practice, so allow browser-side
            # caching to avoid re-downloading the same thumbnail on every re-render.
            _send_png(req, png, cors=True, cache_control="private, max-age=86400, immutable")
            return
        if path == "/preview":
            session_id = (qs.get("session") or [""])[0]
            source = (qs.get("source") or [""])[0]
            png = self._preview_frame(session_id=session_id, source=source)
            if png is None:
                _send_json(req, {"ok": "0", "error": "preview unavailable"}, status=404, cors=True)
                return
            _send_png(req, png, cors=True)
            return
        if path == "/webcam/options":
            try:
                opts = list_webcam_options()
            except Exception:
                opts = []
            _send_json(req, {"ok": "1", "options": opts}, cors=True)
            return
        if path == "/serial/ports":
            ports = []
            try:
                blocked = {
                    "/dev/cu.bluetooth-incoming-port",
                    "/dev/cu.debug-console",
                    "/dev/tty.bluetooth-incoming-port",
                    "/dev/tty.debug-console",
                }
                for p in list_serial_ports():
                    dev = str(p.device or "")
                    desc = str(getattr(p, "description", "") or "")
                    d_low = dev.strip().lower()
                    if d_low in blocked:
                        continue
                    if "bluetooth" in desc.lower() and "serial" not in desc.lower():
                        continue
                    label = f"{dev} - {desc}" if desc else dev
                    ports.append({"device": dev, "label": str(label)})
            except Exception as e:
                _send_json(req, {"ok": "0", "error": str(e)}, status=400, cors=True)
                return
            _send_json(req, {"ok": "1", "ports": ports}, cors=True)
            return
        if path == "/capture":
            session_id = (qs.get("session") or [""])[0]
            source = (qs.get("source") or [""])[0]
            class_name = (qs.get("class") or [""])[0]
            if not session_id or not source or not class_name:
                _send_json(req, {"ok": "0", "error": "missing params"}, status=400, cors=True)
                return
            png = self._capture_single(session_id=session_id, source=source)
            if png is None:
                _send_json(req, {"ok": "0", "error": "capture unavailable"}, status=400, cors=True)
                return
            cfg = self._configs.get(session_id)
            if cfg is None:
                _send_json(req, {"ok": "0", "error": "missing config"}, status=400, cors=True)
                return
            p = _save_png(cfg.dataset_root, class_name, png)
            self._cache_one_sample(cfg.dataset_root, class_name, p, img_size=int(cfg.img_size))
            _send_json(req, {"ok": "1", "class": class_name, "filename": p.name}, cors=True)
            return
        if path == "/live/open":
            session_id = (qs.get("session") or [""])[0]
            source = (qs.get("source") or [""])[0]
            if not session_id or not source:
                _send_json(req, {"ok": "0", "error": "missing params"}, status=400, cors=True)
                return
            # Apply any per-request overrides before opening the stream
            frame_side_raw = (qs.get("frame_side") or [None])[0]
            channels_raw = (qs.get("channels") or [None])[0]
            if frame_side_raw not in (None, "") or channels_raw not in (None, ""):
                kwargs = {}
                if frame_side_raw not in (None, ""):
                    kwargs["serial_frame_side"] = int(float(frame_side_raw))
                if channels_raw not in (None, ""):
                    kwargs["serial_channels"] = int(float(channels_raw))
                self.update_config(session_id, **kwargs)
            self._ensure_live(session_id=session_id, source=source)
            _send_json(req, {"ok": "1"}, cors=True)
            return
        if path == "/live/config":
            session_id = (qs.get("session") or [""])[0]
            if not session_id:
                _send_json(req, {"ok": "0", "error": "missing session"}, status=400, cors=True)
                return
            webcam_index_raw = (qs.get("webcam_index") or [None])[0]
            webcam_id_raw = (qs.get("webcam_id") or [None])[0]
            img_size_raw = (qs.get("img_size") or [None])[0]
            serial_port_raw = (qs.get("serial_port") or [None])[0]
            serial_baud_raw = (qs.get("serial_baud") or [None])[0]
            serial_sync_raw = (qs.get("serial_sync") or [None])[0]
            serial_frame_side_raw = (qs.get("serial_frame_side") or [None])[0]
            serial_channels_raw = (qs.get("serial_channels") or [None])[0]
            direct_tensor_raw = (qs.get("direct_tensor_stream") or [None])[0]
            try:
                if serial_sync_raw not in (None, ""):
                    parse_sync_header(serial_sync_raw)
                if serial_frame_side_raw not in (None, ""):
                    side = int(float(serial_frame_side_raw))
                    if side < 8 or side > 512:
                        raise ValueError("invalid serial_frame_side")
                if serial_channels_raw not in (None, ""):
                    ch = int(float(serial_channels_raw))
                    if ch not in (1, 3):
                        raise ValueError("serial_channels must be 1 or 3")
                cfg = self.update_config(
                    session_id,
                    webcam_index=None if webcam_index_raw in (None, "") else int(float(webcam_index_raw)),
                    webcam_id=None if webcam_id_raw in (None, "") else str(webcam_id_raw),
                    img_size=None if img_size_raw in (None, "") else int(float(img_size_raw)),
                    serial_port=serial_port_raw,
                    serial_baud=None if serial_baud_raw in (None, "") else int(float(serial_baud_raw)),
                    serial_sync=None if serial_sync_raw is None else str(serial_sync_raw),
                    serial_frame_side=None if serial_frame_side_raw in (None, "") else int(float(serial_frame_side_raw)),
                    serial_channels=None if serial_channels_raw in (None, "") else int(float(serial_channels_raw)),
                    direct_tensor_stream=None if direct_tensor_raw in (None, "") else direct_tensor_raw in ("1", "true", "True"),
                )
            except Exception as e:
                _send_json(req, {"ok": "0", "error": str(e)}, status=400, cors=True)
                return
            with self._lock:
                wc_state = self._live.get(self._live_key(session_id, "webcam"))
            wc_actual = dict((wc_state or {}).get("actual") or {})
            _send_json(
                req,
                {
                    "ok": "1",
                    "direct_tensor_stream": "1" if bool(getattr(cfg, "direct_tensor_stream", False)) else "0",
                    "webcam_index": str(int(cfg.webcam_index)),
                    "webcam_id": str(cfg.webcam_id or ""),
                    "webcam_actual_backend": str(wc_actual.get("backend") or ""),
                    "webcam_actual_uid": str(wc_actual.get("unique_id") or ""),
                    "img_size": str(int(cfg.img_size)),
                    "serial_port": str(cfg.serial_port),
                    "serial_baud": str(int(cfg.serial_baud)),
                    "serial_sync": str(cfg.serial_sync),
                    "serial_frame_side": str(int(cfg.serial_frame_side)),
                    "serial_channels": str(int(cfg.serial_channels)),
                },
                cors=True,
            )
            return
        if path == "/live/frame":
            session_id = (qs.get("session") or [""])[0]
            source = (qs.get("source") or [""])[0]
            png = self._get_live_preview(session_id=session_id, source=source)
            if png is None:
                err = self._get_live_error(session_id=session_id, source=source) or "preview unavailable"
                # starting=1 tells the SPA the producer is (re)starting and a
                # frame will follow — it keeps the last good frame on screen
                # instead of blanking, so transient stalls never read as a
                # frozen/black view.
                starting = "1" if self._is_live_starting(session_id=session_id, source=source) else "0"
                _send_json(req, {"ok": "0", "error": err, "starting": starting}, status=404, cors=True)
                return
            _send_png(req, png, cors=True)
            return
        if path == "/preview/predict":
            session_id = (qs.get("session") or [""])[0]
            source = (qs.get("source") or [""])[0]
            preprocess_mode = (qs.get("preprocess") or [""])[0]
            bg_dark_raw = (qs.get("bg_dark") or [""])[0]
            bg_lum_raw  = (qs.get("bg_lum") or [""])[0]
            ood_sign_min_raw = (qs.get("ood_sign_min") or [""])[0]
            ood_sign_max_raw = (qs.get("ood_sign_max") or [""])[0]
            ood_max_prob_raw = (qs.get("ood_max_prob") or [""])[0]
            ood_entropy_max_raw = (qs.get("ood_entropy_max") or [""])[0]
            bg_dark_thresh = int(bg_dark_raw) if str(bg_dark_raw).strip() != "" else None
            bg_lum_thresh  = int(bg_lum_raw)  if str(bg_lum_raw).strip()  != "" else None
            ood_sign_pct_min = float(ood_sign_min_raw) if str(ood_sign_min_raw).strip() != "" else None
            ood_sign_pct_max = float(ood_sign_max_raw) if str(ood_sign_max_raw).strip() != "" else None
            ood_max_prob_min = float(ood_max_prob_raw) if str(ood_max_prob_raw).strip() != "" else None
            ood_entropy_ratio_max = float(ood_entropy_max_raw) if str(ood_entropy_max_raw).strip() != "" else None
            if not session_id or source not in {"webcam", "device"}:
                _send_json(req, {"ok": "0", "error": "missing params"}, status=400, cors=True)
                return
            try:
                self._ensure_live(session_id=session_id, source=source)
                preview_png = self._get_live_preview(session_id=session_id, source=source)
                if preview_png is None:
                    err = self._get_live_error(session_id=session_id, source=source) or "preview unavailable"
                    _send_json(req, {"ok": "0", "error": err}, status=404, cors=True)
                    return
                # Predict using the 96x96 capture_png when possible — this is
                # byte-for-byte the same file the /start (hold capture) worker
                # saves on disk, so the user sees a distribution exactly equal
                # to what their downstream MCU classifier will ingest.
                capture_png = self._get_live_capture(session_id=session_id, source=source)
                predict_png = capture_png if capture_png else preview_png
                pred = self._preview_predict(
                    session_id=session_id,
                    png=predict_png,
                    preprocess_mode=preprocess_mode,
                    bg_dark_thresh=bg_dark_thresh,
                    bg_lum_thresh=bg_lum_thresh,
                    ood_sign_pct_min=ood_sign_pct_min,
                    ood_sign_pct_max=ood_sign_pct_max,
                    ood_max_prob_min=ood_max_prob_min,
                    ood_entropy_ratio_max=ood_entropy_ratio_max,
                    source=source,
                )
            except Exception as e:
                _send_json(req, {"ok": "0", "error": str(e)}, status=400, cors=True)
                return
            _send_json(
                req,
                {
                    "ok": "1",
                    "labels": pred.get("labels") or [],
                    "probs": pred.get("probs") or [],
                    "top_label": str(pred.get("top_label") or ""),
                    "top_prob": float(pred.get("top_prob") or 0.0),
                    "ood": bool(pred.get("ood")),
                    "ood_reason": str(pred.get("ood_reason") or ""),
                    "sign_pct": float(pred.get("sign_pct") or 0.0),
                    "max_prob": float(pred.get("max_prob") or 0.0),
                    "entropy_ratio": float(pred.get("entropy_ratio") or 0.0),
                    "image_b64": base64.b64encode(preview_png).decode("ascii"),
                    "processed_image_b64": str(pred.get("processed_image_b64") or ""),
                    "crop_image_b64": str(pred.get("crop_image_b64") or ""),
                    "full_image_b64": str(pred.get("full_image_b64") or ""),
                    "processed_variant": str(pred.get("processed_variant") or ""),
                    "crop": pred.get("crop"),
                    "model_source": str(pred.get("model_source") or "latest"),
                },
                cors=True,
            )
            return
        if path == "/export/pick_dir":
            picked: Optional[str] = None
            if sys.platform == "darwin":
                try:
                    script = 'POSIX path of (choose folder with prompt "Export folder")'
                    proc = subprocess.run(
                        ["osascript", "-e", script],
                        capture_output=True,
                        text=True,
                        check=False,
                    )
                    if proc.returncode == 0:
                        out = (proc.stdout or "").strip()
                        if out:
                            picked = out
                except Exception:
                    picked = None
            if not picked:
                try:
                    import tkinter as tk
                    from tkinter import filedialog
                except Exception as e2:
                    _send_json(req, {"ok": "0", "error": "no-picker: " + str(e2)}, status=400, cors=True)
                    return
                root = tk.Tk()
                root.withdraw()
                try:
                    root.attributes("-topmost", True)
                except Exception:
                    pass
                try:
                    initial = str(Path.home() / "Documents")
                    try:
                        initial = str(Path.home() / "Desktop")
                    except Exception:
                        pass
                    path = filedialog.askdirectory(title="Export folder", initialdir=initial)
                finally:
                    try:
                        root.destroy()
                    except Exception:
                        pass
                if path:
                    picked = path
            if not picked:
                _send_json(req, {"ok": "0", "canceled": "1"}, cors=True)
                return
            _send_json(req, {"ok": "1", "export_dir": picked}, cors=True)
            return
        if path == "/project/pick_save":
            picked: Optional[str] = None
            if sys.platform == "darwin":
                try:
                    script = 'POSIX path of (choose file name with prompt "Save Project" default name "project.tmproj")'
                    proc = subprocess.run(
                        ["osascript", "-e", script],
                        capture_output=True,
                        text=True,
                        check=False,
                    )
                    if proc.returncode == 0:
                        out = (proc.stdout or "").strip()
                        if out:
                            picked = out
                except Exception:
                    picked = None
            if not picked:
                try:
                    import tkinter as tk
                    from tkinter import filedialog
                except Exception as e2:
                    _send_json(req, {"ok": "0", "error": "no-picker: " + str(e2)}, status=400, cors=True)
                    return
                root = tk.Tk()
                root.withdraw()
                try:
                    root.attributes("-topmost", True)
                except Exception:
                    pass
                try:
                    initial = str(Path.home() / "Documents")
                    try:
                        initial = str(Path.home() / "Desktop")
                    except Exception:
                        pass
                    path = filedialog.asksaveasfilename(
                        title="Save Project",
                        defaultextension=".tmproj",
                        initialfile="project.tmproj",
                        initialdir=initial,
                        filetypes=[("Teachable Machine Project", "*.tmproj")],
                    )
                finally:
                    try:
                        root.destroy()
                    except Exception:
                        pass
                if path:
                    picked = path
            if not picked:
                _send_json(req, {"ok": "0", "canceled": "1"}, cors=True)
                return
            p = Path(picked).expanduser()
            if p.suffix.lower() != ".tmproj":
                p = p.with_suffix(".tmproj")
            _send_json(req, {"ok": "1", "save_path": str(p)}, cors=True)
            return
        if path == "/project/pick_open":
            picked2: Optional[str] = None
            if sys.platform == "darwin":
                try:
                    script = 'POSIX path of (choose file with prompt "Open Project")'
                    proc = subprocess.run(
                        ["osascript", "-e", script],
                        capture_output=True,
                        text=True,
                        check=False,
                    )
                    if proc.returncode == 0:
                        out = (proc.stdout or "").strip()
                        if out:
                            picked2 = out
                except Exception:
                    picked2 = None
            if not picked2:
                try:
                    import tkinter as tk
                    from tkinter import filedialog
                except Exception as e2:
                    _send_json(req, {"ok": "0", "error": "no-picker: " + str(e2)}, status=400, cors=True)
                    return
                root = tk.Tk()
                root.withdraw()
                try:
                    root.attributes("-topmost", True)
                except Exception:
                    pass
                try:
                    initial = str(Path.home() / "Documents")
                    try:
                        initial = str(Path.home() / "Desktop")
                    except Exception:
                        pass
                    path = filedialog.askopenfilename(
                        title="Open Project",
                        initialdir=initial,
                        filetypes=[("Teachable Machine Project", "*.tmproj")],
                    )
                finally:
                    try:
                        root.destroy()
                    except Exception:
                        pass
                if path:
                    picked2 = path
            if not picked2:
                _send_json(req, {"ok": "0", "canceled": "1"}, cors=True)
                return
            p = Path(picked2).expanduser()
            if p.suffix.lower() != ".tmproj":
                _send_json(req, {"ok": "0", "error": "not a .tmproj file"}, status=400, cors=True)
                return
            _send_json(req, {"ok": "1", "open_path": str(p)}, cors=True)
            return
        if path == "/live/capture":
            session_id = (qs.get("session") or [""])[0]
            source = (qs.get("source") or [""])[0]
            class_name = (qs.get("class") or [""])[0]
            if not session_id or not source or not class_name:
                _send_json(req, {"ok": "0", "error": "missing params"}, status=400, cors=True)
                return
            # Single capture consumes the live worker's capture_png cache —
            # NEVER opens a second camera/serial reader (macOS allows one
            # consumer per camera; the serial reader is single-consumer).
            # Wait briefly for a FRESH frame so a click can never save an
            # arbitrarily old one.
            self._ensure_live(session_id=session_id, source=source)
            png = self._get_live_capture_fresh(session_id=session_id, source=source, timeout_s=0.4)
            if png is None:
                err = self._get_live_error(session_id=session_id, source=source) or "capture unavailable"
                _send_json(req, {"ok": "0", "error": err}, status=409, cors=True)
                return
            cfg = self._configs.get(session_id)
            if cfg is None:
                _send_json(req, {"ok": "0", "error": "missing config"}, status=400, cors=True)
                return
            p = _save_png(cfg.dataset_root, class_name, png)
            self._cache_one_sample(cfg.dataset_root, class_name, p, img_size=int(cfg.img_size))
            _send_json(
                req,
                {
                    "ok": "1",
                    "class": class_name,
                    "filename": p.name,
                    "image_b64": base64.b64encode(png).decode("ascii"),
                },
                cors=True,
            )
            return
        if path == "/live/close":
            session_id = (qs.get("session") or [""])[0]
            source = (qs.get("source") or [""])[0]
            self._stop_live(session_id=session_id, source=source if source else None, wait=True, timeout_s=2.0)
            _send_json(req, {"ok": "1"}, cors=True)
            return
        if path == "/live/close-all":
            # Release every live device session — the desktop shell calls this
            # when the user switches away from the training page, so the block
            # editor can upload to the same board.
            self._stop_live_all()
            _send_json(req, {"ok": "1", "closed": 1}, cors=True)
            return
        if path == "/shutdown":
            # Graceful app-close path: release cameras/serial/server AFTER the
            # response is flushed (release_all_resources stops the HTTP
            # server, so it must not run before the client gets its answer).
            _send_json(req, {"ok": "1"}, cors=True)
            try:
                req.wfile.flush()
            except Exception:
                pass
            self.release_all_resources()
            return
        _send_json(req, {"ok": "0", "error": "not found"}, status=404)

    def _handle_options(self, req: BaseHTTPRequestHandler) -> None:
        req.send_response(204)
        req.send_header("Access-Control-Allow-Origin", "*")
        req.send_header("Access-Control-Allow-Headers", "Content-Type")
        req.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        req.send_header("Access-Control-Max-Age", "86400")
        req.end_headers()

    def _handle_post(self, req: BaseHTTPRequestHandler) -> None:
        parsed = urlparse(req.path)
        path = parsed.path
        if path not in {
            "/upload",
            "/train/start",
            "/train/config",
            "/train/recommend",
            "/export/run",
            "/blockcoding-export",
            "/dataset/export",
            "/project/save",
            "/project/open",
            "/project/reset",
            "/classes/rename",
            "/classes/add",
            "/classes/delete",
            "/samples/delete",
            "/samples/clear",
            "/preprocess/preview",
            "/preview/predict_upload",
            "/classes/save_config",
            "/classes/reorder",
        }:
            _send_json(req, {"ok": "0", "error": "not found"}, status=404, cors=True)
            return

        content_len = int(req.headers.get("Content-Length", "0") or "0")
        if content_len <= 0:
            _send_json(req, {"ok": "0", "error": "empty body"}, status=400, cors=True)
            return
        raw = req.rfile.read(content_len)
        payload = json.loads(raw.decode("utf-8"))
        if path == "/blockcoding-export":
            session_id = str(payload.get("session") or "").strip()
            if not session_id:
                _send_json(req, {"ok": "0", "error": "missing session"}, status=400, cors=True)
                return
            try:
                result = self._blockcoding_export(session_id=session_id)
            except Exception as e:
                _send_json(req, {"ok": "0", "error": str(e)}, status=400, cors=True)
                return
            _send_json(req, {"ok": "1", **result}, cors=True)
            return
        if path == "/upload":
            session_id = str(payload.get("session") or "").strip()
            class_name = str(payload.get("class") or "").strip()
            image_b64 = str(payload.get("image_b64") or "").strip()
            if not session_id or not class_name or not image_b64:
                _send_json(req, {"ok": "0", "error": "missing params"}, status=400, cors=True)
                return
            cfg = self._configs.get(session_id)
            if cfg is None:
                _send_json(req, {"ok": "0", "error": "missing config"}, status=400, cors=True)
                return
            try:
                png = base64.b64decode(image_b64, validate=True)
                img = Image.open(_bytes_io(png))
                try:
                    img = ImageOps.exif_transpose(img)
                    img = img.convert("RGB" if img.mode not in {"L", "RGB"} else img.mode)
                    img.load()
                    max_side = 1280
                    w, h = img.size
                    if max(w, h) > max_side:
                        scale = max_side / float(max(w, h))
                        img = img.resize(
                            (max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
                            Image.LANCZOS,
                        )
                    png = _to_png_bytes(img)
                    thumb_size = 96
                    tw, th = img.size
                    tscale = thumb_size / float(max(tw, th))
                    thumb_w = max(1, int(round(tw * tscale)))
                    thumb_h = max(1, int(round(th * tscale)))
                    thumb_img = img.resize((thumb_w, thumb_h), Image.BILINEAR)
                    try:
                        thumb_png = _to_png_bytes(thumb_img)
                        thumb_b64 = base64.b64encode(thumb_png).decode("ascii")
                    finally:
                        try:
                            thumb_img.close()
                        except Exception:
                            pass
                finally:
                    try:
                        img.close()
                    except Exception:
                        pass
                p = _save_png(cfg.dataset_root, class_name, png)
                self._cache_one_sample(cfg.dataset_root, class_name, p, img_size=int(cfg.img_size))
            except Exception as e:
                _send_json(req, {"ok": "0", "error": str(e)}, status=400, cors=True)
                return
            _send_json(
                req,
                {
                    "ok": "1",
                    "class": class_name,
                    "filename": p.name,
                    "thumb_b64": thumb_b64,
                },
                cors=True,
            )
            return
        if path == "/train/start":
            session_id = str(payload.get("session") or "").strip()
            cfg = payload.get("cfg") or {}
            if not session_id:
                _send_json(req, {"ok": "0", "error": "missing session"}, status=400, cors=True)
                return
            try:
                self._start_train(session_id=session_id, cfg=cfg)
            except Exception as e:
                _send_json(req, {"ok": "0", "error": str(e)}, status=400, cors=True)
                return
            _send_json(req, {"ok": "1"}, cors=True)
            return
        if path == "/train/config":
            session_id = str(payload.get("session") or "").strip()
            cfg = payload.get("cfg") or {}
            if not session_id:
                _send_json(req, {"ok": "0", "error": "missing session"}, status=400, cors=True)
                return
            try:
                self._save_train_cfg(session_id=session_id, cfg=cfg)
                # Keep the capture/preview image size in lockstep with the
                # training "Image Size" setting — samples and their cache are
                # stored at this size, never hardcoded 96.
                try:
                    size = int((cfg or {}).get("img_size") or 96)
                    if 8 <= size <= 512:
                        self.update_config(session_id, img_size=size)
                except Exception:
                    pass
            except Exception as e:
                _send_json(req, {"ok": "0", "error": str(e)}, status=400, cors=True)
                return
            _send_json(req, {"ok": "1"}, cors=True)
            return
        if path == "/train/recommend":
            session_id = str(payload.get("session") or "").strip()
            cfg = payload.get("cfg") or {}
            if not session_id:
                _send_json(req, {"ok": "0", "error": "missing session"}, status=400, cors=True)
                return
            try:
                _send_json(req, {"ok": "1", **self._train_recommend(session_id=session_id, cfg=cfg)}, cors=True)
            except Exception as e:
                _send_json(req, {"ok": "0", "error": str(e)}, status=400, cors=True)
            return
        if path == "/preprocess/preview":
            session_id = str(payload.get("session") or "").strip()
            class_name = str(payload.get("class") or "").strip()
            filename = str(payload.get("filename") or "").strip()
            class_config = payload.get("class_config")
            sample_config = payload.get("sample_config")
            if not session_id or not class_name or not filename:
                _send_json(req, {"ok": "0", "error": "missing params"}, status=400, cors=True)
                return
            try:
                preview = self._preprocess_preview(
                    session_id=session_id,
                    class_name=class_name,
                    filename=filename,
                    class_config=class_config,
                    sample_config=sample_config,
                )
            except Exception as e:
                _send_json(req, {"ok": "0", "error": str(e)}, status=400, cors=True)
                return
            _send_json(req, {"ok": "1", **preview}, cors=True)
            return
        if path == "/classes/reorder":
            session_id = str(payload.get("session") or "").strip()
            ordered = payload.get("classes")
            if not session_id or not isinstance(ordered, list) or not ordered:
                _send_json(req, {"ok": "0", "error": "missing classes"}, status=400, cors=True)
                return
            try:
                result = self._classes_reorder(session_id, [str(x) for x in ordered])
            except Exception as e:
                _send_json(req, {"ok": "0", "error": str(e)}, status=400, cors=True)
                return
            _send_json(req, {"ok": "1", "classes": result}, cors=True)
            return
        if path == "/classes/save_config":
            session_id = str(payload.get("session") or "").strip()
            class_preprocess = payload.get("class_preprocess")
            if not session_id:
                _send_json(req, {"ok": "0", "error": "missing session"}, status=400, cors=True)
                return
            try:
                with self._lock:
                    sess_cfg = self._configs.get(session_id)
                if sess_cfg is None:
                    _send_json(req, {"ok": "0", "error": "missing config"}, status=400, cors=True)
                    return
                classes = self._classes_load(sess_cfg.dataset_root)
                self._classes_save(
                    sess_cfg.dataset_root,
                    classes,
                    class_preprocess=normalize_class_preprocess_map(class_preprocess) if class_preprocess else None,
                )
                # Thresholds / ROI config changed — the processed cache must
                # follow so previews and the next training use the new
                # settings immediately.
                try:
                    self._rebuild_processed_cache(
                        sess_cfg.dataset_root,
                        class_preprocess=normalize_class_preprocess_map(class_preprocess) if class_preprocess else None,
                        fast_mode=False,
                        out_size=int(sess_cfg.img_size),
                    )
                except Exception:
                    pass
            except Exception as e:
                _send_json(req, {"ok": "0", "error": str(e)}, status=400, cors=True)
                return
            _send_json(req, {"ok": "1"}, cors=True)
            return
        if path == "/preview/predict_upload":
            session_id = str(payload.get("session") or "").strip()
            image_b64 = str(payload.get("image_b64") or "").strip()
            preprocess_mode = str(payload.get("preprocess") or "").strip()
            bg_dark_thresh = payload.get("bg_dark")
            bg_lum_thresh = payload.get("bg_lum")
            ood_sign_pct_min = payload.get("ood_sign_min")
            ood_sign_pct_max = payload.get("ood_sign_max")
            ood_max_prob_min = payload.get("ood_max_prob")
            ood_entropy_ratio_max = payload.get("ood_entropy_max")
            bg_dark_thresh = int(bg_dark_thresh) if bg_dark_thresh is not None and str(bg_dark_thresh).strip() != "" else None
            bg_lum_thresh  = int(bg_lum_thresh)  if bg_lum_thresh  is not None and str(bg_lum_thresh).strip()  != "" else None
            ood_sign_pct_min = float(ood_sign_pct_min) if ood_sign_pct_min is not None and str(ood_sign_pct_min).strip() != "" else None
            ood_sign_pct_max = float(ood_sign_pct_max) if ood_sign_pct_max is not None and str(ood_sign_pct_max).strip() != "" else None
            ood_max_prob_min = float(ood_max_prob_min) if ood_max_prob_min is not None and str(ood_max_prob_min).strip() != "" else None
            ood_entropy_ratio_max = float(ood_entropy_ratio_max) if ood_entropy_ratio_max is not None and str(ood_entropy_ratio_max).strip() != "" else None
            if not session_id or not image_b64:
                _send_json(req, {"ok": "0", "error": "missing params"}, status=400, cors=True)
                return
            try:
                png = base64.b64decode(image_b64, validate=True)
                pred = self._preview_predict(
                    session_id=session_id,
                    png=png,
                    preprocess_mode=preprocess_mode,
                    bg_dark_thresh=bg_dark_thresh,
                    bg_lum_thresh=bg_lum_thresh,
                    ood_sign_pct_min=ood_sign_pct_min,
                    ood_sign_pct_max=ood_sign_pct_max,
                    ood_max_prob_min=ood_max_prob_min,
                    ood_entropy_ratio_max=ood_entropy_ratio_max,
                )
            except Exception as e:
                _send_json(req, {"ok": "0", "error": str(e)}, status=400, cors=True)
                return
            _send_json(
                req,
                {
                    "ok": "1",
                    **pred,
                    "image_b64": image_b64,
                },
                cors=True,
            )
            return
        if path == "/export/run":
            session_id = str(payload.get("session") or "").strip()
            export_dir = str(payload.get("export_dir") or "").strip()
            model_name = str(payload.get("model_name") or "model").strip() or "model"
            array_name = str(payload.get("array_name") or "g_model").strip() or "g_model"
            overwrite = _is_truthy(payload.get("overwrite"))
            if not session_id or not export_dir:
                _send_json(req, {"ok": "0", "error": "missing fields"}, status=400, cors=True)
                return
            try:
                out_dir = self._export_run(
                    session_id=session_id,
                    export_dir=Path(export_dir).expanduser().resolve(),
                    model_name=model_name,
                    array_name=array_name,
                    overwrite=overwrite,
                )
            except ExportConflictError as e:
                _send_json(
                    req,
                    {
                        "ok": "0",
                        "error": str(e),
                        "needs_confirm": "1",
                        "conflicts": list(e.conflicts),
                    },
                    status=409,
                    cors=True,
                )
                return
            except Exception as e:
                _send_json(req, {"ok": "0", "error": str(e)}, status=400, cors=True)
                return
            out_dir, legacy_removed = out_dir
            _send_json(req, {"ok": "1", "export_dir": str(out_dir), "legacy_removed": legacy_removed}, cors=True)
            return
        if path == "/project/save":
            session_id = str(payload.get("session") or "").strip()
            save_path = str(payload.get("save_path") or "").strip()
            project_state = payload.get("project_state")
            if not session_id or not save_path:
                _send_json(req, {"ok": "0", "error": "missing fields"}, status=400, cors=True)
                return
            try:
                out_path = self._project_save(
                    session_id=session_id,
                    save_path=Path(save_path).expanduser().resolve(),
                    project_state=project_state if isinstance(project_state, dict) else None,
                )
            except Exception as e:
                _send_json(req, {"ok": "0", "error": str(e)}, status=400, cors=True)
                return
            _send_json(req, {"ok": "1", "save_path": str(out_path)}, cors=True)
            return
        if path == "/project/open":
            session_id = str(payload.get("session") or "").strip()
            open_path = str(payload.get("open_path") or "").strip()
            if not session_id or not open_path:
                _send_json(req, {"ok": "0", "error": "missing fields"}, status=400, cors=True)
                return
            try:
                state = self._project_open(session_id=session_id, open_path=Path(open_path).expanduser().resolve())
            except Exception as e:
                _send_json(req, {"ok": "0", "error": str(e)}, status=400, cors=True)
                return
            _send_json(req, {"ok": "1", "state": state}, cors=True)
            return
        if path == "/project/reset":
            session_id = str(payload.get("session") or "").strip()
            confirm = str(payload.get("confirm") or "").strip()
            if not session_id or confirm != "1":
                _send_json(req, {"ok": "0", "error": "missing fields"}, status=400, cors=True)
                return
            try:
                state = self._project_reset(session_id=session_id)
            except Exception as e:
                _send_json(req, {"ok": "0", "error": str(e)}, status=400, cors=True)
                return
            _send_json(req, {"ok": "1", "state": state}, cors=True)
            return
        if path == "/dataset/export":
            session_id = str(payload.get("session") or "").strip()
            export_dir = str(payload.get("export_dir") or "").strip()
            if not session_id or not export_dir:
                _send_json(req, {"ok": "0", "error": "missing fields"}, status=400, cors=True)
                return
            try:
                out_dir = self._dataset_export(
                    session_id=session_id,
                    export_dir=Path(export_dir).expanduser().resolve(),
                )
            except Exception as e:
                _send_json(req, {"ok": "0", "error": str(e)}, status=400, cors=True)
                return
            _send_json(req, {"ok": "1", "export_dir": str(out_dir)}, cors=True)
            return
        if path == "/classes/add":
            session_id = str(payload.get("session") or "").strip()
            if not session_id:
                _send_json(req, {"ok": "0", "error": "missing session"}, status=400, cors=True)
                return
            try:
                classes = self._classes_add(session_id=session_id)
            except Exception as e:
                _send_json(req, {"ok": "0", "error": str(e)}, status=400, cors=True)
                return
            _send_json(req, {"ok": "1", "classes": classes}, cors=True)
            return
        if path == "/samples/delete":
            session_id = str(payload.get("session") or "").strip()
            class_name = str(payload.get("class") or "").strip()
            filename = str(payload.get("filename") or "").strip()
            if not session_id or not class_name or not filename:
                _send_json(req, {"ok": "0", "error": "missing fields"}, status=400, cors=True)
                return
            try:
                state = self._sample_delete(session_id=session_id, class_name=class_name, filename=filename)
            except Exception as e:
                _send_json(req, {"ok": "0", "error": str(e)}, status=400, cors=True)
                return
            _send_json(req, {"ok": "1", "state": state}, cors=True)
            return
        if path == "/samples/clear":
            session_id = str(payload.get("session") or "").strip()
            class_name = str(payload.get("class") or "").strip()
            if not session_id or not class_name:
                _send_json(req, {"ok": "0", "error": "missing fields"}, status=400, cors=True)
                return
            try:
                state = self._samples_clear(session_id=session_id, class_name=class_name)
            except Exception as e:
                _send_json(req, {"ok": "0", "error": str(e)}, status=400, cors=True)
                return
            _send_json(req, {"ok": "1", "state": state}, cors=True)
            return
        if path == "/classes/delete":
            session_id = str(payload.get("session") or "").strip()
            name = str(payload.get("name") or "").strip()
            if not session_id or not name:
                _send_json(req, {"ok": "0", "error": "missing fields"}, status=400, cors=True)
                return
            try:
                classes = self._classes_delete(session_id=session_id, name=name)
            except Exception as e:
                _send_json(req, {"ok": "0", "error": str(e)}, status=400, cors=True)
                return
            _send_json(req, {"ok": "1", "classes": classes}, cors=True)
            return
        if path == "/classes/rename":
            session_id = str(payload.get("session") or "").strip()
            old_name = str(payload.get("old_name") or "").strip()
            new_name = str(payload.get("new_name") or "").strip()
            if not session_id or not old_name or not new_name:
                _send_json(req, {"ok": "0", "error": "missing fields"}, status=400, cors=True)
                return
            try:
                classes = self._classes_rename(session_id=session_id, old_name=old_name, new_name=new_name)
            except Exception as e:
                _send_json(req, {"ok": "0", "error": str(e)}, status=400, cors=True)
                return
            _send_json(req, {"ok": "1", "classes": classes}, cors=True)
            return

        session_id = str(payload.get("session") or "").strip()
        class_name = str(payload.get("class") or "").strip()
        image_b64 = str(payload.get("image_b64") or "").strip()
        if not session_id or not class_name or not image_b64:
            _send_json(req, {"ok": "0", "error": "missing fields"}, status=400, cors=True)
            return

        cfg = self._configs.get(session_id)
        if cfg is None:
            _send_json(req, {"ok": "0", "error": "missing config"}, status=400, cors=True)
            return

        img_bytes = base64.b64decode(image_b64)
        img = Image.open(_bytes_io(img_bytes)).convert("RGB")
        if cfg.crop_box is not None:
            x1, y1, x2, y2 = cfg.crop_box
            img = img.crop((x1, y1, x2, y2))
        img = img.resize((int(cfg.img_size), int(cfg.img_size)))
        out_png = _to_png_bytes(img)
        p = _save_png(cfg.dataset_root, class_name, out_png)
        self._cache_one_sample(cfg.dataset_root, class_name, p, img_size=int(cfg.img_size))
        _send_json(
            req,
            {
                "ok": "1",
                "class": class_name,
                "filename": p.name,
                "image_b64": base64.b64encode(out_png).decode("ascii"),
            },
            cors=True,
        )

    def _classes_meta_path(self, dataset_root: Path) -> Path:
        return dataset_root.parent / "tm_classes.json"

    def _processed_cache_dir(self, dataset_root: Path) -> Path:
        return dataset_root.parent / ".tm_processed_cache"

    def _classes_meta_load(self, dataset_root: Path) -> Dict[str, Any]:
        p = self._classes_meta_path(dataset_root)
        if not p.exists():
            return {}
        try:
            raw = json.loads(p.read_text(encoding="utf-8"))
            return raw if isinstance(raw, dict) else {}
        except Exception:
            return {}

    def _classes_infer_from_dirs(self, dataset_root: Path) -> List[str]:
        out: List[str] = []
        try:
            for p in sorted(dataset_root.iterdir()):
                if p.is_dir() and _list_class_image_files(p):
                    out.append(str(p.name))
        except Exception:
            return []
        return out

    def _classes_load(self, dataset_root: Path) -> List[str]:
        raw = self._classes_meta_load(dataset_root)
        classes = raw.get("classes") if isinstance(raw, dict) else None
        if isinstance(classes, list) and all(isinstance(x, str) for x in classes) and classes:
            return [str(x) for x in classes]
        inferred = self._classes_infer_from_dirs(dataset_root)
        if inferred:
            return inferred
        return ["Class 1", "Class 2"]

    def _class_preprocess_load(self, dataset_root: Path) -> Dict[str, Dict[str, Any]]:
        raw = self._classes_meta_load(dataset_root)
        return normalize_class_preprocess_map(raw.get("class_preprocess"))

    def _sample_preprocess_load(self, dataset_root: Path) -> Dict[str, Dict[str, Dict[str, Any]]]:
        raw = self._classes_meta_load(dataset_root)
        return normalize_sample_preprocess_map(raw.get("sample_preprocess"))

    def _classes_save(
        self,
        dataset_root: Path,
        classes: List[str],
        class_preprocess: Optional[Dict[str, Dict[str, Any]]] = None,
        sample_preprocess: Optional[Dict[str, Dict[str, Dict[str, Any]]]] = None,
    ) -> None:
        p = self._classes_meta_path(dataset_root)
        payload: Dict[str, Any] = {"classes": list(classes)}
        existing = self._class_preprocess_load(dataset_root)
        merged = normalize_class_preprocess_map(class_preprocess if class_preprocess is not None else existing)
        existing_sample = self._sample_preprocess_load(dataset_root)
        merged_sample = normalize_sample_preprocess_map(sample_preprocess if sample_preprocess is not None else existing_sample)
        if merged:
            payload["class_preprocess"] = merged
        if merged_sample:
            payload["sample_preprocess"] = merged_sample
        p.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    def _classes_reorder(self, session_id: str, ordered_classes: List[str]) -> List[str]:
        """Persist a new class ORDER.  The class list order defines the
        model's label indices (training + export read _classes_load), so
        drag-reordering the class cards remaps the model's class index."""
        with self._lock:
            cfg = self._configs.get(session_id)
        if cfg is None:
            raise RuntimeError("missing config")
        current = self._classes_load(cfg.dataset_root)
        if len(ordered_classes) != len(current) or set(ordered_classes) != set(current):
            raise ValueError("class list mismatch")
        self._classes_save(cfg.dataset_root, ordered_classes)
        return list(ordered_classes)

    def _next_class_name(self, existing: List[str]) -> str:
        s = set([str(x).strip() for x in existing if str(x).strip()])
        idx = 1
        while True:
            cand = f"Class {idx}"
            if cand not in s:
                return cand
            idx += 1

    def _classes_add(self, session_id: str) -> List[str]:
        with self._lock:
            cfg = self._configs.get(session_id)
        if cfg is None:
            raise RuntimeError("missing config")
        classes = self._classes_load(cfg.dataset_root)
        class_preprocess = self._class_preprocess_load(cfg.dataset_root)
        sample_preprocess = self._sample_preprocess_load(cfg.dataset_root)
        classes = list(classes) + [self._next_class_name(classes)]
        self._classes_save(cfg.dataset_root, classes, class_preprocess=class_preprocess, sample_preprocess=sample_preprocess)
        return classes

    def _classes_delete(self, session_id: str, name: str) -> List[str]:
        with self._lock:
            cfg = self._configs.get(session_id)
        if cfg is None:
            raise RuntimeError("missing config")
        classes = self._classes_load(cfg.dataset_root)
        name = str(name).strip()
        if not name:
            raise ValueError("Class name is empty.")
        actual_name = name
        if actual_name not in classes:
            safe_name = sanitize_class_name(actual_name)
            for c in classes:
                if sanitize_class_name(c) == safe_name:
                    actual_name = c
                    break
        if actual_name not in classes:
            raise ValueError("Class not found.")
        if len(classes) <= 2:
            raise ValueError("At least 2 classes are required.")
        classes = [c for c in classes if c != actual_name]
        if len(classes) < 2:
            raise ValueError("At least 2 classes are required.")
        class_dir = cfg.dataset_root / sanitize_class_name(actual_name)
        processed_dir = self._processed_cache_dir(cfg.dataset_root) / sanitize_class_name(actual_name)
        class_preprocess = self._class_preprocess_load(cfg.dataset_root)
        sample_preprocess = self._sample_preprocess_load(cfg.dataset_root)
        class_preprocess.pop(actual_name, None)
        sample_preprocess.pop(actual_name, None)
        if class_dir.exists():
            shutil.rmtree(class_dir)
        if processed_dir.exists():
            shutil.rmtree(processed_dir)
        self._classes_save(cfg.dataset_root, classes, class_preprocess=class_preprocess, sample_preprocess=sample_preprocess)
        return classes

    def _classes_rename(self, session_id: str, old_name: str, new_name: str) -> List[str]:
        new_name = str(new_name).strip()
        if not new_name:
            raise ValueError("New class name is empty.")
        with self._lock:
            cfg = self._configs.get(session_id)
        if cfg is None:
            raise RuntimeError("missing config")
        classes = self._classes_load(cfg.dataset_root)
        class_preprocess = self._class_preprocess_load(cfg.dataset_root)
        sample_preprocess = self._sample_preprocess_load(cfg.dataset_root)
        if old_name not in classes:
            raise ValueError("Class not found.")
        if new_name in classes and new_name != old_name:
            raise ValueError("A class with the same name already exists.")

        old_dir = cfg.dataset_root / sanitize_class_name(old_name)
        new_dir = cfg.dataset_root / sanitize_class_name(new_name)
        old_processed_dir = self._processed_cache_dir(cfg.dataset_root) / sanitize_class_name(old_name)
        new_processed_dir = self._processed_cache_dir(cfg.dataset_root) / sanitize_class_name(new_name)
        if old_dir.resolve() != new_dir.resolve():
            if new_dir.exists():
                raise ValueError("Target class folder already exists.")
            if old_dir.exists():
                old_dir.rename(new_dir)
        if old_processed_dir.resolve() != new_processed_dir.resolve():
            if new_processed_dir.exists():
                raise ValueError("Target processed folder already exists.")
            if old_processed_dir.exists():
                old_processed_dir.rename(new_processed_dir)

        updated: List[str] = []
        replaced = False
        for c in classes:
            if (not replaced) and c == old_name:
                updated.append(new_name)
                replaced = True
            else:
                updated.append(c)
        if not replaced:
            raise ValueError("Class not found.")
        if old_name in class_preprocess:
            class_preprocess[new_name] = class_preprocess.pop(old_name)
        if old_name in sample_preprocess:
            sample_preprocess[new_name] = sample_preprocess.pop(old_name)
        self._classes_save(cfg.dataset_root, updated, class_preprocess=class_preprocess, sample_preprocess=sample_preprocess)
        return updated

    def _start_record(self, session_id: str, source: str, class_name: str) -> None:
        self._stop_record(session_id=session_id, wait=True, timeout_s=2.5)
        # BOTH sources keep the live worker running during hold capture.
        # The record thread consumes the live worker's capture_png cache —
        # it never opens a second camera / serial reader.  Stopping the
        # device live reader here used to blind the class-panel view for
        # the whole hold (and the webcam worker starved by the same rule).
        with self._lock:
            cfg = self._configs.get(session_id)
            if cfg is None:
                raise RuntimeError("missing config")
            class_dir = cfg.dataset_root / sanitize_class_name(class_name)
            existing = 0
            try:
                if class_dir.exists():
                    existing = len(_list_class_image_files(class_dir))
            except Exception:
                existing = 0
            self._active[session_id] = {
                "recording": "1",
                "source": source,
                "class": class_name,
                "seq": 0,
                "count": int(existing),
                "filename": "",
                "image_b64": "",
                "error": "",
            }
            if session_id not in self._record_conds:
                self._record_conds[session_id] = threading.Condition()

        with self._lock:
            live_thread = self._live_threads.get(self._live_key(session_id, source))
        t = threading.Thread(
            target=self._record_worker,
            args=(session_id, source, class_name, live_thread),
            daemon=True,
        )
        with self._lock:
            self._record_threads[session_id] = t
        t.start()

    def _stop_record(self, session_id: str, *, wait: bool = True, timeout_s: float = 2.5) -> None:
        thread: Optional[threading.Thread] = None
        with self._lock:
            cur = self._active.get(session_id)
            if cur:
                cur["recording"] = "0"
            thread = self._record_threads.get(session_id)
        cond = self._record_conds.get(session_id)
        if cond is not None:
            with cond:
                cond.notify_all()
        if wait and thread is not None and thread.is_alive() and thread is not threading.current_thread():
            thread.join(timeout=max(0.0, float(timeout_s)))
        with self._lock:
            current = self._record_threads.get(session_id)
            if current is thread and (thread is None or not thread.is_alive()):
                self._record_threads.pop(session_id, None)

    def drop_session(self, session_id: str) -> None:
        """Release ALL per-session state for a finished/returned-home session:
        config, active record state, record condition/thread, live states and
        threads (keyed ``{session_id}:{source}``) and train/preview entries.
        Call after _stop_record + _stop_live so no worker can re-register.
        Safe to call for unknown sessions; idempotent."""
        prefix = f"{session_id}:"
        with self._lock:
            self._configs.pop(session_id, None)
            self._active.pop(session_id, None)
            self._record_conds.pop(session_id, None)
            self._record_threads.pop(session_id, None)
            self._train.pop(session_id, None)
            self._train_conds.pop(session_id, None)
            self._preview.pop(session_id, None)
            self._preview_locks.pop(session_id, None)
            for key in list(self._live):
                if key.startswith(prefix):
                    self._live.pop(key, None)
            for key in list(self._live_threads):
                if key.startswith(prefix):
                    self._live_threads.pop(key, None)
            for key in list(self._live_stops):
                if key.startswith(prefix):
                    self._live_stops.pop(key, None)

    def _wait_next_record(self, session_id: str, since: int, timeout_s: float = 10.0) -> Dict[str, Any]:
        cond = self._record_conds.get(session_id)
        if cond is None:
            with self._lock:
                cur = self._active.get(session_id)
                if cur is None:
                    return {"ok": "0", "error": "missing session"}
                seq = int(cur.get("seq") or 0)
                count = int(cur.get("count") or 0)
                recording = str(cur.get("recording") or "0")
                image_b64 = str(cur.get("image_b64") or "")
                filename = str(cur.get("filename") or "")
                error = str(cur.get("error") or "")
            return {
                "ok": "1",
                "seq": seq,
                "count": count,
                "recording": recording,
                "filename": filename,
                "image_b64": image_b64 if seq > since else "",
                "error": error,
            }
        deadline = time.time() + float(timeout_s)
        while True:
            with self._lock:
                cur = self._active.get(session_id)
                if cur is None:
                    return {"ok": "0", "error": "missing session"}
                seq = int(cur.get("seq") or 0)
                count = int(cur.get("count") or 0)
                recording = str(cur.get("recording") or "0")
                image_b64 = str(cur.get("image_b64") or "")
                filename = str(cur.get("filename") or "")
                error = str(cur.get("error") or "")
            if seq > since or recording != "1":
                return {
                    "ok": "1",
                    "seq": seq,
                    "count": count,
                    "recording": recording,
                    "filename": filename,
                    "image_b64": image_b64 if seq > since else "",
                    "error": error,
                }
            remaining = deadline - time.time()
            if remaining <= 0:
                return {
                    "ok": "1",
                    "seq": seq,
                    "count": count,
                    "recording": recording,
                    "filename": "",
                    "image_b64": "",
                    "error": error,
                }
            with cond:
                cond.wait(timeout=min(remaining, 1.0))

    def _blockcoding_export(self, session_id: str) -> Dict[str, Any]:
        """Copy the latest trained model into the block-coding outbox.

        The AIoScout desktop app watches <app data>/blockcoding_outbox/ and
        imports each complete sub-directory into its persistent model library,
        which the block-coding editor picks up for AI Vision blocks. Output is
        model.tflite + labels.txt (one label per line — the format the mixly
        server's convertTfliteToHeader expects).
        """
        with self._lock:
            cfg = self._configs.get(session_id)
        if cfg is None:
            raise RuntimeError("missing config")
        latest = self._train_result_path(cfg.dataset_root)
        if not latest.exists():
            raise RuntimeError("no trained model yet — train first, then export")
        meta = json.loads(latest.read_text(encoding="utf-8"))
        tflite_path = Path(str(meta.get("tflite_path") or "")).expanduser().resolve()
        if not tflite_path.exists():
            raise RuntimeError("missing .tflite file")
        labels = [str(x).strip() for x in (meta.get("labels") or []) if str(x).strip()]

        stamp = time.strftime("%Y%m%d_%H%M%S")
        outbox = _app_data_dir() / "blockcoding_outbox" / f"{stamp}_{uuid.uuid4().hex[:6]}"
        outbox.mkdir(parents=True, exist_ok=True)
        shutil.copy2(tflite_path, outbox / "model.tflite")
        (outbox / "labels.txt").write_text(("\n".join(labels) + "\n") if labels else "", encoding="utf-8")
        preprocess_info = self._blockcoding_preprocess_info(cfg.dataset_root, labels)
        (outbox / "preprocess.json").write_text(json.dumps(preprocess_info, indent=2), encoding="utf-8")
        return {
            "outbox_dir": str(outbox),
            "labels": labels,
            "model_bytes": tflite_path.stat().st_size,
            "preprocess": {"mode": preprocess_info["mode"], "uniform": preprocess_info["uniform"]},
        }

    def _blockcoding_preprocess_info(self, dataset_root: Path, labels: List[str]) -> Dict[str, Any]:
        """Bake the class filter (preprocess) settings into the block-coding export.

        The block-coding compiler turns this into the on-device crop pipeline:
        dark/lum thresholds drive the sign search box (mirroring
        preprocess_blue_diff_array in image_preprocess.py), so the deployed
        model sees the same input distribution it was trained on.  Runtime
        inference cannot know the class before predicting, so a single device
        pipeline requires uniform settings — mixed configs export mode
        "none" (legacy full-frame) with uniform=False.
        """
        class_pre = self._class_preprocess_load(dataset_root)
        per_class: Dict[str, Any] = {}
        for name in labels:
            cfg = normalize_class_preprocess(class_pre.get(name) or {})
            if cfg.get("mode") == PREPROCESS_MODE_MANUAL_ROI and cfg.get("manual_roi"):
                eff = {"mode": "manual", "roi": [float(v) for v in cfg["manual_roi"]]}
            elif class_clip_mode(name) == CLIP_MODE_JUNCTION:
                eff = {"mode": "junction"}
            else:
                eff = {"mode": "sign", "dark": int(cfg.get("bg_dark_thresh", 0)), "lum": int(cfg.get("bg_lum_thresh", 100))}
            per_class[name] = eff
        if not per_class:
            per_class = {"": {"mode": "sign", "dark": 0, "lum": 100}}
        distinct = {json.dumps(v, sort_keys=True) for v in per_class.values()}
        uniform = len(distinct) == 1
        effective = next(iter(per_class.values())) if uniform else {"mode": "none"}
        info: Dict[str, Any] = {"version": 1, "mode": effective["mode"], "uniform": uniform}
        if effective["mode"] == "sign":
            info["dark"] = int(effective["dark"])
            info["lum"] = int(effective["lum"])
        elif effective["mode"] == "manual":
            info["roi"] = [float(v) for v in effective["roi"]]
        info["classes"] = per_class
        return info

    def _train_result_path(self, dataset_root: Path) -> Path:
        return dataset_root.parent / "tm_train_latest.json"

    def _export_run(self, session_id: str, export_dir: Path, model_name: str, array_name: str, overwrite: bool = False) -> Path:
        with self._lock:
            cfg = self._configs.get(session_id)
        if cfg is None:
            raise RuntimeError("missing config")
        latest = self._train_result_path(cfg.dataset_root)
        if not latest.exists():
            raise RuntimeError("missing trained model")
        meta = json.loads(latest.read_text(encoding="utf-8"))
        tflite_path = Path(str(meta.get("tflite_path") or "")).expanduser().resolve()
        labels = list(meta.get("labels") or []) if isinstance(meta, dict) else []
        img_size = int(meta.get("img_size") or 96) if isinstance(meta, dict) else 96
        color_mode = str(meta.get("color_mode") or "rgb") if isinstance(meta, dict) else "rgb"
        channels = 1 if color_mode.strip().lower() in {"l", "gray", "grayscale", "mono"} else 3
        if not tflite_path.exists():
            raise RuntimeError("missing .tflite file")
        export_dir.mkdir(parents=True, exist_ok=True)
        test_path = export_dir / ".write_test"
        test_path.write_text("ok", encoding="utf-8")
        test_path.unlink(missing_ok=True)
        source_bytes = tflite_path.read_bytes()
        from trainer import export_tflite_c_sources

        base = (model_name or "").strip() or "tm"
        safe_base = "".join([c if (c.isalnum() or c in {"_", "-", "."}) else "_" for c in base])
        safe_base = safe_base.strip("._-") or "tm"
        c_ident = "".join([c if (c.isalnum() or c == "_") else "_" for c in safe_base])
        if not c_ident or not (c_ident[0].isalpha() or c_ident[0] == "_"):
            c_ident = f"tm_{c_ident}" if c_ident else "tm"
        default_array = f"g_{c_ident}_model_data"
        array = (array_name or "").strip()
        if (not array) or array == "g_model":
            array = default_array
        h_name = f"{safe_base}_model_data.h"
        cpp_name = f"{safe_base}_model_data.cpp"
        target_paths = [
            export_dir / f"{safe_base}.tflite",
            export_dir / h_name,
            export_dir / cpp_name,
            export_dir / "model_settings.h",
            export_dir / "model_settings.cpp",
            export_dir / "model_resolver.h",
            export_dir / "labels.txt",
        ]
        dedup_target_paths: List[Path] = []
        seen_target_names = set()
        for path in target_paths:
            name = path.name
            if name in seen_target_names:
                continue
            seen_target_names.add(name)
            dedup_target_paths.append(path)
        conflicts = [p.name for p in dedup_target_paths if p.exists()]
        if conflicts and not overwrite:
            raise ExportConflictError(conflicts)

        src, hdr = export_tflite_c_sources(source_bytes, array_name=array)
        (export_dir / f"{safe_base}.tflite").write_bytes(source_bytes)
        (export_dir / h_name).write_text(hdr, encoding="utf-8")
        (export_dir / cpp_name).write_text(f'#include "{h_name}"\n\n' + src, encoding="utf-8")
        (export_dir / "labels.txt").write_text("\n".join([str(x) for x in labels]) + "\n", encoding="utf-8")
        # Generate model_resolver.h from the actual ops in the exported model
        resolver_h = _generate_model_resolver_h(source_bytes)
        (export_dir / "model_resolver.h").write_text(resolver_h, encoding="utf-8")
        def _c_str(s: str) -> str:
            return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'

        h_guard = "TFLITE_MODEL_SETTINGS_H_"
        model_settings_h = "\n".join(
            [
                "/*",
                "Auto-generated by TFLiteTraining export.",
                "*/",
                "",
                f"#ifndef {h_guard}",
                f"#define {h_guard}",
                "",
                "constexpr int kNumCols = " + str(int(img_size)) + ";",
                "constexpr int kNumRows = " + str(int(img_size)) + ";",
                "constexpr int kNumChannels = " + str(int(channels)) + ";",
                "",
                "constexpr int kMaxImageSize = kNumCols * kNumRows * kNumChannels;",
                "",
                "constexpr int kCategoryCount = " + str(int(len(labels))) + ";",
                "extern const char* kCategoryLabels[kCategoryCount];",
                "",
                f"#endif  // {h_guard}",
                "",
            ]
        )
        label_items = ", ".join([_c_str(str(x)) for x in labels])
        model_settings_cpp = "\n".join(
            [
                "/*",
                "Auto-generated by TFLiteTraining export.",
                "*/",
                "",
                '#include "model_settings.h"',
                "",
                "const char* kCategoryLabels[kCategoryCount] = {",
                f"    {label_items},",
                "};",
                "",
            ]
        )
        (export_dir / "model_settings.h").write_text(model_settings_h, encoding="utf-8")
        (export_dir / "model_settings.cpp").write_text(model_settings_cpp, encoding="utf-8")
        # Stamp the deployed pointer: Preview must keep validating the SAME
        # model that was just exported to the device.  Without this, every
        # later training run silently advances the Preview model while the
        # device keeps running the exported one — train/export/device drift.
        try:
            latest_meta = self._train_result_path(cfg.dataset_root)
            if latest_meta.exists():
                shutil.copyfile(latest_meta, cfg.dataset_root.parent / "deployed.json")
        except Exception:
            pass
        # Fool-proofing: older versions exported model.h / model.cpp as
        # duplicates of {base}_model_data.* — including BOTH in a firmware
        # project causes duplicate-symbol link errors.  We no longer
        # generate them; remove legacy copies left by previous exports
        # (only files recognisably ours, so user files are never touched).
        legacy_removed: List[str] = []
        for _name in ("model.h", "model.cpp"):
            _p = export_dir / _name
            if not _p.exists():
                continue
            try:
                _text = _p.read_text(encoding="utf-8", errors="replace")
            except Exception:
                continue
            if "model_data" in _text and ("unsigned char" in _text or "#include" in _text):
                try:
                    _p.unlink()
                    legacy_removed.append(_name)
                except Exception:
                    pass
        return export_dir, legacy_removed

    def _dataset_export(self, session_id: str, export_dir: Path) -> Path:
        with self._lock:
            cfg = self._configs.get(session_id)
        if cfg is None:
            raise RuntimeError("missing config")
        dataset_root = cfg.dataset_root.resolve()
        if not dataset_root.exists():
            raise RuntimeError("missing dataset")
        classes = self._classes_load(dataset_root)
        export_dir.mkdir(parents=True, exist_ok=True)
        test_path = export_dir / ".write_test"
        test_path.write_text("ok", encoding="utf-8")
        test_path.unlink(missing_ok=True)

        ts = time.strftime("%Y%m%d_%H%M%S")
        out_dir = export_dir / f"tm_dataset_{ts}"
        out_dir.mkdir(parents=True, exist_ok=False)
        for class_name in classes:
            src_dir = dataset_root / sanitize_class_name(class_name)
            dst_dir = out_dir / sanitize_class_name(class_name)
            dst_dir.mkdir(parents=True, exist_ok=True)
            if not src_dir.exists():
                continue
            for src in _list_class_image_files(src_dir):
                shutil.copy2(src, dst_dir / src.name)
        labels_txt = out_dir / "labels.txt"
        labels_json = out_dir / "labels.json"
        labels_txt.write_text("\n".join(classes) + "\n", encoding="utf-8")
        labels_json.write_text(json.dumps({"labels": classes}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return out_dir

    def _project_save(
        self, session_id: str, save_path: Path, project_state: Optional[Dict[str, Any]] = None
    ) -> Path:
        with self._lock:
            cfg = self._configs.get(session_id)
        if cfg is None:
            raise RuntimeError("missing config")
        save_path.parent.mkdir(parents=True, exist_ok=True)
        test_path = save_path.parent / ".write_test"
        test_path.write_text("ok", encoding="utf-8")
        test_path.unlink(missing_ok=True)
        if save_path.suffix.lower() != ".tmproj":
            save_path = save_path.with_suffix(".tmproj")

        dataset_root = cfg.dataset_root
        workspace_root = dataset_root.parent
        classes_meta = self._classes_meta_path(dataset_root)
        processed_cache_dir = self._processed_cache_dir(dataset_root)
        latest = self._train_result_path(dataset_root)
        meta: Dict[str, Any] = {}
        if latest.exists():
            try:
                meta = json.loads(latest.read_text(encoding="utf-8"))
            except Exception:
                meta = {}

        merged_project_state = self._project_state_payload(dataset_root, project_state=project_state)
        manifest = {
            "format": "tmproj",
            "version": 1,
            "project_type": "image",
            "created_at": int(time.time()),
            "dataset_dir": "dataset",
            "dataset_labels_txt": "dataset/labels.txt",
            "dataset_labels_json": "dataset/labels.json",
            "classes_meta": "tm_classes.json" if classes_meta.exists() else "",
            "train_latest": "tm_train_latest.json" if latest.exists() else "",
            "train_run_dir": "",
            "project_state": "tm_project_state.json",
            "processed_cache_dir": "",
        }

        run_dir = Path(str(meta.get("run_dir") or "")).expanduser().resolve() if meta else Path()
        include_run_dir = bool(run_dir) and run_dir.exists() and str(run_dir).startswith(str(workspace_root.resolve()))
        if include_run_dir:
            manifest["train_run_dir"] = "runs/latest"
        class_preprocess = normalize_class_preprocess_map(merged_project_state.get("class_preprocess"))
        sample_preprocess = normalize_sample_preprocess_map(merged_project_state.get("sample_preprocess"))
        if processed_cache_dir.exists():
            shutil.rmtree(processed_cache_dir)
        self._rebuild_processed_cache(
            dataset_root,
            class_preprocess=class_preprocess,
            sample_preprocess=sample_preprocess,
            global_preprocess_mode=str(merged_project_state.get("preprocess_mode") or PREPROCESS_MODE_AUTO_BY_LABEL),
        )
        if processed_cache_dir.exists():
            manifest["processed_cache_dir"] = "processed_cache"

        # Record the display order (newest-first) of every class's samples so
        # project-open can reproduce it — zip round-trips lose sub-second
        # mtimes, which scrambled the sample strip order after reopen.
        sample_order: Dict[str, List[str]] = {}
        if dataset_root.exists():
            for class_dir in sorted([d for d in dataset_root.iterdir() if d.is_dir()]):
                sample_order[str(class_dir.name)] = [p.name for p in _list_class_image_files(class_dir)]
        manifest["sample_order"] = sample_order

        tmp_path = save_path.with_suffix(".tmproj.tmp")
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        with zipfile.ZipFile(tmp_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
            zf.writestr("tm_project_state.json", json.dumps(merged_project_state, ensure_ascii=False, indent=2) + "\n")
            if classes_meta.exists():
                zf.write(classes_meta, arcname="tm_classes.json")
            if latest.exists():
                zf.write(latest, arcname="tm_train_latest.json")
            labels = self._classes_load(dataset_root)
            zf.writestr(Path("dataset/labels.txt").as_posix(), "\n".join(labels) + "\n")
            zf.writestr(
                Path("dataset/labels.json").as_posix(),
                json.dumps({"labels": labels}, ensure_ascii=False, indent=2) + "\n",
            )
            if dataset_root.exists():
                for p in dataset_root.rglob("*"):
                    if p.is_dir():
                        continue
                    rel = p.relative_to(dataset_root)
                    zf.write(p, arcname=str(Path("dataset") / rel))
            if processed_cache_dir.exists():
                for p in processed_cache_dir.rglob("*"):
                    if p.is_dir():
                        continue
                    rel = p.relative_to(processed_cache_dir)
                    zf.write(p, arcname=str(Path("processed_cache") / rel))
            if include_run_dir:
                for p in run_dir.rglob("*"):
                    if p.is_dir():
                        continue
                    rel = p.relative_to(run_dir)
                    zf.write(p, arcname=str(Path("runs/latest") / rel))
        tmp_path.replace(save_path)
        return save_path

    def _project_open(self, session_id: str, open_path: Path) -> Dict[str, Any]:
        with self._lock:
            cfg = self._configs.get(session_id)
            self._train.pop(session_id, None)
            self._preview.pop(session_id, None)
        if cfg is None:
            raise RuntimeError("missing config")
        if open_path.suffix.lower() != ".tmproj":
            raise ValueError("not a .tmproj file")
        if not open_path.exists():
            raise FileNotFoundError(str(open_path))

        dataset_root = cfg.dataset_root
        workspace_root = dataset_root.parent.resolve()
        tmp_dir = workspace_root / f".tmproj_import_{int(time.time())}"
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)
        tmp_dir.mkdir(parents=True, exist_ok=True)

        try:
            with zipfile.ZipFile(open_path, "r") as zf:
                for info in zf.infolist():
                    name = str(info.filename or "")
                    if not name or name.endswith("/"):
                        continue
                    rel = Path(name)
                    if rel.is_absolute() or ".." in rel.parts:
                        raise ValueError("invalid archive path")
                    out = tmp_dir / rel
                    out.parent.mkdir(parents=True, exist_ok=True)
                    with zf.open(info, "r") as src, out.open("wb") as dst:
                        shutil.copyfileobj(src, dst)
                    try:
                        # Restore the original mtime from the zip entry so
                        # copy2 below carries it into the dataset (fresh
                        # extraction-time mtimes scrambled the order).
                        _dt = info.date_time
                        _ts = time.mktime(_dt + (0, 0, -1))
                        os.utime(out, (_ts, _ts))
                    except Exception:
                        pass

            manifest_path = tmp_dir / "manifest.json"
            if not manifest_path.exists():
                raise ValueError("missing manifest.json")
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            dataset_dir = str(manifest.get("dataset_dir") or "dataset")
            in_dataset = tmp_dir / dataset_dir
            if not in_dataset.exists():
                raise ValueError("missing dataset")

            self._project_reset(session_id=session_id)

            for p in in_dataset.rglob("*"):
                if p.is_dir():
                    continue
                rel = p.relative_to(in_dataset)
                out = dataset_root / rel
                out.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(p, out)

            # Authoritative sample order: zip mtimes only have 2-second
            # resolution, so burst captures (several samples per second)
            # would still shuffle — stamp monotonic mtimes in the saved
            # display order (newest-first → largest mtime).
            sample_order = manifest.get("sample_order") or {}
            if isinstance(sample_order, dict):
                _base_t = time.time()
                for _cls_name, _ordered in sample_order.items():
                    if not isinstance(_ordered, list) or not _ordered:
                        continue
                    _cls_dir = dataset_root / sanitize_class_name(str(_cls_name))
                    if not _cls_dir.exists():
                        continue
                    _t = _base_t
                    for _name in _ordered:
                        _p = _cls_dir / Path(str(_name)).name
                        if _p.exists():
                            try:
                                os.utime(_p, (_t, _t))
                            except Exception:
                                pass
                        _t -= 0.01

            classes_meta_in = tmp_dir / "tm_classes.json"
            classes_meta_out = self._classes_meta_path(dataset_root)
            if classes_meta_in.exists():
                shutil.copy2(classes_meta_in, classes_meta_out)
            else:
                classes = self._dataset_labels_load(in_dataset)
                if not classes:
                    dirs = [d.name for d in dataset_root.iterdir() if d.is_dir()]
                    classes = sorted([str(x) for x in dirs]) if dirs else ["Class 1", "Class 2"]
                self._classes_save(dataset_root, classes)

            processed_cache_in = tmp_dir / str(manifest.get("processed_cache_dir") or "")
            processed_cache_out = self._processed_cache_dir(dataset_root)
            if processed_cache_out.exists():
                shutil.rmtree(processed_cache_out)
            if processed_cache_in.exists():
                shutil.copytree(processed_cache_in, processed_cache_out)

            latest_in = tmp_dir / "tm_train_latest.json"
            latest_out = self._train_result_path(dataset_root)
            run_dir_hint = str(manifest.get("train_run_dir") or "")
            if latest_in.exists():
                meta = json.loads(latest_in.read_text(encoding="utf-8"))
                run_src = (tmp_dir / run_dir_hint) if run_dir_hint else None
                run_dst = workspace_root / "runs" / "latest"
                if run_src is not None and run_src.exists():
                    if run_dst.exists():
                        shutil.rmtree(run_dst)
                    run_dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copytree(run_src, run_dst)
                    if isinstance(meta, dict):
                        meta["run_dir"] = str(run_dst)
                        for key in ["keras_model_path", "tflite_path", "model_h_path", "model_cpp_path"]:
                            nm = Path(str(meta.get(key) or "")).name
                            cand = run_dst / nm if nm else None
                            if cand is not None and cand.exists():
                                meta[key] = str(cand)
                latest_out.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            project_state_in = tmp_dir / str(manifest.get("project_state") or "tm_project_state.json")
            project_state: Optional[Dict[str, Any]] = None
            if project_state_in.exists():
                try:
                    loaded = json.loads(project_state_in.read_text(encoding="utf-8"))
                    if isinstance(loaded, dict):
                        project_state = loaded
                except Exception:
                    project_state = None
            return self._project_state_payload(dataset_root, project_state=project_state)
        finally:
            if tmp_dir.exists():
                shutil.rmtree(tmp_dir)

    def _project_reset(self, session_id: str) -> Dict[str, Any]:
        with self._lock:
            cfg = self._configs.get(session_id)
        if cfg is None:
            raise RuntimeError("missing config")
        self._stop_live(session_id=session_id)
        with self._lock:
            self._active.pop(session_id, None)
            # Full per-session teardown: stale live frames, cached
            # interpreters and preview locks must not survive the reset
            # (the workspace they point at is being wiped).
            self._preview.pop(session_id, None)
            self._preview_locks.pop(session_id, None)
            self._train.pop(session_id, None)
            self._live.pop(f"{session_id}:webcam", None)
            self._live.pop(f"{session_id}:device", None)
        dataset_root = cfg.dataset_root
        workspace_root = dataset_root.parent
        classes_meta = self._classes_meta_path(dataset_root)
        processed_cache_dir = self._processed_cache_dir(dataset_root)
        latest = self._train_result_path(dataset_root)
        deployed = workspace_root / "deployed.json"
        if dataset_root.exists():
            for p in list(dataset_root.iterdir()):
                try:
                    if p.is_dir():
                        shutil.rmtree(p)
                    else:
                        p.unlink(missing_ok=True)
                except Exception:
                    continue
        if latest.exists():
            latest.unlink(missing_ok=True)
        # The deployed-model pin must die with the runs it points at —
        # a dangling pin used to hard-fail every /preview/predict and
        # freeze the preview on the old model.
        if deployed.exists():
            deployed.unlink(missing_ok=True)
        if classes_meta.exists():
            classes_meta.unlink(missing_ok=True)
        if processed_cache_dir.exists():
            shutil.rmtree(processed_cache_dir, ignore_errors=True)
        runs_dir = workspace_root / "runs"
        if runs_dir.exists():
            try:
                shutil.rmtree(runs_dir)
            except Exception:
                pass
        self._classes_save(dataset_root, ["Class 1", "Class 2"])
        return self._project_state_payload(dataset_root)

    def _project_train_cfg(
        self, dataset_root: Path, project_state: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        def _pick_cfg(raw: Any) -> Dict[str, Any]:
            src = raw if isinstance(raw, dict) else {}
            out: Dict[str, Any] = {}
            int_fields = {
                "batch_size": 16,
                "epochs": 10,
                "conv1_filters": 8,
                "conv2_filters": 16,
                "dense_units": 32,
            }
            float_fields = {
                "validation_split": 0.2,
                "learning_rate": 0.001,
            }
            for key, fallback in int_fields.items():
                if key not in src:
                    continue
                try:
                    out[key] = int(src.get(key))
                except Exception:
                    out[key] = int(fallback)
            for key, fallback in float_fields.items():
                if key not in src:
                    continue
                try:
                    out[key] = float(src.get(key))
                except Exception:
                    out[key] = float(fallback)
            preprocess_mode = str(src.get("preprocess_mode") or "").strip().lower()
            if preprocess_mode in {"auto_by_label", "manual_roi"}:
                out["preprocess_mode"] = preprocess_mode
            manual_roi = normalize_manual_roi(src.get("manual_roi"))
            if manual_roi is not None:
                out["manual_roi"] = list(manual_roi)
            class_preprocess = normalize_class_preprocess_map(src.get("class_preprocess"))
            if class_preprocess:
                out["class_preprocess"] = class_preprocess
            sample_preprocess = normalize_sample_preprocess_map(src.get("sample_preprocess"))
            if sample_preprocess:
                out["sample_preprocess"] = sample_preprocess
            return out

        if isinstance(project_state, dict):
            cfg = _pick_cfg(project_state.get("train_cfg"))
            if cfg:
                return cfg
        run_cfg = dataset_root.parent / "runs" / "latest" / "train_config.json"
        if run_cfg.exists():
            try:
                cfg = _pick_cfg(json.loads(run_cfg.read_text(encoding="utf-8")))
                if cfg:
                    return cfg
            except Exception:
                pass
        latest = self._train_result_path(dataset_root)
        if latest.exists():
            try:
                meta = json.loads(latest.read_text(encoding="utf-8"))
                if isinstance(meta, dict):
                    return _pick_cfg(meta)
            except Exception:
                pass
        return {}

    def _sample_delete(self, session_id: str, class_name: str, filename: str) -> Dict[str, Any]:
        with self._lock:
            cfg = self._configs.get(session_id)
        if cfg is None:
            raise RuntimeError("missing config")
        class_dir = (cfg.dataset_root / sanitize_class_name(class_name)).resolve()
        target = (class_dir / Path(filename).name).resolve()
        if class_dir != target.parent:
            raise ValueError("invalid sample path")
        if not target.exists():
            raise FileNotFoundError("sample not found")
        target.unlink()
        processed = self._processed_cache_dir(cfg.dataset_root) / sanitize_class_name(class_name) / (Path(filename).stem + ".png")
        processed.unlink(missing_ok=True)
        sample_preprocess = self._sample_preprocess_load(cfg.dataset_root)
        class_map = sample_preprocess.get(class_name)
        if isinstance(class_map, dict):
            class_map.pop(Path(filename).name, None)
            if not class_map:
                sample_preprocess.pop(class_name, None)
            self._classes_save(
                cfg.dataset_root,
                self._classes_load(cfg.dataset_root),
                class_preprocess=self._class_preprocess_load(cfg.dataset_root),
                sample_preprocess=sample_preprocess,
            )
        return self._class_state_payload(cfg.dataset_root, class_name)

    def _samples_clear(self, session_id: str, class_name: str) -> Dict[str, Any]:
        """Delete ALL samples of one class (the class itself stays)."""
        with self._lock:
            cfg = self._configs.get(session_id)
        if cfg is None:
            raise RuntimeError("missing config")
        class_dir = cfg.dataset_root / sanitize_class_name(class_name)
        if not class_dir.exists():
            raise FileNotFoundError("class not found")
        for p in list(class_dir.iterdir()):
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
                try:
                    p.unlink()
                except Exception:
                    pass
        processed_dir = self._processed_cache_dir(cfg.dataset_root) / sanitize_class_name(class_name)
        if processed_dir.exists():
            try:
                shutil.rmtree(processed_dir)
            except Exception:
                pass
        sample_preprocess = self._sample_preprocess_load(cfg.dataset_root)
        if isinstance(sample_preprocess, dict) and class_name in sample_preprocess:
            sample_preprocess.pop(class_name, None)
            self._classes_save(
                cfg.dataset_root,
                self._classes_load(cfg.dataset_root),
                class_preprocess=self._class_preprocess_load(cfg.dataset_root),
                sample_preprocess=sample_preprocess,
            )
        return self._class_state_payload(cfg.dataset_root, class_name)

    def _processed_preview_item_payload(self, path: Path) -> Dict[str, str]:
        return _preview_item_payload(path)

    def _cache_one_sample(self, dataset_root: Path, class_name: str, sample_path: Path, img_size: int = 96) -> None:
        """Process one newly saved sample into the processed cache so the
        class-edit previews and the next training see it immediately."""
        try:
            class_cfg = self._class_preprocess_load(dataset_root).get(class_name)
            config = class_cfg if isinstance(class_cfg, dict) else {"mode": "auto_by_label"}
            out_png = self._preprocess_image_png(
                sample_path.read_bytes(),
                label_name=class_name,
                class_config=config,
                fast_mode=False,
                out_size=int(img_size),
            )
            dst_dir = self._processed_cache_dir(dataset_root) / sanitize_class_name(class_name)
            dst_dir.mkdir(parents=True, exist_ok=True)
            (dst_dir / (sample_path.stem + ".png")).write_bytes(out_png)
        except Exception:
            pass

    def _preprocess_image_png(self, png: bytes, label_name: str, class_config: Any, sample_config: Any = None, return_crop: bool = False, fast_mode: bool = False, out_size: int = 96):
        class_cfg = class_config if isinstance(class_config, dict) else {}
        src_cfg = sample_config if isinstance(sample_config, dict) else {}
        config = normalize_class_preprocess(sample_config if sample_config is not None else class_config)
        mode = str(config.get("mode") or PREPROCESS_MODE_AUTO_BY_LABEL)
        # User-adjustable thresholds from class-level config (shared across samples)
        bg_dark_thresh = int(class_cfg.get("bg_dark_thresh", 0)) if class_cfg else 0
        bg_lum_thresh  = int(class_cfg.get("bg_lum_thresh", 100)) if class_cfg else 100
        # Always B-G; keep RGB colour information.
        img = Image.open(_bytes_io(png)).convert("RGB")
        roi = None
        if mode == PREPROCESS_MODE_MANUAL_ROI:
            src = np.asarray(img)
            roi = manual_roi_to_pixels(src.shape[0], src.shape[1], config.get("manual_roi"))
        result = preprocess_blue_diff_array(np.asarray(img), out_size=int(out_size), roi=roi,
                                         bg_dark_thresh=bg_dark_thresh,
                                         bg_lum_thresh=bg_lum_thresh,
                                         return_crop_box=True,
                                         fast_mode=fast_mode)
        if isinstance(result, tuple) and len(result) >= 5:
            arr, crop_norm, masked_preview, crop_preview, full_preview = result
        elif isinstance(result, tuple) and len(result) >= 4:
            arr, crop_norm, masked_preview, crop_preview = result
            full_preview = None
        elif isinstance(result, tuple) and len(result) == 3:
            arr, crop_norm, masked_preview = result
            crop_preview = None
            full_preview = None
        else:
            arr, crop_norm = result
            masked_preview = None
            crop_preview = None
            full_preview = None
        out = np.asarray(np.clip(arr * 255.0, 0.0, 255.0), dtype=np.uint8)
        if out.ndim == 3:
            out = out[:, :, 0]
        png_bytes = _to_png_bytes(Image.fromarray(out, mode="L"))
        # Build a displayable version of the masked preview for the UI
        if masked_preview is not None:
            mp_arr = np.asarray(np.clip(masked_preview[:,:,0] * 255.0, 0, 255), dtype=np.uint8) if masked_preview.ndim == 3 else np.asarray(np.clip(masked_preview * 255.0, 0, 255), dtype=np.uint8)
            mp_png = _to_png_bytes(Image.fromarray(mp_arr, mode="L"))
        else:
            mp_png = png_bytes
        # Crop-only preview (no threshold)
        if crop_preview is not None:
            cp_arr = np.asarray(np.clip(crop_preview[:,:,0] * 255.0, 0, 255), dtype=np.uint8) if crop_preview.ndim == 3 else np.asarray(np.clip(crop_preview * 255.0, 0, 255), dtype=np.uint8)
            cp_png = _to_png_bytes(Image.fromarray(cp_arr, mode="L"))
        else:
            cp_png = png_bytes
        if return_crop:
            return {"png": png_bytes, "crop": list(crop_norm), "processed_png": mp_png, "crop_png": cp_png}
        return png_bytes

    def _rebuild_processed_cache(
        self,
        dataset_root: Path,
        class_preprocess: Optional[Dict[str, Dict[str, Any]]] = None,
        sample_preprocess: Optional[Dict[str, Dict[str, Dict[str, Any]]]] = None,
        global_preprocess_mode: str = PREPROCESS_MODE_AUTO_BY_LABEL,
        out_size: int = 96,
        fast_mode: bool = True,
    ) -> None:
        classes = self._classes_load(dataset_root)
        cache_root = self._processed_cache_dir(dataset_root)
        cache_root.mkdir(parents=True, exist_ok=True)
        class_preprocess = normalize_class_preprocess_map(class_preprocess if class_preprocess is not None else self._class_preprocess_load(dataset_root))
        sample_preprocess = normalize_sample_preprocess_map(sample_preprocess if sample_preprocess is not None else self._sample_preprocess_load(dataset_root))
        for class_name in classes:
            src_dir = dataset_root / sanitize_class_name(class_name)
            dst_dir = cache_root / sanitize_class_name(class_name)
            if dst_dir.exists():
                shutil.rmtree(dst_dir)
            dst_dir.mkdir(parents=True, exist_ok=True)
            if not src_dir.exists():
                continue
            raw_cfg = class_preprocess.get(class_name, {"mode": global_preprocess_mode})
            config = dict(raw_cfg) if isinstance(raw_cfg, dict) else {"mode": global_preprocess_mode}
            class_sample_map = sample_preprocess.get(class_name, {})
            processed_count = 0
            for src in _list_class_image_files(src_dir):
                try:
                    png = src.read_bytes()
                    sample_cfg = class_sample_map.get(src.name)
                    out_png = self._preprocess_image_png(
                        png,
                        label_name=class_name,
                        class_config=config,
                        fast_mode=fast_mode,
                        sample_config=sample_cfg,
                        out_size=int(out_size),
                    )
                    (dst_dir / f"{src.stem}.png").write_bytes(out_png)
                    processed_count += 1
                except Exception as e:
                    import logging
                    logging.warning(f"preprocess failed for {src.name}: {e}")
            if processed_count == 0 and len(list(_list_class_image_files(src_dir))) > 0:
                import logging
                logging.warning(f"No files processed for class '{class_name}' — check preprocess mode '{config.get('mode')}'")

    def _processed_previews_payload(self, dataset_root: Path, classes: List[str]) -> Dict[str, List[Dict[str, str]]]:
        out: Dict[str, List[Dict[str, str]]] = {}
        cache_root = self._processed_cache_dir(dataset_root)
        for class_name in classes:
            items: List[Dict[str, str]] = []
            class_dir = cache_root / sanitize_class_name(class_name)
            if class_dir.exists():
                for p in sorted(class_dir.glob("*.png"))[:12]:
                    try:
                        items.append(self._processed_preview_item_payload(p))
                    except Exception:
                        continue
            out[class_name] = items
        return out

    def _preprocess_preview(
        self,
        session_id: str,
        class_name: str,
        filename: str,
        class_config: Any,
        sample_config: Any = None,
    ) -> Dict[str, Any]:
        with self._lock:
            cfg = self._configs.get(session_id)
        if cfg is None:
            raise RuntimeError("missing config")
        class_dir = cfg.dataset_root / sanitize_class_name(class_name)
        src = (class_dir / Path(filename).name).resolve()
        if src.parent != class_dir.resolve() or not src.exists():
            raise FileNotFoundError("sample not found")
        result = self._preprocess_image_png(
            src.read_bytes(),
            label_name=class_name,
            class_config=class_config,
            sample_config=sample_config,
            return_crop=True,
            out_size=int(cfg.img_size),
        )
        return {
            "image_b64": base64.b64encode(result["png"]).decode("ascii"),
            "crop": result.get("crop"),
            "processed_image_b64": base64.b64encode(result.get("processed_png", result["png"])).decode("ascii"),
        }

    def _project_state_payload(self, dataset_root: Path, project_state: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        classes = self._classes_load(dataset_root)
        class_preprocess = self._class_preprocess_load(dataset_root)
        sample_preprocess = self._sample_preprocess_load(dataset_root)
        if isinstance(project_state, dict):
            class_preprocess = normalize_class_preprocess_map(project_state.get("class_preprocess") or class_preprocess)
            sample_preprocess = normalize_sample_preprocess_map(project_state.get("sample_preprocess") or sample_preprocess)
        counts: Dict[str, int] = {}
        previews: Dict[str, List[Dict[str, str]]] = {}
        for c in classes:
            c_dir = dataset_root / sanitize_class_name(c)
            cnt = 0
            thumbs: List[Dict[str, str]] = []
            if c_dir.exists():
                try:
                    files = _list_class_image_files(c_dir)
                    cnt = len(files)
                    for p in files[:12]:
                        try:
                            thumbs.append(_preview_item_payload(p))
                        except Exception:
                            continue
                except Exception:
                    cnt = 0
            counts[c] = int(cnt)
            previews[c] = thumbs
        latest = self._train_result_path(dataset_root)
        export_enabled = False
        if latest.exists():
            try:
                meta = json.loads(latest.read_text(encoding="utf-8"))
                tflite_path = Path(str(meta.get("tflite_path") or "")).expanduser().resolve() if isinstance(meta, dict) else Path()
                export_enabled = bool(tflite_path.exists())
            except Exception:
                export_enabled = False
        train_cfg = self._project_train_cfg(dataset_root, project_state=project_state)
        processed_previews = self._processed_previews_payload(dataset_root, classes)
        payload = {
            "classes": list(classes),
            "counts": counts,
            "sample_previews": previews,
            "processed_previews": processed_previews,
            "class_preprocess": class_preprocess,
            "sample_preprocess": sample_preprocess,
            "export_enabled": "1" if export_enabled else "0",
        }
        if train_cfg:
            payload["train_cfg"] = train_cfg
        # Preserve dark/lum thresholds from project_state so they survive save/restore
        if isinstance(project_state, dict):
            if "bg_dark_thresh" in project_state:
                payload["bg_dark_thresh"] = int(project_state["bg_dark_thresh"])
            if "bg_lum_thresh" in project_state:
                payload["bg_lum_thresh"] = int(project_state["bg_lum_thresh"])
        return payload

    def _preview_model_lock(self, session_id: str) -> threading.Lock:
        with self._lock:
            lock = self._preview_locks.get(session_id)
            if lock is None:
                lock = threading.Lock()
                self._preview_locks[session_id] = lock
            return lock

    def _preview_load_model(self, session_id: str) -> Dict[str, Any]:
        with self._lock:
            cfg = self._configs.get(session_id)
        if cfg is None:
            raise RuntimeError("missing config")
        latest = self._train_result_path(cfg.dataset_root)
        # Preview prefers the DEPLOYED model (the one exported to the
        # device) when a deployed.json stamp exists — but ONLY while the
        # stamp is valid.  A dangling pin (exported run wiped by project
        # reset/open) or an unreadable meta used to hard-fail every
        # /preview/predict and freeze the preview on the old model, so any
        # invalid pin falls back to the latest training.
        deployed = cfg.dataset_root.parent / "deployed.json"
        src = None
        meta: Dict[str, Any] = {}
        if deployed.exists():
            try:
                dmeta = json.loads(deployed.read_text(encoding="utf-8"))
                dpath = Path(str((dmeta or {}).get("tflite_path") or "")).expanduser()
                if dpath.exists():
                    src = deployed
                    meta = dmeta if isinstance(dmeta, dict) else {}
            except Exception:
                src = None
        if src is None:
            src = latest
            if src.exists():
                try:
                    meta = json.loads(src.read_text(encoding="utf-8"))
                except Exception:
                    raise RuntimeError("trained model meta is unreadable — retrain or re-export")
        if not src.exists():
            raise RuntimeError("missing trained model")
        if not meta:
            raise RuntimeError("missing trained model meta — retrain or re-export")
        tflite_path = Path(str(meta.get("tflite_path") or "")).expanduser()
        if not tflite_path.exists():
            raise RuntimeError("missing .tflite file — retrain or re-export")
        model_source = "deployed" if src == deployed else "latest"
        labels = meta.get("labels") if isinstance(meta, dict) else None
        if not isinstance(labels, list) or not all(isinstance(x, str) for x in labels):
            labels = []
        mtime = float(tflite_path.stat().st_mtime)
        with self._lock:
            cur = self._preview.get(session_id)
            if cur is not None and str(cur.get("tflite_path") or "") == str(tflite_path) and float(cur.get("tflite_mtime") or 0.0) == mtime:
                return cur
        import tensorflow as tf

        interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()[0]
        output_details = interpreter.get_output_details()[0]
        cur = {
            "tflite_path": str(tflite_path),
            "tflite_mtime": mtime,
            "model_source": str(model_source),
            "labels": list(labels),
            "preprocess_mode": str(meta.get("preprocess_mode") or PREPROCESS_MODE_AUTO_BY_LABEL) if isinstance(meta, dict) else PREPROCESS_MODE_AUTO_BY_LABEL,
            "crop_mode": str(meta.get("crop_mode") or CROP_MODE_CENTER) if isinstance(meta, dict) else CROP_MODE_CENTER,
            "manual_roi": normalize_manual_roi(meta.get("manual_roi")) if isinstance(meta, dict) else None,
            "class_preprocess": normalize_class_preprocess_map(meta.get("class_preprocess")) if isinstance(meta, dict) else {},
            "interpreter": interpreter,
            "input_details": input_details,
            "output_details": output_details,
        }
        with self._lock:
            self._preview[session_id] = cur
        return cur

    def _preview_predict(
        self,
        session_id: str,
        png: bytes,
        preprocess_mode: str = "",
        bg_dark_thresh: Optional[int] = None,
        bg_lum_thresh: Optional[int] = None,
        ood_sign_pct_min: Optional[float] = None,
        ood_sign_pct_max: Optional[float] = None,
        ood_max_prob_min: Optional[float] = None,
        ood_entropy_ratio_max: Optional[float] = None,
        source: str = "",
    ) -> Dict[str, Any]:
        model = self._preview_load_model(session_id=session_id)
        interpreter = model["interpreter"]
        input_details = model["input_details"]
        output_details = model["output_details"]
        labels: List[str] = list(model.get("labels") or [])
        model_preprocess_mode = str(model.get("preprocess_mode") or PREPROCESS_MODE_AUTO_BY_LABEL)
        preprocess_mode = str(preprocess_mode or model_preprocess_mode or PREPROCESS_MODE_AUTO_BY_LABEL).strip().lower()
        # The model defines the crop: new trainings save "auto_search"
        # (shadow-search box = model input, device BG_ENABLE_FOCUS_SEARCH=1);
        # old deployments without the field keep the legacy center crop.
        model_crop_mode = str(model.get("crop_mode") or CROP_MODE_CENTER)
        manual_roi = normalize_manual_roi(model.get("manual_roi"))
        class_preprocess = normalize_class_preprocess_map(model.get("class_preprocess"))
        shape_raw = input_details.get("shape")
        shape = tuple(int(x) for x in (shape_raw if shape_raw is not None else []))
        if len(shape) != 4:
            raise RuntimeError("unsupported input shape")
        _, h, w, c = shape
        # Always RGB — B-G difference is computed inside the pipeline.
        img = Image.open(_bytes_io(png)).convert("RGB")
        # ── Device grayscale streams ARE the device's tensor input ──────
        # TFLite.ino kCaptureGray / kInferGray stream `input->data.int8 + 128`
        # (already center-cropped + stretched).  Re-running
        # prepare_inference_inputs on that frame double-processes it and the
        # host never sees what the device actually fed the model.  When the
        # source is the device and the stream is 1-channel, feed the gray
        # frame straight to the interpreter (int8 = gray − 128) so Preview
        # and device evaluate the EXACT same tensor.
        direct_feed = False
        if source == "device":
            with self._lock:
                cfg = self._configs.get(session_id)
            if cfg is not None and bool(getattr(cfg, "direct_tensor_stream", False)):
                direct_feed = True
        if direct_feed:
            gray = np.asarray(img.convert("L"), dtype=np.uint8)
            if gray.shape != (int(h), int(w)):
                gray = np.asarray(
                    Image.fromarray(gray).resize((int(w), int(h)), Image.BILINEAR),
                    dtype=np.uint8,
                )
            gray_f = gray.astype(np.float32) / 255.0
            arr = np.expand_dims(gray_f, axis=-1)
            d = int(bg_dark_thresh) if bg_dark_thresh is not None else 0
            l = int(bg_lum_thresh) if bg_lum_thresh is not None else 100
            is_sign = (gray > d) & (gray < l)
            # NOTE: sign_pct is computed on the streamed (cropped) frame; the
            # device computes it on the full pre-crop frame, so the two L1
            # gates are close but not identical.
            # Preview variants for the ROI/Orig toggles keep the pre-direct-
            # feed visuals: rebuild them with the normal prepare_inference_
            # inputs pipeline (mask / crop / full-frame views) so the buttons
            # behave exactly as before.  Only the MODEL input ("default") is
            # the direct-fed stream — that part must NOT be re-processed.
            try:
                legacy_views = prepare_inference_inputs(
                    np.asarray(img),
                    out_size=int(w),
                    color_mode="grayscale",
                    preprocess_mode=preprocess_mode,
                    manual_roi=manual_roi,
                    class_preprocess=class_preprocess,
                    bg_dark_thresh=d,
                    bg_lum_thresh=l,
                    crop_mode=model_crop_mode,
                )
            except Exception:
                legacy_views = {}
            prepared = {
                "default": arr,
                "masked_preview": legacy_views.get("masked_preview", arr),
                "crop_preview": legacy_views.get("crop_preview", arr),
                "full_preview": legacy_views.get("full_preview", arr),
                "search_box": legacy_views.get("search_box"),
                "mask_stats": {
                    "sign_pct": float(is_sign.mean() * 100.0),
                    "too_dark_pct": float((gray <= d).mean() * 100.0),
                    "too_bright_pct": float((gray >= l).mean() * 100.0),
                    "bg_dark_thresh": d,
                    "bg_lum_thresh": l,
                },
            }
        else:
            prepared = prepare_inference_inputs(
                np.asarray(img),
                out_size=int(w),
                color_mode="grayscale" if int(c) == 1 else "rgb",
                preprocess_mode=preprocess_mode,
                manual_roi=manual_roi,
                class_preprocess=class_preprocess,
                bg_dark_thresh=(int(bg_dark_thresh) if bg_dark_thresh is not None else None),
                bg_lum_thresh=(int(bg_lum_thresh)  if bg_lum_thresh  is not None else None),
                crop_mode=model_crop_mode,
            )
        qscale = 0.0
        qzp = 0
        q = input_details.get("quantization")
        if isinstance(q, (tuple, list)) and len(q) == 2:
            qscale = float(q[0] or 0.0)
            qzp = int(q[1] or 0)
        dtype = input_details.get("dtype")

        def _invoke_one(arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
            batch = np.expand_dims(arr, axis=0)
            if dtype is not None and dtype != np.float32 and qscale > 0:
                qarr = np.round(batch / qscale + qzp)
                if dtype == np.int8:
                    x = np.clip(qarr, -128, 127).astype(np.int8)
                elif dtype == np.uint8:
                    x = np.clip(qarr, 0, 255).astype(np.uint8)
                else:
                    x = qarr.astype(dtype)
            else:
                x = batch.astype(np.float32)
            out_idx = int(output_details.get("index"))
            in_idx = int(input_details.get("index"))
            with self._preview_model_lock(session_id):
                interpreter.set_tensor(in_idx, x)
                interpreter.invoke()
                out = interpreter.get_tensor(out_idx)
            out = np.asarray(out).reshape((-1,))
            oscale = 0.0
            ozp = 0
            oq = output_details.get("quantization")
            if isinstance(oq, (tuple, list)) and len(oq) == 2:
                oscale = float(oq[0] or 0.0)
                ozp = int(oq[1] or 0)
            odtype = output_details.get("dtype")
            # Raw device-mirror scores: TFLite.ino ood_is_in_distribution /
            # pick_label_and_confidence use (int8 + 128) / (uint8) / (f*255),
            # clamped to [0,255], and derive max_prob + entropy from THOSE
            # values — NOT from dequantised softmax probabilities.  Return
            # the raw tensor so the OOD L2/L3 gates below fire on exactly
            # the same numbers the firmware computes.
            if odtype == np.int8:
                raw_out = np.clip(out.astype(np.int32) + 128, 0, 255).astype(np.int32)
            elif odtype == np.uint8:
                raw_out = out.astype(np.int32)
            else:
                raw_out = np.clip((out.astype(np.float32) * 255.0).astype(np.int32), 0, 255)
            if odtype is not None and odtype != np.float32 and oscale > 0:
                scores = (out.astype(np.float32) - float(ozp)) * float(oscale)
            else:
                scores = out.astype(np.float32)
            # Auto-detect softmax:
            # - If scores sum to ~1 → model already has softmax → use as-is
            # - Otherwise → int8 conversion stripped the softmax op → apply it here
            if abs(float(np.sum(scores)) - 1.0) > 0.1:
                scores = scores - float(np.max(scores))
                expv = np.exp(scores)
                scores = expv / float(np.sum(expv) + 1e-12)
            return np.clip(scores, 0.0, 1.0), raw_out

        variant_probs: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        mask_stats: Any = prepared.get("mask_stats") if isinstance(prepared, dict) else None
        for name, arr in prepared.items():
            if name == "mask_stats":
                continue
            if not isinstance(arr, np.ndarray):
                continue
            try:
                variant_probs[name] = _invoke_one(arr)
            except Exception:
                pass
        probs = None
        raw_out = None
        for k in ("default", "crop_preview", "masked_preview", "full_preview"):
            if k in variant_probs:
                probs, raw_out = variant_probs[k]
                break
        if probs is None and variant_probs:
            probs, raw_out = next(iter(variant_probs.values()))
        if probs is None:
            raise RuntimeError("no usable processed arrays for inference")

        if not labels:
            labels = [f"Class {i+1}" for i in range(int(probs.shape[0]))]
        top_i = int(np.argmax(probs)) if probs.size else 0
        max_prob = float(probs[top_i]) if probs.size else 0.0
        n_classes = max(1, int(probs.shape[0]))
        # Device-exact OOD L2/L3 values (TFLite.ino ood_is_in_distribution):
        # max_prob = raw_best / Σraw, entropy over p = raw/Σraw with natural
        # log, ratio normalised by ln(N) and clamped to 1.0.  float32
        # arithmetic and op order replicate the firmware so host and device
        # reject the same frames.
        raw_total = int(np.sum(raw_out)) if raw_out is not None else 0
        if raw_total <= 0:
            raw_total = 1
        ood_max_prob = float(np.float32(np.max(raw_out)) / np.float32(raw_total)) if raw_out is not None else 0.0
        ood_entropy = np.float32(0.0)
        if raw_out is not None:
            f_total = np.float32(raw_total)
            for v in raw_out:
                p = np.float32(v) / f_total
                if float(p) <= 1e-6:
                    continue
                ood_entropy = ood_entropy - p * np.float32(math.log(float(p)))
        ood_entropy = float(ood_entropy)
        max_entropy = float(math.log(n_classes)) if n_classes > 1 else 1.0
        ood_entropy_ratio = float(np.float32(ood_entropy) / np.float32(max_entropy)) if max_entropy > 0 else 1.0
        ood_entropy_ratio = min(1.0, ood_entropy_ratio)
        # Keep the softmax-side values for the OOD gate names below (they are
        # what the UI labels refer to), but gate on the device-math values.
        max_prob = ood_max_prob
        entropy_ratio = ood_entropy_ratio

        sign_pct = 0.0
        if isinstance(mask_stats, dict):
            try:
                sign_pct = float(mask_stats.get("sign_pct") or 0.0)
            except Exception:
                sign_pct = 0.0

        NO_SIGN_LABEL = "No Sign"

        # Layer 1: sign-mask presence prior (strongest signal, no inference cost)
        # Expected for our training signs: ~2% to ~40% of the (full-frame) pixels
        # have G in [dark, lum).  If too few → empty scene / overexposed; if too
        # many → all-black / all-shadow frame.  Both are clearly "no sign".
        SIGN_PCT_MIN = float(ood_sign_pct_min if ood_sign_pct_min is not None else 0.3)
        SIGN_PCT_MAX = float(ood_sign_pct_max if ood_sign_pct_max is not None else 70.0)

        # Layer 2: softmax calibration.  OOD inputs often get "confidently wrong"
        # by softmax, but combining MAX_PROB threshold with ENTROPY (how "spread
        # out" the votes are) works reliably on small CNNs.  If the net is
        # confident (high prob) AND the decision is sharp (low entropy) → we
        # trust it; otherwise suppress to "No Sign".
        MAX_PROB_MIN = float(ood_max_prob_min if ood_max_prob_min is not None else 0.60)
        ENTROPY_RATIO_MAX = float(ood_entropy_ratio_max if ood_entropy_ratio_max is not None else 0.70)  # 1.0 = uniform / perfectly uncertain
        SIGN_PCT_MIN = max(0.0, min(100.0, SIGN_PCT_MIN))
        SIGN_PCT_MAX = max(0.0, min(100.0, SIGN_PCT_MAX))
        MAX_PROB_MIN = max(0.0, min(1.0, MAX_PROB_MIN))
        ENTROPY_RATIO_MAX = max(0.0, min(1.0, ENTROPY_RATIO_MAX))
        if SIGN_PCT_MIN > SIGN_PCT_MAX:
            SIGN_PCT_MIN, SIGN_PCT_MAX = SIGN_PCT_MAX, SIGN_PCT_MIN

        is_ood = False
        suppress_reason = ""
        if sign_pct < SIGN_PCT_MIN or sign_pct > SIGN_PCT_MAX:
            is_ood = True
            suppress_reason = f"mask sign_pct={sign_pct:.1f}% outside [{SIGN_PCT_MIN:.1f},{SIGN_PCT_MAX:.1f}]"
        elif max_prob < MAX_PROB_MIN:
            is_ood = True
            suppress_reason = f"max_prob={max_prob:.3f} < {MAX_PROB_MIN:.2f}"
        elif entropy_ratio > ENTROPY_RATIO_MAX:
            is_ood = True
            suppress_reason = f"entropy_ratio={entropy_ratio:.3f} > {ENTROPY_RATIO_MAX:.2f}"

        if is_ood:
            # Suppress: declare "No Sign", keep raw probs for debug, push top_prob
            # down to the highest "unconfident" level so downstream decisions don't fire.
            top_label_out = NO_SIGN_LABEL
            top_prob_out = float(0.0)
            labels_out = list(labels) + [NO_SIGN_LABEL]
            probs_out = [float(0.0)] * (len(labels) + 1)
            probs_out[-1] = 1.0
        else:
            top_label_out = str(labels[top_i]) if 0 <= top_i < len(labels) else ""
            top_prob_out = float(max_prob)
            labels_out = list(labels)
            probs_out = [float(x) for x in probs.tolist()]

        # Build a displayable preview of the after-processed model input
        processed_variant = next(iter(prepared.keys()), "")
        # Use masked_preview (thresholded G-channel) for the ROI toggle so the
        # user can see dark/lum threshold effects.  Falls back to model-input array.
        if "masked_preview" in prepared:
            processed_arr = np.asarray(prepared["masked_preview"])
            processed_variant = "masked"
        else:
            processed_arr = np.asarray(prepared.get(processed_variant) if processed_variant in prepared else next(iter(prepared.values())))
        # Crop-only preview (no threshold)
        if "crop_preview" in prepared:
            crop_arr = np.asarray(prepared["crop_preview"])
            crop_png = _to_png_bytes(_model_input_array_to_preview_image(crop_arr))
        else:
            crop_png = None
        # Full-frame thresholded preview (no crop) — for the Orig view
        if "full_preview" in prepared:
            full_arr = np.asarray(prepared["full_preview"])
            full_png = _to_png_bytes(_model_input_array_to_preview_image(full_arr))
        else:
            full_png = None
        processed_img = _model_input_array_to_preview_image(processed_arr)
        processed_png = _to_png_bytes(processed_img)
        # Auto search box (normalized 0-1) for the preview ROI overlay —
        # same contract as the class-edit page ("crop" field).
        search_box: Any = None
        if isinstance(prepared, dict):
            sb = prepared.get("search_box")
            if isinstance(sb, (list, tuple)) and len(sb) == 4:
                search_box = [float(x) for x in sb]
        return {
            "labels": labels_out,
            "probs": probs_out,
            "top_label": top_label_out,
            "top_prob": float(top_prob_out),
            "crop": search_box,
            "model_source": str(model.get("model_source") or "latest"),
            "processed_image_b64": base64.b64encode(processed_png).decode("ascii"),
            "crop_image_b64": base64.b64encode(crop_png).decode("ascii") if crop_png else "",
            "full_image_b64": base64.b64encode(full_png).decode("ascii") if full_png else "",
            "processed_variant": str(processed_variant or ""),
            "ood": bool(is_ood),
            "ood_reason": str(suppress_reason or ""),
            "sign_pct": float(sign_pct),
            "max_prob": float(max_prob),
            "entropy_ratio": float(entropy_ratio),
        }

    def _start_train(self, session_id: str, cfg: Any) -> None:
        with self._lock:
            sess_cfg = self._configs.get(session_id)
            if sess_cfg is None:
                raise RuntimeError("missing config")
            cur = self._train.get(session_id)
            if cur is not None and str(cur.get("running") or "0") == "1":
                return
            self._train[session_id] = {
                "running": "1",
                "done": "0",
                "progress": 0.0,
                "message": "Starting...",
                "error": "",
                "result_path": "",
                "updated_at": time.time(),
            }
            if session_id not in self._train_conds:
                self._train_conds[session_id] = threading.Condition()

        t = threading.Thread(target=self._train_worker, args=(session_id, dict(cfg or {})), daemon=True)
        t.start()

    def _save_train_cfg(self, session_id: str, cfg: Any) -> None:
        """Persist training config (e.g. applied recommendations) so the next
        page render picks it up. Merges with the existing train_config.json."""
        with self._lock:
            sess_cfg = self._configs.get(session_id)
        if sess_cfg is None:
            raise RuntimeError("missing config")
        run_dir = Path(sess_cfg.dataset_root).parent / "runs" / "latest"
        run_dir.mkdir(parents=True, exist_ok=True)
        cfg_path = run_dir / "train_config.json"
        existing: Dict[str, Any] = {}
        if cfg_path.exists():
            try:
                loaded = json.loads(cfg_path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    existing = loaded
            except Exception:
                existing = {}
        merged = dict(existing)
        if isinstance(cfg, dict):
            for k, v in cfg.items():
                if v is not None:
                    merged[str(k)] = v
        cfg_path.write_text(json.dumps(merged, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    def _train_recommend(self, session_id: str, cfg: Any) -> Dict[str, Any]:
        """Recompute dataset-aware recommendations from the CURRENT sample count."""
        from trainer import TrainConfig, recommend_train_params

        with self._lock:
            sess_cfg = self._configs.get(session_id)
        if sess_cfg is None:
            raise RuntimeError("missing config")
        dataset_root = Path(sess_cfg.dataset_root)
        total = 0
        if dataset_root.exists():
            for class_dir in dataset_root.iterdir():
                if class_dir.is_dir():
                    total += len(_list_class_image_files(class_dir))
        src = cfg if isinstance(cfg, dict) else {}
        train_cfg = TrainConfig(
            batch_size=int(src.get("batch_size") or 16),
            epochs=int(src.get("epochs") or 10),
            validation_split=float(src.get("validation_split") or 0.2),
            learning_rate=float(src.get("learning_rate") or 0.001),
        )
        rec_kwargs, reasons = recommend_train_params(total, train_cfg)
        return {
            "train_rec": {
                "total_samples": int(total),
                "batch_size": int(rec_kwargs["batch_size"]),
                "epochs": int(rec_kwargs["epochs"]),
                "learning_rate": float(rec_kwargs["learning_rate"]),
                "validation_split": float(rec_kwargs["validation_split"]),
                "reasons": {
                    "batch": str(reasons[0]),
                    "epochs": str(reasons[1]),
                    "lr": str(reasons[2]),
                    "val_split": str(reasons[3]),
                },
            }
        }

    def _train_status_payload(self, session_id: str) -> Dict[str, Any]:
        with self._lock:
            cur = self._train.get(session_id) or {"running": "0", "done": "0", "progress": 0.0, "message": "", "error": "", "result_path": ""}
            payload = {
                "ok": "1",
                "running": str(cur.get("running") or "0"),
                "done": str(cur.get("done") or "0"),
                "progress": float(cur.get("progress") or 0.0),
                "message": str(cur.get("message") or ""),
                "error": str(cur.get("error") or ""),
                "result_path": str(cur.get("result_path") or ""),
                "class_accuracies": {},
                "warnings": [],
            }
        # Per-class accuracies + self-check warnings from the latest training
        # meta — the SPA renders them in the training panel so weak classes
        # are visible on the interface, not just in the logs.
        try:
            cfg = self._configs.get(session_id)
            if cfg is not None:
                latest = self._train_result_path(cfg.dataset_root)
                if latest.exists():
                    meta = json.loads(latest.read_text(encoding="utf-8"))
                    if isinstance(meta, dict):
                        payload["class_accuracies"] = meta.get("class_accuracies") or {}
                        payload["warnings"] = list(meta.get("warnings") or [])
        except Exception:
            pass
        return payload

    def _train_set(self, session_id: str, *, progress: Optional[float] = None, message: Optional[str] = None, error: Optional[str] = None, done: Optional[bool] = None, result_path: Optional[str] = None) -> None:
        with self._lock:
            cur = self._train.get(session_id)
            if cur is None:
                cur = {"running": "0", "done": "0", "progress": 0.0, "message": "", "error": "", "result_path": ""}
                self._train[session_id] = cur
            if progress is not None:
                cur["progress"] = max(0.0, min(1.0, float(progress)))
            if message is not None:
                cur["message"] = str(message)
            if error is not None:
                cur["error"] = str(error)
            if done is not None:
                cur["done"] = "1" if done else "0"
                if done:
                    cur["running"] = "0"
            if result_path is not None:
                cur["result_path"] = str(result_path)
            cur["updated_at"] = time.time()
        cond = self._train_conds.get(session_id)
        if cond is not None:
            with cond:
                cond.notify_all()

    def _train_worker(self, session_id: str, cfg_dict: Dict[str, Any]) -> None:
        with self._lock:
            sess_cfg = self._configs.get(session_id)
        if sess_cfg is None:
            self._train_set(session_id, error="missing config", done=True)
            return
        try:
            from trainer import TrainConfig, TrainResult, new_run_dir, train_and_export

            project_classes = self._classes_load(sess_cfg.dataset_root)
            project_class_preprocess = self._class_preprocess_load(sess_cfg.dataset_root)
            project_sample_preprocess = self._sample_preprocess_load(sess_cfg.dataset_root)

            raw_class_preprocess = cfg_dict.get("class_preprocess")
            if raw_class_preprocess is None or (isinstance(raw_class_preprocess, dict) and not raw_class_preprocess):
                raw_class_preprocess = project_class_preprocess
            raw_sample_preprocess = cfg_dict.get("sample_preprocess")
            if raw_sample_preprocess is None or (isinstance(raw_sample_preprocess, dict) and not raw_sample_preprocess):
                raw_sample_preprocess = project_sample_preprocess

            cfg = TrainConfig(
                img_size=int(cfg_dict.get("img_size") or 96),
                color_mode="grayscale",
                batch_size=int(cfg_dict.get("batch_size") or 16),
                epochs=int(cfg_dict.get("epochs") or 10),
                validation_split=float(cfg_dict.get("validation_split") or 0.2),
                seed=int(cfg_dict.get("seed") or 42),
                optimizer=str(cfg_dict.get("optimizer") or "adam"),
                learning_rate=float(cfg_dict.get("learning_rate") or 0.001),
                conv1_filters=int(cfg_dict.get("conv1_filters") or 32),
                conv2_filters=int(cfg_dict.get("conv2_filters") or 64),
                dense_units=int(cfg_dict.get("dense_units") or 128),
                representative_samples=int(cfg_dict.get("representative_samples") or 200),
                preprocess_mode=str(cfg_dict.get("preprocess_mode") or PREPROCESS_MODE_AUTO_BY_LABEL),
                crop_mode=str(cfg_dict.get("crop_mode") or CROP_MODE_AUTO_SEARCH),
                manual_roi=normalize_manual_roi(cfg_dict.get("manual_roi")),
                class_preprocess=normalize_class_preprocess_map(raw_class_preprocess),
                sample_preprocess=normalize_sample_preprocess_map(raw_sample_preprocess),
                use_preprocessed_dataset=True,
                class_names_order=list(project_classes) if project_classes else None,
                min_steps_per_epoch=int(cfg_dict.get("min_steps_per_epoch") or 4),
                auto_adjust_batch_and_epochs=bool(cfg_dict.get("auto_adjust_batch_and_epochs", True)),
            )

            workspace_root = sess_cfg.dataset_root.parent
            run_dir = new_run_dir(workspace_root / "runs")
            processed_dataset_dir = self._processed_cache_dir(sess_cfg.dataset_root)

            self._train_set(session_id, progress=0.01, message="Preparing ROI dataset...")
            self._rebuild_processed_cache(
                sess_cfg.dataset_root,
                class_preprocess=cfg.class_preprocess,
                sample_preprocess=cfg.sample_preprocess,
                global_preprocess_mode=cfg.preprocess_mode,
                out_size=int(cfg.img_size),
                fast_mode=(str(cfg.crop_mode or CROP_MODE_CENTER) != CROP_MODE_AUTO_SEARCH),
            )

            def on_progress(p: float, msg: str) -> None:
                self._train_set(session_id, progress=p, message=msg)

            start_msg = "Starting training..."
            if not bool(cfg.auto_adjust_batch_and_epochs):
                # Advisory recommendation when the trainer-side auto-adjust is
                # disabled: surface recommended batch/epochs in the progress
                # message. Purely informational — guarded so it can never
                # break the training flow.
                try:
                    from trainer import recommend_train_params

                    total_samples = sum(
                        len(_list_class_image_files(sess_cfg.dataset_root / sanitize_class_name(c)))
                        for c in project_classes
                    )
                    rec_kwargs, _reasons = recommend_train_params(total_samples, cfg)
                    rec_batch = int(rec_kwargs.get("batch_size") or cfg.batch_size)
                    rec_epochs = int(rec_kwargs.get("epochs") or cfg.epochs)
                    if rec_batch != int(cfg.batch_size) or rec_epochs != int(cfg.epochs):
                        start_msg = (
                            f"Starting training... (recommended: batch {rec_batch}, "
                            f"epochs {rec_epochs} for {total_samples} samples)"
                        )
                except Exception:
                    pass
            self._train_set(session_id, progress=0.03, message=start_msg)
            result: TrainResult = train_and_export(
                dataset_dir=processed_dataset_dir,
                run_dir=run_dir,
                cfg=cfg,
                progress=on_progress,
            )
            latest = self._train_result_path(sess_cfg.dataset_root)
            payload = {
                "run_dir": str(result.run_dir),
                "labels": list(result.labels),
                "keras_model_path": str(result.keras_model_path),
                "tflite_path": str(result.tflite_path),
                "model_h_path": str(result.model_h_path),
                "model_cpp_path": str(result.model_cpp_path),
                "metrics": dict(result.metrics),
                "class_accuracies": dict(result.class_accuracies) if result.class_accuracies else {},
                "warnings": list(result.warnings) if result.warnings else [],
                "img_size": int(cfg.img_size),
                "color_mode": str(cfg.color_mode),
                "preprocess_mode": str(cfg.preprocess_mode),
                "crop_mode": str(cfg.crop_mode or CROP_MODE_AUTO_SEARCH),
                "manual_roi": list(cfg.manual_roi) if cfg.manual_roi is not None else None,
                "class_preprocess": normalize_class_preprocess_map(cfg.class_preprocess),
                "sample_preprocess": normalize_sample_preprocess_map(cfg.sample_preprocess),
                "trained_dataset_dir": str(processed_dataset_dir),
            }
            # Atomic write: a torn meta read by a concurrent /preview/predict
            # used to 400 every prediction until the next training.
            latest_tmp = latest.with_suffix(".json.tmp")
            latest_tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            latest_tmp.replace(latest)
            # Drop the cached preview interpreter AND the deployed-model pin
            # so the very next predict loads the freshly trained model (the
            # pin is re-stamped by the next export — that is its job, not
            # to shadow every later training).
            with self._lock:
                self._preview.pop(session_id, None)
            try:
                (cfg.dataset_root.parent / "deployed.json").unlink(missing_ok=True)
            except Exception:
                pass
            done_msg = "Done."
            if result.warnings:
                done_msg = "Done — WARNINGS: " + " | ".join(result.warnings)
            self._train_set(session_id, progress=1.0, message=done_msg, result_path=str(latest), done=True)
        except Exception as e:
            self._train_set(session_id, error=str(e), done=True)

    def _record_worker(self, session_id: str, source: str, class_name: str, live_thread: Optional[threading.Thread] = None) -> None:
        # Single-owner rule for BOTH sources: the live worker holds the
        # camera / serial reader; the record thread consumes its capture_png
        # cache and never opens a second device handle.  The old fallback
        # opened a 2nd cv2.VideoCapture / SerialFrameReader while the live
        # worker was streaming — on macOS that blinds the shared camera and
        # on serial it corrupts the frame stream (the hold-freeze bug).
        try:
            cfg = self._configs.get(session_id)
            if cfg is None:
                return
            interval = 1.0 / max(1.0, float(cfg.fps))
            if source == "device":
                self._record_device_from_live(session_id, cfg, class_name, interval)
                return
            if source == "webcam":
                try:
                    self._ensure_live(session_id, "webcam")
                except Exception:
                    self._fail_record(session_id, "Camera live preview is unavailable.")
                    return
                if not self._wait_live_capture(session_id, "webcam", timeout_s=2.0):
                    self._fail_record(session_id, "Camera is not producing frames. Check permission or whether it is in use.")
                    return
                self._record_webcam_from_live(session_id, cfg, class_name, interval)
                return
        except Exception as e:
            with self._lock:
                cur = self._active.get(session_id)
                if cur is not None:
                    cur["recording"] = "0"
                    msg = str(e) or "record failed"
                    if len(msg) > 220:
                        msg = msg[:220] + "..."
                    cur["error"] = msg
            cond = self._record_conds.get(session_id)
            if cond is not None:
                with cond:
                    cond.notify_all()
        finally:
            with self._lock:
                cur = self._active.get(session_id)
                if cur is not None:
                    cur["recording"] = "0"
                thread = self._record_threads.get(session_id)
                if thread is threading.current_thread():
                    self._record_threads.pop(session_id, None)
            cond = self._record_conds.get(session_id)
            if cond is not None:
                with cond:
                    cond.notify_all()

    def _live_key(self, session_id: str, source: str) -> str:
        return f"{session_id}:{source}"

    def release_all_resources(self) -> None:
        """Graceful app-shutdown release.  Idempotent — hot-reload and atexit
        may call this more than once.

        Stops every live worker (their finally blocks release the camera /
        serial handles), stops any active record thread, releases every
        AVCaptureSession registered with the mac_camera bridge, and shuts
        the HTTP server down.  Called from the app-level atexit hook and
        the /shutdown endpoint.
        """
        with self._lock:
            keys = list(self._live.keys())
            record_sids = list(self._record_threads.keys())
        for key in keys:
            try:
                session_id, source = key.rsplit(":", 1)
            except Exception:
                continue
            try:
                self._stop_live(session_id=session_id, source=source, wait=True, timeout_s=1.5)
            except Exception:
                continue
        for sid in record_sids:
            try:
                self._stop_record(sid, wait=True, timeout_s=1.5)
            except Exception:
                continue
        try:
            from mac_camera import shutdown_all_caps

            shutdown_all_caps(wait_s=0.2)
        except Exception:
            pass
        try:
            self.stop()
        except Exception:
            pass

    def _stop_live(self, session_id: str, source: Optional[str] = None, *, wait: bool = False, timeout_s: float = 2.0) -> None:
        threads: List[threading.Thread] = []
        with self._lock:
            keys = [self._live_key(session_id, source)] if source else [k for k in self._live if k.startswith(f"{session_id}:")]
            for key in keys:
                state = self._live.get(key)
                if state is not None:
                    state["running"] = False
                stop_evt = self._live_stops.get(key)
                if stop_evt is not None:
                    stop_evt.set()
                thread = self._live_threads.get(key)
                if thread is not None:
                    threads.append(thread)
        if wait:
            deadline = time.time() + max(0.0, float(timeout_s))
            for thread in threads:
                if thread is threading.current_thread():
                    continue
                remaining = deadline - time.time()
                if remaining <= 0:
                    break
                if thread.is_alive():
                    thread.join(timeout=remaining)
        with self._lock:
            for thread in threads:
                for key, current in list(self._live_threads.items()):
                    if current is thread and not thread.is_alive():
                        self._live_threads.pop(key, None)

    def _ensure_live(self, session_id: str, source: str) -> None:
        cfg = self._configs.get(session_id)
        if cfg is None:
            raise RuntimeError("missing config")
        key = self._live_key(session_id, source)
        now = time.time()
        with self._lock:
            for stale_key, stale_state in list(self._live.items()):
                try:
                    if stale_state.get("running"):
                        continue
                    if now - float(stale_state.get("last_touch") or 0.0) > 30.0:
                        self._live.pop(stale_key, None)
                except Exception:
                    continue
            cur = self._live.get(key)
            thread = self._live_threads.get(key)
            if thread is not None and thread.is_alive():
                # The worker is alive (possibly shutting down after a
                # /live/close) — never spawn a second worker for the same
                # device handle; the next poll will respawn once it exits.
                if cur is not None:
                    cur["last_touch"] = now
                    cur["cfg"] = cfg
                return
            gen = 1
            if cur is not None:
                try:
                    gen = int(cur.get("gen") or 0) + 1
                except Exception:
                    gen = 1
            state = {
                "running": True,
                "gen": gen,
                "last_touch": now,
                "_publish_ts": 0.0,
                "_preview_ts": 0.0,
                "_capture_ts": 0.0,
                "source": source,
                "cfg": cfg,
                "preview_png": None,
                "capture_png": None,
                "error": "",
                "actual": None,
            }
            self._live[key] = state
            stop_evt = threading.Event()
            self._live_stops[key] = stop_evt
        t = threading.Thread(target=self._live_worker, args=(key, session_id, source, gen, stop_evt), daemon=True)
        with self._lock:
            self._live_threads[key] = t
        t.start()

    def _get_live_preview(self, session_id: str, source: str, autostart: bool = True) -> Optional[bytes]:
        key = self._live_key(session_id, source)
        with self._lock:
            state = self._live.get(key)
            if state is not None:
                running = bool(state.get("running"))
                ts = float(state.get("_preview_ts") or 0.0)
            else:
                running = False
                ts = 0.0
        needs_revival = False
        if state is None or not running or ts <= 0.0:
            needs_revival = True
        elif time.time() - ts > LIVE_STALE_S:
            # Frames went stale — revive, but only when the producer thread
            # is actually dead.  A live-but-blocked worker (e.g. a serial
            # read_frame in flight) cannot be respawned without double-owning
            # the device, so /live/frame keeps answering "starting" until it
            # publishes again or exits.
            with self._lock:
                thread = self._live_threads.get(key)
            if thread is None or not thread.is_alive():
                needs_revival = True
        if needs_revival and autostart:
            try:
                self._ensure_live(session_id, source)
            except Exception:
                pass
            # Give the fresh worker a short window to publish its first frame.
            deadline = time.time() + 0.4
            while time.time() < deadline:
                with self._lock:
                    state = self._live.get(key)
                    if state is not None:
                        data = state.get("preview_png")
                        if isinstance(data, (bytes, bytearray)):
                            return bytes(data)
                time.sleep(0.01)
        with self._lock:
            state = self._live.get(key)
            if state is None:
                return None
            state["last_touch"] = time.time()
            data = state.get("preview_png")
            return bytes(data) if isinstance(data, (bytes, bytearray)) else None

    def _is_live_starting(self, session_id: str, source: str) -> bool:
        """True when the producer has no frame yet (worker warming up, just
        (re)spawned, or waiting for a device) — the SPA keeps the last good
        frame on screen instead of blanking."""
        key = self._live_key(session_id, source)
        with self._lock:
            state = self._live.get(key)
            if state is None:
                return True
            if not state.get("running"):
                return True
            return state.get("preview_png") is None

    def _get_live_capture_fresh(self, session_id: str, source: str, timeout_s: float = 0.4) -> Optional[bytes]:
        """capture_png from the live cache, but only when it is FRESH
        (published within the last 2 s) — a click must never save a stale or
        arbitrarily old frame."""
        key = self._live_key(session_id, source)
        deadline = time.time() + float(timeout_s)
        while True:
            with self._lock:
                state = self._live.get(key)
                if state is not None:
                    data = state.get("capture_png")
                    ts = float(state.get("_capture_ts") or 0.0)
                    if isinstance(data, (bytes, bytearray)) and ts > 0.0 and (time.time() - ts) <= 2.0:
                        state["last_touch"] = time.time()  # consumer heartbeat
                        return bytes(data)
            if time.time() >= deadline:
                return None
            time.sleep(0.01)

    def _get_live_capture(self, session_id: str, source: str) -> Optional[bytes]:
        key = self._live_key(session_id, source)
        with self._lock:
            state = self._live.get(key)
            if state is None:
                return None
            state["last_touch"] = time.time()
            data = state.get("capture_png")
            return bytes(data) if isinstance(data, (bytes, bytearray)) else None

    def _get_live_error(self, session_id: str, source: str) -> str:
        key = self._live_key(session_id, source)
        with self._lock:
            state = self._live.get(key)
            if state is None:
                return ""
            state["last_touch"] = time.time()
            return str(state.get("error") or "")

    def _live_running(self, key: str, gen: Optional[int] = None) -> bool:
        with self._lock:
            state = self._live.get(key)
            if state is None:
                return False
            if gen is not None and int(state.get("gen") or 0) != int(gen):
                # A successor worker owns this key now — never touch its state.
                return False
            if not state.get("running"):
                return False
            # Consumer-idle lease: a worker that NO consumer (class-panel
            # poll, preview visual/predict, capture read) has touched for
            # LIVE_IDLE_S is reaped — this is what releases the camera when
            # the user closes the webcam view or toggles Input off and no
            # other view still uses the source.  Revival on demand makes
            # this safe: the next /live/frame poll respawns the worker.
            # Never reaped while a hold capture is active for this session
            # (the record thread consumes the capture cache directly).
            now = time.time()
            if now - float(state.get("last_touch") or 0.0) > LIVE_IDLE_S:
                session_id = key.rsplit(":", 1)[0]
                active = self._active.get(session_id) or {}
                if active.get("recording") != "1":
                    state["running"] = False
                    return False
            return True

    def _live_set(self, key: str, *, preview_png: Optional[bytes] = None, capture_png: Optional[bytes] = None, error: Optional[str] = None, actual: Optional[Dict[str, Any]] = None) -> None:
        now = time.time()
        with self._lock:
            state = self._live.get(key)
            if state is None:
                return
            # NOTE: last_touch is deliberately NOT written here.  It tracks
            # CONSUMER activity (HTTP /live/frame + capture reads), and the
            # idle lease in _live_running releases the camera when every
            # view of it is gone (view closed / Input toggled off).  The
            # producer writes _publish_ts, used by staleness checks.
            if preview_png is not None:
                state["preview_png"] = preview_png
                state["_preview_ts"] = now
                state["_publish_ts"] = now
            if capture_png is not None:
                state["capture_png"] = capture_png
                state["_capture_ts"] = now
                state["_publish_ts"] = now
            if error is not None:
                state["error"] = error
            if actual is not None:
                state["actual"] = actual

    def _live_worker(self, key: str, session_id: str, source: str, gen: int, stop_evt: threading.Event) -> None:
        cfg = self._configs.get(session_id)
        if cfg is None:
            return
        try:
            if source == "webcam":
                self._live_webcam(key, session_id, cfg, gen, stop_evt)
            elif source == "device":
                self._live_serial(key, session_id, cfg, gen, stop_evt)
        finally:
            with self._lock:
                state = self._live.get(key)
                if state is not None and int(state.get("gen") or 0) == int(gen):
                    # Only the owning generation may mark its state dead —
                    # a lagging predecessor must not kill its successor.
                    state["running"] = False
                    state["last_touch"] = time.time()
                thread = self._live_threads.get(key)
                if thread is threading.current_thread():
                    self._live_threads.pop(key, None)

    def _live_serial(self, key: str, session_id: str, cfg: SessionConfig, gen: int, stop_evt: threading.Event) -> None:
        if not cfg.serial_port:
            self._live_set(key, error="Serial port is not configured.")
            return
        last_error = ""
        reader = None
        reader_sig = None
        while not stop_evt.is_set() and self._live_running(key, gen):
            # Re-read the live config every iteration so setting changes
            # (baud / frame side / port / channels) take effect immediately.
            fresh = self._configs.get(session_id)
            if fresh is not None:
                cfg = fresh
            sig = (str(cfg.serial_port), int(cfg.serial_baud), str(cfg.serial_sync), int(cfg.serial_frame_side), int(cfg.serial_channels))
            if reader is None or sig != reader_sig:
                # Reuse ONE reader across iterations — per-frame open/close churn
                # becomes dominant at 160×160 and after record contention.
                if reader is not None:
                    try:
                        reader.close()
                    except Exception:
                        pass
                reader = SerialFrameReader(port=cfg.serial_port, baud=int(cfg.serial_baud), sync_header=cfg.serial_sync, frame_side=int(cfg.serial_frame_side), channels=int(cfg.serial_channels))
                reader_sig = sig
                try:
                    reader.open()
                except Exception as e:
                    msg = str(e) or "Unable to open serial device."
                    self._live_set(key, error=msg[:220])
                    last_error = msg
                    reader = None
                    reader_sig = None
                    time.sleep(0.25)
                    continue
            try:
                # Scale timeout with frame size (10 bits/byte at 8N1, ×2.5 margin, floor 1.5 s).
                frame_bytes = int(cfg.serial_frame_side) * int(cfg.serial_frame_side) * int(cfg.serial_channels)
                baud = int(cfg.serial_baud) if int(cfg.serial_baud) > 0 else 921600
                xfer_s = frame_bytes * 10.0 / float(baud)
                timeout = max(1.5, xfer_s * 2.5)
                raw = reader.read_frame(timeout_s=timeout)
                capture_png = _raw96_to_png(raw, crop_box=cfg.crop_box, frame_side=int(cfg.serial_frame_side), out_side=int(cfg.serial_frame_side))
                self._live_set(key, preview_png=capture_png, capture_png=capture_png, error="")
                last_error = ""
            except Exception as e:
                msg = str(e) or "Unable to read from serial device."
                if len(msg) > 220:
                    msg = msg[:220] + "..."
                if msg != last_error:
                    self._live_set(key, error=msg)
                    last_error = msg
                time.sleep(0.06)
        if reader is not None:
            try:
                reader.close()
            except Exception:
                pass

    def _live_webcam(self, key: str, session_id: str, cfg: SessionConfig, gen: int, stop_evt: threading.Event) -> None:
        import cv2

        cap = None
        cam_sig = None
        last_error = ""
        read_failures = 0
        ever_framed = False
        try:
            while not stop_evt.is_set() and self._live_running(key, gen):
                try:
                    fresh = self._configs.get(session_id)
                    if fresh is not None:
                        cfg = fresh
                    uid = str(cfg.webcam_id or "").strip()
                    # Key the open on the uniqueID alone — virtual-camera
                    # connect/disconnect drifts positional indices and must
                    # not force a spurious reopen of the (correct) device.
                    current_sig = uid if uid else f"idx:{int(cfg.webcam_index)}"
                    if cam_sig is None or current_sig != cam_sig:
                        if cap is not None:
                            try:
                                cap.release()
                            except Exception:
                                pass
                            cap = None
                        cam_sig = None
                        permission = ensure_camera_access(webcam_index=int(cfg.webcam_index or 0), probe_open=False, unique_id=uid)
                        if not permission.allowed:
                            last_error = permission.message or "Camera permission denied."
                            self._live_set(key, error=last_error)
                            time.sleep(0.3)
                            continue
                        new_cap, info = _open_working_camera(int(cfg.webcam_index or 0), unique_id=uid)
                        if new_cap is None or not new_cap.isOpened():
                            info_error = str(getattr(info, "error", "") or "") if info is not None else ""
                            last_error = info_error or f"Unable to open webcam (index={cfg.webcam_index}, id={uid[:12] if uid else ''}…). Check settings."
                            self._live_set(key, error=last_error, actual=_camera_open_info_dict(info))
                            time.sleep(0.3)
                            continue
                        cap = new_cap
                        cam_sig = current_sig
                        read_failures = 0
                        last_error = ""
                        self._live_set(key, actual=_camera_open_info_dict(info), error="")
                    ok, frame = cap.read()
                    if not ok or frame is None:
                        read_failures += 1
                        # Only reopen after the stream was KNOWN GOOD — a
                        # camera whose first frame takes >0.8 s (cold start)
                        # must not be torn down by the blind-stream detector.
                        if ever_framed and read_failures >= 10:
                            # The stream went blind (e.g. another process took
                            # the camera) — report it and force a reopen
                            # instead of serving stale frames silently.
                            last_error = "Camera stopped producing frames; reopening..."
                            self._live_set(key, error=last_error)
                            try:
                                cap.release()
                            except Exception:
                                pass
                            cap = None
                            cam_sig = None
                            read_failures = 0
                            ever_framed = False
                            time.sleep(0.1)
                        else:
                            time.sleep(0.08)
                        continue
                    read_failures = 0
                    ever_framed = True
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    preview_frame = frame
                    try:
                        h, w = frame.shape[:2]
                        max_w = 360
                        if w > max_w and h > 0:
                            scaled_h = max(1, int(h * (max_w / float(w))))
                            preview_frame = cv2.resize(frame, (max_w, scaled_h), interpolation=cv2.INTER_AREA)
                    except Exception:
                        preview_frame = frame
                    preview_png = _to_png_bytes(Image.fromarray(preview_frame))
                    img = Image.fromarray(frame).convert("RGB")
                    if cfg.crop_box is not None:
                        x1, y1, x2, y2 = cfg.crop_box
                        img = img.crop((x1, y1, x2, y2))
                    img = img.resize((int(cfg.img_size), int(cfg.img_size)))
                    self._live_set(key, preview_png=preview_png, capture_png=_to_png_bytes(img), error="")
                    time.sleep(0.06)
                except Exception as e:
                    # The live thread must NEVER die: an exception here used
                    # to kill the worker while the cached preview_png kept
                    # being served with HTTP 200 forever — a silently frozen
                    # view.  Publish the error, release the capture and keep
                    # polling instead.
                    msg = str(e) or "webcam error"
                    if msg != last_error:
                        self._live_set(key, error=msg[:220])
                        last_error = msg
                    if cap is not None:
                        try:
                            cap.release()
                        except Exception:
                            pass
                        cap = None
                    cam_sig = None
                    read_failures = 0
                    ever_framed = False
                    time.sleep(0.3)
        finally:
            if cap is not None:
                try:
                    cap.release()
                except Exception:
                    pass

    def _preview_frame(self, session_id: str, source: str) -> Optional[bytes]:
        cfg = self._configs.get(session_id)
        if cfg is None:
            return None
        if source == "device":
            return self.preview_serial_png(cfg.serial_port, int(cfg.serial_baud), str(cfg.serial_sync), frame_side=int(cfg.serial_frame_side), channels=int(cfg.serial_channels))
        if source == "webcam":
            return self.preview_webcam_png(int(cfg.webcam_index), webcam_id=cfg.webcam_id, session_id=session_id)
        return None

    def _capture_single(self, session_id: str, source: str) -> Optional[bytes]:
        cfg = self._configs.get(session_id)
        if cfg is None:
            return None
        if source == "device":
            try:
                reader = SerialFrameReader(port=cfg.serial_port, baud=int(cfg.serial_baud), sync_header=cfg.serial_sync, frame_side=int(cfg.serial_frame_side), channels=int(cfg.serial_channels))
                reader.open()
                try:
                    raw = reader.read_frame()  # auto-scaled timeout (frame_size × baud)
                finally:
                    reader.close()
                return _raw96_to_png(raw, crop_box=cfg.crop_box, frame_side=int(cfg.serial_frame_side), out_side=int(cfg.serial_frame_side))
            except Exception:
                return None
        if source == "webcam":
            permission = ensure_camera_access(webcam_index=int(cfg.webcam_index or 0), probe_open=False, unique_id=str(cfg.webcam_id or ""))
            if not permission.allowed:
                return None
            try:
                import cv2
            except Exception:
                return None
            try:
                cap, _actual_index = _open_working_camera(int(cfg.webcam_index or 0), unique_id=cfg.webcam_id)
            except Exception:
                return None
            if cap is None:
                return None
            ok, frame = cap.read()
            cap.release()
            if not ok or frame is None:
                return None
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame).convert("RGB")
            if cfg.crop_box is not None:
                x1, y1, x2, y2 = cfg.crop_box
                img = img.crop((x1, y1, x2, y2))
            img = img.resize((int(cfg.img_size), int(cfg.img_size)))
            return _to_png_bytes(img)
        return None

    def _is_recording(self, session_id: str) -> bool:
        with self._lock:
            return self._active.get(session_id, {}).get("recording") == "1"

    def _record_serial(self, session_id: str, cfg: SessionConfig, class_name: str, interval: float) -> None:
        reader = SerialFrameReader(port=cfg.serial_port, baud=int(cfg.serial_baud), sync_header=cfg.serial_sync, frame_side=int(cfg.serial_frame_side), channels=int(cfg.serial_channels))
        try:
            reader.open()
            while self._is_recording(session_id):
                raw = reader.read_frame()  # auto-scaled timeout (frame_size × 12 bits/byte / baud)
                png = _raw96_to_png(raw, crop_box=cfg.crop_box, frame_side=int(cfg.serial_frame_side), out_side=int(cfg.serial_frame_side))
                p = _save_png(cfg.dataset_root, class_name, png)
                self._cache_one_sample(cfg.dataset_root, class_name, p, img_size=int(cfg.img_size))
                with self._lock:
                    cur = self._active.get(session_id)
                    if cur is not None and cur.get("recording") == "1":
                        cur["seq"] = int(cur.get("seq") or 0) + 1
                        cur["count"] = int(cur.get("count") or 0) + 1
                        cur["filename"] = str(p.name)
                        cur["image_b64"] = base64.b64encode(png).decode("ascii")
                cond = self._record_conds.get(session_id)
                if cond is not None:
                    with cond:
                        cond.notify_all()
                time.sleep(interval)
        finally:
            try:
                reader.close()
            except Exception:
                pass

    def _record_webcam(self, session_id: str, cfg: SessionConfig, class_name: str, interval: float) -> None:
        # Legacy direct-open record path — normal hold capture now consumes
        # the live worker's capture_png cache (_record_webcam_from_live) so a
        # second camera open can never blind the shared AVCaptureSession.
        permission = ensure_camera_access(webcam_index=int(cfg.webcam_index or 0), probe_open=False, unique_id=str(cfg.webcam_id or ""))
        if not permission.allowed:
            with self._lock:
                self._active[session_id] = {"recording": "0", "error": permission.message}
            return
        try:
            import cv2
        except Exception:
            with self._lock:
                self._active[session_id] = {"recording": "0", "error": "missing opencv-python"}
            return

        cap, info = _open_working_camera(int(cfg.webcam_index or 0), unique_id=cfg.webcam_id)
        if cap is None or not cap.isOpened():
            info_error = str(getattr(info, "error", "") or "") if info is not None else ""
            with self._lock:
                self._active[session_id] = {"recording": "0", "error": info_error or "webcam open failed"}
            return
        try:
            while self._is_recording(session_id):
                ok, frame = cap.read()
                if not ok or frame is None:
                    time.sleep(interval)
                    continue
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame).convert("RGB")
                if cfg.crop_box is not None:
                    x1, y1, x2, y2 = cfg.crop_box
                    img = img.crop((x1, y1, x2, y2))
                img = img.resize((int(cfg.img_size), int(cfg.img_size)))
                png = _to_png_bytes(img)
                p = _save_png(cfg.dataset_root, class_name, png)
                self._cache_one_sample(cfg.dataset_root, class_name, p, img_size=int(cfg.img_size))
                with self._lock:
                    cur = self._active.get(session_id)
                    if cur is not None and cur.get("recording") == "1":
                        cur["seq"] = int(cur.get("seq") or 0) + 1
                        cur["count"] = int(cur.get("count") or 0) + 1
                        cur["filename"] = str(p.name)
                        cur["image_b64"] = base64.b64encode(png).decode("ascii")
                cond = self._record_conds.get(session_id)
                if cond is not None:
                    with cond:
                        cond.notify_all()
                time.sleep(interval)
        finally:
            try:
                cap.release()
            except Exception:
                pass

    def _record_webcam_from_live(self, session_id: str, cfg: SessionConfig, class_name: str, interval: float) -> None:
        """Save capture samples by consuming the 96x96 capture_png cache
        produced by the running _live_webcam worker.

        This is the fast / happy path: it avoids opening a 2nd capture
        object against the same macOS AVCaptureSession (which would fail
        for most built-in cameras), keeps preview/live running concurrently,
        and guarantees the saved samples byte-for-byte match what Preview
        inference consumes (via the same capture_png cache).
        """
        self._record_from_live(session_id, cfg, class_name, interval, self._live_key(session_id, "webcam"), "Camera")

    def _record_device_from_live(self, session_id: str, cfg: SessionConfig, class_name: str, interval: float) -> None:
        """Hold-capture for the serial device via the live worker's capture_png
        cache — the live reader keeps feeding the class-panel view WHILE the
        hold saves samples (the old design stopped the live reader for the
        whole hold, freezing the view)."""
        try:
            self._ensure_live(session_id, "device")
        except Exception:
            self._fail_record(session_id, "Device live preview is unavailable.")
            return
        if not self._wait_live_capture(session_id, "device", timeout_s=2.0):
            self._fail_record(session_id, "Device is not producing frames. Check the serial port and baudrate.")
            return
        self._record_from_live(session_id, cfg, class_name, interval, self._live_key(session_id, "device"), "Device")

    def _record_from_live(self, session_id: str, cfg: SessionConfig, class_name: str, interval: float, key: str, source_label: str) -> None:
        """Shared live-cache consumer for hold capture (webcam + device).

        Consumes only NEW frames (capture_png watermark) and aborts the hold
        with a visible error when the producer stalls — the hold poll loop
        surfaces ``active.error`` as a toast.  No fallback ever opens a
        second device handle.
        """
        last_saved_seen_ts: float = 0.0
        last_frame_at = time.time()
        while self._is_recording(session_id):
            now = time.time()
            capture_png: Optional[bytes] = None
            with self._lock:
                state = self._live.get(key)
                if state is None or not state.get("running"):
                    stale = True
                else:
                    raw_bytes = state.get("capture_png")
                    seen_ts = float(state.get("_capture_ts") or 0.0)
                    if isinstance(raw_bytes, (bytes, bytearray)) and seen_ts > last_saved_seen_ts:
                        capture_png = bytes(raw_bytes)
                        last_saved_seen_ts = seen_ts
                        last_frame_at = now
                        stale = False
                    else:
                        stale = True
            if capture_png is None:
                if now - last_frame_at > 2.0:
                    self._fail_record(session_id, f"{source_label} stopped producing frames during capture. Check the connection and try again.")
                    return
                time.sleep(max(0.015, float(interval) * 0.35))
                continue
            p = _save_png(cfg.dataset_root, class_name, capture_png)
            self._cache_one_sample(cfg.dataset_root, class_name, p, img_size=int(cfg.img_size))
            with self._lock:
                cur = self._active.get(session_id)
                if cur is not None and cur.get("recording") == "1":
                    cur["seq"] = int(cur.get("seq") or 0) + 1
                    cur["count"] = int(cur.get("count") or 0) + 1
                    cur["filename"] = str(p.name)
                    try:
                        cur["image_b64"] = base64.b64encode(capture_png).decode("ascii")
                    except Exception:
                        cur["image_b64"] = ""
            cond = self._record_conds.get(session_id)
            if cond is not None:
                with cond:
                    cond.notify_all()
            next_deadline = time.time() + float(interval)
            sleep_s = next_deadline - time.time()
            if sleep_s > 0:
                time.sleep(sleep_s)

    def _fail_record(self, session_id: str, message: str) -> None:
        """Abort an active hold capture with a user-visible error."""
        with self._lock:
            cur = self._active.get(session_id)
            if cur is not None:
                cur["recording"] = "0"
                cur["error"] = str(message)[:220]
        cond = self._record_conds.get(session_id)
        if cond is not None:
            with cond:
                cond.notify_all()

    def _wait_live_capture(self, session_id: str, source: str, timeout_s: float = 2.0) -> bool:
        """Wait until the live worker has published a FRESH capture frame
        (age ≤ 2 s).  Respawns the worker if it died while waiting."""
        key = self._live_key(session_id, source)
        deadline = time.time() + float(timeout_s)
        while time.time() < deadline:
            with self._lock:
                state = self._live.get(key)
                running = bool(state.get("running")) if state is not None else False
                ts = float(state.get("_capture_ts") or 0.0) if state is not None else 0.0
                if state is not None:
                    state["last_touch"] = time.time()  # consumer heartbeat
            if ts > 0.0 and (time.time() - ts) <= 2.0:
                return True
            if not running:
                try:
                    self._ensure_live(session_id, source)
                except Exception:
                    return False
            time.sleep(0.02)
        return False

    def _class_state_payload(self, dataset_root: Path, class_name: str, limit: Optional[int] = None) -> Dict[str, Any]:
        class_dir = dataset_root / sanitize_class_name(class_name)
        files = _list_class_image_files(class_dir) if class_dir.exists() else []
        previews: List[Dict[str, str]] = []
        view_files = files if limit is None else files[:limit]
        for p in view_files:
            try:
                previews.append(_preview_item_payload(p))
            except Exception:
                continue
        return {
            "ok": "1",
            "class": class_name,
            "count": str(len(files)),
            "previews": previews,
        }

    def _class_filenames_payload(self, dataset_root: Path, class_name: str) -> Dict[str, Any]:
        class_dir = dataset_root / sanitize_class_name(class_name)
        files = _list_class_image_files(class_dir) if class_dir.exists() else []
        return {
            "ok": "1",
            "class": class_name,
            "count": str(len(files)),
            "filenames": [str(p.name) for p in files],
        }

    def _sample_image_png(self, dataset_root: Path, class_name: str, filename: str) -> bytes:
        class_dir = dataset_root / sanitize_class_name(class_name)
        path = (class_dir / str(filename)).resolve()
        if not str(path).startswith(str(class_dir.resolve())) or not path.exists() or not path.is_file():
            raise FileNotFoundError(filename)
        img = Image.open(path)
        try:
            return _to_png_bytes(img.convert("RGB" if img.mode not in {"L", "RGB"} else img.mode))
        finally:
            try:
                img.close()
            except Exception:
                pass

    def _dataset_labels_load(self, dataset_dir: Path) -> List[str]:
        labels_json = dataset_dir / "labels.json"
        if labels_json.exists():
            try:
                raw = json.loads(labels_json.read_text(encoding="utf-8"))
                labels = raw.get("labels") if isinstance(raw, dict) else None
                if isinstance(labels, list):
                    out = [str(x).strip() for x in labels if str(x).strip()]
                    if out:
                        return out
            except Exception:
                pass
        labels_txt = dataset_dir / "labels.txt"
        if labels_txt.exists():
            try:
                out = [line.strip() for line in labels_txt.read_text(encoding="utf-8").splitlines() if line.strip()]
                if out:
                    return out
            except Exception:
                pass
        return []


def _find_free_port(host: str) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, 0))
        return int(s.getsockname()[1])




def list_webcam_options(max_count: int = 6) -> List[Dict[str, str]]:
    """Fresh camera list, enumerated at CALL time.

    AVCaptureDevice order flips when virtual cameras (Iriun etc.) connect or
    disconnect, so snapshots pair names with the wrong cv2 indices.  On
    macOS each entry also carries the device uniqueID — capture by uniqueID
    is stable, so consumers MUST prefer it over the positional index.

    WARNING: this array order is NOT cv2's positional index order — OpenCV's
    AVFoundation backend re-sorts the same device list by uniqueID.  Passing
    the index from this list to cv2.VideoCapture can silently open a
    DIFFERENT camera (e.g. Iriun at index 0 while this list shows the
    built-in first).  Always open by unique_id on macOS.
    """
    options: List[Dict[str, str]] = []
    if sys.platform == "darwin":
        try:
            from AVFoundation import AVCaptureDevice, AVMediaTypeVideo

            devices = list(AVCaptureDevice.devicesWithMediaType_(AVMediaTypeVideo) or [])
            for idx, dev in enumerate(devices[:max_count]):
                name = str(dev.localizedName() or f"Camera {idx}")
                options.append({"index": idx, "label": name, "unique_id": str(dev.uniqueID() or "")})
        except Exception:
            options = []
            try:
                import json as _json
                proc = subprocess.run(
                    ["system_profiler", "SPCameraDataType", "-json"],
                    capture_output=True, text=True, check=False, timeout=10,
                )
                if proc.returncode == 0:
                    cameras = _json.loads(proc.stdout or "{}").get("SPCameraDataType", [])
                    for idx, cam in enumerate(cameras[:max_count]):
                        name = str(cam.get("_name") or cam.get("spcamera_model-id") or f"Camera {idx}")
                        options.append({"index": idx, "label": name, "unique_id": str(cam.get("spcamera_unique-id") or "")})
            except Exception:
                pass
    if not options:
        for idx in range(max_count):
            options.append({"index": idx, "label": f"Camera {idx}", "unique_id": ""})
    return options


def _resolve_webcam_index_from_unique_id(unique_id: Optional[str], max_count: int = 6) -> Optional[int]:
    """Map a macOS AVCaptureDevice uniqueID to its position in the CURRENT
    AVFoundation enumeration — for dropdown preselection ONLY.

    WARNING: this is NOT a cv2 positional index.  cv2's AVFoundation backend
    re-sorts the device array by uniqueID, so passing this number to
    cv2.VideoCapture can open a different camera.  The camera itself is
    opened by uniqueID (mac_camera bridge); the index is purely a UI hint.
    """
    if not unique_id or sys.platform != "darwin":
        return None
    try:
        from AVFoundation import AVCaptureDevice, AVMediaTypeVideo

        devices = list(AVCaptureDevice.devicesWithMediaType_(AVMediaTypeVideo) or [])
        for idx, dev in enumerate(devices[:max_count]):
            if str(dev.uniqueID() or "") == str(unique_id):
                return idx
    except Exception:
        pass
    return None


def _send_json(req: BaseHTTPRequestHandler, obj: Dict[str, Any], status: int = 200, cors: bool = False) -> None:
    data = json.dumps(obj).encode("utf-8")
    try:
        req.send_response(status)
        if cors:
            req.send_header("Access-Control-Allow-Origin", "*")
        req.send_header("Content-Type", "application/json")
        req.send_header("Content-Length", str(len(data)))
        req.end_headers()
        req.wfile.write(data)
    except (BrokenPipeError, ConnectionResetError, OSError):
        # Client disconnected mid-response (SPA aborts fetches with
        # AbortController) — nothing to send to; ignore silently.
        pass


def _send_png(req: BaseHTTPRequestHandler, data: bytes, cors: bool = False, cache_control: str = "no-store, max-age=0") -> None:
    try:
        req.send_response(200)
        if cors:
            req.send_header("Access-Control-Allow-Origin", "*")
        req.send_header("Cache-Control", str(cache_control or "no-store, max-age=0"))
        req.send_header("Content-Type", "image/png")
        req.send_header("Content-Length", str(len(data)))
        req.end_headers()
        req.wfile.write(data)
    except (BrokenPipeError, ConnectionResetError, OSError):
        # Client disconnected mid-response — ignore silently.
        pass


def _bytes_io(data: bytes):
    import io

    return io.BytesIO(data)


def _save_png(dataset_root: Path, class_name: str, png: bytes) -> Path:
    import uuid

    safe = sanitize_class_name(class_name)
    out_dir = dataset_root / safe
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / (uuid.uuid4().hex + ".png")
    p.write_bytes(png)
    return p


def _list_class_image_files(class_dir: Path) -> List[Path]:
    if not class_dir.exists():
        return []
    return sorted(
        [p for p in class_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )


def _preview_item_payload(path: Path) -> Dict[str, str]:
    b64 = base64.b64encode(path.read_bytes()).decode("ascii")
    return {
        "src": f"data:image/png;base64,{b64}",
        "filename": str(path.name),
    }


@dataclass(frozen=True)
class CameraOpenInfo:
    """How (and which) camera _open_working_camera actually opened."""
    backend: str  # "unique_id" (mac_camera bridge) | "cv2_index" | "failed"
    index: int
    unique_id: str
    label: str
    error: str


def _camera_open_info_dict(info: Optional[CameraOpenInfo]) -> Dict[str, Any]:
    if info is None:
        return {}
    return {
        "backend": str(info.backend or ""),
        "index": int(info.index or 0),
        "unique_id": str(info.unique_id or ""),
        "label": str(info.label or ""),
        "error": str(info.error or ""),
    }


_bridge_opener = None
_bridge_opener_checked = False
_bridge_opener_lock = threading.Lock()


def _get_macos_bridge_opener():
    """Lazily import the AVCaptureSession bridge (never at module import).

    mac_camera.py registers an ObjC class on first use; importing it here
    (not at the top of record_controller) keeps processes that never open a
    camera free of that cost, and a broken bridge degrades to None instead
    of breaking ``import record_controller``.
    """
    global _bridge_opener, _bridge_opener_checked
    with _bridge_opener_lock:
        if not _bridge_opener_checked:
            _bridge_opener_checked = True
            try:
                from mac_camera import open_macos_camera_by_unique_id as _opener
                _bridge_opener = _opener
            except Exception:
                _bridge_opener = None
        return _bridge_opener


def _uid_for_webcam_index(camera_index: int, max_count: int = 6) -> str:
    """Best-effort uniqueID for a positional index from a FRESH enumeration.

    Legacy rescue only: old persisted state stored an index without a
    uniqueID.  All new selections carry the uniqueID end-to-end from the
    dropdown, so this is a safety net, not a primary path."""
    try:
        for item in list_webcam_options(max_count=max_count):
            try:
                if int(item.get("index", -1)) == int(camera_index):
                    uid = str(item.get("unique_id") or "")
                    if uid:
                        return uid
            except Exception:
                continue
    except Exception:
        pass
    return ""


def _open_working_camera(preferred_index: int, unique_id: Optional[str] = None, max_probe_index: int = 3):
    """Open the SELECTED camera, never "some camera that opens".

    Resolution order on macOS: (a) unique_id present → open EXACTLY that
    AVCaptureDevice via the mac_camera bridge; on failure return (None, info)
    with NO positional substitution — silently opening the wrong device is
    worse than a visible error (the live worker retries and surfaces the
    message).  (b) no unique_id → derive one from a fresh enumeration (legacy
    state rescue); if the bridge is unavailable, degrade to the positional
    probe.  Non-macOS: positional probe as before.

    Never raises; the second tuple element is a CameraOpenInfo (all existing
    call sites discard it, so the shape change is safe).
    """
    import cv2

    idx = int(preferred_index or 0)
    uid = str(unique_id or "").strip()
    if sys.platform == "darwin" and uid:
        opener = _get_macos_bridge_opener()
        if opener is None:
            return None, CameraOpenInfo("failed", idx, uid, "", "macOS camera bridge unavailable (mac_camera.py missing from the bundle).")
        try:
            cap = opener(uid)
        except Exception as e:
            return None, CameraOpenInfo("failed", idx, uid, "", str(e)[:220])
        if cap is not None and cap.isOpened():
            return cap, CameraOpenInfo("unique_id", idx, uid, "", "")
        return None, CameraOpenInfo("failed", idx, uid, "", f"Selected camera (ID {uid[:12]}…) is not connected or cannot be opened.")
    if sys.platform == "darwin" and not uid:
        derived = _uid_for_webcam_index(idx)
        if derived:
            opener = _get_macos_bridge_opener()
            if opener is not None:
                try:
                    cap = opener(derived)
                except Exception as e:
                    return None, CameraOpenInfo("failed", idx, derived, "", str(e)[:220])
                if cap is not None and cap.isOpened():
                    return cap, CameraOpenInfo("unique_id", idx, derived, "", "")
    candidates = [idx] + [i for i in range(max_probe_index + 1) if i != idx]
    for i in candidates:
        try:
            cap = cv2.VideoCapture(int(i))
        except Exception:
            continue
        opened = bool(cap.isOpened())
        ok = False
        frame = None
        if opened:
            try:
                ok, frame = cap.read()
            except Exception:
                ok = False
        if opened and ok and frame is not None:
            return cap, CameraOpenInfo("cv2_index", int(i), "", "", "")
        try:
            cap.release()
        except Exception:
            pass
    return None, CameraOpenInfo("failed", idx, uid, "", "No camera opened at any probed index.")


def _raw96_to_png(
    raw: bytes,
    crop_box: Optional[Tuple[int, int, int, int]],
    *,
    frame_side: int = 96,
    out_side: int = 96,
) -> bytes:
    n = len(raw)
    side = int(frame_side)
    out_side = int(out_side)
    if n == side * side * 3:
        # RGB frame from data-collection sketch
        arr = np.frombuffer(raw, dtype=np.uint8).reshape((side, side, 3))
        img = Image.fromarray(arr, mode="RGB")
    else:
        # Grayscale / B-G frame (inference sketch)
        arr = np.frombuffer(raw, dtype=np.uint8).reshape((side, side))
        img = Image.fromarray(arr, mode="L")
    if crop_box is not None:
        x1, y1, x2, y2 = crop_box
        img = img.crop((x1, y1, x2, y2))
    img = img.resize((out_side, out_side))
    return _to_png_bytes(img)


def _raw96_preview_png(raw: bytes, frame_side: int = 96) -> bytes:
    side = int(frame_side)
    n = len(raw)
    if n == side * side * 3:
        arr = np.frombuffer(raw, dtype=np.uint8).reshape((side, side, 3))
        img = Image.fromarray(arr, mode="RGB").resize((side * 3, side * 3), Image.NEAREST)
    else:
        arr = np.frombuffer(raw, dtype=np.uint8).reshape((side, side))
        img = Image.fromarray(arr, mode="L").resize((side * 3, side * 3), Image.NEAREST)
    return _to_png_bytes(img)


def _to_png_bytes(img: Image.Image) -> bytes:
    import io

    bio = io.BytesIO()
    img.save(bio, format="PNG")
    return bio.getvalue()


def _model_input_array_to_preview_image(arr: np.ndarray) -> Image.Image:
    """Convert a model-input array (float32/uint8, any range) to a displayable Image."""
    vis = np.asarray(arr)
    if vis.ndim == 3 and vis.shape[-1] == 1:
        vis = vis[:, :, 0]
    if vis.dtype == np.uint8:
        u8 = vis
    else:
        vis = vis.astype(np.float32)
        if vis.size == 0:
            u8 = np.zeros((1, 1), dtype=np.uint8)
        else:
            vmin = float(np.min(vis))
            vmax = float(np.max(vis))
            if vmin >= 0.0 and vmax <= 1.0 + 1e-6:
                u8 = np.clip(np.round(vis * 255.0), 0, 255).astype(np.uint8)
            elif vmin >= -1.0 - 1e-6 and vmax <= 1.0 + 1e-6:
                u8 = np.clip(np.round((vis + 1.0) * 127.5), 0, 255).astype(np.uint8)
            else:
                span = max(vmax - vmin, 1e-6)
                u8 = np.clip(np.round((vis - vmin) * (255.0 / span)), 0, 255).astype(np.uint8)
    if u8.ndim == 2:
        return Image.fromarray(u8, mode="L")
    return Image.fromarray(u8[:, :, :3], mode="RGB")


def make_hold_button_html(label: str, start_url: str, stop_url: str) -> str:
    start_url_js = start_url.replace("'", "\\'")
    stop_url_js = stop_url.replace("'", "\\'")
    label_js = label.replace("<", "&lt;").replace(">", "&gt;")
    return f"""
<div style="display:flex; flex-direction:column; gap:10px;">
  <button id="holdbtn" style="width:100%; padding:14px 12px; border-radius:10px; border:0; background:#0b5fff; color:white; font-weight:700; font-size:16px;">
    {label_js}
  </button>
  <div id="status" style="font-size:12px; color:rgba(0,0,0,.6);">Hold to start recording, release to stop.</div>
</div>
<script>
const btn = document.getElementById('holdbtn');
const status = document.getElementById('status');
let down = false;
async function doFetch(url) {{
  try {{ await fetch(url, {{method:'GET', mode:'no-cors'}}); }} catch (e) {{}}
}}
function onDown() {{
  if (down) return;
  down = true;
  btn.style.background = '#0647c6';
  status.textContent = 'Recording... release to stop.';
  doFetch('{start_url_js}');
}}
function onUp() {{
  if (!down) return;
  down = false;
  btn.style.background = '#0b5fff';
  status.textContent = 'Stopped.';
  doFetch('{stop_url_js}');
}}
btn.addEventListener('mousedown', onDown);
btn.addEventListener('mouseup', onUp);
btn.addEventListener('mouseleave', onUp);
btn.addEventListener('touchstart', function(e){{ e.preventDefault(); onDown(); }}, {{passive:false}});
btn.addEventListener('touchend', function(e){{ e.preventDefault(); onUp(); }}, {{passive:false}});
</script>
"""
