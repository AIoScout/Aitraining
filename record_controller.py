from __future__ import annotations

import base64
import json
import shutil
import socket
import subprocess
import sys
import threading
import time
import uuid
import zipfile
import urllib.request
from dataclasses import dataclass, replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import parse_qs, urlparse

import numpy as np
from PIL import Image

from camera_permission import ensure_camera_access
from serial_device import SerialFrameReader, list_serial_ports, parse_sync_header
from dataset_io import IMAGE_EXTS, sanitize_class_name
from image_preprocess import (
    CLIP_MODE_JUNCTION,
    PREPROCESS_MODE_AUTO_BY_LABEL,
    PREPROCESS_MODE_JUNCTION,
    PREPROCESS_MODE_MANUAL_ROI,
    PREPROCESS_MODE_NONE,
    PREPROCESS_MODE_SIGN,
    class_clip_mode,
    normalize_class_labels_map,
    normalize_class_preprocess,
    normalize_class_preprocess_map,
    normalize_sample_preprocess_map,
    normalize_manual_roi,
    preprocess_for_label,
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


# #region debug-point B:record-controller-device
def _dbg_device_frame_timeout(hypothesis_id: str, location: str, msg: str, data: dict) -> None:
    _p = ".dbg/device-160-frame.env"
    _u = "http://127.0.0.1:7777/event"
    _s = "device-160-frame"
    try:
        try_paths = [".dbg/device-160-frame.env", ".dbg/device-frame-timeout.env"]
        _c = ""
        for _pp in try_paths:
            try:
                with open(_pp, "r", encoding="utf-8") as f:
                    _c = f.read()
                _p = _pp
                break
            except Exception:
                continue
        for _line in _c.splitlines():
            if _line.startswith("DEBUG_SERVER_URL="):
                _u = _line.split("=", 1)[1].strip() or _u
            elif _line.startswith("DEBUG_SESSION_ID="):
                _s = _line.split("=", 1)[1].strip() or _s
    except Exception:
        pass
    try:
        _payload = {
            "sessionId": _s,
            "runId": "pre-fix",
            "hypothesisId": hypothesis_id,
            "location": location,
            "msg": msg,
            "data": data,
            "ts": int(time.time() * 1000),
        }
        urllib.request.urlopen(
            urllib.request.Request(
                _u,
                data=json.dumps(_payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
            ),
            timeout=0.4,
        ).read()
    except Exception:
        pass
# #endregion


@dataclass
class SessionConfig:
    dataset_root: Path
    serial_port: str
    serial_baud: int
    serial_sync: str
    serial_frame_side: int
    webcam_index: int
    fps: float
    crop_box: Optional[Tuple[int, int, int, int]]


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


class RecordController:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._configs: Dict[str, SessionConfig] = {}
        self._active: Dict[str, Dict[str, Any]] = {}
        self._live: Dict[str, Dict[str, Any]] = {}
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

    def set_config(self, session_id: str, cfg: SessionConfig) -> None:
        with self._lock:
            self._configs[session_id] = cfg

    def update_config(
        self,
        session_id: str,
        *,
        webcam_index: Optional[int] = None,
        serial_port: Optional[str] = None,
        serial_baud: Optional[int] = None,
        serial_sync: Optional[str] = None,
        serial_frame_side: Optional[int] = None,
    ) -> SessionConfig:
        with self._lock:
            cfg = self._configs.get(session_id)
            if cfg is None:
                raise RuntimeError("missing config")
            updated = replace(
                cfg,
                webcam_index=int(cfg.webcam_index if webcam_index is None else webcam_index),
                serial_port=str(cfg.serial_port if serial_port is None else serial_port),
                serial_baud=int(cfg.serial_baud if serial_baud is None else serial_baud),
                serial_sync=str(cfg.serial_sync if serial_sync is None else serial_sync),
                serial_frame_side=int(cfg.serial_frame_side if serial_frame_side is None else serial_frame_side),
            )
            self._configs[session_id] = updated
            return updated

    def status(self, session_id: str) -> Dict[str, Any]:
        with self._lock:
            return dict(self._active.get(session_id, {}))

    def preview_webcam_png(self, webcam_index: int) -> Optional[bytes]:
        permission = ensure_camera_access(webcam_index=int(webcam_index))
        if not permission.allowed:
            return None
        try:
            import cv2
        except Exception:
            return None
        cap = cv2.VideoCapture(int(webcam_index))
        if not cap.isOpened():
            return None
        ok, frame = cap.read()
        cap.release()
        if not ok or frame is None:
            return None
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame)
        return _to_png_bytes(img)

    def preview_serial_png(self, port: str, baud: int, sync_header: str = "", frame_side: int = 96) -> Optional[bytes]:
        if not port:
            return None
        try:
            reader = SerialFrameReader(port=port, baud=int(baud), sync_header=sync_header, frame_side=int(frame_side))
            reader.open()
            try:
                raw = reader.read_frame(timeout_s=2.0)
            finally:
                reader.close()
            side = int(frame_side)
            arr = np.frombuffer(raw, dtype=np.uint8).reshape((side, side))
            img = Image.fromarray(arr, mode="L")
            return _to_png_bytes(img)
        except Exception:
            return None

    def _make_handler(self):
        controller = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                try:
                    controller._handle(self)
                except Exception:
                    self.send_response(500)
                    self.send_header("Content-Type", "text/plain")
                    self.end_headers()
                    self.wfile.write(b"error")

            def do_POST(self) -> None:
                try:
                    controller._handle_post(self)
                except Exception:
                    self.send_response(500)
                    self.send_header("Content-Type", "text/plain")
                    self.end_headers()
                    self.wfile.write(b"error")

            def do_OPTIONS(self) -> None:
                try:
                    controller._handle_options(self)
                except Exception:
                    self.send_response(204)
                    self.end_headers()

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
            if not session_id or not class_name:
                _send_json(req, {"ok": "0", "error": "missing params"}, status=400, cors=True)
                return
            cfg = self._configs.get(session_id)
            if cfg is None:
                _send_json(req, {"ok": "0", "error": "missing config"}, status=400, cors=True)
                return
            _send_json(req, self._class_state_payload(cfg.dataset_root, class_name), cors=True)
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
        if path == "/serial/ports":
            ports = []
            try:
                for p in list_serial_ports():
                    label = f"{p.device} - {p.description}" if getattr(p, "description", "") else p.device
                    ports.append({"device": str(p.device), "label": str(label)})
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
            _send_json(req, {"ok": "1", "class": class_name, "filename": p.name}, cors=True)
            return
        if path == "/live/open":
            session_id = (qs.get("session") or [""])[0]
            source = (qs.get("source") or [""])[0]
            if not session_id or not source:
                _send_json(req, {"ok": "0", "error": "missing params"}, status=400, cors=True)
                return
            self._ensure_live(session_id=session_id, source=source)
            _send_json(req, {"ok": "1"}, cors=True)
            return
        if path == "/live/config":
            session_id = (qs.get("session") or [""])[0]
            if not session_id:
                _send_json(req, {"ok": "0", "error": "missing session"}, status=400, cors=True)
                return
            webcam_index_raw = (qs.get("webcam_index") or [None])[0]
            serial_port_raw = (qs.get("serial_port") or [None])[0]
            serial_baud_raw = (qs.get("serial_baud") or [None])[0]
            serial_sync_raw = (qs.get("serial_sync") or [None])[0]
            serial_frame_side_raw = (qs.get("serial_frame_side") or [None])[0]
            try:
                if serial_sync_raw not in (None, ""):
                    parse_sync_header(serial_sync_raw)
                if serial_frame_side_raw not in (None, ""):
                    side = int(float(serial_frame_side_raw))
                    if side < 8 or side > 512:
                        raise ValueError("invalid serial_frame_side")
                cfg = self.update_config(
                    session_id,
                    webcam_index=None if webcam_index_raw in (None, "") else int(float(webcam_index_raw)),
                    serial_port=serial_port_raw,
                    serial_baud=None if serial_baud_raw in (None, "") else int(float(serial_baud_raw)),
                    serial_sync=None if serial_sync_raw is None else str(serial_sync_raw),
                    serial_frame_side=None if serial_frame_side_raw in (None, "") else int(float(serial_frame_side_raw)),
                )
            except Exception as e:
                _send_json(req, {"ok": "0", "error": str(e)}, status=400, cors=True)
                return
            _send_json(
                req,
                {
                    "ok": "1",
                    "webcam_index": str(int(cfg.webcam_index)),
                    "serial_port": str(cfg.serial_port),
                    "serial_baud": str(int(cfg.serial_baud)),
                    "serial_sync": str(cfg.serial_sync),
                    "serial_frame_side": str(int(cfg.serial_frame_side)),
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
                _send_json(req, {"ok": "0", "error": err}, status=404, cors=True)
                return
            _send_png(req, png, cors=True)
            return
        if path == "/preview/predict":
            session_id = (qs.get("session") or [""])[0]
            source = (qs.get("source") or [""])[0]
            preprocess_mode = (qs.get("preprocess") or [""])[0]
            if not session_id or source not in {"webcam", "device"}:
                _send_json(req, {"ok": "0", "error": "missing params"}, status=400, cors=True)
                return
            try:
                self._ensure_live(session_id=session_id, source=source)
                # Brief wait for the worker to produce its first frame. The serial
                # worker may need a moment to open the port and sync to a frame boundary.
                for _ in range(20):
                    png = self._get_live_preview(session_id=session_id, source=source)
                    if png is not None:
                        break
                    time.sleep(0.12)
                if png is None:
                    err = self._get_live_error(session_id=session_id, source=source) or "preview unavailable"
                    _send_json(req, {"ok": "0", "error": err}, status=404, cors=True)
                    return
                pred = self._preview_predict(session_id=session_id, png=png, preprocess_mode=preprocess_mode)
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
                    "image_b64": base64.b64encode(png).decode("ascii"),
                    "processed_image_b64": str(pred.get("processed_image_b64") or ""),
                    "processed_variant": str(pred.get("processed_variant") or ""),
                },
                cors=True,
            )
            return
        if path == "/export/pick_dir":
            if sys.platform != "darwin":
                _send_json(req, {"ok": "0", "error": "unsupported"}, status=400, cors=True)
                return
            try:
                script = 'POSIX path of (choose folder with prompt "Export folder")'
                proc = subprocess.run(
                    ["osascript", "-e", script],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                if proc.returncode != 0:
                    _send_json(req, {"ok": "0", "canceled": "1"}, cors=True)
                    return
                picked = (proc.stdout or "").strip()
                if not picked:
                    _send_json(req, {"ok": "0", "canceled": "1"}, cors=True)
                    return
                _send_json(req, {"ok": "1", "export_dir": picked}, cors=True)
                return
            except Exception as e:
                _send_json(req, {"ok": "0", "error": str(e)}, status=400, cors=True)
                return
        if path == "/project/pick_save":
            if sys.platform != "darwin":
                _send_json(req, {"ok": "0", "error": "unsupported"}, status=400, cors=True)
                return
            try:
                script = 'POSIX path of (choose file name with prompt "Save Project" default name "project.tmproj")'
                proc = subprocess.run(
                    ["osascript", "-e", script],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                if proc.returncode != 0:
                    _send_json(req, {"ok": "0", "canceled": "1"}, cors=True)
                    return
                picked = (proc.stdout or "").strip()
                if not picked:
                    _send_json(req, {"ok": "0", "canceled": "1"}, cors=True)
                    return
                p = Path(picked).expanduser()
                if p.suffix.lower() != ".tmproj":
                    p = p.with_suffix(".tmproj")
                _send_json(req, {"ok": "1", "save_path": str(p)}, cors=True)
                return
            except Exception as e:
                _send_json(req, {"ok": "0", "error": str(e)}, status=400, cors=True)
                return
        if path == "/project/pick_open":
            if sys.platform != "darwin":
                _send_json(req, {"ok": "0", "error": "unsupported"}, status=400, cors=True)
                return
            try:
                script = 'POSIX path of (choose file with prompt "Open Project")'
                proc = subprocess.run(
                    ["osascript", "-e", script],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                if proc.returncode != 0:
                    _send_json(req, {"ok": "0", "canceled": "1"}, cors=True)
                    return
                picked = (proc.stdout or "").strip()
                if not picked:
                    _send_json(req, {"ok": "0", "canceled": "1"}, cors=True)
                    return
                p = Path(picked).expanduser()
                if p.suffix.lower() != ".tmproj":
                    _send_json(req, {"ok": "0", "error": "not a .tmproj file"}, status=400, cors=True)
                    return
                _send_json(req, {"ok": "1", "open_path": str(p)}, cors=True)
                return
            except Exception as e:
                _send_json(req, {"ok": "0", "error": str(e)}, status=400, cors=True)
                return
        if path == "/live/capture":
            session_id = (qs.get("session") or [""])[0]
            source = (qs.get("source") or [""])[0]
            class_name = (qs.get("class") or [""])[0]
            if not session_id or not source or not class_name:
                _send_json(req, {"ok": "0", "error": "missing params"}, status=400, cors=True)
                return
            self._ensure_live(session_id=session_id, source=source)
            png = self._get_live_capture(session_id=session_id, source=source)
            if png is None:
                err = self._get_live_error(session_id=session_id, source=source) or "capture unavailable"
                _send_json(req, {"ok": "0", "error": err}, status=400, cors=True)
                return
            cfg = self._configs.get(session_id)
            if cfg is None:
                _send_json(req, {"ok": "0", "error": "missing config"}, status=400, cors=True)
                return
            p = _save_png(cfg.dataset_root, class_name, png)
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
            self._stop_live(session_id=session_id, source=source if source else None)
            _send_json(req, {"ok": "1"}, cors=True)
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
            "/preprocess/preview",
            "/preprocess/save_sample",
            "/preview/predict_upload",
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
        if path == "/preprocess/preview":
            session_id = str(payload.get("session") or "").strip()
            class_name = str(payload.get("class") or "").strip()
            filename = str(payload.get("filename") or "").strip()
            class_config = payload.get("class_config")
            sample_config = payload.get("sample_config")
            class_labels = normalize_class_labels_map(payload.get("class_labels"))
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
                    class_labels_request=class_labels,
                )
            except Exception as e:
                _send_json(req, {"ok": "0", "error": str(e)}, status=400, cors=True)
                return
            _send_json(req, {"ok": "1", **preview}, cors=True)
            return
        if path == "/preprocess/save_sample":
            session_id = str(payload.get("session") or "").strip()
            class_name = str(payload.get("class") or "").strip()
            filename = str(payload.get("filename") or "").strip()
            sample_config = payload.get("sample_config")
            class_labels = normalize_class_labels_map(payload.get("class_labels"))
            if not session_id or not class_name or not filename:
                _send_json(req, {"ok": "0", "error": "missing params"}, status=400, cors=True)
                return
            try:
                result = self._preprocess_save_sample(
                    session_id=session_id,
                    class_name=class_name,
                    filename=filename,
                    sample_config=sample_config,
                    class_labels_request=class_labels,
                )
            except Exception as e:
                _send_json(req, {"ok": "0", "error": str(e)}, status=400, cors=True)
                return
            _send_json(req, result, cors=True)
            return
        if path == "/classes/save_labels":
            session_id = str(payload.get("session") or "").strip()
            class_labels = normalize_class_labels_map(payload.get("class_labels"))
            if not session_id:
                _send_json(req, {"ok": "0", "error": "missing params"}, status=400, cors=True)
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
                    class_labels=class_labels if class_labels else None,
                )
                self._rebuild_processed_cache(
                    sess_cfg.dataset_root,
                    class_labels=class_labels if class_labels else None,
                )
            except Exception as e:
                _send_json(req, {"ok": "0", "error": str(e)}, status=400, cors=True)
                return
            _send_json(req, {"ok": "1"}, cors=True)
            return
        if path == "/preview/predict_upload":
            session_id = str(payload.get("session") or "").strip()
            image_b64 = str(payload.get("image_b64") or "").strip()
            preprocess_mode = str(payload.get("preprocess") or "").strip()
            if not session_id or not image_b64:
                _send_json(req, {"ok": "0", "error": "missing params"}, status=400, cors=True)
                return
            try:
                png = base64.b64decode(image_b64, validate=True)
                pred = self._preview_predict(session_id=session_id, png=png, preprocess_mode=preprocess_mode)
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
            _send_json(req, {"ok": "1", "export_dir": str(out_dir)}, cors=True)
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
        img = img.resize((96, 96))
        out_png = _to_png_bytes(img)
        p = _save_png(cfg.dataset_root, class_name, out_png)
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

    def _class_labels_load(self, dataset_root: Path) -> Dict[str, str]:
        raw = self._classes_meta_load(dataset_root)
        return normalize_class_labels_map(raw.get("class_labels"))

    def _classes_save(
        self,
        dataset_root: Path,
        classes: List[str],
        class_preprocess: Optional[Dict[str, Dict[str, Any]]] = None,
        sample_preprocess: Optional[Dict[str, Dict[str, Dict[str, Any]]]] = None,
        class_labels: Optional[Dict[str, str]] = None,
    ) -> None:
        p = self._classes_meta_path(dataset_root)
        payload: Dict[str, Any] = {"classes": list(classes)}
        existing = self._class_preprocess_load(dataset_root)
        merged = normalize_class_preprocess_map(class_preprocess if class_preprocess is not None else existing)
        existing_sample = self._sample_preprocess_load(dataset_root)
        merged_sample = normalize_sample_preprocess_map(sample_preprocess if sample_preprocess is not None else existing_sample)
        existing_labels = self._class_labels_load(dataset_root)
        merged_labels = normalize_class_labels_map(class_labels if class_labels is not None else existing_labels)
        if merged:
            payload["class_preprocess"] = merged
        if merged_sample:
            payload["sample_preprocess"] = merged_sample
        if class_labels is not None or existing_labels:
            payload["class_labels"] = merged_labels
        p.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

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
        class_labels = self._class_labels_load(cfg.dataset_root)
        classes = list(classes) + [self._next_class_name(classes)]
        self._classes_save(cfg.dataset_root, classes, class_preprocess=class_preprocess, sample_preprocess=sample_preprocess, class_labels=class_labels)
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
        class_labels = self._class_labels_load(cfg.dataset_root)
        class_preprocess.pop(actual_name, None)
        sample_preprocess.pop(actual_name, None)
        class_labels.pop(actual_name, None)
        if class_dir.exists():
            shutil.rmtree(class_dir)
        if processed_dir.exists():
            shutil.rmtree(processed_dir)
        self._classes_save(cfg.dataset_root, classes, class_preprocess=class_preprocess, sample_preprocess=sample_preprocess, class_labels=class_labels)
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
        class_labels = self._class_labels_load(cfg.dataset_root)
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
        if old_name in class_labels:
            class_labels[new_name] = class_labels.pop(old_name)
        self._classes_save(cfg.dataset_root, updated, class_preprocess=class_preprocess, sample_preprocess=sample_preprocess, class_labels=class_labels)
        return updated

    def _start_record(self, session_id: str, source: str, class_name: str) -> None:
        # Stop any live worker that may be using the same hardware resource,
        # then wait for it to release the resource before starting recording.
        self._stop_live(session_id=session_id, source=source)
        _deadline = time.time() + 2.5
        _key = self._live_key(session_id, source)
        while time.time() < _deadline:
            if not self._live_running(_key):
                break
            time.sleep(0.05)

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

        t = threading.Thread(
            target=self._record_worker,
            args=(session_id, source, class_name),
            daemon=True,
        )
        t.start()

    def _stop_record(self, session_id: str) -> None:
        with self._lock:
            cur = self._active.get(session_id)
            if cur:
                cur["recording"] = "0"
        cond = self._record_conds.get(session_id)
        if cond is not None:
            with cond:
                cond.notify_all()

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

    def _train_result_path(self, dataset_root: Path) -> Path:
        return dataset_root.parent / "tm_train_latest.json"

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
        return {
            "outbox_dir": str(outbox),
            "labels": labels,
            "model_bytes": tflite_path.stat().st_size,
        }

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
        train_class_labels = normalize_class_labels_map(meta.get("class_labels") or {})
        # Merge live class_labels from the dataset so classes left at the
        # default "Sign" value (never explicitly toggled in the UI) still
        # resolve correctly rather than falling through to the keyword
        # heuristic in class_clip_mode, which misclassifies names containing
        # road-direction tokens (END, LEFT, RIGHT, STRAIGHT, etc.).
        live_class_labels = self._class_labels_load(cfg.dataset_root)
        class_labels = {**live_class_labels, **train_class_labels}
        label_types: List[int] = []
        for label in labels:
            if label in class_labels:
                label_types.append(1 if class_labels[label] == "road" else 0)
            else:
                label_types.append(1 if class_clip_mode(label, class_labels=class_labels) == CLIP_MODE_JUNCTION else 0)
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
        from trainer import export_tflite_c_sources, export_tflite_resolver_header

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
            export_dir / "model.h",
            export_dir / "model.cpp",
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
        resolver_hdr = export_tflite_resolver_header(source_bytes)
        (export_dir / f"{safe_base}.tflite").write_bytes(source_bytes)
        (export_dir / h_name).write_text(hdr, encoding="utf-8")
        (export_dir / cpp_name).write_text(f'#include "{h_name}"\n\n' + src, encoding="utf-8")
        (export_dir / "model.h").write_text(hdr, encoding="utf-8")
        (export_dir / "model.cpp").write_text('#include "model.h"\n\n' + src, encoding="utf-8")
        (export_dir / "model_resolver.h").write_text(resolver_hdr, encoding="utf-8")
        (export_dir / "labels.txt").write_text("\n".join([str(x) for x in labels]) + "\n", encoding="utf-8")
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
                "#include <stdint.h>",
                '#include "image_provider.h"',
                "",
                "// Derived from image_provider.h — change OUT_WIDTH/OUT_HEIGHT there.",
                "constexpr int kNumCols = OUT_WIDTH;",
                "constexpr int kNumRows = OUT_HEIGHT;",
                "constexpr int kNumChannels = " + str(int(channels)) + ";",
                "",
                "constexpr int kMaxImageSize = kNumCols * kNumRows * kNumChannels;",
                "",
                "constexpr int kCategoryCount = " + str(int(len(labels))) + ";",
                "extern const char* kCategoryLabels[kCategoryCount];",
                "extern const uint8_t kClassTypes[kCategoryCount];  // 0=sign, 1=road",
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
                "const uint8_t kClassTypes[kCategoryCount] = {",
                f"    {', '.join([str(int(t)) for t in label_types])},",
                "};",
                "",
            ]
        )
        (export_dir / "model_settings.h").write_text(model_settings_h, encoding="utf-8")
        (export_dir / "model_settings.cpp").write_text(model_settings_cpp, encoding="utf-8")
        return export_dir

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
        class_labels = normalize_class_labels_map(merged_project_state.get("class_labels"))
        self._classes_save(
            dataset_root,
            self._classes_load(dataset_root),
            class_preprocess=class_preprocess,
            sample_preprocess=sample_preprocess,
            class_labels=class_labels,
        )
        if processed_cache_dir.exists():
            shutil.rmtree(processed_cache_dir)
        self._rebuild_processed_cache(
            dataset_root,
            class_preprocess=class_preprocess,
            sample_preprocess=sample_preprocess,
            class_labels=class_labels,
        )
        if processed_cache_dir.exists():
            manifest["processed_cache_dir"] = "processed_cache"

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
                    if rel.as_posix() in {"labels.txt", "labels.json"}:
                        continue
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
            fixed_labels = self._classes_load(dataset_root)
            (dataset_root / "labels.txt").write_text("\n".join(fixed_labels) + "\n", encoding="utf-8")
            (dataset_root / "labels.json").write_text(
                json.dumps({"labels": fixed_labels}, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )

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
            if isinstance(project_state, dict):
                class_preprocess = normalize_class_preprocess_map(project_state.get("class_preprocess"))
                sample_preprocess = normalize_sample_preprocess_map(project_state.get("sample_preprocess"))
                # Only pass class_labels if explicitly present in project_state;
                # otherwise preserve whatever was restored from tm_classes.json.
                if "class_labels" in project_state:
                    class_labels = normalize_class_labels_map(project_state["class_labels"])
                else:
                    class_labels = None
                self._classes_save(
                    dataset_root,
                    self._classes_load(dataset_root),
                    class_preprocess=class_preprocess,
                    sample_preprocess=sample_preprocess,
                    class_labels=class_labels,
                )
            else:
                class_preprocess = self._class_preprocess_load(dataset_root)
                sample_preprocess = self._sample_preprocess_load(dataset_root)
                class_labels = self._class_labels_load(dataset_root)
            self._rebuild_processed_cache(
                dataset_root,
                class_preprocess=class_preprocess,
                sample_preprocess=sample_preprocess,
                class_labels=class_labels,
            )
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
        dataset_root = cfg.dataset_root
        workspace_root = dataset_root.parent
        classes_meta = self._classes_meta_path(dataset_root)
        processed_cache_dir = self._processed_cache_dir(dataset_root)
        latest = self._train_result_path(dataset_root)
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
                "img_size": 96,
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
            if preprocess_mode in {"auto_by_label", "manual_roi", "none", "sign", "junction"}:
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

    def _processed_preview_item_payload(self, path: Path) -> Dict[str, str]:
        return _preview_item_payload(path)

    def _preprocess_image_png(self, png: bytes, label_name: str, class_config: Any, sample_config: Any = None, class_labels: Optional[Dict[str, str]] = None) -> bytes:
        config = normalize_class_preprocess(sample_config if sample_config is not None else class_config)
        img = Image.open(_bytes_io(png)).convert("L")
        arr = preprocess_for_label(
            np.asarray(img),
            out_size=96,
            color_mode="grayscale",
            label_name=label_name,
            preprocess_mode=str(config.get("mode") or PREPROCESS_MODE_AUTO_BY_LABEL),
            manual_roi=config.get("manual_roi"),
            class_labels=class_labels,
        )
        out = np.asarray(np.clip(arr[:, :, 0] * 255.0, 0.0, 255.0), dtype=np.uint8)
        return _to_png_bytes(Image.fromarray(out, mode="L"))

    def _rebuild_processed_cache(
        self,
        dataset_root: Path,
        class_preprocess: Optional[Dict[str, Dict[str, Any]]] = None,
        sample_preprocess: Optional[Dict[str, Dict[str, Dict[str, Any]]]] = None,
        class_labels: Optional[Dict[str, str]] = None,
    ) -> None:
        classes = self._classes_load(dataset_root)
        cache_root = self._processed_cache_dir(dataset_root)
        cache_root.mkdir(parents=True, exist_ok=True)
        class_preprocess = normalize_class_preprocess_map(class_preprocess if class_preprocess is not None else self._class_preprocess_load(dataset_root))
        sample_preprocess = normalize_sample_preprocess_map(sample_preprocess if sample_preprocess is not None else self._sample_preprocess_load(dataset_root))
        class_labels = normalize_class_labels_map(class_labels if class_labels is not None else self._class_labels_load(dataset_root))
        for class_name in classes:
            src_dir = dataset_root / sanitize_class_name(class_name)
            dst_dir = cache_root / sanitize_class_name(class_name)
            if dst_dir.exists():
                shutil.rmtree(dst_dir)
            dst_dir.mkdir(parents=True, exist_ok=True)
            if not src_dir.exists():
                continue
            config = class_preprocess.get(class_name, {})
            class_sample_map = sample_preprocess.get(class_name, {})
            for src in _list_class_image_files(src_dir):
                try:
                    png = src.read_bytes()
                    out_png = self._preprocess_image_png(
                        png,
                        label_name=class_name,
                        class_config=config,
                        sample_config=class_sample_map.get(src.name),
                        class_labels=class_labels,
                    )
                    (dst_dir / f"{src.stem}.png").write_bytes(out_png)
                except Exception:
                    continue

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
        class_labels_request: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        with self._lock:
            cfg = self._configs.get(session_id)
        if cfg is None:
            raise RuntimeError("missing config")
        class_dir = cfg.dataset_root / sanitize_class_name(class_name)
        src = (class_dir / Path(filename).name).resolve()
        if src.parent != class_dir.resolve() or not src.exists():
            raise FileNotFoundError("sample not found")
        disk_labels = self._class_labels_load(cfg.dataset_root)
        merged_labels = normalize_class_labels_map(class_labels_request if class_labels_request is not None else {})
        effective_labels = {**disk_labels, **merged_labels} if merged_labels else disk_labels
        out_png = self._preprocess_image_png(
            src.read_bytes(),
            label_name=class_name,
            class_config=class_config,
            sample_config=sample_config,
            class_labels=effective_labels,
        )
        return {"image_b64": base64.b64encode(out_png).decode("ascii")}

    def _preprocess_save_sample(
        self,
        session_id: str,
        class_name: str,
        filename: str,
        sample_config: Any,
        class_labels_request: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        with self._lock:
            cfg = self._configs.get(session_id)
        if cfg is None:
            raise RuntimeError("missing config")
        class_dir = (cfg.dataset_root / sanitize_class_name(class_name)).resolve()
        src = (class_dir / Path(filename).name).resolve()
        if src.parent != class_dir or not src.exists():
            raise FileNotFoundError("sample not found")

        class_preprocess = self._class_preprocess_load(cfg.dataset_root)
        sample_preprocess = self._sample_preprocess_load(cfg.dataset_root)
        class_default = normalize_class_preprocess(class_preprocess.get(class_name))
        normalized = normalize_class_preprocess(sample_config)

        class_map = sample_preprocess.get(class_name, {})
        if normalized == class_default:
            if isinstance(class_map, dict):
                class_map.pop(src.name, None)
                if not class_map:
                    sample_preprocess.pop(class_name, None)
        else:
            if not isinstance(class_map, dict):
                class_map = {}
            class_map[src.name] = normalized
            sample_preprocess[class_name] = class_map

        disk_labels = self._class_labels_load(cfg.dataset_root)
        merged_labels = normalize_class_labels_map(class_labels_request if class_labels_request is not None else {})
        effective_labels = {**disk_labels, **merged_labels} if merged_labels else disk_labels

        self._classes_save(
            cfg.dataset_root,
            self._classes_load(cfg.dataset_root),
            class_preprocess=class_preprocess,
            sample_preprocess=sample_preprocess,
            class_labels=effective_labels,
        )

        processed_dir = self._processed_cache_dir(cfg.dataset_root) / sanitize_class_name(class_name)
        processed_dir.mkdir(parents=True, exist_ok=True)
        out_png = self._preprocess_image_png(
            src.read_bytes(),
            label_name=class_name,
            class_config=class_default,
            sample_config=sample_preprocess.get(class_name, {}).get(src.name),
            class_labels=effective_labels,
        )
        (processed_dir / f"{src.stem}.png").write_bytes(out_png)
        return {
            "ok": "1",
            "sample_preprocess": self._sample_preprocess_load(cfg.dataset_root),
        }

    def _project_state_payload(self, dataset_root: Path, project_state: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        classes = self._classes_load(dataset_root)
        class_preprocess = self._class_preprocess_load(dataset_root)
        sample_preprocess = self._sample_preprocess_load(dataset_root)
        class_labels = self._class_labels_load(dataset_root)
        if isinstance(project_state, dict):
            class_preprocess = normalize_class_preprocess_map(project_state.get("class_preprocess") or class_preprocess)
            sample_preprocess = normalize_sample_preprocess_map(project_state.get("sample_preprocess") or sample_preprocess)
            class_labels = normalize_class_labels_map(project_state.get("class_labels") or class_labels)
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
                    for p in files:
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
            "class_labels": class_labels,
            "export_enabled": "1" if export_enabled else "0",
        }
        if train_cfg:
            payload["train_cfg"] = train_cfg
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
        if not latest.exists():
            raise RuntimeError("missing trained model")
        meta = json.loads(latest.read_text(encoding="utf-8"))
        tflite_path = Path(str(meta.get("tflite_path") or "")).expanduser()
        if not tflite_path.exists():
            raise RuntimeError("missing .tflite file")
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
            "labels": list(labels),
            "preprocess_mode": str(meta.get("preprocess_mode") or PREPROCESS_MODE_AUTO_BY_LABEL) if isinstance(meta, dict) else PREPROCESS_MODE_AUTO_BY_LABEL,
            "manual_roi": normalize_manual_roi(meta.get("manual_roi")) if isinstance(meta, dict) else None,
            "class_preprocess": normalize_class_preprocess_map(meta.get("class_preprocess")) if isinstance(meta, dict) else {},
            "class_labels": normalize_class_labels_map(meta.get("class_labels")) if isinstance(meta, dict) else {},
            "interpreter": interpreter,
            "input_details": input_details,
            "output_details": output_details,
        }
        with self._lock:
            self._preview[session_id] = cur
        return cur

    def _preview_predict(self, session_id: str, png: bytes, preprocess_mode: str = "") -> Dict[str, Any]:
        model = self._preview_load_model(session_id=session_id)
        interpreter = model["interpreter"]
        input_details = model["input_details"]
        output_details = model["output_details"]
        labels: List[str] = list(model.get("labels") or [])
        model_preprocess_mode = str(model.get("preprocess_mode") or PREPROCESS_MODE_AUTO_BY_LABEL)
        preprocess_mode = str(preprocess_mode or model_preprocess_mode or PREPROCESS_MODE_AUTO_BY_LABEL).strip().lower()
        manual_roi = normalize_manual_roi(model.get("manual_roi"))
        class_preprocess = normalize_class_preprocess_map(model.get("class_preprocess"))
        class_labels = normalize_class_labels_map(model.get("class_labels"))
        shape_raw = input_details.get("shape")
        shape = tuple(int(x) for x in (shape_raw if shape_raw is not None else []))
        if len(shape) != 4:
            raise RuntimeError("unsupported input shape")
        _, h, w, c = shape
        img = Image.open(_bytes_io(png)).convert("L" if int(c) == 1 else "RGB")
        prepared = prepare_inference_inputs(
            np.asarray(img),
            out_size=int(w),
            color_mode="grayscale" if int(c) == 1 else "rgb",
            preprocess_mode=preprocess_mode,
            manual_roi=manual_roi,
            class_preprocess=class_preprocess,
            class_labels=class_labels,
        )
        qscale = 0.0
        qzp = 0
        q = input_details.get("quantization")
        if isinstance(q, (tuple, list)) and len(q) == 2:
            qscale = float(q[0] or 0.0)
            qzp = int(q[1] or 0)
        dtype = input_details.get("dtype")

        def _invoke_one(arr: np.ndarray) -> np.ndarray:
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
            if odtype is not None and odtype != np.float32 and oscale > 0:
                scores = (out.astype(np.float32) - float(ozp)) * float(oscale)
            else:
                scores = out.astype(np.float32)
            scores = scores - float(np.max(scores))
            expv = np.exp(scores)
            return expv / float(np.sum(expv) + 1e-12)

        variant_probs = {name: _invoke_one(arr) for name, arr in prepared.items()}

        if preprocess_mode == PREPROCESS_MODE_AUTO_BY_LABEL and "sign" in variant_probs and "junction" in variant_probs and labels:
            sign_probs = variant_probs["sign"]
            junction_probs = variant_probs["junction"]
            probs = np.zeros_like(sign_probs)
            for idx, label_name in enumerate(labels):
                probs[idx] = junction_probs[idx] if class_clip_mode(label_name, class_labels=class_labels) == CLIP_MODE_JUNCTION else sign_probs[idx]
            total = float(np.sum(probs))
            if total > 0.0:
                probs = probs / total
        else:
            probs = next(iter(variant_probs.values()))

        if not labels:
            labels = [f"Class {i+1}" for i in range(int(probs.shape[0]))]
        top_i = int(np.argmax(probs)) if probs.size else 0
        processed_variant = next(iter(prepared.keys()), "")
        if preprocess_mode == PREPROCESS_MODE_AUTO_BY_LABEL and labels:
            top_label = str(labels[top_i]) if 0 <= top_i < len(labels) else ""
            processed_variant = "junction" if class_clip_mode(top_label, class_labels=class_labels) == CLIP_MODE_JUNCTION else "sign"
        processed_arr = np.asarray(prepared.get(processed_variant) if processed_variant in prepared else next(iter(prepared.values())))
        processed_img = _model_input_array_to_preview_image(processed_arr)
        processed_png = _to_png_bytes(processed_img)
        return {
            "labels": labels,
            "probs": [float(x) for x in probs.tolist()],
            "top_label": str(labels[top_i]) if 0 <= top_i < len(labels) else "",
            "top_prob": float(probs[top_i]) if probs.size else 0.0,
            "processed_image_b64": base64.b64encode(processed_png).decode("ascii"),
            "processed_variant": str(processed_variant or ""),
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

    def _train_status_payload(self, session_id: str) -> Dict[str, Any]:
        with self._lock:
            cur = self._train.get(session_id) or {"running": "0", "done": "0", "progress": 0.0, "message": "", "error": "", "result_path": ""}
            return {
                "ok": "1",
                "running": str(cur.get("running") or "0"),
                "done": str(cur.get("done") or "0"),
                "progress": float(cur.get("progress") or 0.0),
                "message": str(cur.get("message") or ""),
                "error": str(cur.get("error") or ""),
                "result_path": str(cur.get("result_path") or ""),
            }

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

            cfg = TrainConfig(
                img_size=int(cfg_dict.get("img_size") or 96),
                color_mode="grayscale",
                batch_size=int(cfg_dict.get("batch_size") or 16),
                epochs=int(cfg_dict.get("epochs") or 10),
                validation_split=float(cfg_dict.get("validation_split") or 0.2),
                seed=int(cfg_dict.get("seed") or 42),
                optimizer=str(cfg_dict.get("optimizer") or "adam"),
                learning_rate=float(cfg_dict.get("learning_rate") or 0.001),
                conv1_filters=int(cfg_dict.get("conv1_filters") or 8),
                conv2_filters=int(cfg_dict.get("conv2_filters") or 16),
                dense_units=int(cfg_dict.get("dense_units") or 32),
                representative_samples=int(cfg_dict.get("representative_samples") or 200),
                preprocess_mode=str(cfg_dict.get("preprocess_mode") or PREPROCESS_MODE_AUTO_BY_LABEL),
                manual_roi=normalize_manual_roi(cfg_dict.get("manual_roi")),
                class_preprocess=normalize_class_preprocess_map(cfg_dict.get("class_preprocess")),
                sample_preprocess=normalize_sample_preprocess_map(cfg_dict.get("sample_preprocess")),
                class_labels=normalize_class_labels_map(cfg_dict.get("class_labels")),
                use_preprocessed_dataset=True,
            )

            workspace_root = sess_cfg.dataset_root.parent
            run_dir = new_run_dir(workspace_root / "runs")
            processed_dataset_dir = self._processed_cache_dir(sess_cfg.dataset_root)

            self._train_set(session_id, progress=0.01, message="Preparing ROI dataset...")
            self._rebuild_processed_cache(
                sess_cfg.dataset_root,
                class_preprocess=cfg.class_preprocess,
                sample_preprocess=cfg.sample_preprocess,
                class_labels=cfg.class_labels,
            )

            def on_progress(p: float, msg: str) -> None:
                self._train_set(session_id, progress=p, message=msg)

            self._train_set(session_id, progress=0.03, message="Starting training...")
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
                "img_size": int(cfg.img_size),
                "color_mode": str(cfg.color_mode),
                "preprocess_mode": str(cfg.preprocess_mode),
                "manual_roi": list(cfg.manual_roi) if cfg.manual_roi is not None else None,
                "class_preprocess": normalize_class_preprocess_map(cfg.class_preprocess),
                "sample_preprocess": normalize_sample_preprocess_map(cfg.sample_preprocess),
                "class_labels": normalize_class_labels_map(cfg.class_labels),
                "trained_dataset_dir": str(processed_dataset_dir),
            }
            latest.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            self._train_set(session_id, progress=1.0, message="Done.", result_path=str(latest), done=True)
        except Exception as e:
            self._train_set(session_id, error=str(e), done=True)

    def _record_worker(self, session_id: str, source: str, class_name: str) -> None:
        cfg = self._configs.get(session_id)
        if cfg is None:
            return
        interval = 1.0 / max(1.0, float(cfg.fps))
        try:
            if source == "device":
                self._record_serial(session_id, cfg, class_name, interval)
                return
            if source == "webcam":
                self._record_webcam(session_id, cfg, class_name, interval)
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

    def _live_key(self, session_id: str, source: str) -> str:
        return f"{session_id}:{source}"

    def _stop_live(self, session_id: str, source: Optional[str] = None) -> None:
        _dbg_device_frame_timeout(
            "C",
            "record_controller.py:_stop_live",
            "[DEBUG] stop live requested",
            {"session_id": session_id, "source": source or "", "thread": threading.current_thread().name},
        )
        with self._lock:
            keys = [self._live_key(session_id, source)] if source else [k for k in self._live if k.startswith(f"{session_id}:")]
            for key in keys:
                state = self._live.get(key)
                if state is not None:
                    state["running"] = False

    def _ensure_live(self, session_id: str, source: str) -> None:
        cfg = self._configs.get(session_id)
        if cfg is None:
            raise RuntimeError("missing config")
        key = self._live_key(session_id, source)
        now = time.time()
        _dbg_device_frame_timeout(
            "C",
            "record_controller.py:_ensure_live",
            "[DEBUG] ensure live entered",
            {
                "session_id": session_id,
                "source": source,
                "key": key,
                "serial_port": str(cfg.serial_port),
                "serial_baud": int(cfg.serial_baud),
                "serial_sync": str(cfg.serial_sync),
                "thread": threading.current_thread().name,
            },
        )
        with self._lock:
            for stale_key, stale_state in list(self._live.items()):
                try:
                    if stale_state.get("running"):
                        continue
                    if now - float(stale_state.get("last_touch") or 0.0) > 30.0:
                        self._live.pop(stale_key, None)
                except Exception:
                    continue
            for other_key, state in list(self._live.items()):
                if other_key.startswith(f"{session_id}:") and other_key != key:
                    state["running"] = False
            cur = self._live.get(key)
            if cur is not None and cur.get("running"):
                cur["last_touch"] = now
                cur["cfg"] = cfg
                _dbg_device_frame_timeout(
                    "C",
                    "record_controller.py:_ensure_live",
                    "[DEBUG] ensure live reused running state",
                    {"session_id": session_id, "source": source, "key": key, "thread": threading.current_thread().name},
                )
                return
            state = {
                "running": True,
                "last_touch": now,
                "source": source,
                "cfg": cfg,
                "preview_png": None,
                "capture_png": None,
                "error": "",
            }
            self._live[key] = state
        _dbg_device_frame_timeout(
            "C",
            "record_controller.py:_ensure_live",
            "[DEBUG] ensure live started worker",
            {"session_id": session_id, "source": source, "key": key, "thread": threading.current_thread().name},
        )
        t = threading.Thread(target=self._live_worker, args=(key, session_id, source), daemon=True)
        t.start()

    def _get_live_preview(self, session_id: str, source: str) -> Optional[bytes]:
        key = self._live_key(session_id, source)
        with self._lock:
            state = self._live.get(key)
            if state is None:
                return None
            state["last_touch"] = time.time()
            data = state.get("preview_png")
            return bytes(data) if isinstance(data, (bytes, bytearray)) else None

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

    def _live_running(self, key: str) -> bool:
        with self._lock:
            state = self._live.get(key)
            if state is None:
                return False
            if not state.get("running"):
                return False
            if time.time() - float(state.get("last_touch") or 0.0) > 8.0:
                state["running"] = False
                return False
            return True

    def _live_set(self, key: str, *, preview_png: Optional[bytes] = None, capture_png: Optional[bytes] = None, error: Optional[str] = None) -> None:
        with self._lock:
            state = self._live.get(key)
            if state is None:
                return
            if preview_png is not None:
                state["preview_png"] = preview_png
            if capture_png is not None:
                state["capture_png"] = capture_png
            if error is not None:
                state["error"] = error

    def _live_worker(self, key: str, session_id: str, source: str) -> None:
        cfg = self._configs.get(session_id)
        if cfg is None:
            return
        try:
            if source == "webcam":
                self._live_webcam(key, cfg)
            elif source == "device":
                self._live_serial(key, cfg)
        finally:
            with self._lock:
                state = self._live.get(key)
                if state is not None:
                    state["running"] = False
                    state["last_touch"] = time.time()

    def _live_serial(self, key: str, cfg: SessionConfig) -> None:
        if not cfg.serial_port:
            self._live_set(key, error="Serial port is not configured.")
            return
        last_error = ""
        _dbg_device_frame_timeout(
            "B",
            "record_controller.py:_live_serial",
            "[DEBUG] live serial worker started",
            {
                "key": key,
                "serial_port": str(cfg.serial_port),
                "serial_baud": int(cfg.serial_baud),
                "serial_sync": str(cfg.serial_sync),
                "thread": threading.current_thread().name,
            },
        )
        reader = SerialFrameReader(
            port=cfg.serial_port,
            baud=int(cfg.serial_baud),
            sync_header=cfg.serial_sync,
            frame_side=int(cfg.serial_frame_side),
        )
        try:
            reader.open()
        except Exception as e:
            self._live_set(key, error=str(e) or "Unable to open serial port.")
            return
        first_frame = True
        try:
            while self._live_running(key):
                try:
                    # First frame after port-open needs longer to sync (reset_input_buffer
                    # dropped any partial frame). Subsequent frames arrive predictably.
                    timeout_s = 3.0 if first_frame else 0.6
                    raw = reader.read_frame(timeout_s=timeout_s)
                    first_frame = False
                    capture_png = _raw96_to_png(
                        raw,
                        crop_box=cfg.crop_box,
                        frame_side=int(cfg.serial_frame_side),
                        out_side=int(cfg.serial_frame_side),
                    )
                    self._live_set(key, preview_png=capture_png, capture_png=capture_png, error="")
                    last_error = ""
                    _dbg_device_frame_timeout(
                        "B",
                        "record_controller.py:_live_serial",
                        "[DEBUG] live serial frame updated",
                        {"key": key, "raw_len": len(raw), "thread": threading.current_thread().name},
                    )
                except Exception as e:
                    msg = str(e) or "Unable to read from serial device."
                    if len(msg) > 220:
                        msg = msg[:220] + "..."
                    if msg != last_error:
                        self._live_set(key, error=msg)
                        last_error = msg
                        _dbg_device_frame_timeout(
                            "B",
                            "record_controller.py:_live_serial",
                            "[DEBUG] live serial frame failed",
                            {"key": key, "error": msg, "thread": threading.current_thread().name},
                        )
                    time.sleep(0.02)
        finally:
            try:
                reader.close()
            except Exception:
                pass
        _dbg_device_frame_timeout(
            "C",
            "record_controller.py:_live_serial",
            "[DEBUG] live serial worker exited",
            {"key": key, "thread": threading.current_thread().name},
        )

    def _live_webcam(self, key: str, cfg: SessionConfig) -> None:
        permission = ensure_camera_access(webcam_index=int(cfg.webcam_index), probe_open=False)
        if not permission.allowed:
            self._live_set(key, error=permission.message or "Camera permission denied.")
            return
        try:
            import cv2
        except Exception:
            self._live_set(key, error="missing opencv-python")
            return
        cap, actual_index = _open_working_camera(int(cfg.webcam_index))
        if cap is None:
            self._live_set(key, error="Unable to open a readable webcam stream.")
            return
        try:
            while self._live_running(key):
                ok, frame = cap.read()
                if not ok or frame is None:
                    time.sleep(0.08)
                    continue
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
                img = img.resize((96, 96))
                self._live_set(key, preview_png=preview_png, capture_png=_to_png_bytes(img), error="")
                time.sleep(0.06)
        finally:
            cap.release()

    def _preview_frame(self, session_id: str, source: str) -> Optional[bytes]:
        cfg = self._configs.get(session_id)
        if cfg is None:
            return None
        if source == "device":
            return self.preview_serial_png(
                cfg.serial_port,
                int(cfg.serial_baud),
                str(cfg.serial_sync),
                frame_side=int(cfg.serial_frame_side),
            )
        if source == "webcam":
            return self.preview_webcam_png(int(cfg.webcam_index))
        return None

    def _capture_single(self, session_id: str, source: str) -> Optional[bytes]:
        cfg = self._configs.get(session_id)
        if cfg is None:
            return None
        if source == "device":
            try:
                _dbg_device_frame_timeout(
                    "B",
                    "record_controller.py:_capture_single",
                    "[DEBUG] device capture started",
                    {
                        "session_id": session_id,
                        "source": source,
                        "serial_port": str(cfg.serial_port),
                        "serial_baud": int(cfg.serial_baud),
                        "serial_sync": str(cfg.serial_sync),
                        "thread": threading.current_thread().name,
                    },
                )
                reader = SerialFrameReader(
                    port=cfg.serial_port,
                    baud=int(cfg.serial_baud),
                    sync_header=cfg.serial_sync,
                    frame_side=int(cfg.serial_frame_side),
                )
                reader.open()
                try:
                    raw = reader.read_frame(timeout_s=2.0)
                finally:
                    reader.close()
                _dbg_device_frame_timeout(
                    "B",
                    "record_controller.py:_capture_single",
                    "[DEBUG] device capture succeeded",
                    {"session_id": session_id, "source": source, "raw_len": len(raw), "thread": threading.current_thread().name},
                )
                return _raw96_to_png(
                    raw,
                    crop_box=cfg.crop_box,
                    frame_side=int(cfg.serial_frame_side),
                    out_side=int(cfg.serial_frame_side),
                )
            except Exception as e:
                _dbg_device_frame_timeout(
                    "A",
                    "record_controller.py:_capture_single",
                    "[DEBUG] device capture failed",
                    {
                        "session_id": session_id,
                        "source": source,
                        "error": str(e),
                        "thread": threading.current_thread().name,
                    },
                )
                return None
        if source == "webcam":
            permission = ensure_camera_access(webcam_index=int(cfg.webcam_index), probe_open=False)
            if not permission.allowed:
                return None
            try:
                import cv2
            except Exception:
                return None
            cap, _actual_index = _open_working_camera(int(cfg.webcam_index))
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
            img = img.resize((96, 96))
            return _to_png_bytes(img)
        return None

    def _is_recording(self, session_id: str) -> bool:
        with self._lock:
            return self._active.get(session_id, {}).get("recording") == "1"

    def _record_serial(self, session_id: str, cfg: SessionConfig, class_name: str, interval: float) -> None:
        reader = SerialFrameReader(
            port=cfg.serial_port,
            baud=int(cfg.serial_baud),
            sync_header=cfg.serial_sync,
            frame_side=int(cfg.serial_frame_side),
        )
        try:
            reader.open()
            while self._is_recording(session_id):
                raw = reader.read_frame(timeout_s=2.0)
                png = _raw96_to_png(
                    raw,
                    crop_box=cfg.crop_box,
                    frame_side=int(cfg.serial_frame_side),
                    out_side=int(cfg.serial_frame_side),
                )
                p = _save_png(cfg.dataset_root, class_name, png)
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
        permission = ensure_camera_access(webcam_index=int(cfg.webcam_index))
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

        cap = cv2.VideoCapture(int(cfg.webcam_index))
        if not cap.isOpened():
            with self._lock:
                self._active[session_id] = {"recording": "0", "error": "webcam open failed"}
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
                img = img.resize((96, 96))
                png = _to_png_bytes(img)
                p = _save_png(cfg.dataset_root, class_name, png)
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
            cap.release()

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


def _send_json(req: BaseHTTPRequestHandler, obj: Dict[str, Any], status: int = 200, cors: bool = False) -> None:
    data = json.dumps(obj).encode("utf-8")
    req.send_response(status)
    if cors:
        req.send_header("Access-Control-Allow-Origin", "*")
    req.send_header("Content-Type", "application/json")
    req.send_header("Content-Length", str(len(data)))
    req.end_headers()
    req.wfile.write(data)


def _send_png(req: BaseHTTPRequestHandler, data: bytes, cors: bool = False) -> None:
    req.send_response(200)
    if cors:
        req.send_header("Access-Control-Allow-Origin", "*")
    req.send_header("Cache-Control", "no-store, max-age=0")
    req.send_header("Content-Type", "image/png")
    req.send_header("Content-Length", str(len(data)))
    req.end_headers()
    req.wfile.write(data)


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


def _open_working_camera(preferred_index: int, max_probe_index: int = 3):
    import cv2

    candidates = [int(preferred_index)] + [i for i in range(max_probe_index + 1) if i != int(preferred_index)]
    for idx in candidates:
        cap = cv2.VideoCapture(int(idx))
        opened = bool(cap.isOpened())
        ok = False
        frame = None
        if opened:
            ok, frame = cap.read()
        if opened and ok and frame is not None:
            return cap, int(idx)
        cap.release()
    return None, None


def _raw96_to_png(
    raw: bytes,
    crop_box: Optional[Tuple[int, int, int, int]],
    *,
    frame_side: int = 96,
    out_side: int = 96,
) -> bytes:
    side = int(frame_side)
    out_side = int(out_side)
    arr = np.frombuffer(raw, dtype=np.uint8).reshape((side, side))
    img = Image.fromarray(arr, mode="L")
    if crop_box is not None:
        x1, y1, x2, y2 = crop_box
        img = img.crop((x1, y1, x2, y2))
    img = img.resize((out_side, out_side))
    return _to_png_bytes(img)


def _raw96_preview_png(raw: bytes) -> bytes:
    arr = np.frombuffer(raw, dtype=np.uint8).reshape((96, 96))
    img = Image.fromarray(arr, mode="L").resize((288, 288), Image.NEAREST)
    return _to_png_bytes(img)


def _to_png_bytes(img: Image.Image) -> bytes:
    import io

    bio = io.BytesIO()
    img.save(bio, format="PNG")
    return bio.getvalue()


def _model_input_array_to_preview_image(arr: np.ndarray) -> Image.Image:
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
