"""Open a macOS camera by AVCaptureDevice uniqueID.

cv2.VideoCapture only opens cameras by positional index, and on macOS its
positional order is NOT the AVCaptureDevice array order the UI lists — OpenCV
re-sorts the device array by uniqueID, so a stored index silently points at a
DIFFERENT camera whenever virtual cameras (Iriun etc.) are installed.  This
module captures via AVCaptureSession against a specific uniqueID, so the
camera the user picked IS the camera that streams.

Exposes ``open_macos_camera_by_unique_id(unique_id)`` → ``MacCameraCapture``
or None, and ``backend_available()`` → ``(ok, reason)``.  ``MacCameraCapture``
mimics the cv2.VideoCapture surface the callers use: ``isOpened()``,
``read()`` → ``(ok, BGR ndarray)``, ``release()``.

IMPORTANT: ``import mac_camera`` must NEVER raise and must be cheap.  All
ObjC imports, libdispatch bindings and the (uuid-named) delegate class are
created lazily on the first open — the PyInstaller datas copy can sit on
sys.path without cost, hot-reloads never re-register an Objective-C class,
and processes that never touch a camera stay ObjC-free.
"""
from __future__ import annotations

import os
import sys
import threading
import time
import uuid
from typing import Optional, Tuple

# Backend is created lazily and exactly once per process (thread-safe).
_backend_lock = threading.Lock()
_backend = None
_backend_error = ""


# --- process-shutdown safety ------------------------------------------------
# A live AVCaptureSession keeps dispatching sample buffers on a GCD workloop
# thread.  If the interpreter starts finalizing while a session is still
# running, the delegate callback's PyGILState_Ensure() reaches take_gil(),
# which calls PyThread_exit_thread() → pthread_exit() on a thread owned by
# libdispatch — macOS traps this ("BUG IN CLIENT OF LIBPTHREAD: pthread_exit()
# called from a thread not created by pthread_create()") and SIGKILLs the
# process (the crash seen when quitting app.py with the camera streaming).
# The live-worker threads are daemons, so they are killed at shutdown WITHOUT
# running their cleanup — this registry + atexit hook is the only reliable
# release point.
_open_caps = {}
_caps_lock = threading.Lock()
_atexit_hooked = False


def _register_open_cap(cap: "MacCameraCapture") -> None:
    global _atexit_hooked
    with _caps_lock:
        _open_caps[id(cap)] = cap
        if not _atexit_hooked:
            _atexit_hooked = True
            try:
                import atexit

                atexit.register(_atexit_shutdown)
            except Exception:
                _atexit_hooked = False


def _unregister_open_cap(cap: "MacCameraCapture") -> None:
    with _caps_lock:
        _open_caps.pop(id(cap), None)


def shutdown_all_caps(wait_s: float = 0.15) -> int:
    """Stop every open AVCaptureSession and detach its delegate.  Returns the
    number of captures released.

    Normal shutdown paths (controller hot-reload, explicit stop) call this
    while the interpreter is fully alive; the brief wait lets a sample-buffer
    block already queued on the dispatch queue drain harmlessly.
    """
    with _caps_lock:
        caps = list(_open_caps.values())
    for cap in caps:
        try:
            cap.release()
        except Exception:
            pass
    if wait_s > 0 and caps:
        time.sleep(wait_s)
    return len(caps)


def _atexit_shutdown() -> None:
    # Runs during interpreter shutdown (before finalization, main thread).
    # Stop the sessions so no NEW frames are dispatched, then give the
    # dispatch queue a short window to drain any sample-buffer block that
    # was queued just before detach — the drain is what prevents the
    # pthread_exit() crash once finalization begins.  We must NOT os._exit
    # here: exiting hard skips every OTHER atexit handler (HTTP server
    # stop, live-worker/serial cleanup in record_controller), which is what
    # leaked resources after closing the app.  The launcher-side SIGTERM
    # handler (desktop_launcher) covers the one path that bypasses atexit
    # entirely.
    shutdown_all_caps(wait_s=0.25)


def _load_backend() -> Optional[dict]:
    """Create the ObjC backend (imports, dispatch queue, delegate class).

    Returns a dict of namespaces or None (with ``backend_available()``
    reporting the reason).  Never raises.
    """
    global _backend, _backend_error
    with _backend_lock:
        if _backend is not None or _backend_error:
            return _backend
        try:
            import ctypes

            import numpy as np
            import objc
            from AVFoundation import (
                AVCaptureDevice,
                AVCaptureDeviceInput,
                AVCaptureSession,
                AVCaptureVideoDataOutput,
                AVMediaTypeVideo,
            )
            from CoreMedia import CMSampleBufferGetImageBuffer
            from Foundation import NSObject
            from Quartz import CoreVideo as CV

            # libdispatch bindings are not always installed (pyobjc 12 split
            # them into a separate package) — call dispatch_queue_create
            # directly from libSystem.
            dispatch = ctypes.CDLL("/usr/lib/libSystem.B.dylib")
            dispatch.dispatch_queue_create.restype = ctypes.c_void_p
            dispatch.dispatch_queue_create.argtypes = [ctypes.c_char_p, ctypes.c_void_p]

            def dispatch_queue_create(label: str):
                raw = dispatch.dispatch_queue_create(label.encode("utf-8"), None)
                # Must be wrapped as an ObjC proxy — passing the bare pointer
                # to the bridge segfaults when AVFoundation dispatches the
                # first sample buffer.
                return objc.objc_object(c_void_p=raw)

            # Unique class name per process: hot-reloads / double imports can
            # then never hit "class is overriding existing Objective-C class"
            # (the crash that killed the previous bridge in the frozen app).
            delegate_class_name = f"_SampleDelegate_{uuid.uuid4().hex}"

            def delegate_initWithCallback_(self, callback):
                self = objc.super(self.__class__, self).init()
                if self is None:
                    return None
                self._callback = callback
                return self

            def delegate_captureOutput_didOutputSampleBuffer_fromConnection_(self, output, sampleBuffer, connection):
                try:
                    self._callback(sampleBuffer)
                except Exception:
                    pass

            delegate_cls = type(
                delegate_class_name,
                (NSObject,),
                {
                    "initWithCallback_": delegate_initWithCallback_,
                    "captureOutput_didOutputSampleBuffer_fromConnection_": delegate_captureOutput_didOutputSampleBuffer_fromConnection_,
                },
            )

            def pointer_to_bytes(addr, n: int):
                # pyobjc 12 returns an objc.varlist for void* returns;
                # as_buffer(n) is the fast C-level copy.  Older pyobjc
                # returned an int (ctypes) or a buffer.
                if hasattr(addr, "as_buffer"):
                    buf = addr.as_buffer(int(n))
                    return buf.tobytes() if hasattr(buf, "tobytes") else bytes(buf)
                if hasattr(addr, "tobytes"):
                    return addr.tobytes()
                if isinstance(addr, int):
                    return ctypes.string_at(addr, int(n))
                return b"".join(addr[0:int(n)])

            def sample_buffer_to_bgr(sample_buffer) -> Optional[np.ndarray]:
                image_buffer = CMSampleBufferGetImageBuffer(sample_buffer)
                if image_buffer is None:
                    return None
                CV.CVPixelBufferLockBaseAddress(image_buffer, 0)
                try:
                    addr = CV.CVPixelBufferGetBaseAddress(image_buffer)
                    w = int(CV.CVPixelBufferGetWidth(image_buffer))
                    h = int(CV.CVPixelBufferGetHeight(image_buffer))
                    stride = int(CV.CVPixelBufferGetBytesPerRow(image_buffer))
                    if not addr or w <= 0 or h <= 0 or stride < w * 4:
                        return None
                    raw = pointer_to_bytes(addr, stride * h)
                    arr = np.frombuffer(raw, dtype=np.uint8).reshape(h, stride)
                    # BGRA row, possibly padded to stride — cut to width,
                    # drop alpha.
                    return np.ascontiguousarray(arr[:, : w * 4].reshape(h, w, 4)[:, :, :3])
                finally:
                    CV.CVPixelBufferUnlockBaseAddress(image_buffer, 0)

            _backend = {
                "AVCaptureDevice": AVCaptureDevice,
                "AVCaptureDeviceInput": AVCaptureDeviceInput,
                "AVCaptureSession": AVCaptureSession,
                "AVCaptureVideoDataOutput": AVCaptureVideoDataOutput,
                "AVMediaTypeVideo": AVMediaTypeVideo,
                "CV": CV,
                "delegate_cls": delegate_cls,
                "dispatch_queue_create": dispatch_queue_create,
                "sample_buffer_to_bgr": sample_buffer_to_bgr,
            }
        except Exception as e:  # pragma: no cover - environment-dependent
            _backend_error = str(e) or "unknown import error"
            return None
        return _backend


def backend_available() -> Tuple[bool, str]:
    """(ok, reason) — whether the AVCaptureSession bridge can be used."""
    if sys.platform != "darwin":
        return False, "not on macOS"
    b = _load_backend()
    if b is not None:
        return True, "ok"
    return False, _backend_error or "backend unavailable"


class MacCameraCapture:
    """Minimal cv2.VideoCapture-compatible wrapper around AVCaptureSession."""

    def __init__(self, backend: dict, device, unique_id: str) -> None:
        self._b = backend
        self._device = device
        self.unique_id = unique_id
        self._session = None
        self._output = None
        self._delegate = None
        self._cond = threading.Condition()
        self._latest_bgr = None
        self._seq = 0
        self._closed = False

    def _start(self) -> bool:
        b = self._b
        session = b["AVCaptureSession"].alloc().init()
        try:
            device_input = b["AVCaptureDeviceInput"].deviceInputWithDevice_error_(self._device, None)
        except Exception:
            return False
        if isinstance(device_input, tuple):
            device_input = device_input[0] if device_input else None
        if device_input is None:
            return False
        session.addInput_(device_input)
        output = b["AVCaptureVideoDataOutput"].alloc().init()
        b_cv = b["CV"]
        output.setVideoSettings_({b_cv.kCVPixelBufferPixelFormatTypeKey: b_cv.kCVPixelFormatType_32BGRA})
        output.setAlwaysDiscardsLateVideoFrames_(True)
        delegate = b["delegate_cls"].alloc().initWithCallback_(self._on_frame)
        queue = b["dispatch_queue_create"]("com.aitraining.maccamera")
        output.setSampleBufferDelegate_queue_(delegate, queue)
        session.addOutput_(output)
        session.startRunning()
        self._session = session
        self._output = output
        self._delegate = delegate
        return True

    def _on_frame(self, sample_buffer) -> None:
        if self._closed:
            # Released capture: drop late frames from the dispatch queue.
            return
        bgr = self._b["sample_buffer_to_bgr"](sample_buffer)
        if bgr is None:
            return
        with self._cond:
            self._latest_bgr = bgr
            self._seq += 1
            self._cond.notify_all()

    def isOpened(self) -> bool:
        return self._session is not None and not self._closed

    def frame_seq(self) -> int:
        """Monotonic counter of frames delivered (diagnostics)."""
        return self._seq

    def read(self, timeout_s: float = 3.0):
        deadline = time.time() + max(0.1, float(timeout_s))
        with self._cond:
            while self._latest_bgr is None and time.time() < deadline and not self._closed:
                self._cond.wait(0.1)
            frame = self._latest_bgr
        if frame is None:
            return False, None
        return True, frame

    def release(self) -> None:
        self._closed = True
        _unregister_open_cap(self)
        session, output, delegate = self._session, self._output, self._delegate
        self._session = None
        self._output = None
        self._delegate = None
        try:
            if session is not None:
                session.stopRunning()
        except Exception:
            pass
        try:
            if output is not None and delegate is not None:
                output.setSampleBufferDelegate_queue_(None, None)
        except Exception:
            pass
        try:
            if delegate is not None:
                delegate._callback = None
        except Exception:
            pass


def _camera_access_status(b) -> int:
    return int(b["AVCaptureDevice"].authorizationStatusForMediaType_(b["AVMediaTypeVideo"]))


def open_macos_camera_by_unique_id(unique_id: str, timeout_s: float = 4.0) -> Optional[MacCameraCapture]:
    """Open the EXACT camera for ``unique_id`` and wait for the first frame.

    TCC pre-flight: requests permission when not_determined (bounded wait);
    denied/restricted returns None — callers surface the reason via
    ``ensure_camera_access``.  Never raises.
    """
    b = _load_backend()
    if b is None:
        return None
    try:
        status = _camera_access_status(b)
        if status in (1, 2):  # restricted / denied
            return None
        if status == 0:  # notDetermined — request, then wait (bounded)
            finished = threading.Event()
            granted_holder = {"granted": False}

            def _completion(granted: bool) -> None:
                granted_holder["granted"] = bool(granted)
                finished.set()

            try:
                b["AVCaptureDevice"].requestAccessForMediaType_completionHandler_(b["AVMediaTypeVideo"], _completion)
            except Exception:
                return None
            deadline = time.time() + max(1.0, float(timeout_s))
            while time.time() < deadline:
                if finished.wait(0.2):
                    break
            if not finished.is_set() or not granted_holder["granted"]:
                return None
        device = b["AVCaptureDevice"].deviceWithUniqueID_(str(unique_id))
    except Exception:
        return None
    if device is None:
        return None
    cap = MacCameraCapture(b, device, str(unique_id))
    try:
        if not cap._start():
            cap.release()
            return None
        ok, frame = cap.read(timeout_s=timeout_s)
        if not ok or frame is None:
            cap.release()
            return None
        _register_open_cap(cap)
        return cap
    except Exception:
        try:
            cap.release()
        except Exception:
            pass
        return None
