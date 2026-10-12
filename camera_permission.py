from __future__ import annotations

import sys
import threading
import time
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class CameraAccessResult:
    allowed: bool
    status: str
    message: str


_STATUS_MESSAGES = {
    "granted": "Camera access is ready.",
    "not_required": "No additional camera permission is required on this platform.",
    "not_determined": "Camera permission has not been granted yet. The system dialog should appear on first use.",
    "denied": "Camera access was denied. Enable it in System Settings → Privacy & Security → Camera.",
    "restricted": "Camera access is restricted by system policy (parental control or enterprise policy).",
    "unavailable": "Unable to read camera permission status. The app will try opening the camera directly.",
    "open_failed": "Camera permission was granted, but the camera cannot be opened. It may be used by another app.",
    "timeout": "Waiting for camera permission. Approve the system dialog and try again.",
}


def _result(status: str, allowed: bool | None = None, message: str | None = None) -> CameraAccessResult:
    if allowed is None:
        allowed = status in {"granted", "not_required", "unavailable"}
    return CameraAccessResult(allowed=bool(allowed), status=status, message=message or _STATUS_MESSAGES[status])


def get_camera_access_status() -> CameraAccessResult:
    if sys.platform != "darwin":
        return _result("not_required")

    try:
        from AVFoundation import AVCaptureDevice, AVMediaTypeVideo
    except Exception:
        return _result("unavailable")

    try:
        status = int(AVCaptureDevice.authorizationStatusForMediaType_(AVMediaTypeVideo))
    except Exception:
        return _result("unavailable")

    if status == 3:
        return _result("granted")
    if status == 2:
        return _result("denied")
    if status == 1:
        return _result("restricted")
    return _result("not_determined", allowed=False)


def ensure_camera_access(webcam_index: int = 0, wait_timeout_s: float = 15.0, probe_open: bool = True, unique_id: Optional[str] = None) -> CameraAccessResult:
    status = get_camera_access_status()
    if status.status in {"denied", "restricted"}:
        return status

    if status.status == "not_determined" and sys.platform == "darwin":
        try:
            from AVFoundation import AVCaptureDevice, AVMediaTypeVideo
        except Exception:
            status = _result("unavailable")
        else:
            finished = threading.Event()
            granted_holder = {"granted": False}

            def _completion(granted: bool) -> None:
                granted_holder["granted"] = bool(granted)
                finished.set()

            AVCaptureDevice.requestAccessForMediaType_completionHandler_(AVMediaTypeVideo, _completion)

            deadline = time.time() + max(1.0, float(wait_timeout_s))
            while time.time() < deadline:
                if finished.wait(0.25):
                    break

            if not finished.is_set():
                return _result("timeout", allowed=False)
            status = _result("granted" if granted_holder["granted"] else "denied", allowed=granted_holder["granted"])

    if not status.allowed:
        return status

    if not probe_open:
        return _result("granted")

    uid = str(unique_id or "").strip()
    if sys.platform == "darwin" and uid:
        # Probe the SELECTED camera by uniqueID via the AVCaptureSession
        # bridge — probing cv2 index 0 as a side effect used to open the
        # virtual camera (Iriun) instead of the user's choice.
        try:
            from mac_camera import open_macos_camera_by_unique_id
        except Exception:
            open_macos_camera_by_unique_id = None
        if open_macos_camera_by_unique_id is not None:
            cap = None
            try:
                cap = open_macos_camera_by_unique_id(uid, timeout_s=4.0)
            except Exception:
                cap = None
            if cap is not None:
                try:
                    cap.release()
                except Exception:
                    pass
                return _result("granted")
            return _result("open_failed", allowed=False)

    try:
        import cv2
    except Exception:
        return status

    cap = cv2.VideoCapture(int(webcam_index or 0))
    if not cap.isOpened():
        cap.release()
        return _result("open_failed", allowed=False)

    ok, _ = cap.read()
    cap.release()
    if not ok:
        return _result("open_failed", allowed=False)
    return _result("granted")
