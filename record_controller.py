from __future__ import annotations

import base64
import json
import math
import re
import shutil
import socket
import subprocess
import sys
import threading
import time
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
from dataset_io import (
    DEFAULT_IMG_SIZE_BY_PROJECT,
    IMAGE_EXTS,
    KNOWN_PROJECT_TYPES,
    PROJECT_FORMAT_LEGACY,
    PROJECT_TYPE_IMAGE,
    PROJECT_TYPE_OCR,
    PROJECT_TYPE_SIGN_OCR,
    app_data_dir,
    is_glyph_project,
    default_img_size,
    project_header,
    read_project_header,
    sanitize_class_name,
)
from image_preprocess import (
    CROP_MODE_AUTO_SEARCH,
    CROP_MODE_CENTER,
    PREPROCESS_MODE_AUTO_BY_LABEL,
    PREPROCESS_MODE_MANUAL_ROI,
    PREPROCESS_MODE_NONE,
    _BG_LUM_THRESH,
    _BG_DIFF_MIN,
    normalize_class_preprocess,
    normalize_class_preprocess_map,
    normalize_sample_preprocess_map,
    normalize_manual_roi,
    manual_roi_to_pixels,
    preprocess_blue_diff_array,
    prepare_inference_inputs,
)
# The OCR project's integer-exact glyph transform.  Imported here (not
# lazily) because the dispatch in _preprocess_image_png is on the hot path
# for every captured sample.
from ocr_preprocess import (
    GLYPH_SIZE,
    INK_DARK,
    NORM_CANON,
    NORM_STRETCH,
    SIGN_SEQUENCE_PAD,
    find_interior_ink_roi,
    find_sign_roi,
    preprocess_glyph,
    preprocess_sign_sequence,
    normalize_sign_sequence,
    normalize_sign_sequences,
)

# The sign-OCR preview's two answers, in one place: the SPA renders
# `match_text` verbatim, so the wording (and the one emoji per branch that
# makes it readable at a glance across a workshop) is decided here rather
# than duplicated in the frontend.
SIGN_MATCH_TEXT = "🟢 YES — MATCH"
SIGN_NOMATCH_TEXT = "🔴 NO — NOT MATCH"

# Template (EMNIST) sample seeding writes files whose names start with this,
# which is how a re-seed can replace its own previous output without ever
# touching a real camera capture in the same class folder.
TEMPLATE_SAMPLE_PREFIX = "emnist_seed_"
TEMPLATE_SAMPLE_DEFAULT_COUNT = 24

# The sign_ocr keys a preview response carries ON TOP of the ordinary
# predict payload.  Listed once so /preview/predict and
# /preview/predict_upload cannot answer the same question with different
# fields — they serve the same SPA renderer.
_SIGN_OCR_PAYLOAD_KEYS = (
    "sign_ocr",
    "sign_sequence",
    "reading_order",
    "slots",
    "slot_images_b64",
    "match",
    "match_text",
    "match_min_prob",
    "sign_box",
)


def sign_ocr_payload_extras(pred: Any) -> Dict[str, Any]:
    """The sign_ocr half of a predict result, or {} for every other project."""
    src = pred if isinstance(pred, dict) else {}
    if not src.get("sign_ocr"):
        return {}
    return {k: src.get(k) for k in _SIGN_OCR_PAYLOAD_KEYS if k in src}


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
    # Training image size (the "Image Size" hyperparameter).  The
    # PROCESSED-CACHE previews and the training tensors are stored at this
    # size, so 160×160 (or 48/128/192) projects are never downsampled to 96.
    #
    # NOT the size of the saved sample: webcam captures are written at camera
    # resolution (bounded by CAPTURE_MAX_SIDE) because the saved PNG is the
    # "Original Sample" the user draws a Manual ROI on.  Storing a 28×28 OCR
    # capture destroyed the scene before the ROI could select anything.
    img_size: int = 96
    # Which project owns this session.  Selects the sample transform used by
    # _preprocess_image_png (and therefore by the processed cache, the class
    # previews and training): "image" = the sign pipeline in image_preprocess,
    # "ocr" = the integer-exact glyph pipeline in ocr_preprocess.  It is a
    # single dispatch key, not a fork of the surrounding lifecycle.
    project_type: str = PROJECT_TYPE_IMAGE


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


def _parse_multipart_form(raw: bytes, content_type: str) -> Dict[str, Any]:
    """Minimal multipart/form-data splitter for /preview/upload.

    Only what that endpoint needs: text fields come back as strings and the
    first file part becomes `image_bytes`.  Deliberately dependency-free —
    the controller already speaks a small hand-rolled HTTP protocol, and a
    full MIME parser would be far more machinery than one image upload.
    """
    m = re.search(r'boundary="?([^";]+)"?', str(content_type or ""), re.I)
    if not m:
        return {}
    boundary = ("--" + m.group(1)).encode("latin-1")
    out: Dict[str, Any] = {}
    for part in raw.split(boundary):
        part = part.strip(b"\r\n")
        if not part or part == b"--" or b"\r\n\r\n" not in part:
            continue
        head, body = part.split(b"\r\n\r\n", 1)
        body = body.rstrip(b"\r\n")
        header = head.decode("utf-8", "replace")
        name_match = re.search(r'name="([^"]+)"', header, re.I)
        if not name_match:
            continue
        name = name_match.group(1)
        if re.search(r"filename=", header, re.I):
            out["image_bytes"] = body
            fname_match = re.search(r'filename="([^"]*)"', header, re.I)
            if fname_match and fname_match.group(1):
                out.setdefault("filename", fname_match.group(1))
        else:
            out[name] = body.decode("utf-8", "replace")
    return out


# ── TrainConfig (de)serialisation ───────────────────────────────────────
# The .tmproj train_cfg is a TrainConfig snapshot, and the set of fields it
# carries is DERIVED from the dataclass rather than listed by hand here.
# v1 hand-listed 12 of the 35 fields in `_project_train_cfg._pick_cfg` and
# silently dropped the rest on every save — all 13 ocr_* mixing / two-phase
# knobs, plus crop_mode (the model-input geometry!), img_size, seed,
# optimizer, and the batch/epoch auto-adjust flags.  A hand-written list
# drifts the moment anyone adds a field; deriving it means a new TrainConfig
# field is serialised by default and the only way to leave one out is to add
# it to `_TRAIN_CFG_UNSERIALISED` and say why.
_TRAIN_CFG_SCHEMA: Optional[Dict[str, Any]] = None

# Fields that are part of the project but deliberately NOT part of the
# train_cfg snapshot, because the archive already carries them somewhere
# better: sample_preprocess lives in tm_classes.json next to the samples it
# overrides, and class_names_order is the class LIST, which is labels.txt and
# the directory order (writing it twice is how the two copies drift apart).
# Excluded here so `missing_train_cfg_fields()` reports a field the file
# SHOULD have carried — never one this format never writes — otherwise every
# freshly saved project would look incomplete on open.  A future field that
# describes the MACHINE rather than the project also belongs in this set.
_TRAIN_CFG_UNSERIALISED: frozenset = frozenset({"sample_preprocess", "class_names_order"})

# Fields whose value is not a scalar and needs the project's own normaliser
# instead of a cast — a tuple, or a nested map the rest of the app already
# knows how to sanitise.
_TRAIN_CFG_COMPLEX_FIELDS = (
    "manual_roi",
    "class_preprocess",
    "sample_preprocess",
    "class_names_order",
    "sign_sequence",
    "sign_sequences",
)

# Enumerated string fields: a value outside the set is forwarded to code that
# switches on it and silently takes a default branch, so it is rejected here
# rather than persisted as if the project were configured that way.
_TRAIN_CFG_ENUMS = {
    "project_type": KNOWN_PROJECT_TYPES,
    # PREPROCESS_MODE_NONE is a v0 constant the settings modal still accepts;
    # it dispatches to the same branch as auto_by_label, so it stays valid
    # rather than being rewritten on save.
    "preprocess_mode": (PREPROCESS_MODE_AUTO_BY_LABEL, PREPROCESS_MODE_MANUAL_ROI, PREPROCESS_MODE_NONE),
    "crop_mode": (CROP_MODE_AUTO_SEARCH, CROP_MODE_CENTER),
    "color_mode": ("grayscale", "rgb"),
    "optimizer": ("adam", "sgd", "rmsprop"),
}


def _train_cfg_schema() -> Dict[str, Any]:
    """name -> dataclass default for TrainConfig, imported lazily and cached.

    Function-local import for the same reason every other `trainer` import in
    this module is: it pulls in TensorFlow, and this module is imported by the
    record controller long before any training happens.  Cached because the
    save and open paths each call through here several times.
    """
    global _TRAIN_CFG_SCHEMA
    if _TRAIN_CFG_SCHEMA is None:
        from dataclasses import fields as _dc_fields

        from trainer import TrainConfig

        _TRAIN_CFG_SCHEMA = {f.name: f.default for f in _dc_fields(TrainConfig) if f.name not in _TRAIN_CFG_UNSERIALISED}
    return _TRAIN_CFG_SCHEMA


def train_cfg_field_names() -> Tuple[str, ...]:
    """Every field name a saved project is expected to carry."""
    return tuple(_train_cfg_schema().keys())


def _coerce_train_cfg_value(name: str, value: Any) -> Tuple[bool, Any]:
    """Coerce one persisted value to the type its dataclass field declares.

    Returns (keep, coerced).  `keep` is False when the value cannot be
    represented as the field's type — a hand-edited JSON file, or a field
    whose meaning changed between versions.  A rejected value falls back to
    the dataclass default rather than being stored as something the trainer
    would then misuse: a string "false" is truthy, and "0.2" is not a number
    of epochs.
    """
    if value is None:
        return False, None
    if name in _TRAIN_CFG_COMPLEX_FIELDS:
        if name == "manual_roi":
            roi = normalize_manual_roi(value)
            return (True, list(roi)) if roi is not None else (False, None)
        if name == "class_names_order":
            if not isinstance(value, (list, tuple)):
                return False, None
            names = [str(v) for v in value if isinstance(v, (str, int, float))]
            return (True, names) if names else (False, None)
        if name == "sign_sequence":
            # Always exactly SIGN_SEQUENCE_LEN characters — the sequence is
            # the model's slot contract, so a truncated one from a
            # hand-edited file is padded rather than rejected, and a long
            # one is cut.  An all-blank sequence carries no information and
            # is dropped so the field falls back to the dataclass default.
            seq = normalize_sign_sequence(value)
            return (True, seq) if any(c != SIGN_SEQUENCE_PAD for c in seq) else (False, None)
        if name == "sign_sequences":
            # The whitelist: complete, deduplicated 3-character plates in
            # first-appearance order.  An empty list is dropped so the
            # primary `sign_sequence` remains the legacy fallback.
            seqs = normalize_sign_sequences(value)
            return (True, seqs) if seqs else (False, None)
        if name == "class_preprocess":
            cfg = normalize_class_preprocess_map(value)
            return (True, cfg) if cfg else (False, None)
        if name == "sample_preprocess":
            cfg = normalize_sample_preprocess_map(value)
            return (True, cfg) if cfg else (False, None)
        return False, None
    if name in _TRAIN_CFG_ENUMS:
        text = str(value).strip().lower()
        return (True, text) if text in _TRAIN_CFG_ENUMS[name] else (False, None)

    # The declared type is the dataclass default's type, so a field added
    # later is coerced correctly without touching this function.
    default = _train_cfg_schema().get(name)
    if isinstance(default, bool):
        # Before the int branch: bool subclasses int, and these fields are
        # strictly boolean on the trainer side.
        return True, _is_truthy(value)
    if isinstance(default, int):
        try:
            return True, int(value)
        except Exception:
            return False, None
    if isinstance(default, float):
        try:
            return True, float(value)
        except Exception:
            return False, None
    if isinstance(default, str):
        text = str(value)
        return (True, text) if text else (False, None)
    # No usable default and not one of the complex fields: hand the JSON back
    # unchanged so a future nested config survives instead of being dropped
    # for lack of a coercion rule.
    return True, value


def pick_train_cfg(raw: Any) -> Dict[str, Any]:
    """Project a JSON blob onto the TrainConfig schema.

    Only keys the blob actually carries come back, so an empty result still
    means "this source had nothing to say" and the callers' fallback chain
    (project state -> runs/latest/train_config.json -> train meta) keeps
    working.  Values are coerced to their declared types and validated
    against the enumerated sets; every field the schema knows about survives.
    """
    src = raw if isinstance(raw, dict) else {}
    out: Dict[str, Any] = {}
    for name in train_cfg_field_names():
        if name not in src:
            continue
        keep, value = _coerce_train_cfg_value(name, src.get(name))
        if keep:
            out[name] = value
    return out


def missing_train_cfg_fields(saved: Any) -> List[str]:
    """Schema fields a saved config does not carry — the legacy marker.

    A v1 archive carries a 12-key train_cfg; a v2 one carries the lot.  The
    load path reports the difference so a user opening an old project is told
    which settings came back at their defaults, instead of discovering it
    from a model trained with a different mix than the file implies.
    """
    src = saved if isinstance(saved, dict) else {}
    return [n for n in train_cfg_field_names() if n not in src]


_TFLITE_OP_RESOLVER_MAP = {
    "ADD": "Add",
    "AVERAGE_POOL_2D": "AveragePool2D",
    "CONCATENATION": "Concatenation",
    "CONV_2D": "Conv2D",
    "DEPTHWISE_CONV_2D": "DepthwiseConv2D",
    "DEQUANTIZE": "Dequantize",
    "FULLY_CONNECTED": "FullyConnected",
    "LEAKY_RELU": "LeakyRelu",
    "LOGISTIC": "Logistic",
    "MAXIMUM": "Maximum",
    "MAX_POOL_2D": "MaxPool2D",
    "MEAN": "Mean",
    "MINIMUM": "Minimum",
    "MUL": "Mul",
    "PACK": "Pack",
    "PAD": "Pad",
    "QUANTIZE": "Quantize",
    "REDUCE_MAX": "ReduceMax",
    "RELU": "Relu",
    "RELU6": "Relu6",
    "RESHAPE": "Reshape",
    "RESIZE_BILINEAR": "ResizeBilinear",
    "RSQRT": "Rsqrt",
    "SHAPE": "Shape",
    "SOFTMAX": "Softmax",
    "STRIDED_SLICE": "StridedSlice",
    "SUB": "Sub",
    "TANH": "Tanh",
    "TRANSPOSE": "Transpose",
}


def _generate_model_resolver_h(tflite_bytes: bytes) -> str:
    """Generate model_resolver.h matching the ops in the exported .tflite."""
    ops = []
    try:
        import tensorflow as tf
        # A default-constructed Interpreter applies the XNNPACK delegate at
        # allocate_tensors().  That does two things to the graph we are about
        # to read: it appends a node literally named `DELEGATE`, and it may
        # FUSE real ops away.  Neither is what the device's TFLM interpreter
        # runs, so the list read from it is the wrong list.  DELEGATE has no
        # Add*() mapping, so every export would print a bogus "will OMIT
        # DELEGATE" warning; worse, a fused-away op would be missing from the
        # resolver entirely and the device would fail AllocateTensors() with
        # only "AllocateTensors() failed" on the serial console.  Ask for the
        # builtin kernels only.
        try:
            interp = tf.lite.Interpreter(
                model_content=tflite_bytes,
                experimental_op_resolver_type=(
                    tf.lite.experimental.OpResolverType
                    .BUILTIN_WITHOUT_DEFAULT_DELEGATES),
            )
        except Exception:
            # Older (or future) TF without that enum: keep the plain
            # constructor, but still drop DELEGATE by name so the fallback
            # degrades to the pre-fix behaviour instead of warning forever.
            interp = tf.lite.Interpreter(model_content=tflite_bytes)
        interp.allocate_tensors()
        for detail in interp._get_ops_details():
            op = str(detail.get("op_name") or "")
            if op and op != "DELEGATE" and op not in ops:
                ops.append(op)
    except Exception:
        pass
    if not ops:
        # Conservative superset of the ops the glyph/image heads can emit.
        # The generated resolver is capacity-exact, so an unused entry costs
        # one resolver slot, never a missing kernel at AllocateTensors().
        ops = list(_TFLITE_OP_RESOLVER_MAP.keys())
    # An op with no Add*() mapping is NOT a cosmetic loss: the generated
    # resolver would be missing that kernel, MicroInterpreter::AllocateTensors()
    # would fail on the device, and the only symptom is a sketch that flashes
    # and then prints "AllocateTensors() failed".  Say it here instead.
    dropped = [op for op in ops if op not in _TFLITE_OP_RESOLVER_MAP]
    if dropped:
        print(
            "[export] WARNING: model_resolver.h will OMIT "
            + ", ".join(dropped)
            + " — no mapping in _TFLITE_OP_RESOLVER_MAP. The device will fail "
            "AllocateTensors(). Add the mapping (and make sure the TFLM build "
            "ships that kernel) before flashing.",
            file=sys.stderr,
        )
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

# Longest side a saved webcam sample keeps.  The saved PNG is the "Original
# Sample" the user draws a Manual ROI on, so downsampling it to the model
# input size (28x28 for OCR) at capture time destroyed the very detail the
# editor exists to select.  Downsampling belongs in the preprocessing
# pipeline, AFTER the crop — _cache_one_sample / _preprocess_image_png
# already do exactly that.  Serial captures are unaffected: the device only
# ever sends 96x96, so there is nothing to preserve.
CAPTURE_MAX_SIDE = 1024


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
        # The most recent uploaded Preview image per session.  /preview/upload
        # keeps it so a Dark/Lum slider move or an ROI/Orig tab switch can
        # re-run inference on the SAME picture without re-uploading it; the
        # SPA sends reuse=1 for those re-evaluations.
        self._preview_uploads: Dict[str, Dict[str, Any]] = {}
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
        project_type: Optional[str] = None,
        fps: Optional[float] = None,
    ) -> SessionConfig:
        with self._lock:
            cfg = self._configs.get(session_id)
            if cfg is None:
                raise RuntimeError("missing config")
            previous_type = str(getattr(cfg, "project_type", "") or "").strip().lower()
            updated = replace(
                cfg,
                project_type=str(cfg.project_type if project_type is None else project_type),
                fps=float(cfg.fps if fps is None else fps),
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
        # Switching INTO a Car License project locks the class list to its
        # target.  Done outside the lock because enforcement reads the
        # config and writes tm_classes.json.  A project OPEN also lands here
        # before the archive is copied, which is harmless: the reset that
        # follows wipes the workspace anyway, and the real enforcement runs
        # again after the archive's sequence is known.
        if (
            str(getattr(updated, "project_type", "") or "").strip().lower()
            == PROJECT_TYPE_SIGN_OCR
            and previous_type != PROJECT_TYPE_SIGN_OCR
        ):
            try:
                saved = self._project_train_cfg(updated.dataset_root)
                seq = saved.get("sign_sequences") or saved.get("sign_sequence")
                if seq:
                    self._enforce_sign_classes(session_id, seq)
            except Exception:
                pass
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
        cfg = self.get_config(session_id) if session_id else None
        frame = _center_crop_webcam_square(
            frame, getattr(cfg, "project_type", "") if cfg is not None else ""
        )
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
                seen = set()
                for p in list_serial_ports():
                    dev = str(p.device or "")
                    # Skip entries we have already emitted — some adapters
                    # enumerate the same COM port under multiple interface
                    # descriptions, which would otherwise produce duplicate
                    # dropdown entries.
                    if dev in seen:
                        continue
                    seen.add(dev)
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
            self._cache_one_sample(cfg.dataset_root, class_name, p, img_size=int(cfg.img_size), project_type=str(cfg.project_type))
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
            project_type_raw = (qs.get("project_type") or [None])[0]
            fps_raw = (qs.get("fps") or [None])[0]
            auto_seed_raw = (qs.get("auto_seed_emnist") or [None])[0]
            include_state_raw = (qs.get("include_state") or [None])[0]
            # The SPA's target sign travels with the auto-fill request so
            # this endpoint can both persist it and hand it to the seeder,
            # which uses it to prepare a synthetic Preview image.
            sign_seq_raw = (qs.get("sign_sequence") or qs.get("sign_seq") or [None])[0]
            reading_order_raw = (qs.get("reading_order") or qs.get("reading") or [None])[0]
            try:
                if serial_sync_raw not in (None, ""):
                    parse_sync_header(serial_sync_raw)
                if fps_raw not in (None, ""):
                    fps_val = float(fps_raw)
                    if fps_val < 1.0 or fps_val > 60.0:
                        raise ValueError("fps must be between 1 and 60")
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
                    project_type=None if project_type_raw in (None, "") else str(project_type_raw).strip().lower(),
                    fps=None if fps_raw in (None, "") else float(fps_raw),
                )
            except Exception as e:
                _send_json(req, {"ok": "0", "error": str(e)}, status=400, cors=True)
                return
            with self._lock:
                wc_state = self._live.get(self._live_key(session_id, "webcam"))
            wc_actual = dict((wc_state or {}).get("actual") or {})
            # The Settings checkbox speaks through this endpoint when it
            # flips: persist the flag, then backfill any class that is
            # already on disk with zero samples.  `include_state=1` is the
            # read-only variant used when the SPA wants the full state back
            # after a live-setting change.
            response: Dict[str, Any] = {
                "ok": "1",
                "direct_tensor_stream": "1" if bool(getattr(cfg, "direct_tensor_stream", False)) else "0",
                "webcam_index": str(int(cfg.webcam_index)),
                "webcam_id": str(cfg.webcam_id or ""),
                "webcam_actual_backend": str(wc_actual.get("backend") or ""),
                "webcam_actual_uid": str(wc_actual.get("unique_id") or ""),
                "img_size": str(int(cfg.img_size)),
                "project_type": str(getattr(cfg, "project_type", PROJECT_TYPE_IMAGE) or PROJECT_TYPE_IMAGE),
                "serial_port": str(cfg.serial_port),
                "serial_baud": str(int(cfg.serial_baud)),
                "serial_sync": str(cfg.serial_sync),
                "serial_frame_side": str(int(cfg.serial_frame_side)),
                "serial_channels": str(int(cfg.serial_channels)),
                # Read-back for the Settings FPS field.  fps PACES the
                # record thread, so a page that showed its local default
                # (8.0) while the session actually ran at another rate was
                # reporting a number the capture did not obey.
                "fps": str(float(cfg.fps)),
            }
            if sign_seq_raw not in (None, "") or reading_order_raw not in (None, ""):
                try:
                    patch: Dict[str, Any] = {}
                    if sign_seq_raw not in (None, ""):
                        patch["sign_sequence"] = normalize_sign_sequence(sign_seq_raw)
                    if reading_order_raw not in (None, ""):
                        order = str(reading_order_raw).strip().lower()
                        if order in ("ltr", "rtl"):
                            patch["reading_order"] = order
                    if patch:
                        self._save_train_cfg(session_id, patch)
                except Exception:
                    pass
            if auto_seed_raw is not None:
                try:
                    enabled = _is_truthy(auto_seed_raw)
                    self._save_train_cfg(
                        session_id,
                        {"auto_seed_emnist_samples": enabled},
                    )
                    if enabled:
                        self._seed_empty_char_classes(
                            session_id,
                            sign_sequence=sign_seq_raw,
                            reading_order=reading_order_raw,
                        )
                except Exception:
                    pass
            if auto_seed_raw is not None or _is_truthy(include_state_raw):
                try:
                    response.update(self._runtime_state_payload(session_id))
                except Exception:
                    pass
            _send_json(req, response, cors=True)
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
            sign_seq_raw = (qs.get("sign_seq") or [""])[0]
            sign_seqs_raw = (qs.get("sign_seqs") or qs.get("sign_sequences") or [""])[0]
            sign_seqs_list = (
                [part.strip() for part in str(sign_seqs_raw).split(",") if part.strip()]
                if str(sign_seqs_raw).strip() else None
            )
            reading_raw = (qs.get("reading") or [""])[0]
            match_min_raw = (qs.get("match_min") or [""])[0]
            bg_dark_thresh = int(bg_dark_raw) if str(bg_dark_raw).strip() != "" else None
            bg_lum_thresh  = int(bg_lum_raw)  if str(bg_lum_raw).strip()  != "" else None
            sign_match_min = float(match_min_raw) if str(match_min_raw).strip() != "" else None
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
                # The DEVICE preview is already box-averaged to the
                # preprocessing side (160 -> 96), exactly like the MCU, and
                # that is what inference must see — the native capture PNG
                # (160) is kept for SAVING samples, not for prediction.  A
                # webcam has no device preprocessing side, so its native
                # capture is the right prediction source.
                if source == "device":
                    predict_png = preview_png
                else:
                    capture_png = self._get_live_capture(session_id=session_id, source=source)
                    predict_png = capture_png if capture_png else preview_png
                if self._is_sign_ocr_session(session_id):
                    pred = self._preview_predict_sign_ocr(
                        session_id=session_id,
                        png=predict_png,
                        sign_sequence=sign_seq_raw,
                        sign_sequences=sign_seqs_list,
                        reading_order=reading_raw,
                        bg_dark_thresh=bg_dark_thresh,
                        bg_lum_thresh=bg_lum_thresh,
                        match_min_prob=sign_match_min,
                        ood_sign_pct_min=ood_sign_pct_min,
                        ood_sign_pct_max=ood_sign_pct_max,
                        ood_max_prob_min=ood_max_prob_min,
                        ood_entropy_ratio_max=ood_entropy_ratio_max,
                        source=source,
                    )
                else:
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
            payload = {
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
            }
            if pred.get("sign_ocr"):
                payload.update(sign_ocr_payload_extras(pred))
            _send_json(req, payload, cors=True)
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
            self._cache_one_sample(cfg.dataset_root, class_name, p, img_size=int(cfg.img_size), project_type=str(cfg.project_type))
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
            "/dataset/export",
            "/project/save",
            "/project/open",
            "/project/reset",
            "/classes/rename",
            "/classes/add",
            "/classes/add_named",
            "/classes/sync_ocr",
            "/classes/delete",
            "/samples/delete",
            "/samples/clear",
            "/samples/seed_emnist",
            "/classes/seed_emnist",
            "/preprocess/preview",
            "/preview/predict_upload",
            "/preview/upload",
            "/preview/synthesize_sign",
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
        content_type = str(req.headers.get("Content-Type", "") or "").lower()
        if path == "/preview/upload" and "application/json" not in content_type:
            # A real file upload: the body IS the PNG/JPG bytes (either raw
            # with an image/* Content-Type or multipart/form-data).  Session
            # and options may ride on the query string so a bare `curl
            # --data-binary @sign.png` works without a JSON envelope.
            qs_params = parse_qs(parsed.query)
            payload: Dict[str, Any] = {
                "session": (qs_params.get("session") or [""])[0],
                "image_bytes": raw,
            }
            if "multipart/form-data" in content_type:
                payload.update(_parse_multipart_form(raw, content_type))
            for key in (
                "reuse",
                "filename",
                "preprocess",
                "dark",
                "bg_dark",
                "lum",
                "bg_lum",
                "sign_sequence",
                "reading_order",
                "match_min",
            ):
                if key in qs_params:
                    payload[key] = (qs_params[key] or [""])[0]
        else:
            payload = json.loads(raw.decode("utf-8"))
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
                self._cache_one_sample(cfg.dataset_root, class_name, p, img_size=int(cfg.img_size), project_type=str(cfg.project_type))
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
                # The auto-fill toggle lives in TrainConfig, so saving a
                # config that turns it ON is exactly the moment to backfill
                # classes that already exist and are still empty — no page
                # reload, no waiting for the next class edit.
                try:
                    if isinstance(cfg, dict) and _is_truthy(cfg.get("auto_seed_emnist_samples")):
                        self._seed_empty_char_classes(
                            session_id,
                            sign_sequence=cfg.get("sign_sequence"),
                            reading_order=cfg.get("reading_order"),
                        )
                except Exception:
                    pass
            except Exception as e:
                _send_json(req, {"ok": "0", "error": str(e)}, status=400, cors=True)
                return
            response = {"ok": "1"}
            try:
                response.update(self._runtime_state_payload(session_id))
            except Exception:
                pass
            _send_json(req, response, cors=True)
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
            sample_preprocess = payload.get("sample_preprocess")
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
                # BOTH maps, and each one only when the caller actually sent
                # it: _classes_save treats None as "keep what is on disk", so a
                # save that carries only one of them cannot blank the other.
                # sample_preprocess had no writer at all before this — the Edit
                # modal's per-sample ROI reached the disk only if the user
                # happened to save the whole .tmproj, so a drag + "Save Sample"
                # silently died on reload.
                self._classes_save(
                    sess_cfg.dataset_root,
                    classes,
                    class_preprocess=normalize_class_preprocess_map(class_preprocess) if class_preprocess else None,
                    # `is not None`, not truthiness: an EMPTY map is what the
                    # client sends after removing the last sample override, and
                    # that has to clear the stored map rather than be read as
                    # "no opinion, keep what is on disk".
                    sample_preprocess=normalize_sample_preprocess_map(sample_preprocess) if sample_preprocess is not None else None,
                )
                # Thresholds / ROI config changed — the processed cache must
                # follow so previews and the next training use the new
                # settings immediately.
                try:
                    self._rebuild_processed_cache(
                        sess_cfg.dataset_root,
                        class_preprocess=normalize_class_preprocess_map(class_preprocess) if class_preprocess else None,
                        sample_preprocess=normalize_sample_preprocess_map(sample_preprocess) if sample_preprocess is not None else None,
                        fast_mode=False,
                        out_size=int(sess_cfg.img_size),
                        project_type=str(sess_cfg.project_type),
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
                if self._is_sign_ocr_session(session_id):
                    pred = self._preview_predict_sign_ocr(
                        session_id=session_id,
                        png=png,
                        sign_sequence=payload.get("sign_sequence"),
                        sign_sequences=payload.get("sign_sequences"),
                        reading_order=payload.get("reading_order"),
                        bg_dark_thresh=bg_dark_thresh,
                        bg_lum_thresh=bg_lum_thresh,
                        match_min_prob=(
                            float(payload.get("match_min"))
                            if payload.get("match_min") is not None and str(payload.get("match_min")).strip() != ""
                            else None
                        ),
                    )
                else:
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
        if path == "/preview/upload":
            # Accepts the image either as JSON (image_b64) or as the raw
            # request body (see the parser at the top of _handle_post), keeps
            # the most recent one per session, and answers with everything
            # the Preview card draws: the ORIGINAL picture, the processed
            # ROI, the per-slot glyphs and the sign verdict (sign projects)
            # or class ranking (image/OCR).  `reuse=1` re-runs inference on
            # the stored image when the Dark/Lum sliders or the ROI/Orig
            # tabs change — no second upload.
            session_id = str(payload.get("session") or "").strip()
            if not session_id:
                _send_json(req, {"ok": "0", "error": "missing session"}, status=400, cors=True)
                return
            image_bytes = payload.get("image_bytes")
            image_b64_in = str(payload.get("image_b64") or "").strip()
            reuse = _is_truthy(payload.get("reuse"))
            filename = str(payload.get("filename") or "").strip()
            if not image_bytes and image_b64_in:
                try:
                    image_bytes = base64.b64decode(image_b64_in, validate=True)
                except Exception as e:
                    _send_json(req, {"ok": "0", "error": f"invalid image data: {e}"}, status=400, cors=True)
                    return
            if not image_bytes and reuse:
                with self._lock:
                    stored = dict(self._preview_uploads.get(session_id) or {})
                image_bytes = stored.get("bytes")
                filename = filename or str(stored.get("filename") or "")
            if not image_bytes:
                _send_json(
                    req,
                    {"ok": "0", "error": "missing image (send image_b64 or raw image bytes)"},
                    status=400,
                    cors=True,
                )
                return
            # Re-encode to a clean PNG BEFORE storing: EXIF-rotated phone
            # photos and palette/CMYK JPEGs would otherwise make the same
            # upload preprocess differently from the preview shown.
            try:
                src_img = Image.open(_bytes_io(image_bytes))
                try:
                    src_img = ImageOps.exif_transpose(src_img)
                    if src_img.mode not in {"L", "RGB"}:
                        src_img = src_img.convert("RGB")
                    orig_png = _to_png_bytes(src_img)
                finally:
                    try:
                        src_img.close()
                    except Exception:
                        pass
            except Exception as e:
                _send_json(req, {"ok": "0", "error": f"unsupported image: {e}"}, status=400, cors=True)
                return
            with self._lock:
                self._preview_uploads[session_id] = {
                    "bytes": orig_png,
                    "filename": filename,
                    # A real file the user chose; the sign auto-preview must
                    # never replace it on the next auto-fill / training run.
                    "synthetic": False,
                }

            def _opt_int(*keys: str) -> Optional[int]:
                for key in keys:
                    raw = payload.get(key)
                    if raw is not None and str(raw).strip() != "":
                        try:
                            return int(float(raw))
                        except (TypeError, ValueError):
                            return None
                return None

            def _opt_float(*keys: str) -> Optional[float]:
                for key in keys:
                    raw = payload.get(key)
                    if raw is not None and str(raw).strip() != "":
                        try:
                            return float(raw)
                        except (TypeError, ValueError):
                            return None
                return None

            bg_dark_thresh = _opt_int("bg_dark", "dark")
            bg_lum_thresh = _opt_int("bg_lum", "lum")
            try:
                if self._is_sign_ocr_session(session_id):
                    pred = self._preview_predict_sign_ocr(
                        session_id=session_id,
                        png=orig_png,
                        sign_sequence=payload.get("sign_sequence"),
                        sign_sequences=payload.get("sign_sequences"),
                        reading_order=payload.get("reading_order"),
                        bg_dark_thresh=bg_dark_thresh,
                        bg_lum_thresh=bg_lum_thresh,
                        match_min_prob=_opt_float("match_min", "match_min_prob"),
                    )
                else:
                    pred = self._preview_predict(
                        session_id=session_id,
                        png=orig_png,
                        preprocess_mode=str(payload.get("preprocess") or "").strip(),
                        bg_dark_thresh=bg_dark_thresh,
                        bg_lum_thresh=bg_lum_thresh,
                        ood_sign_pct_min=_opt_float("ood_sign_min"),
                        ood_sign_pct_max=_opt_float("ood_sign_max"),
                        ood_max_prob_min=_opt_float("ood_max_prob"),
                        ood_entropy_ratio_max=_opt_float("ood_entropy_max"),
                    )
            except Exception as e:
                _send_json(req, {"ok": "0", "error": str(e)}, status=400, cors=True)
                return
            orig_b64 = base64.b64encode(orig_png).decode("ascii")
            _send_json(
                req,
                {
                    "ok": "1",
                    **pred,
                    # `orig` is the untouched upload (EXIF-corrected); `roi`
                    # is the exact model-input rendering.  Kept as separate
                    # keys because the SPA's ROI/Orig toggles use them by
                    # name; image_b64 stays for older callers.
                    "image_b64": orig_b64,
                    "orig_b64": orig_b64,
                    "roi_b64": str(pred.get("processed_image_b64") or ""),
                    "filename": filename or "upload",
                },
                cors=True,
            )
            return
        if path == "/preview/synthesize_sign":
            # The zero-files version of /preview/upload: draw a whole 3-char
            # sign from the project's own samples (EMNIST fallback), store
            # it as the active Preview frame, and immediately run the same
            # sign split + three invokes the upload path runs.  One response
            # shape, so the SPA card does not care which one produced it.
            session_id = str(payload.get("session") or "").strip()
            if not session_id:
                _send_json(req, {"ok": "0", "error": "missing session"}, status=400, cors=True)
                return
            if not self._is_sign_ocr_session(session_id):
                _send_json(
                    req,
                    {"ok": "0", "error": "sign synthesis is only available in a sign_ocr project"},
                    status=400,
                    cors=True,
                )
                return

            def _synth_int(*keys: str) -> Optional[int]:
                for key in keys:
                    raw = payload.get(key)
                    if raw is not None and str(raw).strip() != "":
                        try:
                            return int(float(raw))
                        except (TypeError, ValueError):
                            return None
                return None

            def _synth_float(*keys: str) -> Optional[float]:
                for key in keys:
                    raw = payload.get(key)
                    if raw is not None and str(raw).strip() != "":
                        try:
                            return float(raw)
                        except (TypeError, ValueError):
                            return None
                return None

            try:
                variant = str(payload.get("variant") or "match").strip().lower()
                candidates: List[Any] = [None]
                if variant == "mismatch":
                    # The model only has the project's own classes to answer
                    # with, so an unseen replacement glyph can be MISREAD as
                    # the target character — which would paint a false green
                    # YES on a plate that is genuinely a different sign.
                    # Retry replacement characters until the reading is
                    # actually wrong.
                    seq_for_alt, _order, _seqs, _idx = self._sign_preview_sequence(
                        session_id,
                        chars=payload.get("chars"),
                        sign_sequence=payload.get("sign_sequence") or payload.get("sign_seq"),
                        reading_order=payload.get("reading_order") or payload.get("reading"),
                        sign_sequences=payload.get("sign_sequences"),
                        plate_index=payload.get("plate_index"),
                    )
                    candidates = [
                        c for c in "56789ABCDEFGHIJKLMNOPQRSTUVWXYZ" if c not in seq_for_alt
                    ][:10] or [None]
                else:
                    # Even a MATCH plate can draw an unlucky glyph variant
                    # (a thin EMNIST '1', a slanted 'A') that the current
                    # model reads just under the confidence floor, and a
                    # REVERSED plate can be read as the target by chance.
                    # Redraw — fresh glyph pick, size and jitter — until the
                    # plate shows the verdict the button promises.  Every
                    # variant has an intended answer, and the whole point of
                    # the button is to demonstrate it.
                    candidates = [None] * 8
                synth = None
                pred = None
                for alt_char in candidates:
                    synth = self._synthesize_sign_preview_png(
                        session_id=session_id,
                        variant=variant,
                        chars=payload.get("chars"),
                        sign_sequence=payload.get("sign_sequence") or payload.get("sign_seq"),
                        reading_order=payload.get("reading_order") or payload.get("reading"),
                        alt_char=alt_char,
                        sign_sequences=payload.get("sign_sequences"),
                        plate_index=payload.get("plate_index"),
                    )
                    pred = self._preview_predict_sign_ocr(
                        session_id=session_id,
                        png=synth["png"],
                        sign_sequence=synth["sign_sequence"],
                        sign_sequences=synth.get("sign_sequences"),
                        reading_order=synth["reading_order"],
                        bg_dark_thresh=_synth_int("bg_dark", "dark"),
                        bg_lum_thresh=_synth_int("bg_lum", "lum"),
                        match_min_prob=_synth_float("match_min", "match_min_prob"),
                    )
                    if variant == "match":
                        intended = True
                    elif variant == "reversed":
                        # Reversed pixels read right-to-left are the target.
                        intended = synth["reading_order"] == "rtl"
                    else:
                        intended = False
                    # The LAST attempted plate is the one the synthesizer
                    # stored, so the response, the buffer and the drawn slots
                    # always describe the same frame.
                    if bool(pred.get("match")) == intended:
                        break
                if synth is None or pred is None:
                    raise ValueError("could not draw a sign preview")
                png = synth["png"]
            except Exception as e:
                _send_json(req, {"ok": "0", "error": str(e)}, status=400, cors=True)
                return
            orig_b64 = base64.b64encode(png).decode("ascii")
            _send_json(
                req,
                {
                    "ok": "1",
                    **pred,
                    "image_b64": orig_b64,
                    "orig_b64": orig_b64,
                    "roi_b64": str(pred.get("processed_image_b64") or ""),
                    "filename": synth["filename"],
                    # Provenance for the card's note and for a test that
                    # wants to assert WHICH plate was drawn.
                    "synth_variant": synth["variant"],
                    "synth_chars": synth["chars"],
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
            # Preview-time OOD + preprocessing thresholds, forwarded from the
            # SPA so the device export stamps them into model_settings.h.
            # Defaults mirror the predict endpoint so an export never ships a
            # None / unset value.
            def _num(key, default, cast=float):
                raw = payload.get(key)
                if raw is None or str(raw).strip() == "":
                    return default
                try:
                    return cast(raw)
                except (TypeError, ValueError):
                    return default
            runtime_config = {
                "ood_sign_min": _num("ood_sign_min", 0.3),
                "ood_sign_max": _num("ood_sign_max", 70.0),
                "ood_max_prob": _num("ood_max_prob", 0.60),
                "ood_entropy_max": _num("ood_entropy_max", 0.70),
                "bg_dark_thresh": _num("bg_dark_thresh", 0, cast=int),
                "bg_lum_thresh": _num("bg_lum_thresh", 100, cast=int),
                # sign_ocr only.  String/None pass through; _export_run
                # normalises them and falls back to the TrainConfig when the
                # SPA did not send them (older SPA builds).
                "sign_sequence": payload.get("sign_sequence"),
                "sign_sequences": payload.get("sign_sequences"),
                "capture_frame_side": payload.get("capture_frame_side"),
                "preprocess_frame_side": payload.get("preprocess_frame_side"),
                "reading_order": payload.get("reading_order"),
                "match_min_prob": _num("match_min_prob", 0.60),
            }
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
                    runtime_config=runtime_config,
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
                self._seed_empty_char_classes(session_id)
            except Exception as e:
                _send_json(req, {"ok": "0", "error": str(e)}, status=400, cors=True)
                return
            response = {"ok": "1", "classes": classes}
            try:
                response.update(self._runtime_state_payload(session_id))
            except Exception:
                pass
            _send_json(req, response, cors=True)
            return
        if path == "/classes/add_named":
            # Same response shape as /classes/add so the two are interchangeable
            # on the client: both hand back the whole ordered class list, which
            # is what applyClassesState() wants.
            session_id = str(payload.get("session") or "").strip()
            names = payload.get("names")
            if not session_id:
                _send_json(req, {"ok": "0", "error": "missing session"}, status=400, cors=True)
                return
            if not isinstance(names, list) or not names:
                _send_json(req, {"ok": "0", "error": "missing names"}, status=400, cors=True)
                return
            try:
                classes = self._classes_add_named(
                    session_id=session_id,
                    names=names,
                    prune_empty_placeholders=bool(payload.get("prune_placeholders")),
                )
                self._seed_empty_char_classes(session_id)
            except Exception as e:
                _send_json(req, {"ok": "0", "error": str(e)}, status=400, cors=True)
                return
            response = {"ok": "1", "classes": classes}
            try:
                response.update(self._runtime_state_payload(session_id))
            except Exception:
                pass
            _send_json(req, response, cors=True)
            return
        if path == "/classes/sync_ocr":
            # The road-sign UI has no class editor: its three inline slots ARE
            # the class list.  This endpoint is the one write that makes the
            # stored list (and, with it, the model's label order) agree with
            # them after every edit.  Response shape matches the other class
            # routes so the client can hand it straight to applyClassesState().
            session_id = str(payload.get("session") or "").strip()
            names = payload.get("names")
            seq_raw = payload.get("sign_sequence")
            seqs_raw = payload.get("sign_sequences")
            order_raw = payload.get("reading_order")
            auto_seed_raw = payload.get("auto_seed_emnist_samples")
            if auto_seed_raw is None:
                auto_seed_raw = payload.get("auto_seed_emnist")
            if not session_id:
                _send_json(req, {"ok": "0", "error": "missing session"}, status=400, cors=True)
                return
            # The sequence, when the caller sends it, IS the class list.  The
            # SPA derives `names` from the same slots and keeps sending it for
            # older servers, but taking the sequence here means a client can
            # never sync a class list that disagrees with the target it just
            # committed to the train config.
            sequences = normalize_sign_sequences(seqs_raw)
            if not sequences and seq_raw not in (None, ""):
                sequences = normalize_sign_sequences(seq_raw)
            sequence: Optional[List[str]] = None
            if sequences:
                sequence = normalize_sign_sequence(sequences[0])
                derived: List[str] = []
                for text in sequences:
                    for ch in text:
                        if ch not in derived:
                            derived.append(ch)
                if derived:
                    names = derived
            if not isinstance(names, list) or not names:
                _send_json(req, {"ok": "0", "error": "missing names"}, status=400, cors=True)
                return
            try:
                with self._lock:
                    sess_cfg = self._configs.get(session_id)
                if sess_cfg is None:
                    raise RuntimeError("missing config")
                # Persist the target BEFORE the class sync / backfill: the
                # synthetic preview drawn by the seeder, a later reopen and
                # the export all read it from the project's train config.
                patch: Dict[str, Any] = {}
                if sequences:
                    patch["sign_sequences"] = list(sequences)
                    patch["sign_sequence"] = normalize_sign_sequence(sequences[0])
                if order_raw not in (None, ""):
                    order = str(order_raw).strip().lower()
                    if order in ("ltr", "rtl"):
                        patch["reading_order"] = order
                if patch:
                    self._save_train_cfg(session_id, patch)

                classes = self._classes_sync_ocr(session_id=session_id, names=names)
                # A slot character that just became a class starts empty.
                # Backfill when EITHER the caller's current checkbox state or
                # the project's persisted TrainConfig says auto-fill is on.
                # The two can disagree — the checkbox is browser-local and
                # survives a project switch, while the persisted flag follows
                # the project — and OR-ing them is what makes "on" mean on
                # regardless of which side last wrote it.
                persisted_seed = self._auto_seed_emnist_enabled(sess_cfg.dataset_root)
                incoming_seed = auto_seed_raw is not None and _is_truthy(auto_seed_raw)
                if incoming_seed and not persisted_seed:
                    # Heal the divergence: the next class-write endpoint (and
                    # a later training run) must see what this request said.
                    self._save_train_cfg(session_id, {"auto_seed_emnist_samples": True})
                    persisted_seed = True
                seeded: List[str] = []
                seed_error = ""
                if incoming_seed or persisted_seed:
                    try:
                        seeded = self._seed_empty_char_classes(
                            session_id,
                            sign_sequence=sequence if sequence is not None else seq_raw,
                            reading_order=order_raw,
                            strict=True,
                        )
                    except Exception as exc:
                        # The classes were synced and persisted; report the
                        # seeding failure without turning the whole edit into
                        # a 400 (the SPA can retry with the error visible).
                        seed_error = str(exc)
            except Exception as e:
                _send_json(req, {"ok": "0", "error": str(e)}, status=400, cors=True)
                return
            response = {
                "ok": "1",
                "classes": classes,
                "seeded": seeded,
                "seed_error": seed_error,
            }
            if sequence is not None:
                response["sign_sequence"] = sequence
            if sequences:
                response["sign_sequences"] = list(sequences)
            try:
                # Runtime state is rebuilt AFTER seeding, so counts,
                # sample_previews and train_enabled describe the files that
                # are actually on disk when this response lands.  It also
                # overrides `classes` with the persisted order.
                response.update(self._runtime_state_payload(session_id))
            except Exception:
                pass
            _send_json(req, response, cors=True)
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
        if path in ("/samples/seed_emnist", "/classes/seed_emnist"):
            session_id = str(payload.get("session") or "").strip()
            names = payload.get("classes")
            only_empty = any(
                _is_truthy(payload.get(key))
                for key in ("only_empty", "fill_empty", "all_empty")
            )
            if not session_id:
                _send_json(req, {"ok": "0", "error": "missing session"}, status=400, cors=True)
                return
            try:
                result = self._samples_seed_emnist(
                    session_id=session_id,
                    classes=names if isinstance(names, list) and names else None,
                    count=payload.get("count", TEMPLATE_SAMPLE_DEFAULT_COUNT),
                    only_empty=only_empty,
                    sign_sequence=payload.get("sign_sequence") or payload.get("sign_seq"),
                    reading_order=payload.get("reading_order") or payload.get("reading"),
                )
            except Exception as e:
                _send_json(req, {"ok": "0", "error": str(e)}, status=400, cors=True)
                return
            _send_json(req, {"ok": "1", **result}, cors=True)
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
        self._cache_one_sample(cfg.dataset_root, class_name, p, img_size=int(cfg.img_size), project_type=str(cfg.project_type))
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

    # The names _next_class_name / _tm_apply_project_defaults hand out.  A
    # class still wearing one has never been named by anyone, which is what
    # makes it safe to clean up without asking.
    _PLACEHOLDER_CLASS_RE = re.compile(r"^Class [0-9]+$")

    def _is_empty_placeholder(self, dataset_root: Path, name: str) -> bool:
        """True for an untouched auto-named class: "Class N" with zero samples.

        The sample test uses _list_class_image_files — the same helper the
        workspace payload counts with — so the rule matches the number the user
        sees on the card.  Anything unreadable is KEPT: this decides whether to
        delete someone's class, so it fails closed.
        """
        if not self._PLACEHOLDER_CLASS_RE.match(str(name).strip()):
            return False
        class_dir = dataset_root / sanitize_class_name(name)
        if not class_dir.exists():
            return True  # never captured into — the directory is created lazily
        try:
            return len(_list_class_image_files(class_dir)) == 0
        except Exception:
            return False

    def _classes_add_named(
        self,
        session_id: str,
        names: List[Any],
        prune_empty_placeholders: bool = False,
    ) -> List[str]:
        """Add classes under caller-chosen names — the OCR character selector.

        _classes_add invents "Class N"; this one takes the names the user
        picked.  Nothing else differs, and deliberately so: a class in this
        app is a string in an ordered list plus whatever class_preprocess /
        sample_preprocess entries hang off it, and all of that materialises on
        demand (directories on first capture, preprocess defaults out of
        normalize_class_preprocess).  So appending the name IS creating the
        class, and capture, preview, training and export already treat it as
        first-class without being told it came from a selector.

        Names that are already classes are skipped rather than rejected, so
        the selector can resend its whole selection on every click and stay
        idempotent.  Names that cannot be classes raise — those are bad input,
        not races, and failing loudly beats writing a half-built list.

        prune_empty_placeholders drops untouched "Class N" placeholders in the
        SAME write, so the selector's first click leaves the user with the
        characters they asked for and nothing else.  Only classes with zero
        samples go, so a placeholder someone has captured into is never touched,
        and nothing being added in this request can be a candidate — `added` is
        disjoint from `classes` by the dedupe above.

        Only the app's own naming convention is subject to this.  A user who
        deliberately types "Class 1" as a character in an OCR project is
        wearing the app's namespace, and an empty class by that name will be
        cleaned up on the next pick — which is the intended trade.
        """
        with self._lock:
            cfg = self._configs.get(session_id)
        if cfg is None:
            raise RuntimeError("missing config")
        classes = self._classes_load(cfg.dataset_root)
        class_preprocess = self._class_preprocess_load(cfg.dataset_root)
        sample_preprocess = self._sample_preprocess_load(cfg.dataset_root)

        # Reserved: the OCR trainer appends this one itself as the reject
        # class (ocr_dataset.BACKGROUND_CLASS_NAME).  A user class by that
        # name would give the model two heads wearing one label.  Imported
        # here rather than at module scope to keep this module independent of
        # the OCR stack.
        from ocr_dataset import BACKGROUND_CLASS_NAME

        # Collisions are judged on the sanitised, lowercased name, because
        # that is what actually collides: sanitize_class_name is what turns a
        # class into a directory, so two classes that sanitise alike share
        # their samples.  Case-insensitivity also stops "a" and "A" becoming
        # two model outputs nobody can tell apart on the device.
        taken = {sanitize_class_name(str(c)).lower() for c in classes}
        reserved = {
            BACKGROUND_CLASS_NAME.lower(),
            sanitize_class_name(BACKGROUND_CLASS_NAME).lower(),
        }

        added: List[str] = []
        for raw in names:
            name = str(raw).strip()
            if not name:
                continue
            if len(name) > 64:
                raise ValueError(
                    f"Class name is too long (max 64 characters): {name[:32]}..."
                )
            safe = sanitize_class_name(name)
            if safe.lower() in reserved:
                raise ValueError(f'"{name}" is reserved for the reject class.')
            if safe.lower() in taken:
                continue
            taken.add(safe.lower())
            added.append(name)

        if not added:
            return list(classes)  # nothing to do; not an error

        if prune_empty_placeholders:
            survivors = [c for c in classes if not self._is_empty_placeholder(cfg.dataset_root, c)]
            dropped = [c for c in classes if c not in survivors]
            if dropped:
                # Mirror _classes_delete: a placeholder that is leaving takes
                # its directory, its processed-cache directory and both of its
                # preprocess entries with it.  Leaving any of them behind
                # leaves a class that reappears on the next directory scan.
                for name in dropped:
                    class_preprocess.pop(name, None)
                    sample_preprocess.pop(name, None)
                    class_dir = cfg.dataset_root / sanitize_class_name(name)
                    processed_dir = self._processed_cache_dir(cfg.dataset_root) / sanitize_class_name(name)
                    for d in (class_dir, processed_dir):
                        if d.exists():
                            try:
                                shutil.rmtree(d)
                            except Exception:
                                pass
                classes = survivors

        classes = list(classes) + added
        self._classes_save(cfg.dataset_root, classes, class_preprocess=class_preprocess, sample_preprocess=sample_preprocess)
        return classes

    @staticmethod
    def _sign_sequence_classes(sequence: Any) -> List[str]:
        """The authoritative sign_ocr class list: unique target characters.

        A Car License project has no class editor: its classes ARE the
        characters of `sign_sequence`, in sequence order.  The '?' pad and
        anything that is not one 0-9/A-Z character are not classes.
        """
        out: List[str] = []
        for ch in normalize_sign_sequence(sequence):
            text = str(ch or "").strip().upper()
            if (
                len(text) == 1
                and text != SIGN_SEQUENCE_PAD
                and re.fullmatch(r"[0-9A-Z]", text)
                and text not in out
            ):
                out.append(text)
        return out

    @staticmethod
    def _sign_sequences_classes(sequences: Any) -> List[str]:
        """Union of unique characters across every allowed plate.

        First-appearance order: ["A12", "B18"] -> ["A", "1", "2", "B", "8"].
        Duplicate characters (A12 + A18) contribute one class and keep the
        order in which they first appeared.
        """
        out: List[str] = []
        for text in normalize_sign_sequences(sequences):
            for ch in text:
                if ch not in out:
                    out.append(ch)
        return out

    def _purge_sign_artifact_dirs(
        self, dataset_root: Path, keep: List[str]
    ) -> None:
        """Delete every class folder that is not one of `keep`.

        The three roots that can hold a class are the raw dataset, the
        processed cache and the mixed training set.  A stale "Class 1"/
        "Class 2" left in ANY of them is dangerous: `map_class_to_char`
        reads the trailing digit, so EMNIST injection fills the stale
        folder and it becomes a REAL model label (often the one the target
        digit should have owned).  Only directories are touched, hidden
        names are skipped, and the comparison goes through the same
        sanitiser the folders were created with.
        """
        wanted = {sanitize_class_name(str(n)).lower() for n in keep if str(n).strip()}
        roots = (
            Path(dataset_root),
            self._processed_cache_dir(Path(dataset_root)),
            Path(dataset_root).parent / ".tm_ocr_mixed",
        )
        for root in roots:
            try:
                if not root.exists():
                    continue
                children = list(root.iterdir())
            except Exception:
                continue
            for child in children:
                try:
                    if not child.is_dir():
                        continue
                    name = child.name
                    if name.startswith("."):
                        continue
                    if sanitize_class_name(name).lower() in wanted:
                        continue
                    shutil.rmtree(child, ignore_errors=True)
                except Exception:
                    continue

    def _enforce_sign_classes(
        self, session_id: str, sequence: Any
    ) -> List[str]:
        """Lock the class list and every class folder to the sign sequence.

        Called when a sign project is opened, when its classes are synced
        from the target bar, and at the start of training.  Returns the
        authoritative class list; with no usable target it leaves the
        project untouched (the training path refuses that case explicitly).
        """
        with self._lock:
            cfg = self._configs.get(session_id)
        if cfg is None:
            raise RuntimeError("missing config")
        chars = self._sign_sequences_classes(sequence)
        if not chars:
            return self._classes_load(cfg.dataset_root)
        classes = self._classes_sync_ocr(session_id=session_id, names=chars)
        self._purge_sign_artifact_dirs(cfg.dataset_root, classes)
        return classes

    def _classes_sync_ocr(self, session_id: str, names: List[Any]) -> List[str]:
        """Make the class list exactly the sign's characters, in sequence order.

        The road-sign UI has no class editor of its own: the three inline slots
        ARE the class list.  Every edit — typing a slot, swapping two, opening
        a saved project — calls this, so STATE.classes and the dataset can
        never drift from the sequence the model is being trained to read.

        Keeping a character keeps its samples: a class is a name in an ordered
        list plus its directory, so an existing class contributes its name and
        directory and only the list is rewritten.  A character the sequence
        does NOT name is removed together with its preprocess entries and
        processed cache — in this project type such a class has no card to
        capture into and no slot that can match it, so it would survive only
        as a stale model output that makes every real sign carry an extra
        near-miss.

        Case is reconciled rather than treated as a second class:
        sanitize_class_name() collides "a" with "A", and the device compares
        characters uppercased, so an old lowercase class is renamed to the
        slot's character instead of being duplicated.
        """
        with self._lock:
            cfg = self._configs.get(session_id)
        if cfg is None:
            raise RuntimeError("missing config")

        wanted: List[str] = []
        for raw in names:
            text = str(raw).strip()
            if not text:
                continue
            if len(text) != 1:
                raise ValueError("Sign characters must be one character each.")
            text = text.upper()
            if not re.fullmatch(r"[0-9A-Z]", text):
                raise ValueError(f"Not a sign character: {raw!r}")
            if text not in wanted:
                wanted.append(text)
        if not wanted:
            raise ValueError("The sign needs at least one character.")

        from ocr_dataset import BACKGROUND_CLASS_NAME

        if any(n.lower() == BACKGROUND_CLASS_NAME.lower() for n in wanted):
            raise ValueError(f'"{BACKGROUND_CLASS_NAME}" is reserved for the reject class.')

        classes = self._classes_load(cfg.dataset_root)
        class_preprocess = self._class_preprocess_load(cfg.dataset_root)
        sample_preprocess = self._sample_preprocess_load(cfg.dataset_root)

        # Existing names are matched the way the filesystem matches them —
        # through sanitize_class_name(), case-insensitively — so this never
        # creates a second class beside one that is already training samples.
        wanted_keys = {n.lower() for n in wanted}
        actual_by_key: Dict[str, str] = {}
        for c in classes:
            actual_by_key.setdefault(sanitize_class_name(str(c)).lower(), str(c))

        keep: List[str] = []
        for name in wanted:
            actual = actual_by_key.get(name.lower())
            if actual is None:
                keep.append(name)
                continue
            if actual != name:
                self._rename_class_storage(cfg.dataset_root, actual, name)
                if actual in class_preprocess:
                    class_preprocess[name] = class_preprocess.pop(actual)
                if actual in sample_preprocess:
                    sample_preprocess[name] = sample_preprocess.pop(actual)
            keep.append(name)

        # Decide what leaves BEFORE any rename: a case-variant of a kept
        # character must not be deleted after its directory became the kept
        # one's directory (a no-op rename on a case-insensitive filesystem).
        dropped = [
            c for c in classes
            if sanitize_class_name(str(c)).lower() not in wanted_keys
        ]
        for name in dropped:
            class_preprocess.pop(name, None)
            sample_preprocess.pop(name, None)
            # Mirror _classes_delete: the directory, the processed cache and
            # both preprocess records travel together.  A survivor would
            # otherwise reappear on the next directory scan.
            for d in (
                cfg.dataset_root / sanitize_class_name(name),
                self._processed_cache_dir(cfg.dataset_root) / sanitize_class_name(name),
            ):
                if d.exists():
                    try:
                        shutil.rmtree(d)
                    except Exception:
                        pass

        self._classes_save(
            cfg.dataset_root,
            keep,
            class_preprocess=class_preprocess,
            sample_preprocess=sample_preprocess,
        )
        # The list above is the authority.  Any folder on disk that is not
        # one of its characters — including a stale "Class 1"/"Class 2" that
        # was never in tm_classes.json — is a label waiting to be injected
        # into the next training run, so remove it here as well as in
        # _enforce_sign_classes().
        self._purge_sign_artifact_dirs(cfg.dataset_root, keep)
        return list(keep)

    def _rename_class_storage(self, dataset_root: Path, old_name: str, new_name: str) -> None:
        """Move a class's directory and processed cache to a new spelling.

        Only the case can differ here (the caller has already matched the two
        names through sanitize_class_name), and on macOS those paths are the
        same inode: rename() then reports the destination exists, which is
        success from the class list's point of view.  Both failures are
        non-fatal because the class NAME is what the UI and the model use.
        """
        old_dir = dataset_root / sanitize_class_name(old_name)
        new_dir = dataset_root / sanitize_class_name(new_name)
        if old_dir.exists() and not new_dir.exists():
            try:
                old_dir.rename(new_dir)
            except Exception:
                pass
        old_cache = self._processed_cache_dir(dataset_root) / sanitize_class_name(old_name)
        new_cache = self._processed_cache_dir(dataset_root) / sanitize_class_name(new_name)
        if old_cache.exists() and not new_cache.exists():
            try:
                old_cache.rename(new_cache)
            except Exception:
                pass

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

    def _train_result_path(self, dataset_root: Path) -> Path:
        return dataset_root.parent / "tm_train_latest.json"

    def _export_run(self, session_id: str, export_dir: Path, model_name: str, array_name: str, overwrite: bool = False, runtime_config: Optional[Dict[str, Any]] = None) -> Path:
        with self._lock:
            cfg = self._configs.get(session_id)
        if cfg is None:
            raise RuntimeError("missing config")
        # Preview-time OOD + preprocessing thresholds. Fall back to the same
        # defaults the /preview/predict endpoint uses so the device header is
        # always self-contained even if the SPA didn't send a value.
        rc = runtime_config if isinstance(runtime_config, dict) else {}
        ood_sign_min = float(rc.get("ood_sign_min", 0.3))
        ood_sign_max = float(rc.get("ood_sign_max", 70.0))
        ood_max_prob = float(rc.get("ood_max_prob", 0.60))
        ood_entropy_max = float(rc.get("ood_entropy_max", 0.70))
        bg_dark_thresh = int(rc.get("bg_dark_thresh", 0))
        bg_lum_thresh = int(rc.get("bg_lum_thresh", 100))
        latest = self._train_result_path(cfg.dataset_root)
        if not latest.exists():
            raise RuntimeError("missing trained model")
        meta = json.loads(latest.read_text(encoding="utf-8"))
        tflite_path = Path(str(meta.get("tflite_path") or "")).expanduser().resolve()
        labels = list(meta.get("labels") or []) if isinstance(meta, dict) else []
        img_size = int(meta.get("img_size") or 96) if isinstance(meta, dict) else 96
        # Capture / preprocessing / model sides are separate by design: the
        # camera can stream 160x160 while the model consumes 96x96 (or a
        # glyph canvas at 28x28).  The sketches resize between them.
        try:
            capture_side = int(rc.get("capture_frame_side") or getattr(cfg, "serial_frame_side", 96) or 96)
        except (TypeError, ValueError):
            capture_side = 96
        capture_side = max(8, min(512, capture_side))
        try:
            preprocess_side = int(rc.get("preprocess_frame_side") or 0)
        except (TypeError, ValueError):
            preprocess_side = 0
        if preprocess_side <= 0:
            preprocess_side = 96 if is_glyph_project(getattr(cfg, "project_type", "")) else int(img_size)
        preprocess_side = max(8, min(512, preprocess_side))
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
        # OCR projects feed the device a 28x28 glyph canvas built by the
        # integer-exact front-end (ocr_glyph.c), not the 96x96 sign pipeline.
        # The two device sketches are separate folders now, and each one
        # #errors if handed the other's export — so this macro travels with the
        # model and IS the routing decision.  A project type is a property of
        # the export, not of the sketch that happens to compile it.
        # Detected two ways on purpose: the project type is authoritative, and
        # the "_background" class is the contract the device itself keys on
        # (background_class_id()), so an export carrying one needs the flag
        # even if its project_type metadata says otherwise.
        ocr_export = (
            is_glyph_project(getattr(cfg, "project_type", ""))
            or any(str(x).strip().lower() == "_background" for x in labels)
        )
        # sign_ocr is a glyph project too, but it belongs to its OWN sketch:
        # the sign model reads three 28x28 slots per frame, while TFLite_OCR/
        # is the single-glyph reader.  The routing therefore needs both flags,
        # and each .ino refuses the other's export at compile time.
        sign_export = (
            str(getattr(cfg, "project_type", "") or "").strip().lower()
            == PROJECT_TYPE_SIGN_OCR
        )
        # The target the exported model is expected to read.  runtime_config
        # wins (the SPA forwards the live editor values), then the TrainConfig
        # the model was trained with.  normalize_sign_sequence pads a short
        # sequence, so the arrays below always have SIGN_SEQUENCE_LEN entries.
        # The sequence is a property of the TRAINING RUN, not of the session
        # config: the SPA forwards its live editor state in runtime_config,
        # and the run's train_config.json is the fallback for an older SPA.
        sign_seqs_src = rc.get("sign_sequences")
        sign_seq_src = rc.get("sign_sequence")
        if (
            sign_seqs_src in (None, "", [])
            and (sign_seq_src is None or sign_seq_src == "" or sign_seq_src == [])
        ):
            train_cfg_saved = self._project_train_cfg(cfg.dataset_root)
            sign_seqs_src = train_cfg_saved.get("sign_sequences")
            sign_seq_src = train_cfg_saved.get("sign_sequence")
        else:
            train_cfg_saved = None
        sign_sequences = normalize_sign_sequences(sign_seqs_src)
        if not sign_sequences:
            sign_sequences = normalize_sign_sequences(sign_seq_src)
        sign_seq = normalize_sign_sequence(
            sign_sequences[0] if sign_sequences else sign_seq_src
        )
        if not sign_sequences:
            # Legacy/incomplete target: keep the export well-formed (the
            # device will simply never match a '?' plate) instead of emitting
            # an empty matrix.
            sign_sequences = [normalize_sign_sequence(sign_seq)]
        reading_src = rc.get("reading_order")
        if reading_src is None or str(reading_src).strip() == "":
            if train_cfg_saved is None:
                train_cfg_saved = self._project_train_cfg(cfg.dataset_root)
            reading_src = train_cfg_saved.get("reading_order")
        reading_order = str(reading_src or "ltr").strip().lower()
        if reading_order not in ("ltr", "rtl"):
            reading_order = "ltr"
        try:
            sign_match_min_prob = float(rc.get("match_min_prob", 0.60))
        except (TypeError, ValueError):
            sign_match_min_prob = 0.60
        sign_match_min_prob = max(0.0, min(1.0, sign_match_min_prob))
        # Which rejector did THIS run train?  A `_background` class can only
        # replace OOD if it actually exists in the exported label list.  An
        # old run with no recorded flag used it as the only rejector (the
        # pre-toggle device behaviour), so preserve that for legacy exports.
        has_background_class = any(
            str(x).strip().lower() == "_background" for x in labels
        )
        meta_bg_replace = (
            meta.get("sign_background_replace_ood")
            if isinstance(meta, dict) else None
        )
        sign_background_is_only_rejector = (
            bool(meta_bg_replace)
            if meta_bg_replace is not None
            else bool(has_background_class)
        ) and has_background_class
        # Map each target character to its class index the same way the
        # device's match rule compares: label.upper() == char.upper().  -1
        # means "no such class" and the device treats that slot as never
        # matching (it refuses anything outside [-1, kCategoryCount)).
        upper_labels = [str(x).strip().upper() for x in labels]
        sign_class_indices = []
        for ch in sign_seq:
            try:
                sign_class_indices.append(upper_labels.index(str(ch).strip().upper()))
            except ValueError:
                sign_class_indices.append(-1)
        # One index row per allowed plate; -1 marks a character that has no
        # class in this export (the device treats that slot as never matching).
        sign_class_matrix: List[List[int]] = []
        for plate in sign_sequences:
            row: List[int] = []
            for ch in plate:
                try:
                    row.append(upper_labels.index(str(ch).strip().upper()))
                except ValueError:
                    row.append(-1)
            sign_class_matrix.append(row)
        sign_target_lines = []
        if sign_export:
            sign_target_lines = [
                "// ── Sign OCR target (project_type == sign_ocr) ─────────────────",
                "// The character the model is expected to read in each slot, in",
                "// LABEL order.  kSignReadingOrder applies AFTER the split: 0 =",
                "// left-to-right, 1 = right-to-left (boxes reversed, never mirrored).",
                "// A class index of -1 means the character has no class in this",
                "// export, so that slot can never match.",
                "constexpr char kTargetSignSequence[] = " + _c_str("".join(sign_seq)) + ";",
                "constexpr int kTargetSignClassIndices["
                + str(int(len(sign_class_indices))) + "] = {"
                + ", ".join(str(int(i)) for i in sign_class_indices) + "};",
                "// ── Gatekeeper whitelist (all allowed 3-character plates) ──────",
                "#define TM_SIGN_MULTI_TARGET_EXPORTED 1",
                "constexpr int kTargetSignCount = " + str(int(len(sign_sequences))) + ";",
                "constexpr char kTargetSignSequences[kTargetSignCount][4] = { "
                + ", ".join(_c_str(str(p)) for p in sign_sequences) + " };",
                "constexpr int kTargetSignClassMatrix[kTargetSignCount][3] = {",
                *[
                    "    { " + ", ".join(str(int(v)) for v in row) + " },"
                    for row in sign_class_matrix
                ],
                "};",
                "constexpr int kSignReadingOrder = " + ("1" if reading_order == "rtl" else "0") + ";",
                "constexpr float kSignMatchMinProb = " + f"{sign_match_min_prob:.6f}f;",
                "// 1 = the model's own _background class is the only rejector",
                "//     (OOD Layers 1-3 are bypassed on the device).",
                "// 0 = no _background class; the standard OOD gates + the",
                "//     per-slot kSignMatchMinProb floor do the rejecting.",
                "#define TM_SIGN_USE_BACKGROUND_OOD "
                + ("1" if sign_background_is_only_rejector else "0"),
                "constexpr bool kSignBackgroundReplacesOod = "
                + ("true" if sign_background_is_only_rejector else "false")
                + ";",
                "// Index of \"_background\" in kCategoryLabels, or -1 when the",
                "// run trained without the reject class.",
                "constexpr int kBackgroundClassIndex = "
                + str(int(next(
                    (i for i, x in enumerate(labels)
                     if str(x).strip().lower() == "_background"),
                    -1,
                ))) + ";",
                "// Legacy spellings kept for sketches built before the rename.",
                "#define TM_SIGN_BACKGROUND_IS_THE_ONLY_REJECTOR "
                + ("1" if sign_background_is_only_rejector else "0"),
                "constexpr bool kSignBackgroundIsTheOnlyRejector = "
                + ("true" if sign_background_is_only_rejector else "false")
                + ";",
                "",
            ]
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
                "// ── Resolution layers (capture -> preprocess -> model) ───────────",
                "// The MCU may capture at a different side than the model input;",
                "// the sketches box-average the captured square frame down to the",
                "// preprocessing side before the glyph/ROI transform.",
                "#define TM_MODEL_INPUT_SIDE " + str(int(img_size)),
                "#define TM_PREPROCESS_FRAME_SIDE " + str(int(preprocess_side)),
                "#define TM_CAPTURE_FRAME_SIDE " + str(int(capture_side)),
                "constexpr int kModelInputSide = TM_MODEL_INPUT_SIDE;",
                "constexpr int kPreprocessFrameSide = TM_PREPROCESS_FRAME_SIDE;",
                "constexpr int kCaptureFrameSide = TM_CAPTURE_FRAME_SIDE;",
                "",
                # 1 = OCR glyph model (28x28 canvas, "_background" rejector,
                # ocr_glyph.c front-end) / 0 = legacy 96x96 sign model.
                "// TM_OCR_GLYPH_MODEL says which device sketch this model belongs to:",
                "//   1 = TFLite_OCR/ (single-glyph, 28x28 canvas, \"_background\" rejector)",
                "//   0 = TFLite/ (legacy 96x96) or TFLite_Sign/ (3-character sign reader)",
                "// Each sketch #errors if it is handed another sketch's export.",
                "#define TM_OCR_GLYPH_MODEL " + ("1" if (ocr_export and not sign_export) else "0"),
                "// TM_SIGN_OCR_MODEL is the 3-character sign reader:",
                "//   1 = TFLite_Sign/ (split -> three 28x28 slots -> binary verdict)",
                "//   0 = every other sketch",
                "#define TM_SIGN_OCR_MODEL " + ("1" if sign_export else "0"),
                "",
                "// ── The Dark Thr / Lum Thr window the project was trained through ──",
                "// The device derives its ink bounds exactly as the host does:",
                "//   adjusted = (kDarkThr != 0) || (kLumThr != 100)   // 0/100 = Otsu",
                "//   thresh   = adjusted ? clamp(kLumThr - 1, 1, 254) : -1  (Otsu)",
                "//   dark_thr = adjusted ? clamp(kDarkThr + 1, 0, 255) : 0",
                "// so the defaults are a bit-exact no-op.",
                "#define TM_THRESHOLD_WINDOW_EXPORTED 1",
                "#define kDarkThr " + str(int(bg_dark_thresh)),
                "#define kLumThr " + str(int(bg_lum_thresh)),
                "",
                "constexpr int kCategoryCount = " + str(int(len(labels))) + ";",
                "extern const char* kCategoryLabels[kCategoryCount];",
                "",
                *sign_target_lines,
                "// ── Runtime tuning exported from the Preview panel ───────────────",
                "// OOD (out-of-distribution) rejection thresholds.  The device loop",
                "// uses these the same way the Python preview does so a sign is only",
                "// accepted when sign_pct is in range, max softmax prob is high",
                "// enough, and entropy is low enough.",
                "constexpr float kOodSignMinPct = " + f"{ood_sign_min:.6f}f;",
                "constexpr float kOodSignMaxPct = " + f"{ood_sign_max:.6f}f;",
                "constexpr float kOodMaxProbMin = " + f"{ood_max_prob:.6f}f;",
                "constexpr float kOodEntropyRatioMax = " + f"{ood_entropy_max:.6f}f;",
                "",
                "// Preprocessing brightness/darkness thresholds used by the device",
                "// sign-search front-end.  Pixels darker than kBgDarkThresh become",
                "// white (shadow/background); pixels above kBgLumThresh are treated",
                "// as background.  Values are 0-255 (luma).",
                "constexpr int kBgDarkThresh = " + str(int(bg_dark_thresh)) + ";",
                "constexpr int kBgLumThresh = " + str(int(bg_lum_thresh)) + ";",
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
        # The session's project type is what decides every transform below
        # AND what the Start Page needs in order to route the reopened file
        # back to the right workspace — so it is written here, in the one
        # manifest writer, rather than inferred from the payload later.
        project_type = str(getattr(cfg, "project_type", PROJECT_TYPE_IMAGE) or PROJECT_TYPE_IMAGE)
        # Header first, and in BOTH halves of the archive: the manifest is the
        # door (a reader classifies the file without parsing anything else),
        # and the state file repeats it so the half carrying the settings is
        # self-describing too — a bug report pasted from disk, or a future
        # import tool handed only tm_project_state.json, can still tell what
        # it is holding and whether it predates the complete train_cfg.
        merged_project_state.update(project_header(project_type))
        # The snapshot inside train_cfg is reconciled the same way, so the
        # archive cannot disagree with itself: whatever the session was when
        # it saved is what a later open must report, regardless of what the
        # client put in the payload.
        _saved_train_cfg = merged_project_state.get("train_cfg")
        if isinstance(_saved_train_cfg, dict):
            _saved_train_cfg["project_type"] = project_type
            _saved_train_cfg["img_size"] = int(cfg.img_size)
        manifest: Dict[str, Any] = {
            **project_header(project_type),
            # The v1 spelling of the same thing the magic says.  Kept so a
            # reader written against the old format still recognises the file.
            "format": "tmproj",
            # Persisted alongside the type so reopening restores the model
            # input size too (the processed cache in the archive is already
            # at this size — a reopen that guessed 96 would silently let a
            # later capture overwrite it with the wrong geometry).
            "img_size": int(cfg.img_size),
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
        # The global mode is the FALLBACK for a class with no entry of its own,
        # so it has to come from the same place the session reads it — the
        # train_cfg the SPA actually edits.  (It used to be read from a root
        # key no writer ever set, which meant a save always rebuilt the cache
        # with the default mode.)
        _state_train_cfg = merged_project_state.get("train_cfg")
        _state_mode = ""
        if isinstance(_state_train_cfg, dict):
            _state_mode = str(_state_train_cfg.get("preprocess_mode") or "")
        self._rebuild_processed_cache(
            dataset_root,
            class_preprocess=class_preprocess,
            sample_preprocess=sample_preprocess,
            global_preprocess_mode=str(
                _state_mode or merged_project_state.get("preprocess_mode") or PREPROCESS_MODE_AUTO_BY_LABEL
            ),
            out_size=int(cfg.img_size),
            project_type=project_type,
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

        # ── Flush the client's preprocessing settings into tm_classes.json ──
        # The archive carries tm_classes.json verbatim, so a setting the
        # payload reports but the file was never told would be SAVED AS
        # SOMETHING ELSE — the reader would show a value the writer never
        # agreed to.  class_preprocess has its own endpoint (/classes/
        # save_config) that persists on every edit; sample_preprocess has no
        # such writer and lives in the SPA until something flushes it.  This
        # is that flush, and it happens before the zip is opened so both
        # halves of the archive carry the same settings.
        if isinstance(project_state, dict) and (
            project_state.get("class_preprocess") is not None
            or project_state.get("sample_preprocess") is not None
        ):
            self._classes_save(
                dataset_root,
                self._classes_load(dataset_root),
                class_preprocess=merged_project_state.get("class_preprocess"),
                sample_preprocess=merged_project_state.get("sample_preprocess"),
            )

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

            # What kind of file is this?  Not a gate — every v1 archive on a
            # user's disk still opens — but the answer decides how much of the
            # rest of the payload to trust, and gets reported back so the UI
            # can tell the user which settings came back at their defaults.
            header = read_project_header(manifest)

            # Adopt the archived project type BEFORE anything re-preprocesses
            # a sample, so every later capture / preview / training of this
            # session uses the transform the file was built with.  Old
            # archives have no such key and stay image projects.
            archived_type = str(manifest.get("project_type") or "").strip().lower()
            resolved_type = archived_type if archived_type in KNOWN_PROJECT_TYPES else ""
            if resolved_type:
                self.update_config(session_id, project_type=resolved_type)
            # Resolved here, not just applied: the state payload below has to
            # report the size and type the session is ACTUALLY running with,
            # or the SPA keeps displaying the previous project's 96 while the
            # backend has already switched to 28.
            resolved_size = 0
            archived_size = manifest.get("img_size")
            if archived_size not in (None, ""):
                try:
                    size = int(archived_size)
                    if 8 <= size <= 512:
                        self.update_config(session_id, img_size=size)
                        resolved_size = size
                except Exception:
                    pass
            # An archive with no recorded geometry still declares a project
            # type, and the type implies one.  Without this, opening a legacy
            # OCR file from an Image session left the session at 96x96 while
            # the glyph pipeline ran at 28x28 — the next capture would then
            # write 96x96 frames into a cache the file restored at 28x28.
            if not resolved_size and resolved_type:
                implied = int(default_img_size(resolved_type) or 0)
                if 8 <= implied <= 512:
                    self.update_config(session_id, img_size=implied)
                    resolved_size = implied

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

            # A trained model survives this open only when the ARCHIVE ITSELF
            # bundled one.  The record inside names absolute paths from the
            # machine that saved the file, and those routinely still resolve
            # here (two projects saved in the same default workspace share
            # `runs/latest`), so writing it back would let
            # _project_state_payload report export_enabled="1" off a graph
            # this session never loaded — which is exactly how an Image model
            # stayed "trained" inside a freshly opened OCR project.  Bundled
            # wins; absent means no model, never a stale one.
            latest_in = tmp_dir / "tm_train_latest.json"
            latest_out = self._train_result_path(dataset_root)
            run_dir_hint = str(manifest.get("train_run_dir") or "")
            run_dst = workspace_root / "runs" / "latest"
            restored = False
            if latest_in.exists():
                try:
                    meta = json.loads(latest_in.read_text(encoding="utf-8"))
                except Exception:
                    meta = None
                run_src = (tmp_dir / run_dir_hint) if run_dir_hint else None
                if isinstance(meta, dict) and run_src is not None and run_src.exists():
                    if run_dst.exists():
                        shutil.rmtree(run_dst)
                    run_dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copytree(run_src, run_dst)
                    meta["run_dir"] = str(run_dst)
                    for key in ["keras_model_path", "tflite_path", "model_h_path", "model_cpp_path"]:
                        nm = Path(str(meta.get(key) or "")).name
                        cand = run_dst / nm if nm else None
                        if cand is not None and cand.exists():
                            meta[key] = str(cand)
                    restored = self._run_matches_project(run_dst, resolved_type, resolved_size)
                if restored:
                    latest_out.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
                else:
                    # Nothing usable came out of the archive: leave the
                    # RESET state alone (no latest record, no run directory),
                    # so export_enabled is false and the preview has no
                    # interpreter to reuse.
                    if run_dst.exists():
                        shutil.rmtree(run_dst, ignore_errors=True)
                    latest_out.unlink(missing_ok=True)
            project_state_in = tmp_dir / str(manifest.get("project_state") or "tm_project_state.json")
            project_state: Optional[Dict[str, Any]] = None
            if project_state_in.exists():
                try:
                    loaded = json.loads(project_state_in.read_text(encoding="utf-8"))
                    if isinstance(loaded, dict):
                        project_state = loaded
                except Exception:
                    project_state = None
            if project_state is None:
                project_state = {}
            # Reconcile the two halves of the archive.  The manifest is the
            # header (written by the one save path from the session's own
            # config); the state file is whatever the saver happened to
            # snapshot, and a v1 one carries no project_type at all.  So the
            # manifest wins where the two disagree, and the state is filled in
            # where it is silent — otherwise a legacy project reopens with the
            # SPA still gating the PREVIOUS project's workspace.
            state_cfg = project_state.get("train_cfg")
            # Measured from the FILE's own contents, before anything below is
            # added to it: a setting the manifest supplied is a value the user
            # chose when they saved, not a default filled in on load.
            missing = missing_train_cfg_fields(state_cfg)
            if not isinstance(state_cfg, dict):
                state_cfg = {}
                project_state["train_cfg"] = state_cfg
            if resolved_type:
                project_state["project_type"] = resolved_type
                state_cfg["project_type"] = resolved_type
            if resolved_size:
                state_cfg["img_size"] = resolved_size
            # A sign archive's target is its class list.  Prune any class
            # folder the archive's own target does not name BEFORE the state
            # payload reports counts/labels — a stale "Class 1" here would
            # otherwise be injected with EMNIST samples at the next training
            # and become a real model output.
            if resolved_type == PROJECT_TYPE_SIGN_OCR:
                seq_src = state_cfg.get("sign_sequences") or state_cfg.get("sign_sequence")
                if not seq_src:
                    saved = self._project_train_cfg(
                        dataset_root, project_state=project_state
                    )
                    seq_src = saved.get("sign_sequences") or saved.get("sign_sequence")
                try:
                    self._enforce_sign_classes(session_id, seq_src)
                    # The enforcement may have normalized/added classes;
                    # report the post-prune list to the SPA.
                    project_state["classes"] = self._classes_load(dataset_root)
                except Exception:
                    pass
            project_state["project_format"] = {
                **header,
                "missing_train_cfg_fields": missing,
            }
            return self._project_state_payload(dataset_root, project_state=project_state)
        finally:
            if tmp_dir.exists():
                shutil.rmtree(tmp_dir)

    def _run_matches_project(self, run_dir: Path, project_type: str, img_size: int) -> bool:
        """Does a restored run belong to the project being opened?

        A quantized graph is only usable by the workspace that will run it:
        an OCR archive must not adopt an Image project's 96x96 model — nor a
        28x28 one the other way round — just because both were trained in a
        `runs/latest` this machine still has.  The run's own
        `train_config.json` is the record the trainer wrote, and it carries
        both the project type and the input geometry, so it answers the
        question without guessing.  A run that does not say (an older
        archive, or a hand-made directory) is not refused on this basis
        alone — the archive's own header already gates it.
        """
        cfg_path = run_dir / "train_config.json"
        if not cfg_path.exists():
            return True
        try:
            loaded = json.loads(cfg_path.read_text(encoding="utf-8"))
        except Exception:
            return True
        run_cfg = pick_train_cfg(loaded) if isinstance(loaded, dict) else {}
        if project_type:
            run_type = str(run_cfg.get("project_type") or "").strip().lower()
            if run_type and run_type != project_type:
                return False
        if img_size:
            try:
                run_size = int(run_cfg.get("img_size") or 0)
            except Exception:
                run_size = 0
            if run_size and run_size != int(img_size):
                return False
        return True

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
        # Three sources, most authoritative first: the state being saved (or
        # the one that was just loaded), the run's own train_config.json, and
        # the train meta.  Each is projected through the SAME schema, so a
        # field cannot survive from one source and vanish from another.
        if isinstance(project_state, dict):
            cfg = pick_train_cfg(project_state.get("train_cfg"))
            if cfg:
                return cfg
        run_cfg = dataset_root.parent / "runs" / "latest" / "train_config.json"
        if run_cfg.exists():
            try:
                cfg = pick_train_cfg(json.loads(run_cfg.read_text(encoding="utf-8")))
                if cfg:
                    return cfg
            except Exception:
                pass
        latest = self._train_result_path(dataset_root)
        if latest.exists():
            try:
                meta = json.loads(latest.read_text(encoding="utf-8"))
                if isinstance(meta, dict):
                    return pick_train_cfg(meta)
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

    def _samples_seed_emnist(
        self,
        session_id: str,
        classes: Optional[List[Any]] = None,
        count: Any = TEMPLATE_SAMPLE_DEFAULT_COUNT,
        only_empty: bool = False,
        sign_sequence: Any = None,
        reading_order: Any = None,
    ) -> Dict[str, Any]:
        """Fill character classes with camera-realistic EMNIST template samples.

        The developer shortcut for trying the sign pipeline before a camera
        is wired up.  Each glyph is rendered as a 96x96 DARK-INK-ON-LIGHT
        frame (see `ocr_dataset.template_capture_from_glyph`) so the exact
        same `preprocess_glyph` path Otsu-binarizes and crops it as a real
        capture — raw EMNIST is bright-on-dark and would binarize inverted.

        Seeded files are named `emnist_seed_NN.png`; a re-seed replaces only
        those, never a camera capture in the same class folder.
        """
        with self._lock:
            cfg = self._configs.get(session_id)
        if cfg is None:
            raise RuntimeError("missing config")
        project_type = str(getattr(cfg, "project_type", PROJECT_TYPE_IMAGE) or PROJECT_TYPE_IMAGE)
        if not is_glyph_project(project_type):
            raise ValueError("template samples are only available for OCR projects")
        try:
            n = int(count)
        except (TypeError, ValueError):
            n = TEMPLATE_SAMPLE_DEFAULT_COUNT
        n = max(1, min(500, n))

        from ocr_dataset import (
            TEMPLATE_SIGN_CHARS,
            build_template_capture_samples,
            emnist_cache_dir,
            load_emnist_template_bank,
            map_class_to_char,
        )

        dataset_root = cfg.dataset_root
        available = self._classes_load(dataset_root)
        # Match requested names case-insensitively but seed under the class
        # name the project actually stores (the folder is the class name).
        by_fold = {name.strip().upper(): name for name in available if name.strip()}
        requested: List[str] = []
        if classes:
            for raw in classes:
                name = by_fold.get(str(raw or "").strip().upper())
                if name and name not in requested:
                    requested.append(name)
        else:
            requested = list(available)

        targets: List[str] = []
        skipped: List[str] = []
        for name in requested:
            ch = map_class_to_char(name)
            ch = ch.upper() if ch else ""
            if len(ch) == 1 and ch in TEMPLATE_SIGN_CHARS:
                # only_empty is the auto-fill rule: a class that already holds
                # ANY sample — a camera capture or a previous seed — is left
                # alone.  "Empty" is the same count the class card shows.
                if only_empty:
                    class_dir = dataset_root / sanitize_class_name(name)
                    if class_dir.exists() and _list_class_image_files(class_dir):
                        continue
                targets.append(name)
            else:
                skipped.append(name)
        if not targets:
            if only_empty:
                # Nothing to do is the steady state of an auto-fill pass, not
                # an error: every class is already populated.  The caller
                # still gets the full runtime state so the UI can re-render.
                return {
                    "seeded": [],
                    "skipped": skipped,
                    "count": str(n),
                    "states": {},
                    "processed_previews": {},
                    **self._runtime_state_payload(session_id, dataset_root),
                }
            raise ValueError("no character classes to seed")

        bank = load_emnist_template_bank(cache_root=emnist_cache_dir(app_data_dir()))
        sample_preprocess = self._sample_preprocess_load(dataset_root)
        preprocess_changed = False
        for name in targets:
            char = map_class_to_char(name).upper()
            class_dir = dataset_root / sanitize_class_name(name)
            class_dir.mkdir(parents=True, exist_ok=True)
            processed_dir = self._processed_cache_dir(dataset_root) / sanitize_class_name(name)
            # Replace this seeder's own previous output only.
            for folder in (class_dir, processed_dir):
                if not folder.exists():
                    continue
                for p in list(folder.iterdir()):
                    if (
                        p.is_file()
                        and p.name.startswith(TEMPLATE_SAMPLE_PREFIX)
                        and p.suffix.lower() in IMAGE_EXTS
                    ):
                        try:
                            p.unlink()
                        except Exception:
                            pass
            class_map = sample_preprocess.get(name)
            if isinstance(class_map, dict):
                for fname in [k for k in list(class_map) if str(k).startswith(TEMPLATE_SAMPLE_PREFIX)]:
                    class_map.pop(fname, None)
                    preprocess_changed = True
                if not class_map:
                    sample_preprocess.pop(name, None)
            for i, arr in enumerate(build_template_capture_samples(char, count=n, bank=bank)):
                sample_path = class_dir / f"{TEMPLATE_SAMPLE_PREFIX}{i:02d}.png"
                sample_path.write_bytes(
                    _to_png_bytes(Image.fromarray(np.asarray(arr, dtype=np.uint8), mode="L"))
                )
                self._cache_one_sample(
                    dataset_root,
                    name,
                    sample_path,
                    img_size=int(cfg.img_size),
                    project_type=project_type,
                )
        if preprocess_changed:
            self._classes_save(
                dataset_root,
                available,
                class_preprocess=self._class_preprocess_load(dataset_root),
                sample_preprocess=sample_preprocess,
            )

        # A sign project now has enough glyphs to draw a test sign without
        # the user hunting for a road-sign photo.  Storing it here is what
        # makes Preview -> Upload immediately show a ready-to-test frame;
        # it never replaces a REAL upload, and a failure is ignored.
        if str(project_type) == PROJECT_TYPE_SIGN_OCR:
            self._maybe_autostore_synth_preview(
                session_id,
                sign_sequence=sign_sequence,
                reading_order=reading_order,
            )

        # Full post-seed state, not just the classes this call touched: the
        # SPA applies it verbatim (STATE.classes / counts / sample_previews /
        # train_enabled / train_rec) and re-renders the cards in place, so a
        # class that was empty when the toggle flipped must show "24 Image
        # Samples" before the toast even fades.  `classes` is the project's
        # COMPLETE list; `seeded` names the ones this call actually drew.
        return {
            "seeded": targets,
            "skipped": skipped,
            "count": str(n),
            "states": {
                name: self._class_state_payload(dataset_root, name) for name in targets
            },
            **self._runtime_state_payload(session_id, dataset_root),
        }

    def _auto_seed_emnist_enabled(self, dataset_root: Path) -> bool:
        """Is the project's "fill empty classes" developer shortcut on?

        Read from the persisted train config, never from a request: the flag
        is project state, so a class-write endpoint must honour the same
        value the Settings checkbox last saved even when the caller is a
        background sync that carries no options at all.
        """
        run_cfg = Path(dataset_root).parent / "runs" / "latest" / "train_config.json"
        if not run_cfg.exists():
            return False
        try:
            saved = pick_train_cfg(json.loads(run_cfg.read_text(encoding="utf-8")))
        except Exception:
            return False
        return bool(saved.get("auto_seed_emnist_samples"))

    def _seed_empty_char_classes(
        self,
        session_id: str,
        sign_sequence: Any = None,
        reading_order: Any = None,
        strict: bool = False,
    ) -> List[str]:
        """Auto-fill every EMPTY character class, when the shortcut is on.

        Runs after every class mutation (`/classes/add_named`,
        `/classes/sync_ocr`, ...) and after `/train/config` saves the toggle,
        so a class created before the flag flipped is filled by the very next
        write instead of waiting for the next page load.  Returns the names
        actually seeded.  Failures are swallowed by default — a seeder
        problem must never turn "add a class" into an error — but a caller
        that promises the fill in its own response (`/classes/sync_ocr`
        passes strict=True) needs the failure surfaced instead of silently
        leaving the class at zero.

        `sign_sequence`/`reading_order` are the SPA's current target, when
        the caller has it.  They are forwarded to the seeder so a sign
        project also gets a matching synthetic preview image in the same
        call; a background caller may omit them and the persisted train
        config is used instead.
        """
        with self._lock:
            cfg = self._configs.get(session_id)
        if cfg is None:
            return []
        try:
            if not self._auto_seed_emnist_enabled(cfg.dataset_root):
                return []
            from ocr_dataset import TEMPLATE_SIGN_CHARS, map_class_to_char

            empty: List[str] = []
            for name in self._classes_load(cfg.dataset_root):
                ch = (map_class_to_char(name) or "").upper()
                if len(ch) != 1 or ch not in TEMPLATE_SIGN_CHARS:
                    continue
                class_dir = cfg.dataset_root / sanitize_class_name(name)
                if class_dir.exists() and _list_class_image_files(class_dir):
                    continue
                empty.append(name)
            if not empty:
                return []
            result = self._samples_seed_emnist(
                session_id=session_id,
                classes=empty,
                count=TEMPLATE_SAMPLE_DEFAULT_COUNT,
                only_empty=True,
                sign_sequence=sign_sequence,
                reading_order=reading_order,
            )
            return list(result.get("seeded") or [])
        except Exception:
            if strict:
                raise
            return []

    def _runtime_state_payload(
        self, session_id: str, dataset_root: Optional[Path] = None
    ) -> Dict[str, Any]:
        """The whole workspace state a class mutation can change.

        The SPA's `STATE` is the source of truth for the cards, so a write
        that only returns `classes` leaves counts, thumbnails and the Train
        button stale until a reload.  This payload is the one shape every
        such response carries: the ordered class list, the per-class counts
        and sample previews, whether training is now enabled (the SAME rule
        the client applies — >=2 classes, every class non-empty), and the
        dataset-aware training recommendation.
        """
        with self._lock:
            cfg = self._configs.get(session_id)
        root = Path(dataset_root if dataset_root is not None else (cfg.dataset_root if cfg is not None else ""))
        classes = self._classes_load(root)
        counts: Dict[str, int] = {}
        sample_previews: Dict[str, List[Dict[str, str]]] = {}
        for name in classes:
            class_dir = root / sanitize_class_name(name)
            files = _list_class_image_files(class_dir) if class_dir.exists() else []
            counts[name] = len(files)
            previews: List[Dict[str, str]] = []
            for sample in files[:12]:
                try:
                    previews.append(_preview_item_payload(sample))
                except Exception:
                    continue
            sample_previews[name] = previews
        non_empty = sum(1 for name in classes if counts.get(name, 0) > 0)
        total = sum(counts.values())
        payload: Dict[str, Any] = {
            "classes": classes,
            "counts": counts,
            "sample_previews": sample_previews,
            "processed_previews": self._processed_previews_payload(root, classes),
            "train_enabled": bool(
                len(classes) >= 2
                and non_empty >= 2
                and non_empty == len(classes)
                and total > 0
            ),
        }
        if session_id:
            try:
                payload.update(self._train_recommend(session_id=session_id, cfg={}))
            except Exception:
                pass
        return payload

    def _processed_preview_item_payload(self, path: Path) -> Dict[str, str]:
        return _preview_item_payload(path)

    def _cache_one_sample(
        self,
        dataset_root: Path,
        class_name: str,
        sample_path: Path,
        img_size: int = 96,
        project_type: str = PROJECT_TYPE_IMAGE,
    ) -> None:
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
                project_type=project_type,
            )
            dst_dir = self._processed_cache_dir(dataset_root) / sanitize_class_name(class_name)
            dst_dir.mkdir(parents=True, exist_ok=True)
            (dst_dir / (sample_path.stem + ".png")).write_bytes(out_png)
        except Exception:
            pass

    def _preprocess_image_png(self, png: bytes, label_name: str, class_config: Any, sample_config: Any = None, return_crop: bool = False, fast_mode: bool = False, out_size: int = 96, project_type: str = PROJECT_TYPE_IMAGE):
        class_cfg = class_config if isinstance(class_config, dict) else {}
        src_cfg = sample_config if isinstance(sample_config, dict) else {}
        # Project dispatch — ONE key, at the ONE place every sample enters the
        # pipeline (incremental cache writes, full cache rebuilds, class
        # previews, training).  The OCR branch reuses all of the surrounding
        # lifecycle (cache dirs, class dirs, naming, previews) and only swaps
        # the pixel transform.
        if is_glyph_project(project_type):
            return self._preprocess_image_png_ocr(png, class_config, sample_config, return_crop, out_size)
        config = normalize_class_preprocess(sample_config if sample_config is not None else class_config)
        mode = str(config.get("mode") or PREPROCESS_MODE_AUTO_BY_LABEL)
        # The Edit worktable sends per-sample slider values in `sample_config`;
        # those MUST win over the class record or the live Dark/Lum drag never
        # moves the auto crop (it kept redrawing the class-level window).
        bg_dark_thresh = int(config.get("bg_dark_thresh", 0) or 0)
        bg_lum_thresh  = int(config.get("bg_lum_thresh", 100) or 100)
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

    def _preprocess_image_png_ocr(
        self,
        png: bytes,
        class_config: Any,
        sample_config: Any = None,
        return_crop: bool = False,
        out_size: int = GLYPH_SIZE,
    ):
        """OCR project sample transform: the glyph pipeline in ocr_preprocess.

        Deliberately a *branch* of _preprocess_image_png, not a parallel
        lifecycle: the caller has already done the file I/O, the class-name
        sanitising, the cache-directory choice and the preview plumbing.  All
        this swaps is the pixel transform — exactly what `project_type` is
        for.

        Every step below is integer-only so the device's C port can be
        bit-identical (see ocr_preprocess for the derivations): Otsu by
        cross-multiplied integer variance, vertical projection by int64
        column sums, nearest-neighbour by an exact rational index map.
        """
        config = normalize_class_preprocess(sample_config if sample_config is not None else class_config)
        mode = str(config.get("mode") or PREPROCESS_MODE_AUTO_BY_LABEL)
        polarity = int(config.get("ink_polarity", INK_DARK))
        # NORM_CANON, not NORM_STRETCH: the OCR project mixes EMNIST into
        # every training set, EMNIST ships as 28x28, and CANON is the
        # aspect-preserving convention that keeps a stretched camera
        # capture comparable to it.  STRETCH squashes a tall thin glyph
        # ('1', 'I', 'l') onto the square grid — measured, a 26x48 crop of
        # a '1' became an unreadable blob while the EMNIST half of the SAME
        # class stayed a clean glyph.  See ocr_preprocess.normalise_glyph.
        convention = str(config.get("glyph_convention", NORM_CANON))
        apply_segmentation = bool(config.get("glyph_segment", True))

        src_img = Image.open(_bytes_io(png)).convert("L")
        gray = np.asarray(src_img, dtype=np.uint8)
        full_h, full_w = gray.shape[:2]

        # Manual ROI: crop BEFORE Otsu picks a threshold and BEFORE the
        # vertical projection looks for runs — the same contract the image
        # branch below implements via preprocess_blue_diff_array(roi=...).
        # In manual_roi mode the user's box IS the model input, so pixels
        # outside it must not influence anything that measures ink.  Omitting
        # this made Manual ROI a no-op end to end: the rectangle moved in the
        # UI while the model, the processed cache and training all still saw
        # every pixel of the original frame.
        mode = str(config.get("mode") or PREPROCESS_MODE_AUTO_BY_LABEL)
        roi_px = (
            manual_roi_to_pixels(full_h, full_w, config.get("manual_roi"))
            if mode == PREPROCESS_MODE_MANUAL_ROI
            else None
        )
        roi_origin = (0, 0)
        if roi_px is not None:
            rx1, ry1, rx2, ry2 = roi_px
            sub = gray[ry1:ry2, rx1:rx2]
            if sub.size:
                gray = np.ascontiguousarray(sub)
                roi_origin = (rx1, ry1)

        # The editor's Dark Thr / Lum Thr sliders.  Until now the OCR branch
        # read neither, so moving them changed nothing on screen: the preview
        # re-binarised with exact Otsu every time.  They form the sign's
        # luminance window — ink is what falls inside it.
        #
        # The window is the one the image branch already defines
        # (image_preprocess.preprocess_blue_diff_array), and these two lines
        # reproduce it exactly:
        #
        #     is_sign = (gray > bg_dark_thresh) & (gray < bg_lum_thresh)
        #
        # so BOTH bounds are strict.  binarize_ink's own comparison is
        # inclusive (ink = gray <= t), so the slider values are nudged one
        # LSB inward: ink = (gray <= lum-1) & (gray >= dark+1) is bit-for-bit
        # the same set as the mask the sign previews have always drawn.  The
        # same sliders therefore mean the same thing in both project types,
        # and normalize_class_preprocess has already clamped lum to 50..255
        # and dark to 0..100, so neither nudge leaves binarize_ink's range.
        #
        # Gated on "adjusted away from the defaults", because 0/100 is what
        # the UI ships and what every existing OCR model was trained through:
        # pinning the threshold at an untouched 100 would silently re-cut the
        # training cache of every project in the field.  Defaults therefore
        # still mean exact Otsu.  This is the same test the image branch uses
        # to decide the sliders have been tuned (see the crop branches there).
        bg_dark = int(config.get("bg_dark_thresh", 0) or 0)
        bg_lum = int(config.get("bg_lum_thresh", 100) or 100)
        thresholds_adjusted = (bg_dark != 0) or (bg_lum != 100)
        pinned = max(1, min(254, bg_lum - 1)) if thresholds_adjusted else None
        # Dark=0 means "no lower bound": nudging it to 1 would drop pure
        # black glyph pixels and fragment a thin '1'.  Keep 0 as the
        # inclusive lower edge of the user's window.
        floor = (0 if bg_dark <= 0 else min(255, bg_dark + 1)) if thresholds_adjusted else 0

        result = preprocess_glyph(
            gray,
            out_size=int(out_size),
            polarity=polarity,
            apply_segmentation=apply_segmentation,
            convention=convention,
            thresh=pinned,
            dark_thr=floor,
        )
        glyph = np.asarray(result["glyph"], dtype=np.uint8)
        png_bytes = _to_png_bytes(Image.fromarray(glyph, mode="L"))

        if not return_crop:
            return png_bytes

        # Previews mirror the image branch's contract so the same UI code
        # renders both projects:
        #   processed_png -> the decision mask, cropped to the ROI and shown
        #                    BLACK INK ON PURE WHITE (not the raw 0/255 mask
        #                    the model consumes)
        #   crop_png      -> the pre-decision region (what the camera saw)
        box = result.get("box")
        ox, oy = roi_origin
        ink = result.get("ink")
        ink_arr = np.asarray(ink, dtype=bool) if ink is not None else None
        # ── Auto ROI for the Edit worktable ───────────────────────────────
        # The Preview card already has find_sign_roi; the per-sample editor
        # has only one glyph, so it needs the generic interior-band search:
        # threshold window -> component/border cleanup -> tight padded box.
        # This is the box the SPA draws in green, and the processed pane is
        # re-cut from it so Dark/Lum drags update BOTH live.
        auto_roi = None
        if mode != PREPROCESS_MODE_MANUAL_ROI:
            auto_roi = find_interior_ink_roi(
                gray, thresh=pinned, dark_thr=floor, polarity=polarity,
            )
        if (
            isinstance(auto_roi, dict)
            and str(auto_roi.get("method") or "") == "interior_ink"
            and ink_arr is not None
        ):
            ax1, ay1, ax2, ay2 = (int(v) for v in auto_roi["box"])
            auto_ink = auto_roi.get("ink")
            if auto_ink is not None and np.asarray(auto_ink).shape == ink_arr.shape:
                view = np.asarray(auto_ink, dtype=bool)[ay1:ay2, ax1:ax2]
            else:
                view = ink_arr[ay1:ay2, ax1:ax2]
            if view.size:
                display = np.where(view, 0, 255).astype(np.uint8)
                mp_png = _to_png_bytes(Image.fromarray(display, mode="L"))
            else:
                mp_png = png_bytes
            cp_crop = np.ascontiguousarray(gray[ay1:ay2, ax1:ax2])
            if cp_crop.size:
                cp_png = _to_png_bytes(Image.fromarray(cp_crop, mode="L"))
            else:
                cp_png = png_bytes
            crop_norm = [
                (ax1 + ox) / full_w if full_w else 0.0,
                (ay1 + oy) / full_h if full_h else 0.0,
                (ax2 + ox) / full_w if full_w else 1.0,
                (ay2 + oy) / full_h if full_h else 1.0,
            ]
            return {"png": png_bytes, "crop": crop_norm,
                    "processed_png": mp_png, "crop_png": cp_png,
                    "auto_roi": [ax1, ay1, ax2, ay2],
                    "sign_pct": float(auto_roi.get("sign_pct") or 0.0)}
        # Fallback: the segmented glyph box (or the whole manual ROI).
        if ink_arr is not None:
            view = ink_arr
            if mode != PREPROCESS_MODE_MANUAL_ROI and box is not None:
                ix1, iy1, ix2, iy2 = (int(v) for v in box)
                cropped = ink_arr[iy1:iy2, ix1:ix2]
                if cropped.size:
                    view = cropped
            display = np.where(view, 0, 255).astype(np.uint8)
            mp_png = _to_png_bytes(Image.fromarray(display, mode="L"))
        else:
            mp_png = png_bytes
        # result["box"] is in the coordinates of the frame preprocess_glyph
        # was given, which is the ROI crop in manual mode.  Report it in
        # FULL-frame coordinates so the UI's rectangle lands on the pixels it
        # actually describes.
        crop_norm = [0.0, 0.0, 1.0, 1.0]
        cp_png = png_bytes
        if box is not None:
            x1, y1, x2, y2 = (int(v) for v in box)
            crop = np.ascontiguousarray(gray[y1:y2, x1:x2])
            if crop.size:
                cp_png = _to_png_bytes(Image.fromarray(crop, mode="L"))
            if full_w and full_h:
                crop_norm = [
                    (x1 + ox) / full_w,
                    (y1 + oy) / full_h,
                    (x2 + ox) / full_w,
                    (y2 + oy) / full_h,
                ]
        return {"png": png_bytes, "crop": crop_norm, "processed_png": mp_png, "crop_png": cp_png,
                "auto_roi": None, "sign_pct": 0.0}

    def _rebuild_processed_cache(
        self,
        dataset_root: Path,
        class_preprocess: Optional[Dict[str, Dict[str, Any]]] = None,
        sample_preprocess: Optional[Dict[str, Dict[str, Dict[str, Any]]]] = None,
        global_preprocess_mode: str = PREPROCESS_MODE_AUTO_BY_LABEL,
        out_size: int = 96,
        fast_mode: bool = True,
        project_type: str = PROJECT_TYPE_IMAGE,
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
                        project_type=project_type,
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
            project_type=str(cfg.project_type),
        )
        return {
            "image_b64": base64.b64encode(result["png"]).decode("ascii"),
            "crop": result.get("crop"),
            "auto_roi": result.get("auto_roi"),
            "sign_pct": float(result.get("sign_pct") or 0.0),
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
            project_type = str(project_state.get("project_type") or "").strip().lower()
            if project_type in KNOWN_PROJECT_TYPES:
                payload["project_type"] = project_type
            # The Preview pane's gates.  They are not training parameters, but
            # they ARE deployment parameters: buildExportRuntimeConfig() stamps
            # them into model_settings.h as the device's OOD thresholds.  A
            # project that reopened them at 0.3/70/0.60/0.70 after the user
            # tuned them would re-export a device with different gates than
            # the one that is running.  Echoed verbatim — it is a flat block of
            # UI state that no normaliser owns.
            preview = project_state.get("preview")
            if isinstance(preview, dict):
                payload["preview"] = dict(preview)
        # What the archive said about itself, when the caller resolved one
        # (the open path).  Absent on a fresh payload, which is how the SPA
        # tells "just started" from "opened an old file".
        project_format = project_state.get("project_format") if isinstance(project_state, dict) else None
        if isinstance(project_format, dict):
            payload["project_format"] = dict(project_format)
        return payload

    def _preview_model_lock(self, session_id: str) -> threading.Lock:
        with self._lock:
            lock = self._preview_locks.get(session_id)
            if lock is None:
                lock = threading.Lock()
                self._preview_locks[session_id] = lock
            return lock

    def _is_sign_ocr_session(self, session_id: str) -> bool:
        """Is this session a road-sign sequence project?

        Read from the LIVE session config rather than from anything the
        client sent: which pipeline a prediction runs through is a property
        of the project, and a request that could choose its own would be a
        request that could run the wrong front-end over a model.
        """
        with self._lock:
            cfg = self._configs.get(session_id)
        return str(getattr(cfg, "project_type", "") or "").strip().lower() == PROJECT_TYPE_SIGN_OCR

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
        has_background_class = any(
            str(x).strip().lower() == "_background" for x in labels
        )
        # The setting lives in the RUN meta, not the live train_cfg: toggling
        # it after training must not reinterpret the model that was actually
        # trained.  Legacy runs (no field) that carry a `_background` class
        # used it as the only rejector — the old hard-coded device behaviour.
        bg_replace_raw = meta.get("sign_background_replace_ood") if isinstance(meta, dict) else None
        sign_background_replace_ood = (
            bool(bg_replace_raw)
            if bg_replace_raw is not None
            else bool(has_background_class)
        )
        cur = {
            "tflite_path": str(tflite_path),
            "tflite_mtime": mtime,
            "model_source": str(model_source),
            "labels": list(labels),
            "has_background_class": bool(has_background_class),
            "sign_background_replace_ood": bool(sign_background_replace_ood),
            "sign_sequence": (
                normalize_sign_sequence(meta.get("sign_sequence"))
                if isinstance(meta, dict) and meta.get("sign_sequence") else None
            ),
            "sign_sequences": (
                normalize_sign_sequences(meta.get("sign_sequences"))
                if isinstance(meta, dict) else []
            ),
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

    # ── Road-sign sequences ─────────────────────────────────────────────
    # Everything below is the sign_ocr preview.  It reuses the single-glyph
    # model — the training set is the same per-character glyphs either way,
    # and ocr_trainer's head is built from whatever classes the project
    # holds — so what differs is only how ONE frame becomes N model inputs.
    # That split lives in ocr_preprocess.preprocess_sign_sequence; this is
    # the plumbing that turns its N glyphs into N inferences and combines
    # them into the one answer a sign reader cares about.

    def _invoke_glyph_tensor(
        self, model: Dict[str, Any], session_id: str, arr: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Run one already-prepared glyph array through the model.

        Returns (softmax scores, raw device-mirror output).  A near-copy of
        the quantise-invoke-block inside `_preview_predict`, kept separate
        on purpose: that function's copy is interleaved with the multi-
        variant handling (default / crop / masked / full previews) and its
        behaviour is covered by the image-project tests, so factoring the
        two together would edit a verified path to serve a new one.  Both
        follow the device's contract — TFLite.ino feeds `gray - 128`, which
        is exactly `round(gray/255 / (1/255) - 128)`.
        """
        input_details = model["input_details"]
        output_details = model["output_details"]
        interpreter = model["interpreter"]
        q = input_details.get("quantization")
        qscale = 0.0
        qzp = 0
        if isinstance(q, (tuple, list)) and len(q) == 2:
            qscale = float(q[0] or 0.0)
            qzp = int(q[1] or 0)
        dtype = input_details.get("dtype")
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
        with self._preview_model_lock(session_id):
            interpreter.set_tensor(int(input_details.get("index")), x)
            interpreter.invoke()
            out = interpreter.get_tensor(int(output_details.get("index")))
        out = np.asarray(out).reshape((-1,))
        oq = output_details.get("quantization")
        oscale = 0.0
        ozp = 0
        if isinstance(oq, (tuple, list)) and len(oq) == 2:
            oscale = float(oq[0] or 0.0)
            ozp = int(oq[1] or 0)
        odtype = output_details.get("dtype")
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
        if abs(float(np.sum(scores)) - 1.0) > 0.1:
            scores = scores - float(np.max(scores))
            expv = np.exp(scores)
            scores = expv / float(np.sum(expv) + 1e-12)
        return np.clip(scores, 0.0, 1.0), raw_out

    def _sign_ocr_preprocess_config(self, model: Dict[str, Any], sequence: List[str]) -> Dict[str, Any]:
        """The luminance window / ink polarity the sign is read through.

        A sign's characters are printed in one ink on one background, so
        the sign has ONE preprocessing record — but the editor stores them
        per CLASS, one per character.  They are therefore consulted in
        sequence order and the first one that exists wins, which makes the
        choice deterministic and, when the user has only ever edited one of
        them (the common case, and the only one the UI encourages), the
        obvious one.  No record at all falls back to
        normalize_class_preprocess's own defaults, which is exact Otsu.
        """
        records = model.get("class_preprocess") if isinstance(model, dict) else None
        records = records if isinstance(records, dict) else {}
        for char in sequence:
            rec = records.get(str(char))
            if isinstance(rec, dict) and rec:
                return normalize_class_preprocess(rec)
        return normalize_class_preprocess(None)

    # The three things a Preview synthesis button can draw.  "match" must
    # produce a YES, "mismatch" must produce a NO, and "reversed" is the
    # direction probe: the same plate read left-to-right disagrees, read
    # right-to-left agrees.
    SIGN_SYNTH_VARIANTS = ("match", "reversed", "mismatch")

    def _sign_preview_sequence(
        self,
        session_id: str,
        chars: Any = None,
        sign_sequence: Any = None,
        reading_order: Any = None,
        sign_sequences: Any = None,
        plate_index: Any = None,
    ) -> Tuple[List[str], str, List[str], int]:
        """Resolve (plate, direction, whitelist, plate index) for a synth request.

        The SPA's editor state is the authority when it sends one; a
        background caller (auto-fill, post-training) has no request payload,
        so the project's persisted train config is the fallback.  A '?' pad
        is rejected rather than drawn: it means the user has not set a
        target, and a plate of placeholder glyphs would answer a different
        question than the one the red NO implies.
        """
        from ocr_dataset import TEMPLATE_SIGN_CHARS

        with self._lock:
            cfg = self._configs.get(session_id)
        if cfg is None:
            raise RuntimeError("missing config")
        saved = self._project_train_cfg(Path(cfg.dataset_root))
        seqs = normalize_sign_sequences(sign_sequences)
        if not seqs:
            seqs = normalize_sign_sequences(sign_sequence)
        if not seqs and chars is not None:
            seqs = normalize_sign_sequences(chars)
        if not seqs:
            seqs = normalize_sign_sequences(saved.get("sign_sequences"))
        if not seqs:
            seqs = normalize_sign_sequences(saved.get("sign_sequence"))
        if not seqs:
            raise ValueError(
                "set the project's 3-character target sign before synthesizing a preview"
            )
        try:
            idx = int(plate_index) if plate_index is not None else 0
        except (TypeError, ValueError):
            idx = 0
        idx = max(0, min(len(seqs) - 1, idx))
        seq = normalize_sign_sequence(seqs[idx])
        bad = [c for c in seq if len(str(c)) != 1 or str(c).upper() not in TEMPLATE_SIGN_CHARS]
        if bad:
            raise ValueError(
                "set the project's 3-character target sign before synthesizing a preview"
            )
        seq = [str(c).upper() for c in seq]
        order = str(reading_order or saved.get("reading_order") or "ltr").strip().lower()
        if order not in ("ltr", "rtl"):
            order = "ltr"
        return seq, order, list(seqs), idx

    def _synthesize_sign_preview_png(
        self,
        session_id: str,
        variant: str = "match",
        chars: Any = None,
        sign_sequence: Any = None,
        reading_order: Any = None,
        seed: Optional[int] = None,
        alt_char: Any = None,
        sign_sequences: Any = None,
        plate_index: Any = None,
    ) -> Dict[str, Any]:
        """Draw a composite 3-character sign and make it the active preview.

        The ORDER is the whole test matrix:
          match     the target, laid out the way the project reads it
          reversed  the target, laid out the opposite way (pixels are
                    reordered, never mirrored — the same rule the live
                    pipeline follows)
          mismatch  the target with its LAST character replaced by one the
                    target does not contain

        Storing the bytes in `_preview_uploads` is deliberate: the Preview
        card's upload source, its Dark/Lum re-predict (`reuse=1`) and the
        drawn slots all then read the same stored frame, with no second
        "current synth image" concept to keep in step.
        """
        from ocr_dataset import (
            emnist_cache_dir,
            load_emnist_template_bank,
            synthesize_sign_preview_png,
        )

        with self._lock:
            cfg = self._configs.get(session_id)
        if cfg is None:
            raise RuntimeError("missing config")
        if not self._is_sign_ocr_session(session_id):
            raise ValueError("sign synthesis is only available in a sign_ocr project")
        seq, order, seqs, plate_idx = self._sign_preview_sequence(
            session_id,
            chars=chars,
            sign_sequence=sign_sequence,
            reading_order=reading_order,
            sign_sequences=sign_sequences,
            plate_index=plate_index,
        )
        kind = str(variant or "match").strip().lower()
        if kind not in self.SIGN_SYNTH_VARIANTS:
            raise ValueError(f"unknown synth variant '{variant}'")
        if kind == "match":
            physical = list(seq) if order == "ltr" else list(reversed(seq))
        elif kind == "reversed":
            physical = list(reversed(seq))
        else:
            # A character the target does not use.  The caller may force one
            # (the endpoint retries until the model actually reads it as
            # wrong — an unseen glyph can be misread as the target, which
            # would make a "mismatch" plate show a false YES).  The default
            # starts at 5, clear of the 0/O and 1/I confusions.
            forced = str(alt_char or "").strip().upper()[:1]
            if forced and forced in seq:
                forced = ""
            alt = forced or next(
                (c for c in "56789ABCDEFGHIJKLMNOPQRSTUVWXYZ" if c not in seq), None
            )
            if alt is None:
                raise ValueError("no replacement character available")
            replaced = list(seq)
            replaced[-1] = alt
            physical = replaced if order == "ltr" else list(reversed(replaced))

        bank = load_emnist_template_bank(cache_root=emnist_cache_dir(app_data_dir()))
        png = synthesize_sign_preview_png(
            physical,
            dataset_root=Path(cfg.dataset_root),
            bank=bank,
            seed=seed,
        )
        label = "".join(seq)
        filename = f"synth_{label}_{kind}.png"
        with self._lock:
            self._preview_uploads[session_id] = {
                "bytes": png,
                "filename": filename,
                "synthetic": True,
            }
        return {
            "png": png,
            "filename": filename,
            "variant": kind,
            "chars": physical,
            "sign_sequence": list(seq),
            "sign_sequences": list(seqs),
            "plate_index": int(plate_idx),
            "reading_order": order,
        }

    def _maybe_autostore_synth_preview(
        self,
        session_id: str,
        *,
        sign_sequence: Any = None,
        sign_sequences: Any = None,
        reading_order: Any = None,
        replace_synthetic: bool = False,
    ) -> bool:
        """Prepare a ready-to-preview sign image for a sign_ocr project.

        Called after auto-fill seeds a sign project and after training
        finishes.  A REAL upload is never replaced — the user put it there
        on purpose — but a previous synthetic frame is, because the target
        or the model may have changed since it was drawn.  Failures are
        swallowed: this is a convenience, and a missing preview must never
        turn seeding or training into an error.
        """
        try:
            with self._lock:
                stored = dict(self._preview_uploads.get(session_id) or {})
            if stored.get("bytes") and not (
                replace_synthetic and bool(stored.get("synthetic"))
            ):
                return False
            self._synthesize_sign_preview_png(
                session_id,
                variant="match",
                sign_sequence=sign_sequence,
                sign_sequences=sign_sequences,
                reading_order=reading_order,
            )
            return True
        except Exception:
            return False

    def _preview_predict_sign_ocr(
        self,
        session_id: str,
        png: bytes,
        sign_sequence: Any = None,
        sign_sequences: Any = None,
        reading_order: Any = None,
        bg_dark_thresh: Optional[int] = None,
        bg_lum_thresh: Optional[int] = None,
        match_min_prob: Optional[float] = None,
        ood_sign_pct_min: Optional[float] = None,
        ood_sign_pct_max: Optional[float] = None,
        ood_max_prob_min: Optional[float] = None,
        ood_entropy_ratio_max: Optional[float] = None,
        source: str = "",
    ) -> Dict[str, Any]:
        """Read a whole road sign: locate, split, one inference per glyph.

        Answers the only question a 3-character sign project has — is this
        the sign I trained? — as a single boolean, alongside the three
        per-slot readings it was derived from, so a NO is explainable
        rather than just red.

        The rule is deliberately unforgiving: ALL THREE slots must name
        their own character, none may come back as the reject class, and
        each must clear `match_min_prob`.  A sign reader that says YES when
        two of three characters matched is worse than one that says no.
        """
        order = str(reading_order or "").strip().lower()
        if order not in ("ltr", "rtl"):
            order = "ltr"
        # Same lazy import, same reason as _classes_add_named: ocr_dataset
        # pulls in the OCR training stack, and this module must stay usable
        # without it.
        from ocr_dataset import BACKGROUND_CLASS_NAME

        model = self._preview_load_model(session_id=session_id)
        labels: List[str] = list(model.get("labels") or [])
        # Allowed plates: the live editor list wins, then the run meta (so a
        # reopened project uses the whitelist it was trained against), then
        # the legacy single target.  seq[0] is the primary plate and only
        # supplies slot COUNT / labels; the actual matching below checks every
        # plate in `seqs`.
        seqs = normalize_sign_sequences(sign_sequences)
        if not seqs:
            seqs = normalize_sign_sequences(sign_sequence)
        if not seqs:
            seqs = normalize_sign_sequences(model.get("sign_sequences"))
        if not seqs:
            seqs = normalize_sign_sequences(model.get("sign_sequence"))
        seq = normalize_sign_sequence(seqs[0]) if seqs else normalize_sign_sequence(sign_sequence)
        has_background_class = any(
            str(x).strip().lower() == BACKGROUND_CLASS_NAME.lower()
            for x in labels
        )
        # The run's own rejection policy.  When a `_background` class is the
        # only rejector, Layers 1-3 are bypassed (the device does the same);
        # otherwise the standard OOD gates compose with the per-slot match
        # rule and the probability floor.  A model dict with no explicit flag
        # predates the toggle: having a background class is what old runs
        # used, so preserve that rather than silently changing their verdicts.
        raw_bg_replace = model.get("sign_background_replace_ood")
        background_is_only_rejector = (
            bool(raw_bg_replace)
            if raw_bg_replace is not None
            else bool(has_background_class)
        ) and has_background_class
        shape_raw = model["input_details"].get("shape")
        shape = tuple(int(x) for x in (shape_raw if shape_raw is not None else []))
        if len(shape) != 4:
            raise RuntimeError("unsupported input shape")
        _, h, w, c = shape
        channels = int(c) if int(c) > 0 else 1

        SIGN_PCT_MIN = float(ood_sign_pct_min if ood_sign_pct_min is not None else 0.3)
        SIGN_PCT_MAX = float(ood_sign_pct_max if ood_sign_pct_max is not None else 70.0)
        MAX_PROB_MIN = float(ood_max_prob_min if ood_max_prob_min is not None else 0.60)
        ENTROPY_RATIO_MAX = float(
            ood_entropy_ratio_max if ood_entropy_ratio_max is not None else 0.70
        )
        SIGN_PCT_MIN = max(0.0, min(100.0, SIGN_PCT_MIN))
        SIGN_PCT_MAX = max(0.0, min(100.0, SIGN_PCT_MAX))
        MAX_PROB_MIN = max(0.0, min(1.0, MAX_PROB_MIN))
        ENTROPY_RATIO_MAX = max(0.0, min(1.0, ENTROPY_RATIO_MAX))
        if SIGN_PCT_MIN > SIGN_PCT_MAX:
            SIGN_PCT_MIN, SIGN_PCT_MAX = SIGN_PCT_MAX, SIGN_PCT_MIN

        # The luminance window.  A caller-supplied value (the live preview's
        # Dark/Lum sliders) wins; otherwise the sign's own stored record;
        # otherwise the untouched defaults, which mean exact Otsu.  Same
        # one-LSB-inward nudge as _preprocess_image_png_ocr, for the same
        # reason: the sliders are strict bounds and binarize_ink is not.
        config = self._sign_ocr_preprocess_config(model, seq)
        if bg_dark_thresh is not None:
            config["bg_dark_thresh"] = int(bg_dark_thresh)
        if bg_lum_thresh is not None:
            config["bg_lum_thresh"] = int(bg_lum_thresh)
        config = normalize_class_preprocess(config)
        polarity = int(config.get("ink_polarity", INK_DARK))
        convention = str(config.get("glyph_convention", NORM_CANON))
        bg_dark = int(config.get("bg_dark_thresh", 0) or 0)
        bg_lum = int(config.get("bg_lum_thresh", 100) or 100)
        thresholds_adjusted = (bg_dark != 0) or (bg_lum != 100)
        pinned = max(1, min(254, bg_lum - 1)) if thresholds_adjusted else None
        # Dark=0 means "no lower bound": nudging it to 1 would drop pure
        # black glyph pixels and fragment a thin '1'.  Keep 0 as the
        # inclusive lower edge of the user's window.
        floor = (0 if bg_dark <= 0 else min(255, bg_dark + 1)) if thresholds_adjusted else 0

        src_img = Image.open(_bytes_io(png)).convert("L")
        gray = np.asarray(src_img, dtype=np.uint8)
        full_h, full_w = gray.shape[:2]

        # ── Stage 1: locate the sign plate on the NATIVE frame ────────────
        # The device streams 160x160 (or 96x96); the glyph model is 28x28.
        # Never resize the scene to the glyph size before this point: the
        # split needs the original pixel resolution.  Manual ROI wins when
        # the sign is pinned; otherwise the bright inner plate is detected
        # so the desk/wheels/frame never enter the vertical projection.
        manual_roi = None
        if str(config.get("mode") or "") == PREPROCESS_MODE_MANUAL_ROI:
            manual_roi = config.get("manual_roi")
        if manual_roi is None and str(model.get("preprocess_mode") or "") == PREPROCESS_MODE_MANUAL_ROI:
            manual_roi = model.get("manual_roi")
        roi_info = find_sign_roi(
            gray,
            manual_roi=manual_roi,
            dark_thr=floor if thresholds_adjusted else None,
            lum_thr=pinned if thresholds_adjusted else None,
        )
        rx1, ry1, rx2, ry2 = (int(v) for v in roi_info["box"])
        rx1 = max(0, min(full_w - 1, rx1))
        ry1 = max(0, min(full_h - 1, ry1))
        rx2 = max(rx1 + 1, min(full_w, rx2))
        ry2 = max(ry1 + 1, min(full_h, ry2))
        roi_gray = np.ascontiguousarray(gray[ry1:ry2, rx1:rx2])
        origin_x, origin_y = rx1, ry1

        # ── Stage 2: threshold + split 3 glyphs ───────────────────────────
        # NO second border-cleaning pass here.  `roi_gray` is already the
        # sign interior returned by find_sign_roi; running clean_sign_ink on
        # the cropped ROI treats any glyph near the box edge as "border
        # clutter" and erases it (the `'A'` / broken-`'1'` bug).  The ROI
        # finder strips the outer frame, so the exact `roi_box` content is
        # what the split must see.  All slot crops are clamped inside
        # roi_gray, so no pixel outside `roi_box` can reach a slot.
        reading = preprocess_sign_sequence(
            roi_gray,
            sign_sequence=seq,
            reading_order=order,
            out_size=int(w),
            polarity=polarity,
            apply_segmentation=True,
            convention=convention,
            thresh=pinned,
            dark_thr=floor,
            clean_border=False,
        )
        # OOD Layer 1's input: the ink share of the intact sign ROI.  Same
        # number the image project calls sign_pct, so spmin/spmax mean the
        # same thing in both workspaces.
        ink = reading.get("ink")
        ink_arr = np.asarray(ink, dtype=bool) if ink is not None else None
        sign_pct = (
            float(ink_arr.mean() * 100.0)
            if ink_arr is not None and ink_arr.size
            else 0.0
        )
        sign_pct_in_band = (
            background_is_only_rejector
            or (SIGN_PCT_MIN <= sign_pct <= SIGN_PCT_MAX)
        )

        def _to_full_box(box: Any) -> Any:
            """Slot/sign box from the ROI-local frame back to frame pixels."""
            if not isinstance(box, (list, tuple)) or len(box) != 4:
                return None
            return [
                int(box[0]) + origin_x,
                int(box[1]) + origin_y,
                int(box[2]) + origin_x,
                int(box[3]) + origin_y,
            ]

        SLOT_MIN = max(0.0, min(1.0, float(match_min_prob if match_min_prob is not None else 0.60)))
        slots: List[Dict[str, Any]] = []
        slot_pngs: List[bytes] = []
        all_match = bool(labels)
        ood_reasons: List[str] = []
        for i, slot in enumerate(reading.get("slots") or []):
            glyph = np.asarray(slot.get("glyph"), dtype=np.uint8)
            target = str(slot.get("char") or "")
            label = ""
            prob = 0.0
            raw_out = None
            if labels and glyph.size:
                arr = (glyph.astype(np.float32) / 255.0).reshape((int(h), int(w), channels))
                try:
                    _scores, raw_out = self._invoke_glyph_tensor(model, session_id, arr)
                except Exception:
                    raw_out = None
            if raw_out is not None and raw_out.size:
                total = int(np.sum(raw_out))
                if total <= 0:
                    total = 1
                top_i = int(np.argmax(raw_out))
                # Device-exact confidence (raw_best / sum(raw)) — the same
                # number TFLite.ino's OOD gate and _preview_predict use, so
                # a threshold means one thing across the whole app.
                prob = float(np.float32(np.max(raw_out)) / np.float32(total))
                if 0 <= top_i < len(labels):
                    label = str(labels[top_i])
            is_background = label.startswith("_") or label == BACKGROUND_CLASS_NAME
            slot_max_prob = 0.0
            slot_entropy_ratio = 1.0
            slot_ood = False
            slot_entropy_only = False
            slot_ood_reason = ""
            if raw_out is not None and raw_out.size:
                slot_max_prob, slot_entropy_ratio = _raw_ood_metrics(raw_out)
                if not background_is_only_rejector:
                    # Standard OOD Layers 1-3, mirroring _preview_predict and
                    # the exported device thresholds.
                    if not sign_pct_in_band:
                        slot_ood = True
                        slot_ood_reason = (
                            f"sign_pct={sign_pct:.1f}% outside "
                            f"[{SIGN_PCT_MIN:.1f},{SIGN_PCT_MAX:.1f}]"
                        )
                    elif slot_max_prob < MAX_PROB_MIN:
                        slot_ood = True
                        slot_ood_reason = (
                            f"max_prob={slot_max_prob:.3f} < {MAX_PROB_MIN:.2f}"
                        )
                    elif slot_entropy_ratio > ENTROPY_RATIO_MAX:
                        slot_ood = True
                        slot_entropy_only = True
                        slot_ood_reason = (
                            f"entropy_ratio={slot_entropy_ratio:.3f} > "
                            f"{ENTROPY_RATIO_MAX:.2f}"
                        )
            # A slot is READABLE independent of which plate we compare it to:
            # ink, a real (non-background) label, the confidence floor, and
            # the OOD gates.  The entropy gate alone is NOT decisive: the
            # plate loop below accepts an entropy-flagged slot when its top
            # class is that plate's target character and prob >= SLOT_MIN.
            # A correct 3-way reading used to be reported as NO solely
            # because the model's class distribution can push entropy above
            # 0.70 even when every top class is right.
            accepted = bool(
                labels
                and not is_background
                and label
                and prob >= SLOT_MIN
                and bool(slot.get("ok"))
                and (not slot_ood or slot_entropy_only)
            )
            full_box = _to_full_box(slot.get("box"))
            full_box_norm = None
            if full_box is not None and full_w > 0 and full_h > 0:
                full_box_norm = [
                    float(full_box[0]) / float(full_w),
                    float(full_box[1]) / float(full_h),
                    float(full_box[2]) / float(full_w),
                    float(full_box[3]) / float(full_h),
                ]
            slots.append(
                {
                    "index": int(i),
                    "char": str(target),
                    "label": label,
                    "prob": float(prob),
                    "accepted": bool(accepted),
                    "match": False,
                    "background": bool(is_background),
                    "ood": bool(slot_ood),
                    "ood_reason": str(slot_ood_reason),
                    "_entropy_only": bool(slot_entropy_only),
                    "max_prob": float(slot_max_prob),
                    "entropy_ratio": float(slot_entropy_ratio),
                    "empty": bool(slot.get("empty", True)),
                    "box": full_box,
                    "box_norm": full_box_norm,
                    "roi_box_norm": slot.get("box_norm"),
                }
            )
            display = np.asarray(slot.get("display", glyph), dtype=np.uint8)
            slot_pngs.append(_to_png_bytes(Image.fromarray(display, mode="L")))

        # ── compare the three readings against EVERY allowed plate ───────
        # An exact 3-for-3 plate wins immediately.  If none is exact, keep the
        # closest one (most slots hit) for the WANT headers so a NO is still
        # explainable against the plate the user was testing.
        all_match = False
        chosen_index = -1
        chosen_hits = [False] * len(slots)
        if labels and seqs and len(slots) >= 3:
            best_score = -1
            for pi, plate in enumerate(seqs):
                hits = [
                    bool(slots[s].get("accepted"))
                    and str(slots[s].get("label") or "").upper() == plate[s].upper()
                    for s in range(3)
                ]
                score = sum(1 for h in hits if h)
                if score == 3:
                    chosen_index = pi
                    chosen_hits = hits
                    all_match = True
                    break
                if score > best_score:
                    best_score = score
                    chosen_index = pi
                    chosen_hits = hits
        if chosen_index >= 0:
            chosen_plate = list(seqs[chosen_index])
        else:
            chosen_plate = list(seq) if len(seq) >= 3 else normalize_sign_sequence(seq)
        for s_idx, slot_out in enumerate(slots):
            slot_out["char"] = (
                str(chosen_plate[s_idx]) if s_idx < len(chosen_plate)
                else SIGN_SEQUENCE_PAD
            )
            slot_out["match"] = bool(
                all_match or (s_idx < len(chosen_hits) and chosen_hits[s_idx])
            )

        # Rebuild the OOD failure list from the FINAL per-slot verdict.  An
        # entropy-only flag that was overridden by a correct target match is
        # not a failure; reporting it on a YES was the contradictory banner
        # ("each at 60% or better ... OOD: slot 2: entropy_ratio").
        ood_reasons = []
        for s_idx, slot_out in enumerate(slots):
            if slot_out.get("match") and slot_out.get("_entropy_only"):
                slot_out["ood"] = False
                slot_out["ood_reason"] = ""
            slot_out.pop("_entropy_only", None)
            if slot_out.get("ood") and not slot_out.get("match"):
                reason = str(slot_out.get("ood_reason") or "")
                ood_reasons.append(f"slot {s_idx + 1}: {reason}")

        if all_match:
            plate_text = "".join(chosen_plate)
            if len(seqs) > 1:
                match_text = (
                    f"{SIGN_MATCH_TEXT} · Plate #{chosen_index + 1} "
                    f"({plate_text})"
                )
            else:
                match_text = f"{SIGN_MATCH_TEXT} ({plate_text})"
        else:
            match_text = SIGN_NOMATCH_TEXT
        # Per-slot readings doubled as the bar labels: the preview's output
        # pane renders a distribution for every other source, and for a sign
        # the interesting distribution IS the three slots.
        slot_labels = [f"{s['char']} → {s['label'] or '—'}" for s in slots]
        slot_probs = [float(s["prob"]) for s in slots]

        sign_box = _to_full_box(reading.get("sign_box"))
        # The ROI overlay and the class-edit preview both read `crop` as
        # FULL-frame fractions.  The sign board ROI is the meaningful box
        # here: `sign_box` is only the union of the three glyphs.
        sign_box_norm = None
        if sign_box is not None and full_w > 0 and full_h > 0:
            sign_box_norm = [
                float(sign_box[0]) / float(full_w),
                float(sign_box[1]) / float(full_h),
                float(sign_box[2]) / float(full_w),
                float(sign_box[3]) / float(full_h),
            ]
        roi_box = [int(rx1), int(ry1), int(rx2), int(ry2)]
        roi_box_norm = [
            float(rx1) / float(full_w) if full_w > 0 else 0.0,
            float(ry1) / float(full_h) if full_h > 0 else 0.0,
            float(rx2) / float(full_w) if full_w > 0 else 1.0,
            float(ry2) / float(full_h) if full_h > 0 else 1.0,
        ]
        # User-facing threshold view: inside the Dark..Lum window is BLACK
        # ink; everything else is PURE WHITE.  This is what Orig / both
        # toggles show (with the ROI box on top); the model tensor is built
        # separately and keeps its own polarity.
        inside_full = (gray >= bg_dark) & (gray <= bg_lum)
        full_display = np.where(inside_full, 0, 255).astype(np.uint8)
        # ROI CHECKED must show the EXACT pixels inside the green box drawn
        # on the full view — not a re-cleaned mask.  Same array, same slice:
        # whatever is visible inside the box unchecked is visible checked.
        processed_display = full_display[ry1:ry2, rx1:rx2].copy()
        full_png = _to_png_bytes(Image.fromarray(full_display, mode="L"))
        processed_png = _to_png_bytes(Image.fromarray(processed_display, mode="L"))
        raw_crop_png = _to_png_bytes(Image.fromarray(roi_gray, mode="L"))

        return {
            "labels": slot_labels,
            "probs": slot_probs,
            "top_label": match_text,
            "top_prob": float(min((s["prob"] for s in slots), default=0.0)),
            "crop": roi_box_norm,
            "roi_box": roi_box,
            "roi_box_norm": roi_box_norm,
            "roi_method": str(roi_info.get("method") or ""),
            "model_source": str(model.get("model_source") or "latest"),
            "processed_image_b64": base64.b64encode(processed_png).decode("ascii"),
            "crop_image_b64": base64.b64encode(raw_crop_png).decode("ascii"),
            "full_image_b64": base64.b64encode(full_png).decode("ascii"),
            "processed_variant": "sign-threshold",
            "ood": bool(ood_reasons),
            "ood_reason": " | ".join(ood_reasons),
            "sign_pct": float(sign_pct),
            "max_prob": float(min((s["prob"] for s in slots), default=0.0)),
            "entropy_ratio": float(max((s["entropy_ratio"] for s in slots), default=0.0)),
            # ── the sign_ocr payload ────────────────────────────────────
            "sign_ocr": True,
            "sign_sequence": list(chosen_plate),
            "sign_sequences": list(seqs),
            "matched_sequence": "".join(chosen_plate) if all_match else "",
            "matched_index": int(chosen_index) if all_match else -1,
            "closest_sequence": "".join(chosen_plate) if (not all_match and seqs) else "",
            "plate_count": int(len(seqs)),
            "reading_order": order,
            "slots": slots,
            "slot_images_b64": [base64.b64encode(p).decode("ascii") for p in slot_pngs],
            "match": bool(all_match),
            "match_text": match_text,
            "match_min_prob": float(SLOT_MIN),
            "sign_box": sign_box,
        }

    def _sign_ocr_composite(
        self, gray_local: np.ndarray, reading: Dict[str, Any], origin_x: int, origin_y: int
    ) -> bytes:
        """One image that shows the split: the located sign over its slots.

        Purely a rendering aid — nothing downstream reads it — but it is the
        difference between "NO MATCH" and "NO MATCH, and here is the second
        crop it was looking at", which is the whole diagnostic value of
        segmenting a sign in the first place.
        """
        rows: List[np.ndarray] = []
        try:
            sign_box = reading.get("sign_box")
            if isinstance(sign_box, (list, tuple)) and len(sign_box) == 4:
                x1, y1, x2, y2 = (int(v) for v in sign_box)
                crop = gray_local[y1:y2, x1:x2]
                if crop.size:
                    # Scale to a readable strip: the source may be a 28 px
                    # crop or a 640 px frame, and the pane is a fixed width.
                    scale = max(1, int(round(160.0 / max(1, crop.shape[1]))))
                    crop = np.repeat(np.repeat(crop, scale, axis=0), scale, axis=1)
                    rows.append(np.ascontiguousarray(crop))
            glyph_row = []
            for slot in reading.get("slots") or []:
                g = np.asarray(slot.get("glyph"), dtype=np.uint8)
                if g.size:
                    # 4x with a 1 px separator, so the three slots read as
                    # three crops rather than one blurred strip.
                    glyph_row.append(np.repeat(np.repeat(g, 4, axis=0), 4, axis=1))
                    glyph_row.append(np.full((g.shape[0] * 4, 3), 200, dtype=np.uint8))
            if glyph_row:
                body = np.hstack(glyph_row[:-1])
                width = max(x.shape[1] for x in rows + [body])
                rows.append(np.full((6, width), 160, dtype=np.uint8))
                rows.append(body)
            if not rows:
                rows = [np.zeros((28, 28), dtype=np.uint8)]
            height = sum(r.shape[0] for r in rows)
            width = max(r.shape[1] for r in rows)
            canvas = np.full((height, width), 255, dtype=np.uint8)
            y = 0
            for r in rows:
                canvas[y : y + r.shape[0], : r.shape[1]] = r
                y += r.shape[0]
            return _to_png_bytes(Image.fromarray(canvas, mode="L"))
        except Exception:
            return _to_png_bytes(Image.fromarray(np.asarray(gray_local, dtype=np.uint8), mode="L"))

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

            project_type = str(
                getattr(sess_cfg, "project_type", PROJECT_TYPE_IMAGE)
                or PROJECT_TYPE_IMAGE
            ).strip().lower()

            # ── sign_ocr: the target sequence IS the class list ───────────
            # A stale "Class 1"/"Class 2" directory is not inert: the EMNIST
            # mapper reads its trailing digit and fills it, which turns it
            # into a real model label (and takes the samples away from the
            # target class).  Enforce the sequence BEFORE any cache rebuild,
            # so the mixed set, labels.txt and the model output indices all
            # come from exactly the characters the sign prints.
            saved_train_cfg = self._project_train_cfg(sess_cfg.dataset_root)
            sign_seqs_src = cfg_dict.get("sign_sequences")
            if sign_seqs_src in (None, "", []):
                sign_seqs_src = saved_train_cfg.get("sign_sequences")
            sign_seq_src = cfg_dict.get("sign_sequence")
            if sign_seq_src in (None, "", []):
                sign_seq_src = saved_train_cfg.get("sign_sequence")
            effective_sign_seq: Optional[List[str]] = None
            effective_sign_sequences: Optional[List[str]] = None
            if project_type == PROJECT_TYPE_SIGN_OCR:
                # The whitelist is the authority; the legacy single target is
                # folded in when an older client sends only sign_sequence.
                effective_sign_sequences = normalize_sign_sequences(sign_seqs_src)
                if not effective_sign_sequences:
                    effective_sign_sequences = normalize_sign_sequences(sign_seq_src)
                if not effective_sign_sequences:
                    raise ValueError(
                        "Add at least one 3-character license plate before "
                        "training — its characters are the model's classes."
                    )
                sign_chars = self._sign_sequences_classes(effective_sign_sequences)
                self._enforce_sign_classes(session_id, effective_sign_sequences)
                effective_sign_seq = normalize_sign_sequence(
                    effective_sign_sequences[0]
                )
                # Feature toggle: _background is the ONLY rejector when this
                # is on; when off the model trains on the target characters
                # alone and the standard OOD gates do the rejecting.  The
                # default for a Car License project is OFF.
                raw_bg_replace = cfg_dict.get("sign_background_replace_ood")
                sign_background_replace_ood = (
                    _is_truthy(raw_bg_replace)
                    if raw_bg_replace is not None
                    else False
                )
            else:
                sign_background_replace_ood = False
            background_enabled = (
                sign_background_replace_ood
                if project_type == PROJECT_TYPE_SIGN_OCR
                else _is_truthy(cfg_dict.get("ocr_background_enabled", True))
            )

            project_classes = self._classes_load(sess_cfg.dataset_root)
            project_class_preprocess = self._class_preprocess_load(sess_cfg.dataset_root)
            project_sample_preprocess = self._sample_preprocess_load(sess_cfg.dataset_root)

            raw_class_preprocess = cfg_dict.get("class_preprocess")
            if raw_class_preprocess is None or (isinstance(raw_class_preprocess, dict) and not raw_class_preprocess):
                raw_class_preprocess = project_class_preprocess
            raw_sample_preprocess = cfg_dict.get("sample_preprocess")
            if raw_sample_preprocess is None or (isinstance(raw_sample_preprocess, dict) and not raw_sample_preprocess):
                raw_sample_preprocess = project_sample_preprocess

            # The auto-fill flag is project state, not a hyperparameter, but
            # it rides in TrainConfig and the trainer writes that config into
            # the run.  An older client (or a page whose train_cfg was
            # restored without it) must not silently flip it off in the run
            # it produces: an explicit payload value wins, otherwise the
            # project's persisted choice does.
            auto_seed_samples = (
                _is_truthy(cfg_dict.get("auto_seed_emnist_samples"))
                if cfg_dict.get("auto_seed_emnist_samples") is not None
                else self._auto_seed_emnist_enabled(sess_cfg.dataset_root)
            )

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
                # The session owns the project type; a stray cfg_dict must not
                # be able to train an OCR workspace with the sign transform.
                project_type=project_type,
                # OCR: dataset mixing + transfer learning.  Read only by the
                # OCR strategy, so an image project is unaffected by their
                # presence (and by their absence).
                ocr_emnist_enabled=bool(cfg_dict.get("ocr_emnist_enabled", True)),
                ocr_emnist_split=str(cfg_dict.get("ocr_emnist_split") or "balanced"),
                ocr_emnist_per_class=int(cfg_dict.get("ocr_emnist_per_class") or 200),
                # sign_ocr derives this from the explicit "use _background as
                # the only rejector" toggle (default OFF); OCR keeps its own
                # background-class checkbox.
                ocr_background_enabled=background_enabled,
                ocr_background_samples=int(cfg_dict.get("ocr_background_samples") or 0),
                sign_background_replace_ood=sign_background_replace_ood,
                ocr_transfer_learning=bool(cfg_dict.get("ocr_transfer_learning", True)),
                ocr_pretrain_per_class=int(cfg_dict.get("ocr_pretrain_per_class") or 1200),
                ocr_pretrain_epochs=int(cfg_dict.get("ocr_pretrain_epochs") or 4),
                ocr_head_epochs=int(cfg_dict.get("ocr_head_epochs") or 12),
                ocr_finetune_epochs=int(cfg_dict.get("ocr_finetune_epochs") or 25),
                ocr_head_lr=float(cfg_dict.get("ocr_head_lr") or 0.0015),
                ocr_finetune_lr=float(cfg_dict.get("ocr_finetune_lr") or 0.0002),
                ocr_unfreeze_blocks=int(cfg_dict.get("ocr_unfreeze_blocks") or 1),
                # sign_ocr: the target label the exported device sketch will
                # verify each slot against.  Recorded in train_config.json so
                # an export (or a reopened project) can recover what THIS
                # model was trained to read instead of guessing from the
                # current editor state.
                sign_sequence=effective_sign_seq,
                sign_sequences=effective_sign_sequences,
                reading_order=str(cfg_dict.get("reading_order") or "ltr"),
                auto_seed_emnist_samples=bool(auto_seed_samples),
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
                project_type=str(cfg.project_type),
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
                "project_type": str(cfg.project_type),
                "color_mode": str(cfg.color_mode),
                # Rejection policy actually used by this run.  The sign
                # preview and the exporter read these from the run meta so a
                # later toggle change cannot retroactively reinterpret a
                # model that was trained differently.
                "ocr_background_enabled": bool(cfg.ocr_background_enabled),
                "sign_background_replace_ood": bool(cfg.sign_background_replace_ood),
                "sign_sequence": list(cfg.sign_sequence) if cfg.sign_sequence else None,
                "sign_sequences": list(cfg.sign_sequences) if cfg.sign_sequences else [],
                "preprocess_mode": str(cfg.preprocess_mode),
                "crop_mode": str(cfg.crop_mode or CROP_MODE_AUTO_SEARCH),
                "manual_roi": list(cfg.manual_roi) if cfg.manual_roi is not None else None,
                "class_preprocess": normalize_class_preprocess_map(cfg.class_preprocess),
                "sample_preprocess": normalize_sample_preprocess_map(cfg.sample_preprocess),
                "trained_dataset_dir": str(processed_dataset_dir),
                # Strategy-supplied manifest.  For OCR this is the EMNIST /
                # background mix actually used, so a run can be explained
                # after the fact without re-deriving it from the filenames.
                # "ocr_mix", not "sign_ocr_mix": the sign project runs the same OCR
                # strategy, so the manifest it supplies is the same EMNIST /
                # background mix.  Keyed on the STRATEGY family, because that is
                # what produced it — a name that varied by project type would
                # make every reader of this field branch on it too.
                "ocr_mix": dict(result.meta) if is_glyph_project(cfg.project_type) else {},
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
            # The target the model was just trained against is project
            # state now: persist it so a later auto-fill (which carries no
            # request payload) can still draw a matching preview sign.
            if str(cfg.project_type) == PROJECT_TYPE_SIGN_OCR:
                try:
                    self._save_train_cfg(
                        session_id,
                        {
                            "sign_sequence": cfg.sign_sequence,
                            "sign_sequences": cfg.sign_sequences,
                            "reading_order": cfg.reading_order,
                            "sign_background_replace_ood": bool(cfg.sign_background_replace_ood),
                            "ocr_background_enabled": bool(cfg.ocr_background_enabled),
                            "auto_seed_emnist_samples": bool(cfg.auto_seed_emnist_samples),
                        },
                    )
                except Exception:
                    pass
                # A freshly trained sign project should be testable with one
                # click and no photo: replace any PREVIOUS synthetic frame
                # (the model and maybe the target changed), never a real
                # upload.
                self._maybe_autostore_synth_preview(
                    session_id,
                    sign_sequence=cfg.sign_sequence,
                    sign_sequences=cfg.sign_sequences,
                    reading_order=cfg.reading_order,
                    replace_synthetic=True,
                )
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
                out_side = _preprocess_frame_side(
                    getattr(cfg, "project_type", ""), getattr(cfg, "img_size", 96)
                )
                raw_side = int(cfg.serial_frame_side)
                # preview_png is what Preview/inference sees: box-averaged to
                # the preprocessing side (96), exactly like the MCU.  The
                # capture PNG stays at the NATIVE stream side (160) so a
                # saved sample keeps the resolution the user collected; the
                # training cache downsizes it later.
                preview_png = _raw96_to_png(raw, crop_box=cfg.crop_box, frame_side=raw_side, out_side=int(out_side))
                capture_png = (
                    preview_png
                    if int(out_side) == raw_side
                    else _raw96_to_png(raw, crop_box=cfg.crop_box, frame_side=raw_side, out_side=raw_side)
                )
                self._live_set(key, preview_png=preview_png, capture_png=capture_png, error="")
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
                    # 1:1 on ingest for the glyph projects (see the helper):
                    # the stage is square and the device frame is square.
                    frame = _center_crop_webcam_square(
                        frame, getattr(cfg, "project_type", "")
                    )
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
                    # capture_png is what /live/capture and _record_from_live
                    # write to disk, and what the Manual ROI editor shows as
                    # "Original Sample".  Keep it at camera resolution — the
                    # old `resize((img_size, img_size))` here saved 28x28
                    # thumbnails for an OCR project, so the user was dragging
                    # a ROI on a blown-up thumbnail and the crop had nothing
                    # left to select.  _cache_one_sample still stores the
                    # training tensor at img_size, after the crop.
                    img = _cap_capture_resolution(img)
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
                self._cache_one_sample(cfg.dataset_root, class_name, p, img_size=int(cfg.img_size), project_type=str(cfg.project_type))
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
                frame = _center_crop_webcam_square(
                    frame, getattr(cfg, "project_type", "")
                )
                img = Image.fromarray(frame).convert("RGB")
                if cfg.crop_box is not None:
                    x1, y1, x2, y2 = cfg.crop_box
                    img = img.crop((x1, y1, x2, y2))
                # Same rule as the live worker above: save at camera
                # resolution, let the pipeline downsample after the crop.
                img = _cap_capture_resolution(img)
                png = _to_png_bytes(img)
                p = _save_png(cfg.dataset_root, class_name, png)
                self._cache_one_sample(cfg.dataset_root, class_name, p, img_size=int(cfg.img_size), project_type=str(cfg.project_type))
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
            iter_started = time.time()
            now = iter_started
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
            self._cache_one_sample(cfg.dataset_root, class_name, p, img_size=int(cfg.img_size), project_type=str(cfg.project_type))
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
            # Pace to a fixed PERIOD measured from the start of this iteration,
            # so the sleep absorbs the save+preprocess time instead of being
            # added to it.  Anchoring the deadline at "now" (after the work)
            # made the real period work+interval -- a requested 8 fps delivered
            # ~7.1, and the gap grew with the sample cost.
            sleep_s = (iter_started + float(interval)) - time.time()
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


def _raw_ood_metrics(raw_out: np.ndarray) -> Tuple[float, float]:
    """(max_prob, entropy_ratio) from the raw device-mirror output.

    The same float32 arithmetic and op order as _preview_predict and
    TFLite.ino: raw_best / sum(raw) and natural-log entropy over
    raw_i / sum(raw), normalised by ln(N) and clamped to 1.0.  Module-level
    so the sign preview and the image preview (and their test fakes) share
    one implementation.
    """
    arr = np.asarray(raw_out)
    total = int(np.sum(arr))
    if total <= 0:
        total = 1
    max_prob = float(np.float32(np.max(arr)) / np.float32(total))
    f_total = np.float32(total)
    entropy = np.float32(0.0)
    for value in arr:
        p = np.float32(value) / f_total
        if float(p) <= 1e-6:
            continue
        entropy = entropy - p * np.float32(math.log(float(p)))
    n_classes = max(1, int(arr.size))
    max_entropy = float(math.log(n_classes)) if n_classes > 1 else 1.0
    entropy_ratio = (
        float(np.float32(entropy) / np.float32(max_entropy))
        if max_entropy > 0 else 1.0
    )
    return max_prob, min(1.0, entropy_ratio)


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


# Working resolution the glyph front-end uses on the MCU (captures are
# trained at 96x96 glyph frames) and the image project's fallback model side.
GLYPH_PREPROCESS_FRAME_SIDE = 96


def _preprocess_frame_side(project_type: Any, img_size: Any) -> int:
    """Capture-frame side -> preprocessing side for the live serial stream."""
    try:
        side = int(img_size or 96)
    except (TypeError, ValueError):
        side = 96
    if is_glyph_project(str(project_type or "")):
        return GLYPH_PREPROCESS_FRAME_SIDE
    return max(8, min(512, side))


def _resize_square_box_average_u8(arr: np.ndarray, out_side: int) -> np.ndarray:
    """Exact integer box-average resize used by the host and the MCU sketches.

    The source interval of output pixel o is [o*in//out, (o+1)*in//out)
    (clamped to at least one source pixel), every pixel in that box is
    summed and divided with round-half-up.  No floats and no PIL/BILINEAR
    path, so the same function is reproducible in C on the P4 — it is
    mirrored bit-for-bit by TFLite/frame_resize.h
    (FrameResizeBoxAverageGray / FrameResizeBoxAverageRgb) and pinned by
    tests/test_frame_resize_parity.py.
    """
    a = np.asarray(arr)
    out = max(1, int(out_side))
    if a.ndim < 2 or a.size == 0:
        return a
    h, w = int(a.shape[0]), int(a.shape[1])
    if h == out and w == out:
        return a
    dst = np.empty((out, out) + tuple(a.shape[2:]), dtype=np.uint8)
    for oy in range(out):
        sy0 = (oy * h) // out
        sy1 = max(sy0 + 1, ((oy + 1) * h) // out)
        for ox in range(out):
            sx0 = (ox * w) // out
            sx1 = max(sx0 + 1, ((ox + 1) * w) // out)
            block = a[sy0:sy1, sx0:sx1].astype(np.int64)
            count = int(block.shape[0] * block.shape[1])
            if a.ndim == 2:
                dst[oy, ox] = np.uint8((int(block.sum()) + count // 2) // count)
            else:
                sums = block.reshape((-1, block.shape[-1])).sum(axis=0)
                dst[oy, ox, :] = np.clip((sums + count // 2) // count, 0, 255).astype(np.uint8)
    return dst


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
    arr = np.asarray(img)
    # Exact integer box-average, matching the P4 sketch's resize helper.
    arr = _resize_square_box_average_u8(arr, int(out_side))
    return _to_png_bytes(Image.fromarray(arr, mode=("RGB" if arr.ndim == 3 else "L")))


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


def _center_crop_webcam_square(frame: np.ndarray, project_type: Any) -> np.ndarray:
    """Center-crop a webcam frame to 1:1 for the glyph projects.

    The MCU streams a square frame and the Preview / class-preprocess
    stages are 1:1.  A 16:9 MacBook camera in the same stage would be
    letterboxed (black bars top and bottom), and the ROI coordinates would
    describe a different geometry from the square device frame.  Cropping
    ON INGEST makes webcam and device share one geometry, and every
    downstream consumer (live preview, hold capture, Preview inference)
    sees the same square pixels.

    Image projects keep their native aspect: their pipeline and overlays
    are built around the full webcam frame, and there is no device square
    to match.  No-op for already-square frames.
    """
    if not is_glyph_project(str(project_type or "")):
        return frame
    arr = np.asarray(frame)
    if arr.ndim < 2:
        return frame
    fh, fw = int(arr.shape[0]), int(arr.shape[1])
    side = min(fh, fw)
    if side <= 0 or fh == fw:
        return frame
    x0 = (fw - side) // 2
    y0 = (fh - side) // 2
    return np.ascontiguousarray(arr[y0:y0 + side, x0:x0 + side])


def _cap_capture_resolution(img: Image.Image, max_side: int = CAPTURE_MAX_SIDE) -> Image.Image:
    """Bound a captured sample's long side, preserving its aspect ratio.

    The webcam is the only source that can hand us more than the model input
    size, so it is the only one that needs a ceiling: a 4K frame written at
    capture rate is a lot of disk and PNG-encode time for no extra detail.
    1024 keeps a single character comfortably draggable in the Manual ROI
    editor while staying cheap enough for hold capture.

    Aspect is PRESERVED on purpose — the sample is the user's view of the
    scene, and squashing it square is what made the ROI unusable.
    """
    try:
        w, h = (int(v) for v in img.size)
        long_side = max(w, h)
        if long_side <= 0 or long_side <= int(max_side):
            return img
        scale = float(max_side) / float(long_side)
        return img.resize(
            (max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
            Image.LANCZOS,
        )
    except Exception:
        return img


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
