from __future__ import annotations

import base64
import dataclasses
import json
import shutil
import time
import uuid
import os
import sys
import re
import subprocess
import urllib.request
import zipfile
from html import escape as html_escape
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st
import streamlit.components.v1 as components

from camera_permission import ensure_camera_access, get_camera_access_status
from trainer import TrainConfig, new_run_dir, train_and_export
from ui_styles import inject_teachable_style

from dataset_io import (
    IMAGE_EXTS,
    ImportedData,
    export_classified_dataset,
    export_from_assignments,
    infer_imported_data,
    sanitize_class_name,
    materialize_files,
    materialize_zip_bytes,
    new_output_dir,
)
from serial_device import list_serial_ports, read_frame_png_from_serial
from record_controller import RecordController, SessionConfig, make_hold_button_html


APP_NAME = "TFLiteTraining"


#region debug-point capture-webcam-source-helper
def _dbg_capture_webcam_source(hypothesis_id: str, run_id: str, location: str, msg: str, data: dict) -> None:
    _paths = [".dbg/packaged-webcam-bounce.env", ".dbg/capture-webcam-source.env"]
    _u = "http://127.0.0.1:7777/event"
    _s = "packaged-webcam-bounce"
    try:
        for _p in _paths:
            if not Path(_p).exists():
                continue
            with open(_p, "r", encoding="utf-8") as f:
                for _line in f.read().splitlines():
                    if _line.startswith("DEBUG_SERVER_URL="):
                        _u = _line.split("=", 1)[1].strip() or _u
                    elif _line.startswith("DEBUG_SESSION_ID="):
                        _s = _line.split("=", 1)[1].strip() or _s
            break
        urllib.request.urlopen(
            urllib.request.Request(
                _u,
                data=json.dumps(
                    {
                        "sessionId": _s,
                        "runId": run_id,
                        "hypothesisId": hypothesis_id,
                        "location": location,
                        "msg": msg,
                        "data": data,
                        "ts": int(time.time() * 1000),
                    }
                ).encode("utf-8"),
                headers={"Content-Type": "application/json"},
            ),
            timeout=0.5,
        ).read()
    except Exception:
        pass
#endregion


#region debug-point open-project-layout-helper
def _dbg_open_project_layout(hypothesis_id: str, run_id: str, location: str, msg: str, data: dict) -> None:
    _paths = [".dbg/open-project-layout.env"]
    _u = "http://127.0.0.1:7777/event"
    _s = "open-project-layout"
    try:
        for _p in _paths:
            if not Path(_p).exists():
                continue
            with open(_p, "r", encoding="utf-8") as f:
                for _line in f.read().splitlines():
                    if _line.startswith("DEBUG_SERVER_URL="):
                        _u = _line.split("=", 1)[1].strip() or _u
                    elif _line.startswith("DEBUG_SESSION_ID="):
                        _s = _line.split("=", 1)[1].strip() or _s
            break
        urllib.request.urlopen(
            urllib.request.Request(
                _u,
                data=json.dumps(
                    {
                        "sessionId": _s,
                        "runId": run_id,
                        "hypothesisId": hypothesis_id,
                        "location": location,
                        "msg": msg,
                        "data": data,
                        "ts": int(time.time() * 1000),
                    }
                ).encode("utf-8"),
                headers={"Content-Type": "application/json"},
            ),
            timeout=0.5,
        ).read()
    except Exception:
        pass
#endregion


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
    return (base / APP_NAME).resolve()


APP_DATA_DIR = _app_data_dir()
WORKSPACE_DIR = APP_DATA_DIR / "workspace"
DATASETS_DIR = APP_DATA_DIR / "datasets"


def _documents_dir() -> Path:
    return (Path.home() / "Documents").resolve()


def _default_export_dir() -> Path:
    return (_documents_dir() / APP_NAME / "exports").resolve()


def _pick_directory_dialog(initial_dir: Optional[str] = None) -> Optional[str]:
    if sys.platform == "darwin":
        try:
            start_dir = Path(initial_dir).expanduser().resolve() if initial_dir else _documents_dir()
        except Exception:
            start_dir = _documents_dir()
        try:
            start_posix = str(start_dir).replace("\\", "\\\\").replace('"', '\\"')
            script = (
                'POSIX path of (choose folder with prompt "Choose Folder" '
                f'default location (POSIX file "{start_posix}"))'
            )
            proc = subprocess.run(
                ["osascript", "-e", script],
                capture_output=True,
                text=True,
                check=False,
            )
            if proc.returncode == 0:
                picked = (proc.stdout or "").strip()
                if picked:
                    return picked
        except Exception:
            pass

    try:
        import tkinter as tk
        from tkinter import filedialog
    except Exception:
        return None

    root = tk.Tk()
    root.withdraw()
    try:
        root.attributes("-topmost", True)
    except Exception:
        pass
    try:
        path = filedialog.askdirectory(initialdir=initial_dir or str(_documents_dir()))
    finally:
        try:
            root.destroy()
        except Exception:
            pass
    return path or None


def _pick_tmproj_file_dialog() -> Optional[Path]:
    if sys.platform != "darwin":
        return None
    try:
        proc = subprocess.run(
            ["osascript", "-e", 'POSIX path of (choose file with prompt "Open Project")'],
            capture_output=True,
            text=True,
            check=False,
        )
    except Exception:
        return None
    if proc.returncode != 0:
        return None
    picked = (proc.stdout or "").strip()
    if not picked:
        return None
    p = Path(picked).expanduser()
    if p.suffix.lower() != ".tmproj":
        return None
    return p


def _tmproj_read_manifest(path: Path) -> Dict[str, Any]:
    try:
        with zipfile.ZipFile(path, "r") as zf:
            raw = zf.read("manifest.json")
        data = json.loads(raw.decode("utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _tmproj_detect_project_type(path: Path, manifest: Dict[str, Any]) -> str:
    ptype = str(manifest.get("project_type") or "").strip().lower()
    name = str(path.name or "").lower()
    if "pose" in ptype or "pose" in name:
        return "pose"
    if "image" in ptype or "image" in name:
        return "image"
    return "image"


def _validate_export_inputs(export_dir: Path, model_name: str, array_name: str, tflite_path: Path) -> List[str]:
    errors: List[str] = []
    if not model_name.strip():
        errors.append("Model file prefix cannot be empty.")
    if any(sep in model_name for sep in ("/", "\\", os.sep)):
        errors.append("Model file prefix must not include path separators.")
    if not re.match(r"^[A-Za-z0-9_.-]+$", model_name):
        errors.append("Model file prefix may only contain letters, numbers, '.', '_' and '-'.")
    if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", array_name):
        errors.append("C array name must be a valid identifier (letters/numbers/underscore, not starting with a number).")
    try:
        export_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        errors.append(f"Failed to create export directory: {export_dir} ({e})")
        return errors
    if not export_dir.is_dir():
        errors.append(f"Export path is not a directory: {export_dir}")
    if not tflite_path.exists():
        errors.append("Missing .tflite file. Train the model first.")
    try:
        test_file = export_dir / ".write_test"
        test_file.write_text("ok", encoding="utf-8")
        test_file.unlink(missing_ok=True)
    except Exception as e:
        errors.append(f"Export directory is not writable: {export_dir} ({e})")
    return errors


def _init_session() -> None:
    if "session_id" not in st.session_state:
        resumed_session_id = ""
        try:
            resumed_session_id = str(st.query_params.get("tm_session") or "")
        except Exception:
            try:
                resumed_session_id = str(st.experimental_get_query_params().get("tm_session", [""])[0] or "")
            except Exception:
                resumed_session_id = ""
        st.session_state.session_id = resumed_session_id.strip() or uuid.uuid4().hex
    if "imported" not in st.session_state:
        st.session_state.imported = None
    if "project_type" not in st.session_state:
        resumed_project_type = ""
        try:
            resumed_project_type = str(st.query_params.get("tm_project") or "")
        except Exception:
            try:
                resumed_project_type = str(st.experimental_get_query_params().get("tm_project", [""])[0] or "")
            except Exception:
                resumed_project_type = ""
        st.session_state.project_type = resumed_project_type.strip() or None
    if "step" not in st.session_state:
        st.session_state.step = 0
    if "dataset_dir" not in st.session_state:
        st.session_state.dataset_dir = None
    if "assignments" not in st.session_state:
        st.session_state.assignments = {}
    if "class_names" not in st.session_state:
        st.session_state.class_names = ["Class 1", "Class 2"]
    if "class_rename" not in st.session_state:
        st.session_state.class_rename = {}
    if "last_export_dir" not in st.session_state:
        st.session_state.last_export_dir = str(_default_export_dir())
    if "train_cfg" not in st.session_state:
        st.session_state.train_cfg = TrainConfig()
    if "train_result" not in st.session_state:
        st.session_state.train_result = None
    if "local_import_path" not in st.session_state:
        st.session_state.local_import_path = ""
    if "export_validated_token" not in st.session_state:
        st.session_state.export_validated_token = ""
    if "tm_classes" not in st.session_state:
        st.session_state.tm_classes = ["Class 1", "Class 2"]
    if "tm_device_url" not in st.session_state:
        st.session_state.tm_device_url = ""
    if "tm_device_enabled" not in st.session_state:
        st.session_state.tm_device_enabled = False
    if "tm_serial_port" not in st.session_state:
        st.session_state.tm_serial_port = ""
    if "tm_serial_baud" not in st.session_state:
        # The ESP32-P4 camera firmware streams at 921600 baud (see repo
        # CLAUDE.md) — make that the default so a fresh session just works.
        st.session_state.tm_serial_baud = 921600
    if "tm_serial_sync" not in st.session_state:
        st.session_state.tm_serial_sync = "AA 55 AA"
    if "tm_serial_frame_side" not in st.session_state:
        st.session_state.tm_serial_frame_side = 96
    if "tm_last_device_frame" not in st.session_state:
        st.session_state.tm_last_device_frame = None
    if "tm_capture_open" not in st.session_state:
        st.session_state.tm_capture_open = False
    if "tm_capture_source" not in st.session_state:
        st.session_state.tm_capture_source = ""
    if "tm_capture_class" not in st.session_state:
        st.session_state.tm_capture_class = ""
    if "tm_record_fps" not in st.session_state:
        st.session_state.tm_record_fps = 8.0
    if "tm_webcam_index" not in st.session_state:
        st.session_state.tm_webcam_index = 0
    if "tm_webcam_user_selected" not in st.session_state:
        st.session_state.tm_webcam_user_selected = False
    if "tm_record_crop_box" not in st.session_state:
        st.session_state.tm_record_crop_box = None
    if "tm_crop_mode" not in st.session_state:
        st.session_state.tm_crop_mode = "full"
    if "tm_pending_image" not in st.session_state:
        st.session_state.tm_pending_image = None
    if "tm_pending_class" not in st.session_state:
        st.session_state.tm_pending_class = None
    if "tm_camera_permission_status" not in st.session_state:
        st.session_state.tm_camera_permission_status = ""
    if "tm_camera_permission_note" not in st.session_state:
        st.session_state.tm_camera_permission_note = ""
    if "tm_camera_permission_class" not in st.session_state:
        st.session_state.tm_camera_permission_class = ""
    if "tm_open_source_class" not in st.session_state:
        st.session_state.tm_open_source_class = ""
    if "tm_open_source_kind" not in st.session_state:
        st.session_state.tm_open_source_kind = ""
    if "tm_frontend_notice" not in st.session_state:
        st.session_state.tm_frontend_notice = ""
    if "tm_return_target" not in st.session_state:
        st.session_state.tm_return_target = "home"


def _session_workspace() -> Path:
    p = WORKSPACE_DIR / st.session_state.session_id
    p.mkdir(parents=True, exist_ok=True)
    return p


def _reset_session_workspace() -> None:
    p = WORKSPACE_DIR / st.session_state.session_id
    if p.exists():
        shutil.rmtree(p)
    st.session_state.imported = None
    st.session_state.project_type = None
    st.session_state.step = 0
    st.session_state.dataset_dir = None
    st.session_state.assignments = {}
    st.session_state.class_names = ["Class 1", "Class 2"]
    st.session_state.class_rename = {}
    st.session_state.train_cfg = TrainConfig()
    st.session_state.train_result = None
    st.session_state.export_validated_token = ""
    st.session_state.tm_classes = ["Class 1", "Class 2"]
    st.session_state.tm_device_enabled = False
    st.session_state.tm_crop_mode = "full"
    st.session_state.tm_pending_image = None
    st.session_state.tm_pending_class = None
    st.session_state.tm_last_device_frame = None
    st.session_state.tm_capture_open = False
    st.session_state.tm_capture_source = ""
    st.session_state.tm_capture_class = ""
    st.session_state.tm_record_crop_box = None
    st.session_state.tm_camera_permission_status = ""
    st.session_state.tm_camera_permission_note = ""
    st.session_state.tm_camera_permission_class = ""
    st.session_state.tm_open_source_class = ""
    st.session_state.tm_open_source_kind = ""
    st.session_state.tm_frontend_notice = ""
    st.session_state.tm_webcam_index = 0
    st.session_state.tm_webcam_user_selected = False


def _begin_fresh_tm_session() -> str:
    # #region debug-point B:begin-fresh-session
    _dbg_open_project_layout("B", "pre-fix", "app.py:_begin_fresh_tm_session", "[DEBUG] begin fresh session", {"old_session": str(st.session_state.get("session_id", ""))})
    # #endregion
    _reset_session_workspace()
    st.session_state.session_id = uuid.uuid4().hex
    # #region debug-point B:begin-fresh-session-new
    _dbg_open_project_layout("B", "pre-fix", "app.py:_begin_fresh_tm_session", "[DEBUG] fresh session created", {"new_session": str(st.session_state.session_id)})
    # #endregion
    return st.session_state.session_id


def _read_uploaded_images(uploaded_files) -> List[Tuple[str, bytes]]:
    out: List[Tuple[str, bytes]] = []
    for f in uploaded_files:
        out.append((f.name, f.getvalue()))
    return out


def _render_import_panel() -> None:
    st.subheader("Import images")
    method = st.radio("Import method", ["ZIP file", "Multiple images", "Local folder path"], horizontal=True)
    ws = _session_workspace()

    if method == "ZIP file":
        zip_file = st.file_uploader("Drop or select a .zip file", type=["zip"])
        if zip_file is not None and st.button("Import", type="primary"):
            dest = ws / "import"
            materialize_zip_bytes(zip_file.getvalue(), dest)
            st.session_state.imported = infer_imported_data(dest)
            st.session_state.assignments = {}
            st.session_state.class_rename = {}

    elif method == "Multiple images":
        exts = sorted([e.lstrip(".") for e in IMAGE_EXTS])
        files = st.file_uploader("Select or drop multiple images", type=exts, accept_multiple_files=True)
        if files and st.button("Import", type="primary"):
            dest = ws / "import"
            materialize_files(_read_uploaded_images(files), dest)
            st.session_state.imported = infer_imported_data(dest)
            st.session_state.assignments = {}
            st.session_state.class_rename = {}

    else:
        left, right = st.columns([4, 1])
        with left:
            path_str = st.text_input(
                "Local folder path (classified or unclassified)",
                value=st.session_state.local_import_path,
            )
            st.session_state.local_import_path = path_str
        with right:
            if st.button("Browse...", key="browse_import_dir"):
                picked = _pick_directory_dialog(initial_dir=st.session_state.local_import_path)
                if picked:
                    st.session_state.local_import_path = picked
                    st.rerun()

        if st.button("Load folder", type="primary"):
            path_str = st.session_state.local_import_path
            p = Path(path_str).expanduser()
            if not p.exists() or not p.is_dir():
                st.error("Path does not exist or is not a folder.")
                return
            st.session_state.imported = infer_imported_data(p)
            st.session_state.assignments = {}
            st.session_state.class_rename = {}


def _render_overview(imported: ImportedData) -> None:
    st.subheader("Preview")
    st.markdown(
        f'''
<div class="tm-kv">
  <b>Source</b>: {imported.root_dir}<br/>
  <b>Images</b>: {len(imported.images)}<br/>
  <b>Classified folders</b>: {imported.classified}
</div>
        ''',
        unsafe_allow_html=True,
    )

    cols = st.columns(6)
    sample = imported.images[: min(len(imported.images), 6)]
    for c, p in zip(cols, sample):
        with c:
            st.image(str(p), use_container_width=True)
            st.caption(p.name)


def _render_classified_flow(imported: ImportedData) -> Optional[Path]:
    st.subheader("Rename classes (classified folders)")
    class_names = list(imported.class_to_images.keys())
    st.write({"Detected classes": class_names})

    rename: Dict[str, str] = {}
    for c in class_names:
        rename[c] = st.text_input(f"{c} →", value=st.session_state.class_rename.get(c, c), key=f"rename_{c}")
    st.session_state.class_rename = rename

    st.write({c: len(imported.class_to_images[c]) for c in class_names})

    if st.button("Save as dataset", type="primary"):
        DATASETS_DIR.mkdir(parents=True, exist_ok=True)
        out_dir = new_output_dir(DATASETS_DIR, prefix="classified")
        try:
            export_classified_dataset(imported.class_to_images, rename, out_dir)
        except Exception as e:
            st.error(str(e))
            return None
        st.success(f"Dataset created: {out_dir}")
        return out_dir
    return None


def _render_unclassified_flow(imported: ImportedData) -> Optional[Path]:
    st.subheader("Assign classes (unclassified images)")
    st.session_state.class_names = _render_class_editor(st.session_state.class_names)
    class_names = st.session_state.class_names

    if not class_names:
        st.warning("Create at least 1 class first.")
        return None

    page_size = st.slider("Images per page", min_value=6, max_value=60, value=18, step=6)
    total = len(imported.images)
    max_page = max(1, (total + page_size - 1) // page_size)
    page = st.number_input("Page", min_value=1, max_value=max_page, value=1, step=1)
    start = (page - 1) * page_size
    end = min(total, start + page_size)

    for idx in range(start, end):
        img = imported.images[idx]
        left, right = st.columns([1, 2])
        with left:
            st.image(str(img), use_container_width=True)
        with right:
            default = st.session_state.assignments.get(idx)
            options = ["Unassigned"] + class_names
            try:
                default_index = options.index(default) if default in options else 0
            except Exception:
                default_index = 0
            choice = st.selectbox(
                f"[{idx}] {img.name}",
                options=options,
                index=default_index,
                key=f"assign_{idx}",
            )
            if choice == "Unassigned":
                st.session_state.assignments.pop(idx, None)
            else:
                st.session_state.assignments[idx] = choice

    require_all = st.checkbox("Require all images to be assigned before saving", value=True)
    assigned_count = len(st.session_state.assignments)
    st.write({"Assigned": assigned_count, "Unassigned": total - assigned_count})

    if st.button("Save as dataset", type="primary"):
        DATASETS_DIR.mkdir(parents=True, exist_ok=True)
        out_dir = new_output_dir(DATASETS_DIR, prefix="labeled")
        assignments: List[Optional[str]] = []
        for i in range(total):
            assignments.append(st.session_state.assignments.get(i))
        try:
            export_from_assignments(imported.images, assignments, out_dir, require_all_assigned=require_all)
        except Exception as e:
            st.error(str(e))
            return None
        st.success(f"Dataset created: {out_dir}")
        return out_dir

    return None


def _render_class_editor(class_names: List[str]) -> List[str]:
    st.write("Classes (one per line)")
    txt = st.text_area(" ", value="\n".join(class_names), height=140, key="class_editor")
    names = [x.strip() for x in txt.splitlines()]
    names = [x for x in names if x]
    return names


def _render_steps() -> None:
    steps = ["Import", "Label", "Train", "Export"]
    s = int(st.session_state.step)
    parts: List[str] = []
    for i, name in enumerate(steps):
        if i == s:
            parts.append(f'<span class="tm-step-on">{i+1}. {name}</span>')
        else:
            parts.append(f"{i+1}. {name}")
    st.markdown(f'<div class="tm-steps">{"  ·  ".join(parts)}</div>', unsafe_allow_html=True)


def _next_class_name(existing: List[str]) -> str:
    idx = 1
    existing_set = {sanitize_class_name(name) for name in existing}
    while True:
        candidate = f"Class {idx}"
        if sanitize_class_name(candidate) not in existing_set:
            return candidate
        idx += 1


def _rename_tm_class_dir(old_name: str, new_name: str) -> None:
    old_safe = sanitize_class_name(old_name)
    new_safe = sanitize_class_name(new_name)
    if old_safe == new_safe:
        return
    root = _tm_dataset_dir()
    old_dir = root / old_safe
    new_dir = root / new_safe
    if not old_dir.exists():
        return
    if new_dir.exists():
        raise ValueError(f"Class already exists: {new_safe}")
    old_dir.rename(new_dir)


def _apply_tm_class_names(current_names: List[str], edited_names: List[str]) -> List[str]:
    normalized: List[str] = []
    used: set[str] = set()
    for idx, raw in enumerate(edited_names):
        candidate = sanitize_class_name(raw) if raw.strip() else f"Class {idx + 1}"
        if candidate in used:
            raise ValueError(f"Duplicate class name: {candidate}")
        used.add(candidate)
        normalized.append(candidate)

    for old_name, new_name in zip(current_names, normalized):
        _rename_tm_class_dir(old_name, new_name)

    return normalized


def _remove_tm_class(class_name: str) -> None:
    class_dir = _tm_dataset_dir() / sanitize_class_name(class_name)
    if class_dir.exists():
        shutil.rmtree(class_dir)
    if st.session_state.tm_capture_class == class_name:
        st.session_state.tm_capture_open = False
        st.session_state.tm_capture_source = ""
        st.session_state.tm_capture_class = ""
    if st.session_state.tm_camera_permission_class == class_name:
        st.session_state.tm_camera_permission_class = ""
        st.session_state.tm_camera_permission_note = ""
        st.session_state.tm_camera_permission_status = ""
    if st.session_state.tm_pending_class == class_name:
        st.session_state.tm_pending_class = None
        st.session_state.tm_pending_image = None


def _import_classified_into_workspace(imported: ImportedData, rename: Dict[str, str]) -> List[str]:
    dataset_root = _tm_dataset_dir()
    export_classified_dataset(imported.class_to_images, rename, dataset_root)
    labels: List[str] = []
    for class_name in imported.class_to_images.keys():
        labels.append(sanitize_class_name(rename.get(class_name, class_name)))
    if labels:
        _tm_save_classes_meta(labels)
    return labels


def _tm_dataset_stats() -> Tuple[int, int]:
    dataset_dir = _tm_dataset_dir()
    if not dataset_dir.exists():
        return 0, 0
    class_dirs = sorted([p for p in dataset_dir.iterdir() if p.is_dir()])
    sample_count = 0
    for class_dir in class_dirs:
        sample_count += len(_tm_class_image_files(class_dir))
    return len(class_dirs), sample_count


def _tm_class_image_files(class_dir: Path) -> List[Path]:
    if not class_dir.exists():
        return []
    return sorted(
        [p for p in class_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )


def _tm_train_latest_path() -> Path:
    return _tm_dataset_dir().parent / "tm_train_latest.json"


def _tm_load_train_latest() -> Optional[Dict[str, Any]]:
    p = _tm_train_latest_path()
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def _tm_classes_meta_path() -> Path:
    return _tm_dataset_dir().parent / "tm_classes.json"


def _tm_load_classes_meta() -> Optional[List[str]]:
    p = _tm_classes_meta_path()
    if not p.exists():
        return None
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
        classes = raw.get("classes") if isinstance(raw, dict) else None
        if isinstance(classes, list) and all(isinstance(x, str) for x in classes) and classes:
            return [str(x) for x in classes]
    except Exception:
        return None
    return None


def _tm_save_classes_meta(classes: List[str]) -> None:
    p = _tm_classes_meta_path()
    p.write_text(json.dumps({"classes": list(classes)}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _status_style(status: str) -> str:
    if status in {"granted", "not_required", "trained", "ready"}:
        return "tm-status tm-status-ok"
    if status in {"denied", "restricted", "open_failed", "timeout"}:
        return "tm-status tm-status-bad"
    if status in {"not_determined"}:
        return "tm-status tm-status-warn"
    return "tm-status tm-status-idle"


def _render_tm_workspace_header() -> None:
    class_count, sample_count = _tm_dataset_stats()
    meta = _tm_load_train_latest()
    metrics = dict(meta.get("metrics") or {}) if isinstance(meta, dict) else {}
    trained_label = f"{float(metrics.get('val_accuracy', 0.0)):.1%}" if metrics else "Not trained"
    trained_status = "trained" if metrics else "idle"
    capture_label = st.session_state.tm_capture_source.upper() if st.session_state.tm_capture_open else "Idle"
    camera_status = get_camera_access_status()
    st.markdown(
        f'''
<div class="tm-hero">
  <div class="tm-hero-compact">
    <div class="tm-hero-copy">
      <div class="tm-eyebrow">AI Training Platform</div>
      <h2>Image Training Workspace</h2>
      <p>Collect samples, train, preview, and export in a single workspace.</p>
    </div>
    <div class="tm-hero-stats">
      <div class="tm-stat">
        <div class="tm-stat-label">Classes</div>
        <div class="tm-stat-value">{class_count}</div>
      </div>
      <div class="tm-stat">
        <div class="tm-stat-label">Samples</div>
        <div class="tm-stat-value">{sample_count}</div>
      </div>
      <div class="tm-stat">
        <div class="tm-stat-label">Model</div>
        <div class="tm-stat-value">{html_escape(trained_label)}</div>
        <div class="tm-stat-note"><span class="{_status_style(trained_status)}">{'Ready' if metrics else 'Waiting'}</span></div>
      </div>
      <div class="tm-stat">
        <div class="tm-stat-label">Capture</div>
        <div class="tm-stat-value">{html_escape(capture_label)}</div>
        <div class="tm-stat-note"><span class="{_status_style(camera_status.status)}">{html_escape(camera_status.status.replace('_', ' ').title())}</span></div>
      </div>
    </div>
  </div>
</div>
        ''',
        unsafe_allow_html=True,
    )


def _render_tm_flow_lane() -> None:
    c1, a1, c2, a2, c3 = st.columns([2.2, 0.22, 1.0, 0.22, 1.75], gap="small")
    with c1:
        st.markdown('<div class="tm-flow-step"><span>1</span><strong>Classes</strong><small>Add samples and rename classes</small></div>', unsafe_allow_html=True)
    with a1:
        st.markdown('<div class="tm-flow-arrow">→</div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="tm-flow-step"><span>2</span><strong>Training</strong><small>Train a lightweight model</small></div>', unsafe_allow_html=True)
    with a2:
        st.markdown('<div class="tm-flow-arrow">→</div>', unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="tm-flow-step"><span>3</span><strong>Preview / Export</strong><small>Check inputs and export files</small></div>', unsafe_allow_html=True)


def _render_camera_permission_card() -> None:
    status = get_camera_access_status()
    left, right = st.columns([3, 1.2])
    with left:
        st.markdown(
            f'''
<div class="tm-inline-note">
  <strong>Camera</strong>
  <div style="margin-top:6px;"><span class="{_status_style(status.status)}">{html_escape(status.message)}</span></div>
</div>
            ''',
            unsafe_allow_html=True,
        )
    with right:
        if st.button("Request", key="tm_request_camera_access", use_container_width=True):
            result = ensure_camera_access(int(st.session_state.tm_webcam_index))
            st.session_state.tm_camera_permission_status = result.status
            st.session_state.tm_camera_permission_note = result.message
            st.rerun()

    note = st.session_state.tm_camera_permission_note
    if note:
        level = st.success if st.session_state.tm_camera_permission_status in {"granted", "not_required"} else st.info
        level(note)


def _render_new_project() -> None:
    # #region debug-point B:render-new-project
    _dbg_open_project_layout("B", "pre-fix", "app.py:_render_new_project", "[DEBUG] render new project page", {"session": str(st.session_state.get("session_id", "")), "project_type": str(st.session_state.get("project_type", "")), "query": dict(st.query_params) if hasattr(st, "query_params") else {}})
    # #endregion
    inject_teachable_style()
    st.markdown(
        '''
<div class="tm-hero">
  <div class="tm-hero-grid">
    <div class="tm-hero-copy">
      <div class="tm-eyebrow">Desktop AI Trainer</div>
      <h2>Build classroom-friendly image models without touching code.</h2>
      <p>Inspired by Google Teachable Machine, with a desktop-first workflow for importing data, training, and exporting TFLite + C sources.</p>
      <div class="tm-chip-row">
        <span class="tm-chip">Import local images</span>
        <span class="tm-chip">Capture from webcam</span>
        <span class="tm-chip">Train lightweight CNN</span>
        <span class="tm-chip">Export .tflite + model.cpp</span>
      </div>
    </div>
    <div class="tm-hero-panel">
      <h4>Recommended flow</h4>
      <div class="tm-stat-grid">
        <div class="tm-stat">
          <div class="tm-stat-label">1</div>
          <div class="tm-stat-value">Collect</div>
          <div class="tm-stat-note">Import images or capture samples</div>
        </div>
        <div class="tm-stat">
          <div class="tm-stat-label">2</div>
          <div class="tm-stat-value">Label</div>
          <div class="tm-stat-note">Organize data by class</div>
        </div>
        <div class="tm-stat">
          <div class="tm-stat-label">3</div>
          <div class="tm-stat-value">Train</div>
          <div class="tm-stat-note">Train and quantize to int8 TFLite</div>
        </div>
        <div class="tm-stat">
          <div class="tm-stat-label">4</div>
          <div class="tm-stat-value">Export</div>
          <div class="tm-stat-note">Export files ready for MCU integration</div>
        </div>
      </div>
    </div>
  </div>
</div>
        ''',
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            '''
<div class="tm-card">
  <div class="tm-eyebrow">Ready now</div>
  <h3>Image Project</h3>
  <p>Build an image classification model from images, webcam, or device stream.</p>
  <div class="tm-card-footnote">Great for classroom demos and local training.</div>
</div>
            ''',
            unsafe_allow_html=True,
        )
        with st.popover("Open Image Project", use_container_width=True):
            st.markdown("Choose a start mode")
            if st.button("Start from empty", key="tm_start_empty", type="primary", use_container_width=True):
                _begin_fresh_tm_session()
                st.session_state.tm_return_target = "home"
                st.session_state.project_type = "image"
                st.session_state.step = 0
                _tm_set_query_params(tm_project="image", tm_session=st.session_state.session_id)
                st.rerun()
            if st.button("Start from classified class", key="tm_start_classified", use_container_width=True):
                _begin_fresh_tm_session()
                st.session_state.tm_return_target = "classified-import"
                st.session_state.project_type = "image_classified_import"
                st.session_state.step = 0
                _tm_set_query_params(tm_project="image_classified_import", tm_session=st.session_state.session_id)
                st.rerun()
        with st.popover("Open Project", use_container_width=True):
            st.markdown("Open a saved .tmproj file")
            if st.button("Open .tmproj", key="tm_open_tmproj", type="primary", use_container_width=True):
                p = _pick_tmproj_file_dialog()
                if not p:
                    st.warning("No file selected.")
                    st.stop()
                manifest = _tmproj_read_manifest(p)
                ptype = _tmproj_detect_project_type(p, manifest)
                if ptype != "image":
                    st.error(f"Unsupported project type: {ptype}")
                    st.stop()
                # #region debug-point A:home-open-project-click
                _dbg_open_project_layout("A", "pre-fix", "app.py:_render_new_project", "[DEBUG] home open project selected file", {"picked_path": str(p), "project_type": str(ptype), "session_before_fresh": str(st.session_state.get("session_id", ""))})
                # #endregion
                _begin_fresh_tm_session()
                st.session_state.tm_return_target = "home"
                controller = _get_record_controller()
                controller.set_config(
                    st.session_state.session_id,
                    SessionConfig(
                        dataset_root=_tm_dataset_dir(),
                        serial_port=st.session_state.tm_serial_port,
                        serial_baud=int(st.session_state.tm_serial_baud),
                        serial_sync=str(st.session_state.tm_serial_sync),
                        serial_frame_side=int(st.session_state.tm_serial_frame_side),
                        webcam_index=int(st.session_state.tm_webcam_index),
                        fps=float(st.session_state.tm_record_fps),
                        crop_box=st.session_state.tm_record_crop_box,
                    ),
                )
                state = controller._project_open(st.session_state.session_id, p)
                # #region debug-point A:home-open-project-after-open
                _dbg_open_project_layout("A", "pre-fix", "app.py:_render_new_project", "[DEBUG] home open project restored state", {"session": str(st.session_state.session_id), "classes": list(state.get("classes") or []), "count_keys": list((state.get("counts") or {}).keys())})
                # #endregion
                st.session_state.tm_classes = list(state.get("classes") or ["Class 1", "Class 2"])
                train_cfg_state = state.get("train_cfg") if isinstance(state, dict) else None
                if isinstance(train_cfg_state, dict):
                    # Start from the current defaults and overlay whatever the
                    # project actually stored — fields absent from an old
                    # project keep today's defaults instead of silently
                    # downgrading to stale hardcoded literals.
                    defaults = dataclasses.asdict(st.session_state.train_cfg)
                    known = {k: v for k, v in train_cfg_state.items() if k in defaults}
                    try:
                        st.session_state.train_cfg = TrainConfig(**{**defaults, **known})
                    except TypeError:
                        # Project carried unexpected keys — fall back to the defaults.
                        st.session_state.train_cfg = TrainConfig()
                st.session_state.project_type = "image"
                _tm_set_query_params(tm_project="image", tm_session=st.session_state.session_id)
                st.rerun()
    with c2:
        st.markdown(
            '''
<div class="tm-card">
  <div class="tm-eyebrow">Roadmap</div>
  <h3>Audio Project <span class="tm-badge">Coming soon</span></h3>
  <p>Planned: microphone sampling and audio classification for voice commands.</p>
</div>
            ''',
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            '''
<div class="tm-card">
  <div class="tm-eyebrow">Roadmap</div>
  <h3>Pose Project <span class="tm-badge">Coming soon</span></h3>
  <p>Planned: pose and gesture projects for interactive demos.</p>
</div>
            ''',
            unsafe_allow_html=True,
        )


def _render_train_config(cfg: TrainConfig) -> TrainConfig:
    left, right = st.columns(2)
    with left:
        img_size = st.number_input("Image size", min_value=32, max_value=224, value=int(cfg.img_size), step=8)
        color_mode = st.selectbox("Color mode", options=["rgb", "grayscale"], index=0 if cfg.color_mode == "rgb" else 1)
        batch_size = st.selectbox("Batch size", options=[8, 16, 32, 64], index=[8, 16, 32, 64].index(cfg.batch_size))
        epochs = st.number_input("Epochs", min_value=1, max_value=200, value=int(cfg.epochs), step=1)
        validation_split = st.slider("Validation split", min_value=0.05, max_value=0.5, value=float(cfg.validation_split), step=0.05)
    with right:
        optimizer = st.selectbox("Optimizer", options=["adam", "sgd", "rmsprop"], index=["adam", "sgd", "rmsprop"].index(cfg.optimizer.lower()))
        learning_rate = st.number_input("Learning rate", min_value=1e-5, max_value=1e-1, value=float(cfg.learning_rate), format="%.5f")
        conv1_filters = st.selectbox("Conv1 filters", options=[4, 8, 16, 32], index=[4, 8, 16, 32].index(cfg.conv1_filters))
        conv2_filters = st.selectbox("Conv2 filters", options=[8, 16, 32, 64], index=[8, 16, 32, 64].index(cfg.conv2_filters))
        dense_units = st.selectbox("Dense units", options=[16, 32, 64, 128], index=[16, 32, 64, 128].index(cfg.dense_units))

    return TrainConfig(
        img_size=int(img_size),
        color_mode=str(color_mode),
        batch_size=int(batch_size),
        epochs=int(epochs),
        validation_split=float(validation_split),
        seed=int(cfg.seed),
        optimizer=str(optimizer),
        learning_rate=float(learning_rate),
        conv1_filters=int(conv1_filters),
        conv2_filters=int(conv2_filters),
        dense_units=int(dense_units),
        representative_samples=int(cfg.representative_samples),
        preprocess_mode=str(getattr(cfg, "preprocess_mode", "auto_by_label")),
        manual_roi=getattr(cfg, "manual_roi", None),
    )


def _tm_get_query_param(key: str) -> str:
    try:
        v = st.query_params.get(key)
        if v is None:
            return ""
        if isinstance(v, list):
            return str(v[0]) if v else ""
        return str(v)
    except Exception:
        v = st.experimental_get_query_params().get(key, [""])
        return str(v[0]) if v else ""


def _tm_set_query_params(**params: str) -> None:
    try:
        st.query_params.clear()
        for k, v in params.items():
            if v is not None and str(v) != "":
                st.query_params[k] = str(v)
    except Exception:
        st.experimental_set_query_params(**{k: v for k, v in params.items() if v is not None and str(v) != ""})


def _tm_clear_query_params() -> None:
    try:
        st.query_params.clear()
    except Exception:
        st.experimental_set_query_params()


def _tm_render_page_scroll_reset() -> None:
    components.html(
        '''
        <script>
        function resetFrameBox(node) {
          try {
            if (!node || !node.style) return;
            node.style.height = '';
            node.style.minHeight = '';
            node.style.maxHeight = '';
            node.style.overflow = '';
            node.style.overflowY = '';
            node.style.overflowX = '';
          } catch (e) {}
        }
        try {
          window.scrollTo(0, 0);
          document.documentElement.scrollTop = 0;
          document.body.scrollTop = 0;
        } catch (e) {}
        try {
          if (window.parent) {
            window.parent.scrollTo(0, 0);
            if (window.parent.document) {
              window.parent.document.documentElement.scrollTop = 0;
              window.parent.document.body.scrollTop = 0;
              try {
                const frames = window.parent.document.querySelectorAll('iframe');
                frames.forEach((frame) => {
                  resetFrameBox(frame);
                  resetFrameBox(frame.parentElement);
                });
              } catch (e) {}
            }
          }
        } catch (e) {}
        </script>
        ''',
        height=0,
        width=0,
    )


def _tm_render_shell_reflow_ping(reason: str = "image-project-mount") -> None:
    safe_reason = html_escape(str(reason or "image-project-mount"))
    components.html(
        f'''
        <script>
        (function() {{
          const reason = "{safe_reason}";
          function ping(tag) {{
            try {{
              const targets = [window, window.parent, window.top];
              for (const target of targets) {{
                try {{
                  if (target) target.dispatchEvent(new Event('resize'));
                }} catch (e) {{}}
                try {{
                  if (target) target.dispatchEvent(new Event('orientationchange'));
                }} catch (e) {{}}
                try {{
                  if (target && target.pywebview && target.pywebview.api && typeof target.pywebview.api.request_reflow === 'function') {{
                    target.pywebview.api.request_reflow(String(reason + ':' + tag));
                  }}
                }} catch (e) {{}}
              }}
            }} catch (e) {{}}
          }}
          ping('now');
          window.setTimeout(() => ping('t120'), 120);
          window.setTimeout(() => ping('t360'), 360);
          window.setTimeout(() => ping('t900'), 900);
        }})();
        </script>
        ''',
        height=0,
        width=0,
    )


def _list_camera_options(max_count: int = 6) -> List[Dict[str, str]]:
    options: List[Dict[str, str]] = []
    if sys.platform == "darwin":
        try:
            from AVFoundation import AVCaptureDevice, AVMediaTypeVideo

            devices = list(AVCaptureDevice.devicesWithMediaType_(AVMediaTypeVideo) or [])
            for idx, dev in enumerate(devices[:max_count]):
                name = str(dev.localizedName() or f"Camera {idx}")
                options.append({"index": idx, "label": name})
            #region debug-point E:camera-enumeration
            _dbg_capture_webcam_source("E", "pre-fix", "app.py:_list_camera_options", "[DEBUG] camera options enumerated via AVFoundation", {"platform": sys.platform, "options": options})
            #endregion
        except Exception:
            options = []
            #region debug-point E:camera-enumeration-failed
            _dbg_capture_webcam_source("E", "pre-fix", "app.py:_list_camera_options", "[DEBUG] camera enumeration via AVFoundation failed", {"platform": sys.platform})
            #endregion
    if not options:
        for idx in range(max_count):
            options.append({"index": idx, "label": f"Camera {idx}"})
        #region debug-point E:camera-fallback
        _dbg_capture_webcam_source("E", "pre-fix", "app.py:_list_camera_options", "[DEBUG] camera options fallback used", {"platform": sys.platform, "options": options})
        #endregion
    options = sorted(options, key=lambda item: (1 if _is_virtual_camera_label(str(item.get("label", ""))) else 0, int(item.get("index", 0))))
    return options


def _tm_sample_previews(classes: List[str], limit_per_class: Optional[int] = None) -> Dict[str, List[Dict[str, str]]]:
    out: Dict[str, List[Dict[str, str]]] = {}
    root = _tm_dataset_dir()
    for name in classes:
        class_dir = root / sanitize_class_name(name)
        previews: List[Dict[str, str]] = []
        if class_dir.exists():
            files = _tm_class_image_files(class_dir)
            if limit_per_class is not None:
                files = files[:limit_per_class]
            for p in files:
                try:
                    b64 = base64.b64encode(p.read_bytes()).decode("ascii")
                    previews.append(
                        {
                            "src": f"data:image/png;base64,{b64}",
                            "filename": str(p.name),
                        }
                    )
                except Exception:
                    continue
        out[name] = previews
    return out


def _is_likely_user_serial_port(device: str, description: str) -> bool:
    d = (device or "").strip().lower()
    desc = (description or "").strip().lower()
    blocked = {
        "/dev/cu.bluetooth-incoming-port",
        "/dev/cu.debug-console",
        "/dev/tty.bluetooth-incoming-port",
        "/dev/tty.debug-console",
    }
    if d in blocked:
        return False
    if "bluetooth" in desc and "serial" not in desc:
        return False
    return bool(d)


def _is_virtual_camera_label(label: str) -> bool:
    s = (label or "").strip().lower()
    markers = ("iriun", "virtual", "obs", "camo", "epoccam", "ndi", "droidcam")
    return any(m in s for m in markers)


def _preferred_webcam_index(options: List[Dict[str, str]]) -> int:
    for item in options:
        label = str(item.get("label", ""))
        if not _is_virtual_camera_label(label):
            try:
                return int(item.get("index", 0))
            except Exception:
                continue
    try:
        return int(options[0].get("index", 0)) if options else 0
    except Exception:
        return 0


def _render_tm_old_frontend_html(
    *,
    port: int,
    session_id: str,
    classes: List[str],
    counts: Dict[str, int],
    train_enabled: bool,
    export_enabled: bool,
    notice: str,
    train_cfg: TrainConfig,
    serial_ports: List[Dict[str, str]],
    current_serial_port: str,
    current_serial_baud: int,
    current_serial_sync: str,
    current_serial_frame_side: int,
    webcam_options: List[Dict[str, str]],
    current_webcam_index: int,
    sample_previews: Dict[str, List[Dict[str, str]]],
    initial_open_source_class: str,
    initial_open_source_kind: str,
    class_preprocess: Optional[Dict[str, Dict[str, Any]]] = None,
    sample_preprocess: Optional[Dict[str, Dict[str, Dict[str, Any]]]] = None,
    processed_previews: Optional[Dict[str, List[Dict[str, str]]]] = None,
    class_labels: Optional[Dict[str, str]] = None,
) -> None:
    import json as _json

    payload = {
        "port": int(port),
        "session": str(session_id),
        "return_target": str(st.session_state.get("tm_return_target", "home") or "home"),
        "classes": classes,
        "counts": {k: int(v) for k, v in counts.items()},
        "train_enabled": bool(train_enabled),
        "export_enabled": bool(export_enabled),
        "notice": str(notice or ""),
        "serial_ports": serial_ports,
        "current_serial_port": str(current_serial_port or ""),
        "current_serial_baud": int(current_serial_baud),
        "current_serial_sync": str(current_serial_sync or "AA 55 AA"),
        "current_serial_frame_side": int(current_serial_frame_side),
        "webcam_options": webcam_options,
        "current_webcam_index": int(current_webcam_index),
        "sample_previews": sample_previews,
        "initial_open_source_class": str(initial_open_source_class or ""),
        "initial_open_source_kind": str(initial_open_source_kind or ""),
        "class_preprocess": class_preprocess if class_preprocess is not None else {},
        "sample_preprocess": sample_preprocess if sample_preprocess is not None else {},
        "processed_previews": processed_previews if processed_previews is not None else {},
        "class_labels": class_labels if class_labels is not None else {},
        "train_cfg": {
            "img_size": int(train_cfg.img_size),
            "batch_size": int(train_cfg.batch_size),
            "epochs": int(train_cfg.epochs),
            "validation_split": float(train_cfg.validation_split),
            "learning_rate": float(train_cfg.learning_rate),
            "conv1_filters": int(train_cfg.conv1_filters),
            "conv2_filters": int(train_cfg.conv2_filters),
            "dense_units": int(train_cfg.dense_units),
            "preprocess_mode": str(getattr(train_cfg, "preprocess_mode", "auto_by_label")),
            "manual_roi": list(getattr(train_cfg, "manual_roi", []) or []) or None,
        },
    }
    debug_server_url = "http://127.0.0.1:7777/event"
    debug_session_id = "capture-webcam-source"
    try:
        for env_path in (Path(".dbg/open-project-layout.env"), Path(".dbg/open-project-page-stuck.env"), Path(".dbg/packaged-webcam-bounce.env"), Path(".dbg/capture-webcam-source.env")):
            if env_path.exists():
                for line in env_path.read_text(encoding="utf-8").splitlines():
                    if line.startswith("DEBUG_SERVER_URL="):
                        debug_server_url = line.split("=", 1)[1].strip() or debug_server_url
                    elif line.startswith("DEBUG_SESSION_ID="):
                        debug_session_id = line.split("=", 1)[1].strip() or debug_session_id
                break
    except Exception:
        pass
    payload["debug_server_url"] = debug_server_url
    payload["debug_session_id"] = debug_session_id
    data = _json.dumps(payload)
    components.html(
        f'''
<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f5f7fa;
      --card: #ffffff;
      --surface: #ffffff;
      --surface-soft: #f4f7fb;
      --surface-elevated: #ffffff;
      --surface-frost: rgba(255,255,255,0.72);
      --shadow: 0 2px 8px rgba(0,0,0,0.08);
      --blue: #1a73e8;
      --blue-bg: #e8f0fe;
      --blue-bg-hover: #d2e3fc;
      --text: #202124;
      --text-inverse: #ffffff;
      --muted: #5f6368;
      --line: #c5c8cc;
      --input-bg: #ffffff;
      --input-border: rgba(0,0,0,0.12);
      --overlay: rgba(32,33,36,0.56);
      --danger: #d93025;
      --danger-soft: #fff1f3;
      --radius: 12px;
    }}
    @media (prefers-color-scheme: dark) {{
      :root {{
        color-scheme: dark;
        --bg: #0f141a;
        --card: #1a212b;
        --surface: #18202a;
        --surface-soft: #111821;
        --surface-elevated: #202833;
        --surface-frost: rgba(24,32,42,0.88);
        --shadow: 0 10px 30px rgba(0,0,0,0.32);
        --blue: #8ab4f8;
        --blue-bg: rgba(138,180,248,0.16);
        --blue-bg-hover: rgba(138,180,248,0.24);
        --text: #e8eaed;
        --text-inverse: #0f141a;
        --muted: #a8b0ba;
        --line: rgba(255,255,255,0.12);
        --input-bg: #111821;
        --input-border: rgba(255,255,255,0.12);
        --overlay: rgba(0,0,0,0.62);
        --danger: #f28b82;
        --danger-soft: rgba(242,139,130,0.16);
      }}
    }}
    html, body {{
      height: 100%;
      margin: 0;
      font-family: system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, "Apple Color Emoji", "Segoe UI Emoji";
      color: var(--text);
      background: transparent;
    }}
    .wrap {{
      position: relative;
      background: var(--bg);
      border-radius: 12px;
      padding: 80px 18px 92px 18px;
      min-height: 620px;
      overflow-x: clip;
      overflow-y: clip;
    }}
    .topnav {{
      position: absolute;
      top: 12px;
      left: 18px;
      background: var(--card);
      box-shadow: var(--shadow);
      border-radius: 8px;
      padding: 12px 18px;
      color: var(--blue);
      font-weight: 700;
      font-size: 18px;
      display: inline-flex;
      align-items: center;
      gap: 10px;
      cursor: pointer;
      user-select: none;
      z-index: 1000;
    }}
    .navmenu {{
      position: absolute;
      top: 56px;
      left: 18px;
      width: 220px;
      background: var(--card);
      box-shadow: var(--shadow);
      border-radius: 10px;
      border: 1px solid rgba(0,0,0,0.06);
      padding: 8px;
      z-index: 1001;
      display: none;
    }}
    .navmenu.open {{ display: block; }}
    .navmenu button {{
      width: 100%;
      border: 0;
      background: transparent;
      text-align: left;
      padding: 10px 10px;
      border-radius: 8px;
      cursor: pointer;
      color: var(--text);
      font-weight: 600;
      font-size: 13px;
    }}
    .navmenu button:hover {{ background: rgba(0,0,0,0.05); }}
    .navmenu .danger {{ color: #b42318; }}
    .navmenu .divider {{
      height: 1px;
      background: rgba(0,0,0,0.08);
      margin: 6px 0;
    }}
    .layout {{
      position: relative;
      display: flex;
      width: 100%;
      box-sizing: border-box;
      justify-content: space-between;
      align-items: center;
      gap: 28px;
      z-index: 2;
    }}
    .col-classes {{ width: 500px; display: flex; flex-direction: column; gap: 24px; }}
    .col-train {{
      width: 270px;
      flex: 0 0 270px;
      min-width: 270px;
    }}
    .col-preview {{ width: 320px; }}
    @media (max-width: 1320px) {{
      .wrap {{
        padding: 72px 16px 92px 16px;
      }}
      .layout {{
        gap: 16px;
      }}
      .col-classes {{
        width: 440px;
      }}
      .col-train {{
        width: 230px;
        flex: 0 0 230px;
        min-width: 230px;
      }}
      .col-preview {{
        width: 260px;
      }}
    }}
    @media (max-width: 860px) {{
      .wrap {{
        padding: 72px 12px 92px 12px;
      }}
      .source-panel {{
        grid-template-columns: 1fr;
      }}
      .source-right {{
        border-top: 1px solid rgba(0,0,0,0.08);
      }}
    }}
    .card {{
      background: var(--card);
      border-radius: var(--radius);
      box-shadow: var(--shadow);
      border: none;
    }}
    .class-card {{ padding: 0; position: relative; overflow: hidden; }}
    .class-head {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      padding: 12px 16px;
      border-bottom: 1px solid rgba(0,0,0,0.08);
    }}
    .class-title {{
      display: inline-flex;
      align-items: center;
      gap: 10px;
      font-size: 16px;
      font-weight: 600;
    }}
    .iconbtn {{
      border: 0;
      background: transparent;
      color: var(--muted);
      width: 32px;
      height: 32px;
      border-radius: 6px;
      cursor: pointer;
      display: inline-flex;
      align-items: center;
      justify-content: center;
    }}
    .iconbtn:hover {{ background: rgba(0,0,0,0.05); color: var(--text); }}
    .more {{
      position: absolute;
      top: 8px;
      right: 8px;
    }}
    .divider {{ height: 1px; background: rgba(0,0,0,0.08); margin: 0; }}
    .subhead {{ font-size: 12px; color: var(--muted); margin: 0 0 10px 0; }}
    .btnrow {{
      display: flex;
      gap: 12px;
      padding: 10px 16px 14px 16px;
      flex-wrap: wrap;
      align-items: flex-start;
    }}
    .sample {{
      width: 80px;
      height: 80px;
      border: 0;
      border-radius: 8px;
      background: var(--blue-bg);
      color: var(--blue);
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      gap: 6px;
      font-size: 12px;
      font-weight: 600;
      cursor: pointer;
      user-select: none;
    }}
    .sample:hover {{ background: var(--blue-bg-hover); }}
    .sample svg {{ width: 22px; height: 22px; stroke: var(--blue); fill: none; stroke-width: 2; }}
    .note {{ margin-top: 12px; font-size: 12px; color: var(--muted); line-height: 1.35; }}
    .preview-pane {{
      margin-top: 14px;
      min-height: 180px;
      border-radius: 16px;
      background: rgba(26,115,232,0.07);
      display: flex;
      align-items: center;
      justify-content: center;
      overflow: hidden;
    }}
    .preview-controls {{
      display: flex;
      align-items: center;
      gap: 12px;
      margin-top: 6px;
      min-width: 0;
    }}
    .preview-controls-right {{
      display: inline-flex;
      align-items: center;
      gap: 8px;
      margin-left: auto;
      flex: 0 0 auto;
    }}
    .preview-toggle {{
      display: inline-flex;
      align-items: center;
      gap: 8px;
      font-size: 12px;
      font-weight: 700;
      color: var(--muted);
      user-select: none;
    }}
    .preview-toggle input {{
      width: 16px;
      height: 16px;
      margin: 0;
      accent-color: var(--blue);
    }}
    .preview-select {{
      border: 1px solid var(--input-border);
      border-radius: 8px;
      padding: 7px 10px;
      font-size: 12px;
      font-weight: 700;
      color: var(--text);
      background: var(--input-bg);
    }}
    .preview-pane img {{
      width: 100%;
      height: 100%;
      max-height: 260px;
      object-fit: contain;
      display: block;
      background: #f4f7fb;
    }}
    .preview-empty {{
      padding: 18px;
      text-align: center;
      color: var(--muted);
      font-size: 13px;
      line-height: 1.45;
    }}
    .preview-output {{
      margin-top: 12px;
      display: grid;
      gap: 10px;
    }}
    .out-row {{
      display: grid;
      grid-template-columns: 1fr 140px;
      gap: 10px;
      align-items: center;
      font-size: 12px;
      color: var(--muted);
      font-weight: 700;
    }}
    .out-bar {{
      height: 10px;
      border-radius: 999px;
      background: rgba(0,0,0,0.08);
      overflow: hidden;
      position: relative;
    }}
    .out-fill {{
      height: 100%;
      width: 0%;
      background: rgba(26,115,232,0.55);
      border-radius: 999px;
      transition: width 120ms linear;
    }}
    .out-pct {{
      position: absolute;
      right: 6px;
      top: 50%;
      transform: translateY(-50%);
      font-size: 10px;
      color: rgba(0,0,0,0.55);
      font-weight: 800;
      pointer-events: none;
    }}
    .addclass {{
      border: 2px dashed rgba(0,0,0,0.15);
      background: transparent;
      border-radius: 10px;
      padding: 18px;
      display: flex;
      align-items: center;
      justify-content: center;
      gap: 10px;
      color: var(--muted);
      font-weight: 600;
      cursor: pointer;
      user-select: none;
    }}
    .train-card, .preview-card {{ padding: 16px 18px 18px 18px; position: relative; }}
    .card-head {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      margin-bottom: 12px;
    }}
    .card-head h3 {{ margin: 0; font-size: 14px; font-weight: 700; }}
    .train-status {{
      width: 100%;
      border-radius: 6px;
      background: rgba(0,0,0,0.06);
      color: rgba(0,0,0,0.55);
      padding: 10px 10px;
      font-weight: 700;
      text-align: center;
      margin-bottom: 12px;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
      box-sizing: border-box;
      max-width: 100%;
      font-size: 13px;
    }}
    .train-adv-panel {{
      margin-top: 10px;
      padding-top: 8px;
      border-top: 1px solid rgba(0,0,0,0.08);
      display: none;
    }}
    .train-adv-grid {{
      display: grid;
      gap: 10px;
    }}
    .train-adv-row {{
      display: grid;
      grid-template-columns: 1fr 128px;
      gap: 12px;
      align-items: center;
      font-size: 13px;
      color: rgba(0,0,0,0.62);
      font-weight: 600;
    }}
    .train-adv-row span {{
      line-height: 1.2;
      word-break: normal;
      overflow-wrap: normal;
      white-space: normal;
    }}
    .train-adv-row input,
    .train-adv-row select {{
      width: 100%;
      box-sizing: border-box;
      border: 1px solid var(--input-border);
      border-radius: 8px;
      padding: 10px 12px;
      font-size: 15px;
      font-weight: 700;
      color: var(--blue);
      background: var(--input-bg);
      text-align: right;
      min-width: 128px;
    }}
    .trainbtn {{
      width: 100%;
      border: 0;
      border-radius: 6px;
      background: rgba(0,0,0,0.08);
      color: rgba(0,0,0,0.38);
      padding: 12px 10px;
      font-weight: 700;
      cursor: not-allowed;
      white-space: normal;
      line-height: 1.2;
    }}
    .trainbtn.enabled {{
      background: rgba(26,115,232,0.12);
      color: var(--blue);
      cursor: pointer;
    }}
    .train-progress {{
      margin-top: 10px;
      display: none;
    }}
    .train-progress-bar {{
      width: 100%;
      height: 8px;
      border-radius: 999px;
      background: rgba(0,0,0,0.08);
      overflow: hidden;
    }}
    .train-progress-fill {{
      height: 100%;
      width: 0%;
      background: rgba(26,115,232,0.55);
      border-radius: 999px;
      transition: width 120ms linear;
    }}
    .train-progress-text {{
      margin-top: 6px;
      font-size: 12px;
      color: var(--muted);
      line-height: 1.25;
    }}
    .advanced {{
      margin-top: 10px;
      color: var(--muted);
      font-size: 12px;
      display: inline-flex;
      align-items: center;
      gap: 8px;
      cursor: pointer;
      user-select: none;
      padding: 6px 8px;
      border-radius: 6px;
    }}
    .advanced:hover {{ background: rgba(0,0,0,0.04); }}
    .advanced-row {{
      margin-top: 10px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 10px;
      flex-wrap: wrap;
    }}
    .adv-reset {{
      border: 0;
      background: transparent;
      color: var(--muted);
      font-size: 12px;
      font-weight: 700;
      cursor: pointer;
      padding: 6px 8px;
      border-radius: 6px;
      user-select: none;
    }}
    .adv-reset:hover {{ background: rgba(0,0,0,0.04); color: var(--text); }}
    .exportbtn {{
      border: 0;
      border-radius: 4px;
      background: #f1f3f4;
      padding: 8px 12px;
      display: inline-flex;
      align-items: center;
      gap: 8px;
      color: var(--muted);
      font-weight: 700;
      cursor: pointer;
      user-select: none;
    }}
    .exportbtn:disabled {{
      opacity: 0.55;
      cursor: not-allowed;
    }}
    .exportbtn.blockcoding {{
      background: #eef1ff;
    }}
    .exportbtn.blockcoding svg {{ stroke: #5b5ba8; }}
    .exportbtn svg {{ width: 16px; height: 16px; stroke: var(--muted); fill: none; stroke-width: 2; }}
    .footer {{
      position: fixed;
      right: 18px;
      bottom: 12px;
      font-size: 12px;
      color: rgba(0,0,0,0.45);
      pointer-events: none;
      z-index: 1000;
    }}
    .flow {{
      position: absolute;
      inset: 0;
      width: 100%;
      height: 100%;
      z-index: 1;
      pointer-events: none;
    }}
    .flow path {{ stroke: var(--line); stroke-width: 2; fill: none; }}
    .toast {{
      position: fixed;
      left: 50%;
      top: 18px;
      transform: translateX(-50%);
      background: rgba(32,33,36,0.92);
      color: white;
      padding: 10px 14px;
      border-radius: 10px;
      font-size: 12px;
      z-index: 2001;
      max-width: 720px;
      display: none;
    }}
    .modal-backdrop {{
      position: fixed;
      inset: 0;
      background: rgba(32,33,36,0.56);
      display: none;
      align-items: center;
      justify-content: center;
      z-index: 2200;
      padding: 20px;
    }}
    .modal {{
      width: min(820px, 92vw);
      background: var(--card);
      border-radius: 14px;
      box-shadow: 0 14px 40px rgba(0,0,0,0.28);
      overflow: hidden;
    }}
    .modal-head {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 12px 14px;
      border-bottom: 1px solid rgba(0,0,0,0.08);
    }}
    .modal-head strong {{
      font-size: 14px;
      font-weight: 700;
    }}
    .modal-body {{
      padding: 12px 14px 14px 14px;
    }}
    .modal-actions {{
      display: flex;
      gap: 10px;
      justify-content: flex-end;
      margin-top: 12px;
    }}
    .btn {{
      border: 0;
      border-radius: 8px;
      padding: 10px 12px;
      font-weight: 700;
      cursor: pointer;
      user-select: none;
    }}
    .btn-secondary {{
      background: rgba(0,0,0,0.06);
      color: var(--muted);
    }}
    .btn-secondary:hover {{
      background: rgba(0,0,0,0.09);
      color: var(--text);
    }}
    .btn-primary {{
      background: var(--blue);
      color: white;
    }}
    .btn-primary:hover {{
      background: #1669d4;
    }}
    .preprocess-chip {{
      border: 0;
      border-radius: 999px;
      background: rgba(26,115,232,0.10);
      color: var(--blue);
      padding: 6px 10px;
      font-size: 11px;
      font-weight: 700;
      cursor: pointer;
    }}
    .preprocess-chip:hover {{
      background: rgba(26,115,232,0.16);
    }}
    #classPreprocessModal {{
      z-index: 2400;
      align-items: stretch;
      justify-content: stretch;
      padding: 18px;
    }}
    .class-preprocess-shell {{
      width: min(1240px, calc(100vw - 36px));
      height: min(840px, calc(100vh - 36px));
      background: var(--card);
      border-radius: 18px;
      box-shadow: 0 18px 44px rgba(0,0,0,0.28);
      overflow: hidden;
      display: grid;
      grid-template-columns: minmax(0, 1.35fr) minmax(320px, 0.9fr);
    }}
    .class-preprocess-left {{
      background: var(--blue-bg);
      padding: 18px;
      display: flex;
      flex-direction: column;
      gap: 14px;
      min-width: 0;
      min-height: 0;
    }}
    .class-preprocess-right {{
      background: var(--surface-elevated);
      padding: 18px;
      display: flex;
      flex-direction: column;
      gap: 14px;
      min-width: 0;
      min-height: 0;
      overflow: auto;
    }}
    .class-preprocess-visuals {{
      flex: 1;
      min-height: 0;
      display: grid;
      grid-template-columns: minmax(0, 1.18fr) minmax(260px, 0.82fr);
      gap: 14px;
      align-items: stretch;
    }}
    .class-preprocess-visual-pane {{
      display: flex;
      flex-direction: column;
      gap: 8px;
      min-height: 0;
    }}
    .class-preprocess-visual-title {{
      font-size: 12px;
      font-weight: 700;
      color: var(--muted);
    }}
    .class-preprocess-head {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      color: var(--blue);
      font-size: 15px;
      font-weight: 700;
    }}
    .class-preprocess-stage {{
      flex: 1;
      min-height: 0;
      border-radius: 16px;
      background: var(--surface-frost);
      position: relative;
      overflow: hidden;
      display: flex;
      align-items: center;
      justify-content: center;
      touch-action: none;
      cursor: crosshair;
    }}
    .class-preprocess-stage img {{
      width: 100%;
      height: 100%;
      object-fit: contain;
      display: block;
      user-select: none;
      -webkit-user-drag: none;
    }}
    .roi-overlay {{
      position: absolute;
      border: 2px solid #1a73e8;
      background: rgba(26,115,232,0.14);
      box-shadow: 0 0 0 1px rgba(255,255,255,0.35) inset;
      pointer-events: none;
    }}
    .class-preprocess-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(72px, 1fr));
      gap: 10px;
    }}
    .class-preprocess-thumb {{
      border: 0;
      border-radius: 10px;
      padding: 4px;
      background: rgba(26,115,232,0.08);
      cursor: pointer;
    }}
    .class-preprocess-thumb.active {{
      background: rgba(26,115,232,0.20);
      box-shadow: 0 0 0 2px rgba(26,115,232,0.22) inset;
    }}
    .class-preprocess-thumb.defined {{
      background: rgba(30, 142, 62, 0.14);
      box-shadow: 0 0 0 2px rgba(30, 142, 62, 0.28) inset;
    }}
    .class-preprocess-thumb img {{
      width: 100%;
      aspect-ratio: 1 / 1;
      object-fit: cover;
      border-radius: 8px;
      display: block;
      background: #eef3f9;
    }}
    .class-preprocess-thumb-state {{
      margin-top: 4px;
      font-size: 11px;
      font-weight: 700;
      text-align: center;
      color: var(--muted);
    }}
    .class-preprocess-thumb.defined .class-preprocess-thumb-state {{
      color: #1e8e3e;
    }}
    .preview-mode-tabs {{
      display: inline-flex;
      gap: 6px;
      padding: 4px;
      border-radius: 999px;
      background: rgba(26,115,232,0.08);
      flex: 1 1 auto;
      min-width: 0;
      overflow-x: auto;
      overflow-y: hidden;
      scrollbar-width: none;
    }}
    .preview-mode-tabs::-webkit-scrollbar {{ display: none; }}
    .preview-mode-tab {{
      border: 0;
      background: transparent;
      color: var(--muted);
      border-radius: 999px;
      padding: 6px 10px;
      font-size: 12px;
      font-weight: 700;
      cursor: pointer;
    }}
    .preview-mode-tab.active {{
      background: var(--surface-elevated);
      color: var(--blue);
      box-shadow: 0 1px 4px rgba(0,0,0,0.08);
    }}
    .class-preprocess-fields {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 10px;
    }}
    .class-preprocess-fields label,
    .class-preprocess-side label {{
      display: flex;
      flex-direction: column;
      gap: 6px;
      font-size: 12px;
      font-weight: 700;
      color: var(--muted);
    }}
    .class-preprocess-fields input,
    .class-preprocess-side input,
    .class-preprocess-side select {{
      border: 1px solid var(--input-border);
      border-radius: 10px;
      padding: 9px 10px;
      font-size: 13px;
      color: var(--text);
      background: var(--input-bg);
      width: 100%;
      box-sizing: border-box;
    }}
    .class-preprocess-info {{
      font-size: 12px;
      color: var(--muted);
      line-height: 1.45;
    }}
    .class-preprocess-current {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 10px;
    }}
    .class-preprocess-current .class-preprocess-info {{
      flex: 1;
      min-width: 0;
    }}
    .class-preprocess-result {{
      border-radius: 14px;
      background: var(--surface-soft);
      min-height: 0;
      flex: 1;
      display: flex;
      align-items: center;
      justify-content: center;
      overflow: hidden;
    }}
    .class-preprocess-mode-row {{
      display: flex;
      align-items: end;
      gap: 10px;
    }}
    .class-preprocess-mode-row label {{
      flex: 1;
      min-width: 0;
    }}
    .class-preprocess-mode-save {{
      white-space: nowrap;
      align-self: end;
    }}
    .class-preprocess-status-row {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 10px;
      flex-wrap: wrap;
    }}
    .class-preprocess-status-chip {{
      border-radius: 999px;
      padding: 4px 10px;
      font-size: 11px;
      font-weight: 700;
      background: rgba(26,115,232,0.12);
      color: var(--blue);
    }}
    .class-preprocess-status-chip.dirty {{
      background: rgba(217,119,6,0.14);
      color: #b45309;
    }}
    .class-preprocess-result img {{
      width: 100%;
      height: 100%;
      object-fit: contain;
      display: block;
    }}
    .class-preprocess-actions {{
      display: flex;
      justify-content: flex-end;
      gap: 10px;
      margin-top: auto;
    }}
    @media (max-width: 1100px) {{
      .class-preprocess-shell {{
        grid-template-columns: minmax(0, 1fr);
      }}
      .class-preprocess-visuals {{
        grid-template-columns: minmax(0, 1fr);
      }}
      .class-preprocess-left {{
        min-height: 440px;
      }}
    }}
    .source-panel {{
      margin-top: 16px;
      margin-left: 0;
      margin-right: 0;
      border-top: 1px solid rgba(0,0,0,0.08);
      display: grid;
      grid-template-columns: 1.05fr 1fr;
      min-height: 360px;
      overflow: hidden;
      border-radius: 0 0 12px 12px;
    }}
    .source-left {{
      background: var(--blue-bg);
      padding: 18px;
      display: flex;
      flex-direction: column;
      gap: 14px;
    }}
    .source-right {{
      background: var(--surface-elevated);
      padding: 22px 24px;
      display: flex;
      flex-direction: column;
      max-height: none;
      overflow: hidden;
    }}
    .source-head {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      color: var(--blue);
      font-size: 14px;
      font-weight: 700;
    }}
    .source-tools {{
      display: flex;
      align-items: center;
      gap: 8px;
    }}
    .source-preview-wrap {{
      flex: 1;
      border-radius: 12px;
      overflow: hidden;
      background: color-mix(in srgb, var(--surface-elevated) 55%, transparent);
      min-height: 220px;
      display: flex;
      align-items: center;
      justify-content: center;
    }}
    .preview-frame {{
      width: 100%;
      height: 100%;
      display: block;
      background: var(--surface-soft);
      object-fit: contain;
      image-rendering: auto;
    }}
    .source-note {{
      min-height: 18px;
      font-size: 12px;
      color: var(--muted);
    }}
    .source-actions {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      margin-top: auto;
    }}
    .source-settings {{
      color: var(--blue);
      font-size: 26px;
      line-height: 1;
      cursor: pointer;
    }}
    .source-settings-panel {{
      margin-top: 12px;
      padding: 12px;
      border-radius: 12px;
      background: color-mix(in srgb, var(--surface-elevated) 84%, transparent);
      border: 1px solid rgba(26,115,232,0.16);
      box-shadow: 0 6px 18px rgba(0,0,0,0.05);
      box-sizing: border-box;
      max-width: 100%;
    }}
    .source-settings-grid {{
      display: grid;
      gap: 10px;
      width: 100%;
    }}
    .source-settings-grid label {{
      display: grid;
      gap: 6px;
      font-size: 12px;
      color: var(--muted);
      font-weight: 600;
      width: 100%;
      min-width: 0;
      overflow-wrap: anywhere;
      word-break: break-word;
    }}
    .source-settings-grid input,
    .source-settings-grid select {{
      width: 100%;
      border: 1px solid var(--input-border);
      border-radius: 8px;
      padding: 9px 10px;
      font-size: 13px;
      color: var(--text);
      background: var(--input-bg);
      box-sizing: border-box;
      min-width: 0;
    }}
    .source-settings-actions {{
      display: flex;
      justify-content: flex-end;
      gap: 8px;
      margin-top: 10px;
    }}
    .source-settings-actions button {{
      border: 0;
      border-radius: 8px;
      padding: 8px 12px;
      font-size: 12px;
      font-weight: 700;
      cursor: pointer;
    }}
    .source-settings-save {{
      background: rgba(26,115,232,0.14);
      color: var(--blue);
    }}
    .source-settings-cancel {{
      background: rgba(0,0,0,0.06);
      color: var(--text);
    }}
    .source-right h4 {{
      margin: 0;
      font-size: 14px;
      font-weight: 700;
      color: var(--text);
    }}
    .source-count {{
      margin-top: 6px;
      font-size: 28px;
      font-weight: 700;
      color: var(--text);
    }}
    .source-count small {{
      font-size: 14px;
      font-weight: 600;
      color: var(--muted);
      margin-left: 4px;
    }}
    .device-select {{
      margin-top: 12px;
      width: 100%;
      border: 1px solid var(--input-border);
      border-radius: 8px;
      padding: 10px 12px;
      font-size: 13px;
      color: var(--text);
      background: var(--input-bg);
    }}
    .device-help {{
      margin-top: 10px;
      font-size: 12px;
      color: var(--muted);
      line-height: 1.4;
    }}
    .upload-pick {{
      width: 100%;
      min-height: 144px;
      border: 0;
      border-radius: 12px;
      background: #d6e4f8;
      color: var(--blue);
      font-size: 14px;
      font-weight: 700;
      cursor: pointer;
      padding: 18px;
      text-align: center;
    }}
    .upload-hint {{
      margin-top: 12px;
      color: #7e93b0;
      font-size: 13px;
      line-height: 1.45;
    }}
    .samples-grid {{
      margin-top: 14px;
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 12px;
      align-content: start;
      min-height: 180px;
      padding-right: 4px;
    }}
    .sample-item {{
      position: relative;
    }}
    .sample-thumb {{
      width: 100%;
      aspect-ratio: 1 / 1;
      object-fit: cover;
      border-radius: 10px;
      background: #eef2f6;
      border: 1px solid rgba(0,0,0,0.06);
    }}
    .sample-delete {{
      position: absolute;
      top: 6px;
      right: 6px;
      width: 22px;
      height: 22px;
      border: 0;
      border-radius: 999px;
      background: rgba(32, 33, 36, 0.82);
      color: var(--text-inverse);
      display: inline-flex;
      align-items: center;
      justify-content: center;
      font-size: 13px;
      line-height: 1;
      cursor: pointer;
      opacity: 0;
      pointer-events: none;
      transition: opacity 0.15s ease;
    }}
    .sample-item:hover .sample-delete,
    .sample-item:focus-within .sample-delete {{
      opacity: 1;
      pointer-events: auto;
    }}
    .sample-delete:hover {{
      background: rgba(180, 35, 24, 0.92);
    }}
    .samples-empty {{
      margin-top: 14px;
      border: 1px dashed rgba(0,0,0,0.12);
      border-radius: 12px;
      min-height: 180px;
      display: flex;
      align-items: center;
      justify-content: center;
      color: var(--muted);
      font-size: 13px;
    }}
    .summary-title {{
      padding: 10px 16px;
      font-size: 13px;
      font-weight: 700;
      color: rgba(0,0,0,0.7);
    }}
    .summary-row {{
      padding: 10px 16px 14px 16px;
      display: flex;
      gap: 12px;
      align-items: center;
    }}
    .summary-actions {{
      display: flex;
      gap: 10px;
    }}
    .summary-actions .sample {{
      width: 78px;
      height: 62px;
      flex: 0 0 78px;
      border-radius: 6px;
      gap: 4px;
      font-size: 11px;
    }}
    .summary-samples {{
      min-height: 62px;
      display: flex;
      align-items: center;
      overflow: hidden;
    }}
    .samples-strip {{
      display: flex;
      gap: 10px;
      align-items: center;
      overflow-x: auto;
      padding-bottom: 2px;
    }}
    .samples-strip .sample-thumb {{
      width: 54px;
      min-width: 54px;
      aspect-ratio: 1 / 1;
    }}
    .samples-strip .sample-item {{
      width: 54px;
      min-width: 54px;
    }}
    .samples-strip .sample-delete {{
      width: 18px;
      height: 18px;
      top: 4px;
      right: 4px;
      font-size: 11px;
    }}
    .samples-strip-empty {{
      width: 100%;
      min-height: 62px;
      border: 1px dashed rgba(0,0,0,0.12);
      border-radius: 12px;
      display: flex;
      align-items: center;
      justify-content: center;
      color: var(--muted);
      font-size: 13px;
    }}
    .form-grid {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 12px;
    }}
    .field {{
      display: flex;
      flex-direction: column;
      gap: 6px;
    }}
    .field label {{
      font-size: 12px;
      color: var(--muted);
      font-weight: 600;
    }}
    .field input {{
      border: 1px solid var(--input-border);
      border-radius: 8px;
      padding: 10px 12px;
      font-size: 13px;
      color: var(--text);
      background: var(--input-bg);
    }}
  </style>
</head>
<body>
<div class="toast" id="toast"></div>
<div class="topnav" id="goHome">☰ <span>Teachable Machine</span></div>
<div class="navmenu" id="navMenu">
  <button type="button" id="navOpenProject">Open project</button>
  <button type="button" id="navSaveProject">Save project</button>
  <button type="button" id="navExportDataset">Export dataset</button>
  <button type="button" id="navReturn">Return</button>
  <div class="divider"></div>
  <button type="button" class="danger" id="navResetProject">Reset project</button>
</div>
<div class="modal-backdrop" id="advancedModal">
  <div class="modal">
    <div class="modal-head">
      <strong>Advanced</strong>
      <button class="iconbtn" id="advancedClose" type="button">✕</button>
    </div>
    <div class="modal-body">
      <div class="form-grid">
        <div class="field"><label>Batch Size</label><input id="advBatch" type="number" min="1" step="1"/></div>
        <div class="field"><label>Epochs</label><input id="advEpochs" type="number" min="1" step="1"/></div>
        <div class="field"><label>Validation Split</label><input id="advVal" type="number" min="0.05" max="0.5" step="0.05"/></div>
        <div class="field"><label>Learning Rate</label><input id="advLr" type="number" min="0.00001" step="0.0001"/></div>
        <div class="field"><label>Conv1 Filters</label><input id="advConv1" type="number" min="1" step="1"/></div>
        <div class="field"><label>Conv2 Filters</label><input id="advConv2" type="number" min="1" step="1"/></div>
        <div class="field"><label>Dense Units</label><input id="advDense" type="number" min="1" step="1"/></div>
      </div>
      <div class="modal-actions">
        <button class="btn btn-secondary" id="advancedCancel" type="button">Close</button>
        <button class="btn btn-primary" id="advancedSave" type="button">Save</button>
      </div>
    </div>
  </div>
</div>
<div class="modal-backdrop" id="classPreprocessModal"></div>
<div class="wrap">
  <svg class="flow" viewBox="0 0 1200 700" preserveAspectRatio="none" aria-hidden="true">
    <path id="flowClass1" d="" />
    <path id="flowClass2" d="" />
    <path id="flowTrainPreview" d="" />
  </svg>
  <div class="layout">
    <div class="col-classes" id="classes"></div>
    <div class="col-train">
      <div class="card train-card" id="trainCard">
        <div class="card-head"><h3>Training</h3></div>
        <div class="train-status" id="trainStatus">Not trained</div>
        <button class="trainbtn" id="trainBtn">Train Embedded Model</button>
        <div class="train-progress" id="trainProgress">
          <div class="train-progress-bar"><div class="train-progress-fill" id="trainProgressFill"></div></div>
          <div class="train-progress-text" id="trainProgressText"></div>
        </div>
        <div class="advanced-row">
          <div class="advanced" id="advBtn">Advanced <span id="advChevron">▾</span></div>
          <button class="adv-reset" id="advReset" type="button">Reset</button>
        </div>
        <div class="train-adv-panel" id="trainAdvPanel">
          <div class="train-adv-grid">
            <div class="train-adv-row"><span>Input Size</span><input id="advImgSizeInline" type="number" min="32" max="224" step="8"/></div>
            <div class="train-adv-row"><span>Batch Size</span><input id="advBatchInline" type="number" min="1" step="1"/></div>
            <div class="train-adv-row"><span>Epochs</span><input id="advEpochsInline" type="number" min="1" step="1"/></div>
            <div class="train-adv-row"><span>Validation Split</span><input id="advValInline" type="number" min="0" max="0.5" step="0.05"/></div>
            <div class="train-adv-row"><span>Learning Rate</span><input id="advLrInline" type="number" min="0" step="0.0001"/></div>
            <div class="train-adv-row"><span>Conv1 Filters</span><input id="advConv1Inline" type="number" min="1" step="1"/></div>
            <div class="train-adv-row"><span>Conv2 Filters</span><input id="advConv2Inline" type="number" min="1" step="1"/></div>
            <div class="train-adv-row"><span>Dense Units</span><input id="advDenseInline" type="number" min="1" step="1"/></div>
          </div>
        </div>
      </div>
    </div>
    <div class="col-preview">
      <div class="card preview-card" id="previewCard">
        <div class="card-head">
          <h3>Preview</h3>
          <button class="exportbtn blockcoding" id="exportBlockBtn" title="Send the trained model to the Block Coding editor"><svg viewBox="0 0 24 24"><rect x="3" y="3" width="12" height="12" rx="0"/><path d="M9 21h12V9"/></svg><span>Export to Block Coding</span></button>
          <button class="exportbtn" id="exportBtn"><svg viewBox="0 0 24 24"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/></svg><span>Export Model</span></button>
        </div>
        <div class="preview-controls">
          <label class="preview-toggle"><input id="previewInputToggle" type="checkbox"/><span>Input</span></label>
          <label class="preview-toggle"><input id="previewRoiToggle" type="checkbox"/><span>ROI</span></label>
          <div class="preview-mode-tabs" id="previewModeTabs">
            <button class="preview-mode-tab" type="button" data-preview-mode="auto_by_label">Auto</button>
            <button class="preview-mode-tab" type="button" data-preview-mode="sign">Sign</button>
            <button class="preview-mode-tab" type="button" data-preview-mode="junction">Junction</button>
          </div>
          <div class="preview-controls-right">
            <button class="iconbtn source-settings" type="button" id="previewSettingsToggle" title="Input settings">⚙</button>
          </div>
        </div>
        <div id="previewSettingsHost"></div>
        <div class="preview-pane" id="previewPane"></div>
        <div class="preview-output" id="previewOutput"></div>
        <div class="note" id="previewNote"></div>
      </div>
    </div>
  </div>
</div>
<div class="footer">English | release-2-4-14 - 2.4.14</div>

<script>
(function() {{
  function setStageLabel(text) {{
    try {{
      var id = 'tmBootStageBadge';
      var box = document.getElementById(id);
      if (!box) {{
        box = document.createElement('div');
        box.id = id;
        box.style.position = 'fixed';
        box.style.right = '16px';
        box.style.top = '16px';
        box.style.zIndex = '99998';
        box.style.padding = '8px 10px';
        box.style.borderRadius = '999px';
        box.style.background = 'rgba(32,33,36,0.88)';
          box.style.color = themeVar('--text-inverse', '#fff');
        box.style.font = '11px/1.2 system-ui, -apple-system, Segoe UI, sans-serif';
        box.style.whiteSpace = 'pre-wrap';
        document.body.appendChild(box);
      }}
      box.textContent = 'stage: ' + String(text || 'unknown');
    }} catch (e) {{}}
  }}
  function showBootError(message) {{
    try {{
      var id = 'tmBootErrorBanner';
      var box = document.getElementById(id);
      if (!box) {{
        box = document.createElement('div');
        box.id = id;
        box.style.position = 'fixed';
        box.style.left = '16px';
        box.style.right = '16px';
        box.style.bottom = '16px';
        box.style.zIndex = '99999';
        box.style.padding = '12px 14px';
        box.style.borderRadius = '10px';
          box.style.background = themeVar('--danger-soft', '#fff1f3');
          box.style.border = `1px solid ${{themeVar('--danger', '#f28b82')}}`;
          box.style.color = themeVar('--text', '#202124');
        box.style.font = '12px/1.45 system-ui, -apple-system, Segoe UI, sans-serif';
        box.style.whiteSpace = 'pre-wrap';
        document.body.appendChild(box);
      }}
      box.textContent = 'Frontend boot error: ' + String(message || 'unknown error');
    }} catch (e) {{}}
  }}
  window.__tmStageMark = function(stage) {{
    setStageLabel(stage);
  }};
  setStageLabel('boot-probe-ready');
  window.addEventListener('error', function(event) {{
    showBootError(event && event.message ? event.message : event);
  }});
  window.addEventListener('unhandledrejection', function(event) {{
    var reason = event && event.reason ? event.reason : 'Unhandled promise rejection';
    if (reason && reason.message) reason = reason.message;
    showBootError(reason);
  }});
}})();
</script>

<script>
const STATE = {data};
const baseUrl = `http://127.0.0.1:${{STATE.port}}`;
if (window.__tmStageMark) window.__tmStageMark('script-start');
function dbgEvent(hypothesisId, location, msg, data) {{
  try {{
    fetch(STATE.debug_server_url || 'http://127.0.0.1:7777/event', {{
      method: 'POST',
      headers: {{'Content-Type':'application/json'}},
      body: JSON.stringify({{
        sessionId: String(STATE.debug_session_id || 'open-project-page-stuck'),
        runId: 'pre-fix',
        hypothesisId,
        location,
        msg,
        data,
        ts: Date.now()
      }})
    }}).catch(() => null);
  }} catch (e) {{}}
}}
function logScrollLayers(tag) {{
  try {{
    const frame = window.frameElement || null;
    const frameParent = frame && frame.parentElement ? frame.parentElement : null;
    const frameGrand = frameParent && frameParent.parentElement ? frameParent.parentElement : null;
    let parentIframeCount = -1;
    try {{
      parentIframeCount = window.parent && window.parent.document ? window.parent.document.querySelectorAll('iframe').length : -1;
    }} catch (e) {{}}
    dbgEvent('H6', 'app.py:logScrollLayers', '[DEBUG] scroll layer snapshot', {{
      tag: String(tag || ''),
      parentIframeCount,
      docEl: {{
        clientWidth: document.documentElement ? document.documentElement.clientWidth : 0,
        clientHeight: document.documentElement ? document.documentElement.clientHeight : 0,
        scrollWidth: document.documentElement ? document.documentElement.scrollWidth : 0,
        scrollHeight: document.documentElement ? document.documentElement.scrollHeight : 0,
        overflow: document.documentElement && document.documentElement.style ? String(document.documentElement.style.overflow || '') : '',
        overflowX: document.documentElement && document.documentElement.style ? String(document.documentElement.style.overflowX || '') : '',
        overflowY: document.documentElement && document.documentElement.style ? String(document.documentElement.style.overflowY || '') : '',
      }},
      body: {{
        clientWidth: document.body ? document.body.clientWidth : 0,
        clientHeight: document.body ? document.body.clientHeight : 0,
        scrollWidth: document.body ? document.body.scrollWidth : 0,
        scrollHeight: document.body ? document.body.scrollHeight : 0,
        overflow: document.body && document.body.style ? String(document.body.style.overflow || '') : '',
        overflowX: document.body && document.body.style ? String(document.body.style.overflowX || '') : '',
        overflowY: document.body && document.body.style ? String(document.body.style.overflowY || '') : '',
      }},
      frame: frame ? {{
        clientWidth: frame.clientWidth || 0,
        clientHeight: frame.clientHeight || 0,
        scrollWidth: frame.scrollWidth || 0,
        scrollHeight: frame.scrollHeight || 0,
        overflow: frame.style ? String(frame.style.overflow || '') : '',
        overflowX: frame.style ? String(frame.style.overflowX || '') : '',
        overflowY: frame.style ? String(frame.style.overflowY || '') : '',
      }} : null,
      frameParent: frameParent ? {{
        clientWidth: frameParent.clientWidth || 0,
        clientHeight: frameParent.clientHeight || 0,
        scrollWidth: frameParent.scrollWidth || 0,
        scrollHeight: frameParent.scrollHeight || 0,
        overflow: frameParent.style ? String(frameParent.style.overflow || '') : '',
        overflowX: frameParent.style ? String(frameParent.style.overflowX || '') : '',
        overflowY: frameParent.style ? String(frameParent.style.overflowY || '') : '',
      }} : null,
      frameGrand: frameGrand ? {{
        clientWidth: frameGrand.clientWidth || 0,
        clientHeight: frameGrand.clientHeight || 0,
        scrollWidth: frameGrand.scrollWidth || 0,
        scrollHeight: frameGrand.scrollHeight || 0,
        overflow: frameGrand.style ? String(frameGrand.style.overflow || '') : '',
        overflowX: frameGrand.style ? String(frameGrand.style.overflowX || '') : '',
        overflowY: frameGrand.style ? String(frameGrand.style.overflowY || '') : '',
      }} : null,
    }});
  }} catch (e) {{}}
}}
function sendStreamlitMessage(type, data) {{
  if (window.__tmNavigatingAway) return;
  try {{
    window.parent.postMessage(Object.assign({{
      isStreamlitMessage: true,
      type: type,
    }}, data || {{}}), '*');
  }} catch (e) {{}}
}}
function initStreamlitFrame() {{
  if (window.__tmNavigatingAway) return;
  if (window.__tmStreamlitReady) return;
  window.__tmStreamlitReady = true;
  sendStreamlitMessage('streamlit:componentReady', {{apiVersion: 1}});
}}
let frameHeightRaf = 0;
let layoutResyncTimers = [];
let mountReflowTimers = [];
let homeNavigateTimer = 0;
function syncFrameHeight() {{
  if (window.__tmNavigatingAway) return;
  initStreamlitFrame();
  let nextHeight = 0;
  let nextWidth = 0;
  let metrics = {{}};
  try {{
    const body = document.body;
    const doc = document.documentElement;
    const wrap = document.querySelector('.wrap');
    nextHeight = Math.max(
      body ? body.scrollHeight : 0,
      body ? body.offsetHeight : 0,
      doc ? doc.scrollHeight : 0,
      doc ? doc.offsetHeight : 0,
      doc ? doc.clientHeight : 0,
    );
    nextWidth = Math.max(
      body ? body.scrollWidth : 0,
      body ? body.offsetWidth : 0,
      doc ? doc.scrollWidth : 0,
      doc ? doc.offsetWidth : 0,
      doc ? doc.clientWidth : 0,
      wrap ? Math.round(wrap.getBoundingClientRect().width) : 0,
    );
    metrics = {{
      nextHeight,
      nextWidth,
      bodyScrollHeight: body ? body.scrollHeight : 0,
      bodyOffsetHeight: body ? body.offsetHeight : 0,
      bodyScrollWidth: body ? body.scrollWidth : 0,
      bodyOffsetWidth: body ? body.offsetWidth : 0,
      docScrollHeight: doc ? doc.scrollHeight : 0,
      docOffsetHeight: doc ? doc.offsetHeight : 0,
      docScrollWidth: doc ? doc.scrollWidth : 0,
      docOffsetWidth: doc ? doc.offsetWidth : 0,
      docClientHeight: doc ? doc.clientHeight : 0,
      docClientWidth: doc ? doc.clientWidth : 0,
      wrapWidth: wrap ? Math.round(wrap.getBoundingClientRect().width) : 0,
      wrapHeight: wrap ? Math.round(wrap.getBoundingClientRect().height) : 0,
      windowInnerHeight: window.innerHeight || 0,
      windowInnerWidth: window.innerWidth || 0,
      classCount: Array.isArray(STATE.classes) ? STATE.classes.length : 0,
    }};
  }} catch (e) {{}}
  if (!nextHeight || !Number.isFinite(nextHeight)) return;
  try {{
    const frame = window.frameElement;
    if (frame && frame.style) {{
      frame.style.height = `${{Math.ceil(nextHeight + 12)}}px`;
      frame.style.minHeight = `${{Math.ceil(nextHeight + 12)}}px`;
      frame.style.width = '100%';
      frame.style.maxWidth = '100%';
      if (frame.parentElement && frame.parentElement.style) {{
        frame.parentElement.style.width = '100%';
        frame.parentElement.style.maxWidth = '100%';
        frame.parentElement.style.height = `${{Math.ceil(nextHeight + 12)}}px`;
        frame.parentElement.style.minHeight = `${{Math.ceil(nextHeight + 12)}}px`;
        frame.parentElement.style.overflow = 'hidden';
      }}
    }}
    if (nextWidth && Number.isFinite(nextWidth) && document.body && document.body.style) {{
      document.body.style.width = '100%';
      document.body.style.maxWidth = '100%';
    }}
  }} catch (e) {{}}
  // #region debug-point A:sync-frame-height
  dbgEvent('A', 'app.py:syncFrameHeight', '[DEBUG] syncFrameHeight posting iframe height', metrics);
  // #endregion
  sendStreamlitMessage('streamlit:setFrameHeight', {{height: Math.ceil(nextHeight + 12)}});
}}
function queueFrameHeightSync() {{
  if (window.__tmNavigatingAway) return;
  if (frameHeightRaf) {{
    try {{ cancelAnimationFrame(frameHeightRaf); }} catch (e) {{}}
  }}
  frameHeightRaf = requestAnimationFrame(() => {{
    frameHeightRaf = 0;
    syncFrameHeight();
  }});
}}
function scheduleLayoutResync() {{
  if (window.__tmNavigatingAway) return;
  // #region debug-point B:schedule-layout-resync
  dbgEvent('B', 'app.py:scheduleLayoutResync', '[DEBUG] scheduleLayoutResync queued', {{
    classCount: Array.isArray(STATE.classes) ? STATE.classes.length : 0,
    sampleCounts: Object.assign({{}}, STATE.counts || {{}}),
    previewReady: !!document.getElementById('previewImage'),
    windowInnerHeight: window.innerHeight || 0,
  }});
  // #endregion
  queueFrameHeightSync();
  try {{
    window.requestAnimationFrame(() => {{
      if (window.__tmNavigatingAway) return;
      updateFlow();
      queueFrameHeightSync();
      const t120 = window.setTimeout(() => {{
        if (window.__tmNavigatingAway) return;
        updateFlow();
        queueFrameHeightSync();
      }}, 120);
      const t320 = window.setTimeout(() => {{
        if (window.__tmNavigatingAway) return;
        updateFlow();
        queueFrameHeightSync();
      }}, 320);
      layoutResyncTimers.push(t120, t320);
    }});
  }} catch (e) {{}}
}}
function bindLayoutImageObservers(scope) {{
  const root = scope || document;
  if (!root || !root.querySelectorAll) return;
  root.querySelectorAll('img.sample-thumb, #previewImage').forEach((img) => {{
    if (!img || img.dataset.layoutBound === '1') return;
    img.dataset.layoutBound = '1';
    const onDone = () => scheduleLayoutResync();
    img.addEventListener('load', onDone, {{once: true}});
    img.addEventListener('error', onDone, {{once: true}});
  }});
}}
function requestShellLayoutRefresh(reason) {{
  if (window.__tmNavigatingAway) return;
  // #region debug-point C:request-shell-layout-refresh
  dbgEvent('C', 'app.py:requestShellLayoutRefresh', '[DEBUG] requestShellLayoutRefresh called', {{
    reason: String(reason || ''),
    hasPywebviewWindow: !!(window && window.pywebview && window.pywebview.api),
    hasPywebviewParent: !!(window.parent && window.parent.pywebview && window.parent.pywebview.api),
    hasPywebviewTop: !!(window.top && window.top.pywebview && window.top.pywebview.api),
    innerWidth: window.innerWidth || 0,
    innerHeight: window.innerHeight || 0,
  }});
  // #endregion
  try {{
    const candidates = [window, window.parent, window.top];
    for (const target of candidates) {{
      if (target && target.pywebview && target.pywebview.api && typeof target.pywebview.api.request_reflow === 'function') {{
        target.pywebview.api.request_reflow(String(reason || 'ui-refresh'));
        return;
      }}
    }}
  }} catch (e) {{}}
}}
let openSourceClass = STATE.initial_open_source_class || '';
let openSourceKind = STATE.initial_open_source_kind || '';
// #region debug-point B:initial-open-source
dbgEvent('B', 'app.py:initialOpenSource', '[DEBUG] initial open source from payload', {{
  session: String(STATE.session || ''),
  openSourceClass,
  openSourceKind,
}});
// #endregion
let previewTimer = null;
let previewBlobUrl = '';
let previewRequestInFlight = false;
let captureInFlight = false;
let holdRecording = false;
let holdRecordClass = '';
let holdRecordSource = '';
let holdResumePreviewPredict = false;
let holdPreviewSourceBeforeCapture = 'webcam';
let holdSyncTimer = null;
let holdSeq = 0;
let holdNextToken = 0;
let sourceSwitchInFlight = false;
let sourceSwitchClass = '';
let sourceSwitchKind = '';
let trainInFlight = false;
let trainPollToken = 0;
let sourceSettingsOpen = false;
let previewIntervalMs = 80;
let currentSerialPort = STATE.current_serial_port || '';
let currentWebcamIndex = Number(STATE.current_webcam_index || 0);
let currentSerialBaud = Number(STATE.current_serial_baud || 921600);
let currentSerialSync = String(STATE.current_serial_sync || 'AA 55 AA');
let currentSerialFrameSide = Number(STATE.current_serial_frame_side || 96);
let previewInputOn = false;
let previewSource = 'webcam';
let previewPreprocessMode = 'auto_by_label';
let previewShowRoi = false;
let previewUploadImageSrc = '';
let previewUploadImageB64 = '';
let previewUploadFilename = '';
let previewPredictTimer = null;
let previewPredictInFlight = false;
let previewPredictToken = 0;
let previewSettingsOpen = false;
let classPreprocessOpen = false;
let classPreprocessClass = '';
let classPreprocessDraft = null;
let classPreprocessSampleIndex = 0;
let classPreprocessProcessedSrc = '';
let classPreprocessBusy = false;
let classPreprocessScrollTop = 0;
let classPreprocessDirty = false;
let classPreprocessAbort = null;
let classPreprocessDebounce = null;
let navMenuOpen = false;
let navMenuBound = false;
let resizeBound = false;
STATE.counts = STATE.counts || {{}};
STATE.sample_previews = STATE.sample_previews || {{}};
STATE.class_preprocess = STATE.class_preprocess || {{}};
STATE.sample_preprocess = STATE.sample_preprocess || {{}};
STATE.processed_previews = STATE.processed_previews || {{}};
STATE.class_labels = STATE.class_labels || {{}};
const openSourceStorageKey = `tm-open-source-${{STATE.session}}`;
const trainCfgStorageKey = `tm-train-cfg-${{STATE.session}}`;
const previewStorageKey = `tm-preview-${{STATE.session}}`;
const exportDirStorageKey = `tm-export-dir-${{STATE.session}}`;
const exportNameStorageKey = `tm-export-name-${{STATE.session}}`;
const exportArrayStorageKey = `tm-export-array-${{STATE.session}}`;
let exportDir = '';
let exportModelName = 'model';
let exportArrayName = 'g_model';
function readOpenSourceStorage() {{
  try {{
    const raw = window.localStorage.getItem(openSourceStorageKey);
    if (!raw) return null;
    const data = JSON.parse(raw);
    if (!data || !data.className || !data.kind) return null;
    return {{
      className: String(data.className || ''),
      kind: String(data.kind || '')
    }};
  }} catch (e) {{
    return null;
  }}
}}
function persistOpenSourceState() {{
  try {{
    if (!openSourceClass || !openSourceKind) {{
      window.localStorage.removeItem(openSourceStorageKey);
      return;
    }}
    window.localStorage.setItem(openSourceStorageKey, JSON.stringify({{
      className: openSourceClass,
      kind: openSourceKind
    }}));
  }} catch (e) {{}}
}}
function clearOpenSourceState() {{
  try {{
    window.localStorage.removeItem(openSourceStorageKey);
  }} catch (e) {{}}
}}
if ((!openSourceClass || !openSourceKind) && typeof window !== 'undefined' && window.localStorage) {{
  const restoredOpenSource = readOpenSourceStorage();
  if (restoredOpenSource) {{
    openSourceClass = restoredOpenSource.className;
    openSourceKind = restoredOpenSource.kind;
    // #region debug-point B:restore-open-source-storage
    dbgEvent('B', 'app.py:restoreOpenSourceStorage', '[DEBUG] restored open source from localStorage', {{
      session: String(STATE.session || ''),
      openSourceClass,
      openSourceKind,
    }});
    // #endregion
  }}
}}
if (window.__tmStageMark) window.__tmStageMark('open-source-restored');
function restoreTrainCfgFromStorage() {{
  try {{
    const raw = window.localStorage.getItem(trainCfgStorageKey);
    if (!raw) return;
    const data = JSON.parse(raw);
    if (!data || typeof data !== 'object') return;
    const prev = Object.assign({{}}, STATE.train_cfg || {{}});
    const next = Object.assign({{}}, prev);
    const clampInt = (v, lo, hi, fallback) => {{
      const n = parseInt(String(v), 10);
      if (!isFinite(n)) return fallback;
      return Math.max(lo, Math.min(hi, n));
    }};
    const clampFloat = (v, lo, hi, fallback) => {{
      const n = Number(v);
      if (!isFinite(n)) return fallback;
      return Math.max(lo, Math.min(hi, n));
    }};
    if ('img_size' in data) next.img_size = clampInt(data.img_size, 32, 224, nullish(prev.img_size, 96));
    if ('batch_size' in data) next.batch_size = clampInt(data.batch_size, 1, 512, nullish(prev.batch_size, 32));
    if ('epochs' in data) next.epochs = clampInt(data.epochs, 1, 1000, nullish(prev.epochs, 20));
    if ('validation_split' in data) next.validation_split = clampFloat(data.validation_split, 0, 0.5, nullish(prev.validation_split, 0.25));
    if ('learning_rate' in data) next.learning_rate = clampFloat(data.learning_rate, 0.000001, 1.0, nullish(prev.learning_rate, 0.0016));
    if ('conv1_filters' in data) next.conv1_filters = clampInt(data.conv1_filters, 1, 256, nullish(prev.conv1_filters, 8));
    if ('conv2_filters' in data) next.conv2_filters = clampInt(data.conv2_filters, 1, 512, nullish(prev.conv2_filters, 16));
    if ('dense_units' in data) next.dense_units = clampInt(data.dense_units, 1, 2048, nullish(prev.dense_units, 32));
    if (data.preprocess_mode === 'auto_by_label' || data.preprocess_mode === 'manual_roi' || data.preprocess_mode === 'none') {{
      next.preprocess_mode = String(data.preprocess_mode);
    }}
    if (Array.isArray(data.manual_roi) && data.manual_roi.length === 4) {{
      next.manual_roi = data.manual_roi.map((v, idx) => clampFloat(v, 0, 1, idx < 2 ? 0 : 1));
    }}
    STATE.train_cfg = next;
  }} catch (e) {{}}
}}
function persistTrainCfgStorage() {{
  try {{
    window.localStorage.setItem(trainCfgStorageKey, JSON.stringify(STATE.train_cfg || {{}}));
  }} catch (e) {{}}
}}
function normalizeClassPreprocessConfig(raw) {{
  const src = raw && typeof raw === 'object' ? raw : {{}};
  const mode = ['auto_by_label', 'sign', 'junction', 'manual_roi', 'none'].includes(String(src.mode || ''))
    ? String(src.mode)
    : 'auto_by_label';
  let manual = null;
  if (Array.isArray(src.manual_roi) && src.manual_roi.length === 4) {{
    manual = src.manual_roi.map((v, idx) => {{
      const n = Number(v);
      if (!isFinite(n)) return idx < 2 ? 0 : 1;
      return Math.max(0, Math.min(1, n));
    }});
    manual[2] = Math.max(manual[2], manual[0] + 0.01);
    manual[3] = Math.max(manual[3], manual[1] + 0.01);
    manual[0] = Math.min(manual[0], manual[2] - 0.01);
    manual[1] = Math.min(manual[1], manual[3] - 0.01);
  }}
  return {{ mode, manual_roi: manual }};
}}
function normalizeClassPreprocessMap(raw) {{
  const src = raw && typeof raw === 'object' ? raw : {{}};
  const out = {{}};
  for (const [className, cfg] of Object.entries(src)) {{
    const key = String(className || '').trim();
    if (!key) continue;
    out[key] = normalizeClassPreprocessConfig(cfg);
  }}
  return out;
}}
function getClassPreprocessConfig(className) {{
  const all = STATE.class_preprocess && typeof STATE.class_preprocess === 'object' ? STATE.class_preprocess : {{}};
  return normalizeClassPreprocessConfig(all[className]);
}}
function setClassPreprocessConfig(className, cfg) {{
  const next = Object.assign({{}}, STATE.class_preprocess || {{}});
  next[className] = normalizeClassPreprocessConfig(cfg);
  STATE.class_preprocess = next;
}}
function getClassLabel(className) {{
  const labels = STATE.class_labels && typeof STATE.class_labels === 'object' ? STATE.class_labels : {{}};
  return String(labels[String(className || '')] || 'sign');
}}
function setClassLabel(className, value) {{
  const next = Object.assign({{}}, STATE.class_labels || {{}});
  const v = String(value || 'sign').toLowerCase();
  if (v === 'road' || v === 'sign') {{
    next[String(className || '')] = v;
  }} else {{
    delete next[String(className || '')];
  }}
  STATE.class_labels = next;
  render();
  // Persist to backend so labels survive page reload / Streamlit reruns.
  fetch(`${{baseUrl}}/classes/save_labels`, {{
    method: 'POST',
    headers: {{'Content-Type':'application/json'}},
    body: JSON.stringify({{session: STATE.session, class_labels: Object.assign({{}}, STATE.class_labels || {{}})}})
  }}).catch(() => {{}});
}}
function normalizeSamplePreprocessMap(raw) {{
  const src = raw && typeof raw === 'object' ? raw : {{}};
  const out = {{}};
  for (const [className, value] of Object.entries(src)) {{
    if (!value || typeof value !== 'object') continue;
    const classKey = String(className || '').trim();
    if (!classKey) continue;
    const classMap = {{}};
    for (const [filename, cfg] of Object.entries(value)) {{
      const fileKey = String(filename || '').trim();
      if (!fileKey) continue;
      classMap[fileKey] = normalizeClassPreprocessConfig(cfg);
    }}
    if (Object.keys(classMap).length) out[classKey] = classMap;
  }}
  return out;
}}
function getSampleOwnPreprocessConfig(className, filename) {{
  const map = normalizeSamplePreprocessMap(STATE.sample_preprocess || {{}});
  const classMap = map[String(className || '')] || {{}};
  return normalizeClassPreprocessConfig(classMap[String(filename || '')] || null);
}}
function sampleHasOwnPreprocessConfig(className, filename) {{
  const map = STATE.sample_preprocess && typeof STATE.sample_preprocess === 'object' ? STATE.sample_preprocess : {{}};
  const classMap = map[String(className || '')];
  if (!classMap || typeof classMap !== 'object') return false;
  return Object.prototype.hasOwnProperty.call(classMap, String(filename || ''));
}}
function getSampleEffectivePreprocessConfig(className, filename) {{
  if (sampleHasOwnPreprocessConfig(className, filename)) {{
    return getSampleOwnPreprocessConfig(className, filename);
  }}
  return getClassPreprocessConfig(className);
}}
function samePreprocessConfig(a, b) {{
  const left = normalizeClassPreprocessConfig(a);
  const right = normalizeClassPreprocessConfig(b);
  if (String(left.mode || '') !== String(right.mode || '')) return false;
  const lroi = Array.isArray(left.manual_roi) ? left.manual_roi : null;
  const rroi = Array.isArray(right.manual_roi) ? right.manual_roi : null;
  if (!lroi && !rroi) return true;
  if (!lroi || !rroi || lroi.length !== 4 || rroi.length !== 4) return false;
  for (let i = 0; i < 4; i += 1) {{
    if (Math.abs(Number(lroi[i] || 0) - Number(rroi[i] || 0)) > 1e-6) return false;
  }}
  return true;
}}
function setSamplePreprocessConfig(className, filename, cfg) {{
  const name = String(className || '').trim();
  const file = String(filename || '').trim();
  if (!name || !file) return;
  const next = normalizeSamplePreprocessMap(STATE.sample_preprocess || {{}});
  const classDefault = getClassPreprocessConfig(name);
  const normalized = normalizeClassPreprocessConfig(cfg);
  if (samePreprocessConfig(normalized, classDefault)) {{
    if (next[name] && typeof next[name] === 'object') {{
      delete next[name][file];
      if (!Object.keys(next[name]).length) delete next[name];
    }}
  }} else {{
    if (!next[name] || typeof next[name] !== 'object') next[name] = {{}};
    next[name][file] = normalized;
  }}
  STATE.sample_preprocess = next;
}}
function deleteSamplePreprocessConfig(className, filename) {{
  const next = normalizeSamplePreprocessMap(STATE.sample_preprocess || {{}});
  const name = String(className || '').trim();
  const file = String(filename || '').trim();
  if (next[name] && typeof next[name] === 'object') {{
    delete next[name][file];
    if (!Object.keys(next[name]).length) delete next[name];
  }}
  STATE.sample_preprocess = next;
}}
function selectedClassSample(className) {{
  const items = normalizePreviewList((STATE.sample_previews && STATE.sample_previews[className]) || []);
  if (!items.length) return null;
  const idx = Math.max(0, Math.min(items.length - 1, Number(classPreprocessSampleIndex || 0)));
  classPreprocessSampleIndex = idx;
  return items[idx];
}}
function selectedClassSampleFilename(className) {{
  const sample = selectedClassSample(className);
  return sample ? String(previewFilename(sample) || '') : '';
}}
function classPreprocessModeLabel(mode) {{
  if (mode === 'sign') return 'Sign ROI';
  if (mode === 'junction') return 'Junction ROI';
  if (mode === 'manual_roi') return 'Manual ROI';
  if (mode === 'none') return 'Full Frame';
  return 'Auto';
}}
function samplePreprocessStatus(className, filename) {{
  const cfg = normalizeClassPreprocessConfig(getSampleEffectivePreprocessConfig(className, filename));
  return classPreprocessModeLabel(cfg.mode);
}}
function currentClassPreprocessDirty() {{
  if (!classPreprocessClass) return false;
  const filename = selectedClassSampleFilename(classPreprocessClass);
  if (!filename) return false;
  return !samePreprocessConfig(
    classPreprocessDraft,
    getSampleEffectivePreprocessConfig(classPreprocessClass, filename)
  );
}}
function maybeDiscardClassPreprocessDraft() {{
  classPreprocessDirty = currentClassPreprocessDirty();
  if (!classPreprocessDirty) return true;
  return window.confirm('This sample has unsaved ROI changes. Discard them?');
}}
  async function saveCurrentClassPreprocessSample(closeAfterSave = false) {{
  const filename = selectedClassSampleFilename(classPreprocessClass);
    if (!classPreprocessClass || !filename) return false;
    const sampleConfig = normalizeClassPreprocessConfig(classPreprocessDraft);
    const res = await fetch(`${{baseUrl}}/preprocess/save_sample`, {{
      method: 'POST',
      headers: {{'Content-Type':'application/json'}},
      body: JSON.stringify({{
        session: STATE.session,
        class: classPreprocessClass,
        filename,
        sample_config: sampleConfig,
        class_labels: Object.assign({{}}, STATE.class_labels || {{}})
      }})
    }});
    const data = await res.json().catch(() => ({{ok:'0'}}));
    if (!res.ok || data.ok !== '1') {{
      throw new Error(data.error || 'Unable to save sample preprocess.');
    }}
    STATE.sample_preprocess = normalizeSamplePreprocessMap(data.sample_preprocess || STATE.sample_preprocess || {{}});
    classPreprocessDraft = sampleConfig;
    classPreprocessDirty = false;
    render();
  if (closeAfterSave) {{
    closeClassPreprocessEditor(true);
      return true;
  }}
  renderClassPreprocessModal();
  refreshClassProcessedPreview();
    return true;
}}
function commitCurrentClassPreprocessDraftIfNeeded() {{
  if (!classPreprocessOpen || !classPreprocessClass) return false;
  const filename = selectedClassSampleFilename(classPreprocessClass);
  if (!filename || !currentClassPreprocessDirty()) return false;
  setSamplePreprocessConfig(classPreprocessClass, filename, classPreprocessDraft);
  classPreprocessDirty = false;
  renderClassPreprocessModal();
  return true;
}}
function getClassPreprocessGridColumnCount() {{
  const grid = document.querySelector('#classPreprocessModal .class-preprocess-grid');
  if (!grid) return 1;
  try {{
    const cols = String(window.getComputedStyle(grid).gridTemplateColumns || '')
      .split(' ')
      .map((part) => String(part || '').trim())
      .filter(Boolean);
    return Math.max(1, cols.length || 1);
  }} catch (e) {{
    return 1;
  }}
}}
function rememberClassPreprocessScroll() {{
  const pane = document.getElementById('classPreprocessRightPane');
  if (!pane) return;
  classPreprocessScrollTop = Number(pane.scrollTop || 0);
}}
function restoreClassPreprocessScroll() {{
  const pane = document.getElementById('classPreprocessRightPane');
  if (!pane) return;
  pane.scrollTop = Math.max(0, Number(classPreprocessScrollTop || 0));
}}
async function refreshClassProcessedPreview() {{
  if (!classPreprocessOpen || !classPreprocessClass) return;
  const sample = selectedClassSample(classPreprocessClass);
  if (!sample) {{
    classPreprocessProcessedSrc = '';
    renderClassPreprocessModal();
    return;
  }}
  // Cancel any previous in-flight request so rapid changes don't stack up.
  if (classPreprocessAbort) {{
    classPreprocessAbort.abort();
    classPreprocessAbort = null;
  }}
  const controller = new AbortController();
  classPreprocessAbort = controller;
  classPreprocessBusy = true;
  renderClassPreprocessModal();
  try {{
    const res = await fetch(`${{baseUrl}}/preprocess/preview`, {{
      method: 'POST',
      headers: {{'Content-Type':'application/json'}},
      body: JSON.stringify({{
        session: STATE.session,
        class: classPreprocessClass,
        filename: String(previewFilename(sample) || ''),
        class_config: getClassPreprocessConfig(classPreprocessClass),
        sample_config: normalizeClassPreprocessConfig(classPreprocessDraft),
        class_labels: Object.assign({{}}, STATE.class_labels || {{}})
      }}),
      signal: controller.signal
    }});
    const data = await res.json().catch(() => ({{ok:'0'}}));
    if (!res.ok || data.ok !== '1') throw new Error(data.error || 'Unable to render preprocess preview.');
    classPreprocessProcessedSrc = data.image_b64 ? `data:image/png;base64,${{data.image_b64}}` : '';
  }} catch (err) {{
    if (err && err.name === 'AbortError') return;  // silently ignore aborted requests
    classPreprocessProcessedSrc = '';
    toast(String(err && err.message ? err.message : err));
  }} finally {{
    if (classPreprocessAbort === controller) classPreprocessAbort = null;
    classPreprocessBusy = false;
    renderClassPreprocessModal();
  }}
}}
function debouncedRefreshClassProcessedPreview(delayMs = 150) {{
  if (classPreprocessDebounce) clearTimeout(classPreprocessDebounce);
  classPreprocessDebounce = setTimeout(() => {{
    classPreprocessDebounce = null;
    refreshClassProcessedPreview();
  }}, delayMs);
}}
function openClassPreprocessEditor(className) {{
  classPreprocessClass = String(className || '');
  classPreprocessSampleIndex = 0;
  classPreprocessDraft = normalizeClassPreprocessConfig(
    getSampleEffectivePreprocessConfig(classPreprocessClass, selectedClassSampleFilename(classPreprocessClass))
  );
  classPreprocessProcessedSrc = '';
  classPreprocessDirty = false;
  classPreprocessOpen = true;
  renderClassPreprocessModal();
  refreshClassProcessedPreview();
}}
function closeClassPreprocessEditor(force = false) {{
  if (!force && !maybeDiscardClassPreprocessDraft()) return;
  if (classPreprocessAbort) {{
    classPreprocessAbort.abort();
    classPreprocessAbort = null;
  }}
  if (classPreprocessDebounce) {{
    clearTimeout(classPreprocessDebounce);
    classPreprocessDebounce = null;
  }}
  classPreprocessOpen = false;
  classPreprocessClass = '';
  classPreprocessDraft = null;
  classPreprocessProcessedSrc = '';
  classPreprocessBusy = false;
  classPreprocessDirty = false;
  renderClassPreprocessModal();
}}
  function getClassPreprocessImageContentRect() {{
    const stage = document.getElementById('classPreprocessStage');
    const img = document.getElementById('classPreprocessImage');
    if (!stage || !img) return null;
    const stageRect = stage.getBoundingClientRect();
    if (!stageRect.width || !stageRect.height) return null;
    const naturalW = Number(img.naturalWidth || 0);
    const naturalH = Number(img.naturalHeight || 0);
    if (naturalW <= 0 || naturalH <= 0) {{
      return {{
        stageLeft: stageRect.left,
        stageTop: stageRect.top,
        stageWidth: stageRect.width,
        stageHeight: stageRect.height,
        left: stageRect.left,
        top: stageRect.top,
        width: stageRect.width,
        height: stageRect.height,
        offsetX: 0,
        offsetY: 0,
      }};
    }}
    const scale = Math.min(stageRect.width / naturalW, stageRect.height / naturalH);
    const width = naturalW * scale;
    const height = naturalH * scale;
    const offsetX = (stageRect.width - width) * 0.5;
    const offsetY = (stageRect.height - height) * 0.5;
    return {{
      stageLeft: stageRect.left,
      stageTop: stageRect.top,
      stageWidth: stageRect.width,
      stageHeight: stageRect.height,
      left: stageRect.left + offsetX,
      top: stageRect.top + offsetY,
      width,
      height,
      offsetX,
      offsetY,
    }};
  }}
  function normalizeClassPreprocessClientPoint(clientX, clientY, allowClamp = true) {{
    const rect = getClassPreprocessImageContentRect();
    if (!rect || !rect.width || !rect.height) return null;
    const inside = clientX >= rect.left && clientX <= (rect.left + rect.width)
      && clientY >= rect.top && clientY <= (rect.top + rect.height);
    if (!inside && !allowClamp) return null;
    const px = allowClamp ? Math.max(rect.left, Math.min(rect.left + rect.width, clientX)) : clientX;
    const py = allowClamp ? Math.max(rect.top, Math.min(rect.top + rect.height, clientY)) : clientY;
    return {{
      x: Math.max(0, Math.min(1, (px - rect.left) / rect.width)),
      y: Math.max(0, Math.min(1, (py - rect.top) / rect.height)),
    }};
  }}
  function applyClassPreprocessMode(mode) {{
    const nextMode = ['auto_by_label', 'sign', 'junction', 'manual_roi', 'none'].includes(String(mode || ''))
      ? String(mode)
      : 'auto_by_label';
    rememberClassPreprocessScroll();
    const next = normalizeClassPreprocessConfig(Object.assign({{}}, classPreprocessDraft || {{}}, {{mode: nextMode}}));
    if (!next.manual_roi) next.manual_roi = [0, 0, 1, 1];
    classPreprocessDraft = next;
    classPreprocessDirty = true;
    renderClassPreprocessModal();
    refreshClassProcessedPreview();
  }}
  function moveClassPreprocessSample(delta) {{
    if (!classPreprocessOpen || !classPreprocessClass) return;
    const items = normalizePreviewList((STATE.sample_previews && STATE.sample_previews[classPreprocessClass]) || []);
    if (!items.length) return;
    if (!maybeDiscardClassPreprocessDraft()) return;
    rememberClassPreprocessScroll();
    const nextIndex = Math.max(0, Math.min(items.length - 1, Number(classPreprocessSampleIndex || 0) + Number(delta || 0)));
    if (nextIndex === Number(classPreprocessSampleIndex || 0)) return;
    classPreprocessSampleIndex = nextIndex;
    classPreprocessDraft = normalizeClassPreprocessConfig(
      getSampleEffectivePreprocessConfig(classPreprocessClass, selectedClassSampleFilename(classPreprocessClass))
    );
    classPreprocessDirty = false;
    renderClassPreprocessModal();
    refreshClassProcessedPreview();
  }}
function updateClassPreprocessOverlay(roi) {{
  const overlay = document.getElementById('classPreprocessOverlay');
  if (!overlay) return;
    const valid = Array.isArray(roi) && roi.length === 4 && classPreprocessDraft && String(classPreprocessDraft.mode || 'auto_by_label') === 'manual_roi';
  if (!valid) {{
    overlay.style.display = 'none';
    return;
  }}
    const rect = getClassPreprocessImageContentRect();
    if (!rect || !rect.width || !rect.height) {{
      overlay.style.display = 'none';
      return;
    }}
  const x1 = Math.max(0, Math.min(1, Number(roi[0] || 0)));
  const y1 = Math.max(0, Math.min(1, Number(roi[1] || 0)));
  const x2 = Math.max(x1, Math.min(1, Number(roi[2] || x1)));
  const y2 = Math.max(y1, Math.min(1, Number(roi[3] || y1)));
  overlay.style.display = 'block';
    overlay.style.left = `${{rect.offsetX + x1 * rect.width}}px`;
    overlay.style.top = `${{rect.offsetY + y1 * rect.height}}px`;
    overlay.style.width = `${{Math.max(0, (x2 - x1) * rect.width)}}px`;
    overlay.style.height = `${{Math.max(0, (y2 - y1) * rect.height)}}px`;
}}
function syncClassPreprocessRoiInputs(roi) {{
  if (!Array.isArray(roi) || roi.length !== 4) return;
  const ids = ['classPreprocessX1', 'classPreprocessY1', 'classPreprocessX2', 'classPreprocessY2'];
  ids.forEach((id, idx) => {{
    const el = document.getElementById(id);
    if (!el) return;
    el.value = Number(roi[idx] || 0).toFixed(2);
  }});
}}
function bindClassPreprocessStage() {{
  const stage = document.getElementById('classPreprocessStage');
  const img = document.getElementById('classPreprocessImage');
  if (!stage || !img) return;
  let dragStart = null;
  const applyPoint = (clientX, clientY, finalize = false) => {{
    if (!dragStart || !classPreprocessDraft || String(classPreprocessDraft.mode || 'auto_by_label') !== 'manual_roi') return;
      const start = normalizeClassPreprocessClientPoint(dragStart.x, dragStart.y, true);
      const end = normalizeClassPreprocessClientPoint(clientX, clientY, true);
      if (!start || !end) return;
      const x1 = start.x;
      const y1 = start.y;
      const x2 = end.x;
      const y2 = end.y;
    const nextRoi = [
      Math.min(x1, x2),
      Math.min(y1, y2),
      Math.max(x1, x2),
      Math.max(y1, y2),
    ];
    classPreprocessDraft.manual_roi = nextRoi;
    classPreprocessDraft = normalizeClassPreprocessConfig(classPreprocessDraft);
    classPreprocessDirty = true;
    updateClassPreprocessOverlay(classPreprocessDraft.manual_roi);
    syncClassPreprocessRoiInputs(classPreprocessDraft.manual_roi);
    if (finalize) {{
      renderClassPreprocessModal();
      refreshClassProcessedPreview();
    }}
  }};
  stage.onpointerdown = (e) => {{
    if (!classPreprocessDraft || String(classPreprocessDraft.mode || 'auto_by_label') !== 'manual_roi') return;
      const start = normalizeClassPreprocessClientPoint(e.clientX, e.clientY, false);
      if (!start) return;
      try {{ stage.focus({{preventScroll:true}}); }} catch (err) {{}}
    dragStart = {{x: e.clientX, y: e.clientY}};
    try {{ stage.setPointerCapture(e.pointerId); }} catch (err) {{}}
    applyPoint(e.clientX, e.clientY, false);
    e.preventDefault();
  }};
  stage.onpointermove = (e) => {{
    if (!dragStart) return;
    applyPoint(e.clientX, e.clientY, false);
  }};
  stage.onpointerup = (e) => {{
    if (!dragStart) return;
    applyPoint(e.clientX, e.clientY, true);
    dragStart = null;
  }};
  stage.onpointercancel = () => {{
    dragStart = null;
  }};
    const syncOverlay = () => updateClassPreprocessOverlay(classPreprocessDraft && classPreprocessDraft.manual_roi);
    if (img.complete) syncOverlay();
    img.onload = syncOverlay;
}}
function renderClassPreprocessModal() {{
  const host = document.getElementById('classPreprocessModal');
  if (!host) return;
  if (!classPreprocessOpen || !classPreprocessClass) {{
    host.style.display = 'none';
    host.innerHTML = '';
    return;
  }}
  const sample = selectedClassSample(classPreprocessClass);
  const sampleFilename = sample ? String(previewFilename(sample) || '') : '';
  const cfg = normalizeClassPreprocessConfig(
    classPreprocessDraft || getSampleEffectivePreprocessConfig(classPreprocessClass, sampleFilename)
  );
  classPreprocessDraft = cfg;
  const roi = Array.isArray(cfg.manual_roi) && cfg.manual_roi.length === 4 ? cfg.manual_roi : null;
  const left = roi ? roi[0] * 100 : 0;
  const top = roi ? roi[1] * 100 : 0;
  const width = roi ? Math.max(0, (roi[2] - roi[0]) * 100) : 0;
  const height = roi ? Math.max(0, (roi[3] - roi[1]) * 100) : 0;
  const samples = normalizePreviewList((STATE.sample_previews && STATE.sample_previews[classPreprocessClass]) || []);
  const dirty = currentClassPreprocessDirty();
  classPreprocessDirty = dirty;
  const thumbs = samples.map((item, idx) => `
    <button class="class-preprocess-thumb${{idx === classPreprocessSampleIndex ? ' active' : ''}}${{sampleHasOwnPreprocessConfig(classPreprocessClass, String(previewFilename(item) || '')) ? ' defined' : ''}}" type="button" data-preprocess-sample="${{idx}}">
      <img src="${{escapeHtml(previewSrc(item))}}" alt="Sample ${{idx + 1}}"/>
      <div class="class-preprocess-thumb-state">${{samplePreprocessStatus(classPreprocessClass, String(previewFilename(item) || ''))}}</div>
    </button>
  `).join('');
  host.style.display = 'flex';
  host.innerHTML = `
    <div class="class-preprocess-shell">
      <div class="class-preprocess-left">
        <div class="class-preprocess-head">
          <span>${{escapeHtml(classPreprocessClass)}} · ${{classPreprocessModeLabel(cfg.mode)}}</span>
          <button class="iconbtn" type="button" id="classPreprocessCloseTop">✕</button>
        </div>
        <div class="class-preprocess-visuals">
          <div class="class-preprocess-visual-pane">
            <div class="class-preprocess-visual-title">Original Sample</div>
              <div class="class-preprocess-stage" id="classPreprocessStage" tabindex="0">
              ${{
                sample
                  ? `<img id="classPreprocessImage" src="${{escapeHtml(previewSrc(sample))}}" alt="Raw sample"/>`
                  : `<div class="preview-empty">This class has no samples yet.</div>`
              }}
                <div class="roi-overlay" id="classPreprocessOverlay" style="display:${{roi && cfg.mode === 'manual_roi' ? 'block' : 'none'}};"></div>
            </div>
          </div>
          <div class="class-preprocess-visual-pane">
            <div class="class-preprocess-visual-title">Processed Preview</div>
            <div class="class-preprocess-result">${{classPreprocessProcessedSrc ? `<img src="${{escapeHtml(classPreprocessProcessedSrc)}}" alt="Processed preview"/>` : `<div class="preview-empty">${{classPreprocessBusy ? 'Rendering...' : 'Choose a sample to preview preprocessing.'}}</div>`}}</div>
          </div>
        </div>
            <div class="class-preprocess-info">In Manual ROI mode, drag directly on the original image and save the current sample. Shortcuts: F1 Auto, F2 Sign, F3 Junction, F4 Manual ROI, F5 Full Frame, S Save, D Delete Sample, Esc Close, Arrow Keys Browse Samples.</div>
      </div>
      <div class="class-preprocess-right class-preprocess-side" id="classPreprocessRightPane">
        <div class="class-preprocess-current class-preprocess-status-row">
          <div class="class-preprocess-info">Current Sample: ${{sampleFilename ? escapeHtml(sampleFilename) : 'No sample selected'}} · ${{samplePreprocessStatus(classPreprocessClass, sampleFilename)}}</div>
          <div class="class-preprocess-status-chip${{dirty ? ' dirty' : ''}}">${{dirty ? 'Unsaved' : 'Saved'}}</div>
          <button class="btn btn-secondary" type="button" id="classPreprocessDeleteSample"${{sampleFilename ? '' : ' disabled'}}>Delete Sample</button>
        </div>
        <div class="class-preprocess-mode-row">
          <label>
            Preprocess Mode
            <select id="classPreprocessMode">
              <option value="auto_by_label"${{cfg.mode === 'auto_by_label' ? ' selected' : ''}}>Auto by Label</option>
              <option value="sign"${{cfg.mode === 'sign' ? ' selected' : ''}}>Sign</option>
              <option value="junction"${{cfg.mode === 'junction' ? ' selected' : ''}}>Junction</option>
              <option value="manual_roi"${{cfg.mode === 'manual_roi' ? ' selected' : ''}}>Manual ROI</option>
              <option value="none"${{cfg.mode === 'none' ? ' selected' : ''}}>Full Frame</option>
            </select>
          </label>
          <button class="btn btn-primary class-preprocess-mode-save" type="button" id="classPreprocessSave"${{sampleFilename && dirty ? '' : ' disabled'}}>${{dirty ? 'Save Sample' : 'Saved'}}</button>
        </div>
        <div class="class-preprocess-fields">
          <label>ROI X1<input id="classPreprocessX1" type="number" min="0" max="1" step="0.01" value="${{roi ? roi[0].toFixed(2) : '0.00'}}" ${{cfg.mode === 'manual_roi' ? '' : 'disabled'}}/></label>
          <label>ROI Y1<input id="classPreprocessY1" type="number" min="0" max="1" step="0.01" value="${{roi ? roi[1].toFixed(2) : '0.00'}}" ${{cfg.mode === 'manual_roi' ? '' : 'disabled'}}/></label>
          <label>ROI X2<input id="classPreprocessX2" type="number" min="0" max="1" step="0.01" value="${{roi ? roi[2].toFixed(2) : '1.00'}}" ${{cfg.mode === 'manual_roi' ? '' : 'disabled'}}/></label>
          <label>ROI Y2<input id="classPreprocessY2" type="number" min="0" max="1" step="0.01" value="${{roi ? roi[3].toFixed(2) : '1.00'}}" ${{cfg.mode === 'manual_roi' ? '' : 'disabled'}}/></label>
        </div>
        <div>
          <div class="class-preprocess-info">Samples</div>
          <div class="class-preprocess-grid">${{thumbs || '<div class="preview-empty">No samples yet.</div>'}}</div>
        </div>
        <div class="class-preprocess-actions">
          <button class="btn btn-secondary" type="button" id="classPreprocessCancel">Close</button>
        </div>
      </div>
    </div>
  `;
  restoreClassPreprocessScroll();
  const rightPane = document.getElementById('classPreprocessRightPane');
  if (rightPane) {{
    rightPane.onscroll = () => {{
      classPreprocessScrollTop = Number(rightPane.scrollTop || 0);
    }};
  }}
  const closeTop = document.getElementById('classPreprocessCloseTop');
  const cancel = document.getElementById('classPreprocessCancel');
  const deleteSampleBtn = document.getElementById('classPreprocessDeleteSample');
  const save = document.getElementById('classPreprocessSave');
  const modeSel = document.getElementById('classPreprocessMode');
  const bindRoi = (id, idx) => {{
    const el = document.getElementById(id);
    if (!el) return;
    el.onchange = () => {{
      rememberClassPreprocessScroll();
      const draft = normalizeClassPreprocessConfig(classPreprocessDraft);
      const roiNow = Array.isArray(draft.manual_roi) && draft.manual_roi.length === 4 ? draft.manual_roi.slice() : [0, 0, 1, 1];
      const n = Number(el.value);
      if (!isFinite(n)) return;
      roiNow[idx] = Math.max(0, Math.min(1, n));
      draft.manual_roi = roiNow;
      classPreprocessDraft = normalizeClassPreprocessConfig(draft);
      classPreprocessDirty = true;
      renderClassPreprocessModal();
      debouncedRefreshClassProcessedPreview(120);
    }};
  }};
  bindRoi('classPreprocessX1', 0);
  bindRoi('classPreprocessY1', 1);
  bindRoi('classPreprocessX2', 2);
  bindRoi('classPreprocessY2', 3);
  host.onclick = (e) => {{
    if (e.target === host) closeClassPreprocessEditor();
  }};
  if (closeTop) closeTop.onclick = () => closeClassPreprocessEditor();
  if (cancel) cancel.onclick = () => closeClassPreprocessEditor();
  if (deleteSampleBtn) deleteSampleBtn.onclick = async () => {{
    if (!sampleFilename) return;
    await deleteSample(classPreprocessClass, sampleFilename);
  }};
    if (save) save.onclick = async () => {{
      try {{
        await saveCurrentClassPreprocessSample(false);
      }} catch (err) {{
        toast(String(err && err.message ? err.message : err));
      }}
  }};
  if (modeSel) modeSel.onchange = () => {{
      applyClassPreprocessMode(modeSel.value);
  }};
  host.querySelectorAll('[data-preprocess-sample]').forEach((btn) => {{
    btn.onclick = () => {{
      if (!maybeDiscardClassPreprocessDraft()) return;
      rememberClassPreprocessScroll();
      classPreprocessSampleIndex = Number(btn.getAttribute('data-preprocess-sample') || 0);
      classPreprocessDraft = normalizeClassPreprocessConfig(
        getSampleEffectivePreprocessConfig(classPreprocessClass, selectedClassSampleFilename(classPreprocessClass))
      );
      classPreprocessDirty = false;
      renderClassPreprocessModal();
      refreshClassProcessedPreview();
    }};
  }});
  bindClassPreprocessStage();
      updateClassPreprocessOverlay(classPreprocessDraft && classPreprocessDraft.manual_roi);
    const stage = document.getElementById('classPreprocessStage');
    if (stage) {{
      window.setTimeout(() => {{
        try {{ stage.focus({{preventScroll:true}}); }} catch (err) {{}}
      }}, 0);
    }}
}}
    window.addEventListener('keydown', (e) => {{
    if (!classPreprocessOpen) return;
    if (e.defaultPrevented || e.metaKey || e.ctrlKey || e.altKey) return;
    const target = e.target;
    const tag = String(target && target.tagName || '').toUpperCase();
    const isTypingTarget = !!(target && (target.isContentEditable || tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT'));
    if (isTypingTarget) return;
      const key = String(e.key || '').toUpperCase();
    const modeMap = {{
      F1: 'auto_by_label',
      F2: 'sign',
      F3: 'junction',
      F4: 'manual_roi',
      F5: 'none',
    }};
    const nextMode = modeMap[key];
    if (nextMode) {{
      e.preventDefault();
      applyClassPreprocessMode(nextMode);
      return;
    }}
    if (key === 'ESCAPE' || key === 'ESC') {{
      e.preventDefault();
      closeClassPreprocessEditor();
      return;
    }}
    if (key === 'S') {{
      e.preventDefault();
      if (classPreprocessClass && selectedClassSampleFilename(classPreprocessClass) && currentClassPreprocessDirty()) {{
          saveCurrentClassPreprocessSample(false).catch((err) => {{
            toast(String(err && err.message ? err.message : err));
          }});
      }}
      return;
    }}
    if (key === 'D') {{
      e.preventDefault();
      const sampleFilename = classPreprocessClass ? selectedClassSampleFilename(classPreprocessClass) : '';
      if (classPreprocessClass && sampleFilename) {{
        deleteSample(classPreprocessClass, sampleFilename);
      }}
      return;
    }}
    if (key === 'ARROWLEFT' || key === 'LEFT') {{
      e.preventDefault();
        moveClassPreprocessSample(-1);
      return;
    }}
    if (key === 'ARROWRIGHT' || key === 'RIGHT') {{
      e.preventDefault();
        moveClassPreprocessSample(1);
      return;
    }}
    if (key === 'ARROWUP' || key === 'UP') {{
      e.preventDefault();
        moveClassPreprocessSample(-getClassPreprocessGridColumnCount());
      return;
    }}
    if (key === 'ARROWDOWN' || key === 'DOWN') {{
      e.preventDefault();
        moveClassPreprocessSample(getClassPreprocessGridColumnCount());
    }}
  }}, true);
restoreTrainCfgFromStorage();
if (window.__tmStageMark) window.__tmStageMark('train-cfg-restored');
function restorePreviewState() {{
  try {{
    const raw = window.localStorage.getItem(previewStorageKey);
    if (!raw) return;
    const data = JSON.parse(raw);
    if (!data || typeof data !== 'object') return;
    previewInputOn = !!data.inputOn;
    if (data.source === 'webcam' || data.source === 'device' || data.source === 'upload') previewSource = String(data.source);
    if (['auto_by_label', 'sign', 'junction', 'manual_roi', 'none'].includes(String(data.preprocess || ''))) {{
      previewPreprocessMode = String(data.preprocess);
    }}
    previewShowRoi = !!data.showRoi;
  }} catch (e) {{}}
}}
function persistPreviewState() {{
  try {{
    window.localStorage.setItem(previewStorageKey, JSON.stringify({{
      inputOn: !!previewInputOn,
      source: String(previewSource || 'webcam'),
      preprocess: String(previewPreprocessMode || 'auto_by_label'),
      showRoi: !!previewShowRoi
    }}));
  }} catch (e) {{}}
}}
function clearPreviewUploadState() {{
  previewUploadImageSrc = '';
  previewUploadImageB64 = '';
  previewUploadFilename = '';
}}
  async function fileToB64(file) {{
    const buf = await file.arrayBuffer();
    const bytes = new Uint8Array(buf);
    let bin = '';
    for (let i = 0; i < bytes.length; i += 1) bin += String.fromCharCode(bytes[i]);
    return btoa(bin);
  }}
async function pickPreviewUploadFile() {{
    return await new Promise((resolve) => {{
      const input = document.createElement('input');
      input.type = 'file';
      input.accept = 'image/png,image/jpeg,image/jpg,image/bmp,image/gif';
      input.onchange = async () => {{
        const file = input.files && input.files[0];
        if (!file) {{
          resolve(false);
          return;
        }}
        try {{
          const b64 = await fileToB64(file);
          const mime = String(file.type || 'image/png');
          previewUploadImageB64 = String(b64 || '');
          previewUploadImageSrc = `data:${{mime}};base64,${{b64}}`;
          previewUploadFilename = String(file.name || 'upload');
          previewSource = 'upload';
          persistPreviewState();
          renderPreviewSettings();
          renderPreviewCard();
          if (previewInputOn) await runPreviewUploadPrediction();
          resolve(true);
        }} catch (err) {{
          toast(String(err && err.message ? err.message : err));
          resolve(false);
        }}
      }};
      input.click();
    }});
}}
restorePreviewState();
if (window.__tmStageMark) window.__tmStageMark('preview-restored');
function restoreExportDir() {{
  try {{
    const raw = window.localStorage.getItem(exportDirStorageKey);
    if (raw) exportDir = String(raw || '');
  }} catch (e) {{}}
}}
function persistExportDir() {{
  try {{
    if (exportDir) window.localStorage.setItem(exportDirStorageKey, String(exportDir));
  }} catch (e) {{}}
}}
restoreExportDir();
function restoreExportSettings() {{
  try {{
    const rawName = window.localStorage.getItem(exportNameStorageKey);
    const rawArray = window.localStorage.getItem(exportArrayStorageKey);
    if (rawName) exportModelName = String(rawName || '').trim() || exportModelName;
    if (rawArray !== null && rawArray !== undefined) exportArrayName = String(rawArray || '').trim();
  }} catch (e) {{}}
}}
function persistExportSettings() {{
  try {{
    window.localStorage.setItem(exportNameStorageKey, String(exportModelName || 'model'));
    window.localStorage.setItem(exportArrayStorageKey, String(exportArrayName || 'g_model'));
  }} catch (e) {{}}
}}
restoreExportSettings();
if (window.__tmStageMark) window.__tmStageMark('export-restored');
function parentUrl() {{
  try {{
    return new URL(window.parent.location.href);
  }} catch (e) {{
    return new URL(window.location.href);
  }}
}}
function navigateParent(url) {{
  try {{
    window.parent.location.href = url;
    return;
  }} catch (e) {{}}
  window.location.href = url;
}}
function cleanupWorkspaceFrameBeforeNavigate(reason) {{
  if (window.__tmNavigatingAway) return;
  try {{
    if (frameHeightRaf) {{
      try {{ cancelAnimationFrame(frameHeightRaf); }} catch (e) {{}}
      frameHeightRaf = 0;
    }}
  }} catch (e) {{}}
  try {{
    layoutResyncTimers.forEach((t) => {{
      try {{ clearTimeout(t); }} catch (e) {{}}
    }});
    layoutResyncTimers = [];
    mountReflowTimers.forEach((t) => {{
      try {{ clearTimeout(t); }} catch (e) {{}}
    }});
    mountReflowTimers = [];
    if (homeNavigateTimer) {{
      try {{ clearTimeout(homeNavigateTimer); }} catch (e) {{}}
      homeNavigateTimer = 0;
    }}
  }} catch (e) {{}}
  try {{
    sendStreamlitMessage('streamlit:setFrameHeight', {{height: 1}});
  }} catch (e) {{}}
  window.__tmNavigatingAway = true;
  try {{
    const targets = [];
    if (window.frameElement) targets.push(window.frameElement);
    if (window.frameElement && window.frameElement.parentElement) targets.push(window.frameElement.parentElement);
    if (window.frameElement && window.frameElement.parentElement && window.frameElement.parentElement.parentElement) {{
      targets.push(window.frameElement.parentElement.parentElement);
    }}
    for (const el of targets) {{
      if (!el || !el.style) continue;
      el.style.height = '';
      el.style.minHeight = '';
      el.style.maxHeight = '';
      el.style.width = '';
      el.style.minWidth = '';
      el.style.maxWidth = '';
      el.style.overflow = '';
      el.style.overflowX = '';
      el.style.overflowY = '';
      el.style.position = '';
      el.style.inset = '';
    }}
  }} catch (e) {{}}
  try {{
    for (const el of [document.documentElement, document.body]) {{
      if (!el || !el.style) continue;
      el.style.minHeight = '';
      el.style.maxHeight = '';
      el.style.overflow = '';
      el.style.overflowX = '';
      el.style.overflowY = '';
    }}
    document.documentElement.scrollTop = 0;
    document.body.scrollTop = 0;
    window.scrollTo(0, 0);
  }} catch (e) {{}}
  try {{
    requestShellLayoutRefresh(String(reason || 'navigate-away'));
  }} catch (e) {{}}
  // #region debug-point B:return-home-cleanup
  dbgEvent('B', 'app.py:cleanupWorkspaceFrameBeforeNavigate', '[DEBUG] cleanup workspace frame before navigate', {{
    reason: String(reason || ''),
    hasFrameElement: !!window.frameElement,
    parentHref: parentUrl().toString(),
    innerWidth: window.innerWidth || 0,
    innerHeight: window.innerHeight || 0,
  }});
  // #endregion
}}
async function returnHome() {{
  // #region debug-point B:return-home-start
  dbgEvent('B', 'app.py:returnHome', '[DEBUG] returnHome start', {{
    session: String(STATE.session || ''),
    returnTarget: String(STATE.return_target || 'home'),
    openSourceClass,
    openSourceKind,
    href: parentUrl().toString(),
    innerWidth: window.innerWidth || 0,
    innerHeight: window.innerHeight || 0,
  }});
  // #endregion
  try {{
    await closeSourcePanel(false);
  }} catch (e) {{}}
  stopPreviewPredictLoop();
  previewInputOn = false;
  persistPreviewState();
  const u = parentUrl();
  const target = String(STATE.return_target || 'home');
  const keys = [
    'tm_action', 'open_class', 'open_kind', 'notice',
    'idx', 'name', 'serial_port', 'webcam_index'
  ];
  for (const key of keys) {{
    u.searchParams.delete(key);
  }}
  if (target === 'classified-import') {{
    u.searchParams.set('tm_project', 'image_classified_import');
    u.searchParams.set('tm_session', String(STATE.session || ''));
  }} else {{
    u.searchParams.delete('tm_project');
    u.searchParams.delete('tm_session');
  }}
  // #region debug-point B:return-home-navigate
  dbgEvent('B', 'app.py:returnHome', '[DEBUG] returnHome navigating to home URL', {{
    session: String(STATE.session || ''),
    returnTarget: target,
    targetUrl: u.toString(),
  }});
  // #endregion
  cleanupWorkspaceFrameBeforeNavigate('return-home');
  homeNavigateTimer = window.setTimeout(() => navigateParent(u.toString()), 40);
}}
function reloadParent() {{
  try {{
    window.parent.location.reload();
    return;
  }} catch (e) {{}}
  window.location.reload();
}}
function stopPreviewLoop() {{
  if (previewTimer) {{
    clearInterval(previewTimer);
    previewTimer = null;
  }}
}}
function startPreviewLoop() {{
  stopPreviewLoop();
  refreshPreviewImage();
  previewTimer = window.setInterval(refreshPreviewImage, Math.max(40, Number(previewIntervalMs || 80)));
}}
function stopHoldSyncLoop() {{
  holdNextToken += 1;
  if (holdSyncTimer) {{
    clearInterval(holdSyncTimer);
    holdSyncTimer = null;
  }}
}}
function clearPreviewBlob() {{
  if (previewBlobUrl) {{
    try {{ URL.revokeObjectURL(previewBlobUrl); }} catch (e) {{}}
    previewBlobUrl = '';
  }}
}}
function syncSourceActionButtons(className) {{
  if (!className) return;
  const safe = cssSafe(className);
  const capBtn = document.getElementById(`sourceCapture-${{safe}}`);
  const holdBtn = document.getElementById(`sourceHold-${{safe}}`);
  const switching = sourceSwitchInFlight && sourceSwitchClass === className && (sourceSwitchKind === 'webcam' || sourceSwitchKind === 'device');
  const capturing = captureInFlight && openSourceClass === className;
  const holding = holdRecording && holdRecordClass === className;
  if (capBtn) {{
    capBtn.disabled = switching || capturing || holding;
    capBtn.textContent = switching ? 'Switching...' : (capturing ? 'Capturing...' : 'Capture');
  }}
  if (holdBtn) {{
    holdBtn.disabled = switching || capturing;
    holdBtn.textContent = switching ? 'Switching...' : (holding ? 'Recording...' : 'Hold to Capture');
  }}
}}
async function closeSourcePanel(shouldRender = true) {{
  stopPreviewLoop();
  clearPreviewBlob();
  sourceSettingsOpen = false;
  const prevClass = openSourceClass;
  const prevKind = openSourceKind;
  if (holdRecording) {{
    try {{
      await fetch(`${{baseUrl}}/stop?session=${{encodeURIComponent(STATE.session)}}`);
    }} catch (e) {{}}
    stopHoldSyncLoop();
    holdRecording = false;
    holdRecordClass = '';
    holdRecordSource = '';
  }}
  openSourceClass = '';
  openSourceKind = '';
  clearOpenSourceState();
  if (prevKind) {{
    try {{
      await fetch(`${{baseUrl}}/live/close?session=${{encodeURIComponent(STATE.session)}}&source=${{encodeURIComponent(prevKind)}}`);
    }} catch (e) {{}}
  }}
  if (shouldRender && prevClass) render();
}}
function refreshPreviewImage() {{
  if (!openSourceKind || !openSourceClass || openSourceKind === 'upload') return;
  if (previewRequestInFlight) return;
  const img = document.getElementById(`sourcePreview-${{cssSafe(openSourceClass)}}`);
  const note = document.getElementById(`sourceNote-${{cssSafe(openSourceClass)}}`);
  if (!img || !note) return;
  const url = `${{baseUrl}}/live/frame?session=${{encodeURIComponent(STATE.session)}}&source=${{encodeURIComponent(openSourceKind)}}&_ts=${{Date.now()}}`;
  previewRequestInFlight = true;
  fetch(url).then(async (res) => {{
    if (!res.ok) {{
      let msg = '';
      try {{
        const data = await res.json();
        msg = String(data && data.error ? data.error : '');
      }} catch (e) {{}}
      if (!msg) {{
        msg = openSourceKind === 'device'
          ? 'Unable to read from serial device. Check serial port and baudrate.'
          : 'Unable to open webcam. Check permission or whether the camera is in use.';
      }}
      throw new Error(msg);
    }}
    const contentType = (res.headers.get('content-type') || '').toLowerCase();
    if (!contentType.includes('image/png')) {{
      let msg = 'Preview returned invalid data.';
      try {{
        const data = await res.json();
        msg = String(data && data.error ? data.error : msg);
      }} catch (e) {{}}
      throw new Error(msg);
    }}
    const blob = await res.blob();
    clearPreviewBlob();
    previewBlobUrl = URL.createObjectURL(blob);
    img.src = previewBlobUrl;
    note.textContent = '';
  }}).catch((err) => {{
    clearPreviewBlob();
    img.removeAttribute('src');
    note.textContent = String(err && err.message ? err.message : err);
  }}).finally(() => {{
    previewRequestInFlight = false;
  }});
}}
async function openSourcePanel(kind, className) {{
  // #region debug-point B:open-source-panel
  dbgEvent('B', 'app.py:openSourcePanel', '[DEBUG] openSourcePanel requested', {{kind, className, openSourceClass, openSourceKind}});
  // #endregion
  if (openSourceClass === className && openSourceKind === kind) return;
  await closeSourcePanel(false);
  openSourceClass = className;
  openSourceKind = kind;
  sourceSettingsOpen = false;
  persistOpenSourceState();
  render();
  if (kind === 'device') {{
    await refreshSerialPorts(false);
    render();
  }}
  await syncClassState(className);
  if (openSourceClass === className) updateOpenSamplesPanel(className);
  if (kind === 'webcam' || kind === 'device') {{
    try {{
      const res = await fetch(`${{baseUrl}}/live/open?session=${{encodeURIComponent(STATE.session)}}&source=${{encodeURIComponent(kind)}}`);
      if (!res.ok) {{
        const data = await res.json().catch(() => ({{ok:'0'}}));
        toast(String(data && data.error ? data.error : 'Unable to open live preview.'));
      }}
    }} catch (e) {{}}
    startPreviewLoop();
  }}
}}
async function ensureOpenSourceLive() {{
  // #region debug-point B:ensure-open-source-live
  dbgEvent('B', 'app.py:ensureOpenSourceLive', '[DEBUG] ensureOpenSourceLive enter', {{openSourceClass, openSourceKind}});
  // #endregion
  if (!openSourceKind || !openSourceClass) return;
  if (openSourceKind !== 'webcam' && openSourceKind !== 'device') return;
  try {{
    const res = await fetch(`${{baseUrl}}/live/open?session=${{encodeURIComponent(STATE.session)}}&source=${{encodeURIComponent(openSourceKind)}}`);
    if (!res.ok) {{
      const data = await res.json().catch(() => ({{ok:'0'}}));
      toast(String(data && data.error ? data.error : 'Unable to open live preview.'));
    }}
  }} catch (e) {{}}
  startPreviewLoop();
}}
async function captureSource() {{
  if (!openSourceKind || !openSourceClass) return;
  if (captureInFlight) {{
    dbgEvent('A', 'app.py:captureSource', '[DEBUG] duplicate capture ignored', {{openSourceKind, openSourceClass}});
    return;
  }}
  captureInFlight = true;
  syncSourceActionButtons(openSourceClass);
  dbgEvent('A', 'app.py:captureSource', '[DEBUG] capture clicked', {{openSourceKind, openSourceClass, captureInFlight}});
  const url = `${{baseUrl}}/live/capture?session=${{encodeURIComponent(STATE.session)}}&source=${{encodeURIComponent(openSourceKind)}}&class=${{encodeURIComponent(openSourceClass)}}`;
  try {{
    const res = await fetch(url);
    const data = await res.json().catch(() => ({{ok:'0', error:'capture failed'}}));
    if (!res.ok || data.ok !== '1') {{
      toast(data.error || 'Capture failed.');
      return;
    }}
    if (data.image_b64) {{
      prependSamplePreview(openSourceClass, {{
        src: `data:image/png;base64,${{data.image_b64}}`,
        filename: String(data.filename || '')
      }});
    }}
    incrementSampleCount(openSourceClass, 1);
    recomputeTrainEnabled();
    updateOpenSamplesPanel(openSourceClass);
    toast('1 sample captured.');
  }} finally {{
    captureInFlight = false;
    syncSourceActionButtons(openSourceClass);
  }}
}}
async function capturePreviewInputToClass(className) {{
  const targetClass = String(className || '').trim();
  const sourceKind = String(previewSource || '').trim();
  if (!targetClass || (sourceKind !== 'webcam' && sourceKind !== 'device')) return;
  if (captureInFlight) return;
  captureInFlight = true;
  try {{
    const res = await fetch(`${{baseUrl}}/live/capture?session=${{encodeURIComponent(STATE.session)}}&source=${{encodeURIComponent(sourceKind)}}&class=${{encodeURIComponent(targetClass)}}`);
    const data = await res.json().catch(() => ({{ok:'0', error:'capture failed'}}));
    if (!res.ok || data.ok !== '1') throw new Error(data.error || 'Capture failed.');
    if (data.image_b64) {{
      prependSamplePreview(targetClass, {{
        src: `data:image/png;base64,${{data.image_b64}}`,
        filename: String(data.filename || '')
      }});
    }}
    incrementSampleCount(targetClass, 1);
    recomputeTrainEnabled();
    if (openSourceClass === targetClass) updateOpenSamplesPanel(targetClass);
    render();
    toast(`1 sample captured from ${{sourceLabel(sourceKind)}}.`);
  }} catch (err) {{
    toast(String(err && err.message ? err.message : err));
  }} finally {{
    captureInFlight = false;
  }}
}}
function toast(msg) {{
  const el = document.getElementById('toast');
  if (!msg) return;
  el.textContent = msg;
  el.style.display = 'block';
  setTimeout(() => {{ el.style.display = 'none'; }}, 2400);
}}
function themeVar(name, fallback = '') {{
  try {{
    const root = document.documentElement;
    const value = window.getComputedStyle(root).getPropertyValue(name);
    return String(value || '').trim() || String(fallback || '');
  }} catch (e) {{
    return String(fallback || '');
  }}
}}
function showConfirmDialog(titleText, bodyText, confirmText = 'Confirm') {{
  return new Promise((resolve) => {{
    let hostDocument = document;
    try {{
      if (window.parent && window.parent !== window && window.parent.document && window.parent.document.body) {{
        hostDocument = window.parent.document;
      }}
    }} catch (e) {{}}
    const overlay = hostDocument.createElement('div');
    overlay.style.position = 'fixed';
    overlay.style.inset = '0';
    overlay.style.background = themeVar('--overlay', 'rgba(32,33,36,0.56)');
    overlay.style.display = 'flex';
    overlay.style.alignItems = 'center';
    overlay.style.justifyContent = 'center';
    overlay.style.zIndex = '2400';
    overlay.style.padding = '20px';

    const card = hostDocument.createElement('div');
    card.style.width = 'min(480px, 92vw)';
    card.style.background = themeVar('--surface-elevated', '#fff');
    card.style.borderRadius = '14px';
    card.style.boxShadow = '0 14px 40px rgba(0,0,0,0.28)';
    card.style.padding = '16px';
    card.style.color = themeVar('--text', '#202124');

    const title = hostDocument.createElement('div');
    title.style.fontSize = '16px';
    title.style.fontWeight = '700';
    title.style.marginBottom = '10px';
    title.textContent = String(titleText || 'Are you sure?');

    const body = hostDocument.createElement('div');
    body.style.fontSize = '13px';
    body.style.lineHeight = '1.5';
    body.style.whiteSpace = 'pre-wrap';
    body.textContent = String(bodyText || '');

    const actions = hostDocument.createElement('div');
    actions.style.display = 'flex';
    actions.style.justifyContent = 'flex-end';
    actions.style.gap = '10px';
    actions.style.marginTop = '14px';

    const cancelBtn = hostDocument.createElement('button');
    cancelBtn.type = 'button';
    cancelBtn.textContent = 'Cancel';
    cancelBtn.style.border = `1px solid ${{themeVar('--input-border', 'rgba(0,0,0,0.16)')}}`;
    cancelBtn.style.background = themeVar('--input-bg', '#fff');
    cancelBtn.style.color = themeVar('--text', '#202124');
    cancelBtn.style.padding = '9px 14px';
    cancelBtn.style.borderRadius = '10px';
    cancelBtn.style.cursor = 'pointer';

    const okBtn = hostDocument.createElement('button');
    okBtn.type = 'button';
    okBtn.textContent = String(confirmText || 'Confirm');
    okBtn.style.border = '0';
    okBtn.style.background = themeVar('--danger', '#d93025');
    okBtn.style.color = themeVar('--text-inverse', '#fff');
    okBtn.style.padding = '9px 14px';
    okBtn.style.borderRadius = '10px';
    okBtn.style.cursor = 'pointer';

    const cleanup = (result) => {{
      try {{ overlay.remove(); }} catch (e) {{}}
      resolve(!!result);
    }};
    cancelBtn.onclick = () => cleanup(false);
    okBtn.onclick = () => cleanup(true);
    overlay.onclick = (e) => {{
      if (e.target === overlay) cleanup(false);
    }};

    actions.appendChild(cancelBtn);
    actions.appendChild(okBtn);
    card.appendChild(title);
    card.appendChild(body);
    card.appendChild(actions);
    overlay.appendChild(card);
    hostDocument.body.appendChild(overlay);
  }});
}}
function showOverwriteConfirmDialog(conflicts) {{
  return new Promise((resolve) => {{
    let hostWindow = window;
    let hostDocument = document;
    try {{
      if (window.parent && window.parent !== window && window.parent.document && window.parent.document.body) {{
        hostWindow = window.parent;
        hostDocument = window.parent.document;
      }}
    }} catch (e) {{}}
    const overlay = hostDocument.createElement('div');
    overlay.style.position = 'fixed';
    overlay.style.inset = '0';
    overlay.style.background = themeVar('--overlay', 'rgba(32,33,36,0.56)');
    overlay.style.display = 'flex';
    overlay.style.alignItems = 'center';
    overlay.style.justifyContent = 'center';
    overlay.style.zIndex = '2400';
    overlay.style.padding = '20px';

    const card = hostDocument.createElement('div');
    card.style.width = 'min(560px, 92vw)';
    card.style.background = themeVar('--surface-elevated', '#fff');
    card.style.borderRadius = '14px';
    card.style.boxShadow = '0 14px 40px rgba(0,0,0,0.28)';
    card.style.padding = '16px';
    card.style.color = themeVar('--text', '#202124');

    const title = hostDocument.createElement('div');
    title.style.fontSize = '16px';
    title.style.fontWeight = '700';
    title.style.marginBottom = '10px';
    title.textContent = 'Overwrite existing export files?';

    const body = hostDocument.createElement('div');
    body.style.fontSize = '13px';
    body.style.lineHeight = '1.5';
    body.style.whiteSpace = 'pre-wrap';
    let msg = 'Files already exist in this export folder and will be overwritten.';
    const nl = String.fromCharCode(10);
    if (Array.isArray(conflicts) && conflicts.length) {{
      msg += nl + nl + 'Existing files:' + nl + '- ' + conflicts.join(nl + '- ');
    }}
    body.textContent = msg;

    const actions = hostDocument.createElement('div');
    actions.style.display = 'flex';
    actions.style.justifyContent = 'flex-end';
    actions.style.gap = '10px';
    actions.style.marginTop = '14px';

    const cancelBtn = hostDocument.createElement('button');
    cancelBtn.type = 'button';
    cancelBtn.textContent = 'Cancel';
    cancelBtn.style.border = '0';
    cancelBtn.style.borderRadius = '8px';
    cancelBtn.style.padding = '10px 12px';
    cancelBtn.style.fontWeight = '700';
    cancelBtn.style.cursor = 'pointer';
    cancelBtn.style.background = themeVar('--surface-soft', '#eef1f5');
    cancelBtn.style.color = themeVar('--text', '#202124');

    const okBtn = hostDocument.createElement('button');
    okBtn.type = 'button';
    okBtn.textContent = 'Overwrite';
    okBtn.style.border = '0';
    okBtn.style.borderRadius = '8px';
    okBtn.style.padding = '10px 12px';
    okBtn.style.fontWeight = '700';
    okBtn.style.cursor = 'pointer';
    okBtn.style.background = themeVar('--blue', '#1a73e8');
    okBtn.style.color = themeVar('--text-inverse', '#fff');

    let done = false;
    function finish(value) {{
      if (done) return;
      done = true;
      try {{ overlay.remove(); }} catch (e) {{}}
      resolve(!!value);
    }}

    cancelBtn.onclick = () => finish(false);
    okBtn.onclick = () => finish(true);
    overlay.onclick = (e) => {{
      if (e.target === overlay) finish(false);
    }};

    actions.appendChild(cancelBtn);
    actions.appendChild(okBtn);
    card.appendChild(title);
    card.appendChild(body);
    card.appendChild(actions);
    overlay.appendChild(card);
    try {{
      if (hostWindow && typeof hostWindow.scrollTo === 'function') hostWindow.scrollTo(0, 0);
    }} catch (e) {{}}
    hostDocument.body.appendChild(overlay);
  }});
}}
function renderNavMenu() {{
  const menu = document.getElementById('navMenu');
  if (!menu) return;
  if (navMenuOpen) menu.classList.add('open');
  else menu.classList.remove('open');
  try {{
    window.requestAnimationFrame(() => updateFlow());
  }} catch (e) {{}}
}}
function closeNavMenu() {{
  navMenuOpen = false;
  renderNavMenu();
}}
function toggleNavMenu() {{
  navMenuOpen = !navMenuOpen;
  renderNavMenu();
}}
function applyProjectState(state) {{
  const s = state && typeof state === 'object' ? state : {{}};
  // #region debug-point C:apply-project-state
  dbgEvent('C', 'app.py:applyProjectState', '[DEBUG] applyProjectState received state payload', {{
    classes: Array.isArray(s.classes) ? s.classes.slice() : [],
    countKeys: s.counts && typeof s.counts === 'object' ? Object.keys(s.counts) : [],
    previewKeys: s.sample_previews && typeof s.sample_previews === 'object' ? Object.keys(s.sample_previews) : [],
    exportEnabled: String(s.export_enabled || '0'),
  }});
  // #endregion
  stopHoldSyncLoop();
  stopPreviewPredictLoop();
  stopPreviewLoop();
  clearPreviewBlob();
  previewSettingsOpen = false;
  sourceSettingsOpen = false;
  navMenuOpen = false;
  renderNavMenu();
  holdRecording = false;
  holdRecordClass = '';
  holdRecordSource = '';
  openSourceClass = '';
  openSourceKind = '';
  classPreprocessOpen = false;
  classPreprocessClass = '';
  classPreprocessDraft = null;
  classPreprocessSampleIndex = 0;
  classPreprocessProcessedSrc = '';
  classPreprocessBusy = false;
  if (classPreprocessAbort) {{ classPreprocessAbort.abort(); classPreprocessAbort = null; }}
  if (classPreprocessDebounce) {{ clearTimeout(classPreprocessDebounce); classPreprocessDebounce = null; }}
  clearOpenSourceState();
  previewInputOn = false;
  persistPreviewState();
  trainInFlight = false;
  const cls = Array.isArray(s.classes) ? s.classes.slice() : [];
  STATE.classes = cls;
  const cnt = s.counts && typeof s.counts === 'object' ? s.counts : {{}};
  const nextCounts = {{}};
  for (const [k, v] of Object.entries(cnt)) nextCounts[String(k)] = Number(v || 0);
  STATE.counts = nextCounts;
  const prev = s.sample_previews && typeof s.sample_previews === 'object' ? s.sample_previews : {{}};
  const nextPrev = {{}};
  for (const [k, v] of Object.entries(prev)) nextPrev[String(k)] = normalizePreviewList(v);
  STATE.sample_previews = nextPrev;
  const processedPrev = s.processed_previews && typeof s.processed_previews === 'object' ? s.processed_previews : {{}};
  const nextProcessedPrev = {{}};
  for (const [k, v] of Object.entries(processedPrev)) nextProcessedPrev[String(k)] = normalizePreviewList(v);
  STATE.processed_previews = nextProcessedPrev;
  STATE.class_preprocess = normalizeClassPreprocessMap(s.class_preprocess);
  STATE.sample_preprocess = normalizeSamplePreprocessMap(s.sample_preprocess);
  STATE.class_labels = (s.class_labels && typeof s.class_labels === 'object') ? Object.assign({{}}, s.class_labels) : {{}};
  STATE.export_enabled = String(s.export_enabled || '0') === '1';
  if (s.train_cfg && typeof s.train_cfg === 'object') {{
    STATE.train_cfg = Object.assign({{}}, STATE.train_cfg || {{}}, s.train_cfg);
    persistTrainCfgStorage();
  }}
  recomputeTrainEnabled();
  try {{
    document.documentElement.scrollTop = 0;
    document.documentElement.scrollLeft = 0;
    document.body.scrollTop = 0;
    document.body.scrollLeft = 0;
    window.scrollTo(0, 0);
  }} catch (e) {{}}
  scheduleLayoutResync();
}}
async function exportDataset() {{
  try {{
    const pickRes = await fetch(`${{baseUrl}}/export/pick_dir?session=${{encodeURIComponent(STATE.session)}}`);
    const pick = await pickRes.json().catch(() => ({{ok:'0'}}));
    if (!pickRes.ok || pick.ok !== '1') {{
      if (pick && pick.canceled === '1') return;
      throw new Error(pick.error || 'Unable to choose folder.');
    }}
    const exportDir = String(pick.export_dir || '').trim();
    if (!exportDir) return;
    const res = await fetch(`${{baseUrl}}/dataset/export`, {{
      method: 'POST',
      headers: {{'Content-Type':'application/json'}},
      body: JSON.stringify({{session: STATE.session, export_dir: exportDir}})
    }});
    const data = await res.json().catch(() => ({{ok:'0'}}));
    if (!res.ok || data.ok !== '1') throw new Error(data.error || 'Dataset export failed.');
    toast(`Dataset exported to: ${{String(data.export_dir || exportDir)}}`);
  }} catch (e) {{
    toast(String(e && e.message ? e.message : e));
  }}
}}
async function saveProject() {{
  try {{
    commitCurrentClassPreprocessDraftIfNeeded();
    const pickRes = await fetch(`${{baseUrl}}/project/pick_save?session=${{encodeURIComponent(STATE.session)}}`);
    const pick = await pickRes.json().catch(() => ({{ok:'0'}}));
    if (!pickRes.ok || pick.ok !== '1') {{
      if (pick && pick.canceled === '1') return;
      throw new Error(pick.error || 'Unable to choose file.');
    }}
    const savePath = String(pick.save_path || '').trim();
    if (!savePath) return;
    const res = await fetch(`${{baseUrl}}/project/save`, {{
      method: 'POST',
      headers: {{'Content-Type':'application/json'}},
      body: JSON.stringify({{
        session: STATE.session,
        save_path: savePath,
        project_state: {{
          train_cfg: Object.assign({{}}, STATE.train_cfg || {{}}, {{
            class_preprocess: Object.assign({{}}, STATE.class_preprocess || {{}})
          }}),
          class_preprocess: Object.assign({{}}, STATE.class_preprocess || {{}}),
          sample_preprocess: normalizeSamplePreprocessMap(STATE.sample_preprocess || {{}}),
          class_labels: Object.assign({{}}, STATE.class_labels || {{}})
        }}
      }})
    }});
    const data = await res.json().catch(() => ({{ok:'0'}}));
    if (!res.ok || data.ok !== '1') throw new Error(data.error || 'Save failed.');
    toast(`Saved: ${{String(data.save_path || savePath)}}`);
  }} catch (e) {{
    toast(String(e && e.message ? e.message : e));
  }}
}}
async function openProject() {{
  try {{
    // #region debug-point A:open-project-start
    dbgEvent('A', 'app.py:openProject', '[DEBUG] openProject start', {{
      session: String(STATE.session || ''),
      href: parentUrl().toString(),
      innerWidth: window.innerWidth || 0,
      innerHeight: window.innerHeight || 0,
      classes: Array.isArray(STATE.classes) ? STATE.classes.slice() : [],
    }});
    // #endregion
    const pickRes = await fetch(`${{baseUrl}}/project/pick_open?session=${{encodeURIComponent(STATE.session)}}`);
    const pick = await pickRes.json().catch(() => ({{ok:'0'}}));
    if (!pickRes.ok || pick.ok !== '1') {{
      if (pick && pick.canceled === '1') return;
      throw new Error(pick.error || 'Unable to choose file.');
    }}
    const openPath = String(pick.open_path || '').trim();
    if (!openPath) return;
    const res = await fetch(`${{baseUrl}}/project/open`, {{
      method: 'POST',
      headers: {{'Content-Type':'application/json'}},
      body: JSON.stringify({{session: STATE.session, open_path: openPath}})
    }});
    const data = await res.json().catch(() => ({{ok:'0'}}));
    if (!res.ok || data.ok !== '1') throw new Error(data.error || 'Open failed.');
    const statePayload = data && data.state && typeof data.state === 'object' ? data.state : {{}};
    // #region debug-point D:open-project-response
    dbgEvent('D', 'app.py:openProject', '[DEBUG] openProject response received', {{
      openPath,
      stateClasses: Array.isArray(statePayload.classes) ? statePayload.classes.slice() : [],
      statePreviewKeys: statePayload.sample_previews && typeof statePayload.sample_previews === 'object' ? Object.keys(statePayload.sample_previews) : [],
      stateCounts: statePayload.counts || {{}},
    }});
    // #endregion
    applyProjectState(data.state || {{}});
    render();
    scheduleLayoutResync();
    requestShellLayoutRefresh('open-project');
    window.setTimeout(() => {{
      // #region debug-point A:open-project-post-render-1
      dbgEvent('A', 'app.py:openProject', '[DEBUG] openProject post-render metrics (120ms)', {{
        innerWidth: window.innerWidth || 0,
        innerHeight: window.innerHeight || 0,
        docClientHeight: document.documentElement ? document.documentElement.clientHeight : 0,
        docScrollHeight: document.documentElement ? document.documentElement.scrollHeight : 0,
        bodyScrollHeight: document.body ? document.body.scrollHeight : 0,
      }});
      // #endregion
    }}, 120);
    window.setTimeout(() => {{
      // #region debug-point A:open-project-post-render-2
      dbgEvent('A', 'app.py:openProject', '[DEBUG] openProject post-render metrics (500ms)', {{
        innerWidth: window.innerWidth || 0,
        innerHeight: window.innerHeight || 0,
        docClientHeight: document.documentElement ? document.documentElement.clientHeight : 0,
        docScrollHeight: document.documentElement ? document.documentElement.scrollHeight : 0,
        bodyScrollHeight: document.body ? document.body.scrollHeight : 0,
      }});
      // #endregion
    }}, 500);
  }} catch (e) {{
    toast(String(e && e.message ? e.message : e));
  }}
}}
async function resetProject() {{
  const ok = window.confirm('Reset project? This will delete samples, classes, and trained model.');
  if (!ok) return;
  try {{
    const res = await fetch(`${{baseUrl}}/project/reset`, {{
      method: 'POST',
      headers: {{'Content-Type':'application/json'}},
      body: JSON.stringify({{session: STATE.session, confirm: '1'}})
    }});
    const data = await res.json().catch(() => ({{ok:'0'}}));
    if (!res.ok || data.ok !== '1') throw new Error(data.error || 'Reset failed.');
    applyProjectState(data.state || {{}});
    render();
    scheduleLayoutResync();
    requestShellLayoutRefresh('reset-project');
  }} catch (e) {{
    toast(String(e && e.message ? e.message : e));
  }}
}}
function defaultExportDir() {{
  if (exportDir) return exportDir;
  return `~/Documents/TFLiteTraining/exports`;
}}
async function exportRunWithOverwriteConfirm(exportDirValue, modelNameValue, arrayNameValue) {{
  const payload = {{
    session: STATE.session,
    export_dir: String(exportDirValue || '').trim(),
    model_name: String(modelNameValue || 'model'),
    array_name: String(arrayNameValue || '')
  }};
  let res = await fetch(`${{baseUrl}}/export/run`, {{
    method: 'POST',
    headers: {{'Content-Type':'application/json'}},
    body: JSON.stringify(payload)
  }});
  let data = await res.json().catch(() => ({{ok:'0'}}));
  if (res.ok && data.ok === '1') return data;
  if (data && data.needs_confirm === '1') {{
    const files = Array.isArray(data.conflicts) ? data.conflicts.slice(0, 8) : [];
    const confirmed = await showOverwriteConfirmDialog(files);
    if (!confirmed) {{
      toast('Export canceled.');
      return {{ok:'0', canceled:'1'}};
    }}
    const overwritePayload = {{
      session: payload.session,
      export_dir: payload.export_dir,
      model_name: payload.model_name,
      array_name: payload.array_name,
      overwrite: true
    }};
    res = await fetch(`${{baseUrl}}/export/run`, {{
      method: 'POST',
      headers: {{'Content-Type':'application/json'}},
      body: JSON.stringify(overwritePayload)
    }});
    data = await res.json().catch(() => ({{ok:'0'}}));
    if (res.ok && data.ok === '1') {{
      toast('Existing export files were overwritten.');
      return data;
    }}
  }}
  throw new Error((data && data.error) || ('Export failed (status ' + String(res && res.status ? res.status : 0) + ').'));
}}
async function exportModel() {{
  if (!STATE.export_enabled) return;
  const suggested = defaultExportDir();
  const picked = window.prompt('Export folder', suggested);
  if (picked == null) return;
  exportDir = String(picked || '').trim();
  if (!exportDir) return;
  persistExportDir();
  try {{
    const data = await exportRunWithOverwriteConfirm(exportDir, 'model', 'g_model');
    if (!data || data.canceled === '1') return;
    toast(`Exported to: ${{data.export_dir || exportDir}}`);
  }} catch (err) {{
    toast(String(err && err.message ? err.message : err));
  }}
}}
function showTrainProgress(show, pct, text) {{
  const wrap = document.getElementById('trainProgress');
  const fill = document.getElementById('trainProgressFill');
  const label = document.getElementById('trainProgressText');
  if (!wrap || !fill || !label) return;
  wrap.style.display = show ? 'block' : 'none';
  const clamped = Math.max(0, Math.min(1, Number(pct || 0)));
  fill.style.width = `${{Math.round(clamped * 1000) / 10}}%`;
  label.textContent = String(text || '');
}}
function syncTrainUi() {{
  const trainBtn = document.getElementById('trainBtn');
  if (!trainBtn) return;
  trainBtn.classList.toggle('enabled', !!STATE.train_enabled);
  trainBtn.disabled = !STATE.train_enabled || !!trainInFlight;
}}
async function startTrain() {{
  if (trainInFlight) return;
  if (!STATE.train_enabled) return;
  trainInFlight = true;
  syncTrainUi();
  trainPollToken += 1;
  const token = trainPollToken;
  showTrainProgress(true, 0.01, 'Starting...');
  try {{
    const res = await fetch(`${{baseUrl}}/train/start`, {{
      method: 'POST',
      headers: {{'Content-Type':'application/json'}},
      body: JSON.stringify({{
        session: STATE.session,
        cfg: Object.assign({{}}, STATE.train_cfg || {{}}, {{
          class_preprocess: Object.assign({{}}, STATE.class_preprocess || {{}}),
          sample_preprocess: normalizeSamplePreprocessMap(STATE.sample_preprocess || {{}}),
          class_labels: Object.assign({{}}, STATE.class_labels || {{}})
        }})
      }})
    }});
    const data = await res.json().catch(() => ({{ok:'0'}}));
    if (!res.ok || data.ok !== '1') {{
      throw new Error(data.error || 'Unable to start training.');
    }}
    while (token === trainPollToken) {{
      const stRes = await fetch(`${{baseUrl}}/train/status?session=${{encodeURIComponent(STATE.session)}}`);
      const stData = await stRes.json().catch(() => ({{ok:'0'}}));
      if (!stRes.ok || stData.ok !== '1') {{
        throw new Error(stData.error || 'Unable to read training status.');
      }}
      const p = Number(stData.progress || 0);
      const msg = String(stData.message || '');
      showTrainProgress(true, p, msg);
      if (String(stData.done || '0') === '1') {{
        const err = String(stData.error || '');
        if (err) {{
          toast(err);
          showTrainProgress(true, 1.0, 'Failed.');
        }} else {{
          STATE.export_enabled = true;
          renderTrainStatus();
          renderPreviewCard();
          showTrainProgress(true, 1.0, 'Done.');
          toast('Training complete.');
          const exportBtn = document.getElementById('exportBtn');
          if (exportBtn) exportBtn.disabled = false;
          const exportBlockBtn = document.getElementById('exportBlockBtn');
          if (exportBlockBtn) exportBlockBtn.disabled = false;
        }}
        break;
      }}
      await new Promise((r) => setTimeout(r, 250));
    }}
  }} catch (err) {{
    toast(String(err && err.message ? err.message : err));
    showTrainProgress(false, 0, '');
  }} finally {{
    if (token === trainPollToken) {{
      trainInFlight = false;
      syncTrainUi();
    }}
  }}
}}
function setAction(action, params) {{
  dbgEvent(action === 'open_source' ? 'B' : (action === 'set_webcam_index' ? 'D' : 'A'), 'app.py:setAction', '[DEBUG] setAction navigation', {{action, params, href: parentUrl().toString()}});
  const u = parentUrl();
  u.searchParams.set('tm_action', action);
  if (action !== 'home') {{
    u.searchParams.set('tm_project', 'image');
    u.searchParams.set('tm_session', String(STATE.session || ''));
  }} else {{
    u.searchParams.delete('tm_project');
    u.searchParams.delete('tm_session');
  }}
  for (const [k,v] of Object.entries(params || {{}})) {{
    u.searchParams.set(k, String(v));
  }}
  navigateParent(u.toString());
}}
async function uploadFiles(className, files) {{
  if (!files || files.length === 0) return;
  let uploaded = 0;
  for (const f of files) {{
    const buf = await f.arrayBuffer();
    const bytes = new Uint8Array(buf);
    let bin = '';
    for (let i=0; i<bytes.length; i++) bin += String.fromCharCode(bytes[i]);
    const b64 = btoa(bin);
    const res = await fetch(`${{baseUrl}}/upload`, {{
      method: 'POST',
      headers: {{'Content-Type':'application/json'}},
      body: JSON.stringify({{session: STATE.session, class: className, image_b64: b64}})
    }});
    const data = await res.json().catch(() => ({{ok:'0'}}));
    if (res.ok && data.ok === '1') {{
      uploaded += 1;
      if (data.image_b64) prependSamplePreview(className, {{src: `data:image/png;base64,${{data.image_b64}}`, filename: String(data.filename || '')}});
      else prependSamplePreview(className, {{src: `data:image/*;base64,${{b64}}`, filename: String(data.filename || '')}});
      incrementSampleCount(className, 1);
      recomputeTrainEnabled();
    }}
  }}
  syncTrainUi();
  if (openSourceClass === className) updateOpenSamplesPanel(className);
  toast(uploaded > 0 ? `Uploaded ${{uploaded}} image(s).` : 'Upload failed.');
}}
function cssSafe(name) {{
  return String(name).replace(/[^a-zA-Z0-9_-]/g, '_');
}}
function normalizePreviewItem(item) {{
  if (!item) return null;
  if (typeof item === 'string') {{
    return {{src: String(item), filename: ''}};
  }}
  if (typeof item === 'object') {{
    const src = String(item.src || item.image || '');
    if (!src) return null;
    return {{
      src,
      filename: String(item.filename || '')
    }};
  }}
  return null;
}}
function normalizePreviewList(items) {{
  if (!Array.isArray(items)) return [];
  const out = [];
  for (const item of items) {{
    const normalized = normalizePreviewItem(item);
    if (normalized) out.push(normalized);
  }}
  return out;
}}
function previewSrc(item) {{
  const normalized = normalizePreviewItem(item);
  return normalized ? String(normalized.src || '') : '';
}}
function previewFilename(item) {{
  const normalized = normalizePreviewItem(item);
  return normalized ? String(normalized.filename || '') : '';
}}
function escapeHtml(value) {{
  return String(value || '')
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}}
function nullish(value, fallback) {{
  return value === null || value === undefined ? fallback : value;
}}
function prependSamplePreview(className, item) {{
  const normalized = normalizePreviewItem(item);
  if (!className || !normalized) return;
  const items = Array.isArray(STATE.sample_previews[className]) ? STATE.sample_previews[className].slice() : [];
  items.unshift(normalized);
  STATE.sample_previews[className] = items;
}}
function removePreviewItemLocal(className, filename) {{
  if (!className || !filename) return;
  const items = normalizePreviewList(STATE.sample_previews[className]);
  STATE.sample_previews[className] = items.filter((item) => previewFilename(item) !== filename);
}}
function incrementSampleCount(className, delta) {{
  const next = Math.max(0, Number(STATE.counts[className] || 0) + Number(delta || 0));
  STATE.counts[className] = next;
}}
function recomputeTrainEnabled() {{
  const classes = Array.isArray(STATE.classes) ? STATE.classes : [];
  const counts = STATE.counts || {{}};
  const nonEmpty = classes.filter((name) => Number(counts[name] || 0) > 0).length;
  const total = classes.reduce((sum, name) => sum + Number(counts[name] || 0), 0);
  STATE.train_enabled = classes.length >= 2 && nonEmpty >= 2 && nonEmpty === classes.length && total > 0;
}}
async function syncClassState(className) {{
  if (!className) return;
  try {{
    const res = await fetch(`${{baseUrl}}/class_state?session=${{encodeURIComponent(STATE.session)}}&class=${{encodeURIComponent(className)}}`);
    const data = await res.json().catch(() => ({{ok:'0'}}));
    if (!res.ok || data.ok !== '1') throw new Error(data.error || 'Unable to refresh class state.');
    STATE.counts[className] = Number(data.count || 0);
    STATE.sample_previews[className] = normalizePreviewList(data.previews);
    recomputeTrainEnabled();
    syncTrainUi();
  }} catch (err) {{
    toast(String(err && err.message ? err.message : err));
  }}
}}
async function startHoldCapture() {{
  if (holdRecording || !openSourceClass || !openSourceKind || openSourceKind === 'upload') return;
  try {{
    holdResumePreviewPredict = !!previewInputOn;
    holdPreviewSourceBeforeCapture = String(previewSource || 'webcam');
    if (holdResumePreviewPredict) {{
      stopPreviewPredictLoop();
      try {{
        await fetch(`${{baseUrl}}/live/close?session=${{encodeURIComponent(STATE.session)}}&source=${{encodeURIComponent(holdPreviewSourceBeforeCapture)}}`);
      }} catch (e) {{}}
      await new Promise((r) => setTimeout(r, 120));
    }}
    if (openSourceKind === 'webcam' || openSourceKind === 'device') {{
      stopPreviewLoop();
      clearPreviewBlob();
      try {{
        await fetch(`${{baseUrl}}/live/close?session=${{encodeURIComponent(STATE.session)}}&source=${{encodeURIComponent(openSourceKind)}}`);
      }} catch (e) {{}}
      await new Promise((r) => setTimeout(r, 120));
    }}
    const res = await fetch(`${{baseUrl}}/start?session=${{encodeURIComponent(STATE.session)}}&source=${{encodeURIComponent(openSourceKind)}}&class=${{encodeURIComponent(openSourceClass)}}`);
    const data = await res.json().catch(() => ({{ok:'0'}}));
    if (!res.ok || data.ok !== '1') throw new Error(data.error || 'Unable to start hold capture.');
    holdRecording = true;
    holdRecordClass = openSourceClass;
    holdRecordSource = openSourceKind;
    syncSourceActionButtons(openSourceClass);
    stopHoldSyncLoop();
    holdSeq = 0;
    const token = holdNextToken + 1;
    holdNextToken = token;
    (async () => {{
      while (holdRecording && holdRecordClass && holdNextToken === token) {{
        try {{
          const nextRes = await fetch(`${{baseUrl}}/record/next?session=${{encodeURIComponent(STATE.session)}}&since=${{encodeURIComponent(String(holdSeq))}}`);
          const nextData = await nextRes.json().catch(() => ({{ok:'0'}}));
          if (!holdRecording || holdNextToken !== token) break;
          if (!nextRes.ok || nextData.ok !== '1') throw new Error(nextData.error || 'Unable to read hold capture updates.');
          const serverError = String(nextData.error || '').trim();
          if (serverError) {{
            toast(serverError);
            break;
          }}
          const seq = Number(nextData.seq || holdSeq || 0);
          if (seq > holdSeq && nextData.image_b64) {{
            holdSeq = seq;
            try {{
              const img = document.getElementById(`sourcePreview-${{cssSafe(holdRecordClass)}}`);
              const note = document.getElementById(`sourceNote-${{cssSafe(holdRecordClass)}}`);
              if (img) img.src = `data:image/png;base64,${{nextData.image_b64}}`;
              if (note) note.textContent = '';
            }} catch (e) {{}}
            prependSamplePreview(holdRecordClass, {{
              src: `data:image/png;base64,${{nextData.image_b64}}`,
              filename: String(nextData.filename || '')
            }});
            STATE.counts[holdRecordClass] = Number(nextData.count || STATE.counts[holdRecordClass] || 0);
            recomputeTrainEnabled();
            if (openSourceClass === holdRecordClass) updateOpenSamplesPanel(holdRecordClass);
          }} else {{
            holdSeq = seq;
          }}
          if (String(nextData.recording || '1') !== '1') break;
        }} catch (err) {{
          toast(String(err && err.message ? err.message : err));
          await new Promise((r) => setTimeout(r, 250));
        }}
      }}
    }})().catch(() => {{}});
  }} catch (err) {{
    toast(String(err && err.message ? err.message : err));
  }}
}}
async function stopHoldCapture() {{
  if (!holdRecording) return;
  const className = holdRecordClass;
  const sourceKind = holdRecordSource;
  const shouldResumePreviewPredict = holdResumePreviewPredict;
  const previewSourceBeforeCapture = holdPreviewSourceBeforeCapture === 'device' ? 'device' : 'webcam';
  stopHoldSyncLoop();
  holdRecording = false;
  holdRecordClass = '';
  holdRecordSource = '';
  holdResumePreviewPredict = false;
  holdPreviewSourceBeforeCapture = 'webcam';
  syncSourceActionButtons(className);
  try {{
    await fetch(`${{baseUrl}}/stop?session=${{encodeURIComponent(STATE.session)}}`);
  }} catch (e) {{}}
  await syncClassState(className);
  if (openSourceClass === className) updateOpenSamplesPanel(className);
  if (shouldResumePreviewPredict && previewInputOn) {{
    previewSource = previewSourceBeforeCapture;
    startPreviewPredictLoop();
  }} else if (openSourceClass === className && openSourceKind === sourceKind && (sourceKind === 'webcam' || sourceKind === 'device')) {{
    await ensureOpenSourceLive();
  }}
  toast('Hold capture stopped.');
}}
function buildDeviceOptions(selected) {{
  const ports = Array.isArray(STATE.serial_ports) ? STATE.serial_ports : [];
  const opts = ['<option value="">Select device port</option>'];
  for (const p of ports) {{
    const device = String(p.device || '');
    const label = String(p.label || device);
    const sel = device === selected ? ' selected' : '';
    opts.push(`<option value="${{device.replace(/"/g, '&quot;')}}"${{sel}}>${{label}}</option>`);
  }}
  return opts.join('');
}}
function refillDeviceSelect(selectEl, selectedValue) {{
  if (!selectEl) return;
  const current = String(selectedValue == null ? (selectEl.value || '') : selectedValue);
  selectEl.innerHTML = buildDeviceOptions(current);
  selectEl.value = current;
}}
async function refreshSerialPorts(shouldRerender = true, targetSelectId = '') {{
  try {{
    const res = await fetch(`${{baseUrl}}/serial/ports?session=${{encodeURIComponent(STATE.session)}}&_ts=${{Date.now()}}`);
    const data = await res.json().catch(() => ({{ok:'0'}}));
    if (!res.ok || data.ok !== '1') throw new Error(data.error || 'Unable to refresh serial ports.');
    const ports = Array.isArray(data.ports) ? data.ports : [];
    STATE.serial_ports = ports.map((p) => ({{
      device: String(p && p.device ? p.device : ''),
      label: String(p && p.label ? p.label : (p && p.device ? p.device : '')),
    }})).filter((p) => !!p.device);
    const stillExists = STATE.serial_ports.some((p) => String(p.device || '') === String(currentSerialPort || ''));
    if (!stillExists && currentSerialPort) {{
      STATE.serial_ports = [{{device: String(currentSerialPort), label: String(currentSerialPort)}}].concat(STATE.serial_ports);
    }}
    if (shouldRerender) {{
      render();
    }} else if (targetSelectId) {{
      const selectEl = document.getElementById(targetSelectId);
      refillDeviceSelect(selectEl, currentSerialPort);
    }}
  }} catch (err) {{
    toast(String(err && err.message ? err.message : err));
  }}
}}
function buildWebcamOptions(selected) {{
  const cams = Array.isArray(STATE.webcam_options) ? STATE.webcam_options : [];
  dbgEvent('E', 'app.py:buildWebcamOptions', '[DEBUG] frontend webcam options', {{selected, cams}});
  const opts = ['<option value="">Select camera</option>'];
  for (const c of cams) {{
    const idx = Number(c.index);
    const label = String(c.label || `Camera ${{idx}}`);
    const sel = idx === Number(selected) ? ' selected' : '';
    opts.push(`<option value="${{idx}}"${{sel}}>${{label}}</option>`);
  }}
  return opts.join('');
}}
function buildSerialBaudOptions(selected) {{
  const values = [9600, 19200, 38400, 57600, 115200, 230400, 460800, 921600];
  return values.map((baud) => `<option value="${{baud}}"${{Number(selected) === baud ? ' selected' : ''}}>${{baud}}</option>`).join('');
}}
function buildSerialFrameSideOptions(selected) {{
  const values = [96, 160];
  return values.map((side) => `<option value="${{side}}"${{Number(selected) === side ? ' selected' : ''}}>${{side}}×${{side}}</option>`).join('');
}}
function deviceHelpText() {{
  const ports = Array.isArray(STATE.serial_ports) ? STATE.serial_ports : [];
  if (!ports.length) return 'No compatible serial UART device is currently detected. Connect the board or install the required USB serial driver first.';
  return 'Choose the serial port connected to your device before capturing. Sync is the hex frame header used to find the start of each serial image packet, for example AA 55 AA. Image Size must match the firmware output exactly (e.g. 96x96 or 160x160).';
}}
function webcamHelpText() {{
  return 'Use the settings button (⚙) to choose the camera source for live capture.';
}}
function buildSourceSettingsMarkup(className) {{
  if (openSourceClass !== className || !sourceSettingsOpen || openSourceKind === 'upload') return '';
  if (openSourceKind === 'device') {{
    return `
      <div class="source-settings-panel" id="sourceSettingsPanel-${{cssSafe(className)}}">
        <div class="source-settings-grid">
          <label>
            Image Size
            <select id="sourceFrameSide-${{cssSafe(className)}}">${{buildSerialFrameSideOptions(currentSerialFrameSide)}}</select>
          </label>
          <label>
            Baud Rate
            <select id="sourceBaud-${{cssSafe(className)}}">${{buildSerialBaudOptions(currentSerialBaud)}}</select>
          </label>
          <label>
            Sync Header
            <input id="sourceSync-${{cssSafe(className)}}" value="${{escapeHtml(String(currentSerialSync || 'AA 55 AA'))}}" placeholder="AA 55 AA"/>
          </label>
        </div>
        <div class="source-settings-actions">
          <button class="source-settings-cancel" type="button" id="sourceSettingsCancel-${{cssSafe(className)}}">Close</button>
          <button class="source-settings-save" type="button" id="sourceSettingsSave-${{cssSafe(className)}}">Apply</button>
        </div>
      </div>
    `;
  }}
  return `
    <div class="source-settings-panel" id="sourceSettingsPanel-${{cssSafe(className)}}">
      <div class="source-settings-grid">
        <label>
          Camera
          <select id="sourceCamera-${{cssSafe(className)}}">${{buildWebcamOptions(currentWebcamIndex)}}</select>
        </label>
        <label>
          Preview Refresh
          <select id="sourcePreviewRate-${{cssSafe(className)}}">
            <option value="60"${{Number(previewIntervalMs) === 60 ? ' selected' : ''}}>Fast</option>
            <option value="80"${{Number(previewIntervalMs) === 80 ? ' selected' : ''}}>Balanced</option>
            <option value="120"${{Number(previewIntervalMs) === 120 ? ' selected' : ''}}>Stable</option>
            <option value="180"${{Number(previewIntervalMs) === 180 ? ' selected' : ''}}>Low CPU</option>
          </select>
        </label>
      </div>
      <div class="source-settings-actions">
        <button class="source-settings-cancel" type="button" id="sourceSettingsCancel-${{cssSafe(className)}}">Close</button>
        <button class="source-settings-save" type="button" id="sourceSettingsSave-${{cssSafe(className)}}">Apply</button>
      </div>
    </div>
  `;
}}
async function changeSerialBaud(className, value) {{
  const nextBaud = Number(value || currentSerialBaud || 921600);
  currentSerialBaud = nextBaud;
  STATE.current_serial_baud = currentSerialBaud;
  sourceSwitchInFlight = true;
  sourceSwitchClass = className;
  sourceSwitchKind = 'device';
  syncSourceActionButtons(className);
  try {{
    const res = await fetch(`${{baseUrl}}/live/config?session=${{encodeURIComponent(STATE.session)}}&serial_baud=${{encodeURIComponent(String(currentSerialBaud))}}`);
    const data = await res.json().catch(() => ({{ok:'0'}}));
    if (!res.ok || data.ok !== '1') throw new Error(data.error || 'Unable to update baud rate.');
    if (openSourceClass === className && openSourceKind === 'device') {{
      try {{
        await fetch(`${{baseUrl}}/live/close?session=${{encodeURIComponent(STATE.session)}}&source=device`);
      }} catch (e) {{}}
      await ensureOpenSourceLive();
    }}
  }} catch (err) {{
    toast(String(err && err.message ? err.message : err));
  }} finally {{
    sourceSwitchInFlight = false;
    sourceSwitchClass = '';
    sourceSwitchKind = '';
    syncSourceActionButtons(className);
  }}
}}
async function changeSerialFrameSide(className, value) {{
  const nextSide = Number(value || currentSerialFrameSide || 96);
  currentSerialFrameSide = nextSide;
  STATE.current_serial_frame_side = currentSerialFrameSide;
  sourceSwitchInFlight = true;
  sourceSwitchClass = className;
  sourceSwitchKind = 'device';
  syncSourceActionButtons(className);
  try {{
    const res = await fetch(`${{baseUrl}}/live/config?session=${{encodeURIComponent(STATE.session)}}&serial_frame_side=${{encodeURIComponent(String(currentSerialFrameSide))}}`);
    const data = await res.json().catch(() => ({{ok:'0'}}));
    if (!res.ok || data.ok !== '1') throw new Error(data.error || 'Unable to update device image size.');
    currentSerialFrameSide = Number(data.serial_frame_side || currentSerialFrameSide || 96);
    STATE.current_serial_frame_side = currentSerialFrameSide;
    if (openSourceClass === className && openSourceKind === 'device') {{
      try {{
        await fetch(`${{baseUrl}}/live/close?session=${{encodeURIComponent(STATE.session)}}&source=device`);
      }} catch (e) {{}}
      await ensureOpenSourceLive();
    }}
  }} catch (err) {{
    toast(String(err && err.message ? err.message : err));
  }} finally {{
    sourceSwitchInFlight = false;
    sourceSwitchClass = '';
    sourceSwitchKind = '';
    syncSourceActionButtons(className);
  }}
}}
async function changeSerialSync(className, value) {{
  const nextSync = String(value || currentSerialSync || 'AA 55 AA').trim() || 'AA 55 AA';
  currentSerialSync = nextSync;
  STATE.current_serial_sync = currentSerialSync;
  sourceSwitchInFlight = true;
  sourceSwitchClass = className;
  sourceSwitchKind = 'device';
  syncSourceActionButtons(className);
  try {{
    const res = await fetch(`${{baseUrl}}/live/config?session=${{encodeURIComponent(STATE.session)}}&serial_sync=${{encodeURIComponent(currentSerialSync)}}`);
    const data = await res.json().catch(() => ({{ok:'0'}}));
    if (!res.ok || data.ok !== '1') throw new Error(data.error || 'Unable to update sync header.');
    currentSerialSync = String(data.serial_sync || currentSerialSync || 'AA 55 AA');
    STATE.current_serial_sync = currentSerialSync;
    if (openSourceClass === className && openSourceKind === 'device') {{
      try {{
        await fetch(`${{baseUrl}}/live/close?session=${{encodeURIComponent(STATE.session)}}&source=device`);
      }} catch (e) {{}}
      await ensureOpenSourceLive();
    }}
  }} catch (err) {{
    toast(String(err && err.message ? err.message : err));
  }} finally {{
    sourceSwitchInFlight = false;
    sourceSwitchClass = '';
    sourceSwitchKind = '';
    syncSourceActionButtons(className);
  }}
}}
function toggleSourceSettings(className) {{
  if (openSourceClass !== className || !openSourceKind || openSourceKind === 'upload') return;
  sourceSettingsOpen = !sourceSettingsOpen;
  render();
}}
async function applySourceSettings(className) {{
  if (openSourceClass !== className) return;
  try {{
    if (openSourceKind === 'device') {{
      const sideEl = document.getElementById(`sourceFrameSide-${{cssSafe(className)}}`);
      const baudEl = document.getElementById(`sourceBaud-${{cssSafe(className)}}`);
      const syncEl = document.getElementById(`sourceSync-${{cssSafe(className)}}`);
      await changeSerialFrameSide(className, sideEl ? sideEl.value : String(currentSerialFrameSide));
      await changeSerialBaud(className, baudEl ? baudEl.value : String(currentSerialBaud));
      await changeSerialSync(className, syncEl ? syncEl.value : String(currentSerialSync));
    }} else if (openSourceKind === 'webcam') {{
      const camEl = document.getElementById(`sourceCamera-${{cssSafe(className)}}`);
      const rateEl = document.getElementById(`sourcePreviewRate-${{cssSafe(className)}}`);
      const nextCam = camEl ? camEl.value : String(currentWebcamIndex);
      if (String(nextCam) !== String(currentWebcamIndex)) {{
        await changeWebcamIndex(className, nextCam);
      }}
      previewIntervalMs = Number(rateEl ? rateEl.value : previewIntervalMs || 80);
      if (openSourceClass === className && openSourceKind === 'webcam') startPreviewLoop();
    }}
    sourceSettingsOpen = false;
    render();
  }} catch (err) {{
    toast(String(err && err.message ? err.message : err));
  }}
}}
async function changeDevicePort(className, value) {{
  if (captureInFlight && openSourceClass === className) {{
    toast('Wait for the current capture to finish before switching device.');
    syncSourceActionButtons(className);
    return;
  }}
  if (holdRecording && holdRecordClass === className && holdRecordSource === 'device') {{
    await stopHoldCapture();
  }}
  currentSerialPort = String(value || '');
  STATE.current_serial_port = currentSerialPort;
  sourceSwitchInFlight = true;
  sourceSwitchClass = className;
  sourceSwitchKind = 'device';
  syncSourceActionButtons(className);
  try {{
    const res = await fetch(`${{baseUrl}}/live/config?session=${{encodeURIComponent(STATE.session)}}&serial_port=${{encodeURIComponent(currentSerialPort)}}`);
    const data = await res.json().catch(() => ({{ok:'0'}}));
    if (!res.ok || data.ok !== '1') throw new Error(data.error || 'Unable to update serial port.');
    if (openSourceClass === className && openSourceKind === 'device') {{
      try {{
        await fetch(`${{baseUrl}}/live/close?session=${{encodeURIComponent(STATE.session)}}&source=device`);
      }} catch (e) {{}}
      await ensureOpenSourceLive();
    }}
  }} catch (err) {{
    toast(String(err && err.message ? err.message : err));
  }} finally {{
    sourceSwitchInFlight = false;
    sourceSwitchClass = '';
    sourceSwitchKind = '';
    syncSourceActionButtons(className);
  }}
}}
async function changeWebcamIndex(className, value) {{
  if (captureInFlight && openSourceClass === className) {{
    toast('Wait for the current capture to finish before switching camera.');
    syncSourceActionButtons(className);
    return;
  }}
  if (holdRecording && holdRecordClass === className && holdRecordSource === 'webcam') {{
    await stopHoldCapture();
  }}
  currentWebcamIndex = Number(value || 0);
  STATE.current_webcam_index = currentWebcamIndex;
  sourceSwitchInFlight = true;
  sourceSwitchClass = className;
  sourceSwitchKind = 'webcam';
  syncSourceActionButtons(className);
  try {{
    const res = await fetch(`${{baseUrl}}/live/config?session=${{encodeURIComponent(STATE.session)}}&webcam_index=${{encodeURIComponent(currentWebcamIndex)}}`);
    const data = await res.json().catch(() => ({{ok:'0'}}));
    if (!res.ok || data.ok !== '1') throw new Error(data.error || 'Unable to update webcam.');
    if (openSourceClass === className && openSourceKind === 'webcam') {{
      try {{
        await fetch(`${{baseUrl}}/live/close?session=${{encodeURIComponent(STATE.session)}}&source=webcam`);
      }} catch (e) {{}}
      await ensureOpenSourceLive();
    }}
  }} catch (err) {{
    toast(String(err && err.message ? err.message : err));
  }} finally {{
    sourceSwitchInFlight = false;
    sourceSwitchClass = '';
    sourceSwitchKind = '';
    syncSourceActionButtons(className);
  }}
}}
function sourceLabel(kind) {{
  if (kind === 'device') return 'Device';
  if (kind === 'upload') return 'Upload';
  return 'Webcam';
}}
function sourceSamplesTitle(kind) {{
  if (kind === 'upload') return 'Uploaded Samples';
  if (kind === 'device') return 'Device Samples';
  return 'Camera Samples';
}}
function buildSampleTileMarkup(item, idx) {{
  const src = escapeHtml(previewSrc(item));
  const filename = escapeHtml(previewFilename(item));
  return `
    <div class="sample-item">
      <img class="sample-thumb" src="${{src}}" alt="Sample ${{idx+1}}"/>
      <button class="sample-delete" type="button" data-filename="${{filename}}" aria-label="Delete sample">✕</button>
    </div>
  `;
}}
function buildSamplesMarkup(className) {{
  const items = normalizePreviewList((STATE.sample_previews && STATE.sample_previews[className]) || []);
  if (!items.length) return '<div class="samples-empty">No samples yet.</div>';
  return `<div class="samples-grid">${{items.map((item, idx) => buildSampleTileMarkup(item, idx)).join('')}}</div>`;
}}
function buildSamplesStripMarkup(className) {{
  const items = normalizePreviewList((STATE.sample_previews && STATE.sample_previews[className]) || []);
  if (!items.length) return '<div class="samples-strip-empty">No samples yet.</div>';
  return `<div class="samples-strip">${{items.map((item, idx) => buildSampleTileMarkup(item, idx)).join('')}}</div>`;
}}
function createIcon(kind) {{
  if (kind === 'webcam') return `<svg viewBox="0 0 24 24"><rect x="2" y="6" width="14" height="12" rx="2"></rect><path d="M22 8l-6 4 6 4V8z"></path></svg>`;
  if (kind === 'upload') return `<svg viewBox="0 0 24 24"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path><polyline points="17 8 12 3 7 8"></polyline><line x1="12" y1="3" x2="12" y2="15"></line></svg>`;
  return `<svg viewBox="0 0 24 24"><rect x="3" y="3" width="18" height="18" rx="2"></rect><line x1="9" y1="9" x2="15" y2="9"></line><line x1="9" y1="15" x2="15" y2="15"></line></svg>`;
}}
function setFlowPath(id, d) {{
  const p = document.getElementById(id);
  if (p) p.setAttribute('d', d);
}}
function setClassSampleCountLabel(className) {{
  const count = Number(STATE.counts[className] || 0);
  const summary = document.getElementById(`summaryTitle-${{cssSafe(className)}}`);
  if (summary) summary.textContent = `${{count}} ${{count === 1 ? 'Image Sample' : 'Image Samples'}}`;
  const sourceCount = document.getElementById(`sourceCount-${{cssSafe(className)}}`);
  if (sourceCount) sourceCount.innerHTML = `${{count}}<small>${{count === 1 ? 'Image Sample' : 'Image Samples'}}</small>`;
}}
function latestPreviewImage() {{
  if (openSourceClass) {{
    const items = normalizePreviewList(STATE.sample_previews && STATE.sample_previews[openSourceClass]);
    if (items.length) return previewSrc(items[0]);
  }}
  for (const name of (Array.isArray(STATE.classes) ? STATE.classes : [])) {{
    const items = normalizePreviewList(STATE.sample_previews && STATE.sample_previews[name]);
    if (items.length) return previewSrc(items[0]);
  }}
  return '';
}}
function stopPreviewPredictLoop() {{
  previewPredictToken += 1;
  if (previewPredictTimer) {{
    clearInterval(previewPredictTimer);
    previewPredictTimer = null;
  }}
}}
function renderOutputBars(labels, probs) {{
  const host = document.getElementById('previewOutput');
  if (!host) return;
  const ls = Array.isArray(labels) ? labels : [];
  const ps = Array.isArray(probs) ? probs : [];
  if (!ls.length || !ps.length) {{
    host.innerHTML = '';
    return;
  }}
  host.innerHTML = ls.map((label, idx) => {{
    const p = Math.max(0, Math.min(1, Number(ps[idx] || 0)));
    const pct = Math.round(p * 1000) / 10;
    return `
      <div class="out-row">
        <div>${{String(label)}}</div>
        <div class="out-bar">
          <div class="out-fill" style="width:${{pct}}%"></div>
          <div class="out-pct">${{pct}}%</div>
        </div>
      </div>
    `;
  }}).join('');
}}
async function refreshPreviewPrediction(token) {{
  if (!previewInputOn || !STATE.export_enabled) return;
  if (previewPredictInFlight) return;
  if (token !== previewPredictToken) return;
  previewPredictInFlight = true;
  const pane = document.getElementById('previewPane');
  const note = document.getElementById('previewNote');
  const imgId = 'previewImage';
  try {{
    const res = await fetch(`${{baseUrl}}/preview/predict?session=${{encodeURIComponent(STATE.session)}}&source=${{encodeURIComponent(previewSource)}}&preprocess=${{encodeURIComponent(previewPreprocessMode)}}&_ts=${{Date.now()}}`);
    const data = await res.json().catch(() => ({{ok:'0'}}));
    if (!res.ok || data.ok !== '1') throw new Error(data.error || 'Preview failed.');
    const rawSrc = data.image_b64 ? `data:image/png;base64,${{data.image_b64}}` : '';
    const roiSrc = data.processed_image_b64 ? `data:image/png;base64,${{data.processed_image_b64}}` : '';
    const src = previewShowRoi ? (roiSrc || rawSrc) : rawSrc;
    if (pane) {{
      if (!pane.dataset.ready) {{
        pane.innerHTML = `<img id="${{imgId}}" alt="Preview"/>`;
        pane.dataset.ready = '1';
      }}
      const img = document.getElementById(imgId);
      if (img && src) img.src = src;
    }}
    if (openSourceClass && openSourceKind === previewSource) {{
      const sImg = document.getElementById(`sourcePreview-${{cssSafe(openSourceClass)}}`);
      if (sImg && src) sImg.src = src;
    }}
    renderOutputBars(data.labels || [], data.probs || []);
    if (note) {{
      const top = data.top_label ? `${{String(data.top_label)}}` : '';
      const p = Math.round(Number(data.top_prob || 0) * 1000) / 10;
      const roiMsg = previewShowRoi && data.processed_variant ? ` · ROI: ${{String(data.processed_variant)}}` : '';
      note.textContent = top ? `Top: ${{top}} (${{p}}%)${{roiMsg}}` : roiMsg.trim();
    }}
  }} catch (err) {{
    if (note) note.textContent = String(err && err.message ? err.message : err);
  }} finally {{
    previewPredictInFlight = false;
  }}
}}
async function runPreviewUploadPrediction() {{
  if (!previewInputOn || !STATE.export_enabled) return;
  if (!previewUploadImageB64) {{
    const note = document.getElementById('previewNote');
    if (note) note.textContent = 'Choose an image in settings to test upload preview.';
    return;
  }}
  if (previewPredictInFlight) return;
  previewPredictInFlight = true;
  const pane = document.getElementById('previewPane');
  const note = document.getElementById('previewNote');
  try {{
    const res = await fetch(`${{baseUrl}}/preview/predict_upload`, {{
      method: 'POST',
      headers: {{'Content-Type':'application/json'}},
      body: JSON.stringify({{
        session: STATE.session,
        image_b64: previewUploadImageB64,
        preprocess: previewPreprocessMode,
      }})
    }});
    const data = await res.json().catch(() => ({{ok:'0'}}));
    if (!res.ok || data.ok !== '1') throw new Error(data.error || 'Preview failed.');
    const rawSrc = data.image_b64 ? `data:image/png;base64,${{data.image_b64}}` : previewUploadImageSrc;
    const roiSrc = data.processed_image_b64 ? `data:image/png;base64,${{data.processed_image_b64}}` : '';
    const src = previewShowRoi ? (roiSrc || rawSrc) : rawSrc;
    if (pane) {{
      pane.innerHTML = src ? `<img id="previewImage" src="${{src}}" alt="Preview"/>` : '<div class="preview-empty">Choose an upload image in settings.</div>';
      pane.dataset.ready = '1';
    }}
    renderOutputBars(data.labels || [], data.probs || []);
    if (note) {{
      const top = data.top_label ? `${{String(data.top_label)}}` : '';
      const p = Math.round(Number(data.top_prob || 0) * 1000) / 10;
      const roiMsg = previewShowRoi && data.processed_variant ? ` · ROI: ${{String(data.processed_variant)}}` : '';
      note.textContent = top ? `Upload: ${{String(previewUploadFilename || 'image')}} · Top: ${{top}} (${{p}}%)${{roiMsg}}` : `Upload: ${{String(previewUploadFilename || 'image')}}${{roiMsg}}`;
    }}
  }} catch (err) {{
    if (note) note.textContent = String(err && err.message ? err.message : err);
  }} finally {{
    previewPredictInFlight = false;
  }}
}}
function startPreviewPredictLoop() {{
  stopPreviewPredictLoop();
  previewPredictToken += 1;
  const token = previewPredictToken;
  stopPreviewLoop();
  if (previewSource === 'upload') {{
    runPreviewUploadPrediction();
    return;
  }}
  refreshPreviewPrediction(token);
  previewPredictTimer = window.setInterval(
    () => refreshPreviewPrediction(token),
    Math.max(50, Number(previewIntervalMs || 80))
  );
}}
async function setWebcamIndexGlobal(value) {{
  const next = Number(value || 0);
  currentWebcamIndex = next;
  STATE.current_webcam_index = currentWebcamIndex;
  try {{
    const res = await fetch(`${{baseUrl}}/live/config?session=${{encodeURIComponent(STATE.session)}}&webcam_index=${{encodeURIComponent(currentWebcamIndex)}}`);
    const data = await res.json().catch(() => ({{ok:'0'}}));
    if (!res.ok || data.ok !== '1') throw new Error(data.error || 'Unable to update webcam.');
    if (openSourceKind === 'webcam') {{
      try {{
        await fetch(`${{baseUrl}}/live/close?session=${{encodeURIComponent(STATE.session)}}&source=webcam`);
      }} catch (e) {{}}
      await ensureOpenSourceLive();
    }}
  }} catch (err) {{
    toast(String(err && err.message ? err.message : err));
  }}
}}
async function setDevicePortGlobal(value) {{
  currentSerialPort = String(value || '');
  STATE.current_serial_port = currentSerialPort;
  try {{
    const res = await fetch(`${{baseUrl}}/live/config?session=${{encodeURIComponent(STATE.session)}}&serial_port=${{encodeURIComponent(currentSerialPort)}}`);
    const data = await res.json().catch(() => ({{ok:'0'}}));
    if (!res.ok || data.ok !== '1') throw new Error(data.error || 'Unable to update serial port.');
    if (openSourceKind === 'device') {{
      try {{
        await fetch(`${{baseUrl}}/live/close?session=${{encodeURIComponent(STATE.session)}}&source=device`);
      }} catch (e) {{}}
      await ensureOpenSourceLive();
    }}
  }} catch (err) {{
    toast(String(err && err.message ? err.message : err));
  }}
}}
async function setSerialBaudGlobal(value) {{
  currentSerialBaud = Number(value || currentSerialBaud || 921600);
  STATE.current_serial_baud = currentSerialBaud;
  try {{
    const res = await fetch(`${{baseUrl}}/live/config?session=${{encodeURIComponent(STATE.session)}}&serial_baud=${{encodeURIComponent(String(currentSerialBaud))}}`);
    const data = await res.json().catch(() => ({{ok:'0'}}));
    if (!res.ok || data.ok !== '1') throw new Error(data.error || 'Unable to update baud rate.');
    if (openSourceKind === 'device') {{
      try {{
        await fetch(`${{baseUrl}}/live/close?session=${{encodeURIComponent(STATE.session)}}&source=device`);
      }} catch (e) {{}}
      await ensureOpenSourceLive();
    }}
  }} catch (err) {{
    toast(String(err && err.message ? err.message : err));
  }}
}}
async function setSerialFrameSideGlobal(value) {{
  currentSerialFrameSide = Number(value || currentSerialFrameSide || 96);
  STATE.current_serial_frame_side = currentSerialFrameSide;
  try {{
    const res = await fetch(`${{baseUrl}}/live/config?session=${{encodeURIComponent(STATE.session)}}&serial_frame_side=${{encodeURIComponent(String(currentSerialFrameSide))}}`);
    const data = await res.json().catch(() => ({{ok:'0'}}));
    if (!res.ok || data.ok !== '1') throw new Error(data.error || 'Unable to update device image size.');
    currentSerialFrameSide = Number(data.serial_frame_side || currentSerialFrameSide || 96);
    STATE.current_serial_frame_side = currentSerialFrameSide;
    if (openSourceKind === 'device') {{
      try {{
        await fetch(`${{baseUrl}}/live/close?session=${{encodeURIComponent(STATE.session)}}&source=device`);
      }} catch (e) {{}}
      await ensureOpenSourceLive();
    }}
  }} catch (err) {{
    toast(String(err && err.message ? err.message : err));
  }}
}}
async function setSerialSyncGlobal(value) {{
  currentSerialSync = String(value || currentSerialSync || 'AA 55 AA').trim() || 'AA 55 AA';
  STATE.current_serial_sync = currentSerialSync;
  try {{
    const res = await fetch(`${{baseUrl}}/live/config?session=${{encodeURIComponent(STATE.session)}}&serial_sync=${{encodeURIComponent(currentSerialSync)}}`);
    const data = await res.json().catch(() => ({{ok:'0'}}));
    if (!res.ok || data.ok !== '1') throw new Error(data.error || 'Unable to update sync header.');
    currentSerialSync = String(data.serial_sync || currentSerialSync || 'AA 55 AA');
    STATE.current_serial_sync = currentSerialSync;
    if (openSourceKind === 'device') {{
      try {{
        await fetch(`${{baseUrl}}/live/close?session=${{encodeURIComponent(STATE.session)}}&source=device`);
      }} catch (e) {{}}
      await ensureOpenSourceLive();
    }}
  }} catch (err) {{
    toast(String(err && err.message ? err.message : err));
  }}
}}
function buildPreviewSettingsMarkup() {{
  if (!previewSettingsOpen || !STATE.export_enabled) return '';
  const exportFields = `
    <label>
      Export Name
      <input id="previewExportName" value="${{String(exportModelName || 'model')}}" placeholder="person_detect"/>
    </label>
    <label>
      Array Name (optional)
      <input id="previewExportArray" value="${{String(exportArrayName || 'g_model')}}" placeholder="g_person_detect_model_data"/>
    </label>
    <label>
      Input Source
      <select id="previewInputSource">
        <option value="webcam"${{previewSource === 'webcam' ? ' selected' : ''}}>Webcam</option>
        <option value="device"${{previewSource === 'device' ? ' selected' : ''}}>Device</option>
        <option value="upload"${{previewSource === 'upload' ? ' selected' : ''}}>Upload File</option>
      </select>
    </label>
  `;
  if (previewSource === 'upload') {{
    return `
      <div class="source-settings-panel" id="previewSettingsPanel">
        <div class="source-settings-grid">
          ${{exportFields}}
          <label>
            Test Image
            <button class="upload-pick" type="button" id="previewUploadPick">Choose image</button>
          </label>
          <label>
            Selected File
            <input id="previewUploadName" value="${{escapeHtml(String(previewUploadFilename || 'No file selected'))}}" readonly/>
          </label>
        </div>
        <div class="source-settings-actions">
          <button class="source-settings-cancel" type="button" id="previewSettingsCancel">Close</button>
          <button class="source-settings-save" type="button" id="previewSettingsSave">Apply</button>
        </div>
      </div>
    `;
  }}
  if (previewSource === 'device') {{
    return `
      <div class="source-settings-panel" id="previewSettingsPanel">
        <div class="source-settings-grid">
          ${{exportFields}}
          <label>
            Device Port
            <select id="previewDevicePort">${{buildDeviceOptions(currentSerialPort)}}</select>
          </label>
          <label>
            Image Size
            <select id="previewDeviceFrameSide">${{buildSerialFrameSideOptions(currentSerialFrameSide)}}</select>
          </label>
          <label>
            Baud Rate
            <select id="previewDeviceBaud">${{buildSerialBaudOptions(currentSerialBaud)}}</select>
          </label>
          <label>
            Sync Header
            <input id="previewDeviceSync" value="${{escapeHtml(String(currentSerialSync || 'AA 55 AA'))}}" placeholder="AA 55 AA"/>
          </label>
        </div>
        <div class="source-settings-actions">
          <button class="source-settings-cancel" type="button" id="previewSettingsCancel">Close</button>
          <button class="source-settings-save" type="button" id="previewSettingsSave">Apply</button>
        </div>
      </div>
    `;
  }}
  return `
    <div class="source-settings-panel" id="previewSettingsPanel">
      <div class="source-settings-grid">
        ${{exportFields}}
        <label>
          Camera
          <select id="previewCamera">${{buildWebcamOptions(currentWebcamIndex)}}</select>
        </label>
        <label>
          Preview Refresh
          <select id="previewPreviewRate">
            <option value="60"${{Number(previewIntervalMs) === 60 ? ' selected' : ''}}>Fast</option>
            <option value="80"${{Number(previewIntervalMs) === 80 ? ' selected' : ''}}>Balanced</option>
            <option value="120"${{Number(previewIntervalMs) === 120 ? ' selected' : ''}}>Stable</option>
            <option value="180"${{Number(previewIntervalMs) === 180 ? ' selected' : ''}}>Low CPU</option>
          </select>
        </label>
      </div>
      <div class="source-settings-actions">
        <button class="source-settings-cancel" type="button" id="previewSettingsCancel">Close</button>
        <button class="source-settings-save" type="button" id="previewSettingsSave">Apply</button>
      </div>
    </div>
  `;
}}
function renderPreviewSettings() {{
  const host = document.getElementById('previewSettingsHost');
  if (!host) return;
  host.innerHTML = buildPreviewSettingsMarkup();
  updateFlow();
  if (previewSource === 'device') {{
    const currentPorts = Array.isArray(STATE.serial_ports) ? STATE.serial_ports : [];
    if (!currentPorts.length) {{
      refreshSerialPorts(true).catch(() => {{}});
    }}
  }}
  const cancel = document.getElementById('previewSettingsCancel');
  const save = document.getElementById('previewSettingsSave');
  const portSel = document.getElementById('previewDevicePort');
  const sourceSel = document.getElementById('previewInputSource');
  const uploadPick = document.getElementById('previewUploadPick');
  if (portSel) {{
    portSel.onpointerdown = () => refreshSerialPorts(false, 'previewDevicePort');
    portSel.onmousedown = () => refreshSerialPorts(false, 'previewDevicePort');
  }}
  if (sourceSel) sourceSel.onchange = () => {{
    previewSource = String(sourceSel.value || 'webcam');
    renderPreviewSettings();
  }};
  if (uploadPick) uploadPick.onclick = async () => {{
    await pickPreviewUploadFile();
    renderPreviewSettings();
  }};
  if (cancel) cancel.onclick = () => {{
    previewSettingsOpen = false;
    renderPreviewSettings();
  }};
  if (save) save.onclick = async () => {{
    try {{
      const nameEl = document.getElementById('previewExportName');
      const arrayEl = document.getElementById('previewExportArray');
      const sourceEl = document.getElementById('previewInputSource');
      exportModelName = String(nameEl ? nameEl.value : exportModelName || 'model').trim() || 'model';
      exportArrayName = String(arrayEl ? arrayEl.value : exportArrayName || 'g_model').trim();
      const nextSource = String(sourceEl ? sourceEl.value : previewSource || 'webcam');
      previewSource = nextSource === 'device' ? 'device' : (nextSource === 'upload' ? 'upload' : 'webcam');
      persistPreviewState();
      persistExportSettings();
      if (previewSource === 'device') {{
        const portEl = document.getElementById('previewDevicePort');
        const sideEl = document.getElementById('previewDeviceFrameSide');
        const baudEl = document.getElementById('previewDeviceBaud');
        const syncEl = document.getElementById('previewDeviceSync');
        await setDevicePortGlobal(portEl ? portEl.value : currentSerialPort);
        await setSerialFrameSideGlobal(sideEl ? sideEl.value : String(currentSerialFrameSide));
        await setSerialBaudGlobal(baudEl ? baudEl.value : String(currentSerialBaud));
        await setSerialSyncGlobal(syncEl ? syncEl.value : String(currentSerialSync));
      }} else if (previewSource === 'webcam') {{
        const camEl = document.getElementById('previewCamera');
        const rateEl = document.getElementById('previewPreviewRate');
        if (camEl && String(camEl.value) !== String(currentWebcamIndex)) await setWebcamIndexGlobal(camEl.value);
        if (rateEl) previewIntervalMs = Number(rateEl.value || previewIntervalMs || 80);
      }}
      previewSettingsOpen = false;
      renderPreviewSettings();
      if (previewInputOn) {{
        if (previewPredictTimer) stopPreviewPredictLoop();
        startPreviewPredictLoop();
      }} else if (previewSource !== 'upload') {{
        startPreviewLoop();
      }} else {{
        const note = document.getElementById('previewNote');
        if (note) note.textContent = previewUploadFilename ? `Upload ready: ${{previewUploadFilename}}` : 'Choose an image in settings to test upload preview.';
      }}
    }} catch (err) {{
      toast(String(err && err.message ? err.message : err));
    }}
  }};
}}
function renderTrainStatus() {{
  const el = document.getElementById('trainStatus');
  if (!el) return;
  el.textContent = STATE.export_enabled ? 'Model Trained' : 'Not trained';
}}
function renderPreviewCard() {{
  const pane = document.getElementById('previewPane');
  const note = document.getElementById('previewNote');
  const output = document.getElementById('previewOutput');
  const toggle = document.getElementById('previewInputToggle');
  const roiToggle = document.getElementById('previewRoiToggle');
  const settingsBtn = document.getElementById('previewSettingsToggle');
  const modeTabs = Array.from(document.querySelectorAll('[data-preview-mode]'));
  if (!pane || !note || !output || !toggle || !settingsBtn || !roiToggle) return;
  if (!STATE.export_enabled) {{
    stopPreviewPredictLoop();
    previewSettingsOpen = false;
    renderPreviewSettings();
    pane.removeAttribute('data-ready');
    pane.innerHTML = '<div class="preview-empty">Train a model on the left to enable preview.</div>';
    output.innerHTML = '';
    note.textContent = 'You must train a model on the left before you can preview it here.';
    toggle.checked = false;
    toggle.disabled = true;
    roiToggle.checked = false;
    roiToggle.disabled = true;
    settingsBtn.disabled = true;
    modeTabs.forEach((btn) => {{
      btn.disabled = true;
      btn.classList.remove('active');
    }});
    return;
  }}
  toggle.disabled = false;
  roiToggle.disabled = false;
  settingsBtn.disabled = false;
  modeTabs.forEach((btn) => {{
    const active = String(btn.getAttribute('data-preview-mode') || '') === String(previewPreprocessMode || 'auto_by_label');
    btn.disabled = false;
    btn.classList.toggle('active', active);
  }});
  toggle.checked = !!previewInputOn;
  roiToggle.checked = !!previewShowRoi;
  renderPreviewSettings();
  if (!pane.dataset.ready) {{
    const src = previewSource === 'upload' ? previewUploadImageSrc : latestPreviewImage();
    pane.innerHTML = src ? `<img id="previewImage" src="${{src}}" alt="Preview"/>` : '<div class="preview-empty">Turn on Input to preview live predictions.</div>';
    pane.dataset.ready = '1';
  }}
  bindLayoutImageObservers(pane);
  scheduleLayoutResync();
  if (previewInputOn) {{
    if (!previewPredictTimer) startPreviewPredictLoop();
  }} else {{
    if (previewPredictTimer) stopPreviewPredictLoop();
    output.innerHTML = '';
    note.textContent = previewSource === 'upload'
      ? (previewUploadFilename ? `Upload ready: ${{previewUploadFilename}}` : 'Choose an image in settings to test upload preview.')
      : 'Model ready. Turn on Input to preview it here.';
    if (openSourceClass && (openSourceKind === 'webcam' || openSourceKind === 'device')) startPreviewLoop();
  }}
}}
function bindPreviewControls() {{
  const toggle = document.getElementById('previewInputToggle');
  const roiToggle = document.getElementById('previewRoiToggle');
  const settings = document.getElementById('previewSettingsToggle');
  const modeTabs = Array.from(document.querySelectorAll('[data-preview-mode]'));
  if (!toggle || !roiToggle || !settings) return;
  if (toggle.dataset.bound === '1') return;
  toggle.dataset.bound = '1';
  roiToggle.dataset.bound = '1';
  toggle.onchange = () => {{
    previewInputOn = !!toggle.checked;
    persistPreviewState();
    renderPreviewCard();
  }};
  roiToggle.onchange = () => {{
    previewShowRoi = !!roiToggle.checked;
    persistPreviewState();
    renderPreviewCard();
    if (previewInputOn) {{
      if (previewPredictTimer) stopPreviewPredictLoop();
      startPreviewPredictLoop();
    }}
  }};
  settings.onclick = () => {{
    previewSettingsOpen = !previewSettingsOpen;
    renderPreviewSettings();
  }};
  modeTabs.forEach((btn) => {{
    if (btn.dataset.bound === '1') return;
    btn.dataset.bound = '1';
    btn.onclick = () => {{
      previewPreprocessMode = String(btn.getAttribute('data-preview-mode') || 'auto_by_label');
      persistPreviewState();
      renderPreviewCard();
      if (previewInputOn) {{
        if (previewPredictTimer) stopPreviewPredictLoop();
        startPreviewPredictLoop();
      }}
    }};
  }});
}}
function updateOpenSamplesPanel(className) {{
  const host = document.getElementById(`samplesHost-${{cssSafe(className)}}`);
  if (!host) return;
  host.innerHTML = buildSamplesMarkup(className);
  bindSampleDeleteButtons(host, className);
  bindLayoutImageObservers(host);
  setClassSampleCountLabel(className);
  renderPreviewCard();
  syncTrainUi();
  scheduleLayoutResync();
}}
function applyClassesState(nextClasses, oldClasses) {{
  const prevClasses = Array.isArray(oldClasses) ? oldClasses.slice() : STATE.classes.slice();
  const oldCounts = Object.assign({{}}, STATE.counts || {{}});
  const oldPreviews = Object.assign({{}}, STATE.sample_previews || {{}});
  const oldProcessed = Object.assign({{}}, STATE.processed_previews || {{}});
  const oldClassPreprocess = Object.assign({{}}, STATE.class_preprocess || {{}});
  const oldSamplePreprocess = normalizeSamplePreprocessMap(STATE.sample_preprocess || {{}});
  const nextCounts = {{}};
  const nextPreviews = {{}};
  const nextProcessed = {{}};
  const nextClassPreprocess = {{}};
  const nextSamplePreprocess = {{}};
  (Array.isArray(nextClasses) ? nextClasses : []).forEach((name, idx) => {{
    const prevName = prevClasses[idx];
    nextCounts[name] = Number(nullish(oldCounts[name], nullish(oldCounts[prevName], 0)));
    nextPreviews[name] = normalizePreviewList(
      Array.isArray(oldPreviews[name]) ? oldPreviews[name] : oldPreviews[prevName]
    );
    nextProcessed[name] = normalizePreviewList(
      Array.isArray(oldProcessed[name]) ? oldProcessed[name] : oldProcessed[prevName]
    );
    nextClassPreprocess[name] = normalizeClassPreprocessConfig(oldClassPreprocess[name] || oldClassPreprocess[prevName]);
    const sampleMap = oldSamplePreprocess[name] || oldSamplePreprocess[prevName];
    if (sampleMap && typeof sampleMap === 'object' && Object.keys(sampleMap).length) {{
      nextSamplePreprocess[name] = normalizeSamplePreprocessMap({{ [name]: sampleMap }})[name] || {{}};
    }}
  }});
  STATE.classes = Array.isArray(nextClasses) ? nextClasses.slice() : [];
  STATE.counts = nextCounts;
  STATE.sample_previews = nextPreviews;
  STATE.processed_previews = nextProcessed;
  STATE.class_preprocess = nextClassPreprocess;
  STATE.sample_preprocess = nextSamplePreprocess;
}}
async function deleteSample(className, filename) {{
  if (!className || !filename) return;
  try {{
    const res = await fetch(`${{baseUrl}}/samples/delete`, {{
      method: 'POST',
      headers: {{'Content-Type':'application/json'}},
      body: JSON.stringify({{
        session: STATE.session,
        class: className,
        filename
      }})
    }});
    const data = await res.json().catch(() => ({{ok:'0'}}));
    if (!res.ok || data.ok !== '1') throw new Error(data.error || 'Unable to delete sample.');
    const next = data.state || {{}};
    STATE.counts[className] = Number(next.count || 0);
    STATE.sample_previews[className] = normalizePreviewList(next.previews);
    STATE.processed_previews[className] = normalizePreviewList(
      normalizePreviewList(STATE.processed_previews[className]).filter((item) => String(previewFilename(item) || '') !== String(filename || ''))
    );
    deleteSamplePreprocessConfig(className, filename);
    if (classPreprocessOpen && classPreprocessClass === className) {{
      const remain = normalizePreviewList(STATE.sample_previews[className]);
      if (!remain.length) {{
        classPreprocessSampleIndex = 0;
        classPreprocessProcessedSrc = '';
      }} else if (classPreprocessSampleIndex >= remain.length) {{
        classPreprocessSampleIndex = remain.length - 1;
      }}
      renderClassPreprocessModal();
      refreshClassProcessedPreview();
    }}
    recomputeTrainEnabled();
    syncTrainUi();
    const openHost = document.getElementById(`samplesHost-${{cssSafe(className)}}`);
    if (openHost) {{
      updateOpenSamplesPanel(className);
    }} else {{
      render();
    }}
  }} catch (err) {{
    toast(String(err && err.message ? err.message : err));
  }}
}}
function bindSampleDeleteButtons(scope, className) {{
  if (!scope || !className) return;
  const buttons = scope.querySelectorAll('.sample-delete[data-filename]');
  buttons.forEach((btn) => {{
    if (btn.dataset.bound === '1') return;
    btn.dataset.bound = '1';
    btn.onclick = async (e) => {{
      e.preventDefault();
      e.stopPropagation();
      const filename = String(btn.dataset.filename || '');
      await deleteSample(className, filename);
    }};
  }});
}}
async function renameClass(oldName) {{
  const nextNameRaw = window.prompt('Rename class', oldName);
  if (nextNameRaw == null) return;
  const nextName = String(nextNameRaw).trim();
  if (!nextName || nextName === oldName) return;
  try {{
    const res = await fetch(`${{baseUrl}}/classes/rename`, {{
      method: 'POST',
      headers: {{'Content-Type':'application/json'}},
      body: JSON.stringify({{
        session: STATE.session,
        old_name: oldName,
        new_name: nextName
      }})
    }});
    const data = await res.json().catch(() => ({{ok:'0'}}));
    if (!res.ok || data.ok !== '1') throw new Error(data.error || 'Unable to rename class.');
    applyClassesState(Array.isArray(data.classes) ? data.classes : STATE.classes, STATE.classes);
    if (classPreprocessOpen && classPreprocessClass === oldName) {{
      closeClassPreprocessEditor();
    }}
    if (openSourceClass === oldName) openSourceClass = nextName;
    if (holdRecordClass === oldName) holdRecordClass = nextName;
    persistOpenSourceState();
    recomputeTrainEnabled();
    render();
  }} catch (err) {{
    toast(String(err && err.message ? err.message : err));
  }}
}}
async function addClass() {{
  try {{
    const prev = STATE.classes.slice();
    const res = await fetch(`${{baseUrl}}/classes/add`, {{
      method: 'POST',
      headers: {{'Content-Type':'application/json'}},
      body: JSON.stringify({{session: STATE.session}})
    }});
    const data = await res.json().catch(() => ({{ok:'0'}}));
    if (!res.ok || data.ok !== '1') throw new Error(data.error || 'Unable to add class.');
    applyClassesState(Array.isArray(data.classes) ? data.classes : prev, prev);
    recomputeTrainEnabled();
    render();
  }} catch (err) {{
    toast(String(err && err.message ? err.message : err));
  }}
}}
async function deleteClass(name) {{
  if (STATE.classes.length <= 2) return;
  const ok = await showConfirmDialog(
    'Delete this class?',
    `Class "${{String(name || '')}}" and all of its samples will be removed.`,
    'Delete'
  );
  if (!ok) return;
  try {{
    const prev = STATE.classes.slice();
    const res = await fetch(`${{baseUrl}}/classes/delete`, {{
      method: 'POST',
      headers: {{'Content-Type':'application/json'}},
      body: JSON.stringify({{
        session: STATE.session,
        name
      }})
    }});
    const data = await res.json().catch(() => ({{ok:'0'}}));
    if (!res.ok || data.ok !== '1') throw new Error(data.error || 'Unable to delete class.');
    applyClassesState(Array.isArray(data.classes) ? data.classes : prev.filter((x) => x !== name), prev);
    if (classPreprocessOpen && classPreprocessClass === name) {{
      closeClassPreprocessEditor();
    }}
    if (openSourceClass === name) {{
      openSourceClass = '';
      openSourceKind = '';
      clearOpenSourceState();
    }}
    if (holdRecordClass === name) {{
      holdRecording = false;
      holdRecordClass = '';
      holdRecordSource = '';
    }}
    recomputeTrainEnabled();
    render();
  }} catch (err) {{
    toast(String(err && err.message ? err.message : err));
  }}
}}
function updateFlow() {{
  const wrap = document.querySelector('.wrap');
  const svg = document.querySelector('svg.flow');
  const train = document.getElementById('trainCard');
  const preview = document.getElementById('previewCard');
  const cards = Array.from(document.querySelectorAll('.class-card'));
  if (!wrap || !svg || !train || !preview || cards.length < 1) return;
  const wr = wrap.getBoundingClientRect();
  // #region debug-point E:update-flow
  dbgEvent('E', 'app.py:updateFlow', '[DEBUG] updateFlow geometry snapshot', {{
    wrapWidth: Math.round(wr.width || 0),
    wrapHeight: Math.round(wr.height || 0),
    cardCount: cards.length,
    trainHeight: Math.round((train.getBoundingClientRect().height || 0)),
    previewHeight: Math.round((preview.getBoundingClientRect().height || 0)),
  }});
  // #endregion
  svg.setAttribute('viewBox', `0 0 ${{Math.max(1, Math.round(wr.width))}} ${{Math.max(1, Math.round(wr.height))}}`);
  const t = train.getBoundingClientRect();
  const p = preview.getBoundingClientRect();
  const tx = t.left - wr.left;
  const ty = t.top + t.height * 0.5 - wr.top;
  const tr = t.right - wr.left;
  const py = p.top + p.height * 0.5 - wr.top;
  const px = p.left - wr.left;
  const trainPreviewPath = document.getElementById('flowTrainPreview');
  const existing = Array.from(svg.querySelectorAll('path[id^="flowClass"]'));
  for (const p of existing) {{
    const m = String(p.id || '').match(/^flowClass(\d+)$/);
    if (!m) continue;
    const idx = Number(m[1] || 0);
    if (idx > cards.length) p.remove();
  }}
  for (let i = 0; i < cards.length; i++) {{
    const id = `flowClass${{i + 1}}`;
    let path = document.getElementById(id);
    if (!path) {{
      path = document.createElementNS('http://www.w3.org/2000/svg', 'path');
      path.setAttribute('id', id);
      path.setAttribute('stroke', 'rgba(0,0,0,0.16)');
      path.setAttribute('stroke-width', '2');
      path.setAttribute('fill', 'none');
      if (trainPreviewPath && trainPreviewPath.parentNode === svg) {{
        svg.insertBefore(path, trainPreviewPath);
      }} else {{
        svg.appendChild(path);
      }}
    }}
    const cr = cards[i].getBoundingClientRect();
    const cx = cr.right - wr.left;
    const cy = cr.top + cr.height * 0.5 - wr.top;
    const offset = (i - (cards.length - 1) / 2) * 14;
    setFlowPath(id, `M ${{cx}} ${{cy}} C ${{cx + 44}} ${{cy}}, ${{tx - 44}} ${{ty + offset}}, ${{tx}} ${{ty}}`);
  }}
  setFlowPath('flowTrainPreview', `M ${{tr}} ${{ty}} C ${{tr + 34}} ${{ty}}, ${{px - 34}} ${{py}}, ${{px}} ${{py}}`);
}}
let trainAdvancedOpen = false;
function manualRoiOrDefault(cfg) {{
  const roi = cfg && Array.isArray(cfg.manual_roi) && cfg.manual_roi.length === 4 ? cfg.manual_roi : [0.00, 0.00, 1.00, 1.00];
  return roi.map((v, idx) => {{
    const n = Number(v);
    if (!isFinite(n)) return idx < 2 ? 0 : 1;
    return Math.max(0, Math.min(1, n));
  }});
}}
function syncAdvancedInlineInputs() {{
  const cfg = STATE.train_cfg || {{}};
  const set = (id, value) => {{
    const el = document.getElementById(id);
    if (!el) return;
    el.value = nullish(value, nullish(el.value, ''));
  }};
  set('advImgSizeInline', nullish(cfg.img_size, 96));
  set('advBatchInline', nullish(cfg.batch_size, 32));
  set('advEpochsInline', nullish(cfg.epochs, 20));
  set('advValInline', nullish(cfg.validation_split, 0.25));
  set('advLrInline', nullish(cfg.learning_rate, 0.0016));
  set('advConv1Inline', nullish(cfg.conv1_filters, 8));
  set('advConv2Inline', nullish(cfg.conv2_filters, 16));
  set('advDenseInline', nullish(cfg.dense_units, 32));
}}
function setTrainCfgField(key, rawValue) {{
  const cfg = STATE.train_cfg || {{}};
  let v = rawValue;
  if (key === 'preprocess_mode') {{
    v = String(rawValue || 'auto_by_label');
    if (!['auto_by_label', 'manual_roi', 'none'].includes(v)) return;
  }} else if (key === 'validation_split' || key === 'learning_rate') {{
    v = Number(v);
    if (!isFinite(v)) return;
  }} else {{
    v = parseInt(String(v), 10);
    if (!isFinite(v)) return;
  }}
  if (key === 'img_size') {{
    v = Math.max(32, Math.min(224, v));
  }}
  const next = Object.assign({{}}, cfg);
  next[key] = v;
  STATE.train_cfg = next;
  persistTrainCfgStorage();
  renderTrainStatus();
}}
function setManualRoiField(index, rawValue) {{
  const cfg = Object.assign({{}}, STATE.train_cfg || {{}});
  const roi = manualRoiOrDefault(cfg);
  const n = Number(rawValue);
  if (!isFinite(n)) return;
  roi[index] = Math.max(0, Math.min(1, n));
  roi[2] = Math.max(roi[2], roi[0] + 0.01);
  roi[3] = Math.max(roi[3], roi[1] + 0.01);
  roi[0] = Math.min(roi[0], roi[2] - 0.01);
  roi[1] = Math.min(roi[1], roi[3] - 0.01);
  cfg.manual_roi = roi;
  STATE.train_cfg = cfg;
  persistTrainCfgStorage();
}}
function renderAdvancedPanel() {{
  const panel = document.getElementById('trainAdvPanel');
  const chev = document.getElementById('advChevron');
  if (!panel) return;
  panel.style.display = trainAdvancedOpen ? 'block' : 'none';
  if (chev) chev.textContent = trainAdvancedOpen ? '▴' : '▾';
  if (trainAdvancedOpen) syncAdvancedInlineInputs();
  updateFlow();
}}
function toggleAdvancedInline() {{
  trainAdvancedOpen = !trainAdvancedOpen;
  renderAdvancedPanel();
}}
function bindAdvancedInlineHandlers() {{
  const panel = document.getElementById('trainAdvPanel');
  if (!panel || panel.dataset.bound === '1') return;
  panel.dataset.bound = '1';
  const bindNum = (id, key) => {{
    const el = document.getElementById(id);
    if (!el) return;
    el.addEventListener('input', (e) => setTrainCfgField(key, e.target.value));
    el.addEventListener('change', (e) => setTrainCfgField(key, e.target.value));
  }};
  bindNum('advImgSizeInline', 'img_size');
  bindNum('advBatchInline', 'batch_size');
  bindNum('advEpochsInline', 'epochs');
  bindNum('advValInline', 'validation_split');
  bindNum('advLrInline', 'learning_rate');
  bindNum('advConv1Inline', 'conv1_filters');
  bindNum('advConv2Inline', 'conv2_filters');
  bindNum('advDenseInline', 'dense_units');
}}
function resetTrainCfg() {{
  STATE.train_cfg = {{
    img_size: 96,
    batch_size: 32,
    epochs: 20,
    validation_split: 0.25,
    learning_rate: 0.0016,
    conv1_filters: 8,
    conv2_filters: 16,
    dense_units: 32
  }};
  persistTrainCfgStorage();
  syncAdvancedInlineInputs();
  renderTrainStatus();
  toast('Advanced settings reset.');
}}
function render() {{
  // #region debug-point B:render-entry
  dbgEvent('B', 'app.py:render', '[DEBUG] render start', {{classes: STATE.classes, openSourceClass, openSourceKind, initial_open_source_class: STATE.initial_open_source_class, initial_open_source_kind: STATE.initial_open_source_kind}});
  // #endregion
  const root = document.getElementById('classes');
  root.innerHTML = '';
  for (let i=0; i<STATE.classes.length; i++) {{
    const name = STATE.classes[i];
    const card = document.createElement('div');
    card.className = 'card class-card';
    const head = document.createElement('div');
    head.className = 'class-head';
    const title = document.createElement('div');
    title.className = 'class-title';
    title.textContent = name;
    const preprocessBtn = document.createElement('button');
    preprocessBtn.className = 'preprocess-chip';
    preprocessBtn.type = 'button';
    preprocessBtn.textContent = 'Edit';
    preprocessBtn.onclick = (e) => {{
      if (e) {{
        e.preventDefault();
        e.stopPropagation();
      }}
      openClassPreprocessEditor(name);
    }};
    const edit = document.createElement('button');
    edit.className = 'iconbtn';
    edit.textContent = '✎';
    edit.onclick = () => renameClass(name);
    title.appendChild(preprocessBtn);
    title.appendChild(edit);
    const labelSelect = document.createElement('select');
    labelSelect.className = 'class-label-select';
    labelSelect.innerHTML = '<option value="sign">Sign</option><option value="road">Road</option>';
    labelSelect.value = getClassLabel(name);
    labelSelect.onchange = (e) => {{
      if (e) {{ e.stopPropagation(); }}
      setClassLabel(name, e && e.target ? e.target.value : 'sign');
    }};
    labelSelect.onclick = (e) => {{ e.stopPropagation(); }};
    title.appendChild(labelSelect);
    head.appendChild(title);
    const more = document.createElement('button');
    more.className = 'iconbtn more';
    more.textContent = '⋮';
    more.onclick = async (e) => {{
      if (e) {{
        e.preventDefault();
        e.stopPropagation();
      }}
      await deleteClass(name);
    }};
    card.appendChild(more);
    card.appendChild(head);
    const div = document.createElement('div');
    div.className = 'divider';
    card.appendChild(div);
    const c = Number(STATE.counts[name] || 0);
    const sh = document.createElement('div');
    sh.className = 'summary-title';
    sh.id = `summaryTitle-${{cssSafe(name)}}`;
    sh.textContent = `${{c}} ${{c === 1 ? 'Image Sample' : 'Image Samples'}}`;
    card.appendChild(sh);

    const row = document.createElement('div');
    row.className = openSourceClass === name ? 'btnrow' : 'summary-actions';

    const upWebcam = document.createElement('button');
    upWebcam.className = 'sample';
    upWebcam.innerHTML = createIcon('webcam') + '<div>Webcam</div>';
    upWebcam.onclick = () => openSourcePanel('webcam', name);

    const upDevice = document.createElement('button');
    upDevice.className = 'sample';
    upDevice.innerHTML = createIcon('device') + '<div>Device</div>';
    upDevice.onclick = () => openSourcePanel('device', name);

    const upUpload = document.createElement('button');
    upUpload.className = 'sample';
    upUpload.innerHTML = createIcon('upload') + '<div>Upload</div>';
    upUpload.onclick = () => openSourcePanel('upload', name);

    row.appendChild(upWebcam);
    row.appendChild(upUpload);
    row.appendChild(upDevice);

    if (openSourceClass === name) {{
      // #region debug-point B:render-open-panel
      dbgEvent('B', 'app.py:render', '[DEBUG] rendering expanded source panel', {{name, openSourceKind, sampleCount: c}});
      // #endregion
      card.appendChild(row);
      const selectedPort = currentSerialPort || '';
      const selectedCam = Number(currentWebcamIndex);
      const sampleCount = c;
      const panel = document.createElement('div');
      panel.className = 'source-panel';
      panel.innerHTML = `
        <div class="source-left">
          <div class="source-head">
            <span>${{sourceLabel(openSourceKind)}}</span>
            <div class="source-tools">
              <button class="iconbtn" type="button" title="Close" id="sourceClose-${{cssSafe(name)}}">✕</button>
            </div>
          </div>
          ${{
            openSourceKind === 'upload'
              ? `<button class="upload-pick" type="button" id="uploadPick-${{cssSafe(name)}}">Choose images from your files</button>
                 <div class="upload-hint">Images are added to this class and appear immediately on the right.</div>`
              : (
                openSourceKind === 'device'
                  ? `<select class="device-select" id="deviceSelect-${{cssSafe(name)}}">
                       ${{buildDeviceOptions(selectedPort)}}
                     </select>
                     <div class="device-help">${{deviceHelpText()}}</div>
                     <div class="source-preview-wrap">
                       <img id="sourcePreview-${{cssSafe(name)}}" class="preview-frame" alt="Preview"/>
                     </div>
                     <div class="source-note" id="sourceNote-${{cssSafe(name)}}"></div>
                     ${{buildSourceSettingsMarkup(name)}}
                     <div class="source-actions">
                       <button class="btn btn-primary" type="button" id="sourceCapture-${{cssSafe(name)}}">Capture</button>
                       <button class="btn" type="button" id="sourceHold-${{cssSafe(name)}}">${{holdRecording && holdRecordClass === name ? 'Recording...' : 'Hold to Capture'}}</button>
                       <button class="iconbtn source-settings" type="button" id="sourceSettingsToggle-${{cssSafe(name)}}" title="Source settings">⚙</button>
                     </div>`
                  : `<div class="device-help">${{webcamHelpText()}}</div>
                     <div class="source-preview-wrap">
                       <img id="sourcePreview-${{cssSafe(name)}}" class="preview-frame" alt="Preview"/>
                     </div>
                     <div class="source-note" id="sourceNote-${{cssSafe(name)}}"></div>
                     ${{buildSourceSettingsMarkup(name)}}
                     <div class="source-actions">
                       <button class="btn btn-primary" type="button" id="sourceCapture-${{cssSafe(name)}}">Capture</button>
                       <button class="btn" type="button" id="sourceHold-${{cssSafe(name)}}">${{holdRecording && holdRecordClass === name ? 'Recording...' : 'Hold to Capture'}}</button>
                       <button class="iconbtn source-settings" type="button" id="sourceSettingsToggle-${{cssSafe(name)}}" title="Source settings">⚙</button>
                     </div>`
              )
          }}
        </div>
        <div class="source-right">
          <h4>${{sourceSamplesTitle(openSourceKind)}}</h4>
          <div class="source-count" id="sourceCount-${{cssSafe(name)}}">${{sampleCount}}<small>${{sampleCount === 1 ? 'Image Sample' : 'Image Samples'}}</small></div>
          <div id="samplesHost-${{cssSafe(name)}}">${{buildSamplesMarkup(name)}}</div>
        </div>
      `;
      card.appendChild(panel);
    }} else {{
      const summary = document.createElement('div');
      summary.className = 'summary-row';
      if (previewInputOn && (previewSource === 'webcam' || previewSource === 'device')) {{
        const quick = document.createElement('div');
        quick.className = 'summary-actions';
        quick.innerHTML = `<button class="btn btn-primary" type="button" id="previewCapture-${{cssSafe(name)}}">Capture ${{sourceLabel(previewSource)}}</button>`;
        summary.appendChild(quick);
      }}
      const samples = document.createElement('div');
      samples.className = 'summary-samples';
      samples.innerHTML = buildSamplesStripMarkup(name);
      summary.appendChild(row);
      summary.appendChild(samples);
      card.appendChild(summary);
    }}
    root.appendChild(card);
  }}
  const add = document.createElement('div');
  add.className = 'addclass';
  add.innerHTML = '<span style="font-size:18px;">⊞</span><span>Add a class</span>';
  add.onclick = () => addClass();
  root.appendChild(add);

  const trainBtn = document.getElementById('trainBtn');
  recomputeTrainEnabled();
  syncTrainUi();
  trainBtn.onclick = () => {{
    if (!STATE.train_enabled) return;
    startTrain();
  }};

  const exportBtn = document.getElementById('exportBtn');
  exportBtn.disabled = !STATE.export_enabled;
  exportBtn.onclick = async () => {{
    if (!STATE.export_enabled) return;
    exportBtn.disabled = true;
    try {{
      const pickRes = await fetch(`${{baseUrl}}/export/pick_dir?session=${{encodeURIComponent(STATE.session)}}`);
      const pick = await pickRes.json().catch(() => ({{ok:'0'}}));
      if (!pickRes.ok || pick.ok !== '1') {{
        if (pick && pick.canceled === '1') return;
        throw new Error(pick.error || 'Unable to choose folder.');
      }}
      const exportDir = String(pick.export_dir || '').trim();
      if (!exportDir) return;
      const runData = await exportRunWithOverwriteConfirm(
        exportDir,
        String(exportModelName || 'model'),
        String(exportArrayName || 'g_model')
      );
      if (!runData || runData.canceled === '1') return;
      toast(`Exported to: ${{String(runData.export_dir || exportDir)}}`);
    }} catch (e) {{
      toast(String(e && e.message ? e.message : e));
    }} finally {{
      exportBtn.disabled = !STATE.export_enabled;
    }}
  }};
  const exportBlockBtn = document.getElementById('exportBlockBtn');
  if (exportBlockBtn) {{
    exportBlockBtn.disabled = !STATE.export_enabled;
    exportBlockBtn.onclick = async () => {{
      if (!STATE.export_enabled) return;
      exportBlockBtn.disabled = true;
      try {{
        const res = await fetch(`${{baseUrl}}/blockcoding-export`, {{
          method: 'POST',
          headers: {{'Content-Type':'application/json'}},
          body: JSON.stringify({{session: STATE.session}})
        }});
        const data = await res.json().catch(() => ({{ok:'0'}}));
        if (!res.ok || data.ok !== '1') {{
          throw new Error(data && data.error ? data.error : 'Export to Block Coding failed.');
        }}
        const labelCount = Array.isArray(data.labels) ? data.labels.length : 0;
        toast(`Model sent to Block Coding (${{labelCount}} classes)`);
      }} catch (e) {{
        toast(String(e && e.message ? e.message : e));
      }} finally {{
        exportBlockBtn.disabled = !STATE.export_enabled;
      }}
    }};
  }}
  renderPreviewCard();
  renderTrainStatus();
  bindPreviewControls();
  renderClassPreprocessModal();

  document.getElementById('advBtn').onclick = toggleAdvancedInline;
  const advReset = document.getElementById('advReset');
  if (advReset) advReset.onclick = () => resetTrainCfg();
  bindAdvancedInlineHandlers();
  renderAdvancedPanel();
  const nav = document.getElementById('goHome');
  const navOpen = document.getElementById('navOpenProject');
  const navSave = document.getElementById('navSaveProject');
  const navExportDataset = document.getElementById('navExportDataset');
  const navReturn = document.getElementById('navReturn');
  const navReset = document.getElementById('navResetProject');
  if (nav) nav.onclick = (e) => {{
    if (e) e.stopPropagation();
    toggleNavMenu();
  }};
  if (navOpen) navOpen.onclick = async (e) => {{
    if (e) e.stopPropagation();
    closeNavMenu();
    await openProject();
  }};
  if (navSave) navSave.onclick = async (e) => {{
    if (e) e.stopPropagation();
    closeNavMenu();
    await saveProject();
  }};
  if (navExportDataset) navExportDataset.onclick = async (e) => {{
    if (e) e.stopPropagation();
    closeNavMenu();
    await exportDataset();
  }};
  if (navReturn) navReturn.onclick = async (e) => {{
    if (e) e.stopPropagation();
    closeNavMenu();
    await returnHome();
  }};
  if (navReset) navReset.onclick = async (e) => {{
    if (e) e.stopPropagation();
    closeNavMenu();
    await resetProject();
  }};
  if (!navMenuBound) {{
    document.addEventListener('click', () => closeNavMenu());
    navMenuBound = true;
  }}
  if (openSourceClass) {{
    const safe = cssSafe(openSourceClass);
    const closeBtn = document.getElementById(`sourceClose-${{safe}}`);
    const capBtn = document.getElementById(`sourceCapture-${{safe}}`);
    const holdBtn = document.getElementById(`sourceHold-${{safe}}`);
    const devSel = document.getElementById(`deviceSelect-${{safe}}`);
    const uploadPick = document.getElementById(`uploadPick-${{safe}}`);
    const settingsToggle = document.getElementById(`sourceSettingsToggle-${{safe}}`);
    const settingsSave = document.getElementById(`sourceSettingsSave-${{safe}}`);
    const settingsCancel = document.getElementById(`sourceSettingsCancel-${{safe}}`);
    if (closeBtn) closeBtn.onclick = () => closeSourcePanel();
    if (capBtn) capBtn.onclick = captureSource;
    if (holdBtn) {{
      holdBtn.onpointerdown = (e) => {{
        try {{ holdBtn.setPointerCapture(e.pointerId); }} catch (err) {{}}
        e.preventDefault();
        startHoldCapture();
      }};
      holdBtn.onpointerup = (e) => {{
        e.preventDefault();
        stopHoldCapture();
      }};
      holdBtn.onpointercancel = () => stopHoldCapture();
      holdBtn.onpointerleave = () => stopHoldCapture();
    }}
    if (devSel) devSel.onchange = (e) => changeDevicePort(openSourceClass, e.target.value || '');
    if (devSel) {{
      const selectId = `deviceSelect-${{safe}}`;
      devSel.onpointerdown = () => refreshSerialPorts(false, selectId);
      devSel.onmousedown = () => refreshSerialPorts(false, selectId);
    }}
    if (settingsToggle) settingsToggle.onclick = () => toggleSourceSettings(openSourceClass);
    if (settingsSave) settingsSave.onclick = () => applySourceSettings(openSourceClass);
    if (settingsCancel) settingsCancel.onclick = () => {{
      sourceSettingsOpen = false;
      render();
    }};
    if (uploadPick) uploadPick.onclick = () => {{
      const input = document.createElement('input');
      input.type = 'file';
      input.accept = 'image/*';
      input.multiple = true;
      input.onchange = () => uploadFiles(openSourceClass, input.files);
      input.click();
    }};
    syncSourceActionButtons(openSourceClass);
  }}
  root.querySelectorAll('.class-card').forEach((card, idx) => {{
    const className = STATE.classes[idx];
    bindSampleDeleteButtons(card, className);
    const quickCapBtn = document.getElementById(`previewCapture-${{cssSafe(className)}}`);
    if (quickCapBtn) quickCapBtn.onclick = () => capturePreviewInputToClass(className);
  }});
  bindLayoutImageObservers(root);

  ensureOpenSourceLive();
  updateFlow();
  scheduleLayoutResync();
  if (!resizeBound) {{
    window.addEventListener('resize', () => {{
      // #region debug-point C:window-resize
      dbgEvent('C', 'app.py:window.resize', '[DEBUG] window resize event', {{
        session: String(STATE.session || ''),
        innerWidth: window.innerWidth || 0,
        innerHeight: window.innerHeight || 0,
        docClientHeight: document.documentElement ? document.documentElement.clientHeight : 0,
        docScrollHeight: document.documentElement ? document.documentElement.scrollHeight : 0,
      }});
      // #endregion
      scheduleLayoutResync();
    }});
    resizeBound = true;
  }}
  window.onpointerup = () => stopHoldCapture();
  toast(STATE.notice);
}}
window.addEventListener('load', () => {{
  // #region debug-point A:window-load
  dbgEvent('A', 'app.py:window.load', '[DEBUG] window load event', {{
    session: String(STATE.session || ''),
    innerWidth: window.innerWidth || 0,
    innerHeight: window.innerHeight || 0,
    href: parentUrl().toString(),
  }});
  // #endregion
  queueFrameHeightSync();
  requestShellLayoutRefresh('image-project-mount:load');
}});
window.addEventListener('pagehide', () => {{
  cleanupWorkspaceFrameBeforeNavigate('pagehide');
}});
window.addEventListener('beforeunload', () => {{
  cleanupWorkspaceFrameBeforeNavigate('beforeunload');
}});
window.addEventListener('orientationchange', () => {{
  // #region debug-point C:orientation-change
  dbgEvent('C', 'app.py:orientationchange', '[DEBUG] orientationchange event', {{
    session: String(STATE.session || ''),
    innerWidth: window.innerWidth || 0,
    innerHeight: window.innerHeight || 0,
  }});
  // #endregion
  queueFrameHeightSync();
  logScrollLayers('window-load');
}});
if (window.__tmStageMark) window.__tmStageMark('before-render');
render();
if (window.__tmStageMark) window.__tmStageMark('after-render');
logScrollLayers('after-render');
requestShellLayoutRefresh('image-project-mount');
mountReflowTimers.push(window.setTimeout(() => {{
  if (window.__tmNavigatingAway) return;
  requestShellLayoutRefresh('image-project-mount:t120');
  logScrollLayers('mount-t120');
}}, 120));
mountReflowTimers.push(window.setTimeout(() => {{
  if (window.__tmNavigatingAway) return;
  requestShellLayoutRefresh('image-project-mount:t360');
  logScrollLayers('mount-t360');
}}, 360));
</script>
</body>
</html>
        ''',
        height=1600,
        scrolling=False,
    )


def _render_image_project() -> None:
    # #region debug-point A:render-image-project
    _dbg_open_project_layout("A", "pre-fix", "app.py:_render_image_project", "[DEBUG] render image project page", {"session": str(st.session_state.get("session_id", "")), "project_type": str(st.session_state.get("project_type", "")), "query": dict(st.query_params) if hasattr(st, "query_params") else {}, "workspace": str(_session_workspace())})
    # #endregion
    inject_teachable_style()
    controller = _get_record_controller()
    webcam_options = _list_camera_options()
    preferred_webcam_index = _preferred_webcam_index(webcam_options)
    current_webcam_label = next((str(item.get("label", "")) for item in webcam_options if int(item.get("index", 0)) == int(st.session_state.tm_webcam_index)), "")
    if (not st.session_state.get("tm_webcam_user_selected", False)) and webcam_options:
        st.session_state.tm_webcam_index = int(preferred_webcam_index)
    elif webcam_options and _is_virtual_camera_label(current_webcam_label):
        st.session_state.tm_webcam_index = int(preferred_webcam_index)
    serial_ports = [
        {
            "device": p.device,
            "label": f"{p.device} - {p.description}" if p.description else p.device,
        }
        for p in list_serial_ports()
        if _is_likely_user_serial_port(p.device, p.description)
    ]
    controller.set_config(
        st.session_state.session_id,
        SessionConfig(
            dataset_root=_tm_dataset_dir(),
            serial_port=st.session_state.tm_serial_port,
            serial_baud=int(st.session_state.tm_serial_baud),
            serial_sync=str(st.session_state.tm_serial_sync),
            serial_frame_side=int(st.session_state.tm_serial_frame_side),
            webcam_index=int(st.session_state.tm_webcam_index),
            fps=float(st.session_state.tm_record_fps),
            crop_box=st.session_state.tm_record_crop_box,
        ),
    )

    classes = list(st.session_state.tm_classes) if st.session_state.tm_classes else ["Class 1", "Class 2"]
    if len(classes) == 1:
        classes = classes + ["Class 2"]
    disk_classes = _tm_load_classes_meta()
    if disk_classes:
        classes = disk_classes
    else:
        _tm_save_classes_meta(classes)
    st.session_state.tm_classes = classes

    action = _tm_get_query_param("tm_action").strip()
    notice = ""
    #region debug-point B:action-entry
    _dbg_capture_webcam_source("B", "pre-fix", "app.py:_render_image_project", "[DEBUG] tm_action received", {"action": action, "query": dict(st.query_params) if hasattr(st, "query_params") else {}, "tm_open_source_class": st.session_state.get("tm_open_source_class", ""), "tm_open_source_kind": st.session_state.get("tm_open_source_kind", ""), "tm_webcam_index": st.session_state.get("tm_webcam_index", 0)})
    #endregion
    if action:
        if action == "home":
            # #region debug-point B:action-home
            _dbg_open_project_layout("B", "pre-fix", "app.py:_render_image_project", "[DEBUG] action home triggered", {"session": str(st.session_state.get("session_id", "")), "query_before_clear": dict(st.query_params) if hasattr(st, "query_params") else {}, "project_type_before": str(st.session_state.get("project_type", ""))})
            # #endregion
            st.session_state.tm_open_source_class = ""
            st.session_state.tm_open_source_kind = ""
            st.session_state.tm_frontend_notice = ""
            st.session_state.project_type = None
            _tm_clear_query_params()
            st.rerun()
        if action == "addclass":
            st.session_state.tm_classes = classes + [_next_class_name(classes)]
            _tm_clear_query_params()
            st.rerun()
        if action == "delete":
            try:
                idx = int(_tm_get_query_param("idx") or "-1")
            except Exception:
                idx = -1
            if 0 <= idx < len(classes) and len(classes) > 1:
                _remove_tm_class(classes[idx])
                st.session_state.tm_classes = [c for i, c in enumerate(classes) if i != idx]
            _tm_clear_query_params()
            st.rerun()
        if action == "rename":
            try:
                idx = int(_tm_get_query_param("idx") or "-1")
            except Exception:
                idx = -1
            new_name = _tm_get_query_param("name").strip()
            if 0 <= idx < len(classes) and new_name:
                edited = list(classes)
                edited[idx] = new_name
                updated = _apply_tm_class_names(classes, edited)
                st.session_state.tm_classes = updated
            _tm_clear_query_params()
            st.rerun()
        if action == "set_serial_port":
            st.session_state.tm_serial_port = _tm_get_query_param("serial_port").strip()
            st.session_state.tm_open_source_class = _tm_get_query_param("open_class").strip()
            st.session_state.tm_open_source_kind = _tm_get_query_param("open_kind").strip()
            _tm_clear_query_params()
            st.rerun()
        if action == "set_webcam_index":
            try:
                st.session_state.tm_webcam_index = int(float(_tm_get_query_param("webcam_index") or st.session_state.tm_webcam_index))
                st.session_state.tm_webcam_user_selected = True
            except Exception:
                pass
            st.session_state.tm_open_source_class = _tm_get_query_param("open_class").strip()
            st.session_state.tm_open_source_kind = _tm_get_query_param("open_kind").strip()
            #region debug-point D:set-webcam-index
            _dbg_capture_webcam_source("D", "pre-fix", "app.py:_render_image_project", "[DEBUG] webcam index updated from action", {"tm_webcam_index": st.session_state.tm_webcam_index, "tm_open_source_class": st.session_state.tm_open_source_class, "tm_open_source_kind": st.session_state.tm_open_source_kind})
            #endregion
            _tm_clear_query_params()
            st.rerun()
        if action == "open_source":
            st.session_state.tm_open_source_class = _tm_get_query_param("open_class").strip()
            st.session_state.tm_open_source_kind = _tm_get_query_param("open_kind").strip()
            st.session_state.tm_frontend_notice = _tm_get_query_param("notice").strip()
            #region debug-point B:open-source-action
            _dbg_capture_webcam_source("B", "pre-fix", "app.py:_render_image_project", "[DEBUG] open_source action applied", {"tm_open_source_class": st.session_state.tm_open_source_class, "tm_open_source_kind": st.session_state.tm_open_source_kind, "tm_frontend_notice": st.session_state.tm_frontend_notice})
            #endregion
            _tm_clear_query_params()
            st.rerun()
        if action == "close_source":
            st.session_state.tm_open_source_class = ""
            st.session_state.tm_open_source_kind = ""
            _tm_clear_query_params()
            st.rerun()
        if action == "advanced":
            cfg = st.session_state.train_cfg
            try:
                batch_size = max(1, int(float(_tm_get_query_param("batch_size") or cfg.batch_size)))
                epochs = max(1, int(float(_tm_get_query_param("epochs") or cfg.epochs)))
                validation_split = float(_tm_get_query_param("validation_split") or cfg.validation_split)
                learning_rate = float(_tm_get_query_param("learning_rate") or cfg.learning_rate)
                conv1_filters = max(1, int(float(_tm_get_query_param("conv1_filters") or cfg.conv1_filters)))
                conv2_filters = max(1, int(float(_tm_get_query_param("conv2_filters") or cfg.conv2_filters)))
                dense_units = max(1, int(float(_tm_get_query_param("dense_units") or cfg.dense_units)))
            except Exception:
                notice = "Invalid advanced settings."
            else:
                st.session_state.train_cfg = TrainConfig(
                    img_size=cfg.img_size,
                    color_mode=cfg.color_mode,
                    batch_size=batch_size,
                    epochs=epochs,
                    validation_split=max(0.05, min(0.5, validation_split)),
                    seed=cfg.seed,
                    optimizer=cfg.optimizer,
                    learning_rate=max(0.00001, learning_rate),
                    conv1_filters=conv1_filters,
                    conv2_filters=conv2_filters,
                    dense_units=dense_units,
                    representative_samples=cfg.representative_samples,
                    preprocess_mode=str(getattr(cfg, "preprocess_mode", "auto_by_label")),
                    manual_roi=getattr(cfg, "manual_roi", None),
                )
                notice = "Advanced settings updated."
            _tm_clear_query_params()
            st.rerun()

    counts: Dict[str, int] = {}
    empty_classes: List[str] = []
    total_samples = 0
    for name in classes:
        class_dir = _tm_dataset_dir() / sanitize_class_name(name)
        n = len(_tm_class_image_files(class_dir)) if class_dir.exists() else 0
        counts[name] = n
        total_samples += n
        if n == 0:
            empty_classes.append(name)
    train_ready = len([c for c in classes if counts.get(c, 0) > 0]) >= 2 and not empty_classes and total_samples > 0
    st.session_state.tm_train_ready = train_ready
    if not train_ready:
        if len(classes) < 2:
            st.session_state.tm_train_block_reason = "Training requires at least 2 classes."
        elif empty_classes:
            st.session_state.tm_train_block_reason = "Each class needs at least 1 sample image."
        else:
            st.session_state.tm_train_block_reason = "Add sample images before training."
    else:
        st.session_state.tm_train_block_reason = ""

    if action == "train":
        st.session_state.tm_open_source_class = _tm_get_query_param("open_class").strip()
        st.session_state.tm_open_source_kind = _tm_get_query_param("open_kind").strip()
        _tm_clear_query_params()
        notice = "Training starts in-page."

    if action == "export":
        _tm_clear_query_params()
        meta = _tm_load_train_latest()
        tflite_path = Path(str(meta.get("tflite_path") or "")).expanduser().resolve() if isinstance(meta, dict) else Path()
        labels = list(meta.get("labels") or []) if isinstance(meta, dict) else []
        if (not meta) or (not tflite_path.exists()):
            notice = "Train a model first."
            st.rerun()
        else:
            model_name = str(st.session_state.get("tm_model_name", "model")).strip() or "model"
            array_name = str(st.session_state.get("tm_array_name", "g_model")).strip() or "g_model"
            export_dir = Path(st.session_state.last_export_dir).expanduser().resolve() if st.session_state.last_export_dir.strip() else _default_export_dir()
            errors = _validate_export_inputs(export_dir, model_name, array_name, tflite_path)
            if errors:
                notice = errors[0]
            else:
                export_dir.mkdir(parents=True, exist_ok=True)
                source_bytes = tflite_path.read_bytes()
                from trainer import export_tflite_c_sources

                src, hdr = export_tflite_c_sources(source_bytes, array_name=array_name)
                (export_dir / f"{model_name}.tflite").write_bytes(source_bytes)
                (export_dir / "model.h").write_text(hdr, encoding="utf-8")
                (export_dir / "model.cpp").write_text('#include "model.h"\n\n' + src, encoding="utf-8")
                (export_dir / "labels.txt").write_text("\n".join([str(x) for x in labels]) + "\n", encoding="utf-8")
                notice = f"Exported to: {export_dir}"
            st.rerun()

    if action == "export_browse":
        _tm_clear_query_params()
        meta = _tm_load_train_latest()
        tflite_path = Path(str(meta.get("tflite_path") or "")).expanduser().resolve() if isinstance(meta, dict) else Path()
        labels = list(meta.get("labels") or []) if isinstance(meta, dict) else []
        if (not meta) or (not tflite_path.exists()):
            notice = "Train a model first."
            st.rerun()
        picked = _pick_directory_dialog(initial_dir=str(st.session_state.last_export_dir or ""))
        if picked:
            st.session_state.last_export_dir = str(picked)
        model_name = str(st.session_state.get("tm_model_name", "model")).strip() or "model"
        array_name = str(st.session_state.get("tm_array_name", "g_model")).strip() or "g_model"
        export_dir = Path(st.session_state.last_export_dir).expanduser().resolve() if st.session_state.last_export_dir.strip() else _default_export_dir()
        errors = _validate_export_inputs(export_dir, model_name, array_name, tflite_path)
        if errors:
            notice = errors[0]
        else:
            export_dir.mkdir(parents=True, exist_ok=True)
            source_bytes = tflite_path.read_bytes()
            from trainer import export_tflite_c_sources

            src, hdr = export_tflite_c_sources(source_bytes, array_name=array_name)
            (export_dir / f"{model_name}.tflite").write_bytes(source_bytes)
            (export_dir / "model.h").write_text(hdr, encoding="utf-8")
            (export_dir / "model.cpp").write_text('#include "model.h"\n\n' + src, encoding="utf-8")
            (export_dir / "labels.txt").write_text("\n".join([str(x) for x in labels]) + "\n", encoding="utf-8")
            notice = f"Exported to: {export_dir}"
        st.rerun()

    export_enabled = _tm_load_train_latest() is not None
    if not notice:
        notice = str(st.session_state.get("tm_frontend_notice", "") or "")
    st.session_state.tm_frontend_notice = ""
    sample_previews = _tm_sample_previews(classes, limit_per_class=12)
    dataset_root = _tm_dataset_dir()
    class_preprocess = controller._class_preprocess_load(dataset_root)
    sample_preprocess_data = controller._sample_preprocess_load(dataset_root)
    processed_previews = controller._processed_previews_payload(dataset_root, classes)
    class_labels = controller._class_labels_load(dataset_root)
    initial_open_source_class = str(st.session_state.get("tm_open_source_class", "") or "")
    initial_open_source_kind = str(st.session_state.get("tm_open_source_kind", "") or "")
    _render_tm_old_frontend_html(
        port=int(controller.port),
        session_id=str(st.session_state.session_id),
        classes=classes,
        counts=counts,
        train_enabled=train_ready,
        export_enabled=export_enabled,
        notice=notice,
        train_cfg=st.session_state.train_cfg,
        serial_ports=serial_ports,
        current_serial_port=str(st.session_state.tm_serial_port),
        current_serial_baud=int(st.session_state.tm_serial_baud),
        current_serial_sync=str(st.session_state.tm_serial_sync),
        current_serial_frame_side=int(st.session_state.tm_serial_frame_side),
        webcam_options=webcam_options,
        current_webcam_index=int(st.session_state.tm_webcam_index),
        sample_previews=sample_previews,
        initial_open_source_class=initial_open_source_class,
        initial_open_source_kind=initial_open_source_kind,
        class_preprocess=class_preprocess,
        sample_preprocess=sample_preprocess_data,
        processed_previews=processed_previews,
        class_labels=class_labels,
    )


def _render_classified_import_page() -> None:
    # #region debug-point B:render-classified-import
    _dbg_open_project_layout("B", "pre-fix", "app.py:_render_classified_import_page", "[DEBUG] render classified import page", {"session": str(st.session_state.get("session_id", "")), "project_type": str(st.session_state.get("project_type", "")), "query": dict(st.query_params) if hasattr(st, "query_params") else {}, "local_import_path": str(st.session_state.get("local_import_path", ""))})
    # #endregion
    inject_teachable_style()
    st.markdown(
        '''
<div class="tm-hero">
  <div class="tm-hero-copy">
    <div class="tm-eyebrow">Start from classified class</div>
    <h2>Import a classified image folder</h2>
    <p>Supported layouts include <code>label/image</code> and common <code>train|val|test/label/image</code> datasets. Imported images become class samples and each label becomes a class name.</p>
  </div>
</div>
        ''',
        unsafe_allow_html=True,
    )
    top_left, top_right = st.columns([1, 5])
    with top_left:
        if st.button("← Back", key="tm_back_from_classified"):
            _reset_session_workspace()
            _tm_clear_query_params()
            st.rerun()

    input_left, input_right = st.columns([5, 1], vertical_alignment="bottom")
    prior_path = str(st.session_state.local_import_path or "")
    with input_left:
        st.markdown('<div class="tm-classified-browse-marker"></div>', unsafe_allow_html=True)
        path_str = st.text_input(
            "Classified folder",
            value=st.session_state.local_import_path,
            placeholder="Choose a folder like label/image or train/label/image",
        )
        st.session_state.local_import_path = path_str
    path_changed = str(path_str or "").strip() != str(prior_path or "").strip()
    if path_changed:
        st.session_state.imported = None
        st.session_state.class_rename = {}
    with input_right:
        if st.button("Browse...", key="tm_pick_classified_dir"):
            picked = _pick_directory_dialog(initial_dir=st.session_state.local_import_path)
            if picked:
                st.session_state.local_import_path = picked
                st.session_state.imported = None
                st.session_state.class_rename = {}
                st.rerun()

    path_value = str(st.session_state.local_import_path or "").strip()
    if st.button("Load folder", type="primary", key="tm_read_classified_dir", disabled=(not path_value)):
        p = Path(path_value).expanduser()
        if not p.exists() or not p.is_dir():
            st.error("Path does not exist or is not a folder.")
            st.session_state.imported = None
        else:
            imported = infer_imported_data(p)
            if not imported.class_to_images:
                st.error("No dataset detected. Put images under label folders, or use train|val|test/label/image.")
                st.session_state.imported = None
            elif not imported.classified:
                st.error("This folder is not a supported classified dataset. Use label/image or train|val|test/label/image.")
                st.session_state.imported = None
            else:
                st.session_state.imported = imported
                st.session_state.class_rename = {name: name for name in imported.class_to_images.keys()}
                st.rerun()

    imported = st.session_state.imported
    if imported is None or not imported.classified:
        return

    _render_overview(imported)
    st.markdown("### Rename classes")
    rename: Dict[str, str] = {}
    for class_name, images in imported.class_to_images.items():
        c1, c2 = st.columns([4, 1])
        with c1:
            rename[class_name] = st.text_input(
                f"{class_name} name",
                value=st.session_state.class_rename.get(class_name, class_name),
                key=f"tm_classified_rename_{class_name}",
                label_visibility="collapsed",
            )
        with c2:
            st.markdown(
                f'<div class="{_status_style("ready")}">{len(images)} samples</div>',
                unsafe_allow_html=True,
            )
    st.session_state.class_rename = rename

    if st.button("Import to workspace", type="primary", key="tm_import_classified_workspace"):
        try:
            class_names = _import_classified_into_workspace(imported, rename)
        except Exception as e:
            st.error(str(e))
            return
        # #region debug-point B:classified-import-to-workspace
        _dbg_open_project_layout("B", "pre-fix", "app.py:_render_classified_import_page", "[DEBUG] classified import to workspace", {"session": str(st.session_state.get("session_id", "")), "class_names": list(class_names or []), "query_before": dict(st.query_params) if hasattr(st, "query_params") else {}})
        # #endregion
        st.session_state.tm_classes = class_names or ["Class 1"]
        st.session_state.dataset_dir = str(_tm_dataset_dir())
        st.session_state.imported = None
        st.session_state.project_type = "image"
        _tm_set_query_params(tm_project="image", tm_session=st.session_state.session_id)
        st.rerun()


def _tm_dataset_dir() -> Path:
    ws = _session_workspace()
    return (ws / "tm_dataset").resolve()


@st.cache_resource
def _get_record_controller() -> RecordController:
    c = RecordController()
    c.start()
    return c


def _save_sample_png(class_name: str, png_bytes: bytes) -> Path:
    out_dir = _tm_dataset_dir() / sanitize_class_name(class_name)
    out_dir.mkdir(parents=True, exist_ok=True)
    name = uuid.uuid4().hex + ".png"
    out_path = out_dir / name
    out_path.write_bytes(png_bytes)
    return out_path


def _img_to_png_bytes(img) -> bytes:
    import io
    from PIL import Image

    if isinstance(img, Image.Image):
        im = img
    else:
        im = Image.open(io.BytesIO(img))
    bio = io.BytesIO()
    im.save(bio, format="PNG")
    return bio.getvalue()


def _preprocess_image_to_96x96_gray(png_bytes: bytes, crop_box: Optional[Tuple[int, int, int, int]] = None) -> bytes:
    import io
    from PIL import Image

    im = Image.open(io.BytesIO(png_bytes)).convert("L")
    if crop_box is not None:
        x1, y1, x2, y2 = crop_box
        im = im.crop((x1, y1, x2, y2))
    im = im.resize((96, 96))
    out = io.BytesIO()
    im.save(out, format="PNG")
    return out.getvalue()


def _render_crop_ui() -> None:
    if st.session_state.tm_pending_image is None:
        return
    try:
        from streamlit_drawable_canvas import st_canvas
    except Exception:
        st.error("Missing dependency streamlit-drawable-canvas. ROI crop mode is unavailable.")
        return

    import io
    from PIL import Image

    st.subheader("Crop region (ROI)")
    png_bytes = st.session_state.tm_pending_image
    class_name = st.session_state.tm_pending_class
    im = Image.open(io.BytesIO(png_bytes)).convert("RGB")
    w, h = im.size
    scale = min(520 / max(w, 1), 520 / max(h, 1), 1.0)
    disp_w, disp_h = int(w * scale), int(h * scale)
    im_disp = im.resize((disp_w, disp_h))

    canvas = st_canvas(
        fill_color="rgba(0, 0, 0, 0)",
        stroke_width=3,
        stroke_color="rgba(0, 122, 255, 1)",
        background_image=im_disp,
        update_streamlit=True,
        height=disp_h,
        width=disp_w,
        drawing_mode="rect",
        key="crop_canvas",
    )

    rect = None
    if canvas.json_data and canvas.json_data.get("objects"):
        obj = canvas.json_data["objects"][-1]
        if obj and obj.get("type") == "rect":
            rect = obj

    col_a, col_b, col_c = st.columns([1, 1, 3])
    with col_a:
        if st.button("Cancel", key="crop_cancel"):
            st.session_state.tm_pending_image = None
            st.session_state.tm_pending_class = None
            st.rerun()
    with col_b:
        if st.button("Save", type="primary", key="crop_confirm"):
            if rect is None:
                st.error("Draw a rectangle first.")
                return
            left = int(rect.get("left", 0))
            top = int(rect.get("top", 0))
            rw = int(rect.get("width", 0))
            rh = int(rect.get("height", 0))
            if rw <= 1 or rh <= 1:
                st.error("ROI is too small.")
                return
            x1 = int(left / scale)
            y1 = int(top / scale)
            x2 = int((left + rw) / scale)
            y2 = int((top + rh) / scale)
            x1 = max(0, min(w - 1, x1))
            y1 = max(0, min(h - 1, y1))
            x2 = max(x1 + 1, min(w, x2))
            y2 = max(y1 + 1, min(h, y2))
            out_png = _preprocess_image_to_96x96_gray(png_bytes, crop_box=(x1, y1, x2, y2))
            _save_sample_png(class_name, out_png)
            st.session_state.tm_pending_image = None
            st.session_state.tm_pending_class = None
            st.rerun()

    st.caption(f"Input: {w}x{h}  →  Output: 96x96x1")


def _render_tm_class_panel() -> None:
    _render_crop_ui()
    classes = list(st.session_state.tm_classes) or ["Class 1"]
    edited_names: List[str] = []
    total_samples = 0
    empty_classes: List[str] = []
    if "tm_edit_class_idx" not in st.session_state:
        st.session_state.tm_edit_class_idx = -1

    controller = _get_record_controller()
    controller.set_config(
        st.session_state.session_id,
        SessionConfig(
            dataset_root=_tm_dataset_dir(),
            serial_port=st.session_state.tm_serial_port,
            serial_baud=int(st.session_state.tm_serial_baud),
            serial_sync=str(st.session_state.tm_serial_sync),
            webcam_index=int(st.session_state.tm_webcam_index),
            fps=float(st.session_state.tm_record_fps),
            crop_box=st.session_state.tm_record_crop_box,
        ),
    )

    for idx, name in enumerate(classes):
        class_dir = _tm_dataset_dir() / sanitize_class_name(name)
        samples = _tm_class_image_files(class_dir)
        total_samples += len(samples)
        if len(samples) == 0:
            empty_classes.append(name)
        with st.container():
            st.markdown('<div class="tm-class-card-marker"></div>', unsafe_allow_html=True)
            editing = int(st.session_state.tm_edit_class_idx) == idx
            head_a, head_b, head_c = st.columns([5.6, 0.65, 0.75])
            with head_a:
                if editing:
                    st.markdown('<div class="tm-class-title-field">', unsafe_allow_html=True)
                    edited_name = st.text_input(
                        f"class-name-{idx}",
                        value=name,
                        key=f"tm_class_name_{idx}",
                        label_visibility="collapsed",
                        placeholder=f"Class {idx + 1}",
                    )
                    st.markdown("</div>", unsafe_allow_html=True)
                else:
                    st.markdown(
                        f'<div class="tm-class-title-text"><h3>{html_escape(name)}</h3></div>',
                        unsafe_allow_html=True,
                    )
                    edited_name = name
                edited_names.append(edited_name)
            with head_b:
                if editing:
                    if st.button("✓", key=f"tm_done_edit_{idx}", use_container_width=True):
                        st.session_state.tm_edit_class_idx = -1
                        st.rerun()
                else:
                    if st.button("✎", key=f"tm_edit_{idx}", use_container_width=True):
                        st.session_state.tm_edit_class_idx = idx
                        st.rerun()
            with head_c:
                delete_disabled = len(classes) <= 1
                st.markdown('<div class="tm-class-menu">', unsafe_allow_html=True)
                with st.popover("⋮", use_container_width=True):
                    if st.button("Delete class", key=f"tm_delete_class_{idx}", use_container_width=True, disabled=delete_disabled):
                        _remove_tm_class(name)
                        st.session_state.tm_classes = [item for item in classes if item != name]
                        st.rerun()
                st.markdown("</div>", unsafe_allow_html=True)
            st.markdown('<div class="tm-class-divider"></div>', unsafe_allow_html=True)
            st.markdown('<div class="tm-class-subhead">Add Image Samples:</div>', unsafe_allow_html=True)

            btn_a, btn_b, btn_c = st.columns(3, gap="small")
            with btn_a:
                with st.popover("▢\nWebcam", use_container_width=True):
                    permission = ensure_camera_access(int(st.session_state.tm_webcam_index))
                    st.session_state.tm_camera_permission_status = permission.status
                    st.session_state.tm_camera_permission_note = permission.message
                    st.session_state.tm_camera_permission_class = name
                    if not permission.allowed:
                        st.markdown(
                            f'<div class="tm-camera-note">{html_escape(permission.message)}</div>',
                            unsafe_allow_html=True,
                        )
                    else:
                        cam = st.camera_input("Webcam", key=f"tm_cam_{idx}", label_visibility="collapsed")
                        if cam is not None:
                            png = _img_to_png_bytes(cam.getvalue())
                            out_png = _preprocess_image_to_96x96_gray(png, crop_box=None)
                            _save_sample_png(name, out_png)
                            st.rerun()
            with btn_b:
                with st.popover("▦\nDevice", use_container_width=True):
                    up = st.file_uploader(
                        "Device image",
                        type=["png", "jpg", "jpeg", "bmp", "webp"],
                        accept_multiple_files=True,
                        key=f"tm_device_up_{idx}",
                        label_visibility="collapsed",
                    )
                    if up:
                        for f in up:
                            png = _img_to_png_bytes(f.getvalue())
                            out_png = _preprocess_image_to_96x96_gray(png, crop_box=None)
                            _save_sample_png(name, out_png)
                        st.rerun()
            with btn_c:
                st.markdown('<div class="tm-class-upload">', unsafe_allow_html=True)
                with st.popover("⇧\nUpload", use_container_width=True):
                    upload = st.file_uploader(
                        "Upload",
                        type=["png", "jpg", "jpeg", "bmp", "webp"],
                        accept_multiple_files=True,
                        key=f"tm_up_{idx}",
                        label_visibility="collapsed",
                    )
                    if upload:
                        for f in upload:
                            png = _img_to_png_bytes(f.getvalue())
                            if st.session_state.tm_crop_mode == "roi":
                                st.session_state.tm_pending_image = png
                                st.session_state.tm_pending_class = name
                                st.rerun()
                            out_png = _preprocess_image_to_96x96_gray(png, crop_box=None)
                            _save_sample_png(name, out_png)
                        st.rerun()
                st.markdown("</div>", unsafe_allow_html=True)

            if (
                st.session_state.tm_camera_permission_note
                and st.session_state.tm_camera_permission_class == name
                and st.session_state.tm_camera_permission_status
                not in {"granted", "not_required"}
            ):
                st.markdown(
                    f'<div class="tm-camera-note">{html_escape(st.session_state.tm_camera_permission_note)}</div>',
                    unsafe_allow_html=True,
                )

    if edited_names != classes:
        try:
            updated_names = _apply_tm_class_names(classes, edited_names)
        except Exception as e:
            st.error(str(e))
        else:
            if updated_names != classes:
                name_map = {old: new for old, new in zip(classes, updated_names)}
                st.session_state.tm_classes = updated_names
                for key in ("tm_capture_class", "tm_camera_permission_class", "tm_pending_class"):
                    current = st.session_state.get(key)
                    if current in name_map:
                        st.session_state[key] = name_map[current]
                st.rerun()

    with st.container():
        st.markdown('<div class="tm-add-class-marker"></div>', unsafe_allow_html=True)
        if st.button("⊞ Add a class", key="tm_add_class", use_container_width=True):
            st.session_state.tm_classes = classes + [_next_class_name(classes)]
            st.rerun()
    st.session_state.tm_crop_mode = "full"

    non_empty_classes = [c for c in classes if c not in empty_classes]
    train_ready = len(non_empty_classes) >= 2 and not empty_classes and total_samples > 0
    st.session_state.tm_total_classes = len(classes)
    st.session_state.tm_total_samples = total_samples
    st.session_state.tm_train_ready = train_ready
    if not train_ready:
        if len(classes) < 2:
            st.session_state.tm_train_block_reason = "Training requires at least 2 classes."
        elif empty_classes:
            st.session_state.tm_train_block_reason = "Each class needs at least 1 sample image."
        else:
            st.session_state.tm_train_block_reason = "Add sample images before training."
    else:
        st.session_state.tm_train_block_reason = ""


def _render_hold_capture_panel(controller: RecordController, class_name: str, samples: List[Path]) -> None:
    from urllib.parse import quote

    source = st.session_state.tm_capture_source
    st.markdown('<div class="tm-panel tm-capture-panel">', unsafe_allow_html=True)
    st.markdown(
        f'''
<div class="tm-capture-head">
  <strong>{html_escape(source.title())}</strong>
  <span class="{_status_style('ready')}">Live</span>
</div>
        ''',
        unsafe_allow_html=True,
    )

    preview: Optional[bytes] = None
    if source == "device":
        preview = controller.preview_serial_png(
            st.session_state.tm_serial_port,
            int(st.session_state.tm_serial_baud),
            str(st.session_state.tm_serial_sync),
        )
        if preview is None:
            st.warning("Unable to preview device stream. Check serial port and baudrate.")
    elif source == "webcam":
        permission = ensure_camera_access(int(st.session_state.tm_webcam_index))
        st.session_state.tm_camera_permission_status = permission.status
        st.session_state.tm_camera_permission_note = permission.message
        if not permission.allowed:
            st.error(permission.message)
            st.markdown("</div>", unsafe_allow_html=True)
            return
        preview = controller.preview_webcam_png(int(st.session_state.tm_webcam_index))
        if preview is None:
            st.warning("Unable to preview webcam. Check system permission or whether the camera is in use.")

    crop_box: Optional[Tuple[int, int, int, int]] = None
    if st.session_state.tm_crop_mode == "roi" and preview is not None:
        try:
            from streamlit_drawable_canvas import st_canvas
        except Exception:
            st.error("Missing dependency streamlit-drawable-canvas. ROI crop mode is unavailable.")
        else:
            import io
            from PIL import Image

            im = Image.open(io.BytesIO(preview)).convert("RGB")
            w, h = im.size
            scale = min(520 / max(w, 1), 520 / max(h, 1), 1.0)
            disp_w, disp_h = int(w * scale), int(h * scale)
            im_disp = im.resize((disp_w, disp_h))
            canvas = st_canvas(
                fill_color="rgba(0, 0, 0, 0)",
                stroke_width=3,
                stroke_color="rgba(0, 122, 255, 1)",
                background_image=im_disp,
                update_streamlit=True,
                height=disp_h,
                width=disp_w,
                drawing_mode="rect",
                key="tm_hold_roi_canvas",
            )
            if canvas.json_data and canvas.json_data.get("objects"):
                obj = canvas.json_data["objects"][-1]
                if obj and obj.get("type") == "rect":
                    left = int(obj.get("left", 0))
                    top = int(obj.get("top", 0))
                    rw = int(obj.get("width", 0))
                    rh = int(obj.get("height", 0))
                    if rw > 1 and rh > 1:
                        x1 = int(left / scale)
                        y1 = int(top / scale)
                        x2 = int((left + rw) / scale)
                        y2 = int((top + rh) / scale)
                        x1 = max(0, min(w - 1, x1))
                        y1 = max(0, min(h - 1, y1))
                        x2 = max(x1 + 1, min(w, x2))
                        y2 = max(y1 + 1, min(h, y2))
                        crop_box = (x1, y1, x2, y2)
                        st.session_state.tm_record_crop_box = crop_box

    capture_left, capture_right = st.columns([1.05, 1.05], gap="small")
    with capture_left:
        st.markdown('<div class="tm-capture-stage">', unsafe_allow_html=True)
        if preview is not None:
            st.image(preview, use_container_width=True)
        st.session_state.tm_record_fps = st.slider("FPS", min_value=1.0, max_value=20.0, value=float(st.session_state.tm_record_fps), step=1.0)

        base = f"http://127.0.0.1:{controller.port}"
        q_class = quote(class_name)
        q_sess = quote(st.session_state.session_id)
        q_source = quote(source)
        start_url = f"{base}/start?session={q_sess}&source={q_source}&class={q_class}"
        stop_url = f"{base}/stop?session={q_sess}"
        html = make_hold_button_html("Hold to Record", start_url=start_url, stop_url=stop_url)
        components.html(html, height=88)
        st.markdown("</div>", unsafe_allow_html=True)
    with capture_right:
        st.markdown('<div class="tm-capture-side-head">Samples</div>', unsafe_allow_html=True)
        if samples:
            st.image([str(p) for p in samples[-6:]], width=72)
        else:
            st.markdown('<div class="tm-preview-note">No samples yet.</div>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


def _render_tm_train_panel() -> None:
    dataset_dir = _tm_dataset_dir()
    train_ready = bool(st.session_state.get("tm_train_ready", False))
    train_reason = str(st.session_state.get("tm_train_block_reason", "")).strip()
    with st.container():
        st.markdown('<div class="tm-train-card-marker"></div>', unsafe_allow_html=True)
        st.markdown('<div class="tm-card-head"><h3>Training</h3></div>', unsafe_allow_html=True)
        if st.button("Train Embedded Model", type="primary", key="tm_train_primary", use_container_width=True, disabled=not train_ready):
            if not train_ready:
                return
            st.session_state.export_validated_token = ""
            runs_dir = APP_DATA_DIR / "runs"
            run_dir = new_run_dir(runs_dir)
            cfg = st.session_state.train_cfg
            cfg = TrainConfig(
                img_size=int(getattr(cfg, "img_size", 96)),
                color_mode="grayscale",
                batch_size=cfg.batch_size,
                epochs=cfg.epochs,
                validation_split=cfg.validation_split,
                seed=cfg.seed,
                optimizer=cfg.optimizer,
                learning_rate=cfg.learning_rate,
                conv1_filters=cfg.conv1_filters,
                conv2_filters=cfg.conv2_filters,
                dense_units=cfg.dense_units,
                representative_samples=cfg.representative_samples,
                preprocess_mode=str(getattr(cfg, "preprocess_mode", "auto_by_label")),
                manual_roi=getattr(cfg, "manual_roi", None),
            )
            st.session_state.train_cfg = cfg
            with st.spinner("Training and exporting int8 TFLite..."):
                result = train_and_export(
                    dataset_dir=dataset_dir,
                    run_dir=run_dir,
                    cfg=cfg,
                    model_base_name="model",
                    array_name="g_model",
                )
            st.session_state.train_result = result
        with st.popover("Advanced ▾", use_container_width=True):
            st.session_state.train_cfg = _render_train_config(st.session_state.train_cfg)

def _render_tm_preview_export_panel() -> None:
    result = st.session_state.train_result
    with st.container():
        st.markdown('<div class="tm-preview-card-marker"></div>', unsafe_allow_html=True)
        head_left, head_right = st.columns([1.55, 1.05])
        with head_left:
            st.markdown('<div class="tm-card-head"><h3>Preview</h3></div>', unsafe_allow_html=True)
        with head_right:
            export_clicked = st.button(
                "⇪ Export Model",
                key="tm_export_primary",
                disabled=result is None,
                use_container_width=True,
            )

        if result is None:
            st.markdown(
                '<div class="tm-card-note">You must train a model on the left before you can preview it here.</div>',
                unsafe_allow_html=True,
            )
        elif st.session_state.tm_last_device_frame:
            st.image(st.session_state.tm_last_device_frame, use_container_width=True)
        else:
            st.markdown('<div class="tm-card-note">Model ready.</div>', unsafe_allow_html=True)

    if result is None:
        return

    model_name = str(st.session_state.get("tm_model_name", "model")).strip() or "model"
    array_name = str(st.session_state.get("tm_array_name", "g_model")).strip() or "g_model"
    export_dir = Path(st.session_state.last_export_dir).expanduser().resolve() if st.session_state.last_export_dir.strip() else _default_export_dir()
    current_token = f"{export_dir}|{model_name}|{array_name}"
    if export_clicked:
        errors = _validate_export_inputs(export_dir, model_name, array_name, result.tflite_path)
        if errors:
            st.session_state.export_validated_token = ""
            st.error("\n".join(errors))
            return

        export_dir.mkdir(parents=True, exist_ok=True)
        source_bytes = result.tflite_path.read_bytes()
        from trainer import export_tflite_c_sources

        src, hdr = export_tflite_c_sources(source_bytes, array_name=array_name)
        (export_dir / f"{model_name}.tflite").write_bytes(source_bytes)
        (export_dir / "model.h").write_text(hdr, encoding="utf-8")
        (export_dir / "model.cpp").write_text('#include "model.h"\n\n' + src, encoding="utf-8")
        (export_dir / "labels.txt").write_text("\n".join(result.labels) + "\n", encoding="utf-8")
        st.success(f"Exported to: {export_dir}")


def main() -> None:
    st.set_page_config(page_title="TF Lite Training", layout="wide")
    _init_session()

    if st.session_state.project_type is None:
        _render_new_project()
        return

    if st.session_state.project_type == "image":
        _render_image_project()
    elif st.session_state.project_type == "image_classified_import":
        _render_classified_import_page()
    else:
        st.warning("This project type is not implemented yet.")


if __name__ == "__main__":
    main()
