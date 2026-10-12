from __future__ import annotations

import json
import re
import shutil
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".gif"}
COMMON_SPLIT_NAMES = {
    "train",
    "training",
    "val",
    "valid",
    "validation",
    "test",
    "testing",
}

# ── Project types ───────────────────────────────────────────────────────
# The ONE place a project type is named.  Everything downstream (session
# config, train config, .tmproj manifest, sample transform dispatch) carries
# this string rather than a boolean or a parallel code path, so adding a
# project type never means forking the persistence or training lifecycle.
PROJECT_TYPE_IMAGE = "image"
PROJECT_TYPE_OCR = "ocr"
# The classified-import page is an image-project variant, not a third
# pipeline: it produces the same workspace and trains with the same code.
PROJECT_TYPE_CLASSIFIED_IMPORT = "image_classified_import"

KNOWN_PROJECT_TYPES = (
    PROJECT_TYPE_IMAGE,
    PROJECT_TYPE_OCR,
    PROJECT_TYPE_CLASSIFIED_IMPORT,
)

# Model input size per project type.  The image project's 96 is the default
# in TrainConfig/SessionConfig; OCR works on 28x28 glyphs and reuses the SAME
# img_size plumbing rather than introducing a second size concept.
DEFAULT_IMG_SIZE_BY_PROJECT = {
    PROJECT_TYPE_IMAGE: 96,
    PROJECT_TYPE_CLASSIFIED_IMPORT: 96,
    PROJECT_TYPE_OCR: 28,
}


def default_img_size(project_type: object) -> int:
    """Model input side for a project type (falls back to the image project)."""
    key = str(project_type or PROJECT_TYPE_IMAGE).strip().lower()
    return int(DEFAULT_IMG_SIZE_BY_PROJECT.get(key, 96))


# ── App data directory ──────────────────────────────────────────────────
# Lives here, not in app.py, because the training worker and the EMNIST
# cache also need it and neither may import streamlit.  app.py and
# desktop_launcher.py resolve through this function so there is exactly one
# definition of "where the app keeps its files".
APP_NAME = "TFLiteTraining"


def app_data_dir() -> Path:
    """Writable per-user app data dir (env override, else platform default).

    Falls through to the temp dir if the preferred location cannot be
    written, so a sandboxed or read-only home does not make the app
    unusable — it just makes it forgetful.
    """
    import os
    import sys
    import tempfile

    candidates: List[Path] = []
    env_override = os.getenv("TFLITE_TRAINING_DATA_DIR")
    if env_override:
        candidates.append(Path(env_override).expanduser().resolve())
    if sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support"
        candidates.append((base / APP_NAME).resolve())
    elif os.name == "nt":
        base = Path(os.environ.get("APPDATA", str(Path.home() / "AppData" / "Roaming")))
        candidates.append((base / APP_NAME).resolve())
    else:
        base = Path(os.environ.get("XDG_DATA_HOME", str(Path.home() / ".local" / "share")))
        candidates.append((base / APP_NAME).resolve())
    candidates.append((Path(tempfile.gettempdir()) / APP_NAME).resolve())

    last_err: Optional[Exception] = None
    for candidate in candidates:
        try:
            (candidate / "workspace").mkdir(parents=True, exist_ok=True)
            (candidate / "datasets").mkdir(parents=True, exist_ok=True)
            probe_dir = candidate / "workspace" / "_probe_dir"
            probe_dir.mkdir(parents=True, exist_ok=True)
            probe = probe_dir / ".write_test"
            probe.write_text("ok", encoding="utf-8")
            probe.unlink(missing_ok=True)
            probe_dir.rmdir()
            return candidate
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(f"Failed to prepare app data dir. Last error: {last_err}")


@dataclass(frozen=True)
class ImportedData:
    root_dir: Path
    classified: bool
    images: List[Path]
    class_to_images: Dict[str, List[Path]]


def ensure_empty_dir(dir_path: Path) -> None:
    if dir_path.exists():
        shutil.rmtree(dir_path)
    dir_path.mkdir(parents=True, exist_ok=True)


def is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTS


def scan_images(root_dir: Path) -> List[Path]:
    if not root_dir.exists():
        return []
    return sorted([p for p in root_dir.rglob("*") if is_image_file(p)])


def _safe_extract_zip(zf: zipfile.ZipFile, dest_dir: Path) -> None:
    dest_dir = dest_dir.resolve()
    for member in zf.infolist():
        member_path = Path(member.filename)
        if member_path.is_absolute():
            raise ValueError(f"Illegal absolute path in zip: {member.filename}")
        if ".." in member_path.parts:
            raise ValueError(f"Illegal parent traversal in zip: {member.filename}")
        target_path = (dest_dir / member_path).resolve()
        if dest_dir not in target_path.parents and target_path != dest_dir:
            raise ValueError(f"Illegal path in zip: {member.filename}")
    zf.extractall(dest_dir)


def materialize_zip_file(zip_path: Path, dest_dir: Path) -> Path:
    if not zip_path.exists():
        raise FileNotFoundError(str(zip_path))
    ensure_empty_dir(dest_dir)
    with zipfile.ZipFile(zip_path, "r") as zf:
        _safe_extract_zip(zf, dest_dir)
    return dest_dir


def materialize_zip_bytes(zip_bytes: bytes, dest_dir: Path) -> Path:
    ensure_empty_dir(dest_dir)
    with zipfile.ZipFile(Path(dest_dir) / "__upload__.zip", "w") as _:
        pass
    zip_file_path = dest_dir / "__upload__.zip"
    zip_file_path.write_bytes(zip_bytes)
    with zipfile.ZipFile(zip_file_path, "r") as zf:
        _safe_extract_zip(zf, dest_dir)
    zip_file_path.unlink(missing_ok=True)
    return dest_dir


def materialize_files(files: Sequence[Tuple[str, bytes]], dest_dir: Path) -> Path:
    ensure_empty_dir(dest_dir)
    for name, data in files:
        safe_name = Path(name).name
        (dest_dir / safe_name).write_bytes(data)
    return dest_dir


def _infer_class_to_images_by_subdirs(root_dir: Path) -> Dict[str, List[Path]]:
    class_to_images: Dict[str, List[Path]] = {}
    if not root_dir.exists():
        return class_to_images
    for child in sorted([p for p in root_dir.iterdir() if p.is_dir()]):
        imgs = sorted([p for p in child.rglob("*") if is_image_file(p)])
        if imgs:
            class_to_images[child.name] = imgs
    return class_to_images


def _infer_class_to_images_by_split_dirs(root_dir: Path) -> Dict[str, List[Path]]:
    class_to_images: Dict[str, List[Path]] = {}
    if not root_dir.exists():
        return class_to_images

    split_dirs = [
        p for p in sorted(root_dir.iterdir())
        if p.is_dir() and p.name.strip().lower() in COMMON_SPLIT_NAMES
    ]
    for split_dir in split_dirs:
        for child in sorted([p for p in split_dir.iterdir() if p.is_dir()]):
            imgs = sorted([p for p in child.rglob("*") if is_image_file(p)])
            if not imgs:
                continue
            existing = class_to_images.setdefault(child.name, [])
            existing.extend(imgs)

    return {name: sorted(images) for name, images in class_to_images.items() if images}


def _infer_class_to_images_by_nested_split_dirs(root_dir: Path) -> Dict[str, List[Path]]:
    class_to_images: Dict[str, List[Path]] = {}
    if not root_dir.exists():
        return class_to_images

    for child in sorted([p for p in root_dir.rglob("*") if p.is_dir()]):
        parent_name = child.parent.name.strip().lower()
        if parent_name not in COMMON_SPLIT_NAMES:
            continue
        imgs = sorted([p for p in child.rglob("*") if is_image_file(p)])
        if not imgs:
            continue
        existing = class_to_images.setdefault(child.name, [])
        existing.extend(imgs)

    return {name: sorted(images) for name, images in class_to_images.items() if images}


def infer_imported_data(root_dir: Path) -> ImportedData:
    root_dir = root_dir.resolve()
    all_images = scan_images(root_dir)
    class_to_images = _infer_class_to_images_by_subdirs(root_dir)
    if len(class_to_images) < 2:
        split_class_map = _infer_class_to_images_by_split_dirs(root_dir)
        if len(split_class_map) >= 2:
            class_to_images = split_class_map
    if len(class_to_images) < 2:
        nested_split_map = _infer_class_to_images_by_nested_split_dirs(root_dir)
        if len(nested_split_map) >= 2:
            class_to_images = nested_split_map
    root_level_images = sorted([p for p in root_dir.iterdir() if is_image_file(p)])
    classified = bool(class_to_images) and len(root_level_images) == 0
    if classified:
        images: List[Path] = []
        for imgs in class_to_images.values():
            images.extend(imgs)
        images = sorted(images)
    else:
        class_to_images = {}
        images = all_images
    return ImportedData(
        root_dir=root_dir,
        classified=classified,
        images=images,
        class_to_images=class_to_images,
    )


def sanitize_class_name(name: str) -> str:
    n = name.strip()
    n = re.sub(r"[\s]+", " ", n)
    n = re.sub(r'[\\/:"*?<>|]+', "_", n)
    n = n.strip(" .")
    return n or "class"


def export_labels(labels: List[str], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "labels.txt").write_text("\n".join(labels) + "\n", encoding="utf-8")
    (out_dir / "labels.json").write_text(
        json.dumps({"labels": labels}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def export_classified_dataset(
    class_to_images: Dict[str, List[Path]],
    class_rename: Optional[Dict[str, str]],
    out_dir: Path,
) -> Path:
    ensure_empty_dir(out_dir)
    labels: List[str] = []
    for class_name, images in class_to_images.items():
        new_name = class_rename.get(class_name, class_name) if class_rename else class_name
        new_name = sanitize_class_name(new_name)
        if new_name in labels:
            raise ValueError(f"Duplicated class name after rename: {new_name}")
        labels.append(new_name)
        target_class_dir = out_dir / new_name
        target_class_dir.mkdir(parents=True, exist_ok=True)
        for i, src in enumerate(images):
            dst = target_class_dir / f"{i:06d}{src.suffix.lower()}"
            shutil.copy2(src, dst)
    export_labels(labels, out_dir)
    return out_dir


def export_from_assignments(
    images: Sequence[Path],
    assignments: Sequence[Optional[str]],
    out_dir: Path,
    require_all_assigned: bool = True,
) -> Path:
    if len(images) != len(assignments):
        raise ValueError("images and assignments length mismatch")
    ensure_empty_dir(out_dir)

    class_names: List[str] = []
    for a in assignments:
        if a is None:
            continue
        c = sanitize_class_name(a)
        if c not in class_names:
            class_names.append(c)

    if not class_names:
        raise ValueError("No class names provided")

    for class_name in class_names:
        (out_dir / class_name).mkdir(parents=True, exist_ok=True)

    per_class_idx: Dict[str, int] = {c: 0 for c in class_names}
    for src, a in zip(images, assignments):
        if a is None:
            if require_all_assigned:
                raise ValueError("Found unassigned image")
            continue
        class_name = sanitize_class_name(a)
        if class_name not in per_class_idx:
            raise ValueError(f"Unknown class: {class_name}")
        idx = per_class_idx[class_name]
        per_class_idx[class_name] = idx + 1
        dst = out_dir / class_name / f"{idx:06d}{src.suffix.lower()}"
        shutil.copy2(src, dst)

    export_labels(class_names, out_dir)
    return out_dir


def new_output_dir(base_dir: Path, prefix: str = "dataset") -> Path:
    ts = time.strftime("%Y%m%d_%H%M%S")
    return base_dir / f"{prefix}_{ts}"
