"""EMNIST mixing, class mapping and the N+1 background class.

This module owns *which pixels the OCR model trains on*.  It deliberately
does NOT own the training loop — it materialises a plain directory of PNGs
at `out_size x out_size` in exactly the layout `trainer.load_datasets`
already reads, so the entire existing training/export path (list files →
image_dataset_from_directory → fit → INT8 convert → model.h) is reused
unchanged.  Adding a dataset concept here must never require a change in
trainer.py beyond the strategy seam.

Layout produced:

    <mixed_dir>/<user class>/<nnnnnn>.png     real captures, via the class transform
    <mixed_dir>/<user class>/emnist_<nnnnn>.png   EMNIST, via preprocess_glyph_emnist
    <mixed_dir>/_background/<nnnnnn>.png      the N+1 reject class

Everything written here is 8-bit grayscale, mode "L", out_size square —
the same contract `record_controller._preprocess_image_png_ocr` writes.

── EMNIST source, orientation and labels ────────────────────────────────
Source: the canonical NIST IDX files, mirrored (no auth) at
`huggingface.co/datasets/Heliosoph/EMNIST` — ~33 MB for a whole split.
The official NIST `gzip.zip` is 561 MB and is NOT used.

ORIENTATION IS THE DANGEROUS PART.  EMNIST stores glyphs *transposed*
relative to display orientation; read raw, every character looks rotated.
Getting this wrong rotates every mixed-in sample 90 degrees and silently
poisons the model, so it was settled by measurement, not by reading prose:

  * Font-IoU over all 26 letters (render A-Z with a real TTF, take the best
    translation-aligned IoU against the EMNIST class-mean per candidate
    transform) scored  transpose 11.915  vs  identity 9.681, fliplr 9.329,
    rot180 9.223, flipud 9.158.  A 1.46 margin on an aggregate that no
    single wrong assumption can flip.
  * Empty-quadrant probe on digit '7' agrees (transpose leaves BL empty).
  * `np.rot90(a[:, ::-1]) == a.T` algebraically, and both scored identically.

A per-class quadrant test on 'L' alone gave the WRONG answer — the class
mean of *handwritten* L is a slanted blob, so single-glyph invariants are
not trustworthy here.  Do not "simplify" this back to a visual check.

Labels are taken from the built-in table below (verified against the
distributed `*-mapping.txt`), so a failed mapping download cannot silently
renumber the classes.
"""

from __future__ import annotations

import gzip
import io
import shutil
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

from ocr_preprocess import (
    GLYPH_SIZE,
    INK_BRIGHT,
    INK_DARK,
    NORM_CANON,
    align_source_polarity,
    binarize_ink,
    normalize_ink_polarity,
    normalize_norm_convention,
    otsu_threshold_u8,
    preprocess_glyph,
)

# ── The N+1 reject class ────────────────────────────────────────────────
# Reserved, filesystem-safe, and always placed LAST in the label order so
# its model-output index is exactly N (the device's OOD slot).  Callers
# must not let a user class take this name.
BACKGROUND_CLASS_NAME = "_background"

# How many distinct ways a "nothing here" frame can look.  Weighted by the
# fractions below; see synthesize_background_glyphs() for why each exists.
BG_KIND_FLAT = "flat"        # no Otsu split at all -> the device emits all-zero
BG_KIND_ERASED = "erased"    # a real capture with the glyph removed
BG_KIND_NOISE = "noise"      # sensor grain / paper texture

# ── EMNIST splits ───────────────────────────────────────────────────────
EMNIST_SPLIT_BALANCED = "balanced"
EMNIST_SPLIT_LETTERS = "letters"
EMNIST_SPLIT_DIGITS = "digits"
EMNIST_SPLITS = (EMNIST_SPLIT_BALANCED, EMNIST_SPLIT_LETTERS, EMNIST_SPLIT_DIGITS)
DEFAULT_EMNIST_SPLIT = EMNIST_SPLIT_BALANCED

# label index -> character.  These match the distributed mapping files
# exactly; they are inlined so ingestion never depends on a second
# download that could fail and renumber the classes.
_EMNIST_VOCAB: Dict[str, Tuple[str, ...]] = {
    EMNIST_SPLIT_DIGITS: tuple("0123456789"),
    EMNIST_SPLIT_LETTERS: tuple(chr(ord("A") + i) for i in range(26)),
    # balanced = 0-9, then A-Z merged-case, then the 11 lowercase letters
    # that EMNIST keeps distinct because their uppercase forms differ.
    EMNIST_SPLIT_BALANCED: tuple("0123456789")
    + tuple(chr(ord("A") + i) for i in range(26))
    + tuple("abdefghnqrt"),
}
# labels file is 1-based for `letters`, 0-based everywhere else.
_EMNIST_LABEL_BASE: Dict[str, int] = {
    EMNIST_SPLIT_DIGITS: 0,
    EMNIST_SPLIT_LETTERS: 1,
    EMNIST_SPLIT_BALANCED: 0,
}

_HF_PRIMARY = "https://huggingface.co/datasets/Heliosoph/EMNIST/resolve/main"
_HF_FALLBACK = "https://huggingface.co/datasets/Royc30ne/emnist-{split}/resolve/main"

# Timeouts are deliberately generous: the balanced train file is 28 MB.
_CONNECT_TIMEOUT = 20.0
_READ_TIMEOUT = 180.0


def emnist_label_char(split: str, label: int) -> Optional[str]:
    """EMNIST integer label -> the character it denotes (None if out of range)."""
    vocab = _EMNIST_VOCAB.get(str(split).strip().lower())
    if vocab is None:
        return None
    idx = int(label) - _EMNIST_LABEL_BASE.get(str(split).strip().lower(), 0)
    return vocab[idx] if 0 <= idx < len(vocab) else None


def emnist_char_label(split: str, char: str) -> Optional[int]:
    """Inverse of emnist_label_char: character -> the integer label."""
    key = str(split).strip().lower()
    vocab = _EMNIST_VOCAB.get(key)
    if vocab is None:
        return None
    ch = str(char).strip()
    if len(ch) != 1:
        return None
    try:
        return vocab.index(ch) + _EMNIST_LABEL_BASE.get(key, 0)
    except ValueError:
        # Handwritten case is merged in `balanced`: accept either case.
        try:
            return vocab.index(ch.upper()) + _EMNIST_LABEL_BASE.get(key, 0)
        except ValueError:
            return None


# ════════════════════════════════════════════════════════════════════════
# Acquisition + cache
# ════════════════════════════════════════════════════════════════════════
def emnist_cache_dir(app_data_dir: Path) -> Path:
    """Shared, project-independent EMNIST cache.

    One copy serves every OCR project and every session: the files are
    read-only reference data, so duplicating them per workspace would cost
    tens of megabytes each for no benefit.
    """
    return Path(app_data_dir) / "datasets" / "emnist"


def _split_dir(root: Path, split: str) -> Path:
    return Path(root) / str(split).strip().lower()


def emnist_split_files(split: str, usage: str) -> Tuple[str, str]:
    s = str(split).strip().lower()
    return (
        f"emnist-{s}-{usage}-images-idx3-ubyte.gz",
        f"emnist-{s}-{usage}-labels-idx1-ubyte.gz",
    )


def emnist_is_present(root: Path, split: str) -> bool:
    d = _split_dir(root, split)
    names = [n for usage in ("train", "test") for n in emnist_split_files(split, usage)]
    return all((d / n).exists() for n in names)


def emnist_status(root: Path, split: str) -> Dict[str, Any]:
    """Cheap, side-effect-free description for the UI."""
    d = _split_dir(root, split)
    files = sorted(p.name for p in d.glob("*.gz")) if d.exists() else []
    size = sum(p.stat().st_size for p in d.glob("*.gz")) if d.exists() else 0
    return {
        "split": str(split),
        "dir": str(d),
        "present": emnist_is_present(root, split),
        "files": files,
        "bytes": int(size),
        "classes": len(_EMNIST_VOCAB.get(str(split).strip().lower(), ())),
    }


def _download(url: str, dest: Path, progress: Optional[Callable[[float, str], None]] = None,
              label: str = "") -> None:
    """Stream to a .part file and rename only on success.

    The rename is what makes the cache crash-safe: a half-written IDX file
    that still has the final name would be read as truncated data and
    silently train on garbage.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "AItraining/1.0"})
    with urllib.request.urlopen(req, timeout=_CONNECT_TIMEOUT) as resp:
        total = int(resp.headers.get("Content-Length") or 0)
        got = 0
        with open(part, "wb") as fh:
            while True:
                chunk = resp.read(262144)
                if not chunk:
                    break
                fh.write(chunk)
                got += len(chunk)
                if progress is not None and total:
                    progress(min(1.0, got / total), f"{label} {got // 1048576}/{total // 1048576} MB")
    if total and got != total:
        part.unlink(missing_ok=True)
        raise IOError(f"short download for {url}: {got}/{total} bytes")
    part.replace(dest)


def ensure_emnist_split(
    root: Path,
    split: str = DEFAULT_EMNIST_SPLIT,
    *,
    progress: Optional[Callable[[float, str], None]] = None,
    allow_download: bool = True,
    usages: Sequence[str] = ("train", "test"),
) -> Path:
    """Make `<root>/<split>/` hold both IDX pairs.  Idempotent.

    Raises `FileNotFoundError` if the data is absent and downloading is
    refused or fails — callers turn that into "train on user data only",
    never into a hard stop, so the app stays usable offline.
    """
    root = Path(root)
    s = str(split).strip().lower()
    if s not in _EMNIST_VOCAB:
        raise ValueError(f"unknown EMNIST split: {split!r} (have {EMNIST_SPLITS})")
    d = _split_dir(root, s)
    missing = [n for usage in usages for n in emnist_split_files(s, usage) if not (d / n).exists()]
    if not missing:
        return d
    if not allow_download:
        raise FileNotFoundError(f"EMNIST {s} not cached in {d} and download disabled")

    mirrors = [_HF_PRIMARY, _HF_FALLBACK.format(split=s)]
    errors: List[str] = []
    for i, base in enumerate(mirrors):
        try:
            todo = list(missing)
            for j, name in enumerate(todo):
                if progress is not None:
                    progress(j / max(1, len(todo)), f"Downloading EMNIST {s}: {name}")
                _download(f"{base}/{name}", d / name, progress=progress, label=f"EMNIST {s}")
            if all((d / n).exists() for n in missing):
                return d
        except (urllib.error.URLError, urllib.error.HTTPError, IOError, OSError, TimeoutError) as e:
            errors.append(f"{base}: {e}")
            if i == len(mirrors) - 1:
                break
    raise FileNotFoundError(
        "could not obtain EMNIST {} — tried:\n  {}".format(s, "\n  ".join(errors))
    )


def _read_idx(path: Path) -> np.ndarray:
    """Parse an IDX file (big-endian header, uint8 payload)."""
    with gzip.open(path, "rb") as fh:
        magic = int.from_bytes(fh.read(4), "big")
        ndim = magic & 0xFF
        if magic >> 8 not in (0x08, 0x09, 0x0B, 0x0C, 0x0D, 0x0E):
            raise ValueError(f"not an IDX file: {path}")
        dims = [int.from_bytes(fh.read(4), "big") for _ in range(ndim)]
        data = np.frombuffer(fh.read(), dtype=np.uint8)
    return data.reshape(dims)


@dataclass(frozen=True)
class EmnistData:
    """A loaded split, already in upright (display) orientation."""

    split: str
    images: np.ndarray   # uint8 [N, 28, 28], upright
    labels: np.ndarray   # int64 [N], EMNIST integer labels
    chars: List[Optional[str]]

    def count_for_label(self, label: int) -> int:
        return int((self.labels == int(label)).sum())


def load_emnist(
    root: Path,
    split: str = DEFAULT_EMNIST_SPLIT,
    usages: Sequence[str] = ("train",),
    cache: Optional[Dict[Tuple[str, str], EmnistData]] = None,
) -> EmnistData:
    """Load a split from the cache directory, transposing to upright.

    `cache` is a plain dict the caller can pass across calls; a 47-class
    train file is ~28 MB compressed and ~88 MB in memory, so re-reading it
    per class would dominate the mixing step.
    """
    s = str(split).strip().lower()
    key = (str(Path(root)), s + ":" + ",".join(usages))
    if cache is not None and key in cache:
        return cache[key]

    d = _split_dir(root, s)
    img_parts: List[np.ndarray] = []
    lab_parts: List[np.ndarray] = []
    for usage in usages:
        img_name, lab_name = emnist_split_files(s, usage)
        if not (d / img_name).exists():
            continue
        img_parts.append(_read_idx(d / img_name))
        lab_parts.append(_read_idx(d / lab_name))
    if not img_parts:
        raise FileNotFoundError(f"no EMNIST {s} IDX files under {d}")

    images = np.concatenate(img_parts, axis=0)
    labels = np.concatenate(lab_parts, axis=0).astype(np.int64)
    # ── THE transpose ───────────────────────────────────────────────────
    # Stored glyphs are reflected across the main diagonal; (0,2,1) makes
    # them upright.  See the module docstring for the measurement that
    # established this.  A 90-degree error here is invisible in the loss
    # curve and fatal on the device.
    images = np.ascontiguousarray(images.transpose(0, 2, 1))

    vocab = _EMNIST_VOCAB[s]
    base = _EMNIST_LABEL_BASE[s]
    chars = [chr(0)] * (len(vocab) + base)
    for i, ch in enumerate(vocab):
        chars[base + i] = ch

    data = EmnistData(split=s, images=images, labels=labels, chars=chars)
    if cache is not None:
        cache[key] = data
    return data


# ════════════════════════════════════════════════════════════════════════
# Class-name -> EMNIST character
# ════════════════════════════════════════════════════════════════════════
# Prefixes users put in front of the character itself ("Digit 1",
# "Letter A", "Char 7").  Stripped before matching.
_CLASS_PREFIXES = ("digit", "letter", "char", "character", "symbol", "glyph", "num", "number")


def map_class_to_char(class_name: str) -> Optional[str]:
    """Best-effort single-character reading of a user class name.

    Accepts the whole name first ("A", "7"), then the trailing token
    ("Digit 1", "Stroke L"), then the leading token ("A - alpha").  A name
    that does not reduce to exactly one character maps to None and simply
    gets no EMNIST augmentation — it is never guessed at.
    """
    raw = str(class_name or "").strip()
    if not raw:
        return None
    if len(raw) == 1:
        return raw

    tokens = [t for t in raw.replace("_", " ").replace("-", " ").split() if t]
    if not tokens:
        return None

    def _strip_prefix(ts: List[str]) -> List[str]:
        while ts and ts[0].lower().rstrip(":.") in _CLASS_PREFIXES:
            ts = ts[1:]
        return ts

    stripped = _strip_prefix(list(tokens))
    if len(stripped) == 1 and len(stripped[0]) == 1:
        return stripped[0]
    if stripped and len(stripped[-1]) == 1:
        return stripped[-1]
    if stripped and len(stripped[0]) == 1:
        return stripped[0]
    return None


def resolve_class_map(
    class_names: Sequence[str],
    split: str = DEFAULT_EMNIST_SPLIT,
    class_preprocess: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Dict[str, Optional[int]]:
    """{user class name -> EMNIST label int or None} for every class.

    A per-class override in the class-preprocess record (`emnist_char`)
    wins over the heuristic — that is the escape hatch for a class the
    name-based reading gets wrong, and it rides on the EXISTING per-class
    config record rather than inventing another config file.
    """
    pre = class_preprocess or {}
    out: Dict[str, Optional[int]] = {}
    for name in class_names:
        cfg = pre.get(name) if isinstance(pre.get(name), dict) else {}
        override = str(cfg.get("emnist_char") or "").strip()
        ch = override if override else map_class_to_char(name)
        out[str(name)] = emnist_char_label(split, ch) if ch else None
    return out


# ════════════════════════════════════════════════════════════════════════
# Template samples: EMNIST glyphs drawn as camera-style captures
# ════════════════════════════════════════════════════════════════════════
# The developer shortcut for trying the pipeline before a camera is wired
# up: render EMNIST glyphs as 96x96 DARK-INK-ON-LIGHT frames and store them
# where a capture would go.  The point is NOT to inject already-isolated
# 28x28 glyphs (that is what training-time EMNIST mixing does, bypassing
# segmentation) — it is to produce images the SAME front-end can binarise,
# segment and crop, so Auto ROI, preprocess_glyph and the device C port all
# take their ordinary paths.  Get the polarity backwards and every template
# sample trains as background.
TEMPLATE_SAMPLE_COUNT = 24
TEMPLATE_SAMPLE_SIDE = 96
TEMPLATE_SAMPLE_BG = 242       # light page, like a white car-sign plate
TEMPLATE_SAMPLE_INK = 20       # dark print
TEMPLATE_BANK_FILENAME = "emnist_byclass.npz"
# Classes a road sign can print and this bank can serve.  Lower-case EMNIST
# classes are deliberately NOT folded into upper case here: 'a' and 'A' are
# different glyphs, and silently relabelling one as the other would teach
# the model a shape its class never shows in the UI.
TEMPLATE_SIGN_CHARS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def emnist_bank_path() -> Path:
    """The optional local glyph bank shipped with the app."""
    return Path(__file__).resolve().parent / "models" / TEMPLATE_BANK_FILENAME


def _bank_add(bank: Dict[str, List[np.ndarray]], char: Any, image: np.ndarray) -> None:
    ch = str(char or "").strip().upper()
    if len(ch) != 1 or ch not in TEMPLATE_SIGN_CHARS:
        return
    arr = np.asarray(image)
    if arr.ndim != 2 or arr.shape[0] < 2 or arr.shape[1] < 2:
        return
    bank.setdefault(ch, []).append(np.ascontiguousarray(arr, dtype=np.uint8))


def _bank_from_npz(path: Path) -> Dict[str, List[np.ndarray]]:
    """Read the local bank.  Two layouts are accepted:

        images [N,28,28] + chars [N]                 per-sample character
        images [N,28,28] + labels [N] + label_chars  label LUT

    The file is a cache, not a format: any future producer only has to say
    which character each glyph is.
    """
    bank: Dict[str, List[np.ndarray]] = {}
    with np.load(path, allow_pickle=False) as z:
        if "images" not in z:
            raise ValueError(f"{path} has no 'images' array")
        images = np.asarray(z["images"])
        if images.ndim == 3 and images.shape[1:] == (28, 28):
            pass
        elif images.ndim == 2 and images.shape[1] == 28 * 28:
            images = images.reshape((-1, 28, 28))
        else:
            raise ValueError(f"{path}: unsupported image shape {images.shape}")
        if "chars" in z:
            chars = [str(c) for c in z["chars"]]
        elif "labels" in z and "label_chars" in z:
            lut = [str(c) for c in z["label_chars"]]
            chars = [
                lut[int(l)] if 0 <= int(l) < len(lut) else ""
                for l in np.asarray(z["labels"]).ravel()
            ]
        else:
            raise ValueError(f"{path} has neither 'chars' nor 'labels'/'label_chars'")
    for image, ch in zip(images, chars):
        _bank_add(bank, ch, image)
    return bank


def _bank_from_idx_cache(root: Path) -> Dict[str, List[np.ndarray]]:
    """Fallback for a machine that has already downloaded EMNIST but no npz."""
    bank: Dict[str, List[np.ndarray]] = {}
    for split in (EMNIST_SPLIT_BALANCED, EMNIST_SPLIT_LETTERS, EMNIST_SPLIT_DIGITS):
        try:
            if not emnist_is_present(Path(root), split):
                continue
            data = load_emnist(Path(root), split, usages=("train",))
        except Exception:
            continue
        for image, label in zip(data.images, data.labels):
            i = int(label)
            if 0 <= i < len(data.chars):
                _bank_add(bank, data.chars[i], image)
    return bank


def _template_font(size: int):
    from PIL import ImageFont

    candidates: List[Path] = []
    try:
        import matplotlib

        candidates.append(Path(matplotlib.get_data_path()) / "fonts" / "ttf" / "DejaVuSans-Bold.ttf")
    except Exception:
        pass
    candidates.extend(
        [
            Path("/System/Library/Fonts/Supplemental/Arial Bold.ttf"),
            Path("/System/Library/Fonts/Supplemental/Arial.ttf"),
            Path("/Library/Fonts/Arial Bold.ttf"),
            Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
            Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
            Path("C:/Windows/Fonts/arialbd.ttf"),
            Path("C:/Windows/Fonts/arial.ttf"),
        ]
    )
    for p in candidates:
        try:
            if p.exists():
                return ImageFont.truetype(str(p), int(size))
        except Exception:
            continue
    return ImageFont.load_default()


def _synthetic_glyph_28(char: str) -> np.ndarray:
    """Last-resort glyph when neither the bank nor an EMNIST cache exists.

    Rendered at 4x and downsampled so it has the soft edges of a real
    EMNIST stroke rather than the aliased stair-steps of a bitmap font.
    """
    from PIL import Image, ImageDraw

    ch = str(char or "").strip().upper()[:1]
    if not ch:
        raise ValueError("empty character")
    size = 112
    font = _template_font(int(size * 0.74))
    canvas = Image.new("L", (size, size), 0)
    draw = ImageDraw.Draw(canvas)
    try:
        box = draw.textbbox((0, 0), ch, font=font)
    except Exception:
        box = (0, 0, size, size)
    w = max(1, int(box[2] - box[0]))
    h = max(1, int(box[3] - box[1]))
    # Fit the ink into ~72 % of the 28 px canvas, the same natural margin
    # EMNIST leaves; centre it by its real ink box, not its font metrics.
    scale = min((size * 0.72) / w, (size * 0.72) / h, 1.0)
    if scale < 1.0:
        font = _template_font(max(6, int(int(size * 0.74) * scale)))
        try:
            box = draw.textbbox((0, 0), ch, font=font)
        except Exception:
            box = (0, 0, size, size)
        w = max(1, int(box[2] - box[0]))
        h = max(1, int(box[3] - box[1]))
    draw.text(
        (size / 2.0 - w / 2.0 - box[0], size / 2.0 - h / 2.0 - box[1]),
        ch,
        fill=255,
        font=font,
    )
    return np.asarray(canvas.resize((28, 28), Image.LANCZOS), dtype=np.uint8)


def load_emnist_template_bank(
    npz_path: Optional[Path] = None,
    cache_root: Optional[Path] = None,
) -> Dict[str, List[np.ndarray]]:
    """char -> upright 28x28 glyphs, best source first.

    Local `models/emnist_byclass.npz`, then an existing EMNIST IDX cache,
    then nothing — callers fall back to `_synthetic_glyph_28` per class.
    Never raises and never downloads: seeding is an offline developer
    convenience, so a missing cache must not turn a button click into a
    network operation.
    """
    bank: Dict[str, List[np.ndarray]] = {}
    path = Path(npz_path) if npz_path is not None else emnist_bank_path()
    if path.exists():
        try:
            bank = _bank_from_npz(path)
        except Exception:
            bank = {}
    if not bank and cache_root is not None:
        bank = _bank_from_idx_cache(Path(cache_root))
    return bank


def template_capture_from_glyph(
    glyph: np.ndarray,
    rng: np.random.Generator,
    side: int = TEMPLATE_SAMPLE_SIDE,
) -> np.ndarray:
    """One isolated glyph -> one 96x96 dark-ink-on-light camera-style frame.

    The glyph is cropped to its own ink box, scaled to a sign-sized height
    (38-52 px in a 96 px frame), jittered, rotated a few degrees and
    composited onto a ~242-level page with ~20-level ink, then softened and
    noised.  Every one of those choices exists to keep the result inside
    the distribution the capture front-end was built for: Otsu finds a
    real bimodal split, the projection sees one ink run, and the model
    sees the same gentle anti-aliasing a printed sign has.
    """
    from PIL import Image, ImageFilter

    g = np.asarray(glyph, dtype=np.float32) / 255.0
    if g.ndim != 2 or g.size == 0:
        raise ValueError("glyph must be a non-empty 2-D array")
    ys, xs = np.where(g > 0.08)
    if ys.size:
        g = g[int(ys.min()) : int(ys.max()) + 1, int(xs.min()) : int(xs.max()) + 1]
    side = max(32, int(side))
    target_h = int(rng.integers(38, 53))
    gh, gw = g.shape
    target_w = max(2, int(round(gw * (target_h / float(max(1, gh))))))
    up = Image.fromarray(np.clip(np.rint(g * 255.0), 0, 255).astype(np.uint8), mode="L")
    up = up.resize((target_w, target_h), Image.BICUBIC)
    layer = Image.new("L", (side, side), 0)
    x = (side - target_w) // 2 + int(rng.integers(-5, 6))
    y = (side - target_h) // 2 + int(rng.integers(-5, 6))
    layer.paste(up, (max(0, x), max(0, y)))
    angle = float(rng.uniform(-7.0, 7.0))
    if abs(angle) > 0.5:
        layer = layer.rotate(angle, resample=Image.BICUBIC, center=(side / 2.0, side / 2.0), fillcolor=0)
    layer = layer.filter(ImageFilter.GaussianBlur(float(rng.uniform(0.2, 0.7))))
    alpha = np.asarray(layer, dtype=np.float32) / 255.0
    bg = np.float32(TEMPLATE_SAMPLE_BG) + rng.normal(0.0, 2.0, (side, side)).astype(np.float32)
    ink = np.float32(TEMPLATE_SAMPLE_INK) + rng.normal(0.0, 2.0, (side, side)).astype(np.float32)
    out = bg * (1.0 - alpha) + ink * alpha
    return np.clip(np.rint(out), 0.0, 255.0).astype(np.uint8)


def build_template_capture_samples(
    char: str,
    count: int = TEMPLATE_SAMPLE_COUNT,
    *,
    bank: Optional[Dict[str, List[np.ndarray]]] = None,
    seed: Optional[int] = None,
) -> List[np.ndarray]:
    """`count` camera-style 96x96 samples for one character class.

    Deterministic for a given (char, count): the same click always writes
    the same template set, so a template project is reproducible and two
    runs of the test suite compare against the same pixels.
    """
    ch = str(char or "").strip().upper()[:1]
    if len(ch) != 1 or ch not in TEMPLATE_SIGN_CHARS:
        raise ValueError(f"'{char}' is not a sign character (0-9, A-Z)")
    n = max(1, int(count))
    glyphs = list((bank or {}).get(ch) or [])
    key = int(seed) if seed is not None else int(ch.encode("ascii")[0]) * 97 + len(ch)
    rng = np.random.default_rng(key)
    order = rng.permutation(len(glyphs)) if glyphs else np.zeros(0, dtype=np.int64)
    synthetic = None
    out: List[np.ndarray] = []
    for i in range(n):
        if glyphs:
            g = glyphs[int(order[i % len(order)])]
        else:
            if synthetic is None:
                synthetic = _synthetic_glyph_28(ch)
            g = synthetic
        out.append(template_capture_from_glyph(g, rng))
    return out


def render_template_sign_frame(
    chars: Sequence[str],
    *,
    side: int = TEMPLATE_SAMPLE_SIDE,
    bank: Optional[Dict[str, List[np.ndarray]]] = None,
    seed: int = 0,
) -> np.ndarray:
    """A whole multi-character sign, left-to-right, in one camera-style frame.

    Used to build the demo `A 1 2` / `A 1 7` images.  Glyphs are laid out
    with real gaps so `split_sign_glyph_boxes` finds exactly one run per
    character — the same physical assumption a printed sign satisfies.
    """
    from PIL import Image, ImageFilter

    order = [str(c).strip().upper()[:1] for c in chars if str(c).strip()]
    if not order:
        raise ValueError("no characters")
    side = max(48, int(side))
    rng = np.random.default_rng(int(seed))
    target_h = int(rng.integers(33, 39))
    layers: List[Any] = []
    for ch in order:
        glyphs = list((bank or {}).get(ch) or [])
        g = np.asarray(glyphs[int(rng.integers(0, len(glyphs)))] if glyphs else _synthetic_glyph_28(ch),
                       dtype=np.float32) / 255.0
        ys, xs = np.where(g > 0.08)
        if ys.size:
            g = g[int(ys.min()) : int(ys.max()) + 1, int(xs.min()) : int(xs.max()) + 1]
        h = max(8, target_h + int(rng.integers(-2, 3)))
        w = max(2, int(round(g.shape[1] * (h / float(max(1, g.shape[0]))))))
        layer = Image.fromarray(np.clip(np.rint(g * 255.0), 0, 255).astype(np.uint8), mode="L")
        layers.append(layer.resize((w, h), Image.BICUBIC))
    gap = max(4, int(round(side * 0.075)))
    total = sum(layer.width for layer in layers) + gap * (len(layers) - 1)
    if total > side - 6:
        scale = (side - 6) / float(total)
        layers = [
            layer.resize((max(2, int(round(layer.width * scale))), max(2, int(round(layer.height * scale)))),
                         Image.BICUBIC)
            for layer in layers
        ]
        total = sum(layer.width for layer in layers) + gap * (len(layers) - 1)
    canvas = Image.new("L", (side, side), 0)
    x = max(0, (side - total) // 2)
    for layer in layers:
        y = (side - layer.height) // 2 + int(rng.integers(-2, 3))
        canvas.paste(layer, (x, max(0, y)))
        x += layer.width + gap
    angle = float(rng.uniform(-1.5, 1.5))
    if abs(angle) > 0.3:
        canvas = canvas.rotate(angle, resample=Image.BICUBIC, center=(side / 2.0, side / 2.0), fillcolor=0)
    canvas = canvas.filter(ImageFilter.GaussianBlur(float(rng.uniform(0.25, 0.6))))
    alpha = np.asarray(canvas, dtype=np.float32) / 255.0
    bg = np.float32(TEMPLATE_SAMPLE_BG) + rng.normal(0.0, 2.0, (side, side)).astype(np.float32)
    ink = np.float32(TEMPLATE_SAMPLE_INK) + rng.normal(0.0, 2.0, (side, side)).astype(np.float32)
    out = bg * (1.0 - alpha) + ink * alpha
    return np.clip(np.rint(out), 0.0, 255.0).astype(np.uint8)


# ════════════════════════════════════════════════════════════════════════
# The N+1 background class
# ════════════════════════════════════════════════════════════════════════
def _erase_ink(gray: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Remove the glyph from a real capture by resampling its own ink pixels.

    Each ink pixel takes the value of a randomly chosen NON-ink pixel from
    the same frame (plus a little noise), so the result keeps the frame's
    real illumination gradient, vignetting and grain instead of turning
    into a suspiciously flat patch the model can trivially spot.
    """
    g = np.asarray(gray, dtype=np.uint8)
    t = otsu_threshold_u8(g)
    if t == 0:
        return g.copy()
    ink = binarize_ink(g, t, polarity=INK_DARK)
    if not bool(ink.any()):
        return g.copy()
    donor = g[~ink]
    if donor.size == 0:
        return g.copy()
    out = g.copy()
    picks = rng.integers(0, donor.size, size=int(ink.sum()))
    fill = donor.ravel()[picks].astype(np.int16)
    fill += rng.integers(-6, 7, size=fill.shape)
    out[ink] = np.clip(fill, 0, 255).astype(np.uint8)
    return out


def _noise_frame(h: int, w: int, rng: np.random.Generator) -> np.ndarray:
    """Sensor grain + a smooth illumination gradient, no glyph."""
    base = float(rng.integers(90, 200))
    yy = np.linspace(-1.0, 1.0, h, dtype=np.float32)[:, None]
    xx = np.linspace(-1.0, 1.0, w, dtype=np.float32)[None, :]
    grad = base + rng.uniform(-30, 30) * yy + rng.uniform(-30, 30) * xx
    grain = rng.normal(0.0, rng.uniform(3.0, 14.0), size=(h, w)).astype(np.float32)
    return np.clip(grad + grain, 0, 255).astype(np.uint8)


def synthesize_background_glyphs(
    raw_frames: Sequence[np.ndarray],
    count: int,
    *,
    out_size: int = GLYPH_SIZE,
    polarity: int = INK_DARK,
    convention: str = NORM_CANON,
    seed: int = 42,
    frame_side: Optional[int] = None,
) -> List[Tuple[np.ndarray, str]]:
    """Build `count` (glyph, kind) pairs for the reject class.

    Three sources, because they are the three distinct things the device
    can hand the model when nothing is in view:

    BG_KIND_FLAT    A frame with no Otsu split at all.  This is not a
                    hypothetical: `preprocess_glyph` returns an ALL-ZERO
                    glyph for it, so if the model never sees a zero image
                    during training it has never seen the single most
                    common negative the device produces.
    BG_KIND_ERASED  A real capture with its glyph removed.  Real
                    illumination, real grain, real paper texture — the
                    negatives that actually get confused with a faint mark.
    BG_KIND_NOISE   Grain + gradient with no capture behind it.  Covers a
                    camera pointed at something the user never captured.
    """
    rng = np.random.default_rng(int(seed))
    side = int(frame_side or (raw_frames[0].shape[0] if len(raw_frames) else 96))
    out: List[Tuple[np.ndarray, str]] = []
    if count <= 0:
        return out

    n_flat = max(1, int(round(count * 0.30)))
    n_erased = max(1, int(round(count * 0.40))) if raw_frames else 0
    n_noise = max(1, count - n_flat - n_erased)

    def _run(frame: np.ndarray) -> np.ndarray:
        res = preprocess_glyph(
            frame, out_size=int(out_size), polarity=polarity,
            apply_segmentation=True, convention=convention,
        )
        return np.asarray(res.get("glyph"), dtype=np.uint8)

    # flat: a constant frame has no split, so this exercises the t == 0
    # branch and yields the all-zero glyph the device really emits.
    for _ in range(n_flat):
        level = int(rng.integers(0, 256))
        frame = np.full((side, side), level, dtype=np.uint8)
        out.append((_run(frame), BG_KIND_FLAT))

    for i in range(n_erased):
        src = np.asarray(raw_frames[i % len(raw_frames)], dtype=np.uint8)
        if src.ndim == 3:
            src = src[:, :, 0]
        out.append((_run(_erase_ink(src, rng)), BG_KIND_ERASED))

    for _ in range(n_noise):
        out.append((_run(_noise_frame(side, side, rng)), BG_KIND_NOISE))

    return out[:count] if len(out) > count else out


# ════════════════════════════════════════════════════════════════════════
# Mixing
# ════════════════════════════════════════════════════════════════════════
@dataclass
class OcrMixConfig:
    """Everything that decides the training set's composition."""

    out_size: int = GLYPH_SIZE
    emnist_enabled: bool = True
    emnist_split: str = DEFAULT_EMNIST_SPLIT
    emnist_per_class: int = 200
    # 0 -> auto (mirror `emnist_per_class` so the reject class carries
    # comparable mass to any single glyph class).
    background_samples: int = 0
    background_enabled: bool = True
    # Seeded so two runs on identical data produce an identical training
    # set; changing it re-draws the EMNIST subset.
    seed: int = 42
    # Per-class transform, used for BOTH the user's real captures and the
    # EMNIST injection, so the two halves cannot drift apart.
    class_preprocess: Optional[Dict[str, Dict[str, Any]]] = None
    default_polarity: int = INK_DARK
    default_convention: str = NORM_CANON

    def bg_count(self) -> int:
        return int(self.background_samples) if int(self.background_samples) > 0 else max(1, int(self.emnist_per_class))


@dataclass
class OcrMixResult:
    """What the mixing step produced, for the UI and for train metadata."""

    mixed_dir: Path
    class_names: List[str]          # user classes in order, background LAST
    counts: Dict[str, int]          # class -> total samples written
    user_counts: Dict[str, int]
    emnist_counts: Dict[str, int]
    background_counts: Dict[str, int]
    class_map: Dict[str, Optional[int]]
    emnist_split: str
    emnist_available: bool
    warnings: List[str] = field(default_factory=list)

    @property
    def total(self) -> int:
        return int(sum(self.counts.values()))

    def to_meta(self) -> Dict[str, Any]:
        """JSON-safe summary recorded in the run's metadata."""
        return {
            "emnist_split": self.emnist_split,
            "emnist_available": bool(self.emnist_available),
            "emnist_per_class": dict(self.emnist_counts),
            "user_per_class": dict(self.user_counts),
            "background_count": int(self.background_counts.get(BACKGROUND_CLASS_NAME, 0)),
            "class_map": {k: (v if v is None else int(v)) for k, v in self.class_map.items()},
            "counts": dict(self.counts),
            "total": self.total,
            "mixed_dir": str(self.mixed_dir),
            "warnings": list(self.warnings),
        }


def _write_u8_png(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(arr, dtype=np.uint8), mode="L").save(path)


def _read_gray(path: Path) -> Optional[np.ndarray]:
    try:
        return np.asarray(Image.open(path).convert("L"), dtype=np.uint8)
    except Exception:
        return None


def build_mixed_dataset(
    *,
    processed_dataset_dir: Path,
    mixed_dir: Path,
    class_names: Sequence[str],
    cfg: OcrMixConfig,
    raw_dataset_dir: Optional[Path] = None,
    emnist_root: Optional[Path] = None,
    progress: Optional[Callable[[float, str], None]] = None,
    allow_download: bool = True,
) -> OcrMixResult:
    """Materialise the mixed training set.  Returns what it wrote.

    `processed_dataset_dir` is the project's processed cache — the 28x28
    glyphs `record_controller._rebuild_processed_cache` already produced by
    running every capture through the integer pipeline.  Those are copied
    VERBATIM rather than re-transformed here: the cache is the canonical
    artifact, and copying it is what guarantees the model trains on exactly
    the bytes the device will compute at inference time.  Re-deriving them
    would be a second implementation of the same transform — the one thing
    this codebase does not do.

    `raw_dataset_dir` (optional) is the un-cropped capture directory.  It
    is only used to synthesise realistic background frames; when absent the
    background class is built from the processed glyphs instead, which
    still works but has less of the camera's actual illumination in it.

    The output is a plain image-class-folder directory, so `trainer.py`
    consumes it with no OCR-specific code at all.
    """
    src = Path(processed_dataset_dir)
    raw_src = Path(raw_dataset_dir) if raw_dataset_dir is not None else None
    dest = Path(mixed_dir)
    if dest.exists():
        shutil.rmtree(dest, ignore_errors=True)
    dest.mkdir(parents=True, exist_ok=True)

    class_list = [str(c) for c in class_names]
    # A user class may not squat on the reserved reject name.
    if BACKGROUND_CLASS_NAME in class_list:
        raise ValueError(f"{BACKGROUND_CLASS_NAME!r} is reserved for the reject class")

    pre_map = cfg.class_preprocess or {}
    class_map = resolve_class_map(class_list, cfg.emnist_split, pre_map)

    counts: Dict[str, int] = {}
    user_counts: Dict[str, int] = {}
    emnist_counts: Dict[str, int] = {}
    warnings: List[str] = []

    def _cfg_for(name: str) -> Dict[str, Any]:
        raw = pre_map.get(name)
        return raw if isinstance(raw, dict) else {}

    if progress is not None:
        progress(0.02, "Mixing datasets...")

    # ── 1. The user's own captures (already transformed) ────────────────
    raw_frames: List[np.ndarray] = []
    n_classes = max(1, len(class_list))
    for ci, name in enumerate(class_list):
        src_dir = src / name
        files = sorted(p for p in src_dir.glob("*") if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"})
        written = 0
        for i, p in enumerate(files):
            gray = _read_gray(p)
            if gray is None:
                continue
            _write_u8_png(dest / name / f"{i:06d}.png", gray)
            written += 1
        user_counts[name] = written
        counts[name] = written
        if written == 0:
            warnings.append(f"class '{name}' has no readable images")
        # Background frames come from the RAW captures of the same class
        # when available; a bounded sample is plenty.
        raw_dir = (raw_src / name) if raw_src is not None else src_dir
        for p in sorted(raw_dir.glob("*")) if raw_dir.exists() else []:
            if len(raw_frames) >= 64:
                break
            if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}:
                g = _read_gray(p)
                if g is not None:
                    raw_frames.append(g)
        if progress is not None:
            progress(0.02 + 0.25 * (ci + 1) / n_classes, f"Mixing captures: {name}")

    # ── 2. EMNIST injection ─────────────────────────────────────────────
    emnist_available = False
    emnist_data: Optional[EmnistData] = None
    if cfg.emnist_enabled and int(cfg.emnist_per_class) > 0:
        try:
            if emnist_root is None:
                raise FileNotFoundError("no EMNIST cache directory configured")
            ensure_emnist_split(Path(emnist_root), cfg.emnist_split,
                                progress=progress, allow_download=allow_download)
            emnist_data = load_emnist(Path(emnist_root), cfg.emnist_split, usages=("train",))
            emnist_available = True
        except Exception as e:
            warnings.append(f"EMNIST unavailable, training on user data only: {e}")

    if emnist_data is not None:
        rng = np.random.default_rng(int(cfg.seed))
        for name in class_list:
            label = class_map.get(name)
            if label is None:
                emnist_counts[name] = 0
                continue
            idx = np.flatnonzero(emnist_data.labels == int(label))
            if idx.size == 0:
                emnist_counts[name] = 0
                warnings.append(f"class '{name}' mapped to EMNIST '{emnist_data.chars[label] if label < len(emnist_data.chars) else label}' but the split has no such samples")
                continue
            take = min(int(cfg.emnist_per_class), int(idx.size))
            # Deterministic subset: same seed -> same training set.
            chosen = rng.permutation(idx)[:take]
            cls_cfg = _cfg_for(name)
            # EMNIST is injected through the class's OWN convention so the
            # synthetic half and the captured half are normalised alike.
            convention = str(normalize_norm_convention(cls_cfg.get("glyph_convention", cfg.default_convention)))
            target_polarity = int(cfg.default_polarity)
            for j, k in enumerate(chosen):
                # EMNIST ships bright-on-dark; the project's captures are
                # whatever the camera sees.  Invert the SOURCE so the
                # synthetic half reaches preprocess_glyph already agreeing
                # with the deployment convention.
                src_img = align_source_polarity(
                    emnist_data.images[int(k)],
                    source_polarity=INK_BRIGHT,
                    target_polarity=target_polarity,
                )
                res = preprocess_glyph(
                    src_img,
                    out_size=int(cfg.out_size),
                    polarity=target_polarity,
                    apply_segmentation=False,  # already isolated and centred
                    convention=convention,
                )
                _write_u8_png(dest / name / f"emnist_{j:05d}.png",
                              np.asarray(res.get("glyph"), dtype=np.uint8))
            emnist_counts[name] = take
            counts[name] = counts.get(name, 0) + take
        if progress is not None:
            progress(0.62, "Mixed EMNIST samples")

    unmapped = [n for n in class_list if class_map.get(n) is None]
    if cfg.emnist_enabled and unmapped and emnist_data is not None:
        warnings.append(
            "no EMNIST match for " + ", ".join(f"'{n}'" for n in unmapped)
            + " — these classes train on captures only"
        )

    # ── 3. The N+1 reject class ─────────────────────────────────────────
    background_counts: Dict[str, int] = {}
    if cfg.background_enabled:
        target = int(cfg.bg_count())
        # The background transform is a CAPTURE transform (dark ink on a
        # bright page, glyph somewhere in frame), not the EMNIST one: the
        # device reaches this class from a live frame.
        bg_polarity = cfg.default_polarity
        bg_convention = cfg.default_convention
        for name in class_list:
            c = _cfg_for(name)
            if "glyph_convention" in c:
                bg_convention = str(normalize_norm_convention(c.get("glyph_convention")))
            if "ink_polarity" in c:
                bg_polarity = int(normalize_ink_polarity(c.get("ink_polarity")))
            break
        pairs = synthesize_background_glyphs(
            raw_frames, target, out_size=int(cfg.out_size),
            polarity=bg_polarity, convention=bg_convention, seed=int(cfg.seed) + 977,
        )
        for j, (glyph, _kind) in enumerate(pairs):
            _write_u8_png(dest / BACKGROUND_CLASS_NAME / f"{j:06d}.png", glyph)
        background_counts[BACKGROUND_CLASS_NAME] = len(pairs)
        counts[BACKGROUND_CLASS_NAME] = len(pairs)
        if progress is not None:
            progress(0.72, f"Built {len(pairs)} background samples")

    class_names_out = class_list + ([BACKGROUND_CLASS_NAME] if cfg.background_enabled else [])
    if progress is not None:
        progress(0.75, "Dataset mixed")

    return OcrMixResult(
        mixed_dir=dest,
        class_names=class_names_out,
        counts=counts,
        user_counts=user_counts,
        emnist_counts=emnist_counts,
        background_counts=background_counts,
        class_map=class_map,
        emnist_split=str(cfg.emnist_split),
        emnist_available=bool(emnist_available),
        warnings=warnings,
    )
