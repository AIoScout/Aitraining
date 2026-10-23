"""Transfer-learning strategy for the OCR / character project.

This is the OCR half of the training pipeline.  It is NOT a second training
pipeline: `trainer.train_and_export` remains the single spine (adjust →
prepare → build → fit → evaluate → INT8 → export → self-check), and this
module supplies only the three things that genuinely differ for a
character model:

  prepare()      mix EMNIST + the user's captures + a background reject
                 class into one image-folder directory (ocr_dataset);
  build_model()  load a pretrained glyph backbone, STRIP its head, and
                 attach a fresh Dense(N+1) head;
  fit()          two-phase fine-tuning — head warm-up with the backbone
                 frozen, then a low-LR pass with the last conv block
                 unfrozen.

── Why this backbone and not MobileNetV1 ───────────────────────────────
The brief offered "a heavily pruned MobileNetV1 or a custom pre-trained
lightweight ConvNet".  This module takes the second option, for three
reasons that all point the same way:

1. DOMAIN.  ImageNet weights are trained on 224x224 natural RGB.  Applied
   to a 28x28 binarised glyph the early filters still do something (they
   are edge detectors) but every layer after them is meaningless, and the
   frozen BatchNorm statistics are calibrated for natural-image
   activations at 8x the resolution.  EMNIST-pretraining gives a backbone
   whose features were learned on the actual data distribution.

2. INPUT SHAPE.  MobileNetV1 is a 3-channel network.  The device produces
   one grayscale plane, so using it would force either a channel
   replication on the ESP32 (extra copy of every frame, and a device-side
   change in a pipeline that is currently bit-exact between Python and C)
   or a replication layer inside the model (same cost, moved).

3. ARENA.  The compact stack is the SAME stack the image project already
   runs on the device, so its TFLM arena is a known, already-shipped
   budget.  Measured on the exported INT8 graphs (peak live activation,
   see measure_arena_estimate below): the compact stack at 28x28 needs a
   fraction of what MobileNetV1-0.25 does, and the ESP32-P4 target is a
   32-64 KB arena.

── Two-phase fine-tuning, and why phase 1 is not optional ──────────────
The head is randomly initialised while the backbone is pretrained.  If the
backbone were trainable from step 1, the first few hundred gradients would
flow back through it and destroy the pretrained features before the head
has learned anything to preserve — classic catastrophic forgetting on a
small dataset.  Freezing the backbone for phase 1 lets the head converge
to something sane, and only then is the last conv block released at a 5-10x
lower LR to adapt the high-level features to the user's camera.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import tensorflow as tf

from dataset_io import PROJECT_TYPE_OCR, app_data_dir
from ocr_dataset import (
    BACKGROUND_CLASS_NAME,
    DEFAULT_EMNIST_SPLIT,
    EMNIST_SPLITS,
    OcrMixConfig,
    build_mixed_dataset,
    emnist_cache_dir,
    ensure_emnist_split,
    load_emnist,
)
from ocr_preprocess import (
    INK_BRIGHT,
    INK_DARK,
    NORM_CANON,
    align_source_polarity,
    normalize_ink_polarity,
    normalize_norm_convention,
    preprocess_glyph,
)
from trainer import (
    DatasetBundle,
    FitContext,
    TrainConfig,
    TrainStrategy,
    glyph_head_layers,
    conv_backbone_kwargs,
    conv_backbone_layers,
    load_datasets,
    make_optimizer,
)

# Bump when the backbone's architecture or its training recipe changes, so
# a cached backbone from an older build is never silently reused.
BACKBONE_CACHE_VERSION = "v3"  # v3: compact GAP head; drop any >500 KB cache


# ════════════════════════════════════════════════════════════════════════
# Small shared helpers
# ════════════════════════════════════════════════════════════════════════
def _project_convention(cfg: TrainConfig) -> Tuple[str, int, List[str]]:
    """(glyph convention, ink polarity, classes with a different convention).

    The convention has to be decided ONCE for the whole run because the
    pretrained backbone is convention-specific: a backbone trained on
    MNIST-canonical glyphs is not the right starting point for a project
    that stretches its glyphs to the full grid.  Mixing conventions inside
    one project is therefore reported rather than silently resolved.
    """
    pre = cfg.class_preprocess or {}
    counts: Dict[str, int] = {}
    polarities: Dict[int, int] = {}
    for name, raw in pre.items():
        if not isinstance(raw, dict):
            continue
        if "glyph_convention" in raw:
            conv = str(normalize_norm_convention(raw.get("glyph_convention")))
            counts[conv] = counts.get(conv, 0) + 1
        if "ink_polarity" in raw:
            pol = int(normalize_ink_polarity(raw.get("ink_polarity")))
            polarities[pol] = polarities.get(pol, 0) + 1
    # Defaults MUST equal record_controller._preprocess_image_png_ocr's
    # defaults (NORM_CANON / INK_DARK): that function built the processed
    # cache this module copies, and a different default here would train on
    # a convention the cache was not produced with.
    convention = max(counts, key=counts.get) if counts else NORM_CANON
    polarity = max(polarities, key=polarities.get) if polarities else INK_DARK
    odd = [str(n) for n, raw in pre.items()
           if isinstance(raw, dict) and "glyph_convention" in raw
           and str(normalize_norm_convention(raw.get("glyph_convention"))) != convention]
    return convention, int(polarity), odd


def _emnist_transform(images: np.ndarray, out_size: int, convention: str,
                      target_polarity: int) -> np.ndarray:
    """EMNIST -> model input, through the SAME integer pipeline as mixing.

    `apply_segmentation=False` and `polarity=INK_BRIGHT` are the two EMNIST
    specifics (already-isolated, bright-on-dark).  Reusing preprocess_glyph
    rather than a shortcut is the point: the backbone pretraining and the
    mixed-dataset injection must produce identical pixels or the backbone
    would be pretrained on a distribution the fine-tune never sees — which
    is why the polarity alignment below is the same call the mixer makes.
    """
    out = np.zeros((images.shape[0], int(out_size), int(out_size)), dtype=np.uint8)
    for i in range(images.shape[0]):
        src_img = align_source_polarity(
            images[i], source_polarity=INK_BRIGHT,
            target_polarity=int(target_polarity),
        )
        res = preprocess_glyph(
            src_img, out_size=int(out_size), polarity=int(target_polarity),
            apply_segmentation=False, convention=str(convention),
        )
        out[i] = np.asarray(res.get("glyph"), dtype=np.uint8)
    return out


def _balanced_emnist_subset(
    images: np.ndarray,
    labels: np.ndarray,
    per_class: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Up to `per_class` samples for every label present, deterministically.

    Balanced on purpose: EMNIST `balanced` is only approximately balanced
    and `byclass` is not balanced at all, and an unbalanced pretraining set
    teaches the backbone that common characters are more likely — a bias
    that then has to be unlearned on a 15-sample-per-class fine-tune set.
    """
    rng = np.random.default_rng(int(seed))
    xs: List[np.ndarray] = []
    ys: List[np.ndarray] = []
    for lab in np.unique(labels):
        idx = np.flatnonzero(labels == lab)
        take = min(int(per_class), int(idx.size))
        if take <= 0:
            continue
        chosen = rng.permutation(idx)[:take]
        xs.append(images[chosen])
        ys.append(np.full(take, int(lab), dtype=np.int64))
    if not xs:
        return np.zeros((0, images.shape[1], images.shape[2]), np.uint8), np.zeros((0,), np.int64)
    return np.concatenate(xs, axis=0), np.concatenate(ys, axis=0)


# ════════════════════════════════════════════════════════════════════════
# Backbone: pretraining + cache
# ════════════════════════════════════════════════════════════════════════
def backbone_signature(cfg: TrainConfig, convention: str, polarity: int) -> str:
    """Cache key for a pretrained backbone.

    Everything that changes the weights goes in, so a cached backbone is
    only reused by a run that would have produced the same one.  Polarity
    belongs here for the same reason convention does: the backbone is
    trained on polarity-aligned EMNIST, so a backbone built before the
    alignment existed must not be silently reused.
    """
    payload = {
        "v": BACKBONE_CACHE_VERSION,
        "split": str(cfg.ocr_emnist_split),
        "img_size": int(cfg.img_size),
        "convention": str(convention),
        "polarity": int(polarity),
        "arch": {k: int(v) for k, v in conv_backbone_kwargs(cfg).items()},
        "per_class": int(cfg.ocr_pretrain_per_class),
        "epochs": int(cfg.ocr_pretrain_epochs),
        "batch": int(cfg.batch_size),
        "seed": int(cfg.seed),
        "color_mode": str(cfg.color_mode),
    }
    blob = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


def ensure_pretrained_backbone(
    app_data_dir: Path,
    cfg: TrainConfig,
    *,
    convention: Optional[str] = None,
    progress: Optional[Callable[[float, str], None]] = None,
    allow_download: bool = True,
    force: bool = False,
) -> Optional[Path]:
    """Train (once) and cache a glyph backbone on EMNIST.  Never raises.

    Returns the path to a saved Keras backbone, or None if it could not be
    produced — in which case the caller trains from scratch on the mixed
    dataset, which still works, just with less to start from.  Turning a
    failed download into a hard error here would make the whole OCR
    project unusable offline for no benefit.
    """
    try:
        conv, polarity, _odd = _project_convention(cfg)
        conv = str(convention or conv)
        cache_dir = Path(app_data_dir) / "ocr_backbone"
        cache_dir.mkdir(parents=True, exist_ok=True)
        sig = backbone_signature(cfg, conv, polarity)
        out = cache_dir / f"backbone_{sig}.keras"
        if out.exists() and not force:
            # A cached backbone from the bloated Flatten-head era can be large
            # (the head is not saved, but the cache is still useful to
            # invalidate explicitly).  Rebuild anything obviously oversized
            # instead of silently reusing a multi-hundred-KB artefact.
            try:
                if int(out.stat().st_size) <= 500 * 1024:
                    if progress is not None:
                        progress(0.03, "Reusing cached glyph backbone")
                    return out
                out.unlink(missing_ok=True)
            except Exception:
                pass

        split = str(cfg.ocr_emnist_split)
        data_root = emnist_cache_dir(app_data_dir)
        if progress is not None:
            progress(0.03, f"Preparing EMNIST {split} for backbone pretraining...")
        ensure_emnist_split(data_root, split, progress=progress, allow_download=allow_download)
        data = load_emnist(data_root, split, usages=("train",))

        if progress is not None:
            progress(0.08, "Balancing EMNIST for pretraining...")
        imgs, labs = _balanced_emnist_subset(
            data.images, data.labels, int(cfg.ocr_pretrain_per_class), int(cfg.seed)
        )
        if imgs.shape[0] < 32:
            raise ValueError(f"EMNIST {split} yielded only {imgs.shape[0]} samples")

        if progress is not None:
            progress(0.12, f"Transforming {imgs.shape[0]} EMNIST glyphs...")
        glyphs = _emnist_transform(imgs, int(cfg.img_size), conv, polarity)
        x = (glyphs.astype(np.float32) / 255.0)[..., None]
        # Remap the sparse EMNIST label ids onto a dense 0..K-1 range so the
        # pretraining head is as small as the data warrants.
        uniq = np.unique(labs)
        remap = {int(v): i for i, v in enumerate(uniq)}
        y = np.array([remap[int(v)] for v in labs], dtype=np.int64)

        backbone = tf.keras.Sequential(conv_backbone_layers(cfg), name="backbone")
        inp = tf.keras.layers.Input(shape=x.shape[1:], name="input")
        t = backbone(inp)
        for layer in glyph_head_layers(len(uniq), cfg):
            t = layer(t)
        model = tf.keras.Model(inp, t, name="emnist_pretrain")
        model.compile(optimizer=make_optimizer(cfg), loss="sparse_categorical_crossentropy",
                      metrics=["accuracy"])

        epochs = max(1, int(cfg.ocr_pretrain_epochs))
        batch = max(8, min(int(cfg.batch_size), 256))

        class _Cb(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                if progress is not None:
                    try:
                        progress(0.12 + 0.06 * float(epoch + 1) / float(epochs),
                                 f"Pretraining backbone epoch {int(epoch + 1)}/{epochs}...")
                    except Exception:
                        pass

        model.fit(x, y, epochs=epochs, batch_size=batch, verbose=0,
                  validation_split=0.05, callbacks=[_Cb()])
        # Save the BACKBONE alone: the transfer model must not inherit the
        # 47-way EMNIST head, and loading a model (not a weight list) means
        # no layer names have to line up across processes.
        backbone.save(str(out))
        return out
    except Exception as e:
        print(f"[ocr_trainer] backbone pretraining unavailable: {e}")
        if progress is not None:
            progress(0.12, "Backbone pretraining unavailable — training from scratch")
        return None


# ════════════════════════════════════════════════════════════════════════
# Head surgery + freezing
# ════════════════════════════════════════════════════════════════════════
def set_trainable(module: Any, flag: bool) -> None:
    for layer in getattr(module, "layers", []):
        layer.trainable = bool(flag)


def unfreeze_last_conv_blocks(backbone: Any, n_blocks: int) -> int:
    """Release the last `n_blocks` Conv2D layers (and everything after them).

    "Last conv block" rather than "last N layers" because the tail of the
    backbone is pooling/dropout: unfreezing those alone would release no
    weights and the fine-tune phase would be a no-op.
    """
    layers = list(getattr(backbone, "layers", []))
    conv_idx = [i for i, l in enumerate(layers) if l.__class__.__name__ == "Conv2D"]
    if not conv_idx:
        return 0
    n = max(0, int(n_blocks))
    if n == 0:
        return 0
    start = conv_idx[-min(n, len(conv_idx))]
    for i, layer in enumerate(layers):
        layer.trainable = i >= start
    return len(layers) - start


def build_transfer_model(
    input_shape: Tuple[int, int, int],
    num_classes: int,
    cfg: TrainConfig,
    backbone: Optional[Any] = None,
) -> tf.keras.Model:
    """Backbone + a NEW `num_classes`-way head.

    `num_classes` is N+1 for the OCR project: the user's characters plus
    the reserved `_background` reject class, which is placed last so its
    output index is exactly N.

    When `backbone` is None the conv stack is built fresh — the same
    architecture, just untrained — so the rest of the pipeline is
    identical whether or not pretraining was available.
    """
    if backbone is None:
        backbone = tf.keras.Sequential(conv_backbone_layers(cfg), name="backbone")
    inp = tf.keras.layers.Input(shape=input_shape, name="input")
    x = backbone(inp, training=False)
    for layer in glyph_head_layers(num_classes, cfg):
        x = layer(x)
    model = tf.keras.Model(inp, x, name="ocr_transfer")
    return model


# ════════════════════════════════════════════════════════════════════════
# Calibration: evenly from EMNIST and captures
# ════════════════════════════════════════════════════════════════════════
def build_balanced_calibration_ds(
    mixed_dir: Path,
    class_names: Sequence[str],
    cfg: TrainConfig,
) -> tf.data.Dataset:
    """Representative dataset that cannot be dominated by one source.

    The mixed directory names the two provenances differently — user
    captures are `<n>.png`, EMNIST injections are `emnist_<n>.png` — so the
    calibration set is drawn by round-robining over (class x source) rather
    than by shuffling a flat file list.  A flat shuffle would weight the
    calibration by whichever half happens to have more files, and the INT8
    activation ranges would then be tuned almost entirely to EMNIST, which
    is exactly the half that is NOT the deployment distribution.
    """
    from PIL import Image

    per_class = max(1, int(cfg.representative_samples))
    buckets: List[List[Path]] = []
    for name in class_names:
        d = Path(mixed_dir) / str(name)
        if not d.exists():
            continue
        files = sorted(p for p in d.glob("*.png") if p.is_file())
        user = [p for p in files if not p.name.startswith("emnist_")]
        synth = [p for p in files if p.name.startswith("emnist_")]
        if user:
            buckets.append(user)
        if synth:
            buckets.append(synth)

    xs: List[np.ndarray] = []
    ys: List[np.ndarray] = []
    index = {str(n): i for i, n in enumerate(class_names)}
    cursor = [0] * len(buckets)
    # Round-robin ONE sample at a time across the (class x source) buckets
    # until the requested count is reached.  Taking a fixed number of
    # passes instead would make the set size depend on how many buckets
    # exist rather than on representative_samples — which is what made an
    # earlier version of this function hand the converter ~16 samples and
    # collapse the int8 model (float32 98%, int8 0% on the same inputs).
    while len(xs) < per_class:
        progressed = False
        for b_i, files in enumerate(buckets):
            if len(xs) >= per_class:
                break
            c = cursor[b_i]
            if c >= len(files):
                continue
            cursor[b_i] = c + 1
            progressed = True
            try:
                g = np.asarray(Image.open(files[c]).convert("L"), dtype=np.uint8)
            except Exception:
                continue
            xs.append((g.astype(np.float32) / 255.0)[..., None])
            ys.append(np.int32(index.get(files[c].parent.name, 0)))
        if not progressed:
            break

    if not xs:
        return None  # type: ignore[return-value]

    x = np.stack(xs).astype(np.float32)
    y = np.asarray(ys, dtype=np.int32)
    return tf.data.Dataset.from_tensor_slices((x, y)).batch(1)


# ════════════════════════════════════════════════════════════════════════
# The strategy
# ════════════════════════════════════════════════════════════════════════
def _infer_classes(dataset_dir: Path) -> List[str]:
    try:
        return sorted(p.name for p in Path(dataset_dir).iterdir()
                      if p.is_dir() and any(p.glob("*.png")))
    except Exception:
        return []


def ocr_adjust_cfg(dataset_dir: Path, cfg: TrainConfig) -> TrainConfig:
    """Batch/epoch sizing for a 28x28 glyph set.

    Deliberately re-derived rather than reusing the image project's
    adjuster: the epoch count is not one number for OCR, it is the sum of
    the two fine-tuning phases, and letting the generic adjuster raise
    `epochs` to "hit ~800 updates" would silently change how long phase 1
    runs.  The batch rule (keep >= min_steps steps per epoch) is the same.
    """
    epochs_total = max(1, int(cfg.ocr_head_epochs)) + max(0, int(cfg.ocr_finetune_epochs))
    batch = max(1, int(cfg.batch_size))
    try:
        total = sum(1 for n in _infer_classes(dataset_dir) for _ in (Path(dataset_dir) / n).glob("*.png"))
    except Exception:
        total = 0
    # The mixed set is ~10x the raw capture count, so size the batch off
    # what training will actually see, not off the cache it was built from.
    estimate = total if total else 1
    min_steps = max(1, int(cfg.min_steps_per_epoch or 4))
    max_batch = max(2, int(estimate // min_steps)) if estimate >= min_steps else 2
    # Calibration wants as much of the corpus as it can get.  An OCR corpus
    # is small (a mixed set is a few hundred images), and post-training
    # quantisation needs the activation ranges to be right: too small a
    # representative set collapses the int8 model while the float32 one
    # still looks perfect.  Asking for more than exists is harmless — the
    # builder round-robins until the buckets are exhausted.
    # A bad split name is a hard download error several layers down
    # (ensure_emnist_split raises), and the value can arrive from a web form
    # or localStorage — neither of which is a trustworthy source.  Coerce it
    # here, where "which splits exist" is already known, so every downstream
    # user of cfg.ocr_emnist_split sees a valid name.
    split = str(getattr(cfg, "ocr_emnist_split", "") or "").strip().lower()
    if split not in EMNIST_SPLITS:
        split = DEFAULT_EMNIST_SPLIT
    return replace(
        cfg,
        ocr_emnist_split=split,
        batch_size=int(max(2, min(batch, max_batch))),
        epochs=int(epochs_total),
        representative_samples=max(int(cfg.representative_samples), 600),
    )


def ocr_prepare(dataset_dir: Path, cfg: TrainConfig) -> DatasetBundle:
    """Mix the training set, then load it through the shared dataset loader."""
    processed = Path(dataset_dir).resolve()
    workspace = processed.parent
    raw_dir = workspace / "tm_dataset"
    mixed_dir = workspace / ".tm_ocr_mixed"

    class_names = [str(c) for c in (cfg.class_names_order or [])]
    if not class_names:
        class_names = [c for c in _infer_classes(processed) if c != BACKGROUND_CLASS_NAME]
    if not class_names:
        raise ValueError("no classes with images — capture some samples first")

    convention, polarity, odd_classes = _project_convention(cfg)

    app_data = _app_data_dir()
    backbone_path: Optional[Path] = None
    backbone = None
    warnings: List[str] = []
    if bool(cfg.ocr_transfer_learning):
        backbone_path = ensure_pretrained_backbone(app_data, cfg, convention=convention)
        if backbone_path is not None:
            try:
                backbone = tf.keras.models.load_model(str(backbone_path))
            except Exception as e:
                warnings.append(f"could not load the pretrained backbone ({e}); training from scratch")

    mix_cfg = OcrMixConfig(
        out_size=int(cfg.img_size),
        emnist_enabled=bool(cfg.ocr_emnist_enabled),
        emnist_split=str(cfg.ocr_emnist_split),
        emnist_per_class=int(cfg.ocr_emnist_per_class),
        background_enabled=bool(cfg.ocr_background_enabled),
        background_samples=int(cfg.ocr_background_samples),
        seed=int(cfg.seed),
        class_preprocess=cfg.class_preprocess,
        default_polarity=INK_DARK,
        default_convention=convention,
    )
    mix = build_mixed_dataset(
        processed_dataset_dir=processed,
        mixed_dir=mixed_dir,
        class_names=class_names,
        cfg=mix_cfg,
        raw_dataset_dir=raw_dir if raw_dir.exists() else None,
        emnist_root=emnist_cache_dir(app_data),
        allow_download=bool(cfg.ocr_emnist_enabled),
    )
    warnings.extend(mix.warnings)

    # Train on the mixed directory, in the order the mixer chose — the
    # background class is LAST, so its softmax index is exactly N.
    load_cfg = replace(cfg, class_names_order=list(mix.class_names))
    train_ds, val_ds, calibration_ds, labels, input_shape, class_weights = load_datasets(mixed_dir, load_cfg)

    balanced = build_balanced_calibration_ds(mixed_dir, labels, cfg)
    if balanced is not None:
        calibration_ds = balanced

    if odd_classes:
        warnings.append(
            "these classes use a different glyph convention than the rest "
            f"({convention}): {', '.join(odd_classes)} — the shared backbone is "
            "tuned for the majority convention"
        )

    meta = mix.to_meta()
    meta["backbone_path"] = str(backbone_path) if backbone_path else None
    meta["backbone"] = backbone
    meta["convention"] = convention
    meta["polarity"] = int(polarity)
    meta["warnings"] = warnings
    meta["has_background_source"] = bool(raw_dir.exists())

    return DatasetBundle(
        train_ds=train_ds,
        val_ds=val_ds,
        calibration_ds=calibration_ds,
        class_names=[str(x) for x in labels],
        input_shape=input_shape,
        class_weights=class_weights,
        train_count=sum(mix.counts.values()),
        total_files=sum(mix.counts.values()),
        meta=meta,
        eval_dir=mixed_dir,
    )


def _app_data_dir() -> Path:
    """The app data directory, resolved without importing app.py.

    app.py imports streamlit at module scope, so it cannot be imported
    from a training worker; the rule itself lives in dataset_io, which
    app.py also delegates to.
    """
    return app_data_dir()


def ocr_build_model(bundle: DatasetBundle, cfg: TrainConfig) -> tf.keras.Model:
    backbone = bundle.meta.get("backbone")
    model = build_transfer_model(bundle.input_shape, len(bundle.class_names), cfg, backbone=backbone)
    # Phase 1 always starts fully frozen; ocr_fit releases blocks later.
    if backbone is not None:
        set_trainable(backbone, False)
    model.compile(optimizer=make_optimizer(cfg, float(cfg.ocr_head_lr)),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  metrics=["accuracy"])
    return model


def ocr_fit(model: tf.keras.Model, bundle: DatasetBundle, cfg: TrainConfig, ctx: FitContext) -> None:
    """Phase 1 (head only) then phase 2 (last conv block, lower LR)."""
    backbone = bundle.meta.get("backbone")
    fit_common: Dict[str, Any] = {"verbose": 1, "class_weight": bundle.class_weights}
    if bundle.val_ds is not None:
        fit_common["validation_data"] = bundle.val_ds

    head_epochs = max(1, int(cfg.ocr_head_epochs))
    ft_epochs = max(0, int(cfg.ocr_finetune_epochs))
    if backbone is None:
        # No pretrained features to protect: a single pass is both correct
        # and faster than warming up a head over a random backbone.
        model.fit(bundle.train_ds, epochs=head_epochs + ft_epochs,
                  callbacks=ctx.make_callbacks(head_epochs + ft_epochs, tag="(from scratch)", lo=0.05, hi=0.80),
                  **fit_common)
        return

    model.fit(bundle.train_ds, epochs=head_epochs,
              callbacks=ctx.make_callbacks(head_epochs, tag="(head)", lo=0.05, hi=0.45),
              **fit_common)

    if ft_epochs > 0 and int(cfg.ocr_unfreeze_blocks) > 0:
        released = unfreeze_last_conv_blocks(backbone, int(cfg.ocr_unfreeze_blocks))
        if released > 0:
            # Keras captures the trainable set at compile time, so the
            # unfreeze is only live after a fresh compile.
            model.compile(optimizer=make_optimizer(cfg, float(cfg.ocr_finetune_lr)),
                          loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                          metrics=["accuracy"])
            model.fit(bundle.train_ds, epochs=ft_epochs,
                      callbacks=ctx.make_callbacks(ft_epochs, tag="(fine-tune)", lo=0.45, hi=0.80),
                      **fit_common)


OCR_STRATEGY = TrainStrategy(
    name=PROJECT_TYPE_OCR,
    adjust_cfg=ocr_adjust_cfg,
    prepare=ocr_prepare,
    build_model=ocr_build_model,
    fit=ocr_fit,
)


# ════════════════════════════════════════════════════════════════════════
# Arena estimation (design-time evidence, not a training dependency)
# ════════════════════════════════════════════════════════════════════════
def measure_arena_estimate(tflite_bytes: bytes) -> Dict[str, int]:
    """Estimate the peak live activation bytes of a TFLite graph.

    TFLM's arena is allocated by its own greedy scheduler, which this does
    not reproduce.  What it does give is the quantity that drives the
    arena — the peak of sum(in-flight tensor bytes) over the execution
    order — which is enough to compare two candidate architectures by a
    wide margin (the point of the measurement) without pretending to be
    an exact TFLM number.

    Tensors are walked in model order and each is considered live from the
    layer that produces it until the last layer that consumes it.
    """
    try:
        interp = tf.lite.Interpreter(model_content=tflite_bytes)
    except Exception:
        return {"peak_bytes": 0, "total_bytes": 0, "largest_bytes": 0}

    details = interp.get_tensor_details()
    try:
        ops = interp._get_ops_details()  # type: ignore[attr-defined]
    except Exception:
        ops = []

    sizes: Dict[int, int] = {}
    for d in details:
        if d.get("index") is None:
            continue
        n = 1
        for dim in d.get("shape", []):
            n *= int(dim)
        sizes[int(d["index"])] = max(1, n) * int(np.dtype(d["dtype"]).itemsize)

    if not ops:
        total = sum(sizes.values())
        return {"peak_bytes": max(sizes.values()) if sizes else 0,
                "total_bytes": total,
                "largest_bytes": max(sizes.values()) if sizes else 0}

    last_use: Dict[int, int] = {}
    first_def: Dict[int, int] = {}
    consts = {int(d["index"]) for d in details if d.get("shape") is not None and int(np.prod(d["shape"])) > 0}
    for step, op in enumerate(ops):
        for t in op.get("inputs", []):
            t = int(t)
            if t >= 0:
                last_use[t] = step
        for t in op.get("outputs", []):
            t = int(t)
            if t >= 0:
                first_def.setdefault(t, step)

    peak = 0
    total = sum(sizes.values())
    for step in range(len(ops)):
        live = 0
        for t, s in sizes.items():
            if t in consts and first_def.get(t, -1) != step:
                continue
            if first_def.get(t, 10 ** 9) <= step <= last_use.get(t, -1):
                live += s
        peak = max(peak, live)
    return {"peak_bytes": int(peak), "total_bytes": int(total),
            "largest_bytes": max(sizes.values()) if sizes else 0}
