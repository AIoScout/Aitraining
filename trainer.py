from __future__ import annotations

import json
import math
import os
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import tensorflow as tf
from tensorflow.lite.python.util import convert_bytes_to_c_source

from image_preprocess import (
    CROP_MODE_AUTO_SEARCH,
    PREPROCESS_MODE_AUTO_BY_LABEL,
    normalize_class_preprocess_map,
    normalize_sample_preprocess_map,
    normalize_manual_roi,
    preprocess_for_label,
)
from dataset_io import PROJECT_TYPE_IMAGE, PROJECT_TYPE_OCR, PROJECT_TYPE_SIGN_OCR, is_glyph_project
from ocr_dataset import BACKGROUND_CLASS_NAME, DEFAULT_EMNIST_SPLIT


@dataclass(frozen=True)
class TrainConfig:
    img_size: int = 96
    # Which project produced this run.  Purely provenance for the trainer —
    # every project feeds it the SAME processed-cache PNGs at
    # img_size x img_size, so the training/export code below is shared
    # unchanged.  It is recorded in the run's train_config.json / metadata
    # so the UI and the .tmproj manifest can route the run back to the right
    # workspace after a reopen.
    project_type: str = PROJECT_TYPE_IMAGE
    color_mode: str = "grayscale"
    batch_size: int = 32
    epochs: int = 20
    validation_split: float = 0.25
    seed: int = 42
    optimizer: str = "adam"
    learning_rate: float = 0.0016
    # Larger defaults (32/64/128 conv, 128 dense) — the old 16/32/64
    # architecture silently collapsed to a single class on small stroke-style
    # datasets (END -> LEFT confusion, val_acc 1.0 while one class scored 30 %
    # on its own training images).
    conv1_filters: int = 32
    conv2_filters: int = 64
    dense_units: int = 128
    representative_samples: int = 200
    preprocess_mode: str = PREPROCESS_MODE_AUTO_BY_LABEL
    crop_mode: str = CROP_MODE_AUTO_SEARCH  # auto shadow-search box = model input (device BG_ENABLE_FOCUS_SEARCH=1)
    manual_roi: Optional[Tuple[float, float, float, float]] = None
    class_preprocess: Optional[Dict[str, Dict[str, Any]]] = None
    sample_preprocess: Optional[Dict[str, Dict[str, Dict[str, Any]]]] = None
    use_preprocessed_dataset: bool = False
    class_names_order: Optional[List[str]] = None
    min_steps_per_epoch: int = 4
    auto_adjust_batch_and_epochs: bool = True

    # Reading direction for multi-glyph sequences (road signs: "L 50", "STOP").
    # "ltr" reads left to right, "rtl" right to left. Only the sign-OCR
    # pipeline consumes it; single-character OCR and image projects read one
    # glyph or one label at a time and ignore it.
    reading_order: str = "ltr"

    # The road sign's label, one character per slot, in the order it reads
    # (['A','1','A'] for a sign printed "A 1 A").  This is the TARGET the
    # preview matches against: the model's classes are the individual
    # CHARACTERS, not the sign, so nothing else in the run records which
    # sign is being built.
    #
    # It rides in TrainConfig (and so in the .tmproj) rather than in a store
    # of its own because it is project state like any other — reopening the
    # file has to restore the sign that was being built.
    #
    # Written and read through ocr_preprocess.normalize_sign_sequence, which
    # guarantees exactly SIGN_SEQUENCE_LEN upper-cased characters, so the
    # pipeline never sees a ragged or oversized sequence.  None and an
    # all-placeholder sequence both mean "not set yet".
    sign_sequence: Optional[List[str]] = None

    # ── OCR / character project ─────────────────────────────────────────
    # Only consulted when project_type == PROJECT_TYPE_OCR; the image
    # project's path never reads them, so their presence cannot perturb an
    # existing sign model.
    #
    # Dataset mixing (ocr_dataset.build_mixed_dataset)
    ocr_emnist_enabled: bool = True
    ocr_emnist_split: str = DEFAULT_EMNIST_SPLIT
    ocr_emnist_per_class: int = 200
    ocr_background_enabled: bool = True
    ocr_background_samples: int = 0          # 0 -> mirror ocr_emnist_per_class
    # Transfer learning (ocr_trainer)
    ocr_transfer_learning: bool = True
    ocr_pretrain_per_class: int = 1200       # EMNIST samples/class for the cached backbone
    ocr_pretrain_epochs: int = 4
    ocr_head_epochs: int = 12                # phase 1: head only, backbone frozen
    ocr_finetune_epochs: int = 25            # phase 2: last conv block unfrozen
    ocr_head_lr: float = 0.0015
    ocr_finetune_lr: float = 0.0002
    ocr_unfreeze_blocks: int = 1


@dataclass(frozen=True)
class TrainResult:
    run_dir: Path
    labels: List[str]
    keras_model_path: Path
    tflite_path: Path
    model_h_path: Path
    model_cpp_path: Path
    metrics: Dict[str, float]
    class_accuracies: Optional[Dict[str, float]] = None
    warnings: Optional[List[str]] = None
    # Whatever the strategy wants recorded about how it built the set — the
    # OCR project puts its EMNIST/background mix manifest here.  Typed as a
    # plain dict so the trainer needs to know nothing about any one project.
    meta: Dict[str, Any] = field(default_factory=dict)


def _channels(color_mode: str) -> int:
    return 1 if color_mode == "grayscale" else 3


def _list_image_files(dataset_dir: Path) -> Dict[str, List[Path]]:
    allowed = {".bmp", ".gif", ".jpeg", ".jpg", ".png"}
    out: Dict[str, List[Path]] = {}
    for class_dir in sorted([p for p in dataset_dir.iterdir() if p.is_dir()]):
        files = [p for p in sorted(class_dir.iterdir()) if p.is_file() and p.suffix.lower() in allowed]
        if files:
            out[class_dir.name] = files
    return out


def load_datasets(
    dataset_dir: Path, cfg: TrainConfig
) -> Tuple[tf.data.Dataset, Optional[tf.data.Dataset], tf.data.Dataset, List[str], Tuple[int, int, int], Dict[int, float]]:
    dataset_dir = dataset_dir.resolve()
    image_map = _list_image_files(dataset_dir)
    default_order = list(image_map.keys())
    if cfg.class_names_order:
        provided = [str(x) for x in cfg.class_names_order if str(x) in image_map]
        missed = [str(x) for x in cfg.class_names_order if str(x) not in image_map]
        extra = [x for x in default_order if x not in provided]
        class_names = list(provided) + list(extra)
        if missed:
            print(f"[trainer] class_names_order entries not found in dataset, skipped: {missed}")
    else:
        class_names = list(default_order)
    total_files = sum(len(files) for files in image_map.values())
    if len(class_names) < 2:
        raise ValueError("Need at least 2 classes with images before training.")
    if total_files < 2:
        raise ValueError("Need at least 2 images before training.")
    can_split_validation = cfg.validation_split > 0 and total_files >= 3
    if can_split_validation:
        val_count = int(total_files * float(cfg.validation_split))
        train_count = total_files - val_count
        can_split_validation = val_count >= 1 and train_count >= 1
    if can_split_validation:
        train_ds = tf.keras.utils.image_dataset_from_directory(
            str(dataset_dir),
            labels="inferred",
            class_names=class_names,
            label_mode="int",
            color_mode=cfg.color_mode,
            batch_size=cfg.batch_size,
            image_size=(cfg.img_size, cfg.img_size),
            shuffle=True,
            seed=cfg.seed,
            validation_split=cfg.validation_split,
            subset="training",
        )
        val_ds = tf.keras.utils.image_dataset_from_directory(
            str(dataset_dir),
            labels="inferred",
            class_names=class_names,
            label_mode="int",
            color_mode=cfg.color_mode,
            batch_size=cfg.batch_size,
            image_size=(cfg.img_size, cfg.img_size),
            shuffle=True,
            seed=cfg.seed,
            validation_split=cfg.validation_split,
            subset="validation",
        )
    else:
        train_ds = tf.keras.utils.image_dataset_from_directory(
            str(dataset_dir),
            labels="inferred",
            class_names=class_names,
            label_mode="int",
            color_mode=cfg.color_mode,
            batch_size=cfg.batch_size,
            image_size=(cfg.img_size, cfg.img_size),
            shuffle=True,
            seed=cfg.seed,
        )
        val_ds = None
    input_shape = (cfg.img_size, cfg.img_size, _channels(cfg.color_mode))
    class_name_list = [str(name) for name in class_names]
    preprocess_mode = str(cfg.preprocess_mode or PREPROCESS_MODE_AUTO_BY_LABEL)
    manual_roi = normalize_manual_roi(cfg.manual_roi)
    class_preprocess = normalize_class_preprocess_map(cfg.class_preprocess)
    sample_preprocess = normalize_sample_preprocess_map(cfg.sample_preprocess)
    _ = sample_preprocess
    use_preprocessed_dataset = bool(cfg.use_preprocessed_dataset)

    channels = _channels(cfg.color_mode)
    pad = max(2, int(cfg.img_size) // 12)
    padded_size = int(cfg.img_size) + pad * 2
    augmenter = tf.keras.Sequential(
        [
            tf.keras.layers.RandomTranslation(
                height_factor=0.08,
                width_factor=0.08,
                fill_mode="reflect",
            ),
            tf.keras.layers.RandomZoom(
                height_factor=(-0.08, 0.04),
                width_factor=(-0.08, 0.04),
                fill_mode="reflect",
            ),
            tf.keras.layers.RandomRotation(
                factor=0.035,
                fill_mode="reflect",
            ),
        ],
        name="train_augmenter",
    )

    def focus_tensor_batch(x, y):
        def _focus_one(img: np.ndarray, label: Any):
            label_idx = int(label)
            label_name = class_name_list[label_idx] if 0 <= label_idx < len(class_name_list) else ""
            return preprocess_for_label(
                img,
                out_size=int(cfg.img_size),
                color_mode=str(cfg.color_mode),
                label_name=label_name,
                preprocess_mode=preprocess_mode,
                manual_roi=manual_roi,
                class_preprocess=class_preprocess,
                crop_mode=str(cfg.crop_mode or CROP_MODE_AUTO_SEARCH),
            )

        x = tf.map_fn(
            lambda elems: tf.numpy_function(_focus_one, [elems[0], elems[1]], Tout=tf.float32),
            (x, y),
            fn_output_signature=tf.TensorSpec((int(cfg.img_size), int(cfg.img_size), channels), tf.float32),
        )
        return x, y

    def augment(x, y):
        x, y = focus_tensor_batch(x, y)
        x = tf.image.resize_with_crop_or_pad(x, padded_size, padded_size)
        x = tf.image.random_crop(x, size=[tf.shape(x)[0], int(cfg.img_size), int(cfg.img_size), channels])
        x = augmenter(x, training=True)
        x = tf.image.random_brightness(x, max_delta=0.08)
        x = tf.image.random_contrast(x, lower=0.85, upper=1.15)
        noise = tf.random.normal(tf.shape(x), mean=0.0, stddev=0.02, dtype=tf.float32)
        x = tf.clip_by_value(x + noise, 0.0, 1.0)
        return x, y

    def normalize_only(x, y):
        x = tf.clip_by_value(tf.cast(x, tf.float32) / 255.0, 0.0, 1.0)
        return x, y

    def augment_preprocessed(x, y):
        x, y = normalize_only(x, y)
        x = augmenter(x, training=True)
        x = tf.image.random_brightness(x, max_delta=0.08)
        x = tf.image.random_contrast(x, lower=0.85, upper=1.15)
        noise = tf.random.normal(tf.shape(x), mean=0.0, stddev=0.02, dtype=tf.float32)
        x = tf.clip_by_value(x + noise, 0.0, 1.0)
        return x, y

    total_files_f = float(total_files)
    num_classes_f = float(max(1, len(class_names)))
    class_weights: Dict[int, float] = {}
    for idx, name in enumerate(class_names):
        count = max(1, len(image_map.get(name, [])))
        class_weights[int(idx)] = total_files_f / (num_classes_f * float(count))

    if use_preprocessed_dataset:
        calibration_ds = train_ds.map(normalize_only, num_parallel_calls=tf.data.AUTOTUNE).cache().prefetch(tf.data.AUTOTUNE)
        train_ds = train_ds.map(augment_preprocessed, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)
        if val_ds is not None:
            val_ds = val_ds.map(normalize_only, num_parallel_calls=tf.data.AUTOTUNE).cache().prefetch(tf.data.AUTOTUNE)
    else:
        calibration_ds = train_ds.map(focus_tensor_batch, num_parallel_calls=tf.data.AUTOTUNE).cache().prefetch(tf.data.AUTOTUNE)
        train_ds = train_ds.map(augment, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)
        if val_ds is not None:
            val_ds = val_ds.map(focus_tensor_batch, num_parallel_calls=tf.data.AUTOTUNE).cache().prefetch(tf.data.AUTOTUNE)
    return train_ds, val_ds, calibration_ds, class_names, input_shape, class_weights


def conv_backbone_kwargs(cfg: TrainConfig) -> Dict[str, int]:
    """Filter widths of the shared conv stack.

    Conv3 is DERIVED from conv2 (`max(conv2*2, 64)`), not configured
    separately.  That derivation lived inline in build_model before; it is
    hoisted here because the OCR path has to reproduce the exact same stack
    to be able to load a pretrained version of it.
    """
    return {
        "conv1": cfg.conv1_filters,
        "conv2": cfg.conv2_filters,
        "conv3": max(int(cfg.conv2_filters) * 2, 64),
        "dense": max(int(cfg.dense_units), 64),
    }


def conv_backbone_layers(cfg: TrainConfig) -> List[Any]:
    """The conv feature extractor as FRESH layer instances, no Input, no head.

    One definition, two consumers:
      * build_model()             — stacks it with a head for the image
                                    project's from-scratch classifier;
      * ocr_trainer.build_transfer_model() — uses it as the backbone it
                                    pretrains on EMNIST, freezes, and
                                    grafts a new N+1-way head onto.
    Because both projects share the stack, they also share its arena
    profile on the device — a second architecture would need its own
    memory budget and its own INT8 conversion verification.

    Layers are constructed per call on purpose: a Keras layer instance owns
    weights and cannot belong to two models.
    """
    reg = tf.keras.regularizers.l2(1e-4)
    f = conv_backbone_kwargs(cfg)
    return [
        tf.keras.layers.Conv2D(f["conv1"], (3, 3), activation="relu", padding="same", kernel_regularizer=reg),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.10),
        tf.keras.layers.Conv2D(f["conv2"], (3, 3), activation="relu", padding="same", kernel_regularizer=reg),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.15),
        tf.keras.layers.Conv2D(f["conv3"], (3, 3), activation="relu", padding="same", kernel_regularizer=reg),
        tf.keras.layers.MaxPooling2D((2, 2)),
    ]


def classifier_head_layers(num_classes: int, cfg: TrainConfig) -> List[Any]:
    """Flatten + Dense + Dropout + softmax, as fresh layer instances.

    Shared by the image project's model and — after surgery replaces the
    pretrained EMNIST head — by the OCR model.  The OCR head has
    `num_classes = N + 1`: the user's characters plus the reserved
    background/_background reject class, which is why its index is exactly N.
    """
    reg = tf.keras.regularizers.l2(1e-4)
    dense_units = max(int(cfg.dense_units), 64)
    return [
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(dense_units, activation="relu", kernel_regularizer=reg),
        tf.keras.layers.Dropout(0.35),
        tf.keras.layers.Dense(num_classes, activation="softmax", name="output"),
    ]


def make_optimizer(cfg: TrainConfig, learning_rate: Optional[float] = None) -> tf.keras.optimizers.Optimizer:
    """Build the configured optimizer, optionally at an overridden LR.

    The override matters for transfer learning: head warm-up and
    backbone fine-tuning need the SAME optimizer family at two different
    learning rates, so the lookup lives here instead of being re-derived.
    """
    lr = float(cfg.learning_rate if learning_rate is None else learning_rate)
    opt_name = str(cfg.optimizer).lower()
    if opt_name == "adam":
        return tf.keras.optimizers.Adam(learning_rate=lr)
    if opt_name == "sgd":
        return tf.keras.optimizers.SGD(learning_rate=lr, momentum=0.9)
    if opt_name == "rmsprop":
        return tf.keras.optimizers.RMSprop(learning_rate=lr)
    raise ValueError(f"Unsupported optimizer: {cfg.optimizer}")


def build_model(input_shape: Tuple[int, int, int], num_classes: int, cfg: TrainConfig) -> tf.keras.Model:
    model = tf.keras.Sequential(
        [tf.keras.layers.Input(shape=input_shape, name="input")]
        + conv_backbone_layers(cfg)
        + classifier_head_layers(num_classes, cfg)
    )
    loss = tf.keras.losses.SparseCategoricalCrossentropy()
    model.compile(optimizer=make_optimizer(cfg), loss=loss, metrics=["accuracy"])
    return model


def _representative_data_gen(train_ds: tf.data.Dataset, cfg: TrainConfig):
    n = max(1, int(cfg.representative_samples))
    seen = 0
    for batch_x, _ in train_ds.unbatch().batch(1).take(n):
        x = tf.convert_to_tensor(batch_x, dtype=tf.float32)
        if x.shape.rank != 4:
            x = tf.expand_dims(x, 0)
        yield [x]
        seen += 1
        if seen >= n:
            break


def convert_to_int8_tflite(model: tf.keras.Model, train_ds: tf.data.Dataset, cfg: TrainConfig) -> bytes:
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = lambda: _representative_data_gen(train_ds, cfg)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    try:
        converter.experimental_new_converter = True
    except Exception:
        pass
    try:
        converter.experimental_enable_resource_variables = True
    except Exception:
        pass
    try:
        converter._experimental_calibrate_only = False
    except Exception:
        pass
    try:
        converter._experimental_enable_composite_direct_lowering = True
    except Exception:
        pass
    return converter.convert()


def _keras_predict_fn(model: Any) -> Callable[[np.ndarray], np.ndarray]:
    def fn(batch01: np.ndarray) -> np.ndarray:
        return model.predict(batch01, verbose=0)

    return fn


def _tflite_predict_fn(tflite_bytes: bytes) -> Callable[[np.ndarray], np.ndarray]:
    import tensorflow as tf

    interp = tf.lite.Interpreter(model_content=tflite_bytes)
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]
    qs, qz = inp["quantization"]
    os_, oz = out["quantization"]

    def fn(batch01: np.ndarray) -> np.ndarray:
        outs = []
        for arr01 in batch01:
            arr2d = arr01[:, :, 0] if arr01.ndim == 3 else arr01
            x = np.clip(np.round(arr2d / qs + qz), -128, 127).astype(np.int8)[None, :, :, None]
            interp.set_tensor(inp["index"], x)
            interp.invoke()
            raw = interp.get_tensor(out["index"]).reshape(-1)
            scores = (raw.astype(np.float32) - oz) * os_
            if abs(float(scores.sum()) - 1.0) > 0.1:
                scores = scores - scores.max()
                e = np.exp(scores)
                scores = e / e.sum()
            outs.append(scores)
        return np.stack(outs)

    return fn


def evaluate_class_accuracies(
    predict_fn: Callable[[np.ndarray], np.ndarray],
    dataset_dir: Path,
    labels: List[str],
) -> Dict[str, float]:
    """Per-class accuracy on the processed dataset (normalize-only path).

    This is the training self-check: a model that cannot even classify its
    own training images has collapsed (single-class output) and must never
    ship silently — callers surface the weak classes to the user.
    """
    from PIL import Image

    accs: Dict[str, float] = {}
    for label in labels:
        cls_dir = Path(dataset_dir) / label
        files = sorted([p for p in cls_dir.glob("*.png") if p.is_file()])
        if not files:
            continue
        imgs = []
        for p in files:
            g = np.asarray(Image.open(p).convert("L"), dtype=np.uint8)
            imgs.append(g.astype(np.float32) / 255.0)
        x = np.stack(imgs)[..., None]
        probs = predict_fn(x)
        truth = labels.index(label)
        accs[label] = float((np.argmax(probs, axis=1) == truth).mean())
    return accs


def export_tflite_c_sources(tflite_model: bytes, array_name: str) -> Tuple[str, str]:
    array_name = array_name.strip() or "g_model"
    header_guard = f"{array_name.upper()}_H"
    source_code, header_code = convert_bytes_to_c_source(
        tflite_model,
        array_name=array_name,
        include_guard=header_guard,
        use_tensorflow_license=False,
    )
    return source_code, header_code


def _clamp_batch_for_steps(train_count: int, user_batch: int, min_steps: int = 4) -> int:
    """Largest batch that still leaves at least min_steps steps per epoch."""
    min_steps = max(1, int(min_steps))
    user_batch = max(1, int(user_batch))
    if train_count >= min_steps:
        max_batch_for_steps = max(2, int(train_count // min_steps))
    else:
        max_batch_for_steps = max(2, int(train_count))
    return max(2, min(user_batch, max_batch_for_steps))


def _adjust_cfg_for_dataset_size(dataset_dir: Path, cfg: TrainConfig) -> Tuple[TrainConfig, int, int]:
    from dataclasses import replace

    image_map = _list_image_files(Path(dataset_dir).resolve())
    total_files = int(sum(len(files) for files in image_map.values()))
    if total_files < 2:
        return cfg, total_files, total_files
    val_split = float(cfg.validation_split) if float(cfg.validation_split) > 0 else 0.0
    if val_split > 0 and total_files >= 3:
        val_count = max(1, int(total_files * val_split))
        train_count = max(1, total_files - val_count)
    else:
        train_count = total_files
    if not bool(cfg.auto_adjust_batch_and_epochs):
        return cfg, total_files, train_count

    min_steps = max(1, int(cfg.min_steps_per_epoch or 1))
    user_batch = max(1, int(cfg.batch_size))
    new_batch = _clamp_batch_for_steps(train_count, user_batch, min_steps)
    steps_per_epoch = max(1, int(train_count // new_batch))

    user_epochs = max(1, int(cfg.epochs))
    target_total_updates = 800
    min_total_updates = 300
    suggested_epochs = user_epochs
    if steps_per_epoch >= 1:
        updates_at_user = steps_per_epoch * user_epochs
        if updates_at_user < min_total_updates:
            suggested_epochs = max(user_epochs, int(math.ceil(float(target_total_updates) / float(steps_per_epoch))))
    suggested_epochs = max(user_epochs, min(suggested_epochs, 200))

    if new_batch == user_batch and suggested_epochs == user_epochs:
        return cfg, total_files, train_count

    try:
        new_cfg = replace(cfg, batch_size=int(new_batch), epochs=int(suggested_epochs))
    except Exception:
        return cfg, total_files, train_count
    print(
        f"[trainer] auto-adjust: total={total_files} train={train_count} "
        f"batch {user_batch}->{new_batch}, epochs {user_epochs}->{suggested_epochs} "
        f"(target ~{steps_per_epoch * suggested_epochs} updates)"
    )
    return new_cfg, total_files, train_count


def recommend_train_params(total_samples: int, cfg: TrainConfig) -> Tuple[Dict[str, Any], List[str]]:
    """Compute advisory recommended hyperparameters from the dataset size.

    Returns ``(recommended_kwargs, reasons)`` where ``recommended_kwargs`` keys
    match TrainConfig field names and ``reasons`` holds one human-readable line
    per key, in the same order (batch_size, epochs, learning_rate,
    validation_split). Values are advisory only — the caller decides whether to
    apply them. img_size / conv filters / dense units / optimizer are left to
    the user (no dataset-driven recommendation); img_size should match the
    preprocessing output size (96) rather than the dataset.

    Complementary to ``_adjust_cfg_for_dataset_size``, which stays the single
    authoritative enforcement point inside ``train_and_export``: this mirrors
    its numeric spirit as visible UI hints, but applies different rounding and
    clamps (round-based train count, epochs clamped to [30, 300]) and is not
    gated by ``auto_adjust_batch_and_epochs``.
    """
    total = max(1, int(total_samples))
    val_split = float(cfg.validation_split) if float(cfg.validation_split) > 0 else 0.0
    train_count = max(1, int(round(float(total) * (1.0 - val_split))))

    user_batch = max(1, int(cfg.batch_size))
    new_batch = max(2, min(user_batch, max(2, int(train_count // 4))))
    steps_per_epoch = max(1, int(math.ceil(float(train_count) / float(new_batch))))
    suggested_epochs = int(min(max(int(math.ceil(800.0 / float(steps_per_epoch))), 30), 300))
    suggested_lr = 0.0008 if total < 100 else 0.001
    suggested_val_split = 0.15 if total < 60 else 0.2

    recommended_kwargs: Dict[str, Any] = {
        "batch_size": new_batch,
        "epochs": suggested_epochs,
        "learning_rate": suggested_lr,
        "validation_split": suggested_val_split,
        # Strong capacity: the old 16/32/64 + early-stopping regime silently
        # collapsed to a single class on small stroke-style datasets.
        "conv1_filters": 32,
        "conv2_filters": 64,
        "dense_units": 128,
    }

    reasons: List[str] = []
    if new_batch != user_batch:
        reasons.append(
            f"only {train_count} training samples — batch {new_batch} gives "
            f"{steps_per_epoch} step(s)/epoch; recommend ≥4 steps/epoch"
        )
    else:
        reasons.append(
            f"batch {new_batch} gives {steps_per_epoch} step(s)/epoch on {train_count} training samples"
        )
    reasons.append(
        f"{steps_per_epoch} step(s)/epoch × {suggested_epochs} epochs = "
        f"{steps_per_epoch * suggested_epochs} updates; recommend ~800 total updates"
    )
    if abs(float(cfg.learning_rate) - suggested_lr) > 1e-9:
        if total < 100:
            reasons.append(f"small dataset ({total} samples) — lower lr {suggested_lr} reduces oscillation")
        else:
            reasons.append(f"dataset of {total} samples — lr {suggested_lr} is a solid starting point")
    else:
        reasons.append(f"learning rate {suggested_lr} already matches recommendation")
    if abs(float(cfg.validation_split) - suggested_val_split) > 1e-9:
        if total < 60:
            reasons.append(f"only {total} samples — val split {suggested_val_split} keeps more for training")
        else:
            reasons.append(
                f"val split {suggested_val_split} leaves ~{int(round(float(total) * (1.0 - suggested_val_split)))} "
                f"training samples from {total}"
            )
    else:
        reasons.append(f"val split {suggested_val_split} already matches recommendation")

    return recommended_kwargs, reasons


@dataclass
class DatasetBundle:
    """Everything the training path needs from a prepared dataset.

    A project strategy returns this instead of the old 6-tuple so a strategy
    can carry extra provenance (`meta`) — e.g. the OCR path records how many
    samples came from EMNIST vs the user's captures — without widening a
    positional return value that every caller would then have to unpack.
    """

    train_ds: Any
    val_ds: Any
    calibration_ds: Any
    class_names: List[str]
    input_shape: Tuple[int, int, int]
    class_weights: Dict[int, float]
    train_count: int = 0
    total_files: int = 0
    meta: Dict[str, Any] = field(default_factory=dict)
    # Where the post-training per-class self-check reads its images from.
    # Defaults to `dataset_dir`, which is right whenever training reads the
    # same directory it was handed.  The OCR project trains on a MIXED
    # directory it materialised itself, and the self-check must grade
    # against that one — grading a 28x28 model on 96x96 raw captures would
    # report every class at 0 %.
    eval_dir: Optional[Path] = None


@dataclass
class FitContext:
    """What a strategy's fit function is allowed to use.

    `make_callbacks` is a FACTORY rather than a list because the OCR path
    fits in two phases and each phase needs fresh EarlyStopping /
    ReduceLROnPlateau state — reusing a callback instance across phases
    would carry the first phase's best-weights decision into the second.
    """

    progress: Optional[Callable[[float, str], None]]
    make_callbacks: Callable[..., List[Any]]
    train_ds: Any = None
    val_ds: Any = None
    class_weights: Optional[Dict[int, float]] = None


@dataclass(frozen=True)
class TrainStrategy:
    """One project type's dataset + model + fit policy.

    `train_and_export` below is the shared spine: adjust → prepare → build →
    fit → evaluate → quantise → export → self-check.  Nothing about the
    export, quantisation, per-class self-check or metadata writing is
    project-specific, so a project swaps in a strategy and inherits all of
    it.  This is the seam that keeps the OCR project from becoming a
    parallel copy of the training pipeline.
    """

    name: str
    adjust_cfg: Callable[[Path, TrainConfig], TrainConfig]
    prepare: Callable[[Path, TrainConfig], DatasetBundle]
    build_model: Callable[[DatasetBundle, TrainConfig], tf.keras.Model]
    fit: Callable[[Any, DatasetBundle, TrainConfig, FitContext], None]


def _default_adjust_cfg(dataset_dir: Path, cfg: TrainConfig) -> TrainConfig:
    return _adjust_cfg_for_dataset_size(dataset_dir, cfg)[0]


def _default_prepare(dataset_dir: Path, cfg: TrainConfig) -> DatasetBundle:
    train_ds, val_ds, calibration_ds, class_names, input_shape, class_weights = load_datasets(
        Path(dataset_dir), cfg
    )
    _, total_files, train_count = _adjust_cfg_for_dataset_size(Path(dataset_dir), cfg)
    return DatasetBundle(
        train_ds=train_ds,
        val_ds=val_ds,
        calibration_ds=calibration_ds,
        class_names=class_names,
        input_shape=input_shape,
        class_weights=class_weights,
        train_count=int(train_count),
        total_files=int(total_files),
    )


def _default_build_model(bundle: DatasetBundle, cfg: TrainConfig) -> tf.keras.Model:
    return build_model(bundle.input_shape, len(bundle.class_names), cfg)


def _default_fit(model: Any, bundle: DatasetBundle, cfg: TrainConfig, ctx: FitContext) -> None:
    epochs = int(cfg.epochs)
    fit_kwargs: Dict[str, Any] = {
        "epochs": epochs,
        "verbose": 1,
        "callbacks": ctx.make_callbacks(epochs),
    }
    if bundle.val_ds is not None:
        fit_kwargs["validation_data"] = bundle.val_ds
    fit_kwargs["class_weight"] = bundle.class_weights
    model.fit(bundle.train_ds, **fit_kwargs)


DEFAULT_STRATEGY = TrainStrategy(
    name=PROJECT_TYPE_IMAGE,
    adjust_cfg=_default_adjust_cfg,
    prepare=_default_prepare,
    build_model=_default_build_model,
    fit=_default_fit,
)


def get_train_strategy(project_type: object) -> TrainStrategy:
    """The strategy for a project type; unknown types get the image one."""
    # Membership of the glyph family, not equality with one type: sign_ocr
    # trains the SAME per-character model OCR does (its classes are single
    # characters, its samples are 28x28 glyphs, its EMNIST mix is the same
    # one) — only the preview reads a whole sign at a time.  Naming a single
    # type here is what would make the third glyph project silently train
    # with the 96x96 image strategy.
    if is_glyph_project(project_type):
        # Imported lazily: ocr_trainer imports THIS module for the shared
        # conv stack and head, so a module-level import would be a cycle.
        from ocr_trainer import OCR_STRATEGY

        return OCR_STRATEGY
    return DEFAULT_STRATEGY


def train_and_export(
    dataset_dir: Path,
    run_dir: Path,
    cfg: TrainConfig,
    model_base_name: str = "model",
    array_name: str = "g_model",
    progress: Optional[Callable[[float, str], None]] = None,
) -> TrainResult:
    run_dir = run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    dataset_dir = Path(dataset_dir)
    strategy = get_train_strategy(cfg.project_type)
    cfg = strategy.adjust_cfg(dataset_dir, cfg)

    if progress is not None:
        progress(0.02, "Building datasets...")

    bundle = strategy.prepare(dataset_dir, cfg)
    labels = bundle.class_names
    model = strategy.build_model(bundle, cfg)

    def make_callbacks(epochs_total: int, tag: str = "", lo: float = 0.05, hi: float = 0.80):
        """Fresh callbacks for one fit phase.

        The patience heuristic keys off steps-per-epoch, so it is computed
        here rather than baked in: a short phase needs a longer patience or
        it stops before the head has moved.
        """
        callbacks: List[Any] = []
        if progress is not None:
            total = max(1, int(epochs_total))
            label = f" {tag}" if tag else ""

            class _Progress(tf.keras.callbacks.Callback):
                def on_epoch_end(self, epoch, logs=None):
                    try:
                        p = lo + (hi - lo) * (float(epoch + 1) / float(total))
                        progress(p, f"Training{label} epoch {int(epoch + 1)}/{total}...")
                    except Exception:
                        pass

            callbacks.append(_Progress())

        monitor_loss = "val_loss" if bundle.val_ds is not None else "loss"
        steps_per_epoch_est = max(1, int(bundle.train_count // max(1, int(cfg.batch_size))))
        if steps_per_epoch_est <= 4:
            es_patience, rlr_patience = 12, 6
        elif steps_per_epoch_est <= 8:
            es_patience, rlr_patience = 9, 4
        else:
            es_patience, rlr_patience = 6, 3
        callbacks.append(
            tf.keras.callbacks.EarlyStopping(
                monitor=monitor_loss,
                patience=int(es_patience),
                min_delta=1e-4,
                restore_best_weights=True,
                verbose=0,
            )
        )
        callbacks.append(
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor=monitor_loss,
                factor=0.5,
                patience=int(rlr_patience),
                min_lr=1e-5,
                verbose=0,
            )
        )
        return callbacks

    ctx = FitContext(
        progress=progress,
        make_callbacks=make_callbacks,
        train_ds=bundle.train_ds,
        val_ds=bundle.val_ds,
        class_weights=bundle.class_weights,
    )
    strategy.fit(model, bundle, cfg, ctx)

    eval_ds = bundle.val_ds if bundle.val_ds is not None else bundle.train_ds
    if progress is not None:
        progress(0.85, "Evaluating...")
    loss, acc = model.evaluate(eval_ds, verbose=0)

    keras_model_path = run_dir / f"{model_base_name}.keras"
    model.save(str(keras_model_path))

    if progress is not None:
        progress(0.9, "Converting to int8 TFLite...")
    tflite_bytes = convert_to_int8_tflite(model, bundle.calibration_ds, cfg)
    tflite_path = run_dir / f"{model_base_name}.tflite"
    tflite_path.write_bytes(tflite_bytes)

    # ── Training self-check: per-class accuracy on the training dataset ──
    # Catches the silent single-class collapse (val_acc 1.0 while one class
    # scores ~30 % on its own training images).  Compares float32 vs int8
    # so quantization degradation is surfaced too.
    warnings_out: List[str] = [str(w) for w in (bundle.meta.get("warnings") or [])]
    accs_f32: Dict[str, float] = {}
    self_check_dir = Path(bundle.eval_dir) if bundle.eval_dir is not None else Path(dataset_dir)
    try:
        accs_f32 = evaluate_class_accuracies(_keras_predict_fn(model), self_check_dir, labels)
        accs_i8 = evaluate_class_accuracies(_tflite_predict_fn(tflite_bytes), self_check_dir, labels)
        for label in labels:
            a32 = accs_f32.get(label)
            a8 = accs_i8.get(label)
            if a32 is not None and a32 < 0.80:
                warnings_out.append(
                    f"class '{label}' self-check accuracy {a32:.0%} (< 80%) — add more/varied samples for this class"
                )
            if a32 is not None and a8 is not None and a8 < a32 - 0.10:
                warnings_out.append(
                    f"class '{label}' int8 accuracy {a8:.0%} vs float32 {a32:.0%} — quantization degradation"
                )
    except Exception as e:
        warnings_out.append(f"self-check failed: {e}")
    if warnings_out:
        print("[trainer] SELF-CHECK WARNINGS:")
        for w in warnings_out:
            print("  -", w)

    if progress is not None:
        progress(0.96, "Exporting sources...")
    source_code, header_code = export_tflite_c_sources(tflite_bytes, array_name=array_name)
    model_h_path = run_dir / "model.h"
    model_cpp_path = run_dir / "model.cpp"
    model_h_path.write_text(header_code, encoding="utf-8")
    model_cpp_path.write_text('#include "model.h"\n\n' + source_code, encoding="utf-8")

    (run_dir / "labels.json").write_text(json.dumps({"labels": labels}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (run_dir / "labels.txt").write_text("\n".join(labels) + "\n", encoding="utf-8")
    (run_dir / "train_config.json").write_text(
        json.dumps(asdict(cfg), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if progress is not None:
        progress(1.0, "Done.")

    return TrainResult(
        run_dir=run_dir,
        labels=labels,
        keras_model_path=keras_model_path,
        tflite_path=tflite_path,
        model_h_path=model_h_path,
        model_cpp_path=model_cpp_path,
        metrics={"val_loss": float(loss), "val_accuracy": float(acc)},
        class_accuracies=accs_f32,
        warnings=warnings_out,
        meta=_json_safe_meta(bundle.meta),
    )


def _json_safe_meta(meta: Dict[str, Any]) -> Dict[str, Any]:
    """Strip non-serialisable entries (the OCR strategy carries a live Keras
    backbone in its meta) so the run metadata always writes cleanly."""
    out: Dict[str, Any] = {}
    for k, v in (meta or {}).items():
        try:
            json.dumps(v)
        except (TypeError, ValueError):
            continue
        out[str(k)] = v
    return out


def new_run_dir(base_dir: Path) -> Path:
    ts = time.strftime("%Y%m%d_%H%M%S")
    rand = os.urandom(3).hex()
    return base_dir / f"run_{ts}_{rand}"
