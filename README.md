# AItraining — TFLite Training desktop app

A Streamlit desktop app that trains int8 TFLite models for ESP32-P4, with a
hand-written workspace SPA for capture, preprocessing, training and live
Preview.  Three project types share one UI but train and export different
model shapes.

Packaging instructions (macOS `.dmg` / Windows `.exe`) are at the bottom.

---

## 1. Project types

The Start Page (`✨ Choose a project type`) has three cards:

| Card | Subtitle | Buttons | Model | Export folder |
|---|---|---|---|---|
| **Image Project** (`image`) | *Build an image classification model from images, webcam, or ESP32-P4 device stream. Great for classroom demos and local training for tinyML MCU deployment.* | `Get Started`, `Open Project (.tmproj)`, `Import classified class` | 96×96 default | `TFLite/` |
| **🔠 OCR / Character Project** (`ocr`) | *Train a highly optimized INT8 character recognition model for ESP32. Features EMNIST dataset mixing, bit-exact Vertical Projection, and OOD rejection.* | `Get Started`, `Open Project (.tmproj)` | 28×28 glyph, `0–9 A–Z` + `_background` | `TFLite_OCR/` |
| **🚗 Car License Project** (`sign_ocr`) | *Train a 3-character car-license reader. The same INT8 character model as the OCR project, plus a target sequence, automatic 3-way glyph splitting and a MATCH / NOT MATCH preview against the plate in view.* | `Get Started`, `Open Project (.tmproj)`, `Import classified class` | 3 slots × 28×28 glyphs | `TFLite_Sign/` |

`Open Project (.tmproj)` opens a native file picker.  The archive manifest's
`project_type` is authoritative: a file opened from any card is restored into
the workspace it actually belongs to, and a cross-card switch is announced.
Cancel is a silent no-op; a foreign ZIP is refused at the picker, while older
v1 archives are still readable (missing settings fall back to this build's
defaults and the app reports which ones).

`Get Started` creates a fresh project.  A fresh sign project starts with the
three-slot target `A ▸ B ▸ C` and three matching class cards.

---

## 2. Workspace

Top toolbar:

- **File ▾** — `New project` (⌘N), `Open project` (⌘O), `Save` (⌘S),
  `Save as` (⇧⌘S), `Export dataset`.
- **Edit ▾** — `Edit characters` / `Glyph settings…` (OCR only), `Reset
  project`.
- **Settings** — global webcam/serial input settings, Preview source and OOD
  gates, and the Advanced EMNIST auto-fill controls.
- **Preferences** — display limits and caching.
- **📖 Tutorial** — state-aware walkthrough.  Image, OCR and sign projects
  each have their own curriculum; the sign tour has 10 steps covering target,
  samples, worktable, reading direction, OOD, verdict and export.
- **← Return Home**.
- **Export Model** in the Preview card head.

The workspace is three columns: class cards (`#classes`), `Training`, and
`Preview`.  System dark mode follows `prefers-color-scheme: dark`.

Global shortcuts:

| Key | Action |
|---|---|
| `Esc` | Close the topmost layer (dropdown → worktable → settings → preferences → character/glyph modal → source drawer).  In a text field outside a modal it only blurs. |
| `Space` | With a Webcam/Device drawer open: tap = one capture; hold ≥220 ms = hold capture until release. |
| `⌘/Ctrl+S` | Save; in the worktable, save the current sample |
| `⇧⌘/Ctrl+S` | Save As |
| `⌘/Ctrl+O`, `⌘/Ctrl+N`, `⌘/Ctrl+E` | Open, New, Export |
| `⌘/Ctrl+Enter` | Train (when enabled) |
| `⌘/Ctrl+,` | Settings |

---

## 3. Data collection

Each class card shows its name, sample count (`N Image Sample(s)`), an
**Edit** chip, a clear-samples icon, and three source tiles: **Webcam**,
**Upload**, **Device**.  Clicking a tile opens that source's drawer below the
card; clicking it again (or the `Close` pill) folds it away.

- **Device (ESP32-P4 serial)**: port chip + `Settings` chip, live preview,
  `Capture`, `Hold to Capture`.  Configure baud/frame side/channels in
  Settings → MCU Device Settings.  The firmware must be in a capture mode
  (`mode gray` for grayscale samples, `mode rgb` for RGB).
- **Webcam**: camera chip + `Settings` chip, live preview, `Capture`,
  `Hold to Capture`.  `Space` does the same hands-free.
- **Upload**: `Choose images from your files`; drag-and-drop onto the class
  card also works.
- **⚙ Settings → Advanced Settings → Developer: EMNIST template samples**:
  - `Auto-fill empty classes with EMNIST template samples (24 / class)`
  - `Fill Empty Classes with EMNIST Samples Now`
  - Fills only classes whose on-disk image count is 0, with 24
    dark-ink-on-light 96×96 captures; real captures are never touched.
    The toggle is project state, and changing a sign slot backfills the new
    character in the same request.

In a **sign_ocr** project the class list is owned by `#signSeqBar`: there is
no add/delete/rename/drag on the cards.  Editing a slot updates the cards,
deduplicates repeated characters (the badge reads `Slots 1 & 3`), marks a
previously trained model `⚠ Stale`, and syncs `/classes/sync_ocr`.

**What a sign class sample is:** one 28×28 glyph of that character — the
same single-glyph model an OCR project trains.  Training does not split a
whole road-sign photo into slots; the three-way split runs at Preview/device
inference time.  Use **EMNIST Auto-Fill** for a zero-camera workflow, or
capture/upload a frame where that character is the glyph to learn
(Manual ROI/crop in the worktable isolates it).  A whole 3-character sign
filed under `A` trains `A` on the whole sign, not on `A`.

---

## 4. Sample Worktable (`Edit`)

Click **Edit** on a class card to open the modal:

- **Original Sample** (left) with the green/blue ROI box, and **Processed
  Preview** (right) — for glyph projects this is the `28×28` thresholded
  glyph, shown **black ink on pure white and cropped to the ROI** (manual
  ROI = the user box, auto = the segmented glyph box).  The model tensor
  keeps its own polarity; only the user-facing preview is inverted.
  The modal is centred and both image stages are clean 1:1 squares.
- `Preprocess Mode`: **Auto (find sign)** or **Manual ROI**.
- `Dark Thr` (0–100, default 0) and `Lum Thr` (50–255, default 100).
- Manual ROI fields `ROI X1/Y1/X2/Y2` (0–1); drag on the original image.
- `Current Sample`, `Delete Sample`, sample thumbnails, `Save Sample` /
  `Saved`, and `Close`.
- Shortcuts: `F1` Auto, `F2` Manual ROI, `S` Save, `D` Delete sample, `Esc`
  Close, arrows navigate samples, `Shift`+arrows nudge the ROI.

Defaults 0/100 mean "untouched": exact integer Otsu on the device, no
luminance floor.

---

## 5. Training card & hyperparameters

**Train Embedded Model** is enabled when at least two classes exist and every
class has samples.  The status line shows `Model Trained · tap for accuracy`
(per-class accuracy modal) and adds `⚠ Stale` when the dataset fingerprint no
longer matches the last training.

### Training → Advanced

| Field | Default (image / glyph) | UI range | Notes |
|---|---|---|---|
| Image Size | 96 / **28** | 8–512, step 4 | Model input side; follows the project type |
| Batch Size | 32 | ≥1 | Auto-adjust may lower it for small datasets |
| Epochs | 20 | ≥1 (UI clamp 1–1000) | OCR replaces this total with Head + Fine-tune epochs |
| Validation Split | 0.25 | 0–0.5, step 0.05 | Held-out fraction |
| Learning Rate | 0.0016 | ≥0 | Base head/finetune schedule |
| Conv1 Filters | 32 | ≥1 | Hidden under transfer learning |
| Conv2 Filters | 64 | ≥1 | Hidden under transfer learning |
| Dense Units | 128 | ≥1 | Hidden under transfer learning |
| Ink Polarity | 0 — `Dark ink on bright bg` | `0` / `1` | Glyph projects; global override for every class |
| Sign Reading Direction | `ltr` — `Left to Right (L ➔ R)` | `ltr` / `rtl` | sign_ocr only; RTL reverses boxes, never pixels |

### OCR/sign extras (shown in the same Advanced panel)

| Field | Default | UI range | Notes |
|---|---|---|---|
| Mix EMNIST into every class | on | on/off | Adds synthetic glyphs to every real class |
| EMNIST split | `balanced` | `balanced`, `letters`, `digits` | Which EMNIST subset |
| Synthetic per class | 200 | 0–5000, step 10 | EMNIST glyphs added per class |
| Use `_background` noise class to replace OOD rejection (`#settingsUseBackgroundOod`) | **off** for Car License (`sign_ocr`), on for basic OCR | on/off | Car License ON: trains the extra `_background` class and exports `kSignBackgroundIsTheOnlyRejector = 1`. OFF: target characters only, OOD gates + 60 % floor, exports `0`. Basic OCR maps the same checkbox to `ocr_background_enabled`. |
| Background crops | 0 | 0–5000, step 10 | 0 mirrors Synthetic per class; used only when the toggle above is ON |
| Fine-tune pre-trained backbone | on | on/off | Two-stage transfer learning |
| Pre-train samples / class | 1200 | 100–6000, step 100 | Backbone warm-up corpus |
| Pre-train epochs | 4 | 1–100 | Backbone warm-up |
| Head epochs | 12 | 1–200 | Phase 1: head only, backbone frozen |
| Head learning rate | 0.0015 | ≥0 | Phase 1 LR |
| Fine-tune epochs | 25 | 0–200 | Phase 2: unfreeze last blocks |
| Fine-tune learning rate | 0.0002 | ≥0 | Phase 2 LR |
| Conv blocks to unfreeze | 1 | 0–16 | How many last conv blocks thaw |

**💡 Use recommended** applies dataset-aware `Batch Size`, `Epochs`,
`Learning Rate` and `Validation Split` only.  It targets ≈800 total updates
and ≥4 steps/epoch: `batch = min(current, max(2, train_count // 4))`,
`epochs = clamp(ceil(800 / steps_per_epoch), 30, 300)`,
`lr = 0.0008` below 100 samples else `0.001`, `val = 0.15` below 60 samples
else `0.2`.  Image Size and the architecture are unchanged.  **Reset** returns
the panel to the dataclass defaults (Image Size stays the project's own).

Under the hood:

- **EMNIST mixing** rebuilds a mixed dataset at `img_size`: per-class EMNIST
  glyphs are polarity-aligned to the capture convention, isolated and
  centered (no segmentation), shuffled with the real captures; the optional
  `_background` class receives random non-glyph crops.  In a Car License
  project the class list is locked to the unique target characters — stale
  `Class 1`/`Class 2` folders are deleted from `dataset/`,
  `.tm_processed_cache/` and `.tm_ocr_mixed/` before mixing (the EMNIST
  mapper would otherwise fill them and turn them into real labels).
- **Transfer learning** caches one EMNIST-pretrained backbone keyed by
  architecture, split, convention and polarity.  Phase 1 trains a fresh
  `N+1` classification head with the backbone frozen; phase 2 unfreezes the
  last `conv blocks` and fine-tunes at the lower LR.  If transfer learning is
  off, Conv1/Conv2/Dense are trained from scratch.

---

## 6. Preview

`⚙ Settings → Preview → Preview Image Input Source`:

- **Webcam**
- **Device (MCU)**
- **Local Image File**

The Preview card's `#previewContextBar` shows only the active source's
control (camera select, serial port + `Refresh`, or `📁 Upload Image...` with
a file name).  `Input` starts/stops the prediction loop.

The `ROI` and `Orig` checkboxes are view toggles:

| ROI | Orig | Scene | ROI rectangle |
|---|---|---|---|
| off | off | full **thresholded** frame | drawn (green = auto, blue = manual) |
| on | off | **cropped thresholded** ROI | hidden |
| off | on | full raw frame | drawn |
| on | on | cropped raw ROI | hidden |

`Orig` is the raw/threshold switch: **OFF (default)** shows the Dark/Lum
threshold view (pixels outside the window become pure white `255`, ink stays
dark), and **ON** shows the original unprocessed grayscale.  `ROI` is the
crop switch: **OFF** keeps the full scene with the rectangle drawn on it;
**ON** zooms the pane to the ROI content and hides the rectangle.  The
`.signmatch-slots` below already shows the three glyph crops, so the stage
never repeats them as a strip.

The **Preview stage is a 1:1 square** sized to the card's inner width.  A
96×96 or 160×160 device frame fills it without the wide side padding the
old stage produced, and `image-rendering: pixelated` keeps the small stream
crisp when it is scaled up.  The ROI rectangle is positioned against the
rendered image content (object-fit letterbox offsets), never the wide pane.

The card's `Dark` / `Lum` sliders retune the Preview mask live without
retraining.

### Sign projects

In the **Local Image File** source a sign project also shows:

- `🧪 Synth Match` — compose the current target and assert `YES`.
- `🔀 Synth Reversed` — compose in reversed order, for LTR/RTL probes.
- `❌ Synth Mismatch` — substitute a wrong character and assert `NO`.

They are disabled until a target is set and a model is trained.  Auto-fill
and a finished training run also store a synthetic plate, so Preview has a
ready frame with zero files on disk.

Before splitting, the host sign preview locates the sign board on the
**native frame** (`find_sign_roi`): the pinned Manual ROI when one is set;
when Dark/Lum have been tuned, an **ink-cluster** pass ignores components
within 6 % of the frame edge (fingers/frame/border noise), drops oversized
and tiny blobs — a thin `1` is allowed — and takes the horizontal row of
three character blobs.  The ROI is their union plus one uniform 10 % pad on
all four sides, so the box is tight and symmetric rather than pulled left by
edge clutter.  Otherwise the enclosed bright inner plate is used, with a
centred 70 % fallback.  The ROI is cropped first, border-line/noise ink is
removed, and only then does the three-glyph projection run.

Each slot crop gets a small plate-context margin (8 % of its smaller side,
minimum 2 px) before the glyph transform.  A stroke-only `1` crop otherwise
becomes an almost-black tensor and can score as `_background`.  Under
`NORM_CANON` a narrow glyph (ink bbox `w/h < 0.35`) is additionally widened
to a 3-4 px stroke **before** the nearest-neighbour fit — a 1 px source
stroke can otherwise be skipped entirely by the downscale — matching the
EMNIST `1` samples the model trains on.  The C mirror (`sign_glyph.c`)
implements the identical margin and widening, so host/device parity is
preserved.  The `_background` class is also generated from textured whole
patches (plus a 10 % minority of all-zero blanks) and rejects any centered
stroke-bar prototype that would collide with `1`.

Webcam frames for glyph projects are center-cropped to 1:1 on ingest, so a
16:9 MacBook camera behaves like the square MCU camera (no letterbox bars,
and ROI coordinates are the pixels the model sees).

The response carries `roi_box` (source pixels), `roi_box_norm` / `crop`
(full-frame fractions), a real `sign_pct`, a thresholded full-frame
`full_image_b64`, a cropped threshold `processed_image_b64` and a cropped raw
`crop_image_b64`.  Every user-facing mask is **black ink on pure white**;
each `.signmatch-slot` thumbnail is likewise a black-on-white 28×28 display
glyph.

The output pane is the `.signmatch` banner:

- `🟢 YES — MATCH` or `🔴 NO — NOT MATCH`.
- Three `.signmatch-slot` cells, each showing `WANT <char>`, the read glyph,
  the slot's 28×28 glyph image, and its confidence percentage; green = match,
  red = fail.
- A red banner names the failing slot(s) and the reason (wrong glyph,
  `_background`, no glyph, or below the 0.60 floor).
- Idle states: `◻ SET THE TARGET SIGN` and `◻ NO MODEL YET`.

`Preview Image Input Source`, the four OOD gates (`sign_pct` min/max, max
prob, entropy ratio) and the per-source settings are in Settings → Preview.

---

## 7. Export

`Export Model` asks for a folder (typically the matching sketch folder) and
writes, with default export name `tm`:

```
tm.tflite
tm_model_data.h
tm_model_data.cpp
model_resolver.h
model_settings.h
model_settings.cpp
labels.txt
```

For firmware, copy the **5 compile files** next to the sketch:
`tm_model_data.cpp`, `tm_model_data.h`, `model_resolver.h`,
`model_settings.cpp`, `model_settings.h`.  The default C array symbol is
`g_tm_model_data`.

Sign exports add `TM_SIGN_OCR_MODEL 1`, `TM_OCR_GLYPH_MODEL 0`,
`kTargetSignSequence`, `kTargetSignClassIndices[3]`, `kSignReadingOrder`
(0 LTR / 1 RTL) and `kSignMatchMinProb` (0.60) to `model_settings.h`.
Exporting also stamps `<workspace>/deployed.json`, pinning Preview to the
model that was just exported.

Legacy `model.h`/`model.cpp` duplicates are no longer written and are removed
if present (they caused duplicate-symbol link errors).

---

## 8. Saving projects (`.tmproj`)

**File → Save** (⌘S) saves to the current path; **Save as** (⇧⌘S) picks a new
one.  `.tmproj` is a ZIP:

```
project.tmproj
├── manifest.json            # magic, version 2.0, project_type, img_size
├── tm_project_state.json    # classes, counts, train_cfg, preprocess
├── tm_classes.json
├── tm_train_latest.json     # when a model exists
├── dataset/
│   ├── labels.txt / labels.json
│   └── <class>/<samples>
└── runs/latest/             # when a trained run exists
    ├── model.keras / model.tflite
    ├── model_settings.* / model_resolver.h
    ├── train_config.json
    └── labels.*
```

The archive header is authoritative on reopen.  `train_cfg` is a complete
`TrainConfig` snapshot, including `sign_sequence`, `reading_order`,
`auto_seed_emnist_samples`, OCR mixing/transfer-learning fields and the
preprocessing geometry.  Legacy v1 archives are readable; missing fields fall
back to this build's defaults and the app says which ones.

---

## 9. App data directories

- macOS: `~/Library/Application Support/TFLiteTraining/`
- Windows: `%APPDATA%\TFLiteTraining\`

```
workspace/   current project workspace
datasets/    imported datasets
runs/        training run artifacts
logs/        streamlit.log
ocr_backbone/ cached EMNIST backbones
```

Dataset import accepts `label/image` and `train|val|test/label/image` layouts;
extensions `jpg/jpeg/png/bmp/webp/gif`.  Training always reads grayscale.

---

## 10. Project layout (developers)

- `desktop_launcher.py` — Streamlit server + WebView window
- `app.py` — Streamlit UI + embedded SPA (`_render_tm_old_frontend_html`)
- `record_controller.py` — localhost HTTP controller: live, predict, classes,
  train, export, project save/open
- `image_preprocess.py` — image-project crop/mask/preview transform
- `ocr_preprocess.py` — integer-exact glyph and sign front-end (host mirror)
- `ocr_dataset.py` — EMNIST ingestion, template seeder, sign synthesizer
- `ocr_trainer.py` — EMNIST mixing, backbone pretraining, two-phase fine-tune
- `trainer.py` — image-project trainer, quantization, export sources
- `dataset_io.py` — project types, defaults, import/export
- `serial_device.py` — serial frame reader
- `ui_styles.py`, `camera_permission.py`, `dmg_settings.py`

## 11. macOS (.app + .dmg)

```bash
cd AItraining
python3.11 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
python3 -m pip install -r requirements-dev.txt
python3 -m pip install pyinstaller dmgbuild
python3 -m PyInstaller --clean --noconfirm TFLiteTraining.spec
dmgbuild -s dmg_settings.py "TFLiteTraining" "dist/TFLiteTraining.dmg"
```

Artifacts: `dist/TFLiteTraining.app`, `dist/TFLiteTraining.dmg`.

## 12. Windows (.exe)

```powershell
cd AItraining
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
py -m pip install -r requirements.txt
py -m pip install -r requirements-dev.txt
py -m pip install pyinstaller
python -m PyInstaller --clean --noconfirm TFLiteTraining.spec
```

Artifact: `dist\TFLiteTraining\TFLiteTraining.exe`.

## FAQ

- macOS “unidentified developer”: right-click the app → Open → Confirm.
- macOS webcam permission: allow the app under System Settings → Privacy &
  Security → Camera.
- macOS packaged app does not prompt: run `dist/TFLiteTraining.app`, not the
  terminal entry point.
- Windows missing DLL: install the Microsoft Visual C++ Redistributable
  2015–2022 (x64).
- Python 3.13 + PyInstaller crashes: use Python ≤ 3.12.
