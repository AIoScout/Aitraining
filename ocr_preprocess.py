"""Integer-exact glyph preprocessing for the OCR / Character project.

Every *decision* in this module — the binarisation threshold, the ink
predicate, the segmentation bounds and the resampling index — is computed
with Python integers only.  No `float`, no `np.float32`, no
`hist / hist.sum()`, no `Image.resize`.  This is not stylistic: each of
those four is a *decision* whose outcome flips when the last mantissa bit
differs between CPython/NumPy and the ESP32-P4 firmware, and a flipped
decision does not shift an output pixel by ±1 count — it changes how many
glyphs were found, or which source pixel was sampled, and the two sides
diverge catastrophically from there.

Measured on a 386-image adversarial corpus, a hand-mirrored float
implementation of exactly this pipeline (numpy float64 Otsu + PIL NEAREST
vs. double Otsu + `(int)(x*scale)`) disagreed on 228/386 = 59.1% of
tensors.  The integer-exact form below disagreed on 0/386.

The device mirror (TFLite/image_provider.cpp) must use the SAME integer
expressions, not an equivalent float one.  See the "C port notes" at the
bottom of each function for the exact overflow bounds the C side must
honour.

Pipeline (captures):   RGB -> integer BT.601 luminance -> exact Otsu ->
                       polarity-aware ink -> vertical-projection
                       segmentation -> exact NN resize -> 28x28 ink mask
Pipeline (EMNIST):     the same, with segmentation BYPASSED (EMNIST ships
                       already isolated and centred; the projection scan
                       finds exactly one run on 124/124 measured samples,
                       so it is a no-op there — and the component cleanup
                       normally bundled with it deletes the dots of i/j in
                       4/124 and thin serif strokes in 16/124 more).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ── Glyph geometry ──────────────────────────────────────────────────────
GLYPH_SIZE = 28  # model input side (both axes)
# Canonical MNIST/EMNIST normalisation: the shorter-side fit target and
# the paste offset.  See GLYPH_FIT for when this is used.
_GLYPH_CANON_FIT = 20
_GLYPH_CANON_OFFSET = (GLYPH_SIZE - _GLYPH_CANON_FIT) // 2  # 4

# A camera-segmented '1' is often a 1-2 px vertical stroke.  Scaled to the
# 28x28 canvas under NORM_CANON (longer side -> 20 px) it stays 1-2 px,
# while the EMNIST '1' samples the model trains on are 3-4 px wide.  That
# distribution gap made a clean stroke read as `_background`.  Glyphs whose
# ink bbox is narrower than 35 % of its height get their source stroke
# widened (polarity-aware horizontal min/max) to at least
# _GLYPH_TARGET_STROKE output pixels before the nearest-neighbour resize.
_GLYPH_NARROW_ASPECT_PCT = 35
_GLYPH_TARGET_STROKE = 3
_GLYPH_MAX_STROKE_RADIUS = 4

# ── Ink polarity ────────────────────────────────────────────────────────
# Otsu splits the histogram into a low class [0 .. t] and a high class
# [t+1 .. 255].  Polarity says WHICH of those two classes is the character.
INK_DARK = 0    # dark glyph on a bright page (camera captures, signs)
INK_BRIGHT = 1  # bright glyph on a dark page (EMNIST as distributed)

# ── Segmentation ────────────────────────────────────────────────────────
# Minimum column-run width to keep.  A run narrower than this is noise,
# not a character stroke.  1 = keep everything the projection finds.
MIN_GLYPH_RUN = 2

# ── Normalisation conventions ───────────────────────────────────────────
# STRETCH: fit the glyph bbox onto the full 28x28 grid, one scale factor
#     per axis.  This is the convention the project brief specifies.
# CANON:   MNIST/EMNIST convention — fit the bbox of the ANTI-ALIASED crop
#     so its longer side is 20 px with the aspect ratio preserved, paste
#     centred in 28x28, then shift so the intensity centre of mass sits at
#     (13.5, 13.5).  Measured 2.5x lower same-convention residual floor
#     (14.87 vs 37.14 MAE) and 2.69x less sensitive to a +/-1 px bbox
#     error.  REQUIRED if EMNIST samples are consumed as distributed.
NORM_STRETCH = "stretch"
NORM_CANON = "canon"


# ── Vocabulary normalisers ──────────────────────────────────────────────
# The UI, the class/sample preprocess records and the .tmproj manifest all
# store these as loose JSON.  These two functions are the ONLY place the
# string<->constant mapping is defined, so a value written by one code path
# can never be interpreted differently by another.
def normalize_ink_polarity(value: Any) -> int:
    """→ INK_BRIGHT for the bright-on-dark spellings, INK_DARK otherwise."""
    if isinstance(value, bool):
        return INK_BRIGHT if value else INK_DARK
    if isinstance(value, (int, float)):
        return INK_BRIGHT if int(value) == INK_BRIGHT else INK_DARK
    key = str(value or "").strip().lower()
    if key in {"bright", "bright_on_dark", "1", "light", "inverted", "emnist"}:
        return INK_BRIGHT
    return INK_DARK


def normalize_norm_convention(value: Any) -> str:
    """→ NORM_CANON for the MNIST/EMNIST spellings, NORM_STRETCH otherwise."""
    key = str(value or "").strip().lower()
    if key in {"canon", "canonical", "mnist", "emnist", "centre", "center"}:
        return NORM_CANON
    return NORM_STRETCH


def align_source_polarity(image: np.ndarray, *, source_polarity: int,
                          target_polarity: int) -> np.ndarray:
    """Invert a SOURCE image whose ink is the wrong way round.

    Apply this to the input of preprocess_glyph, never to its output.
    preprocess_glyph's `polarity` argument only decides WHICH SIDE of the
    Otsu split is the glyph — that is, which pixels the bounding box is
    measured from.  It never rewrites the emitted crop (the crop keeps the
    source's intensities, so anti-aliased edges survive).  So a foreign
    source whose ink is on the other side — EMNIST ships bright-on-dark
    while a camera capture is dark-on-bright — arrives in the model's
    input at the opposite end of the intensity range from the captures it
    shares a class with, and the model reads that as contradictory
    supervision.

    Fixing it on the SOURCE rather than on the output matters under
    NORM_CANON: that convention pastes the fitted glyph onto a zero-filled
    canvas, so the crop carries a black padding ring.  Inverting the
    output would turn that ring bright — a halo that appears in no real
    sample — whereas inverting the source lets the padding be generated
    normally from the now-correctly-oriented glyph.

    `target_polarity` is the DEPLOYMENT convention, i.e. what the device
    computes from a live frame.  EMNIST is the half that moves, because it
    is the half that does not exist at inference time.
    """
    arr = np.asarray(image)
    if int(source_polarity) == int(target_polarity):
        return arr.astype(np.uint8, copy=False)
    return (255 - arr.astype(np.uint16)).astype(np.uint8)


# ════════════════════════════════════════════════════════════════════════
# 1. Luminance
# ════════════════════════════════════════════════════════════════════════
def rgb_to_lum_u8(rgb: np.ndarray) -> np.ndarray:
    """Integer BT.601 luminance, round-half-up.  No floats anywhere.

    lum = (r*30 + g*59 + b*11 + 50) // 100

    The device's model-input path computes the same weights in float32
    ((r*30 + g*59 + b*11)/100 + 0.5, then truncate).  Both are
    round-half-up, but the float form is not bit-identical to this one for
    every input, so the OCR stage on the device must use THIS integer
    expression — the whole point of the module is that no decision
    downstream of it can be perturbed by a rounding difference.

    Overflow: max r*30 + g*59 + b*11 = 255*100 = 25500, so the NumPy
    int32 accumulator (and a plain C `int`) cannot overflow.
    """
    src = np.asarray(rgb)
    if src.ndim == 2:  # already single-channel: identity
        return np.ascontiguousarray(src.astype(np.uint8, copy=False))
    if src.shape[-1] == 1:
        return np.ascontiguousarray(src[:, :, 0].astype(np.uint8, copy=False))
    if src.shape[-1] < 3:
        src = np.repeat(src[:, :, :1], 3, axis=2)
    # int32 so the weighted sum has 7 bits of headroom above uint8.
    r = src[:, :, 0].astype(np.int32)
    g = src[:, :, 1].astype(np.int32)
    b = src[:, :, 2].astype(np.int32)
    lum = (r * 30 + g * 59 + b * 11 + 50) // 100
    return np.clip(lum, 0, 255).astype(np.uint8)


# ════════════════════════════════════════════════════════════════════════
# 2. Otsu binarisation — exact integer, cross-multiplied
# ════════════════════════════════════════════════════════════════════════
def otsu_threshold_u8(gray: np.ndarray) -> int:
    """Exact integer Otsu threshold.  Returns t in [0, 255].

    The returned t is the FIRST level of the bright class, i.e. the split
    is `dark = [0 .. t-1]` / `bright = [t .. 255]`, and the ink predicate
    is the EXCLUSIVE `v < t` (dark ink) or `v >= t` (bright ink).  This
    convention — rather than "t is the last dark level, v <= t" — is the
    one pinned by the cross-language reference: it makes the degenerate
    t == 0 case mean "no ink at all", which is the safe answer for a
    frame with no valid split.  The inclusive form would call every pixel
    of an all-black frame ink and emit one whole-frame glyph.

    THE MATH (all integers; see the derivation for why no division is
    needed until the very end, and none is performed at all here):

    For a split at t the histogram partitions into
        class 0 = [0 .. t-1]      count D0, sum S0   (= prefix over < t)
        class 1 = [t .. 255]      count D1, sum S1
    with N = D0 + D1 the pixel count and St = S0 + S1 the total sum.

    Between-class variance is

        sigma_b^2 = w0 * w1 * (mu0 - mu1)^2
                  = (S0*D1 - S1*D0)^2 / (N^2 * D0 * D1)

    N^2 is the same for every t, so maximising sigma_b^2 over t is the
    same as maximising

        num^2 / (D0 * D1),     num = S0*D1 - S1*D0

    and `num` collapses to a single product using D1 = N - D0,
    S1 = St - S0:

        num = S0*(N - D0) - (St - S0)*D0
            = S0*N - S0*D0 - St*D0 + S0*D0
            = N*S0 - St*D0                      <-- computed below

    Comparing two candidate splits c and b without ever dividing
    (D0*D1 > 0 for both, so cross-multiplication preserves the order):

        num_c^2 * (D0_b * D1_b)  >  num_b^2 * (D0_c * D1_c)

    which is `cand_num2 * best_d > best_num2 * cand_d` below.  `num` may
    be negative (the low class can be the brighter one); it is only ever
    squared, so the sign is irrelevant.

    TIE-BREAKING: the comparison is STRICT `>`, so an equal-variance
    candidate never replaces the incumbent — the FIRST maximiser wins.
    That makes the result a pure function of the histogram.

    ZERO / DEGENERATE HANDLING (deterministic, spelled out because the C
    side must reproduce it exactly):
      * total == 0 (empty image)            -> return 0.
      * t == 0 is never evaluated: class 0 would be empty.  The loop runs
        t in [1, 255].
      * splits with D0 == 0 or D1 == 0 are SKIPPED, not scored as zero.
        Scoring them as zero would let a later genuine zero-variance
        candidate tie and lose to the incumbent, changing which t wins.
      * an image with no valid split (a single distinct level, or a blank
        frame) leaves `best_num2` at -1, so the fallback t = 0 is
        returned and the ink predicate `v < 0` matches NOTHING.  That is
        the intended answer: a frame with no split has no glyph.
      * on a flat histogram every split has sigma_b^2 = 0; the first
        valid split (the smallest t with both classes non-empty) wins.

    The histogram itself is the one place NumPy is allowed: `bincount`
    is exact integer counting, not a float computation.  The division
    `hist / hist.sum()` that the naive implementation uses is what makes
    32/386 degenerate frames disagree with this function.
    """
    src = np.asarray(gray)
    if src.size == 0:
        return 0
    counts = np.bincount(src.reshape(-1).astype(np.uint8, copy=False), minlength=256)
    total = int(counts.sum())
    if total <= 0:
        return 0
    levels = np.arange(256, dtype=np.int64)
    total_sum = int((levels * counts).sum())

    best_t = 0
    best_num2 = -1   # -1 == "no candidate yet"; see tie-breaking note
    best_d = 0       # D0*D1 of the incumbent
    d0 = 0           # running D0 = count of values < t
    s0 = 0           # running S0 = sum of level*hist[level] for level < t

    hist = counts.astype(np.int64)
    for t in range(1, 256):       # t = first level of the bright class
        h = int(hist[t - 1])      # grow class 0 to [0 .. t-1]
        d0 += h
        s0 += (t - 1) * h
        d1 = total - d0
        if d0 <= 0 or d1 <= 0:    # empty class -> invalid split, skip
            continue
        num = total * s0 - total_sum * d0          # == S0*D1 - S1*D0
        num2 = num * num                            # Python int: exact
        cand_d = d0 * d1
        if best_num2 < 0 or num2 * best_d > best_num2 * cand_d:
            best_t = t
            best_num2 = num2
            best_d = cand_d
    return best_t


# ════════════════════════════════════════════════════════════════════════
# 3. Ink predicate + vertical-projection segmentation
# ════════════════════════════════════════════════════════════════════════
def binarize_ink(gray: np.ndarray, thresh: int, polarity: int = INK_DARK) -> np.ndarray:
    """Bool ink mask.  Otsu's t separates [0..t] from [t+1..255]; polarity
    says which side is the glyph.

    Written ONCE and shared by every caller — the C side must use the same
    comparison operator, not an equivalent one.  The edge cases genuinely
    differ: at the t == 0 fallback, `v <= t` yields ink for an all-black
    frame while `v < t` yields none — a 100% behavioural divergence, not a
    rounding error.
    """
    src = np.asarray(gray)
    if polarity == INK_DARK:
        return src <= np.uint8(thresh)   # low class = ink
    return src > np.uint8(thresh)        # high class = ink


def projection_runs(ink: np.ndarray, min_run: int = MIN_GLYPH_RUN) -> List[Tuple[int, int]]:
    """Column runs of the vertical projection.  Pure integer accumulation.

    proj[x] = number of ink pixels in column x (an exact int count).
    A run is a maximal span of columns with proj[x] > 0; runs narrower
    than `min_run` are dropped as noise.  Returns exclusive-ended
    (start, end) pairs in ascending order.

    This is the whole segmenter.  Deliberately NO connected-component
    analysis and NO "keep largest blob" cleanup: the dots of i/j and thin
    serif strokes are disconnected from the stem but lie in the SAME
    projection columns, so a column run keeps them; a largest-component
    filter deletes them (measured: 4/124 samples lose the dot, 16/124
    more lose thin strokes, mean 27.5% of ink mass).  A stage that can
    only subtract information from an already-isolated glyph is not worth
    its parity risk.
    """
    if ink.ndim != 2:
        raise ValueError("projection_runs expects a 2-D ink mask")
    h, w = ink.shape
    if h <= 0 or w <= 0:
        return []
    # axis=0 sums down the columns; int64 accumulator cannot overflow
    # (max = h <= a few thousand).
    proj = ink.sum(axis=0, dtype=np.int64)
    runs: List[Tuple[int, int]] = []
    start = -1
    for x in range(w):
        if int(proj[x]) > 0:
            if start < 0:
                start = x
        elif start >= 0:
            if x - start >= min_run:
                runs.append((start, x))
            start = -1
    if start >= 0 and w - start >= min_run:
        runs.append((start, w))
    return runs


def _column_span_bbox(ink: np.ndarray, x1: int, x2: int) -> Optional[Tuple[int, int, int, int]]:
    """Tight exclusive-ended bbox of the ink inside columns [x1, x2).

    x is exact (the projection columns are non-empty by construction);
    y is the first/last ink row in that column span.  Returns None when
    the span holds no ink at all.
    """
    if x2 <= x1:
        return None
    rows = np.flatnonzero(np.asarray(ink[:, x1:x2]).any(axis=1))
    if rows.size == 0:
        return None
    return int(x1), int(rows[0]), int(x2), int(rows[-1]) + 1  # (x1,y1,x2,y2)


def segment_glyph_boxes(ink: np.ndarray, min_run: int = MIN_GLYPH_RUN) -> List[Tuple[int, int, int, int]]:
    """One exclusive-ended bbox per column run, in left-to-right order.

    Use this for multi-character captures (a K-glyph frame).  For the
    single-glyph case the caller wants `segment_glyph_box`.
    """
    boxes: List[Tuple[int, int, int, int]] = []
    for x1, x2 in projection_runs(ink, min_run=min_run):
        box = _column_span_bbox(ink, x1, x2)
        if box is not None:
            boxes.append(box)
    return boxes


def segment_glyph_box(ink: np.ndarray, min_run: int = MIN_GLYPH_RUN) -> Optional[Tuple[int, int, int, int]]:
    """Single union bbox over every KEPT run — the one-glyph-per-frame case.

    The union is taken over the kept per-run boxes, NOT over the whole
    column span from the first to the last run.  The difference is real
    and was caught by the cross-language corpus: a run discarded by
    `min_run` can still contain ink on rows no kept run reaches, and
    re-scanning the span would silently extend the box to include it
    (measured on multi-glyph rows: span-union 76 vs kept-union 75 rows).
    Deriving from `segment_glyph_boxes` keeps this one code path.
    """
    boxes = segment_glyph_boxes(ink, min_run=min_run)
    if not boxes:
        return None
    return (
        boxes[0][0],                      # x1 = leftmost kept run
        min(b[1] for b in boxes),         # y1 = topmost ink among kept runs
        boxes[-1][2],                     # x2 = rightmost kept run
        max(b[3] for b in boxes),         # y2 = bottommost ink among kept runs
    )


# ════════════════════════════════════════════════════════════════════════
# 3b. Sign sequences — reading a multi-glyph sign as one label
#
# A road sign carries its label as SEVERAL glyphs printed side by side:
# "L 50", "3 0" (a speed limit), "A 1".  The whole label is one class, so
# the glyph front-end has to hand the model three separate 28x28 crops,
# one per character, in the order a reader would say them.
#
# That is the same segmentation problem as the single-glyph case with one
# extra requirement the single-glyph case never had: the answer has to be
# EXACTLY `count` boxes.  A frame with three clean gaps and a frame with
# the glyphs touching must both produce three crops, because the model
# always gets three inputs and the label always has three characters —
# "we found two runs" is not a thing the caller can act on.
# ════════════════════════════════════════════════════════════════════════
SIGN_SEQUENCE_LEN = 3  # characters on a road-sign label
# What an unfilled slot holds.  Deliberately a character no sign prints and
# no class is ever named after, so "the third slot is blank" is a state the
# matcher can recognise rather than a '0' it would cheerfully match.
SIGN_SEQUENCE_PAD = "?"


def normalize_sign_sequence(value: Any, length: int = SIGN_SEQUENCE_LEN) -> List[str]:
    """Coerce anything a UI or a manifest can hold into `length` characters.

    Accepts a list/tuple of strings, or a single string read as its
    characters ("A1A" and "A 1 A" both give ['A', '1', 'A']).  Each entry
    is upper-cased and trimmed to one character — a sign label is
    glyph-by-glyph, and the class cards are named for those same single
    characters, so a two-character entry could never match a class.  Short
    input is padded with '?' — the placeholder that cannot collide with a
    real class name — so the result is ALWAYS `length` long and the caller
    never has to handle a ragged sequence.

    This is the one place the shape is decided: the SPA's edit modal, the
    preview pipeline and the saved project all round-trip through it.
    """
    try:
        n = int(length)
    except Exception:
        n = SIGN_SEQUENCE_LEN
    n = max(1, n)
    chars: List[str] = []
    if isinstance(value, str):
        raw: List[Any] = [c for c in value if not c.isspace()]
    elif isinstance(value, (list, tuple)):
        raw = list(value)
    else:
        raw = []
    for item in raw:
        text = "" if item is None else str(item).strip()
        chars.append(text[:1].upper() if text else "")
    chars = [c for c in chars if c]
    if len(chars) > n:
        chars = chars[:n]
    while len(chars) < n:
        chars.append(SIGN_SEQUENCE_PAD)
    return chars


def normalize_sign_sequences(
    value: Any, length: int = SIGN_SEQUENCE_LEN
) -> List[str]:
    """Coerce a whitelist of allowed plates into unique, complete sequences.

    Accepts:
      * one sequence as a string ("A12") or a list of characters
        (["A", "1", "2"]) — the legacy single-target form;
      * a list of sequences (["A12", "B18"], [["A","1","2"], ...]).

    Every plate is uppercased, padded/truncated by normalize_sign_sequence,
    and kept only when all `length` characters are real 0-9/A-Z (a plate with
    a '?' pad is incomplete and is not a valid target).  Duplicates are
    removed in first-appearance order.  Returns [] for no usable plate; the
    caller decides whether that is an error.
    """
    try:
        n = max(1, int(length))
    except Exception:
        n = SIGN_SEQUENCE_LEN
    items: List[Any]
    if isinstance(value, str):
        items = [value]
    elif isinstance(value, (list, tuple)):
        # A bare character list is the legacy single sequence, not a list of
        # one-character plates.
        if len(value) == n and all(
            not isinstance(v, (list, tuple)) and len(str(v).strip()) <= 1
            for v in value
        ):
            items = [value]
        else:
            items = list(value)
    else:
        return []
    out: List[str] = []
    seen = set()
    for item in items:
        if item is None or item == "":
            continue
        if not isinstance(item, (str, list, tuple)):
            continue
        seq = normalize_sign_sequence(item, length=n)
        if len(seq) != n or any(c == SIGN_SEQUENCE_PAD for c in seq):
            continue
        text = "".join(seq)
        if text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


def _union_box(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
    """Exclusive-ended bbox covering both inputs."""
    return (min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]))


def _equal_split_boxes(
    x1: int, x2: int, y1: int, y2: int, count: int
) -> List[Tuple[int, int, int, int]]:
    """`count` equal-width column bands spanning [x1, x2), all at [y1, y2).

    The last band ends exactly on x2 rather than at `x1 + count*band`, so
    the bands always tile the span with no gap and no overlap however the
    integer division falls.  This is the fallback for a frame with no ink
    to find valleys in — the caller still gets its three crops, and they
    are the three crops the geometry alone would predict.
    """
    width = max(0, int(x2) - int(x1))
    boxes: List[Tuple[int, int, int, int]] = []
    for i in range(count):
        bx1 = x1 + (width * i) // count
        bx2 = x2 if i == count - 1 else x1 + (width * (i + 1)) // count
        boxes.append((int(bx1), int(y1), int(bx2), int(y2)))
    return boxes


def _deepest_valley(ink: np.ndarray, x1: int, x2: int) -> Optional[int]:
    """The column boundary inside (x1, x2) that cuts through the least ink.

    Returns a p meaning "cut between column p-1 and column p", i.e. the
    span [x1, x2) becomes [x1, p) + [p, x2).  The cut is scored on the ink
    of the two columns it separates — the amount of stroke it would slice
    — so a cut through the middle of a broad gap, where both neighbours are
    empty, always beats one that clips a stroke.  Ties go to the boundary
    nearest the midpoint, so two equally clean gaps split the span into
    even halves rather than hugging one end.

    None when the span is too narrow to hold a cut at all.
    """
    if x2 - x1 < 2:
        return None
    proj = ink.sum(axis=0, dtype=np.int64)
    mid2 = int(x1) + int(x2)
    best_p: Optional[int] = None
    best_cost: Optional[int] = None
    best_dist: Optional[int] = None
    for p in range(int(x1) + 1, int(x2)):
        cost = int(proj[p - 1]) + int(proj[p])
        dist = abs(2 * p - mid2)
        if best_cost is None or cost < best_cost or (cost == best_cost and dist < best_dist):
            best_p, best_cost, best_dist = p, cost, dist
    return best_p


def split_sign_glyph_boxes(
    ink: np.ndarray, count: int = SIGN_SEQUENCE_LEN, min_run: int = MIN_GLYPH_RUN
) -> List[Tuple[int, int, int, int]]:
    """EXACTLY `count` exclusive-ended glyph boxes, in left-to-right order.

    Three moves, in order, and the last one cannot fail:

    1. THE PROJECTION ALREADY AGREES.  `segment_glyph_boxes` finds exactly
       `count` column runs — three glyphs with three separate ink columns.
       Use them verbatim; this is the ordinary case for a printed sign.
    2. IT DOES NOT.  Reconcile:
       - too many runs (a "4" with a detached stem, a speck of sensor
         noise): repeatedly MERGE THE NARROWEST RUN into its closer
         neighbour.  Narrowest rather than "smallest gap", and the
         difference is measured: on a sign whose three real glyphs sit at
         x=10/30/60 with a 2 px speck at x=90, the smallest gaps are the
         12 px ones BETWEEN the glyphs, so gap-first merging fuses glyphs
         1 and 2 and keeps the speck.  Width is the signal that survives
         both failure modes — a broken stroke's fragment and a noise speck
         are both narrower than the glyphs beside them.
       - too few runs (touching glyphs, a "111" that never separates):
         SPLIT the widest box at its deepest valley, repeatedly.  Widest
         first because the widest box is the one most likely to be holding
         more than one glyph.
    3. NEITHER MOVE IS AVAILABLE — no ink at all, or a span too narrow to
       hold a cut.  Equal thirds of the union bbox.  The caller's contract
       is three crops; this is the last one that can honour it.

    Deterministic: every loop breaks a tie by index order, so the same
    frame always yields the same boxes.
    """
    try:
        n = int(count)
    except Exception:
        n = SIGN_SEQUENCE_LEN
    n = max(1, n)
    arr = np.asarray(ink)
    if arr.ndim != 2 or arr.size == 0:
        return []
    h, w = int(arr.shape[0]), int(arr.shape[1])

    boxes = segment_glyph_boxes(arr, min_run=min_run)
    if not boxes:
        return _equal_split_boxes(0, w, 0, h, n)

    while len(boxes) > n:
        k = min(range(len(boxes)), key=lambda i: (boxes[i][2] - boxes[i][0], i))
        if k == 0:
            j = 1
        elif k == len(boxes) - 1:
            j = k - 1
        else:
            j = k - 1 if (boxes[k][0] - boxes[k - 1][2]) <= (boxes[k + 1][0] - boxes[k][2]) else k + 1
        lo, hi = (j, k) if j < k else (k, j)
        boxes[lo : hi + 1] = [_union_box(boxes[lo], boxes[hi])]

    while len(boxes) < n:
        i = max(range(len(boxes)), key=lambda k: (boxes[k][2] - boxes[k][0], -k))
        bx1, by1, bx2, by2 = boxes[i]
        p = _deepest_valley(arr, bx1, bx2)
        if p is None:
            # This box cannot be cut.  Give up on the reconciliation and
            # answer geometrically over the whole span, so the result is
            # still exactly `n` boxes hugging whatever ink was found.
            whole = _union_box(boxes[0], boxes[-1])
            return _equal_split_boxes(whole[0], whole[2], whole[1], whole[3], n)
        left = _column_span_bbox(arr, bx1, p)
        right = _column_span_bbox(arr, p, bx2)
        # A sub-span with no ink of its own keeps the parent's rows: it is a
        # blank slot, and reporting it as such is more useful than dropping
        # it (the model still gets a 28x28 crop, and it reads as empty).
        boxes[i : i + 1] = [
            left if left is not None else (bx1, by1, p, by2),
            right if right is not None else (p, by1, bx2, by2),
        ]
    return boxes


# ════════════════════════════════════════════════════════════════════════
# 4. Nearest-neighbour resize — exact rational box-centre mapping
# ════════════════════════════════════════════════════════════════════════
def nn_source_indices(in_len: int, out_len: int) -> np.ndarray:
    """Exact integer source-index map: src(o) = floor((2o+1)*in_len/(2*out_len)).

    THE MATH.  Output pixel o covers the source interval
    [(o)*in/out, (o+1)*in/out).  Its centre is (o+0.5)*in/out, and the
    nearest-neighbour source index is the floor of that centre:

        src(o) = floor((o + 0.5) * in_len / out_len)
               = floor((2o + 1) * in_len / (2 * out_len))     <- integer form

    The right-hand side is the left-hand side multiplied through by 2,
    which changes nothing under `floor` for exact arithmetic and — unlike
    the left-hand side — needs no float.  `(2*o + 1)` is odd, so the
    numerator is never an exact multiple of the even denominator at the
    half-way points, which is what makes the tie behaviour deterministic
    instead of dependent on a rounding mode.

    Bounds: o <= out_len - 1 gives numerator <= (2*out_len - 1)*in_len
    < 2*out_len*in_len, so src(o) <= in_len - 1 for all o and src(0) = 0.
    The map is therefore always in range and needs no clamp — a clamp
    would be a second code path that could differ between languages.

    Why this map and not the alternatives (measured over 17,584 sampled
    positions on 314 real crops, vs. PIL Image.NEAREST):
        (int)(x * in/out) float scale ........ 72.4% of positions differ
        floor((o+0.5)*in/out - 0.5) .......... 52.2% differ
        the repo's bilinear centre map ....... 52.2% differ (half a source
            pixel off when reused for NEAREST -> 60.1% tensor divergence)
        THIS map .............................  0.7% differ
    The residual 0.7% is PIL's own internal double accumulation losing a
    unit at exact integers (its accumulator lands on 0.99999999999999989
    at widths 8,16,32,40,64,72,80), which no correctly-rounded integer
    formula can or should reproduce.
    """
    if in_len <= 0 or out_len <= 0:
        raise ValueError("nn_source_indices: in_len and out_len must be >= 1")
    o = np.arange(out_len, dtype=np.int64)
    return ((2 * o + 1) * np.int64(in_len)) // np.int64(2 * out_len)


def nn_resize_u8(src: np.ndarray, out_w: int, out_h: int) -> np.ndarray:
    """Nearest-neighbour resize with the exact rational map above.

    Two independent axis maps — this is an anisotropic stretch (the
    STRETCH convention).  Every index is an integer expression; there is
    no interpolation, no weighting and no float intermediate, so the
    result is bit-identical on any platform that computes the integer
    product/quotient the same way.
    """
    arr = np.asarray(src)
    if arr.ndim != 2:
        raise ValueError("nn_resize_u8 expects a 2-D array")
    in_h, in_w = arr.shape
    sy = nn_source_indices(in_h, int(out_h))
    sx = nn_source_indices(in_w, int(out_w))
    # np.ix_ does the outer-product take in one shot (no row/col temp).
    return arr[np.ix_(sy, sx)]


# ════════════════════════════════════════════════════════════════════════
# 5. The glyph pipeline
# ════════════════════════════════════════════════════════════════════════
def _paste_centred(crop: np.ndarray, canvas: int) -> np.ndarray:
    """Centre `crop` in a canvas×canvas zero field (integer offsets)."""
    h, w = crop.shape
    y0 = (canvas - h) // 2
    x0 = (canvas - w) // 2
    out = np.zeros((canvas, canvas), dtype=crop.dtype)
    out[y0:y0 + h, x0:x0 + w] = crop
    return out


def _scale_to_fit(crop: np.ndarray, target: int) -> np.ndarray:
    """Integer-only aspect-preserving scale so the LONGER side == target.

    Nearest-neighbour throughout, with the same exact rational map, so the
    result stays float-free.  new_len = round(len * target / long_side) is
    computed with integer round-half-up: (2*len*target + long) // (2*long).
    """
    h, w = crop.shape
    long_side = max(h, w)
    if long_side <= 0:
        return crop
    nh = (2 * h * target + long_side) // (2 * long_side)
    nw = (2 * w * target + long_side) // (2 * long_side)
    nh = max(1, min(target, nh))
    nw = max(1, min(target, nw))
    if (nh, nw) == (h, w):
        return crop
    return nn_resize_u8(crop, nw, nh)


def _canon_fit(out_size: int) -> int:
    """NORM_CANON's long-side target for this canvas (20 at 28x28)."""
    return _GLYPH_CANON_FIT if out_size == GLYPH_SIZE else max(
        1, (out_size * _GLYPH_CANON_FIT) // GLYPH_SIZE
    )


def _narrow_stroke_radius(
    ink: np.ndarray,
    crop_w: int,
    crop_h: int,
    out_size: int,
    convention: str,
) -> int:
    """Horizontal dilation radius for a thin single-stroke glyph, else 0.

    A segmented camera '1' is frequently a 1-2 px bar.  Its ink bbox is
    therefore much narrower than it is tall (w/h < 0.35), whereas the
    EMNIST '1' samples used for training are 3-4 px strokes.  After
    NORM_CANON shrinks the long side to 20 px, that camera stroke can end
    up 1 px wide — visually a hairline, and the model's `_background`
    class (built from noisy residuals) is a plausible reading for it.

    This measures the stroke BEFORE the resize and returns the radius of
    a horizontal min/max filter applied to the SOURCE crop, before the
    long-side fit.  Widening after the resize would be too late: nearest-
    neighbour downsampling can skip a 1 px source column entirely (a
    20/35 scale samples source columns 0, 2, 4, 6, 8, 10, ...), so a
    hairline can disappear instead of merely staying thin.  The target
    source width is the width that leaves _GLYPH_TARGET_STROKE pixels
    after the fit: `ceil(3 * long_side / fit)`.  The estimate is
    `total_ink / ink_height` — exact for a vertical bar and still the
    stroke width for a '1' with a small base.  NORM_STRETCH is excluded:
    it stretches the crop to both axes, so a narrow glyph is already a
    wide block and needs no help (and its aspect distortion is the
    convention's defining behaviour).

    Returns 0 for wide glyphs and for strokes already wide enough.
    Integer arithmetic only; the C mirror (sign_glyph.c) computes the
    same value.
    """
    if convention != NORM_CANON:
        return 0
    arr = np.asarray(ink, dtype=bool)
    if arr.ndim != 2 or arr.size == 0:
        return 0
    cols = np.flatnonzero(arr.any(axis=0))
    rows = np.flatnonzero(arr.any(axis=1))
    if cols.size == 0 or rows.size == 0:
        return 0
    bw = int(cols[-1] - cols[0] + 1)
    bh = int(rows[-1] - rows[0] + 1)
    if bw <= 0 or bh <= 0:
        return 0
    if bw * 100 > _GLYPH_NARROW_ASPECT_PCT * bh:
        return 0
    total = int(arr[int(rows[0]):int(rows[-1]) + 1,
                    int(cols[0]):int(cols[-1]) + 1].sum())
    if total <= 0:
        return 0
    # Round-half-up mean ink pixels per ink row = stroke width (integer).
    stroke = max(1, (2 * total + bh) // (2 * bh))
    long_side = max(int(crop_w), int(crop_h))
    if long_side <= 0:
        return 0
    fit = _canon_fit(int(out_size))
    # Source width that leaves at least 3 px after the long-side fit.
    target_src = (_GLYPH_TARGET_STROKE * long_side + fit - 1) // fit
    if stroke >= target_src:
        return 0
    return min(
        _GLYPH_MAX_STROKE_RADIUS,
        (target_src - stroke + 1) // 2,
    )


def _thicken_narrow_stroke(
    crop: np.ndarray, radius: int, polarity: int
) -> np.ndarray:
    """Widen a glyph's stroke horizontally by `radius` pixels each side.

    A 1-D min filter (INK_DARK: dark ink grows into the bright plate) or
    max filter (INK_BRIGHT: bright ink grows into the dark page).  The
    columns added on either side are filled with the plate tone (255 for
    dark ink, 0 for bright ink) so the filter has background to grow into;
    without them a tight crop would clip the expansion at its own edge.
    Only the horizontal axis is filtered — the glyph's height (and the
    CANON centre-of-mass shift) must stay unchanged.

    Integer comparisons only; sign_glyph.c mirrors this loop exactly.
    """
    r = int(radius)
    if r <= 0 or crop.size == 0:
        return crop
    arr = np.asarray(crop, dtype=np.uint8)
    h, w = arr.shape
    fill = 255 if int(polarity) != INK_BRIGHT else 0
    padded = np.full((h, w + 2 * r), fill, dtype=np.uint8)
    padded[:, r:r + w] = arr
    out = np.empty_like(padded)
    for x in range(padded.shape[1]):
        lo = max(0, x - r)
        hi = min(padded.shape[1], x + r + 1)
        window = padded[:, lo:hi]
        if int(polarity) == INK_BRIGHT:
            out[:, x] = window.max(axis=1)
        else:
            out[:, x] = window.min(axis=1)
    return out


def normalise_glyph(crop: np.ndarray, out_size: int = GLYPH_SIZE,
                    convention: str = NORM_STRETCH, *,
                    narrow_radius: int = 0,
                    polarity: int = INK_DARK) -> np.ndarray:
    """Fit an ink crop into an out_size x out_size canvas.

    NORM_STRETCH — bbox stretched onto the full grid (one factor per axis).
        The convention the brief specifies.  Its weakness is that the
        applied scale depends on the glyph's own aspect ratio; measured,
        it is 2.69x more sensitive to a +/-1 px bbox error than NORM_CANON
        and leaves a 2.5x higher residual when EMNIST-as-distributed is
        compared with real captures.

    NORM_CANON — MNIST/EMNIST convention: fit the longer side to 20 with
        the aspect ratio preserved, paste centred, then shift so the
        intensity centre of mass lands on ((out-1)/2, (out-1)/2) using an
        integer translation.  This is the ONLY convention that can be
        applied to EMNIST samples as they ship, and it is what makes the
        EMNIST-mixing strategy work at all.

    narrow_radius / polarity — before the CANON resize, thicken the
        source crop with `_thicken_narrow_stroke`, so a hairline camera
        '1' reaches the 3-4 px stroke width of its EMNIST counterparts
        (and cannot be skipped by the nearest-neighbour downscale).
        `preprocess_glyph` computes the radius from the ink mask (see
        `_narrow_stroke_radius`); callers that paste a mask for display
        pass the default 0 and are unchanged.  Ignored under NORM_STRETCH,
        where the anisotropic scale already widens a narrow glyph.
    """
    arr = np.asarray(crop)
    if arr.size == 0:
        return np.zeros((out_size, out_size), dtype=np.uint8)
    if convention == NORM_STRETCH:
        return nn_resize_u8(arr, out_size, out_size).astype(np.uint8)

    if convention != NORM_CANON:
        raise ValueError(f"unknown glyph convention: {convention!r}")

    fit = _canon_fit(out_size)
    # Widen BEFORE the resize.  Nearest-neighbour downsampling samples
    # every ~long_side/fit-th source column, so a 1 px stroke can be
    # skipped entirely; a 3-4 px source stroke always survives.
    if int(narrow_radius) > 0:
        arr = _thicken_narrow_stroke(arr, int(narrow_radius), int(polarity))
    scaled = _scale_to_fit(arr, fit)
    canvas = _paste_centred(scaled, out_size)

    # Integer centre-of-mass shift: cy = sum(y*I)/sum(I) computed in int64,
    # then compared against the target with integer rounding.  No float.
    field = canvas.astype(np.int64)
    mass = int(field.sum())
    if mass > 0:
        ys = np.arange(out_size, dtype=np.int64)[:, None]
        xs = np.arange(out_size, dtype=np.int64)[None, :]
        cy = int((field * ys).sum()) // mass      # integer division: floor
        cx = int((field * xs).sum()) // mass
        target = out_size // 2                     # (out-1)/2 == out//2 - ... see below
        # Target is (out_size - 1) / 2.  For even out_size that is a half
        # integer, so compare on the doubled lattice to stay in integers.
        dy = _round_half_up(2 * cy - (out_size - 1), 2)
        dx = _round_half_up(2 * cx - (out_size - 1), 2)
        if dy or dx:
            shifted = np.zeros_like(canvas)
            ys0 = max(0, dy)
            ys1 = min(out_size, out_size + dy)
            xs0 = max(0, dx)
            xs1 = min(out_size, out_size + dx)
            shifted[ys0:ys1, xs0:xs1] = canvas[ys0 - dy:ys1 - dy, xs0 - dx:xs1 - dx]
            canvas = shifted
    return canvas.astype(np.uint8)


def _round_half_up(num: int, den: int) -> int:
    """Integer round-half-up division for positive and negative `num`."""
    if num >= 0:
        return (2 * num + den) // (2 * den)
    return -((-2 * num + den) // (2 * den))


def preprocess_glyph(
    src: np.ndarray,
    out_size: int = GLYPH_SIZE,
    polarity: int = INK_DARK,
    apply_segmentation: bool = True,
    min_run: int = MIN_GLYPH_RUN,
    thresh: Optional[int] = None,
    convention: str = NORM_STRETCH,
    dark_thr: int = 0,
) -> Dict[str, Any]:
    """Full glyph transform.  Returns a dict (never raises on bad input).

    src                : RGB (h,w,3) camera frame, or a grayscale (h,w)
                         crop.  RGB goes through integer BT.601 first.
    polarity           : INK_DARK for captures, INK_BRIGHT for EMNIST.
    apply_segmentation : True  -> vertical-projection segmentation finds
                         the glyph (captures: the glyph sits somewhere in
                         a larger frame).
                         False -> the input is already an isolated glyph
                         (EMNIST) and the projection stage is skipped.
                         EMNIST is the one case where the stage is
                         provably useless: measured 1 column run in
                         124/124 samples.
    thresh             : None -> exact Otsu.  An explicit value pins the
                         threshold, which is what the golden-vector
                         conformance suite uses to isolate the stages.
                         thresh=0 disables binarisation entirely (ink
                         stays a real-valued mask driven by `polarity`),
                         which is how an anti-aliased grayscale glyph is
                         fed to NORM_CANON without a hard edge.
    dark_thr           : 0 -> no floor (bit-identical to the pre-existing
                         transform, and what the device's C port always
                         uses).  A positive value drops every pixel darker
                         than it from the ink mask, i.e. the sign's lower
                         luminance bound.  Applied on TOP of the polarity
                         split rather than instead of it, so it only ever
                         tightens the mask and is monotone for both
                         polarities.  The editor's "Dark Thr" slider is
                         this number; the paired "Lum Thr" slider is
                         `thresh`.  See RecordController._preprocess_
                         image_png_ocr for the gate that decides when the
                         two sliders override Otsu at all.

    Returns {"ink", "glyph", "thresh", "box", "boxes", "ok", "empty"}
    where `glyph` is a uint8 (out_size, out_size) GRAYSCALE array ready for
    the model input, and `ink` is the bool mask that produced the box.
    Never raises: a failed transform returns glyph=zeros, ok=False.
    """
    out: Dict[str, Any] = {
        "ink": None, "glyph": None, "thresh": 0, "box": None,
        "boxes": [], "ok": False, "empty": True,
    }
    try:
        gray = rgb_to_lum_u8(src)
        if gray.shape[0] < 1 or gray.shape[1] < 1:
            return out

        t = int(otsu_threshold_u8(gray)) if thresh is None else int(thresh)
        out["thresh"] = t

        if t == 0 and thresh is None:
            # No valid split existed (flat/empty frame).  There is no ink
            # to find; report it rather than inventing one.
            out["ink"] = np.zeros(gray.shape, dtype=bool)
            out["glyph"] = np.zeros((out_size, out_size), dtype=np.uint8)
            return out

        ink = binarize_ink(gray, t, polarity=polarity)
        # The editor's "Dark Thr" floor.  Applied here, after the polarity
        # split, rather than inside binarize_ink: that function is the single
        # comparison the C port mirrors, and dark_thr is a host-side editor
        # knob the device never sets.  Guarded on > 0 so the default path
        # stays bit-identical to that port.
        if dark_thr > 0:
            ink = ink & (gray >= np.uint8(min(255, int(dark_thr))))
        out["ink"] = ink

        if not bool(ink.any()):
            out["glyph"] = np.zeros((out_size, out_size), dtype=np.uint8)
            return out

        if apply_segmentation:
            boxes = segment_glyph_boxes(ink, min_run=min_run)
            box = segment_glyph_box(ink, min_run=min_run)
        else:
            h, w = ink.shape
            box = (0, 0, w, h)
            boxes = [box]
        out["boxes"] = boxes
        out["box"] = box
        if box is None:
            out["glyph"] = np.zeros((out_size, out_size), dtype=np.uint8)
            return out

        x1, y1, x2, y2 = box
        # Normalise the GRAYSCALE crop, not the binary mask.  The ink mask
        # is only ever used to FIND the box; resizing the mask itself
        # discards every anti-aliased edge pixel and measurably fragments
        # thin strokes (16.1% of samples lost ink mass when the binary
        # mask was resized instead of the grayscale).
        crop = gray[y1:y2, x1:x2]
        if crop.size == 0:
            out["glyph"] = np.zeros((out_size, out_size), dtype=np.uint8)
            return out

        # A narrow single-stroke glyph ('1') keeps its 1-2 px stroke through
        # NORM_CANON while the EMNIST '1' samples are 3-4 px.  Measure the
        # ink inside THIS crop and pass a horizontal dilation radius so the
        # model sees the stroke width it was trained on.
        ink_crop = ink[y1:y2, x1:x2]
        narrow_radius = _narrow_stroke_radius(
            ink_crop, int(x2) - int(x1), int(y2) - int(y1),
            int(out_size), convention,
        )
        glyph = normalise_glyph(
            np.ascontiguousarray(crop),
            out_size=out_size,
            convention=convention,
            narrow_radius=narrow_radius,
            polarity=polarity,
        )
        out["glyph"] = glyph
        out["ok"] = True
        out["empty"] = False
        return out
    except Exception:
        out["glyph"] = np.zeros((out_size, out_size), dtype=np.uint8)
        return out


def preprocess_glyph_emnist(src: np.ndarray, out_size: int = GLYPH_SIZE,
                            convention: str = NORM_CANON) -> Dict[str, Any]:
    """EMNIST samples: bright ink on dark, already isolated and centred.

    Same code path as captures — only polarity and the segmentation flag
    change.  Deliberately NOT a separate implementation: a second
    implementation of the same transform is exactly the failure mode this
    module exists to prevent.
    """
    return preprocess_glyph(
        src, out_size=out_size, polarity=INK_BRIGHT,
        apply_segmentation=False, convention=convention,
    )


def preprocess_glyph_capture(src: np.ndarray, out_size: int = GLYPH_SIZE,
                             polarity: int = INK_DARK,
                             convention: str = NORM_STRETCH) -> Dict[str, Any]:
    """Camera capture: dark ink on a bright page, glyph somewhere in frame."""
    return preprocess_glyph(
        src, out_size=out_size, polarity=polarity,
        apply_segmentation=True, convention=convention,
    )


# ════════════════════════════════════════════════════════════════════════
# 7. Sign sequences — the multi-glyph front-end
# ════════════════════════════════════════════════════════════════════════
def _manual_roi_pixels(roi: Any, h: int, w: int) -> Optional[Tuple[int, int, int, int]]:
    """Normalize the editor's 0..1 ROI (or an explicit pixel box)."""
    try:
        vals = [float(v) for v in roi]
    except Exception:
        return None
    if len(vals) != 4:
        return None
    if all(0.0 <= v <= 1.0 for v in vals):
        x1, y1, x2, y2 = (
            int(round(vals[0] * w)),
            int(round(vals[1] * h)),
            int(round(vals[2] * w)),
            int(round(vals[3] * h)),
        )
    else:
        x1, y1, x2, y2 = [int(round(v)) for v in vals]
    x1 = max(0, min(w - 1, x1))
    y1 = max(0, min(h - 1, y1))
    x2 = max(x1 + 1, min(w, x2))
    y2 = max(y1 + 1, min(h, y2))
    return (x1, y1, x2, y2)


def _cv2_module():
    """OpenCV is used only for the host-side sign-board locator.

    The integer glyph/split path does not depend on it; importing lazily keeps
    the parity tests and the device mirror free of an OpenCV dependency.
    """
    try:
        import cv2  # type: ignore
        return cv2
    except Exception:
        return None


def _ink_cluster_roi(
    gray: np.ndarray,
    dark_thr: int,
    lum_thr: int,
) -> Optional[Dict[str, Any]]:
    """Find the three interior character blobs at the user's Dark/Lum window.

    This is the hardware path: the user tunes Dark/Lum until the sign's ink
    stands alone, then the frame still contains fingers, the plastic frame,
    and background clutter.  We take the threshold window they chose, drop
    components touching the outer frame / too large to be a glyph / too
    small to be ink, and look for a horizontal row of three similarly sized
    interior blobs.  The union of that row is the ROI; the 3-way split then
    runs only inside it.
    """
    cv2 = _cv2_module()
    if cv2 is None:
        return None
    h, w = int(gray.shape[0]), int(gray.shape[1])
    dark = max(0, min(255, int(dark_thr)))
    lum = max(0, min(255, int(lum_thr)))
    if lum < dark:
        dark, lum = lum, dark
    ink = ((gray >= dark) & (gray <= lum)).astype(np.uint8)
    try:
        n, _labels, stats, _centroids = cv2.connectedComponentsWithStats(ink, 8)
    except Exception:
        return None
    # Discard anything that reaches into the outer 6 % of the frame: fingers,
    # the plastic frame and bottom-left border clutter live there and used to
    # drag the ROI left (and clip '2' on the right).
    margin = max(2, int(round(min(h, w) * 0.06)))
    # A thin '1' can be only ~35-60 px in a 160 frame; the old 0.15 %
    # floor (38 px) dropped it and left only two candidates.  Keep the
    # floor low enough for a real stroke, high enough for specks.
    min_area = max(6, int(round(h * w * 0.0004)))
    min_height = max(3, int(round(h * 0.10)))
    candidates: List[Dict[str, float]] = []
    for i in range(1, n):
        x, y, cw, ch, area = (int(v) for v in stats[i])
        if area < min_area or cw < 2 or ch < min_height:
            continue
        if x <= margin or y <= margin or x + cw >= w - margin or y + ch >= h - margin:
            continue
        if cw > 0.60 * w or ch > 0.60 * h:
            continue
        if area > 0.40 * w * h:
            continue
        # Narrow does NOT mean noise: a vertical '1' stroke is legitimately
        # 2-4 px wide, while the old 0.15 floor discarded exactly that
        # component and left the trio without its middle character.
        aspect = float(cw) / float(max(1, ch))
        if aspect < 0.03 or aspect > 6.0:
            continue
        fill = float(area) / float(max(1, cw * ch))
        if fill < 0.08:
            continue
        candidates.append({
            "x": x, "y": y, "w": cw, "h": ch, "area": area,
            "cx": x + cw / 2.0, "cy": y + ch / 2.0,
        })
    if not candidates:
        return None
    # Keep the most substantial blobs; clutter tends to be small.  16 gives
    # a hairline '1' room even when several noise blobs outrank it by area.
    candidates.sort(key=lambda c: -c["area"])
    candidates = candidates[:16]
    candidates.sort(key=lambda c: c["cx"])

    best = None
    for i in range(len(candidates) - 2):
        for j in range(i + 1, len(candidates) - 1):
            for k in range(j + 1, len(candidates)):
                trio = (candidates[i], candidates[j], candidates[k])
                gaps = [
                    trio[1]["x"] - (trio[0]["x"] + trio[0]["w"]),
                    trio[2]["x"] - (trio[1]["x"] + trio[1]["w"]),
                ]
                if min(gaps) < 0:
                    continue
                heights = sorted(c["h"] for c in trio)
                med_h = heights[1]
                if max(c["cy"] for c in trio) - min(c["cy"] for c in trio) > max(7, 0.55 * med_h):
                    continue
                if heights[2] > 2.2 * max(1, heights[0]):
                    continue
                # Do NOT compare widths: the middle character of "A12" is a
                # legitimately narrow stroke.  Width is only bounded against
                # the row height (a huge blob is not a character) and gaps
                # are measured against the height, which all three share.
                widths = [c["w"] for c in trio]
                if max(widths) > 3.0 * max(1, med_h):
                    continue
                if max(gaps) > 2.0 * max(1, med_h):
                    continue
                x1 = min(c["x"] for c in trio)
                y1 = min(c["y"] for c in trio)
                x2 = max(c["x"] + c["w"] for c in trio)
                y2 = max(c["y"] + c["h"] for c in trio)
                if (x2 - x1) < 0.22 * w or (y2 - y1) < 0.12 * h:
                    continue
                if (x2 - x1) > 0.92 * w:
                    continue
                cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
                dist = float(np.hypot(cx - w / 2.0, cy - h / 2.0))
                center = max(0.0, 1.0 - 0.5 * dist / max(1.0, 0.5 * np.hypot(w, h)))
                score = float(sum(c["area"] for c in trio)) * center
                if best is None or score > best["score"]:
                    best = {
                        "score": score,
                        "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                        "med_h": float(med_h),
                    }
    if best is None:
        return None
    # Use ONLY the three selected character blobs.  The old extra sweep over
    # "anything in the same band" pulled in bottom-left clutter and made the
    # box lopsided.  Then apply one uniform pad (10 % of the cluster's longer
    # side, equal on all four edges) so the box is symmetric and leaves a
    # little board margin around the glyphs without clipping '2' on the right.
    x1, y1, x2, y2 = best["x1"], best["y1"], best["x2"], best["y2"]
    cw = max(1, int(x2) - int(x1))
    ch = max(1, int(y2) - int(y1))
    pad = max(3, int(round(0.10 * max(cw, ch))))
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(w, x2 + pad)
    y2 = min(h, y2 + pad)
    return {
        "box": [int(x1), int(y1), int(x2), int(y2)],
        "method": "ink_cluster",
        "score": float(best["score"]),
        "threshold": int(lum),
        "area_frac": float((x2 - x1) * (y2 - y1)) / float(max(1, h * w)),
    }


# ── Interior ink ROI (sign search, mirrored by sign_glyph.c) ────────────
# The device cannot run OpenCV or the full three-blob heuristic.  This is a
# deterministic component search both sides can reproduce: 8-connected
# components, a 6 % outer margin and huge-frame filters, then the horizontal
# band with the most components (the row of characters).  The union of that
# band plus one uniform 10 % pad is the ROI.  `preprocess_sign_sequence(...,
# find_roi=True)` and `TFLite_Sign/sign_glyph.c` use it before the split.
_INTERIOR_MARGIN_PCT = 6
_INTERIOR_MAX_COMPONENTS = 256


def _interior_ink_components(ink: np.ndarray) -> List[Dict[str, int]]:
    """Filtered 8-connected ink components, in raster-scan label order.

    Same integer bounds as the C port: an outer margin of 6 % of the shorter
    side, tiny specks, components wider than 85 % or taller than 60 % of the
    frame, and blobs covering >40 % of the area are discarded.  Thin
    single-stroke characters are kept; only frame-scale debris is removed.
    """
    cv2 = _cv2_module()
    arr = np.asarray(ink, dtype=bool)
    if cv2 is None or arr.ndim != 2 or arr.size == 0:
        return []
    h, w = int(arr.shape[0]), int(arr.shape[1])
    margin = max(2, (min(h, w) * _INTERIOR_MARGIN_PCT + 50) // 100)
    min_area = max(6, (h * w * 4 + 5000) // 10000)
    try:
        n, _labels, stats, _centroids = cv2.connectedComponentsWithStats(
            arr.astype(np.uint8), 8
        )
    except Exception:
        return []
    out: List[Dict[str, int]] = []
    for i in range(1, n):
        x, y, cw, ch, area = (int(v) for v in stats[i])
        if area < min_area:
            continue
        if x <= margin or y <= margin or x + cw >= w - margin or y + ch >= h - margin:
            continue
        if cw * 100 > 85 * w or ch * 100 > 60 * h:
            continue
        if area * 100 > 40 * w * h:
            continue
        out.append({
            "x": x, "y": y, "w": cw, "h": ch, "area": area,
        })
        if len(out) >= _INTERIOR_MAX_COMPONENTS:
            break
    return out


def _choose_interior_band(
    candidates: List[Dict[str, int]], w: int, h: int
) -> Optional[Tuple[Tuple[int, int, int, int], Tuple[int, int, int, int]]]:
    """(padded ROI, tight band) for the horizontal row of components.

    A frame line is a lone component; the character row is several (or one
    for a single-character sample).  Ties go to the larger total ink area,
    then to the earlier raster component, so the answer is deterministic and
    the C port can reproduce it exactly.  The tight band is what survives to
    the split (the pad must stay white); the padded ROI is the box the UI
    draws and the OOD sign_pct measures.
    """
    if not candidates:
        return None
    best_idx = 0
    best_key: Optional[Tuple[int, int]] = None
    for i, anchor in enumerate(candidates):
        ah = max(1, int(anchor["h"]))
        acy2 = 2 * int(anchor["y"]) + ah
        band2 = 2 * ah  # +/- 1.0 * height
        count = 0
        total = 0
        for c in candidates:
            cy2 = 2 * int(c["y"]) + int(c["h"])
            if abs(cy2 - acy2) > band2:
                continue
            count += 1
            total += int(c["area"])
        key = (count, total)
        if best_key is None or key > best_key:
            best_key = key
            best_idx = i
    anchor = candidates[best_idx]
    ah = max(1, int(anchor["h"]))
    acy2 = 2 * int(anchor["y"]) + ah
    band2 = 2 * ah
    x1 = y1 = 1 << 30
    x2 = y2 = -1
    for c in candidates:
        cy2 = 2 * int(c["y"]) + int(c["h"])
        if abs(cy2 - acy2) > band2:
            continue
        x1 = min(x1, int(c["x"]))
        y1 = min(y1, int(c["y"]))
        x2 = max(x2, int(c["x"]) + int(c["w"]))
        y2 = max(y2, int(c["y"]) + int(c["h"]))
    if x2 <= x1 or y2 <= y1:
        return None
    cw = x2 - x1
    ch = y2 - y1
    pad = max(2, (max(cw, ch) * 10 + 50) // 100)
    tight = (int(x1), int(y1), int(x2), int(y2))
    roi = (
        max(0, x1 - pad),
        max(0, y1 - pad),
        min(int(w), x2 + pad),
        min(int(h), y2 + pad),
    )
    return roi, tight


def find_interior_ink_roi(
    src: np.ndarray,
    *,
    thresh: Optional[int] = None,
    dark_thr: int = 0,
    polarity: int = INK_DARK,
) -> Dict[str, Any]:
    """Tight interior ROI around the character row, for the Edit worktable.

    Applies the same binarisation the glyph front-end uses (explicit
    `thresh`, else exact Otsu), optionally the Dark Thr floor, then the
    component/band search above.  Returns `{box, method, sign_pct, threshold}`;
    an empty result reports the full frame and method "none" so callers can
    fall back rather than drawing a bogus one-pixel box.
    """
    gray = rgb_to_lum_u8(src)
    if gray.ndim != 2 or gray.size == 0:
        return {"box": [0, 0, 1, 1], "method": "empty",
                "sign_pct": 0.0, "threshold": 0}
    h, w = int(gray.shape[0]), int(gray.shape[1])
    t = int(otsu_threshold_u8(gray)) if thresh is None else int(thresh)
    if t <= 0 and thresh is None:
        ink = np.zeros(gray.shape, dtype=bool)
    else:
        ink = binarize_ink(gray, t, polarity=polarity)
        if int(dark_thr) > 0:
            ink = ink & (gray >= np.uint8(min(255, int(dark_thr))))
    band = _choose_interior_band(_interior_ink_components(ink), w, h)
    if band is None:
        return {"box": [0, 0, w, h], "tight_box": None, "method": "none",
                "ink": ink, "sign_pct": float(ink.mean() * 100.0) if ink.size else 0.0,
                "threshold": int(t)}
    (x1, y1, x2, y2), (tx1, ty1, tx2, ty2) = band
    # Keep only the selected character band; the pad around it must stay
    # pure white in the user-facing crop (a frame line crossing the pad used
    # to leave a black corner there).
    selected = np.zeros_like(ink)
    selected[ty1:ty2, tx1:tx2] = ink[ty1:ty2, tx1:tx2]
    roi_ink = selected[y1:y2, x1:x2]
    pct = float(roi_ink.mean() * 100.0) if roi_ink.size else 0.0
    return {"box": [int(x1), int(y1), int(x2), int(y2)],
            "tight_box": [int(tx1), int(ty1), int(tx2), int(ty2)],
            "method": "interior_ink", "ink": selected, "sign_pct": pct,
            "threshold": int(t)}


def find_sign_roi(
    src: np.ndarray,
    manual_roi: Any = None,
    *,
    dark_thr: Optional[int] = None,
    lum_thr: Optional[int] = None,
    min_area_frac: float = 0.015,
) -> Dict[str, Any]:
    """Locate the bright road-sign plate inside a clutter frame.

    The device frame is a whole scene: a bright sign plate on a dark plastic
    frame, with wheels/sensors/desk clutter around it.  Running Otsu plus a
    vertical projection over that whole frame finds the *frame edge* and the
    clutter, not the three characters.  This first stage finds the inner
    bright plate; the caller crops it before the projection split.

    Returns a dict with:
      box        [x1, y1, x2, y2] in source pixels (exclusive end)
      method     "manual" | "ink_cluster" | "bright_plate" | "center_fallback"
      score      candidate score (0 for manual/fallback)
      threshold  bright-mask threshold used (unused for manual)
      area_frac  plate area / frame area for a detected plate
    """
    gray = rgb_to_lum_u8(src)
    if gray.ndim != 2 or gray.shape[0] < 1 or gray.shape[1] < 1:
        return {"box": [0, 0, 1, 1], "method": "empty", "score": 0.0,
                "threshold": 0, "area_frac": 0.0}
    h, w = int(gray.shape[0]), int(gray.shape[1])

    if manual_roi is not None:
        box = _manual_roi_pixels(manual_roi, h, w)
        if box is not None:
            return {"box": [int(v) for v in box], "method": "manual", "score": 1.0,
                    "threshold": 0, "area_frac": float((box[2] - box[0]) * (box[3] - box[1])) / float(max(1, h * w))}

    # Hardware path: when the user has tuned Dark/Lum, those two sliders are
    # the best description of "ink".  Find the three interior character blobs
    # at that window before falling back to the bright-plate heuristic.
    if dark_thr is not None and lum_thr is not None:
        ink_roi = _ink_cluster_roi(gray, int(dark_thr), int(lum_thr))
        if ink_roi is not None:
            return ink_roi

    cv2 = _cv2_module()
    if cv2 is not None:
        try:
            blur = cv2.GaussianBlur(gray, (5, 5), 0)
            t, _ = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            t = int(t)
            # A flat frame or a weird Otsu split: fall back to a high
            # percentile so "bright plate" still means something.
            if t < 16:
                t = int(np.percentile(blur, 70))
            if t > 245:
                t = int(np.percentile(blur, 55))
            bright = (blur >= t).astype(np.uint8) * 255
            # Small closing only: a kernel wide enough to bridge an 8 px
            # dark sign frame would merge the plate with the desk behind it.
            k = max(3, int(round(min(h, w) * 0.02)))
            if k % 2 == 0:
                k += 1
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
            closed = cv2.morphologyEx(bright, cv2.MORPH_CLOSE, kernel, iterations=1)
            cleaned = cv2.morphologyEx(closed, cv2.MORPH_OPEN,
                                       np.ones((3, 3), np.uint8), iterations=1)
            n, _labels, stats, centroids = cv2.connectedComponentsWithStats(cleaned, 8)
            best = None
            for i in range(1, n):
                x, y, cw, ch, area = (int(v) for v in stats[i])
                if cw <= 0 or ch <= 0:
                    continue
                area_frac = float(area) / float(max(1, h * w))
                if area_frac < min_area_frac:
                    continue
                if cw < max(8, int(round(0.10 * w))) or ch < max(8, int(round(0.10 * h))):
                    continue
                aspect = float(cw) / float(max(1, ch))
                if aspect < 0.20 or aspect > 5.0:
                    continue
                fill = float(area) / float(max(1, cw * ch))
                if fill < 0.25:
                    continue
                cx, cy = float(centroids[i][0]), float(centroids[i][1])
                dist = float(np.hypot(cx - w / 2.0, cy - h / 2.0))
                center = max(0.0, 1.0 - 0.5 * dist / max(1.0, 0.5 * np.hypot(w, h)))
                border = x <= 0 or y <= 0 or x + cw >= w or y + ch >= h
                # A tight capture (the 96x96 template sign) is one bright
                # plate covering the whole frame; a cluttered scene's desk
                # is also border-touching but is split by the dark sign
                # frame / clutter.  Accept a border component only when it
                # really is the frame-filling plate; otherwise prefer the
                # enclosed candidate and fall back to the centre box.
                if border and area_frac < 0.80:
                    continue
                score = area_frac * (0.40 + 0.60 * fill) * center
                if border:
                    score *= 0.90
                if area_frac > 0.90:
                    score *= 0.95
                if best is None or score > best["score"]:
                    pad = max(1, int(round(0.025 * min(cw, ch))))
                    x1 = max(0, x - pad)
                    y1 = max(0, y - pad)
                    x2 = min(w, x + cw + pad)
                    y2 = min(h, y + ch + pad)
                    best = {
                        "box": [x1, y1, x2, y2],
                        "method": "bright_plate",
                        "score": float(score),
                        "threshold": t,
                        "area_frac": area_frac,
                    }
            if best is not None:
                return best
        except Exception:
            pass

    # Deterministic fallback: the sign is normally centred; a 70 % centre
    # box beats returning [0, 0, 1, 1] and feeding the projection the desk.
    x1 = int(round(0.15 * w))
    y1 = int(round(0.15 * h))
    x2 = int(round(0.85 * w))
    y2 = int(round(0.85 * h))
    return {"box": [x1, y1, x2, y2], "method": "center_fallback", "score": 0.0,
            "threshold": 0, "area_frac": float((x2 - x1) * (y2 - y1)) / float(max(1, h * w))}


def clean_sign_ink(
    ink: np.ndarray,
    *,
    margin: Optional[int] = None,
    min_area: Optional[int] = None,
) -> np.ndarray:
    """Remove border-touching frame lines and tiny specks from a sign ROI.

    The sign's plastic frame is a long, dark component touching the ROI edge;
    if it survives into the vertical projection it becomes the widest "run"
    and steals one of the three slots.  The genuine characters are interior
    components.  A speck of sensor noise is also removed by area.

    If the cleanup would remove ALL ink (a genuinely tight sign crop), the
    original mask is returned: an empty slot is worse than a noisy one.
    """
    arr = np.asarray(ink, dtype=bool)
    if arr.ndim != 2 or arr.size == 0:
        return arr
    h, w = arr.shape
    if margin is None:
        margin = max(1, int(round(min(h, w) * 0.02)))
    if min_area is None:
        min_area = max(4, int(round(h * w * 0.0005)))
    cv2 = _cv2_module()
    if cv2 is None:
        out = arr.copy()
        if margin > 0:
            out[:margin, :] = False
            out[-margin:, :] = False
            out[:, :margin] = False
            out[:, -margin:] = False
        return out
    try:
        n, labels, stats, _ = cv2.connectedComponentsWithStats(arr.astype(np.uint8), 8)
        keep = np.zeros_like(arr, dtype=bool)
        for i in range(1, n):
            x, y, cw, ch, area = (int(v) for v in stats[i])
            if area < min_area:
                continue
            touches = (
                (1 if x <= margin else 0)
                + (1 if y <= margin else 0)
                + (1 if x + cw >= w - margin else 0)
                + (1 if y + ch >= h - margin else 0)
            )
            if touches:
                # The plastic frame is a thin line/ring on the ROI edge; it
                # usually touches 2-4 sides.  A real glyph can sit close to
                # one edge in a tight capture, so a single-side,
                # character-sized component is kept.
                thin = max(3, int(round(min(h, w) * 0.04)))
                if touches >= 2 or cw <= thin or ch <= thin:
                    continue
            keep |= (labels == i)
        if not keep.any():
            return arr
        return keep
    except Exception:
        return arr


def preprocess_sign_sequence(
    src: np.ndarray,
    sign_sequence: Any = None,
    reading_order: str = "ltr",
    out_size: int = GLYPH_SIZE,
    polarity: int = INK_DARK,
    apply_segmentation: bool = True,
    convention: str = NORM_CANON,
    thresh: Optional[int] = None,
    dark_thr: int = 0,
    clean_border: bool = False,
    border_margin: Optional[int] = None,
    find_roi: bool = False,
) -> Dict[str, Any]:
    """One sign frame -> one 28x28 glyph PER CHARACTER of the label.

    The single-glyph path asks "where is the glyph?"; this asks "where is
    each of the three glyphs, in the order the label is read?".  Everything
    downstream of the split is the SAME transform — `preprocess_glyph` is
    called once per slot on the slot's own crop — so a model trained on
    single characters is being fed exactly the distribution it was trained
    on, and there is no second normalisation implementation to keep in
    step with the device port.

    reading_order is applied AFTER the split, not before: the split is
    physical (left to right across the frame), and "rtl" means the leftmost
    glyph is the LAST character of the label, so slot 0 must come from the
    rightmost crop.  Applying it to the split rather than to the image
    keeps a mirror out of the pipeline entirely — mirroring the frame would
    hand the model mirrored glyphs, which is a different, wrong, input.

    Returns a dict (never raises):
      slots      : one dict per label character, in LABEL order —
                   {char, glyph, ink, box, box_norm, ok, empty}
      glyphs     : [np.ndarray (out_size, out_size)] in label order
      sign_box   : union bbox of every glyph, in SOURCE pixels
      sign_box_norm : the same box as (x1,y1,x2,y2) fractions of the frame
      ink        : the bool mask the split was taken from
      thresh     : the binarisation threshold actually used
      ok         : True when at least one slot found ink
    """
    seq = normalize_sign_sequence(sign_sequence)
    n = len(seq)
    order = str(reading_order or "ltr").strip().lower()
    if order not in ("ltr", "rtl"):
        order = "ltr"

    blank_glyph = np.zeros((out_size, out_size), dtype=np.uint8)
    out: Dict[str, Any] = {
        "slots": [],
        "glyphs": [blank_glyph.copy() for _ in range(n)],
        "sign_box": None,
        "sign_box_norm": None,
        "roi_box": None,
        "roi_method": "input",
        "sign_pct": 0.0,
        "ink": None,
        "thresh": 0,
        "ok": False,
        "empty": True,
        "reading_order": order,
        "sign_sequence": list(seq),
    }
    try:
        gray = rgb_to_lum_u8(src)
        if gray.ndim != 2 or gray.shape[0] < 1 or gray.shape[1] < 1:
            return out
        h, w = int(gray.shape[0]), int(gray.shape[1])

        t = int(otsu_threshold_u8(gray)) if thresh is None else int(thresh)
        out["thresh"] = t
        if t == 0 and thresh is None:
            # No valid split existed (flat frame).  Same answer as the
            # single-glyph path: there is no ink, so there is nothing to
            # segment — report the frame split geometrically and stop.
            ink = np.zeros(gray.shape, dtype=bool)
        else:
            ink = binarize_ink(gray, t, polarity=polarity)
            if dark_thr > 0:
                ink = ink & (gray >= np.uint8(min(255, int(dark_thr))))
            # The component ROI search below IS the border cleanup when it is
            # enabled; running the OpenCV component filter as well would make
            # the Python side differ from the C port (which cannot run cv2).
            if clean_border and not find_roi:
                ink = clean_sign_ink(ink, margin=border_margin)

        if find_roi and apply_segmentation:
            components = _interior_ink_components(ink)
            # A sign needs more than one character component; a single
            # component means the frame is a tight capture (or the characters
            # touch), where cropping to it could discard edge glyphs.  Fall
            # back to the whole frame and let the split reconcile.
            band = _choose_interior_band(components, w, h) if len(components) >= 2 else None
            if band is not None:
                roi, tight = band
                rx1, ry1, rx2, ry2 = roi
                tx1, ty1, tx2, ty2 = tight
                # Keep only the selected character components; the padded
                # ROI is white around them so a frame line crossing the pad
                # cannot become a fourth projection run.
                band_ink = np.zeros_like(ink)
                band_ink[ty1:ty2, tx1:tx2] = ink[ty1:ty2, tx1:tx2]
                ink = band_ink
                out["roi_box"] = [int(rx1), int(ry1), int(rx2), int(ry2)]
                out["roi_method"] = "interior_ink"
        if out["roi_box"] is None:
            out["roi_box"] = [0, 0, int(w), int(h)]
        rx1, ry1, rx2, ry2 = (int(v) for v in out["roi_box"])
        roi_ink = ink[ry1:ry2, rx1:rx2]
        out["sign_pct"] = (
            float(roi_ink.mean() * 100.0) if roi_ink.size else 0.0
        )
        out["ink"] = ink

        boxes: List[Tuple[int, int, int, int]]
        if apply_segmentation:
            boxes = split_sign_glyph_boxes(ink, count=n)
        else:
            # The frame is already the sign (a tight capture).  Slice it
            # into n equal bands — the caller has told us segmentation has
            # nothing to find.
            boxes = _equal_split_boxes(0, w, 0, h, n)
        if len(boxes) != n:
            return out

        if order == "rtl":
            boxes = list(reversed(boxes))

        for idx, (x1, y1, x2, y2) in enumerate(boxes):
            # Give every glyph a little board context.  A thin '1' segmented
            # to its stroke-only bbox (2-3 px wide) loses the light plate
            # around it, and the resulting mostly-black tensor can score as
            # `_background`; a 2 px margin restores the context the model
            # saw during training.  The margin stays inside the ROI and the
            # gap between characters is normally wider than 4 px.
            bw = max(1, int(x2) - int(x1))
            bh = max(1, int(y2) - int(y1))
            pad = max(2, (min(bw, bh) * 8 + 50) // 100)
            px1 = max(0, int(x1) - pad)
            py1 = max(0, int(y1) - pad)
            px2 = min(w, int(x2) + pad)
            py2 = min(h, int(y2) + pad)
            crop = np.ascontiguousarray(gray[py1:py2, px1:px2])
            if crop.size == 0:
                res: Dict[str, Any] = {"glyph": blank_glyph.copy(), "ok": False, "empty": True}
            else:
                # Segmentation is OFF for the slot: the padded box already
                # contains one glyph plus its plate context; re-running the
                # projection inside it could only re-find the same run — or,
                # on a slot whose ink is a fragment, drop it below `min_run`
                # and blank the slot.
                res = preprocess_glyph(
                    crop,
                    out_size=int(out_size),
                    polarity=polarity,
                    apply_segmentation=False,
                    convention=convention,
                    thresh=thresh,
                    dark_thr=dark_thr,
                )
            glyph = np.asarray(res["glyph"], dtype=np.uint8)
            out["glyphs"][idx] = glyph
            # User-facing thumbnail: BLACK ink on a PURE WHITE 28x28 canvas.
            # The model input (`glyph`) keeps its training polarity/padding;
            # this display variant is only what the Preview slot chip shows.
            slot_ink = res.get("ink")
            if slot_ink is not None and np.asarray(slot_ink).size:
                mask = (np.asarray(slot_ink, dtype=np.uint8) * 255)
                display_mask = normalise_glyph(mask, out_size=int(out_size), convention=convention)
                display = (255 - display_mask).astype(np.uint8)
            else:
                display = np.full((int(out_size), int(out_size)), 255, dtype=np.uint8)
            out["slots"].append(
                {
                    "char": str(seq[idx]),
                    "glyph": glyph,
                    "display": display,
                    "ink": res.get("ink"),
                    "box": [int(px1), int(py1), int(px2), int(py2)],
                    "box_norm": [
                        float(px1) / float(w),
                        float(py1) / float(h),
                        float(px2) / float(w),
                        float(py2) / float(h),
                    ],
                    "ink_box": [int(x1), int(y1), int(x2), int(y2)],
                    "ok": bool(res.get("ok")),
                    "empty": bool(res.get("empty", True)),
                }
            )

        union = _union_box(boxes[0], boxes[-1])
        out["sign_box"] = [int(v) for v in union]
        out["sign_box_norm"] = [
            float(union[0]) / float(w),
            float(union[1]) / float(h),
            float(union[2]) / float(w),
            float(union[3]) / float(h),
        ]
        out["ok"] = any(bool(s.get("ok")) for s in out["slots"])
        out["empty"] = not out["ok"]
        return out
    except Exception:
        return out
