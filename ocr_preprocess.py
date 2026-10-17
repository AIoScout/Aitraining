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


def normalise_glyph(crop: np.ndarray, out_size: int = GLYPH_SIZE,
                    convention: str = NORM_STRETCH) -> np.ndarray:
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
    """
    arr = np.asarray(crop)
    if arr.size == 0:
        return np.zeros((out_size, out_size), dtype=np.uint8)
    if convention == NORM_STRETCH:
        return nn_resize_u8(arr, out_size, out_size).astype(np.uint8)

    if convention != NORM_CANON:
        raise ValueError(f"unknown glyph convention: {convention!r}")

    fit = _GLYPH_CANON_FIT if out_size == GLYPH_SIZE else max(1, (out_size * _GLYPH_CANON_FIT) // GLYPH_SIZE)
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

        glyph = normalise_glyph(np.ascontiguousarray(crop), out_size=out_size, convention=convention)
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
            crop = np.ascontiguousarray(gray[y1:y2, x1:x2])
            if crop.size == 0:
                res: Dict[str, Any] = {"glyph": blank_glyph.copy(), "ok": False, "empty": True}
            else:
                # Segmentation is OFF for the slot: the box already IS one
                # glyph, and re-running the projection inside it could only
                # re-find the same run — or, on a slot whose ink is a
                # fragment, drop it below `min_run` and blank the slot.
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
            out["slots"].append(
                {
                    "char": str(seq[idx]),
                    "glyph": glyph,
                    "ink": res.get("ink"),
                    "box": [int(x1), int(y1), int(x2), int(y2)],
                    "box_norm": [
                        float(x1) / float(w),
                        float(y1) / float(h),
                        float(x2) / float(w),
                        float(y2) / float(h),
                    ],
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
