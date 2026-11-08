"""Size check: estimate the shopper's size from their photo and height (and weight, if given),
compare it with the sizes in stock, and decide which size to draw them in.

Sizes sold: XS, S, M, L, XL (indexes 0-4). Shoppers bigger than XL get indexes 5, 6, ...
("XL + 1", "XL + 2"): never sold, only used to say how far off the biggest size is.

Height and weight are personal data: used in memory for this request, never stored or logged.

The photo has no ruler in it, so the height gives the scale:
- head to feet visible: pixels per cm = person height in pixels / height (high confidence);
- feet out of frame: the torso is the ruler (shoulder to hip joints ~ 0.28 of stature; low);
- no hips either: the head (head height ~ 0.13 of stature; low).
Widths are read across the body (arms left out) at chest, waist and hips, turned into
circumferences as ellipses, less an allowance for the clothes worn. With a weight, a body-mass
estimate is blended in. A photo estimate is good to about one size.
"""

from __future__ import annotations

import hashlib
import math
from typing import Optional

import numpy as np
from PIL import Image

from .preprocess.parsing import (
    BAG, BG, FACE, HAIR, HAT, L_ARM, L_LEG, L_SHOE, PANTS, R_ARM, R_LEG, R_SHOE, SKIRT, SUNGLASSES, parser,
)

SIZES = ["XS", "S", "M", "L", "XL"]
# Upper bound (cm) of each size, unisex. Above XL: one more "size" per 8 cm.
CHEST = [88, 96, 104, 112, 120]
WAIST = [70, 76, 82, 88, 96]
STEP_ABOVE = 8.0
CLOTHES_CM = 4.0  # what the clothes in the photo add to each circumference
NEAR_CM = 2.0  # within this of a size boundary: "between" two sizes


def label(i: int) -> str:
    return SIZES[i] if 0 <= i < len(SIZES) else f"XL + {i - len(SIZES) + 1}"


def size_of(cm: float, bounds: list[int]) -> tuple[int, bool]:
    """(size index, between two sizes). Between rounds up: better a little loose than tight."""
    for i, b in enumerate(bounds):
        if cm < b:
            return (i + 1 if b - cm <= NEAR_CM and i + 1 <= len(bounds) else i), b - cm <= NEAR_CM
    over = cm - bounds[-1]
    i = len(bounds) + int(over // STEP_ABOVE)
    edge = STEP_ABOVE - (over % STEP_ABOVE)
    return (i + 1 if edge <= NEAR_CM else i), edge <= NEAR_CM


def _ellipse(width: float, depth_ratio: float) -> float:
    """Ramanujan's ellipse perimeter for a body section `width` across, `depth_ratio`*width deep."""
    a, b = width / 2, width * depth_ratio / 2
    return math.pi * (3 * (a + b) - math.sqrt((3 * a + b) * (a + 3 * b)))


def _near_arms(y: float, xs: np.ndarray, arms: list[tuple], r: float) -> np.ndarray:
    """Pixels of row y within r of an arm bone (sleeved arms are labelled as clothes, so the parser
    alone can't tell an arm from the torso)."""
    near = np.zeros(len(xs), bool)
    for (ax, ay), (bx, by) in arms:
        vx, vy = bx - ax, by - ay
        L2 = vx * vx + vy * vy or 1e-6
        t = np.clip(((xs - ax) * vx + (y - ay) * vy) / L2, 0, 1)
        near |= np.hypot(xs - (ax + t * vx), y - (ay + t * vy)) < r
    return near


def _row_width(body: np.ndarray, y: int, x0: int, x1: int, arms: list[tuple], r: float) -> Optional[float]:
    """Torso width across row y (median of 5 rows): outermost body pixels in [x0, x1], arms left out."""
    h, w = body.shape
    x0, x1 = max(0, x0), min(w, x1)
    xs_all = np.arange(x0, x1, dtype=np.float32)
    widths = []
    for yy in range(max(0, y - 2), min(h, y + 3)):
        row = body[yy, x0:x1] & ~_near_arms(float(yy), xs_all, arms, r)
        xs = np.nonzero(row)[0]
        if len(xs) > 3:
            widths.append(xs.max() - xs.min() + 1)
    return float(np.median(widths)) if widths else None


_cache: dict[str, dict] = {}


def estimate(img: Image.Image, height_cm: float, weight_kg: Optional[float] = None) -> dict:
    """The shopper's measurements and sizes. Cached in memory by photo + inputs (never on disk)."""
    key = hashlib.sha1(img.convert("RGB").tobytes() + f"{height_cm:.0f}/{weight_kg or 0:.0f}".encode()).hexdigest()
    if key in _cache:
        return _cache[key]
    out = _estimate(img.convert("RGB"), float(height_cm), float(weight_kg) if weight_kg else None)
    if len(_cache) > 256:
        _cache.clear()
    _cache[key] = out
    return out


def _estimate(img: Image.Image, height_cm: float, weight_kg: Optional[float]) -> dict:
    from .live import _pose

    labels = parser.labels(img)
    h, w = labels.shape
    pose = _pose(img)
    none = {"confidence": "none", "top_size": None, "bottom_size": None}
    if pose is None:
        return {**none, "reason": "no_body"}
    P = [(x * w, y * h, v) for x, y, v in pose]
    seen = lambda i: P[i][2] > 0.5  # noqa: E731
    if not (seen(11) and seen(12)):
        return {**none, "reason": "no_shoulders"}

    # ---- the ruler ----
    head = np.isin(labels, [HAIR, FACE, HAT])
    if not head.any():
        return {**none, "reason": "no_head"}
    top_y = int(np.nonzero(head.any(axis=1))[0].min())
    feet = np.isin(labels, [L_SHOE, R_SHOE, L_LEG, R_LEG, PANTS, SKIRT])
    bottom_rows = np.nonzero(feet.any(axis=1))[0]
    confidence = "none"
    px_per_cm = None
    if (seen(27) or seen(28)) and len(bottom_rows) and bottom_rows.max() < h - 3 and top_y > 2:
        px_per_cm = (bottom_rows.max() - top_y) / height_cm
        confidence = "high"
    elif seen(11) and seen(12) and seen(23) and seen(24):
        # Feet out of frame: the torso (shoulder joints to hip joints) is about 0.28 of stature.
        torso_px = math.hypot((P[11][0] + P[12][0] - P[23][0] - P[24][0]) / 2, (P[11][1] + P[12][1] - P[23][1] - P[24][1]) / 2)
        px_per_cm = torso_px / (0.28 * height_cm)
        confidence = "low"
    elif seen(9) and seen(10) and top_y > 2:
        mouth_y = (P[9][1] + P[10][1]) / 2
        head_px = (mouth_y - top_y) / 0.75
        px_per_cm = head_px / (0.13 * height_cm)
        confidence = "low"
    if not px_per_cm or px_per_cm <= 0:
        return {**none, "reason": "no_scale"}

    # ---- frame, from the skeleton (clothes and arm poses don't change it) ----
    sh_y = (P[11][1] + P[12][1]) / 2
    sw_px = math.hypot(P[11][0] - P[12][0], P[11][1] - P[12][1])
    hips = seen(23) and seen(24)
    hip_px = math.hypot(P[23][0] - P[24][0], P[23][1] - P[24][1]) if hips else None
    hip_y = (P[23][1] + P[24][1]) / 2 if hips else None
    torso = (hip_y - sh_y) if hips else sw_px * 1.3
    # Pose shoulder points are joint centres, narrower than the visible shoulders: facing the
    # camera this ratio is about 0.55-0.8 (lower for women); turned sideways, under 0.4.
    if hips and sw_px < 0.4 * torso:
        return {**none, "reason": "turned"}
    # Body proportions: chest ~ 2.6 x shoulder-joint span + 8 cm, waist ~ 4.1 x hip-joint span.
    chest = 2.6 * (sw_px / px_per_cm) + 8.0
    waist = 4.1 * (hip_px / px_per_cm) if hips else None
    hip = None

    # ---- the waist outline, only when both arms are clearly away from the body ----
    mid_x = (P[11][0] + P[12][0]) / 2
    arms_clear = all(
        P[i][2] > 0.5 and (abs(P[i][0] - mid_x) > 0.95 * sw_px or P[i][1] < sh_y) for i in (13, 14, 15, 16)
    )
    if hips and arms_clear:
        body = ~np.isin(labels, [BG, HAIR, HAT, SUNGLASSES, BAG, L_ARM, R_ARM])
        y = int(sh_y + 0.75 * torso)
        width = _row_width(body, y, int(mid_x - 0.9 * sw_px), int(mid_x + 0.9 * sw_px), [], 0)
        if width:
            outline = _ellipse(width, 0.80) / px_per_cm - CLOTHES_CM
            waist = outline if waist is None else 0.5 * waist + 0.5 * outline

    # ---- fullness, from the weight ----
    method = "photo"
    if weight_kg:
        bmi = weight_kg / (height_cm / 100) ** 2
        k = math.sqrt(height_cm / 175)
        chest_b = (100 + 2.0 * (bmi - 25)) * k
        waist_b = (88 + 2.6 * (bmi - 25)) * k
        chest = 0.25 * chest + 0.75 * chest_b  # the skeleton gives the frame; weight gives fullness
        waist = waist_b if waist is None else 0.3 * waist + 0.7 * waist_b
        method = "photo+weight"
        if confidence == "low":
            confidence = "medium"
    elif not (hips and arms_clear):
        # Without a weight, the skeleton gives the frame, not how full the body is.
        confidence = "rough" if confidence == "high" else "low"

    top = size_of(chest, CHEST) if chest else (None, False)
    bottom = size_of(waist, WAIST) if waist else (None, False)
    return {
        "confidence": confidence,
        "method": method,
        "reason": None,
        "chest_cm": round(chest) if chest else None,
        "waist_cm": round(waist) if waist else None,
        "hip_cm": round(hip) if hip else None,
        "top_size": top[0],
        "top_between": top[1],
        "bottom_size": bottom[0],
        "bottom_between": bottom[1],
    }


def needed_size(est: dict, parts: set[str]) -> Optional[int]:
    """The size this shopper needs for an outfit made of these parts (upper / lower / dress)."""
    wants = []
    if parts & {"upper", "dress"} and est.get("top_size") is not None:
        wants.append(est["top_size"])
    if parts & {"lower", "dress"} and est.get("bottom_size") is not None:
        wants.append(est["bottom_size"])
    if not wants:  # e.g. bottoms but the waist couldn't be measured: fall back to the top size
        wants = [s for s in (est.get("top_size"), est.get("bottom_size")) if s is not None]
    return max(wants) if wants else None


def verdict(needed: Optional[int], stock: list[int]) -> dict:
    """Which size to show them in, and how it fits. `off` < 0 = tight, > 0 = loose (sizes)."""
    stock = sorted(set(s for s in stock if 0 <= s < len(SIZES)))
    if needed is None or not stock:
        return {"verdict": "unknown", "needed": None, "worn": None, "off": 0}
    if needed in stock:
        worn = needed
    else:
        smaller = [s for s in stock if s < needed]
        bigger = [s for s in stock if s > needed]
        if smaller and bigger:
            worn = min(bigger) if (min(bigger) - needed) <= (needed - max(smaller)) else max(smaller)
        elif bigger:
            worn = min(bigger)
        else:
            worn = max(smaller)
    off = max(-3, min(3, worn - needed))
    if off == 0:
        v = "fits"
    elif all(s < needed for s in stock):
        v = "snug" if off == -1 else "too_small"
    elif all(s > needed for s in stock):
        v = "loose" if off == 1 else "too_big"
    else:
        v = "close"
    return {"verdict": v, "needed": label(needed), "worn": label(worn), "off": off}
