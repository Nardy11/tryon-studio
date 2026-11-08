"""A ruler along the body, so a garment covers the same part of the user as of the model.

Lengths measured against other clothes go wrong: on a high-waisted skirt the "waist" sits at the
ribs, and a hip-length blazer then measures as a long coat. So every hem is measured against the
skeleton instead (MediaPipe Pose, Apache 2.0, runs on the CPU in ~50 ms):

    0 = shoulders, 1 = hips, 2 = knees, 3 = ankles  (piecewise linear, extrapolated past the ends)

A blazer ending at 1.1 ends just below the hips on anyone; a skirt ending at 2.0 ends at the knee.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

log = logging.getLogger("tryon.body")

MODEL = Path(__file__).resolve().parents[2] / "models" / "pose_landmarker_full.task"

# MediaPipe pose landmark ids, left and right.
_PAIRS = {"shoulder": (11, 12), "hip": (23, 24), "knee": (25, 26), "ankle": (27, 28)}
_ORDER = ["shoulder", "hip", "knee", "ankle"]


@dataclass
class Skeleton:
    """Height (y, in pixels) of each joint line; None when that joint is out of the picture."""

    height: int
    ys: dict[str, Optional[float]]

    def _knots(self) -> tuple[list[float], list[float]]:
        ts, ys = [], []
        for t, name in enumerate(_ORDER):
            y = self.ys.get(name)
            if y is not None:
                ts.append(float(t))
                ys.append(float(y))
        return ts, ys

    def to_body(self, y: float) -> Optional[float]:
        """Pixel row -> body position (0 shoulders ... 3 ankles)."""
        ts, ys = self._knots()
        if len(ts) < 2:
            return None
        return float(_interp(y, ys, ts))

    def to_pixel(self, t: float) -> Optional[float]:
        """Body position -> pixel row on this person."""
        ts, ys = self._knots()
        if len(ts) < 2:
            return None
        return float(_interp(t, ts, ys))


def _interp(x: float, xs: list[float], ys: list[float]) -> float:
    """Linear interpolation that keeps going past both ends (a coat past the last visible joint)."""
    if x <= xs[0]:
        i = 0
    elif x >= xs[-1]:
        i = len(xs) - 2
    else:
        i = int(np.searchsorted(xs, x)) - 1
    x0, x1, y0, y1 = xs[i], xs[i + 1], ys[i], ys[i + 1]
    return y0 + (x - x0) * (y1 - y0) / max(1e-6, x1 - x0)


_lock = threading.Lock()
_landmarker = None


def _get():
    global _landmarker
    if _landmarker is None:
        from mediapipe.tasks.python import BaseOptions, vision

        _landmarker = vision.PoseLandmarker.create_from_options(
            vision.PoseLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=str(MODEL)),
                running_mode=vision.RunningMode.IMAGE,
                num_poses=1,
            )
        )
    return _landmarker


def skeleton(img: Image.Image, min_visibility: float = 0.5) -> Optional[Skeleton]:
    """The joint lines of the (main) person in the photo, or None when no body is found."""
    if not MODEL.is_file():
        log.warning("pose model missing at %s; garment lengths fall back to the torso rule", MODEL)
        return None
    import mediapipe as mp

    rgb = np.ascontiguousarray(np.asarray(img.convert("RGB")))
    with _lock:
        res = _get().detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
    if not res.pose_landmarks:
        return None
    p = res.pose_landmarks[0]
    h = rgb.shape[0]
    ys: dict[str, Optional[float]] = {}
    for name, (a, b) in _PAIRS.items():
        seen = [p[i] for i in (a, b) if p[i].visibility >= min_visibility and -0.05 <= p[i].y <= 1.05]
        ys[name] = float(np.mean([q.y for q in seen]) * h) if seen else None
    if ys["shoulder"] is None or ys["hip"] is None:
        return None
    return Skeleton(height=h, ys=ys)


def length_words(hem: float, piece: str) -> str:
    """'hip-length', 'knee-length' ... for a hem at body position `hem`."""
    if piece in ("skirt", "dress", "trousers"):
        if piece == "trousers":
            return "cropped" if hem < 2.6 else "full-length"
        if hem < 1.6:
            return "mini"
        if hem < 2.25:
            return "knee-length"
        if hem < 2.7:
            return "midi"
        return "maxi"
    # tops, jackets, coats
    if hem < 0.6:
        return "cropped"
    if hem < 0.85:
        return "waist-length"
    if hem < 1.2:
        return "hip-length"
    if hem < 1.75:
        return "mid-thigh"
    if hem < 2.25:
        return "knee-length"
    return "long"


def where(hem: float) -> str:
    """Plain words for the model: 'just below the hips', 'at the knees'."""
    marks = [(0.75, "at the waist"), (1.0, "at the hips"), (1.3, "just below the hips"),
             (1.6, "at mid-thigh"), (1.9, "just above the knees"), (2.1, "at the knees"),
             (2.5, "below the knees, at mid-calf"), (2.8, "above the ankles"), (9, "at the ankles")]
    if hem < 0.6:
        return "above the waist"
    for limit, words in marks:
        if hem < limit + 0.1:
            return words
    return "at the ankles"
