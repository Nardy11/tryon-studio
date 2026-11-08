"""Live filter: an outfit prepared once, then laid onto the camera image in the browser.

For each outfit photo the server returns:
- the clothes alone, cut out of the photo (transparent elsewhere: no face, hair, hands or skin);
- where the body is in that photo (MediaPipe Pose landmarks, as shares of the image size).

The browser tracks the person live and moves each part of the clothes from the model's pose to
theirs. No photo of the person is taken and their video never leaves their device. The outfit
photo passes the same safety and quality checks as in photo mode before anything is returned.
"""

from __future__ import annotations

import base64
import io
from typing import Optional

import numpy as np
from PIL import Image, ImageFilter

from .preprocess.parsing import BELT, DRESS, PANTS, SCARF, SKIRT, UPPER, _dilate, parser

GARMENT = [UPPER, SKIRT, PANTS, DRESS, BELT, SCARF]
MAX_SIDE = 1024  # plenty for a phone screen, small to send


def _pose(img: Image.Image) -> Optional[list[list[float]]]:
    """33 landmarks [x, y, visibility], x and y as shares of the image, or None."""
    from .preprocess import body

    if not body.MODEL.is_file():
        return None
    import mediapipe as mp

    rgb = np.ascontiguousarray(np.asarray(img.convert("RGB")))
    with body._lock:
        res = body._get().detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
    if not res.pose_landmarks:
        return None
    return [[float(p.x), float(p.y), float(p.visibility)] for p in res.pose_landmarks[0]]


def _pose_from_garment(mask: np.ndarray, labels: np.ndarray) -> list[list[float]]:
    """A product photo has no body: place shoulders, hips, knees and ankles on the garment's own
    outline, so the filter can still fit it. MediaPipe's "left" is the wearer's left, which is the
    right side of a photo facing the camera."""
    h, w = mask.shape
    lm = [[0.5, 0.5, 0.0] for _ in range(33)]

    def put(i: int, x: float, y: float) -> None:
        lm[i] = [x / w, y / h, 1.0]

    top = np.isin(labels, [UPPER, DRESS, SCARF]) & mask
    bottom = np.isin(labels, [PANTS, SKIRT]) & mask
    if top.any():
        ys, xs = np.nonzero(top)
        x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
        bw, bh = x1 - x0, y1 - y0
        put(11, x1 - 0.2 * bw, y0 + 0.06 * bh)
        put(12, x0 + 0.2 * bw, y0 + 0.06 * bh)
        put(23, x1 - 0.28 * bw, y1)
        put(24, x0 + 0.28 * bw, y1)
    if bottom.any():
        ys, xs = np.nonzero(bottom)
        x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
        bw, bh = x1 - x0, y1 - y0
        put(23, x1 - 0.25 * bw, y0 + 0.04 * bh)
        put(24, x0 + 0.25 * bw, y0 + 0.04 * bh)
        put(25, x1 - 0.27 * bw, y0 + 0.5 * bh)
        put(26, x0 + 0.27 * bw, y0 + 0.5 * bh)
        put(27, x1 - 0.27 * bw, y1)
        put(28, x0 + 0.27 * bw, y1)
        if not top.any():  # bottoms alone: shoulders far above, so the torso map still works
            put(11, x1 - 0.2 * bw, y0 - 0.9 * bh)
            put(12, x0 + 0.2 * bw, y0 - 0.9 * bh)
    return lm


def outfit_layer(img: Image.Image) -> Optional[dict]:
    """The clothes cut out of an outfit photo, and the pose they were worn in. None if no clothes."""
    img = img.convert("RGB")
    if max(img.size) > MAX_SIDE:
        s = MAX_SIDE / max(img.size)
        img = img.resize((round(img.width * s), round(img.height * s)), Image.LANCZOS)
    labels = parser.labels(img)
    mask = np.isin(labels, GARMENT)
    if mask.sum() < 500:
        return None
    pose = _pose(img)
    source = "photo"
    if pose is None or min(pose[11][2], pose[12][2]) < 0.3:
        pose = _pose_from_garment(mask, labels)
        source = "garment"
    alpha = Image.fromarray((_dilate(mask, 1) * 255).astype(np.uint8), "L").filter(ImageFilter.GaussianBlur(1.0))
    out = img.convert("RGBA")
    out.putalpha(alpha)
    buf = io.BytesIO()
    out.save(buf, format="PNG", optimize=False)
    return {
        "width": img.width,
        "height": img.height,
        "landmarks": pose,
        "pose_from": source,
        "image": "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii"),
    }
