"""Is a photo good enough to try on? Checked before anything runs, so a bad photo costs a retake,
not a GPU run that ends in "We couldn't make this look properly."

Each problem has its own plain request ("too dark", "more than one person"), so the person knows
what to change. Thresholds come from real photos: sharp phone photos measure 178 and up on the
sharpness scale and a blurred one 10 or less; good lighting measures 64-134 and a dark one 21.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
from PIL import Image

from .preprocess.parsing import BG, FACE, L_LEG, PANTS, R_LEG, SKIRT, parser

# Size never refuses a photo on its own; only a picture so small nothing can be seen in it.
TINY = 64
log = logging.getLogger("tryon.quality")
# Edge crispness relative to contrast (see _sharpness). Phone photos 2.2-2.8, webcam frames
# 1.4-2.5 (soft optics, noise reduction), blurred photos 0.5-1.0.
MIN_SHARPNESS = 1.15
DARKEST, BRIGHTEST = 40.0, 215.0
MIN_PERSON_HEIGHT = 0.40  # share of the photo's height the person spans


@dataclass
class Retake:
    code: str
    message: str

    def public(self) -> dict:
        return {"code": self.code, "message": self.message}


_ASK = "Please upload another photo"


def _sharpness(gray: np.ndarray, box: tuple[int, int, int, int]) -> float:
    """How crisp the strongest edges are, relative to the picture's contrast.

    The old measure (variance of the Laplacian) mixes blur with texture: a soft webcam frame of a
    plain shirt scored like a blurred photo and was refused. Blur weakens the steepest edges (the
    outline, a collar) compared with the difference between light and dark, whatever the
    texture, so that ratio separates "soft but usable" from "blurred".
    """
    x0, y0, x1, y1 = box
    crop = gray[y0:y1, x0:x1].astype(np.float32)
    if crop.size == 0:
        return 0.0
    s = min(1.0, 768 / max(crop.shape))  # common scale for big photos; never enlarge small ones
    if s < 1.0:
        crop = cv2.resize(crop, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    gx = cv2.Sobel(crop, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(crop, cv2.CV_32F, 0, 1)
    edges = float(np.percentile(np.hypot(gx, gy), 99))
    contrast = float(np.percentile(crop, 95) - np.percentile(crop, 5)) + 1.0
    return edges / contrast


def _faces(labels: np.ndarray) -> int:
    n, _, stats, _ = cv2.connectedComponentsWithStats((labels == FACE).astype(np.uint8))
    return sum(1 for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] > 0.0015 * labels.size)


def _common(img: Image.Image, labels: np.ndarray, min_side: int, who: str) -> Optional[Retake]:
    if min(img.size) < TINY:
        return Retake("unclear", f"This photo isn't clear enough. {_ASK}.")
    body = labels != BG
    if body.mean() < 0.02:
        return Retake("no_person" if who == "person" else "no_garment",
                      f"We can't find {'you' if who == 'person' else 'the clothes'} in this photo. {_ASK}.")
    gray = cv2.cvtColor(np.asarray(img.convert("RGB")), cv2.COLOR_RGB2GRAY)
    ys, xs = np.nonzero(body)
    light = float(gray[body].mean())
    sharp = _sharpness(gray, (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1))
    # Numbers only, never the photo: enough to tell why a photo was asked again.
    log.info("quality %s: %dx%d light %.0f sharpness %.2f", who, img.width, img.height, light, sharp)
    if light < DARKEST:
        return Retake("too_dark", f"This photo is too dark. {_ASK}, taken in better light.")
    if light > BRIGHTEST:
        return Retake("too_bright", f"This photo is too bright to see the clothes. {_ASK}, out of direct light.")
    if sharp < MIN_SHARPNESS:
        return Retake("blurry", f"This photo isn't clear enough. {_ASK}.")
    return None


def check_person(img: Image.Image) -> Optional[Retake]:
    labels = parser.labels(img)
    if (r := _common(img, labels, TINY, "person")) is not None:
        return r
    faces = _faces(labels)
    if faces == 0:
        return Retake("no_face", f"We can't see your face in this photo. {_ASK} where your face is visible.")
    if faces > 1:
        return Retake("several_people", f"There's more than one person in this photo. {_ASK} with only you in it.")
    ys = np.nonzero((labels != BG).any(axis=1))[0]
    if (ys.max() - ys.min()) / labels.shape[0] < MIN_PERSON_HEIGHT:
        return Retake("too_far", f"You're too far away in this photo. {_ASK}, closer to the camera.")
    from .preprocess.body import skeleton

    if skeleton(img) is None:
        return Retake("no_body", f"We can't see your shoulders and hips in this photo. {_ASK} showing your body from the head down.")
    return None


def check_outfit(img: Image.Image) -> Optional[Retake]:
    labels = parser.labels(img)
    has_person = (labels == FACE).sum() > 0.0015 * labels.size
    if has_person:
        if (r := _common(img, labels, TINY, "outfit")) is not None:
            return r
        if _faces(labels) > 1:
            return Retake("several_people", f"There's more than one person in this outfit photo. {_ASK} showing one outfit.")
        return None
    # A product photo (no one wearing it): size, light and sharpness of the whole picture.
    if min(img.size) < TINY:
        return Retake("unclear", f"This photo isn't clear enough. {_ASK}.")
    gray = cv2.cvtColor(np.asarray(img.convert("RGB")), cv2.COLOR_RGB2GRAY)
    if _sharpness(gray, (0, 0, gray.shape[1], gray.shape[0])) < MIN_SHARPNESS:
        return Retake("blurry", f"This photo isn't clear enough. {_ASK}.")
    return None


def needs_legs(labels: np.ndarray) -> bool:
    """Bottoms and full outfits need the legs in the photo."""
    return float(np.isin(labels, [L_LEG, R_LEG, PANTS, SKIRT]).mean()) < 0.03
