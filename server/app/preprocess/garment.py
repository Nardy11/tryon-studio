"""Garment extraction: turn "a photo of someone wearing it" into "a product photo of it".

Try-on models expect the garment alone, centred on a plain background. A street/lookbook photo
also contains a face, other clothes (beige trousers next to a black sweater), bags and scenery,
and the model blends them together. Here we run the human parser on the garment photo; if it
shows a person, we keep only the pixels of the requested category, paste them on white, and crop.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from PIL import Image, ImageFilter

from .parsing import (
    BAG, BELT, DRESS, FACE, HAIR, L_ARM, L_LEG, PANTS, R_ARM, R_LEG, SCARF, SKIRT, UPPER,
    Category, parser,
)

log = logging.getLogger("tryon.garment")

KEEP = {
    "upper": [UPPER, SCARF],
    "lower": [PANTS, SKIRT, BELT],
    "dress": [UPPER, PANTS, SKIRT, DRESS, BELT, SCARF],
}


def looks_like_person(labels: np.ndarray) -> bool:
    body = np.isin(labels, [FACE, HAIR, L_ARM, R_ARM, L_LEG, R_LEG]).mean()
    return body > 0.01


PART_LABELS = {
    "upper": [UPPER, SCARF],
    "lower": [PANTS, SKIRT],
    "dress": [DRESS],
}


def color_parts(img: Image.Image) -> dict[str, np.ndarray]:
    """Colour samples (Lab pixels) per garment piece, measured on the ORIGINAL garment photo.

    A lookbook photo of a whole outfit (black sweater + cream trousers) must not be averaged into
    one colour: each piece is measured on its own, so each piece in the result can be corrected
    against its own colour. A product photo with no person in it gives one sample: "whole".
    """
    import cv2

    labels = parser.labels(img)
    lab = cv2.cvtColor(np.asarray(img.convert("RGB")), cv2.COLOR_RGB2LAB).astype(np.float32)
    if not looks_like_person(labels):
        from .color import garment_pixels_from_product

        return {"whole": garment_pixels_from_product(img)}
    parts = {}
    kernel = np.ones((5, 5), np.uint8)
    for part, ids in PART_LABELS.items():
        m = cv2.erode(np.isin(labels, ids).astype(np.uint8), kernel).astype(bool)
        if m.sum() >= 300:
            parts[part] = lab[m]
    return parts


def extract_garment(img: Image.Image, category: Category) -> tuple[Image.Image, bool]:
    """Return (garment image, whether extraction happened)."""
    labels = parser.labels(img)
    if not looks_like_person(labels):
        return img, False  # already a product photo

    keep = np.isin(labels, KEEP[category])
    # A bag strap across the chest is labelled "bag": inside the garment's outline, count it as garment
    # so the cut-out has no hole (the model fills it with the garment's own fabric).
    if keep.any():
        ys, xs = np.nonzero(keep)
        box = np.zeros_like(keep)
        box[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1] = True
        keep |= (labels == BAG) & box & _near(keep, 12)
    if keep.mean() < 0.005:
        log.info("no %s found in the garment photo; using it as is", category)
        return img, False

    # Arms or hands crossing the garment leave holes. Close the outline, then paint the holes with
    # the surrounding fabric so the model doesn't copy an arm-shaped white gap.
    import cv2

    k = max(9, int(0.08 * max(img.size)) | 1)
    closed = cv2.morphologyEx(
        keep.astype(np.uint8), cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    ).astype(bool)
    # Only fill what covers the garment: arms, hands and bag straps. Skin seen through the collar or
    # a zip opening is part of the garment's design and must stay (filling it turned a quarter-zip
    # into a closed hoodie-like neck).
    holes = closed & ~keep & np.isin(labels, [L_ARM, R_ARM, BAG])
    src = np.array(img)
    if holes.any():
        # Grow the holes a little: the parser's arm edges are soft, and skin-coloured edge pixels
        # would otherwise be smeared into the fill.
        grow = max(3, int(0.012 * max(img.size)))
        holes = _near(holes, grow) & closed
        src = cv2.inpaint(src, (holes * 255).astype(np.uint8), 9, cv2.INPAINT_TELEA)
        keep = keep | holes

    alpha = Image.fromarray((keep * 255).astype(np.uint8), "L").filter(ImageFilter.GaussianBlur(1.2))
    white = Image.new("RGB", img.size, (255, 255, 255))
    cut = Image.composite(Image.fromarray(src), white, alpha)

    ys, xs = np.nonzero(keep)
    pad = int(0.06 * max(np.ptp(xs), np.ptp(ys))) + 4
    x0, y0 = max(0, xs.min() - pad), max(0, ys.min() - pad)
    x1, y1 = min(img.width, xs.max() + pad + 1), min(img.height, ys.max() + pad + 1)
    cut = cut.crop((x0, y0, x1, y1))
    return _centre_on_white(cut, ratio=3 / 4), True


def _near(mask: np.ndarray, px: int) -> np.ndarray:
    from .parsing import _dilate

    return _dilate(mask, px)


def _centre_on_white(img: Image.Image, ratio: float) -> Image.Image:
    """Pad to a w/h ratio on white, garment centred (how product photos are framed)."""
    w, h = img.size
    if w / h > ratio:
        nw, nh = w, int(round(w / ratio))
    else:
        nw, nh = int(round(h * ratio)), h
    canvas = Image.new("RGB", (nw, nh), (255, 255, 255))
    canvas.paste(img, ((nw - w) // 2, (nh - h) // 2))
    return canvas


def crop_worn_garment(img: Image.Image, category: Category, min_side: int = 512) -> Optional[Image.Image]:
    """The garment as worn in a lookbook photo, real pixels only: no cut-out, no hole filling.

    Editing models that read a reference image (FLUX.2 Klein) copy texture, knit, seams and small
    details far better from the untouched photo than from a cut-out, where filled holes and soft
    edges come out as blotches. The model's face and hair are painted out so they are not copied.
    Small photos are upscaled so the garment gets more of the model's attention (more tokens).
    """
    labels = parser.labels(img)
    if not looks_like_person(labels):
        return None
    ids = KEEP[category] + ([L_ARM, R_ARM] if category in ("upper", "dress") else [])
    keep = np.isin(labels, ids)
    garment = np.isin(labels, KEEP[category])
    if garment.mean() < 0.005:
        return None
    ys, xs = np.nonzero(keep)
    gy = np.nonzero(garment.any(axis=1))[0]
    # Vertical extent from the garment itself (arms reach past a short top); horizontal includes sleeves.
    y0, y1 = int(gy.min()), int(gy.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    pad = int(0.05 * max(x1 - x0, y1 - y0)) + 4
    x0, y0 = max(0, x0 - pad), max(0, y0 - pad)
    # A top's reference stops just under its hem: with the model's trousers in view, the editor
    # copies them onto the user too (beige chinos drawn over navy jeans).
    bottom_pad = max(4, (y1 - y0) // 30) if category == "upper" else pad
    x1, y1 = min(img.width, x1 + pad), min(img.height, y1 + bottom_pad)
    src = np.array(img.convert("RGB"))
    head = _near(np.isin(labels, [FACE, HAIR]), 2)
    # The neck sits under the collar line: keep it (it shows the neckline's shape), hide the face.
    collar = int(gy.min())
    head[collar + max(2, (y1 - y0) // 40) :, :] = False
    src[head] = 255
    crop = Image.fromarray(src).crop((x0, y0, x1, y1))
    if max(crop.size) < min_side:
        s = min_side / max(crop.size)
        crop = crop.resize((round(crop.width * s), round(crop.height * s)), Image.LANCZOS)
    return crop
