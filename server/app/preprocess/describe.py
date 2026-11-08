"""Describe a garment in a few words from its photo: colour per piece and sleeve length.

Prompt-driven engines (FLUX.2 Klein) guess badly from a small lookbook photo: pushed-up knit
sleeves became "short sleeves", a quarter-zip became a jacket. A plain description fixes most of it,
and can be built from what the human parser already sees, with no extra model.
"""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image

from .parsing import L_ARM, PANTS, R_ARM, SKIRT, UPPER, DRESS, Category, parser

# Named colours in RGB. Nearest in Lab wins.
COLOURS = {
    "black": (25, 25, 28), "charcoal": (60, 60, 62), "grey": (128, 128, 128), "light grey": (190, 190, 190),
    "white": (245, 245, 245), "cream": (232, 222, 200), "stone": (210, 200, 188), "beige": (205, 185, 150), "camel": (176, 130, 80),
    "brown": (100, 65, 40), "dark brown": (60, 40, 30), "taupe": (125, 105, 90), "mocha": (120, 85, 65), "khaki": (190, 170, 130), "navy": (25, 35, 75), "blue": (40, 80, 170),
    "light blue": (150, 185, 225), "denim blue": (70, 100, 140), "green": (50, 120, 60), "olive": (100, 100, 50),
    "dark green": (30, 60, 40), "red": (180, 30, 35), "burgundy": (100, 25, 40), "pink": (230, 160, 180),
    "purple": (110, 60, 140), "lilac": (190, 165, 215), "yellow": (230, 200, 50), "orange": (225, 120, 40),
}
_NAMES = list(COLOURS)
_LAB = cv2.cvtColor(np.array([[COLOURS[n] for n in _NAMES]], dtype=np.uint8), cv2.COLOR_RGB2LAB)[0].astype(np.float32)


def colour_name(lab_pixels: np.ndarray) -> str:
    # Median is robust to highlights, shadows and small prints.
    m = np.median(lab_pixels, axis=0)
    # Lightness counts half: it mostly says how bright the room was. Hue survives dim light
    # (navy trousers in a dark lift read as "black" on plain Lab distance).
    chroma = float(np.hypot(m[1] - 128, m[2] - 128))
    # Tint relative to brightness: navy at L=1 has chroma 7 (strong); a black knit at L=25 with
    # chroma 7 is just black with a warm cast.
    if m[0] < 70 and chroma / (m[0] + 5) > 0.5:
        # Very dark fabric (navy in a dim lift photographs at L~1): plain Lab distance calls
        # everything "black". The hue still shows, so match it by hue angle among coloured names.
        names_c = _LAB[:, 1:] - 128
        coloured = np.hypot(names_c[:, 0], names_c[:, 1]) > 8
        ang = np.arctan2(m[2] - 128, m[1] - 128)
        diff = np.abs(np.angle(np.exp(1j * (np.arctan2(names_c[:, 1], names_c[:, 0]) - ang))))
        diff[~coloured] = np.inf
        # Of the names with about the right hue, the darkest (navy over blue, burgundy over red).
        close = np.nonzero(diff <= diff.min() + 0.35)[0]
        return _NAMES[int(close[np.argmin(_LAB[close, 0])])]
    # The colour people name is the lit part of the fabric, not its shadows: a khaki shirt in
    # side light has a median like "taupe". Lightness from the brighter pixels, hue from all.
    lit = float(np.percentile(lab_pixels[:, 0], 70))
    lstar = lit / 2.55
    # Hardly any hue: a grey. A charcoal herringbone with a slight warm cast is still grey, not
    # "taupe" (that name drew a brown jacket for a grey suit).
    if chroma < 5.5:
        for limit, name in ((14, "black"), (42, "charcoal"), (62, "grey"), (82, "light grey")):
            if lstar < limit:
                return name
        return "white"
    m = np.array([lit, m[1], m[2]], np.float32)
    # Lightness at half weight: a brown jacket in shade must not drift to "olive" (a different
    # hue at the same lightness).
    w = np.array([0.5, 1.0, 1.0], np.float32)
    return _NAMES[int(np.argmin(np.linalg.norm((_LAB - m) * w, axis=1)))]


def describe(garment: Image.Image, category: Category) -> str:
    """e.g. "black long-sleeve top", "cream trousers", "black long-sleeve top with cream trousers"."""
    labels = parser.labels(garment)
    lab = cv2.cvtColor(np.asarray(garment.convert("RGB")), cv2.COLOR_RGB2LAB).astype(np.float32)
    kernel = np.ones((5, 5), np.uint8)

    def px(ids):
        m = cv2.erode(np.isin(labels, ids).astype(np.uint8), kernel).astype(bool)
        return lab[m] if m.sum() > 300 else None

    upper, lower, dress = px([UPPER]), px([PANTS, SKIRT]), px([DRESS])
    if upper is None and lower is None and dress is None:
        # A product photo with no person: describe the whole non-background area.
        from .color import garment_pixels_from_product

        whole = garment_pixels_from_product(garment)
        return f"{colour_name(whole)} garment" if len(whole) > 300 else ""

    def top_words() -> str:
        arm_skin = int(np.isin(labels, [L_ARM, R_ARM]).sum())
        top_px = int((labels == UPPER).sum())
        # Visible forearm skin relative to the top: little = long sleeves (maybe pushed up a bit).
        share = arm_skin / max(1, arm_skin + top_px)
        sleeves = "short-sleeve" if share > 0.35 else "long-sleeve"
        return f"{colour_name(upper)} {sleeves} top"

    bottom = "skirt" if (labels == SKIRT).sum() > (labels == PANTS).sum() else "trousers"
    if category == "upper" and upper is not None:
        return top_words()
    if category == "lower" and lower is not None:
        return f"{colour_name(lower)} {bottom}"
    if dress is not None:
        return f"{colour_name(dress)} dress"
    parts = []
    if upper is not None:
        parts.append(top_words())
    if lower is not None:
        parts.append(f"{colour_name(lower)} {bottom}")
    return " with ".join(parts)
