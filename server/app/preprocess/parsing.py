"""Human parsing: which pixels are face, hair, arms, upper clothes, pants...

Used to build the "agnostic" mask the model needs (the area it is allowed to repaint),
and to optionally lock everything outside the garment after a FLUX.2 Klein edit.

Model: mattmdjaga/segformer_b2_clothes (SegFormer-B2, ~110 MB). Its licence on the Hub is
"other" and it was trained on the ATR dataset; verify before shipping it commercially.
"""

from __future__ import annotations

import threading
from typing import Literal

import numpy as np
import torch
from PIL import Image, ImageFilter

REPO = "mattmdjaga/segformer_b2_clothes"

# Label ids of segformer_b2_clothes
BG, HAT, HAIR, SUNGLASSES, UPPER, SKIRT, PANTS, DRESS, BELT = 0, 1, 2, 3, 4, 5, 6, 7, 8
L_SHOE, R_SHOE, FACE, L_LEG, R_LEG, L_ARM, R_ARM, BAG, SCARF = 9, 10, 11, 12, 13, 14, 15, 16, 17

Category = Literal["upper", "lower", "dress"]


class HumanParser:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._model = None
        self._processor = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def _ensure(self) -> None:
        if self._model is not None:
            return
        from transformers import AutoModelForSemanticSegmentation, SegformerImageProcessor

        self._processor = SegformerImageProcessor.from_pretrained(REPO)
        self._model = AutoModelForSemanticSegmentation.from_pretrained(REPO).to(self.device).eval()

    @torch.no_grad()
    def labels(self, img: Image.Image) -> np.ndarray:
        """Return an (H, W) uint8 label map at the image's own size."""
        with self._lock:
            self._ensure()
            inputs = self._processor(images=img, return_tensors="pt").to(self.device)
            logits = self._model(**inputs).logits
            # Upsample the logits to at most ~1024 px (18 float maps at full photo size would take
            # ~1 GB of VRAM), then scale the small label map up with nearest-neighbour.
            w, h = img.size
            s = min(1.0, 1024 / max(w, h))
            mid = (max(1, round(h * s)), max(1, round(w * s)))
            up = torch.nn.functional.interpolate(logits, size=mid, mode="bilinear", align_corners=False)
            small = up.argmax(dim=1)[0].to(torch.uint8).cpu().numpy()
            del logits, up
            if s == 1.0:
                return small
            return np.array(Image.fromarray(small).resize((w, h), Image.NEAREST))

    def unload(self) -> None:
        with self._lock:
            self._model = None
            self._processor = None


parser = HumanParser()


def _dilate(mask: np.ndarray, px: int) -> np.ndarray:
    """Grow a boolean mask by `px` in every direction (square neighbourhood).

    OpenCV, not PIL's MaxFilter: MaxFilter costs grow with the square of the size, and a 70 px
    grow on a 1500 px photo took ~40 s; cv2.dilate does it in milliseconds.
    """
    if px <= 0:
        return mask
    import cv2

    k = np.ones((px * 2 + 1, px * 2 + 1), np.uint8)
    return cv2.dilate(mask.astype(np.uint8), k) > 0


def _bbox_fill(mask: np.ndarray, pad_x: int, pad_y: int) -> np.ndarray:
    """Fill the padded bounding box of `mask` — gives the model room for a looser garment."""
    ys, xs = np.nonzero(mask)
    out = np.zeros_like(mask)
    if len(xs) == 0:
        return out
    h, w = mask.shape
    x0, x1 = max(0, xs.min() - pad_x), min(w, xs.max() + pad_x + 1)
    y0, y1 = max(0, ys.min() - pad_y), min(h, ys.max() + pad_y + 1)
    out[y0:y1, x0:x1] = True
    return out


def split_people(labels: np.ndarray) -> list[np.ndarray]:
    """Boolean column-band masks, one per person, seeded by each detected face.

    Every rule below (shoulder line, waist, hands = bottom of the arm) is about ONE body. In a photo
    of two friends, a single shoulder line taken from the taller one sits across the shorter one's
    jaw. So each face gets a vertical band of the image, split halfway between face centres.
    """
    import cv2

    face = (labels == FACE).astype(np.uint8)
    n, comp, stats, cents = cv2.connectedComponentsWithStats(face, connectivity=8)
    min_area = max(200, int(labels.size * 0.0008))
    xs = sorted(cents[i][0] for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= min_area)
    h, w = labels.shape
    if len(xs) <= 1:
        return [np.ones((h, w), dtype=bool)]
    cuts = [0] + [int((a + b) / 2) for a, b in zip(xs, xs[1:])] + [w]
    bands = []
    for x0, x1 in zip(cuts, cuts[1:]):
        band = np.zeros((h, w), dtype=bool)
        band[:, x0:x1] = True
        bands.append(band)
    return bands


def agnostic_mask(
    labels: np.ndarray, category: Category, top_length: float | None = None, top_hem_frac: float | None = None
) -> Image.Image:
    """White = area the try-on model may repaint. Built per person, then combined.

    `top_hem_frac` is where the new top ends on this person, as a share of the photo height,
    measured on the skeleton (preprocess.body). When it is missing, `top_length` (the top's length
    relative to the torso, 1.0 = ends at the waistband) is applied to this person's torso instead.
    """
    region = np.zeros(labels.shape, dtype=bool)
    for band in split_people(labels):
        own = np.where(band, labels, BG)  # this person only; everyone else counts as background
        region |= _agnostic_one(own, category, top_length, top_hem_frac) & band
    return Image.fromarray((region * 255).astype(np.uint8), mode="L")


def _agnostic_one(
    labels: np.ndarray, category: Category, top_length: float | None = None, top_hem_frac: float | None = None
) -> np.ndarray:
    """The mask for one person. Face, hair, hands and anything they hold stay black."""
    h, w = labels.shape
    unit = max(4, int(min(h, w) * 0.02))
    arms = np.isin(labels, [L_ARM, R_ARM])
    if category == "upper":
        core = np.isin(labels, [UPPER, SCARF]) | arms
    elif category == "lower":
        core = np.isin(labels, [PANTS, SKIRT, BELT, L_LEG, R_LEG])
    else:  # dress
        core = np.isin(labels, [UPPER, SKIRT, PANTS, DRESS, BELT, SCARF]) | arms

    # Only the clothing (and arms) plus a thin margin. No filled bounding box: in a busy photo the
    # box swallows bags, phones and the arm's position, and the model redraws them (a held bag
    # vanished and the arm moved). A wide margin also makes the new garment wider than the body.
    region = _dilate(core, max(2, unit // 3))
    if category in ("lower", "dress"):
        # Bottoms are the opposite: the photo often shows skinny jeans or leggings, and the new
        # garment may be wide-leg. Hugging the legs makes the model copy their shape and colour,
        # and the gap between the legs can never be repainted. So below the waist, open each row
        # from the outside of one leg to the outside of the other, plus a margin.
        legs = np.isin(labels, [PANTS, SKIRT, DRESS, L_LEG, R_LEG])
        rows = np.nonzero(legs.any(axis=1))[0]
        if len(rows):
            waist = int(rows.min())
            span = np.zeros_like(region)
            for y in range(waist, int(rows.max()) + 1):
                xs = np.nonzero(legs[y])[0]
                if len(xs):
                    span[y, max(0, xs.min() - unit) : xs.max() + unit + 1] = True
            region |= span
    # Never repaint the face, hair, headwear, or anything carried (bags); for tops also leave the
    # legs and shoes alone.
    protect = np.isin(labels, [FACE, HAIR, HAT, SUNGLASSES, BAG])
    if category in ("upper", "dress"):
        # The parser labels the neck as "face". Protect it only above the shoulder line, so a new
        # collar or neckline has room and the old collar's trim does not survive as a thin line.
        top_rows = np.nonzero(np.isin(labels, [UPPER, DRESS]).any(axis=1))[0]
        if len(top_rows):
            shoulder_y = int(top_rows.min())
            neck = np.zeros_like(protect)
            neck[shoulder_y - unit :, :] = True
            protect &= ~(neck & (labels == FACE))
            # Hair lying over the clothes (shoulder-length hair on a collar): the new collar and
            # shoulders go under it, so the model redraws those tips over the new garment. Locked,
            # they keep bits of the old top between the strands and the new jacket stops at them
            # in a hard line, which reads as hair cut off.
            tips = (labels == HAIR) & _rows_from(labels.shape, shoulder_y + unit)
            protect &= ~tips
            region |= tips & _dilate(core, unit * 3)
    if category == "upper":
        lower = np.isin(labels, [PANTS, SKIRT, L_LEG, R_LEG, L_SHOE, R_SHOE])
        # Let a top run over the waistband: leave the upper part of the trousers repaintable,
        # otherwise a top longer than the one in the photo is squeezed short.
        # Sized from the torso, not the legs: on a full-length photo a leg-based margin is huge,
        # and whatever the mask covers gets filled (a short vest then grows a band of fake fabric).
        rows = np.nonzero(np.isin(labels, [PANTS, SKIRT, BELT]).any(axis=1))[0]
        top_rows = np.nonzero((labels == UPPER).any(axis=1))[0]
        if len(rows) and len(top_rows):
            waist_y = int(rows.min())
            torso = max(1, waist_y - int(top_rows.min()))
            if top_hem_frac is not None:
                # Measured on the skeleton: the hem lands where it lands on the model's body
                # (a blazer at the hips, a coat at mid-thigh), plus a small margin for the hem itself.
                hem_y = int(top_hem_frac * h) + unit
                extra = (hem_y - waist_y) / torso
            else:
                # The new top's own length, in proportion to this body; at least a little past the waist.
                extra = max(0.10, (top_length or 1.0) - 1.0 + 0.06)
                hem_y = waist_y + int(torso * extra)
            hem_y = max(hem_y, waist_y + int(torso * 0.06))
            lower[:hem_y, :] = False
            if extra > 0.3:
                # A long top (coat) also needs room on either side of the hips: open each row from
                # the outside of the trousers, like for bottoms.
                legs = np.isin(labels, [PANTS, SKIRT, L_LEG, R_LEG])
                for y in range(waist_y, min(labels.shape[0], hem_y)):
                    xs = np.nonzero(legs[y])[0]
                    if len(xs):
                        region[y, max(0, xs.min() - unit) : xs.max() + unit + 1] = True
        protect |= lower
    if category == "lower":
        protect |= np.isin(labels, [UPPER, L_ARM, R_ARM, L_SHOE, R_SHOE])
    region &= ~_dilate(protect, 1)
    if category in ("upper", "dress"):
        # The parser has no "hand" label: hands are part of the arm. Keep the lowest part of each
        # arm (the hand) out of the mask so the model can't swallow it into a sleeve.
        region &= ~hands_mask(labels)
    return region


def hands_mask(labels: np.ndarray, share: float = 0.13) -> np.ndarray:
    """Approximate hands: the bottom `share` of each arm's vertical extent (works for relaxed arms)."""
    out = np.zeros(labels.shape, dtype=bool)
    for arm in (L_ARM, R_ARM):
        arm_px = labels == arm
        rows = np.nonzero(arm_px.any(axis=1))[0]
        if len(rows) < 10:
            continue
        cut = int(rows.max() - (rows.max() - rows.min()) * share)
        hand = arm_px.copy()
        hand[:cut, :] = False
        out |= hand
    return _dilate(out, 2)


def garment_area(labels: np.ndarray, category: Category) -> np.ndarray:
    """Pixels that belong to clothing of this category plus the arms (sleeves change arm coverage)."""
    if category == "upper":
        ids = [UPPER, SCARF, L_ARM, R_ARM]
    elif category == "lower":
        ids = [PANTS, SKIRT, BELT, L_LEG, R_LEG]
    else:
        ids = [UPPER, SKIRT, PANTS, DRESS, BELT, SCARF, L_ARM, R_ARM, L_LEG, R_LEG]
    return np.isin(labels, ids)


def core_hard(head: np.ndarray, labels: np.ndarray, unit: int) -> np.ndarray:
    """The face itself (not hair tips): always fully the user's own pixels."""
    face = head & (labels == FACE)
    return (~_dilate(~face, 2)).astype(np.float32)


def _rows_from(shape: tuple[int, int], y: int) -> np.ndarray:
    m = np.zeros(shape, dtype=bool)
    m[max(0, y) :, :] = True
    return m


def head_mask(labels: np.ndarray) -> np.ndarray:
    """Face, hair, hat, glasses — down to the top of the clothing, never below it.

    The parser labels the neck as "face"; only the part above the top of the clothes counts as head
    here, so a new collar can still be drawn where the neck meets the garment.
    """
    out = np.zeros(labels.shape, dtype=bool)
    for band in split_people(labels):  # per person: each neck ends at that person's own collar
        own = np.where(band, labels, BG)
        head = np.isin(own, [FACE, HAIR, HAT, SUNGLASSES])
        clothes_rows = np.nonzero(np.isin(own, [UPPER, DRESS, SCARF]).any(axis=1))[0]
        if len(clothes_rows):
            y = int(clothes_rows.min())
            head[y:, :] &= ~(own[y:, :] == FACE)
        out |= head & band
    return out


def _erode_mask(m: np.ndarray, px: int) -> np.ndarray:
    return ~_dilate(~m, px)


def lock_head(original: Image.Image, edited: Image.Image, feather_px: int = 3) -> Image.Image:
    """Paste the user's own face and hair back, pixel for pixel. Always applied: we never edit the face.

    Hair tips lying over the clothes are the exception (the garment area includes them, see
    _agnostic_one): the lock fades out across the collar line instead of ending in a hard edge.
    """
    import cv2

    edited = edited.resize(original.size, Image.LANCZOS)
    labels = parser.labels(original)
    head = head_mask(labels)
    if not head.any():
        return edited
    # Shrink by a pixel before feathering so the soft edge falls inside the head, not on the collar.
    core = ~_dilate(~head, 1)
    alpha = np.asarray(
        Image.fromarray((core * 255).astype(np.uint8), mode="L").filter(ImageFilter.GaussianBlur(feather_px)),
        dtype=np.float32,
    ) / 255.0
    h, w = labels.shape
    unit = max(4, int(min(h, w) * 0.02))
    # Hair tips lying over the clothes: keep each strand, not the old top showing between them.
    # Per pixel: does it look like this person's hair, or like the clothes they had on?
    lab = cv2.cvtColor(np.asarray(original.convert("RGB")), cv2.COLOR_RGB2LAB).astype(np.float32)
    wts = np.array([0.5, 1.0, 1.0], np.float32)
    for band in split_people(labels):
        own = np.where(band, labels, BG)
        rows = np.nonzero(np.isin(own, [UPPER, DRESS]).any(axis=1))[0]
        if not len(rows):
            continue
        cut = int(rows.min()) + unit  # same line as the mask uses
        hair = (own == HAIR) & band
        tips = hair & _rows_from(labels.shape, cut - unit)
        ref_zone = hair & ~_rows_from(labels.shape, cut - unit) & _rows_from(labels.shape, cut - 4 * unit)
        cloth = np.isin(own, [UPPER, DRESS]) & ~_dilate(hair, 2)
        if tips.sum() < 20 or ref_zone.sum() < 50 or cloth.sum() < 200:
            alpha = np.where(tips, 0.0, alpha)
            continue
        hair_ref = np.median(lab[ref_zone], 0)
        cpx = lab[cloth]
        k = 3 if len(cpx) > 600 else 1
        if k > 1:
            crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
            _, _, centres = cv2.kmeans(cpx, k, None, crit, 2, cv2.KMEANS_PP_CENTERS)
        else:
            centres = cpx.mean(0, keepdims=True)
        ys, xs = np.nonzero(tips)
        px = lab[ys, xs]
        d_hair = np.linalg.norm((px - hair_ref) * wts, axis=1)
        d_cloth = np.min(np.linalg.norm((px[:, None, :] - centres[None]) * wts, axis=2), axis=1)
        # Hair stays hers unless a pixel is clearly the old top (much nearer its colours than the
        # hair's). Blonde ends look a lot like a white or pink print; dropping every doubtful pixel
        # cut her hair off at the shoulders.
        matte = 1.0 - np.clip((d_hair - d_cloth - 12.0) / 20.0, 0, 1)
        alpha[ys, xs] = np.maximum(alpha[ys, xs], matte)
    alpha = cv2.GaussianBlur(alpha, (0, 0), 0.8)
    alpha = np.maximum(alpha, core_hard(head, labels, unit))
    a = Image.fromarray((np.clip(alpha, 0, 1) * 255).astype(np.uint8), mode="L")
    return Image.composite(original, edited, a)


def garment_rgba(img: Image.Image, category: Category) -> Image.Image:
    """The result with only its clothes opaque: what Live mode lays over the moving camera image.
    Face, hair, hands, skin and background are transparent, so the live person shows through."""
    labels = parser.labels(img)
    ids = {
        "upper": [UPPER, SCARF],
        "lower": [PANTS, SKIRT, BELT],
        "dress": [UPPER, SKIRT, PANTS, DRESS, BELT, SCARF],
    }[category]
    clothes = _dilate(np.isin(labels, ids), 1)
    alpha = Image.fromarray((clothes * 255).astype(np.uint8), mode="L").filter(ImageFilter.GaussianBlur(1.2))
    out = img.convert("RGBA")
    out.putalpha(alpha)
    return out


def lock_outside_garment(
    original: Image.Image, edited: Image.Image, category: Category, feather_px: int = 6
) -> Image.Image:
    """Paste the original pixels back everywhere except the garment area of either image.

    Keeps the face, hair and background byte-identical to the user's photo after an edit.
    """
    edited = edited.resize(original.size, Image.LANCZOS)
    area = garment_area(parser.labels(original), category) | garment_area(
        parser.labels(edited), category
    )
    unit = max(3, int(min(original.size) * 0.012))
    area = _dilate(area, unit)
    alpha = Image.fromarray((area * 255).astype(np.uint8), mode="L").filter(
        ImageFilter.GaussianBlur(feather_px)
    )
    return Image.composite(edited, original, alpha)
