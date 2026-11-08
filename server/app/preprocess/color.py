"""Colour fidelity: pull the generated garment's colour back to the garment photo's colour.

Diffusion try-on models "harmonise" the garment with the scene. Under strong warm light (sunset
photos) CatVTON turned a black sweater mustard-brown while getting its shape and folds right.
We fix that after generation: measure the garment's colour statistics (mean and spread in Lab)
in the product photo and in the result, then shift the result's garment pixels to match.
The local light/dark variation (folds, shadows) is kept; only the overall colour moves.
"""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image, ImageFilter

from .parsing import Category, garment_area, parser, UPPER, PANTS, SKIRT, DRESS, SCARF, BELT

GARMENT_LABELS = {
    "upper": [UPPER, SCARF],
    "lower": [PANTS, SKIRT, BELT],
    "dress": [UPPER, PANTS, SKIRT, DRESS, BELT, SCARF],
}


def match_skin_tone(
    result: Image.Image, original: Image.Image, strength: float = 0.8
) -> tuple[Image.Image, dict]:
    """Match newly generated skin (chest, forearms, hands) to the person's real skin tone.

    The model has to invent skin the old clothes covered, and under strong coloured light it gets
    the tone wrong (pale/grey arms next to a warm, sunlit face). Reference = skin the parser finds
    in the original photo (face, plus any arms that were already visible).
    """
    from .parsing import FACE, L_ARM, R_ARM

    info = {"applied": False}
    original = original.resize(result.size, Image.LANCZOS)
    o_labels = parser.labels(original)
    r_labels = parser.labels(result)
    o_lab, r_lab = _lab(original), _lab(result)

    ref_mask = np.isin(o_labels, [FACE, L_ARM, R_ARM])
    # Skin in the result that wasn't skin in the original = newly generated.
    new_skin = np.isin(r_labels, [FACE, L_ARM, R_ARM]) & ~ref_mask
    if ref_mask.sum() < 400 or new_skin.sum() < 400:
        return result, info

    ref = o_lab[ref_mask]
    src = r_lab[new_skin]
    r_mean, s_mean = ref.mean(0), src.mean(0)
    delta = float(np.linalg.norm(r_mean - s_mean))
    info["delta_e"] = round(delta, 1)
    if delta < 5:
        return result, info

    # Shift colour (a, b) fully and lightness partly: lighting differs across the body.
    shift = (r_mean - s_mean) * np.array([0.6, 1.0, 1.0], dtype=np.float32)
    out = r_lab.copy()
    out[new_skin] = r_lab[new_skin] + shift * strength
    out = np.clip(out, 0, 255).astype(np.uint8)
    corrected = Image.fromarray(cv2.cvtColor(out, cv2.COLOR_LAB2RGB))
    alpha = Image.fromarray((new_skin * 255).astype(np.uint8), "L").filter(ImageFilter.GaussianBlur(3))
    info["applied"] = True
    return Image.composite(corrected, result, alpha), info


def _lab(img: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.asarray(img.convert("RGB")), cv2.COLOR_RGB2LAB).astype(np.float32)


def garment_pixels_from_product(garment: Image.Image) -> np.ndarray:
    """Lab pixels of the garment in a product photo (everything that isn't the plain background)."""
    lab = _lab(garment)
    rgb = np.asarray(garment.convert("RGB")).astype(np.int16)
    corner = np.median(
        np.concatenate([rgb[:8, :8].reshape(-1, 3), rgb[:8, -8:].reshape(-1, 3),
                        rgb[-8:, :8].reshape(-1, 3), rgb[-8:, -8:].reshape(-1, 3)]), axis=0
    )
    fg = np.abs(rgb - corner).sum(axis=2) > 40
    # Drop a thin border so blurry edges against the background don't bias the colour.
    fg = cv2.erode(fg.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    return lab[fg]


RESULT_PART_LABELS = {
    "upper": [UPPER, SCARF],
    "lower": [PANTS, SKIRT, BELT],
    "dress": [DRESS],
}


def match_garment_color(
    result: Image.Image,
    parts: dict[str, np.ndarray],
    category: Category,
    strength: float = 0.9,
) -> tuple[Image.Image, dict]:
    """Correct each garment piece in the result against that piece's own colour.

    `parts` comes from garment.color_parts() on the original garment photo:
    {"upper": px, "lower": px, "dress": px} for a photo of a person, or {"whole": px} for a
    product shot (one garment, whatever the category).
    """
    labels = parser.labels(result)
    lab = _lab(result)
    out_img = result
    info: dict = {"applied": False, "parts": {}}

    if "whole" in parts:
        # Product photo: the single garment maps to the category's area in the result.
        plan = [("whole", parts["whole"], GARMENT_LABELS[category])]
    else:
        wanted = {"upper": ["upper"], "lower": ["lower"], "dress": ["upper", "lower", "dress"]}[category]
        plan = [(p, parts[p], RESULT_PART_LABELS[p]) for p in wanted if p in parts]
        # A one-piece dress in the photo, but the result parser calls it top + skirt: use the dress
        # colour for those regions too.
        if category == "dress" and "dress" in parts and "upper" not in parts:
            plan.append(("dress", parts["dress"], RESULT_PART_LABELS["upper"] + RESULT_PART_LABELS["lower"]))

    for name, ref, result_ids in plan:
        area = np.isin(labels, result_ids)
        if len(ref) < 300 or area.sum() < 300:
            continue
        src = lab[area]
        r_mean, r_std = ref.mean(0), ref.std(0) + 1e-3
        s_mean, s_std = src.mean(0), src.std(0) + 1e-3
        # Distance between average colours in Lab (~2.3 = just noticeable).
        delta = float(np.linalg.norm(r_mean - s_mean))
        info["parts"][name] = round(delta, 1)
        if delta < 6:
            continue
        # Colour (a, b): mean/spread transfer, so a black sweater does not drift tan or green.
        # Lightness: the scene's light decides it. A black sweater measured in sunshine reads
        # charcoal; pulling the result up to that turned a correct deep black into a flat grey.
        # So lightness only comes DOWN (a model that bleached the garment), never up, and its
        # spread is left alone: those are the folds and shadows of the person's own pose.
        scale = np.clip(r_std / s_std, 0.5, 1.5)
        new = (lab - s_mean) * scale + r_mean
        lift = s_mean[0] - r_mean[0]
        new[..., 0] = lab[..., 0] - (lift if lift > 10 else 0.0)
        piece = lab.copy()
        piece[area] = lab[area] * (1 - strength) + new[area] * strength
        corrected = Image.fromarray(cv2.cvtColor(np.clip(piece, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB))
        alpha = Image.fromarray((area * 255).astype(np.uint8), "L").filter(ImageFilter.GaussianBlur(2))
        out_img = Image.composite(corrected, out_img, alpha)
        lab = _lab(out_img)
        info["applied"] = True
    return out_img, info


def match_kept_clothes(result: Image.Image, original: Image.Image, category: Category) -> tuple[Image.Image, dict]:
    """Clothes the user keeps must stay their colour where the model had to draw them anew.

    A long blazer covers the hips; swap it for a short sweater and the model must invent the
    trousers underneath. It does, but from the outfit photo (grey chinos on navy jeans). Here the
    newly drawn part of the kept garment gets the colour of the visible, untouched part.
    """
    kept_ids = {"upper": [PANTS, SKIRT], "lower": [UPPER], "dress": []}[category]
    if not kept_ids:
        return result, {"applied": False}
    result = result.resize(original.size, Image.LANCZOS)
    o_lab, r_lab = _lab(original), _lab(result)
    o_kept = np.isin(parser.labels(original), kept_ids)
    r_kept = np.isin(parser.labels(result), kept_ids)
    changed = np.abs(o_lab - r_lab).max(axis=2) > 14
    changed = cv2.dilate(changed.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    ref_m = cv2.erode((o_kept & r_kept & ~changed).astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    area = r_kept & changed
    if ref_m.sum() < 500 or area.sum() < 300:
        return result, {"applied": False}
    ref, src = o_lab[ref_m], r_lab[area]
    r_mean, r_std = ref.mean(0), ref.std(0) + 1e-3
    s_mean, s_std = src.mean(0), src.std(0) + 1e-3
    delta = float(np.linalg.norm(r_mean - s_mean))
    if delta < 6:
        return result, {"applied": False, "delta_e": round(delta, 1)}
    scale = np.clip(r_std / s_std, 0.6, 1.4)
    new = (r_lab - s_mean) * scale + r_mean
    piece = r_lab.copy()
    piece[area] = new[area]
    corrected = Image.fromarray(cv2.cvtColor(np.clip(piece, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB))
    alpha = Image.fromarray((area * 255).astype(np.uint8), "L").filter(ImageFilter.GaussianBlur(2))
    return Image.composite(corrected, result, alpha), {"applied": True, "delta_e": round(delta, 1)}


_W = np.array([0.5, 1.0, 1.0], np.float32)  # hue weighs more than shade when assigning pixels


def match_layers_color(result: Image.Image, layers: list, strength: float = 0.9) -> tuple[Image.Image, dict]:
    """Colour correction per LAYER, not per body part.

    An open camel coat over a black turtleneck is one "upper clothes" region to the parser; one
    average for it pulled the coat toward brown and the turtleneck toward camel. Here each pixel of
    a region goes to the layer whose colour it is closest to, and each layer is corrected on its
    own: hue fully, lightness only downward (the scene's light decides how bright fabric looks).
    """
    labels = parser.labels(result)
    lab = _lab(result)
    flat = lab.reshape(-1, 3)
    new = flat.copy()
    touched = np.zeros(len(flat), bool)
    info: dict = {"applied": False, "layers": {}}
    regions = {"upper": [UPPER, SCARF], "lower": [PANTS, SKIRT, BELT], "dress": [DRESS]}
    for part, ids in regions.items():
        mine = [layer for layer in layers if layer.part == part and layer.role != "under"]
        if not mine:
            continue
        area = np.flatnonzero(np.isin(labels, ids).ravel())
        if len(area) < 300:
            continue
        px = flat[area]
        refs = np.array([layer.lab for layer in mine], np.float32)
        nearest = np.linalg.norm((px[:, None, :] - refs[None]) * _W, axis=2).argmin(1)
        for i, layer in enumerate(mine):
            sel = area[nearest == i]
            if len(sel) < 200:
                continue
            s_mean = flat[sel].mean(0)
            r = refs[i]
            delta = float(np.linalg.norm(s_mean - r))
            info["layers"][layer.name] = round(delta, 1)
            if delta < 6:
                continue
            shift = np.zeros(3, np.float32)
            shift[1:] = r[1:] - s_mean[1:]
            lift = s_mean[0] - r[0]
            if lift > 10:
                shift[0] = -lift
            new[sel] = flat[sel] + shift * strength
            touched[sel] = True
            info["applied"] = True
    if not info["applied"]:
        return result, info
    corrected = Image.fromarray(
        cv2.cvtColor(np.clip(new.reshape(lab.shape), 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB)
    )
    alpha = Image.fromarray((touched.reshape(lab.shape[:2]) * 255).astype(np.uint8), "L").filter(
        ImageFilter.GaussianBlur(1.5)
    )
    return Image.composite(corrected, result, alpha), info
