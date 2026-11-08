"""Read an outfit photo as a checklist of layers, each with its own colour.

A lookbook photo is rarely one garment: an open camel coat over a black turtleneck with black
trousers is three pieces. The human parser calls the coat and the turtleneck both "upper clothes",
so treating "upper" as one garment averaged them into a beige sweater and lost the coat. Here the
upper region is split by colour into an outer layer (the colour that reaches the sides and the
sleeves) and an inner one (the colour down the middle), plus any under-layer peeking out at the
hem (a white T-shirt), and the bottoms are read on their own.

The result drives three things:
- the prompt: a numbered list of every piece and its colour, so nothing is forgotten;
- the colour check after generation (is every piece there, in its colour?);
- the colour correction, per layer instead of one average.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Optional

import cv2
import numpy as np
from PIL import Image

from .body import length_words, skeleton, where
from .describe import colour_name
from .parsing import (
    DRESS, FACE, L_ARM, L_SHOE, PANTS, R_ARM, R_SHOE, SCARF, SKIRT, UPPER, Category, parser,
)


@dataclass
class Layer:
    role: str  # outer | top | under | bottom | dress | garment
    name: str  # "camel long coat, worn open"
    colour: str  # "camel"
    lab: list[float]  # mean Lab (OpenCV scale) of the piece in the outfit photo
    part: str  # which body part it sits on: upper | lower | dress
    # Where the piece ends on the body (preprocess.body: 0 shoulders, 1 hips, 2 knees, 3 ankles).
    hem: Optional[float] = None
    # Words for the model only ("ending just below the hips"); not shown to people.
    detail: str = ""

    def public(self) -> dict:
        return {"role": self.role, "name": short_name(self.name), "part": self.part}

    def prompt(self) -> str:
        return f"{self.name}, {self.detail}" if self.detail else self.name


@dataclass
class Outfit:
    person: bool
    layers: list[Layer] = field(default_factory=list)
    suggested: Category = "upper"
    # Top length relative to the torso (shoulder line to waistband): 1.0 ends at the waist,
    # 1.6 is a coat to mid-thigh. Used to size the new garment on the user's body in proportion.
    top_length: Optional[float] = None
    # Hem of the top's outermost layer on the body ruler (preprocess.body), when a body was found.
    top_hem: Optional[float] = None
    shoes: Optional[str] = None

    def for_category(self, category: Category) -> list[Layer]:
        parts = {"upper": {"upper"}, "lower": {"lower"}, "dress": {"upper", "lower", "dress"}}[category]
        return [layer for layer in self.layers if layer.part in parts]

    def public(self) -> dict:
        return {
            "person": self.person,
            "suggested": self.suggested,
            "layers": [layer.public() for layer in self.layers],
            "shoes": self.shoes,
        }


def short_name(name: str) -> str:
    """The name people read: without the instructions meant for the model."""
    # Everything after the first comma is for the model ("worn open", "with the same neckline").
    return name.split(", ")[0]


def _erode(mask: np.ndarray, px: int = 2) -> np.ndarray:
    k = np.ones((px * 2 + 1, px * 2 + 1), np.uint8)
    return cv2.erode(mask.astype(np.uint8), k) > 0


def _de(a, b) -> float:
    return float(np.linalg.norm(np.asarray(a, np.float32) - np.asarray(b, np.float32)))


def _distinct(a, b) -> bool:
    """Two colours are two garments only when they differ in hue or tint, or are worlds apart in
    lightness (a black top under a white coat). Light and shadow of one fabric (a herringbone, a
    tweed, a khaki shirt in side light) differ in lightness alone and are one garment."""
    a, b = np.asarray(a, np.float32), np.asarray(b, np.float32)
    return float(np.hypot(a[1] - b[1], a[2] - b[2])) > 10 or abs(float(a[0] - b[0])) > 90


def _garments(clusters: list) -> list:
    """Colour clusters -> garments. Same hue (within 10 in a/b) is one fabric unless its
    lightness has a real gap (more than 80): a khaki shirt in sun and shade, or a herringbone's
    light and dark threads, is one garment; a black top under a white coat is two."""
    groups: list[list] = []
    for c in clusters:
        for g in groups:
            if any(np.hypot(c[0][1] - o[0][1], c[0][2] - o[0][2]) <= 10 for o in g):
                g.append(c)
                break
        else:
            groups.append([c])
    out = []
    for g in groups:
        g = sorted(g, key=lambda c: float(c[0][0]))
        run = [g[0]]
        for c in g[1:]:
            if float(c[0][0]) - float(run[-1][0][0]) > 80:
                out.append(run)
                run = [c]
            else:
                run.append(c)
        out.append(run)
    merged = []
    for run in out:
        idx = np.concatenate([c[1] for c in run])
        weights = np.array([len(c[1]) for c in run], np.float32)
        centre = (np.stack([c[0] for c in run]) * weights[:, None]).sum(0) / weights.sum()
        merged.append((centre.astype(np.float32), idx))
    return sorted(merged, key=lambda c: -len(c[1]))


def _clusters(lab_px: np.ndarray, k: int = 3) -> list[tuple[np.ndarray, np.ndarray]]:
    """k-means in Lab; returns [(centre, member indices)] largest first, near-duplicates merged."""
    if len(lab_px) < k * 50:
        return [(lab_px.mean(0), np.arange(len(lab_px)))]
    data = lab_px.astype(np.float32)
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.5)
    _, lbl, centres = cv2.kmeans(data, k, None, crit, 3, cv2.KMEANS_PP_CENTERS)
    lbl = lbl.ravel()
    groups = [[i] for i in range(k)]
    # Merge clusters that are the same colour (light and shadow of one fabric).
    merged = True
    while merged:
        merged = False
        for a in range(len(groups)):
            for b in range(a + 1, len(groups)):
                ca = data[np.isin(lbl, groups[a])].mean(0)
                cb = data[np.isin(lbl, groups[b])].mean(0)
                # Lightness differences are mostly shading: weigh hue more.
                d = np.linalg.norm((ca - cb) * np.array([0.5, 1, 1], np.float32))
                if d < 14:
                    groups[a] += groups.pop(b)
                    merged = True
                    break
            if merged:
                break
    out = []
    for g in groups:
        idx = np.nonzero(np.isin(lbl, g))[0]
        out.append((data[idx].mean(0), idx))
    return sorted(out, key=lambda c: -len(c[1]))


def analyse_outfit(img: Image.Image) -> Outfit:
    img = img.convert("RGB")
    labels = parser.labels(img)
    lab = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2LAB).astype(np.float32)
    h, w = labels.shape
    body = np.isin(labels, [FACE, L_ARM, R_ARM, UPPER, PANTS, SKIRT, DRESS]).mean()
    if body < 0.02:
        # A product photo: one garment, whatever it is.
        from .color import garment_pixels_from_product

        px = garment_pixels_from_product(img)
        if len(px) < 300:
            return Outfit(person=False)
        c = colour_name(px)
        return Outfit(
            person=False,
            layers=[Layer("garment", f"{c} garment", c, px.mean(0).tolist(), "upper")],
            suggested="upper",
        )

    out = Outfit(person=True)
    sk = skeleton(img)

    def hem_of(rows: np.ndarray) -> Optional[float]:
        # The lowest rows of a piece, ignoring a few stray pixels (a strap, a hand's shadow).
        if sk is None or len(rows) == 0:
            return None
        bottom = float(np.percentile(rows, 98))
        if bottom >= h * 0.97:
            return None  # runs off the bottom of the photo: its real length is unknown
        return round(sk.to_body(bottom), 2)
    upper_m = _erode(np.isin(labels, [UPPER, SCARF]))
    lower_m = _erode(np.isin(labels, [PANTS, SKIRT]))
    dress_m = _erode(labels == DRESS)
    arms = np.isin(labels, [L_ARM, R_ARM])

    if upper_m.sum() > 400:
        ys, xs = np.nonzero(upper_m)
        px = lab[upper_m]
        clusters = _garments(_clusters(px, 4))
        total = len(px)
        big = [c for c in clusters if len(c[1]) / total >= 0.12]
        x0, x1 = xs.min(), xs.max()
        bw = max(1, x1 - x0)
        y0, y1 = ys.min(), ys.max()

        # Sleeves: forearm skin relative to the top (same rule as describe.top_words).
        share = int(arms.sum()) / max(1, int(arms.sum()) + int(upper_m.sum()))
        sleeves = "short-sleeve" if share > 0.35 else "long-sleeve"

        # How long the top is, against the torso (shoulder line to the trousers' waistband).
        lrows = np.nonzero(np.isin(labels, [PANTS, SKIRT]).any(axis=1))[0]
        if len(lrows):
            waist = int(lrows.min())
            torso = max(1, waist - int(y0))
            out.top_length = round(float((y1 - y0) / torso), 2)
        top_hem = hem_of(ys)
        out.top_hem = top_hem
        # Long by the body ruler when there is one; the torso rule misreads high-waisted skirts.
        long_top = top_hem >= 1.6 if top_hem is not None else (out.top_length or 1.0) > 1.3

        def side_share(idx: np.ndarray) -> float:
            cx = xs[idx]
            return float(((cx < x0 + 0.2 * bw) | (cx > x1 - 0.2 * bw)).mean())

        # The outer layer: the big colour that holds the sides and sleeves. The layer under it: the
        # most different colour that sits down the middle (not just the second-largest cluster:
        # light and shadow on a camel coat are two clusters of the same fabric).
        outer = max(big, key=lambda c: side_share(c[1]) * len(c[1])) if big else None
        inner = None
        if outer is not None:
            cands = [
                c for c in clusters
                if c is not outer and len(c[1]) / total >= 0.06 and _distinct(c[0], outer[0])
                and side_share(c[1]) < side_share(outer[1])
            ]
            if cands:
                inner = max(cands, key=lambda c: _de(c[0], outer[0]))
        # The top under a jacket, found by where it is rather than by colour alone (shadows in a
        # tweed are a "different colour" too): a region unlike the jacket that runs down the
        # centre from the neckline. How far down it reaches says open (all the way) or buttoned
        # (a V at the chest).
        opening = _opening(upper_m, lab, outer[0], y0, y1) if outer is not None else None
        # A second colour with no opening down the centre is shading in one fabric, not a layer.
        if inner is not None and opening is None:
            inner = None
        if outer is not None and inner is not None:
            oc = colour_name(px[outer[1]])
            ic = colour_name(lab[opening[2]]) if opening is not None else colour_name(px[inner[1]])
            o_hem = hem_of(ys[outer[1]]) or top_hem
            if o_hem is not None:
                kind = "long coat" if o_hem >= 1.6 else ("coat" if o_hem >= 1.2 else "jacket")
                length = "" if kind == "long coat" else f"{length_words(o_hem, 'top')} "
            else:
                kind, length = ("long coat" if long_top else "jacket"), ""
            # Open or closed: an open jacket shows the top under it all the way down; a buttoned one
            # only at the neckline (a V at the chest).
            depth = opening[3] if opening is not None else (np.percentile(ys[inner[1]], 95) - y0) / max(1, y1 - y0)
            if depth > 0.75:
                worn = "worn open"
            elif depth > 0.5:
                worn = "open at the chest and closed at the waist"
            else:
                worn = "closed, the top under it showing only at the neckline"
            detail = f"ending {where(o_hem)}, no longer and no shorter" if o_hem is not None else ""
            out.layers.append(
                Layer("outer", f"{oc} {length}{kind}, {worn}", oc, outer[0].tolist(), "upper", o_hem, detail)
            )
            out.layers.append(
                Layer("top", f"{ic} top under the {kind}, with the same neckline as in image 2", ic, inner[0].tolist(), "upper")
            )
        else:
            # One garment: all its shades together (light and shadow of the same fabric), not
            # just the biggest cluster, which may be only its shadows.
            main_c = big[0] if big else (px.mean(0), np.arange(len(px)))
            same = np.concatenate([idx for cen, idx in clusters if not _distinct(cen, main_c[0])] or [main_c[1]])
            c = colour_name(px[same])
            centre = np.median(px[same], 0).tolist()
            # "top" is only our word for it: the photo says whether it is a jacket, a cardigan, a
            # shirt or a sweater. Named "top" alone, the model turned a buttoned herringbone jacket
            # into a plain V-neck sweater.
            kind = ("the same kind of garment as in image 2 (jacket, cardigan, shirt or sweater, as shown),"
                    " with its exact collar or neckline, buttons or zip, pockets and fabric texture")
            if top_hem is not None:
                name = f"{c} {sleeves} {length_words(top_hem, 'top')} top"
                detail = f"{kind}, ending {where(top_hem)}"
            else:
                name, detail = f"{c} {sleeves} {'long top' if long_top else 'top'}", kind
            out.layers.append(Layer("top", name, c, centre, "upper", top_hem, detail))

        # An under-layer showing at the hem (white T-shirt under a sweater): a small, distinct
        # colour concentrated in the bottom fifth of the top.
        main = [layer.lab for layer in out.layers]
        for centre, idx in clusters:
            share_c = len(idx) / total
            if not (0.02 <= share_c < 0.12):
                continue
            if min(_de(centre, m) for m in main) < 25:
                continue
            low = ys[idx] > y1 - 0.25 * (y1 - y0)
            if low.mean() > 0.6:
                c = colour_name(px[idx])
                out.layers.append(Layer("under", f"{c} T-shirt hem showing below it", c, centre.tolist(), "upper"))
                break
        # A hem band can also be its own parser region right under the top; check just below.
        # A T-shirt showing at the neckline of a jacket or cardigan: a small, distinct colour at
        # the top centre of the top (a grey tee in a V-neck).
        if not any(layer.role in ("under", "top") and layer is not out.layers[0] for layer in out.layers[1:]):
            main = [layer.lab for layer in out.layers]
            cx = float(np.median(xs))
            for centre, idx in clusters:
                share_c = len(idx) / total
                if not (0.015 <= share_c < 0.15) or any(not _distinct(centre, m) for m in main):
                    continue
                top_part = ys[idx] < y0 + 0.4 * (y1 - y0)
                central = np.abs(xs[idx] - cx) < 0.2 * bw
                if (top_part & central).mean() > 0.6:
                    c = colour_name(px[idx])
                    out.layers.append(Layer("under", f"{c} T-shirt showing at the neckline", c, centre.tolist(), "upper"))
                    break
        if not any(layer.role == "under" for layer in out.layers) and len(lrows):
            band = np.zeros_like(upper_m)
            band[y1 - int(0.06 * (y1 - y0)) : y1 + int(0.04 * (y1 - y0)), x0:x1] = True
            bpx = lab[band & ~arms & (labels != 0)]
            if len(bpx) > 60:
                cent = np.median(bpx, 0)
                if cent[0] > 200 and min(_de(cent, m) for m in main) > 40:
                    out.layers.append(Layer("under", "white T-shirt hem showing below it", "white", cent.tolist(), "upper"))

    if lower_m.sum() > 400:
        px = lab[lower_m]
        c = colour_name(px)
        kind = "skirt" if (labels == SKIRT).sum() > (labels == PANTS).sum() else "trousers"
        # Only the piece's own label: sheer tights under a skirt read as "trousers" and would
        # make the skirt look longer.
        hem = hem_of(np.nonzero(labels == (SKIRT if kind == "skirt" else PANTS))[0])
        length = f"{length_words(hem, kind)} " if hem is not None else ""
        detail = f"ending {where(hem)}" if hem is not None else ""
        out.layers.append(Layer("bottom", f"{c} {length}{kind}", c, np.median(px, 0).tolist(), "lower", hem, detail))

    if dress_m.sum() > 400:
        px = lab[dress_m]
        c = colour_name(px)
        hem = hem_of(np.nonzero(dress_m)[0])
        length = f"{length_words(hem, 'dress')} " if hem is not None else ""
        detail = f"ending {where(hem)}" if hem is not None else ""
        out.layers.append(Layer("dress", f"{c} {length}dress", c, np.median(px, 0).tolist(), "dress", hem, detail))

    shoes = _erode(np.isin(labels, [L_SHOE, R_SHOE]))
    if shoes.sum() > 150:
        out.shoes = f"{colour_name(lab[shoes])} shoes"

    has_upper = any(layer.part == "upper" for layer in out.layers)
    has_lower = any(layer.part in ("lower", "dress") for layer in out.layers)
    out.suggested = "dress" if has_upper and has_lower else ("lower" if has_lower else "upper")
    return out


def _opening(upper_m: np.ndarray, lab: np.ndarray, outer_lab, y0: int, y1: int):
    """The visible part of the top under a jacket: (median Lab, -, pixel mask, depth 0..1) or None."""
    ys, xs = np.nonzero(upper_m)
    if len(xs) == 0:
        return None
    cx = float(np.median(xs))
    bw = float(xs.max() - xs.min())
    strip = np.zeros_like(upper_m)
    strip[:, int(cx - 0.15 * bw) : int(cx + 0.15 * bw) + 1] = True
    w = np.array([0.5, 1, 1], np.float32)
    d = np.linalg.norm((lab - np.asarray(outer_lab, np.float32)) * w, axis=2)
    far = (upper_m & strip & (d > 18)).astype(np.uint8)
    far = cv2.morphologyEx(far, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, comp, stats, _ = cv2.connectedComponentsWithStats(far, 8)
    span = max(1, y1 - y0)
    best = None
    for i in range(1, n):
        top, height, area = stats[i, cv2.CC_STAT_TOP], stats[i, cv2.CC_STAT_HEIGHT], stats[i, cv2.CC_STAT_AREA]
        # Starts at the neckline (top third of the top) and is a real patch, not a fold.
        if top - y0 > 0.3 * span or area < 0.004 * upper_m.sum() or height < 0.08 * span:
            continue
        if best is None or area > stats[best, cv2.CC_STAT_AREA]:
            best = i
    if best is None:
        return None
    m = comp == best
    depth = (stats[best, cv2.CC_STAT_TOP] + stats[best, cv2.CC_STAT_HEIGHT] - y0) / span
    return np.median(lab[m], 0), None, m, float(depth)


def checklist_prompt(layers: list[Layer]) -> str:
    """'1. an open camel long coat; 2. a black turtleneck ...' for the editing model."""
    items = [f"{i}. {layer.prompt()}" for i, layer in enumerate(layers, 1)]
    return "; ".join(items)


def verify(result: Image.Image, layers: list[Layer], original: Optional[Image.Image] = None) -> list[dict]:
    """Is each piece of the checklist in the result, in its colour? One entry per layer.

    The outfit photo and the user's photo have different light (stone trousers in sunshine look
    darker in a lift), so a fixed colour distance calls a correct result "missing". A pixel counts
    for a layer when it is near the layer's colour, or clearly nearer to it than to what the user
    was wearing there before (the change happened, in the right direction).
    """
    labels = parser.labels(result)
    lab = cv2.cvtColor(np.asarray(result.convert("RGB")), cv2.COLOR_RGB2LAB).astype(np.float32)
    ids = {"upper": [UPPER, SCARF], "lower": [PANTS, SKIRT], "dress": [DRESS, UPPER, PANTS, SKIRT]}
    before: dict[str, Optional[np.ndarray]] = {}
    if original is not None:
        o_labels = parser.labels(original)
        o_lab = cv2.cvtColor(np.asarray(original.convert("RGB")), cv2.COLOR_RGB2LAB).astype(np.float32)
        for part, part_ids in ids.items():
            m = _erode(np.isin(o_labels, part_ids))
            before[part] = np.median(o_lab[m], 0) if m.sum() > 200 else None
    w = np.array([0.35, 1, 1], np.float32)
    report = []
    for layer in layers:
        area = _erode(np.isin(labels, ids[layer.part]))
        if area.sum() < 200:
            report.append({"name": layer.name, "ok": False, "share": 0.0})
            continue
        px = lab[area]
        d = np.linalg.norm((px - np.asarray(layer.lab, np.float32)) * w, axis=1)
        hit = d < 20
        old = before.get(layer.part)
        if old is not None and _de(old, layer.lab) > 25:
            d_old = np.linalg.norm((px - old) * w, axis=1)
            hit |= d < 0.6 * d_old
        share = float(hit.mean())
        need = {"outer": 0.15, "top": 0.06, "under": 0.004, "bottom": 0.35, "dress": 0.3, "garment": 0.25}[layer.role]
        report.append({"name": layer.name, "ok": share >= need, "share": round(share, 3)})
    return report


def to_dict(o: Outfit) -> dict:
    return asdict(o)
