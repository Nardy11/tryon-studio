"""Where to crop the photo before the model sees it (more pixels for the clothes)."""

from __future__ import annotations

from PIL import Image


def _focus_box(
    mask: Image.Image, size: tuple[int, int], ratio: float, context: float = 0.12
) -> tuple[int, int, int, int]:
    """Bounding box of the mask, grown by `context` on each side and toward the w/h ratio."""
    import numpy as np

    w, h = size
    ys, xs = np.nonzero(np.asarray(mask) > 127)
    if len(xs) == 0:
        return (0, 0, w, h)
    x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    bw, bh = x1 - x0, y1 - y0
    # Some context so the model sees the neck, hands and surroundings it must blend into.
    x0, x1 = x0 - bw * context, x1 + bw * context
    y0, y1 = y0 - bh * context, y1 + bh * context
    bw, bh = x1 - x0, y1 - y0
    # Grow the short side toward the model's aspect ratio (padding covers whatever is left).
    if bw / bh < ratio:
        extra = bh * ratio - bw
        x0, x1 = x0 - extra / 2, x1 + extra / 2
    else:
        extra = bw / ratio - bh
        y0, y1 = y0 - extra / 2, y1 + extra / 2
    x0, y0 = max(0, int(x0)), max(0, int(y0))
    x1, y1 = min(w, int(round(x1))), min(h, int(round(y1)))
    return (x0, y0, x1, y1)


def _pad_to_ratio(img: Image.Image, ratio: float) -> tuple[Image.Image, tuple[int, int, int, int]]:
    """Pad with edge colour to the target w/h ratio; return the image and the box of the original."""
    w, h = img.size
    if abs(w / h - ratio) < 1e-3:
        return img, (0, 0, w, h)
    if w / h > ratio:  # too wide -> pad top/bottom
        nh = int(round(w / ratio))
        canvas = Image.new("RGB", (w, nh), img.getpixel((0, 0)))
        top = (nh - h) // 2
        canvas.paste(img, (0, top))
        return canvas, (0, top, w, top + h)
    nw = int(round(h * ratio))  # too tall -> pad left/right
    canvas = Image.new("RGB", (nw, h), img.getpixel((0, 0)))
    left = (nw - w) // 2
    canvas.paste(img, (left, 0))
    return canvas, (left, 0, left + w, h)
