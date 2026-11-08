"""Image helpers: decode uploads safely, size them for the model, encode results."""

from __future__ import annotations

import io

from PIL import Image, ImageOps

# Any real photo is accepted: phones shoot up to 200 megapixels. Only absurd sizes (a
# decompression bomb, over 250 MP) are refused.
Image.MAX_IMAGE_PIXELS = 250_000_000

# iPhone photos are HEIC.
try:
    from pillow_heif import register_heif_opener

    register_heif_opener()
except ImportError:  # pragma: no cover
    pass

ALLOWED_FORMATS = {"JPEG", "PNG", "WEBP", "MPO", "HEIF", "AVIF", "BMP", "GIF", "TIFF"}
# Decoded straight to about this size: a 200 MP photo then costs the memory of a 4 MP one.
DECODE_SIDE = 2048


class ImageError(ValueError):
    pass


def decode_upload(data: bytes, max_bytes: int) -> Image.Image:
    if not data:
        raise ImageError("The file is empty.")
    if len(data) > max_bytes:
        raise ImageError(f"The file is larger than {max_bytes // (1024 * 1024)} MB.")
    try:
        img = Image.open(io.BytesIO(data))
        fmt = img.format
        if fmt in ("JPEG", "MPO"):
            img.draft("RGB", (DECODE_SIDE, DECODE_SIDE))  # JPEG decodes at 1/2, 1/4, 1/8 size for free
        img.load()
        if getattr(img, "n_frames", 1) > 1:
            img.seek(0)  # a GIF or multi-page TIFF: its first picture
    except Exception as exc:  # noqa: BLE001 - Pillow raises many types
        raise ImageError("The file is not a readable image.") from exc
    if fmt not in ALLOWED_FORMATS:
        raise ImageError("Use a photo (JPEG, PNG, HEIC or WebP).")
    # Phone photos carry rotation in EXIF; bake it in so the pose detector sees an upright person.
    img = ImageOps.exif_transpose(img)
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        background = Image.new("RGB", img.size, (255, 255, 255))
        background.paste(img.convert("RGBA"), mask=img.convert("RGBA").split()[-1])
        return background
    return img.convert("RGB")


def shrink(img: Image.Image, max_side: int) -> Image.Image:
    """Downscale so the longest side is at most max_side (never upscales)."""
    w, h = img.size
    if max(w, h) <= max_side:
        return img
    s = max_side / max(w, h)
    return img.resize((round(w * s), round(h * s)), Image.LANCZOS)


def fit_size(size: tuple[int, int], max_side: int, multiple: int = 16) -> tuple[int, int]:
    """Scale (w, h) so the longest side is at most max_side, both sides a multiple of `multiple`."""
    w, h = size
    scale = min(1.0, max_side / max(w, h))
    nw = max(multiple, int(round(w * scale / multiple)) * multiple)
    nh = max(multiple, int(round(h * scale / multiple)) * multiple)
    return nw, nh


def encode_png(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=False)
    return buf.getvalue()


def encode_jpeg(img: Image.Image, quality: int = 92) -> bytes:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=quality)
    return buf.getvalue()
