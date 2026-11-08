"""Optional dumps of every intermediate image, for tuning. Off unless TRYON_DEBUG_DIR is set."""

from __future__ import annotations

import os
import threading
from pathlib import Path

import numpy as np
from PIL import Image

_dir = os.environ.get("TRYON_DEBUG_DIR")
_local = threading.local()

if _dir:
    # This writes people's photos to disk. Tuning on a developer machine only; never in production.
    import logging

    logging.getLogger("tryon.debug").warning(
        "DEBUG DUMPS ARE ON: every photo and result is written to %s. Unset TRYON_DEBUG_DIR for real use.", _dir
    )


def begin(job_id: str) -> None:
    _local.prefix = job_id[:6]
    _local.n = 0


def save(name: str, img) -> None:
    if not _dir:
        return
    if isinstance(img, np.ndarray):
        arr = img.astype(np.uint8) * 255 if img.dtype == bool else img
        img = Image.fromarray(arr)
    _local.n = getattr(_local, "n", 0) + 1
    Path(_dir).mkdir(parents=True, exist_ok=True)
    img.save(Path(_dir) / f"{getattr(_local, 'prefix', 'x')}-{_local.n:02d}-{name}.png")


def forget(job_id: str) -> None:
    """Delete every dump of a job (used when its result was refused by the safety check)."""
    if not _dir:
        return
    for f in Path(_dir).glob(f"{job_id[:6]}-*.png"):
        try:
            f.unlink()
        except OSError:
            pass
