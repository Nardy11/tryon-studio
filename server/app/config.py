"""Runtime settings, read once from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    # Which engine a request uses when it does not name one.
    default_engine: str = os.environ.get("TRYON_DEFAULT_ENGINE", "flux-klein")
    # Longest side of the image handed to the model. Lower is faster and lighter on VRAM.
    max_side: int = _int("TRYON_MAX_SIDE", 1024)
    # Uploaded photos are downscaled to this longest side on arrival (results come back this size).
    max_input_side: int = _int("TRYON_MAX_INPUT_SIDE", 1536)
    # Largest upload accepted, in megabytes.
    max_upload_mb: int = _int("TRYON_MAX_UPLOAD_MB", 60)
    # Results live in memory only: deleted when downloaded, or after this many seconds if never collected.
    result_ttl_s: int = _int("TRYON_RESULT_TTL_S", 120)
    # Looks one visitor may start per 10 minutes (the GPU is shared once the site is public).
    looks_per_10min: int = _int("TRYON_LOOKS_PER_10MIN", 12)
    # Most jobs that may wait in the queue before new ones are refused.
    max_queue: int = _int("TRYON_MAX_QUEUE", 20)
    # Klein: quantize transformer + text encoder to 4-bit so it fits an 8 GB GPU.
    klein_4bit: bool = os.environ.get("TRYON_KLEIN_4BIT", "1") != "0"
    # Longest side of the garment reference handed to Klein. Bigger copies more detail but costs
    # time and VRAM (an 8 GB laptop GPU is full at 640: Windows spills into system RAM and a run takes minutes; a 24 GB server can use 1024).
    klein_ref_side: int = _int("TRYON_KLEIN_REF_SIDE", 512)
    klein_repo: str = os.environ.get("TRYON_KLEIN_REPO", "black-forest-labs/FLUX.2-klein-4B")
    web_dist: Path = field(default_factory=lambda: ROOT / "web" / "dist")
    cors_origins: tuple[str, ...] = tuple(
        o.strip()
        for o in os.environ.get(
            "TRYON_CORS_ORIGINS",
            "http://localhost:5175,http://127.0.0.1:5175,https://localhost:5175,https://nardy11.github.io",
        ).split(",")
        if o.strip()
    )


settings = Settings()
