"""Run each engine once on a person + garment photo and save the results, without the web server.

    python server/scripts/smoke_test.py samples/person.jpg samples/garment.jpg [flux-klein]

Prints load time, run time and peak GPU memory for each engine; writes samples/out/<engine>.png.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "server"))

import torch  # noqa: E402

from app.engines.base import TryOnRequest  # noqa: E402
from app.engines.flux_klein import FluxKleinEngine  # noqa: E402
from app.images import decode_upload  # noqa: E402

ENGINES = {"flux-klein": FluxKleinEngine}


def main() -> None:
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    person = decode_upload(Path(sys.argv[1]).read_bytes(), 50 * 1024 * 1024)
    garment = decode_upload(Path(sys.argv[2]).read_bytes(), 50 * 1024 * 1024)
    names = sys.argv[3:] or list(ENGINES)
    out_dir = ROOT / "samples" / "out"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    for name in names:
        engine = ENGINES[name]()
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        engine.load()
        t1 = time.time()
        req = TryOnRequest(person=person, garment=garment, category="upper", view="front", seed=42)
        result = engine.run(req)
        t2 = time.time()
        # A second run shows the warm speed (what users get once the model is loaded).
        engine.run(req)
        t3 = time.time()
        path = out_dir / f"{name}.png"
        result.save(path)
        peak = torch.cuda.max_memory_allocated() / 1e9
        print(
            f"{name:12s} load {t1 - t0:6.1f}s  first run {t2 - t1:6.1f}s  warm run {t3 - t2:6.1f}s"
            f"  peak VRAM {peak:4.1f} GB  -> {path}"
        )
        engine.unload()


if __name__ == "__main__":
    main()
