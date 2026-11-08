"""Run person x garment x category cases through the running API and score colour accuracy per piece.

    python server/scripts/eval_categories.py

For each case: submits the job, saves the result to samples/out/eval-<name>.png, then measures the
average colour of each garment piece in the garment photo vs. in the result (Lab distance, dE:
< 5 hard to notice, 5-10 noticeable, > 15 clearly a different colour). Writes a contact sheet to
samples/out/eval-sheet.jpg.
"""

from __future__ import annotations

import json
import sys
import time
import urllib.request
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "server"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image, ImageDraw, ImageOps  # noqa: E402

from app.images import shrink  # noqa: E402
from app.preprocess.color import GARMENT_LABELS, RESULT_PART_LABELS  # noqa: E402
from app.preprocess.garment import color_parts  # noqa: E402
from app.preprocess.parsing import parser  # noqa: E402

API = "http://127.0.0.1:8000"
DL = Path.home() / "Downloads"
ME = DL / "WhatsApp Image 2025-09-16 at 4.04.23 AM.jpeg"
CASES = [
    ("me-outfit2-top", ME, DL / "outfit2.jpg", "upper"),
    ("me-outfit2-dress", ME, DL / "outfit2.jpg", "dress"),
    ("woman-outfit1-bottoms", ROOT / "samples/person-woman.jpg", DL / "outfit1.jpg", "lower"),
    ("woman-outfit1-dress", ROOT / "samples/person-woman.jpg", DL / "outfit1.jpg", "dress"),
]
ENGINE = sys.argv[1] if len(sys.argv) > 1 else "flux-klein"


def post(person: Path, garment: Path, category: str) -> str:
    boundary = uuid.uuid4().hex
    body = b""
    for name, path in (("person", person), ("garment", garment)):
        body += (f"--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"; filename=\"{path.name}\"\r\n"
                 f"Content-Type: image/jpeg\r\n\r\n").encode() + path.read_bytes() + b"\r\n"
    for k, v in (("engine", ENGINE), ("category", category)):
        body += f"--{boundary}\r\nContent-Disposition: form-data; name=\"{k}\"\r\n\r\n{v}\r\n".encode()
    body += f"--{boundary}--\r\n".encode()
    req = urllib.request.Request(f"{API}/api/tryon", data=body,
                                 headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    return json.load(urllib.request.urlopen(req))["id"]


def wait(job_id: str) -> dict:
    while True:
        job = json.load(urllib.request.urlopen(f"{API}/api/jobs/{job_id}"))
        if job["status"] in ("done", "error"):
            return job
        time.sleep(2)


def lab(img: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.asarray(img.convert("RGB")), cv2.COLOR_RGB2LAB).astype(np.float32)


def score(result: Image.Image, garment: Image.Image, category: str) -> dict:
    parts = color_parts(garment)
    labels = parser.labels(result)
    L = lab(result)
    wanted = {"upper": ["upper"], "lower": ["lower"], "dress": ["upper", "lower", "dress"]}[category]
    scores = {}
    for name, ref in parts.items():
        if name != "whole" and name not in wanted:
            continue
        ids = GARMENT_LABELS[category] if name == "whole" else RESULT_PART_LABELS[name]
        area = np.isin(labels, ids)
        if area.sum() < 300:
            scores[name] = "not in result"
            continue
        scores[name] = round(float(np.linalg.norm(ref.mean(0) - L[area].mean(0))), 1)
    return scores


def main() -> None:
    tiles = []
    for name, person, garment, category in CASES:
        t0 = time.time()
        job = wait(post(person, garment, category))
        if job["status"] != "done":
            print(f"{name:24s} ERROR {job['error']}")
            continue
        data = urllib.request.urlopen(f"{API}/api/jobs/{job['id']}/result").read()
        urllib.request.urlopen(urllib.request.Request(f"{API}/api/jobs/{job['id']}", method="DELETE"))
        out = ROOT / "samples/out" / f"eval-{name}.png"
        out.write_bytes(data)
        result = Image.open(out).convert("RGB")
        g = ImageOps.exif_transpose(Image.open(garment)).convert("RGB")
        s = score(result, g, category)
        print(f"{name:24s} {time.time() - t0:5.0f}s  colour error per piece (dE): {s}", flush=True)
        p = shrink(ImageOps.exif_transpose(Image.open(person)).convert("RGB"), 1536)
        h = 360
        row = [i.resize((int(i.width * h / i.height), h)) for i in (p, g, result)]
        tile = Image.new("RGB", (sum(i.width for i in row) + 40, h + 28), "white")
        x = 0
        for i in row:
            tile.paste(i, (x, 28))
            x += i.width + 20
        ImageDraw.Draw(tile).text((4, 6), f"{name}   dE {s}", fill="black")
        tiles.append(tile)
    if tiles:
        W = max(t.width for t in tiles)
        sheet = Image.new("RGB", (W, sum(t.height + 12 for t in tiles)), "white")
        y = 0
        for t in tiles:
            sheet.paste(t, (0, y))
            y += t.height + 12
        sheet.save(ROOT / "samples/out/eval-sheet.jpg", quality=86)
        print("sheet: samples/out/eval-sheet.jpg")


if __name__ == "__main__":
    main()
