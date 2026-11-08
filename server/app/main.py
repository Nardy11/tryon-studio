"""HTTP API for photo-mode virtual try-on.

POST /api/tryon          multipart: person, garment, [garment_back], engine, category, view, ...
GET  /api/jobs/{id}      status + queue position + timings
GET  /api/jobs/{id}/result   PNG
DELETE /api/jobs/{id}    forget the job now (privacy)
GET  /api/engines        engines, licences, which one is loaded
GET  /api/health         GPU info
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

from .config import settings
from .engines.base import TryOnRequest
from .engines.flux_klein import FluxKleinEngine
from .images import ImageError, decode_upload, shrink
from .jobs import JobManager, QueueFull
from . import quality, safety

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

ENGINES = {
    "flux-klein": FluxKleinEngine(),
}
jobs = JobManager(ENGINES)
safety.warm_up()

app = FastAPI(title="Try-On Studio API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=list(settings.cors_origins),
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["*"],
)

CATEGORIES = {"upper", "lower", "dress"}
VIEWS = {"front", "side", "back"}


@app.get("/api/health")
def health() -> dict:
    gpu = None
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        gpu = {
            "name": torch.cuda.get_device_name(0),
            "vram_total_gb": round(total / 1e9, 1),
            "vram_free_gb": round(free / 1e9, 1),
        }
    return {"ok": True, "gpu": gpu, "active_engine": jobs.active_engine}


@app.get("/api/engines")
def engines() -> dict:
    return {
        "default": settings.default_engine,
        "engines": [
            {
                **e.info.__dict__,
                "loaded": e.loaded,
                "available": not e.missing_files(),
            }
            for e in ENGINES.values()
        ],
    }


async def _read(upload: Optional[UploadFile], name: str, required: bool):
    if upload is None:
        if required:
            raise HTTPException(422, f"Add the {name} photo.")
        return None
    data = await upload.read()
    try:
        img = decode_upload(data, settings.max_upload_mb * 1024 * 1024)
    except ImageError as exc:
        raise HTTPException(422, f"{name.capitalize()} photo: {exc}") from exc
    # Phone photos are ~12 MP. The models work at ~1 MP, and full-size images flood an 8 GB GPU,
    # so shrink on arrival. The result comes back at this size.
    return shrink(img, settings.max_input_side)


@app.post("/api/check-photo")
async def check_photo(person: UploadFile = File(...)) -> dict:
    """Look at a person photo before try-on and say what will limit the result.

    Mask-based try-on keeps the pose and fills the outline of the clothes in the photo, so hidden
    hands (in coat pockets) and missing legs show up directly in the result.
    """
    import asyncio

    import numpy as np

    from .preprocess.parsing import FACE, L_ARM, L_LEG, PANTS, R_ARM, R_LEG, SKIRT, parser

    img = await _read(person, "person", True)
    verdict = await asyncio.to_thread(safety.check_person, img)
    if not verdict.ok:
        return {"blocked": verdict.public(), "retake": None, "warnings": [{"code": verdict.code, "text": verdict.message}],
                "visible": {"arms": False, "legs": False}}
    retake = await asyncio.to_thread(quality.check_person, img)
    if retake is not None:
        return {"blocked": None, "retake": retake.public(), "warnings": [{"code": retake.code, "text": retake.message}],
                "visible": {"arms": False, "legs": False}}
    labels = await asyncio.to_thread(parser.labels, img)
    total = labels.size
    arms = float(np.isin(labels, [L_ARM, R_ARM]).sum()) / total
    legs = float(np.isin(labels, [L_LEG, R_LEG, PANTS, SKIRT]).sum()) / total
    face = float((labels == FACE).sum()) / total
    warnings = []
    if face < 0.002:
        warnings.append({"code": "no_person", "text": "No person found. Use a photo where your face and body are clearly visible."})
    if arms < 0.002:
        warnings.append({
            "code": "hands_hidden",
            "text": "Your hands and arms aren't visible (in pockets or covered). The new garment will copy that pose. For the best fit, let your arms hang relaxed with hands visible.",
        })
    if legs < 0.03:
        warnings.append({
            "code": "no_legs",
            "text": "Your legs aren't in the photo, so Bottoms and Full outfit can't be placed. Use Top, or a full-length photo.",
        })
    return {"blocked": None, "retake": None, "warnings": warnings, "visible": {"arms": arms >= 0.002, "legs": legs >= 0.03}}


@app.post("/api/analyse-outfit")
async def analyse_outfit_endpoint(garment: UploadFile = File(...)) -> dict:
    """Read an outfit photo as a checklist of pieces (coat, top under it, T-shirt hem, trousers)
    and suggest which part to change: the whole outfit when the photo shows top and bottoms."""
    import asyncio

    from .preprocess.layers import analyse_outfit

    img = await _read(garment, "garment", True)
    verdict = await asyncio.to_thread(safety.check_outfit, img)
    if not verdict.ok:
        return {"blocked": verdict.public(), "retake": None, "person": False, "suggested": "upper", "layers": [], "shoes": None}
    retake = await asyncio.to_thread(quality.check_outfit, img)
    if retake is not None:
        return {"blocked": None, "retake": retake.public(), "person": False, "suggested": "upper", "layers": [], "shoes": None}
    outfit = await asyncio.to_thread(analyse_outfit, img)
    if not outfit.layers:
        r = quality.Retake("no_garment", "We can't make out the clothes in this photo. Please upload another photo of the outfit.")
        return {"blocked": None, "retake": r.public(), "person": False, "suggested": "upper", "layers": [], "shoes": None}
    return {**outfit.public(), "blocked": None, "retake": None}


_recent: dict[str, list[float]] = {}


def _visitor(request: Request) -> str:
    # Behind the Cloudflare tunnel the real address is in this header; locally, the socket's.
    return request.headers.get("cf-connecting-ip") or (request.client.host if request.client else "?")


def _rate_limit(request: Request) -> None:
    """A fair share of the one GPU per visitor (the site is public)."""
    import time

    now = time.time()
    who = _visitor(request)
    times = [t for t in _recent.get(who, []) if now - t < 600]
    if len(times) >= settings.looks_per_10min:
        raise HTTPException(429, "You've made a lot of looks in a short time. Please wait a few minutes.")
    times.append(now)
    _recent[who] = times
    if len(_recent) > 5000:
        _recent.clear()


@app.post("/api/tryon", status_code=202)
async def tryon(
    request: Request,
    person: UploadFile = File(...),
    garment: UploadFile = File(...),
    garment_back: Optional[UploadFile] = File(None),
    engine: str = Form(None),
    category: str = Form("upper"),
    view: str = Form("front"),
    seed: int = Form(42),
    lock_outside: bool = Form(False),
    live: bool = Form(False),
) -> dict:
    # No free text reaches the model: the prompt is built only from what the server reads in the
    # photos (fixed colour and garment words). A typed "description" was a way to steer the
    # editing model ("... and take off ..."), so the field is gone and anything sent is ignored.
    # Steps are fixed per engine for the same reason: fewer knobs, fewer ways to get odd output.
    _rate_limit(request)
    engine = engine or settings.default_engine
    if engine not in ENGINES:
        raise HTTPException(422, f"Unknown engine '{engine}'. Use one of: {', '.join(ENGINES)}.")
    if category not in CATEGORIES:
        raise HTTPException(422, "Category must be upper, lower or dress.")
    if view not in VIEWS:
        raise HTTPException(422, "View must be front, side or back.")
    missing = ENGINES[engine].missing_files()
    if missing:
        raise HTTPException(
            503,
            f"{ENGINES[engine].info.label} isn't ready: its model files are still downloading "
            f"(run server/scripts/download_models.py).",
        )
    import asyncio

    person_img = await _read(person, "person", True)
    garment_img = await _read(garment, "garment", True)
    back_img = await _read(garment_back, "garment back", False)
    # Safety gates 1 and 2, before anything is queued, stored or sent to a model.
    checks = [("person", person_img, safety.check_person), ("outfit", garment_img, safety.check_outfit)]
    if back_img is not None:
        checks.append(("outfit back", back_img, safety.check_outfit))
    for _name, img, check in checks:
        verdict = await asyncio.to_thread(check, img)
        if not verdict.ok:
            raise HTTPException(503 if verdict.code == "unavailable" else 422, verdict.message)
    # Quality: a photo that would give a poor result is turned away here, before the GPU is used
    # (a retake costs the customer a moment; a failed run costs a minute and shows an error).
    retake = await asyncio.to_thread(quality.check_person, person_img)
    if retake is None:
        retake = await asyncio.to_thread(quality.check_outfit, garment_img)
    if retake is None and back_img is not None:
        retake = await asyncio.to_thread(quality.check_outfit, back_img)
    if retake is None and category in ("lower", "dress"):
        from .preprocess.parsing import parser

        labels = await asyncio.to_thread(parser.labels, person_img)
        if quality.needs_legs(labels):
            retake = quality.Retake(
                "no_legs", "Your legs aren't in this photo, so this outfit can't be placed. Please upload a full-length photo."
            )
    if retake is not None:
        raise HTTPException(422, retake.message)

    req = TryOnRequest(
        person=person_img,
        garment=garment_img,
        garment_back=back_img,
        category=category,  # type: ignore[arg-type]
        view=view,  # type: ignore[arg-type]
        seed=seed,
        lock_outside=lock_outside,
        live=live,
    )
    try:
        job = jobs.submit(engine, req)
    except QueueFull as exc:
        raise HTTPException(429, str(exc)) from exc
    return job.public(jobs.position(job))


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str) -> dict:
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "This result has expired or never existed.")
    return job.public(jobs.position(job))


@app.get("/api/jobs/{job_id}/result")
def job_result(job_id: str) -> Response:
    job = jobs.get(job_id)
    if job is None or job.result_png is None:
        raise HTTPException(404, "No result for this job yet.")
    png = job.result_png
    # Handed out once: the result leaves the server's memory as soon as it is downloaded.
    jobs.delete(job_id)
    return Response(
        content=png,
        media_type="image/png",
        headers={"Cache-Control": "no-store"},
    )


@app.delete("/api/jobs/{job_id}", status_code=204)
def job_delete(job_id: str) -> Response:
    jobs.delete(job_id)
    return Response(status_code=204)


# Serve the built website from the same origin when it exists (production-style run).
if settings.web_dist.is_dir():
    app.mount("/assets", StaticFiles(directory=settings.web_dist / "assets"), name="assets")

    @app.get("/{path:path}", include_in_schema=False)
    def spa(path: str) -> FileResponse:
        root = settings.web_dist.resolve()
        target = (root / path).resolve()
        if path and target.is_relative_to(root) and target.is_file():
            return FileResponse(target)
        return FileResponse(settings.web_dist / "index.html")
