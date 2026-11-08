"""One GPU, one worker thread, a bounded queue.

- Jobs run strictly one at a time (an 8 GB laptop GPU cannot hold two models at once).
- Only one engine is loaded at a time; switching engines unloads the previous one.
- Inputs and results live in memory only and are dropped after `result_ttl_s`.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
import traceback
import uuid
from dataclasses import dataclass, field
from typing import Optional

import torch

from . import debug, safety
from .config import settings
from .engines.base import Cancelled, Engine, TryOnRequest
from .images import encode_png
from .preprocess.body import skeleton
from .preprocess.color import match_garment_color, match_kept_clothes, match_layers_color, match_skin_tone
from .preprocess.layers import analyse_outfit, checklist_prompt, short_name, verify
from .preprocess.describe import describe
from .preprocess.garment import color_parts, crop_worn_garment, extract_garment
from .preprocess.parsing import garment_rgba, lock_head, lock_outside_garment

log = logging.getLogger("tryon.jobs")


@dataclass
class Job:
    id: str
    engine: str
    view: str
    request: Optional[TryOnRequest]
    status: str = "queued"  # queued | loading | running | done | error
    created: float = field(default_factory=time.time)
    started: Optional[float] = None
    finished: Optional[float] = None
    load_s: float = 0.0
    result_png: Optional[bytes] = None
    error: Optional[str] = None
    # Each piece of the outfit and whether it made it into the result ({name, ok, share}).
    checklist: list = field(default_factory=list)
    # Size verdict (size letters only), from the request.
    fit: Optional[dict] = None
    cancel: threading.Event = field(default_factory=threading.Event)

    def public(self, position: Optional[int]) -> dict:
        return {
            "id": self.id,
            "engine": self.engine,
            "view": self.view,
            "status": self.status,
            "position": position,
            "error": self.error,
            "checklist": [{"name": short_name(c["name"]), "ok": c["ok"]} for c in self.checklist],
            "fit": self.fit,
            "timings": {
                "queued_s": round((self.started or time.time()) - self.created, 2),
                "model_load_s": round(self.load_s, 2),
                "run_s": round((self.finished or time.time()) - self.started, 2)
                if self.started
                else None,
            },
        }


_outfits: dict[str, object] = {}


def _outfit_cached(img):
    """Live mode sends the same outfit with every frame: read it once."""
    import hashlib

    key = hashlib.sha1(img.convert("RGB").tobytes()).hexdigest()
    if key not in _outfits:
        if len(_outfits) > 64:
            _outfits.clear()
        _outfits[key] = analyse_outfit(img)
    return _outfits[key]


def _live_category(person, outfit) -> str:
    """What to change in a live frame: the parts of the outfit whose body part is in view."""
    from .preprocess.parsing import L_LEG, PANTS, R_LEG, SKIRT, parser
    import numpy as np

    labels = parser.labels(person)
    legs = float(np.isin(labels, [L_LEG, R_LEG, PANTS, SKIRT]).mean()) > 0.03
    parts = {layer.part for layer in outfit.layers}
    has_top = bool(parts & {"upper", "dress"})
    has_bottom = bool(parts & {"lower", "dress"})
    if legs and has_top and has_bottom:
        return "dress"
    if has_top or not legs:
        return "upper"
    return "lower"


class QueueFull(RuntimeError):
    pass


class JobManager:
    def __init__(self, engines: dict[str, Engine]) -> None:
        self.engines = engines
        self.jobs: dict[str, Job] = {}
        self._order: list[str] = []
        self._q: "queue.Queue[str]" = queue.Queue()
        self._lock = threading.Lock()
        self._active_engine: Optional[str] = None
        threading.Thread(target=self._worker, name="gpu-worker", daemon=True).start()
        threading.Thread(target=self._janitor, name="janitor", daemon=True).start()

    # ---------- public ----------
    def submit(self, engine: str, req: TryOnRequest) -> Job:
        with self._lock:
            waiting = sum(1 for j in self.jobs.values() if j.status in ("queued", "loading", "running"))
            if waiting >= settings.max_queue:
                raise QueueFull("The queue is full. Try again in a minute.")
            job = Job(id=uuid.uuid4().hex, engine=engine, view=req.view, request=req, fit=req.fit)
            self.jobs[job.id] = job
            self._order.append(job.id)
        self._q.put(job.id)
        return job

    def get(self, job_id: str) -> Optional[Job]:
        return self.jobs.get(job_id)

    def position(self, job: Job) -> Optional[int]:
        if job.status != "queued":
            return None
        with self._lock:
            ahead = [
                j for j in self._order
                if j in self.jobs and self.jobs[j].status in ("queued", "loading", "running")
            ]
        return ahead.index(job.id) if job.id in ahead else None

    def delete(self, job_id: str) -> None:
        with self._lock:
            job = self.jobs.pop(job_id, None)
            if job_id in self._order:
                self._order.remove(job_id)
        if job:
            # A running job stops at its next diffusion step instead of holding the GPU.
            job.cancel.set()
            job.request = None
            job.result_png = None

    @property
    def active_engine(self) -> Optional[str]:
        return self._active_engine

    # ---------- worker ----------
    def _switch_to(self, engine_id: str) -> float:
        if self._active_engine == engine_id and self.engines[engine_id].loaded:
            return 0.0
        t0 = time.time()
        for eid, eng in self.engines.items():
            if eid != engine_id and eng.loaded:
                log.info("unloading %s", eid)
                eng.unload()
        torch.cuda.empty_cache()
        log.info("loading %s", engine_id)
        self.engines[engine_id].load()
        self._active_engine = engine_id
        return time.time() - t0

    def _worker(self) -> None:
        while True:
            job_id = self._q.get()
            job = self.jobs.get(job_id)
            if job is None or job.request is None:
                continue
            job.started = time.time()
            try:
                job.status = "loading"
                job.load_s = self._switch_to(job.engine)
                job.status = "running"
                req = job.request
                if req is None:
                    continue  # deleted while the model was loading
                req.cancel = job.cancel
                debug.begin(job.id)
                debug.save(f"input-person-{req.category}", req.person)
                debug.save("input-outfit", req.garment)
                # The outfit as a checklist of layers (coat, top under it, T-shirt hem, trousers),
                # each with its colour, read from the original photo before any cut-out.
                outfit = _outfit_cached(req.garment)
                if req.category == "auto":
                    req.category = _live_category(req.person, outfit)
                    log.info("job %s: live frame, changing %s", job.id, req.category)
                if outfit.person:
                    req.layers = outfit.for_category(req.category)
                    # Shoes: the whole shoe changes when the outfit photo shows them, else none of it.
                    if outfit.shoes and req.category in ("lower", "dress"):
                        req.shoes = outfit.shoes
                    if req.category in ("upper", "dress"):
                        req.top_length = outfit.top_length
                        # The same hem on the user's body: measured on the model's skeleton, placed
                        # on the user's.
                        if outfit.top_hem is not None:
                            sk = skeleton(req.person)
                            y = sk.to_pixel(outfit.top_hem) if sk is not None else None
                            if y is not None:
                                req.top_hem_frac = max(0.0, min(1.0, y / sk.height))
                                log.info("job %s: top ends at body %.2f -> %.0f%% down the photo", job.id, outfit.top_hem, req.top_hem_frac * 100)
                    log.info("job %s: checklist %s (top length %s)", job.id, checklist_prompt(req.layers), req.top_length)
                # Colour of each garment piece, measured on the original photo before any cut-out.
                parts = color_parts(req.garment)
                # Plain-words description for prompt-driven engines, unless the user typed one.
                if not req.description and req.layers:
                    # From the checklist, so the short name agrees with it ("a long coat", not "a top").
                    names = [l.name.split(",")[0].split(" under ")[0] for l in req.layers if l.role != "under"]
                    req.description = ", ".join(names)
                    log.info("job %s: garment described as %r", job.id, req.description)
                if not req.description:
                    req.description = describe(req.garment, req.category)
                    log.info("job %s: garment described as %r", job.id, req.description)
                # Real pixels of the garment as worn, before any cut-out (Klein copies detail from it).
                req.garment_worn = crop_worn_garment(req.garment, req.category, settings.klein_ref_side)
                # A garment photo that shows a person wearing it gets cut down to the garment alone.
                outfit_photo = req.garment  # untouched, for the result's coverage check
                req.garment, extracted = extract_garment(req.garment, req.category)
                if req.garment_back is not None:
                    req.garment_back, _ = extract_garment(req.garment_back, req.category)
                if extracted:
                    log.info("job %s: garment cut out of a photo of a person", job.id)
                debug.save("garment", req.garment)
                engine = self.engines[job.engine]
                # Generate, then the three result checks (safety.check_result). A refused result
                # gets one more try with another seed (the photos already passed the quality and
                # safety checks, so the model is the likely cause); then it is refused for good.
                verdict = None
                # A too-tight or too-loose look refused twice gets one more try in the shopper's own
                # size, labelled as such (the safety checks always win over showing the bad fit).
                for attempt in range(1 if req.live else (3 if req.fit_off else 2)):
                    if attempt == 2:
                        req.fit_off = 0
                        if job.fit:
                            job.fit = {**job.fit, "drawn_in_own_size": True}
                    if attempt:
                        req.seed += 1009
                        log.info("job %s: result refused (%s), generating once more", job.id, verdict.code)
                    req.check_cancel()
                    out = engine.run(req)
                    debug.save("engine-out", out)
                    # Tick the checklist: is every piece there, in its colour? If one is missing (the coat
                    # left out, the trousers not changed), generate once more and keep the better one.
                    if req.layers:
                        check = verify(out, req.layers, req.person)
                        missing = [c["name"] for c, l in zip(check, req.layers) if not c["ok"] and l.role != "under"]
                        log.info("job %s: checklist %s", job.id, [(c["name"], c["ok"], c["share"]) for c in check])
                        if missing and not req.live:  # live: the next frame is the retry
                            log.info("job %s: missing %s, generating again", job.id, missing)
                            req.seed += 1
                            req.check_cancel()
                            out2 = engine.run(req)
                            check2 = verify(out2, req.layers, req.person)
                            debug.save("engine-out-retry", out2)
                            log.info("job %s: retry checklist %s", job.id, [(c["name"], c["ok"], c["share"]) for c in check2])
                            score = lambda ch: (sum(c["ok"] for c in ch), sum(c["share"] for c in ch))  # noqa: E731
                            if score(check2) > score(check):
                                out, check = out2, check2
                        job.checklist = check
                    # Size check, pass 2: the same look redrawn tight or loose (only when off).
                    if req.fit_off and hasattr(engine, "refit"):
                        req.check_cancel()
                        out = engine.refit(req, out)
                        debug.save("after-refit", out)
                    # Models shift the garment's colour toward the scene's light (black -> brown in a
                    # sunset photo). Pull it back to the garment photo's colour, keeping folds and shading:
                    # per layer when the outfit was read as layers, else per body part.
                    if req.layers:
                        out, color = match_layers_color(out, req.layers)
                        if color.get("applied"):
                            log.info("job %s: colour corrected per layer (dE before: %s)", job.id, color["layers"])
                    else:
                        out, color = match_garment_color(out, parts, req.category)
                        if color.get("applied"):
                            log.info("job %s: garment colour corrected per piece (dE before: %s)", job.id, color["parts"])
                    # Skin the model had to invent (chest, forearms) gets the person's real skin tone.
                    # Clothes the user keeps (their trousers under a new top) keep their own colour where
                    # the model had to redraw them.
                    out, kept = match_kept_clothes(out, req.person, req.category)
                    if kept.get("applied"):
                        log.info("job %s: redrawn kept clothes matched to the photo (was off by dE %s)", job.id, kept["delta_e"])
                    debug.save("after-colour", out)
                    out, skin = match_skin_tone(out, req.person)
                    if skin.get("applied"):
                        log.info("job %s: new skin matched to the person (was off by dE %s)", job.id, skin["delta_e"])
                    if req.lock_outside:
                        out = lock_outside_garment(req.person, out, req.category)
                    # Hard rule: the face and hair in the result are the user's own pixels, always.
                    out = lock_head(req.person, out)
                    verdict = safety.check_result(out, req.person, outfit_photo, req.layers, req.category)
                    if verdict.ok or verdict.code == "unavailable":
                        break
                if not verdict.ok:
                    debug.forget(job.id)
                    job.status = "error"
                    job.error = verdict.message
                    continue
                debug.save("final", out)
                if req.live:
                    # Checked above as a whole picture; only its clothes go back.
                    out = garment_rgba(out, req.category)
                job.result_png = encode_png(out)
                job.status = "done"
            except Cancelled:
                log.info("job %s: cancelled, GPU freed for the next look", job.id)
                job.status = "error"
                job.error = "Cancelled."
            except torch.cuda.OutOfMemoryError:
                log.exception("out of GPU memory")
                job.status = "error"
                job.error = "The GPU ran out of memory. Try a smaller image (TRYON_MAX_SIDE) or restart the server."
                self._recover()
            except Exception as exc:  # noqa: BLE001
                log.error("job %s failed\n%s", job.id, traceback.format_exc())
                job.status = "error"
                job.error = str(exc) or exc.__class__.__name__
            finally:
                job.finished = time.time()
                job.request = None  # drop the user's photos as soon as we are done with them
                # Give back cached blocks: on an 8 GB card a fragmented cache tips the next job into
                # Windows' shared-memory fallback (a 25 s job took over 3 minutes).
                torch.cuda.empty_cache()

    def _recover(self) -> None:
        for eng in self.engines.values():
            if eng.loaded:
                eng.unload()
        self._active_engine = None
        torch.cuda.empty_cache()

    def _janitor(self) -> None:
        while True:
            time.sleep(30)
            cutoff = time.time() - settings.result_ttl_s
            for job_id, job in list(self.jobs.items()):
                if job.finished and job.finished < cutoff:
                    self.delete(job_id)
