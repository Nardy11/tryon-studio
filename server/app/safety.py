"""Content safety: no nudity, no underwear or swimwear try-ons, no extra skin. Fails closed.

Three gates, all on the server, so calling the API directly cannot skip them:

1. Person photo (before anything runs): the person is dressed (no nudity, underwear or swimwear).
   Children are welcome (children's clothes); for them the result check is stricter.
2. Outfit photo, and the back photo (before anything runs): no nudity, and the garment is not
   lingerie, underwear, swimwear, see-through, or topless.
3. Result (before it is stored or returned), three independent checks that must all pass:
   a. the NSFW classifier;
   b. CLIP: does the result look like underwear, swimwear, see-through or nudity;
   c. coverage, by pixels: in each body region (chest, belly, thighs, shins) the result may show
      skin only where the outfit photo shows skin. A sleeveless dress may show arms; a jacket over
      a top may not show a bare chest. The rule is the outfit's own coverage, not "cover
      everything", and never more skin than the outfit has.

Two local models, on the CPU (no photo leaves the server, no GPU memory taken from the try-on):
- Falconsai/nsfw_image_detection (ViT, Apache 2.0): normal vs nsfw.
- openai/clip-vit-large-patch14 (CLIP, MIT): zero-shot "what kind of garment / how old".

If a model cannot be loaded or a check raises, the answer is "blocked": a try-on never runs on an
unchecked photo. The thresholds are deliberately strict; a false refusal costs a retry with
another photo, a false pass is not acceptable.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
from PIL import Image

log = logging.getLogger("tryon.safety")

NSFW_REPO = "Falconsai/nsfw_image_detection"
CLIP_REPO = "openai/clip-vit-large-patch14"

# Probability of "nsfw" above which a photo is refused. Normal fashion photos score < 0.01.
NSFW_INPUT = 0.30
NSFW_RESULT = 0.20  # stricter on what we would show

# What a garment / outfit photo may show. Each prompt is one class; the unsafe ones are summed.
_GARMENT_SAFE = [
    "a photo of a person wearing everyday clothes",
    "a photo of a person wearing a coat or a jacket",
    "a photo of a person wearing a dress",
    "a photo of a person wearing a suit",
    "a photo of a person wearing a t-shirt and jeans",
    "a photo of a person wearing a sweater and trousers",
    "a photo of a person wearing a skirt and a blouse",
    "a photo of a person wearing sportswear",
    "a photo of a person wearing a hijab and modest clothes",
    "a product photo of a shirt",
    "a product photo of trousers",
    "a product photo of a dress",
    "a product photo of a jacket",
]
_GARMENT_UNSAFE = [
    "a photo of a person in lingerie",
    "a photo of a person in underwear",
    "a photo of a person in a bra",
    "a photo of a person in a bikini",
    "a photo of a person in a swimsuit",
    "a photo of a person in swim trunks",
    "a photo of a person wearing see-through clothes",
    "a photo of a naked person",
    "a photo of a topless person",
    "a photo of a shirtless man",
    "a product photo of lingerie",
    "a product photo of underwear",
    "a product photo of a bikini",
]
# Summed probability of the unsafe classes. A fully dressed man in a beige shirt with an open
# collar scored 0.28 (skin-coloured fabric), so the line sits higher; nudity itself is the NSFW
# model's job and is not affected by this number.
GARMENT_UNSAFE = 0.50

_AGE = ["a photo of a child", "a photo of a teenager", "a photo of an adult"]
# Children may use try-on (children's clothes). A photo that looks like a child only makes the
# coverage check stricter; every revealing outfit is refused for everyone anyway.
CHILD = 0.35

# Coverage: how much more skin than the outfit photo a body region may show (share of the
# region's body pixels).
COVER_MARGIN = 0.15
COVER_MARGIN_CHILD = 0.10
# Regions on the body ruler (preprocess.body: 0 shoulders, 1 hips, 2 knees, 3 ankles).
REGIONS = {"chest": (0.0, 0.5), "belly": (0.5, 1.0), "thighs": (1.0, 2.0), "shins": (2.0, 2.9)}
CHANGED = {"upper": ("chest", "belly"), "lower": ("thighs", "shins"), "dress": tuple(REGIONS)}


@dataclass
class Verdict:
    ok: bool
    code: str = "ok"  # ok | nsfw | garment | minor | skin | unavailable
    message: str = ""  # shown to people
    scores: dict = field(default_factory=dict)

    def public(self) -> dict:
        return {"ok": self.ok, "code": self.code, "message": self.message}


_MESSAGES = {
    "nsfw": "This photo can't be used. Please upload a photo where you are dressed.",
    "garment": "This outfit can't be tried on here. Underwear, swimwear and see-through clothes aren't supported.",
    "skin": "We couldn't make this look properly.",
    "result": "We couldn't make this look properly.",
    "unavailable": "The safety check isn't available right now, so try-on is paused. Try again shortly.",
}


def _blocked(code: str, scores: dict) -> Verdict:
    msg_code = "result" if code == "nsfw-result" else code
    return Verdict(False, code.replace("-result", ""), _MESSAGES[msg_code], scores)


class _Models:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._loaded = False
        self.error: Optional[str] = None

    def load(self) -> None:
        with self._lock:
            if self._loaded:
                return
            try:
                from transformers import (
                    AutoImageProcessor,
                    AutoModelForImageClassification,
                    CLIPModel,
                    CLIPProcessor,
                )

                self.nsfw_proc = AutoImageProcessor.from_pretrained(NSFW_REPO)
                self.nsfw = AutoModelForImageClassification.from_pretrained(NSFW_REPO).eval()
                self.nsfw_idx = {v.lower(): int(k) for k, v in self.nsfw.config.id2label.items()}["nsfw"]
                self.clip_proc = CLIPProcessor.from_pretrained(CLIP_REPO)
                self.clip = CLIPModel.from_pretrained(CLIP_REPO).eval()
                with torch.no_grad():
                    self.garment_text = self._text(_GARMENT_SAFE + _GARMENT_UNSAFE)
                    self.age_text = self._text(_AGE)
                self._loaded = True
                self.error = None
                log.info("safety models loaded (CPU)")
            except Exception as exc:  # noqa: BLE001
                self.error = f"{exc.__class__.__name__}: {exc}"
                log.error("safety models failed to load: %s", self.error)

    @property
    def ready(self) -> bool:
        return self._loaded

    def _text(self, prompts: list[str]) -> torch.Tensor:
        t = self.clip_proc(text=prompts, return_tensors="pt", padding=True)
        e = self.clip.get_text_features(**t)
        e = e if isinstance(e, torch.Tensor) else e.pooler_output
        return e / e.norm(dim=-1, keepdim=True)

    @torch.no_grad()
    def nsfw_score(self, img: Image.Image) -> float:
        x = self.nsfw_proc(images=img.convert("RGB"), return_tensors="pt")
        p = self.nsfw(**x).logits.softmax(-1)[0]
        return float(p[self.nsfw_idx])

    @torch.no_grad()
    def clip_probs(self, img: Image.Image, text: torch.Tensor) -> np.ndarray:
        x = self.clip_proc(images=img.convert("RGB"), return_tensors="pt")
        e = self.clip.get_image_features(**x)
        e = e if isinstance(e, torch.Tensor) else e.pooler_output
        e = e / e.norm(dim=-1, keepdim=True)
        logits = self.clip.logit_scale.exp() * e @ text.T
        return logits.softmax(-1)[0].numpy()


models = _Models()


def warm_up() -> None:
    """Load in the background at start-up, so the first request does not wait."""
    threading.Thread(target=models.load, daemon=True).start()


def _ready() -> Optional[Verdict]:
    models.load()
    if not models.ready:
        return _blocked("unavailable", {"error": models.error})
    return None


def _views(img: Image.Image) -> list[Image.Image]:
    """The whole photo plus its upper and lower halves: a small exposed area in a full-length
    photo is a few percent of the frame and can slip under a whole-image classifier."""
    img = img.convert("RGB")
    w, h = img.size
    return [img, img.crop((0, 0, w, int(h * 0.6))), img.crop((0, int(h * 0.4), w, h))]


def _nsfw_max(img: Image.Image) -> float:
    return max(models.nsfw_score(v) for v in _views(img))


def _garment_unsafe(img: Image.Image) -> float:
    n = len(_GARMENT_SAFE)
    return max(float(models.clip_probs(v, models.garment_text)[n:].sum()) for v in _views(img))


_cache: dict[tuple[str, str], Verdict] = {}
_cache_lock = threading.Lock()


def _cached(kind: str, img: Image.Image, fn) -> Verdict:
    """Each photo is checked once (the page checks it on upload, the try-on reuses the answer).
    'unavailable' is never cached, so a recovered model is used at once."""
    import hashlib

    key = (kind, hashlib.sha1(img.convert("RGB").tobytes()).hexdigest())
    with _cache_lock:
        if key in _cache:
            return _cache[key]
    v = fn(img)
    if v.code != "unavailable":
        with _cache_lock:
            if len(_cache) > 512:
                _cache.clear()
            _cache[key] = v
    if not v.ok:
        log.warning("safety: %s photo refused (%s) %s", kind, v.code, v.scores)
    return v


def check_person(img: Image.Image) -> Verdict:
    return _cached("person", img, _check_person)


def check_outfit(img: Image.Image) -> Verdict:
    return _cached("outfit", img, _check_outfit)


def _check_person(img: Image.Image) -> Verdict:
    """Gate 1: the user's photo."""
    if (v := _ready()) is not None:
        return v
    try:
        nsfw = _nsfw_max(img)
        garment = _garment_unsafe(img)
        age = models.clip_probs(_person_crop(img), models.age_text)
        scores = {"nsfw": round(nsfw, 3), "underwear": round(garment, 3),
                  "child": round(float(age[0]), 3), "teen": round(float(age[1]), 3)}
        if nsfw > NSFW_INPUT:
            return _blocked("nsfw", scores)
        if garment > GARMENT_UNSAFE:
            # Already in underwear or swimwear: the model would paint around bare skin.
            return _blocked("nsfw", scores)
        scores["is_child"] = bool(age[0] > CHILD)
        return Verdict(True, scores=scores)
    except Exception as exc:  # noqa: BLE001
        log.exception("person check failed")
        return _blocked("unavailable", {"error": str(exc)})


def _check_outfit(img: Image.Image) -> Verdict:
    """Gate 2: an outfit photo (front or back)."""
    if (v := _ready()) is not None:
        return v
    try:
        nsfw = _nsfw_max(img)
        garment = _garment_unsafe(img)
        scores = {"nsfw": round(nsfw, 3), "underwear": round(garment, 3)}
        if nsfw > NSFW_INPUT:
            return _blocked("nsfw", scores)
        if garment > GARMENT_UNSAFE:
            return _blocked("garment", scores)
        return Verdict(True, scores=scores)
    except Exception as exc:  # noqa: BLE001
        log.exception("outfit check failed")
        return _blocked("unavailable", {"error": str(exc)})


def check_result(
    result: Image.Image,
    original: Image.Image,
    outfit: Optional[Image.Image] = None,
    layers: Optional[list] = None,
    category: str = "dress",
    length: Optional[dict] = None,
) -> Verdict:
    """Gate 3: what we would show. Three independent checks; all must pass.

    `length` (size check): trousers drawn too short on purpose show the ankles and lower shins, and
    only the shins may show that extra skin, in proportion to how short they were meant to be."""
    if (v := _ready()) is not None:
        return v
    try:
        child = bool(check_person(original).scores.get("is_child"))
        nsfw = _nsfw_max(result)
        garment = _garment_unsafe(result)
        over, cover = _extra_skin(result, original, outfit, layers or [], category, child, length)
        scores = {"nsfw": round(nsfw, 3), "underwear": round(garment, 3), "coverage": cover, "child": child}
        failed = None
        if nsfw > NSFW_RESULT or garment > GARMENT_UNSAFE:
            failed = "nsfw-result"
        elif over:
            failed = "skin"
        if failed:
            log.warning("safety: result refused (%s) %s", failed, scores)
            return _blocked(failed, scores)
        return Verdict(True, scores=scores)
    except Exception as exc:  # noqa: BLE001
        log.exception("result check failed")
        return _blocked("unavailable", {"error": str(exc)})


def _person_crop(img: Image.Image) -> Image.Image:
    """Head and shoulders, where age shows; the whole photo when no face is found."""
    from .preprocess.parsing import FACE, HAIR, parser

    labels = parser.labels(img)
    ys, xs = np.nonzero(np.isin(labels, [FACE, HAIR]))
    if len(xs) < 200:
        return img
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    s = max(x1 - x0, y1 - y0)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2 + s * 0.4
    half = s * 1.1
    w, h = img.size
    return img.crop((int(max(0, cx - half)), int(max(0, cy - half)), int(min(w, cx + half)), int(min(h, cy + half))))


def _skin_by_region(img: Image.Image, layer_labs: list) -> Optional[dict]:
    """Share of the body that is bare skin, per region of the body ruler. Skin colour comes from
    the person's own face; pixels the colour of a garment piece never count as skin (a beige
    coat is not bare skin). None when no body is found."""
    import cv2

    from .preprocess.body import skeleton
    from .preprocess.parsing import BAG, BG, FACE, HAIR, HAT, L_ARM, L_SHOE, R_ARM, R_SHOE, SUNGLASSES, parser

    img = img.convert("RGB")
    sk = skeleton(img)
    labels = parser.labels(img)
    face = labels == FACE
    if sk is None or face.sum() < 150:
        return None
    lab = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2LAB).astype(np.float32)
    w8 = np.array([0.3, 1.0, 1.0], np.float32)
    # The face above the shoulders gives the skin colour; the neck below them counts as body.
    shoulder = int(sk.ys["shoulder"])
    face_px = face.copy()
    face_px[shoulder:, :] = False
    ref = np.median(lab[face_px if face_px.sum() > 100 else face], 0)
    skin = np.linalg.norm((lab - ref) * w8, axis=2) < 14
    for g in layer_labs:
        skin &= np.linalg.norm((lab - np.asarray(g, np.float32)) * w8, axis=2) >= 14
    # Arms and hands are left out: a hand holding a phone at the chest is not a bare chest, and
    # bare arms are not what this check is for.
    body = ~np.isin(labels, [BG, HAIR, HAT, SUNGLASSES, BAG, L_SHOE, R_SHOE, L_ARM, R_ARM])
    out = {}
    h = labels.shape[0]
    for name, (a, b) in REGIONS.items():
        ya, yb = sk.to_pixel(a), sk.to_pixel(b)
        if ya is None or yb is None:
            continue
        ya, yb = int(max(0, ya)), int(min(h, yb))
        if yb - ya < 10:
            continue
        m = body[ya:yb]
        if m.sum() < 300:
            continue  # the region is out of the photo (cropped at the knees, ...)
        out[name] = float(skin[ya:yb][m].mean())
    return out


def _extra_skin(
    result: Image.Image,
    original: Image.Image,
    outfit: Optional[Image.Image],
    layers: list,
    category: str,
    child: bool,
    length: Optional[dict] = None,
) -> tuple[bool, dict]:
    """Does any region of the result show more skin than it should?

    Regions the try-on changes are held to the outfit photo (when it shows a person and that
    region); everything else, and outfits without a person, to the user's own photo.
    """
    labs = [layer.lab for layer in layers if getattr(layer, "lab", None) is not None]
    now = _skin_by_region(result, labs) or {}
    before = _skin_by_region(original, labs) or {}
    model = (_skin_by_region(outfit, labs) or {}) if outfit is not None else {}
    margin = COVER_MARGIN_CHILD if child else COVER_MARGIN
    # Below the hem of the outfit's skirt, dress or trousers, legs may be bare: that is what the
    # garment shows (tights and boots on the model are not part of the try-on).
    hems = [layer.hem for layer in layers if getattr(layer, "role", "") in ("bottom", "dress") and getattr(layer, "hem", None)]
    hem = max(hems) if hems else None
    report, over = {}, False
    for region, share in now.items():
        changed = region in CHANGED.get(category, ())
        allowed_from = model.get(region) if changed else None
        if allowed_from is None:
            allowed_from = before.get(region, 0.0)
        if changed and hem is not None:
            a, b = REGIONS[region]
            below = min(1.0, max(0.0, (b - max(a, hem)) / (b - a)))
            allowed_from = max(allowed_from, below)
        limit = allowed_from + margin
        # Trousers too short for the shopper's height (size check): the ankles and lower shins show
        # by design. Shins only; chest, belly and thighs keep the strict rule.
        legs_short = -min(0, int((length or {}).get("legs", 0)))
        if region == "shins" and legs_short:
            limit += 0.3 * legs_short  # matches how far the hem is cut (0.15 + 0.12 per step of the shin)
        report[region] = {"now": round(share, 3), "allowed": round(limit, 3)}
        if share > limit:
            over = True
    return over, report
