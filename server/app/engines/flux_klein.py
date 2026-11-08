"""FLUX.2 [klein] 4B engine — the licence-safe default (Apache 2.0).

Klein is a general image *editing* model with multi-reference input, not a dedicated try-on
model. We give it two (or three) images and an instruction:

    image 1 = the person photo (its size sets the output size)
    image 2 = the garment product photo
    image 3 = the garment's back (only for back views, when the store has one)

On an 8 GB laptop GPU the bf16 weights (~16 GB) do not fit, so by default the transformer and
the Qwen3 text encoder are loaded in 4-bit NF4 (bitsandbytes), about 5 GB in total.
A server GPU (24 GB+) should set TRYON_KLEIN_4BIT=0 for full quality and speed.
"""

from __future__ import annotations

import gc
import logging

import torch
from PIL import Image, ImageFilter

from ..config import settings
from ..images import fit_size
from .base import Engine, EngineInfo, TryOnRequest

log = logging.getLogger("tryon.klein")

REGION = {
    "upper": "top (shirt, sweater or jacket)",
    "lower": "bottoms (trousers, jeans, shorts or skirt)",
    "dress": "outfit",
}

VIEW_HINT = {
    "front": "",
    "side": " The person is photographed from the side, so show the garment from the side as it would look worn.",
    "back": " The person is photographed from behind, so show the back of the garment as it would look worn.",
}


# Said in every prompt. The checks in app/safety.py are what enforce it; this keeps the model
# from drifting there in the first place. The outfit decides coverage: a sleeveless dress shows
# arms, a knee-length skirt shows the lower legs, and nothing more than that.
MODEST = (
    " The new clothes cover the body exactly as they cover the model in image 2: skin shows only"
    " where the outfit in image 2 shows skin, never anywhere else, with no transparency, no gaps and"
    " nothing suggestive. Do not add clothing that is not in image 2. The result is an ordinary"
    " fashion photo."
)


def build_prompt(req: TryOnRequest, has_back: bool, worn: bool = False, keep: str = "") -> str:
    import re

    # Only words the server wrote itself reach here (layers.py, describe.py); the character filter
    # is a second line in case a future change lets other text in.
    what = re.sub(r"[^a-z ,-]", "", (req.description or "garment").lower())[:120] or "garment"
    region = REGION[req.category]
    back = ""
    if req.view == "back":
        back = (
            " Use image 3 for the exact look of the garment's back."
            if has_back
            else " The back of the garment is not shown; keep it plain and consistent with the front's colour, fabric and fit, without the front's prints or logos."
        )
    source = (
        f" Image 2 shows the {what} worn by a model: copy only the clothes, never the model's body, skin, pose or accessories."
        if worn
        else ""
    )
    pieces = ""
    if req.layers:
        from ..preprocess.layers import checklist_prompt

        pieces = (
            f" The new {region} is made of these pieces, and every one of them must be on the person,"
            f" each in exactly its colour from image 2, none left out and none merged into another:"
            f" {checklist_prompt(req.layers)}."
        )
    return (
        f"Virtual try-on. Replace the {region} the person in image 1 is wearing with the {what} shown in image 2.{source}{pieces}"
        f" Match the garment in image 2 exactly: colour, fabric, knit or weave texture, ribbing, cuffs, hem, pattern, print,"
        f" logo placement, collar, sleeves, length, and any layer showing under it (for example a T-shirt hem)."
        f" It fits the person's body and pose naturally, with realistic folds, seams and shadows that match the lighting of image 1."
        f" Nothing of the old {region} remains: no under-layer, collar, print or logo from it, unless it is part of the new garment."
        f" Keep everything else from image 1 exactly the same: the face, identity, hair, skin, hands, body shape, pose,"
        f" the person's own watch, bracelets, rings and anything they hold (never the model's jewellery or bag),"
        f" the other clothes, the background and the camera framing.{VIEW_HINT[req.view]}{back}"
        + (
            f" The person's own {keep} stay exactly as they are, also where the old garment used to cover"
            f" them; never take the {keep.split()[-1]} from image 2."
            if keep
            else ""
        )
        + MODEST
    )


class FluxKleinEngine(Engine):
    info = EngineInfo(
        id="flux-klein",
        label="FLUX.2 Klein 4B",
        licence="Apache 2.0",
        commercial_use=True,
        note="General image-editing model, told to dress you in the garment. The one we can ship.",
    )

    required_files = tuple(
        (settings.klein_repo, f)
        for f in (
            "transformer/diffusion_pytorch_model.safetensors",
            "text_encoder/model-00001-of-00002.safetensors",
            "text_encoder/model-00002-of-00002.safetensors",
            "vae/diffusion_pytorch_model.safetensors",
        )
    )

    def __init__(self) -> None:
        self._pipe = None

    @property
    def loaded(self) -> bool:
        return self._pipe is not None

    def load(self) -> None:
        if self._pipe is not None:
            return
        from diffusers import Flux2KleinPipeline

        kwargs: dict = {"torch_dtype": torch.bfloat16}
        if settings.klein_4bit:
            from diffusers.quantizers import PipelineQuantizationConfig

            kwargs["quantization_config"] = PipelineQuantizationConfig(
                quant_backend="bitsandbytes_4bit",
                quant_kwargs={
                    "load_in_4bit": True,
                    "bnb_4bit_quant_type": "nf4",
                    "bnb_4bit_compute_dtype": torch.bfloat16,
                },
                components_to_quantize=["transformer", "text_encoder"],
            )
        # Measure BEFORE loading: bitsandbytes puts the 4-bit weights on the GPU while loading, so
        # afterwards the card looks full and the check wrongly picked the slow offload mode.
        torch.cuda.empty_cache()
        free, _ = torch.cuda.mem_get_info()
        pipe = Flux2KleinPipeline.from_pretrained(settings.klein_repo, **kwargs)
        # 4-bit weights are ~5 GB; ~1 GB more covers the activations at 1024 px. Below that, move
        # each part to the GPU only while it runs: it works, but ~3x slower (~70 s instead of ~25 s).
        # Browsers, Discord and the desktop share this GPU on a laptop and take 1-2 GB.
        on_gpu = settings.klein_4bit and free > 6.0e9
        if on_gpu:
            pipe.to("cuda")
        else:
            pipe.enable_model_cpu_offload()
        pipe.set_progress_bar_config(disable=True)
        self._pipe = pipe
        log.info(
            "FLUX.2 Klein loaded (4-bit=%s, %s, %.1f GB VRAM was free)",
            settings.klein_4bit, "all on GPU" if on_gpu else "CPU offload: slow", free / 1e9,
        )

    def unload(self) -> None:
        self._pipe = None
        gc.collect()
        torch.cuda.empty_cache()

    @torch.inference_mode()
    def run(self, req: TryOnRequest) -> Image.Image:
        self.load()
        full = req.person
        # Zoom in on the people (more pixels for the clothes) and, at the
        # end, paste back only the clothing area. Klein is a general editor and would otherwise
        # also "improve" faces, hands, bags and the background.
        from ..preprocess.parsing import _dilate, agnostic_mask, parser
        from .crop import _focus_box

        full_labels = parser.labels(full)
        mask_full = agnostic_mask(full_labels, req.category, req.top_length, req.top_hem_frac)
        # The crop must hold the whole head. Cut through the face (the clothes start at the neck,
        # so their box + margin ends mid-face), Klein redraws a person whose head is cut off and
        # recomposes the picture: the body moved 60 px sideways, so the user's real head, put
        # back at the end, no longer sat on Klein's shoulders (hair cut off, smears at the neck).
        import numpy as np

        from ..preprocess.parsing import FACE, HAIR, HAT

        focus = np.asarray(mask_full) > 127
        head = np.isin(full_labels, [FACE, HAIR, HAT])
        if head.any() and req.category in ("upper", "dress"):
            focus = focus | head
            top = int(np.nonzero(head.any(axis=1))[0].min())
            focus[max(0, top - int(0.04 * full.height)), :] = focus[max(0, top - int(0.04 * full.height)), :] | head.any(axis=0)
        crop_box = _focus_box(
            Image.fromarray(focus.astype(np.uint8) * 255), full.size, full.width / full.height, context=0.15
        )
        person = full.crop(crop_box)
        width, height = fit_size(person.size, settings.max_side, multiple=16)
        person_in = person.resize((width, height), Image.LANCZOS)
        worn = req.garment_worn is not None
        garment_in = _fit(req.garment_worn if worn else req.garment, settings.klein_ref_side)
        from .. import debug as _dbg

        _dbg.save("garment-ref", garment_in)
        images = [person_in, garment_in]
        has_back = req.view == "back" and req.garment_back is not None
        if has_back:
            images.append(_fit(req.garment_back, 768))

        # Name the clothes the person keeps ("navy trousers"), or Klein borrows image 2's.
        keep_desc = ""
        if req.category in ("upper", "lower"):
            from ..preprocess.describe import describe

            keep_desc = describe(full, "lower" if req.category == "upper" else "upper")
            # Only a single kept piece makes sense here ("navy trousers"), not a whole outfit.
            piece = ("trousers", "skirt") if req.category == "upper" else ("top",)
            if " with " in keep_desc or not keep_desc.endswith(piece):
                keep_desc = ""
        prompt = build_prompt(req, has_back, worn, keep_desc)
        log.info("klein prompt: %s", prompt)
        from .. import debug

        debug.save("mask", mask_full)
        debug.save("klein-in", person_in)
        import time as _t

        t0 = _t.time()
        log.info("klein inputs: person %s, refs %s", person_in.size, [i.size for i in images[1:]])
        out = self._pipe(
            image=images,
            prompt=prompt,
            width=width,
            height=height,
            num_inference_steps=req.steps or 4,  # the distilled 4B model is made for 4 steps
            guidance_scale=1.0,
            generator=torch.Generator(device="cpu").manual_seed(req.seed),
            callback_on_step_end=_stop_if_cancelled(req),
        ).images[0]
        debug.save("klein-raw", out)
        log.info("klein diffusion took %.1f s", _t.time() - t0)
        t0 = _t.time()
        out = out.resize(person.size, Image.LANCZOS)
        # Keep Klein's pixels only in (and just around) the clothing area: a little wider than the
        # mask, because Klein may draw the new garment slightly looser than the old one.
        import numpy as np

        from ..preprocess.parsing import BG

        from ..preprocess.parsing import BELT, DRESS, L_ARM, PANTS, R_ARM, SCARF, SKIRT, UPPER

        crop_mask = np.asarray(mask_full.crop(crop_box)) > 127
        grow = max(3, int(0.012 * max(person.size)))
        keep = _dilate(crop_mask, grow)
        # Every pixel of the OLD garment is replaced, including the bits the mask protects around
        # the hands: otherwise a blazer cuff survives next to a hand in a pocket (a blue patch on a
        # black sweater). The hands themselves are still the user's (skin is not in this set).
        old_ids = {
            "upper": [UPPER, SCARF],
            "lower": [PANTS, SKIRT, BELT],
            "dress": [UPPER, SCARF, PANTS, SKIRT, BELT, DRESS],
        }[req.category]
        crop_labels = parser.labels(person)
        keep |= _dilate(np.isin(crop_labels, old_ids), 2)
        # Klein drew the new garment over where the old skin was (a long sleeve over a bare arm).
        if req.category in ("upper", "dress"):
            out_labels = parser.labels(out)
            keep |= np.isin(crop_labels, [L_ARM, R_ARM]) & np.isin(out_labels, old_ids) & _dilate(crop_mask, grow * 4)
            # Hands: take Klein's, whole. Mixing the user's hand with Klein's sleeve and pocket made
            # a patchwork at a hand in a pocket; Klein keeps hands, phones and watches well.
            hands = np.isin(crop_labels, [L_ARM, R_ARM]) | np.isin(out_labels, [L_ARM, R_ARM])
            keep |= _dilate(hands, grow) & _dilate(crop_mask, grow * 6)
        # Hair comes only from the user's photo. Klein redraws the head a little off (narrower,
        # shifted), so its hair tips land beside the real ones; once the real hair is locked back
        # they float on the shoulder as a flat-topped chunk. Paint Klein's extra hair over with
        # its own surroundings (the jacket, the wall) before pasting.
        from ..preprocess.parsing import HAIR

        out_labels = parser.labels(out)
        extra_hair = (out_labels == HAIR) & ~_dilate(crop_labels == HAIR, max(2, grow // 2)) & keep
        if extra_hair.sum() > 30:
            import cv2

            hole = _dilate(extra_hair, 3)
            # Where the photo had background (wall above the shoulder), show the photo; only where
            # it had the body does Klein's garment get filled in from around the hole.
            keep &= ~(hole & (crop_labels == BG))
            fill = (hole & (crop_labels != BG)).astype(np.uint8) * 255
            out = Image.fromarray(cv2.inpaint(np.asarray(out.convert("RGB")), fill, 9, cv2.INPAINT_TELEA))
            debug.save("klein-hair-removed", out)
            log.info("klein: painted over %d px of hair Klein drew outside the user's own", int(extra_hair.sum()))
        # Where the photo was clearly background and Klein also drew background, keep the user's
        # pixels: Klein re-renders the background slightly lighter, which shows as an outline.
        edge = max(3, int(0.006 * max(person.size)))
        was_bg = ~_dilate(crop_labels != BG, edge)
        now_bg = out_labels == BG
        keep &= ~(was_bg & now_bg)
        alpha = Image.fromarray((keep * 255).astype(np.uint8), "L")
        alpha = alpha.filter(ImageFilter.GaussianBlur(2))
        debug.save("paste-alpha", alpha)
        result = full.copy()
        result.paste(out, crop_box[:2], alpha)
        log.info("klein paste-back took %.1f s", _t.time() - t0)
        return result


def _fit(img: Image.Image, max_side: int) -> Image.Image:
    w, h = fit_size(img.size, max_side, multiple=16)
    return img.resize((w, h), Image.LANCZOS)


def _stop_if_cancelled(req: TryOnRequest):
    """A diffusers step callback that ends the run when the job was cancelled."""

    def cb(pipe, step, timestep, kwargs):
        req.check_cancel()
        return kwargs

    return cb
