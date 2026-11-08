"""The engine contract every try-on model implements."""

from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Literal, Optional

from PIL import Image

Category = Literal["upper", "lower", "dress"]
View = Literal["front", "side", "back"]


@dataclass
class TryOnRequest:
    person: Image.Image
    garment: Image.Image
    category: Category = "upper"
    view: View = "front"
    # Optional photo of the garment's back, used for back views when the store has one.
    garment_back: Optional[Image.Image] = None
    # The garment as worn in a lookbook photo, untouched pixels (for engines that read a reference
    # image and copy its detail). None for product photos.
    garment_worn: Optional[Image.Image] = None
    # The outfit read as a checklist of layers (preprocess.layers.Layer), only the ones this
    # category changes, and the top's length relative to the torso (1.6 = coat to mid-thigh).
    layers: list = field(default_factory=list)
    top_length: Optional[float] = None
    # Where the new top's hem falls on THIS person, as a share of the photo's height (the outfit's
    # hem measured on the model's skeleton, moved to the same place on the user's skeleton).
    top_hem_frac: Optional[float] = None
    # Free-text hint about the garment ("navy wool crew-neck sweater"); improves prompt-driven engines.
    description: str = ""
    seed: int = 42
    steps: Optional[int] = None
    # Paste the original pixels back outside the garment area (face, hair, background stay exact).
    lock_outside: bool = False
    # Live mode: the result goes back as the new clothes only (transparent elsewhere), for the
    # page to keep on the person's moving camera image.
    live: bool = False
    # Size check: how many sizes the garment shown is off this shopper's size (app/sizing.py).
    # < 0 = drawn tight (all stocked sizes too small), > 0 = drawn loose (all too big).
    fit_off: int = 0
    # The size verdict for this look (size letters only; never height or weight).
    fit: Optional[dict] = None
    # Longest side handed to the model for this request (None = settings.max_side).
    max_side: Optional[int] = None
    # Set when the job is cancelled; engines stop at the next diffusion step.
    cancel: Optional[threading.Event] = None

    def check_cancel(self) -> None:
        if self.cancel is not None and self.cancel.is_set():
            raise Cancelled()


class Cancelled(Exception):
    """The job was cancelled while it was running."""


@dataclass(frozen=True)
class EngineInfo:
    id: str
    label: str
    licence: str
    commercial_use: bool
    note: str


class Engine(ABC):
    info: EngineInfo

    @abstractmethod
    def load(self) -> None:
        """Load weights onto the GPU. Called lazily by the job worker."""

    @abstractmethod
    def unload(self) -> None:
        """Free GPU memory so another engine can load."""

    @abstractmethod
    def run(self, req: TryOnRequest) -> Image.Image:
        """Return the person wearing the garment, same aspect ratio as the person photo."""

    @property
    @abstractmethod
    def loaded(self) -> bool: ...

    # (repo, file) pairs that must be in the local Hugging Face cache before this engine can run.
    required_files: tuple[tuple[str, str], ...] = ()

    def missing_files(self) -> list[str]:
        from huggingface_hub import try_to_load_from_cache
        import os

        missing = []
        for repo, filename in self.required_files:
            path = try_to_load_from_cache(repo, filename)
            if not (isinstance(path, str) and os.path.isfile(path)):
                missing.append(f"{repo}/{filename}")
        return missing
