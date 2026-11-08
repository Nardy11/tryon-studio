"""Download model weights into the Hugging Face cache, so the first try-on doesn't wait for them.

    python server/scripts/download_models.py            # everything
    python server/scripts/download_models.py klein      # one group: klein | parser | safety | pose

Safe on flaky connections: each file is retried, partial files resume, and the run only reports
success once every required file is really on disk. (`snapshot_download` alone silently falls
back to a half-downloaded local copy when the network drops.)
"""

from __future__ import annotations

import os
import sys
import time

from huggingface_hub import hf_hub_download, try_to_load_from_cache

# repo -> files that must exist locally (relative paths in the repo)
GROUPS: dict[str, dict[str, list[str]]] = {
    "parser": {
        "mattmdjaga/segformer_b2_clothes": [
            "config.json",
            "preprocessor_config.json",
            "model.safetensors",
        ],
    },
    # Content safety (CPU): nudity classifier and CLIP for garment type / age.
    "safety": {
        "Falconsai/nsfw_image_detection": ["config.json", "preprocessor_config.json", "model.safetensors"],
        "openai/clip-vit-large-patch14": [
            "config.json", "preprocessor_config.json", "tokenizer_config.json", "tokenizer.json",
            "vocab.json", "merges.txt", "special_tokens_map.json", "model.safetensors",
        ],
    },
    "catvton": {
        "zhengchong/CatVTON": ["mix-48k-1024/attention/model.safetensors"],
        "booksforcharlie/stable-diffusion-inpainting": [
            "model_index.json",
            "scheduler/scheduler_config.json",
            "unet/config.json",
            "unet/diffusion_pytorch_model.bin",  # this mirror only ships .bin weights (3.4 GB)
        ],
        "stabilityai/sd-vae-ft-mse": ["config.json", "diffusion_pytorch_model.safetensors"],
    },
    # ~16 GB in total. The single-file checkpoint in the repo root is skipped (diffusers doesn't need it).
    "klein": {
        "black-forest-labs/FLUX.2-klein-4B": [
            "model_index.json",
            "scheduler/scheduler_config.json",
            "text_encoder/config.json",
            "text_encoder/generation_config.json",
            "text_encoder/model.safetensors.index.json",
            "text_encoder/model-00001-of-00002.safetensors",
            "text_encoder/model-00002-of-00002.safetensors",
            "tokenizer/added_tokens.json",
            "tokenizer/chat_template.jinja",
            "tokenizer/merges.txt",
            "tokenizer/special_tokens_map.json",
            "tokenizer/tokenizer_config.json",
            "tokenizer/vocab.json",
            "tokenizer/tokenizer.json",
            "transformer/config.json",
            "transformer/diffusion_pytorch_model.safetensors",
            "vae/config.json",
            "vae/diffusion_pytorch_model.safetensors",
        ],
    },
}


def have(repo: str, filename: str) -> bool:
    path = try_to_load_from_cache(repo, filename)
    return isinstance(path, str) and os.path.isfile(path)


def fetch(repo: str, filename: str, attempts: int = 50) -> None:
    for i in range(1, attempts + 1):
        try:
            hf_hub_download(repo, filename)
            if have(repo, filename):
                return
            raise RuntimeError("download returned but the file is not in the cache")
        except Exception as exc:  # noqa: BLE001 - network errors come in many types
            wait = min(60, 5 * i)
            print(f"   {filename}: {exc.__class__.__name__}: retry {i}/{attempts} in {wait}s", flush=True)
            time.sleep(wait)
    raise SystemExit(f"Gave up on {repo}/{filename}")


POSE_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task"
)


def fetch_pose() -> None:
    """MediaPipe Pose (Apache 2.0): the body ruler that measures garment lengths."""
    import urllib.request
    from pathlib import Path

    dest = Path(__file__).resolve().parents[1] / "models" / "pose_landmarker_full.task"
    if dest.is_file():
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    print("[pose] pose_landmarker_full.task ...", flush=True)
    urllib.request.urlretrieve(POSE_URL, dest)
    print("[pose] complete", flush=True)


def main() -> None:
    wanted = sys.argv[1:] or [g for g in GROUPS if g != "catvton"] + ["pose"]
    if "pose" in wanted:
        fetch_pose()
        wanted = [g for g in wanted if g != "pose"]
    for group in wanted:
        if group not in GROUPS:
            sys.exit(f"Unknown group '{group}'. Use: {', '.join(GROUPS)}")
        for repo, files in GROUPS[group].items():
            for filename in files:
                if have(repo, filename):
                    continue
                t0 = time.time()
                print(f"[{group}] {repo}/{filename} ...", flush=True)
                fetch(repo, filename)
                print(f"[{group}] {repo}/{filename} done ({time.time() - t0:.0f}s)", flush=True)
        print(f"[{group}] complete", flush=True)
    print("ALL REQUESTED MODELS PRESENT", flush=True)


if __name__ == "__main__":
    main()
