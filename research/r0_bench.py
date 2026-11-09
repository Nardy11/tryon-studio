"""R0 benchmark: StreamDiffusionV2 (Wan 2.1 1.3B, causal, few-step) as a real-time video-to-video base.

Measures, on this machine: steady frames per second, delay per chunk, peak GPU memory, and saves the
output video next to the input for a side-by-side look at identity and background drift.

Run inside the WSL venv (research/setup_wsl.sh), from ~/sdv2 so the model paths resolve:
  python /mnt/c/.../research/r0_bench.py --video <clip.mp4> --height 320 --width 576 --seconds 6
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch

PROMPT = (
    "The person is wearing a khaki long-sleeve button shirt with two chest pockets and a cream sweater "
    "tied over the shoulders, realistic, same face, same room"
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--prompt", default=PROMPT)
    ap.add_argument("--height", type=int, default=320)
    ap.add_argument("--width", type=int, default=576)
    ap.add_argument("--seconds", type=float, default=6.0, help="how much of the input to process")
    ap.add_argument("--mode", default="single-wo", choices=["single", "single-wo"])
    ap.add_argument("--fast", action="store_true", help="TAEHV decoder (+ TensorRT where available)")
    ap.add_argument("--out", required=True, help="output folder")
    a = ap.parse_args()

    from streamdiffusionv2 import StreamDiffusionV2Pipeline, export_video, load_video

    os.makedirs(a.out, exist_ok=True)
    tag = f"{a.height}x{a.width}-{a.mode}{'-fast' if a.fast else ''}"
    torch.cuda.reset_peak_memory_stats()

    t0 = time.time()
    stream = StreamDiffusionV2Pipeline(checkpoint_folder="ckpts/wan_causal_dmd_v2v", mode=a.mode)
    if a.fast:
        stream.enable_acceleration(fast=True)
    stream.prepare(a.prompt)
    load_s = time.time() - t0

    video = load_video(a.video, height=a.height, width=a.width)
    keep = int(a.seconds * 16)  # the pipeline works at 16 fps
    video = video[:keep] if hasattr(video, "__getitem__") else video

    decoded, chunk_times, chunk_frames = [], [], []
    noise_scale = stream.noise_scale
    for video_chunk in stream.chunk_video(video):
        torch.cuda.synchronize()
        c0 = time.time()
        enc = stream.encode_chunk(video, video_chunk, previous_noise_scale=noise_scale, initial_noise_scale=stream.noise_scale)
        noise_scale = enc.noise_scale
        den = stream.denoise_chunk(enc)
        if den is None:
            continue
        out = stream.decode_chunk(den)
        torch.cuda.synchronize()
        chunk_times.append(time.time() - c0)
        chunk_frames.append(len(out))
        decoded.append(out)

    output = np.concatenate(decoded, axis=0)
    export_video(output, os.path.join(a.out, f"r0-{tag}.mp4"), fps=16)

    # Steady state: leave out the first chunk (warm-up, cache filling).
    t = chunk_times[1:] or chunk_times
    f = chunk_frames[1:] or chunk_frames
    result = {
        "tag": tag,
        "gpu": torch.cuda.get_device_name(0),
        "resolution": f"{a.width}x{a.height}",
        "mode": a.mode,
        "fast_decoder": a.fast,
        "load_s": round(load_s, 1),
        "frames_out": int(output.shape[0]),
        "steady_fps": round(sum(f) / sum(t), 2),
        "chunk_delay_s_median": round(float(np.median(t)), 3),
        "frames_per_chunk": int(np.median(f)),
        "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
    }
    print(json.dumps(result))
    with open(os.path.join(a.out, "r0-results.jsonl"), "a", encoding="utf-8") as fh:
        fh.write(json.dumps(result) + "\n")


if __name__ == "__main__":
    main()
