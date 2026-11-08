# Engine V research: a real-time video try-on model

Goal: Live mode like Anywear. The person is redrawn wearing the outfit on every frame, at
15–20 processed fps, self-hosted. Plan and costs: see the plan of 2026-11-08, phases R0–R5.
Data, frames, checkpoints and the research venv stay local (git-ignored).

## Quality bar, read from the reference recording (Anywear, `my-session (1).mp4`)

| | Reference |
|---|---|
| Input | an ordinary laptop webcam: overexposed room, low detail |
| Output | 1280×720 at about 14.5 fps, 499 frames (about 34 s) |
| Redrawn | every frame. The garment stays attached through leaning, arms crossed, turning sideways and walking off |
| Framing | whatever is visible: upper body only at first, more of the body later. Two people at once |
| Identity | faces, background and the room are kept |
| Fidelity | **not photographic.** The garment's shape morphs (it grows above the heads at times), edges wobble, the print shifts. Strength: coherent, live and tied to the body |

So the target is **temporal coherence and live body-following**, not Photo-mode detail at video
speed. That is what Engine V is measured against in R5, alongside fps and latency.

## R0 · Feasibility spike

1. ✅ Reference frames extracted to `research/eval/reference-frames/` (every 1.5 s) and the quality bar above.
2. ⏳ StreamDiffusionV2 (Apache 2.0) on the laptop RTX 4060 8 GB at 320–480p: fps, latency, VRAM, identity drift.
3. ⏳ The same on a rented RTX 4090 and an H100 (budget approval first).
4. Exit: a measured table and the serving-GPU decision.
