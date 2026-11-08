# Fitting Room: plan

## Goal

A reusable try-on block that drops into any shop. A shopper sees themselves wearing a garment
from the front, side and back. It runs on **our own GPU servers**, not a vendor API.

The reference product is Anywear (Decart). It streams the camera to Decart's GPUs over WebRTC, and
their Lucy VTON video model redraws every frame live, at about 2 credits per second.

We start with **photo mode**, which is far cheaper because the GPU works for a few seconds per photo
instead of every second of video. Live video comes later, on the same garment and engine
foundations.

## Phases

| Phase | What | Status |
|---|---|---|
| **1. Website, photo mode, local GPU** | Fitting Room site, FastAPI GPU service, FLUX.2 Klein + CatVTON + 2D overlay, job queue, in-memory privacy | **Built in this repo** |
| 2. Quality pass | Run the evaluation set (below), tune prompts, decide whether Klein needs a try-on LoRA fine-tune | Next |
| 3. Server deployment | Docker image, scale-to-zero GPU workers, object-store-free (in-memory) results, per-tenant API keys, rate limits, metering | |
| 4. Embeddable widget | `<tryon-mirror>` web component + one-line script tag for any shop page | |
| 5. React Native / Expo module and Chrome extension | Same API; the extension adds a "Try on" button to product images on any site | Later |
| 6. Live video | WebRTC (LiveKit) + streaming video model, as researched; needs H100-class GPUs | Later |

## Architecture (phase 1)

```
Browser (web/)                                  GPU server (server/)
──────────────────────────────                  ─────────────────────────────────────────
Fitting Room UI                                  FastAPI  /api/tryon  /api/jobs/{id}
 ├─ photo slots: front / side / back             │
 ├─ garment front (+ optional back)              ▼
 ├─ 2D overlay (MediaPipe Pose, in-browser) ◄─┐  JobManager: bounded queue, ONE worker thread,
 └─ Mirror: curtain compare, download         │  one engine loaded at a time (8 GB GPU)
        ▲                                      │   ├─ FluxKleinEngine (4-bit NF4 on 8 GB GPUs)
        │ poll status, fetch PNG, DELETE job   │   ├─ CatVTONEngine (eval only)
        └──────────────────────────────────────┘   └─ HumanParser (SegFormer) for masks / lock
```

- **One request = one angle.** Front, side and back are separate jobs that share the garment, seed
  and settings, so the results stay consistent.
- **Back views.** If the shop has a back photo of the garment, it goes in as a third reference image.
  Otherwise the prompt tells the model to keep the back plain, without the front's logos or prints.
- **"Keep face, hair and background exact".** After the edit, the original pixels are pasted back
  everywhere outside the garment area (found by the human parser), with a feathered edge.
- **Privacy.** Nothing is written to disk. Input photos are dropped when the job finishes, and the
  result when the site downloads it (or after 10 minutes).

## Why these engines

- **FLUX.2 Klein 4B (Apache 2.0)** is a modern editing model that takes several reference images,
  runs in 4 steps, and is licensed for commercial use. It is not trained for try-on, so prompt
  quality matters. If the evaluation shows weak garment fidelity, phase 2 adds a try-on LoRA
  trained on our own licensed pairs.
- **CatVTON (CC BY-NC-SA)** is a purpose-built try-on model. It's the quality bar for Klein and
  never ships.
- **2D overlay** shows the zero-cost option and why it isn't enough. It also proves the pose
  pipeline that the mobile app can reuse for "step back" guidance.

## Evaluation set (phase 2)

At least 30 people, varied in body type, skin tone and pose (arms crossed, hands in pockets).
Each person gets 5 garments: a tee, a long-sleeve knit, a jacket, trousers and a dress. Each pair is
shot at front, side and back.

What we measure:

- **Garment fidelity.** Colour, pattern and logo placement, judged by people. LPIPS on the garment
  crop.
- **Identity.** Face unchanged (face-embedding similarity before and after).
- **Artefacts.** Hands, hair edges, leftover old clothing.
- **Speed.** p50/p95 per engine and GPU type, model load time, VRAM peak (`scripts/smoke_test.py`
  prints these).

## Cost model (rough, verify in phase 3)

| | Local laptop (RTX 4060, 8 GB, 4-bit) | Server GPU (L4/L40S, bf16) |
|---|---|---|
| Klein, one 1024 px photo | to be measured | ~2–5 s |
| Cost per photo | electricity | ~$0.001–0.002 |
| 4 angles | — | ~$0.005–0.01 |
| Idle | $0 | $0 with scale-to-zero |

## Open questions

- The human parser's licence ("other" on the Hub). We may need a replacement before shipping.
- Garment photos that show a model wearing the item. Phase 2 adds garment extraction (cutting the
  garment out first).
- Content safety. We need an NSFW filter on inputs and outputs before a public launch; the Klein
  model card asks deployers to add one.
