# wizzibiz

Tinder for the dawn chorus. A swipe feed of real bird recordings, where the thing doing the ranking is a neural network trained on mel-spectrograms of the songs you like.

Live: **https://wizzibiz.hheylau.workers.dev**

This is the "actually built it" version of the [HackSheffield 8 submission](https://devpost.com/software/wizzibiz): same idea, but with the dataset downloaded, the DSP written, the model trained and the whole thing deployed on Cloudflare Workers instead of described in a README.

---

## What is actually running

```
xeno-canto 5 s clips (20 species)  ─┐
ESC-50 negatives (49 classes)      ─┴─> parquet ─> 16 kHz mono
                                                    │
                                     ┌──────────────┘
                                     ▼
                     STFT 512 / hop 160  →  64-band mel  →  log
                                     │
                    3 × 1 s windows → pool to 64 × 32 → standardise
                                     │
                                     ▼
                      conv10 → pool → conv20 → pool → 64 → 21
                                     │
             ┌───────────────────────┴───────────────────────┐
             ▼                                               ▼
   species prediction (Worker, quantised)        penultimate 64-d embedding
                                                         │
                                          logistic regression on your swipes
                                                         │
                                   deck re-ranked + pitch contours blended
                                                         │
                                                  generated chorus
```

Everything marked DSP is in `src/dsp.ts`, which is imported by **both** the Node training scripts and the Worker, so training features and inference features are produced by the same code.

## Dataset

| | |
|---|---|
| Positives | [`tglcourse/5s_birdcall_samples_top20`](https://huggingface.co/datasets/tglcourse/5s_birdcall_samples_top20) — 20 species, ~500 five-second clips each, cut from [xeno-canto](https://xeno-canto.org) recordings |
| Negatives | [`ashraq/esc50`](https://huggingface.co/datasets/ashraq/esc50) — ESC-50 environmental sounds, minus `chirping_birds`, used as the `not_a_bird` class |
| Species names | resolved against the [eBird taxonomy API](https://documenter.getpostman.com/view/664302/S1ENwy59) |

Each parquet row carries the WAV bytes inline, at 32 kHz mono, so no external audio tooling is needed.

## The DSP

* decode WAV (8/16/24/32-bit PCM and float32), mix to mono
* resample to 16 kHz with a windowed-sinc interpolator (16 taps, Hann, anti-alias cutoff)
* Hann-windowed STFT, `n_fft = 512`, `hop = 160` → 100 frames/second
* 64 triangular mel filters over 50 Hz – 7.8 kHz, HTK mel scale, area-normalised
* natural log with a floor
* three 1-second analysis windows per clip, each pooled from 100 frames to 32, giving a 64 × 32 frequency × time image, flattened to 2048
* per-band standardisation from dataset statistics (exported alongside the weights)

The pitch track is extracted separately for the generative step: parabolic interpolation around the spectral peak, octave-jump suppression, then a 50-point contour plus a normalised energy envelope. Those contours are what gets blended when you press "generate my dawn chorus".

## The model

| | |
|---|---|
| Input | 64 × 32 log-mel image (1 second, 64 mel bands) |
| Architecture | conv 3×3 ×10 → maxpool 2 → conv 3×3 ×20 → maxpool 2 → dense 64 → dense 21 |
| Parameters | 167,189 |
| Optimiser | Adam, lr 3e-3, batch 64, 12 epochs |
| Augmentation | random analysis window, circular time roll, band masking, gaussian noise, inverse-frequency class weights |
| Split | 85 / 15, stratified by clip, 625 held-out clips |
| Export | int8 per-output-row quantisation → base64, 224 kB in the Worker bundle |

**Result: 41.6% top-1 accuracy over 21 classes** (chance ≈ 4.8%), 84% recall on the `not_a_bird` class.
The first attempt was a 2048 → 96 → 64 MLP on the flattened spectrogram; it plateaued at 33% after 30 epochs, which is what motivated switching to convolutions (33% → 41.6% in 12 epochs, and a fifth of the training time per epoch).

Per-class recall on the held-out split:

| Species | Recall | Species | Recall |
|---|---|---|---|
| Mallard | 63% | Red Crossbill | 52% |
| Curve-billed Thrasher | 58% | Barn Swallow | 46% |
| House Sparrow | 58% | Brown-crested Flycatcher | 41% |
| Gray-breasted Wood-Wren | 41% | Common Raven | 35% |

The weak classes (Red-winged Blackbird, Northern House Wren, Swainson's Thrush at 0%) are the ones where a one-second window lands on a call-free gap, and where the species' songs overlap heavily with the sparrows and wrens they get confused with. More data per class and a longer analysis window would be the next lever.

Accuracy is reported honestly in the UI (it is in `public/clips.json` under `model.valAccuracy`), and the "what's that bird?" panel runs the same quantised weights on the Worker over uploaded audio — verified to reproduce the export-time float predictions exactly.

## What "learns what you like" actually means

* every demo clip ships with the 64-dimensional penultimate embedding from the CNN
* your swipes are logged in D1
* `GET /api/taste` fits a logistic regression (liked vs skipped) over those embeddings and reports **leave-one-out** accuracy, because in-sample accuracy on eight swipes is meaningless
* the same weights re-rank the remaining deck
* the blended pitch contour (median across liked clips, gaps interpolated) is resynthesised in the browser with oscillators and a noise bed

## API

| Route | What it does |
|---|---|
| `GET /api/deck?session=` | remaining clips, re-ranked by your taste model |
| `POST /api/swipe` | `{ session, clip, liked }`, upserted into D1 |
| `GET /api/taste?session=` | fits logistic regression on the embeddings of your swipes, returns accuracy, top species and the blended pitch contour |
| `POST /api/classify` | raw WAV body → species distribution, DSP and inference on the Worker |

Bindings: `DB` (D1), `ASSETS` (Workers static assets, which serve the UI and the 88 demo clips).

## Commands

```bash
npm install
npm run preprocess   # parquet -> cache/raw.f32 (features) + cache/meta.json
npm run train        # -> cache/model.json
npm run export       # -> src/model-data.ts + public/clips/ + public/clips.json
npm run dev          # wrangler dev
npm run deploy       # wrangler deploy
```

The parquet shards are expected in `%TEMP%\opencode\birds\data`; change `DATA_DIR` in `tools/preprocess.ts` to point somewhere else.

## Licence notes

Recordings come from xeno-canto, whose recordings are licensed per-recording (mostly CC-BY-NC). They are shipped here as a small demo deck for a non-commercial hackathon project. ESC-50 is CC BY-NC 3.0.
