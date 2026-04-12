# 24-788 Mini-Project: Generative Models for Kolmogorov Flow

Comparing **Denoising Diffusion (DDPM)** and **Conditional Flow Matching** for
unconditional generation of 2D Kolmogorov Flow vorticity fields.

- **Dataset**: Kolmogorov Flow, `KolmFlow_valid_256.h5`
  ([HuggingFace](https://huggingface.co/datasets/ayz2/temporal_pdes))
  — 256 trajectories × 200 timesteps of 160×160 single-channel vorticity
- **Baseline**: DDPM with cosine schedule, ε-prediction loss (Ho et al. 2020,
  Nichol & Dhariwal 2021)
- **Variant**: Conditional Flow Matching with straight-line paths and Euler
  integration (Lipman et al. 2023)
- **Shared backbone**: 17.6 M-param U-Net. Same architecture for both models
  so the comparison isolates the training objective.

## Task framing

The dataset's default task is a regression problem (predict forward evolution
of the PDE). We instead frame it as an **unconditional generation** task —
each (trajectory, timestep) pair becomes an i.i.d. 160×160 "image" of
vorticity. The project description explicitly allows this re-framing for any
dataset in the curated list.

## Setup

```bash
pip install -r requirements.txt
```

Notebooks auto-detect Colab vs local via `src/env.py`. In Colab, Google Drive
is mounted and the repo path is discovered automatically across both
`My Drive` and `Othercomputers/` (Drive for Desktop).

## Layout

```
code/
├── src/                   # Reusable modules
│   ├── env.py             # Environment + path discovery (local/Colab)
│   ├── data.py            # KolmFlowFrames Dataset, norm stats, split
│   ├── unet.py            # Shared U-Net backbone (17.6M params)
│   ├── diffusion.py       # DDPM: cosine schedule, ε-loss, DDIM sampler
│   ├── flow_matching.py   # CFM: straight-line paths, Euler/Heun/RK4
│   ├── train.py           # Unified training loop (both models)
│   └── evaluate.py        # Energy spectrum, vorticity PDF, Wasserstein-1
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_train.ipynb
│   └── 03_evaluate.ipynb
├── scripts/
│   ├── overfit_test.py    # Sanity: pipeline learns a tiny synthetic batch
│   └── integration_test.py# Full train/checkpoint/resume/preview test
├── report/
│   ├── report.tex
│   ├── references.bib
│   └── neurips_2023.sty
├── data/                  # H5 + norm_stats.json (gitignored)
├── checkpoints/           # Model weights (gitignored)
├── results/               # Figures for the report (gitignored)
├── README.md
├── RUN_INSTRUCTIONS.md    # Step-by-step Colab workflow
└── requirements.txt
```

## Running

See **[`RUN_INSTRUCTIONS.md`](./RUN_INSTRUCTIONS.md)** for the full Colab
workflow. TL;DR:

1. Open `notebooks/01_data_exploration.ipynb` in Colab → run all (downloads
   data, computes stats, visualizes samples).
2. Open `notebooks/02_train.ipynb`, set `MODEL_TYPE = 'ddpm'` → run all
   (~1.5–2 h on T4). Then set `MODEL_TYPE = 'fm'` and re-run the config
   and train cells.
3. Open `notebooks/03_evaluate.ipynb` → run all (generates all figures).
4. Compile the report: `cd report && pdflatex report && bibtex report && pdflatex report && pdflatex report`.

## Reproducing results

Trained checkpoints live in `checkpoints/{ddpm,fm}_main/`. `03_evaluate.ipynb`
is the `reproduce_results` notebook required by the assignment: it loads
checkpoints, regenerates all figures, and writes scalar metrics to
`results/summary.json`.

## Design notes

- **Shared U-Net**: makes the DDPM-vs-FM comparison about objectives, not
  architecture.
- **Trajectory-level split** (230/13/13): prevents frame-level leakage
  between train and test — adjacent timesteps within a trajectory are
  strongly correlated.
- **Continuous-time API** (`t ∈ [0, 1]`): DDPM internally discretizes to
  1000 steps but exposes the same scalar-time interface as FM, so the
  U-Net is byte-identical between the two models.
- **EMA weights** (decay 0.999) for evaluation — standard for diffusion
  models; also improves FM samples.
- **Cosine LR schedule** with 500-step warmup, AdamW, gradient clipping
  at norm 1.0, AMP (FP16) on CUDA only.

## Smoke tests

```bash
# Verify U-Net shape
PYTHONPATH=. python -m src.unet
# Verify DDPM + FM forward + sample
PYTHONPATH=. python -m src.diffusion
PYTHONPATH=. python -m src.flow_matching
# Verify pipeline learns a fixed synthetic batch
PYTHONPATH=. python scripts/overfit_test.py
# Verify full pipeline end-to-end (train, save, resume, preview)
PYTHONPATH=. python scripts/integration_test.py
```
