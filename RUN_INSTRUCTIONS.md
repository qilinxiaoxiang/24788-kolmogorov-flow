# Running the Project on Colab

All training, evaluation, and report compilation steps, in order.

## 0. Prerequisites

- Your local repo (`code/`) is synced to Google Drive via Drive for Desktop.
  In Colab that shows up at either
  `/content/drive/MyDrive/...` or `/content/drive/Othercomputers/My MacBook Pro/...`
  — `src/env.py` auto-discovers which.
- Colab Pro (for GPU runtimes + longer session limits).
- Gradescope + GitHub access for final submission.

## 1. Data + exploration

Open **`notebooks/01_data_exploration.ipynb`** in Colab. Runtime → Change
runtime type → **T4 GPU**. Run all cells.

This:
1. Mounts Drive.
2. Downloads `KolmFlow_valid_256.h5` (~5 GB) to `data/`.
3. Splits the 256 trajectories into 230 train / 13 val / 13 test.
4. Computes global mean/std from 30 sampled training trajectories and
   saves to `data/norm_stats.json`.
5. Saves three sanity-check figures to `results/`:
   `trajectory_evolution.png`, `trajectory_diversity.png`,
   `vorticity_histogram.png`.

**One-time cost: ~5 min.** The downloaded data persists in Drive, so
subsequent notebooks skip the download.

## 2. Train DDPM (baseline)

Open **`notebooks/02_train.ipynb`**. In the "CONFIG" cell, leave
`MODEL_TYPE = 'ddpm'`. Run all.

Checkpoints land in `checkpoints/ddpm_main/`:
- `state.pt` — full training state (rolling; overwritten each save).
- `ema_stepN.pt` — EMA weights at step N (last 3 kept).
- `config.json` — the `TrainConfig` used.
- `samples_step*.png` — visual samples drawn every 2000 steps.

At 40 k steps with batch 32 on a T4, expect **~1.5–2 h**.

If the Colab session drops, just re-run the notebook — `train()` resumes
from the most recent `state.pt`.

## 3. Train Flow Matching (variant)

In the same notebook, change `MODEL_TYPE = 'fm'` and re-run the CONFIG
and train cells. Checkpoints land in `checkpoints/fm_main/`. Expect
similar runtime to DDPM (same backbone, same step count).

## 4. Evaluate

Open **`notebooks/03_evaluate.ipynb`** and run all cells. It loads the
EMA weights from both runs, draws 256 samples from each, and produces:
- `results/samples_panel.png` — real vs DDPM vs FM qualitative grid
- `results/energy_spectrum.png` — radial energy spectrum comparison
- `results/vorticity_pdf.png` — marginal vorticity distribution
- `results/nfe_sweep.png` + `nfe_sweep.csv` — quality vs inference steps
- `results/summary.json` — scalar metrics for the report table

## 5. Compile the report

In `report/`:

```bash
pdflatex report.tex
bibtex report
pdflatex report.tex
pdflatex report.tex
```

Fill in the `\texttt{TBD}` entries in Table 1 and the `[Placeholder]`
figure references using the numbers and PNGs from step 4. The draft
structure, citations, and page budget are already set for a solo report.

## 6. Create GitHub repo + submit

```bash
cd code/
# `.gitignore` already excludes data, checkpoints, results, LaTeX intermediates
git remote add origin git@github.com:<you>/24788-mini-project.git
git push -u origin main
```

Submit PDF via Gradescope. Include the GitHub link in the README (see
`README.md`) or at the end of the report.

---

## Sanity checks that passed locally (MPS)

- `scripts/overfit_test.py` — both models overfit a fixed synthetic batch.
  DDPM loss 0.78 → 0.10 and FM loss 1.63 → 0.38 over 300 steps.
- `scripts/integration_test.py` — full pipeline (train, checkpoint,
  resume, EMA-weighted preview sampling) on synthetic data.
- `src/unet.py`, `src/diffusion.py`, `src/flow_matching.py`,
  `src/evaluate.py` all have runnable `__main__` smoke tests.

## Troubleshooting

- **`src/env.py` raises "Could not locate project repo in Drive"**: your
  repo is synced to a path not in `_search_roots`. Edit `src/env.py` and
  add your Drive path.
- **Out of GPU memory**: reduce `batch_size` in the config cell from 32
  to 16.
- **"Already have the file (X GB)" but X ≠ 5.27**: previous download was
  interrupted. Delete `data/KolmFlow_valid_256.h5` and re-run.
- **Flow Matching sampling diverges**: check `sigma_min` in the config;
  default 0.0 is correct. If it remains unstable, use `method='heun'`
  in the sampler.
