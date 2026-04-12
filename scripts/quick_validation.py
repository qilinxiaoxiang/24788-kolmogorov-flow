"""Quick validation with REAL Kolmogorov Flow data at reduced scale.

Once the H5 file is downloaded, this script:
  1. Computes norm stats on real data
  2. Trains DDPM for 1000 steps at 64x64 (downsampled) — produces a real
     loss curve and 4 sample fields
  3. Same for Flow Matching
  4. Saves both loss curves to results/ so the report has at least
     one real training-dynamics figure

Runs in ~20-40 min on MPS. Full-scale training should still happen on
Colab T4 per RUN_INSTRUCTIONS.md.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.env import DATA_DIR, CKPT_DIR, RESULTS_DIR, summary
from src.data import (
    KolmFlowFrames, NormStats, compute_norm_stats, get_trajectory_split,
    DEFAULT_PATH, N_TIMESTEPS,
)
from src.unet import UNet, count_parameters
from src.diffusion import DDPM
from src.flow_matching import FlowMatching


IMG_SIZE = 64           # downsampled from 160 for speed
BATCH_SIZE = 16
MAX_STEPS = 1000
LR = 2e-4
SEED = 0


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class DownsampledFrames(torch.utils.data.Dataset):
    """Wrap KolmFlowFrames and downsample to IMG_SIZE."""

    def __init__(self, split: str, norm: NormStats):
        self.base = KolmFlowFrames(split=split, norm=norm)
        self.img_size = IMG_SIZE

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        x = self.base[idx]  # [1, 160, 160]
        # Area-averaging pool for anti-aliasing
        x = F.adaptive_avg_pool2d(x, (self.img_size, self.img_size))
        return x


def run_one(model_type: str, device: torch.device, norm: NormStats) -> dict:
    print(f"\n=== {model_type.upper()} quick validation (IMG={IMG_SIZE}, STEPS={MAX_STEPS}) ===")
    torch.manual_seed(SEED)

    # Tiny UNet for speed (about 1.4M params at these settings)
    net = UNet(
        in_ch=1, out_ch=1, base_ch=32, ch_mults=(1, 2, 2), n_res_blocks=2, dropout=0.0
    ).to(device)
    print(f"Params: {count_parameters(net) / 1e6:.2f}M")

    if model_type == "ddpm":
        wrapper = DDPM(net, T=500).to(device)
    else:
        wrapper = FlowMatching(net).to(device)

    ds = DownsampledFrames(split="train", norm=norm)
    loader = torch.utils.data.DataLoader(
        ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, drop_last=True
    )
    print(f"Train frames: {len(ds)}  batches: {len(loader)}")

    opt = torch.optim.AdamW(net.parameters(), lr=LR)
    losses = []

    t0 = time.time()
    step = 0
    data_iter = iter(loader)
    while step < MAX_STEPS:
        try:
            x = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            x = next(data_iter)
        x = x.to(device, non_blocking=True)
        opt.zero_grad(set_to_none=True)
        loss = wrapper.loss(x)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        losses.append(loss.item())
        step += 1

        if step % 50 == 0:
            rate = step / (time.time() - t0)
            recent = sum(losses[-50:]) / 50
            print(f"  [{step:4d}/{MAX_STEPS}] loss_50={recent:.4f}  ({rate:.1f} it/s)")

    dt = time.time() - t0
    print(f"Done in {dt:.1f}s ({MAX_STEPS / dt:.1f} it/s)")

    # Draw a few samples to save
    print("Generating 4 samples...")
    wrapper.eval()
    with torch.no_grad():
        if model_type == "ddpm":
            samples = wrapper.ddim_sample(n=4, n_steps=30, img_size=IMG_SIZE, device=device)
        else:
            samples = wrapper.sample(n=4, img_size=IMG_SIZE, n_steps=30, device=device)
    samples = samples.float().cpu().numpy() * norm.std + norm.mean

    return {"losses": losses, "samples": samples, "wall_clock_s": dt}


def main():
    print(summary())
    device = get_device()
    print(f"device = {device}")

    if not DEFAULT_PATH.exists():
        raise SystemExit(f"Data file {DEFAULT_PATH} not present — run download first.")

    # Fresh norm stats (the file may have been removed earlier)
    stats_path = DATA_DIR / "norm_stats.json"
    if stats_path.exists():
        d = json.loads(stats_path.read_text())
        norm = NormStats(mean=d["mean"], std=d["std"])
        print(f"Loaded norm stats: mean={norm.mean:.4f}, std={norm.std:.4f}")
    else:
        print("Computing norm stats (this reads ~30 trajectories)...")
        norm = compute_norm_stats(n_sample_traj=10)
        stats_path.write_text(json.dumps(norm.to_dict(), indent=2))
        print(f"norm: mean={norm.mean:.4f}, std={norm.std:.4f}")

    results = {}
    for mt in ("ddpm", "fm"):
        results[mt] = run_one(mt, device, norm)

    # Save loss curves side-by-side
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for i, mt in enumerate(("ddpm", "fm")):
        axes[i].plot(results[mt]["losses"], alpha=0.5, lw=0.7)
        # Running mean
        L = np.asarray(results[mt]["losses"])
        w = 25
        if len(L) > w:
            rm = np.convolve(L, np.ones(w) / w, mode="valid")
            axes[i].plot(np.arange(w - 1, len(L)), rm, "k-", lw=1.5, label="running mean (50)")
            axes[i].legend()
        axes[i].set_xlabel("step")
        axes[i].set_ylabel("loss")
        axes[i].set_title(f"{mt.upper()} ({MAX_STEPS} steps @ {IMG_SIZE}x{IMG_SIZE}, MPS)")
        axes[i].grid(alpha=0.3)
    fig.tight_layout()
    out = RESULTS_DIR / "quick_validation_loss.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"Saved {out}")

    # Save sample grids side-by-side
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    for row, mt in enumerate(("ddpm", "fm")):
        for col in range(4):
            s = results[mt]["samples"][col, 0]
            vmax = max(abs(s.min()), abs(s.max()))
            axes[row, col].imshow(s, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
            axes[row, col].axis("off")
        axes[row, 0].set_title(mt.upper(), loc="left", fontsize=11)
    fig.suptitle(f"Quick-validation samples (IMG={IMG_SIZE}, {MAX_STEPS} steps)")
    fig.tight_layout()
    out = RESULTS_DIR / "quick_validation_samples.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"Saved {out}")

    summary_data = {
        "img_size": IMG_SIZE,
        "steps": MAX_STEPS,
        "ddpm_final_loss": float(np.mean(results["ddpm"]["losses"][-50:])),
        "fm_final_loss":   float(np.mean(results["fm"]["losses"][-50:])),
        "ddpm_wall_s": results["ddpm"]["wall_clock_s"],
        "fm_wall_s":   results["fm"]["wall_clock_s"],
        "device": str(device),
    }
    with open(RESULTS_DIR / "quick_validation_summary.json", "w") as f:
        json.dump(summary_data, f, indent=2)
    print(json.dumps(summary_data, indent=2))


if __name__ == "__main__":
    main()
