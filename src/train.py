"""Unified training loop for DDPM and Flow Matching.

Usage (from Python):
    from src.train import train
    train(model_type='ddpm', ...)

Checkpoints + loss curves are saved under CKPT_DIR / run_name.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Literal, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .env import CKPT_DIR, DATA_DIR
from .data import NormStats, make_dataloader, compute_norm_stats, download_if_needed, DEFAULT_PATH
from .unet import UNet, count_parameters
from .diffusion import DDPM
from .flow_matching import FlowMatching


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class TrainConfig:
    model_type: Literal["ddpm", "fm"] = "ddpm"
    run_name: str = "default"

    # Model
    base_ch: int = 64
    ch_mults: tuple = (1, 2, 2, 4)
    n_res_blocks: int = 2
    dropout: float = 0.1
    t_dim: int = 256

    # DDPM-specific
    T: int = 1000

    # Flow matching-specific
    sigma_min: float = 0.0

    # Data
    batch_size: int = 32
    num_workers: int = 2
    burn_in: int = 0
    img_size: int = 160

    # Optimization
    lr: float = 2e-4
    weight_decay: float = 0.0
    ema_decay: float = 0.999
    max_steps: int = 40_000
    warmup_steps: int = 500
    grad_clip: float = 1.0

    # Logging
    log_every: int = 50
    sample_every: int = 2_000
    ckpt_every: int = 2_000
    n_samples_preview: int = 4

    # Environment
    device: str = ""  # "" = auto-detect
    amp: bool = True  # Automatic mixed precision

    seed: int = 0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def pick_device(preferred: str = "") -> torch.device:
    if preferred:
        return torch.device(preferred)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class EMA:
    """Exponential moving average of model parameters (common for diffusion)."""

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for n, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[n].mul_(self.decay).add_(p.detach(), alpha=1.0 - self.decay)

    def apply_to(self, model: nn.Module) -> dict:
        """Swap EMA weights into the model; return original weights for restoration."""
        backup = {}
        for n, p in model.named_parameters():
            if p.requires_grad:
                backup[n] = p.detach().clone()
                p.data.copy_(self.shadow[n])
        return backup

    def restore(self, model: nn.Module, backup: dict) -> None:
        for n, p in model.named_parameters():
            if n in backup:
                p.data.copy_(backup[n])

    def state_dict(self) -> dict:
        return {n: t.detach().cpu() for n, t in self.shadow.items()}

    def load_state_dict(self, sd: dict) -> None:
        for n in self.shadow:
            if n in sd:
                self.shadow[n].copy_(sd[n].to(self.shadow[n].device))


def cosine_lr(step: int, warmup: int, max_steps: int, base_lr: float) -> float:
    if step < warmup:
        return base_lr * step / max(warmup, 1)
    progress = (step - warmup) / max(max_steps - warmup, 1)
    import math
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))


# ---------------------------------------------------------------------------
# Build model
# ---------------------------------------------------------------------------

def build_model(cfg: TrainConfig) -> tuple[nn.Module, nn.Module]:
    """Return (wrapper, net). wrapper = DDPM or FlowMatching; net = UNet."""
    net = UNet(
        in_ch=1,
        out_ch=1,
        base_ch=cfg.base_ch,
        ch_mults=cfg.ch_mults,
        n_res_blocks=cfg.n_res_blocks,
        t_dim=cfg.t_dim,
        dropout=cfg.dropout,
    )
    if cfg.model_type == "ddpm":
        wrapper = DDPM(net, T=cfg.T)
    elif cfg.model_type == "fm":
        wrapper = FlowMatching(net, sigma_min=cfg.sigma_min)
    else:
        raise ValueError(cfg.model_type)
    return wrapper, net


# ---------------------------------------------------------------------------
# Main train loop
# ---------------------------------------------------------------------------

def train(
    cfg: TrainConfig,
    norm: Optional[NormStats] = None,
    resume: bool = True,
) -> Path:
    """Train a model and return the path to the run directory."""
    torch.manual_seed(cfg.seed)

    run_dir = CKPT_DIR / f"{cfg.model_type}_{cfg.run_name}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2, default=str))

    device = pick_device(cfg.device)
    print(f"[train] device = {device}")

    # --- Data ---
    download_if_needed()  # no-op if already downloaded
    if norm is None:
        stats_path = DATA_DIR / "norm_stats.json"
        if stats_path.exists():
            d = json.loads(stats_path.read_text())
            norm = NormStats(mean=d["mean"], std=d["std"])
            print(f"[train] loaded norm: mean={norm.mean:.4f} std={norm.std:.4f}")
        else:
            print("[train] computing norm stats...")
            norm = compute_norm_stats()
            stats_path.write_text(json.dumps(norm.to_dict(), indent=2))
            print(f"[train] norm: mean={norm.mean:.4f} std={norm.std:.4f}")

    loader = make_dataloader(
        split="train",
        batch_size=cfg.batch_size,
        norm=norm,
        num_workers=cfg.num_workers,
        burn_in=cfg.burn_in,
    )
    print(f"[train] train frames: {len(loader.dataset)}  batches/epoch: {len(loader)}")

    # --- Model ---
    wrapper, net = build_model(cfg)
    wrapper.to(device)
    print(f"[train] params: {count_parameters(net) / 1e6:.2f}M")

    opt = torch.optim.AdamW(net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    ema = EMA(net, decay=cfg.ema_decay)

    # AMP: only enabled on CUDA
    use_amp = cfg.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # --- Resume? ---
    state_path = run_dir / "state.pt"
    step = 0
    loss_log = []
    if resume and state_path.exists():
        sd = torch.load(state_path, map_location=device, weights_only=False)
        net.load_state_dict(sd["net"])
        ema.load_state_dict(sd["ema"])
        opt.load_state_dict(sd["opt"])
        step = sd["step"]
        loss_log = sd.get("loss_log", [])
        print(f"[train] resumed from step {step}")

    # --- Training loop ---
    data_iter = iter(loader)
    t0 = time.time()
    running = 0.0
    n_running = 0

    while step < cfg.max_steps:
        try:
            x = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            x = next(data_iter)
        x = x.to(device, non_blocking=True)

        # LR schedule
        lr = cosine_lr(step, cfg.warmup_steps, cfg.max_steps, cfg.lr)
        for g in opt.param_groups:
            g["lr"] = lr

        opt.zero_grad(set_to_none=True)
        if use_amp:
            with torch.amp.autocast("cuda", dtype=torch.float16):
                loss = wrapper.loss(x)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(net.parameters(), cfg.grad_clip)
            scaler.step(opt)
            scaler.update()
        else:
            loss = wrapper.loss(x)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), cfg.grad_clip)
            opt.step()
        ema.update(net)

        step += 1
        running += loss.item()
        n_running += 1

        if step % cfg.log_every == 0:
            avg = running / n_running
            running, n_running = 0.0, 0
            dt = time.time() - t0
            rate = step / dt
            loss_log.append({"step": step, "loss": avg, "lr": lr})
            print(
                f"[{step:6d}/{cfg.max_steps}] loss={avg:.4f} lr={lr:.2e} "
                f"({rate:.1f} it/s)"
            )

        # Checkpoint
        if step % cfg.ckpt_every == 0 or step == cfg.max_steps:
            save_checkpoint(run_dir, step, net, ema, opt, loss_log, asdict(cfg))

        # Periodic samples (saved to disk for visual tracking)
        if step % cfg.sample_every == 0 or step == cfg.max_steps:
            save_preview(run_dir, step, wrapper, ema, cfg, norm)

    return run_dir


def save_checkpoint(run_dir, step, net, ema, opt, loss_log, cfg_dict):
    state_path = run_dir / "state.pt"
    tmp = run_dir / "state.pt.tmp"
    torch.save({
        "step": step,
        "net": net.state_dict(),
        "ema": ema.state_dict(),
        "opt": opt.state_dict(),
        "loss_log": loss_log,
        "config": cfg_dict,
    }, tmp)
    tmp.replace(state_path)

    # Separate EMA-only snapshot for easy evaluation later
    ema_path = run_dir / f"ema_step{step}.pt"
    torch.save({"ema": ema.state_dict(), "step": step, "config": cfg_dict}, ema_path)

    # Keep only the most recent 3 EMA snapshots (plus state.pt)
    emas = sorted(run_dir.glob("ema_step*.pt"), key=lambda p: p.stat().st_mtime)
    for old in emas[:-3]:
        old.unlink()


def save_preview(run_dir, step, wrapper, ema, cfg, norm):
    """Generate a small grid of samples with EMA weights."""
    import matplotlib.pyplot as plt

    device = next(wrapper.net.parameters()).device
    backup = ema.apply_to(wrapper.net)
    wrapper.eval()
    try:
        if cfg.model_type == "ddpm":
            samples = wrapper.ddim_sample(
                n=cfg.n_samples_preview, n_steps=50, img_size=cfg.img_size, device=device
            )
        else:
            samples = wrapper.sample(
                n=cfg.n_samples_preview, n_steps=50, img_size=cfg.img_size, device=device
            )
        samples = samples.cpu().numpy() * norm.std + norm.mean  # denormalize
    finally:
        ema.restore(wrapper.net, backup)
        wrapper.train()

    n = cfg.n_samples_preview
    fig, axes = plt.subplots(1, n, figsize=(3 * n, 3))
    if n == 1:
        axes = [axes]
    for ax, s in zip(axes, samples):
        vmax = max(abs(s.min()), abs(s.max()))
        ax.imshow(s[0], cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax.axis("off")
    fig.suptitle(f"{cfg.model_type} step {step}")
    fig.tight_layout()
    out = run_dir / f"samples_step{step}.png"
    fig.savefig(out, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"[train] saved preview → {out}")


if __name__ == "__main__":
    # Smoke test: tiny, fast, no data needed
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["ddpm", "fm"], default="ddpm")
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--run", default="smoke")
    args = p.parse_args()

    cfg = TrainConfig(
        model_type=args.model,
        run_name=args.run,
        max_steps=args.steps,
        log_every=10,
        sample_every=args.steps,
        ckpt_every=args.steps,
        batch_size=4,
        base_ch=16,
        ch_mults=(1, 2, 2),
        num_workers=0,
    )
    run_dir = train(cfg)
    print(f"Done: {run_dir}")
