"""Colab training launcher v2 — preloads all data into RAM.

Root cause of slow training in v1: h5py random access on Drive (even local SSD
copy) was ~1.5 it/s. GPU utilisation was 0% between batches.

Fix: load all 46000 training frames into a single contiguous float32 tensor
(~4.7 GB, fits in Colab's 12 GB RAM) before training starts. Dataset becomes
an O(1) tensor-indexing op.

Usage:
    python scripts/colab_train_v2.py ddpm
    python scripts/colab_train_v2.py fm
"""
from __future__ import annotations

import gc
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

_HERE = Path(__file__).resolve()
_REPO = _HERE.parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
os.chdir(_REPO)

from src.data import get_trajectory_split, NormStats, N_TIMESTEPS  # noqa: E402


LOCAL_H5 = Path("/content/data.h5")
CACHE_NPY = Path("/content/train_frames.npy")


def preload_to_npy() -> tuple[np.ndarray, NormStats]:
    """Materialize training frames into a single numpy array + norm stats.

    Cached as /content/train_frames.npy so re-runs are instant.
    """
    import h5py

    train_ids, _, _ = get_trajectory_split()

    if CACHE_NPY.exists():
        print(f"[preload] loading cached {CACHE_NPY} ...")
        arr = np.load(CACHE_NPY)
    else:
        print(f"[preload] reading {LOCAL_H5} ...")
        t0 = time.time()
        with h5py.File(LOCAL_H5, "r") as f:
            u = f["valid"]["u"]
            # Select trajectories, take first N_TIMESTEPS per trajectory
            chunks = []
            for i, tid in enumerate(train_ids):
                chunks.append(np.asarray(u[int(tid), :N_TIMESTEPS], dtype=np.float32))
                if i % 20 == 0:
                    print(f"  [{i+1}/{len(train_ids)}] ...")
            arr = np.stack(chunks, axis=0)
        arr = arr.reshape(-1, arr.shape[-2], arr.shape[-1])  # [N_frames, 160, 160]
        print(f"[preload] read in {time.time()-t0:.1f}s, shape={arr.shape}, dtype={arr.dtype}")
        print(f"[preload] saving cache → {CACHE_NPY}")
        np.save(CACHE_NPY, arr)

    # Norm stats from a subset for speed
    rng = np.random.default_rng(0)
    idx = rng.choice(arr.shape[0], size=6000, replace=False)
    sample = arr[idx].astype(np.float64)
    norm = NormStats(mean=float(sample.mean()), std=float(sample.std()))
    print(f"[preload] norm: mean={norm.mean:.4f} std={norm.std:.4f}")
    print(f"[preload] ram used for frames: {arr.nbytes / 1e9:.2f} GB")
    return arr, norm


class RamFrames(Dataset):
    """Zero-overhead per-frame dataset on top of a pre-loaded numpy array."""

    def __init__(self, arr: np.ndarray, norm: NormStats):
        # Keep as numpy for lowest memory; convert per sample on the fly.
        self.arr = arr
        self.mean = norm.mean
        self.std = norm.std

    def __len__(self):
        return self.arr.shape[0]

    def __getitem__(self, idx):
        x = self.arr[idx]  # [160, 160]
        x = (x - self.mean) / self.std
        return torch.from_numpy(x).unsqueeze(0).contiguous()  # [1, 160, 160]


def main(model_type: str) -> None:
    if model_type not in ("ddpm", "fm"):
        raise SystemExit(f"model_type must be ddpm or fm, got {model_type!r}")
    if not LOCAL_H5.exists():
        raise SystemExit(f"{LOCAL_H5} not found")

    arr, norm = preload_to_npy()

    # Build RAM-based DataLoader
    ds = RamFrames(arr, norm)

    # Build training pieces
    from src.unet import UNet, count_parameters
    from src.diffusion import DDPM
    from src.flow_matching import FlowMatching
    from src.train import TrainConfig, EMA, cosine_lr, save_checkpoint, save_preview

    cfg = TrainConfig(
        model_type=model_type,
        run_name="main",
        max_steps=10_000,
        batch_size=128,            # back up to 32 now that data loading is free
        base_ch=48,               # fits A100/L4 easily, works on T4 too
        ch_mults=(1, 2, 2, 4),
        n_res_blocks=2,
        dropout=0.1,
        num_workers=4,            # RAM-backed loader scales well
        log_every=100,
        sample_every=5_000,
        ckpt_every=1_000,
        amp=True,
    )

    loader = DataLoader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        persistent_workers=cfg.num_workers > 0,
        drop_last=True,
    )
    print(f"[launcher] train frames: {len(ds)}  batches/epoch: {len(loader)}")

    net = UNet(
        in_ch=1, out_ch=1, base_ch=cfg.base_ch, ch_mults=cfg.ch_mults,
        n_res_blocks=cfg.n_res_blocks, dropout=cfg.dropout, t_dim=cfg.t_dim,
    )
    if cfg.model_type == "ddpm":
        wrapper = DDPM(net, T=cfg.T)
    else:
        wrapper = FlowMatching(net, sigma_min=cfg.sigma_min)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    wrapper.to(device)
    print(f"[launcher] device={device} params={count_parameters(net)/1e6:.2f}M")

    opt = torch.optim.AdamW(net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    ema = EMA(net, decay=cfg.ema_decay)

    # Setup run dir (within repo, on Drive)
    from src.env import CKPT_DIR
    run_dir = CKPT_DIR / f"{cfg.model_type}_{cfg.run_name}"
    run_dir.mkdir(parents=True, exist_ok=True)
    import json
    from dataclasses import asdict
    (run_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2, default=str))

    # Resume if possible
    state_path = run_dir / "state.pt"
    step = 0
    loss_log: list = []
    if state_path.exists():
        sd = torch.load(state_path, map_location=device, weights_only=False)
        net.load_state_dict(sd["net"])
        ema.load_state_dict(sd["ema"])
        opt.load_state_dict(sd["opt"])
        step = sd["step"]
        loss_log = sd.get("loss_log", [])
        print(f"[launcher] resumed from step {step}")

    use_amp = cfg.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    t0 = time.time()
    running, n_running = 0.0, 0
    data_iter = iter(loader)
    while step < cfg.max_steps:
        try:
            x = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            x = next(data_iter)
        x = x.to(device, non_blocking=True)

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
            print(f"[{step:6d}/{cfg.max_steps}] loss={avg:.4f} lr={lr:.2e} ({rate:.1f} it/s)", flush=True)

        if step % cfg.ckpt_every == 0 or step == cfg.max_steps:
            save_checkpoint(run_dir, step, net, ema, opt, loss_log, asdict(cfg))

        if step % cfg.sample_every == 0 or step == cfg.max_steps:
            save_preview(run_dir, step, wrapper, ema, cfg, norm)

    print(f"[launcher] done. run_dir = {run_dir}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python scripts/colab_train_v2.py {ddpm|fm}", file=sys.stderr)
        sys.exit(1)
    main(sys.argv[1])
