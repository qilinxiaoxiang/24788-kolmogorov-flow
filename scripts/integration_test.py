"""End-to-end integration test using synthetic data.

Verifies that the full train.py path works:
- dataset loading
- model building
- training step
- EMA update
- checkpoint save (state.pt + ema_stepN.pt)
- preview sample generation
- resume from checkpoint

Uses synthetic data so it doesn't require the 5 GB H5 file. Runs in ~1 minute
on MPS/CPU.
"""
import os, sys, shutil, json
from pathlib import Path

import torch
import numpy as np

# Monkey-patch data loading to use synthetic fields before importing train
os.environ["PYTHONPATH"] = str(Path(__file__).resolve().parents[1])


class _SyntheticDS(torch.utils.data.Dataset):
    def __init__(self, n=128, size=32):
        torch.manual_seed(0)
        xs = torch.linspace(0, 2 * np.pi, size)
        X, Y = torch.meshgrid(xs, xs, indexing="ij")
        self.data = torch.stack([
            torch.sin((i % 5 + 1) * X) * torch.cos((i % 4 + 1) * Y)
            for i in range(n)
        ]).unsqueeze(1).float()
        # normalize
        self.data = (self.data - self.data.mean()) / self.data.std()

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.data[idx]


def main():
    from src import data as data_mod
    from src import train as train_mod
    from src.data import NormStats
    from src.train import TrainConfig, train, pick_device

    # Patch loader to skip download + use synthetic.
    # IMPORTANT: we must patch the names in `src.train` (since train.py did
    # `from .data import make_dataloader`), not in src.data.
    def fake_loader(split, batch_size, norm, num_workers=0, shuffle=None, burn_in=0, persistent_workers=False):
        ds = _SyntheticDS(n=64, size=32)
        return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=(split == "train"), num_workers=0)

    train_mod.make_dataloader = fake_loader
    train_mod.download_if_needed = lambda *a, **k: Path("synthetic")
    train_mod.compute_norm_stats = lambda *a, **k: NormStats(mean=0.0, std=1.0)
    # Keep module-level patch too, for completeness
    data_mod.make_dataloader = fake_loader
    data_mod.download_if_needed = lambda *a, **k: Path("synthetic")
    data_mod.compute_norm_stats = lambda *a, **k: NormStats(mean=0.0, std=1.0)

    # Clean previous integration runs
    from src.env import CKPT_DIR, DATA_DIR
    for mt in ("ddpm", "fm"):
        p = CKPT_DIR / f"{mt}_integration"
        if p.exists():
            shutil.rmtree(p)

    # Save a fake norm_stats.json so train.py doesn't try to compute
    (DATA_DIR / "norm_stats.json").write_text(json.dumps({"mean": 0.0, "std": 1.0}))

    for mt in ("ddpm", "fm"):
        print(f"\n=== {mt} integration run ===")
        cfg = TrainConfig(
            model_type=mt,
            run_name="integration",
            base_ch=16,
            ch_mults=(1, 2, 2),
            n_res_blocks=1,
            dropout=0.0,
            batch_size=8,
            num_workers=0,
            max_steps=40,
            warmup_steps=5,
            log_every=10,
            sample_every=20,
            ckpt_every=20,
            n_samples_preview=2,
            img_size=32,
            T=100,
        )
        run_dir = train(cfg)

        # Verify artifacts
        assert (run_dir / "state.pt").exists(), "state.pt missing"
        assert (run_dir / "config.json").exists(), "config.json missing"
        ema_files = list(run_dir.glob("ema_step*.pt"))
        assert len(ema_files) >= 1, f"no EMA snapshots in {run_dir}"
        sample_files = list(run_dir.glob("samples_step*.png"))
        assert len(sample_files) >= 1, f"no preview samples in {run_dir}"
        print(f"[{mt}] artifacts OK: {[p.name for p in run_dir.iterdir() if p.is_file()]}")

        # Verify resume: run 20 more steps, loss log should grow
        cfg2 = TrainConfig(**{**cfg.__dict__, "max_steps": 60})
        run_dir2 = train(cfg2)
        state = torch.load(run_dir2 / "state.pt", map_location="cpu", weights_only=False)
        assert state["step"] == 60, f"expected step=60, got {state['step']}"
        print(f"[{mt}] resume OK, final step = {state['step']}, log length = {len(state['loss_log'])}")

    print("\n=== Integration test: PASS ===")


if __name__ == "__main__":
    main()
