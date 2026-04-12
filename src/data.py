"""Kolmogorov Flow dataset loading.

The raw file is `KolmFlow_valid_256.h5` from:
    https://huggingface.co/datasets/ayz2/temporal_pdes/tree/main/valid

Structure (per the project description):
    valid/u : [256, 200, 160, 160]   # [num_samples, num_timesteps, H, W]
    valid/x : [160]                  # spatial coords (unused here)
    valid/t : [200]                  # time stamps (unused here)

For *unconditional generation*, we treat each (trajectory_i, timestep_t) pair
as an i.i.d. sample. That gives 256 * 200 = 51,200 frames of shape [1, 160, 160].

Normalization is z-score using global mean/std computed over the training split.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from .env import DATA_DIR


HF_URL = (
    "https://huggingface.co/datasets/ayz2/temporal_pdes/"
    "resolve/main/valid/KolmFlow_valid_256.h5"
)
FILE_NAME = "KolmFlow_valid_256.h5"
DEFAULT_PATH = DATA_DIR / FILE_NAME

# Split: 256 trajectories → 230 train / 13 val / 13 test (~90/5/5 by trajectory)
# Splitting by trajectory (not by frame) prevents leakage: frames within the
# same trajectory are strongly correlated.
N_TRAJECTORIES = 256
N_TIMESTEPS = 200
N_TRAIN = 230
N_VAL = 13
# test = remainder

IMG_SIZE = 160


def download_if_needed(path: Path = DEFAULT_PATH, verbose: bool = True) -> Path:
    """Download the H5 file if not already present."""
    if path.exists():
        if verbose:
            size_gb = path.stat().st_size / 1e9
            print(f"[data] Already have {path} ({size_gb:.2f} GB)")
        return path

    import requests
    from tqdm.auto import tqdm

    path.parent.mkdir(parents=True, exist_ok=True)
    if verbose:
        print(f"[data] Downloading {HF_URL} → {path}")

    with requests.get(HF_URL, stream=True, timeout=60) as resp:
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))
        chunk = 1024 * 1024  # 1 MB
        with open(path, "wb") as f, tqdm(
            total=total, unit="B", unit_scale=True, desc=FILE_NAME, disable=not verbose
        ) as pbar:
            for data in resp.iter_content(chunk):
                f.write(data)
                pbar.update(len(data))
    return path


def get_trajectory_split() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Deterministic split of trajectory indices into train/val/test."""
    rng = np.random.default_rng(seed=0)
    perm = rng.permutation(N_TRAJECTORIES)
    train = np.sort(perm[:N_TRAIN])
    val = np.sort(perm[N_TRAIN : N_TRAIN + N_VAL])
    test = np.sort(perm[N_TRAIN + N_VAL :])
    return train, val, test


@dataclass
class NormStats:
    mean: float
    std: float

    def to_dict(self) -> dict:
        return {"mean": self.mean, "std": self.std}


class KolmFlowFrames(Dataset):
    """Per-frame view of Kolmogorov Flow for unconditional generation.

    Each __getitem__ returns a single vorticity frame of shape [1, 160, 160].

    Args:
        split: "train", "val", or "test".
        path: path to the H5 file. Defaults to DATA_DIR / FILE_NAME.
        norm: optional NormStats. If None, caller must normalize externally.
        burn_in: drop the first `burn_in` timesteps of each trajectory
            (transient phase before turbulence is fully developed).
            Default 0 keeps everything.
    """

    def __init__(
        self,
        split: str = "train",
        path: Optional[Path] = None,
        norm: Optional[NormStats] = None,
        burn_in: int = 0,
    ):
        if split not in {"train", "val", "test"}:
            raise ValueError(f"split must be train/val/test, got {split!r}")
        self.split = split
        self.path = Path(path or DEFAULT_PATH)
        self.norm = norm
        self.burn_in = burn_in

        if not self.path.exists():
            raise FileNotFoundError(
                f"{self.path} not found. Call `download_if_needed()` first."
            )

        train_idx, val_idx, test_idx = get_trajectory_split()
        self.traj_ids = {"train": train_idx, "val": val_idx, "test": test_idx}[split]
        self.n_per_traj = N_TIMESTEPS - burn_in
        self.length = len(self.traj_ids) * self.n_per_traj

        # Lazy file handle: opened per-worker on first access.
        self._h5: Optional[h5py.File] = None

    def _ensure_open(self) -> h5py.File:
        if self._h5 is None:
            self._h5 = h5py.File(self.path, "r")
        return self._h5

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> torch.Tensor:
        if idx < 0 or idx >= self.length:
            raise IndexError(idx)
        traj_local = idx // self.n_per_traj
        t_local = idx % self.n_per_traj
        traj_global = int(self.traj_ids[traj_local])
        t_global = t_local + self.burn_in

        h5 = self._ensure_open()
        frame = h5["valid"]["u"][traj_global, t_global]  # [160, 160]
        frame = np.asarray(frame, dtype=np.float32)

        if self.norm is not None:
            frame = (frame - self.norm.mean) / self.norm.std

        return torch.from_numpy(frame).unsqueeze(0)  # [1, 160, 160]

    def __del__(self):
        if self._h5 is not None:
            try:
                self._h5.close()
            except Exception:
                pass


def compute_norm_stats(
    path: Path = DEFAULT_PATH,
    train_ids: Optional[np.ndarray] = None,
    n_sample_traj: int = 30,
    rng_seed: int = 0,
) -> NormStats:
    """Estimate global mean/std from a random subset of training trajectories.

    Computing stats from all 230 * 200 = 46,000 frames is wasteful; a sample
    of ~30 trajectories (6000 frames) gives a stable estimate.
    """
    if train_ids is None:
        train_ids, _, _ = get_trajectory_split()
    rng = np.random.default_rng(rng_seed)
    sample_ids = rng.choice(train_ids, size=min(n_sample_traj, len(train_ids)), replace=False)
    sample_ids = np.sort(sample_ids)

    with h5py.File(path, "r") as h5:
        u = h5["valid"]["u"]
        chunks = [np.asarray(u[int(i)], dtype=np.float64) for i in sample_ids]
    arr = np.concatenate([c.reshape(-1) for c in chunks])
    return NormStats(mean=float(arr.mean()), std=float(arr.std()))


def make_dataloader(
    split: str,
    batch_size: int,
    norm: NormStats,
    num_workers: int = 2,
    shuffle: Optional[bool] = None,
    burn_in: int = 0,
    persistent_workers: bool = True,
) -> DataLoader:
    ds = KolmFlowFrames(split=split, norm=norm, burn_in=burn_in)
    if shuffle is None:
        shuffle = split == "train"
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=persistent_workers and num_workers > 0,
        drop_last=(split == "train"),
    )


if __name__ == "__main__":
    print(f"Default data path: {DEFAULT_PATH}")
    if DEFAULT_PATH.exists():
        print(f"File present ({DEFAULT_PATH.stat().st_size / 1e9:.2f} GB)")
        train, val, test = get_trajectory_split()
        print(f"Train trajectories: {len(train)}  Val: {len(val)}  Test: {len(test)}")
    else:
        print("File not present — call download_if_needed() to fetch it.")
