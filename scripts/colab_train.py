"""Colab training launcher. Runs full 40k-step training for DDPM or FM.

Usage (in a Colab cell):
    !python scripts/colab_train.py ddpm
    !python scripts/colab_train.py fm

Assumes:
  - `/content/data.h5` exists (copy from Drive before running)
  - Working directory is the repo root
  - CUDA T4 GPU

Uses base_ch=48, batch_size=16 to fit T4's 15 GB VRAM.
Resumes automatically if the run_name's state.pt already exists.
"""
from __future__ import annotations

import os
import sys
import gc
from pathlib import Path

# Make `src` importable whether we `cd` into repo or not
_HERE = Path(__file__).resolve()
_REPO = _HERE.parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
os.chdir(_REPO)

import torch


def main(model_type: str) -> None:
    if model_type not in ("ddpm", "fm"):
        raise SystemExit(f"model_type must be ddpm or fm, got {model_type!r}")

    # Patch data paths to point at local SSD copy
    from src import data as D, train as T
    local_h5 = Path("/content/data.h5")
    if not local_h5.exists():
        raise SystemExit(
            f"Expected {local_h5} to exist. Copy it from Drive first:\n"
            "  !cp \"/content/drive/Othercomputers/My MacBook Pro/courses/"
            "24788-Intro_of_DL/project/code/data/KolmFlow_valid_256.h5\" "
            "/content/data.h5"
        )
    D.DEFAULT_PATH = local_h5
    T.DEFAULT_PATH = local_h5

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    cfg = T.TrainConfig(
        model_type=model_type,
        run_name="main",
        max_steps=40_000,
        batch_size=16,
        base_ch=48,              # reduced from 64 to fit T4
        ch_mults=(1, 2, 2, 4),
        n_res_blocks=2,
        dropout=0.1,
        num_workers=2,
        log_every=100,
        sample_every=5_000,
        ckpt_every=2_000,
        amp=True,
    )
    print(f"[launcher] starting {model_type.upper()} training")
    print(f"[launcher] device={('cuda' if torch.cuda.is_available() else 'cpu')}")
    print(f"[launcher] config = {cfg}")

    run_dir = T.train(cfg)
    print(f"\n[launcher] done. run_dir = {run_dir}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python scripts/colab_train.py {ddpm|fm}", file=sys.stderr)
        sys.exit(1)
    main(sys.argv[1])
