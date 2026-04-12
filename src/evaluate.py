"""Evaluation metrics for generated Kolmogorov Flow fields.

For a generative model, we care about whether the *distribution* of generated
fields matches the data distribution — not pixel-level reconstruction.
Metrics used:

1. **Radial energy spectrum** E(k): classic diagnostic for turbulence.
   Kolmogorov Flow should exhibit a characteristic energy cascade. A good
   generator reproduces E(k) across wavenumbers; a bad one gets the wrong
   slope or cutoff.

2. **Vorticity PDF**: histogram of pixel values across many samples. Captures
   whether the marginal statistics (tails, symmetry) are preserved.

3. **Wasserstein-1 distance** between generated and real vorticity PDFs.
   Single scalar summary of (2).

4. **Power spectrum L1 / L2 error** in log space: single scalar summary of (1).

5. **Sampling efficiency sweep**: quality metrics as a function of the number
   of function evaluations (NFE). Directly reveals the DDPM-vs-FM tradeoff.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Energy spectrum (radially averaged)
# ---------------------------------------------------------------------------

def radial_energy_spectrum(
    fields: np.ndarray,
    return_k: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute radially averaged power spectrum of a batch of 2D fields.

    Args:
        fields: [N, H, W] numpy array. For vorticity ω, E(k) = 0.5 |ω̂(k)|^2.
    Returns:
        k : [H//2] wavenumber bins (integer, starting at 1)
        E : [H//2] radially averaged spectrum, averaged across the batch
    """
    if fields.ndim != 3:
        raise ValueError(f"expected [N, H, W], got shape {fields.shape}")
    N, H, W = fields.shape
    assert H == W, "expected square fields"

    # 2D FFT
    fhat = np.fft.fftn(fields, axes=(-2, -1))
    fhat = np.fft.fftshift(fhat, axes=(-2, -1))
    power = 0.5 * (np.abs(fhat) / (H * W)) ** 2  # [N, H, W]

    # Wavenumber magnitude map
    kx = np.fft.fftshift(np.fft.fftfreq(H, d=1.0 / H))
    ky = np.fft.fftshift(np.fft.fftfreq(W, d=1.0 / W))
    KX, KY = np.meshgrid(kx, ky, indexing="ij")
    k_mag = np.sqrt(KX ** 2 + KY ** 2)

    # Radial binning
    k_bins = np.arange(1, H // 2 + 1)
    E = np.zeros_like(k_bins, dtype=np.float64)
    for i, k in enumerate(k_bins):
        mask = (k_mag >= k - 0.5) & (k_mag < k + 0.5)
        if mask.any():
            E[i] = power[:, mask].mean()  # average across samples AND within shell

    if return_k:
        return k_bins, E
    return E


def log_spectrum_error(
    E_ref: np.ndarray,
    E_gen: np.ndarray,
    eps: float = 1e-12,
    norm: str = "l1",
) -> float:
    """L1 or L2 distance between log-spectra (excludes zero bins)."""
    log_ref = np.log10(E_ref + eps)
    log_gen = np.log10(E_gen + eps)
    diff = log_gen - log_ref
    if norm == "l1":
        return float(np.mean(np.abs(diff)))
    elif norm == "l2":
        return float(np.sqrt(np.mean(diff ** 2)))
    raise ValueError(norm)


# ---------------------------------------------------------------------------
# Vorticity PDF + Wasserstein-1
# ---------------------------------------------------------------------------

def vorticity_histogram(
    fields: np.ndarray,
    bins: int = 100,
    value_range: Optional[tuple[float, float]] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return bin_edges, density for the pixel-value PDF across all samples."""
    arr = fields.reshape(-1)
    if value_range is None:
        m = max(abs(arr.min()), abs(arr.max()))
        value_range = (-m, m)
    hist, edges = np.histogram(arr, bins=bins, range=value_range, density=True)
    return edges, hist


def wasserstein1(x: np.ndarray, y: np.ndarray) -> float:
    """1D Wasserstein distance between two samples (CDF-based).

    Equivalent to scipy.stats.wasserstein_distance but vectorized so we don't
    need scipy as a hard dep.
    """
    x = np.sort(x.reshape(-1))
    y = np.sort(y.reshape(-1))
    # Resample to the same size if needed
    n = min(len(x), len(y))
    if len(x) > n:
        idx = np.linspace(0, len(x) - 1, n).astype(int)
        x = x[idx]
    if len(y) > n:
        idx = np.linspace(0, len(y) - 1, n).astype(int)
        y = y[idx]
    return float(np.mean(np.abs(x - y)))


# ---------------------------------------------------------------------------
# Sampling + evaluation convenience
# ---------------------------------------------------------------------------

@dataclass
class EvalResults:
    n_samples: int
    nfe: int
    log_spectrum_l1: float
    wasserstein_vorticity: float
    real_std: float
    gen_std: float


def evaluate_against_real(
    generated: np.ndarray,
    real: np.ndarray,
    bins: int = 100,
) -> EvalResults:
    """Compare generated and real fields with scalar summary metrics."""
    k, E_real = radial_energy_spectrum(real)
    _, E_gen = radial_energy_spectrum(generated)
    spec_l1 = log_spectrum_error(E_real, E_gen, norm="l1")
    w1 = wasserstein1(generated, real)
    return EvalResults(
        n_samples=len(generated),
        nfe=-1,  # filled in by caller
        log_spectrum_l1=spec_l1,
        wasserstein_vorticity=w1,
        real_std=float(real.std()),
        gen_std=float(generated.std()),
    )


@torch.no_grad()
def generate_samples(
    wrapper: torch.nn.Module,
    n: int,
    img_size: int,
    nfe: int,
    device: torch.device,
    model_type: str,
) -> tuple[np.ndarray, int]:
    """Draw n samples from a trained model. Returns (samples_numpy, effective_nfe).

    effective_nfe reported for fair DDPM vs FM comparison (for FM with RK4,
    each step is 4 net evaluations, etc.)
    """
    wrapper.eval()
    batches = []
    remaining = n
    batch_size = 32
    effective_nfe = nfe
    while remaining > 0:
        b = min(batch_size, remaining)
        if model_type == "ddpm":
            if nfe >= wrapper.T:
                x = wrapper.sample(b, img_size=img_size, device=device)
                effective_nfe = wrapper.T
            else:
                x = wrapper.ddim_sample(b, n_steps=nfe, img_size=img_size, device=device)
        elif model_type == "fm":
            x = wrapper.sample(b, img_size=img_size, n_steps=nfe, method="euler", device=device)
        else:
            raise ValueError(model_type)
        batches.append(x.float().cpu().numpy())
        remaining -= b
    return np.concatenate(batches, axis=0).squeeze(1), effective_nfe


if __name__ == "__main__":
    # Smoke test on synthetic data
    np.random.seed(0)
    real = np.random.randn(64, 32, 32)
    gen = real + 0.1 * np.random.randn(*real.shape)
    k, E_real = radial_energy_spectrum(real)
    _, E_gen = radial_energy_spectrum(gen)
    print(f"k bins: {len(k)}, E_real[0]={E_real[0]:.4e}")
    res = evaluate_against_real(gen, real)
    print(f"log-spectrum L1 : {res.log_spectrum_l1:.4f}")
    print(f"Wasserstein-1   : {res.wasserstein_vorticity:.4f}")
    print("evaluate smoke test: OK")
