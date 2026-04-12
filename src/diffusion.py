"""DDPM (Ho et al., NeurIPS 2020) — baseline generative model.

We train the model to predict noise ε given a noisy sample x_t and scalar
timestep t. Implementation notes:

- Continuous-time API externally (t ∈ [0, 1]) for symmetry with Flow Matching,
  but internally discretized to T=1000 steps with a cosine β-schedule
  (Nichol & Dhariwal, 2021) which handles low-variance data like vorticity
  fields better than the original linear schedule.
- Training loss: simplified ε-prediction MSE (x_0 prediction is equivalent).
- Sampling: ancestral DDPM sampler. DDIM deterministic sampler is also
  provided for comparing generation quality at reduced NFE.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Noise schedule
# ---------------------------------------------------------------------------

def cosine_alpha_bar(T: int, s: float = 0.008) -> torch.Tensor:
    """Cosine schedule of \bar{α}_t from Nichol & Dhariwal (2021)."""
    steps = torch.arange(T + 1, dtype=torch.float64) / T
    f = torch.cos((steps + s) / (1 + s) * math.pi / 2) ** 2
    alpha_bar = f / f[0]
    return alpha_bar.to(torch.float32)  # length T+1


@dataclass
class Schedule:
    T: int
    alpha_bar: torch.Tensor   # [T+1]
    alphas: torch.Tensor      # [T]    α_t = ᾱ_t / ᾱ_{t-1}
    betas: torch.Tensor       # [T]    β_t = 1 - α_t
    sqrt_alpha_bar: torch.Tensor      # [T+1]
    sqrt_one_minus_alpha_bar: torch.Tensor  # [T+1]

    @classmethod
    def cosine(cls, T: int = 1000) -> "Schedule":
        ab = cosine_alpha_bar(T)
        alphas = (ab[1:] / ab[:-1]).clamp(min=1e-8, max=1.0)
        betas = (1.0 - alphas).clamp(min=0.0, max=0.999)
        return cls(
            T=T,
            alpha_bar=ab,
            alphas=alphas,
            betas=betas,
            sqrt_alpha_bar=ab.sqrt(),
            sqrt_one_minus_alpha_bar=(1.0 - ab).clamp(min=0.0).sqrt(),
        )

    def to(self, device: torch.device) -> "Schedule":
        return Schedule(
            T=self.T,
            alpha_bar=self.alpha_bar.to(device),
            alphas=self.alphas.to(device),
            betas=self.betas.to(device),
            sqrt_alpha_bar=self.sqrt_alpha_bar.to(device),
            sqrt_one_minus_alpha_bar=self.sqrt_one_minus_alpha_bar.to(device),
        )


# ---------------------------------------------------------------------------
# DDPM wrapper
# ---------------------------------------------------------------------------

class DDPM(nn.Module):
    """Wrap a UNet with DDPM training/sampling logic.

    The UNet receives a continuous t ∈ [0, 1] (t / T) so its time embedding
    is shared with the Flow Matching model.
    """

    def __init__(self, net: nn.Module, T: int = 1000):
        super().__init__()
        self.net = net
        self.T = T
        self._schedule = Schedule.cosine(T)

    @property
    def schedule(self) -> Schedule:
        # Device-aligned lazily
        dev = next(self.net.parameters()).device
        if self._schedule.alpha_bar.device != dev:
            self._schedule = self._schedule.to(dev)
        return self._schedule

    # ---- Forward diffusion ----
    def q_sample(
        self,
        x0: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample x_t ~ q(x_t | x_0). t is integer in [1, T]."""
        if noise is None:
            noise = torch.randn_like(x0)
        s = self.schedule
        sab = s.sqrt_alpha_bar[t].view(-1, 1, 1, 1)
        somab = s.sqrt_one_minus_alpha_bar[t].view(-1, 1, 1, 1)
        xt = sab * x0 + somab * noise
        return xt, noise

    # ---- Training objective ----
    def loss(self, x0: torch.Tensor) -> torch.Tensor:
        """Simplified ε-prediction MSE loss."""
        b = x0.shape[0]
        # Sample integer t uniformly in [1, T]
        t = torch.randint(1, self.T + 1, (b,), device=x0.device)
        xt, noise = self.q_sample(x0, t)
        # Feed net a continuous time in [0, 1]
        t_cont = t.float() / self.T
        eps_pred = self.net(xt, t_cont)
        return F.mse_loss(eps_pred, noise)

    # ---- Ancestral sampling ----
    @torch.no_grad()
    def sample(
        self,
        n: int,
        img_size: int = 160,
        device: Optional[torch.device] = None,
        clip: Optional[float] = None,
        progress: bool = False,
    ) -> torch.Tensor:
        device = device or next(self.net.parameters()).device
        s = self.schedule
        x = torch.randn(n, self.net.in_ch, img_size, img_size, device=device)

        iterator = reversed(range(1, self.T + 1))
        if progress:
            from tqdm.auto import tqdm
            iterator = tqdm(list(iterator), desc="DDPM sampling")

        for t in iterator:
            t_tensor = torch.full((n,), t, device=device, dtype=torch.long)
            t_cont = t_tensor.float() / self.T
            eps = self.net(x, t_cont)

            alpha_t = s.alphas[t - 1]
            alpha_bar_t = s.alpha_bar[t]
            sqrt_one_minus = s.sqrt_one_minus_alpha_bar[t]
            # x_{t-1} mean
            mean = (1.0 / alpha_t.sqrt()) * (x - (s.betas[t - 1] / sqrt_one_minus) * eps)
            if t > 1:
                noise = torch.randn_like(x)
                # Use the "smaller" variance \tilde{β}_t
                alpha_bar_prev = s.alpha_bar[t - 1]
                var = s.betas[t - 1] * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar_t)
                x = mean + var.clamp(min=0.0).sqrt() * noise
            else:
                x = mean

            if clip is not None:
                x = x.clamp(-clip, clip)

        return x

    # ---- DDIM deterministic sampling (reduced NFE) ----
    @torch.no_grad()
    def ddim_sample(
        self,
        n: int,
        n_steps: int = 50,
        img_size: int = 160,
        device: Optional[torch.device] = None,
        eta: float = 0.0,
        clip: Optional[float] = None,
        progress: bool = False,
    ) -> torch.Tensor:
        """DDIM sampler (Song et al., 2021). eta=0 is deterministic."""
        device = device or next(self.net.parameters()).device
        s = self.schedule
        # Pick n_steps time indices evenly from [1, T]
        ts = torch.linspace(self.T, 1, n_steps, device=device).long()

        x = torch.randn(n, self.net.in_ch, img_size, img_size, device=device)
        iterator = range(n_steps)
        if progress:
            from tqdm.auto import tqdm
            iterator = tqdm(list(iterator), desc=f"DDIM ({n_steps} steps)")

        for i in iterator:
            t = ts[i].item()
            t_prev = ts[i + 1].item() if i + 1 < n_steps else 0

            t_tensor = torch.full((n,), t, device=device, dtype=torch.long)
            t_cont = t_tensor.float() / self.T
            eps = self.net(x, t_cont)

            ab_t = s.alpha_bar[t]
            ab_prev = s.alpha_bar[t_prev] if t_prev > 0 else torch.tensor(1.0, device=device)

            # x_0 prediction
            x0_pred = (x - (1 - ab_t).sqrt() * eps) / ab_t.sqrt()
            # Direction term
            sigma = eta * ((1 - ab_prev) / (1 - ab_t) * (1 - ab_t / ab_prev)).clamp(min=0.0).sqrt()
            dir_xt = (1 - ab_prev - sigma ** 2).clamp(min=0.0).sqrt() * eps
            x = ab_prev.sqrt() * x0_pred + dir_xt
            if eta > 0:
                x = x + sigma * torch.randn_like(x)

            if clip is not None:
                x = x.clamp(-clip, clip)

        return x


if __name__ == "__main__":
    from src.unet import UNet
    net = UNet(base_ch=16, ch_mults=(1, 2, 2))  # tiny for smoke test
    ddpm = DDPM(net, T=50)
    x0 = torch.randn(2, 1, 32, 32)
    loss = ddpm.loss(x0)
    print(f"Loss: {loss.item():.4f}")
    samples = ddpm.sample(n=2, img_size=32)
    print(f"Samples: {tuple(samples.shape)}")
    ddim = ddpm.ddim_sample(n=2, n_steps=10, img_size=32)
    print(f"DDIM samples: {tuple(ddim.shape)}")
    print("DDPM smoke test: OK")
