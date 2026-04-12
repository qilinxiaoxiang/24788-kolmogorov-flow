"""Conditional Flow Matching (Lipman et al., ICLR 2023) — the variant.

Core idea: instead of denoising noisy data, train a neural network to predict
the *velocity field* that transports a simple source distribution (isotropic
Gaussian) to the data distribution along straight-line paths.

Paths:
    x_t = (1 - t) * x_0 + t * x_1
    where x_0 ~ N(0, I), x_1 ~ data, t ~ Uniform[0, 1].

Target velocity (the constant derivative of the straight path):
    u_t(x | x_0, x_1) = x_1 - x_0

Loss (flow matching):
    L(θ) = E_{t, x_0, x_1} || v_θ(x_t, t) - (x_1 - x_0) ||^2

Sampling: integrate the learned ODE from t=0 (noise) to t=1 (data).

Why this should differ from DDPM on this task:
- Straight paths (vs curved diffusion paths) → easier to regress, faster
  convergence, and fewer steps needed at inference.
- No stochastic noise during sampling → deterministic, reproducible samples.
- Decoupled source distribution: you could in principle warm-start from a
  non-Gaussian prior (we stick with Gaussian for a fair DDPM comparison).
"""
from __future__ import annotations

from typing import Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class FlowMatching(nn.Module):
    """Wrap a UNet with Conditional Flow Matching train/sample logic."""

    def __init__(self, net: nn.Module, sigma_min: float = 0.0):
        """
        Args:
            net: UNet that maps (x, t) → velocity, with t ∈ [0, 1].
            sigma_min: optional non-zero minimum noise at t=1 (Lipman §4.1).
                Default 0.0 = pure linear interpolation to the clean data.
        """
        super().__init__()
        self.net = net
        self.sigma_min = sigma_min

    # ---- Training ----
    def loss(self, x1: torch.Tensor) -> torch.Tensor:
        """CFM loss: MSE between predicted velocity and (x1 - x0)."""
        b = x1.shape[0]
        device = x1.device
        x0 = torch.randn_like(x1)
        # t ~ U[0, 1]
        t = torch.rand(b, device=device)
        # Interpolation path
        t_b = t.view(-1, 1, 1, 1)
        # With sigma_min > 0: x_t = (1 - (1 - σ_min) t) x_0 + t x_1
        coef0 = 1.0 - (1.0 - self.sigma_min) * t_b
        xt = coef0 * x0 + t_b * x1
        target = x1 - (1.0 - self.sigma_min) * x0
        v_pred = self.net(xt, t)
        return F.mse_loss(v_pred, target)

    # ---- Sampling via ODE integration ----
    @torch.no_grad()
    def sample(
        self,
        n: int,
        img_size: int = 160,
        n_steps: int = 50,
        method: Literal["euler", "rk4", "heun"] = "euler",
        device: Optional[torch.device] = None,
        progress: bool = False,
        return_trajectory: bool = False,
    ) -> torch.Tensor:
        """Integrate dx/dt = v_θ(x, t) from t=0 to t=1."""
        device = device or next(self.net.parameters()).device
        x = torch.randn(n, self.net.in_ch, img_size, img_size, device=device)

        dt = 1.0 / n_steps
        ts = torch.linspace(0.0, 1.0, n_steps + 1, device=device)
        traj = [x.detach().cpu()] if return_trajectory else None

        iterator = range(n_steps)
        if progress:
            from tqdm.auto import tqdm
            iterator = tqdm(list(iterator), desc=f"FM {method} ({n_steps} steps)")

        for i in iterator:
            t = ts[i]
            t_b = t.expand(n)
            if method == "euler":
                v = self.net(x, t_b)
                x = x + dt * v
            elif method == "heun":
                v1 = self.net(x, t_b)
                x_pred = x + dt * v1
                t_next = ts[i + 1].expand(n)
                v2 = self.net(x_pred, t_next)
                x = x + 0.5 * dt * (v1 + v2)
            elif method == "rk4":
                t_b = t.expand(n)
                k1 = self.net(x, t_b)
                k2 = self.net(x + 0.5 * dt * k1, (t + 0.5 * dt).expand(n))
                k3 = self.net(x + 0.5 * dt * k2, (t + 0.5 * dt).expand(n))
                k4 = self.net(x + dt * k3, (t + dt).expand(n))
                x = x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            else:
                raise ValueError(f"unknown method {method!r}")

            if return_trajectory:
                traj.append(x.detach().cpu())

        if return_trajectory:
            return x, traj
        return x


if __name__ == "__main__":
    from src.unet import UNet
    net = UNet(base_ch=16, ch_mults=(1, 2, 2))  # tiny
    fm = FlowMatching(net)
    x1 = torch.randn(2, 1, 32, 32)
    loss = fm.loss(x1)
    print(f"Loss: {loss.item():.4f}")
    samples = fm.sample(n=2, img_size=32, n_steps=10)
    print(f"Euler samples: {tuple(samples.shape)}")
    samples = fm.sample(n=2, img_size=32, n_steps=10, method="rk4")
    print(f"RK4 samples:   {tuple(samples.shape)}")
    print("Flow Matching smoke test: OK")
