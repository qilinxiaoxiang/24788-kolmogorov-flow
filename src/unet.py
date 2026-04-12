"""Shared U-Net backbone for both DDPM (noise prediction) and Flow Matching
(velocity prediction).

Design choices:
- 4 resolution levels: 160 -> 80 -> 40 -> 20 -> 10
- Channel multipliers: [1, 2, 2, 4] x base_ch=64 → (64, 128, 128, 256)
- Group-normalized residual blocks, SiLU activations
- Sinusoidal time embedding → 2-layer MLP → per-block FiLM-style additive bias
- Single attention block at the bottleneck (20x20 is cheap)

This architecture is standard for small-scale 2D diffusion (e.g., CIFAR-10,
Kolmogorov Flow). Using the *same* backbone for both models isolates the
contribution of the training objective (noise prediction vs velocity
prediction).
"""
from __future__ import annotations

import math
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Time embedding
# ---------------------------------------------------------------------------

class SinusoidalTimeEmbedding(nn.Module):
    """Classic transformer-style positional embedding for scalar t in [0, 1]."""

    def __init__(self, dim: int, max_period: float = 10_000.0):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f"dim must be even, got {dim}")
        self.dim = dim
        self.max_period = max_period

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: [B] or [B, 1]
        if t.dim() == 0:
            t = t[None]
        if t.dim() == 2:
            t = t.squeeze(-1)
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(self.max_period)
            * torch.arange(half, device=t.device, dtype=t.dtype) / half
        )
        args = t[:, None] * freqs[None]
        return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class ResBlock(nn.Module):
    """GroupNorm → SiLU → Conv → (+ time bias) → GroupNorm → SiLU → Dropout → Conv."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        t_dim: int,
        groups: int = 8,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.norm1 = nn.GroupNorm(min(groups, in_ch), in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.t_proj = nn.Linear(t_dim, out_ch)
        self.norm2 = nn.GroupNorm(min(groups, out_ch), out_ch)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.t_proj(F.silu(t_emb))[:, :, None, None]
        h = self.conv2(self.dropout(F.silu(self.norm2(h))))
        return h + self.skip(x)


class SelfAttention2D(nn.Module):
    """Single-head self-attention on a 2D feature map, flattened to tokens."""

    def __init__(self, ch: int, groups: int = 8):
        super().__init__()
        self.norm = nn.GroupNorm(min(groups, ch), ch)
        self.qkv = nn.Conv2d(ch, 3 * ch, 1)
        self.proj = nn.Conv2d(ch, ch, 1)
        self.ch = ch

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        qkv = self.qkv(self.norm(x)).reshape(b, 3, c, h * w)
        q, k, v = qkv.unbind(1)  # each [B, C, HW]
        scale = c ** -0.5
        attn = torch.softmax((q.transpose(-2, -1) @ k) * scale, dim=-1)  # [B, HW, HW]
        out = (v @ attn.transpose(-2, -1)).reshape(b, c, h, w)
        return x + self.proj(out)


class Down(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.op = nn.Conv2d(ch, ch, 3, stride=2, padding=1)

    def forward(self, x):
        return self.op(x)


class Up(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.op = nn.ConvTranspose2d(ch, ch, 4, stride=2, padding=1)

    def forward(self, x):
        return self.op(x)


# ---------------------------------------------------------------------------
# U-Net
# ---------------------------------------------------------------------------

class UNet(nn.Module):
    """Lightweight U-Net for 2D generation with scalar time conditioning.

    Works identically for DDPM (input=noisy x, output=predicted noise) and
    Flow Matching (input=interpolated x, output=predicted velocity).
    """

    def __init__(
        self,
        in_ch: int = 1,
        out_ch: int = 1,
        base_ch: int = 64,
        ch_mults: Sequence[int] = (1, 2, 2, 4),
        n_res_blocks: int = 2,
        t_dim: int = 256,
        dropout: float = 0.1,
        attn_at_bottleneck: bool = True,
    ):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.base_ch = base_ch

        # Time embedding MLP
        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(base_ch),
            nn.Linear(base_ch, t_dim),
            nn.SiLU(),
            nn.Linear(t_dim, t_dim),
        )

        # Stem
        self.stem = nn.Conv2d(in_ch, base_ch, 3, padding=1)

        # Build encoder
        self.downs = nn.ModuleList()
        self.skip_chs = [base_ch]
        ch = base_ch
        for i, mult in enumerate(ch_mults):
            out = base_ch * mult
            for _ in range(n_res_blocks):
                self.downs.append(ResBlock(ch, out, t_dim, dropout=dropout))
                ch = out
                self.skip_chs.append(ch)
            if i != len(ch_mults) - 1:
                self.downs.append(Down(ch))
                self.skip_chs.append(ch)

        # Bottleneck
        self.mid1 = ResBlock(ch, ch, t_dim, dropout=dropout)
        self.mid_attn = SelfAttention2D(ch) if attn_at_bottleneck else nn.Identity()
        self.mid2 = ResBlock(ch, ch, t_dim, dropout=dropout)

        # Decoder: mirror encoder, concatenating skip connections
        self.ups = nn.ModuleList()
        skip_chs = list(self.skip_chs)
        for i, mult in reversed(list(enumerate(ch_mults))):
            out = base_ch * mult
            for _ in range(n_res_blocks + 1):
                self.ups.append(ResBlock(ch + skip_chs.pop(), out, t_dim, dropout=dropout))
                ch = out
            if i != 0:
                self.ups.append(Up(ch))

        # Output head
        self.out_norm = nn.GroupNorm(min(8, ch), ch)
        self.out_conv = nn.Conv2d(ch, out_ch, 3, padding=1)
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_embed(t)

        h = self.stem(x)
        skips = [h]
        for m in self.downs:
            if isinstance(m, ResBlock):
                h = m(h, t_emb)
                skips.append(h)
            else:  # Down
                h = m(h)
                skips.append(h)

        h = self.mid1(h, t_emb)
        if isinstance(self.mid_attn, SelfAttention2D):
            h = self.mid_attn(h)
        h = self.mid2(h, t_emb)

        for m in self.ups:
            if isinstance(m, ResBlock):
                s = skips.pop()
                # Guard: spatial dims must match. If they don't (shouldn't happen
                # for standard sizes but sanity-check), interpolate.
                if s.shape[-2:] != h.shape[-2:]:
                    h = F.interpolate(h, size=s.shape[-2:], mode="nearest")
                h = m(torch.cat([h, s], dim=1), t_emb)
            else:  # Up
                h = m(h)

        return self.out_conv(F.silu(self.out_norm(h)))


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    net = UNet(base_ch=64)
    n_params = count_parameters(net)
    print(f"Params: {n_params / 1e6:.2f}M")
    x = torch.randn(2, 1, 160, 160)
    t = torch.rand(2)
    y = net(x, t)
    print(f"Input:  {tuple(x.shape)}")
    print(f"Output: {tuple(y.shape)}")
    assert x.shape == y.shape
    print("Shape check: OK")
