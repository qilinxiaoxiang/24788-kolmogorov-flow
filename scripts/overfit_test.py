"""Sanity-check: can DDPM and FM overfit a tiny fixed batch of synthetic
smooth 2D fields? If loss drops monotonically, the pipeline is wired correctly.

Run from repo root:
    PYTHONPATH=. python scripts/overfit_test.py
"""
import torch
import numpy as np
from src.unet import UNet
from src.diffusion import DDPM
from src.flow_matching import FlowMatching


def make_synthetic_batch(n: int = 8, size: int = 32, device="cpu") -> torch.Tensor:
    """Generate n smooth-looking 2D fields (sum of a few sinusoids).

    Normalized roughly to the regime a vorticity field would be in.
    """
    torch.manual_seed(42)
    b = []
    xs = torch.linspace(0, 2 * np.pi, size)
    X, Y = torch.meshgrid(xs, xs, indexing="ij")
    for i in range(n):
        kx1, ky1 = torch.randint(1, 5, (2,)).tolist()
        kx2, ky2 = torch.randint(1, 5, (2,)).tolist()
        phi1, phi2 = torch.rand(2) * 2 * np.pi
        field = torch.sin(kx1 * X + phi1) * torch.cos(ky1 * Y + phi1) + \
                0.5 * torch.sin(kx2 * X + phi2) * torch.cos(ky2 * Y + phi2)
        b.append(field)
    x = torch.stack(b).unsqueeze(1)  # [n, 1, H, W]
    x = (x - x.mean()) / x.std()
    return x.to(device)


def run_overfit(model_type: str, device: str, steps: int = 300):
    net = UNet(in_ch=1, out_ch=1, base_ch=16, ch_mults=(1, 2, 2), n_res_blocks=1, dropout=0.0)
    if model_type == "ddpm":
        wrapper = DDPM(net, T=100)
    else:
        wrapper = FlowMatching(net)
    wrapper.to(device)

    x = make_synthetic_batch(n=8, size=32, device=device)
    opt = torch.optim.AdamW(net.parameters(), lr=3e-4)

    losses = []
    for step in range(steps):
        opt.zero_grad(set_to_none=True)
        loss = wrapper.loss(x)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        losses.append(loss.item())

    first50 = sum(losses[:50]) / 50
    last50 = sum(losses[-50:]) / 50
    print(f"[{model_type}] first-50 avg loss = {first50:.4f}")
    print(f"[{model_type}] last-50  avg loss = {last50:.4f}")
    print(f"[{model_type}] ratio = {last50 / first50:.3f}  ({'PASS' if last50 < first50 * 0.7 else 'WARN'})")
    return losses


if __name__ == "__main__":
    device = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"device = {device}\n")

    print("=== DDPM overfit ===")
    run_overfit("ddpm", device)

    print("\n=== Flow Matching overfit ===")
    run_overfit("fm", device)
