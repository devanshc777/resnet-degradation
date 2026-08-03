"""Why did the first identity_test run produce a flat plain curve and NaNs for residual?

Two hypotheses:
  H1 undertrained. lr=0.01 / 1500 steps is not enough to converge even at depth 2,
     so every depth reads as "bad" and the curve looks flat. Test: sweep lr and steps
     at depth 2 and see whether MSE goes to ~0.
  H2 residual activations explode. x = relu(W x + x) compounds magnitude with depth,
     and with no normalisation it overflows past ~depth 16. Test: measure the forward
     activation norm by depth at init, no training.

Baseline to beat: predicting the input mean gives MSE = Var(U(0,1)) = 1/12 = 0.0833.
Anything at or above that has learned nothing.
"""

import torch
import torch.nn as nn

from models import build_mlp

DIM = 64
VAR_BASELINE = 1 / 12


def h1_convergence():
    print("H1: is it undertrained? (depth 2, plain)")
    print(f"    baseline MSE from predicting the mean = {VAR_BASELINE:.4f}")
    print(f"{'lr':>8} {'steps':>8} {'final MSE':>12}")
    print("-" * 32)
    for lr in (0.01, 0.05, 0.1, 0.5):
        for steps in (1500, 6000):
            torch.manual_seed(0)
            model = build_mlp("plain", DIM, 2)
            opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
            loss_fn = nn.MSELoss()
            for _ in range(steps):
                x = torch.rand(256, DIM) + 0.1
                loss = loss_fn(model(x), x)
                opt.zero_grad()
                loss.backward()
                opt.step()
            with torch.no_grad():
                x = torch.rand(4096, DIM) + 0.1
                mse = loss_fn(model(x), x).item()
            flag = "  <- learned nothing" if mse >= VAR_BASELINE else ""
            print(f"{lr:>8} {steps:>8} {mse:>12.6f}{flag}")


def h2_activation_growth():
    print("\nH2: do residual activations explode with depth? (at init, no training)")
    print(f"{'depth':>6} {'plain |out|':>14} {'residual |out|':>16}")
    print("-" * 40)
    for depth in (2, 8, 16, 24, 32, 48):
        torch.manual_seed(0)
        x = torch.rand(256, DIM) + 0.1
        norms = {}
        for arch in ("plain", "resnet"):
            torch.manual_seed(0)
            model = build_mlp(arch, DIM, depth)
            with torch.no_grad():
                norms[arch] = model(x).abs().mean().item()
        print(f"{depth:>6} {norms['plain']:>14.4e} {norms['resnet']:>16.4e}")


if __name__ == "__main__":
    h1_convergence()
    h2_activation_growth()
