"""Three more figures that substantiate claims made in the README.

  1 activation growth by depth, at init, no training
       -> substantiates WHY normalisation was needed and what caused the depth-24 NaNs
  2 gradient norm reaching the FIRST layer, at init
       -> substantiates the optimisation story: does signal get to the bottom of the stack
  3 per-seed spread of the identity results
       -> substantiates that the headline curves are not one lucky seed

1 and 2 need no training at all (forward/backward at initialisation, seconds).
3 reads the existing result JSONs.

  uv run python extra_figures.py
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from models import build_mlp

RESULTS = Path(__file__).parent / "results"
DIM = 64
BASELINE = 1 / 12

COLORS = {"plain": "#c1440e", "resnet": "#2e63d9", "preact": "#1a7f5a"}
LABELS = {
    "plain": "plain stack",
    "resnet": "residual, ReLU after the add",
    "preact": "residual, pre-activation (clean path)",
}
DEPTHS = [2, 4, 8, 16, 24, 32, 48]


def style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(True, alpha=0.15, linewidth=0.8)
    ax.set_axisbelow(True)


def fig_activation_growth():
    """Forward pass only, at initialisation, normalisation DISABLED — this is the
    configuration that produced NaNs, so the figure shows the actual cause."""
    out = {a: [] for a in COLORS}
    for depth in DEPTHS:
        for arch in COLORS:
            torch.manual_seed(0)
            x = torch.rand(256, DIM) + 0.1
            torch.manual_seed(0)
            model = build_mlp(arch, DIM, depth, norm=False)
            model.eval()
            with torch.no_grad():
                out[arch].append(model(x).abs().mean().item())

    fig, ax = plt.subplots(figsize=(8.5, 5))
    for arch, ys in out.items():
        ax.plot(DEPTHS, ys, "o-", color=COLORS[arch], lw=2, label=LABELS[arch])
    ax.set_yscale("log")
    ax.set_xlabel("depth (layers)")
    ax.set_ylabel("mean |activation| at output   (log scale)")
    ax.set_title("Why normalisation was needed: residual sums compound with depth\n"
                 "forward pass at initialisation, no normalisation, no training",
                 fontsize=12, loc="left")
    ax.legend(frameon=False)
    style(ax)
    fig.tight_layout()
    p = RESULTS / "activation_growth.png"
    fig.savefig(p, dpi=200)
    print(f"wrote {p}")
    return out


def fig_gradient_reach():
    """Gradient norm on the FIRST layer's weights after one backward pass at init.
    Normalisation ON, matching the configuration the headline results were run in."""
    out = {a: [] for a in COLORS}
    loss_fn = nn.MSELoss()
    for depth in DEPTHS:
        for arch in COLORS:
            torch.manual_seed(0)
            model = build_mlp(arch, DIM, depth, norm=True)
            model.train()
            x = torch.rand(256, DIM) + 0.1
            loss = loss_fn(model(x), x)
            model.zero_grad()
            loss.backward()
            out[arch].append(model.layers[0].weight.grad.norm().item())

    fig, ax = plt.subplots(figsize=(8.5, 5))
    for arch, ys in out.items():
        ax.plot(DEPTHS, ys, "o-", color=COLORS[arch], lw=2, label=LABELS[arch])
    ax.set_yscale("log")
    ax.set_xlabel("depth (layers)")
    ax.set_ylabel("gradient norm at the FIRST layer   (log scale)")
    # Careful with the reading: bigger is NOT better here. Flat with depth is the good
    # property. The plain stack's gradients EXPLODE with depth rather than vanish, which is
    # consistent with He et al. explicitly ruling out vanishing gradients as the cause of
    # degradation (BatchNorm already handles that).
    ax.set_title("Gradient magnitude at the first layer, by depth\n"
                 "flat is good. plain does not vanish, it explodes.",
                 fontsize=12, loc="left")
    ax.text(0.99, 0.02, "one backward pass at initialisation, normalisation on",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=8, color="#666")
    ax.legend(frameon=False, loc="upper left")
    style(ax)
    fig.tight_layout()
    p = RESULTS / "gradient_reach.png"
    fig.savefig(p, dpi=200)
    print(f"wrote {p}")
    return out


def fig_seed_spread():
    """Every individual seed, so the headline means are visibly not one lucky run."""
    data = json.loads((RESULTS / "identity_test.json").read_text())
    runs = {"plain": data["runs"], "resnet": data["runs"]}
    pre = RESULTS / "identity_preact.json"
    if pre.exists():
        runs["preact"] = json.loads(pre.read_text())["runs"]

    fig, ax = plt.subplots(figsize=(8.5, 5))
    ax.axhline(BASELINE, color="#999", ls=":", lw=1.5)
    ax.text(48, BASELINE * 1.1, "learned nothing", ha="right", va="bottom",
            fontsize=8, color="#777")
    for arch, rows in runs.items():
        depths = [r["depth"] for r in rows]
        means = [r[arch]["mean"] for r in rows]
        ax.plot(depths, means, "-", color=COLORS[arch], lw=2, label=LABELS[arch], zorder=2)
        for r in rows:
            for v in r[arch]["all"]:
                ax.plot(r["depth"], v, "o", color=COLORS[arch], ms=4, alpha=0.45, zorder=3)
    ax.set_yscale("log")
    ax.set_xlabel("depth (layers)")
    ax.set_ylabel("final MSE learning f(x) = x   (log scale)")
    ax.set_title("Same result, every seed shown\n"
                 "lines are means, dots are the 3 individual seeds",
                 fontsize=12, loc="left")
    ax.legend(frameon=False, loc="center right")
    style(ax)
    fig.tight_layout()
    p = RESULTS / "seed_spread.png"
    fig.savefig(p, dpi=200)
    print(f"wrote {p}")


if __name__ == "__main__":
    growth = fig_activation_growth()
    grads = fig_gradient_reach()
    fig_seed_spread()

    print("\nactivation |out| at init, no norm:")
    for i, d in enumerate(DEPTHS):
        print(f"  depth {d:>3}  " + "  ".join(f"{a}={growth[a][i]:.3e}" for a in COLORS))
    print("\ngradient norm at first layer, norm on:")
    for i, d in enumerate(DEPTHS):
        print(f"  depth {d:>3}  " + "  ".join(f"{a}={grads[a][i]:.3e}" for a in COLORS))
