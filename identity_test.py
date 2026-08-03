"""Mechanism test: can SGD find an identity mapping through a deep plain stack?

He et al. argue degradation is surprising because a deeper network can always match a
shallower one by setting its extra layers to identity. The capacity is provably there.
This script tests whether the optimiser can actually find it, sweeping depth for a
plain stack and a residual stack with identical parameter counts.

Run:  uv run python identity_test.py
"""

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn

from models import build_mlp

RESULTS = Path(__file__).parent / "results"


MEAN_BASELINE = 1 / 12  # MSE from just predicting the mean of U(0,1). Anything above = learned nothing.


def run_one(arch, dim, depth, steps, lr, seed, device, norm=True):
    torch.manual_seed(seed)
    model = build_mlp(arch, dim, depth, norm).to(device)
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    loss_fn = nn.MSELoss()

    # strictly positive inputs so ReLU is transparent and identity is representable
    # by both architectures. See the note in models.py.
    for _ in range(steps):
        x = torch.rand(256, dim, device=device) + 0.1
        loss = loss_fn(model(x), x)
        opt.zero_grad()
        loss.backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        x = torch.rand(4096, dim, device=device) + 0.1
        final = loss_fn(model(x), x).item()
    return final


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dim", type=int, default=64)
    p.add_argument("--depths", type=int, nargs="+", default=[2, 4, 8, 16, 24, 32])
    # lr 0.01 / 1500 steps failed to converge at any depth (diagnose.py H1) — the whole
    # curve read as "bad" and looked flat. These defaults clear the mean baseline.
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--no-norm", action="store_true",
                   help="disable BatchNorm, reproducing the divergence seen in the first run")
    p.add_argument("--archs", nargs="+", default=["plain", "resnet"],
                   choices=["plain", "resnet", "preact"])
    p.add_argument("--out", default="identity_test.json")
    args = p.parse_args()

    device = "cpu"
    torch.set_num_threads(torch.get_num_threads())
    RESULTS.mkdir(exist_ok=True)

    out = {"config": vars(args), "runs": []}
    print(f"identity mapping test  dim={args.dim}  steps={args.steps}  seeds={args.seeds}")
    print(f"  (MSE >= {MEAN_BASELINE:.4f} means the net learned nothing at all)")
    header = f"{'depth':>6}" + "".join(f"{a:>12}" for a in args.archs)
    print(header)
    print("-" * len(header))

    for depth in args.depths:
        row = {}
        for arch in args.archs:
            losses = [run_one(arch, args.dim, depth, args.steps, args.lr, s, device,
                              norm=not args.no_norm)
                      for s in args.seeds]
            mean = sum(losses) / len(losses)
            row[arch] = {"mean": mean, "all": losses, "beat_baseline": mean < MEAN_BASELINE}
        dead = [a for a in args.archs if not row[a]["beat_baseline"]]
        flag = f"   learned nothing: {', '.join(dead)}" if dead else ""
        # flush: without this, nothing reaches a redirected log until the process exits,
        # which makes a long run look hung when it is fine.
        print(f"{depth:>6}" + "".join(f"{row[a]['mean']:>12.6f}" for a in args.archs) + flag,
              flush=True)
        entry = {"depth": depth, **row}
        if "plain" in row and "resnet" in row:
            entry["ratio"] = row["plain"]["mean"] / max(row["resnet"]["mean"], 1e-12)
        out["runs"].append(entry)

        # Checkpoint after EVERY depth. The previous 2-hour run was killed mid-flight and
        # produced nothing because results were only written at the end. Never again.
        (RESULTS / args.out).write_text(json.dumps(out, indent=2))

    path = RESULTS / args.out
    path.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {path}")


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"took {time.time() - t0:.0f}s")
