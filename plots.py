"""Charts from the JSON logs.

  uv run python plots.py identity   -> results/identity_test.png
  uv run python plots.py cifar      -> results/degradation.png, results/final_errors.png
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS = Path(__file__).parent / "results"

PLAIN = "#c1440e"
RESNET = "#2e63d9"
PREACT = "#1a7f5a"
BASELINE = 1 / 12  # MSE from predicting the input mean; at or above this, nothing was learned


def style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(True, alpha=0.15, linewidth=0.8)
    ax.set_axisbelow(True)


def identity():
    data = json.loads((RESULTS / "identity_test.json").read_text())
    depths = [r["depth"] for r in data["runs"]]
    series = [
        ("plain stack", [r["plain"]["mean"] for r in data["runs"]], PLAIN),
        ("residual, ReLU after the add", [r["resnet"]["mean"] for r in data["runs"]], RESNET),
    ]

    # preact was run separately at an identical config, so it merges legitimately
    pre_path = RESULTS / "identity_preact.json"
    if pre_path.exists():
        pre = json.loads(pre_path.read_text())
        assert [r["depth"] for r in pre["runs"]] == depths, "depth grids differ, refusing to merge"
        for k in ("dim", "steps", "lr", "seeds"):
            assert pre["config"][k] == data["config"][k], f"config mismatch on {k}, refusing to merge"
        series.append(("residual, pre-activation (clean path)",
                       [r["preact"]["mean"] for r in pre["runs"]], PREACT))

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.axhline(BASELINE, color="#999", ls=":", lw=1.5)
    ax.text(depths[-1], BASELINE * 1.12, "predicting the input mean — learned nothing",
            ha="right", va="bottom", fontsize=8, color="#777")
    for label, ys, color in series:
        ax.plot(depths, ys, "o-", color=color, lw=2, label=label)
    ax.set_yscale("log")
    ax.set_xlabel("depth (layers)")
    ax.set_ylabel("final MSE learning f(x) = x   (log scale)")
    ax.set_title("Learning the identity mapping\n"
                 "same layer count, same parameter count. only where the shortcut sits differs.",
                 fontsize=12, loc="left")
    cfg = data["config"]
    ax.text(0.99, 0.02,
            f"dim={cfg['dim']}, {cfg['steps']} SGD steps, {len(cfg['seeds'])} seeds, positive inputs",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=8, color="#666")
    ax.legend(frameon=False, loc="center right")
    style(ax)
    fig.tight_layout()
    out = RESULTS / "identity_test.png"
    fig.savefig(out, dpi=200)
    print(f"wrote {out}")


def _load(tag):
    path = RESULTS / f"{tag}.json"
    return json.loads(path.read_text()) if path.exists() else None


def cifar():
    tags = ["plain20", "plain56", "resnet20", "resnet56"]
    runs = {t: _load(t) for t in tags}
    have = {t: r for t, r in runs.items() if r}
    if not have:
        sys.exit("no CIFAR logs in results/. run train.py first.")
    missing = [t for t in tags if t not in have]
    if missing:
        print(f"warning: missing {missing} — plotting what exists")

    styles = {
        "plain20": dict(color=PLAIN, ls="--", label="plain-20"),
        "plain56": dict(color=PLAIN, ls="-", label="plain-56"),
        "resnet20": dict(color=RESNET, ls="--", label="ResNet-20"),
        "resnet56": dict(color=RESNET, ls="-", label="ResNet-56"),
    }

    # chart 1: the degradation curves
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for key, ax_title in (("train_error", "training error"), ("test_error", "test error")):
        ax = axes[0] if key == "train_error" else axes[1]
        for tag, run in have.items():
            xs = [e["epoch"] for e in run["epochs"]]
            ys = [e[key] for e in run["epochs"]]
            ax.plot(xs, ys, lw=2, **styles[tag])
        ax.set_xlabel("epoch")
        ax.set_title(ax_title, fontsize=11, loc="left")
        style(ax)
    axes[0].set_ylabel("error (%)")
    axes[0].legend(frameon=False)
    any_run = next(iter(have.values()))
    fig.suptitle(
        "Degradation: deeper plain nets train worse. Residual nets do not.\n"
        f"CIFAR-10, {any_run['config']['epochs']} epochs (paper used ~182)",
        fontsize=12, x=0.01, ha="left")
    fig.tight_layout()
    out = RESULTS / "degradation.png"
    fig.savefig(out, dpi=200)
    print(f"wrote {out}")

    # chart 2: final errors, train vs test side by side
    fig, ax = plt.subplots(figsize=(8, 5))
    labels, tr, te = [], [], []
    for tag in tags:
        if tag not in have:
            continue
        last = have[tag]["epochs"][-1]
        labels.append(styles[tag]["label"])
        tr.append(last["train_error"])
        te.append(last["test_error"])
    x = range(len(labels))
    ax.bar([i - 0.2 for i in x], tr, width=0.4, color="#888", label="train error")
    ax.bar([i + 0.2 for i in x], te, width=0.4, color=RESNET, label="test error")
    for i, (a, b) in enumerate(zip(tr, te)):
        ax.text(i - 0.2, a + 0.4, f"{a:.1f}", ha="center", fontsize=8)
        ax.text(i + 0.2, b + 0.4, f"{b:.1f}", ha="center", fontsize=8)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylabel("error (%)")
    ax.set_title("Final error. Deeper plain net is worse on TRAINING error,\n"
                 "which rules out overfitting as the cause.", fontsize=12, loc="left")
    ax.legend(frameon=False)
    style(ax)
    fig.tight_layout()
    out = RESULTS / "final_errors.png"
    fig.savefig(out, dpi=200)
    print(f"wrote {out}")


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "identity"
    {"identity": identity, "cifar": cifar}[which]()
