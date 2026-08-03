"""Train one CIFAR-10 model and log per-epoch train/test error.

Follows He et al. 2015 section 4.2 except for the schedule length, which is
truncated to fit a CPU budget. The truncation is recorded in the output JSON so
any chart made from it can state it honestly.

Paper setup:      45k/5k split, batch 128, SGD momentum 0.9, weight decay 1e-4,
                  lr 0.1 divided by 10 at 32k and 48k of 64k iterations (~182 epochs),
                  augmentation = 4px pad + random 32x32 crop + random horizontal flip.
This script:      same, but --epochs (default 30) with the lr drops rescaled to
                  50% and 75% of the run.

Run:  uv run python train.py --arch plain  --depth 20 --epochs 30
      uv run python train.py --arch resnet --depth 56 --epochs 30
"""

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader, random_split

from keepawake import keep_awake
from models import build

ROOT = Path(__file__).parent
RESULTS = ROOT / "results"
DATA = ROOT / "data"

PAPER_EPOCHS = 182  # 64k iterations at batch 128 over 45k images


def loaders(batch_size, workers):
    train_tf = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    eval_tf = T.Compose([
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])

    full = torchvision.datasets.CIFAR10(DATA, train=True, download=True, transform=train_tf)
    # paper holds out 5k for validation and trains on 45k
    train_set, _ = random_split(full, [45000, 5000], generator=torch.Generator().manual_seed(0))
    test_set = torchvision.datasets.CIFAR10(DATA, train=False, download=True, transform=eval_tf)

    # a second view of the training split without augmentation, for honest training error
    clean = torchvision.datasets.CIFAR10(DATA, train=True, download=False, transform=eval_tf)
    train_clean, _ = random_split(clean, [45000, 5000], generator=torch.Generator().manual_seed(0))

    kw = dict(batch_size=batch_size, num_workers=workers, persistent_workers=workers > 0)
    return (
        DataLoader(train_set, shuffle=True, drop_last=False, **kw),
        DataLoader(train_clean, shuffle=False, **kw),
        DataLoader(test_set, shuffle=False, **kw),
    )


@torch.no_grad()
def error_rate(model, loader, device):
    model.eval()
    wrong = total = 0
    for x, y in loader:
        pred = model(x.to(device)).argmax(1).cpu()
        wrong += (pred != y).sum().item()
        total += y.numel()
    return 100.0 * wrong / total


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arch", choices=["plain", "resnet"], required=True)
    p.add_argument("--depth", type=int, required=True)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = "cpu"
    RESULTS.mkdir(exist_ok=True)
    tag = f"{args.arch}{args.depth}"

    train_loader, train_clean_loader, test_loader = loaders(args.batch_size, args.workers)
    model = build(args.arch, args.depth).to(device)
    n_params = sum(p.numel() for p in model.parameters())

    opt = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
    # paper drops lr at 32k/64k and 48k/64k of training; rescale to this run's length
    milestones = [int(args.epochs * 0.5), int(args.epochs * 0.75)]
    sched = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=milestones, gamma=0.1)
    loss_fn = nn.CrossEntropyLoss()

    log = {
        "arch": args.arch,
        "depth": args.depth,
        "params": n_params,
        "config": vars(args),
        "schedule_note": (
            f"{args.epochs} epochs, truncated from the paper's ~{PAPER_EPOCHS} "
            f"(64k iterations). lr drops rescaled to epochs {milestones}."
        ),
        "epochs": [],
    }
    print(f"{tag}  params={n_params:,}  epochs={args.epochs}  lr drops at {milestones}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        running = 0.0
        for x, y in train_loader:
            loss = loss_fn(model(x.to(device)), y.to(device))
            opt.zero_grad()
            loss.backward()
            opt.step()
            running += loss.item() * y.numel()
        sched.step()

        train_err = error_rate(model, train_clean_loader, device)
        test_err = error_rate(model, test_loader, device)
        row = {
            "epoch": epoch,
            "train_loss": running / 45000,
            "train_error": train_err,
            "test_error": test_err,
            "lr": opt.param_groups[0]["lr"],
            "seconds": time.time() - t0,
        }
        log["epochs"].append(row)
        print(f"  epoch {epoch:>3}  train_err {train_err:6.2f}%  test_err {test_err:6.2f}%  "
              f"loss {row['train_loss']:.4f}  {row['seconds']:.0f}s")

        # write after every epoch so a killed run still leaves usable data
        (RESULTS / f"{tag}.json").write_text(json.dumps(log, indent=2))

    print(f"done. wrote {RESULTS / f'{tag}.json'}")


if __name__ == "__main__":
    with keep_awake("cifar training"):
        main()
