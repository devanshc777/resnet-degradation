"""Plain and residual networks, matching the CIFAR-10 architecture in
He et al. 2015, Deep Residual Learning for Image Recognition (arXiv:1512.03385) section 4.2.

Both families are identical except for the shortcut connection, which is the
whole point: any performance gap is attributable to the shortcut alone.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------
# CIFAR-10 networks (section 4.2)
#
#   first layer   3x3 conv, 16 filters
#   then          6n layers, 3x3 convs, 2n each on feature maps 32/16/8
#                 with 16/32/64 filters, subsampling by stride 2
#   finally       global average pool + 10-way fully connected
#   total depth   6n + 2   ->  n=3 gives 20 layers, n=9 gives 56
# --------------------------------------------------------------------------

class PlainBlock(nn.Module):
    """Two 3x3 convs. No shortcut."""

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        return out


class ResidualBlock(nn.Module):
    """Two 3x3 convs, plus the identity shortcut. The only difference from PlainBlock.

    When the shape changes we use the paper's CIFAR option A: identity mapping with
    stride-2 subsampling and zero-padded channels. Parameter-free, so the residual
    nets have the same parameter count as the plain nets and the comparison stays honest.
    """

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.stride = stride
        self.in_planes = in_planes
        self.planes = planes

    def shortcut(self, x):
        if self.stride == 1 and self.in_planes == self.planes:
            return x
        # option A: subsample, then zero-pad the new channels
        x = x[:, :, ::self.stride, ::self.stride]
        pad = self.planes - self.in_planes
        return F.pad(x, (0, 0, 0, 0, pad // 2, pad - pad // 2))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)   # the one line the paper is about
        return F.relu(out)


class CifarNet(nn.Module):
    def __init__(self, depth, residual, num_classes=10):
        super().__init__()
        assert (depth - 2) % 6 == 0, "depth must be 6n+2 (20, 32, 44, 56, ...)"
        n = (depth - 2) // 6
        block = ResidualBlock if residual else PlainBlock

        self.conv1 = nn.Conv2d(3, 16, 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)

        layers, in_planes = [], 16
        for planes, stride in ((16, 1), (32, 2), (64, 2)):
            for i in range(n):
                layers.append(block(in_planes, planes, stride if i == 0 else 1))
                in_planes = planes
        self.blocks = nn.Sequential(*layers)
        self.fc = nn.Linear(64, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.blocks(out)
        out = F.adaptive_avg_pool2d(out, 1).flatten(1)
        return self.fc(out)


def build(arch, depth):
    """arch is 'plain' or 'resnet'."""
    return CifarNet(depth, residual=(arch == "resnet"))


# --------------------------------------------------------------------------
# Identity-mapping networks (the mechanism test)
#
# The paper's argument for why degradation is surprising: a deeper net can always
# match a shallower one by making the extra layers identity mappings, so the
# capacity is provably there and only the optimizer is at fault.
#
# These nets test exactly that claim in isolation. Learn f(x) = x, sweep depth.
#
# Deliberate control: inputs are strictly positive, so ReLU is transparent and a
# plain stack CAN represent the identity exactly (weights = I). That removes the
# representational confound and leaves a pure optimisation question. With
# zero-centred inputs a plain ReLU stack cannot represent identity at all, which
# would be a different and weaker experiment.
# --------------------------------------------------------------------------

# Normalisation is ON by default, added after the first run diverged to NaN at depth 24+.
# Measured cause (see diagnose.py, H2): mean activation magnitude at init grew 0.68 -> 131
# from depth 2 to 48 in the residual stack, because x = relu(Wx + x) compounds with depth.
# The paper's blocks carry BatchNorm, so adding it to BOTH families is the faithful fix and
# keeps the comparison fair.
#
# REJECTED FIX, on purpose: initialising the residual branch to zero. That is a real and
# common trick, and it would have made the residual net start at exactly the identity, i.e.
# start at the answer this experiment is asking it to find. It would have produced a
# beautiful chart that proves nothing. Both families keep random init.

class PlainMLP(nn.Module):
    def __init__(self, dim, depth, norm=True):
        super().__init__()
        self.layers = nn.ModuleList(nn.Linear(dim, dim) for _ in range(depth))
        self.norms = nn.ModuleList(
            (nn.BatchNorm1d(dim) if norm else nn.Identity()) for _ in range(depth)
        )

    def forward(self, x):
        for layer, norm in zip(self.layers, self.norms):
            x = F.relu(norm(layer(x)))
        return x


class ResidualMLP(nn.Module):
    """Same layers, same parameter count, same normalisation. Only the addition differs."""

    def __init__(self, dim, depth, norm=True):
        super().__init__()
        self.layers = nn.ModuleList(nn.Linear(dim, dim) for _ in range(depth))
        self.norms = nn.ModuleList(
            (nn.BatchNorm1d(dim) if norm else nn.Identity()) for _ in range(depth)
        )

    def forward(self, x):
        for layer, norm in zip(self.layers, self.norms):
            x = F.relu(norm(layer(x)) + x)
        return x


class PreActResidualMLP(nn.Module):
    """Pre-activation variant, testing the hypothesis in the README.

    ResidualMLP computes relu(norm(Wx) + x), so a nonlinearity sits ON the shortcut path in
    every block. Here the norm and relu move BEFORE the weight layer and nothing is applied
    after the addition:

        x = x + W(relu(norm(x)))

    so the path from input to output is a clean sum of residuals with no activation in the way.
    This is the ordering from He et al. 2016, Identity Mappings in Deep Residual Networks
    (arXiv:1603.05027), which is what let them train 1001 layers.

    Same layer count and same parameter count as the other two families.
    """

    def __init__(self, dim, depth, norm=True):
        super().__init__()
        self.layers = nn.ModuleList(nn.Linear(dim, dim) for _ in range(depth))
        self.norms = nn.ModuleList(
            (nn.BatchNorm1d(dim) if norm else nn.Identity()) for _ in range(depth)
        )

    def forward(self, x):
        for layer, norm in zip(self.layers, self.norms):
            x = x + layer(F.relu(norm(x)))
        return x


_MLPS = {"plain": PlainMLP, "resnet": ResidualMLP, "preact": PreActResidualMLP}


def build_mlp(arch, dim, depth, norm=True):
    return _MLPS[arch](dim, depth, norm)
