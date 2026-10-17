# -*- coding: utf-8 -*-
"""小 U-Net：把 (纹理+边缘+透明+距离+深度) 映射成顶点热力图。"""
from __future__ import annotations

import torch
import torch.nn as nn


def _gn(c):
    return nn.GroupNorm(min(8, max(1, c // 8)) if c >= 8 else 1, c)


class DoubleConv(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False), _gn(cout), nn.SiLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False), _gn(cout), nn.SiLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UNet(nn.Module):
    """4 层下采样 U-Net，输出单通道热力图 logits。"""

    def __init__(self, in_ch=9, base=32, depth=4):
        super().__init__()
        chs = [base * (2 ** i) for i in range(depth + 1)]        # 32,64,128,256,512
        self.stem = DoubleConv(in_ch, chs[0])
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        for i in range(depth):
            self.downs.append(nn.Sequential(nn.MaxPool2d(2), DoubleConv(chs[i], chs[i + 1])))
        for i in range(depth - 1, -1, -1):
            self.ups.append(nn.Sequential(
                nn.ConvTranspose2d(chs[i + 1], chs[i], 2, stride=2),
                DoubleConv(chs[i] * 2, chs[i]),
            ))
        self.head = nn.Conv2d(chs[0], 1, 1)

    def forward(self, x):
        skips = []
        x = self.stem(x)
        skips.append(x)
        for d in self.downs:
            x = d(x)
            skips.append(x)
        skips = skips[:-1]
        for u in self.ups:
            s = skips.pop()
            x = u[0](x)
            if x.shape[-2:] != s.shape[-2:]:
                x = nn.functional.interpolate(x, size=s.shape[-2:], mode="nearest")
            x = u[1](torch.cat([x, s], dim=1))
        return self.head(x)


def build(in_ch=9, base=32, depth=4):
    return UNet(in_ch=in_ch, base=base, depth=depth)
