# -*- coding: utf-8 -*-
"""「学建模师风格」的数据管道（基于预缩放缓存）。

缓存由 _tmp/build_cache.py 生成：画布 RGBA png + 画布 深度 png + 顶点 npz。
画布固定 CANVAS x CANVAS，保长宽比居中留边。

输入通道（10 个）:
    0-2  RGB
    3    边缘强度（Sobel 幅值，按 p99 归一）
    4    内部纹理线掩码 line_mask
    5    alpha
    6    到轮廓距离 / DIST_NORM（截断 1）
    7    深度图（画布空间，已按 alpha 掩掉外部）
    8    常数平面 = 顶点数 / 300
    9    常数平面 = 画布内名义点距 / 32
目标: 单通道热力图，每个顶点一个高斯斑（峰值 1，sigma=SIGMA）。
"""
from __future__ import annotations

import glob
import json
import os

import numpy as np

CANVAS = 384
SIGMA = 2.0
DIST_NORM = 48.0
N_CHANNELS = 10

try:
    from .lines import edge_energy, line_mask
except ImportError:
    from autogrid.lines import edge_energy, line_mask


def index_cache(cache_root):
    """返回 [(npz, png, depth_png, model, part)]。"""
    out = []
    for npz in sorted(glob.glob(os.path.join(cache_root, "*", "*.npz"))):
        d = os.path.dirname(npz)
        stem = os.path.basename(npz)[:-4]
        png = os.path.join(d, stem + ".png")
        dpng = os.path.join(d, stem + "_d.png")
        if not os.path.exists(png):
            continue
        model = os.path.basename(d)
        out.append((npz, png, dpng if os.path.exists(dpng) else None, model, stem))
    return out


def _gauss_heat(pts, canvas, sigma):
    heat = np.zeros((canvas, canvas), np.float32)
    rad = int(np.ceil(3.0 * sigma))
    yy, xx = np.mgrid[-rad:rad + 1, -rad:rad + 1].astype(np.float32)
    g = np.exp(-(xx ** 2 + yy ** 2) / (2.0 * sigma * sigma)).astype(np.float32)
    H = W = canvas
    for x, y in pts:
        cx, cy = int(round(float(x))), int(round(float(y)))
        x0, x1 = cx - rad, cx + rad + 1
        y0, y1 = cy - rad, cy + rad + 1
        sx0, sx1 = max(0, -x0), g.shape[1] - max(0, x1 - W)
        sy0, sy1 = max(0, -y0), g.shape[0] - max(0, y1 - H)
        if sx1 <= sx0 or sy1 <= sy0:
            continue
        tx0, tx1 = max(0, x0), min(W, x1)
        ty0, ty1 = max(0, y0), min(H, y1)
        np.maximum(heat[ty0:ty1, tx0:tx1], g[sy0:sy1, sx0:sx1], out=heat[ty0:ty1, tx0:tx1])
    return heat


def load_sample(png, dpng, npz, *, sigma=SIGMA, augment=False, use_depth=True):
    """返回 (img(10,C,C) float32, heat(1,C,C) float32, meta dict)。"""
    from PIL import Image
    from scipy import ndimage as ndi

    rgba = np.asarray(Image.open(png).convert("RGBA"))
    C = rgba.shape[0]
    rgb = rgba[:, :, :3]
    alpha = rgba[:, :, 3]
    content = alpha > 4

    if use_depth and dpng:
        depth = np.asarray(Image.open(dpng).convert("L")).astype(np.float32) / 255.0
        if depth.shape != (C, C):
            depth = np.zeros((C, C), np.float32)
        depth = np.where(content, depth, 0.0)
    else:
        depth = np.zeros((C, C), np.float32)

    dist = ndi.distance_transform_edt(content).astype(np.float32) / DIST_NORM
    np.clip(dist, 0.0, 1.0, out=dist)

    edge = edge_energy(rgb).astype(np.float32)
    peak = float(np.percentile(edge[content], 99.0)) if content.any() else 1.0
    edge = np.clip(edge / max(peak, 1e-6), 0.0, 1.0)
    line = line_mask(rgb, alpha).astype(np.float32)

    z = np.load(npz, allow_pickle=True)
    verts = z["verts"].astype(np.float32)
    nv = int(z["nv"])
    spacing = float(z["spacing"])

    if augment and np.random.rand() < 0.5:
        rgba = rgba[:, ::-1].copy(); rgb = rgba[:, :, :3]; alpha = rgba[:, :, 3]
        content = alpha > 4
        depth = depth[:, ::-1].copy(); dist = dist[:, ::-1].copy()
        edge = edge[:, ::-1].copy(); line = line[:, ::-1].copy()
        verts = verts.copy(); verts[:, 0] = (C - 1) - verts[:, 0]

    img = np.empty((N_CHANNELS, C, C), np.float32)
    img[0] = rgb[:, :, 0] / 255.0
    img[1] = rgb[:, :, 1] / 255.0
    img[2] = rgb[:, :, 2] / 255.0
    img[3] = edge
    img[4] = line
    img[5] = alpha / 255.0
    img[6] = dist
    img[7] = depth
    img[8] = min(nv / 300.0, 2.0)
    img[9] = min(spacing / 32.0, 2.0)
    img[:8] = img[:8] * 2.0 - 1.0

    heat = _gauss_heat(verts, C, sigma)[None]
    meta = dict(nv=nv, spacing=spacing, model=str(z["model"]), part=str(z["part"]),
                size=z["size"].tolist(), verts=verts)
    return img, heat, meta


class StyleDataset:
    def __init__(self, items, *, sigma=SIGMA, augment=False, use_depth=True):
        self.items = list(items)
        self.sigma = sigma
        self.augment = augment
        self.use_depth = use_depth

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        npz, png, dpng, model, part = self.items[i]
        img, heat, meta = load_sample(png, dpng, npz, sigma=self.sigma,
                                      augment=self.augment, use_depth=self.use_depth)
        return img, heat, np.float32(meta["nv"])


def split_items(items, *, holdout_models=("lerelia",), val_frac=0.05, seed=0):
    """留出整个模型 + 其余随机抽 val_frac 做验证。"""
    rng = np.random.RandomState(seed)
    ho = set(holdout_models)
    train, val = [], []
    rest = []
    for it in items:
        if it[3] in ho:
            val.append(it)
        else:
            rest.append(it)
    idx = rng.permutation(len(rest))
    nval = int(len(rest) * val_frac)
    val += [rest[i] for i in idx[:nval]]
    train = [rest[i] for i in idx[nval:]]
    return train, val
