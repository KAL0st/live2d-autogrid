# -*- coding: utf-8 -*-
"""把「部件内部的纹理线」找出来。

卡师傅的布点风格实测：他的内部顶点 40.5% 落在部件内部的纹理线上，而纯随机只有 16%。
所以先把「线」定义出来 —— 这一步的判据必须跟评测口径一致（_tmp/exp_same_budget.py）。
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

__all__ = ["line_mask", "edge_energy"]

_CORE_PX = 3.0        # 离轮廓多远才算“内部”
_PERCENTILE = 92.0    # 内部像素里边缘强度的前 8% 算“线”
_SMOOTH = 5.0


def edge_energy(rgb: np.ndarray, smooth: float = _SMOOTH) -> np.ndarray:
    """灰度图的边缘强度（Sobel 幅值 + 均值滤波）。"""
    g = np.asarray(rgb, dtype=np.float32)
    if g.ndim == 3:
        g = g[..., :3].mean(axis=2)
    if g.max() > 1.5:
        g = g / 255.0
    mag = np.hypot(ndi.sobel(g, axis=1), ndi.sobel(g, axis=0))
    return ndi.uniform_filter(mag, smooth) if smooth and smooth > 0 else mag


def line_mask(rgb: np.ndarray, alpha: np.ndarray, *, threshold: int = 4,
              core_px: float = _CORE_PX, percentile: float = _PERCENTILE,
              smooth: float = _SMOOTH) -> np.ndarray:
    """部件内部的纹理线掩码（布尔图，跟 alpha 同尺寸）。

    rgb   —— (H,W,3|4) 或 (H,W) 的图
    alpha —— (H,W) 的 alpha 通道（0-255 或 0-1）
    core_px —— 离可见轮廓至少这么远的像素才算“内部”
    percentile —— 内部像素里，边缘强度排前 (100-percentile)% 的算线
    """
    a = np.asarray(alpha, dtype=np.float32)
    if a.max() <= 1.5:
        a = a * 255.0
    content = a > threshold
    if content.sum() < 64:
        return np.zeros_like(content)
    d = ndi.distance_transform_edt(content)
    core = content & (d > core_px)
    if core.sum() < 64:
        return np.zeros_like(content)
    e = edge_energy(rgb, smooth)
    thr = float(np.percentile(e[core], percentile))
    return core & (e > thr)
