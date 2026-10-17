# -*- coding: utf-8 -*-
"""从热力图取点 + 与建模师顶点比对的指标。"""
from __future__ import annotations

import numpy as np


def _nms_peaks(h, n, min_dist, threshold, max_candidates=4000):
    """一次贪心 NMS 取点（栅格哈希加速）。"""
    from scipy import ndimage as ndi
    md = max(1.5, float(min_dist))
    mx = ndi.maximum_filter(h, size=max(3, int(2 * md) | 1))
    peaks = (h >= mx - 1e-9) & (h > threshold)
    ys, xs = np.nonzero(peaks)
    if len(ys) == 0:
        ys, xs = np.nonzero(h > threshold)
        if len(ys) == 0:
            return np.zeros((0, 2), np.float32)
    vals = h[ys, xs]
    if len(vals) > max_candidates:
        keep = np.argpartition(-vals, max_candidates)[:max_candidates]
        ys, xs, vals = ys[keep], xs[keep], vals[keep]
    order = np.lexsort((xs, ys, -vals))
    ys, xs, vals = ys[order], xs[order], vals[order]
    cell = md
    grid = {}
    picked = []
    md2 = md * md
    for y, x in zip(ys.tolist(), xs.tolist()):
        gy, gx = int(y // cell), int(x // cell)
        ok = True
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                for (py, px) in grid.get((gy + dy, gx + dx), ()):
                    if (py - y) ** 2 + (px - x) ** 2 < md2:
                        ok = False
                        break
                if not ok:
                    break
            if not ok:
                break
        if ok:
            grid.setdefault((gy, gx), []).append((y, x))
            picked.append((y, x))
            if len(picked) >= n:
                break
    if not picked:
        return np.zeros((0, 2), np.float32)
    return np.array([(x, y) for y, x in picked], np.float32)


def _greedy_spread(h, n, min_dist, threshold):
    """兜底：不考虑局部极大，直接按热力值从高到低贪心撒点（保证能取满 n 个）。"""
    ys, xs = np.nonzero(h > threshold)
    if len(ys) == 0:
        ys, xs = np.nonzero(h > 0)
        if len(ys) == 0:
            return np.zeros((0, 2), np.float32)
    vals = h[ys, xs]
    order = np.argsort(-vals, kind="stable")
    ys, xs = ys[order], xs[order]
    cell = max(1.5, float(min_dist))
    md2 = cell * cell
    grid, picked = {}, []
    for y, x in zip(ys.tolist(), xs.tolist()):
        gy, gx = int(y // cell), int(x // cell)
        ok = True
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                for (py, px) in grid.get((gy + dy, gx + dx), ()):
                    if (py - y) ** 2 + (px - x) ** 2 < md2:
                        ok = False
                        break
                if not ok:
                    break
            if not ok:
                break
        if ok:
            grid.setdefault((gy, gx), []).append((y, x))
            picked.append((y, x))
            if len(picked) >= n:
                break
    return np.array([(x, y) for y, x in picked], np.float32) if picked else np.zeros((0, 2), np.float32)


def extract_points(heat, n, min_dist, threshold=0.05, max_candidates=4000, ensure_n=True):
    """从热力图取 n 个点（贪心 NMS，数量不够时自动放宽间距/阈值），返回 (N,2) 画布坐标 (x,y)。

    ★ 数量不够时必须放宽——否则「他只放了几个点在密集区」的部件会取不满，反向 Chamfer 被冤枉。
    """
    h = np.asarray(heat, np.float32)
    if h.ndim == 3:
        h = h[0]
    n = int(n)
    if n <= 0 or h.size == 0:
        return np.zeros((0, 2), np.float32)
    md = max(1.5, float(min_dist))
    th = float(threshold)
    pts = _nms_peaks(h, n, md, th, max_candidates)
    if not ensure_n:
        return pts
    md0, th0 = md, th
    for _ in range(8):
        if len(pts) >= n:
            break
        md = max(1.6, md * 0.78)
        th = max(5e-3, th * 0.6)
        pts = _nms_peaks(h, n, md, th, max_candidates)
    if len(pts) < n:
        # 热力图太平（峰值不够）→ 兜底撒点，保证点数和基线一致
        md = max(2.0, md0 * 0.5)
        for _ in range(6):
            pts = _greedy_spread(h, n, md, max(th0 * 0.5, 1e-3))
            if len(pts) >= n:
                break
            md = max(1.6, md * 0.7)
    return pts[:n] if len(pts) > n else pts



def _nn_dist(a, b):
    """a 中每个点到 b 的最近距离。"""
    if len(a) == 0 or len(b) == 0:
        return np.zeros(0, np.float32)
    from scipy.spatial import cKDTree
    return cKDTree(b).query(a, k=1)[0].astype(np.float32)


def chamfer(pred, gt, spacing):
    """两个方向的平均最近距离，按名义点距归一。"""
    if len(pred) == 0 or len(gt) == 0 or spacing <= 0:
        return float("nan"), float("nan")
    return float(_nn_dist(pred, gt).mean() / spacing), float(_nn_dist(gt, pred).mean() / spacing)


def online_rate(points, line, content, core, radius=2.0):
    """内部点中落在内部纹理线上的比例。"""
    from scipy import ndimage as ndi
    if len(points) == 0:
        return float("nan"), 0
    H, W = line.shape
    xi = np.clip(np.round(points[:, 0]).astype(int), 0, W - 1)
    yi = np.clip(np.round(points[:, 1]).astype(int), 0, H - 1)
    inner = core[yi, xi]
    if inner.sum() == 0:
        return float("nan"), 0
    dl = ndi.distance_transform_edt(~line.astype(bool))
    on = (dl[yi, xi] <= radius) & inner
    return float(on.sum() / max(inner.sum(), 1)), int(inner.sum())
