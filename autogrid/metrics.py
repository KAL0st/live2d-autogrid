# -*- coding: utf-8 -*-
"""autogrid.metrics —— 网格质量指标（对齐 Bunraku 论文口径）。"""
from __future__ import annotations

import numpy as np
from skimage.draw import polygon as _polygon


def rasterize(vertices: np.ndarray, triangles: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """把三角网格栅格化成布尔掩码。shape=(H,W)，顶点为像素坐标 (x,y)。"""
    h, w = int(shape[0]), int(shape[1])
    cover = np.zeros((h, w), dtype=bool)
    if triangles is None or len(triangles) == 0 or len(vertices) == 0:
        return cover
    for s in triangles:
        pts = vertices[s]
        rr, cc = _polygon(pts[:, 1], pts[:, 0], shape=shape)
        if len(rr):
            cover[rr, cc] = True
    return cover


def coverage(vertices, triangles, visible: np.ndarray) -> float:
    """可见像素覆盖率：网格盖住了多少比例的可见（不透明）像素。越高越好。"""
    total = int(visible.sum())
    if total == 0:
        return float("nan")
    cover = rasterize(vertices, triangles, visible.shape)
    return float((cover & visible).sum()) / total


def spill(vertices, triangles, visible: np.ndarray) -> float:
    """外溢率：网格覆盖的面积里，有多少落在可见像素之外。越低越贴合轮廓。"""
    cover = rasterize(vertices, triangles, visible.shape)
    n = int(cover.sum())
    if n == 0:
        return float("nan")
    return float((cover & ~visible).sum()) / n


def mesh_area_ratio(vertices, triangles, visible: np.ndarray) -> float:
    """网格面积 / 可见像素面积。>1 表示网格比内容大（膨胀所致）。"""
    cover = rasterize(vertices, triangles, visible.shape)
    vis = int(visible.sum())
    if vis == 0:
        return float("nan")
    return float(cover.sum()) / vis


def triangle_quality(vertices: np.ndarray, triangles: np.ndarray) -> dict:
    """三角形质量：最小角分布（Live2D 关心是否存在过尖的三角形）。"""
    if triangles is None or len(triangles) == 0:
        return {"n": 0}
    p = vertices[triangles]                       # (M,3,2)
    a = p[:, 1] - p[:, 0]
    b = p[:, 2] - p[:, 1]
    c = p[:, 0] - p[:, 2]

    def ang(u, v):
        nu = np.linalg.norm(u, axis=1)
        nv = np.linalg.norm(v, axis=1)
        cos = np.clip((u * v).sum(axis=1) / np.maximum(nu * nv, 1e-12), -1, 1)
        return np.degrees(np.arccos(cos))

    angles = np.stack([ang(-a, c), ang(-b, a), ang(-c, b)], axis=1)
    mn = angles.min(axis=1)
    area = 0.5 * np.abs(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0])
    return {
        "n": int(len(triangles)),
        "min_angle_median": float(np.median(mn)),
        "min_angle_p05": float(np.percentile(mn, 5)),
        "degenerate_lt1deg": int((mn < 1.0).sum()),
        "area_median": float(np.median(area)),
    }
