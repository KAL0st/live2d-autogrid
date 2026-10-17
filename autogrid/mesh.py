# -*- coding: utf-8 -*-
"""autogrid.mesh —— alpha 通道驱动的 ArtMesh 自动布点（几何算法，无学习成分）。

配方来自 Bunraku（arXiv:2607.27348）Stage 2：
  "Its mesh is a deterministic function of the layer's alpha channel alone,
   with no learned component and nothing sampled."

步骤：
  1. content = alpha > 4 (255 制)      —— 阈值 4 而非 0：滤掉生成式图层的不可见噪声底
  2. mask    = content 膨胀 3 px        —— 否则边缘三角形会被裁掉，留下白缝
  3. 取所有轮廓（外轮廓 + 孔洞）
  4. 边界预算按各轮廓弧长比例分配
  5. 内部点 = 抖动格点，限制在 content 内
  6. Delaunay 三角化，默认用质心过滤把网格收紧到内容上
  7. 覆盖补齐：只要还有可见像素没被盖住，就在漏洞处补顶点重新三角化
     —— 覆盖不全 = 部件会被切掉，这是工具能不能用的硬指标

坐标约定：像素坐标，原点左上，与 PNG 1:1。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import Delaunay
from skimage import measure

ALPHA_THRESHOLD = 4      # 255 制；Bunraku 实测值，不要改成 0
DILATE_PX = 3.0          # Bunraku 实测值
BOUNDARY_FRACTION = 0.55 # 约 55% 顶点给边界，45% 给内部

# 默认疏密：从 758 个真实 ArtMesh（两个 Live2D 模型）的建模师手布网格反推。
# 建模师的布点间距不是固定值，而是随部件变大而变松（实测按轮廓总长分桶）：
#   轮廓 41-197px  ->  每 8.2 px 一个顶点      轮廓 687-1112px -> 每 23.8 px
#   轮廓 197-392px ->  每 12.0 px              轮廓 1112-8098px -> 每 24.6 px（封顶）
# 拟合出来就是 间距 = min(24.6, 0.75*sqrt(轮廓总长))。
SPACING_SCALE = 0.75       # 间距 = SPACING_SCALE * sqrt(轮廓总长)
SPACING_MAX = 24.6         # 间距上限（大部件再大也不会更松）
SPACING_MIN = 4.0          # 间距下限（小部件再小也不会更密）
AREA_SPACING_COEF = 0.40   # 成块部件的面积兜底：顶点数 >= 0.40*sqrt(面积)
AREA_PER_VERTEX = 279.0    # 老口径（--ppv 用）：每个顶点摊到的可见面积 px^2
LENGTH_PER_VERTEX = 15.0   # 老口径：每个顶点摊到的轮廓长度 px
MIN_VERTICES = 8
MAX_VERTICES = 600
COVER_TARGET = 0.999       # 覆盖补齐的目标覆盖率


@dataclass
class MeshResult:
    vertices: np.ndarray            # (N, 2) 像素坐标 (x, y)
    triangles: np.ndarray           # (M, 3) int32
    boundary_count: int
    interior_count: int
    contour_count: int
    meta: dict = field(default_factory=dict)

    @property
    def n_vertices(self) -> int:
        return int(self.vertices.shape[0])

    @property
    def n_triangles(self) -> int:
        return int(self.triangles.shape[0])


def disk(radius: float) -> np.ndarray:
    r = int(np.ceil(radius))
    y, x = np.ogrid[-r:r + 1, -r:r + 1]
    return (x * x + y * y) <= (radius * radius + 1e-9)


def content_mask(alpha: np.ndarray, threshold: int = ALPHA_THRESHOLD) -> np.ndarray:
    """可见内容掩码。alpha 可以是 0-255 或 0-1，自动判别。"""
    a = np.asarray(alpha)
    if a.dtype.kind == "f" and a.max() <= 1.0 + 1e-6:
        a = a * 255.0
    return a > threshold


def fetch_contours(mask: np.ndarray, min_length: float = 6.0) -> list[np.ndarray]:
    """所有等值轮廓，返回 [(n,2) 的 (x,y) 数组]。外轮廓与孔洞都返回。"""
    padded = np.pad(mask.astype(np.float32), 1)
    raw = measure.find_contours(padded, 0.5)
    out: list[np.ndarray] = []
    for c in raw:
        rc = c - 1.0                                   # 去掉 padding 偏移
        if len(rc) < 3:
            continue
        seg = np.linalg.norm(np.diff(rc, axis=0), axis=1).sum()
        if seg < min_length:
            continue
        out.append(np.stack([rc[:, 1], rc[:, 0]], axis=1))   # (row,col) -> (x,y)
    out.sort(key=lambda p: -polyline_length(p))
    return out


def polyline_length(poly: np.ndarray) -> float:
    p = np.vstack([poly, poly[:1]])
    return float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum())


def resample_closed(poly: np.ndarray, n: int) -> np.ndarray:
    """在闭合折线上按弧长均匀取 n 个点（不重复首尾）。"""
    if n <= 0 or len(poly) < 2:
        return np.zeros((0, 2))
    p = np.vstack([poly, poly[:1]])
    seg = np.linalg.norm(np.diff(p, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    total = s[-1]
    if total <= 0:
        return np.zeros((0, 2))
    t = (np.arange(n) / float(n)) * total
    return np.stack([np.interp(t, s, p[:, 0]), np.interp(t, s, p[:, 1])], axis=1)


def sample_boundary(contours: list[np.ndarray], budget: int) -> np.ndarray:
    """按弧长比例把 boundary 预算分配给各轮廓，逐条均匀采样。"""
    if budget <= 0 or not contours:
        return np.zeros((0, 2))
    lengths = np.array([polyline_length(c) for c in contours], dtype=np.float64)
    total = lengths.sum()
    if total <= 0:
        return np.zeros((0, 2))
    raw = lengths / total * budget
    # 每个轮廓至少 3 个点才可能围成面；够分的按比例
    counts = np.where(raw < 3, 3, np.floor(raw).astype(int))
    pts = [resample_closed(c, int(n)) for c, n in zip(contours, counts)]
    pts = [p for p in pts if len(p)]
    return np.vstack(pts) if pts else np.zeros((0, 2))


def sample_on_mask(mask: np.ndarray, n: int, rng: np.random.Generator,
                  r: float | None = None, max_tries: int = 5,
                  max_candidates: int = 60000) -> np.ndarray:
    """在 mask 的像素上撒「互相至少隔 r 像素」的点（贪心 Poisson disk）。

    用来把顶点摆到纹理线上 —— 建模师就是这么干的，而且要摆得均匀，
    不能一条线上挤一堆。
    """
    if n <= 0:
        return np.zeros((0, 2))
    ys, xs = np.nonzero(mask)
    m = len(ys)
    if m == 0:
        return np.zeros((0, 2))
    if m > max_candidates:
        keep = rng.choice(m, max_candidates, replace=False)
        ys, xs, m = ys[keep], xs[keep], len(keep)
    if r is None or r <= 0:
        r = math.sqrt(m / float(n)) * 0.75
    acc: list = []
    for _ in range(max_tries):
        cell = max(r, 1e-3)
        grid: dict = {}
        acc = []
        r2 = r * r
        for idx in rng.permutation(m):
            y = int(ys[idx]); x = int(xs[idx])
            ky = int(y // cell); kx = int(x // cell)
            ok = True
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    for py, px in grid.get((ky + dy, kx + dx), ()):
                        if (py - y) * (py - y) + (px - x) * (px - x) < r2:
                            ok = False
                            break
                    if not ok:
                        break
                if not ok:
                    break
            if ok:
                acc.append((float(x), float(y)))
                grid.setdefault((ky, kx), []).append((y, x))
                if len(acc) >= n:
                    break
        if len(acc) >= n * 0.8 or r < 0.9:
            break
        r *= 0.75
    if len(acc) > n:
        sel = rng.choice(len(acc), n, replace=False)
        acc = [acc[i] for i in sorted(sel)]
    return np.asarray(acc, dtype=np.float64) if acc else np.zeros((0, 2))


def sample_interior(content: np.ndarray, n: int, rng: np.random.Generator,
                    jitter: float = 0.45, max_tries: int = 6,
                    line_mask: np.ndarray | None = None,
                    line_fraction: float = 0.0) -> np.ndarray:
    """在 content 内撒点。给了 line_mask 就把一部分点摆到纹理线上。

    line_fraction=0 时就是原来的 Bunraku 抖动格点，行为完全不变。
    """
    if n <= 0:
        return np.zeros((0, 2))
    line_pts = np.zeros((0, 2))
    if line_mask is not None and line_fraction > 0:
        line_pts = sample_on_mask(line_mask, int(round(n * line_fraction)), rng)
    n_lat = n - len(line_pts)
    lat = _interior_lattice(content, n_lat, rng, jitter, max_tries,
                            avoid=line_pts if len(line_pts) else None)
    parts = [p for p in (line_pts, lat) if len(p)]
    return np.vstack(parts) if parts else np.zeros((0, 2))


def _interior_lattice(content: np.ndarray, n: int, rng: np.random.Generator,
                      jitter: float = 0.45, max_tries: int = 6,
                      avoid: np.ndarray | None = None) -> np.ndarray:
    """在 content 内撒抖动格点（Bunraku 的 jittered interior lattice）。"""
    if n <= 0:
        return np.zeros((0, 2))
    h, w = content.shape
    area = float(content.sum())
    if area <= 0:
        return np.zeros((0, 2))
    step = max(1.6, np.sqrt(area / float(n)))
    pts = np.zeros((0, 2))
    for _ in range(max_tries):
        xs = np.arange(step * 0.5, w, step)
        ys = np.arange(step * 0.5, h, step)
        gx, gy = np.meshgrid(xs, ys)
        pts = np.stack([gx.ravel(), gy.ravel()], axis=1).astype(np.float64)
        pts += rng.uniform(-jitter, jitter, pts.shape) * step
        xi = np.clip(np.round(pts[:, 0]).astype(int), 0, w - 1)
        yi = np.clip(np.round(pts[:, 1]).astype(int), 0, h - 1)
        pts = pts[content[yi, xi]]
        if len(pts) == 0:
            step *= 0.6
            continue
        if avoid is not None and len(avoid):
            from scipy.spatial import cKDTree
            d, _ = cKDTree(avoid).query(pts)
            pts = pts[d > 0.55 * step]
            if len(pts) == 0:
                step *= 0.6
                continue
        if len(pts) <= n:
            return pts
        if len(pts) <= n * 1.35:
            idx = rng.choice(len(pts), n, replace=False)
            return pts[np.sort(idx)]
        step *= 1.08
    if len(pts) == 0:
        return np.zeros((0, 2))
    if avoid is not None and len(avoid):
        from scipy.spatial import cKDTree
        d, _ = cKDTree(avoid).query(pts)
        pts = pts[d > max(1.5, 0.3 * step)]
        if len(pts) == 0:
            return np.zeros((0, 2))
    idx = rng.choice(len(pts), min(n, len(pts)), replace=False)
    return pts[np.sort(idx)]


def _bary_lattice(k: int) -> tuple[np.ndarray, np.ndarray]:
    us, vs = [], []
    for i in range(k + 1):
        for j in range(k + 1 - i):
            us.append(i / float(k))
            vs.append(j / float(k))
    return np.asarray(us), np.asarray(vs)


def triangle_hits_mask(points: np.ndarray, simplices: np.ndarray, mask: np.ndarray,
                       k: int = 4) -> np.ndarray:
    """逐个三角形判断：它有没有盖到 mask 里的像素（重心格点采样）。"""
    p0, p1, p2 = points[simplices[:, 0]], points[simplices[:, 1]], points[simplices[:, 2]]
    u, v = _bary_lattice(k)
    w0, w1, w2 = (1.0 - u - v)[None, :, None], u[None, :, None], v[None, :, None]
    s = p0[:, None, :] * w0 + p1[:, None, :] * w1 + p2[:, None, :] * w2    # (M,S,2)
    h, w = mask.shape
    xi = np.clip(np.round(s[:, :, 0]).astype(int), 0, w - 1)
    yi = np.clip(np.round(s[:, :, 1]).astype(int), 0, h - 1)
    return mask[yi, xi].any(axis=1)


def triangulate(points: np.ndarray, keep_mask: np.ndarray | None = None,
                clip: str | bool = "centroid") -> np.ndarray:
    """Delaunay 三角化 + 按内容裁剪三角形。

    clip="touch"   （默认）丢掉「完全落在内容之外」的三角形：质心在内容里的、
                   以及擦到内容的都留下。覆盖最保险，但可能留下跨凹陷的大三角。
    clip="centroid" 只留质心落在内容里的三角形。网格最贴合，但擦边的三角形会被
                   误删、把部件边缘切掉 —— 要靠覆盖补齐（refine_coverage）补回顶点。
    clip=False / "none" 全部保留（网格 = 点集凸包），空白处也铺满，外溢最大。
    """
    if len(points) < 3:
        return np.zeros((0, 3), dtype=np.int32)
    tri = Delaunay(points)
    simplices = tri.simplices.astype(np.int32)
    if clip in (False, "none", None) or keep_mask is None:
        return simplices
    cent = points[simplices].mean(axis=1)
    h, w = keep_mask.shape
    xi = np.clip(np.round(cent[:, 0]).astype(int), 0, w - 1)
    yi = np.clip(np.round(cent[:, 1]).astype(int), 0, h - 1)
    cent_ok = keep_mask[yi, xi]
    if clip == "centroid":
        return simplices[cent_ok]
    return simplices[cent_ok | triangle_hits_mask(points, simplices, keep_mask)]


def prune_degenerate(points: np.ndarray, tris: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """去掉零面积三角形。"""
    if not len(tris):
        return tris
    p0, p1, p2 = points[tris[:, 0]], points[tris[:, 1]], points[tris[:, 2]]
    area2 = np.abs((p1[:, 0] - p0[:, 0]) * (p2[:, 1] - p0[:, 1]) -
                   (p2[:, 0] - p0[:, 0]) * (p1[:, 1] - p0[:, 1]))
    return tris[area2 > eps]


def suggest_budget(alpha: np.ndarray, *, dilate_px: float = DILATE_PX,
                   spacing_scale: float = SPACING_SCALE, spacing_max: float = SPACING_MAX,
                   spacing_min: float = SPACING_MIN, area_coef: float = AREA_SPACING_COEF,
                   area_per_vertex: float | None = None,
                   length_per_vertex: float | None = None,
                   minimum: int = MIN_VERTICES, maximum: int = MAX_VERTICES,
                   threshold: int = ALPHA_THRESHOLD) -> int:
    """按内容自己推荐顶点数。

    主判据是轮廓长度：细长部件（发丝、勾线）按面积算会严重欠采样、网格整片崩掉，
    所以间距按 间距 = clamp(spacing_scale*sqrt(轮廓总长)) 随部件变大而变松 ——
    这正是实测里建模师的做法。成块部件再用 0.40*sqrt(面积) 兜底。

    想退回「每个顶点摊 N 像素」的老口径，传 area_per_vertex=N 即可（CLI 的 --ppv）。
    """
    content = content_mask(alpha, threshold)
    area = float(content.sum())
    if area <= 0:
        return minimum
    mask = ndi.binary_dilation(content, structure=disk(dilate_px)) if dilate_px > 0 else content
    perim = sum(polyline_length(c) for c in fetch_contours(mask))
    if area_per_vertex is not None or length_per_vertex is not None:
        apv = AREA_PER_VERTEX if area_per_vertex is None else area_per_vertex
        lpv = LENGTH_PER_VERTEX if length_per_vertex is None else length_per_vertex
        n = max(area / max(apv, 1e-6), perim / max(lpv, 1e-6))
    else:
        spacing = min(spacing_max, max(spacing_min, spacing_scale * math.sqrt(max(perim, 1.0))))
        n = max(perim / spacing, area_coef * math.sqrt(area))
    return int(np.clip(round(n), minimum, maximum))


def _stamp(points: np.ndarray, shape) -> np.ndarray:
    h, w = shape
    m = np.zeros((h, w), dtype=bool)
    xi = np.clip(np.round(points[:, 0]).astype(int), 0, w - 1)
    yi = np.clip(np.round(points[:, 1]).astype(int), 0, h - 1)
    m[yi, xi] = True
    return m


def refine_coverage(points: np.ndarray, tris: np.ndarray, content: np.ndarray,
                    mask: np.ndarray, *, clip: str = "centroid", target: float = COVER_TARGET,
                    max_extra: int = 0, chunk: int = 8) -> tuple[np.ndarray, np.ndarray, dict]:
    """覆盖补齐：还有可见像素没被网格盖住，就在漏洞里挑离已有点最远的像素补顶点。

    覆盖不全 = 部件会被切掉。这是保证工具「能用」的兜底步骤。
    """
    from .metrics import rasterize

    total = float(content.sum())
    extra = 0
    info = {"added": 0, "coverage": 1.0}
    if total <= 0 or max_extra <= 0:
        return points, tris, info

    while extra < max_extra:
        cov = float((rasterize(points, tris, content.shape) & content).sum()) / total
        if cov >= target:
            info["coverage"] = cov
            return points, tris, info
        mesh = rasterize(points, tris, content.shape)
        miss = content & ~mesh
        if not miss.any():
            info["coverage"] = cov
            return points, tris, info
        # 未覆盖像素到「最近已有顶点」的距离；挑最远的几个补上
        d = ndi.distance_transform_edt(~_stamp(points, content.shape))
        d = np.where(miss, d, -1.0)
        add = []
        for _ in range(min(chunk, max_extra - extra)):
            i = int(np.argmax(d))
            y, x = np.unravel_index(i, d.shape)
            if d[y, x] <= 0:
                break
            add.append((float(x), float(y)))
            yy, xx = np.ogrid[:d.shape[0], :d.shape[1]]
            d = np.minimum(d, np.sqrt((yy - y) ** 2 + (xx - x) ** 2))
            d = np.where(miss, d, -1.0)
        if not add:
            break
        points = np.vstack([points, np.asarray(add, dtype=np.float64)])
        extra += len(add)
        tris = prune_degenerate(points, triangulate(points, keep_mask=mask, clip=clip))
    info["added"] = extra
    info["coverage"] = float((rasterize(points, tris, content.shape) & content).sum()) / total
    return points, tris, info


def auto_mesh(alpha: np.ndarray, budget: int, *, threshold: int = ALPHA_THRESHOLD,
              dilate_px: float = DILATE_PX, boundary_fraction: float = BOUNDARY_FRACTION,
              clip: str = "centroid", cover: bool = True, cover_extra: float = 4.0,
              seed: int = 0, image: np.ndarray | None = None,
              line_fraction: float = 0.0,
              line_kwargs: dict | None = None) -> MeshResult:
    """从 alpha 通道生成网格。budget = 初始目标顶点数（覆盖补齐可能再加最多 cover_extra 倍）。

    默认只看 alpha（纯几何）。传了 image（RGB / RGBA）并给 line_fraction>0，
    就会把一部分内部顶点摆到「部件内部的纹理线」上 —— 这是建模师的手上习惯
    （实测他的内部顶点 40.5% 落在线上）。line_kwargs 透传给 lines.line_mask。
    """
    rng = np.random.default_rng(seed)
    content = content_mask(alpha, threshold)
    empty = MeshResult(np.zeros((0, 2)), np.zeros((0, 3), np.int32), 0, 0, 0)
    if not content.any():
        return empty
    mask = ndi.binary_dilation(content, structure=disk(dilate_px)) if dilate_px > 0 else content

    contours = fetch_contours(mask)
    n_b = int(round(budget * boundary_fraction))
    n_i = max(0, budget - n_b)

    lm = None
    if image is not None and line_fraction > 0:
        from .lines import line_mask as _lm
        lm = _lm(image, alpha, threshold=threshold, **(line_kwargs or {}))

    bp = sample_boundary(contours, n_b)
    ip = sample_interior(content, n_i, rng, line_mask=lm, line_fraction=line_fraction)

    pts = np.vstack([p for p in (bp, ip) if len(p)]) if (len(bp) or len(ip)) else np.zeros((0, 2))
    if len(pts) < 3:
        return MeshResult(pts, np.zeros((0, 3), np.int32), len(bp), len(ip), len(contours))

    tris = prune_degenerate(pts, triangulate(pts, keep_mask=mask, clip=clip))
    info = {"added": 0, "coverage": None}
    if cover:
        pts, tris, info = refine_coverage(
            pts, tris, content, mask, clip=clip,
            max_extra=int(max(8, round(budget * cover_extra))))

    return MeshResult(vertices=pts, triangles=tris, boundary_count=int(len(bp)),
                      interior_count=int(len(ip)), contour_count=len(contours),
                      meta={"threshold": threshold, "dilate_px": dilate_px,
                            "boundary_fraction": boundary_fraction, "clip": clip,
                            "cover_added": int(info["added"]),
                            "coverage": info["coverage"],
                            "line_fraction": line_fraction,
                            "line_px": int(lm.sum()) if lm is not None else 0})

def snap_to_content(points: np.ndarray, content: np.ndarray) -> np.ndarray:
    """把落在可见区域外的点吸附到最近的可见像素（模型预测可能压到边缘外）。"""
    H, W = content.shape
    pts = np.array(points, dtype=np.float64).reshape(-1, 2)
    if not len(pts):
        return pts
    xi = np.clip(np.round(pts[:, 0]).astype(int), 0, W - 1)
    yi = np.clip(np.round(pts[:, 1]).astype(int), 0, H - 1)
    outside = ~content[yi, xi]
    if outside.any():
        ind = ndi.distance_transform_edt(~content, return_indices=True)[1]
        pts[outside, 0] = ind[1][yi[outside], xi[outside]]
        pts[outside, 1] = ind[0][yi[outside], xi[outside]]
    return np.clip(pts, [0.0, 0.0], [W - 1.0, H - 1.0])


def mesh_from_points(points: np.ndarray, alpha: np.ndarray, *, threshold: int = ALPHA_THRESHOLD,
                     dilate_px: float = DILATE_PX, clip: str = "centroid",
                     cover: bool = True, cover_extra: float = 1.0,
                     min_extra: int = 8) -> MeshResult:
    """用给定的顶点建网格（Delaunay + 质心裁剪 + 覆盖补齐）。

    给「模型预测出来的点」用：几何收尾交给这套确定性代码，保证覆盖不会崩。
    """
    content = content_mask(alpha, threshold)
    empty = MeshResult(np.zeros((0, 2)), np.zeros((0, 3), np.int32), 0, 0, 0)
    if not content.any():
        return empty
    mask = ndi.binary_dilation(content, structure=disk(dilate_px)) if dilate_px > 0 else content

    pts = snap_to_content(points, content)
    if len(pts) >= 3:
        keep = [0]
        for i in range(1, len(pts)):
            if float(np.min(np.linalg.norm(pts[keep] - pts[i], axis=1))) > 0.6:
                keep.append(i)
        pts = pts[keep]
    n_in = int(len(pts))
    if len(pts) < 3:
        return MeshResult(pts, np.zeros((0, 3), np.int32), 0, n_in, 0)

    tris = prune_degenerate(pts, triangulate(pts, keep_mask=mask, clip=clip))
    info = {"added": 0, "coverage": None}
    if cover:
        pts, tris, info = refine_coverage(
            pts, tris, content, mask, clip=clip,
            max_extra=int(max(min_extra, round(n_in * cover_extra))))
    return MeshResult(vertices=pts, triangles=tris, boundary_count=0, interior_count=n_in,
                      contour_count=0,
                      meta={"threshold": threshold, "dilate_px": dilate_px, "clip": clip,
                            "cover_added": int(info["added"]), "coverage": info["coverage"],
                            "source": "points", "n_input": n_in})

