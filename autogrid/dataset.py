# -*- coding: utf-8 -*-
"""autogrid.dataset —— 把 .cmo3 导出成「图片 + 建模师真顶点」训练集。

和旧 dataset/ 的区别（旧那套是坏的）：
  · 顶点取 CArtMeshSource 的 uvs，按【逻辑页】宽度换算（图宽 / m00），与图片同一坐标系
  · 图片按「可见内容 ∪ 顶点包围盒」紧裁，保证顶点一定落在图内
  · 每个样本都自检覆盖率，写进 JSON，坏样本当场就能看出来
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from .cmo3 import load_parts
from .mesh import content_mask, fetch_contours, polyline_length, suggest_budget
from .metrics import coverage, spill, rasterize


def union_box(alpha: np.ndarray, verts: np.ndarray, pad: int = 4):
    """可见内容 ∪ 顶点的包围盒，向外留 pad 像素。"""
    h, w = alpha.shape
    m = content_mask(alpha)
    if m.any():
        ys, xs = np.nonzero(m)
        x0, x1, y0, y1 = int(xs.min()), int(xs.max()) + 1, int(ys.min()), int(ys.max()) + 1
    else:
        x0, y0, x1, y1 = 0, 0, w, h
    if len(verts):
        x0 = min(x0, int(math.floor(float(verts[:, 0].min()))))
        y0 = min(y0, int(math.floor(float(verts[:, 1].min()))))
        x1 = max(x1, int(math.ceil(float(verts[:, 0].max()))) + 1)
        y1 = max(y1, int(math.ceil(float(verts[:, 1].max()))) + 1)
    return max(0, x0 - pad), max(0, y0 - pad), min(w, x1 + pad), min(h, y1 + pad)


def export_model(cmo3, outdir, *, pad: int = 4, min_visible: int = 64,
                 min_area: int = 100, min_vertices: int = 4, limit: int = 0,
                 save_canvas: bool = True, isolate: bool = True, progress=None) -> list[dict]:
    """把一个 .cmo3 导出成训练集，返回样本清单。"""
    from PIL import Image

    cmo3 = Path(cmo3)
    outdir = Path(outdir) / cmo3.stem
    outdir.mkdir(parents=True, exist_ok=True)
    parts = load_parts(cmo3, pad=2, include_uv_bounds=True, min_visible=min_visible)
    if limit:
        parts = parts[:limit]
    samples = []
    for k, q in enumerate(parts):
        alpha = q.alpha
        vis = content_mask(alpha)
        area = int(vis.sum())
        if area < min_area or len(q.uv_px) < min_vertices or len(q.triangles) == 0:
            continue
        x0, y0, x1, y1 = union_box(alpha, q.uv_px, pad)
        img = q.image[y0:y1, x0:x1]
        verts = q.uv_px - np.array([x0, y0], dtype=np.float64)
        canvas = q.canvas - np.array([x0 - q.origin[0], y0 - q.origin[1]], dtype=np.float64) if save_canvas and q.canvas is not None else None
        stem = "%s_%d" % ("".join(c for c in q.name if c.isalnum() or c in "-_") or "part", q.xs_id)
        cov_raw = coverage(verts, q.triangles, content_mask(img[:, :, 3]))
        if isolate:
            # 图集里邻居部件的矩形包围盒会和本部件咬合，裁切框里常混进别人的画。
            # 只保留本部件网格盖到的像素，样本才是「一张图 = 一个部件」。
            keep = rasterize(verts, q.triangles, img.shape[:2])
            img = np.where(keep[:, :, None], img, 0).astype(np.uint8)
        Image.fromarray(img, "RGBA").save(outdir / (stem + ".png"))
        v2 = content_mask(img[:, :, 3])
        rec = {
            "model": cmo3.stem, "part": q.name, "xs_id": int(q.xs_id), "texture": q.texture,
            "image": "%s/%s.png" % (cmo3.stem, stem), "size": [int(img.shape[1]), int(img.shape[0])],
            "n_vertices": int(len(verts)), "n_triangles": int(len(q.triangles)),
            "coverage": round(float(coverage(verts, q.triangles, v2)), 4) if v2.any() else None,
            "spill": round(float(spill(verts, q.triangles, v2)), 4) if v2.any() else None,
            "coverage_raw": round(float(cov_raw), 4),
            "area": area, "suggest_vertices": int(suggest_budget(img[:, :, 3])),
            "logical": [float(q.logical[0]), float(q.logical[1])],
            "origin": [int(x0), int(y0)],
            "vertices": [[round(float(a), 3), round(float(b), 3)] for a, b in verts],
            "triangles": [[int(a), int(b), int(c)] for a, b, c in q.triangles],
        }
        if canvas is not None:
            rec["canvas"] = [[round(float(a), 3), round(float(b), 3)] for a, b in canvas]
        (outdir / (stem + ".json")).write_text(json.dumps(rec, ensure_ascii=False, indent=1), encoding="utf-8")
        samples.append(rec)
        if progress and k % 50 == 0:
            progress("%s ... %d/%d（已收 %d）" % (cmo3.name, k, len(parts), len(samples)))
    return samples


def write_manifest(samples: list[dict], outdir) -> dict:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    mp = outdir / "manifest.json"
    fresh = [{k: s[k] for k in ("model", "part", "image", "size", "n_vertices", "coverage", "spill", "area")} for s in samples]
    cur = {s["model"] for s in samples}
    if mp.exists():  # 保留以前导出过的其它模型
        try:
            prev = json.loads(mp.read_text(encoding="utf-8")).get("samples", [])
            fresh = [r for r in prev if r.get("model") not in cur] + fresh
        except Exception:
            pass
    samples = fresh
    cov = np.array([s["coverage"] for s in samples if s.get("coverage") is not None], dtype=np.float64)
    nv = np.array([s["n_vertices"] for s in samples], dtype=np.float64)
    models = sorted({s["model"] for s in samples})
    man = {
        "n_samples": len(samples), "n_models": len(models), "models": models,
        "coverage": {"mean": round(float(cov.mean()), 4), "median": round(float(np.median(cov)), 4),
                     "p05": round(float(np.percentile(cov, 5)), 4), "min": round(float(cov.min()), 4)} if len(cov) else None,
        "vertices": {"median": int(np.median(nv)), "mean": round(float(nv.mean()), 1), "max": int(nv.max())} if len(nv) else None,
        "samples": [{k: s[k] for k in ("model", "part", "image", "size", "n_vertices", "coverage", "spill", "area")} for s in samples],
    }
    (outdir / "manifest.json").write_text(json.dumps(man, ensure_ascii=False, indent=1), encoding="utf-8")
    return man
