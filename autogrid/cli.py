# -*- coding: utf-8 -*-
"""autogrid.cli —— 命令行入口。

用法：
  python -m autogrid.cli <输入...> [--out 输出目录] [--ppv 每个顶点摊到的像素] [--budget N] [--sheet]

输入可以是：
  * 一个 .cmo3 文件      —— 逐个 ArtMesh 部件自动布点（会用建模师的真顶点做对比）
  * 一个 .png 文件       —— 直接对整张图布点
  * 一个目录             —— 目录下所有 .png / .cmo3

输出（默认写到 <输入同级>/autogrid_out/）：
  <名称>.overlay.png   原图 + 自动网格（绿线绿点）+ 建模师网格（红线，若来自 cmo3）
  <名称>.json          顶点、三角形、以及（若有）建模师顶点
  sheet.png            所有部件的总览图
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .mesh import auto_mesh, content_mask, suggest_budget
from .metrics import coverage, spill

def budget_for(image: np.ndarray, budget=None, ppv: float | None = None) -> int:
    """顶点预算：默认按内容自适应（见 mesh.suggest_budget）；给了 --ppv 则走老口径。"""
    if budget:
        return int(budget)
    if ppv:
        return suggest_budget(image[:, :, 3], area_per_vertex=ppv)
    return suggest_budget(image[:, :, 3])


def draw_mesh(im: Image.Image, verts, tris, color, lw=1):
    d = ImageDraw.Draw(im)
    for a, b, c in np.asarray(tris):
        pts = [(float(verts[a][0]), float(verts[a][1])),
               (float(verts[b][0]), float(verts[b][1])),
               (float(verts[c][0]), float(verts[c][1]))]
        d.polygon(pts, outline=color)
    for x, y in np.asarray(verts):
        d.ellipse([x - lw, y - lw, x + lw, y + lw], fill=color)
    return im


def base_image(image: np.ndarray) -> Image.Image:
    """在白底上合成，方便肉眼看网格。"""
    im = Image.fromarray(image, "RGBA")
    bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
    bg.alpha_composite(im)
    return bg.convert("RGB")


def run_one(image: np.ndarray, budget: int, gt_verts=None, gt_tris=None, model=None):
    alpha = image[:, :, 3]
    vis = content_mask(alpha)
    res = model.mesh(image, budget) if model is not None else auto_mesh(alpha, budget)
    out = {
        "budget": int(budget),
        "n_vertices": res.n_vertices,
        "n_triangles": res.n_triangles,
        "auto": {"coverage": None, "spill": None},
    }
    if vis.any() and res.n_triangles:
        out["auto"]["coverage"] = round(float(coverage(res.vertices, res.triangles, vis)), 4)
        out["auto"]["spill"] = round(float(spill(res.vertices, res.triangles, vis)), 4)
    if gt_verts is not None and gt_tris is not None and len(gt_tris) and vis.any():
        out["modeler"] = {
            "n_vertices": int(len(gt_verts)),
            "coverage": round(float(coverage(gt_verts, gt_tris, vis)), 4),
            "spill": round(float(spill(gt_verts, gt_tris, vis)), 4),
        }
    return res, out


def render(image: np.ndarray, res, out_png: Path, gt_verts=None, gt_tris=None, title=""):
    im = base_image(image)
    if gt_verts is not None and gt_tris is not None and len(gt_tris):
        draw_mesh(im, gt_verts, gt_tris, (230, 40, 40), 1)
    draw_mesh(im, res.vertices, res.triangles, (0, 150, 0), 1)
    im.save(out_png)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="autogrid", description="图像 + 网格顶点自动排布")
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("--out", default=None)
    ap.add_argument("--ppv", type=float, default=None, help="老口径：每个顶点摊到的像素数（越小越密；不给则按内容自适应）")
    ap.add_argument("--budget", type=int, default=None, help="直接指定顶点总数（覆盖 --ppv）")
    ap.add_argument("--limit", type=int, default=0, help="每个输入最多处理多少个部件")
    ap.add_argument("--min-visible", type=int, default=16, help="可见像素少于此数的部件跳过")
    ap.add_argument("--model", default=None, help="用训练好的风格模型布点（给 .pt 路径）；不给就用纯几何算法")
    ap.add_argument("--no-depth", action="store_true", help="用 --model 时关掉深度通道")
    args = ap.parse_args(argv)

    style = None
    if args.model:
        from .smart import StyleModel
        style = StyleModel(args.model, use_depth=(False if args.no_depth else None))

    files: list[Path] = []
    for s in args.inputs:
        p = Path(s)
        if p.is_dir():
            files += sorted(p.glob("*.png")) + sorted(p.glob("*.cmo3"))
        else:
            files.append(p)
    if not files:
        print("没有找到输入文件"); return 1

    outdir = Path(args.out) if args.out else (files[0].parent / "autogrid_out")
    outdir.mkdir(parents=True, exist_ok=True)
    print("输出目录：%s" % outdir)

    tiles = []
    total = 0
    for f in files:
        if f.suffix.lower() == ".cmo3":
            from .cmo3 import load_parts
            parts = load_parts(f, min_visible=args.min_visible)
            if args.limit:
                parts = parts[:args.limit]
            print("%s：%d 个部件" % (f.name, len(parts)))
            for k, q in enumerate(parts):
                b = budget_for(q.image, args.budget, args.ppv)
                res, rep = run_one(q.image, b, q.uv_px, q.triangles, style)
                stem = "%s_%03d_%s" % (f.stem, k, "".join(ch for ch in q.name if ch.isalnum() or ch in "-_") or "part")
                render(q.image, res, outdir / (stem + ".overlay.png"), q.uv_px, q.triangles)
                rep["method"] = "style-model" if style is not None else "geometry"
                rep.update({"source": f.name, "part": q.name, "texture": q.texture,
                            "size": [int(q.image.shape[1]), int(q.image.shape[0])]})
                rep["vertices"] = [[round(float(a), 3), round(float(b2), 3)] for a, b2 in res.vertices]
                rep["triangles"] = [[int(a), int(b2), int(c)] for a, b2, c in res.triangles]
                (outdir / (stem + ".json")).write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
                tiles.append((stem, q.image, q.uv_px, q.triangles, res))
                total += 1
                if k % 50 == 0:
                    print("   ... %d/%d" % (k, len(parts)))
        else:
            im = Image.open(f).convert("RGBA")
            arr = np.asarray(im)
            b = budget_for(arr, args.budget, args.ppv)
            res, rep = run_one(arr, b, model=style)
            render(arr, res, outdir / (f.stem + ".overlay.png"))
            rep.update({"source": f.name, "part": f.stem, "size": [int(arr.shape[1]), int(arr.shape[0])]})
            rep["vertices"] = [[round(float(a), 3), round(float(b2), 3)] for a, b2 in res.vertices]
            rep["triangles"] = [[int(a), int(b2), int(c)] for a, b2, c in res.triangles]
            (outdir / (f.stem + ".json")).write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
            tiles.append((f.stem, arr, None, None, res))
            total += 1

    if tiles:
        make_sheet(tiles, outdir / "sheet.png")
    print("完成：%d 个部件 -> %s" % (total, outdir))
    return 0


def make_sheet(tiles, out_png: Path, cell=None, cols=None):
    """把所有部件的缩略图拼一张总览图。部件多时自动缩小格子，避免出一张上亿像素的图。"""
    n = len(tiles)
    if cols is None:
        cols = 6 if n <= 120 else (12 if n <= 480 else 16)
    if cell is None:
        cell = 220 if n <= 120 else (150 if n <= 480 else 110)
    rows = (n + cols - 1) // cols
    sheet = Image.new("RGB", (cols * cell, rows * cell), (255, 255, 255))
    for i, (name, img, gv, gt, res) in enumerate(tiles):
        im = base_image(img)
        if gv is not None and gt is not None and len(gt):
            draw_mesh(im, gv, gt, (230, 40, 40), 1)
        draw_mesh(im, res.vertices, res.triangles, (0, 150, 0), 1)
        im.thumbnail((cell - 8, cell - 8))
        sheet.paste(im, ((i % cols) * cell + 4, (i // cols) * cell + 4))
    sheet.save(out_png)


if __name__ == "__main__":
    sys.exit(main())
