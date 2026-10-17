# -*- coding: utf-8 -*-
"""autogrid.cmo3 —— 从 .cmo3 里按 ArtMesh 导出「图像 + 网格顶点」配对（正确版）。

相对旧版 _tools/export_dataset.cpp 的两处关键修正：

  (1) 旧版只读 uvs 数组，把「UV 的像素化」当成了网格顶点；真正的网格顶点在同级的
      CArtMeshSource/positions（画布绝对坐标，与 CEditableMesh2/point 逐值相同）。

  (2) UV 到图像像素的正确换算是：  像素 = uv * 逻辑页宽
      其中 逻辑页宽 = 图像宽 / m00，m00 来自 GTexture2D/transformImageResource01toLogical01。
      图像只是贴在逻辑页的左上角，而 UV 是按整个逻辑页计的。旧版用 像素 = uv * 图像宽，
      于是网格被整体朝左上角压缩。m00 恰为 1 的图（宽度已经是 64 的整数倍）从来没错过 ——
      这正是"只有部分样本对不上"的原因。
"""
from __future__ import annotations

import io
from dataclasses import dataclass, field
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image

from .caff import parse as caff_parse, read_entry

DEFAULT_PAD = 2


@dataclass
class Part:
    """一个 ArtMesh 的「图像 + 网格」配对，图像与顶点严格同坐标系。"""

    name: str
    xs_id: int
    image: np.ndarray          # (H,W,4) uint8 RGBA；已按网格 UV 包围盒裁剪并透明补边
    uv_px: np.ndarray          # (N,2) 与 image 对齐的像素坐标
    triangles: np.ndarray      # (M,3) int32
    canvas: np.ndarray         # (N,2) 画布坐标（建模师真正的顶点）
    texture: str = ""
    page: tuple = (0, 0)       # 图像本身尺寸
    logical: tuple = (0, 0)    # 逻辑页尺寸 = 图像尺寸 / m00,m11
    origin: tuple = (0, 0)     # 裁剪框左上角在纹理像素中的位置
    meta: dict = field(default_factory=dict)

    @property
    def alpha(self) -> np.ndarray:
        return self.image[:, :, 3]

    def to_dict(self) -> dict:
        return {
            "id": self.name,
            "xsId": self.xs_id,
            "texture": self.texture,
            "size": [int(self.image.shape[1]), int(self.image.shape[0])],
            "page": [int(self.page[0]), int(self.page[1])],
            "logical": [round(float(self.logical[0]), 3), round(float(self.logical[1]), 3)],
            "origin": [int(self.origin[0]), int(self.origin[1])],
            "vertices": [round(float(v), 4) for v in self.uv_px.reshape(-1)],
            "triangles": [int(t) for t in self.triangles.reshape(-1)],
            "canvas": [round(float(v), 4) for v in self.canvas.reshape(-1)],
        }


def extract_main_xml(cmo3_path) -> str:
    data, key, entries = caff_parse(str(cmo3_path))
    for e in entries:
        if e["tag"] == "main_xml":
            return read_entry(data, key, e).decode("utf-8", "replace")
    raise ValueError("容器里没有 main_xml")


def _kids(el, name):
    return [c for c in el if c.attrib.get("xs.n") == name]


def _floats(el):
    return [float(x) for x in (el.text or "").split()]


def _ints(el):
    return [int(x) for x in (el.text or "").split()]


def _tex_name(gt) -> str:
    for c in gt.iter("s"):
        if c.attrib.get("xs.n") == "name":
            return (c.text or "").strip()
    return ""


def load_parts(cmo3_path, *, pad: int = DEFAULT_PAD, include_uv_bounds: bool = True,
               min_visible: int = 0) -> list[Part]:
    """读出 .cmo3 里每个 ArtMesh 的（图像, 顶点, 三角形）。

    include_uv_bounds=True 时图像会以透明像素补足到完整的 UV 包围盒，
    保证网格顶点一定落在图像内（否则 UV 越界部分会被裁掉）。
    """
    cmo3_path = Path(cmo3_path)
    data, key, entries = caff_parse(str(cmo3_path))
    xml_txt = extract_main_xml(cmo3_path)
    root = ET.fromstring(xml_txt)

    by_id = {el.attrib["xs.id"]: el for el in root.iter() if "xs.id" in el.attrib}
    tex_cache: dict = {}

    def decode_tex(path_hint, size, want_wh):
        ck = (path_hint, size)
        if ck in tex_cache:
            return tex_cache[ck]
        cands = [e for e in entries if e["size"] == size and e["path"] == path_hint]
        if not cands:
            cands = [e for e in entries if e["size"] == size]
        out = None
        for e in cands:
            try:
                b = read_entry(data, key, e)
            except Exception:
                continue
            if b[:8] != b"\x89PNG\r\n\x1a\n":
                continue
            try:
                im = Image.open(io.BytesIO(b)).convert("RGBA")
            except Exception:
                continue
            if want_wh and (im.width, im.height) != want_wh:
                continue
            out = np.asarray(im)
            break
        tex_cache[ck] = out
        return out

    parts: list[Part] = []
    for m in root.iter("CArtMeshSource"):
        sid = m.attrib.get("xs.id")
        if not sid:
            continue
        xs_id = int(sid.lstrip("#"))
        tr = _kids(m, "texture")
        ue = _kids(m, "uvs")
        ie = _kids(m, "indices")
        pe = _kids(m, "positions")
        if not (tr and ue and ie):
            continue
        gt = by_id.get(tr[0].attrib.get("xs.ref"))
        if gt is None:
            continue
        sr = _kids(gt, "srcImageResource")
        if not sr:
            continue
        res = by_id.get(sr[0].attrib.get("xs.ref"))
        if res is None or "width" not in res.attrib:
            continue
        W, H = int(res.attrib["width"]), int(res.attrib["height"])
        size = int(res.attrib.get("imageFileBuf_size", -1))
        path_hint = ""
        for f in res.iter("file"):
            if f.attrib.get("xs.n") == "imageFileBuf":
                path_hint = f.attrib.get("path", "")
        af = _kids(gt, "transformImageResource01toLogical01")
        m00 = float(af[0].attrib.get("m00", 1.0)) if af else 1.0
        m11 = float(af[0].attrib.get("m11", 1.0)) if af else 1.0
        if abs(m00) < 1e-9:
            m00 = 1.0
        if abs(m11) < 1e-9:
            m11 = 1.0
        WL, HL = W / m00, H / m11

        tex = decode_tex(path_hint, size, (W, H))
        if tex is None:
            continue
        uv = _floats(ue[0])
        us = np.asarray(uv[0::2], dtype=np.float64)
        vs = np.asarray(uv[1::2], dtype=np.float64)
        n = min(len(us), len(vs))
        us, vs = us[:n], vs[:n]
        idx = np.asarray(_ints(ie[0]), dtype=np.int64)
        if idx.size == 0 or idx.size % 3 or idx.max() >= n:
            continue
        tris = idx.reshape(-1, 3).astype(np.int32)

        if include_uv_bounds:
            x0 = int(np.floor(us.min() * WL)) - pad
            y0 = int(np.floor(vs.min() * HL)) - pad
            x1 = int(np.ceil(us.max() * WL)) + pad
            y1 = int(np.ceil(vs.max() * HL)) + pad
        else:
            x0 = max(0, int(np.floor(us.min() * WL)) - pad)
            y0 = max(0, int(np.floor(vs.min() * HL)) - pad)
            x1 = min(W, int(np.ceil(us.max() * WL)) + pad)
            y1 = min(H, int(np.ceil(vs.max() * HL)) + pad)
        Wc, Hc = max(1, x1 - x0), max(1, y1 - y0)
        img = np.zeros((Hc, Wc, 4), dtype=np.uint8)
        sx0, sy0 = max(0, x0), max(0, y0)
        sx1, sy1 = min(W, x1), min(H, y1)
        if sx1 > sx0 and sy1 > sy0:
            img[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = tex[sy0:sy1, sx0:sx1]
        if min_visible and int((img[:, :, 3] > 4).sum()) < min_visible:
            continue
        uv_px = np.stack([us * WL - x0, vs * HL - y0], axis=1)
        if pe:
            cv = np.asarray(_floats(pe[0]), dtype=np.float64)
            canvas = np.stack([cv[0::2], cv[1::2]], axis=1)[:n]
            if len(canvas) != n:
                canvas = np.zeros((n, 2))
        else:
            canvas = np.zeros((n, 2))
        parts.append(Part(
            name=m.attrib.get("name") or ("ArtMesh%d" % xs_id),
            xs_id=xs_id, image=img, uv_px=uv_px, triangles=tris, canvas=canvas,
            texture=_tex_name(gt), page=(W, H), logical=(WL, HL), origin=(x0, y0),
            meta={"m00": m00, "m11": m11},
        ))
    return parts
