# -*- coding: utf-8 -*-
"""用训练好的「风格模型」布点：输入 RGBA 图 → 输出 MeshResult。

预处理与训练时（autogrid.style_data）逐字一致：
  画布 384x384 保长宽比居中留边；10 个输入通道；顶点热力图 → 取点 → 交给几何后处理。
模型的职责只是「决定点放哪里」，三角形仍由 autogrid.mesh 生成，所以不会崩。
"""
from __future__ import annotations

import numpy as np
from PIL import Image

from .lines import edge_energy, line_mask
from .mesh import ALPHA_THRESHOLD, content_mask, mesh_from_points, suggest_budget

CANVAS = 384
DIST_NORM = 48.0
N_CHANNELS = 10
DEPTH_MODEL = "depth-anything/Depth-Anything-V2-Small-hf"

_DEPTH_CACHE = {}


def _depth_net(device):
    if "model" not in _DEPTH_CACHE:
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation
        proc = AutoImageProcessor.from_pretrained(DEPTH_MODEL, local_files_only=True)
        net = AutoModelForDepthEstimation.from_pretrained(DEPTH_MODEL, local_files_only=True)
        net = net.to(device).eval()
        _DEPTH_CACHE["model"], _DEPTH_CACHE["proc"] = net, proc
    return _DEPTH_CACHE["model"], _DEPTH_CACHE["proc"]


def depth_native(rgba, device="cuda", max_side=512):
    """原生尺寸的 8bit 深度图（口径与批量生成时一致：长边 <=512、按自身 min/max 归一）。"""
    import torch
    net, proc = _depth_net(device)
    H, W = rgba.shape[:2]
    rgb = Image.fromarray(rgba[:, :, :3]).convert("RGB")
    sc = max_side / float(max(W, H))
    tw, th = max(1, int(round(W * sc))), max(1, int(round(H * sc)))
    inputs = proc(images=[rgb], return_tensors="pt").to(device)
    with torch.no_grad():
        out = net(**inputs).predicted_depth
    d = torch.nn.functional.interpolate(out[:1].unsqueeze(1), size=(th, tw),
                                        mode="bicubic", align_corners=False)[0, 0].cpu().numpy()
    lo, hi = float(d.min()), float(d.max())
    dn = (d - lo) / max(hi - lo, 1e-6)
    im = Image.fromarray((dn * 255.0 + 0.5).astype(np.uint8))
    if (tw, th) != (W, H):
        im = im.resize((W, H), Image.BILINEAR)
    return np.asarray(im).astype(np.uint8)


def to_canvas(rgba, C=CANVAS):
    """缩到画布并返回 (canvas_rgba, ox, oy, tw, th)。"""
    H, W = rgba.shape[:2]
    sc = C / float(max(H, W))
    tw, th = max(1, int(round(W * sc))), max(1, int(round(H * sc)))
    ox, oy = (C - tw) // 2, (C - th) // 2
    small = Image.fromarray(rgba, "RGBA").resize((tw, th), Image.BILINEAR)
    canvas = Image.new("RGBA", (C, C), (0, 0, 0, 0))
    canvas.paste(small, (ox, oy))
    return np.asarray(canvas), ox, oy, tw, th


def build_input(canvas_rgba, depth_canvas, nv, spacing):
    """10 通道输入，与 style_data.load_sample 的通道口径一致。"""
    from scipy import ndimage as ndi
    C = canvas_rgba.shape[0]
    rgb = canvas_rgba[:, :, :3]
    alpha = canvas_rgba[:, :, 3]
    content = alpha > ALPHA_THRESHOLD
    dist = ndi.distance_transform_edt(content).astype(np.float32) / DIST_NORM
    np.clip(dist, 0.0, 1.0, out=dist)
    edge = edge_energy(rgb).astype(np.float32)
    peak = float(np.percentile(edge[content], 99.0)) if content.any() else 1.0
    edge = np.clip(edge / max(peak, 1e-6), 0.0, 1.0)
    line = line_mask(rgb, alpha).astype(np.float32)
    if depth_canvas is None:
        depth = np.zeros((C, C), np.float32)
    else:
        depth = np.asarray(depth_canvas).astype(np.float32) / 255.0
        depth = np.where(content, depth, 0.0)
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
    return img


class StyleModel:
    r"""训练好的布点模型。用法：
        m = StyleModel(r"...\style_unet.pt")
        res = m.mesh(rgba)          # -> MeshResult
    """

    def __init__(self, ckpt, device=None, use_depth=None, verbose=True):
        import torch
        from .nets import build
        self.torch = torch
        cuda = torch.cuda.is_available()
        self.device = device or ("cuda" if cuda else "cpu")
        sd = torch.load(ckpt, map_location=self.device)
        targs = sd.get("args", {}) or {}
        self.net = build(in_ch=N_CHANNELS, base=int(targs.get("base", 32)))
        self.net = self.net.to(self.device).eval()
        self.net.load_state_dict(sd["model"])
        if use_depth is None:
            use_depth = not bool(targs.get("no_depth", False))
        self.use_depth = bool(use_depth)
        self.ckpt = str(ckpt)
        if verbose:
            print("风格模型：%s  (device=%s, 深度通道=%s)" % (ckpt, self.device, "开" if self.use_depth else "关"))

    @property
    def torch_amp(self):
        return self.torch.autocast("cuda", dtype=self.torch.bfloat16, enabled=(self.device == "cuda"))

    def depth_canvas(self, rgba, ox, oy, tw, th):
        """给画布算深度通道；失败就退回全 0（模型仍能跑，只是少了这一路信息）。"""
        if not self.use_depth:
            return None
        try:
            dn = depth_native(rgba, self.device)
        except Exception as e:      # 没装 transformers / 没下模型
            print("   （深度图不可用，改用全 0 通道：%s）" % type(e).__name__)
            self.use_depth = False
            return None
        C = CANVAS
        dep = np.zeros((C, C), np.uint8)
        small = Image.fromarray(dn).resize((tw, th), Image.BILINEAR)
        dep[oy:oy + th, ox:ox + tw] = np.asarray(small)
        return dep

    def points(self, rgba, budget=None):
        """返回 (原生像素坐标的顶点 (N,2), 预算)。"""
        from .style_eval import extract_points
        canvas, ox, oy, tw, th = to_canvas(rgba)
        content = canvas[:, :, 3] > ALPHA_THRESHOLD
        area = float(content.sum())
        nv = int(budget) if budget else int(suggest_budget(rgba[:, :, 3]))
        spacing = float(np.sqrt(max(area, 1.0) / max(nv, 1)))
        dep = self.depth_canvas(rgba, ox, oy, tw, th)
        img = build_input(canvas, dep, nv, spacing)
        with self.torch.no_grad(), self.torch_amp:
            out = self.net(self.torch.from_numpy(img[None]).to(self.device))
        hm = self.torch.sigmoid(out.float())[0, 0].cpu().numpy()
        pc = extract_points(hm, nv, max(2.0, spacing * 0.45))
        H, W = rgba.shape[:2]
        if len(pc) == 0:
            return np.zeros((0, 2), np.float32), nv
        pts = np.stack([(pc[:, 0] - ox) * (W / tw), (pc[:, 1] - oy) * (H / th)], 1)
        return pts.astype(np.float32), nv

    def mesh(self, rgba, budget=None, *, cover=True, cover_extra=1.6):
        pts, nv = self.points(rgba, budget)
        return mesh_from_points(pts, rgba[:, :, 3], cover=cover, cover_extra=cover_extra)
