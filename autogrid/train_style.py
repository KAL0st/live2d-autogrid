# -*- coding: utf-8 -*-
"""训「学建模师风格」的布点模型（热力图回归）。

用法:
    python -m autogrid.train_style --epochs 30 --batch 8
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from autogrid.nets import build
from autogrid.style_data import (CANVAS, N_CHANNELS, SIGMA, StyleDataset,
                                 index_cache, split_items)
from autogrid.style_eval import chamfer, extract_points, online_rate


def log(msg, path):
    line = "[%s] %s" % (time.strftime("%H:%M:%S"), msg)
    print(line, flush=True)
    if path:
        with open(path, "a", encoding="utf-8") as f:
            f.write(line + "\n")


def make_loss(sigma=SIGMA, w_pos=20.0, w_count=0.1):
    k = 2.0 * np.pi * sigma * sigma

    def fn(logits, target):
        bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        w = 1.0 + w_pos * target
        l_heat = (bce * w).mean()
        prob = torch.sigmoid(logits)
        cnt_t = target.sum(dim=(1, 2, 3)) / k
        cnt_p = prob.sum(dim=(1, 2, 3)) / k
        l_cnt = (torch.log1p(cnt_p) - torch.log1p(cnt_t)).abs().mean()
        return l_heat + w_count * l_cnt, float(l_heat.detach()), float(l_cnt.detach())

    return fn


@torch.no_grad()
def evaluate(model, items, *, no_depth, eval_n, device, seed=0, radius=2.0):
    model.eval()
    rng = np.random.RandomState(seed)
    idx = np.arange(len(items))
    if eval_n and eval_n < len(idx):
        idx = rng.choice(idx, eval_n, replace=False)
    ds = StyleDataset([items[i] for i in idx], augment=False, use_depth=not no_depth)
    ch_a, ch_b, on, nin, npts, ncore, gtn = [], [], [], [], [], [], []
    dv = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)
    for i, (img, heat, nv) in enumerate(dv):
        img = img.to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(device == "cuda")):
            out = model(img)
        p = torch.sigmoid(out.float())[0, 0].cpu().numpy()
        it = ds.items[i]
        _, png, dpng, _, _ = it
        from autogrid.style_data import load_sample
        _, _, meta = load_sample(png, dpng, it[0], augment=False, use_depth=not no_depth)
        gt = np.stack([meta["verts"][:, 0], meta["verts"][:, 1]], 1)
        n = int(meta["nv"])
        md = max(2.0, meta["spacing"] * 0.45)
        pred = extract_points(p, n, md)
        a, b = chamfer(pred, gt, meta["spacing"])
        ch_a.append(a); ch_b.append(b)
        # 压线率
        from PIL import Image
        from scipy import ndimage as ndi
        rgba = np.asarray(Image.open(png).convert("RGBA"))
        content = rgba[:, :, 3] > 4
        core = ndi.distance_transform_edt(content) > 3.0
        from autogrid.lines import line_mask
        lm = line_mask(rgba[:, :, :3], rgba[:, :, 3])
        r, ni = online_rate(pred, lm, content, core, radius)
        npts.append(len(pred)); ncore.append(int(core.sum())); gtn.append(n)
        if not np.isnan(r):
            on.append(r); nin.append(ni)
    model.train()
    return dict(chamfer_fwd=float(np.nanmean(ch_a)) if ch_a else float("nan"),
                chamfer_bwd=float(np.nanmean(ch_b)) if ch_b else float("nan"),
                online=float(np.mean(on)) if on else float("nan"),
                n_eval=len(ch_a),
                n_core=int(np.sum(nin)),
                mean_pred_pts=float(np.mean(npts)) if npts else 0.0,
                mean_gt_pts=float(np.mean(gtn)) if gtn else 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=r"D:\live2d-autogrid\缓存")
    ap.add_argument("--out", default=r"D:\live2d-autogrid\_out\style_unet.pt")
    ap.add_argument("--log", default=r"D:\live2d-autogrid\_tmp\train_log.txt")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--base", type=int, default=32)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--holdout", default="lerelia")
    ap.add_argument("--no-depth", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--eval-every", type=int, default=3)
    ap.add_argument("--eval-n", type=int, default=48)
    ap.add_argument("--resume", default="")
    ap.add_argument("--tag", default="run")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(0)

    items = index_cache(args.cache)
    if not items:
        raise SystemExit("缓存为空: " + args.cache)
    train_it, val_it = split_items(items, holdout_models=(args.holdout,), val_frac=0.05)
    if args.limit:
        train_it = train_it[:args.limit]
        val_it = val_it[:max(8, args.limit // 20)]
    log("样本 训练%d 验证%d（留出模型 %s）| 画布 %d | 深度 %s | 设备 %s"
        % (len(train_it), len(val_it), args.holdout, CANVAS,
           "off" if args.no_depth else "on", device), args.log)

    ds = StyleDataset(train_it, augment=True, use_depth=not args.no_depth)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=args.workers,
                    drop_last=True, persistent_workers=(args.workers > 0), pin_memory=True)

    model = build(in_ch=N_CHANNELS, base=args.base).to(device)
    if args.resume and os.path.exists(args.resume):
        sd = torch.load(args.resume, map_location=device)
        model.load_state_dict(sd["model"] if "model" in sd else sd)
        log("续训自 " + args.resume, args.log)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    steps = max(1, len(dl)) * args.epochs
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps, pct_start=0.1)
    loss_fn = make_loss()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    best = 1e9
    gstep = 0
    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        model.train()
        agg = np.zeros(4)
        nb = 0
        for img, heat, nv in dl:
            img = img.to(device, non_blocking=True)
            heat = heat.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(device == "cuda")):
                out = model(img)
                loss, lh, lc = loss_fn(out.float(), heat)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step(); sched.step()
            gstep += 1; nb += 1
            agg += [float(loss.detach()), lh, lc, float(out.detach().float().max())]
            if nb % 100 == 0:
                el = time.time() - t0
                log("epoch %d/%d  step %d/%d  loss %.4f (热力 %.4f 计数 %.4f)  lr %.2e  %.0fs"
                    % (ep, args.epochs, nb, len(dl), agg[0]/nb, agg[1]/nb, agg[2]/nb,
                       sched.get_last_lr()[0], el), args.log)
        log("== epoch %d 完成  平均 loss %.4f  用时 %.1f 分钟" % (ep, agg[0]/max(nb, 1), (time.time()-t0)/60), args.log)

        if ep % args.eval_every == 0 or ep == args.epochs:
            try:
                m = evaluate(model, val_it, no_depth=args.no_depth, eval_n=args.eval_n, device=device)
                log("   验证  n=%d  预测->他 %.4f  他->预测 %.4f  压线率 %.1f%%  (平均出点 %.0f 个 / 他 %.0f 个, 内部点 %d 个)"
                    % (m["n_eval"], m["chamfer_fwd"], m["chamfer_bwd"], 100*m["online"],
                       m["mean_pred_pts"], m["mean_gt_pts"], m["n_core"]), args.log)
            except Exception as e:
                import traceback
                m = {"chamfer_fwd": float("nan")}
                log("   验证失败: %s\n%s" % (e, traceback.format_exc()), args.log)
            score = m["chamfer_fwd"]
            torch.save(dict(model=model.state_dict(), args=vars(args), canvas=CANVAS,
                            sigma=SIGMA, metrics=m), args.out.replace(".pt", "_last.pt"))
            if score == score and score < best:
                best = score
                torch.save(dict(model=model.state_dict(), args=vars(args), canvas=CANVAS,
                                sigma=SIGMA, metrics=m), args.out)
                log("   保存新最优 -> %s (%.4f)" % (args.out, best), args.log)
    log("训练结束 best=%.4f 用时 %.1f 分钟" % (best, (time.time()-t0)/60), args.log)
    print("TRAIN_DONE", flush=True)


if __name__ == "__main__":
    main()
