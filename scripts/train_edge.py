"""Edge-weighted density + peak sampling (no center-regressed fixed points)."""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from torch import nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train as T  # noqa: E402


def alpha_edge_weight(images: torch.Tensor, dilate: int = 2) -> torch.Tensor:
    """Weight map from RGBA alpha edges. images: (B,4,H,W) in [0,1]."""
    alpha = images[:, 3:4]
    sobel_x = torch.tensor(
        [[[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]], dtype=images.dtype, device=images.device
    )
    sobel_y = torch.tensor(
        [[[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]]], dtype=images.dtype, device=images.device
    )
    gx = F.conv2d(alpha, sobel_x, padding=1)
    gy = F.conv2d(alpha, sobel_y, padding=1)
    edge = (gx.square() + gy.square()).sqrt()
    edge = edge / edge.amax(dim=(2, 3), keepdim=True).clamp(min=1e-6)
    if dilate > 0:
        k = dilate * 2 + 1
        edge = F.max_pool2d(edge, kernel_size=k, stride=1, padding=dilate)
    # Interior low, edges high; floor so empty regions still get some signal.
    return 0.25 + 1.75 * edge


def rasterize_edge_aware(vertices, lengths, size, images, radius: int = 1):
    target = T.rasterize_points(vertices, lengths, size, radius=radius)
    weight = alpha_edge_weight(images)
    # Boost GT pixels that sit on edges.
    return target, weight


def nms_peaks(density: torch.Tensor, k: int, kernel: int = 5) -> torch.Tensor:
    """Non-diff NMS peak pick. density (B,1,H,W) -> (B,K,2) in [0,1] xy."""
    b, _, h, w = density.shape
    pooled = F.max_pool2d(density, kernel_size=kernel, stride=1, padding=kernel // 2)
    peaks = density * (density == pooled).float()
    flat = peaks.view(b, -1)
    topv, topi = flat.topk(min(k, flat.shape[1]), dim=1)
    ys = (topi // w).float()
    xs = (topi % w).float()
    # If fewer real peaks, keep zeros for empties (masked later by score).
    xy = torch.stack([xs / max(w - 1, 1), ys / max(h - 1, 1)], dim=-1)
    if xy.shape[1] < k:
        pad = xy.new_zeros((b, k - xy.shape[1], 2))
        xy = torch.cat([xy, pad], dim=1)
        topv = F.pad(topv, (0, k - topv.shape[1]))
    return xy, topv


def soft_peak_points(logits: torch.Tensor, k: int, temperature: float = 0.5) -> torch.Tensor:
    """Differentiable-ish multi-peak sampling via repeated soft masking."""
    b, _, h, w = logits.shape
    device = logits.device
    probs = torch.sigmoid(logits)
    yy, xx = torch.meshgrid(
        torch.linspace(0, 1, h, device=device),
        torch.linspace(0, 1, w, device=device),
        indexing="ij",
    )
    grid = torch.stack([xx, yy], dim=-1).view(1, h * w, 2).expand(b, -1, -1)
    points = []
    work = probs.view(b, -1)
    for _ in range(k):
        weights = F.softmax(work / max(temperature, 1e-3), dim=1)
        pt = (weights.unsqueeze(-1) * grid).sum(dim=1)
        points.append(pt)
        # Suppress neighborhood in soft fashion.
        dist2 = (grid - pt.unsqueeze(1)).square().sum(dim=-1)
        work = work * (1.0 - torch.exp(-dist2 / 0.002))
    return torch.stack(points, dim=1)


class EdgeDensityNet(nn.Module):
    def __init__(self, in_channels: int = 4, base_channels: int = 32, num_points: int = 64):
        super().__init__()
        self.num_points = num_points
        self.inner = T.DensityVertexNet(
            in_channels=in_channels, num_points=num_points, base_channels=base_channels
        )

    def forward(self, images: torch.Tensor):
        density_logits, _hard_points = self.inner(images)
        # Replace center-regressed points with soft peaks from density.
        soft_points = soft_peak_points(density_logits, self.num_points)
        return density_logits, soft_points


def boundary_weights(vertices, lengths, images, size: int) -> torch.Tensor:
    """Per-GT-point weight: higher if near alpha edge."""
    edge = alpha_edge_weight(images)  # B,1,H,W
    b, n, _ = vertices.shape
    weights = vertices.new_ones((b, n))
    xs = (vertices[..., 0] * (size - 1)).round().long().clamp(0, size - 1)
    ys = (vertices[..., 1] * (size - 1)).round().long().clamp(0, size - 1)
    for i in range(b):
        count = int(lengths[i])
        if count <= 0:
            continue
        weights[i, :count] = edge[i, 0, ys[i, :count], xs[i, :count]].clamp(0.5, 2.0)
    return weights


def weighted_chamfer(pred, target, lengths, point_weights) -> torch.Tensor:
    lengths = lengths.to(device=pred.device, dtype=torch.long)
    target = target.to(device=pred.device, dtype=pred.dtype)
    point_weights = point_weights.to(device=pred.device, dtype=pred.dtype)
    count = target.shape[1]
    if count == 0:
        return pred.new_zeros(())
    slots = torch.arange(count, device=pred.device).unsqueeze(0)
    mask = slots < lengths.unsqueeze(1)
    dist2 = torch.cdist(pred, target, p=2).square()
    dist2 = dist2.masked_fill(~mask[:, None, :], float("inf"))
    pred_to_target = dist2.min(dim=2).values.mean(dim=1)
    nearest = dist2.min(dim=1)
    target_to_pred = nearest.values
    w = torch.where(mask, point_weights, torch.zeros_like(point_weights))
    denom = w.sum(dim=1).clamp(min=1e-6)
    target_to_pred = torch.where(mask, target_to_pred, torch.zeros_like(target_to_pred))
    weighted = (target_to_pred * w).sum(dim=1) / denom
    return (pred_to_target + weighted).mean()


def edge_bce(logits, target, edge_weight) -> torch.Tensor:
    # Element-wise weight: edges amplify both positives and the map overall.
    weight = 0.5 + edge_weight
    # Standard pos imbalance still handled lightly.
    positives = (target * weight).sum().clamp(min=1.0)
    negatives = ((1 - target) * weight).sum().clamp(min=1.0)
    pos_weight = (negatives / positives).clamp(1.0, 30.0).reshape(1)
    loss = F.binary_cross_entropy_with_logits(logits, target, reduction="none", pos_weight=pos_weight)
    loss = loss * weight
    return loss.mean()


def compute_losses(model, images, vertices, lengths, args):
    density_logits, pred_points = model(images)
    target, edge_w = rasterize_edge_aware(
        vertices, lengths, images.shape[-1], images, radius=args.point_radius
    )
    bce = edge_bce(density_logits, target, edge_w)
    pw = boundary_weights(vertices, lengths, images, images.shape[-1])
    chamfer = weighted_chamfer(pred_points, vertices, lengths, pw)
    loss = args.chamfer_weight * chamfer + args.bce_weight * bce
    return {"loss": loss, "chamfer": chamfer, "bce": bce}


def run_epoch(model, loader, optimizer, device, args, train: bool):
    model.train(train)
    totals = {"loss": 0.0, "chamfer": 0.0, "bce": 0.0}
    seen = 0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for batch in loader:
            images, vertices, lengths = T._move_batch(batch, device)
            if train:
                optimizer.zero_grad(set_to_none=True)
            stats = compute_losses(model, images, vertices, lengths, args)
            if train:
                stats["loss"].backward()
                optimizer.step()
            bs = images.shape[0]
            seen += bs
            for name in totals:
                totals[name] += float(stats[name].detach()) * bs
    if seen == 0:
        return {k: 0.0 for k in totals}
    return {k: v / seen for k, v in totals.items()}


def draw_overlay(rgba, gt_xy, pred_xy, canvas: int) -> Image.Image:
    width, height = rgba.size
    scale = canvas / float(max(width, height))
    new_w = max(1, min(canvas, int(round(width * scale))))
    new_h = max(1, min(canvas, int(round(height * scale))))
    resized = rgba.resize((new_w, new_h), Image.Resampling.BILINEAR)
    canvas_image = Image.new("RGBA", (canvas, canvas), (0, 0, 0, 0))
    canvas_image.paste(resized, (0, 0))
    rgb = Image.new("RGB", (canvas, canvas), (20, 20, 24))
    rgb.paste(canvas_image, mask=canvas_image.split()[-1])
    draw = ImageDraw.Draw(rgb)

    def paint(points, color, radius=2):
        if points is None or len(points) == 0:
            return
        for x, y in points:
            px = float(x) * (canvas - 1)
            py = float(y) * (canvas - 1)
            draw.ellipse((px - radius, py - radius, px + radius, py + radius), fill=color)

    _, gt_norm = T.prepare_sample(rgba, gt_xy, canvas)
    paint(gt_norm.numpy(), (80, 220, 80), 2)
    paint(pred_xy, (240, 70, 70), 2)
    return rgb


def save_previews(model, val_pairs, args, device, out_dir: Path, count: int = 8):
    out_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    picks = list(val_pairs[:count]) if len(val_pairs) <= count else random.Random(args.seed).sample(list(val_pairs), count)
    paths = []
    with torch.no_grad():
        for pair in picks:
            with Image.open(pair.image_path) as image:
                rgba = image.convert("RGBA")
            verts = T.load_vertices(pair.json_path)
            if verts is None:
                continue
            tensor, _ = T.prepare_sample(rgba, verts, args.image_size)
            images = tensor.unsqueeze(0).to(device)
            logits, soft_pts = model(images)
            # Prefer hard NMS peaks for preview readability.
            hard, _ = nms_peaks(torch.sigmoid(logits), args.num_points)
            pred = hard[0].cpu().numpy()
            # Drop near-zero peaks (empty pads).
            dens = torch.sigmoid(logits)[0, 0].cpu().numpy()
            keep = []
            h, w = dens.shape
            for x, y in pred:
                px = int(round(x * (w - 1)))
                py = int(round(y * (h - 1)))
                if dens[py, px] > 0.15:
                    keep.append([x, y])
            if not keep:
                keep = soft_pts[0].cpu().numpy().tolist()
            preview = draw_overlay(rgba, verts, np.asarray(keep, dtype=np.float32), args.image_size)
            dest = out_dir / f"{pair.role}__{pair.sample_id}.png"
            preview.save(dest)
            paths.append(str(dest))
    return paths


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, default=Path("dataset"))
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--patience", type=int, default=6)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--image-size", type=int, default=128)
    p.add_argument("--num-points", type=int, default=64)
    p.add_argument("--base-channels", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--val-ratio", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=Path("runs"))
    p.add_argument("--preview-dir", type=Path, default=Path("runs/preview2"))
    p.add_argument("--preview-count", type=int, default=8)
    p.add_argument("--chamfer-weight", type=float, default=1.0)
    p.add_argument("--bce-weight", type=float, default=1.5)
    p.add_argument("--point-radius", type=int, default=1)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--device", default="auto")
    args = p.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    pairs = T.discover_pairs(args.data)
    train_pairs, val_pairs = T.split_pairs(pairs, args.val_ratio, args.seed)
    print(f"pairs={len(pairs)} train={len(train_pairs)} val={len(val_pairs)}", flush=True)

    device = T.resolve_device(args.device)
    model = EdgeDensityNet(num_points=args.num_points, base_channels=args.base_channels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    print(f"device={device} parameters={sum(x.numel() for x in model.parameters())}", flush=True)

    gen = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        T.ArtMeshDataset(train_pairs, args.image_size),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        collate_fn=T.collate_batch,
        generator=gen,
    )
    val_loader = DataLoader(
        T.ArtMeshDataset(val_pairs, args.image_size),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        collate_fn=T.collate_batch,
    )

    history = []
    best_val = float("inf")
    best_state = None
    best_epoch = 0
    stale = 0

    for epoch in range(1, args.epochs + 1):
        tr = run_epoch(model, train_loader, optimizer, device, args, True)
        va = run_epoch(model, val_loader, optimizer, device, args, False)
        row = {
            "epoch": epoch,
            "train_loss": tr["loss"],
            "train_chamfer": tr["chamfer"],
            "train_bce": tr["bce"],
            "val_loss": va["loss"],
            "val_chamfer": va["chamfer"],
            "val_bce": va["bce"],
        }
        history.append(row)
        print(
            f"epoch {epoch}/{args.epochs} train_loss={row['train_loss']:.6f} "
            f"val_loss={row['val_loss']:.6f} train_chamfer={row['train_chamfer']:.6f} "
            f"train_bce={row['train_bce']:.6f}",
            flush=True,
        )
        if row["val_loss"] + 1e-6 < best_val:
            best_val = row["val_loss"]
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= args.patience:
                print(f"early stop at epoch {epoch} (best_epoch={best_epoch} best_val={best_val:.6f})", flush=True)
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    ckpt = out / "checkpoint_edge.pt"
    metrics = {
        "variant": "edge_peak",
        "best_epoch": best_epoch,
        "best_val_loss": best_val,
        "stopped_epoch": history[-1]["epoch"],
        "epochs": history,
    }
    torch.save({"model_state": model.state_dict(), "metrics": metrics, "model_kwargs": {
        "num_points": args.num_points, "base_channels": args.base_channels
    }}, ckpt)
    (out / "metrics_edge.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {ckpt}", flush=True)

    paths = save_previews(model, val_pairs, args, device, Path(args.preview_dir), args.preview_count)
    print(f"previews={len(paths)} dir={args.preview_dir}", flush=True)
    for path in paths:
        print(f"preview {path}", flush=True)


if __name__ == "__main__":
    main()
