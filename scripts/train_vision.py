"""Pretrained vision backbone + rule baseline offsets (not density→points)."""
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
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ResNet18_Weights, resnet18

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train as T  # noqa: E402


def letterbox_rgba(image: Image.Image, canvas: int):
    rgba = image.convert("RGBA")
    width, height = rgba.size
    scale = canvas / float(max(width, height))
    new_w = max(1, min(canvas, int(round(width * scale))))
    new_h = max(1, min(canvas, int(round(height * scale))))
    resized = rgba.resize((new_w, new_h), Image.Resampling.BILINEAR)
    canvas_image = Image.new("RGBA", (canvas, canvas), (0, 0, 0, 0))
    canvas_image.paste(resized, (0, 0))
    return canvas_image, scale, new_w, new_h, width, height


def contour_baseline(alpha: np.ndarray, k: int) -> np.ndarray:
    """Uniform-ish samples along alpha silhouette + a few interior fills. Returns (K,2) in [0,1]."""
    h, w = alpha.shape
    mask = alpha > 0.15
    if not mask.any():
        # fallback grid
        xs = np.linspace(0.2, 0.8, int(np.sqrt(k)))
        ys = np.linspace(0.2, 0.8, int(np.ceil(k / max(len(xs), 1))))
        pts = np.array([(x, y) for y in ys for x in xs], dtype=np.float32)[:k]
        if len(pts) < k:
            pts = np.pad(pts, ((0, k - len(pts)), (0, 0)))
        return pts

    # Boundary pixels via simple erosion difference
    from scipy import ndimage  # optional

    try:
        eroded = ndimage.binary_erosion(mask, iterations=1)
        border = mask & ~eroded
    except Exception:
        # Pure numpy 4-neigh erosion
        eroded = np.zeros_like(mask)
        eroded[1:-1, 1:-1] = mask[1:-1, 1:-1] & mask[:-2, 1:-1] & mask[2:, 1:-1] & mask[1:-1, :-2] & mask[1:-1, 2:]
        border = mask & ~eroded

    ys, xs = np.where(border)
    if len(xs) == 0:
        ys, xs = np.where(mask)
    order = np.argsort(np.arctan2(ys - ys.mean(), xs - xs.mean()))
    xs, ys = xs[order], ys[order]
    n_border = max(1, int(k * 0.75))
    idx = np.linspace(0, len(xs) - 1, n_border).astype(int)
    border_pts = np.stack([xs[idx] / max(w - 1, 1), ys[idx] / max(h - 1, 1)], axis=1)

    n_in = k - len(border_pts)
    if n_in > 0:
        iy, ix = np.where(mask)
        pick = np.linspace(0, len(ix) - 1, n_in).astype(int) if len(ix) else np.zeros(n_in, dtype=int)
        if len(ix) == 0:
            interior = np.zeros((n_in, 2), dtype=np.float32)
        else:
            # stride sample for stability
            rng = np.random.RandomState(0)
            choose = rng.choice(len(ix), size=n_in, replace=len(ix) < n_in)
            interior = np.stack([ix[choose] / max(w - 1, 1), iy[choose] / max(h - 1, 1)], axis=1)
        pts = np.concatenate([border_pts, interior], axis=0)
    else:
        pts = border_pts[:k]
    if len(pts) < k:
        pts = np.pad(pts, ((0, k - len(pts)), (0, 0)))
    return pts.astype(np.float32)[:k]


def contour_baseline_numpy(alpha: np.ndarray, k: int) -> np.ndarray:
    """Contour baseline without scipy."""
    h, w = alpha.shape
    mask = alpha > 0.15
    if not mask.any():
        grid = int(np.ceil(np.sqrt(k)))
        xs = np.linspace(0.25, 0.75, grid)
        ys = np.linspace(0.25, 0.75, grid)
        pts = np.array([(x, y) for y in ys for x in xs], dtype=np.float32)[:k]
        if len(pts) < k:
            pts = np.concatenate([pts, np.zeros((k - len(pts), 2), dtype=np.float32)], 0)
        return pts
    eroded = np.zeros_like(mask, dtype=bool)
    eroded[1:-1, 1:-1] = (
        mask[1:-1, 1:-1]
        & mask[:-2, 1:-1]
        & mask[2:, 1:-1]
        & mask[1:-1, :-2]
        & mask[1:-1, 2:]
    )
    border = mask & ~eroded
    ys, xs = np.where(border)
    if len(xs) == 0:
        ys, xs = np.where(mask)
    ang = np.arctan2(ys - ys.mean(), xs - xs.mean())
    order = np.argsort(ang)
    xs, ys = xs[order], ys[order]
    n_border = max(1, int(round(k * 0.75)))
    idx = np.linspace(0, len(xs) - 1, n_border).astype(int)
    border_pts = np.stack([xs[idx] / max(w - 1, 1), ys[idx] / max(h - 1, 1)], axis=1).astype(np.float32)
    n_in = k - len(border_pts)
    if n_in > 0:
        iy, ix = np.where(mask)
        rng = np.random.RandomState(abs(hash(alpha.tobytes())) % (2**32))
        choose = rng.choice(len(ix), size=n_in, replace=len(ix) < n_in)
        interior = np.stack(
            [ix[choose] / max(w - 1, 1), iy[choose] / max(h - 1, 1)], axis=1
        ).astype(np.float32)
        pts = np.concatenate([border_pts, interior], axis=0)
    else:
        pts = border_pts
    if len(pts) < k:
        pts = np.concatenate([pts, np.zeros((k - len(pts), 2), dtype=np.float32)], 0)
    return pts[:k]


class VisionOffsetDataset(Dataset):
    def __init__(self, pairs, image_size: int, num_points: int):
        self.pairs = list(pairs)
        self.image_size = image_size
        self.num_points = num_points
        # ImageNet norm for RGB; keep alpha separate channel stacked later as 4ch via RGB only for backbone
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        pair = self.pairs[index]
        with Image.open(pair.image_path) as image:
            rgba = image.convert("RGBA")
        verts = T.load_vertices(pair.json_path)
        if verts is None:
            raise RuntimeError(pair.json_path)
        canvas, *_ = letterbox_rgba(rgba, self.image_size)
        arr = np.asarray(canvas, dtype=np.float32) / 255.0  # H,W,4
        alpha = arr[..., 3]
        baseline = contour_baseline_numpy(alpha, self.num_points)
        # GT normalized with same letterbox as train.prepare_sample
        _, gt = T.prepare_sample(rgba, verts, self.image_size)
        rgb = torch.from_numpy(arr[..., :3]).permute(2, 0, 1)
        rgb = (rgb - self.mean) / self.std
        alpha_t = torch.from_numpy(alpha).unsqueeze(0)
        # 4ch input: normalized rgb + raw alpha (backbone uses rgb; alpha for head)
        x = torch.cat([rgb, alpha_t], dim=0)
        return x, torch.from_numpy(baseline), gt, torch.tensor(gt.shape[0], dtype=torch.long)


def collate(batch):
    images = torch.stack([b[0] for b in batch])
    baseline = torch.stack([b[1] for b in batch])
    lengths = torch.stack([b[3] for b in batch])
    max_len = int(lengths.max()) if len(batch) else 0
    gt = images.new_zeros((len(batch), max_len, 2))
    for i, item in enumerate(batch):
        n = item[2].shape[0]
        if n:
            gt[i, :n] = item[2]
    return images, baseline, gt, lengths


class VisionOffsetNet(nn.Module):
    def __init__(self, num_points: int = 64, freeze_backbone: bool = True):
        super().__init__()
        self.num_points = num_points
        weights = ResNet18_Weights.IMAGENET1K_V1
        backbone = resnet18(weights=weights)
        # Drop FC; keep through layer4
        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        if freeze_backbone:
            for p in list(self.stem.parameters()) + list(self.layer1.parameters()) + list(self.layer2.parameters()):
                p.requires_grad = False
            # fine-tune deeper a bit
            for p in list(self.layer3.parameters()) + list(self.layer4.parameters()):
                p.requires_grad = True
        feat_dim = 512
        self.alpha_conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Sequential(
            nn.Linear(feat_dim + 16 + num_points * 2, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_points * 2),
            nn.Tanh(),  # residual in [-1,1], scaled small
        )
        self.offset_scale = 0.15

    def forward(self, images: torch.Tensor, baseline: torch.Tensor):
        rgb = images[:, :3]
        alpha = images[:, 3:4]
        x = self.stem(rgb)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        glob = F.adaptive_avg_pool2d(x, 1).flatten(1)
        a = self.alpha_conv(alpha).flatten(1)
        inp = torch.cat([glob, a, baseline.reshape(baseline.shape[0], -1)], dim=1)
        delta = self.head(inp).view(-1, self.num_points, 2) * self.offset_scale
        points = (baseline + delta).clamp(0.0, 1.0)
        return points, delta


def chamfer(pred, target, lengths):
    return T.chamfer_loss(pred, target, lengths)


def run_epoch(model, loader, optimizer, device, train: bool):
    model.train(train)
    total = 0.0
    seen = 0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for images, baseline, gt, lengths in loader:
            images = images.to(device)
            baseline = baseline.to(device)
            gt = gt.to(device)
            lengths = lengths.to(device)
            if train:
                optimizer.zero_grad(set_to_none=True)
            pred, _ = model(images, baseline)
            loss = chamfer(pred, gt, lengths)
            if train:
                loss.backward()
                optimizer.step()
            bs = images.shape[0]
            total += float(loss.detach()) * bs
            seen += bs
    return total / max(seen, 1)


def draw_overlay(rgba, gt_xy, base_xy, pred_xy, canvas: int) -> Image.Image:
    canvas_image, *_ = letterbox_rgba(rgba, canvas)
    rgb = Image.new("RGB", (canvas, canvas), (20, 20, 24))
    rgb.paste(canvas_image, mask=canvas_image.split()[-1])
    draw = ImageDraw.Draw(rgb)

    def paint(pts, color, r=2):
        if pts is None or len(pts) == 0:
            return
        for x, y in pts:
            px = float(x) * (canvas - 1)
            py = float(y) * (canvas - 1)
            draw.ellipse((px - r, py - r, px + r, py + r), fill=color)

    _, gt_norm = T.prepare_sample(rgba, gt_xy, canvas)
    paint(base_xy, (80, 160, 255), 1)  # blue = rule baseline
    paint(gt_norm.numpy(), (80, 220, 80), 2)  # green GT
    paint(pred_xy, (240, 70, 70), 2)  # red pred
    return rgb


def save_previews(model, pairs, args, device, out_dir: Path, count=8):
    out_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    picks = list(pairs[:count]) if len(pairs) <= count else random.Random(args.seed).sample(list(pairs), count)
    ds = VisionOffsetDataset(picks, args.image_size, args.num_points)
    paths = []
    with torch.no_grad():
        for i, pair in enumerate(picks):
            x, baseline, gt, length = ds[i]
            images = x.unsqueeze(0).to(device)
            base = baseline.unsqueeze(0).to(device)
            pred, _ = model(images, base)
            with Image.open(pair.image_path) as image:
                rgba = image.convert("RGBA")
            verts = T.load_vertices(pair.json_path)
            preview = draw_overlay(
                rgba,
                verts,
                baseline.numpy(),
                pred[0].cpu().numpy(),
                args.image_size,
            )
            dest = out_dir / f"{pair.role}__{pair.sample_id}.png"
            preview.save(dest)
            paths.append(str(dest))
    return paths


def main():
    p = argparse.ArgumentParser(description="ResNet18 pretrained + contour baseline offsets")
    p.add_argument("--data", type=Path, default=Path("dataset"))
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--image-size", type=int, default=128)
    p.add_argument("--num-points", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--val-ratio", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=Path("runs"))
    p.add_argument("--preview-dir", type=Path, default=Path("runs/preview3"))
    p.add_argument("--preview-count", type=int, default=8)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--device", default="auto")
    p.add_argument("--unfreeze-all", action="store_true")
    args = p.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    pairs = T.discover_pairs(args.data)
    train_pairs, val_pairs = T.split_pairs(pairs, args.val_ratio, args.seed)
    print(f"pairs={len(pairs)} train={len(train_pairs)} val={len(val_pairs)}", flush=True)
    print("logic=ImageNet ResNet18 (frozen early) + alpha-contour baseline + offset head → Chamfer vs GT", flush=True)

    device = T.resolve_device(args.device)
    model = VisionOffsetNet(num_points=args.num_points, freeze_backbone=not args.unfreeze_all).to(device)
    params = [x for x in model.parameters() if x.requires_grad]
    optimizer = torch.optim.Adam(params, lr=args.lr)
    print(f"device={device} trainable={sum(x.numel() for x in params)}", flush=True)

    gen = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        VisionOffsetDataset(train_pairs, args.image_size, args.num_points),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        collate_fn=collate,
        generator=gen,
    )
    val_loader = DataLoader(
        VisionOffsetDataset(val_pairs, args.image_size, args.num_points),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        collate_fn=collate,
    )

    history = []
    best_val = float("inf")
    best_state = None
    best_epoch = 0
    stale = 0
    for epoch in range(1, args.epochs + 1):
        tr = run_epoch(model, train_loader, optimizer, device, True)
        va = run_epoch(model, val_loader, optimizer, device, False)
        history.append({"epoch": epoch, "train_chamfer": tr, "val_chamfer": va})
        print(f"epoch {epoch}/{args.epochs} train_chamfer={tr:.6f} val_chamfer={va:.6f}", flush=True)
        if va + 1e-6 < best_val:
            best_val = va
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
    ckpt = out / "checkpoint_vision.pt"
    metrics = {
        "variant": "resnet18_contour_offset",
        "best_epoch": best_epoch,
        "best_val_chamfer": best_val,
        "stopped_epoch": history[-1]["epoch"],
        "epochs": history,
        "legend": {"blue": "rule baseline", "green": "GT", "red": "baseline+offset"},
    }
    torch.save({"model_state": model.state_dict(), "metrics": metrics}, ckpt)
    (out / "metrics_vision.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {ckpt}", flush=True)
    paths = save_previews(model, val_pairs, args, device, Path(args.preview_dir), args.preview_count)
    print(f"previews={len(paths)} dir={args.preview_dir}", flush=True)
    for path in paths:
        print(f"preview {path}", flush=True)


if __name__ == "__main__":
    main()
