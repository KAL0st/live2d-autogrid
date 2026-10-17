"""Longer smoke train with early stopping + val point previews."""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader

# Allow `python scripts/run_longer.py` from repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import train as T  # noqa: E402


def draw_overlay(image_rgba: Image.Image, gt_xy: np.ndarray, pred_xy: np.ndarray, canvas: int) -> Image.Image:
    """Letterbox image to canvas and paint GT (lime) + pred (red) points."""
    tensor, _ = T.prepare_sample(image_rgba, gt_xy.reshape(-1).tolist(), canvas)
    # Rebuild letterboxed RGBA for drawing (same as prepare_sample).
    rgba = image_rgba.convert("RGBA")
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

    def paint(points: np.ndarray, color, radius: int = 2):
        if points is None or len(points) == 0:
            return
        for x, y in points:
            px = float(x) * (canvas - 1)
            py = float(y) * (canvas - 1)
            draw.ellipse((px - radius, py - radius, px + radius, py + radius), fill=color)

    # GT in letterbox-normalized coords
    _, gt_norm = T.prepare_sample(image_rgba, gt_xy.reshape(-1).tolist(), canvas)
    paint(gt_norm.numpy(), (80, 220, 80), radius=2)
    paint(pred_xy, (240, 70, 70), radius=2)
    return rgb


def save_previews(model, val_pairs, args, device, out_dir: Path, count: int = 8) -> list[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    picks = list(val_pairs[:count]) if len(val_pairs) <= count else random.Random(args.seed).sample(val_pairs, count)
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
            _, pred = model(images)
            pred_xy = pred[0].detach().cpu().numpy()
            preview = draw_overlay(rgba, verts, pred_xy, args.image_size)
            dest = out_dir / f"{pair.role}__{pair.sample_id}.png"
            preview.save(dest)
            paths.append(str(dest))
    return paths


def main():
    parser = argparse.ArgumentParser(description="Longer smoke train with early stop + previews")
    parser.add_argument("--data", type=Path, default=Path("dataset"))
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=8, help="early-stop patience on val_loss")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--num-points", type=int, default=64)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("runs"))
    parser.add_argument("--preview-dir", type=Path, default=Path("runs/preview"))
    parser.add_argument("--preview-count", type=int, default=8)
    parser.add_argument("--chamfer-weight", type=float, default=1.0)
    parser.add_argument("--bce-weight", type=float, default=1.0)
    parser.add_argument("--point-radius", type=int, default=1)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    # Reuse train.Namespace-compatible object for run_epoch
    ns = argparse.Namespace(**vars(args))

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    pairs = T.discover_pairs(args.data)
    if not pairs:
        raise SystemExit(f"no pairs under {args.data}")
    train_pairs, val_pairs = T.split_pairs(pairs, args.val_ratio, args.seed)
    print(f"pairs={len(pairs)} train={len(train_pairs)} val={len(val_pairs)}", flush=True)

    device = T.resolve_device(args.device)
    model = T.DensityVertexNet(
        in_channels=4, num_points=args.num_points, base_channels=args.base_channels
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    print(f"device={device} parameters={sum(p.numel() for p in model.parameters())}", flush=True)

    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        T.ArtMeshDataset(train_pairs, args.image_size),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        collate_fn=T.collate_batch,
        generator=generator,
    )
    val_loader = DataLoader(
        T.ArtMeshDataset(val_pairs, args.image_size),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        collate_fn=T.collate_batch,
    ) if val_pairs else None

    history = []
    best_val = float("inf")
    best_state = None
    best_epoch = 0
    stale = 0

    for epoch in range(1, args.epochs + 1):
        train_stats = T.run_epoch(model, train_loader, optimizer, device, ns, train=True)
        val_stats = (
            T.run_epoch(model, val_loader, optimizer, device, ns, train=False)
            if val_loader is not None
            else None
        )
        row = {
            "epoch": epoch,
            "train_loss": train_stats["loss"],
            "train_chamfer": train_stats["chamfer"],
            "train_bce": train_stats["bce"],
            "val_loss": None if val_stats is None else val_stats["loss"],
            "val_chamfer": None if val_stats is None else val_stats["chamfer"],
            "val_bce": None if val_stats is None else val_stats["bce"],
        }
        history.append(row)
        val_text = "n/a" if row["val_loss"] is None else f"{row['val_loss']:.6f}"
        print(
            f"epoch {epoch}/{args.epochs} train_loss={row['train_loss']:.6f} "
            f"val_loss={val_text} train_chamfer={row['train_chamfer']:.6f} "
            f"train_bce={row['train_bce']:.6f}",
            flush=True,
        )

        if row["val_loss"] is not None:
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
    checkpoint_path = out / "checkpoint.pt"
    metrics = {
        "train_pairs": len(train_pairs),
        "val_pairs": len(val_pairs),
        "best_epoch": best_epoch,
        "best_val_loss": None if best_val == float("inf") else best_val,
        "stopped_epoch": history[-1]["epoch"] if history else 0,
        "epochs": history,
    }
    torch.save(
        {
            "model_state": model.state_dict(),
            "epoch": best_epoch,
            "model_kwargs": {
                "in_channels": 4,
                "num_points": args.num_points,
                "base_channels": args.base_channels,
            },
            "metrics": metrics,
        },
        checkpoint_path,
    )
    metrics_path = out / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {checkpoint_path} and {metrics_path}", flush=True)

    preview_paths = save_previews(model, val_pairs, args, device, Path(args.preview_dir), args.preview_count)
    print(f"previews={len(preview_paths)} dir={args.preview_dir}", flush=True)
    for path in preview_paths:
        print(f"preview {path}", flush=True)


if __name__ == "__main__":
    main()
