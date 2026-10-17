"""Smoke-train a small CNN on Live2D ArtMesh crops (PNG + JSON).

Merges every role folder under the data root, holds out about 10% for
validation, and fits a density map plus a fixed set of points. This is a
smoke path for vertex density, not an end-to-end mesh generator.

Example (from the repo root):
    python scripts/train.py --data dataset --epochs 2 --batch-size 8
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset


@dataclass(frozen=True)
class Pair:
    image_path: Path
    json_path: Path
    role: str
    sample_id: str


def _as_xy(vertices) -> np.ndarray | None:
    try:
        array = np.asarray(vertices, dtype=np.float32)
    except (TypeError, ValueError):
        return None
    if array.size == 0:
        return np.zeros((0, 2), dtype=np.float32)
    if array.ndim == 1:
        if array.size % 2 != 0:
            return None
        array = array.reshape(-1, 2)
    if array.ndim != 2 or array.shape[1] != 2:
        return None
    if not np.isfinite(array).all():
        return None
    return np.ascontiguousarray(array, dtype=np.float32)


def load_vertices(json_path: Path) -> np.ndarray | None:
    try:
        payload = json.loads(Path(json_path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    coords = _as_xy(payload.get("vertices", []))
    if coords is None or coords.shape[0] == 0:
        return None
    return coords


def discover_pairs(root: Path) -> list[Pair]:
    """Find PNG+JSON pairs with at least one vertex under role folders."""
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"data directory not found: {root}")

    search_dirs: list[tuple[Path, str]] = []
    if any(root.glob("*.json")):
        search_dirs.append((root, root.name))
    search_dirs.extend((path, path.name) for path in sorted(root.iterdir()) if path.is_dir())

    pairs: list[Pair] = []
    for directory, role in search_dirs:
        for json_path in sorted(directory.glob("*.json")):
            image_path = json_path.with_suffix(".png")
            if not image_path.is_file():
                continue
            if load_vertices(json_path) is None:
                continue
            pairs.append(
                Pair(
                    image_path=image_path,
                    json_path=json_path,
                    role=role,
                    sample_id=json_path.stem,
                )
            )
    pairs.sort(key=lambda pair: (pair.role, pair.sample_id, str(pair.json_path)))
    return pairs


def prepare_sample(image: Image.Image, vertices, canvas: int):
    """Letterbox an RGBA crop to a square canvas and map vertices to [0, 1].

    The crop keeps its aspect ratio, is scaled from the long side, and is
    pasted at the top-left. Vertex pixels use that same scale.
    """
    if canvas < 1:
        raise ValueError(f"canvas must be positive, got {canvas}")
    rgba = image.convert("RGBA")
    width, height = rgba.size
    if width < 1 or height < 1:
        raise ValueError("image has no pixels")

    scale = canvas / float(max(width, height))
    new_w = max(1, min(canvas, int(round(width * scale))))
    new_h = max(1, min(canvas, int(round(height * scale))))
    resized = rgba.resize((new_w, new_h), Image.Resampling.BILINEAR)
    canvas_image = Image.new("RGBA", (canvas, canvas), (0, 0, 0, 0))
    canvas_image.paste(resized, (0, 0))
    pixels = np.asarray(canvas_image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(pixels).permute(2, 0, 1).contiguous()

    coords = _as_xy(vertices)
    if coords is None:
        raise ValueError("vertices must be a flat even-length list or an Nx2 array")
    if coords.shape[0] == 0:
        points = torch.zeros((0, 2), dtype=torch.float32)
    else:
        xy = coords.copy()
        xy[:, 0] = np.clip(xy[:, 0] * (new_w / float(width)) / canvas, 0.0, 1.0)
        xy[:, 1] = np.clip(xy[:, 1] * (new_h / float(height)) / canvas, 0.0, 1.0)
        points = torch.from_numpy(np.ascontiguousarray(xy, dtype=np.float32))
    return tensor, points


def split_pairs(pairs, val_ratio: float, seed: int):
    """Shuffle pairs and hold out about val_ratio of them, keeping one train pair."""
    items = list(pairs)
    rng = random.Random(seed)
    rng.shuffle(items)
    count = len(items)
    if count <= 1 or val_ratio <= 0:
        return items, []
    val_count = int(round(count * float(val_ratio)))
    val_count = min(max(val_count, 1), count - 1)
    return items[val_count:], items[:val_count]


class ArtMeshDataset(Dataset):
    def __init__(self, pairs: list[Pair], image_size: int):
        self.pairs = list(pairs)
        self.image_size = int(image_size)

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int):
        pair = self.pairs[index]
        with Image.open(pair.image_path) as image:
            loaded = image.convert("RGBA")
        vertices = load_vertices(pair.json_path)
        if vertices is None:
            raise RuntimeError(f"no vertices in {pair.json_path}")
        return prepare_sample(loaded, vertices, self.image_size)


def collate_batch(batch):
    images = torch.stack([item[0] for item in batch], dim=0)
    lengths = torch.tensor([item[1].shape[0] for item in batch], dtype=torch.long)
    max_len = int(lengths.max()) if len(batch) else 0
    vertices = images.new_zeros((len(batch), max_len, 2))
    for index, (_, points) in enumerate(batch):
        count = points.shape[0]
        if count:
            vertices[index, :count] = points
    return images, vertices, lengths


def _groups(channels: int) -> int:
    return 8 if channels % 8 == 0 else 1


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False),
            nn.GroupNorm(_groups(out_channels), out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(_groups(out_channels), out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DensityVertexNet(nn.Module):
    """CNN encoder with a full-resolution density head and K normalized points."""

    def __init__(self, in_channels: int = 4, num_points: int = 64, base_channels: int = 32):
        super().__init__()
        if base_channels < 1:
            raise ValueError("base_channels must be positive")
        if num_points < 1:
            raise ValueError("num_points must be positive")
        width = base_channels
        self.num_points = num_points
        self.base_channels = width
        self.stem = ConvBlock(in_channels, width, stride=1)
        self.down = nn.ModuleList(
            [
                ConvBlock(width, width * 2, stride=2),
                ConvBlock(width * 2, width * 4, stride=2),
                ConvBlock(width * 4, width * 8, stride=2),
                ConvBlock(width * 8, width * 8, stride=2),
            ]
        )
        self.up = nn.ModuleList(
            [
                UpBlock(width * 8, width * 4),
                UpBlock(width * 4, width * 2),
                UpBlock(width * 2, width),
                UpBlock(width, width),
            ]
        )
        self.density_head = nn.Conv2d(width, 1, kernel_size=1)
        self.vertex_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(width * 8, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_points * 2),
        )

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.stem(images)
        for layer in self.down:
            features = layer(features)
        density = features
        for layer in self.up:
            density = layer(density)
        if density.shape[-2:] != images.shape[-2:]:
            density = F.interpolate(
                density, size=images.shape[-2:], mode="bilinear", align_corners=False
            )
        density_logits = self.density_head(density)
        points = torch.sigmoid(self.vertex_head(features)).view(-1, self.num_points, 2)
        return density_logits, points


def chamfer_loss(pred: torch.Tensor, target: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    """Mean bidirectional squared Chamfer distance. `lengths` masks padded targets."""
    lengths = lengths.to(device=pred.device, dtype=torch.long)
    target = target.to(device=pred.device, dtype=pred.dtype)
    count = target.shape[1]
    if count == 0:
        return pred.new_zeros(())
    slots = torch.arange(count, device=pred.device).unsqueeze(0)
    mask = slots < lengths.unsqueeze(1)
    dist2 = torch.cdist(pred, target, p=2).square()
    dist2 = dist2.masked_fill(~mask[:, None, :], float("inf"))
    pred_to_target = dist2.min(dim=2).values
    target_to_pred = dist2.min(dim=1).values
    target_to_pred = torch.where(mask, target_to_pred, torch.zeros_like(target_to_pred))
    denom = mask.sum(dim=1).clamp(min=1).to(pred.dtype)
    per_sample = pred_to_target.mean(dim=1) + target_to_pred.sum(dim=1) / denom
    return per_sample.mean()


def rasterize_points(
    vertices: torch.Tensor,
    lengths: torch.Tensor,
    size: int,
    radius: int = 1,
) -> torch.Tensor:
    """Paint normalized points into a {0, 1} density target of shape (B, 1, size, size)."""
    lengths = lengths.to(device=vertices.device, dtype=torch.long)
    batch = vertices.shape[0]
    target = vertices.new_zeros((batch, 1, size, size))
    if vertices.shape[1] == 0:
        return target
    xs = (vertices[..., 0] * (size - 1)).round().long().clamp(0, size - 1)
    ys = (vertices[..., 1] * (size - 1)).round().long().clamp(0, size - 1)
    for index in range(batch):
        count = int(lengths[index])
        if count <= 0:
            continue
        target[index, 0, ys[index, :count], xs[index, :count]] = 1.0
    if radius > 0:
        kernel = radius * 2 + 1
        target = F.max_pool2d(target, kernel_size=kernel, stride=1, padding=radius)
    return target


def density_bce(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    positives = target.sum().clamp(min=1.0)
    negatives = (target.numel() - target.sum()).clamp(min=1.0)
    pos_weight = (negatives / positives).clamp(1.0, 20.0).reshape(1)
    return F.binary_cross_entropy_with_logits(logits, target, pos_weight=pos_weight)


def compute_losses(model, images, vertices, lengths, chamfer_weight: float, bce_weight: float, point_radius: int):
    density_logits, pred_points = model(images)
    chamfer = chamfer_loss(pred_points, vertices, lengths)
    target = rasterize_points(vertices, lengths, images.shape[-1], radius=point_radius)
    bce = density_bce(density_logits, target)
    loss = chamfer_weight * chamfer + bce_weight * bce
    return {"loss": loss, "chamfer": chamfer, "bce": bce}


def optimize_batch(
    model,
    optimizer,
    images,
    vertices,
    lengths,
    chamfer_weight: float,
    bce_weight: float,
    point_radius: int = 1,
):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    stats = compute_losses(
        model, images, vertices, lengths, chamfer_weight, bce_weight, point_radius
    )
    stats["loss"].backward()
    optimizer.step()
    return {name: float(value.detach()) for name, value in stats.items()}


def _move_batch(batch, device: torch.device):
    images, vertices, lengths = batch
    return images.to(device), vertices.to(device), lengths.to(device)


def run_epoch(model, loader, optimizer, device, args, train: bool) -> dict[str, float]:
    model.train(train)
    totals = {"loss": 0.0, "chamfer": 0.0, "bce": 0.0}
    seen = 0
    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for batch in loader:
            images, vertices, lengths = _move_batch(batch, device)
            if train:
                optimizer.zero_grad(set_to_none=True)
            stats = compute_losses(
                model,
                images,
                vertices,
                lengths,
                args.chamfer_weight,
                args.bce_weight,
                args.point_radius,
            )
            if train:
                stats["loss"].backward()
                optimizer.step()
            batch_size = images.shape[0]
            seen += batch_size
            for name in totals:
                totals[name] += float(stats[name].detach()) * batch_size
    if seen == 0:
        return {name: 0.0 for name in totals}
    return {name: value / seen for name, value in totals.items()}


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but torch.cuda.is_available() is false")
    return torch.device(name)


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Smoke-train a small CNN on Live2D ArtMesh PNG+JSON crops. "
            "Predicts a vertex density map (BCE) and a fixed set of points (Chamfer)."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog="Example: python scripts/train.py --data dataset --epochs 2 --batch-size 8",
    )
    parser.add_argument("--data", type=Path, default=Path("dataset"), help="root with role folders of PNG+JSON pairs")
    parser.add_argument("--epochs", type=int, default=2, help="training epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="batch size")
    parser.add_argument("--image-size", type=int, default=128, help="square canvas; multiple of 16")
    parser.add_argument("--num-points", type=int, default=64, help="fixed number of predicted points")
    parser.add_argument("--base-channels", type=int, default=32, help="CNN width; 32 is about 1.8M parameters")
    parser.add_argument("--lr", type=float, default=1e-3, help="Adam learning rate")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="fraction of pairs held out for validation")
    parser.add_argument("--seed", type=int, default=0, help="split and initialization seed")
    parser.add_argument("--out", type=Path, default=Path("runs"), help="checkpoint and metrics directory")
    parser.add_argument("--chamfer-weight", type=float, default=1.0, help="weight on bidirectional Chamfer loss")
    parser.add_argument("--bce-weight", type=float, default=1.0, help="weight on density BCE loss")
    parser.add_argument("--point-radius", type=int, default=1, help="density-target splat radius in pixels")
    parser.add_argument("--workers", type=int, default=0, help="DataLoader worker count")
    parser.add_argument("--device", default="auto", help="auto, cpu, or cuda")
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> dict:
    if args.epochs < 1:
        raise SystemExit("--epochs must be >= 1")
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be >= 1")
    if args.image_size < 16 or args.image_size % 16 != 0:
        raise SystemExit(f"--image-size must be a multiple of 16, got {args.image_size}")
    if args.num_points < 1:
        raise SystemExit("--num-points must be >= 1")
    if not 0.0 <= args.val_ratio < 1.0:
        raise SystemExit("--val-ratio must be in [0, 1)")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    pairs = discover_pairs(args.data)
    if not pairs:
        raise SystemExit(
            f"no PNG+JSON pairs with vertices under {args.data}. "
            "Expected dataset/<role>/*.png and a matching .json."
        )
    train_pairs, val_pairs = split_pairs(pairs, args.val_ratio, args.seed)
    role_counts = dict(sorted(Counter(pair.role for pair in pairs).items()))
    print(
        f"pairs={len(pairs)} train={len(train_pairs)} val={len(val_pairs)} roles={role_counts}",
        flush=True,
    )

    device = resolve_device(args.device)
    model = DensityVertexNet(
        in_channels=4,
        num_points=args.num_points,
        base_channels=args.base_channels,
    ).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    print(f"device={device} parameters={parameter_count}", flush=True)

    generator = torch.Generator()
    generator.manual_seed(args.seed)
    train_loader = DataLoader(
        ArtMeshDataset(train_pairs, args.image_size),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        collate_fn=collate_batch,
        generator=generator,
    )
    val_loader = None
    if val_pairs:
        val_loader = DataLoader(
            ArtMeshDataset(val_pairs, args.image_size),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            collate_fn=collate_batch,
        )

    history = []
    for epoch in range(1, args.epochs + 1):
        train_stats = run_epoch(model, train_loader, optimizer, device, args, train=True)
        val_stats = (
            run_epoch(model, val_loader, optimizer, device, args, train=False)
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
            f"epoch {epoch}/{args.epochs} "
            f"train_loss={row['train_loss']:.6f} val_loss={val_text} "
            f"train_chamfer={row['train_chamfer']:.6f} train_bce={row['train_bce']:.6f}",
            flush=True,
        )

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    checkpoint_path = out / "checkpoint.pt"
    metrics = {
        "train_pairs": len(train_pairs),
        "val_pairs": len(val_pairs),
        "parameters": parameter_count,
        "roles": role_counts,
        "epochs": history,
    }
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "epoch": args.epochs,
            "model_kwargs": {
                "in_channels": 4,
                "num_points": args.num_points,
                "base_channels": args.base_channels,
            },
            "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
            "metrics": metrics,
        },
        checkpoint_path,
    )
    metrics_path = out / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {checkpoint_path} and {metrics_path}", flush=True)
    return metrics


def main(argv=None) -> None:
    run(parse_args(argv))


if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError as exc:
        print(exc, file=sys.stderr)
        sys.exit(1)
