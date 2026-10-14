"""Behavior tests for the ArtMesh smoke trainer.

These import scripts/train.py by path so the script stays runnable without
making scripts/ a package.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
TRAIN_PATH = ROOT / "scripts" / "train.py"


def load_train():
    spec = importlib.util.spec_from_file_location("artmesh_train", TRAIN_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {TRAIN_PATH}")
    module = importlib.util.module_from_spec(spec)
    # dataclasses looks up the class module in sys.modules during class creation.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def write_pair(directory: Path, stem: str, size, vertices, rgb=(20, 40, 60, 255)):
    directory.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGBA", (size[0], size[1]), rgb)
    image.save(directory / f"{stem}.png")
    payload = {
        "id": stem,
        "texture": "3",
        "size": [size[0], size[1]],
        "part": "",
        "vertices": vertices,
        "triangles": [],
        "notes": {},
    }
    (directory / f"{stem}.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )


class TrainModuleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.train = load_train()

    def test_discover_pairs_merges_roles_and_skips_empty_vertices(self):
        train = self.train
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_pair(root / "lerelia", "mesh_a", (10, 20), [0, 0, 10, 20])
            write_pair(root / "猫娘", "mesh_b", (8, 8), [1, 2, 3, 4, 5, 6])
            write_pair(root / "lerelia", "empty", (8, 8), [])
            write_pair(root / "lerelia", "odd", (8, 8), [1, 2, 3])
            (root / "lerelia" / "orphan.json").write_text(
                json.dumps({"id": "orphan", "vertices": [0, 0]}),
                encoding="utf-8",
            )
            Image.new("RGBA", (4, 4), (0, 0, 0, 255)).save(
                root / "lerelia" / "no_json.png"
            )

            pairs = train.discover_pairs(root)

        ids = sorted((pair.role, pair.sample_id) for pair in pairs)
        self.assertEqual(ids, [("lerelia", "mesh_a"), ("猫娘", "mesh_b")])

    def test_letterbox_maps_vertices_onto_unit_canvas(self):
        import torch

        train = self.train
        # 4x2 image on a 32 canvas: uniform scale 8, pasted at the top-left.
        # Bottom-right source corner (4, 2) lands at normalized (1, 0.5).
        image = Image.new("RGBA", (4, 2), (10, 20, 30, 40))
        tensor, verts = train.prepare_sample(
            image, [[0, 0], [4, 2], [2, 1]], canvas=32
        )
        self.assertEqual(tuple(tensor.shape), (4, 32, 32))
        # Pillow bilinear does not keep a flat color bit-exact, but the pasted
        # block stays near the source RGBA and the padded rows stay empty.
        source = torch.tensor([10, 20, 30, 40], dtype=torch.float32) / 255.0
        content = tensor[:, :16, :]
        self.assertLess(float((content.mean(dim=(1, 2)) - source).abs().max()), 0.02)
        self.assertEqual(float(tensor[:, 16:, :].abs().sum()), 0.0)
        self.assertAlmostEqual(float(verts[0, 0]), 0.0, places=5)
        self.assertAlmostEqual(float(verts[0, 1]), 0.0, places=5)
        self.assertAlmostEqual(float(verts[1, 0]), 1.0, places=5)
        self.assertAlmostEqual(float(verts[1, 1]), 0.5, places=5)
        self.assertAlmostEqual(float(verts[2, 0]), 0.5, places=5)
        self.assertAlmostEqual(float(verts[2, 1]), 0.25, places=5)
        self.assertTrue(float(verts.min()) >= 0.0)
        self.assertTrue(float(verts.max()) <= 1.0)

    def test_split_holds_out_about_ten_percent_and_is_seeded(self):
        train = self.train
        pairs = [f"id-{i}" for i in range(10)]
        train_pairs, val_pairs = train.split_pairs(pairs, val_ratio=0.1, seed=7)
        self.assertEqual(len(train_pairs), 9)
        self.assertEqual(len(val_pairs), 1)
        self.assertEqual(set(train_pairs) | set(val_pairs), set(pairs))
        again_train, again_val = train.split_pairs(pairs, val_ratio=0.1, seed=7)
        self.assertEqual(train_pairs, again_train)
        self.assertEqual(val_pairs, again_val)
        only_train, no_val = train.split_pairs(["solo"], val_ratio=0.1, seed=0)
        self.assertEqual(only_train, ["solo"])
        self.assertEqual(no_val, [])

    def test_default_model_is_a_few_million_parameters_and_emits_density_and_points(self):
        import torch

        train = self.train
        model = train.DensityVertexNet()
        count = sum(parameter.numel() for parameter in model.parameters())
        self.assertGreaterEqual(count, 1_000_000)
        self.assertLess(count, 8_000_000)
        images = torch.rand(2, 4, 128, 128)
        density, points = model(images)
        self.assertEqual(tuple(density.shape), (2, 1, 128, 128))
        self.assertEqual(tuple(points.shape), (2, model.num_points, 2))
        self.assertGreaterEqual(float(points.detach().min()), 0.0)
        self.assertLessEqual(float(points.detach().max()), 1.0)

    def test_chamfer_is_zero_for_identical_sets_and_raster_marks_points(self):
        import torch

        train = self.train
        points = torch.tensor([[[0.0, 0.0], [1.0, 1.0]], [[0.25, 0.75], [0.25, 0.75]]])
        lengths = torch.tensor([2, 1])
        loss = train.chamfer_loss(points, points, lengths)
        self.assertAlmostEqual(float(loss), 0.0, places=6)

        shifted = points.clone()
        shifted[..., 0] = (shifted[..., 0] + 0.2).clamp(0, 1)
        moved = train.chamfer_loss(shifted, points, lengths)
        self.assertGreater(float(moved), 0.0)

        target = train.rasterize_points(points, lengths, size=8, radius=0)
        self.assertEqual(tuple(target.shape), (2, 1, 8, 8))
        self.assertEqual(float(target[0, 0, 0, 0]), 1.0)
        self.assertEqual(float(target[0, 0, 7, 7]), 1.0)
        self.assertEqual(float(target[1, 0].sum()), 1.0)

    def test_one_step_updates_weights_with_finite_loss(self):
        import torch

        train = self.train
        torch.manual_seed(0)
        model = train.DensityVertexNet(num_points=4, base_channels=8)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
        images = torch.rand(2, 4, 32, 32)
        vertices = torch.rand(2, 5, 2)
        lengths = torch.tensor([5, 3])
        before = torch.cat([parameter.detach().reshape(-1) for parameter in model.parameters()])
        stats = train.optimize_batch(
            model,
            optimizer,
            images,
            vertices,
            lengths,
            chamfer_weight=1.0,
            bce_weight=1.0,
        )
        after = torch.cat([parameter.detach().reshape(-1) for parameter in model.parameters()])
        self.assertTrue(torch.isfinite(torch.tensor(stats["loss"])))
        self.assertGreater(float((before - after).abs().sum()), 0.0)

    def test_cli_help_and_synthetic_run_write_a_checkpoint(self):
        help_run = subprocess.run(
            [sys.executable, str(TRAIN_PATH), "--help"],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(help_run.returncode, 0, help_run.stderr)
        self.assertIn("--epochs", help_run.stdout)
        self.assertIn("--batch-size", help_run.stdout)
        self.assertIn("--data", help_run.stdout)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = root / "dataset"
            for index in range(4):
                write_pair(
                    data / ("lerelia" if index % 2 == 0 else "猫娘"),
                    f"mesh_{index}",
                    (6, 10),
                    [0, 0, 6, 5, 3, 10, 1, 2],
                )
            out = root / "runs"
            completed = subprocess.run(
                [
                    sys.executable,
                    str(TRAIN_PATH),
                    "--data",
                    str(data),
                    "--epochs",
                    "1",
                    "--batch-size",
                    "2",
                    "--image-size",
                    "32",
                    "--num-points",
                    "4",
                    "--base-channels",
                    "8",
                    "--out",
                    str(out),
                    "--seed",
                    "1",
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr + completed.stdout)
            checkpoint = out / "checkpoint.pt"
            metrics = out / "metrics.json"
            self.assertTrue(checkpoint.is_file(), completed.stdout)
            self.assertTrue(metrics.is_file())
            payload = json.loads(metrics.read_text(encoding="utf-8"))
            self.assertEqual(payload["train_pairs"] + payload["val_pairs"], 4)
            self.assertGreaterEqual(payload["val_pairs"], 1)
            self.assertEqual(len(payload["epochs"]), 1)
            self.assertTrue(payload["epochs"][0]["train_loss"] == payload["epochs"][0]["train_loss"])


if __name__ == "__main__":
    unittest.main()
