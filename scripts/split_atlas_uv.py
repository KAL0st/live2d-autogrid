# -*- coding: utf-8 -*-
"""Entry A: UV AABB crop from cmo3 (embedded atlas) → dataset/ part_id.png + part_id.json.

Does NOT require side PNGs — textures are extracted from the cmo3 via the
local export_dataset helper (links Live2DReforge Cmo3Document / PngCodec).

Usage:
  python split_atlas_uv.py --model path/to/model.cmo3 --out ../dataset/charname/
  python split_atlas_uv.py --model path/to/model.cmo3 --out ../dataset/charname/ --list 20 --pad 2

--atlas is accepted for API compatibility with the README stub but ignored
(atlas pages live inside the cmo3).
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "_tools"
EXE = TOOLS / "export_dataset.exe"
QT_BIN = Path(r"D:\Qt\6.8.3\msvc2022_64\bin")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Entry A UV-AABB split → PNG+JSON training pairs")
    ap.add_argument("--model", required=True, help="Input .cmo3 (atlas embedded)")
    ap.add_argument("--out", required=True, help="Output directory for part_id.png/.json")
    ap.add_argument("--atlas", default=None, help="Ignored (compat); atlas is inside cmo3")
    ap.add_argument("--pad", type=int, default=2, help="UV AABB padding in pixels (default 2)")
    ap.add_argument("--list", type=int, default=0, dest="list_n",
                    help="Export only first N meshes (0 = all)")
    args = ap.parse_args(argv)

    model = Path(args.model)
    out = Path(args.out)
    if not model.is_file():
        print(f"ERROR: model not found: {model}", file=sys.stderr)
        return 1
    if not EXE.is_file():
        print(
            f"ERROR: missing {EXE}\n"
            f"  Build it with: {TOOLS / 'build_export_dataset.bat'}",
            file=sys.stderr,
        )
        return 1

    out.mkdir(parents=True, exist_ok=True)
    cmd = [str(EXE), str(model), str(out), "--pad", str(args.pad)]
    if args.list_n > 0:
        cmd += ["--list", str(args.list_n)]
    if args.atlas:
        print(f"note: --atlas ignored (embedded textures used): {args.atlas}")

    env = os.environ.copy()
    if QT_BIN.is_dir():
        env["PATH"] = str(QT_BIN) + os.pathsep + env.get("PATH", "")

    print("running:", " ".join(cmd))
    r = subprocess.run(cmd, env=env)
    return r.returncode


if __name__ == "__main__":
    raise SystemExit(main())
