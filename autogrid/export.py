# -*- coding: utf-8 -*-
"""导出训练集：python -m autogrid.export <模型.cmo3 ...> [--out 训练集]"""
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from autogrid.dataset import export_model, write_manifest


def main(argv):
    import argparse
    ap = argparse.ArgumentParser(prog="导出训练集")
    ap.add_argument("inputs", nargs="+", help=".cmo3 文件或目录")
    ap.add_argument("--out", default=str(HERE / "训练集"))
    ap.add_argument("--limit", type=int, default=0, help="每个模型最多导出多少个部件（调试用）")
    ap.add_argument("--min-area", type=int, default=100, help="可见像素少于此数的部件丢掉")
    a = ap.parse_args(argv)

    files = []
    for s in a.inputs:
        p = Path(s)
        if p.is_dir():
            files += sorted(p.glob("*.cmo3"))
        else:
            files.append(p)
    if not files:
        print("没找到 .cmo3")
        return 1

    all_s = []
    for f in files:
        print("=== %s（%.1f MB）===" % (f.name, f.stat().st_size / 1048576.0))
        s = export_model(f, a.out, limit=a.limit, min_area=a.min_area,
                         progress=lambda m: print("   " + m))
        print("   导出 %d 个样本" % len(s))
        all_s += s
    man = write_manifest(all_s, a.out)
    print("\n总计 %d 个样本 / %d 个模型 -> %s" % (man["n_samples"], man["n_models"], a.out))
    if man["coverage"]:
        print("建模师网格覆盖率：mean=%.4f 中位=%.4f p05=%.4f 最小=%.4f" %
              (man["coverage"]["mean"], man["coverage"]["median"], man["coverage"]["p05"], man["coverage"]["min"]))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
