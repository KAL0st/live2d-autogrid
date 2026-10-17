# -*- coding: utf-8 -*-
"""双击「自动布点.bat」时跑的入口。

没给参数：读本目录下的「输入」文件夹里的 .cmo3 / .png，结果写到「输出」。
给了参数（拖拽文件/文件夹到 bat 上）：把这些路径当输入。
"""
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from autogrid.cli import main   # noqa: E402


def run():
    args = sys.argv[1:]
    # 「智能布点.bat」会额外传一个 --model，表示用训练好的风格模型
    smart = "--model" in args
    args = [a for a in args if a != "--model"]
    extra = ["--model", str(HERE / "_out" / "style_unet.pt")] if smart else []
    out = str(HERE / "输出")
    if args:
        return main(args + ["--out", out] + extra)
    ind = HERE / "输入"
    ind.mkdir(exist_ok=True)
    found = sorted(ind.glob("*.cmo3")) + sorted(ind.glob("*.png"))
    if not found:
        print("「输入」文件夹里还没有东西。")
        print("把 .cmo3 或 .png 复制进去，再双击一次这个 bat：")
        print("   " + str(ind))
        return 1
    if smart:
        print("用「学你风格」的模型布点……（第一张图会慢十几秒：要加载深度估计模型）")
    return main([str(ind), "--out", out] + extra)


if __name__ == "__main__":
    sys.exit(run())
