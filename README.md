# live2d-autogrid

后续独立项目：**自动布点 + 识纹理**（从 Live2DReforge 迁出，2026-10-14）。

## 目标
- 输出：**ArtMesh 顶点**（不是变形器控制点）
- 数据：全部为卡师傅**自有**素材
- 路线：算法布点为主；小模型只学疏密/边缘；**不**训端到端整网

## 分割入口（脚本方案，实现待开工）
- **A（推荐，有 moc3/cmo3）**：`scripts/split_atlas_uv.py --model … --atlas … --out dataset/`
  - UV → 图集 AABB（可 pad）→ 裁 PNG+alpha → 顶点落到局部像素 → 写规范 json
  - 可对接重构仓现有 `meshsplit analyze`
- **B（仅大图）**：`scripts/split_atlas_cc.py --atlas … --out dataset/`
  - 连通域裁块；无顶点时 `vertices: []` 待后标
  - 可对接 `imgcut`

## 标注字段
一对样本 = `part_id.png`（PNG+alpha）+ `part_id.json`：

| 字段 | 必填 | 说明 |
|---|---|---|
| id / texture / size / part | 是 | part 词表：hair/face/body/cloth/acc |
| vertices | 是 | 像素坐标，原点左上，与 PNG 1:1 |
| triangles / density_hints / notes | 否 | |

硬规则：一部件一文件；顶点在不透明区内；先 50 对试跑再扩 200–500。

## 推荐栈
- 软件：Python + PyTorch/OpenCV；落地 ONNX Runtime
- 硬件：8–12GB 显存起步
- 参数量：分割约 5M–30M；疏密约 1M–10M

## 与重构仓关系
- 重构侧快照：`D:\live2d重构\docs\36-网格自动布点方案.md`（已标迁出）
- **不要**再往重构 README/docs 堆本主题

## 目录
- `dataset/` 训练对
- `scripts/` 分割脚本（占位）与 `train.py` 冒烟训练
- `docs/` 方案正文（可从 Notion / docs/36 快照迁入）

## 冒烟训练

`scripts/train.py` 在已经切好的 `dataset/<角色>/*.png` 与同名 json 上做小模型试跑。各角色样本会合并，空的 `vertices` 会被跳过，再按约 90/10 划分训练和验证。图像按长边缩放到正方形画布（默认 128，左上角对齐后填充），顶点使用同一次缩放，归一化到 `[0, 1]`。

网络是一个小 CNN（默认约 1.8M 参数）：密度图头对着由顶点栅格化出来的目标做 BCE，同时预测固定 K 个点，用双向 Chamfer 去贴近 ArtMesh 顶点。它只学习疏密和点的分布，不是端到端整网。

```bash
python -m pip install -r requirements.txt
python scripts/train.py --data dataset --epochs 2 --batch-size 8
```

权重和每一轮的损失写到 `runs/checkpoint.pt` 与 `runs/metrics.json`。检测到 CUDA 时自动使用，否则用 CPU。常用参数：`--image-size 128`、`--num-points 64`、`--lr 1e-3`、`--val-ratio 0.1`、`--out runs`。
