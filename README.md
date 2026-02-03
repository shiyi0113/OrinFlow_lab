# OrinFlow Lab

OrinFlow 的 PC 端子项目。在开发机上将 YOLO26 系列模型优化为高性能、低精度损失的 ONNX 模型，用于后续传输到 Jetson Orin Nano Super 通过 TensorRT 部署推理。

## 优化路线

所有路线的最终产物都是 `.onnx`，在板端通过 `trtexec --int8` 构建 TensorRT engine 部署。

```
路线 1（PTQ）:       FP32.pt → export → FP32.onnx → calibrate → quantize → INT8.onnx (QDQ)
路线 2（QAT）:       FP32.pt → qat（校准 + QAT 微调 + 导出）→ INT8.onnx (QDQ)
路线 3（SAT + PTQ）: FP32.pt → sparsify（SAT 微调）→ sparse.pt → 路线 1 → INT8.onnx (QDQ + 稀疏)
路线 4（SAT + QAT）: FP32.pt → sparse-qat（SAT 微调 + QAT 微调 + 导出）→ INT8.onnx (QDQ + 稀疏)
```

- **路线 1** 最简单，适合大多数场景作为起点
- **路线 2** PTQ 精度损失不可接受时使用，需 GPU 训练
- **路线 3** 在路线 1 基础上叠加稀疏加速，板端需 `trtexec --sparsity=enable`
- **路线 4** 理论收益最大（稀疏 + 量化联合优化），实现复杂度最高，板端需 `trtexec --sparsity=enable`

## 命令一览

| 命令 | 说明 | 路线 |
|------|------|------|
| `orinflow-export` | PyTorch `.pt` → ONNX（含 simplify） | 1, 3 |
| `orinflow-calibrate` | 从数据集生成 `.npy` 校准数据 | 1, 3 |
| `orinflow-quantize` | INT8 / INT4 / FP8 后训练量化 (PTQ) | 1, 3 |
| `orinflow-qat` | 量化感知训练，直接输出带 QDQ 节点的 ONNX | 2 |
| `orinflow-sparsify` | SparseGPT / magnitude 剪枝 + SAT 微调 | 3 |
| `orinflow-sparse-qat` | 2:4 稀疏 + QAT 联合训练，输出带 QDQ + 稀疏的 ONNX | 4 |
| `orinflow-evaluate` | 精度 (mAP50/mAP50-95) 与性能基准，支持 PT↔ONNX 对比 | 全部 |

## 目录结构

```
OrinFlow_lab/
├── orinflow/                # 核心库
│   ├── config.py            # 路径与默认参数
│   ├── export/              # Stage 1: ONNX 导出
│   ├── data/                # Stage 2: 校准数据 & 预处理
│   ├── optimize/            # Stage 3: PTQ / QAT / 2:4 稀疏
│   └── evaluate/            # Stage 4: 精度与性能评估
├── scripts/                 # CLI 入口脚本
├── models/
│   ├── source/              # PyTorch 源模型 (.pt)
│   └── onnx/
│       ├── raw/             # 导出的原始 ONNX
│       └── optimized/       # 量化/稀疏后的 ONNX
├── datasets/                # 数据集 YAML 配置与校准文件
├── Dockerfile
└── pyproject.toml
```

## 安装

### Docker（推荐）

```bash
docker build -t orinflow .
docker run --gpus all -it -v $(pwd):/workspace orinflow

# 容器内
pip install -e ".[dev]"
```

## 使用示例

### 路线 1: PTQ

```bash
orinflow-export -m yolo26n.pt --imgsz 640 --opset 13
orinflow-calibrate --yaml coco128.yaml --split train --num-images 500
orinflow-quantize -m yolo26n.onnx --calib coco128.npy --mode int8 --method entropy
orinflow-evaluate -m yolo26n --yaml coco128.yaml --compare
```

### 路线 2: QAT

```bash
orinflow-qat -m yolo26n.pt --data coco128.yaml --mode int8 --epochs 10
orinflow-evaluate -m models/onnx/raw/yolo26n_qat_int8.onnx --yaml coco128.yaml
```

### 路线 3: SAT + PTQ

```bash
orinflow-sparsify -m yolo26n.pt --data coco128.yaml --epochs 10
orinflow-export -m yolo26n_sparse.pt --imgsz 640 --opset 13
orinflow-calibrate --yaml coco128.yaml --split train --num-images 500
orinflow-quantize -m yolo26n_sparse.onnx --calib coco128.npy --mode int8 --method entropy
# 板端: trtexec --onnx=yolo26n_sparse.onnx --int8 --sparsity=enable
```

### 路线 4: SAT + QAT

```bash
orinflow-sparse-qat -m yolo26n.pt --data coco128.yaml \
    --sat-epochs 10 --qat-epochs 10 --qat-mode int8
# 板端: trtexec --onnx=yolo26n_sparse_qat_int8.onnx --int8 --sparsity=enable
```

## 待办事项

- [ ] **路线 4 验证**: `orinflow-sparse-qat` 尚未经过实际训练验证，需确认 `mtq.quantize()` 在 sparsity DynamicModule 上叠加量化时行为正确，以及 QAT 梯度更新不会破坏稀疏 mask
- [ ] **评估模块适配**: `orinflow-evaluate` 对 QAT 导出的 ONNX（路线 2、4）和稀疏 ONNX（路线 3、4）的兼容性需要测试
- [ ] **板端部署脚本**: 提供 Jetson Orin 上的 `trtexec` 构建与推理脚本，区分各路线所需的参数组合（`--int8`、`--sparsity=enable`）
- [ ] **蒸馏**: 在QAT或SAT的训练过程中添加蒸馏。（对yolo模型的收益待考证）

## 技术栈

- [Ultralytics](https://github.com/ultralytics/ultralytics) — YOLO 模型加载、训练、导出
- [NVIDIA ModelOpt](https://github.com/NVIDIA/Model-Optimizer) — PTQ / QAT / 2:4 稀疏
- [ONNX Runtime](https://github.com/microsoft/onnxruntime) — ONNX 推理基准测试

## 许可证

MIT
