# OrinFlow Lab

OrinFlow 的 PC 端子项目。在开发机上将 YOLO26 系列模型优化为高性能、低精度损失的 ONNX 模型，用于后续传输到 Jetson Orin Nano Super 通过 TensorRT 部署推理。

## 流水线

```
yolo26x.pt ──▶ Export ──▶ Calibrate ──▶ Quantize / Sparsify ──▶ Evaluate ──▶ 部署就绪的.onnx
```

| 阶段 | 命令 | 说明 |
|------|------|------|
| 导出 | `orinflow-export` | PyTorch `.pt` → ONNX |
| 校准 | `orinflow-calibrate` | 从数据集生成 `.npy` 校准数据 |
| PTQ 量化 | `orinflow-quantize` | INT8 / INT4 / FP8 后训练量化 |
| QAT 量化 | `orinflow-qat` | 量化感知训练，输出带 QDQ 节点的 ONNX |
| 2:4 稀疏 | `orinflow-sparsify` | SparseGPT / magnitude 剪枝 + SAT 微调 |
| 评估 | `orinflow-evaluate` | 精度 (mAP50/mAP50-95) 与性能基准，支持 PT↔ONNX 对比 |

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

```bash
# 1. 导出 ONNX
orinflow-export -m yolo26n.pt --imgsz 640 --opset 13

# 2. 生成校准数据
orinflow-calibrate --yaml coco128.yaml --split train --num-images 500

# 3a. PTQ 量化（推荐 INT8 + entropy 作为起点）
orinflow-quantize -m yolo26n.onnx --calib coco128.npy --mode int8 --method entropy

# 3b. QAT 量化感知训练
orinflow-qat -m yolo26n.pt --data coco128.yaml --mode int8 --epochs 10

# 3c. 2:4 结构化稀疏
orinflow-sparsify -m yolo26n.pt --data coco128.yaml --epochs 10

# 4. 评估（PT vs ONNX 对比）
orinflow-evaluate -m yolo26n --yaml coco128.yaml --compare
```

## 技术栈

- [Ultralytics](https://github.com/ultralytics/ultralytics) — YOLO 模型加载、训练、导出
- [NVIDIA ModelOpt](https://github.com/NVIDIA/Model-Optimizer) — PTQ / QAT / 2:4 稀疏
- [ONNX Runtime](https://github.com/microsoft/onnxruntime) — ONNX 推理基准测试

## 许可证

MIT
