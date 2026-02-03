#!/usr/bin/env python3
"""Evaluate and compare model accuracy.
# 单模型精度评估
orinflow-evaluate -m models/source/yolo26n.pt --yaml coco128.yaml

# ONNX 模型精度评估（Blackwell GPU 需加 --device cpu）
orinflow-evaluate -m models/onnx/optimized/yolo26n.onnx --yaml coco128.yaml --device cpu

# PT vs ONNX 精度对比
orinflow-evaluate -m yolo26n --yaml coco128.yaml --compare

"""
import argparse

from orinflow.evaluate import evaluate_model, compare_models

def main():
    parser = argparse.ArgumentParser(description="Evaluate model accuracy")
    parser.add_argument("-m", "--model", type=str, required=True,
                        help="Model path (.pt/.onnx) for single eval, or model name for --compare")
    parser.add_argument("--yaml", type=str, default="coco128.yaml", help="Dataset YAML file")
    parser.add_argument("--compare", action="store_true",
                        help="Compare PT vs quantized ONNX (pass model name without extension)")
    parser.add_argument("--device", type=str, default=None,
                        help="Device: 'cpu' or GPU ID (default: auto-detect). Use 'cpu' for GPUs without ORT/torchvision CUDA support.")
    args = parser.parse_args()

    # Parse device
    device = None
    if args.device is not None:
        if args.device.lower() == "cpu":
            device = "cpu"
        else:
            try:
                device = int(args.device)
            except ValueError:
                parser.error(f"Invalid device: {args.device}. Use 'cpu' or GPU ID (0, 1, ...)")

    if args.compare:
        compare_models(args.model, args.yaml, device=device)
    else:
        result = evaluate_model(args.model, args.yaml, device=device)
        print(f"\n{'=' * 50}")
        print(f"Model:    {result.model_name} ({result.model_format})")
        print(f"mAP50:    {result.map50:.4f}")
        print(f"mAP50-95: {result.map50_95:.4f}")
        if result.size_mb is not None:
            print(f"Size:     {result.size_mb:.2f} MB")


if __name__ == "__main__":
    main()
