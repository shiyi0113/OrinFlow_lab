#!/usr/bin/env python3
"""Evaluate and compare model accuracy.
# 单模型精度评估
python scripts/evaluate.py -m models/yolo11n.pt --yaml coco128.yaml

# 单模型性能评估
python scripts/evaluate.py -m models/yolo11n.onnx --yaml coco128.yaml --mode performance

# PT vs ONNX 完整对比（3 步自动执行）
python scripts/evaluate.py -m yolo11n --yaml coco128.yaml --compare

"""
import argparse

from orinflow.evaluate import evaluate_model, compare_models

def main():
    parser = argparse.ArgumentParser(description="Evaluate model accuracy and performance")
    parser.add_argument("-m", "--model", type=str, required=True,
                        help="Model path (.pt/.onnx) for single eval, or model name for --compare")
    parser.add_argument("--yaml", type=str, default="coco128.yaml", help="Dataset YAML file")
    parser.add_argument("--mode", type=str, default="accuracy", choices=["accuracy", "performance"],
                        help="Evaluation mode (default: accuracy)")
    parser.add_argument("--compare", action="store_true",
                        help="Compare PT vs quantized ONNX (pass model name without extension)")
    args = parser.parse_args()

    if args.compare:
        # Compare mode: 3-step PT baseline + ONNX accuracy + ONNX performance
        compare_models(args.model, args.yaml)
    else:
        # Single model evaluation
        result = evaluate_model(args.model, args.yaml, mode=args.mode)
        print(f"\n{'=' * 50}")
        print(f"Model:    {result.model_name} ({result.model_format})")
        print(f"mAP50:    {result.map50:.4f}")
        print(f"mAP50-95: {result.map50_95:.4f}")
        if result.size_mb is not None:
            print(f"Size:     {result.size_mb:.2f} MB")
        if result.speed is not None:
            print(f"Speed:    {result.speed}")


if __name__ == "__main__":
    main()
