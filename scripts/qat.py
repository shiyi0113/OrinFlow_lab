#!/usr/bin/env python3
"""Apply Quantization-Aware Training (QAT) to YOLO model."""

import argparse

from orinflow.optimize.qat import quantize_aware_finetune


def main():
    parser = argparse.ArgumentParser(
        description="Quantization-Aware Training for YOLO model (FakeQuantize + QAT fine-tuning)"
    )
    parser.add_argument("-m", "--model", type=str, required=True, help="Source model filename (e.g. yolo26n.pt)")
    parser.add_argument("--data", type=str, default="coco128.yaml", help="Dataset YAML filename")
    parser.add_argument(
        "--mode", type=str, default="int8", choices=["int8", "fp8", "int4_awq", "w4a8"],
        help="Quantization mode (default: int8)",
    )
    parser.add_argument("--epochs", type=int, default=10, help="QAT fine-tuning epochs")
    parser.add_argument("--batch", type=int, default=16, help="Training batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="QAT learning rate")
    parser.add_argument("--calib-batch", type=int, default=4, help="Calibration batch size")
    parser.add_argument("--calib-images", type=int, default=512, help="Max calibration images (default: 512)")
    parser.add_argument(
        "--exclude", type=str, nargs="*", default=None,
        help="Glob patterns for layers to exclude from quantization (e.g. 'model.0.' 'model.22.')",
    )
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size")
    parser.add_argument("--device", type=int, default=0, help="CUDA device ID")
    parser.add_argument("--output", type=str, default=None, help="Output model path")
    parser.add_argument(
        "--no-onnx", action="store_true",
        help="Skip ONNX export (by default exports .onnx with QDQ nodes to models/onnx/raw/)",
    )
    args = parser.parse_args()

    quantize_aware_finetune(
        model_name=args.model,
        data_yaml=args.data,
        mode=args.mode,
        epochs=args.epochs,
        batch=args.batch,
        lr=args.lr,
        calib_batch=args.calib_batch,
        calib_images=args.calib_images,
        exclude=args.exclude,
        imgsz=args.imgsz,
        device=args.device,
        output_path=args.output,
        export_onnx=not args.no_onnx,
    )


if __name__ == "__main__":
    main()
