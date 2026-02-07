#!/usr/bin/env python3
"""Export PyTorch YOLO model to ONNX format."""

import argparse

from orinflow.export import export_to_onnx

def main():
    parser = argparse.ArgumentParser(description="Export PyTorch model to ONNX")
    parser.add_argument("-m", "--model", type=str, default="yolo26n.pt", help="Model name")

    parser.add_argument("--imgsz", type=int, default=640, help="Input image size")
    parser.add_argument("--opset", type=int, default=13, help="ONNX opset version")
    parser.add_argument("--half", action="store_true", help="Export as FP16")
    parser.add_argument("--dynamic", action="store_true", help="Enable dynamic batch size")

    parser.add_argument("--device", type=str, default=None, choices=["cpu", "0", "1", "2", "3"], help="Device: 'cpu' or GPU ID (default: auto-detect)")
    args = parser.parse_args()

    export_to_onnx(
        model_name=args.model,
        input_size=args.imgsz,
        opset=args.opset,
        half=args.half,
        dynamic=args.dynamic,
        device=args.device,
    )

if __name__ == "__main__":
    main()
