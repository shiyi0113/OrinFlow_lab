#!/usr/bin/env python3
"""Export PyTorch YOLO model to ONNX format."""

import argparse

from orinflow.export import export_to_onnx

def main():
    parser = argparse.ArgumentParser(description="Export PyTorch model to ONNX")
    parser.add_argument("-m", "--model", type=str, default="yolo26n.pt", help="Model name")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size")
    parser.add_argument("--opset", type=int, default=19, help="ONNX opset version")
    parser.add_argument("--half", action="store_true", help="Export as FP16")
    parser.add_argument("--dynamic", action="store_true", help="Enable dynamic batch size")
    parser.add_argument("--device", type=str, default=None, help="Device: 'cpu' or GPU ID (default: auto-detect)")
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

    export_to_onnx(
        model_name=args.model,
        input_size=args.imgsz,
        opset=args.opset,
        half=args.half,
        dynamic=args.dynamic,
        device=device,
    )

if __name__ == "__main__":
    main()
