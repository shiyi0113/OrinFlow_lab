#!/usr/bin/env python3
"""Quantize ONNX model using PTQ."""

import argparse
from orinflow.optimize import quantize_onnx

def main():
    parser = argparse.ArgumentParser(description="Quantize ONNX model (PTQ)")
    parser.add_argument("-m", "--model", type=str, default="yolo26n.onnx", help="Model filename (e.g. yolo26n.onnx)")
    
    parser.add_argument("--data", type=str, default="coco.yaml", help="Dataset YAML file (e.g. coco.yaml)")
    parser.add_argument("--num-images", type=int, default=512, help="Number of calibration images")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size")
    
    parser.add_argument("--mode", type=str, default="int8", choices=["int8", "int4", "fp8"], help="Quantization mode")
    parser.add_argument("--method", type=str, default="entropy", choices=["entropy", "max", "awq_clip"], help="Calibration method")
    parser.add_argument("--exclude", type=str, nargs="*", default=None, help="Node patterns to exclude")

    parser.add_argument("--device", type=str, default=None, help="Calibration device (cuda:0, cpu). Default: auto-detect")
    args = parser.parse_args()

    quantize_onnx(
        model_name=args.model,
        data_yaml=args.data,
        num_calib_images=args.num_images, 
        imgsz=args.imgsz, 
        quantize_mode=args.mode,
        calibration_method=args.method,
        nodes_to_exclude=args.exclude,
        device=args.device,
    )

if __name__ == "__main__":
    main()