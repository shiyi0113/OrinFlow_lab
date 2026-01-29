#!/usr/bin/env python3
"""Quantize ONNX model using PTQ."""

import argparse

from orinflow.optimize import quantize_onnx

def main():
    parser = argparse.ArgumentParser(description="Quantize ONNX model (PTQ)")
    parser.add_argument("-m", "--model", type=str, default="yolo26n.onnx", help="Model name")
    parser.add_argument("--calib", type=str, default="coco128.npy",help="Calibration data name")
    parser.add_argument("--mode", type=str, default="int8", choices=["int8", "int4", "fp8"], help="Quantization mode")
    parser.add_argument("--method", type=str, default="entropy", choices=["entropy", "max", "awq_clip"], help="Calibration method")
    parser.add_argument("--exclude", type=str, nargs="*", default=None, help="Node patterns to exclude")
    args = parser.parse_args()

    # 节点排除
    # model = onnx.load(model_path)
    # exclude_nodes = []
    # for node in model.graph.node:
    #     # if "model.23" in node.name:             #  Detect
    #     #     exclude_nodes.append(node.name)
    #     # if "model.10" in node.name:             #  C2PSA
    #     #     exclude_nodes.append(node.name)
    # print(f"排除节点数: {len(exclude_nodes)}")
    
    quantize_onnx(
        model_name=args.model,
        calib_name=args.calib,
        quantize_mode=args.mode,
        calibration_method=args.method,
        nodes_to_exclude=args.exclude,
    )

if __name__ == "__main__":
    main()
