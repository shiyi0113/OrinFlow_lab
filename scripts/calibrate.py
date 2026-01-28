#!/usr/bin/env python3
"""Prepare calibration data for quantization."""

import argparse

from orinflow.data import prepare_calibration_data

def main():
    parser = argparse.ArgumentParser(description="Prepare calibration data")
    parser.add_argument("--yaml", type=str, default="coco128.yaml", help="Dataset YAML file")
    parser.add_argument("--split", type=str, default="train", choices=["train", "val", "test"], help="Dataset split")
    parser.add_argument("--num-images", type=int, default=500, help="Number of calibration images")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size")
    parser.add_argument("--output", type=str, default=None, help="Output file name")
    args = parser.parse_args()

    prepare_calibration_data(
        yaml_file=args.yaml,
        split=args.split,
        num_images=args.num_images,
        input_size=args.imgsz,
        output_name=args.output,
    )


if __name__ == "__main__":
    main()
