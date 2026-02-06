#!/usr/bin/env python3
"""Apply 2:4 structured sparsity to YOLO model with SAT fine-tuning."""

import argparse

from orinflow.optimize.sparsify import sparsify_and_finetune


def main():
    parser = argparse.ArgumentParser(
        description="Sparsify YOLO model (2:4 structured sparsity + SAT fine-tuning)"
    )
    parser.add_argument("-m", "--model", type=str, required=True, help="Source model filename (e.g. yolo26x.pt)")
    parser.add_argument("--data", type=str, default="coco.yaml", help="Dataset YAML filename")
    parser.add_argument("--mode", type=str, default="sparsegpt", choices=["sparsegpt", "sparse_magnitude"],
        help="Sparsity algorithm (default: sparsegpt)",
    )
    parser.add_argument("--epochs", type=int, default=10, help="SAT fine-tuning epochs")
    parser.add_argument("--batch", type=int, default=16, help="Training batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="SAT learning rate")
    parser.add_argument("--calib-batch", type=int, default=4, help="Calibration batch size for SparseGPT")
    parser.add_argument("--calib-images", type=int, default=512, help="Max calibration images for SparseGPT (default: 512)")
    parser.add_argument("--exclude", type=str, nargs="*", default=None, help="Glob patterns for layers to exclude (e.g. 'model.0.' 'model.22.')")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size")
    parser.add_argument("--device", type=int, default=0, help="CUDA device ID")
    parser.add_argument("--output", type=str, default=None, help="Output model path")
    args = parser.parse_args()

    sparsify_and_finetune(
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
    )


if __name__ == "__main__":
    main()
