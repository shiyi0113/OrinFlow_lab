#!/usr/bin/env python3
"""Apply combined 2:4 Sparsity + Quantization-Aware Training (SAT + QAT)."""

import argparse

from orinflow.optimize.sparse_qat import sparse_quantize_aware_finetune


def main():
    parser = argparse.ArgumentParser(
        description="Combined 2:4 Sparsity + QAT for YOLO model (Route 4: SAT -> QAT -> ONNX with QDQ + sparsity)"
    )
    parser.add_argument("-m", "--model", type=str, required=True, help="Source model filename (e.g. yolo26n.pt)")
    parser.add_argument("--data", type=str, default="coco.yaml", help="Dataset YAML filename")
    parser.add_argument("--sparsity-mode", type=str, default="sparsegpt", choices=["sparsegpt", "sparse_magnitude"],
        help="Sparsity algorithm (default: sparsegpt)",
    )
    parser.add_argument("--qat-mode", type=str, default="int8", choices=["int8", "fp8", "int4_awq", "w4a8"],
        help="Quantization mode (default: int8)",
    )
    parser.add_argument("--sat-epochs", type=int, default=10, help="SAT fine-tuning epochs")
    parser.add_argument("--qat-epochs", type=int, default=10, help="QAT fine-tuning epochs")
    parser.add_argument("--batch", type=int, default=16, help="Training batch size")
    parser.add_argument("--sat-lr", type=float, default=1e-3, help="SAT learning rate (default: 1e-3)")
    parser.add_argument("--qat-lr", type=float, default=1e-3, help="QAT learning rate (default: 1e-3)")
    parser.add_argument("--optimizer", type=str, default="AdamW", help="Optimizer (default: AdamW; also SGD, Adam, RAdam)")
    parser.add_argument("--workers", type=int, default=8, help="Dataloader workers")
    parser.add_argument("--calib-batch", type=int, default=4, help="Calibration batch size")
    parser.add_argument("--calib-images", type=int, default=512, help="Max calibration images (default: 512)")
    parser.add_argument("--exclude-sparse", type=str, nargs="*", default=None,
        help="Glob patterns for layers to exclude from sparsification (e.g. 'model.0.' 'model.22.')",
    )
    parser.add_argument("--exclude-quant", type=str, nargs="*", default=None,
        help="Glob patterns for layers to exclude from quantization (e.g. 'model.0.' 'model.22.')",
    )
    parser.add_argument("--calibrator", type=str, default="max", choices=["histogram", "max"],
        help="Activation calibrator: 'max' (global max, default) or 'histogram' (entropy-based, may clip too aggressively for CNN)",
    )
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size")
    parser.add_argument("--device", type=int, default=0, help="CUDA device ID")
    parser.add_argument("--warmup-epochs", type=float, default=1.0,
        help="LR warmup epochs (default: 1.0; fine-tuning needs shorter warmup)",
    )
    parser.add_argument("--close-mosaic", type=int, default=0,
        help="Disable mosaic for last N epochs (default: 0, disabled entirely for fine-tuning)",
    )
    parser.add_argument("--freeze", type=int, default=None,
        help="Freeze first N backbone layers (e.g. 10). Reduces trainable params for large models",
    )
    args = parser.parse_args()

    sparse_quantize_aware_finetune(
        model_name=args.model,
        data_yaml=args.data,
        sparsity_mode=args.sparsity_mode,
        qat_mode=args.qat_mode,
        sat_epochs=args.sat_epochs,
        qat_epochs=args.qat_epochs,
        batch=args.batch,
        sat_lr=args.sat_lr,
        qat_lr=args.qat_lr,
        optimizer=args.optimizer,
        workers=args.workers,
        calib_batch=args.calib_batch,
        calib_images=args.calib_images,
        exclude_sparse=args.exclude_sparse,
        exclude_quant=args.exclude_quant,
        calibrator=args.calibrator,
        imgsz=args.imgsz,
        device=args.device,
        warmup_epochs=args.warmup_epochs,
        close_mosaic=args.close_mosaic,
        freeze=args.freeze,
    )


if __name__ == "__main__":
    main()
