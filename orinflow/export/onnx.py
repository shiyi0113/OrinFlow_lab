"""ONNX export utilities for YOLO models."""

import shutil
from pathlib import Path

from orinflow.config import DEFAULT_INPUT_SIZE, DEFAULT_OPSET, SOURCE_MODELS_DIR,ONNX_RAW_DIR


def export_to_onnx(
    model_name: str,
    input_size: int = DEFAULT_INPUT_SIZE,
    opset: int = DEFAULT_OPSET,
    half: bool = False,
    simplify: bool = True,
    dynamic: bool = False,
    device: str | int | None = None,
    output_dir: Path | None = None,
) -> Path:
    """
    Export PyTorch YOLO model to ONNX format.

    Args:
        model_name: Model name (without .pt extension)
        input_size: Input image size
        opset: ONNX opset version
        half: Export as FP16
        simplify: Simplify ONNX model
        dynamic: Enable dynamic batch size
        device: Device to use ('cpu', 0, 1, etc.). None for auto-detect.
        output_dir: Output directory, defaults to models/onnx_raw

    Returns:
        Path to exported ONNX model
    """
    import torch
    from ultralytics import YOLO

    if device is None:
        device = 0 if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model_path = SOURCE_MODELS_DIR / f"{model_name}.pt"
    if output_dir is None:
        output_dir = ONNX_RAW_DIR
    output_path = output_dir / f"{model_name}.onnx"

    print(f"Loading model: {model_path}")
    model = YOLO(model_path)

    print("Exporting to ONNX...")
    exported_path = model.export(
        format="onnx",
        opset=opset,
        half=half,
        simplify=simplify,
        dynamic=dynamic,
        imgsz=input_size,
        device=device,
    )

    # Move to output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    if Path(exported_path).exists() and Path(exported_path) != output_path:
        shutil.move(exported_path, output_path)
        print(f"Model exported to: {output_path}")
    elif output_path.exists():
        print(f"Model exported to: {output_path}")
    else:
        raise FileNotFoundError(f"Export failed, file not found: {exported_path}")

    return output_path
