"""Quantization utilities (PTQ and QAT)."""

from pathlib import Path
import numpy as np

from orinflow.config import CALIB_DIR, ONNX_OPTIMIZED_DIR, ONNX_RAW_DIR


def quantize_onnx(
    model_name: str,
    calib_name: str,
    quantize_mode: str = "int8",
    calibration_method: str = "entropy",
    nodes_to_exclude: list[str] | None = None,
    input_dir: Path | None = None,
    output_dir: Path | None = None,
) -> Path:
    """
    Quantize ONNX model using ModelOpt PTQ.

    Args:
        model_name: Model name (without .onnx extension)
        calib_name: Calibration data name (without _calib.npy extension)
        quantize_mode: Quantization mode (int8, int4, fp8)
        calibration_method: Calibration method (entropy, max, awq_clip, etc.)
        nodes_to_exclude: List of node name patterns to exclude from quantization
        input_dir: Input directory, defaults to models/onnx_raw
        output_dir: Output directory, defaults to models/onnx_quant

    Returns:
        Path to quantized ONNX model
    """
    from modelopt.onnx.quantization import quantize

    if input_dir is None:
        input_dir = ONNX_RAW_DIR
    if output_dir is None:
        output_dir = ONNX_OPTIMIZED_DIR

    model_path = input_dir / model_name
    calib_path = CALIB_DIR / calib_name
    output_path = output_dir / model_name

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not calib_path.exists():
        raise FileNotFoundError(f"Calibration data not found: {calib_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load calibration data
    calib_data = np.load(calib_path)
    print(f"Calibration data shape: {calib_data.shape}")

    # Quantize
    print(f"Quantizing {model_path} with {quantize_mode} mode...")
    quantize(
        onnx_path=str(model_path),
        output_path=str(output_path),
        calibration_data=calib_data,
        calibration_method=calibration_method,
        quantize_mode=quantize_mode,
        nodes_to_exclude=nodes_to_exclude,
        calibration_eps=["cuda:0"],
        calibration_shapes="images:1x3x640x640",
    )

    print(f"Quantization complete: {output_path}")
    return output_path