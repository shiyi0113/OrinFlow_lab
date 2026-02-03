"""Model evaluation and comparison utilities."""

import os
from dataclasses import dataclass
from pathlib import Path

from orinflow.config import DATA_DIR, ONNX_OPTIMIZED_DIR, SOURCE_MODELS_DIR


@dataclass
class EvalResult:
    """Evaluation result container."""

    model_name: str
    model_format: str
    map50: float
    map50_95: float
    ap: list | None = None
    size_mb: float | None = None

def evaluate_model(
    model_path: str | Path,
    data_yaml: str | Path,
    task: str = "detect",
    device: str | int | None = None,
) -> EvalResult:
    """
    Evaluate model accuracy on validation dataset.

    Uses low confidence threshold (0.001) and high max_det (300) for
    the most accurate mAP measurement.

    Args:
        model_path: Path to model file (.pt or .onnx)
        data_yaml: Path to dataset YAML config
        task: Task type (detect, segment, classify)
        device: Device to use ('cpu', 0, 1, etc.). None for auto-detect.

    Returns:
        EvalResult with mAP metrics
    """
    from ultralytics import YOLO

    model_path = Path(model_path)
    data_yaml = Path(data_yaml)

    if not data_yaml.is_absolute():
        data_yaml = DATA_DIR / data_yaml

    # Determine format
    model_format = "PyTorch" if model_path.suffix == ".pt" else "ONNX"
    is_pt = model_path.suffix == ".pt"

    val_kwargs = dict(
        data=str(data_yaml), task=task, imgsz=640, workers=0,
        conf=0.001, max_det=300, plots=True, save_json=True, verbose=True,
    )
    if device is not None:
        val_kwargs["device"] = device
    if is_pt:
        val_kwargs.update(batch=16, half=True)
    print(f"Evaluating {model_format} model: {model_path.name}")

    model = YOLO(model_path, task=task)
    metrics = model.val(**val_kwargs)

    size_mb = os.path.getsize(model_path) / 1e6 if model_path.exists() else None

    return EvalResult(
        model_name=model_path.stem,
        model_format=model_format,
        map50=metrics.box.map50,
        map50_95=metrics.box.map,
        ap=metrics.box.ap.tolist(),
        size_mb=size_mb,
    )


def compare_models(
    model_name: str,
    data_yaml: str | Path,
    device: str | int | None = None,
) -> dict:
    """
    Compare PyTorch and quantized ONNX model accuracy.

    Runs two evaluations:
    1. PT accuracy baseline (batch=16, half=True, conf=0.001, max_det=300)
    2. ONNX accuracy verification (conf=0.001, max_det=300)

    Args:
        model_name: Model name (without extension)
        data_yaml: Path to dataset YAML config
        device: Device to use ('cpu', 0, 1, etc.). None for auto-detect.

    Returns:
        Comparison report dict
    """
    pt_path = SOURCE_MODELS_DIR / f"{model_name}.pt"
    onnx_path = ONNX_OPTIMIZED_DIR / f"{model_name}.onnx"

    results = []

    # 1. Evaluate PyTorch baseline
    if pt_path.exists():
        print(f"\n[1/2] PT accuracy baseline: {pt_path.name}")
        pt_result = evaluate_model(pt_path, data_yaml, device=device)
        results.append(pt_result)
        print(f"  mAP50: {pt_result.map50:.4f}, mAP50-95: {pt_result.map50_95:.4f}")
    else:
        print(f"Warning: PyTorch model not found: {pt_path}")
        pt_result = None

    # 2. Evaluate ONNX accuracy
    if onnx_path.exists():
        print(f"\n[2/2] ONNX accuracy verification: {onnx_path.name}")
        onnx_result = evaluate_model(onnx_path, data_yaml, device=device)
        results.append(onnx_result)
        print(f"  mAP50: {onnx_result.map50:.4f}, mAP50-95: {onnx_result.map50_95:.4f}")
    else:
        print(f"Warning: ONNX model not found: {onnx_path}")
        onnx_result = None

    # Generate report
    report = {
        "model_name": model_name,
        "results": results,
    }

    if pt_result and onnx_result:
        diff_map50 = pt_result.map50 - onnx_result.map50
        diff_map = pt_result.map50_95 - onnx_result.map50_95
        drop_rate = (diff_map / pt_result.map50_95) * 100 if pt_result.map50_95 > 0 else 0
        size_ratio = pt_result.size_mb / onnx_result.size_mb if onnx_result.size_mb else None

        report["comparison"] = {
            "map50_drop": diff_map50,
            "map50_95_drop": diff_map,
            "drop_rate_percent": drop_rate,
            "size_reduction": size_ratio,
        }

        # Print comparison table
        print("\n" + "=" * 60)
        print("Comparison Report")
        print("=" * 60)
        print(f"{'Metric':<16} {'PT':>10} {'ONNX':>10} {'Delta':>10}")
        print("-" * 60)
        print(f"{'mAP50':<16} {pt_result.map50:>10.4f} {onnx_result.map50:>10.4f} {-diff_map50:>+10.4f}")
        print(f"{'mAP50-95':<16} {pt_result.map50_95:>10.4f} {onnx_result.map50_95:>10.4f} {-diff_map:>+10.4f}")
        if pt_result.size_mb is not None and onnx_result.size_mb is not None:
            print(f"{'Size (MB)':<16} {pt_result.size_mb:>10.2f} {onnx_result.size_mb:>10.2f} {size_ratio:>9.1f}x")
        print("-" * 60)
        print(f"Accuracy loss: {drop_rate:.2f}%")

        if drop_rate < 1.0:
            print("Conclusion: Minimal accuracy loss, quantization successful!")
        elif drop_rate < 5.0:
            print("Conclusion: Moderate accuracy loss, consider QAT or layer exclusion.")
        else:
            print("Conclusion: Significant accuracy loss! Investigate sensitive layers.")


    return report
