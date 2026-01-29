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
    speed: dict | None = None
    size_mb: float | None = None

def evaluate_model(
    model_path: str | Path,
    data_yaml: str | Path,
    task: str = "detect",
    mode: str = "accuracy",
) -> EvalResult:
    """
    Evaluate model accuracy or performance on validation dataset.

    Args:
        model_path: Path to model file (.pt or .onnx)
        data_yaml: Path to dataset YAML config
        task: Task type (detect, segment, classify)
        mode: Evaluation mode
            - "accuracy": low conf(0.001), high max_det(300), with plots/save_json
            - "performance": deployment-realistic conf(0.25), max_det(100), no plots

    Returns:
        EvalResult with mAP metrics and/or speed
    """
    from ultralytics import YOLO

    model_path = Path(model_path)
    data_yaml = Path(data_yaml)

    if not data_yaml.is_absolute():
        data_yaml = DATA_DIR / data_yaml

    # Determine format
    model_format = "PyTorch" if model_path.suffix == ".pt" else "ONNX"
    is_pt = model_path.suffix == ".pt"

    val_kwargs = dict(data=str(data_yaml), task=task, imgsz=640, workers=0)
    if mode == "performance":
        val_kwargs.update(conf=0.25, max_det=100, plots=False, save_json=False, verbose=False)
    else:
        # accuracy mode
        val_kwargs.update(conf=0.001, max_det=300, plots=True, save_json=True, verbose=True)
        if is_pt:
            val_kwargs.update(batch=16, half=True)
    print(f"Evaluating {model_format} model ({mode}): {model_path.name}")

    model = YOLO(model_path)
    metrics = model.val(**val_kwargs)

    size_mb = os.path.getsize(model_path) / 1e6 if model_path.exists() else None

    return EvalResult(
        model_name=model_path.stem,
        model_format=model_format,
        map50=metrics.box.map50,
        map50_95=metrics.box.map,
        ap=metrics.box.ap.tolist(),
        speed=metrics.speed,
        size_mb=size_mb,
    )


def compare_models(
    model_name: str,
    data_yaml: str | Path,
) -> dict:
    """
    Compare PyTorch and quantized ONNX model accuracy and performance.

    Runs three evaluations:
    1. PT accuracy baseline (batch=16, half=True, conf=0.001, max_det=300)
    2. ONNX accuracy verification (conf=0.001, max_det=300)
    3. ONNX performance benchmark (conf=0.25, max_det=100)

    The ONNX result merges accuracy metrics from run 2 and speed from run 3.

    Args:
        model_name: Model name (without extension)
        data_yaml: Path to dataset YAML config

    Returns:
        Comparison report dict
    """
    pt_path = SOURCE_MODELS_DIR / f"{model_name}.pt"
    onnx_path = ONNX_OPTIMIZED_DIR / f"{model_name}.onnx"

    results = []

    # 1. Evaluate PyTorch baseline (accuracy mode)
    if pt_path.exists():
        print(f"\n[1/3] PT accuracy baseline: {pt_path.name}")
        pt_result = evaluate_model(pt_path, data_yaml, mode="accuracy")
        results.append(pt_result)
        print(f"  mAP50: {pt_result.map50:.4f}, mAP50-95: {pt_result.map50_95:.4f}")
    else:
        print(f"Warning: PyTorch model not found: {pt_path}")
        pt_result = None

    # 2 & 3. Evaluate ONNX (accuracy + performance)
    if onnx_path.exists():
        # Accuracy verification
        print(f"\n[2/3] ONNX accuracy verification: {onnx_path.name}")
        onnx_result = evaluate_model(onnx_path, data_yaml, mode="accuracy")
        print(f"  mAP50: {onnx_result.map50:.4f}, mAP50-95: {onnx_result.map50_95:.4f}")

        # Performance benchmark
        print(f"\n[3/3] ONNX performance benchmark: {onnx_path.name}")
        perf_result = evaluate_model(onnx_path, data_yaml, mode="performance")

        # Merge: keep accuracy metrics, replace speed with deployment-realistic speed
        onnx_result.speed = perf_result.speed
        results.append(onnx_result)
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

        # Extract total inference speed (preprocess + inference + postprocess)
        def _total_speed(speed: dict | None) -> float | None:
            if speed is None:
                return None
            return sum(speed.get(k, 0) for k in ("preprocess", "inference", "postprocess"))

        pt_speed = _total_speed(pt_result.speed)
        onnx_speed = _total_speed(onnx_result.speed)

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
        if pt_speed is not None and onnx_speed is not None:
            print(f"{'Speed (ms)':<16} {pt_speed:>10.2f} {onnx_speed:>10.2f} {pt_speed / onnx_speed:>9.1f}x")
        print("-" * 60)
        print(f"Accuracy loss: {drop_rate:.2f}%")

        if drop_rate < 1.0:
            print("Conclusion: Minimal accuracy loss, quantization successful!")
        elif drop_rate < 5.0:
            print("Conclusion: Moderate accuracy loss, consider QAT or layer exclusion.")
        else:
            print("Conclusion: Significant accuracy loss! Investigate sensitive layers.")


    return report