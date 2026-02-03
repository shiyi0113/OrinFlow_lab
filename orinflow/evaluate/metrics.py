"""Model evaluation and comparison utilities."""

import gc
import os
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from orinflow.config import DATA_DIR, ONNX_OPTIMIZED_DIR, SOURCE_MODELS_DIR


_DEFAULT_CONF = 0.001
_DEFAULT_MAX_DET = 300
_DEFAULT_IMGSZ = 640
_DEFAULT_FLUSH_INTERVAL = 500


@dataclass
class EvalResult:
    """Evaluation result container."""

    model_name: str
    model_format: str
    map50: float
    map50_95: float
    ap: list | None = None
    size_mb: float | None = None


@contextmanager
def _batched_metrics(flush_interval: int = _DEFAULT_FLUSH_INTERVAL):
    """
    Context manager that patches DetMetrics to flush stats to disk periodically.

    This prevents memory accumulation when evaluating large datasets.
    """
    from ultralytics.utils.metrics import DetMetrics

    # Store original methods
    original_init = DetMetrics.__init__
    original_update = DetMetrics.update_stats
    original_process = DetMetrics.process

    def patched_init(self, names={}):
        original_init(self, names)
        # Add batch tracking attributes
        self._flush_interval = flush_interval
        self._image_count = 0
        self._batch_count = 0
        self._batch_dir = None
        self._batch_files = []

    def patched_update(self, stat):
        original_update(self, stat)
        self._image_count += 1

        # Flush to disk periodically
        if self._flush_interval > 0 and self._image_count % self._flush_interval == 0:
            _flush_stats_to_disk(self)

    def patched_process(self, save_dir=Path("."), plot=False, on_plot=None):
        # Load all batched stats from disk first
        _load_all_batches(self)
        # Then call original process
        return original_process(self, save_dir, plot, on_plot)

    def _flush_stats_to_disk(metrics_obj):
        """Flush current stats to disk and clear memory."""
        if not metrics_obj.stats["tp"]:
            return

        if metrics_obj._batch_dir is None:
            metrics_obj._batch_dir = Path(tempfile.mkdtemp(prefix="ultralytics_stats_"))

        batch_file = metrics_obj._batch_dir / f"batch_{metrics_obj._batch_count:04d}.npz"
        np.savez_compressed(
            batch_file,
            tp=np.concatenate(metrics_obj.stats["tp"], 0) if metrics_obj.stats["tp"] else np.array([]),
            conf=np.concatenate(metrics_obj.stats["conf"], 0) if metrics_obj.stats["conf"] else np.array([]),
            pred_cls=np.concatenate(metrics_obj.stats["pred_cls"], 0) if metrics_obj.stats["pred_cls"] else np.array([]),
            target_cls=np.concatenate(metrics_obj.stats["target_cls"], 0) if metrics_obj.stats["target_cls"] else np.array([]),
            target_img=np.concatenate(metrics_obj.stats["target_img"], 0) if metrics_obj.stats["target_img"] else np.array([]),
        )
        metrics_obj._batch_files.append(batch_file)
        metrics_obj._batch_count += 1

        # Clear in-memory stats
        for v in metrics_obj.stats.values():
            v.clear()

        # print(f"  [Memory] Flushed batch {metrics_obj._batch_count} ({metrics_obj._image_count} images)")
        gc.collect()

    def _load_all_batches(metrics_obj):
        """Load all batched stats from disk and merge with current in-memory stats."""
        if not metrics_obj._batch_files:
            return

        all_stats = {k: [] for k in metrics_obj.stats.keys()}

        # Load from disk
        for batch_file in metrics_obj._batch_files:
            data = np.load(batch_file)
            for k in all_stats.keys():
                if len(data[k]) > 0:
                    all_stats[k].append(data[k])

        # Add current in-memory stats
        for k in all_stats.keys():
            all_stats[k].extend(metrics_obj.stats[k])

        # Replace stats with merged data
        metrics_obj.stats = {k: v if v else [] for k, v in all_stats.items()}

        # Cleanup temp files
        if metrics_obj._batch_dir and metrics_obj._batch_dir.exists():
            import shutil
            shutil.rmtree(metrics_obj._batch_dir)
            metrics_obj._batch_dir = None
            metrics_obj._batch_files = []

        # print(f"  [Memory] Merged {metrics_obj._batch_count} batches for final mAP calculation")

    # Apply patches
    DetMetrics.__init__ = patched_init
    DetMetrics.update_stats = patched_update
    DetMetrics.process = patched_process

    try:
        yield
    finally:
        # Restore original methods
        DetMetrics.__init__ = original_init
        DetMetrics.update_stats = original_update
        DetMetrics.process = original_process


def evaluate_model(
    model_path: str | Path,
    data_yaml: str | Path,
    device: str | None = None,
    fallback_to_cpu: bool = True,
) -> EvalResult:
    """
    Evaluate model accuracy on validation dataset using ultralytics val().

    Uses memory-efficient batch evaluation with accuracy-first settings:
    - conf=0.001 (low threshold for accurate mAP)
    - max_det=300 (high detection limit)
    - Automatic disk flush every 500 images to prevent memory issues

    Args:
        model_path: Path to model file (.pt or .onnx)
        data_yaml: Path to dataset YAML config
        device: Device to use for inference ("cpu", "0", "cuda:0", etc.). Default: GPU ("0").
        fallback_to_cpu: If True and GPU fails, automatically retry with CPU.

    Returns:
        EvalResult with mAP metrics
    """
    from ultralytics import YOLO

    model_path = Path(model_path)
    data_yaml = Path(data_yaml)

    if not data_yaml.is_absolute():
        data_yaml = DATA_DIR / data_yaml

    model_format = "PyTorch" if model_path.suffix == ".pt" else "ONNX"
    is_pt = model_path.suffix == ".pt"
    size_mb = os.path.getsize(model_path) / 1e6 if model_path.exists() else None

    if device is None:
        device = "0"

    val_kwargs = dict(
        data=str(data_yaml),
        imgsz=_DEFAULT_IMGSZ,
        workers=0,
        conf=_DEFAULT_CONF,
        max_det=_DEFAULT_MAX_DET,
        plots=False,
        save_json=False,
        verbose=True,
        device=device,
    )
    if is_pt:
        val_kwargs.update(batch=16, half=True)

    print(f"Evaluating: {model_path.name}")
    print(f"  conf={_DEFAULT_CONF}, max_det={_DEFAULT_MAX_DET}, imgsz={_DEFAULT_IMGSZ}, device={device}")

    model = YOLO(model_path, task="detect")

    # Use patched metrics for memory-efficient evaluation
    with _batched_metrics(flush_interval=_DEFAULT_FLUSH_INTERVAL):
        try:
            metrics = model.val(**val_kwargs)
        except Exception as e:
            # Fallback to CPU if GPU fails (e.g., QAT ONNX on Blackwell/RTX 50xx)
            if fallback_to_cpu and device != "cpu":
                print(f"  Warning: GPU inference failed ({type(e).__name__}), retrying with CPU...")
                val_kwargs["device"] = "cpu"
                model = YOLO(model_path, task="detect")
                metrics = model.val(**val_kwargs)
            else:
                raise

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
    device: str | None = None,
    onnx_device: str | None = None,
) -> dict:
    """
    Compare PyTorch and quantized ONNX model accuracy.

    Args:
        model_name: Model name (without extension)
        data_yaml: Path to dataset YAML config
        device: Device for PT model evaluation (default: GPU)
        onnx_device: Device for ONNX model evaluation (default: GPU, auto fallback to CPU on error)

    Returns:
        Comparison report dict
    """
    pt_path = SOURCE_MODELS_DIR / f"{model_name}.pt"
    onnx_path = ONNX_OPTIMIZED_DIR / f"{model_name}.onnx"

    results = []

    if pt_path.exists():
        print(f"\n[1/2] PT baseline: {pt_path.name}")
        pt_result = evaluate_model(pt_path, data_yaml, device=device)
        results.append(pt_result)
        print(f"  mAP50: {pt_result.map50:.4f}, mAP50-95: {pt_result.map50_95:.4f}")
    else:
        print(f"Warning: PyTorch model not found: {pt_path}")
        pt_result = None

    if onnx_path.exists():
        print(f"\n[2/2] ONNX: {onnx_path.name}")
        onnx_result = evaluate_model(onnx_path, data_yaml, device=onnx_device)
        results.append(onnx_result)
        print(f"  mAP50: {onnx_result.map50:.4f}, mAP50-95: {onnx_result.map50_95:.4f}")
    else:
        print(f"Warning: ONNX model not found: {onnx_path}")
        onnx_result = None

    report = {"model_name": model_name, "results": results}

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
        print(f"Accuracy drop: {drop_rate:.2f}%")

        if drop_rate < 1.0:
            print("✓ Minimal accuracy loss, quantization successful!")
        elif drop_rate < 5.0:
            print("⚠ Moderate accuracy loss, consider layer exclusion.")
        else:
            print("✗ Significant accuracy loss! Investigate sensitive layers.")

    return report
