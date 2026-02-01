"""Quantization-Aware Training (QAT) with NVIDIA ModelOpt.

Uses ModelOpt's ``mtq.quantize()`` to insert FakeQuantize nodes (TensorQuantizer)
into the model, calibrates activation ranges, then fine-tunes with QAT so the
model learns to compensate for quantization error via the Straight-Through
Estimator (STE).

TensorQuantizer is a regular nn.Module -- it is pickle-safe and supports
deepcopy, so unlike sparsity's DynamicModule we do NOT need a save_model()
override.

FakeQuantize nodes automatically export as Q/DQ nodes during
torch.onnx.export(), so the user runs ``orinflow-export`` separately after QAT.

Typical workflow:
    orinflow-qat -m yolo11n.pt --data coco128.yaml --mode int8 --epochs 10
    orinflow-export -m yolo11n_qat_int8.pt --imgsz 640
"""

import copy
import fnmatch
from pathlib import Path

import torch

import modelopt.torch.quantization as mtq

from orinflow.config import DATA_DIR, SOURCE_MODELS_DIR

# Mapping from user-facing mode strings to ModelOpt quantization preset configs.
# INT8 is the recommended starting point for CNN/YOLO models.
# INT4/W4A8 presets are primarily designed for LLMs and may not perform well on CNNs.
QAT_MODE_MAP: dict[str, dict] = {
    "int8": mtq.INT8_DEFAULT_CFG,
    "fp8": mtq.FP8_DEFAULT_CFG,
    "int4_awq": mtq.INT4_AWQ_CFG,
    "w4a8": mtq.W4A8_AWQ_BETA_CFG,
}


def _resolve_qat_config(mode: str) -> dict:
    """Resolve a user-facing mode string to a ModelOpt quantization config.

    Args:
        mode: One of "int8", "fp8", "int4_awq", "w4a8".

    Returns:
        Deep copy of the corresponding ModelOpt preset config dict.

    Raises:
        ValueError: If mode is not recognized.
    """
    if mode not in QAT_MODE_MAP:
        valid = ", ".join(sorted(QAT_MODE_MAP.keys()))
        raise ValueError(f"Unknown QAT mode '{mode}'. Valid modes: {valid}")
    return copy.deepcopy(QAT_MODE_MAP[mode])


def _apply_quantizer_exclusions(model: torch.nn.Module, exclude: list[str]) -> None:
    """Disable quantizers on layers matching exclusion patterns.

    Uses ``mtq.disable_quantizer()`` with fnmatch-style glob patterns.
    Patterns without wildcards are auto-wrapped as ``*pattern*``.

    Args:
        model: The quantized model (with TensorQuantizer nodes).
        exclude: Glob patterns for quantizer names to disable.
    """
    for pattern in exclude:
        glob_pat = pattern if ("*" in pattern or "?" in pattern) else f"*{pattern}*"
        matched = [name for name, _ in model.named_modules() if fnmatch.fnmatch(name, glob_pat)]
        if matched:
            preview = ", ".join(matched[:5])
            suffix = f" ... ({len(matched)} total)" if len(matched) > 5 else ""
            print(f"  Exclude '{pattern}' -> {preview}{suffix}")
        else:
            print(f"  Warning: '{pattern}' did not match any modules.")
        mtq.disable_quantizer(model, glob_pat)


def _make_qat_trainer_cls():
    """Create a DetectionTrainer subclass for QAT fine-tuning.

    Deferred import to avoid loading ultralytics at module level.
    """
    from ultralytics.models.yolo.detect import DetectionTrainer

    class QATDetectionTrainer(DetectionTrainer):
        """DetectionTrainer that handles ModelOpt TensorQuantizer parameters.

        TensorQuantizer modules may set some parameters to requires_grad=False.
        The base _setup_train() would force them back to True with a noisy
        "setting requires_grad=True for frozen layer" warning. Since QAT needs
        gradients on all float params (quantization error propagated via STE),
        we pre-enable them to suppress the warnings.

        No save_model() override needed: TensorQuantizer is pickle-safe.
        """

        def _setup_train(self):
            """Enable gradients on all float params before base setup."""
            for v in self.model.parameters():
                if not v.requires_grad and v.dtype.is_floating_point:
                    v.requires_grad = True
            super()._setup_train()

    return QATDetectionTrainer


def quantize_aware_finetune(
    model_name: str,
    data_yaml: str,
    mode: str = "int8",
    epochs: int = 10,
    batch: int = 16,
    lr: float = 1e-4,
    calib_batch: int = 4,
    calib_images: int = 512,
    exclude: list[str] | None = None,
    imgsz: int = 640,
    device: int = 0,
    output_path: Path | None = None,
) -> Path:
    """Apply quantization and fine-tune with QAT.

    Pipeline:
        1. Load YOLO model
        2. Insert FakeQuantize nodes via mtq.quantize() + calibrate
        3. Optionally disable quantizers on excluded layers
        4. QAT fine-tuning (STE-based gradient flow through FakeQuantize)
        5. Save final quantized model

    The saved .pt model contains TensorQuantizer nodes. When exported to ONNX
    via ``orinflow-export``, FakeQuantize nodes automatically become Q/DQ
    (QuantizeLinear/DequantizeLinear) nodes.

    Args:
        model_name: Source model filename (e.g. "yolo11n.pt").
        data_yaml: Dataset YAML filename (e.g. "coco128.yaml").
        mode: Quantization mode -- "int8", "fp8", "int4_awq", or "w4a8".
        epochs: QAT fine-tuning epochs.
        batch: Training batch size.
        lr: Learning rate for QAT.
        calib_batch: Batch size for calibration data forwarding.
        calib_images: Max number of images for calibration (default 512).
        exclude: Glob patterns for layers to exclude from quantization
            (fnmatch-style). Patterns without wildcards are auto-wrapped
            as ``*pattern*``. E.g. ``["model.0.", "model.22."]``.
        imgsz: Input image size.
        device: CUDA device ID.
        output_path: Output .pt path. Defaults to
            models/source/{stem}_qat_{mode}.pt.

    Returns:
        Path to saved QAT model.
    """
    from ultralytics import YOLO

    QATDetectionTrainer = _make_qat_trainer_cls()

    model_path = SOURCE_MODELS_DIR / model_name
    data_path = DATA_DIR / data_yaml
    cuda_device = f"cuda:{device}"

    if output_path is None:
        output_path = SOURCE_MODELS_DIR / f"{model_path.stem}_qat_{mode}.pt"
    output_path = Path(output_path)

    # ── 1. Load model ──────────────────────────────────────────────
    print(f"Loading model: {model_path}")
    model_yolo = YOLO(str(model_path))
    model_yolo.to(cuda_device)
    model_yolo.eval()

    # ── 2. Quantize + Calibrate ────────────────────────────────────
    # Build calibration dataloader via a temporary trainer.
    # setup_model() is required because build_dataset accesses self.model.stride.
    tmp_trainer = QATDetectionTrainer(overrides=dict(
        model=str(model_path),
        data=str(data_path),
        imgsz=imgsz,
    ))
    tmp_trainer.setup_model()
    calib_loader = tmp_trainer.get_dataloader(
        tmp_trainer.data["train"], batch_size=calib_batch, rank=-1, mode="train",
    )

    max_iters = max(1, calib_images // calib_batch)

    def forward_loop(model):
        """Forward calibration data through model for activation range collection."""
        for i, batch_data in enumerate(calib_loader):
            if i >= max_iters:
                break
            model(batch_data["img"].to(cuda_device).float() / 255.0)

    qat_config = _resolve_qat_config(mode)
    print(f"Quantizing model with {mode} mode "
          f"(calibration: {max_iters} batches x {calib_batch} = ~{max_iters * calib_batch} images)...")
    mtq.quantize(model_yolo.model, qat_config, forward_loop)

    # ── 3. Apply exclusions ────────────────────────────────────────
    if exclude:
        print("Resolving exclusion patterns:")
        _apply_quantizer_exclusions(model_yolo.model, exclude)

    print("Quantization summary:")
    mtq.print_quant_summary(model_yolo.model)

    # ── 4. QAT fine-tuning ─────────────────────────────────────────
    # Inject quantized model directly into the trainer. setup_model()
    # checks isinstance(self.model, nn.Module) and returns immediately,
    # preserving the TensorQuantizer wrappers.
    # FakeQuantize uses STE for backward pass — no manual hooks needed.
    print("Starting Quantization-Aware Training (QAT)...")
    qat_trainer = QATDetectionTrainer(overrides=dict(
        model=str(model_path),
        data=str(data_path),
        imgsz=imgsz,
        epochs=epochs,
        batch=batch,
        lr0=lr,
        device=device,
        workers=0,
    ))
    qat_trainer.model = model_yolo.model
    qat_trainer.train()

    # ── 5. Save ───────────────────────────────────────────────────
    model_yolo.model = qat_trainer.model
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model_yolo.save(str(output_path))
    print(f"Saved QAT model to: {output_path}")
    print(f"To export to ONNX with QDQ nodes: orinflow-export -m {output_path.name}")

    return output_path
