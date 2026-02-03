"""Quantization-Aware Training (QAT) with NVIDIA ModelOpt.

Uses ModelOpt's ``mtq.quantize()`` to insert FakeQuantize nodes (TensorQuantizer)
into the model, calibrates activation ranges, then fine-tunes with QAT so the
model learns to compensate for quantization error via the Straight-Through
Estimator (STE).

While TensorQuantizer itself is a regular nn.Module, ``mtq.quantize()`` wraps
layers in DynamicModule subclasses (QuantConv2d, QuantLinear) that CANNOT be
pickled.  The QATDetectionTrainer's ``save_model()`` deepcopies the model and
calls ``DynamicModule.export()`` to restore pickle-safe classes for intermediate
training checkpoints.

To preserve QDQ information for TensorRT INT8 deployment, the pipeline exports
ONNX directly from the in-memory model (with DynamicModule wrappers intact).
FakeQuantize nodes become Q/DQ (QuantizeLinear/DequantizeLinear) ONNX ops via
TensorQuantizer's symbolic functions.

Typical workflow:
    orinflow-qat -m yolo11n.pt --data coco128.yaml --mode int8 --epochs 10
    # Produces .onnx with QDQ nodes in models/onnx/raw/
    # Use with trtexec --int8 for deployment
"""

import copy
import fnmatch
from pathlib import Path

import torch

import modelopt.torch.quantization as mtq

from orinflow.config import DATA_DIR, DEFAULT_OPSET, ONNX_OPTIMIZED_DIR, SOURCE_MODELS_DIR, print_trtexec_hint

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


def _export_qat_onnx(
    model: torch.nn.Module,
    output_path: Path,
    imgsz: int,
    device: str,
    opset: int = DEFAULT_OPSET,
) -> Path:
    """Export QAT model to ONNX with QDQ nodes preserved.

    Uses ``torch.onnx.export()`` directly instead of ``YOLO.export()`` because
    the latter calls ``model.fuse()`` which destroys QuantConv2d/TensorQuantizer
    wrappers.

    TensorQuantizer's ``symbolic()`` method automatically converts FakeQuantize
    ops to QuantizeLinear/DequantizeLinear (Q/DQ) ONNX nodes during tracing.

    Args:
        model: The QAT-trained model with DynamicModule wrappers still intact.
        output_path: Path for the output .onnx file.
        imgsz: Input image size (square).
        device: CUDA device string (e.g. "cuda:0").
        opset: ONNX opset version (>= 13 for QDQ ops).

    Returns:
        Path to the exported ONNX file.
    """
    export_model = copy.deepcopy(model)
    export_model.eval()
    export_model.float()

    # Set export mode on Detect head so it outputs raw predictions (no NMS).
    # This mirrors what ultralytics' Exporter does before torch.onnx.export().
    for m in export_model.modules():
        if hasattr(m, "export"):
            m.export = True

    dummy = torch.randn(1, 3, imgsz, imgsz, device=device)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        export_model,
        dummy,
        str(output_path),
        opset_version=opset,
        do_constant_folding=True,
        input_names=["images"],
        output_names=["output0"],
        dynamo=False,
    )

    return output_path


def _export_dynamic_modules(model: torch.nn.Module) -> None:
    """Recursively export all DynamicModules back to regular modules in-place.

    DynamicModule.export() restores __class__ to the original (e.g. QuantConv2d -> Conv2d),
    making the module pickle-safe. This is the general mechanism underlying mts.export()
    for sparsity; for quantization, mtq has no public export() yet, so we call
    DynamicModule.export() directly.
    """
    from modelopt.torch.opt.dynamic import DynamicModule

    for _name, child in list(model.named_children()):
        if isinstance(child, DynamicModule):
            child.export()
        _export_dynamic_modules(child)


def _make_qat_trainer_cls():
    """Create a DetectionTrainer subclass for QAT fine-tuning.

    Deferred import to avoid loading ultralytics at module level.
    """
    from ultralytics.models.yolo.detect import DetectionTrainer

    class QATDetectionTrainer(DetectionTrainer):
        """DetectionTrainer that handles ModelOpt QuantModule (DynamicModule).

        QuantConv2d/QuantLinear are dynamically generated DynamicModule classes
        that cannot be pickled (same issue as sparsity's SparseConv2d).
        We override save_model() to deepcopy + export DynamicModules back to
        regular modules before torch.save.

        Also overrides _setup_train() to suppress requires_grad warnings.
        """

        def _setup_train(self):
            """Enable gradients on all float params before base setup."""
            for v in self.model.parameters():
                if not v.requires_grad and v.dtype.is_floating_point:
                    v.requires_grad = True
            super()._setup_train()

        def save_model(self):
            """Export DynamicModule copies to pickle-safe modules before saving."""
            orig_ema = self.ema.ema
            try:
                ema_copy = copy.deepcopy(orig_ema)
                _export_dynamic_modules(ema_copy)
                self.ema.ema = ema_copy
                super().save_model()
            finally:
                self.ema.ema = orig_ema

        def final_eval(self):
            """Run final evaluation, catching CUDA kernel errors from NMS warmup.

            final_eval() loads the saved checkpoint via AutoBackend, whose NMS
            warmup may fail on GPUs without matching torchvision CUDA kernels
            (e.g. Blackwell/RTX 50xx). Per-epoch validation is unaffected since
            it uses the in-memory model directly.
            """
            try:
                return super().final_eval()
            except Exception as e:
                if "no kernel image" in str(e):
                    print(f"Warning: Final validation skipped ({type(e).__name__}: {e})")
                    print("Training completed. Run orinflow-evaluate separately for metrics.")
                else:
                    raise

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
) -> Path:
    """Apply quantization and fine-tune with QAT.

    Pipeline:
        1. Load YOLO model
        2. Insert FakeQuantize nodes via mtq.quantize() + calibrate
        3. Optionally disable quantizers on excluded layers
        4. QAT fine-tuning (STE-based gradient flow through FakeQuantize)
        5. Export ONNX with QDQ nodes (from in-memory model)

    The .onnx output contains Q/DQ (QuantizeLinear/DequantizeLinear) nodes and
    can be used directly with ``trtexec --int8`` for TensorRT INT8 deployment.

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

    Returns:
        Path to exported ONNX file with QDQ nodes.
    """
    from ultralytics import YOLO

    QATDetectionTrainer = _make_qat_trainer_cls()

    model_path = SOURCE_MODELS_DIR / model_name
    data_path = DATA_DIR / data_yaml
    cuda_device = f"cuda:{device}"

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
        amp=False,
    ))
    qat_trainer.model = model_yolo.model
    qat_trainer.train()

    # ── 5. Export ONNX (with QDQ nodes) ─────────────────────────
    # Must happen BEFORE DynamicModule stripping, since TensorQuantizer's
    # symbolic() method converts FakeQuantize -> Q/DQ during torch.onnx.export().
    # We use torch.onnx.export() directly (not YOLO.export()) because the latter
    # calls model.fuse() which destroys QuantConv2d wrappers.
    stem = Path(model_name).stem
    onnx_path = ONNX_OPTIMIZED_DIR / f"{stem}_{mode.upper()}_qat.onnx"
    print(f"Exporting ONNX with QDQ nodes: {onnx_path}")
    _export_qat_onnx(qat_trainer.model, onnx_path, imgsz, cuda_device)
    print(f"Saved QAT ONNX to: {onnx_path}")
    print_trtexec_hint(onnx_path)

    return onnx_path
