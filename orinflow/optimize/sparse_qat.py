"""Combined 2:4 Sparsity + Quantization-Aware Training (SAT + QAT).

Implements Route 4: applies 2:4 structured sparsity first, then quantization-aware
training on the sparsified model, producing an ONNX with both QDQ nodes and sparse
weights.

The key difference from running SAT and QAT sequentially (as separate commands) is
that sparsity masks remain active (via DynamicModule wrappers) throughout the QAT
phase, preventing gradient updates from filling in pruned weights.

Pipeline:
    1. Load YOLO model
    2. Sparsify with SparseGPT/magnitude (DynamicModule wrappers with masks)
    3. SAT fine-tuning (masks enforced by DynamicModule)
    4. Insert FakeQuantize nodes via mtq.quantize() + calibrate
    5. QAT fine-tuning (sparsity masks + FakeQuantize both active)
    6. Export ONNX with QDQ nodes (sparse weights preserved)

Typical workflow:
    orinflow-sparse-qat -m yolo26n.pt --data coco128.yaml --sat-epochs 10 --qat-epochs 10
    # Produces .onnx with QDQ nodes + sparse weights in models/onnx/raw/
    # Use with trtexec --int8 --sparsity=enable for deployment
"""

import copy
from pathlib import Path

import modelopt.torch.quantization as mtq
import modelopt.torch.sparsity as mts

from orinflow.config import DATA_DIR, ONNX_RAW_DIR, SOURCE_MODELS_DIR, print_trtexec_hint
from orinflow.optimize.qat import (
    _apply_quantizer_exclusions,
    _export_dynamic_modules,
    _export_qat_onnx,
    _histogram_recalibrate,
    _resolve_qat_config,
)
from orinflow.optimize.sparsify import _build_sparsity_rules, check_sparsity


def _make_sparse_qat_trainer_cls():
    """Create a DetectionTrainer subclass for combined SAT + QAT training.

    Deferred import to avoid loading ultralytics at module level.
    """
    from ultralytics.models.yolo.detect import DetectionTrainer

    class SparseQATDetectionTrainer(DetectionTrainer):
        """DetectionTrainer for combined sparsity + quantization training.

        Handles DynamicModule checkpointing for both SAT and QAT phases.
        Set the ``phase`` attribute to control save_model behavior:

        - "sat": exports sparsity DynamicModules via mts.export()
        - "qat": exports all DynamicModules (sparsity + quantization)
        """

        phase = "sat"

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
                if self.phase == "sat":
                    self.ema.ema = mts.export(ema_copy)
                else:
                    _export_dynamic_modules(ema_copy)
                    self.ema.ema = ema_copy
                super().save_model()
            finally:
                self.ema.ema = orig_ema

        def final_eval(self):
            """Run final evaluation, catching CUDA kernel errors from NMS warmup."""
            try:
                return super().final_eval()
            except Exception as e:
                if "no kernel image" in str(e):
                    print(f"Warning: Final validation skipped ({type(e).__name__}: {e})")
                    print("Training completed. Run orinflow-evaluate separately for metrics.")
                else:
                    raise

    return SparseQATDetectionTrainer


def sparse_quantize_aware_finetune(
    model_name: str,
    data_yaml: str,
    sparsity_mode: str = "sparsegpt",
    qat_mode: str = "int8",
    sat_epochs: int = 10,
    qat_epochs: int = 10,
    batch: int = 16,
    sat_lr: float = 1e-4,
    qat_lr: float = 1e-4,
    optimizer: str = "SGD",
    workers: int = 8,
    calib_batch: int = 4,
    calib_images: int = 512,
    exclude_sparse: list[str] | None = None,
    exclude_quant: list[str] | None = None,
    calibrator: str = "histogram",
    imgsz: int = 640,
    device: int = 0,
) -> Path:
    """Apply 2:4 sparsity then QAT, producing ONNX with QDQ nodes + sparse weights.

    Pipeline:
        1. Load YOLO model
        2. Sparsify with SparseGPT/magnitude + calibrate
        3. SAT fine-tuning (sparsity masks enforced by DynamicModule)
        4. Insert FakeQuantize nodes via mtq.quantize() + calibrate
        5. QAT fine-tuning (sparsity masks + FakeQuantize both active)
        6. Export ONNX with QDQ nodes (sparse weights preserved)

    The sparsity DynamicModule wrappers are kept throughout the QAT phase,
    ensuring that pruned weights remain zero during gradient updates.

    Args:
        model_name: Source model filename (e.g. "yolo26n.pt").
        data_yaml: Dataset YAML filename (e.g. "coco128.yaml").
        sparsity_mode: Sparsity algorithm -- "sparsegpt" or "sparse_magnitude".
        qat_mode: Quantization mode -- "int8", "fp8", "int4_awq", or "w4a8".
        sat_epochs: SAT fine-tuning epochs.
        qat_epochs: QAT fine-tuning epochs.
        batch: Training batch size (shared for SAT and QAT).
        sat_lr: Learning rate for SAT.
        qat_lr: Learning rate for QAT.
        optimizer: Optimizer name (e.g. "AdamW", "SGD", "Adam", "RAdam").
        workers: Number of dataloader workers.
        calib_batch: Batch size for calibration (sparsity + quantization).
        calib_images: Max number of images for calibration (default 512).
        exclude_sparse: Glob patterns for layers to exclude from sparsification.
        exclude_quant: Glob patterns for layers to exclude from quantization.
        calibrator: Calibrator for activation quantizers -- "histogram" (default,
            entropy-based, robust to outliers) or "max" (global max, fast but
            outlier-sensitive). Weight quantizers always use "max".
        imgsz: Input image size.
        device: CUDA device ID.

    Returns:
        Path to exported ONNX file with QDQ nodes and sparse weights.
    """
    from ultralytics import YOLO

    from orinflow.optimize.qat import CALIBRATOR_CHOICES

    if calibrator not in CALIBRATOR_CHOICES:
        valid = ", ".join(CALIBRATOR_CHOICES)
        raise ValueError(f"Unknown calibrator '{calibrator}'. Valid choices: {valid}")

    SparseQATTrainer = _make_sparse_qat_trainer_cls()

    model_path = SOURCE_MODELS_DIR / model_name
    data_path = DATA_DIR / data_yaml
    cuda_device = f"cuda:{device}"

    # ── 1. Load model ──────────────────────────────────────────────
    print(f"Loading model: {model_path}")
    model_yolo = YOLO(str(model_path))
    model_yolo.to(cuda_device)
    model_yolo.eval()

    # ── Build calibration dataloader (shared for sparsity + quantization) ──
    tmp_trainer = SparseQATTrainer(overrides=dict(
        model=str(model_path),
        data=str(data_path),
        imgsz=imgsz,
    ))
    tmp_trainer.setup_model()
    calib_loader = tmp_trainer.get_dataloader(
        tmp_trainer.data["train"], batch_size=calib_batch, rank=-1, mode="train",
    )
    max_iters = max(1, calib_images // calib_batch)

    # ── 2. Sparsify ───────────────────────────────────────────────
    def collect_func(batch_data):
        """Extract and normalize images from batch for SparseGPT calibration."""
        return batch_data["img"].to(cuda_device).float() / 255.0

    print(
        f"Applying {sparsity_mode} sparsification "
        f"(calibration: {max_iters} batches x {calib_batch} = ~{max_iters * calib_batch} images)..."
    )
    sparsify_config = {
        "data_loader": calib_loader,
        "collect_func": collect_func,
        "max_iter_data_loader": max_iters,
    }

    if exclude_sparse:
        print("Resolving sparsity exclusion patterns:")
        rules = _build_sparsity_rules(model_yolo.model, exclude_sparse)
        sparsify_mode_spec = (sparsity_mode, rules)
    else:
        sparsify_mode_spec = sparsity_mode

    mts.sparsify(model_yolo.model, mode=sparsify_mode_spec, config=sparsify_config)

    print("Post-sparsification:")
    check_sparsity(model_yolo.model)

    # ── 3. SAT fine-tuning ─────────────────────────────────────────
    print("Starting Sparsity-Aware Training (SAT)...")
    sat_trainer = SparseQATTrainer(overrides=dict(
        model=str(model_path),
        data=str(data_path),
        imgsz=imgsz,
        epochs=sat_epochs,
        batch=batch,
        optimizer=optimizer,
        lr0=sat_lr,
        device=device,
        workers=workers,
    ))
    sat_trainer.model = model_yolo.model
    sat_trainer.train()

    print("Post-SAT:")
    check_sparsity(sat_trainer.model)

    # ── 4. Quantize + Calibrate ────────────────────────────────────
    # Apply quantization ON TOP of sparsity DynamicModules.
    # The sparsity wrappers remain intact, enforcing masks during QAT.
    def forward_loop(model):
        """Forward calibration data through model for activation range collection."""
        for i, batch_data in enumerate(calib_loader):
            if i >= max_iters:
                break
            model(batch_data["img"].to(cuda_device).float() / 255.0)

    qat_config = _resolve_qat_config(qat_mode)
    print(
        f"Quantizing sparsified model with {qat_mode} mode (calibrator: {calibrator}, "
        f"calibration: {max_iters} batches x {calib_batch} = ~{max_iters * calib_batch} images)..."
    )
    mtq.quantize(sat_trainer.model, qat_config, forward_loop)

    if calibrator == "histogram":
        print("Recalibrating activations with histogram/entropy...")
        _histogram_recalibrate(sat_trainer.model, forward_loop)

    if exclude_quant:
        print("Resolving quantization exclusion patterns:")
        _apply_quantizer_exclusions(sat_trainer.model, exclude_quant)

    print("Quantization summary:")
    mtq.print_quant_summary(sat_trainer.model)

    # ── 5. QAT fine-tuning ─────────────────────────────────────────
    # Both sparsity masks and FakeQuantize are active simultaneously.
    # DynamicModule enforces sparsity in forward pass, TensorQuantizer
    # simulates quantization. Gradients flow through STE for quantization
    # and masked weights stay zero.
    print("Starting Quantization-Aware Training (QAT) on sparsified model...")
    qat_trainer = SparseQATTrainer(overrides=dict(
        model=str(model_path),
        data=str(data_path),
        imgsz=imgsz,
        epochs=qat_epochs,
        batch=batch,
        optimizer=optimizer,
        lr0=qat_lr,
        device=device,
        workers=workers,
        amp=False,
    ))
    qat_trainer.phase = "qat"
    qat_trainer.model = sat_trainer.model
    qat_trainer.train()

    print("Post-QAT:")
    check_sparsity(qat_trainer.model)

    # ── 6. Export ONNX (with QDQ nodes + sparse weights) ───────────
    # Must happen BEFORE DynamicModule stripping, since TensorQuantizer's
    # symbolic() converts FakeQuantize -> Q/DQ and sparsity masks ensure
    # zero weights are preserved in the traced graph.
    stem = Path(model_name).stem
    onnx_path = ONNX_RAW_DIR / f"{stem}_sparse_qat_{qat_mode}.onnx"
    print(f"Exporting ONNX with QDQ nodes + sparse weights: {onnx_path}")
    _export_qat_onnx(qat_trainer.model, onnx_path, imgsz, cuda_device)
    print(f"Saved sparse QAT ONNX to: {onnx_path}")
    print_trtexec_hint(onnx_path, sparse=True)

    return onnx_path
