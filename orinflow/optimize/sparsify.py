"""2:4 structured sparsity with Sparsity-Aware Training (SAT).

Uses NVIDIA ModelOpt's SparseGPT/magnitude pruning to apply 2:4 sparsity,
then fine-tunes with SAT to recover accuracy.

ModelOpt's DynamicModule (SparseConv2d etc.) automatically enforces masks
in the forward pass during training. However these dynamic classes can't be
pickled by torch.save, which ultralytics calls every epoch for checkpoints.
We solve this by subclassing DetectionTrainer and overriding save_model()
to temporarily swap in exported (pickle-safe) copies of the model for saving.

Typical workflow:
    orinflow-sparsify -m yolo26x.pt --data coco_medium.yaml --epochs 10
"""

import copy
import fnmatch
from pathlib import Path

import torch

import modelopt.torch.sparsity as mts

from orinflow.config import DATA_DIR, SOURCE_MODELS_DIR


def check_sparsity(model: torch.nn.Module) -> float:
    """Compute and print global sparsity ratio for weight tensors (ndim >= 2).

    Handles both regular models (counts zeros in param.data) and ModelOpt
    DynamicModule models (reads _weight_mask buffers, since the underlying
    param.data is not zeroed until export).

    Returns:
        Sparsity ratio in [0, 1].
    """
    masked = 0
    total = 0
    # Try DynamicModule masks first (pre-export state)
    for module in model.modules():
        mask = getattr(module, "_weight_mask", None)
        if mask is not None:
            masked += torch.sum(~mask).item()
            total += mask.numel()
    if total > 0:
        ratio = masked / total
        print(f"Global Sparsity: {100 * ratio:.2f}% (from DynamicModule masks)")
        return ratio
    # Fallback: count zeros in param.data (post-export state)
    zeros = 0
    for param in model.parameters():
        if param.ndim >= 2:
            zeros += torch.sum(param == 0).item()
            total += param.numel()
    ratio = zeros / total if total > 0 else 0.0
    print(f"Global Sparsity: {100 * ratio:.2f}%")
    return ratio


def _build_sparsity_rules(model: torch.nn.Module, exclude: list[str]) -> dict:
    """Build ModelOpt sparsity rules from exclude glob patterns.

    Scans the model for Conv2d/Linear layers and matches them against the
    provided patterns.  Patterns without wildcards are auto-wrapped as
    ``*pattern*`` for convenience.

    Args:
        model: The PyTorch model.
        exclude: Glob patterns (fnmatch-style) for layers to exclude.

    Returns:
        Rules dict compatible with ModelOpt's ``(mode, rules)`` tuple.
    """
    rules: dict[str, dict[str, dict | None]] = {
        "nn.Conv2d": {"*": {}},
        "nn.Linear": {"*": {}},
    }

    # Collect Conv2d/Linear layer names for matching preview
    target_layers = {
        name: module
        for name, module in model.named_modules()
        if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear))
    }

    for pattern in exclude:
        glob_pat = pattern if ("*" in pattern or "?" in pattern) else f"*{pattern}*"
        matched = [n for n in target_layers if fnmatch.fnmatch(n, glob_pat)]
        if matched:
            preview = ", ".join(matched[:5])
            suffix = f" ... ({len(matched)} total)" if len(matched) > 5 else ""
            print(f"  Exclude '{pattern}' -> {preview}{suffix}")
        else:
            print(f"  Warning: '{pattern}' did not match any Conv2d/Linear layers.")
        rules["nn.Conv2d"][glob_pat] = None
        rules["nn.Linear"][glob_pat] = None

    return rules


def _make_sparse_trainer_cls():
    """Create a DetectionTrainer subclass that can checkpoint DynamicModules.

    Deferred import to avoid loading ultralytics at module level.
    """
    from ultralytics.models.yolo.detect import DetectionTrainer

    class SparseDetectionTrainer(DetectionTrainer):
        """DetectionTrainer that handles ModelOpt DynamicModule checkpointing.

        DynamicModule (SparseConv2d, SparseLinear) can be deepcopied but not
        pickled.  ultralytics' save_model() does deepcopy → torch.save, so we
        intercept and export the copies to regular modules before pickling.
        """

        def _setup_train(self):
            """Enable gradients on all float params before setup.

            DynamicModule wrappers may set some params to requires_grad=False.
            The base _setup_train() would force them back to True with a noisy
            "setting requires_grad=True for frozen layer" warning.  Since SAT
            needs gradients on all params anyway (sparsity is enforced by masks
            in the forward pass, not by disabling gradients), we pre-enable
            them to avoid the warnings.
            """
            for v in self.model.parameters():
                if not v.requires_grad and v.dtype.is_floating_point:
                    v.requires_grad = True
            super()._setup_train()

        def save_model(self):
            """Export DynamicModule copies to pickle-safe modules before saving."""
            orig_ema = self.ema.ema
            try:
                self.ema.ema = mts.export(copy.deepcopy(orig_ema))
                super().save_model()
            finally:
                self.ema.ema = orig_ema

    return SparseDetectionTrainer


def sparsify_and_finetune(
    model_name: str,
    data_yaml: str,
    mode: str = "sparsegpt",
    epochs: int = 10,
    batch: int = 16,
    lr: float = 1e-4,
    optimizer: str = "SGD",
    workers: int = 8,
    calib_batch: int = 4,
    calib_images: int = 512,
    exclude: list[str] | None = None,
    imgsz: int = 640,
    device: int = 0,
    output_path: Path | None = None,
) -> Path:
    """Apply 2:4 structured sparsity and fine-tune with SAT.

    Pipeline:
        1. Load YOLO model
        2. Sparsify with SparseGPT / magnitude (creates DynamicModule wrappers)
        3. SAT fine-tuning (masks enforced automatically by DynamicModule)
        4. Export (bake masks into weights, remove wrappers)
        5. Save final sparse model

    Args:
        model_name: Source model filename (e.g. "yolo26x.pt")
        data_yaml: Dataset YAML filename (e.g. "coco_medium.yaml")
        mode: Sparsity algorithm — "sparsegpt" (needs calibration data) or "sparse_magnitude"
        epochs: SAT fine-tuning epochs
        batch: Training batch size
        lr: Learning rate for SAT
        optimizer: Optimizer name (e.g. "AdamW", "SGD", "Adam", "RAdam").
        workers: Number of dataloader workers.
        calib_batch: Batch size for SparseGPT calibration
        calib_images: Max number of images for SparseGPT calibration (default 512)
        exclude: Glob patterns for layers to exclude from sparsification (fnmatch-style).
            Patterns without wildcards are auto-wrapped as ``*pattern*``.
            E.g. ``["model.0.", "model.22."]`` excludes stem and detect head.
        imgsz: Input image size
        device: CUDA device ID
        output_path: Output .pt path. Defaults to models/source/{stem}_sparse.pt

    Returns:
        Path to saved sparse model.
    """
    from ultralytics import YOLO

    SparseDetectionTrainer = _make_sparse_trainer_cls()

    model_path = SOURCE_MODELS_DIR / model_name
    data_path = DATA_DIR / data_yaml
    cuda_device = f"cuda:{device}"

    if output_path is None:
        output_path = SOURCE_MODELS_DIR / f"{model_path.stem}_sparse.pt"
    output_path = Path(output_path)

    # ── 1. Load model ──────────────────────────────────────────────
    print(f"Loading model: {model_path}")
    model_yolo = YOLO(str(model_path))
    model_yolo.to(cuda_device)
    model_yolo.eval()

    # ── 2. Sparsify ───────────────────────────────────────────────
    # Build calibration dataloader via a temporary trainer.
    # setup_model() is required because build_dataset accesses self.model.stride.
    tmp_trainer = SparseDetectionTrainer(overrides=dict(
        model=str(model_path),
        data=str(data_path),
        imgsz=imgsz,
    ))
    tmp_trainer.setup_model()
    calib_loader = tmp_trainer.get_dataloader(
        tmp_trainer.data["train"], batch_size=calib_batch, rank=-1, mode="train",
    )

    def collect_func(batch):
        return batch["img"].to(cuda_device).float() / 255.0

    max_iters = max(1, calib_images // calib_batch)
    print(f"Applying {mode} sparsification (calibration: {max_iters} batches × {calib_batch} = ~{max_iters * calib_batch} images)...")
    sparsify_config = {
        "data_loader": calib_loader,
        "collect_func": collect_func,
        "max_iter_data_loader": max_iters,
    }

    # Build mode spec: plain string or (mode, rules) tuple with exclusions
    if exclude:
        print("Resolving exclusion patterns:")
        rules = _build_sparsity_rules(model_yolo.model, exclude)
        sparsify_mode = (mode, rules)
    else:
        sparsify_mode = mode

    mts.sparsify(model_yolo.model, mode=sparsify_mode, config=sparsify_config)

    print("Post-sparsification:")
    check_sparsity(model_yolo.model)

    # ── 3. SAT fine-tuning ─────────────────────────────────────────
    # Inject sparsified model directly into the trainer. setup_model()
    # checks isinstance(self.model, nn.Module) and returns immediately
    # (trainer.py:677), preserving the DynamicModule wrappers.
    # Masks are enforced automatically by DynamicModule._get_weight()
    # on every forward pass — no manual hooks needed.
    # SparseDetectionTrainer.save_model() handles the pickle issue by
    # exporting a deepcopy before torch.save.
    print("Starting Sparsity-Aware Training (SAT)...")
    sat_trainer = SparseDetectionTrainer(overrides=dict(
        model=str(model_path),
        data=str(data_path),
        imgsz=imgsz,
        epochs=epochs,
        batch=batch,
        optimizer=optimizer,
        lr0=lr,
        device=device,
        workers=workers,
    ))
    sat_trainer.model = model_yolo.model
    sat_trainer.train()

    print("Post-SAT:")
    check_sparsity(sat_trainer.model)

    # ── 4. Export: bake masks into weights ─────────────────────────
    print("Exporting sparse model (baking masks into weights)...")
    final_model = mts.export(sat_trainer.model)

    print("Post-export:")
    check_sparsity(final_model)

    # ── 5. Save ───────────────────────────────────────────────────
    model_yolo.model = final_model
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model_yolo.save(str(output_path))
    print(f"Saved sparse model to: {output_path}")

    return output_path
