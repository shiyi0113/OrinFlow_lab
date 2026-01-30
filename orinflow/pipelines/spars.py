"""Path C: Sparse Optimization Pipeline (Tensor Core Acceleration).

Steps:
1. Apply 2:4 structured sparsity
2. (Optional) Sparsity-Aware Training (SAT)
3. INT8 quantization
4. Export to ONNX

Use case:
- Target platform supports 2:4 sparsity (Jetson Orin, Ampere+ GPUs)
- Want additional ~2x acceleration from Tensor Core sparsity
- Acceptable accuracy requirements (~1-2% loss)

Hardware requirements:
- NVIDIA Ampere architecture or newer (A100, RTX 30xx, Jetson Orin)
- TensorRT 10.0+
"""

from pathlib import Path
from typing import Callable

import torch

from orinflow.config import ONNX_QUANT_DIR, PYTORCH_MODELS_DIR


def run_sparse_optimization(
    model_name: str,
    calib_loader,
    collect_func: Callable | None = None,
    train_loader=None,
    val_loader=None,
    score_func: Callable | None = None,
    sparse_mode: str = "sparsegpt",
    enable_sat: bool = False,
    sat_epochs: int = 20,
    enable_qat: bool = False,
    qat_epochs: int = 10,
    learning_rate: float = 1e-4,
    device: str = "cuda",
) -> dict:
    """
    Run Path C: Sparse optimization pipeline.

    Args:
        model_name: Model name (without .pt extension)
        calib_loader: Calibration data loader for sparsification
        collect_func: Function to extract model input from batch
        train_loader: Training data loader (required for SAT/QAT)
        val_loader: Validation data loader
        score_func: Function to evaluate model
        sparse_mode: Sparsification mode (sparsegpt, sparse_magnitude)
        enable_sat: Enable Sparsity-Aware Training
        sat_epochs: Number of SAT epochs
        enable_qat: Enable Quantization-Aware Training after sparsity
        qat_epochs: Number of QAT epochs
        learning_rate: Learning rate for training
        device: Device to train on

    Returns:
        Pipeline result dict
    """
    import modelopt.torch.opt as mto
    import modelopt.torch.quantization as mtq
    import modelopt.torch.sparsity as mts
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import CosineAnnealingLR

    print("=" * 60)
    print("Path C: Sparse Optimization Pipeline (2:4 Sparsity)")
    print("=" * 60)

    result = {"model_name": model_name, "path": "C", "steps": []}

    # Default collect function
    if collect_func is None:
        collect_func = lambda x: x[0].to(device)

    # Load model
    model_path = PYTORCH_MODELS_DIR / f"{model_name}.pt"
    print(f"\nLoading model: {model_path}")
    model = torch.load(model_path)
    model.to(device)
    model.eval()

    # Step 1: Sparsification
    print(f"\n[Step 1/4] Applying 2:4 sparsity ({sparse_mode})...")
    try:
        model, search_result = mts.sparsify(
            model,
            mode=sparse_mode,
            config={
                "data_loader": calib_loader,
                "collect_func": collect_func,
                "verbose": True,
            },
        )

        result["steps"].append({"name": "sparsify", "status": "success", "mode": sparse_mode})

        # Save sparse state
        sparse_state_path = PYTORCH_MODELS_DIR / f"{model_name}_sparse.pth"
        mto.save(model, sparse_state_path)
        print(f"Sparse state saved to: {sparse_state_path}")

    except Exception as e:
        result["steps"].append({"name": "sparsify", "status": "failed", "error": str(e)})
        print(f"Sparsification failed: {e}")
        return result

    # Step 2: SAT (optional)
    if enable_sat:
        if train_loader is None:
            print("Warning: SAT enabled but no train_loader provided, skipping")
        else:
            print(f"\n[Step 2/4] Sparsity-Aware Training for {sat_epochs} epochs...")
            try:
                model.train()
                optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
                scheduler = CosineAnnealingLR(optimizer, T_max=sat_epochs, eta_min=1e-6)

                for epoch in range(sat_epochs):
                    for batch in train_loader:
                        images, targets = batch
                        images = images.to(device)

                        optimizer.zero_grad()
                        loss = model(images, targets)
                        loss.backward()
                        optimizer.step()

                    scheduler.step()

                    if (epoch + 1) % 5 == 0 and score_func:
                        metric = score_func(model)
                        print(f"SAT Epoch {epoch+1}/{sat_epochs}: metric = {metric:.4f}")

                result["steps"].append({"name": "sat", "status": "success", "epochs": sat_epochs})

                # Save SAT state
                sat_state_path = PYTORCH_MODELS_DIR / f"{model_name}_sat.pth"
                mto.save(model, sat_state_path)
                print(f"SAT state saved to: {sat_state_path}")

            except Exception as e:
                result["steps"].append({"name": "sat", "status": "failed", "error": str(e)})
                print(f"SAT failed: {e}")
    else:
        print("\n[Step 2/4] Skipping SAT (disabled)")

    # Step 3: Quantization
    print("\n[Step 3/4] INT8 quantization...")
    try:
        def forward_loop(m):
            for batch in calib_loader:
                inputs = collect_func(batch)
                with torch.no_grad():
                    m(inputs)

        if enable_qat and train_loader is not None:
            # QAT
            model = mtq.quantize(model, mtq.INT8_DEFAULT_CFG, forward_loop=None)
            model.train()

            optimizer = AdamW(model.parameters(), lr=learning_rate * 0.1)

            for epoch in range(qat_epochs):
                for batch in train_loader:
                    images, targets = batch
                    images = images.to(device)

                    optimizer.zero_grad()
                    loss = model(images, targets)
                    loss.backward()
                    optimizer.step()

                if score_func:
                    metric = score_func(model)
                    print(f"QAT Epoch {epoch+1}/{qat_epochs}: metric = {metric:.4f}")

            result["steps"].append({"name": "qat", "status": "success", "epochs": qat_epochs})
        else:
            # PTQ
            model = mtq.quantize(model, mtq.INT8_DEFAULT_CFG, forward_loop)
            result["steps"].append({"name": "ptq", "status": "success"})

        print("Quantization complete")

    except Exception as e:
        result["steps"].append({"name": "quantize", "status": "failed", "error": str(e)})
        print(f"Quantization failed: {e}")
        return result

    # Step 4: Export
    print("\n[Step 4/4] Exporting sparse model to ONNX...")
    try:
        # Freeze sparse weights
        model = mts.export(model)

        model.eval()
        model.cpu()
        dummy_input = torch.randn(1, 3, 640, 640)

        output_path = ONNX_QUANT_DIR / f"{model_name}_sparse_int8.onnx"
        ONNX_QUANT_DIR.mkdir(parents=True, exist_ok=True)

        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            opset_version=17,
            input_names=["images"],
            output_names=["output"],
        )

        result["steps"].append({"name": "export", "status": "success", "output": str(output_path)})
        print(f"Sparse ONNX exported to: {output_path}")

    except Exception as e:
        result["steps"].append({"name": "export", "status": "failed", "error": str(e)})
        print(f"ONNX export failed: {e}")
        return result

    print("\n" + "=" * 60)
    print("Path C: Sparse Optimization Complete!")
    print(f"Output: {output_path}")
    print("=" * 60)
    print("\nTensorRT deployment with sparsity:")
    print(f"  trtexec --onnx={output_path} --int8 --sparsity=enable")

    return result
