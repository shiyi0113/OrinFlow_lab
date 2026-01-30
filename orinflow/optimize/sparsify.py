"""2:4 Structured sparsity utilities using ModelOpt."""

from typing import Any, Callable


def sparsify_model(
    model,
    mode: str = "sparsegpt",
    data_loader=None,
    collect_func: Callable | None = None,
    forward_loop: Callable | None = None,
) -> tuple[Any, dict]:
    """
    Apply 2:4 structured sparsity to PyTorch model.

    2:4 sparsity means 2 out of every 4 consecutive weights are zero.
    This enables hardware acceleration on NVIDIA Tensor Cores (Ampere+).

    Args:
        model: PyTorch model (nn.Module)
        mode: Sparsification mode
            - "sparsegpt": Hessian-based optimal sparsification (recommended, better accuracy)
            - "sparse_magnitude": Magnitude-based sparsification (faster, slightly lower accuracy)
        data_loader: Calibration data loader
        collect_func: Function to extract model input from batch
        forward_loop: Alternative to data_loader + collect_func

    Returns:
        Tuple of (sparsified_model, search_result)
    """
    import modelopt.torch.sparsity as mts

    config = {"verbose": True}

    if forward_loop is not None:
        config["forward_loop"] = forward_loop
    elif data_loader is not None:
        config["data_loader"] = data_loader
        if collect_func is None:
            # Default collect function
            import torch

            collect_func = lambda x: x[0].cuda() if torch.cuda.is_available() else x[0]
        config["collect_func"] = collect_func
    else:
        raise ValueError("Either data_loader or forward_loop must be provided")

    print(f"Sparsifying model with {mode} mode (2:4 structured sparsity)...")
    model, search_result = mts.sparsify(model, mode=mode, config=config)
    print("Sparsification complete")

    return model, search_result


def export_sparse_model(model):
    """
    Export sparse model by freezing sparse weights.

    After calling this, the sparsity mask is no longer enforced.
    Call this before ONNX export.

    Args:
        model: Sparsified PyTorch model

    Returns:
        Exported model with frozen sparse weights
    """
    import modelopt.torch.sparsity as mts

    print("Exporting sparse model (freezing sparse weights)...")
    model = mts.export(model)
    print("Sparse export complete")
    return model


def save_sparse_state(model, path: str) -> None:
    """Save sparse model state for later fine-tuning."""
    import modelopt.torch.opt as mto

    mto.save(model, path)
    print(f"Sparse state saved to: {path}")


def restore_sparse_state(model, path: str):
    """Restore sparse model from saved state."""
    import modelopt.torch.opt as mto

    model = mto.restore(model, path)
    print(f"Sparse state restored from: {path}")
    return model
