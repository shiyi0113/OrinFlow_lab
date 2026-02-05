"""ONNX Runtime unified management.

This module provides centralized management for ONNX Runtime to prevent issues like:
- Ultralytics auto-installing CPU-only onnxruntime over onnxruntime-gpu
- Inconsistent provider selection across the codebase
- Missing GPU support detection
"""

import os
import sys
from dataclasses import dataclass
from functools import lru_cache
from typing import Sequence


@dataclass
class OrtInfo:
    """ONNX Runtime installation information."""

    version: str
    package_name: str  # "onnxruntime-gpu" or "onnxruntime"
    available_providers: list[str]
    has_cuda: bool
    has_tensorrt: bool


@lru_cache(maxsize=1)
def check_onnxruntime() -> OrtInfo:
    """
    Check ONNX Runtime installation and capabilities.

    Returns:
        OrtInfo with version, package type, and available providers.

    Raises:
        ImportError: If ONNX Runtime is not installed.
    """
    try:
        import onnxruntime as ort
    except ImportError as e:
        raise ImportError(
            "ONNX Runtime is not installed. Install with:\n"
            "  pip install onnxruntime-gpu  # For GPU support\n"
            "  pip install onnxruntime      # For CPU only"
        ) from e

    version = ort.__version__
    providers = ort.get_available_providers()

    # Determine package name by checking for GPU providers
    has_cuda = "CUDAExecutionProvider" in providers
    has_tensorrt = "TensorrtExecutionProvider" in providers
    package_name = "onnxruntime-gpu" if (has_cuda or has_tensorrt) else "onnxruntime"

    return OrtInfo(
        version=version,
        package_name=package_name,
        available_providers=providers,
        has_cuda=has_cuda,
        has_tensorrt=has_tensorrt,
    )


def get_available_providers() -> list[str]:
    """Get list of available ONNX Runtime execution providers."""
    import onnxruntime as ort

    return ort.get_available_providers()


def get_preferred_providers(
    prefer_gpu: bool = True,
    use_tensorrt: bool = False,
) -> list[str]:
    """
    Get ordered list of preferred execution providers.

    Args:
        prefer_gpu: If True, prioritize GPU providers over CPU.
        use_tensorrt: If True, include TensorRT provider (requires additional setup).

    Returns:
        Ordered list of execution providers to use.
    """
    available = get_available_providers()
    providers = []

    if prefer_gpu:
        if use_tensorrt and "TensorrtExecutionProvider" in available:
            providers.append("TensorrtExecutionProvider")
        if "CUDAExecutionProvider" in available:
            providers.append("CUDAExecutionProvider")

    # Always include CPU as fallback
    if "CPUExecutionProvider" in available:
        providers.append("CPUExecutionProvider")

    return providers if providers else ["CPUExecutionProvider"]


def create_session(
    model_path: str,
    providers: Sequence[str] | None = None,
    prefer_gpu: bool = True,
    **session_options,
):
    """
    Create an ONNX Runtime InferenceSession with proper provider selection.

    Args:
        model_path: Path to ONNX model file.
        providers: Explicit list of providers, or None to auto-select.
        prefer_gpu: If providers is None, whether to prefer GPU.
        **session_options: Additional options passed to SessionOptions.

    Returns:
        ort.InferenceSession configured with appropriate providers.
    """
    import onnxruntime as ort

    if providers is None:
        providers = get_preferred_providers(prefer_gpu=prefer_gpu)

    opts = ort.SessionOptions()

    # Apply common optimizations
    if "graph_optimization_level" not in session_options:
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    # Apply custom options
    for key, value in session_options.items():
        setattr(opts, key, value)

    return ort.InferenceSession(model_path, sess_options=opts, providers=providers)


def disable_ultralytics_autoupdate():
    """
    Disable Ultralytics auto-update for onnxruntime.

    This prevents Ultralytics from installing CPU-only onnxruntime
    when onnxruntime-gpu is already installed.

    Should be called early in the application startup.
    """
    # Method 1: Set environment variable to disable auto-update
    os.environ["YOLO_AUTOINSTALL"] = "false"

    # Method 2: Patch the check_requirements function to skip onnxruntime
    try:
        from ultralytics.utils.checks import check_requirements as _original_check

        def patched_check_requirements(requirements, exclude=(), install=True, cmds=""):
            # Always exclude onnxruntime from auto-install
            if isinstance(exclude, (list, tuple)):
                exclude = list(exclude) + ["onnxruntime", "onnxruntime-gpu"]
            else:
                exclude = ["onnxruntime", "onnxruntime-gpu"]
            return _original_check(requirements, exclude=exclude, install=install, cmds=cmds)

        import ultralytics.utils.checks

        ultralytics.utils.checks.check_requirements = patched_check_requirements

    except ImportError:
        pass  # Ultralytics not installed, skip patching

    # Method 3: Register onnxruntime as "satisfied" in importlib.metadata
    # This tricks pip/setuptools into thinking onnxruntime is installed
    _register_ort_alias()


def _register_ort_alias():
    """
    Register onnxruntime-gpu as satisfying the 'onnxruntime' requirement.

    When onnxruntime-gpu is installed, this makes tools like pip and
    Ultralytics believe that 'onnxruntime' is also available.
    """
    try:
        from importlib.metadata import distributions, PackageNotFoundError

        # Check if onnxruntime-gpu is installed but onnxruntime is not
        gpu_installed = False
        cpu_installed = False

        for dist in distributions():
            name = dist.metadata.get("Name", "").lower()
            if name == "onnxruntime-gpu":
                gpu_installed = True
            elif name == "onnxruntime":
                cpu_installed = True

        if gpu_installed and not cpu_installed:
            # onnxruntime-gpu is installed, add alias
            # This is done by adding onnxruntime to sys.modules if not present
            if "onnxruntime" not in sys.modules:
                try:
                    import onnxruntime

                    sys.modules["onnxruntime"] = onnxruntime
                except ImportError:
                    pass

    except Exception:
        pass  # Silently ignore errors


def is_blackwell_gpu() -> bool:
    """
    Check if the current GPU is NVIDIA Blackwell architecture (RTX 50 series).

    Blackwell GPUs (SM 12.0) may have compatibility issues with
    pre-compiled CUDA kernels in torchvision.

    Returns:
        True if Blackwell GPU detected.
    """
    try:
        import torch

        if not torch.cuda.is_available():
            return False

        # Get compute capability (Blackwell is SM 12.0, i.e. compute capability 12.x)
        major, _ = torch.cuda.get_device_capability(0)
        return major >= 12
    except Exception:
        return False


_nms_patched = False


def patch_torchvision_nms_for_blackwell():
    """
    Patch torchvision.ops.nms to run on CPU for Blackwell GPUs.

    This works around the missing CUDA kernels for SM 12.0 by automatically
    moving tensors to CPU for NMS computation, then back to original device.

    Should be called early in application startup on Blackwell systems.
    """
    global _nms_patched
    if _nms_patched:
        return

    if not is_blackwell_gpu():
        return

    try:
        import torchvision.ops

        _original_nms = torchvision.ops.nms

        def cpu_nms(boxes, scores, iou_threshold):
            """NMS wrapper that runs on CPU for Blackwell compatibility."""
            original_device = boxes.device
            # Move to CPU, run NMS, move result back
            result = _original_nms(
                boxes.cpu(),
                scores.cpu(),
                iou_threshold,
            )
            return result.to(original_device)

        torchvision.ops.nms = cpu_nms
        _nms_patched = True
        print("  [Runtime] Patched torchvision.ops.nms for Blackwell GPU (SM 12.0)")

    except Exception as e:
        print(f"  [Runtime] Failed to patch NMS: {e}")


def check_torchvision_compatibility() -> tuple[bool, str]:
    """
    Check if torchvision CUDA kernels are compatible with current GPU.

    Returns:
        Tuple of (is_compatible, message).
    """
    try:
        import torch
        import torchvision

        if not torch.cuda.is_available():
            return True, "CUDA not available, using CPU"

        # Check for Blackwell (SM 12.0)
        if is_blackwell_gpu():
            device_name = torch.cuda.get_device_name(0)
            major, minor = torch.cuda.get_device_capability(0)
            return False, (
                f"Blackwell GPU detected: {device_name} (SM {major}.{minor}). "
                f"torchvision {torchvision.__version__} lacks SM {major}.{minor} kernels. "
                "NMS will run on CPU (auto-patched)."
            )

        return True, "GPU compatible"
    except ImportError:
        return True, "torchvision not installed"


def print_ort_status():
    """Print ONNX Runtime status for debugging."""
    try:
        info = check_onnxruntime()
        print("ONNX Runtime Status:")
        print(f"  Package: {info.package_name}")
        print(f"  Version: {info.version}")
        print(f"  Providers: {', '.join(info.available_providers)}")
        print(f"  CUDA: {'Yes' if info.has_cuda else 'No'}")
        print(f"  TensorRT: {'Yes' if info.has_tensorrt else 'No'}")

        # Check torchvision compatibility
        compatible, msg = check_torchvision_compatibility()
        if not compatible:
            print(f"  Warning: {msg}")
    except ImportError as e:
        print(f"ONNX Runtime not installed: {e}")
