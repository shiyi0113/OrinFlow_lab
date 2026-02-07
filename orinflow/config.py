"""Project configuration and path definitions."""

from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Model directories
MODELS_DIR = PROJECT_ROOT / "models"
SOURCE_MODELS_DIR = MODELS_DIR / "source"
ONNX_RAW_DIR = MODELS_DIR / "onnx" / "raw"
ONNX_OPTIMIZED_DIR = MODELS_DIR / "onnx" / "optimized"

# Data directories
DATA_DIR = PROJECT_ROOT / "datasets"
TEST_DIR = DATA_DIR / "test"

# Default settings
DEFAULT_INPUT_SIZE = 640
DEFAULT_OPSET = 13
DEFAULT_CALIB_SAMPLES = 500


def ensure_dirs():
    """Ensure all required directories exist."""
    for d in [SOURCE_MODELS_DIR, ONNX_RAW_DIR, ONNX_OPTIMIZED_DIR, DATA_DIR]:
        d.mkdir(parents=True, exist_ok=True)


def print_trtexec_hint(onnx_path: Path, *, sparse: bool = False) -> None:
    """Print recommended trtexec command for Jetson edge deployment.

    Args:
        onnx_path: Path to the ONNX model file.
        sparse: Whether the model has 2:4 structured sparsity.
    """
    onnx_name = onnx_path.name
    engine_name = onnx_path.stem + ".engine"

    flags = [
        f"--onnx={onnx_name}",
        f"--saveEngine={engine_name}",
        "--int8",
        "--fp16",
    ]
    if sparse:
        flags.append("--sparsity=enable")

    cmd = "trtexec " + " \\\n         ".join(flags)
    print(f"{'─' * 60}")
    print(f"Deployment command (copy to Jetson):{cmd}")
    print(f"{'─' * 60}")
