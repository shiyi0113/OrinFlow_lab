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
CALIB_DIR = DATA_DIR / "calib"

# Default settings
DEFAULT_INPUT_SIZE = 640
DEFAULT_OPSET = 19
DEFAULT_CALIB_SAMPLES = 500


def ensure_dirs():
    """Ensure all required directories exist."""
    for d in [SOURCE_MODELS_DIR, ONNX_RAW_DIR, ONNX_OPTIMIZED_DIR,DATA_DIR]:
        d.mkdir(parents=True, exist_ok=True)
