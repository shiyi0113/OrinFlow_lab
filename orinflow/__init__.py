"""OrinFlow - YOLO Model Optimization Pipeline for Edge Deployment."""

__version__ = "1.0.0"

# Disable Ultralytics auto-install of onnxruntime to prevent GPU version override
# This must run early, before any Ultralytics imports
def _init_runtime():
    """Initialize runtime settings on module import."""
    try:
        from orinflow.runtime import disable_ultralytics_autoupdate

        disable_ultralytics_autoupdate()
    except ImportError:
        pass  # runtime module not yet available during install


_init_runtime()
