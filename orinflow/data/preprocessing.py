"""Image preprocessing utilities for YOLO models."""

import cv2
import numpy as np


def letterbox(
    img: np.ndarray,
    new_shape: tuple[int, int] = (640, 640),
    color: tuple[int, int, int] = (114, 114, 114),
    auto: bool = False,
    scale_fill: bool = False,
    scaleup: bool = True,
    stride: int = 32,
    center: bool = True,
) -> np.ndarray:
    """
    Letterbox transformation: resize image while maintaining aspect ratio and pad to target size.

    Args:
        img: Input image, shape (H, W, 3)
        new_shape: Target size (height, width)
        color: Padding color, default gray (114, 114, 114)
        auto: Minimum rectangle mode, padding aligned to stride
        scale_fill: Stretch mode, don't maintain aspect ratio
        scaleup: Allow upscaling, False means only downscale
        stride: Model stride
        center: Center the image, False means top-left alignment

    Returns:
        Processed image, shape (new_shape[0], new_shape[1], 3)
    """
    shape = img.shape[:2]  # current shape [height, width]

    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Calculate scale ratio
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:
        r = min(r, 1.0)

    # Calculate new unpadded size and padding
    new_unpad = (round(shape[1] * r), round(shape[0] * r))  # (width, height)
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]

    if auto:  # minimum rectangle, padding aligned to stride
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)
    elif scale_fill:  # stretch mode
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])

    if center:
        dw /= 2
        dh /= 2

    # Resize
    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)

    # Pad
    top, bottom = (round(dh - 0.1), round(dh + 0.1)) if center else (0, round(dh + 0.1) * 2)
    left, right = (round(dw - 0.1), round(dw + 0.1)) if center else (0, round(dw + 0.1) * 2)
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

    return img


def preprocess_for_yolo(
    image: np.ndarray,
    input_size: tuple[int, int] = (640, 640),
    stride: int = 32,
    auto: bool = False,
) -> np.ndarray:
    """
    Preprocess image for YOLO model inference/quantization calibration.

    Args:
        image: Input image, BGR format, shape (H, W, 3), dtype uint8
        input_size: Target size (height, width), default (640, 640)
        stride: Model stride, used for auto mode padding alignment, default 32
        auto: Whether to use minimum rectangle mode (dynamic shape), recommended False for calibration

    Returns:
        Preprocessed image, shape (1, 3, H, W), dtype float32, range [0, 1]
    """
    # 1. Letterbox: maintain aspect ratio + padding
    img = letterbox(image, input_size, stride=stride, auto=auto)

    # 2. BGR -> RGB
    img = img[..., ::-1]

    # 3. HWC -> CHW, add batch dimension
    img = img.transpose((2, 0, 1))[None]

    # 4. Contiguous memory
    img = np.ascontiguousarray(img)

    # 5. Normalize to [0, 1]
    img = img.astype(np.float32) / 255.0

    return img
