from __future__ import annotations

import cv2
import numpy as np


VALID_COLOR_SPACES = {"bgr", "ycrcb", "lab"}


def _validate_kernel_size(kernel_size: int) -> int:
    if not isinstance(kernel_size, int):
        raise TypeError("kernel_size must be int")
    if kernel_size <= 0 or kernel_size % 2 == 0:
        raise ValueError("kernel_size must be a positive odd integer")
    return kernel_size


def _restore_luminance(original_bgr: np.ndarray, processed_bgr: np.ndarray) -> np.ndarray:
    original_ycrcb = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2YCrCb)
    processed_ycrcb = cv2.cvtColor(processed_bgr, cv2.COLOR_BGR2YCrCb)
    processed_ycrcb[:, :, 0] = original_ycrcb[:, :, 0]
    return cv2.cvtColor(processed_ycrcb, cv2.COLOR_YCrCb2BGR)


def apply_median_filter(
    image_bgr: np.ndarray,
    kernel_size: int = 3,
    color_space: str = "bgr",
    preserve_luminance: bool = False,
) -> np.ndarray:
    if image_bgr is None:
        raise ValueError("image_bgr is None")
    if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise ValueError(f"Expected BGR HxWx3 image, got shape={getattr(image_bgr, 'shape', None)}")

    kernel_size = _validate_kernel_size(kernel_size)
    color_space = color_space.lower()
    if color_space not in VALID_COLOR_SPACES:
        raise ValueError(f"Unsupported color_space: {color_space}. Valid options: {sorted(VALID_COLOR_SPACES)}")

    img = image_bgr.copy()

    if color_space == "bgr":
        result = cv2.medianBlur(img, kernel_size)
    elif color_space == "ycrcb":
        ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        ycrcb[:, :, 0] = cv2.medianBlur(ycrcb[:, :, 0], kernel_size)
        result = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
    else:  # lab
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        lab[:, :, 0] = cv2.medianBlur(lab[:, :, 0], kernel_size)
        result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    if preserve_luminance:
        result = _restore_luminance(image_bgr, result)

    return np.clip(result, 0, 255).astype(np.uint8)
