from __future__ import annotations

import cv2
import numpy as np


VALID_COLOR_SPACES = {"bgr", "ycrcb", "lab"}


def _restore_luminance(original_bgr: np.ndarray, processed_bgr: np.ndarray) -> np.ndarray:
    original_ycrcb = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2YCrCb)
    processed_ycrcb = cv2.cvtColor(processed_bgr, cv2.COLOR_BGR2YCrCb)
    processed_ycrcb[:, :, 0] = original_ycrcb[:, :, 0]
    return cv2.cvtColor(processed_ycrcb, cv2.COLOR_YCrCb2BGR)


def apply_bilateral_filter(
    image_bgr: np.ndarray,
    d: int = 5,
    sigma_color: float = 25.0,
    sigma_space: float = 25.0,
    color_space: str = "bgr",
    preserve_luminance: bool = False,
) -> np.ndarray:
    if image_bgr is None:
        raise ValueError("image_bgr is None")
    if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise ValueError(f"Expected BGR HxWx3 image, got shape={getattr(image_bgr, 'shape', None)}")
    if d <= 0:
        raise ValueError("d must be > 0")
    if sigma_color < 0 or sigma_space < 0:
        raise ValueError("sigma_color and sigma_space must be >= 0")

    color_space = color_space.lower()
    if color_space not in VALID_COLOR_SPACES:
        raise ValueError(f"Unsupported color_space: {color_space}. Valid options: {sorted(VALID_COLOR_SPACES)}")

    img = image_bgr.copy()

    if color_space == "bgr":
        result = cv2.bilateralFilter(img, d=d, sigmaColor=sigma_color, sigmaSpace=sigma_space)
    elif color_space == "ycrcb":
        ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        ycrcb[:, :, 0] = cv2.bilateralFilter(ycrcb[:, :, 0], d=d, sigmaColor=sigma_color, sigmaSpace=sigma_space)
        result = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
    else:  # lab
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        lab[:, :, 0] = cv2.bilateralFilter(lab[:, :, 0], d=d, sigmaColor=sigma_color, sigmaSpace=sigma_space)
        result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    if preserve_luminance:
        result = _restore_luminance(image_bgr, result)

    return np.clip(result, 0, 255).astype(np.uint8)
