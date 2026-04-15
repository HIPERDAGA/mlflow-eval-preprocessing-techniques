"""PIQE metric helper.

Uses pypiqe, a Python implementation of MATLAB's PIQE.
"""
from __future__ import annotations

from typing import Union

import cv2
import numpy as np
from PIL import Image
from pypiqe import piqe

ImageInput = Union[str, np.ndarray, Image.Image]


def _to_gray_uint8(image: ImageInput) -> np.ndarray:
    if isinstance(image, str):
        arr = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
        if arr is None:
            raise ValueError(f"Could not read image from path: {image}")
        return arr
    if isinstance(image, Image.Image):
        return np.asarray(image.convert("L"), dtype=np.uint8)
    if isinstance(image, np.ndarray):
        if image.ndim == 2:
            return image.astype(np.uint8)
        if image.ndim == 3:
            return cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    raise TypeError("Unsupported image input type.")


def compute_piqe(image: ImageInput) -> float:
    gray = _to_gray_uint8(image)
    score, _, _, _ = piqe(gray)
    return float(score)
