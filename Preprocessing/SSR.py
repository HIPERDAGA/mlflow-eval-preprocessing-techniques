from __future__ import annotations

from typing import Literal

import cv2
import numpy as np

ColorSpace = Literal["bgr", "ycrcb", "lab"]

_VALID_COLOR_SPACES = {"bgr", "ycrcb", "lab"}
_EPS = 1e-6


def _validate_image(image_bgr: np.ndarray) -> np.ndarray:
    if image_bgr is None:
        raise ValueError("image_bgr no puede ser None")
    if not isinstance(image_bgr, np.ndarray):
        raise TypeError(f"Se esperaba np.ndarray, recibido: {type(image_bgr)!r}")
    if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise ValueError(f"Se esperaba imagen BGR HxWx3, recibido shape={image_bgr.shape}")
    if image_bgr.dtype != np.uint8:
        image_bgr = np.clip(image_bgr, 0, 255).astype(np.uint8)
    return image_bgr


def _retinex_single_scale(channel: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        raise ValueError(f"sigma debe ser > 0, recibido {sigma}")
    channel_f = channel.astype(np.float32) + 1.0
    blurred = cv2.GaussianBlur(channel_f, ksize=(0, 0), sigmaX=sigma, sigmaY=sigma)
    return np.log(channel_f + _EPS) - np.log(blurred + _EPS)


def _normalize_to_uint8(channel: np.ndarray, gain: float = 1.0, offset: float = 0.0) -> np.ndarray:
    channel = channel * gain + offset
    ch_min = float(channel.min())
    ch_max = float(channel.max())
    if np.isclose(ch_max, ch_min):
        return np.zeros_like(channel, dtype=np.uint8)
    channel = (channel - ch_min) / (ch_max - ch_min)
    return np.clip(channel * 255.0, 0, 255).astype(np.uint8)


def _preserve_mean_luminance(original: np.ndarray, processed: np.ndarray) -> np.ndarray:
    original_mean = float(original.mean())
    processed_mean = float(processed.mean())
    if processed_mean <= 0:
        return processed
    scale = original_mean / processed_mean
    return np.clip(processed.astype(np.float32) * scale, 0, 255).astype(np.uint8)


def apply_ssr(
    image_bgr: np.ndarray,
    sigma: float = 80.0,
    color_space: ColorSpace = "lab",
    preserve_luminance: bool = True,
    gain: float = 1.0,
    offset: float = 0.0,
) -> np.ndarray:
    """Aplica Single Scale Retinex (SSR) a una imagen BGR y devuelve BGR uint8."""
    image_bgr = _validate_image(image_bgr)

    if color_space not in _VALID_COLOR_SPACES:
        raise ValueError(f"color_space no válido: {color_space}. Use uno de {_VALID_COLOR_SPACES}")

    if color_space == "bgr":
        channels = cv2.split(image_bgr)
        processed_channels = []
        for channel in channels:
            retinex = _retinex_single_scale(channel, sigma=sigma)
            processed_channels.append(_normalize_to_uint8(retinex, gain=gain, offset=offset))
        return cv2.merge(processed_channels)

    if color_space == "ycrcb":
        ycrcb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2YCrCb)
        y, cr, cb = cv2.split(ycrcb)
        y_processed = _normalize_to_uint8(_retinex_single_scale(y, sigma=sigma), gain=gain, offset=offset)
        if preserve_luminance:
            y_processed = _preserve_mean_luminance(y, y_processed)
        return cv2.cvtColor(cv2.merge([y_processed, cr, cb]), cv2.COLOR_YCrCb2BGR)

    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l_processed = _normalize_to_uint8(_retinex_single_scale(l, sigma=sigma), gain=gain, offset=offset)
    if preserve_luminance:
        l_processed = _preserve_mean_luminance(l, l_processed)
    return cv2.cvtColor(cv2.merge([l_processed, a, b]), cv2.COLOR_LAB2BGR)
