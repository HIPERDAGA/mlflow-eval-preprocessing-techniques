from __future__ import annotations

import cv2
import numpy as np


def _validate_bgr_uint8(image_bgr):
    if image_bgr is None:
        raise ValueError('image_bgr es None')
    if not isinstance(image_bgr, np.ndarray):
        raise TypeError(f'Se esperaba np.ndarray, recibido: {type(image_bgr)}')
    if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise ValueError(f'Se esperaba imagen BGR HxWx3, recibido shape={image_bgr.shape}')
    if image_bgr.dtype != np.uint8:
        image_bgr = np.clip(image_bgr, 0, 255).astype(np.uint8)
    return image_bgr


def _single_scale_retinex(channel: np.ndarray, sigma: float) -> np.ndarray:
    channel_f = channel.astype(np.float32) + 1.0
    blurred = cv2.GaussianBlur(channel_f, (0, 0), sigmaX=sigma, sigmaY=sigma)
    return np.log(channel_f) - np.log(blurred + 1e-6)


def _msr(image_rgb: np.ndarray, sigmas, weights) -> np.ndarray:
    image_rgb = image_rgb.astype(np.float32) + 1.0
    retinex = np.zeros_like(image_rgb, dtype=np.float32)
    for sigma, weight in zip(sigmas, weights):
        blurred = cv2.GaussianBlur(image_rgb, (0, 0), sigmaX=sigma, sigmaY=sigma)
        retinex += weight * (np.log(image_rgb) - np.log(blurred + 1e-6))
    return retinex


def apply_msrcr(
    image_bgr,
    sigmas=(15, 80, 250),
    weights=None,
    alpha=125.0,
    beta=46.0,
    gain=1.0,
    offset=0.0,
    preserve_luminance=True,
):
    """Aplica Multi-Scale Retinex with Color Restoration (MSRCR)."""
    image_bgr = _validate_bgr_uint8(image_bgr)
    sigmas = [float(s) for s in sigmas]
    if any(s <= 0 for s in sigmas):
        raise ValueError('Todos los sigmas deben ser > 0')

    if weights is None:
        weights = [1.0 / len(sigmas)] * len(sigmas)
    else:
        weights = [float(w) for w in weights]
        if len(weights) != len(sigmas):
            raise ValueError('weights debe tener la misma longitud que sigmas')
        total = sum(weights)
        if total <= 0:
            raise ValueError('La suma de weights debe ser > 0')
        weights = [w / total for w in weights]

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) + 1.0
    msr = _msr(image_rgb, sigmas, weights)

    intensity_sum = np.sum(image_rgb, axis=2, keepdims=True)
    color_restoration = beta * (np.log(alpha * image_rgb) - np.log(intensity_sum + 1e-6))
    msrcr = gain * (msr * color_restoration) + offset

    msrcr_norm = np.zeros_like(msrcr, dtype=np.uint8)
    for c in range(3):
        msrcr_norm[:, :, c] = cv2.normalize(msrcr[:, :, c], None, 0, 255, cv2.NORM_MINMAX)
    processed_bgr = cv2.cvtColor(msrcr_norm, cv2.COLOR_RGB2BGR)

    if preserve_luminance:
        original_mean = float(np.mean(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)))
        processed_gray = cv2.cvtColor(processed_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
        processed_mean = float(np.mean(processed_gray)) + 1e-6
        gain_factor = original_mean / processed_mean
        processed_bgr = np.clip(processed_bgr.astype(np.float32) * gain_factor, 0, 255).astype(np.uint8)

    return processed_bgr
