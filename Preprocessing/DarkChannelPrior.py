from __future__ import annotations

import cv2
import numpy as np


def _ensure_uint8_bgr(image_bgr: np.ndarray) -> np.ndarray:
    if image_bgr is None:
        raise ValueError("image_bgr is None")
    if not isinstance(image_bgr, np.ndarray):
        raise TypeError(f"Expected np.ndarray, got {type(image_bgr)}")
    if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise ValueError(f"Expected BGR image HxWx3, got shape={image_bgr.shape}")
    if image_bgr.dtype != np.uint8:
        image_bgr = np.clip(image_bgr, 0, 255).astype(np.uint8)
    return image_bgr


def _dark_channel(image_rgb: np.ndarray, patch_size: int) -> np.ndarray:
    min_per_pixel = np.min(image_rgb, axis=2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (patch_size, patch_size))
    return cv2.erode(min_per_pixel, kernel)


def _estimate_atmospheric_light(image_rgb: np.ndarray, dark: np.ndarray, top_percent: float) -> np.ndarray:
    h, w = dark.shape
    num_pixels = h * w
    k = max(1, int(num_pixels * top_percent))

    flat_dark = dark.reshape(-1)
    flat_img = image_rgb.reshape(-1, 3)
    indices = np.argpartition(flat_dark, -k)[-k:]

    brightest = flat_img[indices]
    atmospheric_light = brightest[np.argmax(np.sum(brightest, axis=1))]
    return np.maximum(atmospheric_light, 1e-6)


def _estimate_transmission(image_rgb: np.ndarray, atmospheric_light: np.ndarray, patch_size: int, omega: float) -> np.ndarray:
    normed = image_rgb / atmospheric_light.reshape(1, 1, 3)
    dark_normed = _dark_channel(normed, patch_size)
    transmission = 1.0 - omega * dark_normed
    return np.clip(transmission, 0.0, 1.0)


def _recover_radiance(image_rgb: np.ndarray, transmission: np.ndarray, atmospheric_light: np.ndarray, t0: float) -> np.ndarray:
    transmission = np.maximum(transmission, t0)
    j = (image_rgb - atmospheric_light.reshape(1, 1, 3)) / transmission[:, :, None] + atmospheric_light.reshape(1, 1, 3)
    return np.clip(j, 0.0, 255.0).astype(np.uint8)


def _preserve_mean_luminance(original_bgr: np.ndarray, processed_bgr: np.ndarray) -> np.ndarray:
    orig_ycrcb = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2YCrCb)
    proc_ycrcb = cv2.cvtColor(processed_bgr, cv2.COLOR_BGR2YCrCb)

    orig_y = orig_ycrcb[:, :, 0].astype(np.float32)
    proc_y = proc_ycrcb[:, :, 0].astype(np.float32)

    orig_mean = float(orig_y.mean())
    proc_mean = float(proc_y.mean())
    if proc_mean <= 1e-6:
        return processed_bgr

    scale = orig_mean / proc_mean
    proc_ycrcb[:, :, 0] = np.clip(proc_y * scale, 0, 255).astype(np.uint8)
    return cv2.cvtColor(proc_ycrcb, cv2.COLOR_YCrCb2BGR)


def apply_dark_channel_prior(
    image_bgr: np.ndarray,
    patch_size: int = 15,
    omega: float = 0.95,
    t0: float = 0.1,
    atmospheric_top_percent: float = 0.001,
    preserve_luminance: bool = False,
) -> np.ndarray:
    """
    Apply Dark Channel Prior dehazing on a BGR image.

    Parameters
    ----------
    image_bgr : np.ndarray
        Input image in BGR uint8 format.
    patch_size : int
        Window size used to compute the dark channel. Typical values: 7, 15, 31.
    omega : float
        Haze removal strength. Typical values: 0.85, 0.90, 0.95.
    t0 : float
        Minimum transmission bound to avoid over-amplification. Typical values: 0.1, 0.15, 0.2.
    atmospheric_top_percent : float
        Fraction of brightest dark-channel pixels used for atmospheric light estimation.
        Example values: 0.001, 0.005, 0.01.
    preserve_luminance : bool
        Whether to rescale luminance to approximate the original image mean luminance.
    """
    image_bgr = _ensure_uint8_bgr(image_bgr)

    if patch_size % 2 == 0 or patch_size < 3:
        raise ValueError("patch_size must be odd and >= 3")
    if not (0.0 < omega <= 1.0):
        raise ValueError("omega must be in (0, 1]")
    if not (0.0 < t0 <= 1.0):
        raise ValueError("t0 must be in (0, 1]")
    if not (0.0 < atmospheric_top_percent <= 0.05):
        raise ValueError("atmospheric_top_percent must be in (0, 0.05]")

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)

    dark = _dark_channel(image_rgb, patch_size)
    atmospheric_light = _estimate_atmospheric_light(image_rgb, dark, atmospheric_top_percent)
    transmission = _estimate_transmission(image_rgb, atmospheric_light, patch_size, omega)
    recovered_rgb = _recover_radiance(image_rgb, transmission, atmospheric_light, t0)
    recovered_bgr = cv2.cvtColor(recovered_rgb, cv2.COLOR_RGB2BGR)

    if preserve_luminance:
        recovered_bgr = _preserve_mean_luminance(image_bgr, recovered_bgr)

    return recovered_bgr
