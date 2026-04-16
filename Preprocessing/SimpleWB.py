from __future__ import annotations

from pathlib import Path
from typing import Union

import cv2
import numpy as np


def _compute_percentile_bounds(channel: np.ndarray, clip_percent: float) -> tuple[float, float]:
    if clip_percent <= 0:
        return float(channel.min()), float(channel.max())
    low = np.percentile(channel, clip_percent)
    high = np.percentile(channel, 100 - clip_percent)
    if high <= low:
        return float(channel.min()), float(channel.max())
    return float(low), float(high)


def _rescale_channel(channel: np.ndarray, clip_percent: float) -> np.ndarray:
    low, high = _compute_percentile_bounds(channel, clip_percent)
    clipped = np.clip(channel, low, high)
    scaled = (clipped - low) * (255.0 / max(high - low, 1e-8))
    return np.clip(scaled, 0, 255)


def simple_wb(
    image_bgr: np.ndarray,
    clip_percent: float = 1.0,
    preserve_luminance: bool = True,
    channel_gain_limit: float = 2.0,
) -> np.ndarray:
    if image_bgr is None or image_bgr.size == 0:
        raise ValueError("Imagen vacia o invalida.")

    image = image_bgr.astype(np.float32)
    original_mean = float(np.mean(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)))

    channels = cv2.split(image)
    balanced_channels = []
    channel_means = []

    for channel in channels:
        balanced = _rescale_channel(channel, clip_percent)
        balanced_channels.append(balanced)
        channel_means.append(float(np.mean(balanced)))

    target_mean = float(np.mean(channel_means))
    adjusted_channels = []
    for balanced, mean_val in zip(balanced_channels, channel_means):
        gain = target_mean / max(mean_val, 1e-8)
        gain = min(gain, channel_gain_limit)
        adjusted_channels.append(np.clip(balanced * gain, 0, 255))

    out = cv2.merge(adjusted_channels)

    if preserve_luminance:
        out_gray_mean = float(np.mean(cv2.cvtColor(out.astype(np.uint8), cv2.COLOR_BGR2GRAY)))
        lum_gain = original_mean / max(out_gray_mean, 1e-8)
        out = np.clip(out * lum_gain, 0, 255)

    return out.astype(np.uint8)


def apply_simple_wb(
    image_bgr: np.ndarray,
    clip_percent: float,
    preserve_luminance: bool,
    channel_gain_limit: float,
) -> np.ndarray:
    return simple_wb(
        image_bgr=image_bgr,
        clip_percent=clip_percent,
        preserve_luminance=preserve_luminance,
        channel_gain_limit=channel_gain_limit,
    )


def process_image(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    clip_percent: float = 1.0,
    preserve_luminance: bool = True,
    channel_gain_limit: float = 2.0,
) -> Path:
    input_path = Path(input_path)
    output_path = Path(output_path)
    image = cv2.imread(str(input_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"No se pudo leer la imagen: {input_path}")

    processed = simple_wb(
        image_bgr=image,
        clip_percent=clip_percent,
        preserve_luminance=preserve_luminance,
        channel_gain_limit=channel_gain_limit,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(output_path), processed)
    if not ok:
        raise IOError(f"No se pudo guardar la imagen procesada: {output_path}")
    return output_path
