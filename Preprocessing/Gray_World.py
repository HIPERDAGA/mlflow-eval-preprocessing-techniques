"""Gray World white balance preprocessing.

This implementation exposes exactly the parameters requested for the experiment:
- preserve_luminance
- channel_gain_limit
"""
from __future__ import annotations

import argparse
import os
from typing import Tuple

import cv2
import numpy as np


class GrayWorld:
    def __init__(self, preserve_luminance: bool = True, channel_gain_limit: float = 1.5) -> None:
        if channel_gain_limit <= 0:
            raise ValueError("channel_gain_limit must be > 0.")
        self.preserve_luminance = preserve_luminance
        self.channel_gain_limit = float(channel_gain_limit)

    def apply(self, image_rgb: np.ndarray) -> np.ndarray:
        if image_rgb is None or image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
            raise ValueError("Expected an RGB image with shape HxWx3.")

        img = image_rgb.astype(np.float32)
        channel_means = img.reshape(-1, 3).mean(axis=0)
        gray_mean = float(channel_means.mean())

        gains = gray_mean / (channel_means + 1e-8)
        gains = np.clip(gains, 1.0 / self.channel_gain_limit, self.channel_gain_limit)

        balanced = img * gains.reshape(1, 1, 3)

        if self.preserve_luminance:
            original_lum = self._luminance(img)
            balanced_lum = self._luminance(balanced)
            luminance_ratio = original_lum / (balanced_lum + 1e-8)
            balanced *= luminance_ratio

        return np.clip(balanced, 0, 255).astype(np.uint8)

    @staticmethod
    def _luminance(image_rgb: np.ndarray) -> float:
        r, g, b = image_rgb[..., 0], image_rgb[..., 1], image_rgb[..., 2]
        return float((0.2126 * r + 0.7152 * g + 0.0722 * b).mean())


def apply_gray_world(
    image_rgb: np.ndarray,
    preserve_luminance: bool = True,
    channel_gain_limit: float = 1.5,
) -> np.ndarray:
    processor = GrayWorld(
        preserve_luminance=preserve_luminance,
        channel_gain_limit=channel_gain_limit,
    )
    return processor.apply(image_rgb)


def read_image_rgb(path: str) -> np.ndarray:
    image_bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError(f"Could not read image from: {path}")
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def save_image_rgb(path: str, image_rgb: np.ndarray) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    if not cv2.imwrite(path, image_bgr):
        raise IOError(f"Could not save image to: {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply Gray World white balance to a single image.")
    parser.add_argument("--input", required=True, help="Input image path")
    parser.add_argument("--output", required=True, help="Output image path")
    parser.add_argument("--preserve_luminance", action="store_true", help="Preserve global luminance")
    parser.add_argument("--channel_gain_limit", type=float, default=1.5, help="Maximum absolute per-channel gain")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image_rgb = read_image_rgb(args.input)
    output_rgb = apply_gray_world(
        image_rgb=image_rgb,
        preserve_luminance=args.preserve_luminance,
        channel_gain_limit=args.channel_gain_limit,
    )
    save_image_rgb(args.output, output_rgb)
    print(f"Saved processed image to: {args.output}")


if __name__ == "__main__":
    main()
