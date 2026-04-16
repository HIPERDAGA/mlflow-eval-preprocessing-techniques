from __future__ import annotations

from pathlib import Path
from typing import Union

import cv2
import numpy as np


VALID_COLOR_SPACES = {"bgr", "ycrcb", "lab"}


def _validate_inputs(
    image_bgr: np.ndarray,
    gamma: float,
    color_space: str,
) -> str:
    if image_bgr is None or image_bgr.size == 0:
        raise ValueError("Imagen vacía o inválida.")

    if image_bgr.dtype != np.uint8:
        raise ValueError("La imagen debe ser uint8.")

    if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise ValueError("Se espera una imagen BGR de 3 canales.")

    if gamma <= 0:
        raise ValueError("gamma debe ser > 0.")

    color_space = color_space.lower().strip()
    if color_space not in VALID_COLOR_SPACES:
        raise ValueError(f"color_space debe ser uno de {VALID_COLOR_SPACES}, recibido: {color_space}")

    return color_space


def _build_gamma_lut(gamma: float) -> np.ndarray:
    values = np.arange(256, dtype=np.float32) / 255.0
    corrected = np.power(values, gamma) * 255.0
    return np.clip(corrected, 0, 255).astype(np.uint8)


def _preserve_global_luminance(original_bgr: np.ndarray, processed_bgr: np.ndarray) -> np.ndarray:
    original_gray_mean = float(np.mean(cv2.cvtColor(original_bgr, cv2.COLOR_BGR2GRAY)))
    processed_gray_mean = float(np.mean(cv2.cvtColor(processed_bgr, cv2.COLOR_BGR2GRAY)))

    if processed_gray_mean <= 1e-8:
        return processed_bgr

    gain = original_gray_mean / processed_gray_mean
    adjusted = np.clip(processed_bgr.astype(np.float32) * gain, 0, 255)
    return adjusted.astype(np.uint8)


def _apply_gamma_to_bgr(image_bgr: np.ndarray, gamma: float) -> np.ndarray:
    lut = _build_gamma_lut(gamma)
    return cv2.LUT(image_bgr, lut)


def _apply_gamma_to_luma_ycrcb(image_bgr: np.ndarray, gamma: float) -> np.ndarray:
    lut = _build_gamma_lut(gamma)
    image_ycrcb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2YCrCb)
    channels = list(cv2.split(image_ycrcb))
    channels[0] = cv2.LUT(channels[0], lut)
    merged = cv2.merge(channels)
    return cv2.cvtColor(merged, cv2.COLOR_YCrCb2BGR)


def _apply_gamma_to_luma_lab(image_bgr: np.ndarray, gamma: float) -> np.ndarray:
    lut = _build_gamma_lut(gamma)
    image_lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    channels = list(cv2.split(image_lab))
    channels[0] = cv2.LUT(channels[0], lut)
    merged = cv2.merge(channels)
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


def gamma_correction(
    image_bgr: np.ndarray,
    gamma: float = 1.0,
    color_space: str = "ycrcb",
    preserve_luminance: bool = False,
) -> np.ndarray:
    color_space = _validate_inputs(image_bgr, gamma, color_space)

    if color_space == "bgr":
        out_bgr = _apply_gamma_to_bgr(image_bgr, gamma)
    elif color_space == "ycrcb":
        out_bgr = _apply_gamma_to_luma_ycrcb(image_bgr, gamma)
    elif color_space == "lab":
        out_bgr = _apply_gamma_to_luma_lab(image_bgr, gamma)
    else:
        raise ValueError(f"Espacio de color no soportado: {color_space}")

    if preserve_luminance:
        out_bgr = _preserve_global_luminance(image_bgr, out_bgr)

    return out_bgr


def apply_gamma_correction(
    image_bgr: np.ndarray,
    gamma: float,
    color_space: str,
    preserve_luminance: bool,
) -> np.ndarray:
    return gamma_correction(
        image_bgr=image_bgr,
        gamma=gamma,
        color_space=color_space,
        preserve_luminance=preserve_luminance,
    )


def process_image(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    gamma: float = 1.0,
    color_space: str = "ycrcb",
    preserve_luminance: bool = False,
) -> Path:
    input_path = Path(input_path)
    output_path = Path(output_path)

    image = cv2.imread(str(input_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"No se pudo leer la imagen: {input_path}")

    processed = gamma_correction(
        image_bgr=image,
        gamma=gamma,
        color_space=color_space,
        preserve_luminance=preserve_luminance,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(output_path), processed)
    if not ok:
        raise IOError(f"No se pudo guardar la imagen procesada: {output_path}")

    return output_path
