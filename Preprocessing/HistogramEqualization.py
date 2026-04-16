from __future__ import annotations

from pathlib import Path
from typing import Tuple, Union

import cv2
import numpy as np


VALID_COLOR_SPACES = {"ycrcb", "lab", "hsv"}
VALID_METHODS = {"global_he", "clahe"}


def _validate_inputs(
    image_bgr: np.ndarray,
    method: str,
    color_space: str,
) -> tuple[str, str]:
    if image_bgr is None or image_bgr.size == 0:
        raise ValueError("Imagen vacía o inválida.")

    if image_bgr.dtype != np.uint8:
        raise ValueError("La imagen debe ser uint8.")

    if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise ValueError("Se espera una imagen BGR de 3 canales.")

    method = method.lower().strip()
    color_space = color_space.lower().strip()

    if method not in VALID_METHODS:
        raise ValueError(f"method debe ser uno de {VALID_METHODS}, recibido: {method}")

    if color_space not in VALID_COLOR_SPACES:
        raise ValueError(f"color_space debe ser uno de {VALID_COLOR_SPACES}, recibido: {color_space}")

    return method, color_space


def _convert_from_bgr(image_bgr: np.ndarray, color_space: str) -> np.ndarray:
    if color_space == "ycrcb":
        return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2YCrCb)
    if color_space == "lab":
        return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    if color_space == "hsv":
        return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    raise ValueError(f"Espacio de color no soportado: {color_space}")


def _convert_to_bgr(image_cs: np.ndarray, color_space: str) -> np.ndarray:
    if color_space == "ycrcb":
        return cv2.cvtColor(image_cs, cv2.COLOR_YCrCb2BGR)
    if color_space == "lab":
        return cv2.cvtColor(image_cs, cv2.COLOR_LAB2BGR)
    if color_space == "hsv":
        return cv2.cvtColor(image_cs, cv2.COLOR_HSV2BGR)
    raise ValueError(f"Espacio de color no soportado: {color_space}")


def _get_luma_channel_index(color_space: str) -> int:
    if color_space in {"ycrcb", "lab", "hsv"}:
        return 0
    raise ValueError(f"Espacio de color no soportado: {color_space}")


def _preserve_global_luminance(original_bgr: np.ndarray, processed_bgr: np.ndarray) -> np.ndarray:
    original_gray_mean = float(np.mean(cv2.cvtColor(original_bgr, cv2.COLOR_BGR2GRAY)))
    processed_gray_mean = float(np.mean(cv2.cvtColor(processed_bgr, cv2.COLOR_BGR2GRAY)))

    if processed_gray_mean <= 1e-8:
        return processed_bgr

    gain = original_gray_mean / processed_gray_mean
    adjusted = np.clip(processed_bgr.astype(np.float32) * gain, 0, 255)
    return adjusted.astype(np.uint8)


def global_histogram_equalization(
    image_bgr: np.ndarray,
    color_space: str = "ycrcb",
    preserve_luminance: bool = True,
) -> np.ndarray:
    color_space = color_space.lower().strip()
    image_cs = _convert_from_bgr(image_bgr, color_space)
    channel_idx = _get_luma_channel_index(color_space)

    channels = list(cv2.split(image_cs))
    channels[channel_idx] = cv2.equalizeHist(channels[channel_idx])
    equalized_cs = cv2.merge(channels)

    out_bgr = _convert_to_bgr(equalized_cs, color_space)

    if preserve_luminance:
        out_bgr = _preserve_global_luminance(image_bgr, out_bgr)

    return out_bgr


def clahe_histogram_equalization(
    image_bgr: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: Tuple[int, int] = (8, 8),
    color_space: str = "ycrcb",
    preserve_luminance: bool = True,
) -> np.ndarray:
    color_space = color_space.lower().strip()
    image_cs = _convert_from_bgr(image_bgr, color_space)
    channel_idx = _get_luma_channel_index(color_space)

    channels = list(cv2.split(image_cs))
    clahe = cv2.createCLAHE(clipLimit=float(clip_limit), tileGridSize=tile_grid_size)
    channels[channel_idx] = clahe.apply(channels[channel_idx])
    equalized_cs = cv2.merge(channels)

    out_bgr = _convert_to_bgr(equalized_cs, color_space)

    if preserve_luminance:
        out_bgr = _preserve_global_luminance(image_bgr, out_bgr)

    return out_bgr


def histogram_equalization(
    image_bgr: np.ndarray,
    method: str = "global_he",
    color_space: str = "ycrcb",
    preserve_luminance: bool = True,
    clip_limit: float = 2.0,
    tile_grid_size: Tuple[int, int] = (8, 8),
) -> np.ndarray:
    method, color_space = _validate_inputs(image_bgr, method, color_space)

    if method == "global_he":
        return global_histogram_equalization(
            image_bgr=image_bgr,
            color_space=color_space,
            preserve_luminance=preserve_luminance,
        )

    if method == "clahe":
        return clahe_histogram_equalization(
            image_bgr=image_bgr,
            clip_limit=clip_limit,
            tile_grid_size=tile_grid_size,
            color_space=color_space,
            preserve_luminance=preserve_luminance,
        )

    raise ValueError(f"Método no soportado: {method}")


def apply_histogram_equalization(
    image_bgr: np.ndarray,
    method: str,
    color_space: str,
    preserve_luminance: bool,
    clip_limit: float = 2.0,
    tile_grid_size: Tuple[int, int] = (8, 8),
) -> np.ndarray:
    return histogram_equalization(
        image_bgr=image_bgr,
        method=method,
        color_space=color_space,
        preserve_luminance=preserve_luminance,
        clip_limit=clip_limit,
        tile_grid_size=tile_grid_size,
    )


def process_image(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    method: str = "global_he",
    color_space: str = "ycrcb",
    preserve_luminance: bool = True,
    clip_limit: float = 2.0,
    tile_grid_size: Tuple[int, int] = (8, 8),
) -> Path:
    input_path = Path(input_path)
    output_path = Path(output_path)

    image = cv2.imread(str(input_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"No se pudo leer la imagen: {input_path}")

    processed = histogram_equalization(
        image_bgr=image,
        method=method,
        color_space=color_space,
        preserve_luminance=preserve_luminance,
        clip_limit=clip_limit,
        tile_grid_size=tile_grid_size,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(output_path), processed)
    if not ok:
        raise IOError(f"No se pudo guardar la imagen procesada: {output_path}")

    return output_path
