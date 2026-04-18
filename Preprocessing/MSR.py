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


def _extract_working_channels(image_bgr: np.ndarray, color_space: str):
    color_space = color_space.lower()
    if color_space == 'bgr':
        converted = image_bgr.copy()
        return converted, converted, None
    if color_space == 'ycrcb':
        converted = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2YCrCb)
        y, cr, cb = cv2.split(converted)
        return y, converted, ('ycrcb', cr, cb)
    if color_space == 'lab':
        converted = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(converted)
        return l, converted, ('lab', a, b)
    raise ValueError("color_space debe ser uno de: 'bgr', 'ycrcb', 'lab'")


def _merge_back(processed, metadata):
    if metadata is None:
        return processed
    kind = metadata[0]
    if kind == 'ycrcb':
        _, cr, cb = metadata
        merged = cv2.merge([processed, cr, cb])
        return cv2.cvtColor(merged, cv2.COLOR_YCrCb2BGR)
    if kind == 'lab':
        _, a, b = metadata
        merged = cv2.merge([processed, a, b])
        return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
    raise ValueError(f'Metadato de color no soportado: {kind}')


def _single_scale_retinex(channel: np.ndarray, sigma: float) -> np.ndarray:
    channel_f = channel.astype(np.float32) + 1.0
    blurred = cv2.GaussianBlur(channel_f, (0, 0), sigmaX=sigma, sigmaY=sigma)
    retinex = np.log(channel_f) - np.log(blurred + 1e-6)
    return retinex


def _normalize_to_uint8(image_f: np.ndarray) -> np.ndarray:
    normalized = cv2.normalize(image_f, None, 0, 255, cv2.NORM_MINMAX)
    return np.clip(normalized, 0, 255).astype(np.uint8)


def apply_msr(
    image_bgr,
    sigmas=(15, 80, 250),
    weights=None,
    color_space='bgr',
    preserve_luminance=True,
):
    """
    Aplica Multi-Scale Retinex (MSR).

    Args:
        image_bgr: imagen BGR uint8.
        sigmas: tupla o lista de escalas gaussianas.
        weights: pesos por escala. Si es None, usa pesos uniformes.
        color_space: 'bgr', 'ycrcb' o 'lab'.
        preserve_luminance: si True, reescala para aproximar el promedio original.

    Returns:
        Imagen procesada en BGR uint8.
    """
    image_bgr = _validate_bgr_uint8(image_bgr)

    if sigmas is None or len(sigmas) == 0:
        raise ValueError('sigmas no puede estar vacío')

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

    working, converted, metadata = _extract_working_channels(image_bgr, color_space)

    if color_space == 'bgr':
        channels = cv2.split(working)
        processed_channels = []
        for ch in channels:
            acc = np.zeros_like(ch, dtype=np.float32)
            for sigma, weight in zip(sigmas, weights):
                acc += weight * _single_scale_retinex(ch, sigma)
            processed_channels.append(_normalize_to_uint8(acc))
        processed_bgr = cv2.merge(processed_channels)
        if preserve_luminance:
            original_mean = float(np.mean(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)))
            processed_gray = cv2.cvtColor(processed_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
            processed_mean = float(np.mean(processed_gray)) + 1e-6
            gain = original_mean / processed_mean
            processed_bgr = np.clip(processed_bgr.astype(np.float32) * gain, 0, 255).astype(np.uint8)
        return processed_bgr

    original_mean = float(np.mean(working))
    acc = np.zeros_like(working, dtype=np.float32)
    for sigma, weight in zip(sigmas, weights):
        acc += weight * _single_scale_retinex(working, sigma)
    processed = _normalize_to_uint8(acc)

    if preserve_luminance:
        processed_mean = float(np.mean(processed)) + 1e-6
        gain = original_mean / processed_mean
        processed = np.clip(processed.astype(np.float32) * gain, 0, 255).astype(np.uint8)

    return _merge_back(processed, metadata)
