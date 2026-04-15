"""BRISQUE metric helper.

Uses pyiqa for a robust no-reference BRISQUE implementation.
"""
from __future__ import annotations

from typing import Union

import numpy as np
from PIL import Image
import torch
import pyiqa

ImageInput = Union[str, np.ndarray, Image.Image]


class BRISQUEMetric:
    def __init__(self, device: str | None = None) -> None:
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.metric = pyiqa.create_metric("brisque", device=self.device)

    def _to_tensor(self, image: ImageInput) -> torch.Tensor:
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):
            if image.ndim != 3:
                raise ValueError("Expected HxWxC array for image input.")
            image = Image.fromarray(image.astype(np.uint8)).convert("RGB")
        elif isinstance(image, Image.Image):
            image = image.convert("RGB")
        else:
            raise TypeError("Unsupported image input type.")

        arr = np.asarray(image).astype(np.float32) / 255.0
        tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
        return tensor.to(self.device)

    def score(self, image: ImageInput) -> float:
        with torch.no_grad():
            value = self.metric(self._to_tensor(image))
        return float(value.detach().cpu().item())


def compute_brisque(image: ImageInput, device: str | None = None) -> float:
    return BRISQUEMetric(device=device).score(image)
