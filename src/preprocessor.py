"""Pipeline de procesamiento digital de imágenes (OpenCV) para Smart ALPR.

Provee recorte con margen, escala, filtros bilaterales, umbralización
adaptativa/Otsu y operaciones morfológicas.
"""

from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray


class ImagePreprocessor:
    """Procesador morfológico y de realce para matrículas vehiculares."""

    def __init__(self, scale_factor: float = 3.0, margin_ratio: float = 0.12) -> None:
        self.scale_factor = scale_factor
        self.margin_ratio = margin_ratio

    def to_grayscale(self, image: NDArray[np.uint8]) -> NDArray[np.uint8]:
        """Convierte imagen BGR a escala de grises."""
        if len(image.shape) == 2:
            return image
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    def process_pipeline(self, patch: NDArray[np.uint8]) -> dict[str, Any]:
        """Ejecuta el pipeline morfológico devolviendo fases intermedias."""
        gray = self.to_grayscale(patch)
        bilateral = cv2.bilateralFilter(gray, d=11, sigmaColor=17, sigmaSpace=17)
        _, otsu = cv2.threshold(bilateral, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        return {
            "gray": gray,
            "bilateral": bilateral,
            "otsu": otsu,
        }
