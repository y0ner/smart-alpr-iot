"""Pipeline principal de detección de placas vehiculares e inferencia OCR.

Conecta YOLOv8/v11, el preprocesamiento morfológico y EasyOCR.
"""

from typing import Any

import numpy as np
from numpy.typing import NDArray

from src.config import YOLO_CONF_THRESHOLD


class ALPRDetector:
    """Motor de detección e inferencia ALPR."""

    def __init__(self, conf_threshold: float = YOLO_CONF_THRESHOLD) -> None:
        self.conf_threshold = conf_threshold

    def is_valid_frame(self, frame: NDArray[np.uint8]) -> bool:
        """Comprueba que el frame sea un arreglo válido de NumPy no vacío."""
        return isinstance(frame, np.ndarray) and frame.size > 0

    def get_status(self) -> dict[str, Any]:
        """Devuelve el estado de configuración del detector."""
        return {
            "conf_threshold": self.conf_threshold,
            "status": "ready",
        }
