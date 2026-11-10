"""Pipeline de procesamiento digital de imágenes (OpenCV) para Smart ALPR IoT.

Implementa recorte con margen porcentual de seguridad, escalado bicúbico,
filtrado bilateral para preservación de bordes, binarización dual (Otsu y adaptativa),
operaciones morfológicas y corrección de inclinación (deskewing).
"""

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray

from src.config import CROP_MARGIN_RATIO, RESIZE_SCALE_FACTOR


@dataclass(frozen=True)
class PreprocessedPlate:
    """Contenedor de todas las etapas intermedias del pipeline de visión artificial."""

    original_crop: NDArray[np.uint8]
    resized: NDArray[np.uint8]
    gray: NDArray[np.uint8]
    filtered: NDArray[np.uint8]
    binary_otsu: NDArray[np.uint8]
    binary_adaptive: NDArray[np.uint8]
    morph: NDArray[np.uint8]
    deskewed: NDArray[np.uint8]
    final_for_ocr: NDArray[np.uint8]
    rotation_angle: float

    def __getitem__(self, key: str) -> Any:
        """Permite acceso por subíndice como diccionario para compatibilidad."""
        if hasattr(self, key):
            return getattr(self, key)
        if key == "bilateral":
            return self.filtered
        if key == "otsu":
            return self.binary_otsu
        raise KeyError(key)

    def __contains__(self, key: str) -> bool:
        """Permite evaluar pertenencia de claves como diccionario."""
        valid_keys = {
            "original_crop",
            "resized",
            "gray",
            "filtered",
            "bilateral",
            "binary_otsu",
            "otsu",
            "binary_adaptive",
            "morph",
            "deskewed",
            "final_for_ocr",
            "rotation_angle",
        }
        return key in valid_keys

    def to_dict(self) -> dict[str, Any]:
        """Convierte los artefactos visuales a diccionario para visualizadores y telemetría."""
        return {
            "original_crop": self.original_crop,
            "resized": self.resized,
            "gray": self.gray,
            "filtered": self.filtered,
            "binary_otsu": self.binary_otsu,
            "binary_adaptive": self.binary_adaptive,
            "morph": self.morph,
            "deskewed": self.deskewed,
            "final_for_ocr": self.final_for_ocr,
            "rotation_angle": self.rotation_angle,
        }


class ImagePreprocessor:
    """Procesador digital de imágenes clásico para optimización de caracteres vehiculares."""

    def __init__(
        self,
        scale_factor: float = RESIZE_SCALE_FACTOR,
        margin_ratio: float = CROP_MARGIN_RATIO,
    ) -> None:
        self.scale_factor = scale_factor
        self.margin_ratio = margin_ratio

    def crop_with_margin(
        self,
        image: NDArray[np.uint8],
        bbox: tuple[int, int, int, int],
    ) -> NDArray[np.uint8]:
        """Recorta la región de interés (ROI) aplicando un margen porcentual de seguridad.

        Args:
            image: Frame completo en formato BGR o escala de grises.
            bbox: Coordenadas de la caja detectada (x1, y1, x2, y2).

        Returns:
            Parche recortado contenido dentro de los límites físicos de la imagen.
        """
        h_img, w_img = image.shape[:2]
        x1, y1, x2, y2 = bbox

        box_w = max(1, x2 - x1)
        box_h = max(1, y2 - y1)

        margin_x = int(box_w * self.margin_ratio)
        margin_y = int(box_h * self.margin_ratio)

        crop_x1 = max(0, x1 - margin_x)
        crop_y1 = max(0, y1 - margin_y)
        crop_x2 = min(w_img, x2 + margin_x)
        crop_y2 = min(h_img, y2 + margin_y)

        # Si las coordenadas son degeneradas, recortar al menos 1 píxel válido
        if crop_x2 <= crop_x1 or crop_y2 <= crop_y1:
            return image[max(0, y1) : min(h_img, y2), max(0, x1) : min(w_img, x2)]

        return image[crop_y1:crop_y2, crop_x1:crop_x2]

    def resize_patch(
        self,
        patch: NDArray[np.uint8],
        interpolation: int = cv2.INTER_CUBIC,
    ) -> NDArray[np.uint8]:
        """Escala el recorte usando interpolación bicúbica para densificar caracteres."""
        if self.scale_factor <= 1.0 or patch.size == 0:
            return patch

        new_w = max(1, int(patch.shape[1] * self.scale_factor))
        new_h = max(1, int(patch.shape[0] * self.scale_factor))
        return cv2.resize(patch, (new_w, new_h), interpolation=interpolation)

    @staticmethod
    def to_grayscale(image: NDArray[np.uint8]) -> NDArray[np.uint8]:
        """Convierte una matriz BGR a monocromática de un solo canal (uint8)."""
        if len(image.shape) == 2:
            return image
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    @staticmethod
    def apply_bilateral_filter(
        gray: NDArray[np.uint8],
        diameter: int = 11,
        sigma_color: float = 17.0,
        sigma_space: float = 17.0,
    ) -> NDArray[np.uint8]:
        """Aplica filtro bilateral para suavizar lámina reflectiva preservando bordes."""
        return cv2.bilateralFilter(
            gray,
            d=diameter,
            sigmaColor=sigma_color,
            sigmaSpace=sigma_space,
        )

    @staticmethod
    def binarize_otsu(filtered: NDArray[np.uint8]) -> NDArray[np.uint8]:
        """Aplica binarización automática mediante umbral óptimo de Otsu."""
        _, otsu = cv2.threshold(
            filtered, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        return otsu

    @staticmethod
    def binarize_adaptive(
        filtered: NDArray[np.uint8],
        block_size: int = 19,
        c_param: int = 9,
    ) -> NDArray[np.uint8]:
        """Aplica umbralización adaptativa gaussiana para compensar iluminación no uniforme."""
        # El tamaño del bloque debe ser un número entero impar mayor a 1
        bs = block_size if block_size % 2 == 1 else block_size + 1
        return cv2.adaptiveThreshold(
            filtered,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            blockSize=max(3, bs),
            C=c_param,
        )

    @staticmethod
    def apply_morphology(binary: NDArray[np.uint8]) -> NDArray[np.uint8]:
        """Aplica apertura para remover tornillos y cierre para consolidar trazos."""
        # Kernel rectangular pequeño (3x3) para apertura (eliminar artefactos menores a trazos)
        kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        # Kernel horizontal (3x2) para cierre (unir segmentos desconectados en caracteres)
        kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 2))

        # Invertir si las letras son oscuras sobre fondo claro para operar sobre los caracteres
        # En placas colombianas, fondo claro (255) y letras oscuras (0)
        es_fondo_blanco = np.mean(binary[0:3, :]) > 127
        work_img = cv2.bitwise_not(binary) if es_fondo_blanco else binary

        opened = cv2.morphologyEx(work_img, cv2.MORPH_OPEN, kernel_open)
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel_close)

        return cv2.bitwise_not(closed) if es_fondo_blanco else closed

    @staticmethod
    def deskew_image(gray_or_binary: NDArray[np.uint8]) -> tuple[NDArray[np.uint8], float]:
        """Detecta la inclinación de la matrícula y aplica corrección afín.

        Returns:
            Tupla (imagen_alineada, angulo_detectado_grados).
        """
        # Invertir para que los caracteres sean blancos sobre fondo negro
        thresh = cv2.bitwise_not(gray_or_binary) if np.mean(gray_or_binary) > 127 else gray_or_binary
        coords = np.column_stack(np.where(thresh > 0))

        if len(coords) < 50:
            return gray_or_binary, 0.0

        angle = cv2.minAreaRect(coords)[-1]
        # minAreaRect retorna ángulos entre [-90, 0)
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle

        # Si el ángulo es irrelevante (< 0.5 grados) o excesivo (> 35 grados por mala detección), no rotar
        if abs(angle) < 0.5 or abs(angle) > 35.0:
            return gray_or_binary, 0.0

        (h, w) = gray_or_binary.shape[:2]
        center = (w // 2, h // 2)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(
            gray_or_binary,
            matrix,
            (w, h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE,
        )
        return rotated, round(float(angle), 2)

    def process_pipeline(
        self,
        image: NDArray[np.uint8],
        bbox: tuple[int, int, int, int] | None = None,
    ) -> PreprocessedPlate:
        """Ejecuta el pipeline completo de visión clásica devolviendo todas las fases.

        Secuencia de procesamiento:
        1. Recorte con margen porcentual de seguridad (12%).
        2. Escalado bicúbico 3.0x para magnificar fuentes y remaches.
        3. Conversión a escala de grises.
        4. Filtro bilateral preservador de contornos.
        5. Umbralización dual (Otsu global y adaptativa gaussiana).
        6. Operaciones morfológicas rectangulares (apertura y cierre).
        7. Corrección de rotación / alineación horizontal (deskewing).
        """
        # Fase 1: Recorte con margen
        crop = self.crop_with_margin(image, bbox) if bbox is not None else image.copy()

        # Fase 2: Escalado bicúbico
        resized = self.resize_patch(crop)

        # Fase 3: Escala de grises
        gray = self.to_grayscale(resized)

        # Fase 4: Filtro Bilateral
        filtered = self.apply_bilateral_filter(gray)

        # Fase 5: Binarización dual
        binary_otsu = self.binarize_otsu(filtered)
        binary_adaptive = self.binarize_adaptive(filtered)

        # Fase 6: Operaciones Morfológicas
        morph = self.apply_morphology(binary_adaptive)

        # Fase 7: Deskewing / Alineación
        deskewed, angle = self.deskew_image(morph)

        # Imagen final recomendada para el motor OCR
        final_for_ocr = deskewed

        return PreprocessedPlate(
            original_crop=crop,
            resized=resized,
            gray=gray,
            filtered=filtered,
            binary_otsu=binary_otsu,
            binary_adaptive=binary_adaptive,
            morph=morph,
            deskewed=deskewed,
            final_for_ocr=final_for_ocr,
            rotation_angle=angle,
        )
