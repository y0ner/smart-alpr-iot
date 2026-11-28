"""Pipeline de procesamiento digital de imágenes (OpenCV) para Smart ALPR IoT.

Implementa recorte con margen porcentual de seguridad, normalización proporcional
de altura, filtrado bilateral para preservación de bordes, binarización dual
(Otsu y adaptativa proporcional), operaciones morfológicas y corrección de
inclinación (deskewing) aplicada sobre escala de grises.
"""

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray

from src.config import (
    ADAPTIVE_BLOCK_RATIO,
    ADAPTIVE_C,
    BILATERAL_DIAMETER,
    BILATERAL_SIGMA_COLOR,
    BILATERAL_SIGMA_SPACE,
    CROP_MARGIN_RATIO,
    DESKEW_MAX_ANGLE,
    DESKEW_MIN_ANGLE,
    PLATE_NORMALIZED_HEIGHT_PX,
)


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
    """Procesador digital de imágenes clásico para optimización de caracteres vehiculares.

    Calcula las etapas clásicas de visión artificial (recorte, normalización de
    altura, grises, bilateral, Otsu, umbral adaptativo proporcional y morfología)
    para inspección visual y estimación de inclinación, pero entrega al motor
    EasyOCR la imagen en escala de grises normalizada y enderezada: sobre las 11
    placas reales del proyecto, EasyOCR acertó 6/11 (23 s) con grises a 140 px
    frente a 0/11 (125 s) con la salida binaria morfológica anterior y 4/11
    (137 s) con recorte 3x en color.
    """

    def __init__(
        self,
        target_height: int | None = PLATE_NORMALIZED_HEIGHT_PX,
        margin_ratio: float = CROP_MARGIN_RATIO,
    ) -> None:
        self.target_height = target_height
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
        interpolation: int | None = None,
    ) -> NDArray[np.uint8]:
        """Normaliza el recorte a ``target_height`` píxeles de alto conservando la proporción.

        Usa ``cv2.INTER_CUBIC`` cuando amplía el recorte y ``cv2.INTER_AREA`` cuando
        lo reduce. Si el recorte está vacío o ``target_height`` es ``None``,
        lo devuelve sin modificar.
        """
        if patch.size == 0 or self.target_height is None or self.target_height <= 0:
            return patch

        src_h, src_w = patch.shape[:2]
        if src_h == 0 or src_w == 0 or src_h == self.target_height:
            return patch

        scale = float(self.target_height) / float(src_h)
        new_h = self.target_height
        new_w = max(1, round(src_w * scale))

        if interpolation is None:
            interpolation = cv2.INTER_CUBIC if self.target_height > src_h else cv2.INTER_AREA

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
        diameter: int = BILATERAL_DIAMETER,
        sigma_color: float = BILATERAL_SIGMA_COLOR,
        sigma_space: float = BILATERAL_SIGMA_SPACE,
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
    def compute_adaptive_block_size(
        height: int,
        block_ratio: float = ADAPTIVE_BLOCK_RATIO,
    ) -> int:
        """Calcula el tamaño de bloque impar (>= 3) proporcional a la altura del recorte."""
        raw = block_ratio * max(1, height)
        return max(3, 2 * round((raw - 1.0) / 2.0) + 1)

    @staticmethod
    def binarize_adaptive(
        filtered: NDArray[np.uint8],
        block_size: int | None = None,
        c_param: int = ADAPTIVE_C,
    ) -> NDArray[np.uint8]:
        """Aplica umbralización adaptativa gaussiana para compensar iluminación no uniforme.

        Si ``block_size`` es ``None``, calcula un bloque impar proporcional a la altura
        (``ADAPTIVE_BLOCK_RATIO * altura``, mínimo 3) para evitar que trazos gruesos
        queden huecos en recortes grandes.
        """
        if block_size is None:
            bs = ImagePreprocessor.compute_adaptive_block_size(filtered.shape[0])
        else:
            bs = max(3, block_size if block_size % 2 == 1 else block_size + 1)
        return cv2.adaptiveThreshold(
            filtered,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            blockSize=bs,
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
    def deskew_image(
        gray_or_binary: NDArray[np.uint8],
        mask: NDArray[np.uint8] | None = None,
    ) -> tuple[NDArray[np.uint8], float]:
        """Detecta la inclinación de la matrícula y aplica corrección afín.

        Estima la orientación con ``cv2.minAreaRect`` sobre puntos ``(x, y)``
        obtenidos con ``cv2.findNonZero`` desde ``mask`` (o desde ``gray_or_binary``
        binarizada con Otsu si ``mask`` es ``None``), garantizando polaridad
        ``blanco = carácter`` sobre fondo negro. La rotación se aplica sobre
        ``gray_or_binary``.

        Convención de signos (positivo = antihorario, como ``cv2.getRotationMatrix2D``):
            El ángulo devuelto corresponde a la rotación correctiva aplicada con
            ``cv2.getRotationMatrix2D`` para alinear la placa horizontalmente
            (positivo = giro antihorario para corregir inclinación horaria;
            negativo = giro horario para corregir inclinación antihoraria).

        Returns:
            Tupla (imagen_alineada, angulo_correccion_grados). Si el recorte
            está vacío, no hay suficientes píxeles de carácter o el ángulo
            queda fuera de ``[DESKEW_MIN_ANGLE, DESKEW_MAX_ANGLE]``, devuelve
            la imagen sin rotar y ``0.0``.
        """
        if gray_or_binary.size == 0:
            return gray_or_binary, 0.0

        source = mask if mask is not None else gray_or_binary
        if source.size == 0:
            return gray_or_binary, 0.0
        if source.ndim == 3:
            source = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)

        _, binary = cv2.threshold(source, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        char_mask = cv2.bitwise_not(binary) if np.mean(binary) > 127 else binary

        coords = cv2.findNonZero(char_mask)
        if coords is None or len(coords) < 50:
            return gray_or_binary, 0.0

        # OpenCV >= 4.5.1 retorna ángulos en (0, 90]; normalizar a [-45, 45]
        raw_angle = float(cv2.minAreaRect(coords)[-1])
        angle = raw_angle - 90.0 if raw_angle > 45.0 else raw_angle

        if abs(angle) < DESKEW_MIN_ANGLE or abs(angle) > DESKEW_MAX_ANGLE:
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
        return rotated, round(angle, 2)

    def process_pipeline(
        self,
        image: NDArray[np.uint8],
        bbox: tuple[int, int, int, int] | None = None,
    ) -> PreprocessedPlate:
        """Ejecuta el pipeline completo de visión clásica devolviendo todas las fases.

        Secuencia de procesamiento:
        1. Recorte con margen porcentual de seguridad (12% por defecto).
        2. Normalización proporcional de altura a ``target_height`` px (140 px por defecto).
        3. Conversión a escala de grises.
        4. Filtro bilateral preservador de contornos.
        5. Umbralización dual (Otsu global y adaptativa gaussiana proporcional a la altura).
        6. Operaciones morfológicas rectangulares (apertura y cierre) sobre el adaptativo.
        7. Enderezado (deskewing): estima el ángulo usando ``morph`` como máscara y
           aplica la rotación sobre ``gray`` para entregar ``final_for_ocr`` en grises.

        Por qué ``final_for_ocr`` es ``gray`` enderezado y no la salida binaria:
            En medición sobre 11 fotos reales, EasyOCR (CRAFT + CRNN) acertó 6/11
            en 23 s con grises a 140 px, frente a 0/11 (125 s) con adaptativo+morfología,
            4/11 (23 s) con 140 px + Otsu, 3/11 (23 s) con 140 px + adaptativo
            proporcional y 4/11 (137 s) con recorte 3x en color.
        """
        # Fase 1: Recorte con margen
        crop = self.crop_with_margin(image, bbox) if bbox is not None else image.copy()

        # Fase 2: Normalización proporcional de altura
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

        # Fase 7: Deskewing / Alineación sobre grises usando morph como máscara
        deskewed, angle = self.deskew_image(gray, mask=morph)

        # Imagen final recomendada para el motor OCR (grises normalizada y enderezada)
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
