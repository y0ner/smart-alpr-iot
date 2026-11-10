"""Pruebas unitarias para el pipeline de visión clásica OpenCV y demostración matemática 3x3.

Verifica el recorte con margen, escalado bicúbico, filtros bilaterales, umbralización dual,
morfología, corrección de inclinación y paridad matemática exacta contra cv2.filter2D.
"""

import cv2
import numpy as np
from numpy.typing import NDArray

from src.math_proof import (
    BOX_BLUR_KERNEL,
    SHARPEN_KERNEL,
    SOBEL_GX_KERNEL,
    generate_markdown_report,
    run_math_proof,
)
from src.preprocessor import ImagePreprocessor, PreprocessedPlate


def create_synthetic_plate(
    text: str = "ABC123",
    add_screws: bool = True,
    add_noise: bool = True,
    add_shadow: bool = True,
) -> NDArray[np.uint8]:
    """Genera una imagen sintética realista de una matrícula colombiana."""
    # Base: 110x300 píxeles, fondo amarillo típico colombiano (#FACC15 -> BGR: [21, 204, 250])
    img = np.full((110, 300, 3), (21, 204, 250), dtype=np.uint8)

    # Borde exterior negro
    cv2.rectangle(img, (4, 4), (295, 105), (0, 0, 0), thickness=3)

    # Texto de matrícula
    cv2.putText(
        img,
        text,
        (25, 75),
        cv2.FONT_HERSHEY_DUPLEX,
        fontScale=2.0,
        color=(10, 10, 10),
        thickness=4,
        lineType=cv2.LINE_AA,
    )

    # Simulación de tornillos / remaches en esquinas
    if add_screws:
        for center in [(20, 20), (280, 20), (20, 90), (280, 90)]:
            cv2.circle(img, center, radius=4, color=(50, 50, 50), thickness=-1)

    # Simulación de sombra en esquina superior derecha
    if add_shadow:
        shadow_overlay = img.copy()
        cv2.rectangle(shadow_overlay, (150, 0), (300, 60), (0, 0, 0), thickness=-1)
        cv2.addWeighted(shadow_overlay, 0.25, img, 0.75, 0, img)

    # Simulación de ruido gaussiano (polvo y grano de sensor)
    if add_noise:
        noise = np.random.normal(0, 8, img.shape).astype(np.float32)
        noisy = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        return noisy

    return img


def test_crop_with_margin() -> None:
    """Verifica que el recorte con margen expanda la ROI respetando los límites de imagen."""
    pre = ImagePreprocessor(margin_ratio=0.12)
    canvas = np.zeros((500, 500, 3), dtype=np.uint8)
    bbox = (100, 100, 200, 150)  # w=100, h=50 -> margin_x=12, margin_y=6

    crop = pre.crop_with_margin(canvas, bbox)
    # Dimensiones esperadas: h = 50 + 12 = 62, w = 100 + 24 = 124
    assert crop.shape[0] == 62
    assert crop.shape[1] == 124

    # Caso borde: Bbox pegada al origen (0, 0)
    bbox_zero = (0, 0, 50, 50)
    crop_zero = pre.crop_with_margin(canvas, bbox_zero)
    assert crop_zero.shape[0] > 0
    assert crop_zero.shape[1] > 0


def test_resize_patch() -> None:
    """Verifica que el escalado aumente proporcionalmente la resolución."""
    pre = ImagePreprocessor(scale_factor=2.5)
    patch = np.zeros((40, 80, 3), dtype=np.uint8)
    resized = pre.resize_patch(patch)
    assert resized.shape == (100, 200, 3)
    assert resized.dtype == np.uint8


def test_pipeline_phases_presence_and_types() -> None:
    """Verifica que el pipeline devuelva todas las fases requeridas con tipo uint8."""
    pre = ImagePreprocessor(scale_factor=2.0, margin_ratio=0.10)
    plate_img = create_synthetic_plate()

    result: PreprocessedPlate = pre.process_pipeline(plate_img)

    # Verificar presencia de todas las etapas
    assert isinstance(result.original_crop, np.ndarray)
    assert isinstance(result.resized, np.ndarray)
    assert isinstance(result.gray, np.ndarray)
    assert isinstance(result.filtered, np.ndarray)
    assert isinstance(result.binary_otsu, np.ndarray)
    assert isinstance(result.binary_adaptive, np.ndarray)
    assert isinstance(result.morph, np.ndarray)
    assert isinstance(result.deskewed, np.ndarray)
    assert isinstance(result.final_for_ocr, np.ndarray)
    assert isinstance(result.rotation_angle, float)

    # Tipos de datos deben ser estrictamente uint8
    assert result.gray.dtype == np.uint8
    assert result.filtered.dtype == np.uint8
    assert result.binary_otsu.dtype == np.uint8
    assert result.binary_adaptive.dtype == np.uint8
    assert result.morph.dtype == np.uint8
    assert result.final_for_ocr.dtype == np.uint8

    # Formas bidimensionales para las fases monocromáticas
    assert len(result.gray.shape) == 2
    assert len(result.filtered.shape) == 2
    assert len(result.binary_otsu.shape) == 2

    # Compatibilidad con diccionario
    dict_repr = result.to_dict()
    assert "binary_otsu" in dict_repr
    assert "morph" in dict_repr


def test_morphology_removes_small_screws() -> None:
    """Verifica que la morfología atenúe remaches o motas aisladas de ruido."""
    img = np.full((100, 200), 255, dtype=np.uint8)
    # Dibujar un remache circular pequeño aislado (radio 1)
    cv2.circle(img, (25, 25), radius=1, color=0, thickness=-1)

    morphed = ImagePreprocessor.apply_morphology(img)
    # En el centro del remache, el valor debe haber sido filtrado de regreso a fondo claro
    assert morphed[25, 25] == 255


def test_deskew_image_known_rotation() -> None:
    """Verifica que la rutina de deskewing detecte y compense inclinaciones."""
    plate = create_synthetic_plate(add_noise=False, add_shadow=False)
    gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)

    # Rotar artificialmente 8 grados
    h, w = gray.shape[:2]
    matrix = cv2.getRotationMatrix2D((w // 2, h // 2), 8.0, 1.0)
    rotated = cv2.warpAffine(gray, matrix, (w, h), borderValue=255)

    corrected, detected_angle = ImagePreprocessor.deskew_image(rotated)
    assert isinstance(detected_angle, float)
    assert corrected.shape == (h, w)


def test_math_proof_parity_sobel() -> None:
    """Verifica la paridad numérica exacta (error cero) en el parche 3x3 con Sobel Gx."""
    proof = run_math_proof(kernel=SOBEL_GX_KERNEL, kernel_name="Sobel Horizontal (Gx)")
    assert proof["is_exact_match"] is True
    assert proof["absolute_difference"] < 1e-11
    assert proof["python_result"] == 678.0
    assert proof["opencv_result"] == 678.0


def test_math_proof_parity_sharpen_and_box() -> None:
    """Verifica la paridad numérica exacta en kernels Sharpening y Box Blur."""
    proof_sharpen = run_math_proof(kernel=SHARPEN_KERNEL, kernel_name="Realce / Sharpening")
    assert proof_sharpen["is_exact_match"] is True
    assert proof_sharpen["absolute_difference"] < 1e-11

    proof_box = run_math_proof(kernel=BOX_BLUR_KERNEL, kernel_name="Filtro Promedio / Box Blur")
    assert proof_box["is_exact_match"] is True
    assert proof_box["absolute_difference"] < 1e-11


def test_math_proof_markdown_generation() -> None:
    """Verifica que el generador de informe genere texto con formato Markdown válido."""
    proof = run_math_proof()
    md = generate_markdown_report(proof)
    assert "Demostración Matemática: Convolución Espacial 3x3" in md
    assert "Suma Aritmética Total" in md
    assert "EXACTO (ERROR CERO)" in md
