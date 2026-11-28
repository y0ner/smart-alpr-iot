"""Pruebas unitarias para el pipeline de visión clásica OpenCV y demostración matemática 3x3.

Verifica el recorte con margen, normalización proporcional de altura, filtros
bilaterales, umbralización dual (Otsu y adaptativa proporcional), morfología,
corrección de inclinación y paridad matemática exacta contra cv2.filter2D.
"""

import cv2
import numpy as np
import pytest
from numpy.typing import NDArray

from src.config import (
    ADAPTIVE_C,
    DESKEW_MAX_ANGLE,
    DESKEW_MIN_ANGLE,
    PLATE_NORMALIZED_HEIGHT_PX,
)
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


def test_resize_patch(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifica la normalización de altura, conservación de proporción e interpolación."""
    pre = ImagePreprocessor(target_height=PLATE_NORMALIZED_HEIGHT_PX)
    interpolations: list[int] = []
    real_resize = cv2.resize

    def spy_resize(src: NDArray[np.uint8], dsize: tuple[int, int], **kwargs: int) -> NDArray[np.uint8]:
        interpolations.append(int(kwargs["interpolation"]))
        return real_resize(src, dsize, **kwargs)

    monkeypatch.setattr(cv2, "resize", spy_resize)

    # Reducción desde 900x1800 -> 140x280 con INTER_AREA
    large_patch = np.zeros((900, 1800, 3), dtype=np.uint8)
    resized_large = pre.resize_patch(large_patch)
    expected_w_large = round(1800 * (PLATE_NORMALIZED_HEIGHT_PX / 900))
    assert resized_large.shape[0] == PLATE_NORMALIZED_HEIGHT_PX
    assert abs(resized_large.shape[1] - expected_w_large) <= 1
    assert resized_large.dtype == np.uint8

    # Ampliación desde 40x80 -> 140x280 con INTER_CUBIC
    small_patch = np.zeros((40, 80, 3), dtype=np.uint8)
    resized_small = pre.resize_patch(small_patch)
    expected_w_small = round(80 * (PLATE_NORMALIZED_HEIGHT_PX / 40))
    assert resized_small.shape[0] == PLATE_NORMALIZED_HEIGHT_PX
    assert abs(resized_small.shape[1] - expected_w_small) <= 1
    assert resized_small.dtype == np.uint8

    assert interpolations == [cv2.INTER_AREA, cv2.INTER_CUBIC]

    # Recorte vacío se devuelve tal cual
    empty_patch = np.empty((0, 0, 3), dtype=np.uint8)
    assert pre.resize_patch(empty_patch).size == 0


def test_resize_patch_target_height_none_preserves_size() -> None:
    """Verifica que target_height=None desactive el reescalado."""
    pre = ImagePreprocessor(target_height=None)
    patch = np.zeros((900, 1800, 3), dtype=np.uint8)
    resized = pre.resize_patch(patch)
    assert resized.shape == (900, 1800, 3)

    plate = create_synthetic_plate(add_noise=False, add_shadow=False)
    res = pre.process_pipeline(plate)
    assert res.resized.shape == plate.shape


def test_pipeline_phases_presence_and_types() -> None:
    """Verifica que el pipeline devuelva todas las fases requeridas con tipo uint8."""
    pre = ImagePreprocessor(target_height=PLATE_NORMALIZED_HEIGHT_PX, margin_ratio=0.10)
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

    # Altura normalizada en las fases procesadas
    assert result.resized.shape[0] == PLATE_NORMALIZED_HEIGHT_PX
    assert result.final_for_ocr.shape[0] == PLATE_NORMALIZED_HEIGHT_PX

    # Tipos de datos deben ser estrictamente uint8
    assert result.gray.dtype == np.uint8
    assert result.filtered.dtype == np.uint8
    assert result.binary_otsu.dtype == np.uint8
    assert result.binary_adaptive.dtype == np.uint8
    assert result.morph.dtype == np.uint8
    assert result.deskewed.dtype == np.uint8
    assert result.final_for_ocr.dtype == np.uint8

    # Formas bidimensionales para las fases monocromáticas
    assert len(result.gray.shape) == 2
    assert len(result.filtered.shape) == 2
    assert len(result.binary_otsu.shape) == 2
    assert len(result.final_for_ocr.shape) == 2

    # Compatibilidad con diccionario y alias
    dict_repr = result.to_dict()
    assert "binary_otsu" in dict_repr
    assert "morph" in dict_repr
    assert np.array_equal(result["bilateral"], result.filtered)
    assert np.array_equal(result["otsu"], result.binary_otsu)


def test_final_for_ocr_is_grayscale_and_not_binary() -> None:
    """Verifica que final_for_ocr sea 2D uint8 en grises (más de 2 valores) con ruido."""
    pre = ImagePreprocessor()
    plate = create_synthetic_plate(add_noise=True, add_shadow=True)
    result = pre.process_pipeline(plate)

    assert result.final_for_ocr.ndim == 2
    assert result.final_for_ocr.dtype == np.uint8
    assert len(np.unique(result.final_for_ocr)) > 2
    assert np.array_equal(result.final_for_ocr, result.deskewed)


def test_adaptive_block_size_proportional_and_odd(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifica que el bloque adaptativo sea impar, >= 3 y crezca con la altura."""
    heights = [4, 10, 40, 140, 300, 900]
    blocks = [ImagePreprocessor.compute_adaptive_block_size(h) for h in heights]

    for bs in blocks:
        assert bs >= 3
        assert bs % 2 == 1

    assert blocks == sorted(blocks)
    assert blocks[-1] > blocks[3] > blocks[2]

    recorded_calls: list[tuple[int, int]] = []
    real_adaptive = cv2.adaptiveThreshold

    def spy_adaptive(
        src: NDArray[np.uint8],
        maxValue: float,
        adaptiveMethod: int,
        thresholdType: int,
        blockSize: int,
        C: float,
    ) -> NDArray[np.uint8]:
        recorded_calls.append((int(blockSize), int(C)))
        return real_adaptive(src, maxValue, adaptiveMethod, thresholdType, blockSize, C)

    monkeypatch.setattr(cv2, "adaptiveThreshold", spy_adaptive)

    small_gray = np.full((40, 120), 180, dtype=np.uint8)
    tall_gray = np.full((300, 900), 180, dtype=np.uint8)
    ImagePreprocessor.binarize_adaptive(small_gray)
    ImagePreprocessor.binarize_adaptive(tall_gray)

    assert len(recorded_calls) == 2
    (bs_small, c_small), (bs_tall, c_tall) = recorded_calls
    assert bs_small >= 3 and bs_small % 2 == 1
    assert bs_tall >= 3 and bs_tall % 2 == 1
    assert bs_tall > bs_small
    assert c_small == ADAPTIVE_C and c_tall == ADAPTIVE_C


def test_adaptive_threshold_900px_avoids_hollow_strokes() -> None:
    """Verifica que en 900 px de alto las letras de binary_adaptive queden rellenas."""
    base_plate = create_synthetic_plate(add_noise=False, add_shadow=False)
    plate_900 = cv2.resize(base_plate, (1800, 900), interpolation=cv2.INTER_CUBIC)
    gray_900 = ImagePreprocessor.to_grayscale(plate_900)
    filtered_900 = ImagePreprocessor.apply_bilateral_filter(gray_900)

    def text_dark_fraction(binary: NDArray[np.uint8]) -> float:
        h, w = binary.shape[:2]
        roi = binary[int(0.2 * h) : int(0.75 * h), int(0.08 * w) : int(0.92 * w)]
        return float(np.mean(roi == 0))

    fixed_19 = ImagePreprocessor.binarize_adaptive(filtered_900, block_size=19)
    frac_fixed_19 = text_dark_fraction(fixed_19)

    res_normalized = ImagePreprocessor().process_pipeline(plate_900)
    res_unscaled = ImagePreprocessor(target_height=None).process_pipeline(plate_900)

    assert text_dark_fraction(res_normalized.binary_adaptive) >= 2.0 * frac_fixed_19
    assert text_dark_fraction(res_unscaled.binary_adaptive) >= 2.0 * frac_fixed_19


def test_morphology_removes_small_screws() -> None:
    """Verifica que la morfología atenúe remaches o motas aisladas de ruido."""
    img = np.full((100, 200), 255, dtype=np.uint8)
    # Dibujar un remache circular pequeño aislado (radio 1)
    cv2.circle(img, (25, 25), radius=1, color=0, thickness=-1)

    morphed = ImagePreprocessor.apply_morphology(img)
    # En el centro del remache, el valor debe haber sido filtrado de regreso a fondo claro
    assert morphed[25, 25] == 255


@pytest.mark.parametrize("input_rotation", [8.0, -8.0])
def test_deskew_image_known_rotation(input_rotation: float) -> None:
    """Verifica que deskew_image mida con precisión ángulos de +8° y -8° y enderece."""
    plate = create_synthetic_plate(add_noise=False, add_shadow=False)
    gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)

    h, w = gray.shape[:2]
    matrix = cv2.getRotationMatrix2D((w // 2, h // 2), input_rotation, 1.0)
    rotated = cv2.warpAffine(gray, matrix, (w, h), borderValue=255)

    corrected, correction_angle = ImagePreprocessor.deskew_image(rotated)
    expected_correction = -input_rotation

    assert isinstance(correction_angle, float)
    assert corrected.shape == (h, w)
    assert (correction_angle > 0) == (expected_correction > 0)
    assert abs(correction_angle - expected_correction) <= 1.5

    # La imagen corregida debe quedar horizontal (sin requerir nueva rotación)
    _, residual_angle = ImagePreprocessor.deskew_image(corrected)
    assert residual_angle == 0.0


def test_deskew_image_straight_and_out_of_range_plates() -> None:
    """Verifica que una placa recta o fuera de límites devuelva 0.0 sin rotar."""
    plate = create_synthetic_plate(add_noise=False, add_shadow=False)
    gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)

    straight_out, straight_angle = ImagePreprocessor.deskew_image(gray)
    assert straight_angle == 0.0
    assert np.array_equal(straight_out, gray)

    h, w = gray.shape[:2]
    # Ángulo menor a DESKEW_MIN_ANGLE
    tiny_angle = DESKEW_MIN_ANGLE * 0.4
    m_tiny = cv2.getRotationMatrix2D((w // 2, h // 2), tiny_angle, 1.0)
    tiny_rot = cv2.warpAffine(gray, m_tiny, (w, h), borderValue=255)
    tiny_out, tiny_detected = ImagePreprocessor.deskew_image(tiny_rot)
    assert tiny_detected == 0.0
    assert np.array_equal(tiny_out, tiny_rot)

    # Ángulo mayor a DESKEW_MAX_ANGLE
    excessive_angle = DESKEW_MAX_ANGLE + 6.0
    m_big = cv2.getRotationMatrix2D((w // 2, h // 2), excessive_angle, 1.0)
    big_rot = cv2.warpAffine(gray, m_big, (w, h), borderValue=255)
    big_out, big_detected = ImagePreprocessor.deskew_image(big_rot)
    assert big_detected == 0.0
    assert np.array_equal(big_out, big_rot)


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
