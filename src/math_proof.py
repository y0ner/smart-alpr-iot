"""Demostración matemática reproducible: Convolución espacial 3x3 pura vs OpenCV.

Ejecuta el cálculo analítico paso a paso de filtrado espacial sobre un parche
real de 3x3 píxeles de una placa vehicular, demostrando paridad numérica
exacta (diferencia cero) frente a cv2.filter2D.
Alimenta la Sección C.2 del Análisis Técnico.
"""

from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray

# Parche representativo 3x3 de borde de carácter (Letra oscura ~45-55 vs Lámina clara ~205-215)
DEFAULT_PATCH_3X3: list[list[float]] = [
    [45.0, 50.0, 210.0],
    [40.0, 55.0, 215.0],
    [42.0, 48.0, 205.0],
]

# Kernel Sobel Horizontal Gx para detección de bordes verticales de caracteres
SOBEL_GX_KERNEL: list[list[float]] = [
    [-1.0, 0.0, 1.0],
    [-2.0, 0.0, 2.0],
    [-1.0, 0.0, 1.0],
]

# Kernel de Enfoque / Realce (Sharpening)
SHARPEN_KERNEL: list[list[float]] = [
    [0.0, -1.0, 0.0],
    [-1.0, 5.0, -1.0],
    [0.0, -1.0, 0.0],
]

# Kernel de Suavizado por Promedio (Box Blur 3x3)
BOX_BLUR_KERNEL: list[list[float]] = [
    [1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0],
    [1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0],
    [1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0],
]


def convolve_center_pure_python(
    patch: list[list[float]],
    kernel: list[list[float]],
) -> tuple[float, list[dict[str, Any]]]:
    """Calcula la correlación/convolución 2D en el píxel central (1, 1) con aritmética pura.

    Returns:
        Tupla (resultado_escalar, lista_de_multiplicaciones_detalladas).
    """
    total = 0.0
    terms: list[dict[str, Any]] = []

    for r in range(3):
        for c in range(3):
            p_val = float(patch[r][c])
            k_val = float(kernel[r][c])
            prod = p_val * k_val
            total += prod
            terms.append(
                {
                    "row": r,
                    "col": c,
                    "pixel": p_val,
                    "kernel": k_val,
                    "product": prod,
                }
            )

    return total, terms


def convolve_center_opencv(
    patch: list[list[float]] | NDArray[np.float64],
    kernel: list[list[float]] | NDArray[np.float64],
) -> float:
    """Ejecuta cv2.filter2D con precisión float64 y extrae el valor central (1, 1)."""
    p_mat = np.array(patch, dtype=np.float64)
    k_mat = np.array(kernel, dtype=np.float64)

    # cv2.filter2D aplica correlación cruzada bidimensional sin rotar el kernel
    out = cv2.filter2D(p_mat, ddepth=cv2.CV_64F, kernel=k_mat, borderType=cv2.BORDER_CONSTANT)
    return float(out[1, 1])


def run_math_proof(
    patch: list[list[float]] | None = None,
    kernel: list[list[float]] | None = None,
    kernel_name: str = "Sobel Horizontal (Gx)",
) -> dict[str, Any]:
    """Ejecuta la comparación rigurosa entre aritmética pura de Python y OpenCV."""
    p = patch or DEFAULT_PATCH_3X3
    k = kernel or SOBEL_GX_KERNEL

    py_val, terms = convolve_center_pure_python(p, k)
    cv_val = convolve_center_opencv(p, k)
    abs_diff = abs(py_val - cv_val)
    is_exact_match = abs_diff < 1e-11

    return {
        "kernel_name": kernel_name,
        "patch": p,
        "kernel": k,
        "python_result": py_val,
        "opencv_result": cv_val,
        "absolute_difference": abs_diff,
        "is_exact_match": is_exact_match,
        "terms": terms,
    }


def generate_markdown_report(proof: dict[str, Any]) -> str:
    """Genera la derivación matemática paso a paso en formato Markdown."""
    patch = proof["patch"]
    kernel = proof["kernel"]
    terms = proof["terms"]

    lines: list[str] = [
        f"### Demostración Matemática: Convolución Espacial 3x3 — {proof['kernel_name']}",
        "",
        "#### 1. Matriz del Parche de Imagen $P$ (3x3 píxeles en escala de grises):",
        "```text",
        f"| {patch[0][0]:6.1f}  {patch[0][1]:6.1f}  {patch[0][2]:6.1f} |",
        f"| {patch[1][0]:6.1f}  {patch[1][1]:6.1f}  {patch[1][2]:6.1f} |",
        f"| {patch[2][0]:6.1f}  {patch[2][1]:6.1f}  {patch[2][2]:6.1f} |",
        "```",
        "",
        "#### 2. Matriz del Kernel Espacial $K$ (3x3):",
        "```text",
        f"| {kernel[0][0]:6.2f}  {kernel[0][1]:6.2f}  {kernel[0][2]:6.2f} |",
        f"| {kernel[1][0]:6.2f}  {kernel[1][1]:6.2f}  {kernel[1][2]:6.2f} |",
        f"| {kernel[2][0]:6.2f}  {kernel[2][1]:6.2f}  {kernel[2][2]:6.2f} |",
        "```",
        "",
        "#### 3. Operación Aritmética de Filtrado Espacial en el Píxel Central $(1, 1)$:",
        r"$$R(1, 1) = \sum_{i=0}^{2} \sum_{j=0}^{2} P(i, j) \cdot K(i, j)$$",
        "",
        r"| Coordenada $(i, j)$ | Intensidad Píxel $P(i, j)$ | Peso Kernel $K(i, j)$ | Producto Parcial $P \cdot K$ |",
        "| :---: | :---: | :---: | :---: |",
    ]

    for t in terms:
        lines.append(
            f"| `({t['row']}, {t['col']})` | `{t['pixel']:6.1f}` | `{t['kernel']:6.2f}` | **`{t['product']:+8.2f}`** |"
        )

    lines.extend(
        [
            "",
            f"**Suma Aritmética Total (Python Puro):** `{proof['python_result']:.4f}`  ",
            f"**Respuesta `cv2.filter2D` (OpenCV):** `{proof['opencv_result']:.4f}`  ",
            f"**Diferencia Numérica Absoluta:** `{proof['absolute_difference']:.1e}`  ",
            f"**Resultado de Paridad:** {'✅ EXACTO (ERROR CERO)' if proof['is_exact_match'] else '❌ DISCREPANCIA'}",
        ]
    )

    return "\n".join(lines)


if __name__ == "__main__":
    report_sobel = run_math_proof(kernel=SOBEL_GX_KERNEL, kernel_name="Sobel Horizontal (Gx)")
    print(generate_markdown_report(report_sobel))
