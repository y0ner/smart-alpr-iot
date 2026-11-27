"""Pruebas del evaluador experimental A/B (Reto Extra) y de sus métricas."""

import pytest

from src.benchmark_ab import (
    BenchmarkEvaluator,
    BenchmarkReport,
    calculate_cer,
    create_synthetic_benchmark_dataset,
    levenshtein_distance,
)


def test_levenshtein_and_cer_metrics() -> None:
    """Verifica el cálculo de distancia de edición y CER."""
    assert levenshtein_distance("ABC123", "ABC123") == 0
    assert levenshtein_distance("ABC123", "AB0123") == 1
    assert levenshtein_distance("ABC123", "XYZ999") == 6

    assert calculate_cer("ABC123", "ABC123") == 0.0
    assert calculate_cer("ABO123", "ABC123") == pytest.approx(1.0 / 6.0, rel=1e-3)
    assert calculate_cer("", "ABC123") == 1.0


def test_benchmark_report_markdown_generation() -> None:
    """Verifica la generación del informe comparativo del Reto Extra A/B."""
    dataset = create_synthetic_benchmark_dataset(num_samples=2)
    evaluator = BenchmarkEvaluator(use_gpu=False)
    report: BenchmarkReport = evaluator.evaluate_dataset(dataset)

    md = report.to_markdown()
    assert "Evaluación Experimental A/B: Reto Extra" in md
    assert "Modo A: Recorte Crudo" in md
    assert "Modo B: Pipeline OpenCV" in md
    assert "Curva y Tabla de Sensibilidad del Umbral de Confianza YOLO" in md
    assert len(report.sensitivity_curve) == 6
