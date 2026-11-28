"""Pruebas del evaluador experimental A/B (Reto Extra) y de sus métricas."""

import json
from pathlib import Path

import numpy as np
import pytest

from src.benchmark_ab import (
    BenchmarkEvaluator,
    BenchmarkReport,
    LabeledSample,
    RealSamplesReport,
    calculate_cer,
    create_synthetic_benchmark_dataset,
    evaluate_real_samples,
    levenshtein_distance,
    load_labeled_samples,
    main,
)
from src.config import PROCESSED_DATA_DIR, SCHEMA_SQL_PATH, SQLITE_DB_PATH
from src.db import DatabaseManager
from src.detector import ALPRPipeline, DetectionBBox


class _FakeOCRReader:
    """Lector OCR falso para probar BenchmarkEvaluator sin cargar EasyOCR."""

    def readtext(
        self,
        img: np.ndarray,
        allowlist: str = "",
        detail: int = 1,
        paragraph: bool = False,
    ) -> list[tuple[list[list[int]], str, float]]:
        del allowlist, detail, paragraph
        text = "UJR456" if img.ndim == 2 else "UJR45"
        conf = 0.92 if img.ndim == 2 else 0.55
        return [([[0, 0], [50, 0], [50, 20], [0, 20]], text, conf)]


def _build_memory_pipeline() -> ALPRPipeline:
    """Construye un pipeline en memoria sin escritura a disco para pruebas."""
    db = DatabaseManager(db_path=":memory:")
    db.init_db(schema_path=SCHEMA_SQL_PATH)
    return ALPRPipeline(use_gpu=False, db_manager=db, save_captures=False)


def test_levenshtein_and_cer_metrics() -> None:
    """Verifica el cálculo de distancia de edición y CER."""
    assert levenshtein_distance("ABC123", "ABC123") == 0
    assert levenshtein_distance("ABC123", "AB0123") == 1
    assert levenshtein_distance("ABC123", "XYZ999") == 6

    assert calculate_cer("ABC123", "ABC123") == 0.0
    assert calculate_cer("ABO123", "ABC123") == pytest.approx(1.0 / 6.0, rel=1e-3)
    assert calculate_cer("", "ABC123") == 1.0


def test_benchmark_report_markdown_generation() -> None:
    """Verifica el informe sintético sin curvas inventadas ni carga de EasyOCR."""
    dataset = create_synthetic_benchmark_dataset(num_samples=2)
    evaluator = BenchmarkEvaluator(use_gpu=False)
    evaluator._reader = _FakeOCRReader()

    report: BenchmarkReport = evaluator.evaluate_dataset(dataset)
    md = report.to_markdown()

    assert "Evaluación Experimental A/B: Reto Extra" in md
    assert "lote sintético" in md
    assert "Modo A: Recorte Crudo" in md
    assert "Modo B: Pipeline OpenCV" in md
    assert "Sensibilidad del umbral YOLO: sin medir" in md
    assert "sin medir" in md
    assert "Precisión Estimada" not in md
    assert "Curva y Tabla de Sensibilidad" not in md
    assert not hasattr(report, "sensitivity_curve")
    assert report.mode_a_raw.exact_matches == 0
    assert report.mode_b_preprocessed.exact_matches == 1
    assert "mejoró" in md

    parsed = json.loads(json.dumps(report.to_dict()))
    assert parsed["dataset_type"] == "lote sintético"


def test_load_labeled_samples_real_ground_truth() -> None:
    """Verifica que load_labeled_samples cargue las 11 muestras reales existentes."""
    samples = load_labeled_samples()

    assert len(samples) == 11
    assert all(isinstance(s, LabeledSample) for s in samples)
    assert all(s.ruta.exists() and len(s.placa) == 6 for s in samples)


def test_load_labeled_samples_missing_files(tmp_path: Path) -> None:
    """Verifica errores claros si falta el CSV o alguna imagen referenciada."""
    with pytest.raises(FileNotFoundError):
        load_labeled_samples(tmp_path / "no_existe.csv")

    csv_file = tmp_path / "verdad.csv"
    csv_file.write_text("archivo,placa\nfaltante.jpg,AGH430\n", encoding="utf-8")
    with pytest.raises(FileNotFoundError):
        load_labeled_samples(csv_file)


def test_evaluate_real_samples_mode_b_improves(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifica conteo exacto, CER y conclusión 'mejoró' cuando Modo B supera a Modo A."""
    samples = load_labeled_samples()[:3]  # Todas corresponden a AGH430
    pipeline = _build_memory_pipeline()

    def fake_ocr(img: np.ndarray) -> tuple[str, float]:
        if img.ndim == 2:
            return "AGH430", 0.90
        return "AGH439", 0.50

    monkeypatch.setattr(pipeline, "extract_ocr_text", fake_ocr)

    report: RealSamplesReport = evaluate_real_samples(samples, pipeline=pipeline)
    md = report.to_markdown()

    assert report.mode_a_raw.total_samples == 3
    assert report.mode_a_raw.exact_matches == 0
    assert report.mode_b_preprocessed.exact_matches == 3
    assert report.mode_a_raw.accuracy_percentage == 0.0
    assert report.mode_b_preprocessed.accuracy_percentage == 100.0
    assert report.mode_a_raw.mean_cer == pytest.approx(1.0 / 6.0, abs=1e-4)
    assert report.mode_b_preprocessed.mean_cer == 0.0
    assert report.detector_status == "respaldo, sin pesos YOLO"
    assert "mejoró" in md
    assert "sin medir" in md
    assert "Precisión Estimada" not in md
    assert "respaldo, sin pesos YOLO" in md

    serialized = json.loads(json.dumps(report.to_dict()))
    assert len(serialized["per_sample"]) == 3
    assert serialized["detector"] == "respaldo, sin pesos YOLO"


def test_evaluate_real_samples_mode_b_worsens(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifica que la conclusión indique 'empeoró' cuando Modo B rinde menos que Modo A."""
    samples = load_labeled_samples()[:3]  # Todas corresponden a AGH430
    pipeline = _build_memory_pipeline()

    def fake_ocr(img: np.ndarray) -> tuple[str, float]:
        if img.ndim == 2:
            return "AGH439", 0.50
        return "AGH430", 0.90

    monkeypatch.setattr(pipeline, "extract_ocr_text", fake_ocr)

    report = evaluate_real_samples(samples, pipeline=pipeline)
    md = report.to_markdown()

    assert report.mode_a_raw.exact_matches == 3
    assert report.mode_b_preprocessed.exact_matches == 0
    assert "empeoró" in md


def test_evaluate_real_samples_tie_and_multibox_selection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verifica empate sin adjetivos y prioridad de lectura válida sobre inválida."""
    sample = load_labeled_samples()[0]
    pipeline = _build_memory_pipeline()

    monkeypatch.setattr(
        pipeline,
        "detect_plates",
        lambda img: [
            DetectionBBox(x1=0, y1=0, x2=min(100, img.shape[1]), y2=min(40, img.shape[0]), confidence=0.9),
            DetectionBBox(x1=0, y1=0, x2=min(120, img.shape[1]), y2=min(50, img.shape[0]), confidence=0.7),
        ],
    )

    calls = {"raw": 0, "prep": 0}

    def fake_ocr(img: np.ndarray) -> tuple[str, float]:
        if img.ndim == 2:
            calls["prep"] += 1
            # Caja 1 inválida con mayor confianza (0.95); caja 2 válida con menor confianza (0.65)
            return ("INVALID", 0.95) if calls["prep"] == 1 else ("AGH430", 0.65)
        calls["raw"] += 1
        # Caja 1 inválida con 0.80; caja 2 válida con 0.70
        return ("NO12", 0.80) if calls["raw"] == 1 else ("AGH430", 0.70)

    monkeypatch.setattr(pipeline, "extract_ocr_text", fake_ocr)

    report = evaluate_real_samples([sample], pipeline=pipeline)
    md = report.to_markdown()

    assert report.per_sample[0].boxes_count == 2
    assert report.per_sample[0].prediction_a == "AGH430"
    assert report.per_sample[0].prediction_b == "AGH430"
    assert report.mode_a_raw.exact_matches == 1
    assert report.mode_b_preprocessed.exact_matches == 1
    assert "mismos aciertos exactos" in md


def test_evaluate_real_samples_does_not_write_db_or_processed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verifica que pipeline=None use BD en memoria y no escriba en db/ ni data/processed/."""
    sample = load_labeled_samples()[0]
    db_existed_before = SQLITE_DB_PATH.exists()
    db_mtime_before = SQLITE_DB_PATH.stat().st_mtime_ns if db_existed_before else None
    processed_before = (
        sorted(p.name for p in PROCESSED_DATA_DIR.iterdir())
        if PROCESSED_DATA_DIR.exists()
        else []
    )

    monkeypatch.setattr(
        ALPRPipeline,
        "extract_ocr_text",
        lambda _self, img: ("AGH430", 0.88) if img.ndim == 2 else ("AGH430", 0.82),
    )

    report = evaluate_real_samples([sample], pipeline=None)
    assert report.mode_b_preprocessed.exact_matches == 1

    if db_existed_before:
        assert SQLITE_DB_PATH.stat().st_mtime_ns == db_mtime_before
    else:
        assert not SQLITE_DB_PATH.exists()

    processed_after = (
        sorted(p.name for p in PROCESSED_DATA_DIR.iterdir())
        if PROCESSED_DATA_DIR.exists()
        else []
    )
    assert processed_after == processed_before


def test_cli_synthetic_and_json_output(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Verifica la interfaz CLI con --sintetico y --json sin cargar EasyOCR."""
    monkeypatch.setattr(BenchmarkEvaluator, "_get_reader", lambda _self: _FakeOCRReader())
    json_out = tmp_path / "salida.json"

    exit_code = main(["--sintetico", "2", "--json", str(json_out)])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "lote sintético" in captured.out
    assert "sin medir" in captured.out
    assert json_out.exists()
    data = json.loads(json_out.read_text(encoding="utf-8"))
    assert data["dataset_type"] == "lote sintético"


def test_cli_real_samples_and_json_output(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Verifica la interfaz CLI por defecto sobre muestras reales con --json sin cargar EasyOCR."""
    samples = load_labeled_samples()[:2]
    monkeypatch.setattr("src.benchmark_ab.load_labeled_samples", lambda: samples)
    monkeypatch.setattr(
        ALPRPipeline,
        "extract_ocr_text",
        lambda _self, img: ("AGH430", 0.91) if img.ndim == 2 else ("AGH43", 0.45),
    )
    json_out = tmp_path / "real.json"

    exit_code = main(["--json", str(json_out)])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "Placas Reales Etiquetadas" in captured.out
    assert "mejoró" in captured.out
    assert "sin medir" in captured.out
    data = json.loads(json_out.read_text(encoding="utf-8"))
    assert data["dataset_type"] == "placas reales etiquetadas"
    assert len(data["per_sample"]) == 2


def test_evaluate_real_samples_invalid_fallback_and_empty_boxes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verifica fallback a mayor confianza para CER y manejo seguro sin cajas detectadas."""
    samples = load_labeled_samples()[:2]
    pipeline = _build_memory_pipeline()

    call_idx = {"photo": 0, "ocr": 0}

    def fake_detect(img: np.ndarray) -> list[DetectionBBox]:
        call_idx["photo"] += 1
        if call_idx["photo"] == 1:
            return [
                DetectionBBox(x1=0, y1=0, x2=min(80, img.shape[1]), y2=min(30, img.shape[0]), confidence=0.7),
                DetectionBBox(x1=0, y1=0, x2=min(90, img.shape[1]), y2=min(35, img.shape[0]), confidence=0.8),
            ]
        return []

    def fake_ocr(_img: np.ndarray) -> tuple[str, float]:
        call_idx["ocr"] += 1
        if call_idx["ocr"] in (1, 2):
            return "AB", 0.30
        return "AGH43", 0.75

    monkeypatch.setattr(pipeline, "detect_plates", fake_detect)
    monkeypatch.setattr(pipeline, "extract_ocr_text", fake_ocr)

    report = evaluate_real_samples(samples, pipeline=pipeline)
    assert report.per_sample[0].boxes_count == 2
    assert report.per_sample[0].prediction_a == "AGH43"
    assert report.per_sample[0].prediction_b == "AGH43"
    assert report.per_sample[0].cer_a == pytest.approx(1.0 / 6.0, abs=1e-4)
    assert report.per_sample[1].boxes_count == 0
    assert report.per_sample[1].prediction_a == ""
    assert report.per_sample[1].cer_a == 1.0

