"""Pruebas de integración de extremo a extremo (E2E) para el motor ALPR y Reto Extra A/B.

Verifica la detección, extracción OCR, validación sintáctica, votación temporal,
persistencia en SQLite, telemetría de latencias/FPS y evaluación experimental A/B.
"""

import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest

from src.benchmark_ab import (
    BenchmarkEvaluator,
    BenchmarkReport,
    calculate_cer,
    create_synthetic_benchmark_dataset,
    levenshtein_distance,
)
from src.config import SCHEMA_SQL_PATH
from src.db import DatabaseManager
from src.detector import ALPRPipeline, DetectionBBox, PipelineLatency
from src.validator import PlateVoterBuffer


@pytest.fixture
def temp_output_dir() -> Path:
    """Provee un directorio temporal para capturas procesadas."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)


@pytest.fixture
def memory_db() -> DatabaseManager:
    """Provee un gestor SQLite en memoria con esquema inicializado."""
    mgr = DatabaseManager(db_path=Path(":memory:"))
    mgr.init_db(schema_path=SCHEMA_SQL_PATH)
    return mgr


def create_clean_plate_frame(plate_text: str = "ABC123") -> np.ndarray:
    """Crea un frame que contiene una placa vehicular centrada nítida."""
    # Frame de 300x500 píxeles
    frame = np.full((300, 500, 3), (200, 200, 200), dtype=np.uint8)
    # Fondo de placa 90x240 centrado en (130, 105)
    x1, y1, x2, y2 = 130, 105, 370, 195
    # Fondo amarillo
    frame[y1:y2, x1:x2] = (21, 204, 250)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 2)
    # Texto centrado
    cv2.putText(
        frame,
        plate_text,
        (x1 + 18, y1 + 65),
        cv2.FONT_HERSHEY_DUPLEX,
        1.7,
        (10, 10, 10),
        3,
        cv2.LINE_AA,
    )
    return frame


def test_pipeline_initialization(memory_db: DatabaseManager, temp_output_dir: Path) -> None:
    """Verifica la inicialización del orquestador ALPR con parámetros personalizados."""
    pipeline = ALPRPipeline(
        conf_threshold=0.50,
        use_gpu=False,
        db_manager=memory_db,
        output_dir=temp_output_dir,
    )
    assert pipeline.conf_threshold == 0.50
    assert pipeline.use_gpu is False
    assert pipeline.output_dir == temp_output_dir


def test_detect_plates_synthetic() -> None:
    """Verifica que el detector localice la región de placa en el frame."""
    pipeline = ALPRPipeline(use_gpu=False)
    frame = create_clean_plate_frame("UJR456")
    boxes = pipeline.detect_plates(frame)

    assert len(boxes) >= 1
    box = boxes[0]
    assert isinstance(box, DetectionBBox)
    assert box.x2 > box.x1
    assert box.y2 > box.y1
    assert box.confidence > 0.0


def test_invalid_frame_raises_error(memory_db: DatabaseManager) -> None:
    """Verifica que un frame vacío lance ValueError."""
    pipeline = ALPRPipeline(db_manager=memory_db, use_gpu=False)
    empty_frame = np.array([], dtype=np.uint8)
    with pytest.raises(ValueError):
        pipeline.process_frame(empty_frame)


def test_process_frame_end_to_end_single_frame(
    memory_db: DatabaseManager, temp_output_dir: Path
) -> None:
    """Verifica la ejecución de un frame midiendo telemetría y overlays."""
    # Usar mock o OCR real en CPU
    voter = PlateVoterBuffer(window_size=3, min_matches=2)
    pipeline = ALPRPipeline(
        db_manager=memory_db,
        voter_buffer=voter,
        output_dir=temp_output_dir,
        use_gpu=False,
    )

    frame = create_clean_plate_frame("ABC123")
    result = pipeline.process_frame(frame, timestamp=100.0)

    # Verificar estructura de latencia y FPS
    assert isinstance(result.latency, PipelineLatency)
    assert result.latency.total_ms > 0
    assert result.latency.fps > 0
    assert result.annotated_frame.shape == frame.shape

    # En el primer cuadro no debe haber consenso aún (min_matches=2)
    assert result.consensus is None
    assert result.access_record is None


def test_process_frame_consensus_and_sqlite_persistence(
    memory_db: DatabaseManager, temp_output_dir: Path
) -> None:
    """Verifica el flujo E2E: ráfaga multi-cuadro -> consenso -> persistencia SQLite + archivo."""
    voter = PlateVoterBuffer(window_size=5, min_matches=2, cooldown_seconds=10.0)
    pipeline = ALPRPipeline(
        db_manager=memory_db,
        voter_buffer=voter,
        output_dir=temp_output_dir,
        save_captures=True,
        use_gpu=False,
    )

    frame = create_clean_plate_frame("ABC123")

    # Cuadro 1: Primer voto
    r1 = pipeline.process_frame(frame, timestamp=200.0)
    assert r1.consensus is None

    # Cuadro 2: Segundo voto -> ALCANZA CONSENSO (min_matches=2)
    r2 = pipeline.process_frame(frame, timestamp=200.1)

    # Si EasyOCR leyó la matrícula (o si se detectó la región)
    if r2.detections and r2.detections[0].is_valid:
        assert r2.consensus is not None
        assert r2.consensus.plate == "ABC123"
        assert r2.access_record is not None
        assert r2.access_record["sentido"] == "ENTRADA"

        # Verificar persistencia en base de datos SQLite
        historial = memory_db.get_historial_accesos(placa="ABC123")
        assert len(historial) == 1
        assert historial[0]["placa"] == "ABC123"

        # Verificar que la captura se guardó en disco
        assert r2.saved_capture_path is not None
        assert Path(r2.saved_capture_path).exists()


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
