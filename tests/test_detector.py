"""Pruebas de integración de extremo a extremo (E2E) para el motor ALPR.

Verifica la detección, extracción OCR, validación sintáctica, votación temporal,
persistencia en SQLite y telemetría de latencias/FPS.
"""

import tempfile
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import pytest

from src.config import (
    FALLBACK_MAX_CANDIDATES,
    OCR_EARLY_STOP_CONFIDENCE,
    SCHEMA_SQL_PATH,
)
from src.db import DatabaseManager
from src.detector import (
    ALPRPipeline,
    DetectionBBox,
    PipelineLatency,
    PlateDetectionResult,
)
from src.validator import PlateValidator, PlateVoterBuffer


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
    memory_db: DatabaseManager, temp_output_dir: Path, monkeypatch: pytest.MonkeyPatch
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
    # OCR determinista para verificar consenso y persistencia sin cargar EasyOCR
    monkeypatch.setattr(pipeline, "extract_ocr_text", lambda _img: ("ABC123", 0.9))

    frame = create_clean_plate_frame("ABC123")

    # Cuadro 1: Primer voto
    r1 = pipeline.process_frame(frame, timestamp=200.0)
    assert r1.consensus is None
    assert r1.detections and r1.detections[0].is_valid

    # Cuadro 2: Segundo voto -> ALCANZA CONSENSO (min_matches=2)
    r2 = pipeline.process_frame(frame, timestamp=200.1)
    assert r2.detections and r2.detections[0].is_valid
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


def test_detect_plates_fallback_caps_candidates() -> None:
    """El respaldo nunca devuelve más de FALLBACK_MAX_CANDIDATES cajas en un cuadro ruidoso."""
    pipeline = ALPRPipeline(use_gpu=False)
    rng = np.random.default_rng(42)
    noisy_frame = np.full((600, 800, 3), 128, dtype=np.uint8)
    for _ in range(40):
        x1 = int(rng.integers(20, 650))
        y1 = int(rng.integers(20, 550))
        w = int(rng.integers(60, 180))
        h = int(rng.integers(25, 70))
        cv2.putText(
            noisy_frame,
            "NOISE88",
            (x1, y1),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (20, 20, 20),
            2,
        )
        cv2.line(noisy_frame, (x1, y1), (x1 + w, y1 + h // 2), (240, 240, 240), 2)

    boxes = pipeline.detect_plates(noisy_frame)

    assert 1 <= len(boxes) <= FALLBACK_MAX_CANDIDATES
    assert all(b.source == "respaldo" for b in boxes)
    assert all(0.0 <= b.confidence <= 1.0 for b in boxes)
    scores = [b.confidence for b in boxes]
    assert scores == sorted(scores, reverse=True)


def test_detect_plates_fallback_nms_merges_overlapping_boxes() -> None:
    """Dos cajas candidatas casi iguales o muy solapadas quedan en una sola por NMS."""
    pipeline = ALPRPipeline(use_gpu=False)
    frame = np.full((320, 520, 3), 230, dtype=np.uint8)
    # Contorno externo en U que encierra sin tocar a la placa interna
    cv2.rectangle(frame, (75, 65), (81, 220), (0, 0, 0), -1)
    cv2.rectangle(frame, (354, 65), (360, 220), (0, 0, 0), -1)
    for dx in range(75, 360, 8):
        cv2.rectangle(frame, (dx, 212), (dx + 3, 220), (0, 0, 0), -1)

    cv2.rectangle(frame, (106, 80), (328, 180), (21, 204, 250), -1)
    cv2.rectangle(frame, (106, 80), (328, 180), (0, 0, 0), 3)
    cv2.putText(frame, "ABC123", (118, 148), cv2.FONT_HERSHEY_DUPLEX, 1.6, (0, 0, 0), 3, cv2.LINE_AA)

    boxes = pipeline.detect_plates(frame)

    assert len(boxes) == 1
    assert boxes[0].source == "respaldo"


def test_detect_plates_fallback_scores_clean_plate_above_noise() -> None:
    """El puntaje heurístico de una placa sintética nítida supera al de una región de ruido."""
    pipeline = ALPRPipeline(use_gpu=False)
    clean_frame = create_clean_plate_frame("ABC123")

    noise_frame = np.full((300, 500, 3), (200, 200, 200), dtype=np.uint8)
    cv2.line(noise_frame, (120, 180), (360, 100), (20, 20, 20), 2)
    cv2.line(noise_frame, (120, 100), (360, 180), (20, 20, 20), 2)

    clean_boxes = pipeline.detect_plates(clean_frame)
    noise_boxes = pipeline.detect_plates(noise_frame)

    assert clean_boxes[0].source == "respaldo"
    assert noise_boxes[0].source == "respaldo"
    assert 0.0 <= noise_boxes[0].confidence < clean_boxes[0].confidence <= 1.0


def test_detect_plates_fallback_uniform_image_returns_full_frame() -> None:
    """Una imagen sin bordes devuelve una sola caja cuadro_completo con confianza 0.0."""
    pipeline = ALPRPipeline(use_gpu=False)
    uniform_frame = np.full((300, 500, 3), 128, dtype=np.uint8)

    boxes = pipeline.detect_plates(uniform_frame)

    assert len(boxes) == 1
    assert boxes[0].source == "cuadro_completo"
    assert boxes[0].confidence == 0.0
    assert boxes[0].box_xyxy == (0, 0, 500, 300)


def _create_two_plates_frame() -> np.ndarray:
    """Crea un frame sintético con dos placas nítidas bien separadas."""
    frame = np.full((450, 700, 3), (200, 200, 200), dtype=np.uint8)
    for x1, y1, text in [(60, 60, "ABC123"), (380, 240, "XYZ789")]:
        x2, y2 = x1 + 240, y1 + 90
        frame[y1:y2, x1:x2] = (21, 204, 250)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 2)
        cv2.putText(
            frame,
            text,
            (x1 + 18, y1 + 65),
            cv2.FONT_HERSHEY_DUPLEX,
            1.7,
            (10, 10, 10),
            3,
            cv2.LINE_AA,
        )
    return frame


def test_process_frame_early_stop_on_valid_high_confidence(
    memory_db: DatabaseManager, temp_output_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """La parada temprana deja de leer cajas tras la primera placa válida con confianza >= umbral."""
    pipeline = ALPRPipeline(
        db_manager=memory_db,
        output_dir=temp_output_dir,
        save_captures=False,
        use_gpu=False,
    )
    frame = _create_two_plates_frame()
    assert len(pipeline.detect_plates(frame)) >= 2

    calls = 0

    def fake_ocr(_img: np.ndarray) -> tuple[str, float]:
        nonlocal calls
        calls += 1
        return ("ABC123", max(0.9, OCR_EARLY_STOP_CONFIDENCE))

    monkeypatch.setattr(pipeline, "extract_ocr_text", fake_ocr)
    result = pipeline.process_frame(frame, timestamp=400.0, persist_on_consensus=False)

    assert calls == 1
    assert len(result.detections) == 1
    assert result.detections[0].is_valid
    assert result.detections[0].ocr_confidence >= OCR_EARLY_STOP_CONFIDENCE


def test_process_frame_continues_when_below_early_stop_confidence(
    memory_db: DatabaseManager, temp_output_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Si la primera lectura es inválida o tiene baja confianza OCR, sigue con las demás cajas."""
    pipeline = ALPRPipeline(
        db_manager=memory_db,
        output_dir=temp_output_dir,
        save_captures=False,
        use_gpu=False,
    )
    frame = _create_two_plates_frame()
    expected_boxes = len(pipeline.detect_plates(frame))
    assert expected_boxes >= 2

    calls = 0

    def fake_low_conf_ocr(_img: np.ndarray) -> tuple[str, float]:
        nonlocal calls
        calls += 1
        return ("ABC123", max(0.0, OCR_EARLY_STOP_CONFIDENCE - 0.1))

    monkeypatch.setattr(pipeline, "extract_ocr_text", fake_low_conf_ocr)
    result = pipeline.process_frame(frame, timestamp=410.0, persist_on_consensus=False)

    assert calls == expected_boxes
    assert len(result.detections) == expected_boxes


def test_render_telemetry_labels_fallback_as_respaldo(
    memory_db: DatabaseManager, temp_output_dir: Path
) -> None:
    """En el renderizado, respaldo/cuadro_completo muestran 'respaldo' sin score; YOLO no."""
    pipeline = ALPRPipeline(
        db_manager=memory_db,
        output_dir=temp_output_dir,
        save_captures=False,
        use_gpu=False,
    )
    frame = np.zeros((300, 500, 3), dtype=np.uint8)
    latency = PipelineLatency(yolo_ms=1.0, preprocessor_ms=2.0, ocr_ms=3.0, total_ms=6.0, fps=166.7)
    detections = [
        PlateDetectionResult(
            bbox=DetectionBBox(x1=20, y1=40, x2=220, y2=120, confidence=0.51, source="respaldo"),
            raw_ocr_text="ABC123",
            validation=PlateValidator.correct_and_validate("ABC123"),
            ocr_confidence=0.88,
        ),
        PlateDetectionResult(
            bbox=DetectionBBox(x1=0, y1=0, x2=500, y2=300, confidence=0.0, source="cuadro_completo"),
            raw_ocr_text="",
            validation=PlateValidator.correct_and_validate(""),
            ocr_confidence=0.0,
        ),
        PlateDetectionResult(
            bbox=DetectionBBox(x1=30, y1=160, x2=230, y2=240, confidence=0.95, source="yolo"),
            raw_ocr_text="XYZ789",
            validation=PlateValidator.correct_and_validate("XYZ789"),
            ocr_confidence=0.91,
        ),
    ]

    with patch("src.detector.cv2.putText") as mock_put_text:
        pipeline._render_industrial_telemetry(frame, detections, latency, None)

    rendered_labels = [call.args[1] for call in mock_put_text.call_args_list]
    assert any("respaldo" in lbl and "0.51" not in lbl for lbl in rendered_labels)
    assert sum("respaldo" in lbl for lbl in rendered_labels) == 2
    assert any("XYZ789" in lbl and "respaldo" not in lbl for lbl in rendered_labels)


def test_process_frame_uses_session_voter_buffer(
    memory_db: DatabaseManager, temp_output_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verifica que la votación vaya al buffer de la sesión y no al del pipeline."""
    pipeline = ALPRPipeline(
        db_manager=memory_db,
        voter_buffer=PlateVoterBuffer(window_size=3, min_matches=2),
        output_dir=temp_output_dir,
        save_captures=False,
        use_gpu=False,
    )
    # OCR determinista: no carga EasyOCR
    monkeypatch.setattr(pipeline, "extract_ocr_text", lambda _img: ("ABC123", 0.9))
    sesion = PlateVoterBuffer(window_size=3, min_matches=2)
    frame = create_clean_plate_frame("ABC123")

    r1 = pipeline.process_frame(frame, timestamp=300.0, voter_buffer=sesion)
    r2 = pipeline.process_frame(frame, timestamp=300.1, voter_buffer=sesion)

    assert r1.consensus is None
    assert r2.consensus is not None and r2.consensus.plate == "ABC123"
    assert r2.access_record is not None
    # El buffer propio del pipeline no recibió ningún voto
    assert len(pipeline.voter_buffer._buffer) == 0
    assert len(sesion._buffer) == 2
    assert pipeline.yolo_loaded is False
