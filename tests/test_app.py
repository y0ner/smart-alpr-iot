"""Pruebas unitarias para la interfaz Streamlit y archivos de despliegue Docker."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src import __version__
from src.app import (
    INDUSTRIAL_CSS,
    compute_frame_hash,
    get_alpr_system,
    get_or_process_photo,
    get_session_voter_buffer,
    load_real_samples_truth,
    register_manual_access,
    render_header,
    run_video_stream,
    select_primary_detection,
)
from src.config import (
    BASE_DIR,
    COOLDOWN_SECONDS,
    REAL_SAMPLES_DIR,
    SCHEMA_SQL_PATH,
    VIDEO_FRAME_STRIDE,
    VOTER_MIN_MATCHES,
    VOTER_WINDOW_SIZE,
    YOLO_CONF_THRESHOLD,
)
from src.db import DatabaseManager
from src.detector import (
    ALPRPipeline,
    DetectionBBox,
    FrameProcessingResult,
    PipelineLatency,
    PlateDetectionResult,
)
from src.validator import PlateVoterBuffer, ValidationResult


def _make_fake_result(plate: str = "ABC123", valid: bool = True, access_record: dict | None = None) -> FrameProcessingResult:
    """Construye un resultado de pipeline sintético sin ejecutar YOLO ni EasyOCR."""
    validation = ValidationResult(
        is_valid=valid,
        original_text=plate,
        corrected_plate=plate,
        vehicle_type="CARRO" if valid else "DESCONOCIDO",
        corrections=[],
    )
    det = PlateDetectionResult(
        bbox=DetectionBBox(x1=10, y1=10, x2=100, y2=50, confidence=0.9),
        raw_ocr_text=plate,
        validation=validation,
        ocr_confidence=0.92,
        preprocessed=None,
    )
    return FrameProcessingResult(
        timestamp=1000.0,
        latency=PipelineLatency(yolo_ms=1.0, preprocessor_ms=2.0, ocr_ms=3.0, total_ms=6.0, fps=166.7),
        detections=[det],
        consensus=None,
        access_record=access_record,
        saved_capture_path=None,
        annotated_frame=np.zeros((60, 120, 3), dtype=np.uint8),
    )


def test_app_css_conforms_to_industrial_standard() -> None:
    """Verifica que los estilos inyectados cumplan las reglas anti-AI slop."""
    assert "#090A0F" in INDUSTRIAL_CSS  # Fondo base
    assert "#11131A" in INDUSTRIAL_CSS  # Superficie tarjetas
    assert "#1E2230" in INDUSTRIAL_CSS  # Borde sutil
    assert "#10B981" in INDUSTRIAL_CSS  # Verde esmeralda (Acento OK)
    assert "#FACC15" in INDUSTRIAL_CSS  # Amarillo placa colombiana
    assert "JetBrains Mono" in INDUSTRIAL_CSS  # Tipografía monoespaciada
    assert "Inter" in INDUSTRIAL_CSS  # Tipografía suiza UI


def test_render_header_invokes_markdown() -> None:
    """Verifica que la cabecera muestre el estado real del detector sin textos falsos."""
    with patch("streamlit.markdown") as mock_markdown:
        render_header(yolo_loaded=False)
        mock_markdown.assert_called_once()
        args, kwargs = mock_markdown.call_args
        html_fallback = args[0]
        assert "SMART ALPR IOT" in html_fallback
        assert "Detector de respaldo" in html_fallback
        assert "badge-alert" in html_fallback
        assert "ONLINE" not in html_fallback
        assert "-prod" not in html_fallback
        assert __version__ in html_fallback
        assert kwargs.get("unsafe_allow_html") is True

    with patch("streamlit.markdown") as mock_markdown:
        render_header(yolo_loaded=True)
        html_yolo = mock_markdown.call_args[0][0]
        assert "YOLO cargado" in html_yolo
        assert "badge-ok" in html_yolo
        assert "ONLINE" not in html_yolo
        assert "-prod" not in html_yolo


def test_get_alpr_system_instances(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifica que el singleton use la ruta configurada sin tocar la base real ni EasyOCR."""
    test_db_path = tmp_path / "t.db"
    monkeypatch.setattr("src.app.SQLITE_DB_PATH", test_db_path)
    get_alpr_system.clear()
    try:
        db, pipeline = get_alpr_system()
        assert isinstance(db, DatabaseManager)
        assert isinstance(pipeline, ALPRPipeline)
        assert Path(db.db_path_str) == test_db_path
        assert test_db_path.exists()
        assert pipeline.conf_threshold == YOLO_CONF_THRESHOLD
    finally:
        get_alpr_system.clear()


def test_compute_frame_hash_stable_and_sensitive() -> None:
    """Verifica que la huella SHA-1 del cuadro sea estable y cambie al alterar un píxel."""
    frame = np.zeros((24, 48, 3), dtype=np.uint8)
    h1 = compute_frame_hash(frame)
    h2 = compute_frame_hash(frame.copy())
    assert h1 == h2
    assert len(h1) == 40

    modified = frame.copy()
    modified[12, 24, 1] = 1
    assert compute_frame_hash(modified) != h1


def test_load_real_samples_truth(tmp_path: Path) -> None:
    """Verifica que verdad.csv devuelva 11 filas válidas y que cada archivo exista."""
    rows = load_real_samples_truth()
    assert len(rows) == 11
    for row in rows:
        assert row.get("archivo")
        assert row.get("placa")
        assert (REAL_SAMPLES_DIR / row["archivo"]).is_file()

    assert load_real_samples_truth(tmp_path / "inexistente.csv") == []


def test_get_session_voter_buffer_isolated_per_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verifica que cada sesión obtenga su propio PlateVoterBuffer configurado."""
    session_a: dict[str, object] = {}
    session_b: dict[str, object] = {}

    monkeypatch.setattr("src.app.st.session_state", session_a)
    buf_a1 = get_session_voter_buffer()
    buf_a2 = get_session_voter_buffer()
    assert isinstance(buf_a1, PlateVoterBuffer)
    assert buf_a1 is buf_a2
    assert buf_a1.window_size == VOTER_WINDOW_SIZE
    assert buf_a1.min_matches == VOTER_MIN_MATCHES
    assert buf_a1.cooldown_seconds == COOLDOWN_SECONDS

    monkeypatch.setattr("src.app.st.session_state", session_b)
    buf_b = get_session_voter_buffer()
    assert buf_b is not buf_a1


def test_get_or_process_photo_caches_by_hash_and_never_persists() -> None:
    """Verifica que las fotos se procesen una sola vez por huella y sin persistir."""
    fake_result = _make_fake_result("AGH430", valid=True)
    mock_pipeline = MagicMock()
    mock_pipeline.process_frame.return_value = fake_result
    voter = PlateVoterBuffer(window_size=3, min_matches=2)
    session_state: dict[str, object] = {}

    frame = np.zeros((30, 60, 3), dtype=np.uint8)
    res1, h1, processed1 = get_or_process_photo(frame, mock_pipeline, voter, session_state)
    assert processed1 is True
    assert res1 is fake_result
    mock_pipeline.process_frame.assert_called_once()
    call_kwargs = mock_pipeline.process_frame.call_args.kwargs
    assert call_kwargs["persist_on_consensus"] is False
    assert call_kwargs["voter_buffer"] is voter

    # Misma imagen en rerun posterior -> reutiliza caché sin volver a inferir
    res2, h2, processed2 = get_or_process_photo(frame.copy(), mock_pipeline, voter, session_state)
    assert processed2 is False
    assert h2 == h1
    assert res2 is fake_result
    assert mock_pipeline.process_frame.call_count == 1

    # Botón "Volver a procesar" -> fuerza nueva ejecución
    _, _, processed3 = get_or_process_photo(
        frame, mock_pipeline, voter, session_state, force_reprocess=True
    )
    assert processed3 is True
    assert mock_pipeline.process_frame.call_count == 2

    # Foto distinta -> procesa una vez nueva
    other_frame = frame.copy()
    other_frame[0, 0, 0] = 255
    _, h4, processed4 = get_or_process_photo(other_frame, mock_pipeline, voter, session_state)
    assert processed4 is True
    assert h4 != h1
    assert mock_pipeline.process_frame.call_count == 3


def test_register_manual_access_alternates_entry_and_exit() -> None:
    """Verifica el registro manual explícito y rechazo de lecturas inválidas."""
    db = DatabaseManager(db_path=Path(":memory:"))
    db.init_db(schema_path=SCHEMA_SQL_PATH)
    try:
        valid_det = _make_fake_result("UJR456", valid=True).detections[0]
        r_in = register_manual_access(db, valid_det)
        assert r_in["placa"] == "UJR456"
        assert r_in["sentido"] == "ENTRADA"

        r_out = register_manual_access(db, valid_det)
        assert r_out["sentido"] == "SALIDA"
        assert r_out["permanencia_texto"] is not None

        invalid_det = _make_fake_result("BAD", valid=False).detections[0]
        with pytest.raises(ValueError):
            register_manual_access(db, invalid_det)
    finally:
        db.close()


def test_select_primary_detection_prefers_valid() -> None:
    """Verifica la selección prioritaria de detección válida con mayor confianza."""
    invalid = _make_fake_result("XX", valid=False).detections[0]
    valid = _make_fake_result("WGL451", valid=True).detections[0]
    assert select_primary_detection([invalid, valid]) is valid
    assert select_primary_detection([]) is None


def test_run_video_stream_respects_stride_and_releases_capture() -> None:
    """Verifica muestreo por stride, persistencia por consenso y release() en finally."""
    frames = [np.full((20, 40, 3), idx, dtype=np.uint8) for idx in range(7)]
    read_calls = iter([(True, f) for f in frames] + [(False, None)])

    fake_cap = SimpleNamespace(
        read=lambda: next(read_calls, (False, None)),
        release=MagicMock(),
    )
    voter = PlateVoterBuffer(window_size=4, min_matches=2)
    mock_pipeline = MagicMock()
    mock_pipeline.process_frame.side_effect = [
        _make_fake_result("ABC123", valid=True, access_record=None),
        _make_fake_result("ABC123", valid=True, access_record={"id": 1, "placa": "ABC123", "sentido": "ENTRADA"}),
        _make_fake_result("ABC123", valid=True, access_record=None),
    ]

    last_res, events, total_frames = run_video_stream(
        cap=fake_cap,
        pipeline=mock_pipeline,
        voter_buffer=voter,
        frame_stride=VIDEO_FRAME_STRIDE,
    )

    assert total_frames == 7
    # Para 7 cuadros con stride=3 se procesan índices 0, 3 y 6 (3 llamadas)
    assert mock_pipeline.process_frame.call_count == 3
    for call in mock_pipeline.process_frame.call_args_list:
        assert call.kwargs["persist_on_consensus"] is True
        assert call.kwargs["voter_buffer"] is voter
    assert last_res is not None
    assert len(events) == 1
    assert events[0]["placa"] == "ABC123"
    fake_cap.release.assert_called_once()


def test_run_video_stream_releases_capture_on_error() -> None:
    """Verifica que release() se ejecute siempre aunque falle el procesamiento."""
    fake_cap = SimpleNamespace(
        read=lambda: (True, np.zeros((20, 40, 3), dtype=np.uint8)),
        release=MagicMock(),
    )
    mock_pipeline = MagicMock()
    mock_pipeline.process_frame.side_effect = RuntimeError("Fallo simulado")

    with pytest.raises(RuntimeError):
        run_video_stream(
            cap=fake_cap,
            pipeline=mock_pipeline,
            voter_buffer=PlateVoterBuffer(),
            frame_stride=VIDEO_FRAME_STRIDE,
        )

    fake_cap.release.assert_called_once()


def test_deployment_files_exist() -> None:
    """Verifica la existencia y contenido de Dockerfile, docker-compose.yml y run.sh."""
    dockerfile = BASE_DIR / "Dockerfile"
    docker_compose = BASE_DIR / "docker-compose.yml"
    run_sh = BASE_DIR / "run.sh"

    assert dockerfile.exists() and dockerfile.is_file()
    assert docker_compose.exists() and docker_compose.is_file()
    assert run_sh.exists() and run_sh.is_file()

    content_df = dockerfile.read_text(encoding="utf-8")
    assert "FROM python:3.12-slim" in content_df
    assert "streamlit" in content_df
    assert "8501" in content_df

    content_dc = docker_compose.read_text(encoding="utf-8")
    assert "smart_alpr_iot_console" in content_dc
    assert "8501:8501" in content_dc

    content_run = run_sh.read_text(encoding="utf-8")
    assert "streamlit run src/app.py" in content_run
