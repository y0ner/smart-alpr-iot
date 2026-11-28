"""Centro de Control y Monitoreo ALPR de Grado Industrial (Smart ALPR IoT).

Implementa la consola SCADA de misión crítica, telemetría de visión artificial,
auditoría de base de datos relacional y laboratorio del Reto Extra A/B.
Diseñado conforme al estándar anti-AI slop (.agents/rules/industrial-ui-standard.md).
"""

import csv
import hashlib
import sys
import tempfile
from collections.abc import Callable, MutableMapping
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from numpy.typing import NDArray

# Asegurar importación de módulos del proyecto
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src import __version__
from src.benchmark_ab import BenchmarkEvaluator, create_synthetic_benchmark_dataset
from src.config import (
    COOLDOWN_SECONDS,
    DEFAULT_CAMERA_SOURCE,
    REAL_SAMPLES_DIR,
    SAMPLES_GROUND_TRUTH_CSV,
    SCHEMA_SQL_PATH,
    SQLITE_DB_PATH,
    VIDEO_FRAME_STRIDE,
    VOTER_MIN_MATCHES,
    VOTER_WINDOW_SIZE,
    YOLO_CONF_THRESHOLD,
)
from src.db import DatabaseManager, VehicleType
from src.detector import ALPRPipeline, FrameProcessingResult, PlateDetectionResult
from src.math_proof import (
    DEFAULT_PATCH_3X3,
    SOBEL_GX_KERNEL,
    generate_markdown_report,
    run_math_proof,
)
from src.validator import PlateVoterBuffer

# -----------------------------------------------------------------------------
# INYECCIÓN DE ESTILOS INDUSTRIALES ANTI-AI SLOP (SCADA / TELEMETRÍA VIAL)
# -----------------------------------------------------------------------------

INDUSTRIAL_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    color: #F3F4F6;
    background-color: #090A0F;
}

/* Ocultar elementos decorativos innecesarios de Streamlit */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* Contenedor principal sin márgenes inflados */
.block-container {
    padding-top: 1.5rem;
    padding-bottom: 2rem;
    padding-left: 2rem;
    padding-right: 2rem;
    max-width: 1400px;
}

/* Sidebar Industrial */
section[data-testid="stSidebar"] {
    background-color: #11131A !important;
    border-right: 1px solid #1E2230 !important;
}

/* Tarjetas Técnicas Sobrias (contenedores Streamlit con borde) */
div[data-testid="stVerticalBlockBorderWrapper"],
div[data-testid="stVerticalBlock"][class*="st-key-card_"] {
    background-color: #11131A !important;
    border: 1px solid #1E2230 !important;
    border-radius: 6px !important;
    padding: 16px !important;
}

.industrial-card-header {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.8rem;
    color: #9CA3AF;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 8px;
    border-bottom: 1px solid #1E2230;
    padding-bottom: 6px;
}

/* Placa Vehicular Colombiana Realista */
.colombian-plate-card {
    background-color: #FACC15;
    border: 3px solid #11131A;
    border-radius: 6px;
    padding: 8px 16px;
    text-align: center;
    box-shadow: 0 2px 4px rgba(0,0,0,0.4);
    display: inline-block;
    min-width: 200px;
}
.colombian-plate-text {
    font-family: 'JetBrains Mono', monospace;
    font-size: 2.2rem;
    font-weight: 700;
    color: #11131A;
    letter-spacing: 4px;
    line-height: 1.1;
    margin: 0;
}
.colombian-plate-country {
    font-family: 'Inter', sans-serif;
    font-size: 0.65rem;
    font-weight: 700;
    color: #1F2937;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-top: 2px;
}

/* Badges de Telemetría Funcional */
.badge-ok {
    background: rgba(16, 185, 129, 0.12);
    color: #10B981;
    border: 1px solid rgba(16, 185, 129, 0.3);
    padding: 4px 10px;
    border-radius: 4px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.75rem;
    font-weight: 600;
    display: inline-block;
}
.badge-alert {
    background: rgba(245, 158, 11, 0.12);
    color: #F59E0B;
    border: 1px solid rgba(245, 158, 11, 0.3);
    padding: 4px 10px;
    border-radius: 4px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.75rem;
    font-weight: 600;
    display: inline-block;
}
.badge-danger {
    background: rgba(239, 68, 68, 0.12);
    color: #EF4444;
    border: 1px solid rgba(239, 68, 68, 0.3);
    padding: 4px 10px;
    border-radius: 4px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.75rem;
    font-weight: 600;
    display: inline-block;
}

/* Tipografía de Métricas */
.metric-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.8rem;
    font-weight: 700;
    color: #F3F4F6;
}
.metric-label {
    font-family: 'Inter', sans-serif;
    font-size: 0.8rem;
    color: #9CA3AF;
    text-transform: uppercase;
}
</style>
"""


# -----------------------------------------------------------------------------
# FUNCIONES PURAS Y GESTIÓN DE ESTADO POR SESIÓN
# -----------------------------------------------------------------------------


def compute_frame_hash(frame: NDArray[np.uint8]) -> str:
    """Calcula una huella SHA-1 determinista a partir de los píxeles del cuadro."""
    return hashlib.sha1(frame.tobytes()).hexdigest()


def load_real_samples_truth(
    csv_path: Path = SAMPLES_GROUND_TRUTH_CSV,
) -> list[dict[str, str]]:
    """Lee la verdad de referencia de las muestras reales desde verdad.csv."""
    if not csv_path.exists():
        return []
    with csv_path.open(encoding="utf-8", newline="") as csv_file:
        return [dict(row) for row in csv.DictReader(csv_file)]


def build_synthetic_sample_frame(plate_code: str) -> NDArray[np.uint8]:
    """Genera una muestra sintética reproducible de carril vehicular."""
    canvas = np.full((360, 640, 3), (180, 180, 185), dtype=np.uint8)
    x1, y1, x2, y2 = 200, 140, 440, 220
    canvas[y1:y2, x1:x2] = (21, 204, 250)
    cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 0, 0), 3)
    cv2.putText(
        canvas,
        plate_code,
        (x1 + 18, y1 + 58),
        cv2.FONT_HERSHEY_DUPLEX,
        1.7,
        (10, 10, 10),
        3,
        cv2.LINE_AA,
    )
    cv2.circle(canvas, (x1 + 12, y1 + 12), 3, (30, 30, 30), -1)
    cv2.circle(canvas, (x2 - 12, y1 + 12), 3, (30, 30, 30), -1)
    return canvas


def select_primary_detection(
    detections: list[PlateDetectionResult],
) -> PlateDetectionResult | None:
    """Selecciona la detección principal priorizando placas válidas por confianza OCR."""
    if not detections:
        return None
    valid = [det for det in detections if det.is_valid]
    if valid:
        return max(valid, key=lambda det: det.ocr_confidence)
    return max(detections, key=lambda det: det.ocr_confidence)


def get_session_voter_buffer() -> PlateVoterBuffer:
    """Obtiene o crea el buffer de votación temporal exclusivo de la sesión actual."""
    if "voter_buffer" not in st.session_state:
        st.session_state["voter_buffer"] = PlateVoterBuffer(
            window_size=VOTER_WINDOW_SIZE,
            min_matches=VOTER_MIN_MATCHES,
            cooldown_seconds=COOLDOWN_SECONDS,
        )
    return st.session_state["voter_buffer"]


def get_or_process_photo(
    frame: NDArray[np.uint8],
    pipeline: ALPRPipeline,
    voter_buffer: PlateVoterBuffer,
    session_state: MutableMapping[str, Any],
    force_reprocess: bool = False,
) -> tuple[FrameProcessingResult, str, bool]:
    """Procesa una foto solo si cambió su huella o si se solicitó reprocesar."""
    frame_hash = compute_frame_hash(frame)
    cached_hash = session_state.get("photo_frame_hash")
    cached_result = session_state.get("photo_result")

    if not force_reprocess and cached_hash == frame_hash and cached_result is not None:
        return cached_result, frame_hash, False

    result = pipeline.process_frame(
        frame,
        persist_on_consensus=False,
        voter_buffer=voter_buffer,
    )
    session_state["photo_frame_hash"] = frame_hash
    session_state["photo_result"] = result
    session_state["photo_access_record"] = None
    return result, frame_hash, True


def register_manual_access(
    db: DatabaseManager,
    detection: PlateDetectionResult,
) -> dict[str, Any]:
    """Persiste un acceso manualmente en SQLite tras validar la matrícula."""
    if not detection.is_valid or detection.vehicle_type not in ("CARRO", "MOTO"):
        raise ValueError("Solo se pueden registrar placas válidas de CARRO o MOTO.")
    vehicle_type: VehicleType = "CARRO" if detection.vehicle_type == "CARRO" else "MOTO"
    return db.registrar_acceso(
        placa=detection.plate,
        tipo=vehicle_type,
        confianza=detection.ocr_confidence,
    )


def run_video_stream(
    cap: Any,
    pipeline: ALPRPipeline,
    voter_buffer: PlateVoterBuffer,
    frame_stride: int = VIDEO_FRAME_STRIDE,
    on_processed_frame: Callable[[FrameProcessingResult, list[dict[str, Any]], int], None]
    | None = None,
    max_frames: int | None = None,
) -> tuple[FrameProcessingResult | None, list[dict[str, Any]], int]:
    """Recorre una captura OpenCV procesando 1 de cada N cuadros y libera en finally."""
    stride = max(1, frame_stride)
    frame_idx = 0
    last_result: FrameProcessingResult | None = None
    consensus_events: list[dict[str, Any]] = []

    try:
        while max_frames is None or frame_idx < max_frames:
            ok, frame = cap.read()
            if not ok or frame is None:
                break

            if frame_idx % stride == 0:
                res = pipeline.process_frame(
                    frame,
                    persist_on_consensus=True,
                    voter_buffer=voter_buffer,
                )
                last_result = res
                if res.access_record is not None:
                    consensus_events.append(res.access_record)
                if on_processed_frame is not None:
                    on_processed_frame(res, consensus_events, frame_idx)

            frame_idx += 1
    finally:
        cap.release()

    return last_result, consensus_events, frame_idx


@st.cache_resource
def get_alpr_system() -> tuple[DatabaseManager, ALPRPipeline]:
    """Inicializa en memoria compartida el gestor de base de datos y el pipeline ALPR."""
    db = DatabaseManager(db_path=SQLITE_DB_PATH)
    # El esquema usa IF NOT EXISTS: inicializar siempre cubre también un archivo vacío
    db.init_db(schema_path=SCHEMA_SQL_PATH)

    pipeline = ALPRPipeline(
        conf_threshold=YOLO_CONF_THRESHOLD,
        use_gpu=False,
        db_manager=db,
        save_captures=True,
    )
    return db, pipeline


def render_header(yolo_loaded: bool) -> None:
    """Renderiza el encabezado industrial superior con el estado real del detector."""
    if yolo_loaded:
        detector_badge = '<span class="badge-ok">YOLO cargado</span>'
    else:
        detector_badge = (
            '<span class="badge-alert">Detector de respaldo (sin pesos YOLO)</span>'
        )

    st.markdown(
        f"""
        <div style="background-color: #11131A; border: 1px solid #1E2230; padding: 14px 20px; border-radius: 6px; margin-bottom: 18px; display: flex; justify-content: space-between; align-items: center;">
            <div>
                <h1 style="color: #F3F4F6; font-size: 1.15rem; font-weight: 600; margin: 0; font-family: 'Inter', sans-serif;">
                    SMART ALPR IOT &bull; CONSOLA DE TELEMETRÍA Y ACCESO VIAL
                </h1>
                <p style="color: #9CA3AF; margin: 3px 0 0 0; font-family: 'JetBrains Mono', monospace; font-size: 0.78rem;">
                    Terminal Perimetral: Riohacha (La Guajira) &bull; Centro Comercial Viva Wajiira / Uniguajira
                </p>
            </div>
            <div style="text-align: right;">
                {detector_badge}
                <span style="color: #6B7280; font-family: 'JetBrains Mono', monospace; font-size: 0.75rem; margin-left: 10px;">
                    v{__version__}
                </span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# -----------------------------------------------------------------------------
# VISTA 1: CONSOLA DE OPERACIÓN EN VIVO (CÁMARA / VIDEO / IMAGEN)
# -----------------------------------------------------------------------------


def _render_events_table(
    events_slot: Any,
    events: list[dict[str, Any]],
    empty_message: str,
) -> None:
    """Renderiza la tabla de accesos registrados por consenso en un st.empty()."""
    if not events:
        events_slot.caption(empty_message)
        return

    df_events = pd.DataFrame(
        [
            {
                "ID": ev["id"],
                "Placa": ev["placa"],
                "Tipo": ev["tipo"],
                "Sentido": ev["sentido"],
                "Confianza": round(float(ev["confianza"]), 4),
                "Permanencia": ev.get("permanencia_texto") or "-",
                "Momento": ev["momento"],
            }
            for ev in reversed(events)
        ]
    )
    events_slot.dataframe(df_events, width="stretch", hide_index=True)


def _render_stream_telemetry(
    res: FrameProcessingResult,
    events: list[dict[str, Any]],
    frame_idx: int,
    yolo_loaded: bool,
    frame_slot: Any,
    reading_slot: Any,
    latency_slot: Any,
    events_slot: Any,
) -> None:
    """Actualiza los marcadores st.empty() durante la reproducción de video o cámara."""
    detector_tag = "YOLO" if yolo_loaded else "respaldo por bordes"
    frame_slot.image(
        res.annotated_frame,
        channels="BGR",
        width="stretch",
        caption=f"Cuadro #{frame_idx} | Detector: {detector_tag}",
    )

    primary_det = select_primary_detection(res.detections)
    if primary_det is not None:
        validity = "FORMATO VÁLIDO" if primary_det.is_valid else "FORMATO INVÁLIDO"
        reading_slot.markdown(
            f"**Última lectura:** `{primary_det.plate}` ({primary_det.vehicle_type}) "
            f"| **Confianza OCR:** `{primary_det.ocr_confidence:.2%}` | `{validity}`"
        )
    else:
        reading_slot.info("Última lectura: sin matrícula detectada en el cuadro.")

    latency_slot.markdown(
        f"**Latencia medida:** `{res.latency.total_ms:.1f} ms` (`{res.latency.fps:.1f} FPS`) "
        f"| Detección: `{res.latency.yolo_ms:.1f} ms` "
        f"| Preprocesamiento: `{res.latency.preprocessor_ms:.1f} ms` "
        f"| OCR: `{res.latency.ocr_ms:.1f} ms`"
    )

    _render_events_table(
        events_slot,
        events,
        "Sin accesos registrados por consenso en esta secuencia.",
    )


def _render_photo_results(
    selected_frame: NDArray[np.uint8],
    pipeline: ALPRPipeline,
    db: DatabaseManager,
    voter_buffer: PlateVoterBuffer,
    expected_plate: str | None,
) -> None:
    """Ejecuta o reutiliza la inferencia de una fotografía y habilita el registro manual."""
    force_reprocess = st.button("Volver a procesar", key="btn_reprocess_photo")
    res, frame_hash, _ = get_or_process_photo(
        frame=selected_frame,
        pipeline=pipeline,
        voter_buffer=voter_buffer,
        session_state=st.session_state,
        force_reprocess=force_reprocess,
    )

    primary_det = select_primary_detection(res.detections)
    detector_caption = (
        "Detección YOLO + Telemetría de Latencias"
        if pipeline.yolo_loaded
        else "Detección por respaldo morfológico (sin pesos YOLO) + Telemetría de Latencias"
    )

    # 1. Visor Principal con Overlays
    with st.container(border=True, key="card_visor"):
        st.markdown(
            '<div class="industrial-card-header">Visor Perimetral con Telemetría Activa</div>',
            unsafe_allow_html=True,
        )
        st.image(
            res.annotated_frame,
            channels="BGR",
            width="stretch",
            caption=detector_caption,
        )

    # 2. Tarjeta de Estado de Acceso y Validación Colombiana
    with st.container(border=True, key="card_validacion"):
        st.markdown(
            '<div class="industrial-card-header">Estado de Acceso & Validación de Placa</div>',
            unsafe_allow_html=True,
        )

        col_plate_render, col_plate_info = st.columns([1.2, 1.8])

        with col_plate_render:
            if primary_det is not None:
                st.markdown(
                    f"""
                    <div class="colombian-plate-card">
                        <div class="colombian-plate-text">{primary_det.plate}</div>
                        <div class="colombian-plate-country">COLOMBIA &bull; {primary_det.vehicle_type}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.info("No se localizó matrícula vehicular en el cuadro actual.")

        with col_plate_info:
            if expected_plate is not None:
                matches_truth = (
                    primary_det is not None and primary_det.plate == expected_plate
                )
                truth_badge = (
                    '<span class="badge-ok">COINCIDE</span>'
                    if matches_truth
                    else '<span class="badge-danger">NO COINCIDE</span>'
                )
                st.markdown(
                    f"**Placa de referencia:** `{expected_plate}` | {truth_badge}",
                    unsafe_allow_html=True,
                )

            if primary_det is not None:
                status_badge = (
                    '<span class="badge-ok">FORMATO OFICIAL VÁLIDO</span>'
                    if primary_det.is_valid
                    else '<span class="badge-danger">FORMATO INVÁLIDO</span>'
                )
                st.markdown(f"**Validación:** {status_badge}", unsafe_allow_html=True)
                st.markdown(
                    f"**Tipo Detectado:** `{primary_det.vehicle_type}` | "
                    f"**Confianza OCR:** `{primary_det.ocr_confidence:.2%}` | "
                    f"**Latencia:** `{res.latency.total_ms:.1f} ms`"
                )
                if primary_det.validation.corrections:
                    st.markdown(
                        f"**Corrección Posicional:** `{', '.join(primary_det.validation.corrections)}`"
                    )

                if primary_det.is_valid:
                    if st.button(
                        "Registrar acceso",
                        key=f"btn_register_{frame_hash}",
                        type="primary",
                    ):
                        saved_record = register_manual_access(db, primary_det)
                        st.session_state["photo_access_record"] = {
                            "frame_hash": frame_hash,
                            "record": saved_record,
                        }

                    stored_access = st.session_state.get("photo_access_record")
                    if stored_access and stored_access.get("frame_hash") == frame_hash:
                        rec = stored_access["record"]
                        st.markdown(
                            f"**Transacción SQLite:** `{rec['sentido']}` (ID: `{rec['id']}`)"
                        )
                        if rec.get("permanencia_texto"):
                            st.markdown(
                                f"**Permanencia Registrada:** `{rec['permanencia_texto']}`"
                            )

    # 3. Inspector de Visión Artificial: Tira Horizontal de 5 Fases Morfológicas
    if primary_det is not None and primary_det.preprocessed is not None:
        prep = primary_det.preprocessed
        with st.container(border=True, key="card_morfologia"):
            st.markdown(
                '<div class="industrial-card-header">Inspector de Visión Artificial: Pipeline Morfológico (Demostración de Técnicas del Curso)</div>',
                unsafe_allow_html=True,
            )

            f1, f2, f3, f4, f5 = st.columns(5)
            with f1:
                st.caption("1. Recorte con margen")
                st.image(prep.original_crop, channels="BGR", width="stretch")
            with f2:
                st.caption("2. Escala de grises")
                st.image(prep.gray, width="stretch")
            with f3:
                st.caption("3. Filtro Bilateral")
                st.image(prep.filtered, width="stretch")
            with f4:
                st.caption("4. Umbral Adaptativo")
                st.image(prep.binary_adaptive, width="stretch")
            with f5:
                st.caption("5. Morfología Final")
                st.image(prep.morph, width="stretch")


def _render_video_file_mode(
    uploaded_video: Any,
    pipeline: ALPRPipeline,
    voter_buffer: PlateVoterBuffer,
) -> None:
    """Procesa un archivo de video subido guardándolo en temporal y recorriéndolo por stride."""
    with st.container(border=True, key="card_visor_video"):
        st.markdown(
            '<div class="industrial-card-header">Visor de Video en Secuencia</div>',
            unsafe_allow_html=True,
        )
        frame_slot = st.empty()

    with st.container(border=True, key="card_telemetria_video"):
        st.markdown(
            '<div class="industrial-card-header">Telemetría y Accesos Registrados por Consenso</div>',
            unsafe_allow_html=True,
        )
        reading_slot = st.empty()
        latency_slot = st.empty()
        events_slot = st.empty()

    if uploaded_video is None:
        frame_slot.info(
            "Suba un archivo de video (MP4, AVI o MOV) para procesar cuadros con votación temporal."
        )
        return

    video_bytes = uploaded_video.getvalue()
    video_hash = hashlib.sha1(video_bytes).hexdigest()
    reprocess = st.button("Volver a procesar video", key="btn_reprocess_video")
    cached_summary = st.session_state.get("video_summary")

    if reprocess or st.session_state.get("video_hash") != video_hash or cached_summary is None:
        suffix = Path(uploaded_video.name).suffix or ".mp4"
        tmp_path: Path | None = None
        cap: cv2.VideoCapture | None = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
                tmp_file.write(video_bytes)
                tmp_path = Path(tmp_file.name)

            cap = cv2.VideoCapture(str(tmp_path))
            if not cap.isOpened():
                st.error("No se pudo abrir el archivo de video proporcionado.")
                return

            def _on_frame(
                res: FrameProcessingResult,
                events: list[dict[str, Any]],
                frame_idx: int,
            ) -> None:
                _render_stream_telemetry(
                    res=res,
                    events=events,
                    frame_idx=frame_idx,
                    yolo_loaded=pipeline.yolo_loaded,
                    frame_slot=frame_slot,
                    reading_slot=reading_slot,
                    latency_slot=latency_slot,
                    events_slot=events_slot,
                )

            last_res, events, total_frames = run_video_stream(
                cap=cap,
                pipeline=pipeline,
                voter_buffer=voter_buffer,
                frame_stride=VIDEO_FRAME_STRIDE,
                on_processed_frame=_on_frame,
            )
            st.session_state["video_hash"] = video_hash
            st.session_state["video_summary"] = {
                "last_res": last_res,
                "events": events,
                "total_frames": total_frames,
            }
        finally:
            if cap is not None:
                cap.release()
            if tmp_path is not None:
                tmp_path.unlink(missing_ok=True)
        cached_summary = st.session_state.get("video_summary")

    if cached_summary and cached_summary.get("last_res") is not None:
        _render_stream_telemetry(
            res=cached_summary["last_res"],
            events=cached_summary["events"],
            frame_idx=max(0, cached_summary["total_frames"] - 1),
            yolo_loaded=pipeline.yolo_loaded,
            frame_slot=frame_slot,
            reading_slot=reading_slot,
            latency_slot=latency_slot,
            events_slot=events_slot,
        )
        st.caption(
            f"Video procesado: {cached_summary['total_frames']} cuadros leídos "
            f"(1 de cada {VIDEO_FRAME_STRIDE} analizado)."
        )
    else:
        frame_slot.warning("El archivo de video no entregó cuadros válidos.")


def _render_live_camera_mode(
    streaming: bool,
    pipeline: ALPRPipeline,
    voter_buffer: PlateVoterBuffer,
) -> None:
    """Lee cuadros en vivo desde la cámara local mientras el toggle esté activo."""
    with st.container(border=True, key="card_visor_camara"):
        st.markdown(
            '<div class="industrial-card-header">Visor Perimetral de Cámara Local en Vivo</div>',
            unsafe_allow_html=True,
        )
        frame_slot = st.empty()

    with st.container(border=True, key="card_telemetria_camara"):
        st.markdown(
            '<div class="industrial-card-header">Lectura en Vivo, Latencia y Accesos por Consenso</div>',
            unsafe_allow_html=True,
        )
        reading_slot = st.empty()
        latency_slot = st.empty()
        events_slot = st.empty()

    live_events: list[dict[str, Any]] = st.session_state.setdefault("live_events", [])

    if not streaming:
        frame_slot.info(
            "Active 'Transmitir' en el panel izquierdo para iniciar la lectura en vivo."
        )
        _render_events_table(
            events_slot,
            live_events,
            "Sin accesos registrados por consenso en esta sesión.",
        )
        return

    cap = cv2.VideoCapture(DEFAULT_CAMERA_SOURCE)
    try:
        if not cap.isOpened():
            frame_slot.error(
                f"No se pudo abrir la cámara local (fuente {DEFAULT_CAMERA_SOURCE}). "
                "Verifique el dispositivo conectado; en contenedores Docker puede no haber cámara."
            )
            return

        def _on_live_frame(
            res: FrameProcessingResult,
            _stream_events: list[dict[str, Any]],
            frame_idx: int,
        ) -> None:
            if res.access_record is not None:
                live_events.append(res.access_record)
            _render_stream_telemetry(
                res=res,
                events=live_events,
                frame_idx=frame_idx,
                yolo_loaded=pipeline.yolo_loaded,
                frame_slot=frame_slot,
                reading_slot=reading_slot,
                latency_slot=latency_slot,
                events_slot=events_slot,
            )

        run_video_stream(
            cap=cap,
            pipeline=pipeline,
            voter_buffer=voter_buffer,
            frame_stride=VIDEO_FRAME_STRIDE,
            on_processed_frame=_on_live_frame,
        )
    finally:
        cap.release()


def view_live_operation(
    pipeline: ALPRPipeline,
    db: DatabaseManager | None = None,
) -> None:
    """Consola principal de monitoreo en tiempo real con visor de 5 fases morfológicas."""
    st.markdown("### Consola de Monitoreo e Inferencia en Tiempo Real")
    db_manager = db or pipeline.db_manager
    voter_buffer = get_session_voter_buffer()

    col_ctrl, col_main = st.columns([1, 2.5])

    source_options = [
        "Muestras sintéticas (1-Click)",
        "Placas reales",
        "Cargar imagen desde archivo",
        "Foto con la cámara del navegador",
        "Video (archivo)",
        "Cámara local en vivo",
    ]

    with col_ctrl, st.container(border=True, key="card_ingesta"):
        st.markdown(
            '<div class="industrial-card-header">Fuente de Video / Ingesta</div>',
            unsafe_allow_html=True,
        )

        source_type = st.radio(
            "Seleccionar Entrada:",
            source_options,
            index=0,
        )

        selected_frame: NDArray[np.uint8] | None = None
        expected_plate: str | None = None
        uploaded_video: Any = None
        live_streaming = False

        if source_type == "Muestras sintéticas (1-Click)":
            sample_choice = st.selectbox(
                "Seleccionar Muestra Sintética:",
                [
                    "Carro Particular (UJR456 - Sombra Cenital)",
                    "Carro Particular (VIV100 - Borde Nítido)",
                    "Moto Transporte (WJR78A - Formato Oficial)",
                    "Carro Confuso (KLM9O1 - Corrección O->0)",
                ],
            )
            plate_code = sample_choice.split("(")[1].split(" ")[0]
            selected_frame = build_synthetic_sample_frame(plate_code)

        elif source_type == "Placas reales":
            truth_rows = load_real_samples_truth()
            available_rows = [
                row
                for row in truth_rows
                if row.get("archivo") and (REAL_SAMPLES_DIR / row["archivo"]).is_file()
            ]
            if available_rows:
                files = [row["archivo"] for row in available_rows]
                rows_by_file = {row["archivo"]: row for row in available_rows}
                file_choice = st.selectbox(
                    "Seleccionar Placa Real:",
                    files,
                    index=0,
                )
                truth_row = rows_by_file[file_choice]
                expected_plate = truth_row.get("placa")
                selected_frame = cv2.imread(str(REAL_SAMPLES_DIR / file_choice))
                st.caption(f"Ruta: `data/samples/placas_de_prueba/{file_choice}`")
                st.markdown(
                    f"**Referencia:** `{expected_plate}` "
                    f"({truth_row.get('tipo', '-')}, {truth_row.get('servicio', '-')})"
                )
                if truth_row.get("nota"):
                    st.caption(f"Nota: {truth_row['nota']}")
            else:
                st.warning("No se encontraron placas reales en `data/samples/placas_de_prueba/`.")

        elif source_type == "Cargar imagen desde archivo":
            st.caption(
                "Sugerencia: puedes seleccionar cualquier imagen de `data/samples/placas_de_prueba/`."
            )
            uploaded = st.file_uploader(
                "Subir archivo de imagen (JPG/PNG)",
                type=["jpg", "jpeg", "png"],
            )
            if uploaded is not None:
                file_bytes = np.frombuffer(uploaded.getvalue(), dtype=np.uint8)
                selected_frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        elif source_type == "Foto con la cámara del navegador":
            cam_picture = st.camera_input("Capturar cuadro desde cámara del navegador")
            if cam_picture is not None:
                file_bytes = np.frombuffer(cam_picture.getvalue(), dtype=np.uint8)
                selected_frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        elif source_type == "Video (archivo)":
            uploaded_video = st.file_uploader(
                "Subir archivo de video (MP4/AVI/MOV)",
                type=["mp4", "avi", "mov"],
            )
            st.caption(
                f"Análisis por consenso: 1 de cada {VIDEO_FRAME_STRIDE} cuadros "
                f"(ventana {VOTER_WINDOW_SIZE}, mínimo {VOTER_MIN_MATCHES} votos)."
            )

        elif source_type == "Cámara local en vivo":
            live_streaming = st.toggle("Transmitir", value=False, key="toggle_live_camera")
            st.caption(
                f"Fuente local: `{DEFAULT_CAMERA_SOURCE}` | "
                f"Muestreo: 1 de cada {VIDEO_FRAME_STRIDE} cuadros."
            )

    with col_main:
        if source_type == "Video (archivo)":
            _render_video_file_mode(uploaded_video, pipeline, voter_buffer)
        elif source_type == "Cámara local en vivo":
            _render_live_camera_mode(live_streaming, pipeline, voter_buffer)
        elif selected_frame is not None:
            _render_photo_results(
                selected_frame=selected_frame,
                pipeline=pipeline,
                db=db_manager,
                voter_buffer=voter_buffer,
                expected_plate=expected_plate,
            )
        else:
            st.info("Seleccione o cargue una fuente de imagen para inicializar la telemetría.")


# -----------------------------------------------------------------------------
# VISTA 2: AUDITORÍA Y CONTROL DE PARQUEADERO (REPORTE CLIENTE)
# -----------------------------------------------------------------------------


def view_database_audit(db: DatabaseManager) -> None:
    """Dashboard transaccional para administradores del estacionamiento."""
    st.markdown("### Auditoría Transaccional y Control de Estacionamiento")

    # KPIs Superiores
    metricas = db.get_metricas_dashboard()
    k1, k2, k3, k4 = st.columns(4)
    kpi_items = [
        (k1, "card_kpi_ocupacion", f"{metricas['ocupacion_total']}", "Vehículos en Sitio"),
        (
            k2,
            "card_kpi_tipos",
            f"{metricas['carros_dentro']} / {metricas['motos_dentro']}",
            "Carros / Motos",
        ),
        (k3, "card_kpi_hoy", f"{metricas['total_accesos_hoy']}", "Accesos de Hoy"),
        (
            k4,
            "card_kpi_confianza",
            f"{metricas['confianza_promedio'] * 100:.1f}%",
            "Confianza Media",
        ),
    ]
    for col, key, value, label in kpi_items:
        with col, st.container(border=True, key=key):
            st.markdown(
                f'<div class="metric-value">{value}</div><div class="metric-label">{label}</div>',
                unsafe_allow_html=True,
            )

    # Filtros y Tabla de Auditoría
    with st.container(border=True, key="card_historial"):
        st.markdown(
            '<div class="industrial-card-header">Historial de Accesos Registrados (SQLite)</div>',
            unsafe_allow_html=True,
        )

        f_col1, f_col2, f_col3 = st.columns([1.5, 1, 1])
        with f_col1:
            filtro_placa = st.text_input("Filtrar por Placa (Búsqueda exacta):", "").strip().upper()
        with f_col2:
            limite = st.selectbox("Límite de registros:", [25, 50, 100], index=1)
        with f_col3:
            if st.button("Actualizar Tabla"):
                st.rerun()

        registros = db.get_historial_accesos(
            limite=limite, placa=filtro_placa if filtro_placa else None
        )

        if registros:
            df = pd.DataFrame(registros)
            st.dataframe(
                df[
                    [
                        "id",
                        "placa",
                        "tipo",
                        "sentido",
                        "confianza",
                        "momento",
                        "ruta_captura",
                    ]
                ],
                width="stretch",
                hide_index=True,
            )
        else:
            st.warning(
                "No se encontraron registros de accesos con los criterios de búsqueda especificados."
            )

    # Flujo por Hora
    with st.container(border=True, key="card_flujo"):
        st.markdown(
            '<div class="industrial-card-header">Volumen de Flujo por Hora (Hoy)</div>',
            unsafe_allow_html=True,
        )
        flujo = db.get_flujo_por_hora()
        if flujo:
            df_flujo = pd.DataFrame(flujo)
            st.bar_chart(df_flujo.set_index("hora")[["entradas", "salidas"]])
        else:
            st.info("Sin movimientos registrados para el día actual.")


# -----------------------------------------------------------------------------
# VISTA 3: LABORATORIO TÉCNICO Y RETO EXTRA A/B (CON VS. SIN PREPROCESAMIENTO)
# -----------------------------------------------------------------------------


def view_technical_lab() -> None:
    """Laboratorio para demostración del Reto Extra (+0.5 pts) y convolución matemática 3x3."""
    st.markdown("### Laboratorio de Visión Artificial: Reto Extra y Análisis Teórico")

    tab_ab, tab_math = st.tabs(
        [
            "Evaluador A/B (Lote Sintético)",
            "Demostración Matemática 3x3 (Sección C.2)",
        ]
    )

    with tab_ab, st.container(border=True, key="card_benchmark_ab"):
        st.markdown(
            '<div class="industrial-card-header">Comparativa Experimental sobre Lote Sintético: Modo A (Crudo) vs. Modo B (Pipeline OpenCV)</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            """
            Este experimento ejecuta **EasyOCR** sobre un **lote sintético** de placas generadas bajo dos condiciones:
            - **Modo A (Sin Preprocesamiento):** La caja detectada se entrega directamente al OCR.
            - **Modo B (Con Preprocesamiento):** Se aplica la tira morfológica completa (Grises, Bilateral, Binarización y Apertura).
            """
        )

        num_samples = st.slider("Cantidad de placas sintéticas a evaluar:", 2, 12, 6)

        if st.button("Ejecutar Benchmark Sintético A/B", type="primary"):
            with st.spinner("Ejecutando inferencia dual sobre lote sintético..."):
                dataset = create_synthetic_benchmark_dataset(num_samples=num_samples)
                evaluator = BenchmarkEvaluator(use_gpu=False)
                report = evaluator.evaluate_dataset(dataset)

                st.markdown(report.to_markdown())

    with tab_math, st.container(border=True, key="card_math_proof"):
        st.markdown(
            '<div class="industrial-card-header">Cálculo Analítico Paso a Paso: Parche 3x3 vs. cv2.filter2D</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            """
            Demostración requerida por la **Sección C.2 de la Guía Oficial**:
            Demuestra que la convolución analítica realizada a mano con aritmética pura de Python
            coincide con **exactitud absoluta (diferencia numérica cero)** con la función nativa `cv2.filter2D`.
            """
        )

        proof = run_math_proof(
            patch=DEFAULT_PATCH_3X3,
            kernel=SOBEL_GX_KERNEL,
            kernel_name="Sobel Horizontal (Gx)",
        )
        st.markdown(generate_markdown_report(proof))


# -----------------------------------------------------------------------------
# PUNTO DE ENTRADA PRINCIPAL (MAIN)
# -----------------------------------------------------------------------------


def main() -> dict[str, Any]:
    """Punto de ejecución de la aplicación Streamlit."""
    st.set_page_config(
        page_title="Smart ALPR IoT | Telemetría Vial",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(INDUSTRIAL_CSS, unsafe_allow_html=True)

    db, pipeline = get_alpr_system()
    render_header(yolo_loaded=pipeline.yolo_loaded)

    detector_status = (
        "Detector: YOLO"
        if pipeline.yolo_loaded
        else "Detector: respaldo por bordes (sin pesos YOLO)"
    )

    with st.sidebar:
        st.markdown(
            """
            <div style="padding: 10px 0 16px 0; border-bottom: 1px solid #1E2230; margin-bottom: 14px;">
                <span style="font-family: 'JetBrains Mono', monospace; font-size: 0.72rem; color: #9CA3AF; text-transform: uppercase;">
                    Panel de Control SCADA
                </span>
                <h3 style="color: #F3F4F6; margin: 4px 0 0 0; font-size: 1.05rem;">Smart ALPR IoT</h3>
            </div>
            """,
            unsafe_allow_html=True,
        )

        nav_choice = st.radio(
            "Seleccionar Módulo Operativo:",
            [
                "01. Consola de Operación en Vivo",
                "02. Control de Parqueadero (BD)",
                "03. Laboratorio Reto Extra A/B",
            ],
            index=0,
        )

        st.markdown(
            f"""
            <div style="margin-top: 30px; padding: 12px; background: #090A0F; border: 1px solid #1E2230; border-radius: 4px;">
                <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.75rem; color: #9CA3AF;">
                    ESTADO OPERATIVO<br>
                    &bull; Base de datos: SQLite<br>
                    &bull; {detector_status}<br>
                    &bull; OCR: EasyOCR (es)
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if nav_choice.startswith("01"):
        view_live_operation(pipeline, db)
    elif nav_choice.startswith("02"):
        view_database_audit(db)
    elif nav_choice.startswith("03"):
        view_technical_lab()

    return {"status": "ok", "view": nav_choice}


if __name__ == "__main__":
    main()
