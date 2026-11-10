"""Centro de Control y Monitoreo ALPR de Grado Industrial (Smart ALPR IoT).

Implementa la consola SCADA de misión crítica, telemetría de visión artificial,
auditoría de base de datos relacional y laboratorio del Reto Extra A/B.
Diseñado conforme al estándar anti-AI slop (.agents/rules/industrial-ui-standard.md).
"""

import os
import sys
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

from src.benchmark_ab import BenchmarkEvaluator, create_synthetic_benchmark_dataset
from src.config import SCHEMA_SQL_PATH, SQLITE_DB_PATH
from src.db import DatabaseManager
from src.detector import ALPRPipeline, FrameProcessingResult
from src.math_proof import (
    DEFAULT_PATCH_3X3,
    SOBEL_GX_KERNEL,
    generate_markdown_report,
    run_math_proof,
)

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

/* Tarjetas Técnicas Sobrias */
.industrial-card {
    background-color: #11131A;
    border: 1px solid #1E2230;
    border-radius: 6px;
    padding: 16px;
    margin-bottom: 14px;
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


@st.cache_resource
def get_alpr_system() -> tuple[DatabaseManager, ALPRPipeline]:
    """Inicializa en memoria compartida el gestor de base de datos y el pipeline ALPR."""
    db = DatabaseManager(db_path=SQLITE_DB_PATH)
    if not SQLITE_DB_PATH.exists():
        db.init_db(schema_path=SCHEMA_SQL_PATH)

    pipeline = ALPRPipeline(
        conf_threshold=0.45,
        use_gpu=False,
        db_manager=db,
        save_captures=True,
    )
    return db, pipeline


def render_header() -> None:
    """Renderiza el encabezado industrial superior con estado de red y servidor."""
    st.markdown(
        """
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
                <span class="badge-ok">&bull; SISTEMA ONLINE</span>
                <span style="color: #6B7280; font-family: 'JetBrains Mono', monospace; font-size: 0.75rem; margin-left: 10px;">
                    v0.1.0-prod
                </span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# -----------------------------------------------------------------------------
# VISTA 1: CONSOLA DE OPERACIÓN EN VIVO (CÁMARA / VIDEO / IMAGEN)
# -----------------------------------------------------------------------------


def view_live_operation(pipeline: ALPRPipeline) -> None:
    """Consola principal de monitoreo en tiempo real con visor de 5 fases morfológicas."""
    st.markdown("### Consola de Monitoreo e Inferencia en Tiempo Real")

    col_ctrl, col_main = st.columns([1, 2.5])

    with col_ctrl:
        st.markdown('<div class="industrial-card">', unsafe_allow_html=True)
        st.markdown(
            '<div class="industrial-card-header">Fuente de Video / Ingesta</div>',
            unsafe_allow_html=True,
        )

        source_type = st.radio(
            "Seleccionar Entrada:",
            [
                "Muestras de Tráfico (1-Click)",
                "Placas Reales Descargadas (1-Click)",
                "Cargar Imagen desde Archivo",
                "Cámara Web Local",
            ],
            index=0,
        )

        selected_frame: NDArray[np.uint8] | None = None

        if source_type == "Muestras de Tráfico (1-Click)":
            sample_choice = st.selectbox(
                "Seleccionar Muestra de Tráfico:",
                [
                    "Carro Particular (UJR456 - Sombra Cenital)",
                    "Carro Particular (VIV100 - Borde Nítido)",
                    "Moto Transporte (WJR78A - Formato Oficial)",
                    "Carro Confuso (KLM9O1 - Corrección O->0)",
                ],
            )
            # Generar muestra sintética reproducible para demostración inmediata
            plate_code = sample_choice.split("(")[1].split(" ")[0]
            # Frame simula carril vial
            canvas = np.full((360, 640, 3), (180, 180, 185), dtype=np.uint8)
            # Placa centrada
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
            # Tornillos
            cv2.circle(canvas, (x1 + 12, y1 + 12), 3, (30, 30, 30), -1)
            cv2.circle(canvas, (x2 - 12, y1 + 12), 3, (30, 30, 30), -1)
            selected_frame = canvas

        elif source_type == "Placas Reales Descargadas (1-Click)":
            real_samples_dir = BASE_DIR / "data" / "samples" / "placas_de_prueba"
            available_files = (
                sorted([
                    f
                    for f in os.listdir(real_samples_dir)
                    if f.lower().endswith((".jpg", ".png", ".jpeg"))
                ])
                if real_samples_dir.exists()
                else []
            )
            if available_files:
                file_choice = st.selectbox(
                    "Seleccionar Placa Real Descargada:",
                    available_files,
                    index=0,
                )
                img_path = str(real_samples_dir / file_choice)
                selected_frame = cv2.imread(img_path)
                st.caption(f"📁 Ruta: `~/Desktop/placas_de_prueba/{file_choice}`")
            else:
                st.warning("No se encontraron placas en el directorio de pruebas.")

        elif source_type == "Cargar Imagen desde Archivo":
            st.caption("💡 *Tip: Puedes seleccionar cualquier imagen de `~/Desktop/placas_de_prueba/`*")
            uploaded = st.file_uploader(
                "Subir archivo de imagen (JPG/PNG)", type=["jpg", "jpeg", "png"]
            )
            if uploaded is not None:
                file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
                selected_frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        elif source_type == "Cámara Web Local":
            cam_picture = st.camera_input("Capturar cuadro desde cámara local")
            if cam_picture is not None:
                file_bytes = np.asarray(bytearray(cam_picture.read()), dtype=np.uint8)
                selected_frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        st.markdown("</div>", unsafe_allow_html=True)

    with col_main:
        if selected_frame is not None:
            # Procesar el cuadro a través del pipeline ALPR completo
            res: FrameProcessingResult = pipeline.process_frame(selected_frame)

            # 1. Visor Principal con Overlays
            st.markdown('<div class="industrial-card">', unsafe_allow_html=True)
            st.markdown(
                '<div class="industrial-card-header">Visor Perimetral con Telemetría Activa</div>',
                unsafe_allow_html=True,
            )
            st.image(
                res.annotated_frame,
                channels="BGR",
                use_container_width=True,
                caption="Detección YOLO + Telemetría de Latencias",
            )
            st.markdown("</div>", unsafe_allow_html=True)

            # 2. Tarjeta de Estado de Acceso y Validación Colombiana
            st.markdown('<div class="industrial-card">', unsafe_allow_html=True)
            st.markdown(
                '<div class="industrial-card-header">Estado de Acceso & Validación de Placa</div>',
                unsafe_allow_html=True,
            )

            col_plate_render, col_plate_info = st.columns([1.2, 1.8])

            with col_plate_render:
                if res.detections:
                    d = res.detections[0]
                    st.markdown(
                        f"""
                        <div class="colombian-plate-card">
                            <div class="colombian-plate-text">{d.plate}</div>
                            <div class="colombian-plate-country">COLOMBIA &bull; {d.vehicle_type}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                else:
                    st.info("No se localizó matrícula vehicular en el cuadro actual.")

            with col_plate_info:
                if res.detections:
                    d = res.detections[0]
                    status_badge = (
                        '<span class="badge-ok">FORMATO OFICIAL VÁLIDO</span>'
                        if d.is_valid
                        else '<span class="badge-danger">FORMATO INVÁLIDO</span>'
                    )
                    st.markdown(f"**Validación:** {status_badge}", unsafe_allow_html=True)
                    st.markdown(
                        f"**Tipo Detectado:** `{d.vehicle_type}` | **Confianza OCR:** `{d.ocr_confidence:.2%}`"
                    )
                    if d.validation.corrections:
                        st.markdown(
                            f"**Corrección Posicional:** `{', '.join(d.validation.corrections)}`"
                        )
                    if res.access_record:
                        st.markdown(
                            f"**Transacción SQLite:** `{res.access_record['sentido']}` (ID: {res.access_record['id']})"
                        )
                        if res.access_record.get("permanencia_texto"):
                            st.markdown(
                                f"**Permanencia Registrada:** `{res.access_record['permanencia_texto']}`"
                            )

            st.markdown("</div>", unsafe_allow_html=True)

            # 3. Inspector de Visión Artificial: Tira Horizontal de 5 Fases Morfológicas
            if res.detections and res.detections[0].preprocessed is not None:
                p = res.detections[0].preprocessed
                st.markdown('<div class="industrial-card">', unsafe_allow_html=True)
                st.markdown(
                    '<div class="industrial-card-header">Inspector de Visión Artificial: Pipeline Morfológico (Demostración de Técnicas del Curso)</div>',
                    unsafe_allow_html=True,
                )

                f1, f2, f3, f4, f5 = st.columns(5)
                with f1:
                    st.caption("1. Recorte Margen (3x)")
                    st.image(p.original_crop, channels="BGR", use_container_width=True)
                with f2:
                    st.caption("2. Escala de Grises")
                    st.image(p.gray, use_container_width=True)
                with f3:
                    st.caption("3. Filtro Bilateral")
                    st.image(p.filtered, use_container_width=True)
                with f4:
                    st.caption("4. Umbral Adaptativo")
                    st.image(p.binary_adaptive, use_container_width=True)
                with f5:
                    st.caption("5. Morfología Final")
                    st.image(p.morph, use_container_width=True)

                st.markdown("</div>", unsafe_allow_html=True)
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

    with k1:
        st.markdown('<div class="industrial-card">', unsafe_allow_html=True)
        st.markdown(
            f'<div class="metric-value">{metricas["ocupacion_total"]}</div><div class="metric-label">Vehículos en Sitio</div>',
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with k2:
        st.markdown('<div class="industrial-card">', unsafe_allow_html=True)
        st.markdown(
            f'<div class="metric-value">{metricas["carros_dentro"]} / {metricas["motos_dentro"]}</div><div class="metric-label">Carros / Motos</div>',
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with k3:
        st.markdown('<div class="industrial-card">', unsafe_allow_html=True)
        st.markdown(
            f'<div class="metric-value">{metricas["total_accesos_hoy"]}</div><div class="metric-label">Accesos de Hoy</div>',
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with k4:
        st.markdown('<div class="industrial-card">', unsafe_allow_html=True)
        st.markdown(
            f'<div class="metric-value">{metricas["confianza_promedio"] * 100:.1f}%</div><div class="metric-label">Confianza Media</div>',
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    # Filtros y Tabla de Auditoría
    st.markdown('<div class="industrial-card">', unsafe_allow_html=True)
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
        # Formatear columnas para visualización clara
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
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.warning(
            "No se encontraron registros de accesos con los criterios de búsqueda especificados."
        )

    st.markdown("</div>", unsafe_allow_html=True)

    # Flujo por Hora
    st.markdown('<div class="industrial-card">', unsafe_allow_html=True)
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
    st.markdown("</div>", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# VISTA 3: LABORATORIO TÉCNICO Y RETO EXTRA A/B (CON VS. SIN PREPROCESAMIENTO)
# -----------------------------------------------------------------------------


def view_technical_lab() -> None:
    """Laboratorio para demostración del Reto Extra (+0.5 pts) y convolución matemática 3x3."""
    st.markdown("### Laboratorio de Visión Artificial: Reto Extra y Análisis Teórico")

    tab_ab, tab_math = st.tabs(
        [
            "Evaluador A/B: Reto Extra (+0.5 pts)",
            "Demostración Matemática 3x3 (Sección C.2)",
        ]
    )

    with tab_ab:
        st.markdown('<div class="industrial-card">', unsafe_allow_html=True)
        st.markdown(
            '<div class="industrial-card-header">Comparativa Experimental: Modo A (Crudo) vs. Modo B (Pipeline OpenCV)</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            """
            Este experimento ejecuta **EasyOCR** bajo dos condiciones sobre el mismo lote de prueba:
            - **Modo A (Sin Preprocesamiento):** La caja detectada se entrega directamente al OCR.
            - **Modo B (Con Preprocesamiento):** Se aplica la tira morfológica completa (Grises, Bilateral, Binarización y Apertura).
            """
        )

        num_samples = st.slider("Cantidad de placas de prueba a evaluar:", 2, 12, 6)

        if st.button("Ejecutar Benchmark Experimental A/B", type="primary"):
            with st.spinner("Ejecutando inferencia dual sobre lote experimental..."):
                dataset = create_synthetic_benchmark_dataset(num_samples=num_samples)
                evaluator = BenchmarkEvaluator(use_gpu=False)
                report = evaluator.evaluate_dataset(dataset)

                st.markdown(report.to_markdown())

        st.markdown("</div>", unsafe_allow_html=True)

    with tab_math:
        st.markdown('<div class="industrial-card">', unsafe_allow_html=True)
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
        st.markdown("</div>", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# PUNTO DE ENTRADA PRINCIPAL (MAIN)
# -----------------------------------------------------------------------------


def main() -> dict[str, Any]:
    """Punto de ejecución de la aplicación Streamlit."""
    st.set_page_config(
        page_title="Smart ALPR IoT | Telemetría Vial",
        page_icon="🚗",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(INDUSTRIAL_CSS, unsafe_allow_html=True)

    db, pipeline = get_alpr_system()
    render_header()

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
            """
            <div style="margin-top: 30px; padding: 12px; background: #090A0F; border: 1px solid #1E2230; border-radius: 4px;">
                <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.75rem; color: #9CA3AF;">
                    ESTADO OPERATIVO<br>
                    &bull; Backend: SQLite (WAL)<br>
                    &bull; Detector: YOLO + Heurística<br>
                    &bull; OCR: EasyOCR (es)<br>
                    &bull; Acento: Industrial Dark
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if nav_choice.startswith("01"):
        view_live_operation(pipeline)
    elif nav_choice.startswith("02"):
        view_database_audit(db)
    elif nav_choice.startswith("03"):
        view_technical_lab()

    return {"status": "ok", "view": nav_choice}


if __name__ == "__main__":
    main()
