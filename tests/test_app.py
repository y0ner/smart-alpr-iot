"""Pruebas unitarias para la interfaz Streamlit y archivos de despliegue Docker."""

from unittest.mock import patch

from src.app import INDUSTRIAL_CSS, get_alpr_system, render_header
from src.config import BASE_DIR
from src.db import DatabaseManager
from src.detector import ALPRPipeline


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
    """Verifica que la cabecera industrial se renderice sin excepciones."""
    with patch("streamlit.markdown") as mock_markdown:
        render_header()
        mock_markdown.assert_called_once()
        args, kwargs = mock_markdown.call_args
        assert "SMART ALPR IOT" in args[0]
        assert kwargs.get("unsafe_allow_html") is True


def test_get_alpr_system_instances() -> None:
    """Verifica que el singleton del sistema devuelva instancias operativas."""
    db, pipeline = get_alpr_system()
    assert isinstance(db, DatabaseManager)
    assert isinstance(pipeline, ALPRPipeline)


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
