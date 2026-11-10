"""Pruebas unitarias de scaffolding, verificación de entorno y estructura del proyecto."""

from pathlib import Path

import numpy as np

from src.config import (
    BASE_DIR,
    DATA_DIR,
    DB_DIR,
    DOCS_DIR,
    MODELS_DIR,
    NOTEBOOKS_DIR,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
    SAMPLES_DATA_DIR,
    SCHEMA_SQL_PATH,
    TESTS_DIR,
)
from src.db import DatabaseManager
from src.detector import ALPRDetector
from src.preprocessor import ImagePreprocessor
from src.validator import PlateValidator


def test_directory_structure_exists() -> None:
    """Verifica la existencia de todos los directorios requeridos por la guía."""
    required_dirs = [
        BASE_DIR / ".agents" / "skills",
        BASE_DIR / ".agents" / "rules",
        DATA_DIR,
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        SAMPLES_DATA_DIR,
        DB_DIR,
        DOCS_DIR,
        MODELS_DIR,
        NOTEBOOKS_DIR,
        BASE_DIR / "src",
        TESTS_DIR,
    ]
    for directory in required_dirs:
        assert directory.exists(), f"El directorio {directory} no existe."
        assert directory.is_dir(), f"{directory} no es un directorio."


def test_required_files_exist() -> None:
    """Verifica que todos los archivos requeridos existan."""
    required_files = [
        SCHEMA_SQL_PATH,
        DOCS_DIR / "ficha_cliente.md",
        DOCS_DIR / "analisis_tecnico_A_E.md",
        BASE_DIR / ".agents" / "rules" / "industrial-ui-standard.md",
        BASE_DIR / ".agents" / "skills" / "ui-ux-pro-max" / "SKILL.md",
        BASE_DIR / ".agents" / "skills" / "frontend-design" / "SKILL.md",
        BASE_DIR / "src" / "__init__.py",
        BASE_DIR / "src" / "config.py",
        BASE_DIR / "src" / "db.py",
        BASE_DIR / "src" / "preprocessor.py",
        BASE_DIR / "src" / "validator.py",
        BASE_DIR / "src" / "detector.py",
        BASE_DIR / "src" / "app.py",
        BASE_DIR / "requirements.txt",
        BASE_DIR / "README.md",
    ]
    for file_path in required_files:
        assert file_path.exists(), f"El archivo {file_path} no existe."
        assert file_path.is_file(), f"{file_path} no es un archivo."


def test_database_manager_in_memory() -> None:
    """Verifica que DatabaseManager inicialice y consulte SQLite correctamente."""
    db_manager = DatabaseManager(db_path=Path(":memory:"))
    db_manager.init_db(schema_path=SCHEMA_SQL_PATH)
    status = db_manager.ping()
    assert status["status"] == "connected"
    assert "sqlite_version" in status


def test_plate_validator_basic() -> None:
    """Verifica la lógica básica de clasificación de placas."""
    assert PlateValidator.classify_type("ABC123") == "CARRO"
    assert PlateValidator.classify_type("XYZ12A") == "MOTO"
    assert PlateValidator.classify_type("INVALID") == "DESCONOCIDO"


def test_image_preprocessor_dummy() -> None:
    """Verifica que el preprocesador OpenCV responda a una matriz sintética."""
    preprocessor = ImagePreprocessor(scale_factor=1.0)
    dummy_img = np.zeros((100, 200, 3), dtype=np.uint8)
    dummy_img[20:80, 20:180] = 255

    res = preprocessor.process_pipeline(dummy_img)
    assert "gray" in res
    assert "bilateral" in res
    assert "otsu" in res
    assert res["gray"].shape == (100, 200)


def test_detector_basic() -> None:
    """Verifica que el detector valide frames correctamente."""
    detector = ALPRDetector()
    valid_frame = np.ones((50, 50, 3), dtype=np.uint8)
    empty_frame = np.array([], dtype=np.uint8)

    assert detector.is_valid_frame(valid_frame) is True
    assert detector.is_valid_frame(empty_frame) is False
    status = detector.get_status()
    assert status["status"] == "ready"
