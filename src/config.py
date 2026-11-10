"""Módulo de configuración global del sistema Smart ALPR IoT.

Centraliza rutas absolutas, hiperparámetros de visión artificial y umbrales
de detección para garantizar que no existan rutas ni constantes hardcodeadas.
"""

from pathlib import Path
from typing import Final

# Rutas Base del Proyecto
BASE_DIR: Final[Path] = Path(__file__).resolve().parent.parent
DATA_DIR: Final[Path] = BASE_DIR / "data"
RAW_DATA_DIR: Final[Path] = DATA_DIR / "raw"
PROCESSED_DATA_DIR: Final[Path] = DATA_DIR / "processed"
SAMPLES_DATA_DIR: Final[Path] = DATA_DIR / "samples"
MODELS_DIR: Final[Path] = BASE_DIR / "models"
DOCS_DIR: Final[Path] = BASE_DIR / "docs"
NOTEBOOKS_DIR: Final[Path] = BASE_DIR / "notebooks"
TESTS_DIR: Final[Path] = BASE_DIR / "tests"
DB_DIR: Final[Path] = BASE_DIR / "db"
SCHEMA_SQL_PATH: Final[Path] = DB_DIR / "schema.sql"
SQLITE_DB_PATH: Final[Path] = DB_DIR / "smart_alpr.db"

# Hiperparámetros de Visión Artificial y Detección
YOLO_MODEL_PATH: Final[Path] = MODELS_DIR / "best.pt"
YOLO_CONF_THRESHOLD: Final[float] = 0.45
CROP_MARGIN_RATIO: Final[float] = 0.12
RESIZE_SCALE_FACTOR: Final[float] = 3.0

# Hiperparámetros de OCR y Validación
OCR_ALLOWLIST: Final[str] = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
VOTER_WINDOW_SIZE: Final[int] = 7
VOTER_MIN_MATCHES: Final[int] = 4
COOLDOWN_SECONDS: Final[float] = 15.0

# Parámetros de Cámara e Inferencia
DEFAULT_CAMERA_SOURCE: Final[int] = 0
TARGET_FPS: Final[int] = 30
