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
RESIZE_SCALE_FACTOR: Final[float] = 3.0  # obsoleto: lo reemplaza PLATE_NORMALIZED_HEIGHT_PX

# Preprocesamiento (medido sobre las 11 fotos reales el 2026-11-27: con el recorte
# llevado a 140 px de alto y en grises, EasyOCR acertó 6/11; a 120 px, 3/11; a 180 px, 5/11)
PLATE_NORMALIZED_HEIGHT_PX: Final[int] = 140
BILATERAL_DIAMETER: Final[int] = 9
BILATERAL_SIGMA_COLOR: Final[float] = 30.0
BILATERAL_SIGMA_SPACE: Final[float] = 30.0
# Bloque del umbral adaptativo como fracción de la altura del recorte (se redondea a impar)
ADAPTIVE_BLOCK_RATIO: Final[float] = 0.3
ADAPTIVE_C: Final[int] = 9
# Enderezado: no se rota por debajo del mínimo ni por encima del máximo (grados)
DESKEW_MIN_ANGLE: Final[float] = 0.5
DESKEW_MAX_ANGLE: Final[float] = 15.0

# Hiperparámetros de OCR y Validación
OCR_ALLOWLIST: Final[str] = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
VOTER_WINDOW_SIZE: Final[int] = 7
VOTER_MIN_MATCHES: Final[int] = 4
COOLDOWN_SECONDS: Final[float] = 15.0

# Parámetros de Cámara e Inferencia
DEFAULT_CAMERA_SOURCE: Final[int] = 0
TARGET_FPS: Final[int] = 30
# En video y cámara en vivo se procesa 1 de cada N cuadros (el OCR en CPU tarda)
VIDEO_FRAME_STRIDE: Final[int] = 3

# Muestras reales con verdad de referencia (placa leída a ojo en cada foto)
REAL_SAMPLES_DIR: Final[Path] = SAMPLES_DATA_DIR / "placas_de_prueba"
SAMPLES_GROUND_TRUTH_CSV: Final[Path] = REAL_SAMPLES_DIR / "verdad.csv"

# Detector de respaldo (sin pesos YOLO): candidatos por gradiente y morfología
FALLBACK_ASPECT_RATIO_RANGE: Final[tuple[float, float]] = (1.6, 4.2)
# Placa colombiana de automóvil: 33 × 16 cm
PLATE_CAR_ASPECT_RATIO: Final[float] = 33.0 / 16.0
FALLBACK_MIN_AREA_PX: Final[int] = 1200
FALLBACK_MAX_CANDIDATES: Final[int] = 5
FALLBACK_NMS_IOU: Final[float] = 0.3
# Se deja de leer candidatos al encontrar una placa válida con esta confianza OCR
OCR_EARLY_STOP_CONFIDENCE: Final[float] = 0.6
