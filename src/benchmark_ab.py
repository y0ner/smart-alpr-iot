"""Módulo de evaluación experimental y Reto Extra A/B (Smart ALPR IoT).

Compara de manera reproducible el rendimiento de EasyOCR bajo dos condiciones:
- Modo A (Sin Preprocesamiento): Recorte crudo directo al OCR.
- Modo B (Con Preprocesamiento OpenCV): la salida `final_for_ocr` de `src/preprocessor.py`.

Calcula Tasa de Acierto Exacto (%), Tasa de Error de Caracteres (CER),
Distancia de Levenshtein, confianza media y latencia medida por muestra.
Alimenta la Sección D del Análisis Técnico sobre placas reales etiquetadas
y mantiene compatibilidad con el evaluador de lote sintético de la consola.
"""

import argparse
import csv
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray

# Permitir ejecución directa del script desde cualquier directorio
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.config import OCR_ALLOWLIST, SAMPLES_GROUND_TRUTH_CSV, SCHEMA_SQL_PATH
from src.db import DatabaseManager
from src.detector import ALPRPipeline
from src.preprocessor import ImagePreprocessor
from src.validator import PlateValidator


def levenshtein_distance(s1: str, s2: str) -> int:
    """Calcula la distancia de edición de Levenshtein entre dos cadenas."""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def calculate_cer(predicted: str, ground_truth: str) -> float:
    """Calcula el Character Error Rate (CER) normalizado."""
    if not ground_truth:
        return 0.0 if not predicted else 1.0
    dist = levenshtein_distance(predicted, ground_truth)
    return round(min(1.0, float(dist) / float(len(ground_truth))), 4)


@dataclass(frozen=True)
class ModeMetrics:
    """Métricas cuantitativas de una modalidad experimental de OCR."""

    mode_name: str
    total_samples: int
    exact_matches: int
    accuracy_percentage: float
    mean_cer: float
    mean_confidence: float
    mean_latency_ms: float


@dataclass(frozen=True)
class LabeledSample:
    """Muestra real etiquetada con su matrícula de referencia y ruta en disco."""

    archivo: str
    placa: str
    ruta: Path


@dataclass(frozen=True)
class RealSampleResult:
    """Resultado por fotografía real de la comparativa A/B."""

    archivo: str
    ground_truth: str
    prediction_a: str
    confidence_a: float
    cer_a: float
    match_a: bool
    latency_a_ms: float
    prediction_b: str
    confidence_b: float
    cer_b: float
    match_b: bool
    latency_b_ms: float
    boxes_count: int
    detection_ms: float
    preprocess_ms: float


def _compute_mode_metrics(
    mode_name: str,
    total_samples: int,
    exact_matches: int,
    cer_sum: float,
    conf_sum: float,
    lat_sum_ms: float,
) -> ModeMetrics:
    """Consolida métricas agregadas de acierto, CER, confianza y latencia media."""
    return ModeMetrics(
        mode_name=mode_name,
        total_samples=total_samples,
        exact_matches=exact_matches,
        accuracy_percentage=round((exact_matches / total_samples) * 100.0, 2),
        mean_cer=round(cer_sum / total_samples, 4),
        mean_confidence=round(conf_sum / total_samples, 4),
        mean_latency_ms=round(lat_sum_ms / total_samples, 2),
    )


def _build_conclusion_text(mode_a: ModeMetrics, mode_b: ModeMetrics) -> str:
    """Genera la conclusión cuantitativa del informe según los resultados medidos."""
    diff_matches = mode_b.exact_matches - mode_a.exact_matches
    diff_acc = round(mode_b.accuracy_percentage - mode_a.accuracy_percentage, 1)
    diff_cer_pp = round((mode_b.mean_cer - mode_a.mean_cer) * 100.0, 2)

    if mode_b.exact_matches > mode_a.exact_matches:
        return (
            f"El preprocesamiento (Modo B) mejoró la tasa de acierto exacto en "
            f"{diff_acc:+.1f} puntos porcentuales ({mode_b.exact_matches}/{mode_b.total_samples} "
            f"frente a {mode_a.exact_matches}/{mode_a.total_samples} en Modo A, "
            f"{diff_matches:+d} aciertos; variación de CER medio: {diff_cer_pp:+.2f} pp)."
        )
    if mode_b.exact_matches < mode_a.exact_matches:
        return (
            f"El preprocesamiento (Modo B) empeoró la tasa de acierto exacto en "
            f"{abs(diff_acc):.1f} puntos porcentuales ({mode_b.exact_matches}/{mode_b.total_samples} "
            f"frente a {mode_a.exact_matches}/{mode_a.total_samples} en Modo A, "
            f"{diff_matches:+d} aciertos; variación de CER medio: {diff_cer_pp:+.2f} pp)."
        )
    return (
        f"El preprocesamiento (Modo B) obtuvo los mismos aciertos exactos que el Modo A "
        f"({mode_b.exact_matches}/{mode_b.total_samples}, diferencia de {diff_acc:+.1f} puntos "
        f"porcentuales; variación de CER medio: {diff_cer_pp:+.2f} pp)."
    )


def _format_comparison_section(
    mode_a: ModeMetrics,
    mode_b: ModeMetrics,
    accuracy_gain_percentage: float,
    cer_reduction_percentage: float,
    section_number: int = 1,
) -> list[str]:
    """Formatea la tabla comparativa Modo A vs Modo B y su conclusión calculada."""
    conf_diff = mode_b.mean_confidence - mode_a.mean_confidence
    lat_diff = mode_b.mean_latency_ms - mode_a.mean_latency_ms
    conclusion = _build_conclusion_text(mode_a, mode_b)
    return [
        f"#### {section_number}. Tabla Comparativa de Rendimiento OCR (Modo A vs. Modo B)",
        "",
        "| Métrica de Evaluación | Modo A: Recorte Crudo (Sin Preprocesamiento) | Modo B: Pipeline OpenCV (Con Preprocesamiento) | Ganancia / Impacto Técnico |",
        "| :--- | :---: | :---: | :---: |",
        f"| **Tasa de Acierto Exacto (Accuracy)** | `{mode_a.accuracy_percentage:5.1f}%` ({mode_a.exact_matches}/{mode_a.total_samples}) | `{mode_b.accuracy_percentage:5.1f}%` ({mode_b.exact_matches}/{mode_b.total_samples}) | **`{accuracy_gain_percentage:+.1f}%`** |",
        f"| **Character Error Rate (CER Medio)** | `{mode_a.mean_cer * 100:5.2f}%` | `{mode_b.mean_cer * 100:5.2f}%` | **`{cer_reduction_percentage:+.2f}%`** |",
        f"| **Confianza Media de EasyOCR** | `{mode_a.mean_confidence:5.3f}` | `{mode_b.mean_confidence:5.3f}` | **`{conf_diff:+.3f}`** |",
        f"| **Latencia de Inferencia por Muestra** | `{mode_a.mean_latency_ms:5.1f} ms` | `{mode_b.mean_latency_ms:5.1f} ms` | `{lat_diff:+5.1f} ms` |",
        "",
        f"> **Conclusión:** {conclusion}",
    ]


def _format_yolo_unmeasured_section(section_number: int) -> list[str]:
    """Genera la nota explícita indicando que la sensibilidad YOLO aún no se ha medido."""
    note = (
        "La curva precisión/recall por umbral de confianza del detector YOLO está **sin medir** "
        "en este módulo porque requiere los pesos entrenados (`models/best.pt`) y cajas "
        "etiquetadas de validación; la curva P/R real sale de la validación del notebook de entrenamiento."
    )
    return [
        f"#### {section_number}. Sensibilidad del umbral YOLO: sin medir",
        "",
        note,
    ]


@dataclass(frozen=True)
class BenchmarkReport:
    """Informe consolidado de la comparativa A/B sobre el lote sintético."""

    mode_a_raw: ModeMetrics
    mode_b_preprocessed: ModeMetrics
    accuracy_gain_percentage: float
    cer_reduction_percentage: float

    def to_markdown(self) -> str:
        """Formatea el reporte experimental completo en tablas GitHub Flavored Markdown."""
        lines: list[str] = [
            "### Evaluación Experimental A/B: Reto Extra (lote sintético)",
            "",
            *_format_comparison_section(
                self.mode_a_raw,
                self.mode_b_preprocessed,
                self.accuracy_gain_percentage,
                self.cer_reduction_percentage,
                section_number=1,
            ),
            "",
            *_format_yolo_unmeasured_section(section_number=2),
        ]
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Serializa el reporte del lote sintético a un diccionario JSON-compatible."""
        return {
            "dataset_type": "lote sintético",
            "accuracy_gain_percentage": float(self.accuracy_gain_percentage),
            "cer_reduction_percentage": float(self.cer_reduction_percentage),
            "conclusion": _build_conclusion_text(self.mode_a_raw, self.mode_b_preprocessed),
            "yolo_threshold_sensitivity": (
                "sin medir (requiere models/best.pt y validación del notebook de entrenamiento)"
            ),
            "mode_a_raw": asdict(self.mode_a_raw),
            "mode_b_preprocessed": asdict(self.mode_b_preprocessed),
        }


@dataclass(frozen=True)
class RealSamplesReport:
    """Informe consolidado de la comparativa A/B medida sobre las fotos reales."""

    mode_a_raw: ModeMetrics
    mode_b_preprocessed: ModeMetrics
    accuracy_gain_percentage: float
    cer_reduction_percentage: float
    per_sample: list[RealSampleResult]
    yolo_loaded: bool

    @property
    def detector_status(self) -> str:
        """Indica si las cajas salieron de YOLO o del detector de respaldo."""
        return "YOLO" if self.yolo_loaded else "respaldo, sin pesos YOLO"

    def to_markdown(self) -> str:
        """Formatea la evaluación real con tabla resumen, detalle por foto y estado del detector."""
        lines: list[str] = [
            "### Evaluación Experimental A/B sobre Placas Reales Etiquetadas",
            "",
            f"- **Detector activo:** {self.detector_status} ({self.mode_a_raw.total_samples} fotos reales)",
            "",
            *_format_comparison_section(
                self.mode_a_raw,
                self.mode_b_preprocessed,
                self.accuracy_gain_percentage,
                self.cer_reduction_percentage,
                section_number=1,
            ),
            "",
            "#### 2. Detalle por Foto Real",
            "",
            "| Archivo | Verdad | Lectura A | Acierto A | Lectura B | Acierto B | Cajas | Latencia A (ms) | Latencia B (ms) |",
            "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
        ]

        for item in self.per_sample:
            pred_a = item.prediction_a if item.prediction_a else "-"
            pred_b = item.prediction_b if item.prediction_b else "-"
            acierto_a = "Sí" if item.match_a else "No"
            acierto_b = "Sí" if item.match_b else "No"
            lines.append(
                f"| `{item.archivo}` | `{item.ground_truth}` | `{pred_a}` | {acierto_a} | "
                f"`{pred_b}` | {acierto_b} | {item.boxes_count} | "
                f"`{item.latency_a_ms:.1f}` | `{item.latency_b_ms:.1f}` |"
            )

        lines.extend(["", *_format_yolo_unmeasured_section(section_number=3)])
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Serializa el reporte sobre placas reales a un diccionario compatible con JSON."""
        return {
            "dataset_type": "placas reales etiquetadas",
            "detector": self.detector_status,
            "yolo_loaded": bool(self.yolo_loaded),
            "accuracy_gain_percentage": float(self.accuracy_gain_percentage),
            "cer_reduction_percentage": float(self.cer_reduction_percentage),
            "conclusion": _build_conclusion_text(self.mode_a_raw, self.mode_b_preprocessed),
            "yolo_threshold_sensitivity": (
                "sin medir (requiere models/best.pt y validación del notebook de entrenamiento)"
            ),
            "mode_a_raw": asdict(self.mode_a_raw),
            "mode_b_preprocessed": asdict(self.mode_b_preprocessed),
            "per_sample": [asdict(item) for item in self.per_sample],
        }


def load_labeled_samples(csv_path: Path = SAMPLES_GROUND_TRUTH_CSV) -> list[LabeledSample]:
    """Carga las muestras reales etiquetadas desde el CSV de verdad de referencia."""
    resolved_csv = Path(csv_path)
    if not resolved_csv.exists():
        raise FileNotFoundError(
            f"No se encontró el CSV de verdad de referencia en: {resolved_csv}"
        )

    samples: list[LabeledSample] = []
    with open(resolved_csv, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "archivo" not in reader.fieldnames or "placa" not in reader.fieldnames:
            raise ValueError(
                f"El archivo CSV {resolved_csv} debe contener las columnas 'archivo' y 'placa'."
            )

        for line_num, row in enumerate(reader, start=2):
            archivo = (row.get("archivo") or "").strip()
            placa = (row.get("placa") or "").strip().upper()
            if not archivo or not placa:
                raise ValueError(
                    f"Fila {line_num} inválida en {resolved_csv}: faltan 'archivo' o 'placa'."
                )
            ruta = resolved_csv.parent / archivo
            if not ruta.exists():
                raise FileNotFoundError(
                    f"No se encontró la foto etiquetada '{archivo}' esperada en: {ruta}"
                )
            samples.append(LabeledSample(archivo=archivo, placa=placa, ruta=ruta))

    if not samples:
        raise ValueError(f"El archivo CSV {resolved_csv} no contiene muestras etiquetadas.")

    return samples


def _normalize_ocr_prediction(raw_text: str, confidence: float) -> tuple[bool, str, float]:
    """Aplica validación colombiana y conserva el texto limpio para cálculo de CER."""
    val = PlateValidator.correct_and_validate(raw_text)
    text = val.corrected_plate if val.is_valid else PlateValidator.clean_text(raw_text)
    return val.is_valid, text, float(confidence)


def _select_best_prediction(candidates: list[tuple[bool, str, float]]) -> tuple[str, float]:
    """Elige la lectura válida con mayor confianza o, si ninguna es válida, la de mayor confianza."""
    if not candidates:
        return "", 0.0
    valid = [item for item in candidates if item[0]]
    pool = valid if valid else candidates
    _, best_text, best_conf = max(pool, key=lambda item: (item[2], len(item[1]) > 0))
    return best_text, float(best_conf)


def evaluate_real_samples(
    samples: list[LabeledSample],
    pipeline: ALPRPipeline | None = None,
) -> RealSamplesReport:
    """Mide el pipeline real en Modo A y Modo B sobre fotos reales sin escribir en disco ni BD."""
    if not samples:
        raise ValueError("El lote de muestras reales para el benchmark no puede estar vacío.")

    own_db: DatabaseManager | None = None
    if pipeline is None:
        own_db = DatabaseManager(db_path=":memory:")
        own_db.init_db(schema_path=SCHEMA_SQL_PATH)
        pipeline = ALPRPipeline(
            use_gpu=False,
            db_manager=own_db,
            save_captures=False,
        )

    per_sample: list[RealSampleResult] = []
    matches_a = 0
    cer_sum_a = 0.0
    conf_sum_a = 0.0
    lat_sum_a = 0.0

    matches_b = 0
    cer_sum_b = 0.0
    conf_sum_b = 0.0
    lat_sum_b = 0.0

    try:
        for sample in samples:
            img = cv2.imread(str(sample.ruta))
            if img is None:
                raise ValueError(
                    f"No se pudo decodificar la imagen '{sample.archivo}' desde: {sample.ruta}"
                )

            t_det_0 = time.perf_counter()
            boxes = pipeline.detect_plates(img)
            det_ms = (time.perf_counter() - t_det_0) * 1000.0

            prep_ms = 0.0
            ocr_a_ms = 0.0
            ocr_b_ms = 0.0
            candidates_a: list[tuple[bool, str, float]] = []
            candidates_b: list[tuple[bool, str, float]] = []

            for bbox in boxes:
                t_prep_0 = time.perf_counter()
                prep = pipeline.preprocessor.process_pipeline(img, bbox=bbox.box_xyxy)
                prep_ms += (time.perf_counter() - t_prep_0) * 1000.0

                t_ocr_a0 = time.perf_counter()
                raw_a, conf_raw_a = pipeline.extract_ocr_text(prep.original_crop)
                ocr_a_ms += (time.perf_counter() - t_ocr_a0) * 1000.0
                candidates_a.append(_normalize_ocr_prediction(raw_a, conf_raw_a))

                t_ocr_b0 = time.perf_counter()
                raw_b, conf_raw_b = pipeline.extract_ocr_text(prep.final_for_ocr)
                ocr_b_ms += (time.perf_counter() - t_ocr_b0) * 1000.0
                candidates_b.append(_normalize_ocr_prediction(raw_b, conf_raw_b))

            pred_a, conf_a = _select_best_prediction(candidates_a)
            pred_b, conf_b = _select_best_prediction(candidates_b)

            truth = sample.placa.strip().upper()
            match_a = pred_a == truth
            match_b = pred_b == truth
            cer_a = calculate_cer(pred_a, truth)
            cer_b = calculate_cer(pred_b, truth)

            lat_a_ms = det_ms + ocr_a_ms
            lat_b_ms = det_ms + prep_ms + ocr_b_ms

            if match_a:
                matches_a += 1
            if match_b:
                matches_b += 1
            cer_sum_a += cer_a
            cer_sum_b += cer_b
            conf_sum_a += conf_a
            conf_sum_b += conf_b
            lat_sum_a += lat_a_ms
            lat_sum_b += lat_b_ms

            per_sample.append(
                RealSampleResult(
                    archivo=sample.archivo,
                    ground_truth=truth,
                    prediction_a=pred_a,
                    confidence_a=round(conf_a, 4),
                    cer_a=cer_a,
                    match_a=match_a,
                    latency_a_ms=round(lat_a_ms, 2),
                    prediction_b=pred_b,
                    confidence_b=round(conf_b, 4),
                    cer_b=cer_b,
                    match_b=match_b,
                    latency_b_ms=round(lat_b_ms, 2),
                    boxes_count=len(boxes),
                    detection_ms=round(det_ms, 2),
                    preprocess_ms=round(prep_ms, 2),
                )
            )
    finally:
        if own_db is not None:
            own_db.close()

    total = len(samples)
    metrics_a = _compute_mode_metrics(
        mode_name="Modo A (Sin Preprocesamiento)",
        total_samples=total,
        exact_matches=matches_a,
        cer_sum=cer_sum_a,
        conf_sum=conf_sum_a,
        lat_sum_ms=lat_sum_a,
    )
    metrics_b = _compute_mode_metrics(
        mode_name="Modo B (Con Preprocesamiento OpenCV)",
        total_samples=total,
        exact_matches=matches_b,
        cer_sum=cer_sum_b,
        conf_sum=conf_sum_b,
        lat_sum_ms=lat_sum_b,
    )

    acc_gain = metrics_b.accuracy_percentage - metrics_a.accuracy_percentage
    cer_reduc = (metrics_a.mean_cer - metrics_b.mean_cer) * 100.0

    return RealSamplesReport(
        mode_a_raw=metrics_a,
        mode_b_preprocessed=metrics_b,
        accuracy_gain_percentage=round(acc_gain, 2),
        cer_reduction_percentage=round(cer_reduc, 2),
        per_sample=per_sample,
        yolo_loaded=bool(getattr(pipeline, "yolo_loaded", False)),
    )


class BenchmarkEvaluator:
    """Evaluador experimental de extremo a extremo para el Reto Extra."""

    def __init__(self, preprocessor: ImagePreprocessor | None = None, use_gpu: bool = False) -> None:
        self.preprocessor = preprocessor or ImagePreprocessor()
        self.use_gpu = use_gpu
        self._reader: Any = None

    def _get_reader(self) -> Any:
        if self._reader is None:
            import easyocr

            self._reader = easyocr.Reader(["es"], gpu=self.use_gpu, verbose=False)
        return self._reader

    def run_ocr_on_image(self, img: NDArray[np.uint8]) -> tuple[str, float, float]:
        """Ejecuta EasyOCR midiendo texto extraído, confianza y latencia en ms."""
        reader = self._get_reader()
        t0 = time.perf_counter()
        results = reader.readtext(img, allowlist=OCR_ALLOWLIST, detail=1, paragraph=False)
        latency_ms = (time.perf_counter() - t0) * 1000.0

        if not results:
            return "", 0.0, latency_ms

        sorted_res = sorted(results, key=lambda it: it[0][0][0])
        raw_text = "".join(it[1] for it in sorted_res)
        conf = sum(float(it[2]) for it in sorted_res) / len(sorted_res)
        cleaned = PlateValidator.clean_text(raw_text)

        return cleaned, round(conf, 4), round(latency_ms, 2)

    def evaluate_dataset(
        self,
        samples: list[tuple[NDArray[np.uint8], str]],
    ) -> BenchmarkReport:
        """Evalúa un conjunto de imágenes bajo Modo A y Modo B con validación estricta."""
        matches_a = 0
        cer_sum_a = 0.0
        conf_sum_a = 0.0
        lat_sum_a = 0.0

        matches_b = 0
        cer_sum_b = 0.0
        conf_sum_b = 0.0
        lat_sum_b = 0.0

        n = len(samples)
        if n == 0:
            raise ValueError("El lote de muestras para el benchmark no puede estar vacío.")

        for raw_img, true_plate in samples:
            # Modo A (Recorte directo)
            pred_a, conf_a, lat_a = self.run_ocr_on_image(raw_img)
            val_a = PlateValidator.correct_and_validate(pred_a)
            final_a = val_a.corrected_plate if val_a.is_valid else pred_a

            if final_a == true_plate:
                matches_a += 1
            cer_sum_a += calculate_cer(final_a, true_plate)
            conf_sum_a += conf_a
            lat_sum_a += lat_a

            # Modo B (Pipeline OpenCV)
            t_prep_0 = time.perf_counter()
            prep_res = self.preprocessor.process_pipeline(raw_img)
            prep_ms = (time.perf_counter() - t_prep_0) * 1000.0

            pred_b, conf_b, lat_b = self.run_ocr_on_image(prep_res.final_for_ocr)
            val_b = PlateValidator.correct_and_validate(pred_b)
            final_b = val_b.corrected_plate if val_b.is_valid else pred_b

            if final_b == true_plate:
                matches_b += 1
            cer_sum_b += calculate_cer(final_b, true_plate)
            conf_sum_b += conf_b
            lat_sum_b += lat_b + prep_ms

        metrics_a = _compute_mode_metrics(
            mode_name="Modo A (Sin Preprocesamiento)",
            total_samples=n,
            exact_matches=matches_a,
            cer_sum=cer_sum_a,
            conf_sum=conf_sum_a,
            lat_sum_ms=lat_sum_a,
        )
        metrics_b = _compute_mode_metrics(
            mode_name="Modo B (Con Preprocesamiento OpenCV)",
            total_samples=n,
            exact_matches=matches_b,
            cer_sum=cer_sum_b,
            conf_sum=conf_sum_b,
            lat_sum_ms=lat_sum_b,
        )

        acc_gain = metrics_b.accuracy_percentage - metrics_a.accuracy_percentage
        cer_reduc = (metrics_a.mean_cer - metrics_b.mean_cer) * 100.0

        return BenchmarkReport(
            mode_a_raw=metrics_a,
            mode_b_preprocessed=metrics_b,
            accuracy_gain_percentage=round(acc_gain, 2),
            cer_reduction_percentage=round(cer_reduc, 2),
        )


def create_synthetic_benchmark_dataset(num_samples: int = 12) -> list[tuple[NDArray[np.uint8], str]]:
    """Genera un lote sintético de placas colombianas con tornillos, sombras y ruido."""
    plates_car = ["UJR456", "VIV100", "COL123", "ABC987", "KLM901", "ABO123"]
    plates_moto = ["WJR78A", "XYZ18D", "MTO45D", "GUA82B", "RIO33C", "VWA99E"]
    candidates = (plates_car + plates_moto)[:num_samples]

    dataset: list[tuple[NDArray[np.uint8], str]] = []
    for text in candidates:
        # Fondo amarillo reflectivo
        img = np.full((100, 280, 3), (21, 204, 250), dtype=np.uint8)
        # Marco perimetral
        cv2.rectangle(img, (3, 3), (276, 96), (0, 0, 0), 2)
        # Caracteres
        cv2.putText(
            img,
            text,
            (22, 70),
            cv2.FONT_HERSHEY_DUPLEX,
            fontScale=1.9,
            color=(10, 10, 10),
            thickness=4,
            lineType=cv2.LINE_AA,
        )
        # Tornillos y sombra
        for center in [(18, 18), (262, 18)]:
            cv2.circle(img, center, 3, (40, 40, 40), -1)
        shadow = img.copy()
        cv2.rectangle(shadow, (140, 0), (280, 50), (0, 0, 0), -1)
        cv2.addWeighted(shadow, 0.2, img, 0.8, 0, img)

        dataset.append((img, text))

    return dataset


def main(argv: list[str] | None = None) -> int:
    """Punto de entrada CLI para ejecutar el benchmark A/B sobre fotos reales o lote sintético."""
    parser = argparse.ArgumentParser(
        description="Evaluación experimental A/B (Modo A crudo vs. Modo B preprocesado)."
    )
    parser.add_argument(
        "--json",
        type=Path,
        default=None,
        metavar="RUTA",
        help="Ruta opcional donde guardar el resultado serializado en JSON.",
    )
    parser.add_argument(
        "--sintetico",
        type=int,
        default=None,
        metavar="N",
        help="Ejecuta el benchmark sobre un lote sintético de N muestras.",
    )
    args = parser.parse_args(argv)

    if args.sintetico is not None:
        if args.sintetico <= 0:
            parser.error("--sintetico requiere un entero positivo.")
        dataset = create_synthetic_benchmark_dataset(num_samples=args.sintetico)
        evaluator = BenchmarkEvaluator(use_gpu=False)
        report: BenchmarkReport | RealSamplesReport = evaluator.evaluate_dataset(dataset)
    else:
        samples = load_labeled_samples()
        report = evaluate_real_samples(samples)

    print(report.to_markdown())

    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps(report.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
