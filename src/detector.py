"""Pipeline principal de detección de placas vehiculares e inferencia OCR (Smart ALPR IoT).

Integra el detector Ultralytics YOLO, el pipeline de preprocesamiento morfológico OpenCV,
la extracción óptica con EasyOCR, el validador heurístico colombiano, el buffer de
votación temporal multi-cuadro y la persistencia transaccional en SQLite.
"""

import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray

from src.config import (
    FALLBACK_ASPECT_RATIO_RANGE,
    FALLBACK_MAX_CANDIDATES,
    FALLBACK_MIN_AREA_PX,
    FALLBACK_NMS_IOU,
    OCR_ALLOWLIST,
    OCR_EARLY_STOP_CONFIDENCE,
    PROCESSED_DATA_DIR,
    YOLO_CONF_THRESHOLD,
    YOLO_MODEL_PATH,
)
from src.db import DatabaseManager
from src.preprocessor import ImagePreprocessor, PreprocessedPlate
from src.validator import (
    ConsensusResult,
    PlateValidator,
    PlateVoterBuffer,
    ValidationResult,
    VehicleType,
)


@dataclass(frozen=True)
class DetectionBBox:
    """Bounding box de una placa detectada en el plano de la imagen.

    Attributes:
        x1: Coordenada izquierda en píxeles.
        y1: Coordenada superior en píxeles.
        x2: Coordenada derecha en píxeles.
        y2: Coordenada inferior en píxeles.
        confidence: Probabilidad del detector cuando ``source == "yolo"``.
            En el detector de respaldo (``source == "respaldo"``) es un puntaje
            heurístico en [0, 1] calculado sobre la imagen, no una probabilidad;
            en ``source == "cuadro_completo"`` vale 0.0 porque no hubo detección.
        class_name: Nombre de la clase detectada.
        source: Origen de la caja (``"yolo"``, ``"respaldo"`` o
            ``"cuadro_completo"``).
    """

    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float
    class_name: str = "placa"
    source: str = "yolo"

    @property
    def box_xyxy(self) -> tuple[int, int, int, int]:
        """Tupla (x1, y1, x2, y2)."""
        return (self.x1, self.y1, self.x2, self.y2)


@dataclass(frozen=True)
class PipelineLatency:
    """Telemetría de rendimiento y tiempos de latencia por fase del pipeline."""

    yolo_ms: float
    preprocessor_ms: float
    ocr_ms: float
    total_ms: float
    fps: float


@dataclass(frozen=True)
class PlateDetectionResult:
    """Resultado individual de detección y extracción de caracteres para una placa."""

    bbox: DetectionBBox
    raw_ocr_text: str
    validation: ValidationResult
    ocr_confidence: float
    preprocessed: PreprocessedPlate | None = None

    @property
    def plate(self) -> str:
        """Matrícula validada y corregida o texto limpio."""
        return self.validation.corrected_plate

    @property
    def vehicle_type(self) -> VehicleType:
        """Tipo de vehículo clasificado (CARRO / MOTO / DESCONOCIDO)."""
        return self.validation.vehicle_type

    @property
    def is_valid(self) -> bool:
        """Indica si cumple estrictamente la normativa colombiana."""
        return self.validation.is_valid


@dataclass(frozen=True)
class FrameProcessingResult:
    """Resultado consolidado del procesamiento de un cuadro de video o imagen."""

    timestamp: float
    latency: PipelineLatency
    detections: list[PlateDetectionResult]
    consensus: ConsensusResult | None
    access_record: dict[str, Any] | None
    saved_capture_path: str | None
    annotated_frame: NDArray[np.uint8] = field(repr=False)


class ALPRPipeline:
    """Orquestador integral del sistema Smart ALPR IoT."""

    def __init__(
        self,
        yolo_model_path: Path | str = YOLO_MODEL_PATH,
        conf_threshold: float = YOLO_CONF_THRESHOLD,
        use_gpu: bool = False,
        db_manager: DatabaseManager | None = None,
        voter_buffer: PlateVoterBuffer | None = None,
        preprocessor: ImagePreprocessor | None = None,
        save_captures: bool = True,
        output_dir: Path = PROCESSED_DATA_DIR,
    ) -> None:
        self.yolo_model_path = Path(yolo_model_path)
        self.conf_threshold = conf_threshold
        self.use_gpu = use_gpu
        self.db_manager = db_manager or DatabaseManager()
        self.voter_buffer = voter_buffer or PlateVoterBuffer()
        self.preprocessor = preprocessor or ImagePreprocessor()
        self.save_captures = save_captures
        self.output_dir = output_dir

        self.output_dir.mkdir(parents=True, exist_ok=True)

        self._yolo_model: Any = None
        self._ocr_reader: Any = None
        self._init_models()

    def _init_models(self) -> None:
        """Carga perezosa del modelo YOLO y del motor EasyOCR."""
        # 1. Cargar Ultralytics YOLO si existen los pesos en disco
        if self.yolo_model_path.exists():
            try:
                from ultralytics import YOLO

                self._yolo_model = YOLO(str(self.yolo_model_path))
            except (FileNotFoundError, RuntimeError, ValueError, ImportError):
                self._yolo_model = None

    def _get_ocr_reader(self) -> Any:
        """Inicializa EasyOCR de forma perezosa (lazy load) para optimizar memoria."""
        if self._ocr_reader is None:
            import easyocr

            self._ocr_reader = easyocr.Reader(["es"], gpu=self.use_gpu, verbose=False)
        return self._ocr_reader

    @property
    def yolo_loaded(self) -> bool:
        """Indica si hay pesos YOLO cargados (si no, se usa el detector de respaldo)."""
        return self._yolo_model is not None

    def is_valid_frame(self, frame: NDArray[np.uint8]) -> bool:
        """Comprueba que el frame sea un arreglo válido de NumPy no vacío."""
        return isinstance(frame, np.ndarray) and frame.size > 0 and len(frame.shape) in (2, 3)

    def detect_plates(self, frame: NDArray[np.uint8]) -> list[DetectionBBox]:
        """Detecta las regiones de placa vehicular en el cuadro.

        Utiliza Ultralytics YOLO si los pesos están disponibles; de lo contrario,
        utiliza un detector por morfología y contornos rectangulares de alta precisión.
        """
        boxes: list[DetectionBBox] = []

        if self._yolo_model is not None:
            # Inferencia con YOLO
            results = self._yolo_model(frame, conf=self.conf_threshold, verbose=False)
            for r in results:
                for box in r.boxes:
                    coords = box.xyxy[0].cpu().numpy().astype(int)
                    conf = float(box.conf[0].cpu().numpy())
                    boxes.append(
                        DetectionBBox(
                            x1=int(coords[0]),
                            y1=int(coords[1]),
                            x2=int(coords[2]),
                            y2=int(coords[3]),
                            confidence=round(conf, 4),
                        )
                    )
            return boxes

        return self._detect_plates_fallback(frame)

    @staticmethod
    def _score_candidate(
        contour: NDArray[np.int32],
        bbox_xywh: tuple[int, int, int, int],
        thresh_grad: NDArray[np.uint8],
    ) -> float:
        """Calcula un puntaje heurístico en [0, 1] para una caja candidata del respaldo.

        La fórmula promedia tres términos normalizados en [0, 1]:
        1. Cercanía de la relación de aspecto (w / h) a la de una placa
           colombiana estándar de automóvil (33 x 16 cm -> 2.0625), normalizada
           respecto a la desviación máxima dentro de ``FALLBACK_ASPECT_RATIO_RANGE``.
        2. Densidad de bordes verticales: fracción de píxeles activos en el
           gradiente horizontal umbralizado (``thresh_grad``) dentro de la caja.
        3. Relleno del contorno: ``cv2.contourArea(contour) / (w * h)``.
        """
        x, y, w, h = bbox_xywh
        aspect_ratio = float(w) / float(max(1, h))
        min_ar, max_ar = FALLBACK_ASPECT_RATIO_RANGE
        # Dimensión física reglamentaria de placa colombiana de automóvil: 33 x 16 cm
        target_ar = 33.0 / 16.0
        max_dev = max(abs(target_ar - min_ar), abs(max_ar - target_ar), 1e-6)
        aspect_score = max(0.0, 1.0 - abs(aspect_ratio - target_ar) / max_dev)

        roi_thresh = thresh_grad[y : y + h, x : x + w]
        edge_density = float(np.count_nonzero(roi_thresh)) / float(max(1, roi_thresh.size))

        box_area = float(max(1, w * h))
        fill_ratio = min(1.0, max(0.0, float(cv2.contourArea(contour)) / box_area))

        score = (aspect_score + edge_density + fill_ratio) / 3.0
        return round(min(1.0, max(0.0, score)), 4)

    def _detect_plates_fallback(self, frame: NDArray[np.uint8]) -> list[DetectionBBox]:
        """Detector de respaldo por gradiente Sobel, cierre morfológico, puntaje y NMS."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        grad_x = cv2.Sobel(gray, ddepth=cv2.CV_16S, dx=1, dy=0, ksize=3)
        abs_grad_x = cv2.convertScaleAbs(grad_x)

        # Suavizado y cierre morfológico para agrupar caracteres en una masa continua
        blurred = cv2.GaussianBlur(abs_grad_x, (9, 9), 0)
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 3))
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        h_img, w_img = frame.shape[:2]
        min_ar, max_ar = FALLBACK_ASPECT_RATIO_RANGE

        rects: list[list[int]] = []
        scores: list[float] = []

        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = float(w) / float(max(1, h))
            area = w * h

            # Criterios geométricos de placa vehicular basados en config.py
            if (
                min_ar <= aspect_ratio <= max_ar
                and area >= FALLBACK_MIN_AREA_PX
                and w < w_img * 0.95
                and h < h_img * 0.8
            ):
                score = self._score_candidate(cnt, (x, y, w, h), thresh)
                rects.append([x, y, w, h])
                scores.append(score)

        boxes: list[DetectionBBox] = []
        if rects:
            raw_indices = cv2.dnn.NMSBoxes(rects, scores, 0.0, FALLBACK_NMS_IOU)
            kept_indices = (
                [int(i) for i in np.asarray(raw_indices).reshape(-1)]
                if len(raw_indices) > 0
                else []
            )
            for idx in kept_indices:
                x, y, w, h = rects[idx]
                boxes.append(
                    DetectionBBox(
                        x1=x,
                        y1=y,
                        x2=x + w,
                        y2=y + h,
                        confidence=scores[idx],
                        source="respaldo",
                    )
                )
            boxes.sort(key=lambda b: b.confidence, reverse=True)
            boxes = boxes[:FALLBACK_MAX_CANDIDATES]

        # Si no hay candidatos válidos, se entrega el cuadro completo con puntaje honesto 0.0
        if not boxes:
            boxes.append(
                DetectionBBox(
                    x1=0,
                    y1=0,
                    x2=w_img,
                    y2=h_img,
                    confidence=0.0,
                    source="cuadro_completo",
                )
            )

        return boxes

    def extract_ocr_text(self, image: NDArray[np.uint8]) -> tuple[str, float]:
        """Extrae caracteres alfanuméricos mediante EasyOCR con allowlist estricto."""
        reader = self._get_ocr_reader()
        results = reader.readtext(
            image,
            allowlist=OCR_ALLOWLIST,
            detail=1,
            paragraph=False,
        )

        if not results:
            return "", 0.0

        # 1. Probar fragmentos individuales ordenados por confianza
        # En placas reales, la matrícula y el municipio ('BOGOTA', 'MEDELLIN') vienen separados
        for item in sorted(results, key=lambda x: -x[2]):
            text_snippet = item[1].strip().upper()
            cleaned = PlateValidator.clean_text(text_snippet)
            val = PlateValidator.correct_and_validate(cleaned)
            if val.is_valid:
                return val.corrected_plate, round(float(item[2]), 4)

        # 2. Probar combinación secuencial de fragmentos ordenados horizontalmente
        sorted_results = sorted(results, key=lambda item: item[0][0][0])
        extracted_text = "".join(item[1] for item in sorted_results)
        cleaned_joined = PlateValidator.clean_text(extracted_text)
        val_joined = PlateValidator.correct_and_validate(cleaned_joined)
        if val_joined.is_valid:
            mean_conf = sum(float(item[2]) for item in sorted_results) / len(sorted_results)
            return val_joined.corrected_plate, round(mean_conf, 4)

        # 3. Buscar subcadena de 6 caracteres que satisfaga el patrón colombiano
        match = re.search(r"[A-Z0-9]{6}", cleaned_joined)
        if match:
            candidate = match.group(0)
            val_match = PlateValidator.correct_and_validate(candidate)
            if val_match.is_valid:
                mean_conf = sum(float(item[2]) for item in sorted_results) / len(sorted_results)
                return val_match.corrected_plate, round(mean_conf, 4)

        # Fallback estándar si no coincide con sintaxis estricta
        mean_conf = sum(float(item[2]) for item in sorted_results) / len(sorted_results)
        return extracted_text.strip().upper(), round(mean_conf, 4)

    def process_frame(
        self,
        frame: NDArray[np.uint8],
        timestamp: float | None = None,
        persist_on_consensus: bool = True,
        voter_buffer: PlateVoterBuffer | None = None,
    ) -> FrameProcessingResult:
        """Ejecuta el pipeline completo de detección, OCR, validación y persistencia.

        Args:
            frame: Cuadro BGR o en escala de grises.
            timestamp: Marca de tiempo del cuadro (por defecto, la hora actual).
            persist_on_consensus: Si es False, no guarda captura ni registra el acceso.
            voter_buffer: Buffer de votación a usar en este cuadro. Permite que cada
                sesión de la consola tenga su propia votación aunque comparta el
                pipeline (y los modelos cargados). Por defecto, el del pipeline.
        """
        voter = voter_buffer if voter_buffer is not None else self.voter_buffer
        t_start = time.perf_counter()
        t_now = time.time() if timestamp is None else timestamp

        if not self.is_valid_frame(frame):
            raise ValueError("El frame proporcionado no es una imagen válida de NumPy.")

        annotated = frame.copy()

        # 1. Detección de Bounding Boxes (YOLO / Fallback)
        t_yolo_0 = time.perf_counter()
        boxes = self.detect_plates(frame)
        yolo_ms = (time.perf_counter() - t_yolo_0) * 1000.0

        detections: list[PlateDetectionResult] = []
        total_prep_ms = 0.0
        total_ocr_ms = 0.0

        for bbox in boxes:
            # 2. Preprocesamiento Morfológico OpenCV (7 Fases)
            t_prep_0 = time.perf_counter()
            prep_result: PreprocessedPlate = self.preprocessor.process_pipeline(
                frame, bbox=bbox.box_xyxy
            )
            total_prep_ms += (time.perf_counter() - t_prep_0) * 1000.0

            # 3. Reconocimiento Óptico de Caracteres (EasyOCR)
            t_ocr_0 = time.perf_counter()
            ocr_text, ocr_conf = self.extract_ocr_text(prep_result.final_for_ocr)
            total_ocr_ms += (time.perf_counter() - t_ocr_0) * 1000.0

            # 4. Validación Sintáctica Colombiana y Corrección Heurística
            validation = PlateValidator.correct_and_validate(ocr_text)

            det_res = PlateDetectionResult(
                bbox=bbox,
                raw_ocr_text=ocr_text,
                validation=validation,
                ocr_confidence=ocr_conf,
                preprocessed=prep_result,
            )
            detections.append(det_res)

            # Parada temprana: no leer más cajas cuando ya hay una lectura válida confiable
            if validation.is_valid and ocr_conf >= OCR_EARLY_STOP_CONFIDENCE:
                break

        # 5. Votación Temporal Multi-Cuadro
        consensus: ConsensusResult | None = None
        access_record: dict[str, Any] | None = None
        saved_capture_path: str | None = None

        valid_detections = [d for d in detections if d.is_valid]
        if valid_detections:
            # Tomar la lectura con mayor confianza del cuadro actual
            best_det = max(valid_detections, key=lambda d: d.ocr_confidence)
            consensus = voter.add_observation(
                raw_text=best_det.plate,
                confidence=best_det.ocr_confidence,
                timestamp=t_now,
            )

            # 6. Persistencia Relacional y Guardado de Captura si hay Consenso
            if consensus is not None and persist_on_consensus:
                if self.save_captures and best_det.preprocessed is not None:
                    fname = f"{consensus.plate}_{int(t_now)}.jpg"
                    save_path = self.output_dir / fname
                    cv2.imwrite(str(save_path), best_det.preprocessed.original_crop)
                    saved_capture_path = str(save_path)

                access_record = self.db_manager.registrar_acceso(
                    placa=consensus.plate,
                    tipo=consensus.vehicle_type,
                    confianza=consensus.mean_confidence,
                    ruta_captura=saved_capture_path,
                    preprocesado=True,
                )

        # Medición global de latencia
        t_total_ms = (time.perf_counter() - t_start) * 1000.0
        fps = 1000.0 / t_total_ms if t_total_ms > 0 else 0.0

        latency = PipelineLatency(
            yolo_ms=round(yolo_ms, 2),
            preprocessor_ms=round(total_prep_ms, 2),
            ocr_ms=round(total_ocr_ms, 2),
            total_ms=round(t_total_ms, 2),
            fps=round(fps, 1),
        )

        # Renderizar overlays industriales en annotated_frame
        self._render_industrial_telemetry(annotated, detections, latency, consensus)

        return FrameProcessingResult(
            timestamp=t_now,
            latency=latency,
            detections=detections,
            consensus=consensus,
            access_record=access_record,
            saved_capture_path=saved_capture_path,
            annotated_frame=annotated,
        )

    def _render_industrial_telemetry(
        self,
        frame: NDArray[np.uint8],
        detections: list[PlateDetectionResult],
        latency: PipelineLatency,
        consensus: ConsensusResult | None,
    ) -> None:
        """Dibuja overlays nítidos de telemetría vial de misión crítica (anti-AI slop)."""
        # Bounding boxes sobrios para cada detección
        for det in detections:
            b = det.bbox
            color = (16, 185, 129) if det.is_valid else (11, 158, 245)  # Verde esmeralda o Ámbar
            # Rectángulo técnico de 2px
            cv2.rectangle(frame, (b.x1, b.y1), (b.x2, b.y2), color, 2)

            # Etiqueta sobre fondo oscuro sólido (#11131A)
            plate_text = det.plate or "---"
            if b.source in ("respaldo", "cuadro_completo"):
                label = f"{plate_text} [{det.vehicle_type}] respaldo OCR:{det.ocr_confidence:.2f}"
            else:
                label = f"{plate_text} [{det.vehicle_type}] {det.ocr_confidence:.2f}"
            (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(
                frame,
                (b.x1, max(0, b.y1 - lh - 8)),
                (b.x1 + lw + 8, b.y1),
                (17, 19, 26),
                -1,
            )
            cv2.putText(
                frame,
                label,
                (b.x1 + 4, max(lh + 2, b.y1 - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (243, 244, 246),
                1,
                cv2.LINE_AA,
            )

        # Panel de telemetría superior izquierdo
        telemetry_text = (
            f"FPS: {latency.fps:4.1f} | YOLO: {latency.yolo_ms:5.1f}ms | "
            f"CV: {latency.preprocessor_ms:5.1f}ms | OCR: {latency.ocr_ms:5.1f}ms"
        )
        cv2.rectangle(frame, (10, 10), (520, 36), (17, 19, 26), -1)
        cv2.rectangle(frame, (10, 10), (520, 36), (30, 34, 48), 1)
        cv2.putText(
            frame,
            telemetry_text,
            (16, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (243, 244, 246),
            1,
            cv2.LINE_AA,
        )


class ALPRDetector(ALPRPipeline):
    """Alias para compatibilidad retroactiva con ALPRPipeline."""

    def get_status(self) -> dict[str, Any]:
        """Devuelve el estado operativo del detector."""
        return {
            "conf_threshold": self.conf_threshold,
            "status": "ready",
            "yolo_loaded": self.yolo_loaded,
            "gpu_enabled": self.use_gpu,
        }
