# Brief P2 · Detector de respaldo honesto y más rápido

- **Rama:** `fix/detector-respaldo` · **Worktree:** `~/Desktop/Projects/worktrees/smart-alpr-fix-detector-respaldo`
- **Lee primero:** `AGENTS.md` (reglas obligatorias).
- **Objetivo:** que, sin pesos YOLO, el detector entregue pocos candidatos con un puntaje calculado de la imagen (no una confianza fija), y que el pipeline deje de leer con OCR cuando ya encontró la placa.

## Contexto (por qué)

- `models/best.pt` no existe todavía, así que `ALPRPipeline.detect_plates` (`src/detector.py`, ~línea 159) siempre usa el respaldo por gradiente Sobel + cierre morfológico (~líneas 185-229).
- Cada caja del respaldo lleva `confidence=0.85` fijo (~línea 213), y la caja de "cuadro completo" también (~línea 225). La consola muestra ese 0.85 como si fuera una confianza medida: es una cifra inventada.
- En la foto `AHB965.jpg` el respaldo devolvió **23 cajas**; cada una pasa por preprocesamiento y EasyOCR (≈0,4-0,7 s por caja en CPU), así que esa foto tardó 15,9 s. La meta del cliente es < 2,5 s por lectura.
- `process_frame` (~línea 275) lee todas las cajas aunque la primera ya haya dado una placa válida.
- La prueba `test_process_frame_consensus_and_sqlite_persistence` (`tests/test_detector.py`) solo comprueba algo **si** EasyOCR lee bien (`if r2.detections and ...`): si no lee, pasa sin verificar nada.
- Constantes ya creadas por el jefe en `src/config.py`: `FALLBACK_ASPECT_RATIO_RANGE`, `FALLBACK_MIN_AREA_PX`, `FALLBACK_MAX_CANDIDATES`, `FALLBACK_NMS_IOU`, `OCR_EARLY_STOP_CONFIDENCE`.
- API de OpenCV disponible (solo lectura): `.venv/lib/python3.12/site-packages/cv2/` — `cv2.dnn.NMSBoxes(bboxes, scores, score_threshold, nms_threshold)` existe.

## Archivos permitidos

- `src/detector.py`
- `tests/test_detector.py`

No toques la sección de YOLO (cuando hay pesos se usa la confianza real del modelo), ni la firma de `process_frame`, ni la propiedad `yolo_loaded`.

## Qué hacer

1. **Origen de cada caja.** Agrega a `DetectionBBox` un campo `source: str` con valor por defecto `"yolo"`. El respaldo usa `"respaldo"` y la caja de cuadro completo `"cuadro_completo"`. Documenta en el docstring que, en el respaldo, `confidence` es un **puntaje heurístico** en [0, 1], no una probabilidad.
2. **Puntaje calculado.** Sustituye los `0.85` del respaldo por un puntaje en [0, 1] calculado de la región, por ejemplo el promedio de:
   - cercanía de la relación de aspecto a la de una placa colombiana de carro (33 × 16 cm → ≈2,06), normalizada dentro de `FALLBACK_ASPECT_RATIO_RANGE`;
   - densidad de bordes verticales (fracción de píxeles del gradiente umbralizado dentro de la caja);
   - relleno: área del contorno / área de la caja.
   Usa `FALLBACK_ASPECT_RATIO_RANGE` y `FALLBACK_MIN_AREA_PX` en lugar de los literales `1.6`, `4.2` y `1200`. El cálculo va en un método propio (p. ej. `_score_candidate`) con docstring que explique la fórmula.
3. **Menos candidatos.** Aplica supresión de no máximos (`cv2.dnn.NMSBoxes` con `FALLBACK_NMS_IOU`) y conserva como máximo `FALLBACK_MAX_CANDIDATES` cajas, ordenadas de mayor a menor puntaje.
4. **Cuadro completo.** Si no hay candidatos, devuelve una sola caja del cuadro completo con `source="cuadro_completo"` y `confidence=0.0` (no sabemos si hay placa: es lo honesto).
5. **Parada temprana.** En `process_frame`, recorre las cajas en el orden recibido y deja de leer en cuanto una detección sea válida (`validation.is_valid`) con `ocr_confidence >= OCR_EARLY_STOP_CONFIDENCE`. Las cajas no leídas no se agregan a `detections`.
6. **Etiqueta en la imagen anotada.** En `_render_industrial_telemetry`, si la caja viene del respaldo o del cuadro completo, la etiqueta dice `respaldo` en vez de mostrar un número que parezca confianza del detector (la confianza OCR sí se puede seguir mostrando, porque es medida).
7. **Pruebas** en `tests/test_detector.py`, todas sin cargar EasyOCR (usa `monkeypatch.setattr(pipeline, "extract_ocr_text", ...)` como en `test_process_frame_uses_session_voter_buffer`):
   - el respaldo nunca devuelve más de `FALLBACK_MAX_CANDIDATES` cajas en un cuadro con mucho ruido (p. ej. texto y líneas aleatorias con semilla fija);
   - dos cajas casi iguales quedan en una sola (NMS);
   - el puntaje de una placa sintética nítida es mayor que el de una región de ruido;
   - una imagen sin bordes (gris uniforme) devuelve una caja `cuadro_completo` con `confidence == 0.0`;
   - la parada temprana: con un OCR falso que devuelve una placa válida con 0.9 en la primera caja, `len(result.detections) == 1` aunque haya más cajas;
   - reescribe `test_process_frame_consensus_and_sqlite_persistence` con OCR falso para que **siempre** verifique consenso, registro en SQLite y captura en disco (sin el `if`).

## Criterios de aceptación

- [ ] `grep -n "0\.85" src/detector.py` no devuelve nada.
- [ ] `grep -n "1200\|1\.6 <=\|<= 4\.2" src/detector.py` no devuelve nada (salen de `config.py`).
- [ ] Las seis pruebas del paso 7 existen y no cargan EasyOCR.
- [ ] `grep -n "if r2.detections" tests/test_detector.py` no devuelve nada.
- [ ] `.venv/bin/ruff check src tests` sin problemas.

## Qué se probará en el entorno real (lo hace el humano)

Abrir la consola (`./run.sh`), elegir 3 placas reales y comprobar que la imagen anotada muestra pocas cajas, que dicen `respaldo` y que cada lectura tarda menos que antes.
