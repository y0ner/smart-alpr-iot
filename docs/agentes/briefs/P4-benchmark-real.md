# Brief P4 · Benchmark A/B medido sobre las placas reales

- **Rama:** `fix/benchmark-real` · **Worktree:** `~/Desktop/Projects/worktrees/smart-alpr-fix-benchmark-real`
- **Lee primero:** `AGENTS.md` (reglas obligatorias; en especial la regla 5, "nada de cifras inventadas").
- **Objetivo:** que el Reto Extra (OCR con y sin preprocesamiento) se mida sobre las 11 fotos reales etiquetadas, sin ninguna cifra calculada con fórmulas inventadas.

## Contexto (por qué)

Todo está en `src/benchmark_ab.py`:

- **Curva inventada:** `evaluate_dataset` (~líneas 235-250) "calcula" precisión y *recall* por umbral con `prec = min(0.99, 0.82 + th*0.22)` y `rec = max(0.68, 0.98 - th*0.35)` y el informe la presenta como "análisis de sensibilidad empírico". No sale de ninguna medición. Medirla de verdad exige pesos YOLO y cajas etiquetadas; eso lo hará el notebook de entrenamiento (otro brief), no este módulo.
- **Conclusión fija:** `to_markdown` (~línea 108) afirma siempre que el preprocesamiento mejora el OCR, aunque los números digan lo contrario (en las fotos reales, hoy el pipeline completo acierta 0/11 y el recorte sin preprocesar 4/11).
- **Solo datos sintéticos:** el único lote es `create_synthetic_benchmark_dataset` (texto dibujado con `cv2.putText`). La guía del curso exige medir con datos reales no vistos.
- **Verdad de referencia:** `SAMPLES_GROUND_TRUTH_CSV` (`data/samples/placas_de_prueba/verdad.csv`) con columnas `archivo,placa,tipo,servicio,fuente,nota`; las fotos están en `REAL_SAMPLES_DIR`. Ambas constantes ya existen en `src/config.py`.
- La consola (`src/app.py`, otro brief en paralelo) usa `create_synthetic_benchmark_dataset(num_samples=...)`, `BenchmarkEvaluator(use_gpu=False).evaluate_dataset(dataset)` y `report.to_markdown()`: **esas firmas deben seguir funcionando**.
- El detector (`ALPRPipeline.detect_plates`) y el preprocesador (`ImagePreprocessor.process_pipeline`) cambian por dentro en otros briefs, pero sus firmas no cambian: úsalos, no copies su lógica.
- Las pruebas del benchmark ya están en `tests/test_benchmark.py` (el jefe las movió desde `tests/test_detector.py`).

## Archivos permitidos

- `src/benchmark_ab.py`
- `tests/test_benchmark.py`

## Qué hacer

1. **Quitar lo inventado.** Elimina `SensitivityThreshold`, el campo `sensitivity_curve`, el bucle de umbrales y la sección "Curva y Tabla de Sensibilidad" del informe. En su lugar, el informe incluye una sección corta **"Sensibilidad del umbral YOLO: sin medir"** que explica que requiere `models/best.pt` y que la curva P/R real sale de la validación del notebook de entrenamiento.
2. **Conclusión calculada.** La conclusión del informe se genera a partir de los números: si Modo B acierta más que Modo A, dice que mejoró y cuánto; si acierta menos, dice que empeoró y cuánto; si igual, lo dice. Nada de adjetivos fijos.
3. **Cargar la verdad.** `load_labeled_samples(csv_path: Path = SAMPLES_GROUND_TRUTH_CSV) -> list[LabeledSample]` (dataclass con `archivo`, `placa`, `ruta`), con error claro si falta un archivo.
4. **Medir el pipeline real.** `evaluate_real_samples(samples, pipeline: ALPRPipeline | None = None) -> RealSamplesReport`:
   - por cada foto: cajas con `pipeline.detect_plates(img)` (medir el tiempo de detección una sola vez), y por cada caja `pipeline.preprocessor.process_pipeline(img, bbox=...)`;
   - **Modo A (sin preprocesamiento):** OCR sobre `prep.original_crop`; **Modo B (con preprocesamiento):** OCR sobre `prep.final_for_ocr`. Usa `pipeline.extract_ocr_text` para ambos;
   - la predicción de cada modo es la lectura válida de mayor confianza entre las cajas; si ninguna es válida, el texto de la caja con más confianza (para el CER);
   - por modo: aciertos exactos, % de acierto, CER medio, confianza media y latencia media por foto (detección + preprocesamiento + OCR, medida con `time.perf_counter`);
   - por foto: archivo, verdad, lectura A, lectura B, acierto A, acierto B, cajas y latencias;
   - si `pipeline` es `None`, crea uno con `DatabaseManager(db_path=":memory:")` inicializado y `save_captures=False`: el benchmark **no escribe** en `db/` ni en `data/processed/`;
   - `RealSamplesReport.to_markdown()` (tabla resumen + tabla por foto + estado del detector: "YOLO" o "respaldo, sin pesos YOLO") y `to_dict()` serializable a JSON.
5. **Línea de comandos.** `python -m src.benchmark_ab` mide las fotos reales e imprime el Markdown; `--json <ruta>` además guarda `to_dict()`; `--sintetico N` corre el lote sintético anterior.
6. **Compatibilidad.** `levenshtein_distance`, `calculate_cer`, `create_synthetic_benchmark_dataset`, `BenchmarkEvaluator.evaluate_dataset` y `BenchmarkReport.to_markdown` siguen existiendo con la misma firma; el informe sintético se rotula "lote sintético".
7. **Pruebas** en `tests/test_benchmark.py`, sin cargar EasyOCR (inyecta un OCR falso: en `BenchmarkEvaluator` asigna `evaluator._reader` a un objeto con `readtext`; en `evaluate_real_samples` pasa un pipeline con `extract_ocr_text` reemplazado por `monkeypatch`):
   - `load_labeled_samples()` devuelve 11 muestras y todas las rutas existen;
   - con un OCR falso que acierta en Modo B y falla en Modo A, el informe cuenta exactamente esos aciertos, el CER y la conclusión "mejoró";
   - con el caso contrario, la conclusión dice "empeoró";
   - el Markdown contiene "sin medir" y no contiene "Precisión Estimada";
   - actualiza `test_benchmark_report_markdown_generation` a la nueva estructura (sin `sensitivity_curve`).

## Criterios de aceptación

- [ ] `grep -n "0\.82\|0\.22\|0\.98\|0\.35\|estimated_precision\|estimated_recall\|Sensitivity" src/benchmark_ab.py` no devuelve nada.
- [ ] `grep -n "significativamente" src/benchmark_ab.py` no devuelve nada (sin conclusiones fijas).
- [ ] `evaluate_real_samples` no escribe en `db/smart_alpr.db` ni en `data/processed/`.
- [ ] Las pruebas del paso 7 existen y no cargan EasyOCR.
- [ ] `.venv/bin/ruff check src tests` sin problemas.

## Qué se probará en el entorno real (lo hace el humano)

`.venv/bin/python -m src.benchmark_ab --json /tmp/medicion.json` imprime las dos tablas con las 11 fotos y la conclusión coincide con los números.
