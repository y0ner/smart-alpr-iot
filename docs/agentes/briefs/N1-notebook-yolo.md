# Brief N1 · Notebook de Colab para entrenar YOLO de verdad

- **Rama:** `feat/notebook-yolo` · **Worktree:** `~/Desktop/Projects/worktrees/smart-alpr-feat-notebook-yolo`
- **Lee primero:** `AGENTS.md` (reglas obligatorias; en especial la regla 5, "nada de cifras inventadas").
- **Objetivo:** que Yoner abra el notebook en Google Colab con GPU, ponga su dataset etiquetado, pulse "Ejecutar todo" y obtenga `best.pt` con métricas reales sobre datos no vistos.

## Contexto (por qué)

- `notebooks/entrenamiento_y_metricas.ipynb` nunca se ejecutó (ninguna celda tiene salida) y el dataset es una "simulación de estructura `data.yaml`" con la descarga de Roboflow comentada. Por eso `models/best.pt` no existe y la consola usa siempre el detector de respaldo.
- El proyecto final exige: YOLO entrenado, datos propios, métricas sobre datos no vistos (partición de prueba) y 3 casos de falla.
- La consola carga los pesos de `models/best.pt` con `ultralytics` **8.4.x** (versión instalada: consúltala en `.venv/lib/python3.12/site-packages/ultralytics/__init__.py`). El notebook debe instalar esa misma versión para que los pesos sean compatibles.
- API de Ultralytics (solo lectura): `.venv/lib/python3.12/site-packages/ultralytics/`. Las curvas por confianza de la validación están en `ultralytics/utils/metrics.py` (`px` de 1000 umbrales, curvas P y R por clase). **No inventes métodos**: verifica cada atributo que uses en esos archivos.
- No hay red en tu *sandbox*: no puedes ejecutar el notebook. Por eso cada celda debe ser simple, sin pasos ocultos, y fallar con un mensaje claro si falta algo.

## Archivos permitidos

- `notebooks/entrenamiento_y_metricas.ipynb` (se reescribe completo)

## Qué hacer

Escribe el notebook (formato nbformat 4, JSON válido, **sin salidas**, celdas en español) con estas secciones:

1. **Portada:** qué hace, que se ejecuta en Colab con GPU (Entorno de ejecución → T4) y cuánto tarda aproximadamente.
2. **Comprobar GPU** (`!nvidia-smi`) e **instalar** `ultralytics==<versión exacta instalada>`.
3. **Configuración en una sola celda:** `ORIGEN_DATASET` = `"drive"` o `"roboflow"`; ruta del zip en Drive; para Roboflow, `workspace`, `project` y `version` como cadenas **vacías** que el usuario llena, y la API key leída de los secretos de Colab (`google.colab.userdata.get("ROBOFLOW_API_KEY")`). Nunca escribas una clave ni inventes un nombre de dataset. Hiperparámetros: modelo base `yolo11n.pt`, `EPOCHS=80`, `IMGSZ=640`, `BATCH=16`, `PATIENCE=20`, `SEED=42`.
4. **Obtener el dataset:** opción Drive (montar, descomprimir) u opción Roboflow (`pip install roboflow`, descarga en formato `yolov8`). Si la configuración está vacía, `raise ValueError` con instrucciones.
5. **Revisar el dataset:** leer `data.yaml`, exigir particiones `train`, `val` y `test` (si falta `test`, explicar cómo crearla y detenerse), contar imágenes y etiquetas por partición, y mostrar 6 imágenes de entrenamiento con sus cajas dibujadas.
6. **Entrenar** con `YOLO(MODELO_BASE).train(...)` usando la configuración.
7. **Evaluar en la partición de prueba (datos no vistos):** `model.val(split="test")`; imprimir precisión, *recall*, mAP50 y mAP50-95 tal como los devuelve Ultralytics, y mostrar las curvas guardadas.
8. **Sensibilidad del umbral de confianza (medida):** a partir de las curvas P y R por confianza de esa misma validación, tabla de precisión, *recall* y F1 en los umbrales 0.25, 0.35, 0.45, 0.55, 0.65 y 0.75, más el umbral con mejor F1. Guardarla en `sensibilidad_umbral.csv`. Esta tabla reemplaza la curva que antes se inventaba en `src/benchmark_ab.py`.
9. **Tres casos de falla:** predecir sobre la partición de prueba, emparejar con las etiquetas por IoU ≥ 0,5 y mostrar hasta 3 imágenes con falsos negativos o falsos positivos (caja real en verde, predicha en rojo), con una celda de texto para que el humano escriba por qué falló cada una.
10. **Exportar:** copiar `best.pt` y `sensibilidad_umbral.csv` a Drive y descargarlos (`google.colab.files.download`), con la instrucción final: colocar `best.pt` en `models/best.pt` del repo y correr `.venv/bin/python -m src.benchmark_ab --json medicion.json` para medir el pipeline completo con las 11 fotos reales.
11. **Si el dataset es propio:** celda de texto con cómo etiquetar fotos propias (Roboflow o CVAT, clase única `placa`, exportar en formato YOLO con particiones 70/20/10).

## Criterios de aceptación

- [ ] `.venv/bin/python -c "import json; nb=json.load(open('notebooks/entrenamiento_y_metricas.ipynb')); assert nb['nbformat']==4; assert all(not c.get('outputs') for c in nb['cells'] if c['cell_type']=='code')"` termina sin error.
- [ ] Cada celda de código compila: un script que extrae las celdas, quita las líneas que empiezan por `!` o `%` y llama `compile()` no da errores.
- [ ] `grep -n "Simulación\|simulad\|api_key=\"" notebooks/entrenamiento_y_metricas.ipynb` no devuelve nada.
- [ ] La versión instalada de `ultralytics` es la misma que la de `.venv`.
- [ ] Cada atributo de Ultralytics que usa el paso 8 existe en `ultralytics/utils/metrics.py` (cítalo en tu respuesta con archivo y línea).

## Qué se probará en el entorno real (lo hace el humano)

Abrir el notebook en Colab con GPU T4, llenar la configuración con su dataset y ejecutar todo: debe terminar con `best.pt` descargado, la tabla de sensibilidad y 3 casos de falla.
