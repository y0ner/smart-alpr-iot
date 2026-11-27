# Brief P3 · Consola Streamlit: procesar una vez, registrar con intención y video en vivo

- **Rama:** `fix/consola-streamlit` · **Worktree:** `~/Desktop/Projects/worktrees/smart-alpr-fix-consola-streamlit`
- **Lee primero:** `AGENTS.md` (reglas obligatorias) y `.agents/rules/industrial-ui-standard.md`.
- **Objetivo:** que la consola no registre accesos por accidente, muestre solo estados reales y lea placas en video y cámara en vivo con la votación por sesión.

## Contexto (por qué)

Todo está en `src/app.py`:

- **Registros accidentales:** `view_live_operation` (~línea 319) llama `pipeline.process_frame(selected_frame)` en **cada** *rerun* de Streamlit. Cada clic en cualquier control vuelve a procesar la misma foto y le suma un voto a la votación; al cuarto clic se registra una ENTRADA y después una SALIDA, sin que pase ningún carro.
- **Votación compartida:** `get_alpr_system` (~línea 172) está en `@st.cache_resource`, así que el pipeline (y su `voter_buffer`) es el mismo para todas las sesiones del navegador. El jefe ya agregó a `ALPRPipeline.process_frame` el parámetro `voter_buffer=` para pasar una votación propia por sesión, y la propiedad `pipeline.yolo_loaded`.
- **Tarjetas vacías:** el patrón `st.markdown('<div class="industrial-card">')` … `st.markdown("</div>")` (~líneas 224/314, 322/333, 336/384, 389/412, 431-460, 463/505, 508/519, 539/562, 565/584) no envuelve nada: Streamlit cierra cada `st.markdown` por separado y el `<div>` queda vacío.
- **Estados falsos:** cabecera con "SISTEMA ONLINE" fijo y "v0.1.0-prod" (~líneas 201-203); barra lateral con "Backend: SQLite (WAL)" (WAL no se activa en `src/db.py`), "Detector: YOLO + Heurística" fijo y "Acento: Industrial Dark" (~líneas 630-637).
- **Constantes sueltas:** `conf_threshold=0.45` (~línea 179) en vez de `YOLO_CONF_THRESHOLD`.
- **Rutas falsas:** las leyendas dicen `~/Desktop/placas_de_prueba/` (~líneas 295 y 300); la carpeta real es `data/samples/placas_de_prueba/` (`REAL_SAMPLES_DIR` en `src/config.py`).
- **Emojis** (prohibidos por el estándar): `page_icon="🚗"`, "📁", "💡".
- **API deprecada:** `use_container_width=True` en `st.image` y `st.dataframe`. En la Streamlit instalada (1.67) se usa `width="stretch"`. Consulta (solo lectura) `.venv/lib/python3.12/site-packages/streamlit/elements/image.py` y `.../arrow.py`.
- **Sin tiempo real:** la única cámara es `st.camera_input` (una foto). El curso puntúa "tiempo real".
- **Verdad de referencia:** `SAMPLES_GROUND_TRUTH_CSV` (`data/samples/placas_de_prueba/verdad.csv`, columnas `archivo,placa,tipo,servicio,fuente,nota`) dice qué placa hay de verdad en cada foto.
- Constantes ya creadas por el jefe en `src/config.py`: `VIDEO_FRAME_STRIDE`, `DEFAULT_CAMERA_SOURCE`, `REAL_SAMPLES_DIR`, `SAMPLES_GROUND_TRUTH_CSV`, `YOLO_CONF_THRESHOLD`, `VOTER_WINDOW_SIZE`, `VOTER_MIN_MATCHES`, `COOLDOWN_SECONDS`.

## Archivos permitidos

- `src/app.py`
- `tests/test_app.py`

No cambies la pestaña del benchmark A/B más allá de las tarjetas y los textos (otro brief cambia `src/benchmark_ab.py` en paralelo; sigue usando `create_synthetic_benchmark_dataset` y `BenchmarkEvaluator` tal como están y rotula ese lote como **sintético**).

## Qué hacer

1. **Tarjetas reales.** Reemplaza cada par `<div class="industrial-card">`/`</div>` por `with st.container(border=True):` y el encabezado por el `st.markdown` con la clase `industrial-card-header` dentro del contenedor. Ajusta el CSS para que los contenedores con borde usen los tokens de tarjeta (`#11131A`, borde `#1E2230`, radio 6 px). Conserva en `INDUSTRIAL_CSS` los colores y fuentes que comprueba `test_app_css_conforms_to_industrial_standard`.
2. **Votación por sesión.** Guarda en `st.session_state` un `PlateVoterBuffer` por sesión (créalo con las constantes de `config.py`) y pásalo siempre con `voter_buffer=`. `get_alpr_system` sigue en caché para compartir los modelos cargados, y usa `YOLO_CONF_THRESHOLD`.
3. **Fotos: procesar una vez y registrar con intención.** Para las fuentes de foto (muestras sintéticas, placas reales, archivo subido, foto de la cámara del navegador):
   - calcula una huella del cuadro (`hashlib.sha1(frame.tobytes()).hexdigest()`) en una función pura y reutiliza el resultado guardado en `st.session_state` si la huella no cambió; solo se vuelve a procesar con una foto distinta o con el botón "Volver a procesar";
   - procesa con `persist_on_consensus=False`;
   - si la lectura es válida, muestra un botón **"Registrar acceso"** que llama a `db.registrar_acceso(...)` con la placa, el tipo y la confianza OCR, y muestra el sentido (ENTRADA/SALIDA) y la permanencia que devuelve. Sin clic, no se registra nada.
4. **Video y cámara en vivo.** Agrega dos fuentes:
   - **"Video (archivo)"**: `st.file_uploader` para `mp4`, `avi`, `mov`; guarda el archivo en un temporal (`tempfile`) y recórrelo con `cv2.VideoCapture`;
   - **"Cámara local en vivo"**: `cv2.VideoCapture(DEFAULT_CAMERA_SOURCE)` mientras un `st.toggle("Transmitir")` esté activo; si la cámara no abre, muestra un error claro (en Docker no hay cámara).
   En ambas, procesa 1 de cada `VIDEO_FRAME_STRIDE` cuadros con `persist_on_consensus=True` y la votación de la sesión, y actualiza con `st.empty()` la imagen anotada, la última lectura, la latencia medida y una lista de los accesos registrados por consenso. Libera siempre la captura (`release()` en `finally`). La antigua "Cámara Web Local" pasa a llamarse **"Foto con la cámara del navegador"**.
5. **Estados reales.** La cabecera recibe el estado del pipeline (`render_header(yolo_loaded: bool)`): con YOLO muestra `badge-ok` "YOLO cargado"; sin YOLO, `badge-alert` "Detector de respaldo (sin pesos YOLO)". La versión sale de `src.__version__`, sin "-prod". En la barra lateral: "Base de datos: SQLite", "Detector: YOLO" o "Detector: respaldo por bordes (sin pesos YOLO)", "OCR: EasyOCR (es)". Quita "Acento: Industrial Dark".
6. **Placas reales con su verdad.** En "Placas reales", lee `verdad.csv` con una función pura (`csv.DictReader`), muestra la ruta real `data/samples/placas_de_prueba/<archivo>`, la placa de referencia, y al leer, `badge-ok` "COINCIDE" o `badge-danger` "NO COINCIDE".
7. **Sin emojis ni API deprecada:** sin `page_icon` con emoji, sin "📁"/"💡"; `width="stretch"` en lugar de `use_container_width`. El visor de etapas dice "1. Recorte con margen" (sin "3x": el escalado cambia en otro brief) y sigue usando los campos `original_crop`, `gray`, `filtered`, `binary_adaptive` y `morph`.
8. **Pruebas** en `tests/test_app.py`:
   - `test_get_alpr_system_instances` no debe tocar `db/smart_alpr.db`: usa `monkeypatch.setattr("src.app.SQLITE_DB_PATH", tmp_path / "t.db")` y `get_alpr_system.clear()` antes y después;
   - la huella de un cuadro es estable y cambia si cambia un píxel;
   - el lector de `verdad.csv` devuelve 11 filas con `archivo` y `placa`, y cada archivo existe;
   - `render_header(yolo_loaded=False)` incluye "Detector de respaldo" y no incluye "ONLINE" ni "-prod";
   - `grep` en las pruebas: ninguna depende de EasyOCR.

## Criterios de aceptación

- [ ] `grep -n 'class="industrial-card"' src/app.py` y `grep -n '"</div>"' src/app.py` no devuelven nada (sin divs abiertos y cerrados por separado).
- [ ] `grep -n "use_container_width\|SISTEMA ONLINE\|-prod\|WAL\|Desktop/placas\|0\.45" src/app.py` no devuelve nada.
- [ ] `grep -nP "[\x{1F300}-\x{1FAFF}]" src/app.py` no devuelve nada.
- [ ] `process_frame` se llama siempre con `voter_buffer=` (`grep -n "process_frame(" src/app.py`).
- [ ] Ninguna fuente de foto llama `registrar_acceso` sin el botón "Registrar acceso".
- [ ] Las pruebas del paso 8 existen; `.venv/bin/ruff check src tests` sin problemas.

## Qué se probará en el entorno real (lo hace el humano)

1. `./run.sh`, elegir una placa real y hacer varios clics en otros controles: la tabla de la base de datos no debe cambiar hasta pulsar "Registrar acceso".
2. Subir un video corto de un carro entrando: debe aparecer una ENTRADA por consenso.
3. "Cámara local en vivo" con la webcam del portátil y una placa impresa.
4. Abrir la consola en dos pestañas: la votación de una no afecta a la otra.
