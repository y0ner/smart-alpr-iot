# Brief P1 · Preprocesador: altura normalizada, OCR sobre grises y enderezado correcto

- **Rama:** `fix/preprocesador-altura` · **Worktree:** `~/Desktop/Projects/worktrees/smart-alpr-fix-preprocesador-altura`
- **Lee primero:** `AGENTS.md` (reglas obligatorias).
- **Objetivo:** que la imagen que el pipeline entrega a EasyOCR mejore la lectura en vez de empeorarla, sin dejar de calcular y mostrar las etapas del curso (grises, bilateral, umbrales, morfología, enderezado).

## Contexto (por qué)

Todo está en `src/preprocessor.py`:

- **Escala fija:** `resize_patch` (~línea 122) multiplica el recorte por `RESIZE_SCALE_FACTOR = 3.0` sin mirar su tamaño. Un recorte de una foto de 900 px de alto termina en 2700 px.
- **Letras huecas:** `binarize_adaptive` (~línea 166) usa un bloque fijo de 19 px. En un recorte de 2700 px de alto los trazos miden más de 100 px, así que el bloque solo "ve" bordes: las letras quedan como contornos huecos (etapas `binary_adaptive` y `morph`), y la salida final (`final_for_ocr = deskewed`, ~línea 277) es esa imagen hueca.
- **Medición del jefe** sobre las 11 fotos reales (mismas cajas del detector, distintas imágenes al OCR):

  | Imagen que recibe EasyOCR | Aciertos | Tiempo total de OCR |
  |---|---|---|
  | `final_for_ocr` actual (adaptativo + morfología) | 0 / 11 | 125 s |
  | Recorte ×3 en color, sin filtros | 4 / 11 | 137 s |
  | **Recorte llevado a 140 px de alto, en grises** | **6 / 11** | **23 s** |
  | 140 px + Otsu | 4 / 11 | 23 s |
  | 140 px + adaptativo (bloque ≈ 0,3 × altura) | 3 / 11 | 23 s |

  Conclusión: EasyOCR (red CRAFT + CRNN) lee mejor en grises de altura moderada que en binario; la binarización y la morfología siguen siendo útiles para **mostrar** las técnicas y para estimar la inclinación, pero no deben ser la entrada del OCR.
- **Enderezado con la convención vieja:** `deskew_image` (~línea 202) pasa a `cv2.minAreaRect` coordenadas `(fila, columna)` de `np.where` (al revés: OpenCV espera `(x, y)`) e interpreta el ángulo como si estuviera en [-90, 0), que es la convención anterior a OpenCV 4.5.1. Con la versión instalada el ángulo está en (0, 90]. La prueba actual (`test_deskew_image_known_rotation`) no comprueba el ángulo.
- Constantes nuevas ya creadas por el jefe en `src/config.py`: `PLATE_NORMALIZED_HEIGHT_PX`, `BILATERAL_DIAMETER`, `BILATERAL_SIGMA_COLOR`, `BILATERAL_SIGMA_SPACE`, `ADAPTIVE_BLOCK_RATIO`, `ADAPTIVE_C`, `DESKEW_MAX_ANGLE`, `DESKEW_MIN_ANGLE`. Deja de importar `RESIZE_SCALE_FACTOR` (el jefe lo borrará de `config.py` al fusionar).

## Archivos permitidos

- `src/preprocessor.py`
- `tests/test_preprocessor.py`

Los campos de `PreprocessedPlate` (`original_crop`, `resized`, `gray`, `filtered`, `binary_otsu`, `binary_adaptive`, `morph`, `deskewed`, `final_for_ocr`, `rotation_angle`), `process_pipeline(image, bbox=None)`, `crop_with_margin`, `apply_morphology`, `deskew_image` y el acceso tipo diccionario (`res["bilateral"]`, `res["otsu"]`) **no cambian de nombre**: la consola, el detector y el benchmark los usan.

## Qué hacer

1. **Constructor:** `ImagePreprocessor(target_height: int | None = PLATE_NORMALIZED_HEIGHT_PX, margin_ratio: float = CROP_MARGIN_RATIO)`. Quita `scale_factor`. `target_height=None` significa "no reescalar".
2. **Fase 2 · altura normalizada:** `resize_patch` lleva el recorte a `target_height` píxeles de alto conservando la proporción; `cv2.INTER_CUBIC` si agranda y `cv2.INTER_AREA` si reduce. Recortes vacíos se devuelven tal cual.
3. **Fase 4 · bilateral** con `BILATERAL_DIAMETER`, `BILATERAL_SIGMA_COLOR`, `BILATERAL_SIGMA_SPACE`.
4. **Fase 5 · umbral adaptativo proporcional:** el bloque es `ADAPTIVE_BLOCK_RATIO × altura` redondeado al impar más cercano (mínimo 3) y `C = ADAPTIVE_C`. Otsu sin cambios.
5. **Fase 7 · enderezado correcto:** `deskew_image(gray_or_binary, mask=None)`:
   - estima el ángulo con `cv2.minAreaRect` sobre puntos `(x, y)` (usa `cv2.findNonZero` sobre la máscara de caracteres: blanco = carácter);
   - normaliza el ángulo de la convención actual de OpenCV al rango [-45, 45];
   - no rota si `abs(ángulo) < DESKEW_MIN_ANGLE` o `abs(ángulo) > DESKEW_MAX_ANGLE` (devuelve la imagen y `0.0`);
   - documenta en el docstring la convención de signos (positivo = antihorario, como `cv2.getRotationMatrix2D`).
6. **Entrada del OCR:** en `process_pipeline`, estima el ángulo con la morfología (`morph`) como máscara y aplica la rotación a la imagen **en grises normalizada** (`gray`). `deskewed` y `final_for_ocr` son ese gris enderezado (uint8, 2D). Actualiza el docstring de la clase y de `process_pipeline` con la secuencia real y el porqué (la tabla de arriba, resumida).
7. **Pruebas** en `tests/test_preprocessor.py`:
   - un recorte de 900 × 1800 y otro de 40 × 80 salen con 140 px de alto (o `PLATE_NORMALIZED_HEIGHT_PX`) y la proporción conservada (±1 px);
   - `target_height=None` no cambia el tamaño;
   - el bloque adaptativo es impar, ≥ 3 y crece con la altura;
   - **el enderezado mide bien:** una placa sintética sin ruido rotada +8° y −8° (fondo blanco, como `test_deskew_image_known_rotation`) da un ángulo detectado de signo correcto y |error| ≤ 1,5°; una placa recta da `0.0`;
   - `final_for_ocr` es 2D, uint8 y **no** es binaria (tiene más de 2 valores distintos) en una placa sintética con ruido;
   - en una placa sintética de 900 px de alto, la fracción de píxeles oscuros de `binary_adaptive` dentro de la zona del texto es al menos el doble que con el bloque fijo de 19 px (demuestra que las letras ya no quedan huecas);
   - actualiza `test_resize_patch` y `test_pipeline_phases_presence_and_types` a la nueva API.

## Criterios de aceptación

- [ ] `grep -n "RESIZE_SCALE_FACTOR\|scale_factor\|block_size: int = 19\|diameter: int = 11" src/preprocessor.py` no devuelve nada.
- [ ] `grep -n "np.where" src/preprocessor.py` no devuelve nada dentro de `deskew_image`.
- [ ] Las pruebas del paso 7 existen.
- [ ] Puedes correr y deben pasar: `.venv/bin/python -m pytest -q tests/test_preprocessor.py tests/test_validator.py` (no cargan EasyOCR).
- [ ] `.venv/bin/ruff check src tests` sin problemas.
- [ ] (Lo mide el jefe) Con las fotos reales, el Modo B del benchmark acierta al menos lo mismo que el gris a 140 px de la tabla.

## Qué se probará en el entorno real (lo hace el humano)

En la consola, elegir 3 placas reales y mirar el inspector de etapas: las letras del umbral adaptativo deben verse rellenas, y la lectura debe ser más rápida que antes.
