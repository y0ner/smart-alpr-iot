# Brief D1 · README, análisis técnico y licencia con datos reales

- **Rama:** `docs/documentos-medidos` · **Worktree:** `~/Desktop/Projects/worktrees/smart-alpr-docs-documentos-medidos`
- **Lee primero:** `AGENTS.md` (reglas obligatorias; en especial la regla 5, "nada de cifras inventadas").
- **Objetivo:** que el README y el análisis técnico que leerá el docente describan el sistema tal como es hoy, con los créditos correctos y solo cifras medidas.

## Contexto (por qué)

- **Créditos equivocados:** `README.md` (~líneas 256-264) dice "Docente Titular: Ing. Carlos Redondo", "Semestre: 2024-II / 2026" y un solo autor.
- **Licencia sin archivo:** el README (~línea 270) dice MIT pero no hay `LICENSE`.
- **Cifras inventadas en el README** (~líneas 215-221): "CER = 24.5 %", "CER = 2.78 %", "+34.2 % de ganancia", "Análisis de Sensibilidad YOLO". Ninguna salió de una medición.
- **Datos desactualizados en el README:** "60 tests" (~línea 168), estructura del repositorio sin `website/`, `AGENTS.md`, `docs/agentes/` ni `verdad.csv`, y sin enlace al sitio.
- **`docs/analisis_tecnico_A_E.md`:**
  - enlaces `file:///home/yoner/...` que no abren en GitHub (decenas; ver `grep -n "file:///" docs/analisis_tecnico_A_E.md`);
  - describe código que no existe (~líneas 80-85: `BEGIN IMMEDIATE`, `INSERT OR IGNORE`, `JULIANDAY`);
  - dice que la moto lleva "letra reglamentaria A-K" (~línea 78), pero `src/validator.py` acepta cualquier letra en la sexta posición;
  - sección D (~líneas 251-300) con la tabla y la curva de sensibilidad inventadas, "pruebas de campo" (~línea 320) y "40 s → 1.8 s" (~línea 381) que nunca se midieron.

### Créditos correctos (dados por el humano)

- **Autores:** Yoner De Jesus Isaza Vides y Kerlon Andres Causado Ojeda (escríbelos exactamente así).
- **Docente:** Eduardo Luis Sierra Fragozo.
- **Asignatura:** Internet de las Cosas, Grupo A3. Universidad de La Guajira, Facultad de Ingeniería, Ingeniería de Sistemas, Riohacha, noviembre de 2026.
- **Licencia:** MIT. `Copyright (c) 2026 Yoner De Jesus Isaza Vides y Kerlon Andres Causado Ojeda`.

### Enlaces

- Sitio: https://smart-alpr-iot.vercel.app
- Cuaderno en Colab: https://colab.research.google.com/github/y0ner/smart-alpr-iot/blob/main/notebooks/entrenamiento_y_metricas.ipynb

### Cifras medidas (las únicas que puedes usar)

Fotos: 11 fotos de 9 placas distintas en `data/samples/placas_de_prueba` (tres son del mismo carro), descargadas de internet, fuente pendiente de citar. No son un conjunto de prueba independiente. Detector: respaldo por bordes (no hay pesos YOLO).

| Medida | Antes (2026-11-27) | Después, Modo A (recorte sin procesar) | Después, Modo B (con preprocesamiento) |
|---|---|---|---|
| Aciertos exactos | 0/11 (pipeline) y 4/11 (recorte ×3) | 6/11 | 6/11 |
| CER medio | sin medir | 37,88 % | 34,85 % |
| Tiempo por foto (CPU) | 2,5 – 15,9 s | 1,85 s promedio | 1,20 s promedio (0,4 – 2,2 s) |

Fuente del "después": `python -m src.benchmark_ab` (2026-11-28). El tiempo del Modo A incluye la carga de EasyOCR en la primera foto.

Parámetro más sensible (experimento del 2026-11-27, mismas cajas del detector anterior, imagen entregada a EasyOCR):

| Imagen al OCR | Aciertos | OCR total |
|---|---|---|
| Salida binaria anterior (adaptativo + morfología) | 0/11 | 125 s |
| Recorte ×3 en color | 4/11 | 137 s |
| Grises a 120 px | 3/11 | 36 s |
| **Grises a 140 px** | **6/11** | 34 s |
| Grises a 180 px | 5/11 | 52 s |
| 140 px + Otsu | 4/11 | 23 s |
| 140 px + adaptativo proporcional | 3/11 | 23 s |

Casos de falla reales (después de las correcciones): EMY923 (el detector de respaldo no ubica la placa: sin lectura), WGL451 (placa blanca de taxi: lee LOC10A), AHB965 (lee la franja "STFE DE BOGOTÁ": STFC), SJS444 (Modo B lee ISU544; antes unía el municipio: 9HSOUEDRADAS444), WHK086 (lee RHK086: la W parece R).

Pruebas: 92 de Python (`pytest -q tests`) y 10 del sitio (`node --test website/tests/*.test.mjs`), todas en verde.

## Archivos permitidos

- `README.md`
- `docs/analisis_tecnico_A_E.md`
- `LICENSE` (**nuevo**)

## Qué hacer

1. **`LICENSE`:** texto MIT estándar con la línea de copyright de arriba.
2. **README, créditos:** autores, docente, asignatura y grupo como arriba; quita "Semestre 2024-II".
3. **README, enlaces:** al principio, enlaces al sitio y al cuaderno de Colab (botón "Open in Colab" en Markdown está bien).
4. **README, resultados:** reemplaza "Resumen de Resultados del Benchmark A/B" por la primera tabla de arriba y una línea con el comando que la reproduce. Elimina toda mención a "Análisis de Sensibilidad YOLO" como algo medido: di que la curva P/R real sale del cuaderno de Colab cuando haya pesos.
5. **README, datos al día:** número de pruebas (92 + 10), estructura del repositorio (agrega `website/`, `AGENTS.md`, `docs/agentes/`, `data/samples/placas_de_prueba/verdad.csv`), y una sección corta "Sitio web" que diga cómo verlo en local (`cd website && python3 -m http.server 8000`). Conserva el estilo visual del README (emojis de encabezado incluidos: el README no es la consola).
6. **Análisis técnico A–E:**
   - convierte todos los `file:///home/yoner/Desktop/Projects/smart-alpr-iot/...` en enlaces relativos (`../src/...`), y los anclas `#Lnn` déjalos solo si siguen siendo correctas o quítalas;
   - corrige la sección de persistencia para describir lo que hace `src/db.py` (consultas parametrizadas, `registrar_acceso` alterna ENTRADA/SALIDA consultando el último acceso y calcula la permanencia en Python con `datetime`);
   - corrige la regla de moto: sexta posición letra (cualquiera, salvo O, Q, I, L que se corrigen a dígito y dan CARRO);
   - sección D: reemplaza la tabla y la curva inventadas por las dos tablas de arriba (D.1 antes/después; D.3 parámetro más sensible: altura del recorte) y explica por qué EasyOCR lee mejor en grises a altura moderada que en binario; la sensibilidad del umbral de YOLO queda "sin medir, requiere pesos";
   - sección E: reemplaza "pruebas de campo" y los casos inventados por los casos de falla reales de arriba; quita "40 s → 1.8 s" y cualquier cifra sin medir (puedes citar las metas de la ficha del cliente como metas, no como logros).
7. No inventes nada: si un dato no está en este brief o en el código, no lo escribas.

## Criterios de aceptación

- [ ] `grep -n "Carlos Redondo\|2024-II\|24.5\|2.78\|34.2\|JULIANDAY\|BEGIN IMMEDIATE\|INSERT OR IGNORE\|A-K\|pruebas de campo\|file:///" README.md docs/analisis_tecnico_A_E.md` no devuelve nada.
- [ ] `LICENSE` existe con la línea de copyright exacta.
- [ ] El README enlaza `https://smart-alpr-iot.vercel.app` y el cuaderno de Colab.
- [ ] Todas las cifras de resultados del README y del análisis están en este brief.
- [ ] `.venv/bin/ruff check src tests` sigue sin problemas (no tocas código).

## Qué se probará en el entorno real (lo hace el humano)

Abrir el README en GitHub y el análisis técnico: los enlaces abren, los créditos son correctos y las tablas coinciden con el sitio.
