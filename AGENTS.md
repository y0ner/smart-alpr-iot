# Manual operativo para agentes de IA — Smart ALPR IoT

Rige cómo trabajan los agentes de IA en este repositorio. Si algo de este
archivo choca con una instrucción de un brief, gana este archivo.

## 1. Roles

| Rol | Quién | Qué hace |
|---|---|---|
| **Humano** | Yoner | Aprueba cada oleada, autoriza los merges, entrena YOLO en Colab y prueba la consola con cámara y placas reales. |
| **Agente jefe** | Claude Code | Escribe los briefs, crea ramas y *worktrees*, revisa, corre pruebas, abre los PR y fusiona a `dev` con autorización. |
| **Agentes trabajadores** | Gemini (CLI `agy`) | Implementan **una tarea por rama**, en su propio *worktree*, tocando solo los archivos de su brief. |

## 2. Flujo (una tarea = una rama = un PR)

1. El jefe crea el *worktree* en `~/Desktop/Projects/worktrees/smart-alpr-<rama>`.
2. El trabajador recibe `docs/agentes/briefs/<id>-<tema>.md` y trabaja **solo** en su *worktree*.
3. El trabajador hace commits pequeños y **no hace push**.
4. El jefe revisa (sección 5), hace push y abre el PR contra `dev`.
5. Si algo falla, el jefe devuelve el error exacto con archivo y línea; el trabajador corrige en la misma rama.
6. Con la revisión aprobada y la autorización del humano, el jefe fusiona con *squash* a `dev`.
7. `main` solo recibe cambios después de la prueba del humano.

## 3. Comandos

El entorno virtual está en `.venv/` (en el *worktree* es un enlace al del repo principal).

| Para qué | Comando | ¿Lo corre el trabajador? |
|---|---|---|
| Análisis estático | `.venv/bin/ruff check src tests` | **Sí**, las veces que quiera. |
| Pruebas sin OCR | `.venv/bin/python -m pytest -q tests/test_preprocessor.py tests/test_validator.py tests/test_db.py` | Solo si su brief lo dice. |
| Pruebas completas | `.venv/bin/python -m pytest -q tests` | **No**: cargan EasyOCR (≈1,5 GB de RAM) y las corre el jefe, una a la vez. |
| Consola | `./run.sh` | **No**. |

No corras `ruff format` sobre archivos completos: el repo no está formateado
con él y el diff quedaría ilegible.

## 4. Reglas para los trabajadores

1. **Aislamiento:** solo los "Archivos permitidos" del brief. Si hace falta otro, te detienes y lo reportas.
2. **Sin merge, push, rebase ni cambio de rama.**
3. **Sin dependencias nuevas** (el *sandbox* no tiene red): se piden al jefe.
4. **Código completo:** sin `TODO`/`FIXME`, funciones vacías, datos simulados en producción ni errores silenciados (`except: pass`, `except Exception` sin registrar ni relanzar).
5. **Nada de cifras inventadas.** Toda métrica que se muestre como medida (precisión, CER, latencia, confianza, curvas) sale de una medición real sobre datos etiquetados. Si algo no se puede medir todavía, el texto dice "sin medir" y explica qué falta. Prohibido: fórmulas que "estiman" precisión o *recall*, confianzas fijas (`confidence=0.85`) y textos de estado que no consultan el estado real.
6. **Hiperparámetros en `src/config.py`.** Ningún umbral ni tamaño suelto en el código. Si tu brief no incluye `config.py` y necesitas una constante nueva, pídesela al jefe.
7. **Pruebas:** toda lógica nueva llega con su prueba. No debilites ni borres pruebas existentes para que pasen; si una prueba vieja choca con el brief, explícalo en tu respuesta.
8. **Datos:** no modifiques `data/samples/` ni la verdad de referencia (`data/samples/placas_de_prueba/verdad.csv`).
9. **Diseño de la consola:** tokens de `.agents/rules/industrial-ui-standard.md` (fondo `#090A0F`, superficies `#11131A`, bordes `#1E2230`, verde `#10B981`, ámbar `#F59E0B`, rojo `#EF4444`, amarillo placa `#FACC15`; Inter y JetBrains Mono). Sin emojis, sin degradados.
10. **Idioma:** español en comentarios, docstrings, textos de usuario y mensajes de commit. Los nombres de funciones siguen el estilo del archivo que editas.
11. **Commits:** `tipo(ámbito): descripción` en español (p. ej. `fix(preprocessor): normaliza la altura del recorte`).

### Alcance prohibido

- Enviar imágenes, placas o registros de acceso a servicios externos (APIs de OCR, analítica, telemetría).
- Cambiar el esquema de la base de datos (`db/schema.sql`) sin un brief que lo pida.
- Presentar como "en producción", "en línea" o "probado en campo" algo que no se ha probado así.

## 5. Lista de revisión de cada PR (jefe)

1. **[Aislamiento]** Solo cambió los archivos permitidos.
2. **[Integridad]** Sin `TODO`, simulaciones en producción, cifras inventadas ni errores silenciados.
3. **[Análisis]** `.venv/bin/ruff check src tests` sin problemas.
4. **[Pruebas]** `.venv/bin/python -m pytest -q tests` en verde (local y CI).
5. **[Criterios]** Cada criterio de aceptación del brief se puede señalar en el diff.
6. **[Entorno real]** El PR dice qué debe probar el humano.
