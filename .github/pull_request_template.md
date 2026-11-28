## Qué cambia

<!-- Una o dos frases. -->

**Brief:** `docs/agentes/briefs/<id>-<tema>.md`

## Criterios de aceptación del brief

- [ ] …

## Revisión del agente jefe

- [ ] **Aislamiento:** solo se tocaron los archivos permitidos.
- [ ] **Integridad:** sin `TODO`/`FIXME`, sin simulaciones en producción, sin cifras inventadas, sin errores silenciados.
- [ ] **Análisis estático:** `ruff check src tests` sin problemas.
- [ ] **Pruebas:** `pytest -q tests` en verde (local y CI), con número de pruebas.

## Qué probar en el entorno real

<!-- Pasos concretos, o "No aplica". -->
