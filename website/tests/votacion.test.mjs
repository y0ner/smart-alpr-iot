import assert from "node:assert/strict";
import test from "node:test";
import { BufferVotacion } from "../js/votacion.js";

test("4 de 7 lecturas iguales registran ENTRADA, respetan espera de 15 s y luego emiten SALIDA", () => {
  const votador = new BufferVotacion();
  const t0 = 1_700_000_000_000;

  // 7 lecturas mezcladas donde ABC123 aparece 4 veces (incluyendo una corrección OCR)
  const secuencia = [
    ["ABC123", 0.85, t0],
    ["XYZ999", 0.80, t0 + 500],
    ["ABCI23", 0.75, t0 + 1000],
    ["DEF456", 0.70, t0 + 1500],
    ["ABC123", 0.90, t0 + 2000],
    ["GHI789", 0.65, t0 + 2500]
  ];

  for (const [texto, conf, ts] of secuencia) {
    const r = votador.agregarLectura(texto, conf, ts);
    assert.equal(r.aceptada, true);
    assert.equal(r.consenso, false, `no debe haber consenso antes de 4 votos (${texto})`);
  }

  // Cuarta lectura válida de ABC123 dentro de la ventana de 7: dispara ENTRADA
  const rEntrada = votador.agregarLectura("ABC123", 0.82, t0 + 3000);
  assert.equal(rEntrada.aceptada, true);
  assert.equal(rEntrada.consenso, true);
  assert.equal(rEntrada.enEspera, false);
  assert.equal(rEntrada.acceso.placa, "ABC123");
  assert.equal(rEntrada.acceso.sentido, "ENTRADA");
  assert.equal(rEntrada.acceso.votos, 4);

  // Dentro de los 15 s de espera no se repite el registro
  const rEspera = votador.agregarLectura("ABC123", 0.92, t0 + 10_000);
  assert.equal(rEspera.aceptada, true);
  assert.equal(rEspera.consenso, false);
  assert.equal(rEspera.enEspera, true);
  assert.equal(rEspera.restanteSeg, 8);

  // Al borde (14.999 s después de t0 + 3000) sigue en espera
  const rBorde = votador.agregarLectura("ABC123", 0.91, t0 + 17_999);
  assert.equal(rBorde.consenso, false);
  assert.equal(rBorde.enEspera, true);

  // Cumplidos los 15 s (>= 15000 ms después del consenso), la siguiente emisión es SALIDA
  const rSalida = votador.agregarLectura("ABC123", 0.88, t0 + 18_000);
  assert.equal(rSalida.aceptada, true);
  assert.equal(rSalida.consenso, true);
  assert.equal(rSalida.enEspera, false);
  assert.equal(rSalida.acceso.placa, "ABC123");
  assert.equal(rSalida.acceso.sentido, "SALIDA");
});

test("lecturas con confianza menor a 0.35 o formato inválido no votan ni ocupan el buffer", () => {
  const votador = new BufferVotacion();
  const t0 = 1_700_000_000_000;

  // Tres lecturas válidas previas
  votador.agregarLectura("UJR456", 0.80, t0);
  votador.agregarLectura("UJR456", 0.80, t0 + 200);
  votador.agregarLectura("UJR456", 0.80, t0 + 400);

  // Lecturas inválidas o con baja confianza no entran en la ventana de 7
  const descartes = [
    ["9HSOUE444", 0.95, "invalida"],
    ["", 0.99, "invalida"],
    ["3BC123", 0.90, "invalida"],
    ["UJR456", 0.34, "baja_confianza"],
    ["UJR456", 0.0, "baja_confianza"],
    ["UJR456", Number.NaN, "baja_confianza"]
  ];

  for (const [texto, conf, motivoEsperado] of descartes) {
    const r = votador.agregarLectura(texto, conf, t0 + 600);
    assert.equal(r.aceptada, false);
    assert.equal(r.consenso, false);
    assert.equal(r.motivoDescarte, motivoEsperado);
  }

  assert.equal(votador.obtenerEstado().buffer.length, 3);

  // La cuarta lectura con confianza exacta en el límite (0.35) sí vota y cierra el consenso
  const rLimite = votador.agregarLectura("UJR456", 0.35, t0 + 1000);
  assert.equal(rLimite.aceptada, true);
  assert.equal(rLimite.consenso, true);
  assert.equal(rLimite.acceso.sentido, "ENTRADA");
});
