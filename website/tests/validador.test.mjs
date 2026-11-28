import assert from "node:assert/strict";
import test from "node:test";
import { limpiarTextoPlaca, validarPlaca } from "../js/validador.js";

const CASOS_ESPERADOS = [
  ["KLM9O1", true, "KLM901", "CARRO"],
  ["8OG12A", true, "BOG12A", "MOTO"],
  ["AUH43O", true, "AUH430", "CARRO"],
  ["WGL45I", true, "WGL451", "CARRO"],
  ["TKI-O33", true, "TKI033", "CARRO"],
  ["9HSOUE444", false, "9HSOUE444", "DESCONOCIDO"],
  ["ABC123", true, "ABC123", "CARRO"],
  ["XYZ18D", true, "XYZ18D", "MOTO"],
  ["0I2S6B", true, "OIZ56B", "MOTO"],
  ["ABCDEF", false, "ABCDEF", "DESCONOCIDO"],
  ["AB1234", true, "ABI234", "CARRO"],
  ["ABC12Q", true, "ABC120", "CARRO"],
  ["3BC123", false, "3BC123", "DESCONOCIDO"],
  ["ABCX23", false, "ABCX23", "DESCONOCIDO"],
  ["abc 12 3", true, "ABC123", "CARRO"],
  ["", false, "", "DESCONOCIDO"]
];

test("validarPlaca cubre los 16 casos de referencia de src/validator.py", () => {
  for (const [entrada, validaEsperada, placaEsperada, tipoEsperado] of CASOS_ESPERADOS) {
    const res = validarPlaca(entrada);
    assert.equal(res.valida, validaEsperada, `validez inesperada para "${entrada}"`);
    assert.equal(res.tipo, tipoEsperado, `tipo inesperado para "${entrada}"`);
    if (validaEsperada) {
      assert.equal(res.placa, placaEsperada, `placa corregida inesperada para "${entrada}"`);
    }
  }
});

test("limpiarTextoPlaca normaliza mayúsculas y quita caracteres ajenos", () => {
  assert.equal(limpiarTextoPlaca("  tki-o33 \n"), "TKIO33");
  assert.equal(limpiarTextoPlaca(null), "");
  assert.equal(limpiarTextoPlaca(undefined), "");
});

test("validarPlaca reporta el motivo exacto de rechazo y número de correcciones", () => {
  const rCorto = validarPlaca("AB12");
  assert.equal(rCorto.valida, false);
  assert.equal(rCorto.errorClave, "longitud");

  const rLetra = validarPlaca("3BC123");
  assert.equal(rLetra.valida, false);
  assert.equal(rLetra.errorClave, "letra");
  assert.equal(rLetra.errorPos, 1);
  assert.equal(rLetra.errorChar, "3");

  const rDigito = validarPlaca("ABCX23");
  assert.equal(rDigito.valida, false);
  assert.equal(rDigito.errorClave, "digito");
  assert.equal(rDigito.errorPos, 4);
  assert.equal(rDigito.errorChar, "X");

  const rCorr = validarPlaca("0I2S6B");
  assert.equal(rCorr.valida, true);
  assert.equal(rCorr.correcciones, 3);
});
