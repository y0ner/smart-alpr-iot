import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";
import { etapas } from "../js/datos.js";
import { normalizarIdioma } from "../js/idioma.js";
import { IDIOMAS, TEXTOS, traducir } from "../js/textos.js";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const WEBSITE_DIR = path.resolve(__dirname, "..");

function extraerClavesHtml(rutaRelativa) {
  const html = fs.readFileSync(path.join(WEBSITE_DIR, rutaRelativa), "utf8");
  const claves = new Set();

  for (const m of html.matchAll(/data-i18n="([^"]+)"/g)) {
    claves.add(m[1]);
  }
  for (const m of html.matchAll(/data-i18n-attr="([^"]+)"/g)) {
    for (const par of m[1].split(",")) {
      const idx = par.indexOf(":");
      if (idx !== -1) {
        claves.add(par.slice(idx + 1).trim());
      }
    }
  }
  return { html, claves };
}

test("es y en tienen exactamente las mismas claves no vacías", () => {
  assert.deepEqual(IDIOMAS, ["es", "en"]);
  const clavesEs = Object.keys(TEXTOS.es).sort();
  const clavesEn = Object.keys(TEXTOS.en).sort();
  assert.deepEqual(clavesEs, clavesEn);

  for (const clave of clavesEs) {
    assert.equal(typeof TEXTOS.es[clave], "string", `es["${clave}"] debe ser cadena`);
    assert.equal(typeof TEXTOS.en[clave], "string", `en["${clave}"] debe ser cadena`);
    assert.ok(TEXTOS.es[clave].trim().length > 0, `es["${clave}"] no debe estar vacío`);
    assert.ok(TEXTOS.en[clave].trim().length > 0, `en["${clave}"] no debe estar vacío`);
  }
});

test("toda clave data-i18n y data-i18n-attr de index.html y 404.html existe en es y en", () => {
  for (const archivo of ["index.html", "404.html"]) {
    const { claves } = extraerClavesHtml(archivo);
    assert.ok(claves.size > 0, `${archivo} debe tener claves i18n`);
    for (const clave of claves) {
      assert.ok(Object.hasOwn(TEXTOS.es, clave), `falta "${clave}" de ${archivo} en TEXTOS.es`);
      assert.ok(Object.hasOwn(TEXTOS.en, clave), `falta "${clave}" de ${archivo} en TEXTOS.en`);
    }
  }
});

test("todas las etapas de datos.js tienen título, descripción y alt en ambos idiomas", () => {
  for (const etapa of etapas.antes) {
    for (const sufijo of ["titulo", "que", "alt"]) {
      const clave = `etapas.item.${etapa.clave}.${sufijo}`;
      assert.ok(Object.hasOwn(TEXTOS.es, clave), `falta ${clave} en es`);
      assert.ok(Object.hasOwn(TEXTOS.en, clave), `falta ${clave} en en`);
    }
  }
});

test("todas las etiquetas img de index.html declaran alt, width y height", () => {
  const { html } = extraerClavesHtml("index.html");
  const imgs = [...html.matchAll(/<img\b[^>]*>/g)].map((m) => m[0]);
  assert.ok(imgs.length >= 3);
  for (const tag of imgs) {
    assert.match(tag, /\balt="/, `falta alt en ${tag}`);
    assert.match(tag, /\bwidth="\d+"/, `falta width en ${tag}`);
    assert.match(tag, /\bheight="\d+"/, `falta height en ${tag}`);
  }
});

test("normalizarIdioma y traducir funcionan con códigos BCP-47 y parámetros", () => {
  assert.equal(normalizarIdioma("es"), "es");
  assert.equal(normalizarIdioma("es-CO"), "es");
  assert.equal(normalizarIdioma("ES-419"), "es");
  assert.equal(normalizarIdioma("en-US"), "en");
  assert.equal(normalizarIdioma("fr-FR"), "en");
  assert.equal(normalizarIdioma(undefined), "en");

  assert.equal(
    traducir("en", "votacion.contador", { placa: "UJR456", votos: 3, k: 4 }),
    "UJR456: 3 of 4 votes"
  );
});
