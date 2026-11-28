/**
 * Punto de entrada de módulos ES del sitio estático.
 */

import { iniciarIdioma } from "./idioma.js";
import { montarLaboratorio } from "./laboratorio.js";
import { montarResultados } from "./resultados.js";
import { iniciarTema } from "./tema.js";
import { montarValidador } from "./validador.js";
import { montarVotacion } from "./votacion.js";

const suscriptoresIdioma = [];

const gestorIdioma = iniciarIdioma(() => {
  for (const fn of suscriptoresIdioma) {
    fn();
  }
});

const gestorTema = iniciarTema(gestorIdioma.obtenerIdioma);
suscriptoresIdioma.push(() => gestorTema.actualizarEtiqueta());

if (document.getElementById("etapas-grid")) {
  const res = montarResultados({ obtenerIdioma: gestorIdioma.obtenerIdioma });
  suscriptoresIdioma.push(() => res.repintar());
}

if (document.getElementById("lectura")) {
  const val = montarValidador({ obtenerIdioma: gestorIdioma.obtenerIdioma });
  suscriptoresIdioma.push(() => val.repintar());

  if (document.getElementById("slots")) {
    const vot = montarVotacion({
      obtenerIdioma: gestorIdioma.obtenerIdioma,
      obtenerPlacaActual: val.obtenerPlacaActual
    });
    suscriptoresIdioma.push(() => vot.repintar());
  }
}

if (document.getElementById("lab-visor")) {
  const lab = montarLaboratorio({ obtenerIdioma: gestorIdioma.obtenerIdioma });
  suscriptoresIdioma.push(() => lab.repintar());
}
