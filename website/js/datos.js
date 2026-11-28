// Datos medidos que muestra el sitio. Los genera el agente jefe a partir de las
// mediciones del repositorio (src/benchmark_ab.py y las salidas reales del pipeline).
// No se editan a mano. `null` significa "todavía sin medir".

export const muestras = [
  {
    "archivo": "placa_bogota_AGH430.jpg",
    "placa": "AGH430",
    "img": "img/muestras/placa_bogota_AGH430.webp",
    "ancho": 180,
    "alto": 83
  },
  {
    "archivo": "placa_bogota_AHB965.jpg",
    "placa": "AHB965",
    "img": "img/muestras/placa_bogota_AHB965.webp",
    "ancho": 1280,
    "alto": 575
  },
  {
    "archivo": "placa_bogota_AGH430_b.jpg",
    "placa": "AGH430",
    "img": "img/muestras/placa_bogota_AGH430_b.webp",
    "ancho": 1024,
    "alto": 768
  },
  {
    "archivo": "placa_bogota_AGH430_c.jpg",
    "placa": "AGH430",
    "img": "img/muestras/placa_bogota_AGH430_c.webp",
    "ancho": 1024,
    "alto": 768
  },
  {
    "archivo": "placa_cartagena_WGL451.jpg",
    "placa": "WGL451",
    "img": "img/muestras/placa_cartagena_WGL451.webp",
    "ancho": 1280,
    "alto": 853
  },
  {
    "archivo": "placa_dosquebradas_SJS444.jpg",
    "placa": "SJS444",
    "img": "img/muestras/placa_dosquebradas_SJS444.webp",
    "ancho": 640,
    "alto": 479
  },
  {
    "archivo": "placa_itagui_TKI033.jpg",
    "placa": "TKI033",
    "img": "img/muestras/placa_itagui_TKI033.webp",
    "ancho": 640,
    "alto": 480
  },
  {
    "archivo": "placa_medellin_WHK086.jpg",
    "placa": "WHK086",
    "img": "img/muestras/placa_medellin_WHK086.webp",
    "ancho": 640,
    "alto": 481
  },
  {
    "archivo": "placa_popayan_EMY923.jpg",
    "placa": "EMY923",
    "img": "img/muestras/placa_popayan_EMY923.webp",
    "ancho": 1280,
    "alto": 853
  },
  {
    "archivo": "placa_sevilla_NOJ122.jpg",
    "placa": "NOJ122",
    "img": "img/muestras/placa_sevilla_NOJ122.webp",
    "ancho": 640,
    "alto": 480
  },
  {
    "archivo": "placa_sonson_LFB645.jpg",
    "placa": "LFB645",
    "img": "img/muestras/placa_sonson_LFB645.webp",
    "ancho": 640,
    "alto": 480
  }
];

export const etapas = {
  "antes": [
    {
      "clave": "captura",
      "img": "img/etapas/antes/01-captura.webp",
      "ancho": 720,
      "alto": 480,
      "param": {
        "es": "YOLO · conf ≥ 0.45",
        "en": "YOLO · conf ≥ 0.45"
      }
    },
    {
      "clave": "recorte",
      "img": "img/etapas/antes/02-recorte.webp",
      "ancho": 360,
      "alto": 176,
      "param": {
        "es": "margen 12 % · escala ×3 bicúbica",
        "en": "12% margin · 3× bicubic upscale"
      }
    },
    {
      "clave": "grises",
      "img": "img/etapas/antes/03-grises.webp",
      "ancho": 360,
      "alto": 176,
      "param": {
        "es": "Y = 0.299R + 0.587G + 0.114B",
        "en": "Y = 0.299R + 0.587G + 0.114B"
      }
    },
    {
      "clave": "bilateral",
      "img": "img/etapas/antes/04-bilateral.webp",
      "ancho": 360,
      "alto": 176,
      "param": {
        "es": "d = 11 · σ color = 17 · σ espacio = 17",
        "en": "d = 11 · σ color = 17 · σ space = 17"
      }
    },
    {
      "clave": "umbral",
      "img": "img/etapas/antes/05-umbral.webp",
      "ancho": 360,
      "alto": 176,
      "param": {
        "es": "gaussiano · bloque 19 × 19 · C = 9",
        "en": "Gaussian · 19 × 19 block · C = 9"
      }
    },
    {
      "clave": "morfologia",
      "img": "img/etapas/antes/06-morfologia.webp",
      "ancho": 360,
      "alto": 176,
      "param": {
        "es": "apertura 3 × 3 · cierre 3 × 2",
        "en": "3 × 3 opening · 3 × 2 closing"
      }
    }
  ],
  "despues": null
};

export const medicion = {
  "antes": {
    "fecha": "2026-11-27",
    "detector": "respaldo",
    "total": 11,
    "aciertosSinPre": 4,
    "aciertosConPre": 0,
    "segundosMin": 2.5,
    "segundosMax": 15.9,
    "filas": [
      {
        "archivo": "placa_bogota_AGH430.jpg",
        "placa": "AGH430",
        "sinPre": "AGH430",
        "conPre": null,
        "cajas": 1,
        "ms": 5658.4
      },
      {
        "archivo": "placa_bogota_AHB965.jpg",
        "placa": "AHB965",
        "sinPre": null,
        "conPre": null,
        "cajas": 23,
        "ms": 6492.2
      },
      {
        "archivo": "placa_bogota_AGH430_b.jpg",
        "placa": "AGH430",
        "sinPre": "AGH430",
        "conPre": "AGI435",
        "cajas": 3,
        "ms": 3617.6
      },
      {
        "archivo": "placa_bogota_AGH430_c.jpg",
        "placa": "AGH430",
        "sinPre": "AUH430",
        "conPre": "AUH430",
        "cajas": 3,
        "ms": 2545.6
      },
      {
        "archivo": "placa_cartagena_WGL451.jpg",
        "placa": "WGL451",
        "sinPre": "AOG686",
        "conPre": null,
        "cajas": 8,
        "ms": 14271.0
      },
      {
        "archivo": "placa_dosquebradas_SJS444.jpg",
        "placa": "SJS444",
        "sinPre": "ISU544",
        "conPre": null,
        "cajas": 1,
        "ms": 7119.1
      },
      {
        "archivo": "placa_itagui_TKI033.jpg",
        "placa": "TKI033",
        "sinPre": "TKI033",
        "conPre": null,
        "cajas": 2,
        "ms": 4955.1
      },
      {
        "archivo": "placa_medellin_WHK086.jpg",
        "placa": "WHK086",
        "sinPre": "RHK086",
        "conPre": null,
        "cajas": 1,
        "ms": 15852.7
      },
      {
        "archivo": "placa_popayan_EMY923.jpg",
        "placa": "EMY923",
        "sinPre": null,
        "conPre": null,
        "cajas": 11,
        "ms": 7852.1
      },
      {
        "archivo": "placa_sevilla_NOJ122.jpg",
        "placa": "NOJ122",
        "sinPre": "IVI22Y",
        "conPre": "NOA122",
        "cajas": 5,
        "ms": 5614.1
      },
      {
        "archivo": "placa_sonson_LFB645.jpg",
        "placa": "LFB645",
        "sinPre": "LFB645",
        "conPre": "SOA50M",
        "cajas": 2,
        "ms": 8635.4
      }
    ]
  },
  "despues": null
};
