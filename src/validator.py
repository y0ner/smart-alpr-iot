"""Módulo de validación sintáctica de placas vehiculares colombianas.

Implementa corrección morfológica de caracteres, clasificación (CARRO / MOTO)
y buffer de votación multi-cuadro para filtrado temporal.
"""

from typing import Literal

VehicleType = Literal["CARRO", "MOTO", "DESCONOCIDO"]


class PlateValidator:
    """Validador sintáctico y heurístico de matrículas colombianas."""

    @staticmethod
    def clean_text(raw_text: str) -> str:
        """Limpia caracteres ajenos al allowlist alfanumérico."""
        return "".join(c for c in raw_text.upper() if c.isalnum())

    @staticmethod
    def classify_type(plate: str) -> VehicleType:
        """Determina el tipo preliminar de vehículo según el formato."""
        clean = PlateValidator.clean_text(plate)
        if len(clean) == 6:
            if clean[5].isdigit():
                return "CARRO"
            if clean[5].isalpha():
                return "MOTO"
        return "DESCONOCIDO"
