"""Módulo de validación sintáctica de placas vehiculares colombianas.

Implementa corrección morfológica posicional de caracteres, clasificación
estricta (CARRO / MOTO) y buffer de votación multi-cuadro para filtrado temporal.
"""

import re
import time
from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Final, Literal

from src.config import (
    COOLDOWN_SECONDS,
    OCR_ALLOWLIST,
    VOTER_MIN_MATCHES,
    VOTER_WINDOW_SIZE,
)

VehicleType = Literal["CARRO", "MOTO", "DESCONOCIDO"]

# Mapeos de corrección posicional basados en confusiones morfológicas comunes de OCR
DIGIT_TO_LETTER: Final[dict[str, str]] = {
    "0": "O",
    "1": "I",
    "2": "Z",
    "5": "S",
    "6": "G",
    "8": "B",
    "4": "A",
}

LETTER_TO_DIGIT: Final[dict[str, str]] = {
    "O": "0",
    "Q": "0",
    "D": "0",
    "I": "1",
    "L": "1",
    "Z": "2",
    "S": "5",
    "B": "8",
    "G": "6",
    "A": "4",
}

# Caracteres de confusión que en la posición 5 siempre corresponden al dígito de un CARRO
POS5_CONFUSED_DIGITS: Final[dict[str, str]] = {
    "O": "0",
    "Q": "0",
    "I": "1",
    "L": "1",
}

REGEX_CARRO: Final[re.Pattern[str]] = re.compile(r"^[A-Z]{3}[0-9]{3}$")
REGEX_MOTO: Final[re.Pattern[str]] = re.compile(r"^[A-Z]{3}[0-9]{2}[A-Z]$")


@dataclass(frozen=True)
class ValidationResult:
    """Resultado detallado de la validación y corrección sintáctica."""

    is_valid: bool
    original_text: str
    corrected_plate: str
    vehicle_type: VehicleType
    corrections: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ConsensusResult:
    """Resultado de consenso emitido por el buffer de votación temporal."""

    plate: str
    vehicle_type: VehicleType
    mean_confidence: float
    votes: int
    window_size: int
    timestamp: float


class PlateValidator:
    """Validador sintáctico y heurístico de matrículas colombianas."""

    @staticmethod
    def clean_text(raw_text: str) -> str:
        """Limpia caracteres ajenos al allowlist alfanumérico en mayúsculas."""
        if not raw_text:
            return ""
        allowlist_set = set(OCR_ALLOWLIST)
        return "".join(c for c in raw_text.upper() if c in allowlist_set)

    @classmethod
    def correct_and_validate(cls, raw_text: str) -> ValidationResult:
        """Aplica corrección heurística posicional y valida formato colombiano.

        Estructura de matrícula colombiana (6 caracteres):
        - Posiciones 0, 1, 2: Letras obligatorias (AAA)
        - Posiciones 3, 4: Dígitos obligatorios (00)
        - Posición 5: Dígito para CARRO (0) o Letra para MOTO (A)
        """
        cleaned = cls.clean_text(raw_text)

        # La placa oficial colombiana consta estrictamente de 6 caracteres
        if len(cleaned) != 6:
            return ValidationResult(
                is_valid=False,
                original_text=raw_text,
                corrected_plate=cleaned,
                vehicle_type="DESCONOCIDO",
                corrections=["Longitud inválida (requiere exactamente 6 caracteres)"],
            )

        chars = list(cleaned)
        corrections: list[str] = []

        # 1. Posiciones 0, 1, 2: Deben ser letras (A-Z)
        for i in range(3):
            ch = chars[i]
            if ch.isdigit() and ch in DIGIT_TO_LETTER:
                letra_corregida = DIGIT_TO_LETTER[ch]
                chars[i] = letra_corregida
                corrections.append(f"Pos {i}: '{ch}' -> '{letra_corregida}'")
            elif not chars[i].isalpha():
                return ValidationResult(
                    is_valid=False,
                    original_text=raw_text,
                    corrected_plate="".join(chars),
                    vehicle_type="DESCONOCIDO",
                    corrections=[f"Posición {i} no es letra válida: '{chars[i]}'"],
                )

        # 2. Posiciones 3, 4: Deben ser dígitos (0-9)
        for i in (3, 4):
            ch = chars[i]
            if ch.isalpha() and ch in LETTER_TO_DIGIT:
                num_corregido = LETTER_TO_DIGIT[ch]
                chars[i] = num_corregido
                corrections.append(f"Pos {i}: '{ch}' -> '{num_corregido}'")
            elif not chars[i].isdigit():
                return ValidationResult(
                    is_valid=False,
                    original_text=raw_text,
                    corrected_plate="".join(chars),
                    vehicle_type="DESCONOCIDO",
                    corrections=[f"Posición {i} no es dígito válido: '{chars[i]}'"],
                )

        # 3. Posición 5: Define si es CARRO (dígito) o MOTO (letra)
        pos5 = chars[5]

        # Caso 3.1: Letras de confusión inequívocas como dígitos (O, Q, I, L no existen como sufijo de moto)
        if pos5 in POS5_CONFUSED_DIGITS:
            num_corregido = POS5_CONFUSED_DIGITS[pos5]
            chars[5] = num_corregido
            corrections.append(f"Pos 5: '{pos5}' -> '{num_corregido}' (CARRO)")
            placa_final = "".join(chars)
            if REGEX_CARRO.match(placa_final):
                return ValidationResult(
                    is_valid=True,
                    original_text=raw_text,
                    corrected_plate=placa_final,
                    vehicle_type="CARRO",
                    corrections=corrections,
                )

        # Caso 3.2: Ya es dígito -> Es un CARRO
        if pos5.isdigit():
            placa_final = "".join(chars)
            if REGEX_CARRO.match(placa_final):
                return ValidationResult(
                    is_valid=True,
                    original_text=raw_text,
                    corrected_plate=placa_final,
                    vehicle_type="CARRO",
                    corrections=corrections,
                )

        # Caso 3.3: Es letra -> Es una MOTO
        if pos5.isalpha():
            placa_final = "".join(chars)
            if REGEX_MOTO.match(placa_final):
                return ValidationResult(
                    is_valid=True,
                    original_text=raw_text,
                    corrected_plate=placa_final,
                    vehicle_type="MOTO",
                    corrections=corrections,
                )

        return ValidationResult(
            is_valid=False,
            original_text=raw_text,
            corrected_plate="".join(chars),
            vehicle_type="DESCONOCIDO",
            corrections=["Posición 5 incompatible con formato Carro (dígito) o Moto (letra)"],
        )

    @classmethod
    def classify_type(cls, plate: str) -> VehicleType:
        """Determina el tipo de vehículo aplicando previamente corrección posicional."""
        res = cls.correct_and_validate(plate)
        return res.vehicle_type if res.is_valid else "DESCONOCIDO"


class PlateVoterBuffer:
    """Buffer de votación temporal multi-cuadro para estabilización de lecturas."""

    def __init__(
        self,
        window_size: int = VOTER_WINDOW_SIZE,
        min_matches: int = VOTER_MIN_MATCHES,
        cooldown_seconds: float = COOLDOWN_SECONDS,
        min_confidence: float = 0.35,
    ) -> None:
        self.window_size = window_size
        self.min_matches = min_matches
        self.cooldown_seconds = cooldown_seconds
        self.min_confidence = min_confidence

        # Buffer FIFO de lecturas validadas: (placa, tipo, confianza, timestamp)
        self._buffer: deque[tuple[str, VehicleType, float, float]] = deque(maxlen=window_size)
        # Registro de timestamps de últimas emisiones confirmadas por placa
        self._recent_emissions: dict[str, float] = {}

    def add_observation(
        self,
        raw_text: str,
        confidence: float,
        timestamp: float | None = None,
    ) -> ConsensusResult | None:
        """Recibe una lectura de cuadro y emite consenso si se alcanza el umbral de votos."""
        t_now = time.time() if timestamp is None else timestamp

        # Validar y corregir sintaxis
        val_res = PlateValidator.correct_and_validate(raw_text)
        if not val_res.is_valid or confidence < self.min_confidence:
            return None

        # Insertar observación válida al buffer deslizante
        self._buffer.append((val_res.corrected_plate, val_res.vehicle_type, confidence, t_now))

        # Contabilizar votos en la ventana actual
        plates_in_window = [item[0] for item in self._buffer]
        counts = Counter(plates_in_window)
        most_common_plate, vote_count = counts.most_common(1)[0]

        # Verificar si alcanza el consenso mínimo
        if vote_count >= self.min_matches:
            # Comprobar ventana de enfriamiento (cooldown) contra duplicados
            last_emitted_time = self._recent_emissions.get(most_common_plate, 0.0)
            if t_now - last_emitted_time < self.cooldown_seconds:
                # Placa en período de enfriamiento: suprimir emisión duplicada
                return None

            # Recuperar tipos y confianzas de los votos que respaldan a la placa ganadora
            matching_obs = [item for item in self._buffer if item[0] == most_common_plate]
            vtype = matching_obs[-1][1]
            mean_conf = sum(item[2] for item in matching_obs) / len(matching_obs)

            # Registrar emisión en el mapa de enfriamiento
            self._recent_emissions[most_common_plate] = t_now

            return ConsensusResult(
                plate=most_common_plate,
                vehicle_type=vtype,
                mean_confidence=round(mean_conf, 4),
                votes=vote_count,
                window_size=len(self._buffer),
                timestamp=t_now,
            )

        return None

    def clear(self) -> None:
        """Reinicia el buffer y el historial de enfriamiento."""
        self._buffer.clear()
        self._recent_emissions.clear()

    def get_buffer_state(self) -> list[dict[str, object]]:
        """Devuelve una instantánea del estado del buffer para la telemetría en vivo."""
        return [
            {
                "placa": p,
                "tipo": t,
                "confianza": c,
                "timestamp": ts,
            }
            for p, t, c, ts in self._buffer
        ]
