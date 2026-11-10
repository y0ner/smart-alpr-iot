"""Pruebas unitarias para validación colombiana, corrección heurística y PlateVoterBuffer.

Cubre casos borde de confusión morfológica, distinción Carro vs Moto, rechazo de
cadenas inválidas y simulación temporal multi-cuadro con control de enfriamiento.
"""

import pytest

from src.validator import (
    ConsensusResult,
    PlateValidator,
    PlateVoterBuffer,
    ValidationResult,
)


def test_clean_text_strips_invalid_chars() -> None:
    """Verifica la eliminación de caracteres espurios manteniendo el allowlist."""
    assert PlateValidator.clean_text("abc-123") == "ABC123"
    assert PlateValidator.clean_text("XYZ.98A*") == "XYZ98A"
    assert PlateValidator.clean_text("   mto 45d   ") == "MTO45D"
    assert PlateValidator.clean_text("") == ""


@pytest.mark.parametrize(
    "raw_input, expected_plate, expected_type",
    [
        # Caso borde exigido en el curso: número 0 en posición 2 corregido a O
        ("AB0123", "ABO123", "CARRO"),
        # Caso borde exigido: letra O en posición 4 corregida a 0
        ("KLM9O1", "KLM901", "CARRO"),
        # Caso borde exigido: letra B en posición 4 corregida a 8 en formato MOTO
        ("XYZ1BD", "XYZ18D", "MOTO"),
        # Casos estándar limpios
        ("UJR456", "UJR456", "CARRO"),
        ("WJR78A", "WJR78A", "MOTO"),
        # Correcciones adicionales de confusión morfológica
        ("0AB123", "OAB123", "CARRO"),  # Posición 0: 0 -> O
        ("A1C456", "AIC456", "CARRO"),  # Posición 1: 1 -> I
        ("AB5123", "ABS123", "CARRO"),  # Posición 2: 5 -> S
        ("AB8123", "ABB123", "CARRO"),  # Posición 2: 8 -> B
        ("AB6123", "ABG123", "CARRO"),  # Posición 2: 6 -> G
        ("ABCQI3", "ABC013", "CARRO"),  # Posiciones 3 y 4: Q -> 0, I -> 1
        ("ABCLZ3", "ABC123", "CARRO"),  # Posiciones 3 y 4: L -> 1, Z -> 2
        ("ABCSB3", "ABC583", "CARRO"),  # Posiciones 3 y 4: S -> 5, B -> 8
        ("MOT120", "MOT120", "CARRO"),  # Termina en dígito -> CARRO
        ("MOT12O", "MOT120", "CARRO"),  # Termina en O corregible a 0 -> CARRO
    ],
)
def test_corrections_and_validations(
    raw_input: str, expected_plate: str, expected_type: str
) -> None:
    """Verifica que las heurísticas posicionales corrijan con precisión."""
    res: ValidationResult = PlateValidator.correct_and_validate(raw_input)
    assert res.is_valid is True
    assert res.corrected_plate == expected_plate
    assert res.vehicle_type == expected_type
    assert PlateValidator.classify_type(raw_input) == expected_type


@pytest.mark.parametrize(
    "invalid_input",
    [
        "ABC12",        # Menos de 6 caracteres
        "ABC1234",      # Más de 6 caracteres
        "123456",       # Todo números (sin letras al inicio)
        "ABCDEF",       # Todo letras (sin números al centro)
        "AB?123",       # Carácter no salvable
        "",             # Cadena vacía
    ],
)
def test_invalid_plates_rejection(invalid_input: str) -> None:
    """Verifica que matrículas que no cumplen el estándar colombiano sean rechazadas."""
    res: ValidationResult = PlateValidator.correct_and_validate(invalid_input)
    assert res.is_valid is False
    assert res.vehicle_type == "DESCONOCIDO"
    assert PlateValidator.classify_type(invalid_input) == "DESCONOCIDO"


def test_voter_buffer_stream_burst_simulation() -> None:
    """Simula una ráfaga continua de 10 cuadros de video con ruido y cooldown."""
    buffer = PlateVoterBuffer(
        window_size=7,
        min_matches=4,
        cooldown_seconds=15.0,
        min_confidence=0.40,
    )

    t0 = 1000.0  # Timestamp base en segundos

    # Cuadros 1 a 3: Lecturas dispersas o con baja confianza
    # Frame 1: Lectura errónea
    r1 = buffer.add_observation("XYZ999", confidence=0.70, timestamp=t0 + 0.1)
    assert r1 is None  # Apenas 1 voto

    # Frame 2: Confianza insuficiente (< min_confidence=0.40) -> debe ser ignorada
    r2 = buffer.add_observation("ABC123", confidence=0.30, timestamp=t0 + 0.2)
    assert r2 is None

    # Frame 3: Lectura ruidosa inválida
    r3 = buffer.add_observation("INVALID_PLATE", confidence=0.85, timestamp=t0 + 0.3)
    assert r3 is None

    # Cuadros 4 a 7: Ráfaga consistente de "ABC123" (con alguna pequeña variación corregible)
    # Frame 4: ABC123 (Voto 1)
    assert buffer.add_observation("ABC123", confidence=0.88, timestamp=t0 + 0.4) is None
    # Frame 5: ABC123 (Voto 2)
    assert buffer.add_observation("ABC123", confidence=0.92, timestamp=t0 + 0.5) is None
    # Frame 6: ABC123 con O en vez de 0 -> se corrige a ABC123 (Voto 3)
    assert buffer.add_observation("ABC123", confidence=0.90, timestamp=t0 + 0.6) is None
    # Frame 7: ABC123 (Voto 4 -> ALCANZA min_matches=4!)
    consensus: ConsensusResult | None = buffer.add_observation(
        "ABC123", confidence=0.94, timestamp=t0 + 0.7
    )
    assert consensus is not None
    assert consensus.plate == "ABC123"
    assert consensus.vehicle_type == "CARRO"
    assert consensus.votes == 4
    assert consensus.mean_confidence >= 0.90

    # Cuadros 8 a 10: Continúa pasando el mismo vehículo frente a la cámara
    # Frame 8 (t0 + 0.8s): Dentro del período de enfriamiento (15s) -> debe retornar None
    assert buffer.add_observation("ABC123", confidence=0.95, timestamp=t0 + 0.8) is None
    # Frame 9 (t0 + 1.0s): Bloqueado por cooldown
    assert buffer.add_observation("ABC123", confidence=0.93, timestamp=t0 + 1.0) is None

    # Simular paso de 20 segundos (> cooldown de 15s)
    # Al transcurrir el cooldown, la primera lectura que acumule el umbral emite consenso
    t_post_cooldown = t0 + 20.0
    cons_reentry = buffer.add_observation("ABC123", confidence=0.91, timestamp=t_post_cooldown + 0.1)

    assert cons_reentry is not None
    assert cons_reentry.plate == "ABC123"
    assert cons_reentry.timestamp == t_post_cooldown + 0.1

    # Las siguientes lecturas consecutivas inmediatas vuelven a entrar en cooldown
    assert buffer.add_observation("ABC123", confidence=0.92, timestamp=t_post_cooldown + 0.2) is None


def test_voter_buffer_clear_and_state() -> None:
    """Verifica que el método clear restablezca el buffer y que get_buffer_state funcione."""
    buffer = PlateVoterBuffer(window_size=5, min_matches=3)
    buffer.add_observation("MTO45A", confidence=0.88, timestamp=100.0)
    buffer.add_observation("MTO45A", confidence=0.90, timestamp=100.1)

    state = buffer.get_buffer_state()
    assert len(state) == 2
    assert state[0]["placa"] == "MTO45A"

    buffer.clear()
    assert len(buffer.get_buffer_state()) == 0
    assert len(buffer._recent_emissions) == 0
