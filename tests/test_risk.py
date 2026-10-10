"""
Tests de la logica critica de negocio: la clasificacion en niveles de
riesgo. Es la unica regla "propia" de la API (el resto es el modelo),
asi que vale la pena fijar sus limites exactos.
"""

from __future__ import annotations

import pytest

from app.risk import UMBRAL_LOW_MEDIUM, UMBRAL_MEDIUM_HIGH, calcular_risk_level


@pytest.mark.parametrize(
    "probabilidad, esperado",
    [
        (0.0, "LOW"),
        (0.29, "LOW"),
        (UMBRAL_LOW_MEDIUM, "MEDIUM"),   # el limite pertenece al nivel de arriba
        (0.45, "MEDIUM"),
        (0.59, "MEDIUM"),
        (UMBRAL_MEDIUM_HIGH, "HIGH"),    # idem
        (0.85, "HIGH"),
        (1.0, "HIGH"),
    ],
)
def testeo_risk_level_respeta_los_umbrales(probabilidad, esperado):
    assert calcular_risk_level(probabilidad) == esperado


def testeo_umbrales_estan_ordenados():
    """Si alguien cambia los umbrales y los desordena, la clasificacion deja de tener sentido."""
    assert 0 < UMBRAL_LOW_MEDIUM < UMBRAL_MEDIUM_HIGH < 1
