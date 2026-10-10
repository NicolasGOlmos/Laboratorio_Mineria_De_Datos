"""
Traduce la probabilidad del modelo a un nivel de riesgo categorico
(LOW / MEDIUM / HIGH), que es lo que un CRM puede usar directamente
para decidir una accion (nada / oferta liviana / llamado prioritario).

Justificacion de los umbrales (ver tambien README, seccion "Umbrales
de riesgo"): en la Entrega 1 definimos que priorizamos Recall porque
un falso negativo (no detectar a alguien que se va) sale mas caro que
un falso positivo. Mismo criterio aca: preferimos umbrales que
clasifiquen como HIGH a mas clientes en vez de menos, para no dejar
pasar casos reales de churn, aun a costa de que el equipo de
retencion reciba algunos falsos positivos.

- probability < 0.30          -> LOW     (bajo riesgo, no actuar)
- 0.30 <= probability < 0.60   -> MEDIUM  (seguimiento liviano)
- probability >= 0.60          -> HIGH    (contacto prioritario)

Estos umbrales NO salen de un analisis estadistico del dataset de
esta entrega (eso quedaria para una proxima iteracion, comparando
el costo real de cada accion de retencion contra el valor de vida
del cliente). Por ahora son un punto de partida razonable y
documentado, pensado para facilitar que el equipo de negocio los
ajuste sin tocar codigo del modelo.
"""

from __future__ import annotations

UMBRAL_LOW_MEDIUM = 0.30
UMBRAL_MEDIUM_HIGH = 0.60


def calcular_risk_level(probability: float) -> str:
    if probability < UMBRAL_LOW_MEDIUM:
        return "LOW"
    if probability < UMBRAL_MEDIUM_HIGH:
        return "MEDIUM"
    return "HIGH"
