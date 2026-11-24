"""
Logging estructurado (una linea JSON por evento).

Por que JSON: cada evento sale como un objeto con campos fijos
(timestamp, level, message, request_id, latency_ms...), asi se puede
filtrar con herramientas simples (grep, jq) o mandar despues a
cualquier plataforma de logs sin parsear texto libre.

Variable de entorno:
    LOG_LEVEL -> INFO por defecto. DEBUG muestra tambien los /health.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone

NOMBRE_LOGGER = "churn-api"

# Atributos que trae todo LogRecord de fabrica. Lo que NO este aca lo
# pusimos nosotros via extra={...} y es lo que queremos en el JSON.
_CAMPOS_ESTANDAR = set(logging.LogRecord("", 0, "", 0, "", (), None).__dict__) | {
    "message",
    "asctime",
}


class FormatoJson(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        evento = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(
                timespec="milliseconds"
            ),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for clave, valor in record.__dict__.items():
            if clave not in _CAMPOS_ESTANDAR:
                evento[clave] = valor
        if record.exc_info:
            evento["exception"] = self.formatException(record.exc_info)
        return json.dumps(evento, default=str, ensure_ascii=False)


def configurar_logging() -> logging.Logger:
    """Deja el logger de la API listo. Es idempotente: se puede llamar varias veces."""
    logger = logging.getLogger(NOMBRE_LOGGER)
    logger.setLevel(os.environ.get("LOG_LEVEL", "INFO").upper())

    if not any(isinstance(h.formatter, FormatoJson) for h in logger.handlers):
        handler = logging.StreamHandler()
        handler.setFormatter(FormatoJson())
        logger.addHandler(handler)

    return logger
