"""
API de inferencia para el modelo de churn.

Endpoints:
  GET  /health   -> confirma que el servicio esta operativo y que
                     version de modelo tiene cargada (y de que commit,
                     run y dataset viene).
  POST /predict  -> recibe las caracteristicas de UN cliente, corre
                     el pipeline completo (preprocesamiento + modelo)
                     y devuelve prediccion + probabilidad + nivel de
                     riesgo.

Observabilidad: cada request recibe un request_id (el que mande el
cliente en el header X-Request-ID, o uno nuevo), que se devuelve en la
respuesta y aparece en TODOS los logs de esa request. Los logs salen
en JSON, una linea por evento.

Se ejecuta con: uvicorn app.main:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import logging
import os
import re
import time
from contextlib import asynccontextmanager
from uuid import uuid4

import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from app.logging_config import configurar_logging
from app.model_loader import ModeloCargado, ModeloNoDisponibleError, cargar_modelo
from app.risk import calcular_risk_level
from app.schemas import ClientePredictRequest, HealthResponse, PredictResponse

logger = configurar_logging()

# Solo aceptamos request_id "razonables" que mande el cliente: evita que
# alguien meta basura (o textos enormes) en los logs y en los headers.
_REQUEST_ID_VALIDO = re.compile(r"^[A-Za-z0-9._-]{1,64}$")

# Guardamos el modelo cargado en un dict mutable (en vez de una
# variable global simple) para poder reemplazarlo facilmente desde
# los tests, sin reiniciar el proceso.
estado: dict[str, ModeloCargado | None] = {"modelo": None}


def _metadata_de_build() -> dict[str, str | None]:
    """Commit y version del dataset con los que se construyo la imagen (los inyecta el build)."""

    def _leer(nombre: str) -> str | None:
        valor = os.environ.get(nombre, "").strip()
        return valor if valor and valor != "unknown" else None

    return {"git_commit": _leer("GIT_COMMIT"), "dataset_version": _leer("DATASET_VERSION")}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Se ejecuta UNA sola vez, al arrancar la aplicacion.
    try:
        estado["modelo"] = cargar_modelo()
        logger.info(
            "model_loaded",
            extra={
                "model_name": estado["modelo"].model_name,
                "model_version": estado["modelo"].model_version,
                "run_id": estado["modelo"].run_id,
                **_metadata_de_build(),
            },
        )
    except ModeloNoDisponibleError as exc:
        # No tiramos la aplicacion abajo: preferimos que arranque en
        # un estado "degradado" y que /health lo refleje, en vez de
        # que el contenedor entero crashee y Docker lo reinicie en
        # loop sin dar ninguna pista de que paso.
        logger.error("model_load_failed", extra={"error": str(exc)})
        estado["modelo"] = None
    yield
    estado["modelo"] = None


app = FastAPI(
    title="Customer Churn API",
    description="Servicio de prediccion de abandono de clientes de telecomunicaciones.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.middleware("http")
async def trazar_request(request: Request, call_next):
    """Asigna request_id, mide la latencia y deja log de inicio y fin de cada request."""
    request_id = request.headers.get("X-Request-ID", "")
    if not _REQUEST_ID_VALIDO.match(request_id):
        request_id = str(uuid4())
    request.state.request_id = request_id

    # /health lo consulta el healthcheck de Docker cada pocos segundos:
    # lo mandamos a DEBUG para que no tape los eventos que importan.
    es_health = request.url.path == "/health"
    nivel_normal = logging.DEBUG if es_health else logging.INFO
    base = {"request_id": request_id, "method": request.method, "path": request.url.path}

    inicio = time.perf_counter()
    logger.log(nivel_normal, "request_started", extra=base)

    try:
        response = await call_next(request)
    except Exception:
        # Algo no previsto exploto: lo dejamos en el log con el traceback
        # y devolvemos un 500 prolijo (con el request_id para rastrearlo).
        latencia_ms = round((time.perf_counter() - inicio) * 1000, 2)
        logger.exception("request_failed", extra={**base, "latency_ms": latencia_ms})
        response = JSONResponse(
            status_code=500,
            content={"detail": "Error interno del servidor.", "request_id": request_id},
        )

    latencia_ms = round((time.perf_counter() - inicio) * 1000, 2)
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Process-Time-Ms"] = str(latencia_ms)

    if response.status_code >= 500:
        nivel = logging.ERROR
    elif response.status_code >= 400:
        nivel = logging.WARNING
    else:
        nivel = nivel_normal
    logger.log(
        nivel,
        "request_finished",
        extra={**base, "status_code": response.status_code, "latency_ms": latencia_ms},
    )
    return response


@app.get(
    "/health",
    response_model=HealthResponse,
    responses={503: {"model": HealthResponse, "description": "El modelo no esta cargado (servicio degradado)."}},
)
def health():
    """Readiness: 200 solo si el servicio PUEDE predecir; 503 si no tiene modelo.

    Es lo que consulta el healthcheck de Docker. Si devolviera 200 con el
    modelo caido, Docker marcaria el contenedor como saludable aunque no
    pueda responder ninguna prediccion.
    """
    modelo = estado["modelo"]
    metadata = _metadata_de_build()
    if modelo is None:
        degradado = HealthResponse(
            status="degraded",
            model_name="desconocido",
            model_version=None,
            **metadata,
        )
        return JSONResponse(status_code=503, content=degradado.model_dump())
    return HealthResponse(
        status="ok",
        model_name=modelo.model_name,
        model_version=modelo.model_version,
        run_id=modelo.run_id,
        **metadata,
    )


@app.post("/predict", response_model=PredictResponse)
def predict(request: Request, cliente: ClientePredictRequest):
    request_id = request.state.request_id
    modelo = estado["modelo"]
    if modelo is None:
        # 503: el servicio esta arriba pero no puede cumplir el
        # pedido porque no tiene modelo cargado. Distinto de un 500,
        # que implicaria un bug nuestro procesando la request.
        logger.error("model_unavailable", extra={"request_id": request_id})
        raise HTTPException(
            status_code=503,
            detail="El modelo no esta disponible en este momento. Reintentar mas tarde.",
        )

    # El pipeline espera un DataFrame con las mismas columnas que vio
    # en entrenamiento. customerID se excluye a proposito, igual que
    # en el entrenamiento (no es una feature).
    datos_cliente = cliente.model_dump(exclude={"customerID"})
    fila = pd.DataFrame([datos_cliente])

    inicio = time.perf_counter()
    try:
        probabilidad = float(modelo.pipeline.predict_proba(fila)[0, 1])
    except Exception as exc:
        logger.exception("inference_failed", extra={"request_id": request_id})
        raise HTTPException(
            status_code=500,
            # El detalle real queda solo en el log (con traceback y request_id);
            # al consumidor se le devuelve un mensaje controlado.
            detail={"message": "Error interno al ejecutar la inferencia.", "request_id": request_id},
        ) from exc
    inferencia_ms = round((time.perf_counter() - inicio) * 1000, 2)

    prediccion = "CHURN" if probabilidad >= 0.5 else "NO_CHURN"
    nivel_riesgo = calcular_risk_level(probabilidad)

    # Ojo: NO se loguean los datos del cliente (son datos personales);
    # solo el resultado y lo necesario para trazar la solicitud.
    logger.info(
        "inference",
        extra={
            "request_id": request_id,
            "model_version": modelo.model_version,
            "prediction": prediccion,
            "probability": round(probabilidad, 4),
            "risk_level": nivel_riesgo,
            "inference_ms": inferencia_ms,
        },
    )

    return PredictResponse(
        prediction=prediccion,
        probability=round(probabilidad, 4),
        risk_level=nivel_riesgo,
        model_version=modelo.model_version,
        request_id=request_id,
    )
