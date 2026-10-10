"""
API de inferencia para el modelo de churn.

Endpoints:
  GET  /health   -> confirma que el servicio esta operativo y que
                     version de modelo tiene cargada.
  POST /predict  -> recibe las caracteristicas de UN cliente, corre
                     el pipeline completo (preprocesamiento + modelo)
                     y devuelve prediccion + probabilidad + nivel de
                     riesgo.

Se ejecuta con: uvicorn app.main:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, HTTPException

from app.model_loader import ModeloCargado, ModeloNoDisponibleError, cargar_modelo
from app.risk import calcular_risk_level
from app.schemas import ClientePredictRequest, HealthResponse, PredictResponse

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("churn-api")

# Guardamos el modelo cargado en un dict mutable (en vez de una
# variable global simple) para poder reemplazarlo facilmente desde
# los tests, sin reiniciar el proceso.
estado: dict[str, ModeloCargado | None] = {"modelo": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Se ejecuta UNA sola vez, al arrancar la aplicacion.
    try:
        estado["modelo"] = cargar_modelo()
        logger.info(
            "Modelo cargado: %s v%s",
            estado["modelo"].model_name,
            estado["modelo"].model_version,
        )
    except ModeloNoDisponibleError as exc:
        # No tiramos la aplicacion abajo: preferimos que arranque en
        # un estado "degradado" y que /health lo refleje, en vez de
        # que el contenedor entero crashee y Docker lo reinicie en
        # loop sin dar ninguna pista de que paso.
        logger.error("No se pudo cargar el modelo al arrancar: %s", exc)
        estado["modelo"] = None
    yield
    estado["modelo"] = None


app = FastAPI(
    title="Customer Churn API",
    description="Servicio de prediccion de abandono de clientes de telecomunicaciones.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
def health():
    modelo = estado["modelo"]
    if modelo is None:
        return HealthResponse(
            status="degraded",
            model_name="desconocido",
            model_version=None,
        )
    return HealthResponse(
        status="ok",
        model_name=modelo.model_name,
        model_version=modelo.model_version,
    )


@app.post("/predict", response_model=PredictResponse)
def predict(cliente: ClientePredictRequest):
    modelo = estado["modelo"]
    if modelo is None:
        # 503: el servicio esta arriba pero no puede cumplir el
        # pedido porque no tiene modelo cargado. Distinto de un 500,
        # que implicaria un bug nuestro procesando la request.
        raise HTTPException(
            status_code=503,
            detail="El modelo no esta disponible en este momento. Reintentar mas tarde.",
        )

    # El pipeline espera un DataFrame con las mismas columnas que vio
    # en entrenamiento. customerID se excluye a proposito, igual que
    # en el entrenamiento (no es una feature).
    datos_cliente = cliente.model_dump(exclude={"customerID"})
    fila = pd.DataFrame([datos_cliente])

    try:
        probabilidad = float(modelo.pipeline.predict_proba(fila)[0, 1])
    except Exception as exc:
        logger.exception("Error corriendo el pipeline de inferencia")
        raise HTTPException(
            status_code=500,
            detail=f"Error interno al ejecutar el pipeline de inferencia: {exc}",
        ) from exc

    prediccion = "CHURN" if probabilidad >= 0.5 else "NO_CHURN"
    nivel_riesgo = calcular_risk_level(probabilidad)

    return PredictResponse(
        prediction=prediccion,
        probability=round(probabilidad, 4),
        risk_level=nivel_riesgo,
        model_version=modelo.model_version,
    )
