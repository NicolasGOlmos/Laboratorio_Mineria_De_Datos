"""
Carga el modelo registrado desde el MLflow Model Registry (en DagsHub)
una sola vez, al arrancar la aplicacion, y lo deja en memoria para
todas las requests siguientes. No se reconecta ni recarga en cada
prediccion: eso seria lento y, mas importante, la version del modelo
que sirve tiene que quedar fija mientras el servicio esta arriba
(sino un deploy de un modelo nuevo podria cambiar el comportamiento
de la API a mitad de una sesion de alguien).

Toda la configuracion sale de variables de entorno (regla de
seguridad del proyecto: nada de tokens o URLs hardcodeadas):

    MLFLOW_TRACKING_URI       -> URL del servidor MLflow (DagsHub)
    MLFLOW_TRACKING_USERNAME  -> usuario de DagsHub
    MLFLOW_TRACKING_PASSWORD  -> token de DagsHub
    MODEL_NAME                -> nombre del modelo registrado (default: customer-churn-classifier)
    MODEL_VERSION             -> version puntual a cargar. Si no se
                                  especifica, se usa la ULTIMA version
                                  registrada (no un "stage", porque
                                  los stages estan deprecados en MLflow).
"""

from __future__ import annotations

import os

import mlflow
from mlflow.tracking import MlflowClient

DEFAULT_MODEL_NAME = "customer-churn-classifier"


class ModeloNoDisponibleError(Exception):
    """Se lanza cuando el modelo no pudo cargarse (red, credenciales, no existe, etc.)."""


class ModeloCargado:
    """Envoltorio simple: guarda el pipeline ya cargado + metadata util para /health."""

    def __init__(self, pipeline, model_name: str, model_version: str, run_id: str | None = None):
        self.pipeline = pipeline
        self.model_name = model_name
        self.model_version = model_version
        # Run de MLflow con el que se entreno esta version del modelo.
        self.run_id = run_id


def _resolver_version(client: MlflowClient, model_name: str) -> str:
    """Si MODEL_VERSION no esta seteada, busca la version mas alta registrada.

    Ojo: segun la version/backend de MLflow, ModelVersion.version puede
    venir como int o como str. Forzamos str siempre, porque es lo que
    espera el contrato de la API (PredictResponse.model_version: str)
    y lo que se necesita para armar el model_uri ("models:/nombre/N").
    """
    version_env = os.environ.get("MODEL_VERSION")
    if version_env:
        return str(version_env)

    versiones = client.search_model_versions(f"name='{model_name}'")
    if not versiones:
        raise ModeloNoDisponibleError(
            f"No hay ninguna version registrada para el modelo '{model_name}'."
        )
    ultima = max(versiones, key=lambda v: int(v.version))
    return str(ultima.version)


def cargar_modelo() -> ModeloCargado:
    """Conecta a MLflow (via variables de entorno) y trae el modelo registrado.

    Lanza ModeloNoDisponibleError si algo falla, para que quien llama
    (el startup de FastAPI) decida que hacer con eso en vez de que la
    excepcion original de MLflow se propague tal cual.
    """
    model_name = os.environ.get("MODEL_NAME", DEFAULT_MODEL_NAME)

    try:
        client = MlflowClient()
        model_version = _resolver_version(client, model_name)
        model_uri = f"models:/{model_name}/{model_version}"
        pipeline = mlflow.sklearn.load_model(model_uri)
    except ModeloNoDisponibleError:
        raise
    except Exception as exc:  # cualquier error de red, auth, modelo inexistente, etc.
        raise ModeloNoDisponibleError(
            f"No se pudo cargar el modelo '{model_name}' desde MLflow: {exc}"
        ) from exc

    # El run de origen es solo metadata de trazabilidad: si no se puede
    # leer, no vale la pena tirar abajo un modelo que ya cargo bien.
    try:
        run_id = client.get_model_version(model_name, model_version).run_id
    except Exception:
        run_id = None

    return ModeloCargado(
        pipeline=pipeline,
        model_name=model_name,
        model_version=model_version,
        run_id=run_id,
    )
