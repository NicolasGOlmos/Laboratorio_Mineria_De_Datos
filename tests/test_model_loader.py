"""
Tests de la carga del modelo desde el Model Registry y de como se
comporta la API cuando el modelo NO esta disponible.
"""

from __future__ import annotations

import pytest

from app.model_loader import ModeloNoDisponibleError, cargar_modelo
from tests.test_api import CLIENTE_VALIDO


def testeo_cargar_modelo_trae_pipeline_y_version_como_texto(modelo_registrado_local, monkeypatch):
    for clave, valor in modelo_registrado_local.items():
        monkeypatch.setenv(clave, valor)
    monkeypatch.delenv("MODEL_VERSION", raising=False)

    modelo = cargar_modelo()

    assert modelo.model_name == "customer-churn-classifier-test"
    assert isinstance(modelo.model_version, str)
    assert hasattr(modelo.pipeline, "predict_proba")


def testeo_cargar_modelo_respeta_model_version_fija(modelo_registrado_local, monkeypatch):
    for clave, valor in modelo_registrado_local.items():
        monkeypatch.setenv(clave, valor)
    monkeypatch.setenv("MODEL_VERSION", "1")

    assert cargar_modelo().model_version == "1"


def testeo_cargar_modelo_inexistente_lanza_error_claro(modelo_registrado_local, monkeypatch):
    for clave, valor in modelo_registrado_local.items():
        monkeypatch.setenv(clave, valor)
    monkeypatch.setenv("MODEL_NAME", "modelo-que-no-existe")
    monkeypatch.delenv("MODEL_VERSION", raising=False)

    with pytest.raises(ModeloNoDisponibleError):
        cargar_modelo()


def testeo_api_sin_modelo_health_y_predict_devuelven_503(modelo_registrado_local, monkeypatch):
    """Si el modelo no carga, la API levanta igual pero avisa: /health y /predict dan 503.

    Importante: /health tiene que devolver 503 (no 200 con status=degraded),
    porque es lo que usa el healthcheck de Docker para marcar el contenedor
    como no saludable.
    """
    for clave, valor in modelo_registrado_local.items():
        monkeypatch.setenv(clave, valor)
    monkeypatch.setenv("MODEL_NAME", "modelo-que-no-existe")
    monkeypatch.delenv("MODEL_VERSION", raising=False)

    from fastapi.testclient import TestClient

    from app.main import app

    with TestClient(app) as cliente:
        health = cliente.get("/health")
        assert health.status_code == 503
        assert health.json()["status"] == "degraded"

        predict = cliente.post("/predict", json=CLIENTE_VALIDO)
        assert predict.status_code == 503
