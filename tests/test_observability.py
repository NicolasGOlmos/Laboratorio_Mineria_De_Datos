"""
Tests de observabilidad: request_id, logs estructurados, latencia y
datos de trazabilidad en /health.
"""

from __future__ import annotations

import json
import logging

from app.logging_config import FormatoJson
from tests.test_api import CLIENTE_VALIDO


def _eventos(caplog, mensaje: str):
    return [r for r in caplog.records if r.getMessage() == mensaje]


def testeo_response_trae_header_request_id_y_tiempo(cliente_api):
    respuesta = cliente_api.post("/predict", json=CLIENTE_VALIDO)
    assert respuesta.headers["X-Request-ID"]
    assert float(respuesta.headers["X-Process-Time-Ms"]) >= 0


def testeo_request_id_del_body_coincide_con_el_header(cliente_api):
    respuesta = cliente_api.post("/predict", json=CLIENTE_VALIDO)
    assert respuesta.json()["request_id"] == respuesta.headers["X-Request-ID"]


def testeo_respeta_el_request_id_que_manda_el_cliente(cliente_api):
    respuesta = cliente_api.post(
        "/predict", json=CLIENTE_VALIDO, headers={"X-Request-ID": "mi-trace-123"}
    )
    assert respuesta.headers["X-Request-ID"] == "mi-trace-123"
    assert respuesta.json()["request_id"] == "mi-trace-123"


def testeo_descarta_un_request_id_invalido_y_genera_otro(cliente_api):
    respuesta = cliente_api.post(
        "/predict", json=CLIENTE_VALIDO, headers={"X-Request-ID": "con espacios y {llaves}"}
    )
    assert respuesta.headers["X-Request-ID"] != "con espacios y {llaves}"
    assert len(respuesta.headers["X-Request-ID"]) > 0


def testeo_predict_deja_logs_de_inicio_inferencia_y_fin_con_el_mismo_request_id(
    cliente_api, caplog
):
    with caplog.at_level(logging.DEBUG, logger="churn-api"):
        respuesta = cliente_api.post(
            "/predict", json=CLIENTE_VALIDO, headers={"X-Request-ID": "trace-abc"}
        )
    assert respuesta.status_code == 200

    inicio = _eventos(caplog, "request_started")
    inferencia = _eventos(caplog, "inference")
    fin = _eventos(caplog, "request_finished")

    assert inicio and inferencia and fin
    assert inicio[-1].request_id == inferencia[-1].request_id == fin[-1].request_id == "trace-abc"
    assert fin[-1].status_code == 200
    assert fin[-1].latency_ms >= 0
    assert inferencia[-1].model_version == respuesta.json()["model_version"]
    assert inferencia[-1].inference_ms >= 0


def testeo_el_log_de_inferencia_no_incluye_datos_del_cliente(cliente_api, caplog):
    with caplog.at_level(logging.DEBUG, logger="churn-api"):
        cliente_api.post("/predict", json=CLIENTE_VALIDO)
    formateador = FormatoJson()
    for registro in caplog.records:
        linea = formateador.format(registro)
        assert "7590-VHVEG" not in linea
        assert "Electronic check" not in linea


def testeo_una_entrada_invalida_se_loguea_como_warning_con_422(cliente_api, caplog):
    with caplog.at_level(logging.DEBUG, logger="churn-api"):
        respuesta = cliente_api.post("/predict", json={**CLIENTE_VALIDO, "Contract": "Weekly"})
    assert respuesta.status_code == 422

    fin = _eventos(caplog, "request_finished")[-1]
    assert fin.status_code == 422
    assert fin.levelno == logging.WARNING


def testeo_health_no_ensucia_los_logs_en_nivel_info(cliente_api, caplog):
    with caplog.at_level(logging.INFO, logger="churn-api"):
        cliente_api.get("/health")
    assert not _eventos(caplog, "request_finished")


def testeo_los_logs_salen_como_json_valido(cliente_api, caplog):
    with caplog.at_level(logging.DEBUG, logger="churn-api"):
        cliente_api.post("/predict", json=CLIENTE_VALIDO)
    formateador = FormatoJson()
    linea = formateador.format(_eventos(caplog, "request_finished")[-1])
    datos = json.loads(linea)
    assert {"timestamp", "level", "message", "request_id", "latency_ms", "status_code"} <= set(datos)


def testeo_health_expone_trazabilidad_run_commit_y_dataset(cliente_api, monkeypatch):
    monkeypatch.setenv("GIT_COMMIT", "abc1234")
    monkeypatch.setenv("DATASET_VERSION", "md5-de-prueba")

    cuerpo = cliente_api.get("/health").json()

    assert cuerpo["run_id"]  # el run de MLflow del que salio el modelo
    assert cuerpo["git_commit"] == "abc1234"
    assert cuerpo["dataset_version"] == "md5-de-prueba"


def testeo_health_sin_metadata_de_build_devuelve_none(cliente_api, monkeypatch):
    monkeypatch.delenv("GIT_COMMIT", raising=False)
    monkeypatch.setenv("DATASET_VERSION", "unknown")

    cuerpo = cliente_api.get("/health").json()

    assert cuerpo["git_commit"] is None
    assert cuerpo["dataset_version"] is None


def testeo_error_de_modelo_no_disponible_se_loguea_con_request_id(
    modelo_registrado_local, monkeypatch, caplog
):
    for clave, valor in modelo_registrado_local.items():
        monkeypatch.setenv(clave, valor)
    monkeypatch.setenv("MODEL_NAME", "modelo-que-no-existe")
    monkeypatch.delenv("MODEL_VERSION", raising=False)

    from fastapi.testclient import TestClient

    from app.main import app

    with caplog.at_level(logging.DEBUG, logger="churn-api"):
        with TestClient(app) as cliente:
            respuesta = cliente.post(
                "/predict", json=CLIENTE_VALIDO, headers={"X-Request-ID": "trace-503"}
            )

    assert respuesta.status_code == 503
    no_disponible = _eventos(caplog, "model_unavailable")
    assert no_disponible and no_disponible[-1].request_id == "trace-503"
    assert no_disponible[-1].levelno == logging.ERROR
    assert _eventos(caplog, "model_load_failed")
