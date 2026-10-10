"""
Tests de la API, cubriendo lo pedido en la consigna:
  - Carga del modelo o pipeline.
  - Ejecucion de una prediccion valida.
  - Estructura y tipos de la respuesta.
  - Endpoint /health.
  - Endpoint /predict.
  - Entradas invalidas y comportamiento esperado ante errores.
"""

from __future__ import annotations

CLIENTE_VALIDO = {
    "customerID": "7590-VHVEG",
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 1,
    "PhoneService": "No",
    "MultipleLines": "No phone service",
    "InternetService": "DSL",
    "OnlineSecurity": "No",
    "OnlineBackup": "Yes",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 29.85,
    "TotalCharges": 29.85,
}


# --- /health -----------------------------------------------------------

def testeo_health_devuelve_ok_y_modelo_cargado(cliente_api):
    """Si el modelo se cargo bien al arrancar, /health tiene que decir status=ok."""
    respuesta = cliente_api.get("/health")

    assert respuesta.status_code == 200
    cuerpo = respuesta.json()
    assert cuerpo["status"] == "ok"
    assert cuerpo["model_name"] == "customer-churn-classifier-test"
    assert cuerpo["model_version"] is not None


# --- /predict: carga del modelo + prediccion valida --------------------

def testeo_predict_con_cliente_valido_devuelve_200(cliente_api):
    """El caso feliz: un cliente con todos los campos correctos."""
    respuesta = cliente_api.post("/predict", json=CLIENTE_VALIDO)
    assert respuesta.status_code == 200


def testeo_predict_estructura_y_tipos_de_la_respuesta(cliente_api):
    """La respuesta tiene que tener exactamente los campos del contrato, con los tipos correctos."""
    respuesta = cliente_api.post("/predict", json=CLIENTE_VALIDO)
    cuerpo = respuesta.json()

    assert set(cuerpo.keys()) == {
        "prediction", "probability", "risk_level", "model_version", "request_id",
    }
    assert cuerpo["prediction"] in {"CHURN", "NO_CHURN"}
    assert isinstance(cuerpo["probability"], float)
    assert 0.0 <= cuerpo["probability"] <= 1.0
    assert cuerpo["risk_level"] in {"LOW", "MEDIUM", "HIGH"}
    assert isinstance(cuerpo["model_version"], str)
    assert isinstance(cuerpo["request_id"], str)
    assert len(cuerpo["request_id"]) > 0


def testeo_predict_genera_un_request_id_distinto_por_llamada(cliente_api):
    """Dos requests identicas no deberian compartir el mismo request_id (es por-request, no fijo)."""
    r1 = cliente_api.post("/predict", json=CLIENTE_VALIDO)
    r2 = cliente_api.post("/predict", json=CLIENTE_VALIDO)
    assert r1.json()["request_id"] != r2.json()["request_id"]


def testeo_predict_sin_total_charges_no_rompe(cliente_api):
    """TotalCharges puede venir vacio (clientes muy nuevos); el pipeline tiene que poder imputarlo."""
    cliente_sin_total = {**CLIENTE_VALIDO, "TotalCharges": None}
    respuesta = cliente_api.post("/predict", json=cliente_sin_total)
    assert respuesta.status_code == 200


def testeo_predict_ignora_customer_id_para_la_prediccion(cliente_api):
    """customerID es solo trazabilidad: dos clientes identicos salvo el ID deben dar la misma prediccion."""
    cliente_a = {**CLIENTE_VALIDO, "customerID": "AAAA-1111"}
    cliente_b = {**CLIENTE_VALIDO, "customerID": "BBBB-2222"}

    respuesta_a = cliente_api.post("/predict", json=cliente_a).json()
    respuesta_b = cliente_api.post("/predict", json=cliente_b).json()

    assert respuesta_a["probability"] == respuesta_b["probability"]


def testeo_predict_funciona_sin_mandar_customer_id(cliente_api):
    """customerID es opcional: la request tiene que funcionar igual sin mandarlo."""
    cliente_sin_id = {k: v for k, v in CLIENTE_VALIDO.items() if k != "customerID"}
    respuesta = cliente_api.post("/predict", json=cliente_sin_id)
    assert respuesta.status_code == 200


# --- Entradas invalidas --------------------------------------------------

def testeo_predict_rechaza_campo_faltante(cliente_api):
    """Si falta un campo obligatorio (ej. Contract), Pydantic debe rechazarlo con 422."""
    cliente_incompleto = {k: v for k, v in CLIENTE_VALIDO.items() if k != "Contract"}
    respuesta = cliente_api.post("/predict", json=cliente_incompleto)
    assert respuesta.status_code == 422


def testeo_predict_rechaza_valor_categorico_invalido(cliente_api):
    """Un valor fuera de la lista permitida (ej. Contract='Weekly') tiene que ser rechazado."""
    cliente_invalido = {**CLIENTE_VALIDO, "Contract": "Weekly"}
    respuesta = cliente_api.post("/predict", json=cliente_invalido)
    assert respuesta.status_code == 422


def testeo_predict_rechaza_tenure_negativo(cliente_api):
    """tenure no puede ser negativo (restriccion ge=0 del schema)."""
    cliente_invalido = {**CLIENTE_VALIDO, "tenure": -5}
    respuesta = cliente_api.post("/predict", json=cliente_invalido)
    assert respuesta.status_code == 422


def testeo_predict_rechaza_monthly_charges_como_texto(cliente_api):
    """MonthlyCharges tiene que ser numerico; si mandan texto no convertible, 422."""
    cliente_invalido = {**CLIENTE_VALIDO, "MonthlyCharges": "gratis"}
    respuesta = cliente_api.post("/predict", json=cliente_invalido)
    assert respuesta.status_code == 422


def testeo_predict_con_body_vacio_devuelve_422(cliente_api):
    """Un body vacio tiene que fallar la validacion, no caerse con un 500."""
    respuesta = cliente_api.post("/predict", json={})
    assert respuesta.status_code == 422
