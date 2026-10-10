"""
Fixtures compartidas para los tests de la API.

La idea central: en vez de mockear mlflow.sklearn.load_model (lo que
dejaria sin probar si la carga real funciona), entrenamos un modelo
de verdad -chiquito y rapido- y lo registramos en un MLflow local
(sqlite, en un directorio temporal). Asi la API carga un modelo REAL
desde un Model Registry REAL, igual que en produccion, solo que
local en vez de contra DagsHub.
"""

from __future__ import annotations

import os
import tempfile

import mlflow
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUMERIC_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges"]
CATEGORICAL_FEATURES = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]

MODEL_NAME_TEST = "customer-churn-classifier-test"


def _dataset_sintetico() -> pd.DataFrame:
    """Un dataset chiquito pero con al menos un ejemplo de cada categoria,
    para que el OneHotEncoder tenga algo que aprender en cada columna."""
    filas = [
        {
            "gender": "Female", "SeniorCitizen": 0, "Partner": "Yes", "Dependents": "No",
            "tenure": 1, "PhoneService": "No", "MultipleLines": "No phone service",
            "InternetService": "DSL", "OnlineSecurity": "No", "OnlineBackup": "Yes",
            "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "No",
            "StreamingMovies": "No", "Contract": "Month-to-month", "PaperlessBilling": "Yes",
            "PaymentMethod": "Electronic check", "MonthlyCharges": 29.85, "TotalCharges": 29.85,
            "Churn": 1,
        },
        {
            "gender": "Male", "SeniorCitizen": 1, "Partner": "No", "Dependents": "No",
            "tenure": 34, "PhoneService": "Yes", "MultipleLines": "No",
            "InternetService": "Fiber optic", "OnlineSecurity": "Yes", "OnlineBackup": "No",
            "DeviceProtection": "Yes", "TechSupport": "No", "StreamingTV": "Yes",
            "StreamingMovies": "No", "Contract": "One year", "PaperlessBilling": "No",
            "PaymentMethod": "Mailed check", "MonthlyCharges": 56.95, "TotalCharges": 1889.5,
            "Churn": 0,
        },
        {
            "gender": "Female", "SeniorCitizen": 0, "Partner": "No", "Dependents": "Yes",
            "tenure": 58, "PhoneService": "Yes", "MultipleLines": "Yes",
            "InternetService": "No", "OnlineSecurity": "No internet service",
            "OnlineBackup": "No internet service", "DeviceProtection": "No internet service",
            "TechSupport": "No internet service", "StreamingTV": "No internet service",
            "StreamingMovies": "No internet service", "Contract": "Two year",
            "PaperlessBilling": "No", "PaymentMethod": "Bank transfer (automatic)",
            "MonthlyCharges": 20.05, "TotalCharges": 1162.9, "Churn": 0,
        },
        {
            "gender": "Male", "SeniorCitizen": 0, "Partner": "Yes", "Dependents": "Yes",
            "tenure": 2, "PhoneService": "Yes", "MultipleLines": "No",
            "InternetService": "Fiber optic", "OnlineSecurity": "No", "OnlineBackup": "No",
            "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "No",
            "StreamingMovies": "No", "Contract": "Month-to-month", "PaperlessBilling": "Yes",
            "PaymentMethod": "Credit card (automatic)", "MonthlyCharges": 75.3,
            "TotalCharges": 151.65, "Churn": 1,
        },
    ] * 5  # repetido para que el RandomForest tenga algo minimamente estable que aprender

    return pd.DataFrame(filas)


def _armar_pipeline() -> Pipeline:
    preprocesador = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]),
                NUMERIC_FEATURES,
            ),
            (
                "cat",
                Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]),
                CATEGORICAL_FEATURES,
            ),
        ]
    )
    return Pipeline(
        steps=[
            ("preprocessor", preprocesador),
            ("classifier", RandomForestClassifier(n_estimators=5, max_depth=3, random_state=42)),
        ]
    )


@pytest.fixture(scope="session")
def modelo_registrado_local():
    """Entrena, loguea y registra un modelo de prueba en un MLflow local.

    Devuelve un dict con las variables de entorno que hay que setear
    para que app.model_loader lo encuentre, como si fuera DagsHub.
    """
    tmp_dir = tempfile.mkdtemp(prefix="mlflow_test_")
    tracking_uri = f"sqlite:///{tmp_dir}/mlflow.db"
    mlflow.set_tracking_uri(tracking_uri)

    df = _dataset_sintetico()
    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    pipeline = _armar_pipeline()
    pipeline.fit(X, y)

    with mlflow.start_run(run_name="test-run"):
        mlflow.sklearn.log_model(
            pipeline,
            artifact_path="model",
            input_example=X.head(2),
            serialization_format="cloudpickle",
        )
        run_id = mlflow.active_run().info.run_id

    model_uri = f"runs:/{run_id}/model"
    mlflow.register_model(model_uri=model_uri, name=MODEL_NAME_TEST)

    return {
        "MLFLOW_TRACKING_URI": tracking_uri,
        "MODEL_NAME": MODEL_NAME_TEST,
    }


@pytest.fixture()
def cliente_api(modelo_registrado_local, monkeypatch):
    """Devuelve un TestClient de FastAPI, con la app apuntando al modelo de prueba."""
    for key, value in modelo_registrado_local.items():
        monkeypatch.setenv(key, value)
    monkeypatch.delenv("MODEL_VERSION", raising=False)

    # Import diferido: la app lee las variables de entorno recien
    # cuando arranca (lifespan), no al importar el modulo, pero
    # importamos aca adentro igual para evitar cachear un modulo
    # ya cargado de un test anterior con env vars distintas.
    from fastapi.testclient import TestClient

    from app.main import app

    with TestClient(app) as client:
        yield client
