"""
Contratos de entrada y salida de la API, con Pydantic.

Los nombres de campo y sus valores permitidos salen directo del
diccionario de datos del dataset (ver metadata/data_dictionary.csv
del paquete de datos original). customerID es opcional: si no se
manda, generamos un request_id igual; si se manda, lo usamos para
poder rastrear la prediccion hasta el cliente real.
"""

from __future__ import annotations

from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator


class ClientePredictRequest(BaseModel):
    """Lo que la API espera recibir en POST /predict: las caracteristicas de UN cliente."""

    customerID: str | None = Field(
        default=None,
        description="Identificador del cliente, opcional, solo para trazabilidad.",
    )

    gender: Literal["Male", "Female"]
    SeniorCitizen: Literal[0, 1]
    Partner: Literal["Yes", "No"]
    Dependents: Literal["Yes", "No"]

    tenure: int = Field(ge=0, le=100, description="Meses de antiguedad del cliente.")

    PhoneService: Literal["Yes", "No"]
    MultipleLines: Literal["Yes", "No", "No phone service"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: Literal["Yes", "No", "No internet service"]
    OnlineBackup: Literal["Yes", "No", "No internet service"]
    DeviceProtection: Literal["Yes", "No", "No internet service"]
    TechSupport: Literal["Yes", "No", "No internet service"]
    StreamingTV: Literal["Yes", "No", "No internet service"]
    StreamingMovies: Literal["Yes", "No", "No internet service"]
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: Literal["Yes", "No"]
    PaymentMethod: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ]

    MonthlyCharges: float = Field(ge=0, description="Cargo mensual en dolares.")
    TotalCharges: float | None = Field(
        default=None,
        ge=0,
        description="Cargo total acumulado. Puede venir vacio (clientes muy nuevos).",
    )

    @field_validator("TotalCharges", mode="before")
    @classmethod
    def vacio_a_none(cls, valor):
        # En el dataset original TotalCharges a veces viene como string
        # vacio (" "). Si alguien manda eso, lo tratamos como None en
        # vez de dejar que Pydantic tire un error de tipo confuso.
        if isinstance(valor, str) and valor.strip() == "":
            return None
        return valor

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
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
            ]
        }
    }


class PredictResponse(BaseModel):
    """Lo que la API devuelve en POST /predict."""

    # protected_namespaces=() porque Pydantic por defecto reserva el
    # prefijo "model_" para su propia configuracion interna, y
    # nuestros campos model_version/model_name lo pisan a proposito
    # (son los nombres que pide el contrato de la consigna).
    model_config = {"protected_namespaces": ()}

    prediction: Literal["CHURN", "NO_CHURN"]
    probability: float = Field(ge=0, le=1)
    risk_level: Literal["LOW", "MEDIUM", "HIGH"]
    model_version: str
    request_id: str = Field(default_factory=lambda: str(uuid4()))


class HealthResponse(BaseModel):
    """Lo que la API devuelve en GET /health."""

    model_config = {"protected_namespaces": ()}

    status: Literal["ok", "degraded"]
    model_name: str
    model_version: str | None = None

    # Trazabilidad: permiten relacionar lo que esta sirviendo la API con
    # el codigo (commit), el entrenamiento (run de MLflow) y los datos
    # (version del dataset) que le dieron origen.
    run_id: str | None = None
    git_commit: str | None = None
    dataset_version: str | None = None
