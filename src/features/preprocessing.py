"""
Este es el UNICO lugar donde definimos como se transforman los datos
antes de entrar al modelo. Tanto el entrenamiento como (mas adelante)
la inferencia en produccion usan este mismo pipeline, asi nos
aseguramos de que a los datos se les aplique siempre exactamente la
misma transformacion, sin importar en que momento se use.
"""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Numericas: pueden tener faltantes (TotalCharges) y las escalamos
# porque despues probamos un modelo lineal (LogisticRegression), que
# es sensible a la escala de las variables.
NUMERIC_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges"]

# SeniorCitizen es 0/1 pero la tratamos como categorica a proposito,
# para no asumir una relacion de orden que en realidad no existe.
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


def armar_preprocesamiento() -> ColumnTransformer:
    """Arma el ColumnTransformer con todo el preprocesamiento.

    Numericas -> se imputan con la mediana y se escalan.
    Categoricas -> se imputan con la moda y se codifican con one-hot.
    """
    transformador_numerico = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    transformador_categorico = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocesador = ColumnTransformer(
        transformers=[
            ("num", transformador_numerico, NUMERIC_FEATURES),
            ("cat", transformador_categorico, CATEGORICAL_FEATURES),
        ]
    )
    return preprocesador


def listar_columnas() -> list[str]:
    #Devuelve la lista completa de columnas que entran al pipeline (util para debug).
    return NUMERIC_FEATURES + CATEGORICAL_FEATURES
