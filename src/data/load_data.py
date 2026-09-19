"""
Este modulo se encarga solamente de leer el csv original, corregir los
tipos de columnas que vienen mal, y separar el dataset en features (X)
y target (y). No hace nada mas que eso: ni entrena, ni preprocesa.
"""

from __future__ import annotations

import pandas as pd

# Nombres de columnas que usamos en todo el proyecto, para no repetir
# el string "Churn" o "customerID" a mano en cada archivo.
ID_COLUMN = "customerID"
TARGET_COLUMN = "Churn"


def cargar_datos_crudos(path: str) -> pd.DataFrame:
    """Lee el CSV original tal cual viene y arregla los tipos de dato.

    Ojo: esto NO imputa ni elimina nada, solo corrige el tipado.
    Los nulos de TotalCharges quedan como NaN a propósito, para que
    se traten mas adelante en el pipeline de preprocesamiento.
    """
    df = pd.read_csv(path)

    # TotalCharges viene como texto en el csv original, y algunos
    # valores estan vacios o con espacios. Forzamos a numerico y lo
    # que no se puede convertir queda como NaN (se imputa despues).
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # SeniorCitizen es logicamente un si/no pero viene como int64.
    df["SeniorCitizen"] = df["SeniorCitizen"].astype(int)

    return df


def separar_features_y_target(
    df: pd.DataFrame,
    target_column: str = TARGET_COLUMN,
    id_column: str = ID_COLUMN,
):
    """Separa el DataFrame en features (X) y target (y).

    customerID se descarta de las features: es un identificador de
    cliente, no una variable que le sirva al modelo para predecir nada
    (regla del negocio, no una decision tecnica nuestra).
    """
    y = (df[target_column] == "Yes").astype(int)
    X = df.drop(columns=[target_column, id_column])
    return X, y
