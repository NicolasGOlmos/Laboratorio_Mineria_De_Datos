import numpy as np
import pandas as pd
import pytest

from src.data.load_data import cargar_datos_crudos, separar_features_y_target
from src.features.preprocessing import armar_preprocesamiento

RUTA_DATOS_CRUDOS = "data/raw/customer_churn_historical.csv"


@pytest.fixture(scope="module")
def features_y_target():
    datos = cargar_datos_crudos(RUTA_DATOS_CRUDOS)
    return separar_features_y_target(datos)


def testeo_valores_faltantes(features_y_target):
    # TotalCharges tiene nulos; por esto el preprocesador no puede tener errores al corregirlos.
    features, _ = features_y_target
    assert features["TotalCharges"].isna().sum() > 0

    preprocesador = armar_preprocesamiento()
    features_transformadas = preprocesador.fit_transform(features)
    matriz = features_transformadas.toarray() if hasattr(features_transformadas, "toarray") else features_transformadas

    assert not np.isnan(matriz).any()


def testeo_salida_es_numerica(features_y_target):
    # Despues de imputar y codificar, todo tiene que quedar en formato numerico para el modelo
    features, _ = features_y_target
    preprocesador = armar_preprocesamiento()
    features_transformadas = preprocesador.fit_transform(features)
    matriz = features_transformadas.toarray() if hasattr(features_transformadas, "toarray") else features_transformadas

    assert np.issubdtype(matriz.dtype, np.number)


def testeo_preprocesador_transforma_entrenamiento_inferencia(features_y_target):
    # Ajustado con datos de entrenamiento, tiene que transformar de forma identica datos nuevos.
    features, _ = features_y_target
    preprocesador = armar_preprocesamiento()
    preprocesador.fit(features.iloc[:5000])

    transformacion_a = preprocesador.transform(features.iloc[5000:5010])
    transformacion_b = preprocesador.transform(features.iloc[5000:5010])

    matriz_a = transformacion_a.toarray() if hasattr(transformacion_a, "toarray") else transformacion_a
    matriz_b = transformacion_b.toarray() if hasattr(transformacion_b, "toarray") else transformacion_b
    np.testing.assert_array_equal(matriz_a, matriz_b)
