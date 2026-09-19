import numpy as np

from src.evaluation.evaluate import calcular_metricas


def testeo_prediccion_perfecta():
    #Si el modelo acierta absolutamente todo, todas las metricas tienen que dar el maximo posible.
    real = np.array([0, 0, 1, 1])
    prediccion = np.array([0, 0, 1, 1])
    probabilidad = np.array([0.1, 0.2, 0.9, 0.95])

    metricas = calcular_metricas(real, prediccion, probabilidad)

    assert metricas["accuracy"] == 1.0
    assert metricas["precision"] == 1.0
    assert metricas["recall"] == 1.0
    assert metricas["f1"] == 1.0
    assert metricas["false_negatives"] == 0
    assert metricas["false_positives"] == 0


def testeo_detectar_falso_negativo():
    # Un cliente que iba a hacer churn (1) pero fue predicho como estable (0) tiene que contar como falso negativo.
    real = np.array([0, 0, 1, 1])
    prediccion = np.array([0, 0, 0, 1])
    probabilidad = np.array([0.1, 0.2, 0.4, 0.95])

    metricas = calcular_metricas(real, prediccion, probabilidad)

    assert metricas["false_negatives"] == 1
    assert metricas["recall"] == 0.5
