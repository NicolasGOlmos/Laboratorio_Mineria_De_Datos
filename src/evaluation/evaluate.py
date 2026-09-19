"""
Calculamos las metricas de cada modelo para decidir cual predice de mejora manera.
Accuracy solo se usa como referencia pero no para seleccionar el modelo ganador, por que el target esta desbalanceado.
En este trabajo utilizamos Recall de la clase Churn=1, porque permite analizar los falsos negativos.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def calcular_metricas(y_true, y_pred, y_proba) -> dict:
    # Calcula las metricas relevantes para un clasificador binario.
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_proba),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
    }


def reporte_de_metricas(nombre: str, metricas: dict) -> None:
    # Imprime en consola un resumen de las metricas de un run.
    print(f"\n--- {nombre} ---")
    for clave in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
        print(f"{clave:>10}: {metricas[clave]:.4f}")
    print(
        f"Matriz de confusion: TN={metricas['true_negatives']} "
        f"FP={metricas['false_positives']} "
        f"FN={metricas['false_negatives']} "
        f"TP={metricas['true_positives']}"
    )
