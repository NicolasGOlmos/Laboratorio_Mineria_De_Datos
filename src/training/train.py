"""
Este es el script principal de entrenamiento. Prueba 6 configuraciones
distintas de modelos (una posta de baseline, dos lineales y tres de
arboles) y le manda todo a MLflow: parametros, metricas y el pipeline
entero (preprocesamiento + modelo) como artefacto descargable.

No depende de que corras celdas de notebook a mano: se ejecuta asi,
parado en la raiz del proyecto:

    python -m src.training.train
"""

from __future__ import annotations

import argparse
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.data.load_data import separar_features_y_target, TARGET_COLUMN
from src.evaluation.evaluate import calcular_metricas, mostrar_reporte_de_metricas
from src.features.preprocessing import armar_pipeline_de_preprocesamiento

NOMBRE_EXPERIMENTO = "customer-churn-model-selection"
SEMILLA = 42


def definir_modelos_a_comparar() -> list[dict]:
    """Define que modelos y configuraciones vamos a comparar.

    No es una lista al azar: cada modelo esta ahi por una razon
    puntual (ver README para la justificacion completa de cada uno).
    Cada entrada de esta lista termina siendo un Run independiente en
    MLflow.
    """
    return [
        {
            "run_name": "baseline-dummy",
            "family": "baseline",
            "estimator": DummyClassifier(strategy="stratified", random_state=SEMILLA),
            "params": {"strategy": "stratified"},
        },
        {
            "run_name": "logreg-l2-balanced",
            "family": "linear",
            "estimator": LogisticRegression(
                penalty="l2",
                C=1.0,
                class_weight="balanced",
                max_iter=1000,
                random_state=SEMILLA,
            ),
            "params": {"penalty": "l2", "C": 1.0, "class_weight": "balanced"},
        },
        {
            "run_name": "logreg-l2-C0.1",
            "family": "linear",
            "estimator": LogisticRegression(
                penalty="l2",
                C=0.1,
                class_weight="balanced",
                max_iter=1000,
                random_state=SEMILLA,
            ),
            "params": {"penalty": "l2", "C": 0.1, "class_weight": "balanced"},
        },
        {
            "run_name": "rf-100-depth6",
            "family": "tree",
            "estimator": RandomForestClassifier(
                n_estimators=100,
                max_depth=6,
                class_weight="balanced",
                random_state=SEMILLA,
            ),
            "params": {
                "n_estimators": 100,
                "max_depth": 6,
                "class_weight": "balanced",
            },
        },
        {
            "run_name": "rf-300-depth10",
            "family": "tree",
            "estimator": RandomForestClassifier(
                n_estimators=300,
                max_depth=10,
                class_weight="balanced",
                random_state=SEMILLA,
            ),
            "params": {
                "n_estimators": 300,
                "max_depth": 10,
                "class_weight": "balanced",
            },
        },
        {
            "run_name": "rf-300-depth-none",
            "family": "tree",
            "estimator": RandomForestClassifier(
                n_estimators=300,
                max_depth=None,
                min_samples_leaf=5,
                class_weight="balanced",
                random_state=SEMILLA,
            ),
            "params": {
                "n_estimators": 300,
                "max_depth": None,
                "min_samples_leaf": 5,
                "class_weight": "balanced",
            },
        },
    ]


def entrenar_y_comparar_modelos(train_path: str, test_path: str, seed: int = SEMILLA) -> pd.DataFrame:
    """Entrena cada modelo de la lista, lo evalua contra el test set y lo loguea en MLflow."""
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    X_train, y_train = separar_features_y_target(train_df)
    X_test, y_test = separar_features_y_target(test_df)

    mlflow.set_experiment(NOMBRE_EXPERIMENTO)

    resultados = []

    for config in definir_modelos_a_comparar():
        with mlflow.start_run(run_name=config["run_name"]):
            preprocesador = armar_pipeline_de_preprocesamiento()
            pipeline = Pipeline(
                steps=[
                    ("preprocessor", preprocesador),
                    ("classifier", config["estimator"]),
                ]
            )

            pipeline.fit(X_train, y_train)

            y_pred = pipeline.predict(X_test)
            y_proba = pipeline.predict_proba(X_test)[:, 1]
            metricas = calcular_metricas(y_test, y_pred, y_proba)

            # --- esto es lo que queda registrado en MLflow ---
            mlflow.log_param("family", config["family"])
            mlflow.log_params(config["params"])
            mlflow.log_param("seed", seed)
            mlflow.log_param("n_train", len(X_train))
            mlflow.log_param("n_test", len(X_test))

            mlflow.log_metrics(
                {
                    "accuracy": metricas["accuracy"],
                    "precision": metricas["precision"],
                    "recall": metricas["recall"],
                    "f1": metricas["f1"],
                    "roc_auc": metricas["roc_auc"],
                }
            )

            mlflow.sklearn.log_model(
                pipeline,
                name="model",
                input_example=X_train.head(2),
                serialization_format="cloudpickle",
            )

            mostrar_reporte_de_metricas(config["run_name"], metricas)

            run_id = mlflow.active_run().info.run_id
            resultados.append(
                {
                    "run_name": config["run_name"],
                    "run_id": run_id,
                    "family": config["family"],
                    **{
                        k: metricas[k]
                        for k in ["accuracy", "precision", "recall", "f1", "roc_auc"]
                    },
                }
            )

    return pd.DataFrame(resultados)


def main():
    parser = argparse.ArgumentParser(description="Entrena y compara los modelos candidatos")
    parser.add_argument("--train", type=str, default="data/processed/train.csv")
    parser.add_argument("--test", type=str, default="data/processed/test.csv")
    parser.add_argument("--output", type=str, default="modelos/resumen_resultados.csv")
    args = parser.parse_args()

    resultados_df = entrenar_y_comparar_modelos(args.train, args.test)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    resultados_df.to_csv(output_path, index=False)

    print("\n=== Resumen de todos los runs ===")
    print(resultados_df.sort_values("recall", ascending=False).to_string(index=False))


if __name__ == "__main__":
    main()
