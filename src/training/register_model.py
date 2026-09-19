"""
Este script NO reentrena nada. Toma el run_id de un run que ya existe
(el que loguea src/training/train.py) y lo registra
formalmente en el MLflow Model Registry, dejando bien claro de donde
salio ese modelo (run_id, experiment, metricas).

Se ejecuta asi, con el run_id del modelo que elegiste como ganador:

    python -m src.training.register_model --run-id <ID>
"""

from __future__ import annotations

import argparse

import mlflow
from mlflow.tracking import MlflowClient

NOMBRE_MODELO = "customer-churn-classifier"


def registrar_modelo_ganador(run_id: str, model_name: str = NOMBRE_MODELO):
    # Registra en el Model Registry el modelo de un run puntual, con trazabilidad completa.
    client = MlflowClient()
    model_uri = f"runs:/{run_id}/model"

    resultado = mlflow.register_model(model_uri=model_uri, name=model_name)

    """"Dejamos tags explicitos para poder reconstruir despues de que
    run y que experiment salio este modelo, sin tener que adivinar. """
    run = client.get_run(run_id)
    client.set_model_version_tag(
        name=model_name,
        version=resultado.version,
        key="source_experiment_id",
        value=run.info.experiment_id,
    )
    client.set_model_version_tag(
        name=model_name,
        version=resultado.version,
        key="source_run_id",
        value=run_id,
    )
    client.update_model_version(
        name=model_name,
        version=resultado.version,
        description=(
            f"Registrado desde run_id={run_id}. "
            f"Recall={run.data.metrics.get('recall'):.4f}, "
            f"ROC-AUC={run.data.metrics.get('roc_auc'):.4f}. "
            "Elegido por maximizar Recall (nos importa mas no dejar "
            "pasar un cliente que se va a ir, que molestar de mas a "
            "uno que se iba a quedar)."
        ),
    )

    print(f"Modelo registrado: {model_name} v{resultado.version}")
    print(f"Viene del run_id={run_id}, experiment_id={run.info.experiment_id}")
    return resultado


def main():
    parser = argparse.ArgumentParser(description="Registra el modelo ganador en el Model Registry")
    parser.add_argument("--run-id", type=str, required=True)
    parser.add_argument("--model-name", type=str, default=NOMBRE_MODELO)
    args = parser.parse_args()
    registrar_modelo_ganador(args.run_id, args.model_name)


if __name__ == "__main__":
    main()
