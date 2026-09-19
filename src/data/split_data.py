"""
Aca armamos la particion train/test. La corremos como script aparte
(no adentro del entrenamiento) para que el resultado quede guardado
en disco y se pueda testear de forma aislada, sin depender de que el
entrenamiento haya corrido antes.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.data.load_data import cargar_datos_crudos, TARGET_COLUMN

SEMILLA_POR_DEFECTO = 42
PORCENTAJE_TEST_POR_DEFECTO = 0.2


def dividir_en_entrenamiento_y_prueba(
    df: pd.DataFrame,
    test_size: float = PORCENTAJE_TEST_POR_DEFECTO,
    seed: int = SEMILLA_POR_DEFECTO,
):
    """Divide el DataFrame en train/test, estratificado por Churn.

    Estratificamos porque el target esta desbalanceado (~26% de
    churn). Si no lo hicieramos, un split al azar podria dejar, por
    mala suerte, mucho mas o mucho menos churn en un lado que en otro.
    """
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=seed,
        stratify=df[TARGET_COLUMN],
    )
    return train_df, test_df


def main():
    parser = argparse.ArgumentParser(description="Genera el split reproducible train/test")
    parser.add_argument(
        "--input",
        type=str,
        default="data/raw/customer_churn_historical.csv",
    )
    parser.add_argument("--output-dir", type=str, default="data/processed")
    parser.add_argument("--test-size", type=float, default=PORCENTAJE_TEST_POR_DEFECTO)
    parser.add_argument("--seed", type=int, default=SEMILLA_POR_DEFECTO)
    args = parser.parse_args()

    df = cargar_datos_crudos(args.input)
    train_df, test_df = dividir_en_entrenamiento_y_prueba(
        df, test_size=args.test_size, seed=args.seed
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / "train.csv"
    test_path = output_dir / "test.csv"

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"Train: {train_df.shape} -> {train_path}")
    print(f"Test:  {test_df.shape} -> {test_path}")
    print(f"Tasa de churn en train: {(train_df[TARGET_COLUMN] == 'Yes').mean():.4f}")
    print(f"Tasa de churn en test:  {(test_df[TARGET_COLUMN] == 'Yes').mean():.4f}")


if __name__ == "__main__":
    main()
