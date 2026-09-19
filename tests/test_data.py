import pandas as pd
import pytest

from src.data.load_data import cargar_datos_crudos, separar_features_y_target, TARGET_COLUMN, ID_COLUMN
from src.data.split_data import dividir_en_entrenamiento_y_prueba

RUTA_DATOS_CRUDOS = "data/raw/customer_churn_historical.csv"


@pytest.fixture(scope="module")
def datos_crudos():
    return cargar_datos_crudos(RUTA_DATOS_CRUDOS)


def testeo_dataset_columnas_claves(datos_crudos):
    # El dataset debe conservar el target y el identificador de cliente tal como vienen.
    assert TARGET_COLUMN in datos_crudos.columns
    assert ID_COLUMN in datos_crudos.columns
    assert len(datos_crudos) == 7043


def testeo_totalcharges_tipado(datos_crudos):
    # TotalCharges viene como texto en el csv original; tiene que quedar como numero.
    assert pd.api.types.is_numeric_dtype(datos_crudos["TotalCharges"])


def testeo_id_churn_no_predictoras(datos_crudos):
    # customerID y Churn no tienen que terminar como variables predictoras.
    features, target = separar_features_y_target(datos_crudos)
    assert ID_COLUMN not in features.columns
    assert TARGET_COLUMN not in features.columns
    assert set(target.unique()) <= {0, 1}


def testeo_la_particion_es_reproducible(datos_crudos):
    """Correr la particion dos veces con el mismo seed tiene que dar exactamente el mismo resultado."""
    entrenamiento_1, prueba_1 = dividir_en_entrenamiento_y_prueba(datos_crudos, seed=42)
    entrenamiento_2, prueba_2 = dividir_en_entrenamiento_y_prueba(datos_crudos, seed=42)

    pd.testing.assert_frame_equal(
        entrenamiento_1.reset_index(drop=True), entrenamiento_2.reset_index(drop=True)
    )
    pd.testing.assert_frame_equal(
        prueba_1.reset_index(drop=True), prueba_2.reset_index(drop=True)
    )


def testeo_cambiar_semilla(datos_crudos):
    # Con un seed distinto, tienen que caer filas distintas en entrenamiento.
    entrenamiento_1, _ = dividir_en_entrenamiento_y_prueba(datos_crudos, seed=42)
    entrenamiento_2, _ = dividir_en_entrenamiento_y_prueba(datos_crudos, seed=7)
    assert not entrenamiento_1.index.equals(entrenamiento_2.index)


def testeo_la_particion_mantiene_proporcion(datos_crudos):
    # La particion estratificada no tiene que alterar la proporcion real de clientes que abandonan.
    entrenamiento, prueba = dividir_en_entrenamiento_y_prueba(datos_crudos, seed=42)
    proporcion_total = (datos_crudos[TARGET_COLUMN] == "Yes").mean()
    proporcion_entrenamiento = (entrenamiento[TARGET_COLUMN] == "Yes").mean()
    proporcion_prueba = (prueba[TARGET_COLUMN] == "Yes").mean()

    assert abs(proporcion_entrenamiento - proporcion_total) < 0.02
    assert abs(proporcion_prueba - proporcion_total) < 0.02


def testeo_no_repetir_entrenamiento_y_prueba(datos_crudos):
    # Chequea que no haya fuga de datos: un cliente no puede estar en los dos conjuntos a la vez.
    entrenamiento, prueba = dividir_en_entrenamiento_y_prueba(datos_crudos, seed=42)
    clientes_repetidos = set(entrenamiento[ID_COLUMN]) & set(prueba[ID_COLUMN])
    assert len(clientes_repetidos) == 0
