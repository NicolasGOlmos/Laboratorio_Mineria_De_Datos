# Analisis de Customer Churn

En este proyecto vamos a armar un servicio de ML para detectar los clientes que estan
en riesgo de darse de baja del servicio de una empresa de telecomunicaciones.
Para ello vamos a utilizar datos historicos, los servicios contratados,
antiguedad y las formas de pago.

Como primera instancia vamos a comenzar haciendo el analisis de los datos,
luego vamos a entrenar diferentes modelos, los vamos a comparar y registrarlos
en el Model Registry. 


## El problema que tratamos de solventar.

Tenemos una clasificacion binaria supervsida, donde tenemos como variable predictora 
a "Churn" (Yes/No). El modelo debera devolver la probalidad que el cliente abandone el servicio,
para esto vamos a utilizar una salida derivada en LOW, MEDIUM, HIGH.

Viendo los resultados de los testeos que hicimos vamos a priorizar la metrica "RECALL" de la clase 
Churn=1. Ya que lo prioritario de solventar son los falsos negativos, porque si la empresa no
realiza algun accion para retener al cliente que ya tenia pensando irse es muy malo para el negocio.
En cambio si la empresa llama a un cliente por un falso positivo es mucho mas barato, solamente se
gasta en el costo de hacer llamada, ya que el cliente pensaba quedarse igualmente.


## Organizacion del repositorio.

```
Analisis_customer_churn/
├── data/
│   ├── raw/                  # dataset histórico, tal como vino (versionado con DVC)
│   └── processed/            # train.csv / test.csv, generados por el split
├── notebooks/
│   └── Analisis_Exploratorio.ipynb   # EDA, solo para explorar, no es código productivo
├── src/
│   ├── data/                 # cargar y particionar los datos
│   ├── features/             # el pipeline de preprocesamiento
│   ├── training/             # entrenar, comparar y registrar modelos
│   ├── evaluation/           # cálculo de métricas
│   └── inference/            # (para la próxima etapa: el servicio de predicción)
├── tests/                    # pytest: datos, preprocesamiento, métricas
├── models/                   # acá cae el resumen de resultados de cada corrida
├── app/                      # (próxima etapa: FastAPI)
├── monitoring/               # (próxima etapa: drift, Evidently)
├── requirements.txt
├── pytest.ini
├── README.md
├── SETUP_GIT_DVC.md
└── .vscode/settings.json
```

## El dataset

Partimos de `data/raw/customer_churn_historical.csv`: 7.043 clientes,
21 columnas. Un par de cosas a tener en cuenta:

- `customerID` es solo un identificador, así que lo sacamos de las
  features a propósito — no le sirve al modelo para predecir nada.
- `TotalCharges` tiene 26 valores vacíos que vienen así en el dataset
  original. No los tocamos al cargar los datos: se resuelven recién
  en el pipeline de preprocesamiento, imputándolos por la mediana.

## Cómo correr todo esto en tu máquina

### 1. Instalar dependencias

```bash
git clone <URL_DEL_REPO>
cd Analisis_customer_churn
python -m venv .venv
source .venv/bin/activate          # en Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. Traer los datos (DVC)

```bash
dvc pull
```

Esto baja `customer_churn_historical.csv` desde el remote de DagsHub
(los comandos de conexión están en `SETUP_GIT_DVC.md`).

### 3. Generar el split de entrenamiento y prueba

```bash
python -m src.data.split_data
```

La función que hace esto es `dividir_en_entrenamiento_y_prueba()`,
adentro de `src/data/split_data.py`. Divide 80/20, siempre con
`seed=42` y estratificando por `Churn`, así que correrlo dos veces te
da exactamente el mismo resultado — esto está comprobado en
`tests/test_data.py`.

### 4. Entrenar y comparar los modelos

```bash
export MLFLOW_TRACKING_URI="https://dagshub.com/<usuario>/<repo>.mlflow"
export MLFLOW_TRACKING_USERNAME="<usuario_dagshub>"
export MLFLOW_TRACKING_PASSWORD="<token_dagshub>"

python -m src.training.train
```

Esto corre `entrenar_y_comparar_modelos()`, que prueba 6
configuraciones distintas — no elegidas al azar, sino pensadas para
cubrir un rango razonable de opciones:

| Run                 | Familia | Configuración                                                        |
|---------------------|---------|----------------------------------------------------------------------|
| `baseline-dummy`    | Baseline| `DummyClassifier(strategy="stratified")`                             |
| `logreg-l2-balanced`| Lineal  | `LogisticRegression(C=1.0, class_weight="balanced")`                 |
| `logreg-l2-C0.1`    | Lineal  | `LogisticRegression(C=0.1, class_weight="balanced")`                 |
| `rf-100-depth6`     | Árbol   | `RandomForest(n_estimators=100, max_depth=6)`                        |
| `rf-300-depth10`    | Árbol   | `RandomForest(n_estimators=300, max_depth=10)`                       |
| `rf-300-depth-none` | Árbol   | `RandomForest(n_estimators=300, max_depth=None, min_samples_leaf=5)` |

Cada run queda logueado en MLflow con sus parámetros, sus métricas
(`accuracy`, `precision`, `recall`, `f1`, `roc_auc`) y el pipeline
completo (preprocesamiento + modelo) como artefacto descargable.

### 5. Los resultados que nos dio (test set)

| run_name           | family   | accuracy   | precision  | recall     | f1         | roc_auc|
|--------------------|----------|------------|------------|------------|------------|--------|
| baseline-dummy     | baseline | 0.6217     | 0.2853     | 0.2876     | 0.2865     | 0.5146 |
| logreg-l2-balanced | linear   | 0.7211     | 0.4812     | 0.7231     | 0.5779     | 0.8117 |
| logreg-l2-C0.1     | linear   | 0.7211     | 0.4812     | 0.7231     | 0.5779     | 0.8119 |
| **rf-100-depth6**  | **tree** | **0.7232** | **0.4842** | **0.7419** | **0.5860** | 0.8011 |
| rf-300-depth10     | tree     | 0.7573     | 0.5352     | 0.6129     | 0.5714     | 0.8052 |
| rf-300-depth-none  | tree     | 0.7544     | 0.5295     | 0.6263     | 0.5739     | 0.8034 |

### 6. Qué modelo elegimos y por qué

Nos quedamos con **`rf-100-depth6`**:
`RandomForestClassifier(n_estimators=100, max_depth=6, class_weight="balanced")`.

Es el que saca **mayor Recall** de todos (0.7419), que es justo la
métrica que definimos como prioridad. Su ROC-AUC (0.8011) está a la
par del resto, y de hecho tiene el F1 más alto de la tabla. Los
RandomForest con 300 árboles logran mejor accuracy y precision, pero
a costa de perder recall — y ese trade-off no nos conviene para este
caso de negocio: preferimos detectar más churners reales, aunque eso
signifique contactar de más a algunos que en realidad no se iban a ir.

Un dato que vale la pena remarcar: **todos** los modelos entrenados
superan ampliamente al baseline (recall pasa de 0.29 a 0.74), así que
las variables del dataset sí tienen poder predictivo real sobre el
churn — no es ruido.

### 7. Registrar el modelo ganador

```bash
python -m src.training.register_model --run-id <RUN_ID_DE_rf-100-depth6>
```

Esto ejecuta `registrar_modelo_ganador()`, que registra el modelo
como `customer-churn-classifier` en el Model Registry de MLflow, y le
deja tags (`source_run_id`, `source_experiment_id`) para poder
reconstruir después exactamente de qué run salió, sin tener que
adivinar nada.

### 8. Correr los tests

```bash
pytest tests/ -v
```

Tenemos 12 tests que cubren: que el split sea reproducible, que
mantenga la proporción real de churn, que no haya clientes repetidos
entre train y test, que el preprocesamiento no deje pasar valores
faltantes, que transforme igual en entrenamiento y en inferencia, y
que las métricas detecten bien un falso negativo.

## Versionar los datos con DVC + DagsHub

```bash
dvc init

dvc remote add origin https://dagshub.com/<usuario>/<repo>.dvc
dvc remote modify origin --local auth basic
dvc remote modify origin --local user <usuario_dagshub>
dvc remote modify origin --local password <token_dagshub>

dvc add data/raw/customer_churn_historical.csv
git add data/raw/customer_churn_historical.csv.dvc data/raw/.gitignore
git commit -m "data: versionar dataset historico con DVC"

dvc push
```

El `.csv` en sí **no se sube a Git** — solo el archivo `.dvc`, que es
un puntero chiquito de texto. El contenido real del dataset vive en
DagsHub.

## Cómo ver los experimentos en DagsHub

Con las variables de entorno de la sección 4 configuradas, todos los
runs, parámetros, métricas y el modelo registrado quedan visibles acá:

```
https://dagshub.com/<usuario>/<repo>.mlflow
```

## Trazabilidad: cómo reconstruir de dónde salió un modelo

1. El modelo en el Model Registry tiene el tag `source_run_id`, que
   apunta al Run exacto de MLflow que lo generó.
2. Ese Run tiene logueados los parámetros del modelo y del split
   (`seed`, `n_train`, `n_test`).
3. El dataset que se usó está versionado en DVC — el commit de Git
   asociado al `.dvc` identifica exactamente qué versión de los datos
   se usó para entrenar.
4. El código que generó el Run es el del commit de Git correspondiente
   (usamos el tag `entrega-1` sobre el commit exacto que presentamos).

## Algunas decisiones que tomamos (y por qué)

- Tratamos `SeniorCitizen` (que es 0/1) como variable **categórica**,
  no numérica, porque no hay una relación de orden real entre esos
  valores — no tiene sentido decir que "1 es más" que "0".
- El `StandardScaler` solo tiene efecto real en el modelo lineal, pero
  lo aplicamos igual a todos los modelos para no duplicar lógica de
  preprocesamiento (un pipeline único para todo, en vez de uno por
  modelo).
- El threshold de decisión (por ahora el 0.5 que usa `.predict()` por
  defecto) todavía no lo ajustamos — eso lo vamos a definir en la
  etapa de serving, junto con los umbrales de LOW/MEDIUM/HIGH.
- Las carpetas `src/inference/`, `app/` y `monitoring/` están vacías
  a propósito: son el lugar donde va a vivir el código de las próximas
  etapas (servicio con FastAPI y monitoreo de drift), pero no forman
  parte de esta entrega.

## Por si algo no anda

- **`ModuleNotFoundError: No module named 'src'`**: te falta correr
  el módulo desde la raíz del proyecto con `python -m`, no entrando a
  la carpeta y ejecutando el archivo directo.
- **Pylance marca el import en rojo pero el código corre bien**: es
  solo el linter de VS Code, ya está resuelto con
  `.vscode/settings.json` (`python.analysis.extraPaths`).
- **`pytest` no encuentra los tests o tira `ModuleNotFoundError`**:
  fijate que `pytest.ini` tenga la línea `pythonpath = .` — sin eso,
  corriendo `pytest` directo (no `python -m pytest`) el proyecto no
  queda en el path de búsqueda de Python.
