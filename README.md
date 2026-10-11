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
git clone https://github.com/NicolasGOlmos/Laboratorio_Mineria_De_Datos.git
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

**En PowerShell (Windows):**
```powershell
$env:MLFLOW_TRACKING_URI="https://dagshub.com/nicolas.olmos/Laboratorio_Mineria_De_Datos.git.mlflow"
$env:MLFLOW_TRACKING_USERNAME="<usuario_dagshub>"
$env:MLFLOW_TRACKING_PASSWORD="<token_dagshub>"

python -m src.training.train
```

**En Mac/Linux/Git Bash:**
```bash
export MLFLOW_TRACKING_URI="https://dagshub.com/nicolas.olmos/Laboratorio_Mineria_De_Datos.git.mlflow"
export MLFLOW_TRACKING_USERNAME="<usuario_dagshub>"
export MLFLOW_TRACKING_PASSWORD="<token_dagshub>"

python -m src.training.train
```

Estas variables **no persisten** entre sesiones de terminal — si
cerrás y abrís VS Code, hay que volver a exportarlas antes de entrenar
o registrar el modelo.

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
**Nota sobre la versión de MLflow**: usamos `mlflow==2.16.2` a
propósito (fijado en `requirements.txt`), no la 3.x. El servidor de
MLflow que expone DagsHub todavía no soporta bien el mecanismo nuevo
de "Logged Models" que trae MLflow 3.x, y el registro del modelo
falla con `MlflowException: Unable to find a logged model...` si se
usa esa versión. Por la misma razón, en `train.py` el modelo se
loguea con `artifact_path="model"` (el esquema clásico), no con el
parámetro `name="model"` que usa la sintaxis nueva.


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
- **`pip install -r requirements.txt` falla compilando `pyarrow`**:
  estás en Python 3.14 (demasiado nueva). Usá 3.11, 3.12 o 3.13.
- **`pip install` falla con un conflicto entre `mlflow` y `numpy`**:
  puede pasar en Python 3.13 si la versión de `numpy` que se
  intenta instalar es previa a que numpy soportara esa versión de
  Python. Con las versiones ya fijadas en este `requirements.txt` no
  debería pasar, pero si aparece, la solución más simple es recrear el
  entorno virtual con Python 3.12 en vez de 3.13.
- **`ImportError: cannot import name '...' from '...'`**: significa
  que el nombre de una función cambió en un archivo pero no se
  actualizó en el archivo que la importa (o viceversa). Comparar el
  nombre exacto de la función en ambos archivos y corregir el que
  esté desactualizado.
- **`MlflowException: Unable to find a logged model with artifact_path
  model under run ...`** al correr `register_model`: es la versión de
  MLflow. Confirmá que `requirements.txt` tenga `mlflow==2.16.2` (no
  3.x) y que `train.py` use `artifact_path="model"` en el
  `log_model()` — después hay que volver a entrenar (los runs viejos,
  logueados con la versión incorrecta, no se pueden registrar).
- **Variables de entorno de MLflow "se olvidan"**: no persisten entre
  sesiones de terminal. Si cerraste y abriste VS Code, hay que
  volver a exportarlas antes de entrenar o registrar

  ---


# Etapa 2: API de inferencia

En la etapa 1 dejamos un modelo registrado en MLflow (`customer-churn-classifier`).
En esta etapa lo convertimos en una API REST que recibe los datos de un
cliente y responde la probabilidad, el nivel de riesgo y si es probable que se vaya (churn), 
todo corriendo dentro de Docker.


## Qué hay de nuevo en el repo

```
app/
  main.py            -> la API: GET /health y POST /predict
  schemas.py         -> contratos de entrada y salida (Pydantic)
  model_loader.py    -> trae el modelo desde el Model Registry de MLflow
  risk.py            -> traduce la probabilidad a LOW / MEDIUM / HIGH
tests/
  conftest.py        -> arma un modelo de prueba en un MLflow local
  test_api.py        -> tests de /health, /predict y entradas invalidas
  test_model_loader.py -> carga del modelo y comportamiento si no esta disponible
  test_risk.py       -> limites de los niveles de riesgo
Dockerfile
docker-compose.yml
.dockerignore
.env.example         -> plantilla de variables de entorno (sin secretos)
requirements-api.txt -> dependencias de produccion (lo que va en la imagen)
requirements-test.txt-> lo anterior + herramientas para testear
```

## Cómo correrlo de cero

Necesitás Python 3.12, Git y Docker Desktop.

**1. Clonar e instalar**

```bash
git clone https://github.com/NicolasGOlmos/Laboratorio_Mineria_De_Datos.git
cd <repo>
python -m venv .venv
# Windows (PowerShell): .venv\Scripts\Activate.ps1
# Linux/Mac:            source .venv/bin/activate
pip install -r requirements-test.txt
pip install -r requirements.txt
configurar las credenciales de DagsHub (ver sección de DVC de la etapa 1)
dvc pull
```

**2. Correr los tests**

```bash
pytest tests/ -v
```

Los tests de la API no necesitan internet ni credenciales: entrenan un modelo
chiquito, lo registran en un MLflow local (sqlite, en una carpeta temporal) y la API
lo carga de ahi, igual que haria con DagsHub.

**3. Configurar las credenciales**

```bash
cp .env.example .env        # en PowerShell: copy .env.example .env
```

Abrí `.env` y completá tu usuario y token de DagsHub. Ese archivo está en el
`.gitignore` y en el `.dockerignore`: nunca se sube al repo ni entra a la imagen.

| Variable | Qué es |
|---|---|
| `MLFLOW_TRACKING_URI` | URL del MLflow de DagsHub (`https://dagshub.com/<usuario>/<repo>.mlflow`) |
| `MLFLOW_TRACKING_USERNAME` | Tu usuario de DagsHub |
| `MLFLOW_TRACKING_PASSWORD` | Tu token de DagsHub (no la contraseña) |
| `MODEL_NAME` | Nombre del modelo registrado (`customer-churn-classifier`) |
| `MODEL_VERSION` | Opcional. Si va vacío, se usa la última versión registrada |
| `HOST_PORT` | Opcional. Puerto de tu máquina donde se publica la API (por defecto 8000) |

**4. Construir la imagen y levantar el servicio**

```bash
docker build -t customer-churn-api:latest .
docker compose up --build
```

Cuando todo salió bien, el log muestra:

```
Modelo cargado: customer-churn-classifier v1
Application startup complete.
```

Para apagarlo: `Ctrl+C` y después `docker compose down`.

**5. Probar la API**

- Documentación interactiva (Swagger): http://localhost:8000/docs
- Estado del servicio: http://localhost:8000/health

```bash
curl http://localhost:8000/health
```

```json
{"status": "ok", "model_name": "customer-churn-classifier", "model_version": "1"}
```

Para predecir, guardá un cliente en `cliente.json`:

```json
{
  "customerID": "7590-VHVEG",
  "gender": "Female",
  "SeniorCitizen": 0,
  "Partner": "Yes",
  "Dependents": "No",
  "tenure": 1,
  "PhoneService": "No",
  "MultipleLines": "No phone service",
  "InternetService": "DSL",
  "OnlineSecurity": "No",
  "OnlineBackup": "Yes",
  "DeviceProtection": "No",
  "TechSupport": "No",
  "StreamingTV": "No",
  "StreamingMovies": "No",
  "Contract": "Month-to-month",
  "PaperlessBilling": "Yes",
  "PaymentMethod": "Electronic check",
  "MonthlyCharges": 29.85,
  "TotalCharges": 29.85
}
```

y mandalo:

```bash
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d @cliente.json
# Windows PowerShell: curl.exe -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d "@cliente.json"
```

```json
{
  "prediction": "CHURN",
  "probability": 0.6105,
  "risk_level": "HIGH",
  "model_version": "1",
  "request_id": "c2aaa8bb-b9ec-45ee-ac94-b3fbf9a5b9ae"
}
```

## El contrato de la API

**`GET /health`**: dice si el servicio está operativo y qué versión del modelo tiene
cargada. Responde `ok` si el modelo se cargó, o `degraded` si no.

**`POST /predict`**: recibe las características de un cliente y devuelve:

| Campo | Qué es |
|---|---|
| `prediction` | `CHURN` o `NO_CHURN` (corte en probabilidad 0.5) |
| `probability` | Probabilidad de churn, entre 0 y 1 |
| `risk_level` | `LOW` (< 0.30), `MEDIUM` (0.30 a 0.60) o `HIGH` (>= 0.60) |
| `model_version` | Versión del modelo del Registry que respondió |
| `request_id` | Identificador único de cada llamada, para trazar |

Los datos de entrada se validan con Pydantic **antes** de llegar al modelo:
los campos categóricos solo aceptan los valores que existen en el dataset
(por ejemplo `Contract` solo acepta `Month-to-month`, `One year` o `Two year`),
`tenure` y `MonthlyCharges` no pueden ser negativos, y faltar un campo obligatorio
es error. Una entrada inválida devuelve **422** con el detalle de qué falló.
`TotalCharges` y `customerID` son opcionales (`customerID` es solo para trazabilidad y
no influye en la predicción).

| Código | Cuándo |
|---|---|
| 200 | Predicción correcta |
| 422 | Entrada inválida (campo faltante, valor fuera de lista, número negativo, etc.) |
| 503 | El modelo no está disponible (por ejemplo, credenciales incorrectas al arrancar) |
| 500 | Falló el pipeline al predecir (error inesperado) |

## Decisiones de diseño

- **El modelo sale del Model Registry, no de un archivo.** Así la API queda atada al
  modelo y al run reales de la etapa 1 (misma trazabilidad), y cambiar de modelo es
  registrar una versión nueva, sin tocar la imagen.
- **Se carga una sola vez, al arrancar.** No en cada request: es más rápido y la versión
  que responde queda fija mientras el servicio esté arriba.
- **Si el modelo no carga, la API igual levanta.** `/health` pasa a `degraded` y
  `/predict` responde 503. Preferimos eso a que el contenedor se caiga en bucle sin
  explicar nada.
- **`mlflow-skinny` en la imagen.** La API solo es *cliente* de MLflow (habla con
  DagsHub por HTTPS), así que no hace falta el paquete completo con sus dependencias de
  servidor. La imagen queda más liviana. Quedó fijado en la misma versión que se usó
  para entrenar (2.16.2).
- **Dos archivos de dependencias.** `requirements-api.txt` es lo que va a producción;
  `requirements-test.txt` le suma pytest, httpx y lo necesario para simular el Registry
  en local.
- **Umbrales de riesgo.** 0.30 y 0.60 son una decisión nuestra, alineada con priorizar el
  recall que justificamos en la etapa 1. Están en `app/risk.py` y tienen tests.
- **Sin secretos en el repo ni en la imagen.** Todo entra por variables de entorno;
  `docker-compose.yml` las lee del `.env` local.

## Por si algo no anda

- **`Docker Desktop - Virtual Machine Platform not enabled`**: en PowerShell como
  administrador, `Enable-WindowsOptionalFeature -Online -FeatureName VirtualMachinePlatform`
  y reiniciar. Si Docker pide virtualización, hay que activarla en la BIOS
  (Intel VT-x / AMD SVM).
- **`/health` responde `degraded` o `/predict` da 503**: el modelo no se pudo cargar.
  Mirá el log (`docker compose logs api`): lo más común es un token mal copiado, una
  `MLFLOW_TRACKING_URI` incorrecta o un `MODEL_NAME` que no existe en el Registry.
- **`port is already allocated`**: el 8000 está ocupado (por ejemplo por un `uvicorn`
  que quedó abierto). Cerralo, o cambiá `HOST_PORT` en el `.env`.
- **Los tests fallan al importar mlflow con un error de SQLAlchemy**: instalaste una
  SQLAlchemy 2.1. Usá `pip install -r requirements-test.txt`, que la deja en 2.0.x.
- **Las variables del `.env` no se ven con `uvicorn` a mano**: uvicorn no lee el `.env`;
  eso lo hace Docker Compose. Para probar sin Docker hay que setear las variables en la
  terminal (`$env:MLFLOW_TRACKING_URI="..."` en PowerShell, `export ...` en bash).

## Evidencias

Capturas del flujo completo en `docs/evidencias/`

## Entrega

La versión evaluada de esta etapa está marcada con el tag `entrega-2`.