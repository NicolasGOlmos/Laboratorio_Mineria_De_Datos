# Guía de conexión: GitHub + DVC + DagsHub

Esta guía asume que ya tenés el proyecto armado localmente (carpeta
`customer-churn-ml/`, con `src/`, `data/`, `tests/`, etc.) y que ya
creaste el repositorio vacío en GitHub y el proyecto en DagsHub.

Reemplazá estos placeholders por tus datos reales en todos los
comandos de abajo:

| Placeholder | Qué va ahí |
|---|---|
| `<GITHUB_USER>` | Tu usuario u organización de GitHub |
| `<REPO>` | Nombre del repositorio (el mismo en GitHub y en DagsHub) |
| `<DAGSHUB_USER>` | Tu usuario de DagsHub |
| `<DAGSHUB_TOKEN>` | Token de acceso de DagsHub (Settings → Tokens) |

## 1. Subir el código a GitHub

```bash
cd customer-churn-ml
git init
git branch -M main
git remote add origin https://github.com/<GITHUB_USER>/<REPO>.git

git add .
git commit -m "primer commit: estructura del proyecto, pipeline y tests"
git push -u origin main
```

Con el `.gitignore` que ya tenemos, esto sube el código
(`src/`, `tests/`, `notebooks/`), el `README.md`, `requirements.txt`,
`pytest.ini` y `.vscode/settings.json` — pero **no** sube los `.csv`
de datos ni los artefactos de MLflow local, que quedan excluidos a
propósito.

## 2. Inicializar DVC y conectar el remote de DagsHub

DagsHub expone automáticamente un remote de DVC por cada repositorio,
en esta URL:

```
https://dagshub.com/<DAGSHUB_USER>/<REPO>.dvc
```

```bash
dvc init

dvc remote add origin https://dagshub.com/<DAGSHUB_USER>/<REPO>.dvc
dvc remote default origin

# Autenticación con token (nunca uses la contraseña de tu cuenta acá)
dvc remote modify origin --local auth basic
dvc remote modify origin --local user <DAGSHUB_USER>
dvc remote modify origin --local password <DAGSHUB_TOKEN>
```

El flag `--local` guarda estas credenciales en `.dvc/config.local`,
que Git ignora automáticamente. Así el token nunca termina expuesto
en el repositorio (regla de seguridad del proyecto: nada de tokens ni
contraseñas commiteadas).

## 3. Versionar el dataset con DVC

```bash
dvc add data/raw/customer_churn_historical.csv

git add data/raw/customer_churn_historical.csv.dvc data/raw/.gitignore .dvc/config
git commit -m "data: versionar dataset historico con DVC"
git push

dvc push
```

Fijate que acá hay **dos pushes distintos**: `git push` sube el
archivo `.dvc` (un puntero de texto, chiquito) a GitHub, y `dvc push`
sube el contenido real del CSV a DagsHub. Los datos nunca se suben
directo con `git`.

**Verificación:** en la pestaña "Data" de tu proyecto en DagsHub,
deberías ver `customer_churn_historical.csv` listado con su hash y
tamaño.

## 4. Conectar el entrenamiento al MLflow de DagsHub

DagsHub también expone un servidor de MLflow por repositorio:

```
https://dagshub.com/<DAGSHUB_USER>/<REPO>.mlflow
```

```bash
export MLFLOW_TRACKING_URI="https://dagshub.com/<DAGSHUB_USER>/<REPO>.mlflow"
export MLFLOW_TRACKING_USERNAME="<DAGSHUB_USER>"
export MLFLOW_TRACKING_PASSWORD="<DAGSHUB_TOKEN>"

python -m src.training.train
```

En Windows PowerShell, `export` se reemplaza por:

```powershell
$env:MLFLOW_TRACKING_URI="https://dagshub.com/<DAGSHUB_USER>/<REPO>.mlflow"
$env:MLFLOW_TRACKING_USERNAME="<DAGSHUB_USER>"
$env:MLFLOW_TRACKING_PASSWORD="<DAGSHUB_TOKEN>"
```

No hace falta tocar nada del código: `src/training/train.py` no tiene
ninguna URL hardcodeada, así que apenas seteás estas variables de
entorno, MLflow manda todo (params, métricas, el modelo) directo a
DagsHub en vez de guardarlo solo en tu máquina.

**Verificación:** en la pestaña "Experiments" de DagsHub, deberían
aparecer los 6 runs del experimento `customer-churn-model-selection`
(`baseline-dummy`, `logreg-l2-balanced`, `logreg-l2-C0.1`,
`rf-100-depth6`, `rf-300-depth10`, `rf-300-depth-none`).

## 5. Registrar el modelo candidato

```bash
python -m src.training.register_model --run-id <RUN_ID_DEL_MODELO_GANADOR>
```

El `run_id` lo sacás de la salida de la terminal del paso anterior, o
directamente de la tabla de runs en la UI de DagsHub.

**Verificación:** en la pestaña "Models" de DagsHub debería aparecer
`customer-churn-classifier`, con su versión y los tags de
trazabilidad (`source_run_id`, `source_experiment_id`).

## 6. Configurar secrets para que GitHub Actions también funcione

En GitHub: `Settings → Secrets and variables → Actions → New repository secret`

Creá dos secrets:
- `DAGSHUB_USER` → tu usuario de DagsHub
- `DAGSHUB_TOKEN` → tu token de DagsHub

Con esto, el workflow en `.github/workflows/tests.yml` puede hacer
`dvc pull` y correr los tests automáticamente en cada push, sin que
las credenciales queden expuestas en ningún lado del código.

## 7. Tag de la entrega

Sobre el commit exacto que vas a presentar como Entrega 1:

```bash
git tag entrega-1
git push origin entrega-1
```

Esto deja el requisito de evidencia mínima cubierto: "Tag Git
entrega-1 sobre el commit exacto presentado".

## Orden mental para no perderte

```
git push          -> sube el codigo (y los .dvc, que son solo punteros)
dvc push          -> sube el csv real y pesado
python -m src...  -> entrena, y MLflow manda todo solo a DagsHub
```
