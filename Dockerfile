# Python 3.12: misma version que se valido en la Entrega 1 (mlflow 2.16.2
# + pandas/numpy pineados funcionan bien aca; 3.13 tiene problemas de
# wheels con numpy/pyarrow y 3.14 todavia no tiene wheels para varias
# de estas librerias). "-slim" para no cargar un sistema completo.
FROM python:3.12-slim

# Evita que Python genere .pyc y fuerza a que los logs salgan sin buffer
# (asi se ven en tiempo real con "docker compose logs -f").
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Primero copiamos solo el archivo de dependencias de produccion para
# aprovechar el cache de capas de Docker: si el codigo cambia pero las
# dependencias no, esta capa no se vuelve a instalar.
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

# Reción ahora copiamos el codigo de la aplicacion.
COPY app ./app

# Documentativo: el puerto real que se publica se define en
# docker-compose.yml (o con -p en "docker run"), esto solo declara
# que el proceso de adentro escucha en el 8000.
EXPOSE 8000

# Nada de secretos acá: MLFLOW_TRACKING_URI, MLFLOW_TRACKING_USERNAME,
# MLFLOW_TRACKING_PASSWORD, MODEL_NAME y MODEL_VERSION se inyectan en
# tiempo de ejecucion (docker-compose.yml con un .env, o "docker run -e").
# La imagen no los contiene hardcodeados en ningun lado.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
