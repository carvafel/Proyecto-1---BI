# Imagen base alineada con tu entorno local (3.10) para evitar sorpresas con wheels
FROM python:3.10-slim

# Buenas prácticas de Python
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Instalar utilidades del sistema que usaremos (curl para healthcheck)
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

# Directorio de trabajo
WORKDIR /app

# Copiamos primero requirements para aprovechar cache
COPY requirements.txt .

# Recomendación: fija versiones compatibles con tu modelo
# (si ya las tienes fijadas en requirements.txt no hace falta tocar nada aquí)
RUN pip install --no-cache-dir -r requirements.txt

# Copiamos el resto del proyecto
COPY . .

# Usuario no-root
RUN useradd -u 1000 -m appuser && chown -R appuser:appuser /app
USER appuser

# Exponer puerto
EXPOSE 8000


# Import path de tu app: scripts.main_api:app
CMD ["python", "-m", "uvicorn", "scripts.main_api:app", "--host", "0.0.0.0", "--port", "8000"]