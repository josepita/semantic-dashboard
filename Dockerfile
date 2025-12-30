# Dockerfile base para Embedding Insights Suite
# ==================================================
# Soporta las 3 aplicaciones con requirements compartidos

FROM python:3.12-slim

# Metadata
LABEL maintainer="Embedding Insights"
LABEL description="SEO Multi-Project Suite with Streamlit"
LABEL version="1.0.0"

# Variables de entorno para Python
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Crear directorio de trabajo
WORKDIR /app

# Copiar archivos de requirements primero (para cache de Docker)
COPY shared/requirements.txt /app/shared/requirements.txt
COPY apps/gsc-insights/requirements.txt /app/apps/gsc-insights/requirements.txt
COPY apps/content-analyzer/requirements.txt /app/apps/content-analyzer/requirements.txt
COPY apps/linking-optimizer/requirements.txt /app/apps/linking-optimizer/requirements.txt

# Instalar dependencias comunes
RUN pip install --upgrade pip && \
    pip install -r /app/shared/requirements.txt

# Descargar modelos de spaCy (necesario para NLP)
RUN python -m spacy download es_core_news_sm || echo "Spanish model not available" && \
    python -m spacy download en_core_web_sm || echo "English model downloaded"

# Copiar código de la aplicación
COPY . /app/

# Crear directorio para workspace (persistent volume)
RUN mkdir -p /app/workspace/projects && \
    mkdir -p /app/workspace/exports && \
    chmod -R 777 /app/workspace

# Exponer puerto de Streamlit (por defecto 8501)
EXPOSE 8501

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Comando por defecto (se sobrescribe en docker-compose)
CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
