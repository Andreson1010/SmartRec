FROM python:3.11-slim

WORKDIR /app

# Dependências de sistema necessárias para scipy/numpy
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Instala dependências Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copia o código-fonte
COPY api/ api/
COPY ml/ ml/
COPY data/processed/ data/processed/

# Variáveis de ambiente padrão (sobrescritas pelo docker-compose)
ENV MLFLOW_TRACKING_URI=http://mlflow:5000
ENV API_PORT=8000
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
