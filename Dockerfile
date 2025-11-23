FROM python:3.11-slim

# Evita problemas de buffering
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Copiar apenas requirements antes para melhor uso de cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar o restante do código
COPY . .

EXPOSE 8000

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]

