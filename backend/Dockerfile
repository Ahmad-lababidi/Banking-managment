FROM python:3.11-slim

WORKDIR /app
COPY backend/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt
COPY backend/ /app/backend/
COPY model/ /app/model/
COPY logs/ /app/logs/
ENV PYTHONPATH=/app PYTHONUNBUFFERED=1 MODEL_DIR=/app/model/artifacts LOG_DIR=/app/logs
EXPOSE 8000
CMD ["uvicorn", "backend.api:app", "--host", "0.0.0.0", "--port", "8000"]
