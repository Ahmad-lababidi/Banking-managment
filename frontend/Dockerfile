FROM python:3.11-slim

WORKDIR /app
COPY frontend/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt
COPY frontend/ /app/frontend/
COPY model/data/Train.csv /app/data/Train.csv
ENV PYTHONUNBUFFERED=1 API_BASE_URL=http://backend:8000
EXPOSE 8501
CMD ["streamlit", "run", "frontend/app.py", "--server.address=0.0.0.0", "--server.port=8501"]
