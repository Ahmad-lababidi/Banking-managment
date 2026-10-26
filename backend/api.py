from functools import lru_cache
import json
import logging
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from model.config import METRICS_FILE, MODEL_FILE
from model.utils import add_features

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("credit_api")
app = FastAPI(title="Bank Credit Score API", version="1.0.0")


class PredictionRequest(BaseModel):
    features: dict[str, Any] = Field(description="Customer feature names and values")
    model: str | None = Field(default=None, description="Optional model name")


@lru_cache(maxsize=1)
def load_models():
    if not MODEL_FILE.exists():
        raise HTTPException(status_code=503, detail="Model is not ready yet.")
    return joblib.load(MODEL_FILE)


@app.get("/")
def root():
    return {"message": "Credit score API is running", "docs": "/docs"}


@app.get("/health")
def health():
    return {"status": "ok", "model_ready": MODEL_FILE.exists()}


@app.get("/metrics")
def metrics():
    if not METRICS_FILE.exists():
        raise HTTPException(status_code=503, detail="Metrics are not ready yet.")
    return json.loads(METRICS_FILE.read_text(encoding="utf-8"))


@app.post("/predict")
def predict(payload: PredictionRequest):
    bundle = load_models()
    name = payload.model or bundle["selected_model"]
    if name not in bundle["models"]:
        raise HTTPException(status_code=400, detail=f"Choose one of: {', '.join(bundle['models'])}")
    try:
        row = add_features(pd.DataFrame([payload.features])).reindex(columns=bundle["feature_columns"])
        model = bundle["models"][name]
        probabilities = model.predict_proba(row)[0]
        labels = model.classes_
        log.info("Prediction completed with %s", name)
        return {"model": name, "prediction": str(labels[probabilities.argmax()]),
                "probabilities": {str(label): round(float(value), 4)
                                  for label, value in zip(labels, probabilities)}}
    except HTTPException:
        raise
    except Exception as exc:
        log.exception("Prediction failed")
        raise HTTPException(status_code=400, detail="Invalid feature names or values.") from exc
