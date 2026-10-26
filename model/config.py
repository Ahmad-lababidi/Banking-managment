import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
MODEL_DIR = Path(os.getenv("MODEL_DIR", str(ROOT)))
if os.getenv("CLEANED_DIR"):
    CLEANED_DIR = Path(os.environ["CLEANED_DIR"])
else:
    CLEANED_DIR = ROOT / "cleaned_data"
LOG_DIR = Path(os.getenv("LOG_DIR", str(ROOT.parent / "logs")))
TRAIN_FILE = DATA_DIR / "Train.csv"
TEST_FILE = DATA_DIR / "Test.csv"
MODEL_FILE = MODEL_DIR / "models.joblib"
METRICS_FILE = MODEL_DIR / "model_metrics.json"
TARGET = "Credit_Score"
RANDOM_STATE = 42
