import logging
from logging.handlers import RotatingFileHandler

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

from .config import LOG_DIR


def get_logger(name: str) -> logging.Logger:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = RotatingFileHandler(LOG_DIR / "project.log", maxBytes=1_000_000, backupCount=2, encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
        logger.addHandler(handler)
        logger.addHandler(logging.StreamHandler())
    return logger


def add_features(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.drop(columns=["ID", "Credit_Score"], errors="ignore").copy()
    income = pd.to_numeric(data["Annual_Income"], errors="coerce").replace(0, np.nan)
    salary = pd.to_numeric(data["Monthly_Inhand_Salary"], errors="coerce").replace(0, np.nan)
    data["Annual_Income"] = income.fillna(salary * 12)
    income = data["Annual_Income"]
    salary = salary.fillna(income / 12)
    data["Monthly_Inhand_Salary"] = salary

    emi = pd.to_numeric(data["Total_EMI_per_month"], errors="coerce").replace(0, np.nan)
    valid = emi.notna() & salary.notna() & (salary > 0)
    emi_salary_ratio = (emi[valid] / salary[valid]).median() if valid.any() else 0.1
    data["Total_EMI_per_month"] = emi.fillna(salary * emi_salary_ratio)
    data["Debt_to_Annual_Income"] = data["Outstanding_Debt"] / income
    data["EMI_to_Monthly_Salary"] = data["Total_EMI_per_month"] / salary
    return data.drop(columns=["Monthly_Inhand_Salary"], errors="ignore")


def make_preprocessor(features: pd.DataFrame) -> ColumnTransformer:
    numeric = features.select_dtypes(include="number").columns.tolist()
    categorical = features.select_dtypes(exclude="number").columns.tolist()
    numeric_pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", MinMaxScaler(clip=True))])
    categorical_pipe = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                                 ("encode", OneHotEncoder(handle_unknown="ignore"))])
    return ColumnTransformer([("numeric", numeric_pipe, numeric), ("categorical", categorical_pipe, categorical)])
