import logging
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from .config import CLEANED_DIR, TARGET, TEST_FILE, TRAIN_FILE
from .utils import add_features

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("data_preparation")


def prepare_data():
    log.info("Reading raw CSVs; originals will be left unchanged")
    train = pd.read_csv(TRAIN_FILE)
    test = pd.read_csv(TEST_FILE)
    y_train = train[TARGET].copy()
    y_test = test[TARGET].copy()
    x_train = add_features(train)
    x_test = add_features(test)

    numeric_columns = x_train.select_dtypes(include="number").columns.tolist()
    categorical_columns = x_train.select_dtypes(exclude="number").columns.tolist()

    medians = x_train[numeric_columns].median()
    train_numeric = x_train[numeric_columns].fillna(medians)
    test_numeric = x_test[numeric_columns].fillna(medians)
    scaler = MinMaxScaler(clip=True)
    train_numeric = pd.DataFrame(scaler.fit_transform(train_numeric), columns=numeric_columns, index=train_numeric.index)
    test_numeric = pd.DataFrame(scaler.transform(test_numeric), columns=numeric_columns, index=test_numeric.index)
    train_numeric.columns = [f"numeric_{name}" for name in numeric_columns]
    test_numeric.columns = train_numeric.columns

    modes = {column: x_train[column].mode(dropna=True).iloc[0] for column in categorical_columns}
    train_categories = x_train[categorical_columns].fillna(modes)
    test_categories = x_test[categorical_columns].fillna(modes)
    prefix = [f"category_{name}" for name in categorical_columns]
    train_categories = pd.get_dummies(train_categories, prefix=prefix, dtype="int8")
    test_categories = pd.get_dummies(test_categories, prefix=prefix, dtype="int8")
    test_categories = test_categories.reindex(columns=train_categories.columns, fill_value=0)

    cleaned_train = pd.concat([train_numeric, train_categories], axis=1)
    cleaned_test = pd.concat([test_numeric, test_categories], axis=1)
    cleaned_train[TARGET] = y_train.to_numpy()
    cleaned_test[TARGET] = y_test.to_numpy()

    CLEANED_DIR.mkdir(parents=True, exist_ok=True)
    train_path = CLEANED_DIR / "Cleaned_Train.csv"
    test_path = CLEANED_DIR / "Cleaned_Test.csv"
    cleaned_train.to_csv(train_path, index=False)
    cleaned_test.to_csv(test_path, index=False)
    log.info("Saved separate cleaned files to %s", CLEANED_DIR)
    return train_path, test_path


if __name__ == "__main__":
    for path in prepare_data():
        print(path)
