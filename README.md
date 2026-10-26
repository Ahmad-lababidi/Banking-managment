# Bank Credit Score Classification

This project predicts whether a customer's credit score is Poor, Standard, or Good. It contains a model training workflow, a FastAPI backend, and a Streamlit frontend.

## Start with Docker

Install and start Docker Desktop. Open a terminal in the project folder and run:

```bash
docker compose up --build
```

On first start, the trainer prepares the separate cleaned CSV files, trains three models, and saves model metrics. The API starts after training completes.

- Streamlit app: <http://localhost:8501>
- FastAPI interactive docs: <http://localhost:8000/docs>
- API health: <http://localhost:8000/health>

Stop the app with `Ctrl+C`. Run `docker compose down` to stop and remove the containers. The trained model remains in a Docker volume for the next run. The cleaned CSV files are saved in `model/cleaned_data/`.

## What is in each folder

- `backend/` contains the FastAPI app and its Dockerfile.
- `frontend/` contains the Streamlit app and its Dockerfile.
- `model/` contains preprocessing and training code.
- `model/data/` contains the original `Train.csv`, `Test.csv`, and `Fields.txt` files.
- `model/cleaned_data/` contains separate cleaned copies of the train and test data.
- `model/artifacts/` is where locally trained models and metrics are saved.
- `logs/` stores the training log.
- `docker-compose.yml` builds and runs the trainer, backend, and frontend together.

## Run with Python instead

Install Python 3.10 or newer. From the project folder, create and activate a virtual environment, then install the dependencies:

```bash
python -m venv .venv
```

On Windows, activate it with `.venv\Scripts\activate`. On macOS or Linux, use `source .venv/bin/activate`.

```bash
pip install -r requirements.txt
python -m model.train_model
```

Start the API in one terminal:

```bash
uvicorn backend.api:app --host 127.0.0.1 --port 8000
```

Start the frontend in another terminal:

```bash
streamlit run frontend/app.py
```

To create only the cleaned CSV files, run `python -m model.prepare_data`. The original CSV files are not changed.

## Data preparation and model training

The preparation step fills missing numbers using training medians, fills missing categories using the most common training category, scales numeric columns, adds debt-to-income and EMI-to-salary ratios, and one-hot encodes categories. It learns these values from the training data and reuses them for the test data. It does not overwrite the raw CSVs.

Training uses a stratified validation split and compares logistic regression, a decision tree, and a random forest. Three-fold cross-validation tunes the random forest. Accuracy and weighted F1 are reported for validation and the supplied test set. Each model and split also records per-class precision, recall, F1, support, error rate, the most common incorrect prediction, and a confusion matrix. Balanced class weights account for the uneven class counts.

## API example

Send a `POST` request to `http://localhost:8000/predict` with customer values:

```json
{
  "model": "random_forest",
  "features": {
    "Age": 33,
    "Annual_Income": 50000,
    "Monthly_Inhand_Salary": 4000,
    "Interest_Rate": 14,
    "Credit_Mix": "Standard",
    "Outstanding_Debt": 1400,
    "Credit_Utilization_Ratio": 32,
    "Payment_of_Min_Amount": "Yes",
    "Total_EMI_per_month": 100,
    "Monthly_Balance": 400
  }
}
```

Choose `logistic_regression`, `decision_tree`, or `random_forest`. Leave out `model` to use the model selected by validation weighted F1. The response contains the predicted class and its probabilities.

The frontend shows training data summaries, evaluation metrics, per-class error analysis for validation or test data, confusion matrices, and prediction results. It has no data download controls. The test CSV is not included in the frontend container.
