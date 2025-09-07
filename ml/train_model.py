"""
train_model.py
==============

This script trains a machine learning model to predict book prices based on
metadata scraped from synthetic HTML pages. The input dataset should
contain at least the following columns: ``category``, ``rating``,
``availability``, ``description`` and ``price``. Categorical features
are one‑hot encoded and text features are summarised by their length.

The resulting regression model and preprocessing pipeline are saved to
disk as a ``joblib`` file. An evaluation report is written to a
markdown file summarising model performance metrics on a held‑out test
set.

Usage:
    python train_model.py

Outputs:
    models/price_model.pkl         - Serialized model pipeline
    models/model_performance.md    - Markdown report of evaluation metrics
"""

import os
import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer

# Paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "scraped_books_local.csv")
MODEL_DIR = os.path.join(PROJECT_ROOT, "ml", "models")
MODEL_FILE = os.path.join(MODEL_DIR, "price_model.pkl")
REPORT_FILE = os.path.join(MODEL_DIR, "model_performance.md")


def load_data(path: str) -> pd.DataFrame:
    """Load the dataset from a CSV file."""
    return pd.read_csv(path)


def preprocess_features(df: pd.DataFrame) -> tuple:
    """Prepare feature matrix X and target vector y.

    This function creates an additional feature for description length and
    returns the feature matrix and target vector.

    Args:
        df: Input DataFrame.

    Returns:
        X: DataFrame of features
        y: Series of target values (prices)
    """
    # Compute description length in words
    df = df.copy()
    df['description_length'] = df['description'].fillna('').apply(lambda x: len(str(x).split()))

    feature_cols = ['category', 'rating', 'availability', 'description_length']
    X = df[feature_cols]
    y = df['price']
    return X, y


def build_model_pipeline(categorical_features: list, numeric_features: list) -> Pipeline:
    """Construct a preprocessing and regression pipeline."""
    # One-hot encode categorical features
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')

    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', categorical_transformer, categorical_features),
            ('num', 'passthrough', numeric_features),
        ]
    )

    # Use RandomForestRegressor for its ability to handle non-linearities
    model = RandomForestRegressor(n_estimators=200, random_state=42)

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', model),
    ])
    return pipeline


def evaluate_model(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """Evaluate the model and return a dictionary of metrics."""
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = mse ** 0.5
    r2 = r2_score(y_test, y_pred)
    return {'MAE': mae, 'RMSE': rmse, 'R2': r2}


def save_report(metrics: dict, path: str) -> None:
    """Write a markdown report summarising model performance."""
    lines = [
        "# Model Performance Report\n",
        "This report summarises the performance of the RandomForestRegressor on a 20% held‑out test set.\n",
        "## Metrics\n",
    ]
    for name, value in metrics.items():
        lines.append(f"* **{name}**: {value:.4f}\n")
    with open(path, 'w', encoding='utf-8') as f:
        f.writelines(lines)


def main() -> None:
    # Ensure model directory exists
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Load data
    df = load_data(DATA_PATH)
    X, y = preprocess_features(df)

    categorical_features = ['category']
    numeric_features = ['rating', 'availability', 'description_length']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Build and train model
    pipeline = build_model_pipeline(categorical_features, numeric_features)
    pipeline.fit(X_train, y_train)

    # Evaluate
    metrics = evaluate_model(pipeline, X_test, y_test)
    print("Model evaluation:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    # Save model and report
    joblib.dump(pipeline, MODEL_FILE)
    save_report(metrics, REPORT_FILE)
    print(f"Model saved to {MODEL_FILE}")
    print(f"Report saved to {REPORT_FILE}")


if __name__ == '__main__':
    main()