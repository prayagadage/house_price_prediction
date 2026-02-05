
import pandas as pd
import numpy as np
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error

from data_preprocessing import (
    load_data,
    drop_unused_columns,
    clean_size,
    reduce_locality_cardinality,
    parse_floor,
    log_transform_target
)


def build_pipeline(cat_features, num_features):
    """
    Build preprocessing + Ridge regression pipeline
    """

    # Numerical preprocessing
    numeric_transformer = Pipeline(steps=[
        ("scaler", StandardScaler())
    ])

    # Categorical preprocessing
    categorical_transformer = Pipeline(steps=[
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    # Column-wise preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, num_features),
            ("cat", categorical_transformer, cat_features)
        ]
    )

    # Ridge Regression model
    model = Ridge(alpha=1.0)

    # Full pipeline
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    return pipeline


def main():
    # ---------------------------
    # Load & preprocess data
    # ---------------------------
    # Define paths relative to this script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    
    data_path = os.path.join(project_root, "data", "raw", "House_Rent_Dataset.csv")
    model_path = os.path.join(project_root, "models", "rent_prediction_model.pkl")

    df = load_data(data_path)
    df = drop_unused_columns(df)
    df = clean_size(df)
    df = reduce_locality_cardinality(df)
    df = parse_floor(df)
    df = log_transform_target(df)

    # Features & target
    X = df.drop(columns=["Rent"])
    y = df["Rent"]

    # Categorical & numerical columns
    categorical_features = [
        "Area Type",
        "Area Locality",
        "City",
        "Furnishing Status",
        "Tenant Preferred"
    ]

    numerical_features = [
        "BHK",
        "Size",
        "Bathroom",
        "current_floor",
        "total_floors"
    ]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Build and train pipeline
    pipeline = build_pipeline(categorical_features, numerical_features)
    pipeline.fit(X_train, y_train)

    # Predictions
    y_pred = pipeline.predict(X_test)

    # Evaluation (log scale)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)

    print(f"RMSE (log scale): {rmse:.4f}")
    print(f"MAE  (log scale): {mae:.4f}")

    # Save trained pipeline
    joblib.dump(pipeline, model_path)
    print(f"✅ Model saved successfully at {model_path}")


if __name__ == "__main__":
    main()
