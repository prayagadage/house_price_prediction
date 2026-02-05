import joblib
import numpy as np
import pandas as pd

import os

# Define model path relative to this script
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
MODEL_PATH = os.path.join(project_root, "models", "rent_prediction_model.pkl")


def load_model():
    return joblib.load(MODEL_PATH)


def predict_rent(input_data: dict):
    """
    input_data: dictionary with raw feature values
    returns: predicted rent in INR
    """
    model = load_model()

    df = pd.DataFrame([input_data])
    log_rent_pred = model.predict(df)[0]

    rent_pred = np.exp(log_rent_pred)
    return round(rent_pred, 2)


if __name__ == "__main__":
    sample_input = {
        "BHK": 2,
        "Size": 900,
        "Bathroom": 2,
        "current_floor": 2,
        "total_floors": 5,
        "Area Type": "Super Area",
        "Area Locality": "Other",
        "City": "Bangalore",
        "Furnishing Status": "Semi-Furnished",
        "Tenant Preferred": "Bachelors/Family"
    }

    print("Predicted Rent (₹):", predict_rent(sample_input))
