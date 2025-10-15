import os
from typing import Any, Dict
import pandas as pd
import numpy as np


MODEL_PATH = "app/saved_models/nn_stress_model.h5"
SCALER_PATH = "app/saved_models/nn_scaler.joblib"
ENCODERS_PATH = "app/saved_models/nn_label_encoders.joblib"
TARGET_ENCODER_PATH = "app/saved_models/nn_target_encoder.joblib"


# Globals that will hold either the real artifacts or dev fallbacks
_model = None
scaler = None
label_encoders = {}
label_encoder = None


def _use_dev_fallback():
    """Create lightweight fallback objects so the API can run without TF/joblib/models."""

    class MockScaler:
        def transform(self, df):
            # try to return numeric array; convert non-numeric to 0
            arr = []
            for _, row in df.iterrows():
                values = []
                for v in row.tolist():
                    try:
                        values.append(float(v))
                    except Exception:
                        values.append(0.0)
                arr.append(values)
            return np.array(arr, dtype=float)

    class MockModel:
        def predict(self, X):
            # Simple heuristic: higher sum -> higher stress
            sums = np.sum(X, axis=1)
            out = []
            for s in sums:
                if s > 30:
                    out.append([0.05, 0.2, 0.75])  # high
                elif s > 12:
                    out.append([0.1, 0.7, 0.2])    # medium
                else:
                    out.append([0.8, 0.15, 0.05])  # low
            return np.array(out)

    class MockTargetEncoder:
        def inverse_transform(self, arr):
            mapping = {0: "low", 1: "medium", 2: "high"}
            return [mapping.get(int(a), "unknown") for a in arr]

    return MockModel(), MockScaler(), {}, MockTargetEncoder()


def _load_artifacts():
    """Attempt to load real model artifacts; fall back to dev mocks on any failure."""
    global _model, scaler, label_encoders, label_encoder

    try:
        import joblib
        import tensorflow as tf

        # load model if available
        if os.path.exists(MODEL_PATH):
            _model = tf.keras.models.load_model(MODEL_PATH)
        else:
            _model = None

        # load scaler / encoders if available
        if os.path.exists(SCALER_PATH):
            scaler = joblib.load(SCALER_PATH)
        if os.path.exists(ENCODERS_PATH):
            label_encoders = joblib.load(ENCODERS_PATH)
        if os.path.exists(TARGET_ENCODER_PATH):
            label_encoder = joblib.load(TARGET_ENCODER_PATH)

        # If any required piece is missing, fall back
        if _model is None or scaler is None or label_encoder is None:
            raise RuntimeError("Missing model artifacts; using dev fallback")

    except Exception:
        # install lightweight fallbacks for development
        _model, scaler, label_encoders, label_encoder = _use_dev_fallback()


# Initialize on import
_load_artifacts()


def preprocess_input(input_dict: Dict[str, Any]):
    """Prepare a DataFrame from input dict, apply label encoders if available, and scale.

    The function is defensive so it works with both real and mock artifacts.
    """
    df = pd.DataFrame([input_dict])

    for col in df.columns:
        # fill missing values
        if df[col].isna().any():
            if col in label_encoders and hasattr(df[col], "mode"):
                mode_vals = df[col].mode()
                if not mode_vals.empty:
                    df[col] = df[col].fillna(mode_vals[0])
                else:
                    df[col] = df[col].fillna(0)
            else:
                # try numeric fill, otherwise 0
                try:
                    df[col] = pd.to_numeric(df[col])
                    df[col] = df[col].fillna(df[col].mean())
                except Exception:
                    df[col] = df[col].fillna(0)

        # apply label encoders when available
        if col in label_encoders:
            try:
                df[col] = label_encoders[col].transform(df[col])
            except Exception:
                # fallback: factorize categories
                df[col] = pd.factorize(df[col])[0]

    try:
        X_scaled = scaler.transform(df)
    except Exception:
        # final fallback: convert to numeric array
        X_scaled = df.apply(pd.to_numeric, errors="coerce").fillna(0).values

    return X_scaled


def predict_stress(input_dict: Dict[str, Any]):
    X_scaled = preprocess_input(input_dict)
    predictions = _model.predict(X_scaled)
    pred_class = np.argmax(predictions, axis=1)
    return label_encoder.inverse_transform(pred_class)[0]

