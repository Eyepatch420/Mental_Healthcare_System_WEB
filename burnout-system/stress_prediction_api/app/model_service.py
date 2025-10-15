import pandas as pd
import tensorflow as tf
import joblib
import numpy as np

MODEL_PATH = "app/saved_models/nn_stress_model.h5"
SCALER_PATH = "app/saved_models/nn_scaler.joblib"
ENCODERS_PATH = "app/saved_models/nn_label_encoders.joblib"
TARGET_ENCODER_PATH = "app/saved_models/nn_target_encoder.joblib"

# Load at app start
model = tf.keras.models.load_model(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
label_encoders = joblib.load(ENCODERS_PATH)
label_encoder = joblib.load(TARGET_ENCODER_PATH)

def preprocess_input(input_dict):
    df = pd.DataFrame([input_dict])

    for col in df.columns:
        if pd.isna(df[col]).any():
            if col in label_encoders:
                df[col] = df[col].fillna(df[col].mode()[0])
            else:
                df[col] = df[col].fillna(df[col].mean())

        if col in label_encoders:
            df[col] = label_encoders[col].transform(df[col])

    X_scaled = scaler.transform(df)
    return X_scaled

def predict_stress(input_dict):
    X_scaled = preprocess_input(input_dict)
    predictions = model.predict(X_scaled)
    pred_class = np.argmax(predictions, axis=1)
    return label_encoder.inverse_transform(pred_class)[0]
