# Stress Prediction API

This folder contains a minimal FastAPI scaffold for serving a neural-network-based stress prediction model.

Files added:

- `app/` - API package with `main.py`, `model_service.py`, and `preprocessing.py`.
- `requirements.txt` - Python dependencies for the API.
- `Dockerfile` - Containerizes the API.
- `app/saved_models/` - placeholder location for model artifacts.

Model artifacts expected in `app/saved_models/`:

- `nn_stress_model.h5` - trained Keras model
- `nn_scaler.joblib` - fitted scaler (optional)
- `nn_label_encoders.joblib` - dict of fitted label encoders (optional)
- `nn_target_encoder.joblib` - target label encoder (optional)

Replace the placeholder files in `saved_models/` with your real artifacts before using the service.

Run locally (recommended in a virtualenv):

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

Using Docker:

```bash
docker build -t stress-api:latest .
docker run -p 8000:8000 stress-api:latest
```

API endpoints:

- `GET /health` - health check
- `POST /predict` - JSON body: {"data": {"feature1": value1, ...}}

Notes:

- The scaffold is intentionally defensive — if model artifacts are missing the service returns a fallback prediction. Replace placeholders with your trained model and encoders for production use.
