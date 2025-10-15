from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.model_service import predict_stress

app = FastAPI(title="Stress Level Prediction API", version="1.0.0")

# Enable CORS for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class StressInput(BaseModel):
    gender: str
    NationalITy: str
    PlaceofBirth: str
    StageID: str
    GradeID: str
    Topic: str
    GPA: float | None = None
    ClassesPerWeek: float
    WorkloadIndex: float
    ExamPressure: float
    # Add other required fields found in CSV header


@app.post("/predict")
def get_prediction(payload: StressInput):
    input_data = payload.dict()
    try:
        prediction = predict_stress(input_data)
        return {"stress_level": prediction}
    except Exception as e:
        return {"error": str(e)}
