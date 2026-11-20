
from pathlib import Path

import joblib
import pandas as pd

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


# ---------------------------------------
# 1. Load trained model
# ---------------------------------------

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "model.joblib"

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

model = joblib.load(MODEL_PATH)


# ---------------------------------------
# 2. Initialize FastAPI
# ---------------------------------------

app = FastAPI(
    title="Predictive Maintenance API",
    description="Machine failure prediction using Random Forest",
    version="1.0.0",
)

# Allow the Vercel dashboard (and local Vite development) to call this API.
# Keep the list explicit rather than allowing arbitrary origins.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "https://predictive-maintenance-mlops-frontend.vercel.app",
        "https://predictive-maintenance-mlop-git-5e4400-ankits-projects-5b336671.vercel.app",
    ],
    allow_origin_regex=r"https://predictive-maintenance-mlop.*-ankits-projects-5b336671\\.vercel\\.app",
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type"],
)


# ---------------------------------------
# 3. Define input data structure
# ---------------------------------------

class MachineData(BaseModel):
    machine_type: str = Field(pattern="^(L|M|H)$")
    air_temperature: float = Field(gt=0)
    process_temperature: float = Field(gt=0)
    rotational_speed: int = Field(gt=0)
    torque: float = Field(ge=0)
    tool_wear: int = Field(ge=0)


# ---------------------------------------
# 4. Root endpoint
# ---------------------------------------

@app.get("/")
def home():
    return {
        "message": "Predictive Maintenance API",
        "model": "Random Forest",
        "status": "running",
    }


# ---------------------------------------
# 5. Health check endpoint
# ---------------------------------------

@app.get("/health")
def health():
    return {
        "api_status": "healthy",
        "model_status": "loaded",
    }


# ---------------------------------------
# 6. Prediction endpoint
# ---------------------------------------

@app.post("/predict")
def predict(data: MachineData):

    try:
        # Convert API input into model feature format
        input_data = pd.DataFrame([{
            "Type": data.machine_type,
            "Air temperature [K]": data.air_temperature,
            "Process temperature [K]": data.process_temperature,
            "Rotational speed [rpm]": data.rotational_speed,
            "Torque [Nm]": data.torque,
            "Tool wear [min]": data.tool_wear,
        }])

        # Generate prediction
        prediction = int(model.predict(input_data)[0])

        # Calculate failure probability
        probability = float(
            model.predict_proba(input_data)[0][1]
        )

        return {
            "machine_failure": prediction,
            "prediction": (
                "Failure detected" if prediction == 1
                else "No failure detected"
            ),
            "failure_probability": round(probability, 4),
        }

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=str(error),
        )
