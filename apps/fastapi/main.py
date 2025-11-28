import os
import joblib
import pandas as pd
from datetime import datetime
from fastapi import FastAPI
from pydantic import BaseModel

# =============================
# Configuración
# =============================

# Ruta absoluta del modelo local
MODEL_PATH = os.getenv(
    "MODEL_PATH",
    "/Users/juanse/mlops-final/mlops-platform/data/models/chunk_007/model.pkl"
)

app = FastAPI(
    title="House Price Inference (Local Model)",
    description="API que realiza inferencias usando un modelo cargado desde el sistema de archivos.",
)

model = None


# =============================
# Esquema entrada/salida
# =============================

class HouseFeatures(BaseModel):
    bed: float
    bath: float
    acre_lot: float
    house_size: float


class PredictionResponse(BaseModel):
    prediction: float
    model_path: str
    timestamp: datetime


# =============================
# Utilidades
# =============================

def load_model():
    """Carga el modelo desde un archivo .pkl local."""
    global model

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"No se encontró el modelo en {MODEL_PATH}")

    model = joblib.load(MODEL_PATH)


# =============================
# Eventos
# =============================

@app.on_event("startup")
def startup_event():
    load_model()


# =============================
# Endpoints
# =============================

@app.get("/")
def root():
    return {
        "message": "FastAPI is running with a local model!",
        "model_path": MODEL_PATH,
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(features: HouseFeatures):
    if model is None:
        raise RuntimeError("Modelo no cargado.")

    df = pd.DataFrame([features.dict()])
    y_pred = model.predict(df)
    pred_value = float(y_pred[0])

    ts = datetime.utcnow()

    return PredictionResponse(
        prediction=pred_value,
        model_path=MODEL_PATH,
        timestamp=ts,
    )
