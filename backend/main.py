from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib

from utils import map_model_size,evaluate_gpus

class ModelConfig(BaseModel):
    model_size: str
    batch_size: int
    learning_rate: float
    seq_length: int
    run_id: int

try:
    bundle = joblib.load("model_bundle.pkl")
    model = bundle['model']
    preprocessor = bundle['preprocessor']
except Exception as e:
    raise RuntimeError(f"Failed to load model bundle: {e}")

app = FastAPI()

@app.get("/")
def home():
    return {"message":"Welcome to GPUlympics"}

@app.get("/health")
def health():
    if model and preprocessor:
        return {"status":"OK"}
    else:
        raise HTTPException(status_code=503,detail="Model or preprocessor or both not loaded")

@app.post("/predict")
def predict(config: ModelConfig):
    try:
        results,best_gpus = evaluate_gpus(
            model = model,
            preprocessor = preprocessor,
            model_size = config.model_size,
            batch_size = config.batch_size,
            learning_rate = config.learning_rate,
            seq_length = config.seq_length,
            run_id = config.run_id
        )
    
        return {
            "gpu_comparison":results,
            "best_gpus":best_gpus
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")




