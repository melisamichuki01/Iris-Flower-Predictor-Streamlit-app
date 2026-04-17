import pickle
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List

# ── Load model once at startup ──────────────────────
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

CLASS_NAMES = ["setosa", "versicolor", "virginica"]

app = FastAPI(title="Iris Classifier API", version="1.0.0")

# ── Schemas ─────────────────────────────────────────
class PredictRequest(BaseModel):
    features: List[float]  # [sepal_l, sepal_w, petal_l, petal_w]

class PredictResponse(BaseModel):
    prediction: int
    class_name: str
    confidence: float

# ── Endpoints ────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/v1/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if len(req.features) != 4:
        raise HTTPException(status_code=400, detail="Need exactly 4 features")

    proba = model.predict_proba([req.features])[0]
    pred  = int(proba.argmax())

    return PredictResponse(
        prediction=pred,
        class_name=CLASS_NAMES[pred],
        confidence=round(float(proba.max()), 4),
    )