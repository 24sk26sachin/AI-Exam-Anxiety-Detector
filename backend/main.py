from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from .model_utils import get_predictor

app = FastAPI(title="AI Based Exam Anxiety Detector API")

class PredictionRequest(BaseModel):
    text: str

class PredictionResponse(BaseModel):
    anxiety_level: str
    confidence: float

@app.get("/")
def read_root():
    return {"message": "Welcome to the AI Based Exam Anxiety Detector API"}

@app.post("/predict", response_model=PredictionResponse)
def predict_anxiety(request: PredictionRequest):
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text input cannot be empty")
    
    try:
        predictor = get_predictor()
        result = predictor.predict(request.text)
        return PredictionResponse(
            anxiety_level=result["prediction"],
            confidence=result["confidence"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
