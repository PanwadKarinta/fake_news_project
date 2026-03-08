from fastapi import FastAPI
from pydantic import BaseModel
import joblib

model = joblib.load("best_random_forest_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

app = FastAPI()

class NewsRequest(BaseModel):
    text: str

@app.post("/predict")
def predict_news(request: NewsRequest):
    transformed_text = vectorizer.transform([request.text])

    prediction = model.predict(transformed_text)[0]
    probability = model.predict_proba(transformed_text).max()

    return {
        "prediction": prediction,
        "confidence": round(float(probability) * 100, 2)
    }
