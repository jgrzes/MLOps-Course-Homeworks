import joblib
from fastapi import FastAPI
from sentence_transformers import SentenceTransformer

from api.models.sentiment import PredictRequest, PredictResponse

app = FastAPI()

LABELS = {0: "negative", 1: "neutral", 2: "positive"}

transformer = SentenceTransformer("model/sentence_transformer.model")
classifier = joblib.load("model/classifier.joblib")


@app.post("/predict")
def predict(request: PredictRequest) -> PredictResponse:
    embedding = transformer.encode(request.text)
    prediction = classifier.predict([embedding])[0]
    label = LABELS[prediction]
    return PredictResponse(prediction=label)
