import joblib
import pytest
from fastapi.testclient import TestClient
from sentence_transformers import SentenceTransformer

from app import app

client = TestClient(app)


def test_model_loads():
    transformer = SentenceTransformer("model/sentence_transformer.model")
    classifier = joblib.load("model/classifier.joblib")

    assert transformer is not None
    assert classifier is not None


@pytest.mark.parametrize(
    "text",
    [
        "I love scikit-fingerprints!",
        "This is okay.",
        "I hate this.",
    ],
)
def test_inferenece(text):
    response = client.post("/predict", json={"text": text})
    data = response.json()

    assert response.status_code == 200
    assert "prediction" in data
    assert data["prediction"] in ["negative", "neutral", "positive"]


def test_output_format():
    response = client.post("/predict", json={"text": "I love scikit-fingerprints!"})

    try:
        data = response.json()
    except ValueError:
        assert False, "Response is not valid JSON"

    assert isinstance(data, dict)
    assert "prediction" in data
    assert isinstance(data["prediction"], str)
    assert data["prediction"] in ["negative", "neutral", "positive"]


@pytest.mark.parametrize(
    "invalid_text",
    [
        "",
        12345,
        None,
        ["This is a list"],
        {"key": "value"},
    ],
)
def test_invalid_input(invalid_text):
    response = client.post("/predict", json={"text": invalid_text})
    data = response.json()

    assert response.status_code == 422
    assert "detail" in data
