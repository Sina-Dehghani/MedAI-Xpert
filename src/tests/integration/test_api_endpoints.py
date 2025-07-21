import pytest
import base64
import numpy as np
from PIL import Image
from io import BytesIO
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch
from src.apis.main import app, PredictionService


# Use a fixture to provide a test client
@pytest.fixture(scope="module")
def test_client():
    # Temporarily override PredictionService initialization for testing
    # We'll mock the internal predict methods later
    with patch(
        "src.prediction.prediction_service.PredictionService._initialize_models"
    ):
        client = TestClient(app)
        yield client


# Mock the PredictionService instance within the app for integration tests
@pytest.fixture(autouse=True)
def mock_prediction_service():
    mock_service = MagicMock(spec=PredictionService)
    # Configure mock_service.get_comprehensive_prediction to return a dummy response
    mock_service.get_comprehensive_prediction.return_value = {
        "patient_id": "PTest001",
        "overall_risk_score": 0.65,
        "image_analysis_results": {"predicted_class": 1, "probabilities": [0.3, 0.7]},
        "ehr_analysis_results": {"nlp_prediction": {"label": "LABEL_1", "score": 0.8}},
        "graph_analysis_results": [
            {"node_index": 0, "predicted_class": 1, "probabilities": [0.2, 0.8]}
        ],
        "recommendations": ["Risk detected.", "Further assessment needed."],
        "explainability": {
            "image_explanation": "mock image XAI",
            "nlp_explanation": "mock NLP XAI",
        },
    }
    # Patch the instance of PredictionService used by the FastAPI app
    with patch("src.apis.main.prediction_service", new=mock_service):
        yield mock_service


def test_health_check(test_client):
    response = test_client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "service_ready": True}


def test_predict_endpoint_success(test_client, mock_prediction_service):
    # Create a dummy image (e.g., 1x1 black image) and base64 encode it
    dummy_image = Image.fromarray(np.zeros((1, 1), dtype=np.uint8))
    buffered = BytesIO()
    dummy_image.save(buffered, format="PNG")
    encoded_image = base64.b64encode(buffered.getvalue()).decode("utf-8")

    request_payload = {
        "image_data": {"image_bytes": encoded_image},
        "ehr_data": {
            "patient_id": "PTest001",
            "age": 60,
            "gender": "male",
            "symptoms": ["cough"],
            "lab_results": {"CRP": 10},
            "clinical_notes": "Patient has a mild cough.",
        },
        "graph_node_indices": [0],
    }

    response = test_client.post("/predict", json=request_payload)
    assert response.status_code == 200
    response_json = response.json()

    # Assert that the mocked service method was called
    mock_prediction_service.get_comprehensive_prediction.assert_called_once()

    # Assert on the structure of the response based on the mock
    assert response_json["patient_id"] == "PTest001"
    assert response_json["overall_risk_score"] == 0.65
    assert "image_analysis_results" in response_json
    assert "ehr_analysis_results" in response_json
    assert "graph_analysis_results" in response_json
    assert "recommendations" in response_json
    assert "explainability" in response_json
    assert response_json["explainability"]["image_explanation"] == "mock image XAI"


def test_predict_endpoint_no_data(test_client, mock_prediction_service):
    # Test with an empty request body (optional data)
    request_payload = {
        "ehr_data": {
            "patient_id": "PTest002",
            "age": 30,
            "gender": "female",
            "symptoms": [],
            "lab_results": {},
            "clinical_notes": None,
        }
    }
    response = test_client.post("/predict", json=request_payload)
    assert response.status_code == 200
    mock_prediction_service.get_comprehensive_prediction.assert_called_once()
    assert response.json()["patient_id"] == "PTest002"


def test_predict_endpoint_malformed_input(test_client):
    # Test with missing required field (e.g., image_bytes for ImageData)
    malformed_payload = {
        "image_data": {"invalid_field": "test"},  # Missing image_bytes
        "ehr_data": {"patient_id": "PTest003", "age": 25, "gender": "male"},
    }
    response = test_client.post("/predict", json=malformed_payload)
    assert response.status_code == 422  # FastAPI's validation error
