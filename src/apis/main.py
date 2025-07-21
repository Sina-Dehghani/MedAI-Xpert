import logging
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException
from typing import List, Dict, Any, Optional

from config.settings import settings
from src.utils.logging_config import setup_logging
from src.prediction.prediction_service import PredictionService  # Will create this next

setup_logging()
logger = logging.getLogger(__name__)

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.API_VERSION,
    description="An AI-powered Clinical Decision Support System API for early disease detection and personalized treatment planning.",
)

# Initialize Prediction Service (Singleton or dependency injection in real app)
# For simplicity, we initialize it directly. In a larger app, consider using Depends() for FastAPI.
try:
    prediction_service = PredictionService()
    logger.info("PredictionService initialized for API.")
except Exception as e:
    logger.error(f"Failed to initialize PredictionService: {e}")
    prediction_service = None  # Handle gracefully


# --- Pydantic Models for Request/Response ---
class ImageData(BaseModel):
    """Represents medical image data for prediction."""

    image_bytes: str = Field(..., description="Base64 encoded image data.")
    # In a real scenario, this might be a path to an image on a shared volume or S3 URL


class EHRData(BaseModel):
    """Represents structured EHR data for prediction."""

    patient_id: str
    age: int
    gender: str
    symptoms: List[str] = []
    lab_results: Dict[str, Any] = {}
    clinical_notes: Optional[str] = None  # For NLP analysis


class GraphDataNode(BaseModel):
    """Represents a node in the disease graph for GNN prediction."""

    node_id: int
    node_features: List[
        float
    ]  # Features associated with the node (e.g., disease characteristics)


class PredictionRequest(BaseModel):
    """Combines different data types for a comprehensive prediction request."""

    image_data: Optional[ImageData] = None
    ehr_data: Optional[EHRData] = None
    graph_node_indices: Optional[List[int]] = None  # For GNN predictions


class PredictionResponse(BaseModel):
    """Represents the combined prediction output."""

    patient_id: str
    overall_risk_score: float
    image_analysis_results: Optional[Dict[str, Any]] = None
    ehr_analysis_results: Optional[Dict[str, Any]] = None
    graph_analysis_results: Optional[List[Dict[str, Any]]] = None
    recommendations: List[str]
    explainability: Dict[str, Any]


# --- API Endpoints ---
@app.get("/health", summary="Health Check", tags=["System"])
async def health_check():
    """Checks the health of the API."""
    if prediction_service is None:
        raise HTTPException(
            status_code=503, detail="Prediction service not initialized."
        )
    return {"status": "ok", "service_ready": prediction_service is not None}


@app.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Get Comprehensive Patient Prediction",
    tags=["Prediction"],
)
async def get_comprehensive_prediction(request: PredictionRequest):
    """
    Receives multimodal patient data and returns a comprehensive prediction
    including risk scores, analyses from different AI models, and recommendations.
    """
    if prediction_service is None:
        raise HTTPException(status_code=503, detail="Prediction service not available.")

    try:
        results = await prediction_service.get_comprehensive_prediction(request)
        return results
    except Exception as e:
        logger.error(f"Error during prediction: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {e}")


@app.post("/predict/image", summary="Predict from Image Data", tags=["Prediction"])
async def predict_image(image_data: ImageData):
    # This could be a separate endpoint if needed, or subsumed by /predict
    return {"message": "Image prediction endpoint (for specific testing)"}


@app.post("/predict/ehr", summary="Predict from EHR Data", tags=["Prediction"])
async def predict_ehr(ehr_data: EHRData):
    # Similar to image, for specific testing
    return {"message": "EHR prediction endpoint (for specific testing)"}
