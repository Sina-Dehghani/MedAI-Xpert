import base64
import logging
import numpy as np
from PIL import Image
from io import BytesIO
from typing import Dict, Any, List

from config.settings import settings
from src.utils.logging_config import setup_logging
from src.models.nlp.text_classifier import TextClassifier
from src.models.computer_vision.image_detector import ImageDetector
from src.models.graph_neural_networks.disease_graph_model import DiseaseGraphModel
from src.apis.main import (
    PredictionRequest,
    PredictionResponse,
    ImageData,
)  # Import Pydantic models

setup_logging()
logger = logging.getLogger(__name__)


class PredictionService:
    """
    A Facade service to orchestrate predictions from various AI models.
    Handles data preparation, model inference, and result aggregation.
    """

    def __init__(self):
        self.image_detector: ImageDetector = None
        self.text_classifier: TextClassifier = None
        self.disease_graph_model: DiseaseGraphModel = None
        self._initialize_models()
        logger.info("PredictionService initialized.")

    def _initialize_models(self):
        """
        Initializes and loads all necessary AI models.
        This would ideally be managed by a dependency injection framework or a model registry.
        """
        try:
            # Initialize Image Detector
            # Dummy num_classes = 2 for simple binary classification (e.g., disease present/absent)
            self.image_detector = ImageDetector(num_classes=2)
            self.image_detector.load(settings.IMAGE_MODEL_PATH)

        except Exception as e:
            logger.error(f"Failed to load ImageDetector: {e}")
            self.image_detector = None

        try:
            # Initialize Text Classifier
            # Dummy num_labels = 2 for binary classification (e.g., fraud/not fraud)
            self.text_classifier = TextClassifier(num_labels=2)
            self.text_classifier.load(settings.NLP_MODEL_PATH)

        except Exception as e:
            logger.error(f"Failed to load TextClassifier: {e}")
            self.text_classifier = None

        try:
            # Initialize Disease Graph Model
            # Dummy values, adjust based on your GNN graph structure
            self.disease_graph_model = DiseaseGraphModel(
                num_node_features=2, num_classes=2
            )
            # Note: GNN often needs graph data loaded separately or alongside.
            # For this demo, we assume the graph data is prepared/available when loading the model.
            self.disease_graph_model.load(
                settings.NLP_MODEL_PATH
            )  # Placeholder, needs its own model path
            # A real GNN might also need to load its graph data here
            # self.disease_graph_model._data_graph = preloaded_graph_data

        except Exception as e:
            logger.error(f"Failed to load DiseaseGraphModel: {e}")
            self.disease_graph_model = None

        if not any(
            [self.image_detector, self.text_classifier, self.disease_graph_model]
        ):
            logger.warning(
                "No AI models were successfully loaded. Prediction service will be limited."
            )

    async def get_comprehensive_prediction(
        self, request: PredictionRequest
    ) -> PredictionResponse:
        """
        Orchestrates predictions from multiple AI models based on the input request.
        """
        image_analysis_results = None
        ehr_analysis_results = None
        graph_analysis_results = None
        overall_risk_score = 0.0
        recommendations = []
        explainability = {}

        # 1. Image Analysis
        if request.image_data and self.image_detector:
            try:
                # Decode base64 image
                image_bytes = base64.b64decode(request.image_data.image_bytes)
                img = Image.open(BytesIO(image_bytes)).convert(
                    "L"
                )  # Convert to grayscale for simple CNN
                img_np = np.array(img)  # Convert to NumPy array

                img_pred = self.image_detector.predict(img_np)
                image_analysis_results = img_pred
                explainability["image_explanation"] = self.image_detector.explain(
                    img_np, img_pred
                )
                overall_risk_score += (
                    image_analysis_results.get("probabilities", [0.0, 0.0])[1] * 0.4
                )  # Arbitrary weight
                recommendations.append(
                    f"Image analysis suggests class: {image_analysis_results.get('predicted_class')}"
                )
            except Exception as e:
                logger.error(f"Image analysis failed: {e}")
                image_analysis_results = {"error": str(e)}

        # 2. EHR Analysis (NLP on clinical notes, structured data analysis)
        if request.ehr_data and self.text_classifier:
            try:
                ehr_analysis_results = {}
                # NLP on clinical notes
                if request.ehr_data.clinical_notes:
                    nlp_pred = self.text_classifier.predict(
                        request.ehr_data.clinical_notes
                    )
                    ehr_analysis_results["nlp_prediction"] = nlp_pred
                    explainability["nlp_explanation"] = self.text_classifier.explain(
                        request.ehr_data.clinical_notes, nlp_pred
                    )
                    overall_risk_score += (
                        nlp_pred.get("score", 0.0) * 0.3
                    )  # Arbitrary weight
                    recommendations.append(
                        f"NLP analysis on notes suggests: {nlp_pred.get('label')}"
                    )

                # Simple structured data analysis (e.g., age, lab results)
                if request.ehr_data.age > 60:
                    overall_risk_score += 0.1
                    recommendations.append("Patient age indicates higher risk.")
                if request.ehr_data.get("lab_results", {}).get("CRP", 0) > 10:
                    overall_risk_score += 0.15
                    recommendations.append(
                        "Elevated CRP requires further investigation."
                    )

            except Exception as e:
                logger.error(f"EHR analysis failed: {e}")
                ehr_analysis_results = {"error": str(e)}

        # 3. Graph Neural Network Analysis
        if request.graph_node_indices and self.disease_graph_model:
            try:
                graph_pred = self.disease_graph_model.predict(
                    request.graph_node_indices
                )
                graph_analysis_results = graph_pred
                # Example: Aggregate GNN insights into risk score
                for node_res in graph_pred:
                    overall_risk_score += (
                        node_res.get("probabilities", [0.0, 0.0])[1] * 0.2
                    )  # Arbitrary weight
                    recommendations.append(
                        f"GNN suggests node {node_res.get('node_index')} is class {node_res.get('predicted_class')}."
                    )
            except Exception as e:
                logger.error(f"Graph analysis failed: {e}")
                graph_analysis_results = {"error": str(e)}

        # Final aggregation and normalization of risk score (conceptual)
        overall_risk_score = min(
            1.0, max(0.0, overall_risk_score)
        )  # Clamp between 0 and 1

        # Populate the PredictionResponse
        response = PredictionResponse(
            patient_id=request.ehr_data.patient_id if request.ehr_data else "N/A",
            overall_risk_score=overall_risk_score,
            image_analysis_results=image_analysis_results,
            ehr_analysis_results=ehr_analysis_results,
            graph_analysis_results=graph_analysis_results,
            recommendations=(
                recommendations
                if recommendations
                else ["No specific recommendations generated."]
            ),
            explainability=explainability,
        )
        logger.info(
            f"Comprehensive prediction for patient {response.patient_id} completed."
        )
        return response
