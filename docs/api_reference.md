# API Reference

The MediPredict API provides endpoints for comprehensive patient prediction and specific model inferences.

## Base URL

`http://localhost:8000` (for local development)

## Endpoints

### 1. Health Check

`GET /health`

* **Summary:** Checks the health of the API and its underlying services.
* **Response:**
    ```json
    {
      "status": "ok",
      "service_ready": true
    }
    ```

### 2. Get Comprehensive Patient Prediction

`POST /predict`

* **Summary:** Receives multimodal patient data and returns a comprehensive prediction, including risk scores, analyses from different AI models, and recommendations.
* **Request Body (JSON):**
    ```json
    {
      "image_data": {
        "image_bytes": "base64_encoded_image_string"
      },
      "ehr_data": {
        "patient_id": "P12345",
        "age": 65,
        "gender": "male",
        "symptoms": ["cough", "fever"],
        "lab_results": {"CRP": 15, "WBC": 12000},
        "clinical_notes": "Patient presented with a persistent cough and elevated CRP levels."
      },
      "graph_node_indices": [0, 3]
    }
    ```
* **Response Body (JSON - example):**
    ```json
    {
      "patient_id": "P12345",
      "overall_risk_score": 0.75,
      "image_analysis_results": {
        "predicted_class": 1,
        "probabilities": [0.2, 0.8]
      },
      "ehr_analysis_results": {
        "nlp_prediction": {
          "label": "LABEL_1",
          "score": 0.92
        }
      },
      "graph_analysis_results": [
        {
          "node_index": 0,
          "predicted_class": 1,
          "probabilities": [0.1, 0.9]
        }
      ],
      "recommendations": [
        "Image analysis suggests class: 1",
        "NLP analysis on notes suggests: LABEL_1",
        "Patient age indicates higher risk.",
        "Elevated CRP requires further investigation."
      ],
      "explainability": {
        "image_explanation": "Conceptual heatmap or feature importance for image regions.",
        "nlp_explanation": "Key phrases contributing to the prediction of LABEL_1: [Conceptual important words from 'Patient presented with a persistent cough and elevated CRP levels.']"
      }
    }
    ```

---

*(Further documentation for specific endpoints, error codes, authentication, etc. can be added here.)*