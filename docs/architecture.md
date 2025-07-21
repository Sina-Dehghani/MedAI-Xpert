# Architecture Overview

MediPredict is designed with a scalable microservices architecture to handle the complexities of multimodal medical data and AI model deployment.

## Core Components:

* **Data Ingestion Service:** Responsible for extracting, transforming, and loading data from various sources (PACS, EHR systems, genomic sequencers). Implemented with `IETLPipeline` and concrete strategies (`ImageETLPipeline`, `EHRETLPipeline`).
* **AI Model Services:**
    * **Image Analysis Service:** Hosts Computer Vision models (`ImageDetector`) for image pre-processing, feature extraction, and disease detection.
    * **NLP Analysis Service:** Manages NLP pipelines (`TextClassifier`) for processing clinical notes and unstructured text.
    * **Graph Analysis Service:** Hosts Graph Neural Networks (`DiseaseGraphModel`) for modeling complex disease relationships and patient pathways.
* **Prediction & Fusion Service:** Orchestrates calls to individual AI models, fuses multimodal insights, and generates comprehensive patient predictions and recommendations. (Implemented as `PredictionService` facade).
* **API Gateway:** Provides a unified RESTful API (`FastAPI` in `src/apis/main.py`) for external systems to interact with MediPredict.
* **Data Warehouse/Lake:** Centralized repository for cleaned, processed, and raw medical data (PostgreSQL, Cassandra, Redis - conceptual integration).
* **MLOps Platform:** Tools for experiment tracking (MLflow/Weights & Biases), model versioning (DVC), continuous integration/deployment (CI/CD), and monitoring (Prometheus/Grafana - conceptual).

---

## Architecture Diagram

This project's architecture is visually represented using [Mermaid syntax](https://mermaid.js.org/).

**To view the diagram:**

* **On GitHub:** The diagram will render automatically when viewing this `architecture.md` file in the project's GitHub repository.
* **In Local Editors:** If your Markdown editor (like PyCharm's built-in preview) does not display the diagram, you may need a specific plugin (e.g., "Mermaid" or "Markdown with Mermaid" plugin for PyCharm).
* **Mermaid Live Editor:** You can copy the code block below and paste it into the [Mermaid Live Editor](https://mermaid.live/) to see an interactive visualization.

```mermaid
graph TD
    A[External Systems] --> B(API Gateway);
    B --> C{Prediction Service};
    C --> D1(Image Analysis Model);
    C --> D2(NLP Analysis Model);
    C --> D3(GNN Model);
    D1 --> E1(Medical Images);
    D2 --> E2(EHR Data - Clinical Notes);
    D3 --> E3(EHR Data - Graph Structured);
    E1 -- ETL --> F1(Data Warehouse/Lake);
    E2 -- ETL --> F1;
    E3 -- ETL --> F1;
    F1 --> G(MLOps Platform);
    G --> D1;
    G --> D2;
    G --> D3;
    C --> H[Recommendations & Explanations];
    H --> B;