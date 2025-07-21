import torch
import logging
from typing import Any, Dict, List
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

from src.utils.logging_config import setup_logging
from src.models.base_model import BaseModel, ExplainableBaseModel

setup_logging()
logger = logging.getLogger(__name__)


class TextClassifier(ExplainableBaseModel):
    """
    Implements a text classification model using Hugging Face Transformers.
    """

    def __init__(
        self, model_name: str = "distilbert-base-uncased", num_labels: int = 2
    ):
        super().__init__(model_name=f"TextClassifier_{model_name}")
        self.pretrained_model_name = model_name
        self.num_labels = num_labels
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=num_labels
        )
        self._pipeline = None  # For inference
        logger.info(f"TextClassifier initialized with {model_name}.")

    def _load_model_weights(self, model_path: str):
        """Loads Hugging Face model from a local path."""
        try:
            self._model = AutoModelForSequenceClassification.from_pretrained(model_path)
            self._model.eval()
            self._pipeline = pipeline(
                "text-classification", model=self._model, tokenizer=self._tokenizer
            )
            logger.info(f"TextClassifier model loaded from {model_path}.")

        except Exception as e:
            logger.error(f"Error loading TextClassifier weights from {model_path}: {e}")
            raise

    def _train_model(self, data: List[Dict[str, Any]], config: Dict[str, Any]):
        """
        Trains the text classification model (conceptual, using Trainer API in real scenario).
        `data` expects a list of dicts: [{'text': str, 'label': int}, ...]
        `config` expects {'epochs': int, 'batch_size': int, 'output_dir': str}
        """
        logger.warning(
            "Training method for TextClassifier is conceptual. Use HuggingFace Trainer for proper training."
        )
        # This would typically involve data collators, Trainer API, and Dataset objects
        # For a minimal example:
        texts = [item["text"] for item in data]
        labels = torch.tensor([item["label"] for item in data])
        encodings = self._tokenizer(
            texts, truncation=True, padding=True, return_tensors="pt"
        )

        optimizer = torch.optim.AdamW(
            self._model.parameters(), lr=config.get("learning_rate", 5e-5)
        )
        criterion = torch.nn.CrossEntropyLoss()

        self._model.train()
        for epoch in range(config.get("epochs", 3)):
            for i in range(0, len(texts), config.get("batch_size", 8)):
                batch_inputs = {
                    k: v[i : i + config.get("batch_size", 8)]
                    for k, v in encodings.items()
                }
                batch_labels = labels[i : i + config.get("batch_size", 8)]

                optimizer.zero_grad()
                outputs = self._model(**batch_inputs, labels=batch_labels)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
            logger.info(f"Epoch {epoch+1}/{config['epochs']}, Loss: {loss.item():.4f}")

        output_dir = config.get("output_dir", "models/text_classifier")
        self._model.save_pretrained(output_dir)
        self._tokenizer.save_pretrained(output_dir)
        logger.info(f"TextClassifier model saved to {output_dir}.")

    def _perform_prediction(self, data: str) -> Dict[str, Any]:
        """
        Performs prediction on a single text string.
        """
        if self._pipeline is None:
            # Initialize pipeline if not loaded
            self._pipeline = pipeline(
                "text-classification", model=self._model, tokenizer=self._tokenizer
            )
        result = self._pipeline(data)[0]
        logger.debug(f"TextClassifier prediction for '{data[:50]}...': {result}")

        return result

    def _evaluate_model(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Evaluates the model on test data.
        `data` expects a list of dicts: [{'text': str, 'label': int}, ...]
        """
        if self._pipeline is None:
            self._pipeline = pipeline(
                "text-classification", model=self._model, tokenizer=self._tokenizer
            )

        true_labels = [item["label"] for item in data]
        predictions = []
        for item in data:
            res = self._pipeline(item["text"])[0]
            # Assuming 'LABEL_0', 'LABEL_1' format for binary classification
            pred_label = int(res["label"].split("_")[1])
            predictions.append(pred_label)

        # Simple accuracy calculation
        correct_predictions = sum(
            1 for true, pred in zip(true_labels, predictions) if true == pred
        )
        accuracy = correct_predictions / len(true_labels) if true_labels else 0
        logger.info(f"TextClassifier evaluation accuracy: {accuracy:.4f}")

        return {"accuracy": accuracy}

    def explain(self, data: str, prediction: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generates explanation for text classification (conceptual, e.g., LIME or SHAP).
        """
        logger.info("Generating explanation for text classification (conceptual XAI).")
        explanation_text = f"Key phrases contributing to the prediction of {prediction.get('label')}: [Conceptual important words from '{data}']"
        return {"explanation": explanation_text, "method": "Conceptual LIME/SHAP"}


if __name__ == "__main__":
    setup_logging()

    # Initialize and train model (conceptual training)
    classifier = TextClassifier(num_labels=2)  # e.g., fraud/not fraud
    train_data = [
        {"text": "This is a normal medical record, no issues found.", "label": 0},
        {
            "text": "Patient shows signs of potential fraud with suspicious claims.",
            "label": 1,
        },
        {"text": "Routine checkup, everything seems fine.", "label": 0},
        {"text": "Duplicate claim detected, investigation required.", "label": 1},
    ]
    train_config = {
        "epochs": 1,
        "batch_size": 2,
        "learning_rate": 1e-5,
        "output_dir": "models/test_text_classifier",
    }
    classifier.train(train_data, train_config)

    # Load the trained model
    classifier.load("models/test_text_classifier")

    # Perform prediction
    text_to_predict = "Suspicious activity detected in patient records."
    prediction_result = classifier.predict(text_to_predict)
    print(f"Prediction Result: {prediction_result}")

    # Evaluate model
    eval_data = [
        {"text": "No unusual patterns in the claims.", "label": 0},
        {"text": "High likelihood of fraudulent activity.", "label": 1},
    ]
    eval_metrics = classifier.evaluate(eval_data)
    print(f"Evaluation Metrics: {eval_metrics}")

    # Get explanation
    explanation = classifier.explain(text_to_predict, prediction_result)
    print(f"Explanation: {explanation}")
