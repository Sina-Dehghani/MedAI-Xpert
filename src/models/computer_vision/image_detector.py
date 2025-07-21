import torch
import logging
import torch.nn as nn
from typing import Any, Dict, List
import torchvision.transforms as transforms

from src.utils.logging_config import setup_logging
from src.models.base_model import BaseModel, ExplainableBaseModel

# from captum.attr import IntegratedGradients # Example for XAI

setup_logging()
logger = logging.getLogger(__name__)


class SimpleCNN(nn.Module):
    """A simple CNN for demonstration purposes."""

    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(32, num_classes)
        logger.info("SimpleCNN architecture initialized.")

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class ImageDetector(ExplainableBaseModel):
    """
    Implements an image detection model, inheriting from ExplainableBaseModel.
    Uses a simple CNN. In a real project, this would be a more complex model (e.g., U-Net, YOLO).
    """

    def __init__(self, num_classes: int = 2):
        super().__init__(model_name="ImageDetector")
        self._model = SimpleCNN(num_classes=num_classes)
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize((64, 64)),  # Example fixed size
                transforms.Normalize(mean=[0.5], std=[0.5]),  # Example for grayscale
            ]
        )
        # self.ig = IntegratedGradients(self._model) # For XAI demonstration
        logger.info("ImageDetector initialized with SimpleCNN.")

    def _load_model_weights(self, model_path: str):
        """Loads PyTorch model state dict."""
        try:
            self._model.load_state_dict(torch.load(model_path))
            self._model.eval()
            logger.info(f"ImageDetector model weights loaded from {model_path}.")
        except Exception as e:
            logger.error(f"Error loading ImageDetector weights: {e}")
            raise

    def _train_model(self, data: List[Dict[str, Any]], config: Dict[str, Any]):
        """
        Trains the image detection model.
        `data` expects a list of dicts: [{'image': np_array, 'label': int}, ...]
        `config` expects {'epochs': int, 'learning_rate': float}
        """
        self._model.train()
        optimizer = torch.optim.Adam(
            self._model.parameters(), lr=config.get("learning_rate", 0.001)
        )
        criterion = nn.CrossEntropyLoss()

        # Dummy data conversion for training (in real scenario, use DataLoader)
        images_tensor = torch.stack(
            [self.transform(img_data["image"]) for img_data in data]
        )
        labels_tensor = torch.tensor(
            [img_data["label"] for img_data in data], dtype=torch.long
        )

        for epoch in range(config.get("epochs", 10)):
            optimizer.zero_grad()
            outputs = self._model(images_tensor)
            loss = criterion(outputs, labels_tensor)
            loss.backward()
            optimizer.step()
            logger.info(f"Epoch {epoch+1}/{config['epochs']}, Loss: {loss.item():.4f}")

        # Save model after training
        output_model_path = config.get("output_model_path", "models/image_detector.pth")
        torch.save(self._model.state_dict(), output_model_path)
        logger.info(f"ImageDetector model saved to {output_model_path}.")

    def _perform_prediction(self, data: Any) -> Dict[str, Any]:
        """
        Performs prediction on image data.
        `data` expects a numpy array representing the image.
        """
        self._model.eval()
        with torch.no_grad():
            input_tensor = self.transform(data).unsqueeze(0)  # Add batch dimension
            outputs = self._model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)[0].tolist()
            predicted_class = torch.argmax(outputs, dim=1).item()
            logger.debug(
                f"ImageDetector prediction: class {predicted_class}, probs {probabilities}"
            )
        return {"predicted_class": predicted_class, "probabilities": probabilities}

    def _evaluate_model(self, data: Any) -> Dict[str, Any]:
        """
        Evaluates the model on test data.
        `data` expects a list of dicts: [{'image': np_array, 'label': int}, ...]
        """
        self._model.eval()
        correct_predictions = 0
        total_predictions = 0

        with torch.no_grad():
            for item in data:
                input_tensor = self.transform(item["image"]).unsqueeze(0)
                outputs = self._model(input_tensor)
                predicted_class = torch.argmax(outputs, dim=1).item()
                if predicted_class == item["label"]:
                    correct_predictions += 1
                total_predictions += 1
        accuracy = (
            correct_predictions / total_predictions if total_predictions > 0 else 0
        )
        logger.info(f"ImageDetector evaluation accuracy: {accuracy:.4f}")

        return {"accuracy": accuracy}

    def explain(self, data: Any, prediction: Any) -> Dict[str, Any]:
        """
        Generates explanation for the image prediction using Grad-CAM (conceptual).
        """
        # This is a placeholder for actual XAI implementation like Grad-CAM
        # Requires libraries like captum or custom implementation
        logger.info("Generating explanation for image detection (conceptual XAI).")
        explanation_map = "Conceptual heatmap or feature importance for image regions."
        return {"explanation": explanation_map, "method": "Conceptual Grad-CAM"}


if __name__ == "__main__":
    import numpy as np

    setup_logging()

    # Create dummy image data
    dummy_image_data = np.random.randint(
        0, 255, size=(100, 100), dtype=np.uint8
    )  # Grayscale image
    dummy_train_data = [
        {
            "image": np.random.randint(0, 255, size=(100, 100), dtype=np.uint8),
            "label": 0,
        }
        for _ in range(20)
    ] + [
        {
            "image": np.random.randint(0, 255, size=(100, 100), dtype=np.uint8),
            "label": 1,
        }
        for _ in range(20)
    ]
    dummy_eval_data = [
        {
            "image": np.random.randint(0, 255, size=(100, 100), dtype=np.uint8),
            "label": 0,
        }
        for _ in range(5)
    ] + [
        {
            "image": np.random.randint(0, 255, size=(100, 100), dtype=np.uint8),
            "label": 1,
        }
        for _ in range(5)
    ]

    # Initialize and train model
    detector = ImageDetector()
    train_config = {
        "epochs": 2,
        "learning_rate": 0.01,
        "output_model_path": "models/test_image_detector.pth",
    }
    detector.train(dummy_train_data, train_config)

    # Load the trained model
    detector.load("models/test_image_detector.pth")

    # Perform prediction
    prediction_result = detector.predict(dummy_image_data)
    print(f"Prediction Result: {prediction_result}")

    # Evaluate model
    eval_metrics = detector.evaluate(dummy_eval_data)
    print(f"Evaluation Metrics: {eval_metrics}")

    # Get explanation
    explanation = detector.explain(dummy_image_data, prediction_result)
    print(f"Explanation: {explanation}")
