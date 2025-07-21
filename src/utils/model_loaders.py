import os
import torch
import logging
from typing import Union, Type, Dict, Any
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from src.models.base_model import IModel
from src.utils.logging_config import setup_logging
from src.models.nlp.text_classifier import TextClassifier
from src.models.computer_vision.image_detector import ImageDetector
from src.models.graph_neural_networks.disease_graph_model import DiseaseGraphModel

setup_logging()
logger = logging.getLogger(__name__)


class ModelFactory:
    """
    A Factory for creating and loading different types of AI models.
    Adheres to Factory Method Design Pattern.
    """

    _model_map: Dict[str, Type[IModel]] = {
        "ImageDetector": ImageDetector,
        "TextClassifier": TextClassifier,
        "DiseaseGraphModel": DiseaseGraphModel,
        # Add other model types here
    }

    @classmethod
    def create_model(cls, model_type: str, **kwargs) -> IModel:
        """
        Creates an instance of a specified model type.
        """
        model_class = cls._model_map.get(model_type)
        if not model_class:
            logger.error(f"Unknown model type: {model_type}")
            raise ValueError(f"Unknown model type: {model_type}")

        try:
            model_instance = model_class(**kwargs)
            logger.info(f"Created model instance of type: {model_type}")
            return model_instance

        except TypeError as e:
            logger.error(f"Error creating model {model_type} with kwargs {kwargs}: {e}")
            raise


class ModelLoader:
    """
    Centralized utility for loading AI models.
    Could be extended for integration with MLflow Model Registry.
    """

    @staticmethod
    def load_model_instance(model_instance: IModel, model_path: str):
        """
        Loads weights into an already instantiated model.
        """
        try:
            model_instance.load(model_path)
            logger.info(
                f"Successfully loaded weights for {model_instance.model_name} from {model_path}."
            )
        except Exception as e:
            logger.error(
                f"Failed to load weights for {model_instance.model_name} from {model_path}: {e}"
            )
            raise

    @staticmethod
    def load_hf_transformer_model(model_name_or_path: str, num_labels: int = None):
        """Loads a Hugging Face transformer model and tokenizer."""
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
            model = AutoModelForSequenceClassification.from_pretrained(
                model_name_or_path, num_labels=num_labels
            )
            logger.info(
                f"Loaded Hugging Face model and tokenizer from {model_name_or_path}."
            )
            return tokenizer, model

        except Exception as e:
            logger.error(
                f"Failed to load Hugging Face model from {model_name_or_path}: {e}"
            )
            raise


if __name__ == "__main__":
    setup_logging()
    from config.settings import settings  # Ensure settings are available

    # Example usage of ModelFactory
    print("\n--- ModelFactory Example ---")
    try:
        # Create an ImageDetector instance
        image_detector_model = ModelFactory.create_model("ImageDetector", num_classes=2)
        print(f"Created: {image_detector_model.model_name}")

        # Create a TextClassifier instance
        text_classifier_model = ModelFactory.create_model(
            "TextClassifier", model_name="distilbert-base-uncased", num_labels=2
        )
        print(f"Created: {text_classifier_model.model_name}")

        # Example of unknown model type
        # unknown_model = ModelFactory.create_model("UnknownModel")

    except ValueError as e:
        print(f"Error: {e}")

    except Exception as e:
        print(f"Unexpected error: {e}")

    # Usage of ModelLoader
    print("\n--- ModelLoader Example (Conceptual) ---")
    # For actual loading, ensure the paths exist or mock them
    try:
        # Assume a dummy model file exists for ImageDetector
        dummy_image_model_path = "models/dummy_image_detector.pth"
        torch.save(
            {"dummy_key": "dummy_value"}, dummy_image_model_path
        )  # Create a dummy file

        # Instantiate a model first, then load weights
        img_det = ModelFactory.create_model("ImageDetector", num_classes=2)
        ModelLoader.load_model_instance(img_det, dummy_image_model_path)
        print(f"Attempted to load ImageDetector from {dummy_image_model_path}")

        os.remove(dummy_image_model_path)  # Clean up dummy file

    except Exception as e:
        print(f"Error during ModelLoader example: {e}")
