import os
import logging
from typing import Any, Dict
from abc import ABC, abstractmethod

from src.utils.logging_config import setup_logging
from src.models.interfaces import IModel, IExplainableModel

setup_logging()
logger = logging.getLogger(__name__)


class BaseModel(IModel, ABC):
    """
    Abstract base class for all AI models, implementing common lifecycle methods.
    Uses Template Method Pattern for load/train/predict/evaluate.
    """

    def __init__(self, model_name: str):
        self.model_name = model_name
        self._model = None  # Internal model instance
        logger.info(f"BaseModel '{model_name}' initialized.")

    def load(self, model_path: str):
        """Template method to load the model."""
        if not os.path.exists(model_path):
            logger.warning(
                f"Model path '{model_path}' not found for {self.model_name}. Skipping load."
            )
            return  # Allow models to be created without pre-existing weights for initial training

        try:
            self._load_model_weights(model_path)
            logger.info(f"Model '{self.model_name}' loaded from '{model_path}'.")
        except Exception as e:
            logger.error(
                f"Failed to load model '{self.model_name}' from '{model_path}': {e}"
            )
            raise

    def train(self, data: Any, config: Dict[str, Any]):
        """Template method to train the model."""
        logger.info(f"Starting training for model '{self.model_name}'...")

        try:
            self._train_model(data, config)
            logger.info(f"Training completed for model '{self.model_name}'.")

        except Exception as e:
            logger.error(f"Failed to train model '{self.model_name}': {e}")
            raise

    def predict(self, data: Any) -> Any:
        """Template method to perform prediction."""
        if self._model is None:
            logger.error(f"Model '{self.model_name}' is not loaded. Cannot predict.")
            raise RuntimeError(f"Model '{self.model_name}' not loaded.")

        logger.debug(f"Performing prediction with model '{self.model_name}'.")
        try:
            return self._perform_prediction(data)

        except Exception as e:
            logger.error(f"Failed to predict with model '{self.model_name}': {e}")
            raise

    def evaluate(self, data: Any) -> Dict[str, Any]:
        """Template method to evaluate the model."""
        if self._model is None:
            logger.error(f"Model '{self.model_name}' is not loaded. Cannot evaluate.")
            raise RuntimeError(f"Model '{self.model_name}' not loaded.")

        logger.info(f"Evaluating model '{self.model_name}'.")
        try:
            return self._evaluate_model(data)

        except Exception as e:
            logger.error(f"Failed to evaluate model '{self.model_name}': {e}")
            raise

    @abstractmethod
    def _load_model_weights(self, model_path: str):
        """Concrete implementation for loading model weights."""
        pass

    @abstractmethod
    def _train_model(self, data: Any, config: Dict[str, Any]):
        """Concrete implementation for model training."""
        pass

    @abstractmethod
    def _perform_prediction(self, data: Any) -> Any:
        """Concrete implementation for prediction logic."""
        pass

    @abstractmethod
    def _evaluate_model(self, data: Any) -> Dict[str, Any]:
        """Concrete implementation for model evaluation."""
        pass


class ExplainableBaseModel(BaseModel, IExplainableModel, ABC):
    """
    Abstract base class for explainable AI models.
    """

    def __init__(self, model_name: str):
        super().__init__(model_name)
        logger.info(f"ExplainableBaseModel '{model_name}' initialized.")

    @abstractmethod
    def explain(self, data: Any, prediction: Any) -> Dict[str, Any]:
        """Concrete implementation for generating explanations."""
        pass
