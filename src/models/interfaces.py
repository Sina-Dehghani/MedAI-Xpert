from typing import Any, Dict
from abc import ABC, abstractmethod


class IModel(ABC):
    """
    Base interface for all AI models.
    Adheres to Interface Segregation Principle (ISP) and Liskov Substitution Principle (LSP).
    """

    @abstractmethod
    def load(self, model_path: str):
        """Loads the model weights and configuration."""
        pass

    @abstractmethod
    def predict(self, data: Any) -> Any:
        """Performs prediction on input data."""
        pass

    @abstractmethod
    def train(self, data: Any, config: Dict[str, Any]):
        """Trains the model on input data."""
        pass

    @abstractmethod
    def evaluate(self, data: Any) -> Dict[str, Any]:
        """Evaluates the model and returns metrics."""
        pass


class IExplainableModel(IModel):
    """
    Interface for models that can provide explanations for their predictions.
    Extends IModel, demonstrating ISP.
    """

    @abstractmethod
    def explain(self, data: Any, prediction: Any) -> Dict[str, Any]:
        """Generates explanations for a given prediction."""
        pass
