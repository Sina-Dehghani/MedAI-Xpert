from typing import Any, Dict, List
from abc import ABC, abstractmethod


class IDataSource(ABC):
    """
    Interface for data source connectors.
    Adheres to Interface Segregation Principle (ISP) and Liskov Substitution Principle (LSP).
    """

    @abstractmethod
    def connect(self) -> Any:
        """Establishes connection to the data source."""
        pass

    @abstractmethod
    def disconnect(self, connection: Any):
        """Closes connection to the data source."""
        pass


class IDataExtractor(ABC):
    """
    Interface for extracting raw data.
    """

    @abstractmethod
    def extract(self, source_config: Dict[str, Any]) -> Any:
        """Extracts raw data from the source."""
        pass


class IDataTransformer(ABC):
    """
    Interface for transforming raw data into a usable format.
    """

    @abstractmethod
    def transform(self, raw_data: Any) -> Any:
        """Transforms raw data."""
        pass


class IDataLoader(ABC):
    """
    Interface for loading transformed data into a data store.
    """

    @abstractmethod
    def load(self, transformed_data: Any, destination_config: Dict[str, Any]):
        """Loads transformed data into a destination."""
        pass


class IETLPipeline(ABC):
    """
    Interface for a complete ETL pipeline.
    Adheres to Single Responsibility Principle (SRP) for orchestrating ETL.
    """

    @abstractmethod
    def run_pipeline(
        self, source_config: Dict[str, Any], destination_config: Dict[str, Any]
    ):
        """Runs the full ETL pipeline."""
        pass
