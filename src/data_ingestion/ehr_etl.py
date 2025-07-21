import os
import logging
import pandas as pd
from typing import Dict, Any, List

from src.utils.logging_config import setup_logging
from src.data_ingestion.interfaces import (
    IDataSource,
    IDataExtractor,
    IDataTransformer,
    IDataLoader,
    IETLPipeline,
)

setup_logging()
logger = logging.getLogger(__name__)


class EHRDataSource(IDataSource):
    """Connects to an EHR data source (e.g., CSV, database, FHIR server)."""

    def connect(self) -> str:
        """Returns a placeholder for EHR connection."""
        logger.info("Connecting to EHR data source.")
        return "data/raw/ehr_data.csv"  # Placeholder for a CSV file

    def disconnect(self, connection: str):
        logger.info(f"Disconnected from {connection}.")


class EHRExtractor(IDataExtractor):
    """Extracts EHR data from a given source (e.g., CSV, database query result)."""

    def extract(self, source_config: Dict[str, Any]) -> pd.DataFrame:
        """
        Extracts EHR data. `source_config` expects {'path': str, 'format': str}.
        """
        data_path = source_config.get("path")
        data_format = source_config.get("format", "csv")

        if data_format == "csv":
            df = pd.read_csv(data_path)
        elif data_format == "json":
            df = pd.read_json(data_path)
        # Add more formats like database connections, FHIR etc.
        else:
            raise ValueError(f"Unsupported EHR data format: {data_format}")
        logger.info(f"Extracted {len(df)} records from EHR data.")
        return df


class EHRTransformer(IDataTransformer):
    """Transforms raw EHR data (e.g., cleaning, normalization, feature engineering)."""

    def transform(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        """
        Performs basic cleaning and normalization on EHR data.
        """
        df = raw_data.copy()
        # 1. Handle missing values
        df = df.fillna(method="ffill").fillna(method="bfill")
        # 2. Convert data types
        for col in ["age", "lab_result_a", "lab_result_b"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        # 3. Normalize text fields for NLP (e.g., clinical notes)
        if "clinical_notes" in df.columns:
            df["clinical_notes"] = (
                df["clinical_notes"].astype(str).str.lower().str.strip()
            )

        logger.info(f"Transformed EHR data, resulting in {len(df)} records.")
        return df


class EHRDataLoader(IDataLoader):
    """Loads transformed EHR data into a relational database."""

    def load(self, transformed_data: pd.DataFrame, destination_config: Dict[str, Any]):
        """
        Loads transformed EHR data into a database.
        `destination_config` expects {'db_connection': Any, 'table_name': str}
        """
        db_connection = destination_config.get("db_connection")
        table_name = destination_config.get("table_name", "ehr_records")

        if db_connection:
            logger.info(
                f"Loading {len(transformed_data)} records into '{table_name}' table (conceptual)."
            )
            # Example: transformed_data.to_sql(table_name, con=db_connection, if_exists='append', index=False)
            # We'll save to CSV
            transformed_data.to_csv("data/processed/ehr_processed.csv", index=False)
            logger.info("EHR data loaded to 'data/processed/ehr_processed.csv'.")
        else:
            logger.warning(
                "No database connection provided for EHR data loading. Skipping DB load."
            )
        logger.info("EHR data loaded successfully.")


class EHRETLPipeline(IETLPipeline):
    """Orchestrates the ETL process for EHR data."""

    def __init__(
        self,
        extractor: IDataExtractor,
        transformer: IDataTransformer,
        loader: IDataLoader,
    ):
        self.extractor = extractor
        self.transformer = transformer
        self.loader = loader
        logger.info("EHRETLPipeline initialized.")

    def run_pipeline(
        self, source_config: Dict[str, Any], destination_config: Dict[str, Any]
    ):
        """
        Runs the full ETL pipeline for EHR data.
        """
        logger.info("Starting EHR ETL pipeline...")
        raw_data = self.extractor.extract(source_config)
        transformed_data = self.transformer.transform(raw_data)
        self.loader.load(transformed_data, destination_config)
        logger.info("EHR ETL pipeline completed.")


if __name__ == "__main__":
    # Create a dummy EHR CSV file for testing

    test_ehr_dir = "data/raw/"
    os.makedirs(test_ehr_dir, exist_ok=True)
    dummy_ehr_data = pd.DataFrame(
        {
            "patient_id": [1, 2, 3],
            "age": [50, 60, 45],
            "gender": ["M", "F", "M"],
            "diagnosis": ["Flu", "COVID-19", "Hypertension"],
            "clinical_notes": [
                "patient has mild fever and cough",
                "severe respiratory issues",
                "controlled BP",
            ],
        }
    )
    dummy_ehr_data.to_csv(os.path.join(test_ehr_dir, "ehr_data.csv"), index=False)
    logger.info(
        f"Created dummy EHR file at {os.path.join(test_ehr_dir, 'ehr_data.csv')}"
    )

    ehr_extractor = EHRExtractor()
    ehr_transformer = EHRTransformer()
    ehr_loader = EHRDataLoader()
    ehr_etl_pipeline = EHRETLPipeline(ehr_extractor, ehr_transformer, ehr_loader)

    source_config = {
        "path": os.path.join(test_ehr_dir, "ehr_data.csv"),
        "format": "csv",
    }
    destination_config = {
        "db_connection": None,
        "table_name": "ehr_records",
    }  # No actual DB connection for demo

    ehr_etl_pipeline.run_pipeline(source_config, destination_config)
