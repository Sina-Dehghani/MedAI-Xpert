import os
import pydicom
import logging
import pandas as pd
from typing import Dict, Any, List

from src.utils.logging_config import setup_logging  # Ensure logging is set up
from src.data_ingestion.interfaces import (
    IDataSource,
    IDataExtractor,
    IDataTransformer,
    IDataLoader,
    IETLPipeline,
)

# Setup logging for this module
setup_logging()
logger = logging.getLogger(__name__)


class DICOMDataSource(IDataSource):
    """Connects to a DICOM image directory."""

    def connect(self) -> str:
        """Returns the path to the DICOM directory."""
        logger.info("Connecting to DICOM data source.")
        # In a real scenario, this might connect to a PACS system or cloud storage.
        return "data/raw/dicom_images"  # Placeholder

    def disconnect(self, connection: str):
        """Simulates disconnecting."""
        logger.info(f"Disconnected from {connection}.")


class DICOMExtractor(IDataExtractor):
    """Extracts DICOM files from a given directory."""

    def extract(self, source_config: Dict[str, Any]) -> List[pydicom.FileDataset]:
        """
        Extracts DICOM datasets from specified paths.
        `source_config` expects {'paths': List[str]}
        """
        dicom_files = []
        for path in source_config.get("paths", []):
            if os.path.isdir(path):
                for root, _, files in os.walk(path):
                    for file in files:
                        if file.endswith(".dcm"):  # Simple check for DICOM files
                            try:
                                dicom_files.append(
                                    pydicom.dcmread(os.path.join(root, file))
                                )
                                logger.debug(f"Extracted DICOM: {file}")
                            except Exception as e:
                                logger.warning(f"Could not read DICOM file {file}: {e}")
            elif os.path.isfile(path) and path.endswith(".dcm"):
                try:
                    dicom_files.append(pydicom.dcmread(path))
                    logger.debug(f"Extracted single DICOM: {path}")
                except Exception as e:
                    logger.warning(f"Could not read DICOM file {path}: {e}")
        logger.info(f"Extracted {len(dicom_files)} DICOM files.")
        return dicom_files


class DICOMTransformer(IDataTransformer):
    """Transforms DICOM datasets into a structured format (e.g., NumPy arrays, metadata Pandas DataFrame)."""

    def transform(self, raw_data: List[pydicom.FileDataset]) -> Dict[str, Any]:
        """
        Transforms a list of DICOM datasets.
        Returns a dict with 'images' (list of numpy arrays) and 'metadata' (pandas DataFrame).
        """
        images = []
        metadata_records = []
        for ds in raw_data:
            try:
                images.append(ds.pixel_array)
                # Extract key metadata
                meta = {
                    "PatientID": getattr(ds, "PatientID", None),
                    "StudyInstanceUID": getattr(ds, "StudyInstanceUID", None),
                    "SeriesInstanceUID": getattr(ds, "SeriesInstanceUID", None),
                    "Modality": getattr(ds, "Modality", None),
                    "SOPInstanceUID": getattr(ds, "SOPInstanceUID", None),
                    "Rows": getattr(ds, "Rows", None),
                    "Columns": getattr(ds, "Columns", None),
                    "PixelSpacing": getattr(ds, "PixelSpacing", None),
                    "BitsStored": getattr(ds, "BitsStored", None),
                }
                metadata_records.append(meta)
                logger.debug(f"Transformed DICOM for PatientID: {meta['PatientID']}")
            except Exception as e:
                logger.error(f"Error transforming DICOM dataset: {e}")
        logger.info(
            f"Transformed {len(images)} images and {len(metadata_records)} metadata records."
        )
        return {"images": images, "metadata": pd.DataFrame(metadata_records)}


class ImageDataLoader(IDataLoader):
    """Loads transformed image data (NumPy arrays, Pandas DataFrame) into a data store."""

    def load(
        self, transformed_data: Dict[str, Any], destination_config: Dict[str, Any]
    ):
        """
        Loads transformed images (e.g., to disk) and metadata (e.g., to a database).
        `destination_config` expects {'image_dir': str, 'metadata_db_connection': Any}
        """
        image_dir = destination_config.get("image_dir", "data/processed/images")
        os.makedirs(image_dir, exist_ok=True)

        images = transformed_data.get("images", [])
        metadata_df = transformed_data.get("metadata")

        # Placeholder for saving images and metadata
        for i, img in enumerate(images):
            # Save images as .npy files
            # np.save(os.path.join(image_dir, f'image_{i}.npy'), img)
            pass  # Placeholder for actual saving logic
        logger.info(f"Saved {len(images)} images to {image_dir} (conceptual).")

        if metadata_df is not None:
            # In a real scenario, this would load into a database (PostgreSQL, Cassandra etc.)
            # metadata_df.to_sql('image_metadata', con=destination_config['metadata_db_connection'], if_exists='append', index=False)
            metadata_df.to_csv(os.path.join(image_dir, "metadata.csv"), index=False)
            logger.info(
                f"Loaded {len(metadata_df)} metadata records to CSV (conceptual)."
            )
        logger.info("Image data loaded successfully.")


class ImageETLPipeline(IETLPipeline):
    """Orchestrates the ETL process for image data."""

    def __init__(
        self,
        extractor: IDataExtractor,
        transformer: IDataTransformer,
        loader: IDataLoader,
    ):
        self.extractor = extractor
        self.transformer = transformer
        self.loader = loader
        logger.info("ImageETLPipeline initialized.")

    def run_pipeline(
        self, source_config: Dict[str, Any], destination_config: Dict[str, Any]
    ):
        """
        Runs the full ETL pipeline for image data.
        """
        logger.info("Starting Image ETL pipeline...")
        raw_data = self.extractor.extract(source_config)
        transformed_data = self.transformer.transform(raw_data)
        self.loader.load(transformed_data, destination_config)
        logger.info("Image ETL pipeline completed.")


if __name__ == "__main__":
    # Create a dummy DICOM file for testing
    import numpy as np
    from pydicom.uid import generate_uid
    from pydicom.dataset import Dataset, FileDataset

    test_data_dir = "data/raw/dicom_test_data"
    os.makedirs(test_data_dir, exist_ok=True)

    filename_dicom = os.path.join(test_data_dir, "test_image.dcm")
    file_meta = Dataset()
    file_meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.1"  # CT Image Storage
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.TransferSyntaxUID = "1.2.840.10008.1.2.1"  # Explicit VR Little Endian

    ds = FileDataset(filename_dicom, {}, file_meta=file_meta, preamble=b"\0" * 128)
    ds.PatientName = "Test^Patient"
    ds.PatientID = "12345"
    ds.Modality = "CT"
    ds.Rows = 100
    ds.Columns = 100
    ds.PixelData = np.random.randint(0, 255, size=(100, 100), dtype=np.uint16).tobytes()

    ds.save_as(filename_dicom)
    logger.info(f"Created dummy DICOM file at {filename_dicom}")

    image_extractor = DICOMExtractor()
    image_transformer = DICOMTransformer()
    image_loader = ImageDataLoader()
    image_etl_pipeline = ImageETLPipeline(
        image_extractor, image_transformer, image_loader
    )

    source_config = {"paths": [test_data_dir]}
    destination_config = {"image_dir": "data/processed/images_output"}

    image_etl_pipeline.run_pipeline(source_config, destination_config)
